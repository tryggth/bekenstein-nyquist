"""
Unit and falsification tests for Milestone 1:
The N x N Unitary Block-Leakage Sandbox (`toy_model_1_unitary_leakage.py`).
"""

import pytest
import torch

from toy_model_1_unitary_leakage import (
    ASSERT_TOL,
    COMPLEX_DTYPE,
    build_bandlimit_projector,
    factorize_into_elementary_gates,
    generate_nonlinear_unitary,
    numerical_rank,
    partition_blocks,
    run_adversarial_falsification_check,
    run_depth_tradeoff_sweep,
    verify_kinematic_commutator_identities,
)


def test_bandlimit_projector_properties() -> None:
    N, d = 10, 4
    P, Q = build_bandlimit_projector(N=N, d=d)
    I_N = torch.eye(N, dtype=COMPLEX_DTYPE)
    assert P.dtype == COMPLEX_DTYPE
    assert torch.linalg.norm(P @ P - P).item() < ASSERT_TOL
    assert torch.linalg.norm(P.mH - P).item() < ASSERT_TOL
    assert torch.linalg.norm(Q @ Q - Q).item() < ASSERT_TOL
    assert torch.linalg.norm(P @ Q).item() < ASSERT_TOL
    assert torch.linalg.norm(P + Q - I_N).item() < ASSERT_TOL
    assert numerical_rank(P) == d
    assert numerical_rank(Q) == N - d


@pytest.mark.parametrize("method", ["qr", "hamiltonian"])
@pytest.mark.parametrize("N,d", [(10, 4), (12, 5), (8, 3)])
def test_kinematic_gram_and_commutator_rank_identities(method: str, N: int, d: int) -> None:
    P_Lambda, _ = build_bandlimit_projector(N=N, d=d)
    U_nl = generate_nonlinear_unitary(N=N, d=d, method=method, seed=101)
    kin = verify_kinematic_commutator_identities(U_nl, P_Lambda, d=d)

    r = kin["r"]
    assert r == min(d, N - d)
    assert kin["rank_A"] == r
    assert kin["rank_B"] == r
    assert kin["rank_commutator"] == 2 * r
    assert kin["unitary_err"] < ASSERT_TOL
    assert kin["gram_row_err"] < ASSERT_TOL
    assert kin["gram_col_err"] < ASSERT_TOL
    assert kin["comm_decomp_err"] < ASSERT_TOL


@pytest.mark.parametrize("method", ["qr", "hamiltonian"])
@pytest.mark.parametrize("target_r", [0, 1, 2, 3, 4])
def test_depth_tradeoff_exact_threshold_at_r(method: str, target_r: int) -> None:
    N, d = 10, 4
    P_Lambda, Q_Lambda = build_bandlimit_projector(N=N, d=d)
    U_nl = generate_nonlinear_unitary(
        N=N, d=d, method=method, target_rank=target_r, seed=200 + target_r
    )
    kin = verify_kinematic_commutator_identities(U_nl, P_Lambda, d=d)
    assert kin["r"] == target_r
    assert kin["rank_commutator"] == 2 * target_r

    aligned_sweep, coord_sweep = run_depth_tradeoff_sweep(U_nl, P_Lambda, d=d, max_depth=N - d)
    for D in range(0, N - d + 1):
        res_a = aligned_sweep[D]
        res_c = coord_sweep[D]
        # Verify every elementary gate has leakage rank <= 1 and commutator rank <= 2
        for gate in res_a.gates:
            assert gate.leakage_rank <= 1
            assert gate.commutator_rank <= 2
            assert numerical_rank(gate.P_Ck) == 1
            assert torch.linalg.norm(P_Lambda @ gate.P_Ck).item() < ASSERT_TOL

        if D < target_r:
            assert not res_a.exact_isometric_success
            assert not res_a.exact_compression_success
            assert not res_c.exact_compression_success
            assert res_a.forward_leakage_rank == D < target_r
            assert res_a.commutator_rank == 2 * D < 2 * target_r
        else:
            assert res_a.exact_isometric_success
            assert res_a.exact_compression_success
            assert res_c.exact_compression_success
            assert res_a.forward_leakage_rank == target_r
            assert res_a.commutator_rank == 2 * target_r


def test_adversarial_subcritical_depth_falsification() -> None:
    N, d = 10, 4
    P_Lambda, _ = build_bandlimit_projector(N=N, d=d)
    U_nl = generate_nonlinear_unitary(N=N, d=d, method="qr", seed=42)
    adv = run_adversarial_falsification_check(U_nl, P_Lambda, d=d, num_random_trials=100)
    assert adv["max_observed_comm_rank"] <= 2 * (adv["target_rank_r"] - 1)
    assert adv["min_observed_iso_err"] >= adv["eckart_young_lower_bound"] - 1e-12
    assert adv["eckart_young_lower_bound"] > 0.0


def test_lean4_formalization_complete_no_sorrys() -> None:
    from pathlib import Path
    import re
    import subprocess

    lean_path = Path(__file__).resolve().parent.parent / "UnitaryBandwidthDepth.lean"
    assert lean_path.exists()
    content = lean_path.read_text(encoding="utf-8")

    # Strip comments before checking for sorry
    no_block_comments = re.sub(r"/-.*?-/", "", content, flags=re.DOTALL)
    no_comments = re.sub(r"--.*$", "", no_block_comments, flags=re.MULTILINE)
    assert "sorry" not in no_comments

    required_identifiers = [
        "IsElementaryClockGate",
        "rank_forwardLeakage_le_one_of_isElementaryClockGate",
        "forwardLeakage_rank_eq_reverseLeakage_rank",
        "commutator_rank_eq_two_mul_forwardLeakage_rank",
        "forwardLeakage_rank_eq_of_compression_eq",
        "unitary_bandwidth_depth_tradeoff_compression",
        "unitary_bandwidth_depth_tradeoff_clock_wires",
    ]
    for ident in required_identifiers:
        assert ident in no_comments, f"Missing Lean 4 identifier: {ident}"

    lake_bin = Path("/home/tryggth2009/.elan/bin/lake")
    if lake_bin.exists():
        proc = subprocess.run(
            [str(lake_bin), "env", "lean", "UnitaryBandwidthDepth.lean"],
            cwd=str(lean_path.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Lean compile failed:\n{proc.stdout}\n{proc.stderr}"
        assert proc.stdout.strip() == "", f"Unexpected Lean warnings:\n{proc.stdout}"


@pytest.mark.parametrize(
    "lean_filename,required_identifiers",
    [
        (
            "NyquistRotorGates.lean",
            [
                "givensRotor_mem_unitaryGroup",
                "givensRotor_isElementaryClockGate",
                "givensRotor_in_phase_mul",
                "givensRotor_in_phase_double",
                "givensRotor_anti_phase_mul_eq_one",
                "givensRotor_ramsey_fringe_entry",
                "pauli_saturation_reflection",
                "pauli_exclusion_blockade_ker_finrank",
                "stage1UVStabilizer_mem_unitaryGroup",
                "stage2MeanReadback_preserves_uv_mode",
                "stage2MeanReadback_mem_unitaryGroup",
                "milestone4TripartiteRouter_mem_unitaryGroup",
                "spatial_compression_add_clock_leakage_eq_proj",
            ],
        ),
        (
            "NyquistHiggsDuality.lean",
            [
                "complete_square",
                "pos_𝓵_lower_bound",
                "minimum_iff_normSq_eq",
                "saturation_linear_relaxation",
                "subNyquist_unique_minimum_zero",
                "superNyquist_condensate_minimum",
                "superNyquist_horizon_restoration",
                "quarticCoupling_cutoff_independent",
            ],
        ),
        (
            "NyquistGaugeTransport.lean",
            [
                "cayleyTransform_mul_conjTranspose_self",
                "cayleyTransform_mem_unitaryGroup",
                "splitFormAdvection_skewHermitian",
                "covariantClockAdvection_skewHermitian",
                "twoPathHolonomy_gauge_invariant",
                "wilsonLoop3_gauge_invariant",
                "covariantClockAdvection_gauge_equivariant",
            ],
        ),
    ],
)
def test_lean4_phase_a_physics_extensions_no_sorrys(
    lean_filename: str, required_identifiers: list[str]
) -> None:
    from pathlib import Path
    import re
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    lean_path = repo_root / lean_filename
    assert lean_path.exists(), f"Missing Lean 4 file: {lean_filename}"
    content = lean_path.read_text(encoding="utf-8")

    no_block_comments = re.sub(r"/-.*?-/", "", content, flags=re.DOTALL)
    no_comments = re.sub(r"--.*$", "", no_block_comments, flags=re.MULTILINE)
    for forbidden in ("sorry", "admit", "axiom"):
        assert not re.search(rf"\b{forbidden}\b", no_comments), (
            f"Found '{forbidden}' in {lean_filename}"
        )

    for ident in required_identifiers:
        assert ident in no_comments, f"Missing Lean 4 identifier '{ident}' in {lean_filename}"

    lake_bin = Path("/home/tryggth2009/.elan/bin/lake")
    if lake_bin.exists():
        proc = subprocess.run(
            [str(lake_bin), "env", "lean", lean_filename],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Lean compile failed for {lean_filename}:\n{proc.stdout}\n{proc.stderr}"
        assert proc.stdout.strip() == "", f"Unexpected Lean warnings in {lean_filename}:\n{proc.stdout}"


@pytest.mark.parametrize(
    "lean_filename,olean_name,required_identifiers",
    [
        (
            "PhyslibHiggsBridge.lean",
            "PhyslibHiggsBridge.olean",
            [
                "Physlib.Particles.StandardModel.HiggsBoson.Potential",
                "toPhyslibPotential",
                "toPhyslibPotential_toFun_eq_nyquistEffectivePotential",
                "subNyquist_physlib_isMinOn_iff_zero",
                "superNyquist_physlib_isMinOn_iff_condensate",
                "superNyquist_physlib_horizon_restoration",
            ],
        ),
        (
            "QuantumInfoClockBridge.lean",
            "QuantumInfoClockBridge.olean",
            [
                "QuantumInfo.Channels.CPTP",
                "QuantumInfo.Entropy.VonNeumann",
                "bipartiteUnitaryChannel",
                "bipartiteUnitaryChannel_isUnitary",
                "givensRotorCPTP",
                "givensRotorCPTP_isUnitary",
                "milestone4TripartiteCPTP",
                "milestone4TripartiteCPTP_isUnitary",
                "reducedSpatialFluidChannel",
                "reducedSpatialFluidChannel_tracePreserving",
                "spatialChannel_admits_unitary_purification",
                "bipartite_pure_global_entropy_zero",
                "spatial_shock_entropy_eq_clock_entropy",
                "bipartite_pure_mutual_info_eq_two_mul_spatial_entropy",
            ],
        ),
    ],
)
def test_lean4_phase_b_upstream_physlib_bridges(
    lean_filename: str, olean_name: str, required_identifiers: list[str]
) -> None:
    from pathlib import Path
    import re

    repo_root = Path(__file__).resolve().parent.parent
    lean_path = repo_root / lean_filename
    assert lean_path.exists(), f"Missing Phase B Lean 4 file: {lean_filename}"
    content = lean_path.read_text(encoding="utf-8")

    no_block_comments = re.sub(r"/-.*?-/", "", content, flags=re.DOTALL)
    no_comments = re.sub(r"--.*$", "", no_block_comments, flags=re.MULTILINE)
    for forbidden in ("sorry", "admit", "axiom"):
        assert not re.search(rf"\b{forbidden}\b", no_comments), (
            f"Found '{forbidden}' in {lean_filename}"
        )

    for ident in required_identifiers:
        assert ident in no_comments, f"Missing Lean 4 identifier '{ident}' in {lean_filename}"

    olean_path = (
        repo_root / ".worktrees" / "physlib-test" / ".lake" / "build" / "lib" / "lean" / olean_name
    )
    if olean_path.exists():
        header = olean_path.read_bytes()[:5]
        assert header == b"olean", f"Invalid .olean header for {olean_name}: {header!r}"
