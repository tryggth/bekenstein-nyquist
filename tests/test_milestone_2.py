"""
Unit and falsification tests for Milestone 2:
The 1+1D Unitary Nyquist Rotor Lattice (`toy_model_2_nyquist_rotor_pde.py`).
"""

from pathlib import Path
import inspect
import pytest
import torch

from toy_model_2_nyquist_rotor_pde import (
    COMPLEX_DTYPE,
    FLOAT_DTYPE,
    RotorLatticeConfig,
    compute_effective_transport_velocity,
    evaluate_lax_commutator_and_saturation,
    execute_sequential_givens_rotations,
    initial_condition_riemann_step,
    initial_condition_sine,
    make_periodic_grid,
    run_milestone_2_benchmarks,
    run_unitary_nyquist_rotor,
    verify_pi_rotor_destructive_interference,
)


def test_anti_cheat_no_oracle_time_arguments_in_flux_evaluator() -> None:
    """
    Verify Global Anti-Specification-Gaming Guardrail 1 & 4:
    `evaluate_lax_commutator_and_saturation` and `run_unitary_nyquist_rotor` must NOT accept
    `t_current`, `t_breaking`, or `t_star`, and passing an identical spatial field u(x)
    must always return identical (F_j, Sigma_j, m_j).
    """
    sig_eval = inspect.signature(evaluate_lax_commutator_and_saturation)
    forbidden = {"t_current", "t_breaking", "t_star"}
    assert forbidden.isdisjoint(sig_eval.parameters.keys()), (
        f"Forbidden oracle time parameters found in evaluate_lax_commutator_and_saturation: {sig_eval.parameters.keys()}"
    )
    assert list(sig_eval.parameters.keys()) == ["u", "config"]

    sig_runner = inspect.signature(run_unitary_nyquist_rotor)
    assert forbidden.isdisjoint(sig_runner.parameters.keys()), (
        f"Forbidden oracle time parameters found in run_unitary_nyquist_rotor: {sig_runner.parameters.keys()}"
    )

    cfg = RotorLatticeConfig(lambda_bek=64)
    x, _, _ = make_periodic_grid(cfg.lambda_bek)
    u_test = initial_condition_riemann_step(x)
    F1, S1, m1, d1 = evaluate_lax_commutator_and_saturation(u_test, cfg)
    F2, S2, m2, d2 = evaluate_lax_commutator_and_saturation(u_test.clone(), cfg)
    assert torch.equal(F1, F2)
    assert torch.equal(S1, S2)
    assert torch.equal(m1, m2)
    assert torch.equal(d1, d2)


def test_lattice_and_impedance_angle_calibration() -> None:
    cfg = RotorLatticeConfig(lambda_bek=64, nu_ml_intensive=1.0)
    assert cfg.n_x == 128
    assert abs(cfg.dx - float(torch.pi / 64.0)) < 1e-15
    assert abs(cfg.theta_nyq - 1.0) < 1e-15
    assert abs(cfg.dtau_ml - float(torch.pi / cfg.nu_ml)) < 1e-12
    # Impedance matching: tan(vartheta) = sqrt(Delta k / Lambda_Bek) with Delta k = 1
    tan_theta = float(torch.tan(torch.tensor(cfg.impedance_angle, dtype=FLOAT_DTYPE)).item())
    expected_tan = float(torch.sqrt(torch.tensor(1.0 / 64.0, dtype=FLOAT_DTYPE)).item())
    assert abs(tan_theta - expected_tan) < 1e-15


def test_sub_nyquist_vs_super_nyquist_trigger_evaluation() -> None:
    cfg = RotorLatticeConfig(lambda_bek=64)
    x, _, _ = make_periodic_grid(cfg.lambda_bek)
    u_sine = initial_condition_sine(x)

    # For smooth sub-Nyquist sine wave, F_j <= theta_Nyq, Sigma_j > 0, m_j = 0 everywhere
    F_sub, Sigma_sub, m_sub, _ = evaluate_lax_commutator_and_saturation(u_sine, cfg)
    assert torch.all(F_sub <= cfg.theta_nyq)
    assert torch.all(Sigma_sub > 0.0)
    assert int(torch.sum(m_sub).item()) == 0

    # For a steep Riemann shock, F_j > theta_Nyq and Sigma_j < 0 at the shock core
    u_riemann = initial_condition_riemann_step(x, u_left=1.0, u_right=0.2)
    F_sup, Sigma_sup, m_sup, _ = evaluate_lax_commutator_and_saturation(u_riemann, cfg)
    assert torch.any(F_sup > cfg.theta_nyq)
    assert torch.any(Sigma_sup < 0.0)
    assert int(torch.sum(m_sup).item()) > 0


def test_galilean_invariant_effective_transport_velocity() -> None:
    cfg = RotorLatticeConfig(lambda_bek=64, velocity_delay_gamma=1.0e-5)
    x, _, _ = make_periodic_grid(cfg.lambda_bek)
    u = initial_condition_riemann_step(x)
    _, _, m_j, _ = evaluate_lax_commutator_and_saturation(u, cfg)
    w_eff = compute_effective_transport_velocity(u, m_j, cfg)
    # Shifting u -> u + C must shift w_eff -> w_eff + (2/3) C (strict Galilean invariance)
    c_shift = 1.75
    w_eff_shifted = compute_effective_transport_velocity(u + c_shift, m_j, cfg)
    assert torch.allclose(w_eff_shifted - w_eff, torch.full_like(w_eff, (2.0 / 3.0) * c_shift), atol=1e-14)


def test_pi_rotor_destructive_interference_verification() -> None:
    outcomes = verify_pi_rotor_destructive_interference()
    assert outcomes["phi_0_max_drift"] < 1e-13
    assert outcomes["phi_pi_max_drift"] < 1e-13
    assert outcomes["phi_pi_tv"] > 1.25 * outcomes["phi_0_tv"]


def test_single_step_givens_unitarity_to_machine_precision() -> None:
    cfg = RotorLatticeConfig(lambda_bek=64)
    x, _, _ = make_periodic_grid(cfg.lambda_bek)
    u = initial_condition_riemann_step(x).to(COMPLEX_DTYPE)
    clock_registers = torch.zeros((cfg.n_x, 128), dtype=COMPLEX_DTYPE)
    clock_pointers = torch.zeros(cfg.n_x, dtype=torch.long)
    m_j = torch.zeros(cfg.n_x, dtype=torch.long)
    m_j[40] = 12
    m_j[41] = 8

    norm_before = float((torch.sum(torch.abs(u) ** 2) + torch.sum(torch.abs(clock_registers) ** 2)).item())
    u_after, clock_after, _ = execute_sequential_givens_rotations(
        u.clone(), clock_registers, clock_pointers, m_j, vartheta=cfg.impedance_angle
    )
    norm_after = float((torch.sum(torch.abs(u_after) ** 2) + torch.sum(torch.abs(clock_after) ** 2)).item())
    assert abs(norm_after - norm_before) / norm_before < 1e-15
    assert float(torch.sum(torch.abs(clock_after) ** 2).item()) > 0.0


def test_full_milestone_2_comparative_benchmarks(tmp_path: Path) -> None:
    results = run_milestone_2_benchmarks(lambda_bek=64, save_dir=tmp_path)
    assert (tmp_path / "milestone_2_benchmarks.png").exists()

    for key in ("riemann", "sine"):
        rotor = results[key]["rotor"]
        godunov = results[key]["godunov"]
        spectral = results[key]["spectral"]

        # Unitary norm conservation to ~10^-16
        assert rotor.max_norm_drift < 1e-14
        # Falsification of dissipative Godunov
        assert godunov.max_norm_drift > 1e-2
        # Falsification of naive spectral Gibbs ringing
        assert spectral.total_variation_history[-1] > 2.0 * rotor.total_variation_history[-1]
        assert rotor.gibbs_overshoot < 0.25 * spectral.gibbs_overshoot
