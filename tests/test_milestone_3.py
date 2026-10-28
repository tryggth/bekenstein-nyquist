"""
Unit and falsification tests for Milestone 3:
Causal DAG Critical-Path Scaling (`toy_model_3_dag_complexity.py`).
"""

from pathlib import Path
import inspect
import pytest
import torch

import toy_model_3_dag_complexity as m3_mod
from toy_model_2_nyquist_rotor_pde import evaluate_lax_commutator_and_saturation
from toy_model_3_dag_complexity import (
    SpacetimeCausalDAG,
    build_multi_shock_gas_initial_condition,
    build_single_shock_initial_condition,
    run_milestone_3_complexity_sweep,
    simulate_dag_complexity_trajectory,
)


def test_anti_cheat_unified_first_principles_flux_evaluator() -> None:
    """
    Verify Global Anti-Specification-Gaming Guardrails in Milestone 3:
    `evaluate_macro_lax_commutator_flux` must not exist; Milestone 3 must use the unified
    state-only `evaluate_lax_commutator_and_saturation(u, config)` with zero oracle time parameters.
    """
    assert not hasattr(m3_mod, "evaluate_macro_lax_commutator_flux"), (
        "Shortcut evaluate_macro_lax_commutator_flux must be removed from toy_model_3_dag_complexity.py"
    )
    sig = inspect.signature(evaluate_lax_commutator_and_saturation)
    assert list(sig.parameters.keys()) == ["u", "config"]


def test_spacetime_causal_dag_explicit_vertices_and_edges() -> None:
    dag = SpacetimeCausalDAG(n_x=8, record_explicit_edges=True)
    assert dag.num_vertices == 8
    assert dag.num_edges == 0

    # Step 1: pure sub-Nyquist parallel step (m_j = 0 everywhere)
    m_zero = torch.zeros(8, dtype=torch.long)
    depth_1, work_1, edges_1 = dag.advance_macro_step(m_zero)
    assert depth_1 == 1
    assert work_1 == 16
    assert edges_1 == 24
    assert len(dag.edges) == 24

    # Step 2: super-Nyquist shock core at site j=3 with m_3 = 5 sequential Givens rotations
    m_shock = torch.zeros(8, dtype=torch.long)
    m_shock[3] = 5
    depth_2, work_2, edges_2 = dag.advance_macro_step(m_shock)
    assert depth_2 == 2 + 5 == 7
    assert work_2 == 16 + 8 + 5 == 29
    assert edges_2 == 24 + 24 + 5 == 53


def test_analytical_breaking_times() -> None:
    _, _, t_star_sine = build_single_shock_initial_condition(64)
    assert abs(t_star_sine - 1.0) < 1e-4

    _, _, t_star_gas = build_multi_shock_gas_initial_condition(64, target_t_star=0.80)
    assert abs(t_star_gas - 0.80) < 1e-12


def test_phase_transition_and_lambda_scaling(tmp_path: Path) -> None:
    results = run_milestone_3_complexity_sweep(
        cutoffs=(64, 128, 256), t_final=1.40, dt=0.02, save_dir=tmp_path
    )
    assert (tmp_path / "milestone_3_dag_scaling.png").exists()

    for ic_key in ("single_shock", "multi_shock_gas"):
        m64 = results[ic_key][64]
        m128 = results[ic_key][128]
        m256 = results[ic_key][256]

        # 1. Pre-catastrophe O(1) depth per step independent of Lambda_Bek
        assert abs(m64.elliptic_slope_per_step - 1.0) < 1e-12
        assert abs(m128.elliptic_slope_per_step - 1.0) < 1e-12
        assert abs(m256.elliptic_slope_per_step - 1.0) < 1e-12

        # 2. Post-catastrophe Omega((T - t*) * Lambda_Bek) linear scaling
        n_steps = len(m64.times) - 1
        ex64 = m64.dag_depth_history[-1] - n_steps
        ex128 = m128.dag_depth_history[-1] - n_steps
        ex256 = m256.dag_depth_history[-1] - n_steps
        assert ex64 > 0
        assert 1.6 <= (ex128 / ex64) <= 2.4
        assert 1.6 <= (ex256 / ex128) <= 2.4

        # 3. Strict 1-to-1 accounting between physical Givens rotations and DAG rotor vertices
        for m in (m64, m128, m256):
            expected_substep_sum = sum(m.substep_sum_history)
            dag_rotor_vertices = m.total_work_history[-1] - len(m.times) * m.n_x
            assert m.total_physical_givens_rotations == expected_substep_sum == dag_rotor_vertices
            assert m.total_physical_givens_rotations > 0


def test_strict_1to1_physical_givens_rotations_vs_dag_vertices() -> None:
    """
    Verify directly on `simulate_dag_complexity_trajectory` across Lambda_Bek in {64, 128}
    (where n_sub = 1 and n_sub > 1 respectively) that:
      total_physical_givens_rotations == sum_t sum_j m_j(t) == |V(T)| - (N_steps + 1) * N_x.
    """
    for lb in (64, 128):
        _, u0, t_star = build_single_shock_initial_condition(lb)
        m = simulate_dag_complexity_trajectory(
            ic_name="Single-Shock 1-to-1 Check",
            u0=u0,
            t_star=t_star,
            lambda_bek=lb,
            t_final=1.25,
            dt=0.02,
        )
        n_steps = len(m.times) - 1
        expected_rotations = sum(m.substep_sum_history)
        dag_rotor_vertices = m.total_work_history[-1] - (n_steps + 1) * m.n_x
        assert m.total_physical_givens_rotations > 0
        assert m.total_physical_givens_rotations == expected_rotations == dag_rotor_vertices
