#!/usr/bin/env python3
"""
Unit, Falsification, and Anti-Specification-Gaming Tests for Milestone 4:
Two-Shock Collision, Coherent Clock-Readback Gate & Ramsey Mass-Annihilation Interferometer
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch

from toy_model_2_nyquist_rotor_pde import (
    evaluate_lax_commutator_and_saturation,
    make_periodic_grid,
)
import toy_model_4_shock_collision_gate as tm4
from toy_model_4_shock_collision_gate import (
    TwoShockCollisionResult,
    build_two_shock_collision_ic,
    execute_bidirectional_givens_collision,
    locate_compressive_shocks,
    run_milestone_4_collision_experiments,
    verify_two_pulse_clock_readback_gate,
)


@pytest.fixture(scope="module")
def milestone_4_results(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the full Milestone 4 experimental suites once per test module."""
    out_dir = tmp_path_factory.mktemp("milestone_4_outputs")
    return run_milestone_4_collision_experiments(lambda_bek=64, save_dir=out_dir)


def test_no_oracle_time_or_position_parameters() -> None:
    """
    Guardrail 1: Verify that no function evaluating Lax commutator defect, shock localization,
    or bidirectional clock-rotor coupling accepts oracle collision time or position parameters
    (t_current, t_coll, x_coll, t_breaking, t_star).
    """
    forbidden_params = {"t_current", "t_coll", "x_coll", "t_breaking", "t_star", "step", "time"}
    for fn in (
        evaluate_lax_commutator_and_saturation,
        locate_compressive_shocks,
        execute_bidirectional_givens_collision,
    ):
        sig_params = set(inspect.signature(fn).parameters.keys())
        overlap = sig_params.intersection(forbidden_params)
        assert not overlap, f"Function {fn.__name__} accepts forbidden oracle parameters: {overlap}"


def test_no_post_step_norm_clamping_or_mag_bin_in_source() -> None:
    """
    Guardrail 2 & 4: Verify that toy_model_4_shock_collision_gate.py contains no post-step
    scalar norm rescaling, clamping hacks, or non-local amplitude-binning (mag_bin) logic.
    """
    src = inspect.getsource(tm4)
    forbidden_snippets = [
        "E_0 - E_clock",
        "u = u * scale",
        "u *= scale",
        "torch.sqrt((initial_total_norm_sq",
        "mag_bin",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in src, f"Forbidden snippet found in source: {snippet}"


def test_suite_a_two_pulse_clock_readback_gate() -> None:
    """
    Suite A: Verify the microscopic 2-pulse gate truth table:
      - G_pi(vartheta)^m G_0(vartheta)^m = I_2 (coherent negation / readback to < 10^-15)
      - G_0(vartheta)^m G_0(vartheta)^m = G_0(2*m*vartheta) (constructive angle doubling to < 10^-15)
      - Lattice 2-site Haar + SU(2) Givens 2-pulse inversion and mean-mode readback to < 10^-15.
    """
    metrics = verify_two_pulse_clock_readback_gate(vartheta=0.12435499454676144, m_pulses=8)
    assert metrics["anti_phase_u_error"] < 1e-15
    assert metrics["anti_phase_c_error"] < 1e-15
    assert metrics["anti_phase_norm_drift"] < 1e-14
    assert metrics["in_phase_u_error"] < 1e-15
    assert metrics["in_phase_c_error"] < 1e-15
    assert metrics["in_phase_norm_drift"] < 1e-14
    assert metrics["lattice_haar_u_error"] < 1e-15
    assert metrics["lattice_haar_c_error"] < 1e-15
    assert metrics["haar_unitarity_drift"] < 1e-15


def test_two_shock_initial_condition_and_localization() -> None:
    """
    Verify that build_two_shock_collision_ic initializes two compressive shocks around
    x_{1,0} = -1.0 and x_{2,0} = -0.5 on [-pi, pi) with Lambda_Bek = 64, and that
    locate_compressive_shocks accurately recovers their initial positions purely from u_0(x).
    """
    x_grid, _, dx = make_periodic_grid(64)
    u0 = build_two_shock_collision_ic(x_grid)
    x1, x2, sep = locate_compressive_shocks(u0, x_grid)

    assert abs(x1 - (-1.0)) < dx
    assert abs(x2 - (-0.5)) < dx
    assert abs(sep - 0.5) < dx


def test_suite_b_unclamped_unitarity_and_one_to_one_accounting(milestone_4_results: dict) -> None:
    """
    Suite B Assertion 1 & Guardrail 3: Verify that both Delta phi = 0 and Delta phi = pi
    conserve total Hilbert-space norm ||u||_2^2 + sum |c|^2 to < 10^-13 relative drift and
    satisfy exact 1-to-1 gate accounting (total_physical_givens_rotations == sum_t sum_j m_j(t)).
    """
    res_0: TwoShockCollisionResult = milestone_4_results["in_phase"]
    res_pi: TwoShockCollisionResult = milestone_4_results["anti_phase"]

    assert res_0.max_norm_drift < 1e-13
    assert res_pi.max_norm_drift < 1e-13

    sum_mj_0 = sum(int(torch.sum(m).item()) for m in res_0.substep_counts_history)
    sum_mj_pi = sum(int(torch.sum(m).item()) for m in res_pi.substep_counts_history)

    assert res_0.total_physical_givens_rotations == sum_mj_0 == res_0.total_requested_substeps
    assert res_pi.total_physical_givens_rotations == sum_mj_pi == res_pi.total_requested_substeps


def test_suite_b_clock_mass_annihilation_and_spatial_recovery(milestone_4_results: dict) -> None:
    """
    Suite B Assertion 2: Verify that in the anti-phase collision (Delta phi = pi), coherent
    readback into the smooth Haar mean mode discharges >80% of the stored clock-mass condensate:
      M_clock(Delta phi = pi) < 0.20 * M_clock(Delta phi = 0) (< 0.55 required)
    and post-collision spatial flow energy satisfies
      ||u(Delta phi = pi)||_2^2 > ||u(Delta phi = 0)||_2^2.
    """
    res_0: TwoShockCollisionResult = milestone_4_results["in_phase"]
    res_pi: TwoShockCollisionResult = milestone_4_results["anti_phase"]

    m_clock_0 = res_0.clock_norm_sq_history[-1]
    m_clock_pi = res_pi.clock_norm_sq_history[-1]
    e_sp_0 = res_0.spatial_norm_sq_history[-1]
    e_sp_pi = res_pi.spatial_norm_sq_history[-1]

    assert m_clock_pi < 0.20 * m_clock_0
    assert e_sp_pi > e_sp_0


def test_suite_b_rankine_hugoniot_coalescence_speed(milestone_4_results: dict) -> None:
    """
    Suite B Assertion 3: Verify that the merged shock propagates within 10% of the analytical
    coalesced Rankine-Hugoniot speed s_merged = (u_L + u_R)/2 = 0.70.
    """
    res_0: TwoShockCollisionResult = milestone_4_results["in_phase"]
    res_pi: TwoShockCollisionResult = milestone_4_results["anti_phase"]

    assert abs(res_0.measured_merged_speed - 0.70) / 0.70 < 0.10
    assert abs(res_pi.measured_merged_speed - 0.70) / 0.70 < 0.10


def test_suite_b_no_uv_spike_avalanche_or_substep_runaway(milestone_4_results: dict) -> None:
    """
    Suite B Assertion 4 (Anti-Spike & Smoothness):
      - Maximum post-collision anti-phase amplitude satisfies max_x |u_{Delta phi = pi}(x, T)| < 1.35.
      - Total Variation satisfies TV(u_{Delta phi = pi}(T)) < 1.35 * TV(u_{Delta phi = 0}(T)).
      - Total sub-step count sum_{t,j} m_j(t) for Delta phi = pi stays within 1.5x of Delta phi = 0.
    """
    res_0: TwoShockCollisionResult = milestone_4_results["in_phase"]
    res_pi: TwoShockCollisionResult = milestone_4_results["anti_phase"]

    assert res_pi.final_max_abs_u < 1.35
    assert res_pi.final_total_variation < 1.35 * res_0.final_total_variation
    assert res_pi.total_requested_substeps <= 1.50 * res_0.total_requested_substeps


def test_suite_c_ramsey_interference_fringe_smooth_bell_shape(milestone_4_results: dict) -> None:
    """
    Suite C: Verify that across Delta phi in {0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4, 2*pi},
    M_clock(Delta phi) and ||u(T, Delta phi)||_2^2 trace a smooth, bell-shaped / sinusoidal
    interference fringe peaking at Delta phi in {0, 2*pi} and reaching a smooth minimum at
    Delta phi = pi without a singular 1-point V-notch.
    """
    masses = milestone_4_results["sweep_clock_masses"]
    energies = milestone_4_results["sweep_spatial_energies"]
    assert len(masses) == 9
    assert len(energies) == 9

    # Strictly decreasing clock mass (increasing spatial energy) on [0, pi]
    for k in range(4):
        assert masses[k] > masses[k + 1]
        assert energies[k] < energies[k + 1]

    # Strictly increasing clock mass (decreasing spatial energy) on [pi, 2*pi]
    for k in range(4, 8):
        assert masses[k] < masses[k + 1]
        assert energies[k] > energies[k + 1]

    # Exact 2*pi periodicity and reflection symmetry around pi
    for k in range(4):
        assert abs(masses[k] - masses[8 - k]) < 1e-12
        assert abs(energies[k] - energies[8 - k]) < 1e-12

    # Smooth sinusoidal / bell-shaped trough around pi (no singular 1-point V-notch at pi):
    # The drop from 3*pi/4 -> pi is smaller than the steepest mid-fringe drop from pi/4 -> pi/2
    assert (masses[3] - masses[4]) < (masses[1] - masses[2])

    fig_path: Path = milestone_4_results["figure_path"]
    assert fig_path.exists() and fig_path.stat().st_size > 10000
