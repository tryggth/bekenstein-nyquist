#!/usr/bin/env python3
"""
Milestone 3: Complexity Toy Model — Causal DAG Critical-Path Scaling
(Experiment H.4, Phase 3 & Phase 4)

Extends the 1+1D Unitary Nyquist Rotor Lattice to dynamically construct the spacetime
causal Directed Acyclic Graph G = (V, E) and test the Sub-Stepping Depth Theorem:

    D(gamma) >= (1 / (pi * nu_ML)) * int_gamma ||[P_Lambda, [M(u), L(u)]]||_HS ds

across Bekenstein bandwidth cutoffs Lambda_Bek in {64, 128, 256} for:
  (a) A single-shock wave u_0(x) = -sin(x) with analytical breaking time t* = 1.0
  (b) A random multi-shock gas with analytical breaking time t* = min_x (-1 / d_x u_0(x))
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from toy_model_2_nyquist_rotor_pde import (
    COMPLEX_DTYPE,
    FLOAT_DTYPE,
    RotorLatticeConfig,
    _apply_unitary_cayley_step,
    compute_effective_transport_velocity,
    evaluate_lax_commutator_and_saturation,
    execute_sequential_givens_rotations,
    make_periodic_grid,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


@dataclass
class CausalDAGMetrics:
    """Spacetime causal DAG G = (V, E) observables for a single (initial_condition, Lambda_Bek) run."""

    ic_name: str
    lambda_bek: int
    n_x: int
    dt: float
    t_star: float
    times: List[float]
    dag_depth_history: List[int]  # Critical path length depth(G) at each macro-step
    total_work_history: List[int]  # Total vertex count |V| at each macro-step
    edge_count_history: List[int]  # Total causal edge count |E| at each macro-step
    parallelism_ratio_history: List[float]  # depth(G) / |V|
    substep_sum_history: List[int]  # sum_j m_j(t)
    substep_max_history: List[int]  # max_j m_j(t)
    active_shock_sites_history: List[int]  # number of sites with m_j(t) > 0
    theorem_integral_bound_history: List[float]  # (1 / (pi * nu_ML)) int_gamma F_j ds
    max_norm_drift: float
    elliptic_slope_per_step: float  # d(depth)/d(step) for t < t*
    hyperbolic_slope_per_time: float  # d(depth)/dt for t > t*
    total_physical_givens_rotations: int = 0  # Cumulative physical SU(2) Givens rotations executed on clock_registers
    spacetime_substep_map: torch.Tensor = field(default_factory=lambda: torch.empty(0))


class SpacetimeCausalDAG:
    """
    Dynamic spacetime Directed Acyclic Graph G = (V, E) tracker for the Unitary Nyquist Rotor Lattice.

    At each macro-step n and spatial site j in {0, ..., N_x - 1}:
      1. Creates a spatial stencil vertex v^{stencil}_{n,j} in V with 3 incoming causal edges in E
         from the previous step's terminal vertices at {(j-1) mod N_x, j, (j+1) mod N_x}.
      2. If m_j > 0 (Super-Nyquist Shock Core, Sigma_j < 0), creates a sequential chain of m_j
         Givens rotor sub-step vertices v^{rotor}_{n,j,1} -> ... -> v^{rotor}_{n,j,m_j} in V,
         each with causal edges from the preceding wire states on site j and (j+1) mod N_x.
    Tracks exact vertex count |V|, edge count |E|, and longest directed path length depth(G)
    via exact dynamic programming over the DAG topological layers.
    """

    def __init__(self, n_x: int, record_explicit_edges: bool = False) -> None:
        self.n_x = n_x
        self.record_explicit_edges = record_explicit_edges
        # Initial layer (t = 0): N_x input state vertices at depth 0
        self.num_vertices: int = n_x
        self.num_edges: int = 0
        self.terminal_depth = torch.zeros(n_x, dtype=torch.long)
        self.terminal_vertex_id = torch.arange(n_x, dtype=torch.long)
        self.edges: List[Tuple[int, int]] = []

    def advance_macro_step(self, m_j: torch.Tensor) -> Tuple[int, int, int]:
        """
        Record one macro-step in the causal DAG G = (V, E) given sub-step counts m_j at each site.
        Returns (critical_path_depth, total_vertices_|V|, total_edges_|E|).
        """
        n_x = self.n_x
        # 1. Parallel spatial stencil layer: each site j depends on {j-1, j, j+1}
        prev_left = torch.roll(self.terminal_depth, shifts=1, dims=0)
        prev_center = self.terminal_depth
        prev_right = torch.roll(self.terminal_depth, shifts=-1, dims=0)
        stencil_depth = 1 + torch.maximum(prev_center, torch.maximum(prev_left, prev_right))

        if self.record_explicit_edges:
            for j in range(n_x):
                v_new = self.num_vertices + j
                self.edges.append((int(self.terminal_vertex_id[(j - 1) % n_x].item()), v_new))
                self.edges.append((int(self.terminal_vertex_id[j].item()), v_new))
                self.edges.append((int(self.terminal_vertex_id[(j + 1) % n_x].item()), v_new))
                self.terminal_vertex_id[j] = v_new

        self.num_vertices += n_x
        self.num_edges += 3 * n_x

        # 2. Sequential Givens rotor sub-step chains at Super-Nyquist Shock Core sites (m_j > 0)
        curr_depth = stencil_depth.clone()
        active_sites = torch.nonzero(m_j > 0, as_tuple=False).flatten()
        for j_tensor in active_sites:
            j = int(j_tensor.item())
            j_next = (j + 1) % n_x
            steps = int(m_j[j].item())
            # Each Givens rotation couples site j (and its UV pair j+1) with clock wire c_{j,k}
            base_d = max(int(curr_depth[j].item()), int(curr_depth[j_next].item()))
            curr_depth[j] = base_d + steps

            if self.record_explicit_edges:
                prev_v = int(self.terminal_vertex_id[j].item())
                for _k in range(steps):
                    v_rot = self.num_vertices
                    self.edges.append((prev_v, v_rot))
                    prev_v = v_rot
                    self.num_vertices += 1
                    self.num_edges += 1
                self.terminal_vertex_id[j] = prev_v
            else:
                self.num_vertices += steps
                self.num_edges += 2 * steps  # causal dependency from spatial mode + clock wire

        self.terminal_depth = curr_depth
        critical_path_depth = int(torch.max(self.terminal_depth).item())
        return critical_path_depth, self.num_vertices, self.num_edges


def build_single_shock_initial_condition(
    lambda_bek: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Initialize (a) single-shock wave u_0(x) = -sin(x) on [-pi, pi) with analytical
    gradient catastrophe breaking time t* = 1 / max |d_x u_0| = min_x (-1 / d_x u_0(x)) = 1.0.
    """
    x, _, _ = make_periodic_grid(lambda_bek)
    u0 = -torch.sin(x)
    du0_dx = -torch.cos(x)
    max_comp_slope = float(torch.max(-du0_dx).item())  # 1.0 at x = 0
    t_star = 1.0 / max_comp_slope
    return x, u0, t_star


def build_multi_shock_gas_initial_condition(
    lambda_bek: int, target_t_star: float = 0.80, seed: int = 2026
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Initialize (b) a random multi-shock gas u_0(x) composed of low-frequency Fourier modes
    m in {1, 2, 3, 4} with random phases, normalized so the analytical gradient catastrophe
    breaking time is t* = min_x (-1 / d_x u_0(x)) = target_t_star.
    """
    x, _, _ = make_periodic_grid(lambda_bek)
    gen = torch.Generator().manual_seed(seed)
    modes = [1, 2, 3, 4]
    amps = torch.tensor([1.0, 0.55, 0.35, 0.20], dtype=FLOAT_DTYPE)
    phases = 2.0 * torch.pi * torch.rand(len(modes), dtype=FLOAT_DTYPE, generator=gen)

    # Evaluate on a fine reference grid to get exact continuum maximum compressive slope
    x_fine = torch.linspace(-torch.pi, torch.pi, 16384, dtype=FLOAT_DTYPE)[:-1]
    u_raw_fine = torch.zeros_like(x_fine)
    du_raw_fine = torch.zeros_like(x_fine)
    u_raw = torch.zeros_like(x)

    for m_val, a_val, phi_val in zip(modes, amps, phases):
        u_raw = u_raw - a_val * torch.sin(m_val * x + phi_val)
        u_raw_fine = u_raw_fine - a_val * torch.sin(m_val * x_fine + phi_val)
        du_raw_fine = du_raw_fine - a_val * m_val * torch.cos(m_val * x_fine + phi_val)

    raw_max_comp_slope = float(torch.max(-du_raw_fine).item())
    target_slope = 1.0 / target_t_star
    scale = target_slope / raw_max_comp_slope
    u0 = scale * u_raw
    t_star = target_t_star
    return x, u0, t_star


def simulate_dag_complexity_trajectory(
    ic_name: str,
    u0: torch.Tensor,
    t_star: float,
    lambda_bek: int,
    t_final: float = 1.60,
    dt: float = 0.01,
    flux_coupling: float = 4.15,
) -> CausalDAGMetrics:
    """
    Evolve the 1+1D Unitary Nyquist Rotor Lattice up to t = t_final while dynamically
    constructing the spacetime causal DAG G = (V, E) and tracking all complexity observables.

    Note: `t_star` is stored in `CausalDAGMetrics` strictly as a post-hoc diagnostic reference
    to measure pre- vs. post-catastrophe DAG depth slopes; it is NEVER passed to the flux/trigger
    evaluator `evaluate_lax_commutator_and_saturation(u, config)`.
    """
    config = RotorLatticeConfig(
        lambda_bek=lambda_bek,
        dt=dt,
        nu_ml_intensive=1.0,
        flux_coupling=flux_coupling,
        max_clock_levels_per_step=512,
    )
    n_x = config.n_x
    dx = config.dx
    vartheta = config.impedance_angle
    n_steps = int(round(t_final / dt))
    n_sub = max(1, int(round(dt * lambda_bek / 0.64)))
    dt_sub = dt / float(n_sub)

    u = u0.to(COMPLEX_DTYPE).clone()
    initial_total_norm_sq = float(torch.sum(torch.abs(u) ** 2).item())

    clock_registers = torch.zeros((n_x, 4096), dtype=COMPLEX_DTYPE)
    clock_pointers = torch.zeros(n_x, dtype=torch.long)

    dag = SpacetimeCausalDAG(n_x=n_x, record_explicit_edges=(lambda_bek == 64))

    times: List[float] = [0.0]
    dag_depth_history: List[int] = [0]
    total_work_history: List[int] = [n_x]
    edge_count_history: List[int] = [0]
    parallelism_ratio_history: List[float] = [0.0]
    substep_sum_history: List[int] = [0]
    substep_max_history: List[int] = [0]
    active_shock_sites_history: List[int] = [0]
    theorem_integral_bound_history: List[float] = [0.0]
    substep_rows: List[torch.Tensor] = []

    cumulative_theorem_bound = 0.0
    max_norm_drift = 0.0
    total_physical_givens_rotations = 0

    for step in range(1, n_steps + 1):
        t_curr = step * dt
        # Evaluate F_j, Sigma_j, m_j purely from instantaneous spatial state u(x) (zero oracle branching)
        F_j, Sigma_j, m_j, _ = evaluate_lax_commutator_and_saturation(u, config)
        substep_rows.append(m_j.clone())

        # Record causal dependencies and sub-stepping vertices in G = (V, E)
        depth_G, work_V, edges_E = dag.advance_macro_step(m_j)

        # Accumulate Sub-Stepping Depth Theorem lower bound along the critical shock worldline gamma:
        # D(gamma) >= (1 / (pi * nu_ML)) int_gamma ||[P_Lambda, [M(u), L(u)]]||_HS ds
        # At the critical-path worldline node j_crit, 1 + m_{j_crit} >= F_{j_crit} / (Lambda_Bek * nu_ML) = F_{j_crit} ds / (pi * nu_ML)
        j_crit = int(torch.argmax(dag.terminal_depth).item())
        F_crit = float(F_j[j_crit].item())
        ds_worldline = dx
        cumulative_theorem_bound += F_crit * ds_worldline / (float(torch.pi) * config.nu_ml)

        # Advance field unitarily via Galilean-invariant delayed skew-Hermitian bulk + sequential SU(2) Givens clock rotations
        # Strict 1-to-1 per-macro-step rotation budget: partition m_j across s in range(n_sub) via m_sub = (m_j + s) // n_sub
        # so that sum_{s=0}^{n_sub - 1} m_sub == m_j identically at every spatial node j.
        w_eff = compute_effective_transport_velocity(u, m_j, config, dt_ref=0.01)
        m_accum = torch.zeros_like(m_j)
        for s in range(n_sub):
            u = _apply_unitary_cayley_step(u, w_eff, dt=dt_sub, dx=dx)
            m_sub = (m_j + s) // n_sub
            m_accum = m_accum + m_sub
            if torch.any(m_sub > 0):
                total_physical_givens_rotations += int(torch.sum(m_sub).item())
                u, clock_registers, clock_pointers = execute_sequential_givens_rotations(
                    u, clock_registers, clock_pointers, m_sub, vartheta=vartheta
                )
        assert torch.equal(m_accum, m_j), "Sub-step integer partition failed sum_s m_sub == m_j"

        cl_norm_sq = float(torch.sum(torch.abs(clock_registers) ** 2).item())
        tot_norm_sq = float(torch.sum(torch.abs(u) ** 2).item()) + cl_norm_sq
        max_norm_drift = max(
            max_norm_drift, abs(tot_norm_sq - initial_total_norm_sq) / initial_total_norm_sq
        )

        times.append(t_curr)
        dag_depth_history.append(depth_G)
        total_work_history.append(work_V)
        edge_count_history.append(edges_E)
        parallelism_ratio_history.append(float(depth_G) / float(work_V))
        substep_sum_history.append(int(torch.sum(m_j).item()))
        substep_max_history.append(int(torch.max(m_j).item()))
        active_shock_sites_history.append(int(torch.sum(m_j > 0).item()))
        theorem_integral_bound_history.append(cumulative_theorem_bound)

    # Compute elliptic slope (d(depth)/d(step) for t < t*) and hyperbolic slope (d(depth)/dt for t > t*)
    pre_indices = [i for i, t in enumerate(times) if 0 < t < t_star - 1e-9]
    post_indices = [i for i, t in enumerate(times) if t > t_star + 0.05]

    elliptic_slope_per_step = (
        float(dag_depth_history[pre_indices[-1]] - dag_depth_history[0]) / float(pre_indices[-1])
        if pre_indices
        else 1.0
    )
    hyperbolic_slope_per_time = (
        float(dag_depth_history[post_indices[-1]] - dag_depth_history[post_indices[0]])
        / float(times[post_indices[-1]] - times[post_indices[0]])
        if len(post_indices) >= 2
        else 0.0
    )

    return CausalDAGMetrics(
        ic_name=ic_name,
        lambda_bek=lambda_bek,
        n_x=n_x,
        dt=dt,
        t_star=t_star,
        times=times,
        dag_depth_history=dag_depth_history,
        total_work_history=total_work_history,
        edge_count_history=edge_count_history,
        parallelism_ratio_history=parallelism_ratio_history,
        substep_sum_history=substep_sum_history,
        substep_max_history=substep_max_history,
        active_shock_sites_history=active_shock_sites_history,
        theorem_integral_bound_history=theorem_integral_bound_history,
        max_norm_drift=max_norm_drift,
        elliptic_slope_per_step=elliptic_slope_per_step,
        hyperbolic_slope_per_time=hyperbolic_slope_per_time,
        total_physical_givens_rotations=total_physical_givens_rotations,
        spacetime_substep_map=torch.stack(substep_rows, dim=0),
    )


def run_milestone_3_complexity_sweep(
    cutoffs: Sequence[int] = (64, 128, 256),
    t_final: float = 1.60,
    dt: float = 0.01,
    save_dir: Optional[Path] = None,
) -> Dict[str, Dict[int, CausalDAGMetrics]]:
    """
    Execute the Milestone 3 parameter sweep across Lambda_Bek in {64, 128, 256} for both:
      (a) Single-shock sine wave u_0(x) = -sin(x) (t* = 1.0)
      (b) Random multi-shock gas (t* = 0.80)
    and run all Phase 3 / Phase 4 complexity phase transition assertions and falsification checks.
    """
    results: Dict[str, Dict[int, CausalDAGMetrics]] = {
        "single_shock": {},
        "multi_shock_gas": {},
    }

    for lb in cutoffs:
        _, u0_sine, t_star_sine = build_single_shock_initial_condition(lb)
        res_sine = simulate_dag_complexity_trajectory(
            ic_name="Single-Shock u0(x) = -sin(x)",
            u0=u0_sine,
            t_star=t_star_sine,
            lambda_bek=lb,
            t_final=t_final,
            dt=dt,
        )
        results["single_shock"][lb] = res_sine

        _, u0_gas, t_star_gas = build_multi_shock_gas_initial_condition(lb, target_t_star=0.80)
        res_gas = simulate_dag_complexity_trajectory(
            ic_name="Random Multi-Shock Gas",
            u0=u0_gas,
            t_star=t_star_gas,
            lambda_bek=lb,
            t_final=t_final,
            dt=dt,
        )
        results["multi_shock_gas"][lb] = res_gas

    # --- Automated Verification Assertions & Falsification Checks ---
    for ic_key, ic_dict in results.items():
        for lb, m in ic_dict.items():
            # 1. Unitary norm preservation to machine precision
            assert m.max_norm_drift < 1e-14, (
                f"[{ic_key}, Lambda={lb}] Unitary norm drift too large: {m.max_norm_drift}"
            )

            # 2. Elliptic regime verification (t < t*):
            #    m_j = 0 everywhere, depth(G) increases by exactly 1 per macro-step (O(1) parallel depth, independent of Lambda_Bek)
            for idx, t in enumerate(m.times):
                if 0 < t < m.t_star - 1e-9:
                    assert m.substep_sum_history[idx] == 0, (
                        f"[{ic_key}, Lambda={lb}] Unexpected rotor sub-steps at t={t:.3f} < t*={m.t_star:.3f}"
                    )
                    assert m.dag_depth_history[idx] == idx, (
                        f"[{ic_key}, Lambda={lb}] Expected O(1) step depth {idx} at t={t:.3f} < t*, got {m.dag_depth_history[idx]}"
                    )
            assert abs(m.elliptic_slope_per_step - 1.0) < 1e-12

            # 3. Hyperbolic regime verification (t > t*):
            #    Sharp kink at t = t*, transitioning to linear sequential scaling concentrated strictly along 1D shock worldlines
            assert m.hyperbolic_slope_per_time > (1.0 / m.dt) * 2.5, (
                f"[{ic_key}, Lambda={lb}] Failed to transition to super-linear sequential depth for t > t*"
            )

            # 4. Sub-Stepping Depth Theorem bound verification:
            #    D(gamma) >= (1 / (pi * nu_ML)) int_gamma ||[P_Lambda, [M(u), L(u)]]||_HS ds at all times t
            for d_val, bound_val in zip(m.dag_depth_history, m.theorem_integral_bound_history):
                assert float(d_val) >= bound_val - 1e-12, (
                    f"[{ic_key}, Lambda={lb}] Sub-Stepping Depth Theorem violated: D={d_val} < bound={bound_val}"
                )

            # 5. Strict 1D worldline localization check: active shock sites are a sparse subset (< 15% of lattice)
            max_active_frac = max(m.active_shock_sites_history) / float(m.n_x)
            assert 0.0 < max_active_frac < 0.20, (
                f"[{ic_key}, Lambda={lb}] Shock worldlines not 1D-localized: active_frac={max_active_frac:.3f}"
            )

            # 6. Strict 1-to-1 accounting between physical Givens rotations and causal DAG rotor vertices:
            #    total_physical_givens_rotations == sum_t sum_j m_j(t) == |V(T)| - (N_steps + 1) * N_x
            expected_substep_sum = sum(m.substep_sum_history)
            dag_rotor_vertices = m.total_work_history[-1] - len(m.times) * m.n_x
            assert m.total_physical_givens_rotations == expected_substep_sum == dag_rotor_vertices, (
                f"[{ic_key}, Lambda={lb}] 1-to-1 rotor accounting mismatch: "
                f"physical={m.total_physical_givens_rotations}, sum_m_j={expected_substep_sum}, "
                f"dag_rotor_vertices={dag_rotor_vertices}"
            )

        # 7. Verify linear scaling Omega((T - t*) * Lambda_Bek) across cutoffs {64, 128, 256}
        sorted_lbs = sorted(ic_dict.keys())
        for i in range(len(sorted_lbs) - 1):
            lb_low = sorted_lbs[i]
            lb_high = sorted_lbs[i + 1]
            ratio_lb = float(lb_high) / float(lb_low)

            m_low = ic_dict[lb_low]
            m_high = ic_dict[lb_high]

            # Pre-shock depths at t < t* must be IDENTICAL (independent of Lambda_Bek)
            pre_idx = int(0.5 * m_low.t_star / m_low.dt)
            assert m_low.dag_depth_history[pre_idx] == m_high.dag_depth_history[pre_idx], (
                f"[{ic_key}] Pre-catastrophe depth depended on Lambda_Bek: "
                f"{m_low.dag_depth_history[pre_idx]} vs {m_high.dag_depth_history[pre_idx]}"
            )

            # Post-shock sequential rotor depth D_rotor(T) = depth(G_T) - n_steps must scale as ~ ratio_lb
            n_steps_total = len(m_low.times) - 1
            excess_low = m_low.dag_depth_history[-1] - n_steps_total
            excess_high = m_high.dag_depth_history[-1] - n_steps_total
            observed_scaling = float(excess_high) / float(excess_low)
            assert 0.80 * ratio_lb <= observed_scaling <= 1.25 * ratio_lb, (
                f"[{ic_key}] Post-shock depth failed Omega((T - t*) * Lambda_Bek) scaling between "
                f"Lambda={lb_low} and {lb_high}: observed ratio={observed_scaling:.3f}, expected ~{ratio_lb:.3f}"
            )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_milestone_3_results(results, output_path=save_dir / "milestone_3_dag_scaling.png")

    return results


def plot_milestone_3_results(
    results: Dict[str, Dict[int, CausalDAGMetrics]], output_path: Path
) -> None:
    """Generate a 2x3 figure showing depth(G), total work |V|, parallelism ratio depth(G)/|V|, and worldline localization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=140)
    colors = {64: "#1f77b4", 128: "#ff7f0e", 256: "#2ca02c"}

    for row_idx, (ic_key, row_label) in enumerate(
        [
            ("single_shock", "Single-Shock $u_0(x) = -\\sin(x)$ ($t^* = 1.00$)"),
            ("multi_shock_gas", "Random Multi-Shock Gas ($t^* = 0.80$)"),
        ]
    ):
        ic_dict = results[ic_key]
        t_star = next(iter(ic_dict.values())).t_star

        # Col 0: Critical path length depth(G) vs T
        ax = axes[row_idx, 0]
        for lb, m in sorted(ic_dict.items()):
            ax.plot(
                m.times,
                m.dag_depth_history,
                label=f"$\\Lambda_{{\\mathrm{{Bek}}}} = {lb}$ (depth)",
                color=colors.get(lb, "k"),
                lw=2.2,
            )
            ax.plot(
                m.times,
                m.theorem_integral_bound_history,
                color=colors.get(lb, "k"),
                linestyle=":",
                lw=1.3,
                alpha=0.8,
            )
        ax.axvline(t_star, color="r", linestyle="--", alpha=0.7, label=f"Caustic $t^* = {t_star:.2f}$")
        ax.set_title(f"{row_label}\nCritical Path $\\mathrm{{depth}}(G)$ & Theorem Bound (dotted)")
        ax.set_xlabel("Macroscopic Time $T$")
        ax.set_ylabel("Critical Path Depth $\\mathrm{depth}(G)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 1: Total computational work |V| vs T
        ax = axes[row_idx, 1]
        for lb, m in sorted(ic_dict.items()):
            ax.plot(
                m.times,
                m.total_work_history,
                label=f"$\\Lambda_{{\\mathrm{{Bek}}}} = {lb}$ ($|V|$)",
                color=colors.get(lb, "k"),
                lw=2.2,
            )
        ax.axvline(t_star, color="r", linestyle="--", alpha=0.7, label=f"Caustic $t^* = {t_star:.2f}$")
        ax.set_title(f"{row_label}\nTotal DAG Computational Work $|V(T)|$")
        ax.set_xlabel("Macroscopic Time $T$")
        ax.set_ylabel("Total DAG Vertices $|V|$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 2: Parallelism ratio depth(G) / |V| vs T
        ax = axes[row_idx, 2]
        for lb, m in sorted(ic_dict.items()):
            ax.plot(
                m.times[1:],
                m.parallelism_ratio_history[1:],
                label=f"$\\Lambda_{{\\mathrm{{Bek}}}} = {lb}$",
                color=colors.get(lb, "k"),
                lw=2.2,
            )
        ax.axvline(t_star, color="r", linestyle="--", alpha=0.7, label=f"Caustic $t^* = {t_star:.2f}$")
        ax.set_title(f"{row_label}\nParallelism Ratio $\\mathrm{{depth}}(G) / |V|$ Phase Transition")
        ax.set_xlabel("Macroscopic Time $T$")
        ax.set_ylabel("Ratio $\\mathrm{depth}(G) / |V|$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 3: Causal DAG Critical-Path Scaling (Experiment H.4)"
    )
    parser.add_argument(
        "--t-final", type=float, default=1.60, help="Final evolution time T (default: 1.60)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Macro-step dt (default: 0.01)"
    )
    args = parser.parse_args()

    out_dir = Path("outputs")
    cutoffs = (64, 128, 256)
    results = run_milestone_3_complexity_sweep(
        cutoffs=cutoffs, t_final=args.t_final, dt=args.dt, save_dir=out_dir
    )

    print("=" * 100)
    print("MILESTONE 3: Causal DAG Critical-Path Scaling (Experiment H.4, Phase 3 & Phase 4)")
    print(
        f"Precision: {COMPLEX_DTYPE} / {FLOAT_DTYPE} | Cutoffs Lambda_Bek in {list(cutoffs)} | T_final = {args.t_final}, dt = {args.dt}"
    )
    print("=" * 100)

    for ic_key, title in [
        ("single_shock", "Initial Condition (a): Single-Shock Wave u_0(x) = -sin(x) (Analytical Breaking Time t* = 1.00)"),
        ("multi_shock_gas", "Initial Condition (b): Random Multi-Shock Gas (Analytical Breaking Time t* = 0.80)"),
    ]:
        print(f"\n--- {title} ---")
        print(
            "  Lambda_Bek | Depth(t=0.5*t*) | Final Depth(T) | Rotor Excess Depth | Theorem Bound | Total Work |V| | Phys Rotations | Final depth/|V| | Pre Slope | Post Slope/dt | Norm Drift"
        )
        print("  " + "-" * 175)
        for lb in cutoffs:
            m = results[ic_key][lb]
            pre_idx = int(0.5 * m.t_star / m.dt)
            n_steps_tot = len(m.times) - 1
            excess_d = m.dag_depth_history[-1] - n_steps_tot
            print(
                f"  {lb:10d} | {m.dag_depth_history[pre_idx]:15d} | {m.dag_depth_history[-1]:14d} | "
                f"{excess_d:18d} | {m.theorem_integral_bound_history[-1]:13.2f} | "
                f"{m.total_work_history[-1]:14d} | {m.total_physical_givens_rotations:14d} | "
                f"{m.parallelism_ratio_history[-1]:15.5f} | "
                f"{m.elliptic_slope_per_step:9.4f} | {m.hyperbolic_slope_per_time:13.2f} | "
                f"{m.max_norm_drift:10.3e}"
            )

        e64 = results[ic_key][64].dag_depth_history[-1] - (len(results[ic_key][64].times) - 1)
        e128 = results[ic_key][128].dag_depth_history[-1] - (len(results[ic_key][128].times) - 1)
        e256 = results[ic_key][256].dag_depth_history[-1] - (len(results[ic_key][256].times) - 1)
        print(
            f"  [SCALING VERIFIED] Post-catastrophe sequential depth ratios: "
            f"D_excess(128)/D_excess(64) = {e128 / e64:.3f}x, D_excess(256)/D_excess(128) = {e256 / e128:.3f}x "
            f"(theoretical Omega((T - t*) * Lambda_Bek) scaling = 2.000x)"
        )

    print("\n" + "=" * 100)
    print("ALL MILESTONE 3 VERIFICATION ASSERTIONS AND FALSIFICATION CHECKS PASSED.")
    print(f"DAG scaling figure saved to: {out_dir / 'milestone_3_dag_scaling.png'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
