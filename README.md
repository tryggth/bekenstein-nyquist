# Bekenstein–Nyquist Shock Worldline Conjecture

Formal Lean 4 verification, 1+1D unitary hydrodynamic rotor simulations, and monograph for the **Bekenstein–Nyquist Shock Worldline Conjecture (BNSWC)**.

## Repository Structure

- **[`bekenstein_nyquist_monograph.pdf`](bekenstein_nyquist_monograph.pdf)** & **[`bekenstein_nyquist_monograph.tex`](bekenstein_nyquist_monograph.tex)** — Complete 16-page monograph (*Theoretical Architecture, Axiomatic Foundations, and Formal Verification of the Bekenstein–Nyquist Shock Worldline Conjecture*).
- **[`UnitaryBandwidthDepth.lean`](UnitaryBandwidthDepth.lean)** — Machine-checked Lean 4 formalization of the finite-dimensional bandwidth–depth tradeoff ($D \ge \mathrm{rank}([P_{\Lambda}, U_{\text{nl}}]) / 2$) with zero `sorry` axioms.
- **[`NyquistRotorGates.lean`](NyquistRotorGates.lean)** — Machine-checked Lean 4 formalization of Phase A Test A1: exact $\mathrm{SU}(2)$ Givens rotor gates, in-phase constructive clock-mass creation ($G_0(\vartheta_2)G_0(\vartheta_1) = G_0(\vartheta_1 + \vartheta_2)$), anti-phase $\pi$-rotor negation gate ($G_\pi(\vartheta)G_0(\vartheta) = I_2$), Ramsey fringe formula, Pauli exclusion blockade, and Milestone 4 tripartite 3-mode Haar router unitarity.
- **[`NyquistHiggsDuality.lean`](NyquistHiggsDuality.lean)** — Machine-checked Lean 4 formalization of Phase A Test A2: Bekenstein–Nyquist saturation functional $\Sigma_0(x,t) = 1 - \frac{F_0(x,t)}{\Lambda_{\text{Bek}}\nu_{\text{ML}}}$, Physlib-compatible Mexican-hat Higgs potential bridge (`complete_square`, `pos_𝓵_lower_bound`, `minimum_iff_normSq_eq`), spontaneous clock-condensate symmetry breaking for $\Sigma_0 < 0$, and horizon restoration.
- **[`NyquistGaugeTransport.lean`](NyquistGaugeTransport.lean)** — Machine-checked Lean 4 formalization of Phase A Test A3: unconditional unitarity of the Cayley / Crank–Nicolson integrator (`cayleyTransform_mem_unitaryGroup`), skew-Hermiticity of split-form Burgers and gauge-covariant clock advection, and local $\mathrm{U}(1)$ gauge invariance of two-shock Ramsey collision holonomy and 3-site Wilson loops.
- **[`lean_physics_extension_plan.md`](lean_physics_extension_plan.md)** — Roadmap and compatibility audit for extending the Lean 4 formalization across `leanprover-community/physlib` (`QuantumInfo` + `Physlib`) and `LionSR/TNLean`.
- **[`toy_model_1_unitary_leakage.py`](toy_model_1_unitary_leakage.py)** — Milestone 1 numerical verification of commutator rank duality and sequential clock-wire dilation bounds.
- **[`toy_model_2_nyquist_rotor_pde.py`](toy_model_2_nyquist_rotor_pde.py)** — Milestone 2 (Experiment H.3): $1+1\text{D}$ Unitary Nyquist Rotor Lattice for inviscid Burgers shocks with translation-invariant $\mathcal{H}_{2\Lambda}$ analytic-signal spillover sensing, Galilean-invariant wave-frame delay, and 5-bond Haar–Givens rotations.
- **[`toy_model_3_dag_complexity.py`](toy_model_3_dag_complexity.py)** — Milestone 3 (Experiment H.4): Spacetime causal dependency DAG critical-path scaling across $\Lambda_{\text{Bek}} \in \{64, 128, 256\}$ and 1-to-1 physical Givens rotation accounting.
- **[`toy_model_4_shock_collision_gate.py`](toy_model_4_shock_collision_gate.py)** — Milestone 4 (Phase 3.2 & Section 4.4): Two-shock collision interferometry, orthogonal 2-site Haar clock readback ($\neg$ gate evading the Monotone Planar CVP $\mathrm{NC}^3$ trap), and Ramsey clock-mass annihilation fringes ($\Delta\phi \in [0, 2\pi]$).
- **[`tests/`](tests/)** — Complete 43-test verification and anti-specification-gaming suite (`test_milestone_1.py` through `test_milestone_4.py`).
- **[`outputs/`](outputs/)** — Benchmark figures (`milestone_2_benchmarks.png`, `milestone_3_dag_scaling.png`, `milestone_4_shock_collision.png`).
- **[`walkthrough.md`](walkthrough.md)** — Detailed mathematical and numerical verification walkthrough.

## Verification

```bash
# Build and verify all 4 Lean 4 modules (UnitaryBandwidthDepth, NyquistRotorGates, NyquistHiggsDuality, NyquistGaugeTransport)
lake build

# Run the 43-test Python & Lean 4 verification suite
./.venv/bin/pytest -v
```
