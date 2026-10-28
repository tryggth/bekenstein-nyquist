# Bekenstein–Nyquist Shock Worldline Conjecture

Formal Lean 4 verification, 1+1D unitary hydrodynamic rotor simulations, and monograph for the **Bekenstein–Nyquist Shock Worldline Conjecture (BNSWC)**.

## Repository Structure

- **[`bekenstein_nyquist_monograph.pdf`](bekenstein_nyquist_monograph.pdf)** & **[`bekenstein_nyquist_monograph.tex`](bekenstein_nyquist_monograph.tex)** — Complete 16-page monograph (*Theoretical Architecture, Axiomatic Foundations, and Formal Verification of the Bekenstein–Nyquist Shock Worldline Conjecture*).
- **[`UnitaryBandwidthDepth.lean`](UnitaryBandwidthDepth.lean)** — Machine-checked Lean 4 formalization of the finite-dimensional bandwidth–depth tradeoff ($D \ge \mathrm{rank}([P_{\Lambda}, U_{\text{nl}}]) / 2$) with zero `sorry` axioms.
- **[`toy_model_1_unitary_leakage.py`](toy_model_1_unitary_leakage.py)** — Milestone 1 numerical verification of commutator rank duality and sequential clock-wire dilation bounds.
- **[`toy_model_2_nyquist_rotor_pde.py`](toy_model_2_nyquist_rotor_pde.py)** — Milestone 2 (Experiment H.3): $1+1\text{D}$ Unitary Nyquist Rotor Lattice for inviscid Burgers shocks with translation-invariant $\mathcal{H}_{2\Lambda}$ analytic-signal spillover sensing, Galilean-invariant wave-frame delay, and 5-bond Haar–Givens rotations.
- **[`toy_model_3_dag_complexity.py`](toy_model_3_dag_complexity.py)** — Milestone 3 (Experiment H.4): Spacetime causal dependency DAG critical-path scaling across $\Lambda_{\text{Bek}} \in \{64, 128, 256\}$ and 1-to-1 physical Givens rotation accounting.
- **[`toy_model_4_shock_collision_gate.py`](toy_model_4_shock_collision_gate.py)** — Milestone 4 (Phase 3.2 & Section 4.4): Two-shock collision interferometry, orthogonal 2-site Haar clock readback ($\neg$ gate evading the Monotone Planar CVP $\mathrm{NC}^3$ trap), and Ramsey clock-mass annihilation fringes ($\Delta\phi \in [0, 2\pi]$).
- **[`tests/`](tests/)** — Complete 40-test verification and anti-specification-gaming suite (`test_milestone_1.py` through `test_milestone_4.py`).
- **[`outputs/`](outputs/)** — Benchmark figures (`milestone_2_benchmarks.png`, `milestone_3_dag_scaling.png`, `milestone_4_shock_collision.png`).
- **[`walkthrough.md`](walkthrough.md)** — Detailed mathematical and numerical verification walkthrough.

## Verification

```bash
# Run Lean 4 formal verification
lake env lean UnitaryBandwidthDepth.lean

# Run the 40-test Python verification suite
./.venv/bin/pytest -v
```
