# Bekenstein–Nyquist Shock Worldline Conjecture

Formal Lean 4 verification, 1+1D unitary hydrodynamic rotor simulations, and monograph for the **Bekenstein–Nyquist Shock Worldline Conjecture (BNSWC)**.

## Foundational Premises (Axioms A1–A4)

1. **Axiom A1 — Bekenstein Spatial Bandlimitation ($P_{\Lambda}$):** Observable spatial fields reside in a Paley–Wiener subspace $`V_{\Lambda} = P_{\Lambda}\mathcal{H}_{\text{tot}} = PW_{\Lambda_{\text{Bek}}}(\mathbb{R})`$ defined by an idempotent, self-adjoint ultraviolet wavenumber projector $P_{\Lambda}^2 = P_{\Lambda} = P_{\Lambda}^\dagger$ with hard cutoff $|k| \le \Lambda_{\text{Bek}}$ derived from the Bekenstein bound $I_{\max} \le \frac{2\pi E R}{\hbar c \ln 2}$.
2. **Axiom A2 — Margolus–Levitin Rate Limit on State Changes ($\nu_{\text{ML}}$):** Any local state transition coupling $V_{\Lambda}$ to an orthogonal degree of freedom proceeds through elementary rank-1 quantum channels ($\mathrm{rank}((I - P_{\Lambda}) W_k P_{\Lambda}) \le 1$) bounded by the Margolus–Levitin transition frequency $\nu_{\text{ML}} \le \frac{2E}{\pi\hbar}$.
3. **Axiom A3 — Scale-Generating Nonlinearity ($[P_{\Lambda}, U_{\text{nl}}] \ne 0$):** Nonlinear convective steepening multiplies modes across the Brillouin boundary so the unconstrained short-time propagator $U_{\text{nl}} = \exp(\Delta t M(u))$ fails to commute with the spatial bandlimit projector $P_{\Lambda}$.
4. **Axiom A4 — Reflectionless Local Unitarity ($U^\dagger U = I$):** Time evolution on $`\mathcal{H}_{\text{tot}}`$ is strictly unitary and causally local, forbidding both dissipative Landauer bit erasure ($`E_{\min} = n k_B T \ln 2`$) and unphysical Umklapp aliasing / Gibbs reflection at $|k| = \Lambda_{\text{Bek}}$.

### From Established Axioms to Machine-Checked Theorems

Once Axioms A1–A4 are locked in, the conserved trade-off between spatial bandwidth, register precision, and sequential time-stepping is derived as a strict mathematical theorem rather than assumed as an analogy. Specifically, via Stinespring and Sz.-Nagy unitary dilation, preserving unitarity ($U^\dagger U = I$) across the bandlimit projector $P_{\Lambda}$ when $[P_{\Lambda}, U_{\text{nl}}] \ne 0$ mathematically forces the existence of an orthogonal worldline clock fiber $`\mathcal{H}_{\text{clock}} = V_{\Lambda}^{\perp}`$ and bounds the minimum sequential circuit depth by $D \ge \mathrm{rank}([P_{\Lambda}, U_{\text{nl}}]) / 2$. This algebraic core—along with its formal bridges to Landau–Ginzburg–Higgs criticality and quantum channels—has been machine-checked in Lean 4 ([`UnitaryBandwidthDepth.lean`](UnitaryBandwidthDepth.lean)) with zero unproven `sorry` axioms.

## Repository Structure

- **[`bekenstein_nyquist_monograph.pdf`](bekenstein_nyquist_monograph.pdf)** & **[`bekenstein_nyquist_monograph.tex`](bekenstein_nyquist_monograph.tex)** — Complete 18-page monograph (*Theoretical Architecture, Axiomatic Foundations, and Formal Verification of the Bekenstein–Nyquist Shock Worldline Conjecture*).
- **[`UnitaryBandwidthDepth.lean`](UnitaryBandwidthDepth.lean)** — Machine-checked Lean 4 formalization of the finite-dimensional bandwidth–depth tradeoff ($D \ge \mathrm{rank}([P_{\Lambda}, U_{\text{nl}}]) / 2$) with zero `sorry` axioms.
- **[`NyquistRotorGates.lean`](NyquistRotorGates.lean)** — Machine-checked Lean 4 formalization of Phase A Test A1: exact $\mathrm{SU}(2)$ Givens rotor gates, in-phase constructive clock-mass creation ($G_0(\vartheta_2)G_0(\vartheta_1) = G_0(\vartheta_1 + \vartheta_2){}$), anti-phase $\pi$-rotor negation gate ($G_\pi(\vartheta)G_0(\vartheta) = I_2$), Ramsey fringe formula, Pauli exclusion blockade, and Milestone 4 tripartite 3-mode Haar router unitarity.
- **[`NyquistHiggsDuality.lean`](NyquistHiggsDuality.lean)** — Machine-checked Lean 4 formalization of Phase A Test A2: Bekenstein–Nyquist saturation functional $\Sigma_0(x,t) = 1 - \frac{F_0(x,t)}{\Lambda_{\text{Bek}}\nu_{\text{ML}}}$, Physlib-compatible Mexican-hat Higgs potential bridge (`complete_square`, `pos_𝓵_lower_bound`, `minimum_iff_normSq_eq`), spontaneous clock-condensate symmetry breaking for $\Sigma_0 < 0$, and horizon restoration.
- **[`NyquistGaugeTransport.lean`](NyquistGaugeTransport.lean)** — Machine-checked Lean 4 formalization of Phase A Test A3: unconditional unitarity of the Cayley / Crank–Nicolson integrator (`cayleyTransform_mem_unitaryGroup`), skew-Hermiticity of split-form Burgers and gauge-covariant clock advection, and local $\mathrm{U}(1)$ gauge invariance of two-shock Ramsey collision holonomy and 3-site Wilson loops.
- **[`PhyslibHiggsBridge.lean`](PhyslibHiggsBridge.lean)** — Phase B direct upstream bridge importing `Physlib.Particles.StandardModel.HiggsBoson.Potential` from `leanprover-community/physlib`, proving exact pointwise equivalence between `StandardModel.HiggsField.Potential.toFun` and `nyquistEffectivePotential` and invoking `isMinOn_iff_field_of_μSq_nonpos_𝓵_pos` / `isMinOn_iff_field_of_μSq_nonneg_𝓵_pos` directly on `SpaceTime`.
- **[`QuantumInfoClockBridge.lean`](QuantumInfoClockBridge.lean)** — Phase B direct upstream bridge importing `QuantumInfo.Channels.CPTP` and `QuantumInfo.Entropy.VonNeumann` from `leanprover-community/physlib`, certifying our Givens and Milestone 4 tripartite routers as `CPTPMap` channels, proving Stinespring purification duality (`CPTPMap.purify_trace`), and proving zero global Landauer entropy (`Sᵥₙ_of_pure_zero`) with apparent spatial shock entropy equal to worldline clock entanglement entropy (`Sᵥₙ_of_partial_eq`).
- **[`lean_physics_extension_plan.md`](lean_physics_extension_plan.md)** — Roadmap and compatibility audit for extending the Lean 4 formalization across `leanprover-community/physlib` (`QuantumInfo` + `Physlib`) and `LionSR/TNLean`.
- **[`toy_model_1_unitary_leakage.py`](toy_model_1_unitary_leakage.py)** — Milestone 1 numerical verification of commutator rank duality and sequential clock-wire dilation bounds.
- **[`toy_model_2_nyquist_rotor_pde.py`](toy_model_2_nyquist_rotor_pde.py)** — Milestone 2 (Experiment H.3): $1+1\text{D}$ Unitary Nyquist Rotor Lattice for inviscid Burgers shocks with translation-invariant $\mathcal{H}_{2\Lambda}$ analytic-signal spillover sensing, Galilean-invariant wave-frame delay, and 5-bond Haar–Givens rotations.
- **[`toy_model_3_dag_complexity.py`](toy_model_3_dag_complexity.py)** — Milestone 3 (Experiment H.4): Spacetime causal dependency DAG critical-path scaling across $\Lambda_{\text{Bek}} \in \lbrace 64, 128, 256 \rbrace$ and 1-to-1 physical Givens rotation accounting.
- **[`toy_model_4_shock_collision_gate.py`](toy_model_4_shock_collision_gate.py)** — Milestone 4 (Phase 3.2 & Section 4.4): Two-shock collision interferometry, orthogonal 2-site Haar clock readback ($\neg$ gate evading the Monotone Planar CVP $\mathrm{NC}^3$ trap), and Ramsey clock-mass annihilation fringes ($\Delta\phi \in [0, 2\pi]$).
- **[`tests/`](tests/)** — Complete 45-test verification and anti-specification-gaming suite (`test_milestone_1.py` through `test_milestone_4.py`).
- **[`outputs/`](outputs/)** — Benchmark figures (`milestone_2_benchmarks.png`, `milestone_3_dag_scaling.png`, `milestone_4_shock_collision.png`).
- **[`walkthrough.md`](walkthrough.md)** — Detailed mathematical and numerical verification walkthrough.

## Verification

```bash
# Build and verify all 4 Lean 4 modules (UnitaryBandwidthDepth, NyquistRotorGates, NyquistHiggsDuality, NyquistGaugeTransport)
lake build

# Run the 45-test Python & Lean 4 verification suite
./.venv/bin/pytest -v
```
