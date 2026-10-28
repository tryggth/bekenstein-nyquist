import UnitaryBandwidthDepth
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

/-!
# Phase A (Extension 1): Landau–Ginzburg–Higgs & Nyquist Horizon Criticality (`NyquistHiggsDuality.lean`)

Formalizes the structural duality between the Landau–Ginzburg–Higgs mechanism
(`Physlib.Particles.StandardModel.HiggsBoson.Potential`) and the Bekenstein–Nyquist
worldline clock fiber (Monograph Section 4.5.3, Equations 19–20, and Table 9):

1. `HiggsPotential`: Exact mirror of `StandardModel.HiggsField.Potential` in `Physlib`
   (`- P.μ2 * normSq + P.𝓵 * normSq * normSq`), along with `complete_square`,
   `pos_𝓵_lower_bound`, and `minimum_iff_normSq_eq`.
2. `toHiggsPotential`: Explicit sign-convention bridge `μ2 := -σ₀, 𝓵 := lam` mapping the
   Bekenstein–Nyquist Saturation Functional `σ₀(x,t) = 1 - F₀(x,t) / (Λ_Bek * ν_ML)` to
   `Physlib`'s quadratic Higgs mass parameter.
3. `saturation_linear_relaxation`: Formal proof that linear commutator defect absorption
   `F(|c|²) = F₀ - κ |c|²` shifts the saturation functional to `Σ(|c|²) = σ₀ + 2 λ |c|²`
   with `λ = κ / (2 Λ_Bek ν_ML)`.
4. `subNyquist_unique_minimum_zero`: In the sub-Nyquist elliptic bulk (`σ₀ > 0, λ > 0`),
   the unique global minimum on `c_normSq ≥ 0` is `c_normSq = 0` (`m_j = 0`).
5. `superNyquist_condensate_minimum` & `superNyquist_horizon_restoration`: In the super-Nyquist
   hyperbolic shock core (`σ₀ < 0, λ > 0`), symmetry breaks to the strictly positive equilibrium
   clock condensate `c_eq² = -σ₀ / (2λ) > 0`, which restores the effective Tricomi–Keldysh symbol
   to the critical Sonic/Nyquist Horizon `Σ(c_eq²) = σ₀ + 2λ c_eq² = 0`.
6. `quarticCoupling_cutoff_independent`: Proves that under Paley–Wiener linear scaling
   `κ = c₁ Λ_Bek` and `θ_Nyq = Λ_Bek ν_ML`, the quartic Higgs–Nyquist coupling `λ = c₁ / (2 ν_ML)`
   is strictly independent of the ultraviolet cutoff `Λ_Bek`.
-/

noncomputable section

namespace BekensteinNyquist

/-! ### 1. Exact Mirror of `Physlib.Particles.StandardModel.HiggsBoson.Potential` -/

/-- Mirror of `StandardModel.HiggsField.Potential` from `Physlib`:
    parameterized by `μ2 : ℝ` (with `-μ2` as the quadratic coefficient) and quartic coupling `𝓵 : ℝ`. -/
structure HiggsPotential where
  μ2 : ℝ
  𝓵 : ℝ

namespace HiggsPotential

/-- Evaluation of `HiggsPotential` on the squared norm `normSq = ‖φ‖² ≥ 0`:
    `V(normSq) = - P.μ2 * normSq + P.𝓵 * normSq * normSq`. -/
def toFun (P : HiggsPotential) (normSq : ℝ) : ℝ :=
  - P.μ2 * normSq + P.𝓵 * normSq * normSq

/-- Completion of the square for the Higgs potential (`Physlib` `Potential.complete_square`). -/
theorem complete_square (P : HiggsPotential) (h : P.𝓵 ≠ 0) (normSq : ℝ) :
    P.toFun normSq = P.𝓵 * (normSq - P.μ2 / (2 * P.𝓵)) ^ 2 - (P.μ2 ^ 2 / (4 * P.𝓵)) := by
  unfold toFun
  field_simp
  ring

/-- Global lower bound `- (μ2² / (4 𝓵)) ≤ P.toFun normSq` when `0 < P.𝓵`
    (`Physlib` `Potential.pos_𝓵_quadDiscrim_zero_bound`). -/
theorem pos_𝓵_lower_bound (P : HiggsPotential) (h : 0 < P.𝓵) (normSq : ℝ) :
    - (P.μ2 ^ 2 / (4 * P.𝓵)) ≤ P.toFun normSq := by
  rw [P.complete_square (ne_of_gt h) normSq]
  have hsq : 0 ≤ (normSq - P.μ2 / (2 * P.𝓵)) ^ 2 := sq_nonneg _
  have hmul : 0 ≤ P.𝓵 * (normSq - P.μ2 / (2 * P.𝓵)) ^ 2 := mul_nonneg (le_of_lt h) hsq
  linarith

/-- Characterization of the global minimum of `HiggsPotential` (`Physlib` `quadDiscrim_eq_zero_iff_normSq`):
    `P.toFun normSq = - (P.μ2² / (4 P.𝓵)) ↔ normSq = P.μ2 / (2 P.𝓵)`. -/
theorem minimum_iff_normSq_eq (P : HiggsPotential) (h : 0 < P.𝓵) (normSq : ℝ) :
    P.toFun normSq = - (P.μ2 ^ 2 / (4 * P.𝓵)) ↔ normSq = P.μ2 / (2 * P.𝓵) := by
  rw [P.complete_square (ne_of_gt h) normSq]
  constructor
  · intro hEq
    have hzero : P.𝓵 * (normSq - P.μ2 / (2 * P.𝓵)) ^ 2 = 0 := by linarith
    have hsq0 : (normSq - P.μ2 / (2 * P.𝓵)) ^ 2 = 0 :=
      (mul_eq_zero.mp hzero).resolve_left (ne_of_gt h)
    have hsub0 : normSq - P.μ2 / (2 * P.𝓵) = 0 := sq_eq_zero_iff.mp hsq0
    linarith
  · intro hNorm
    rw [hNorm, sub_self]
    ring

end HiggsPotential

/-! ### 2. Bekenstein–Nyquist Saturation Functional & Sign-Convention Bridge -/

/-- Bekenstein–Nyquist Saturation Functional `Σ(F) = 1 - F / θ_Nyq`, where `θ_Nyq = Λ_Bek * ν_ML`. -/
def saturationFunctional (F θ_Nyq : ℝ) : ℝ :=
  1 - F / θ_Nyq

/-- Linear reduction of the active spatial Lax commutator defect `F(|c|²) = F₀ - κ |c|²`
    as ultraviolet norm `c_normSq = |c|²` is rotated into the worldline clock fiber. -/
def relaxedDefect (F₀ κ c_normSq : ℝ) : ℝ :=
  F₀ - κ * c_normSq

/-- Quartic self-coupling `λ = κ / (2 θ_Nyq)` induced by linear commutator relaxation. -/
def quarticCoupling (κ θ_Nyq : ℝ) : ℝ :=
  κ / (2 * θ_Nyq)

/-- **Linear Relaxation of the Saturation Functional (Monograph Eq. 19)**:
    Rotating norm `c_normSq = |c|²` into the clock fiber shifts the instantaneous
    saturation functional from `σ₀ = saturationFunctional F₀ θ_Nyq` to `σ₀ + 2 λ |c|²`. -/
theorem saturation_linear_relaxation (F₀ κ θ_Nyq c_normSq : ℝ) (hθ : θ_Nyq ≠ 0) :
    saturationFunctional (relaxedDefect F₀ κ c_normSq) θ_Nyq =
      saturationFunctional F₀ θ_Nyq + 2 * (quarticCoupling κ θ_Nyq) * c_normSq := by
  unfold saturationFunctional relaxedDefect quarticCoupling
  field_simp
  ring

/-- Effective Landau–Ginzburg–Higgs potential `V_eff(|c|²) = σ₀ |c|² + λ |c|⁴` (Monograph Eq. 20). -/
def nyquistEffectivePotential (σ₀ lam c_normSq : ℝ) : ℝ :=
  σ₀ * c_normSq + lam * c_normSq ^ 2

/-- Derivative of `nyquistEffectivePotential` with respect to `c_normSq = |c|²` recovers
    the relaxed saturation functional `Σ(|c|²) = σ₀ + 2 λ |c|²`. -/
def dV_dcNormSq (σ₀ lam c_normSq : ℝ) : ℝ :=
  σ₀ + 2 * lam * c_normSq

/-- Canonical bridge mapping Bekenstein–Nyquist parameters `(σ₀, lam)` to `Physlib`'s
    `HiggsPotential` via `μ2 := -σ₀` and `𝓵 := lam`. -/
def toHiggsPotential (σ₀ lam : ℝ) : HiggsPotential :=
  { μ2 := -σ₀, 𝓵 := lam }

/-- **Sign-Convention Bridge Identity**:
    `nyquistEffectivePotential σ₀ lam c_normSq` is identically equal to
    `(toHiggsPotential σ₀ lam).toFun c_normSq`. -/
theorem nyquistEffectivePotential_eq_higgsPotential (σ₀ lam c_normSq : ℝ) :
    nyquistEffectivePotential σ₀ lam c_normSq = (toHiggsPotential σ₀ lam).toFun c_normSq := by
  unfold nyquistEffectivePotential toHiggsPotential HiggsPotential.toFun
  ring

/-! ### 3. Horizon Criticality Duality Theorems (Elliptic Bulk vs. Hyperbolic Shock Core) -/

/-- **Sub-Nyquist Elliptic Bulk Unbroken Phase (`σ₀ > 0, λ > 0`)**:
    In the sub-Nyquist bulk where `σ₀ > 0`, `nyquistEffectivePotential σ₀ lam c_normSq ≥ 0`
    for all `c_normSq ≥ 0`, and vanishes if and only if `c_normSq = 0` (no clock condensation, `m_j = 0`). -/
theorem subNyquist_unique_minimum_zero
    (σ₀ lam c_normSq : ℝ) (hσ : 0 < σ₀) (hlam : 0 < lam) (hc : 0 ≤ c_normSq) :
    0 ≤ nyquistEffectivePotential σ₀ lam c_normSq ∧
      (nyquistEffectivePotential σ₀ lam c_normSq = 0 ↔ c_normSq = 0) := by
  unfold nyquistEffectivePotential
  have h1 : 0 ≤ σ₀ * c_normSq := mul_nonneg (le_of_lt hσ) hc
  have h2 : 0 ≤ lam * c_normSq ^ 2 := mul_nonneg (le_of_lt hlam) (sq_nonneg c_normSq)
  refine ⟨by linarith, ?_⟩
  constructor
  · intro hzero
    by_contra hne
    have hc_pos : 0 < c_normSq := lt_of_le_of_ne hc (Ne.symm hne)
    have h1_pos : 0 < σ₀ * c_normSq := mul_pos hσ hc_pos
    linarith
  · intro hc0
    rw [hc0]
    ring

/-- Equilibrium clock condensate norm squared `c_eq² = -σ₀ / (2λ)` in the super-Nyquist shock core. -/
def equilibriumClockNormSq (σ₀ lam : ℝ) : ℝ :=
  -σ₀ / (2 * lam)

/-- **Super-Nyquist Shock Core Spontaneous Symmetry Breaking (`σ₀ < 0, λ > 0`)**:
    When `σ₀ < 0` and `λ > 0`:
    1. The equilibrium clock condensate norm squared `equilibriumClockNormSq σ₀ lam = -σ₀ / (2λ)` is strictly positive.
    2. For all `c_normSq`, `nyquistEffectivePotential σ₀ lam c_normSq ≥ - (σ₀² / (4λ))`.
    3. The global minimum `- (σ₀² / (4λ))` is attained if and only if `c_normSq = equilibriumClockNormSq σ₀ lam`. -/
theorem superNyquist_condensate_minimum
    (σ₀ lam : ℝ) (hσ : σ₀ < 0) (hlam : 0 < lam) :
    0 < equilibriumClockNormSq σ₀ lam ∧
    (∀ c_normSq : ℝ, - (σ₀ ^ 2 / (4 * lam)) ≤ nyquistEffectivePotential σ₀ lam c_normSq) ∧
    (∀ c_normSq : ℝ,
      nyquistEffectivePotential σ₀ lam c_normSq = - (σ₀ ^ 2 / (4 * lam)) ↔
        c_normSq = equilibriumClockNormSq σ₀ lam) := by
  refine ⟨?_, ?_, ?_⟩
  · unfold equilibriumClockNormSq
    have hneg : 0 < -σ₀ := neg_pos.mpr hσ
    have hden : 0 < 2 * lam := by linarith
    exact div_pos hneg hden
  · intro c_normSq
    rw [nyquistEffectivePotential_eq_higgsPotential]
    simpa [toHiggsPotential, neg_sq] using
      (toHiggsPotential σ₀ lam).pos_𝓵_lower_bound hlam c_normSq
  · intro c_normSq
    rw [nyquistEffectivePotential_eq_higgsPotential]
    simpa [toHiggsPotential, equilibriumClockNormSq, neg_sq] using
      (toHiggsPotential σ₀ lam).minimum_iff_normSq_eq hlam c_normSq

/-- **Sonic / Nyquist Horizon Restoration Identity (`Σ(|c_eq|²) = 0`)**:
    Condensing clock norm `c_eq² = equilibriumClockNormSq σ₀ lam = -σ₀ / (2λ)` inside a
    super-Nyquist shock core (`σ₀ < 0`) drives the relaxed saturation functional
    `dV_dcNormSq σ₀ lam (equilibriumClockNormSq σ₀ lam) = σ₀ + 2λ c_eq²` to zero,
    pinning the composite system back to the critical Sonic/Nyquist Horizon `Σ = 0`. -/
theorem superNyquist_horizon_restoration (σ₀ lam : ℝ) (hlam : lam ≠ 0) :
    dV_dcNormSq σ₀ lam (equilibriumClockNormSq σ₀ lam) = 0 := by
  unfold dV_dcNormSq equilibriumClockNormSq
  field_simp
  ring

/-- **Cutoff-Independent Higgs–Nyquist Stiffness (`O(Λ_Bek⁰)`)**:
    When the spillover absorption rate scales linearly with the Bekenstein cutoff (`κ = c₁ * Λ_Bek`)
    and the Nyquist threshold is `θ_Nyq = Λ_Bek * ν_ML` (`Λ_Bek ≠ 0`), the quartic coupling
    `quarticCoupling (c₁ * Λ_Bek) (Λ_Bek * ν_ML)` equals `c₁ / (2 * ν_ML)`, independent of `Λ_Bek`. -/
theorem quarticCoupling_cutoff_independent (c₁ Λ_Bek ν_ML : ℝ) (hΛ : Λ_Bek ≠ 0) :
    quarticCoupling (c₁ * Λ_Bek) (Λ_Bek * ν_ML) = c₁ / (2 * ν_ML) := by
  unfold quarticCoupling
  by_cases hν : ν_ML = 0
  · subst hν
    simp
  · field_simp

end BekensteinNyquist
