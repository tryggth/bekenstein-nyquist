import Physlib.Particles.StandardModel.HiggsBoson.Potential
import NyquistHiggsDuality

/-!
# Phase B Bridge 1 — Direct Upstream `Physlib` Higgs Potential Integration (`PhyslibHiggsBridge.lean`)

This module imports `Physlib.Particles.StandardModel.HiggsBoson.Potential` directly from
`leanprover-community/physlib` and bridges it with `NyquistHiggsDuality.lean`, proving that:
1. Under the parameter map `toPhyslibPotential σ₀ lam = ⟨-σ₀, lam⟩` (`μ² = -Σ₀`, `𝓵 = λ`),
   upstream `StandardModel.HiggsField.Potential.toFun φ x` is identically equal to the
   Bekenstein–Nyquist clock-condensate free-energy density
   `nyquistEffectivePotential σ₀ lam (‖φ‖_H^2 x)` at every spacetime point `x : SpaceTime`.
2. Applying upstream `StandardModel.HiggsField.Potential.isMinOn_iff_field_of_μSq_nonpos_𝓵_pos`
   and `StandardModel.HiggsField.Potential.isMinOn_iff_field_of_μSq_nonneg_𝓵_pos` directly:
   - In the **sub-Nyquist regime** (`0 ≤ σ₀`), the global minimum on `SpaceTime` occurs if and
     only if the clock section vanishes (`φ x = 0`).
   - In the **super-Nyquist shock regime** (`σ₀ ≤ 0`), the global minimum on `SpaceTime` occurs
     if and only if the pointwise squared norm equals the spontaneous clock-condensate VEV
     `‖φ‖_H^2 x = equilibriumClockNormSq σ₀ lam = -σ₀ / (2 * lam)`, restoring the sonic horizon
     `dV_dcNormSq σ₀ lam (‖φ‖_H^2 x) = 0`.
-/

namespace BekensteinNyquist

open StandardModel
open StandardModel.HiggsField

/-- Map the Bekenstein–Nyquist bare saturation `σ₀` and quartic stiffness `lam` directly into
    an upstream `Physlib` `StandardModel.HiggsField.Potential` structure (`μ2 = -σ₀`, `𝓵 = lam`). -/
def toPhyslibPotential (σ₀ lam : ℝ) : StandardModel.HiggsField.Potential where
  μ2 := -σ₀
  𝓵 := lam

/-- **Exact Pointwise Equivalence with Upstream `Physlib` `Potential.toFun`**:
    For any smooth section `φ : HiggsField` and spacetime point `x : SpaceTime`,
    `(toPhyslibPotential σ₀ lam).toFun φ x` equals `nyquistEffectivePotential σ₀ lam (‖φ‖_H^2 x)`. -/
theorem toPhyslibPotential_toFun_eq_nyquistEffectivePotential
    (σ₀ lam : ℝ) (φ : HiggsField) (x : SpaceTime) :
    (toPhyslibPotential σ₀ lam).toFun φ x =
      nyquistEffectivePotential σ₀ lam (‖φ‖_H^2 x) := by
  simp only [StandardModel.HiggsField.Potential.toFun, toPhyslibPotential,
    nyquistEffectivePotential]
  ring

/-- **Upstream `Physlib` Sub-Nyquist Vacuum Theorem (`0 ≤ σ₀`)**:
    Directly invokes `StandardModel.HiggsField.Potential.isMinOn_iff_field_of_μSq_nonpos_𝓵_pos`
    to prove that in the smooth sub-Nyquist regime (`0 ≤ σ₀`), `(φ, x)` minimizes the potential
    on `SpaceTime` if and only if `φ x = 0`. -/
theorem subNyquist_physlib_isMinOn_iff_zero
    (σ₀ lam : ℝ) (hσ : 0 ≤ σ₀) (hlam : 0 < lam) (φ : HiggsField) (x : SpaceTime) :
    IsMinOn (fun (φ, x) => (toPhyslibPotential σ₀ lam).toFun φ x) Set.univ (φ, x) ↔
      φ x = 0 := by
  have hμ2 : (toPhyslibPotential σ₀ lam).μ2 ≤ 0 := by
    simp [toPhyslibPotential, hσ]
  exact (toPhyslibPotential σ₀ lam).isMinOn_iff_field_of_μSq_nonpos_𝓵_pos hlam hμ2 φ x

/-- **Upstream `Physlib` Super-Nyquist Clock-Condensate Theorem (`σ₀ ≤ 0`)**:
    Directly invokes `StandardModel.HiggsField.Potential.isMinOn_iff_field_of_μSq_nonneg_𝓵_pos`
    to prove that when the Nyquist–Margolus–Levitin ceiling is breached (`σ₀ ≤ 0`), `(φ, x)`
    minimizes the potential on `SpaceTime` if and only if
    `‖φ‖_H^2 x = equilibriumClockNormSq σ₀ lam`. -/
theorem superNyquist_physlib_isMinOn_iff_condensate
    (σ₀ lam : ℝ) (hσ : σ₀ ≤ 0) (hlam : 0 < lam) (φ : HiggsField) (x : SpaceTime) :
    IsMinOn (fun (φ, x) => (toPhyslibPotential σ₀ lam).toFun φ x) Set.univ (φ, x) ↔
      ‖φ‖_H^2 x = equilibriumClockNormSq σ₀ lam := by
  have hμ2 : 0 ≤ (toPhyslibPotential σ₀ lam).μ2 := by
    simp [toPhyslibPotential, hσ]
  exact (toPhyslibPotential σ₀ lam).isMinOn_iff_field_of_μSq_nonneg_𝓵_pos hlam hμ2 φ x

/-- **Upstream `Physlib` Sonic-Horizon Restoration at the Potential Minimum**:
    Whenever `(φ, x)` minimizes `(toPhyslibPotential σ₀ lam).toFun` in the super-Nyquist regime
    (`σ₀ ≤ 0`, `0 < lam`), the effective Bekenstein–Nyquist saturation functional vanishes
    identically (`dV_dcNormSq σ₀ lam (‖φ‖_H^2 x) = 0`), restoring the sonic horizon. -/
theorem superNyquist_physlib_horizon_restoration
    (σ₀ lam : ℝ) (hσ : σ₀ ≤ 0) (hlam : 0 < lam) (φ : HiggsField) (x : SpaceTime)
    (hmin : IsMinOn (fun (φ, x) => (toPhyslibPotential σ₀ lam).toFun φ x) Set.univ (φ, x)) :
    dV_dcNormSq σ₀ lam (‖φ‖_H^2 x) = 0 := by
  have hnorm : ‖φ‖_H^2 x = equilibriumClockNormSq σ₀ lam :=
    (superNyquist_physlib_isMinOn_iff_condensate σ₀ lam hσ hlam φ x).mp hmin
  rw [hnorm]
  exact superNyquist_horizon_restoration σ₀ lam (ne_of_gt hlam)

end BekensteinNyquist
