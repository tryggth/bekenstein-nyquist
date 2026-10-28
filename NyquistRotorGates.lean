import UnitaryBandwidthDepth
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Phase A (Extensions 2 & 3A): SU(2) Givens Rotors, π-Rotor Negation, Pauli Blockade & Tripartite Haar Routing (`NyquistRotorGates.lean`)

Extends `UnitaryBandwidthDepth.lean` with machine-checked formal proofs for:
1. **2×2 Complex-Phase SU(2) Givens Rotors on `BlockIdx 1 1` (`givensRotor`)**:
   - Defined directly on `BlockIdx 1 1 = Fin 1 ⊕ Fin 1` (`Sum.inl 0` = spatial UV mode `u_UV ∈ V_Λ`,
     `Sum.inr 0` = orthogonal clock wire `c ∈ V_Λ^⊥`) via `Matrix.fromBlocks`.
   - Exact unitarity `givensRotor_mem_unitaryGroup` (`Gᴴ * G = 1` and `G * Gᴴ = 1`).
   - Direct compatibility with Phase 5's `IsElementaryClockGate`: `givensRotor_isElementaryClockGate`.
   - Constructive in-phase clock-mass creation (`Δϕ = 0`, `ζ = 1`):
     `givensRotor ϑ₂ 1 * givensRotor ϑ₁ 1 = givensRotor (ϑ₁ + ϑ₂) 1`.
   - Coherent anti-phase clock-mass annihilation / logical negation (`¬`) gate (`Δϕ = π`, `ζ = -1`):
     `givensRotor ϑ (-1) * givensRotor ϑ 1 = 1` (evading the Monotone Planar CVP `NC³` trap).
   - Shock Mach–Zehnder / Ramsey interference fringe formula (`givensRotor_ramsey_fringe_entry`):
     the net UV-to-clock forward leakage after two pulses is `(1 + ζ) sin ϑ cos ϑ`.
2. **Pauli Exclusion Blockade on 1D Clock Wires**:
   - `pauli_saturation_reflection`: When a 1D clock wire is driven to saturation at `ϑ = π / 2`
     (`sin(π/2) = 1, cos(π/2) = 0`), a second in-phase `π / 2` pulse cannot compress additional
     amplitude into the clock wire (`forwardLeakage = 0` at `2 * (π / 2) = π`).
   - `pauli_exclusion_blockade_ker_finrank`: By `rank_forwardLeakage_le_one_of_isElementaryClockGate`
     and rank-nullity, any elementary 1D clock gate `IsElementaryClockGate d m c W` has
     `finrank (ker (forwardLeakage P W)) ≥ d + m - 1`, forbidding simultaneous compression of two
     independent spatial UV modes into a single clock wire.
3. **Milestone 4 Tripartite 3-Mode Haar Router `(ū_b, u_{b,UV}, c_slot)` (`TripartiteIdx`)**:
   - Formalizes the 3-mode subspace `Fin 1 ⊕ BlockIdx 1 1` (`ū_b ⊕ u_{b,UV} ⊕ c_slot`), proving
     exact unitarity of Stage 1 (`u_{b,UV} ↔ c_slot`), Stage 2 (`ū_b ↔ c_slot`), and their composite
     router, and proving `stage2MeanReadback_preserves_uv_mode` (zero UV spike excitation during readback).
4. **Direct-Sum Norm & Purity Defect Identity (`spatial_compression_add_clock_leakage_eq_proj`)**:
   - Proves `(P U P)ᴴ (P U P) + Aᴴ A = P`, establishing that apparent spatial norm dissipation
     is identically equal to the clock-fiber Gram matrix `Aᴴ A`.
-/

noncomputable section

open Matrix

namespace BekensteinNyquist

/-! ### 1. 1×1 Scalar Block Helpers & 2×2 SU(2) Givens Rotors on `BlockIdx 1 1` -/

/-- Embeds a complex scalar `z : ℂ` as a `1 × 1` matrix on `Fin 1`. -/
def scalarMat1 (z : ℂ) : Matrix (Fin 1) (Fin 1) ℂ :=
  fun _ _ => z

@[simp]
theorem scalarMat1_apply (z : ℂ) (i j : Fin 1) : scalarMat1 z i j = z := rfl

@[simp]
theorem scalarMat1_zero : scalarMat1 0 = 0 := by
  ext i j; rfl

@[simp]
theorem scalarMat1_one : scalarMat1 1 = 1 := by
  ext i j
  fin_cases i; fin_cases j
  rfl

@[simp]
theorem scalarMat1_add (a b : ℂ) : scalarMat1 a + scalarMat1 b = scalarMat1 (a + b) := by
  ext i j; rfl

@[simp]
theorem scalarMat1_sub (a b : ℂ) : scalarMat1 a - scalarMat1 b = scalarMat1 (a - b) := by
  ext i j; rfl

@[simp]
theorem scalarMat1_neg (a : ℂ) : -scalarMat1 a = scalarMat1 (-a) := by
  ext i j; rfl

@[simp]
theorem scalarMat1_mul (a b : ℂ) : scalarMat1 a * scalarMat1 b = scalarMat1 (a * b) := by
  ext i j
  simp [Matrix.mul_apply, scalarMat1]

@[simp]
theorem scalarMat1_conjTranspose (a : ℂ) : (scalarMat1 a)ᴴ = scalarMat1 (star a) := by
  ext i j; rfl

/-- Helper: `star (r : ℂ) = (r : ℂ)` for any real `r : ℝ`. -/
@[simp]
theorem star_ofReal (r : ℝ) : star (r : ℂ) = (r : ℂ) :=
  Complex.conj_ofReal r

/-- Fundamental complex-cast Pythagorean identity: `cos² ϑ + sin² ϑ = 1`. -/
theorem ofReal_cos_sq_add_sin_sq (ϑ : ℝ) :
    (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + (Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ) = 1 := by
  norm_cast
  have h := Real.cos_sq_add_sin_sq ϑ
  linarith [sq (Real.cos ϑ), sq (Real.sin ϑ)]

/-- General 2×2 complex-phase Givens rotor matrix on `BlockIdx 1 1 = Fin 1 ⊕ Fin 1`,
    where `Sum.inl 0` is the spatial Nyquist mode `u_UV ∈ V_Λ`, `Sum.inr 0` is the
    orthogonal clock wire `c ∈ V_Λ^⊥`, `ϑ : ℝ` is the rotation angle, and `ζ : ℂ` (`star ζ * ζ = 1`)
    is the rotor phase factor `e^{iϕ}`:
    `G(ϑ, ζ) = fromBlocks (cos ϑ) (-star ζ * sin ϑ) (ζ * sin ϑ) (cos ϑ)`. -/
def givensRotor (ϑ : ℝ) (ζ : ℂ) : Matrix (BlockIdx 1 1) (BlockIdx 1 1) ℂ :=
  Matrix.fromBlocks
    (scalarMat1 (Real.cos ϑ : ℂ))
    (scalarMat1 (- star ζ * (Real.sin ϑ : ℂ)))
    (scalarMat1 (ζ * (Real.sin ϑ : ℂ)))
    (scalarMat1 (Real.cos ϑ : ℂ))

/-- Left unitarity of `givensRotor ϑ ζ` when `star ζ * ζ = 1`: `(givensRotor ϑ ζ)ᴴ * givensRotor ϑ ζ = 1`. -/
theorem givensRotor_conjTranspose_mul_self (ϑ : ℝ) (ζ : ℂ) (hζ : star ζ * ζ = 1) :
    (givensRotor ϑ ζ)ᴴ * givensRotor ϑ ζ = 1 := by
  have hζ_comm : ζ * star ζ = 1 := mul_comm (star ζ) ζ ▸ hζ
  have htrig := ofReal_cos_sq_add_sin_sq ϑ
  unfold givensRotor
  rw [Matrix.fromBlocks_conjTranspose, Matrix.fromBlocks_multiply]
  simp only [scalarMat1_conjTranspose, star_ofReal, star_mul, star_neg, star_star,
    scalarMat1_mul, scalarMat1_add]
  have h11 : (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) +
      (Real.sin ϑ : ℂ) * star ζ * (ζ * (Real.sin ϑ : ℂ)) = 1 := by
    calc
      (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + (Real.sin ϑ : ℂ) * star ζ * (ζ * (Real.sin ϑ : ℂ))
        = (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + (star ζ * ζ) * ((Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ)) := by ring
      _ = (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + 1 * ((Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ)) := by rw [hζ]
      _ = 1 := by rw [one_mul, htrig]
  have h12 : (Real.cos ϑ : ℂ) * (-star ζ * (Real.sin ϑ : ℂ)) +
      (Real.sin ϑ : ℂ) * star ζ * (Real.cos ϑ : ℂ) = 0 := by ring
  have h21 : (Real.sin ϑ : ℂ) * -ζ * (Real.cos ϑ : ℂ) +
      (Real.cos ϑ : ℂ) * (ζ * (Real.sin ϑ : ℂ)) = 0 := by ring
  have h22 : (Real.sin ϑ : ℂ) * -ζ * (-star ζ * (Real.sin ϑ : ℂ)) +
      (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) = 1 := by
    calc
      (Real.sin ϑ : ℂ) * -ζ * (-star ζ * (Real.sin ϑ : ℂ)) + (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ)
        = (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + (ζ * star ζ) * ((Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ)) := by ring
      _ = (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + 1 * ((Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ)) := by rw [hζ_comm]
      _ = 1 := by rw [one_mul, htrig]
  rw [h11, h12, h21, h22, scalarMat1_one, scalarMat1_zero, Matrix.fromBlocks_one]

/-- Right unitarity of `givensRotor ϑ ζ` when `star ζ * ζ = 1`: `givensRotor ϑ ζ * (givensRotor ϑ ζ)ᴴ = 1`. -/
theorem givensRotor_mul_conjTranspose_self (ϑ : ℝ) (ζ : ℂ) (hζ : star ζ * ζ = 1) :
    givensRotor ϑ ζ * (givensRotor ϑ ζ)ᴴ = 1 := by
  have hζ_comm : ζ * star ζ = 1 := mul_comm (star ζ) ζ ▸ hζ
  have htrig := ofReal_cos_sq_add_sin_sq ϑ
  unfold givensRotor
  rw [Matrix.fromBlocks_conjTranspose, Matrix.fromBlocks_multiply]
  simp only [scalarMat1_conjTranspose, star_ofReal, star_mul, star_neg, star_star,
    scalarMat1_mul, scalarMat1_add]
  have h11 : (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) +
      (-star ζ * (Real.sin ϑ : ℂ)) * ((Real.sin ϑ : ℂ) * -ζ) = 1 := by
    calc
      (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + (-star ζ * (Real.sin ϑ : ℂ)) * ((Real.sin ϑ : ℂ) * -ζ)
        = (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + (star ζ * ζ) * ((Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ)) := by ring
      _ = (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + 1 * ((Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ)) := by rw [hζ]
      _ = 1 := by rw [one_mul, htrig]
  have h12 : (Real.cos ϑ : ℂ) * ((Real.sin ϑ : ℂ) * star ζ) +
      (-star ζ * (Real.sin ϑ : ℂ)) * (Real.cos ϑ : ℂ) = 0 := by ring
  have h21 : ζ * (Real.sin ϑ : ℂ) * (Real.cos ϑ : ℂ) +
      (Real.cos ϑ : ℂ) * ((Real.sin ϑ : ℂ) * -ζ) = 0 := by ring
  have h22 : ζ * (Real.sin ϑ : ℂ) * ((Real.sin ϑ : ℂ) * star ζ) +
      (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) = 1 := by
    calc
      ζ * (Real.sin ϑ : ℂ) * ((Real.sin ϑ : ℂ) * star ζ) + (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ)
        = (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + (ζ * star ζ) * ((Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ)) := by ring
      _ = (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + 1 * ((Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ)) := by rw [hζ_comm]
      _ = 1 := by rw [one_mul, htrig]
  rw [h11, h12, h21, h22, scalarMat1_one, scalarMat1_zero, Matrix.fromBlocks_one]

/-- **Unitarity of `givensRotor`**:
    For any `ϑ : ℝ` and `ζ : ℂ` with `star ζ * ζ = 1`, `givensRotor ϑ ζ ∈ Matrix.unitaryGroup (BlockIdx 1 1) ℂ`. -/
theorem givensRotor_mem_unitaryGroup (ϑ : ℝ) (ζ : ℂ) (hζ : star ζ * ζ = 1) :
    givensRotor ϑ ζ ∈ Matrix.unitaryGroup (BlockIdx 1 1) ℂ := by
  rw [Matrix.mem_unitaryGroup_iff]
  exact givensRotor_mul_conjTranspose_self ϑ ζ hζ

/-- Every `givensRotor ϑ ζ` satisfies the Phase 5 `IsElementaryClockGate 1 1 0` predicate. -/
theorem givensRotor_isElementaryClockGate (ϑ : ℝ) (ζ : ℂ) :
    IsElementaryClockGate 1 1 0 (givensRotor ϑ ζ) := by
  intro i hi col
  fin_cases i
  exact False.elim (hi rfl)

/-- **In-Phase Constructive Clock-Mass Creation (`Δϕ = 0`, `ζ = 1`)**:
    Two sequential in-phase Givens rotations add their angles coherently:
    `givensRotor ϑ₂ 1 * givensRotor ϑ₁ 1 = givensRotor (ϑ₁ + ϑ₂) 1`. -/
theorem givensRotor_in_phase_mul (ϑ₁ ϑ₂ : ℝ) :
    givensRotor ϑ₂ 1 * givensRotor ϑ₁ 1 = givensRotor (ϑ₁ + ϑ₂) 1 := by
  have h11 : (Real.cos ϑ₂ : ℂ) * (Real.cos ϑ₁ : ℂ) + -(Real.sin ϑ₂ : ℂ) * (Real.sin ϑ₁ : ℂ) =
      (Real.cos (ϑ₁ + ϑ₂) : ℂ) := by
    norm_cast
    rw [Real.cos_add]
    ring
  have h12 : (Real.cos ϑ₂ : ℂ) * -(Real.sin ϑ₁ : ℂ) + -(Real.sin ϑ₂ : ℂ) * (Real.cos ϑ₁ : ℂ) =
      -(Real.sin (ϑ₁ + ϑ₂) : ℂ) := by
    norm_cast
    rw [Real.sin_add]
    ring
  have h21 : (Real.sin ϑ₂ : ℂ) * (Real.cos ϑ₁ : ℂ) + (Real.cos ϑ₂ : ℂ) * (Real.sin ϑ₁ : ℂ) =
      (Real.sin (ϑ₁ + ϑ₂) : ℂ) := by
    norm_cast
    rw [Real.sin_add]
    ring
  have h22 : (Real.sin ϑ₂ : ℂ) * -(Real.sin ϑ₁ : ℂ) + (Real.cos ϑ₂ : ℂ) * (Real.cos ϑ₁ : ℂ) =
      (Real.cos (ϑ₁ + ϑ₂) : ℂ) := by
    norm_cast
    rw [Real.cos_add]
    ring
  unfold givensRotor
  rw [Matrix.fromBlocks_multiply]
  simp only [star_one, one_mul, neg_one_mul, scalarMat1_mul, scalarMat1_add, h11, h12, h21, h22]

/-- **In-Phase Angle Doubling (`G₀(ϑ) G₀(ϑ) = G₀(2ϑ)`)**:
    Special case of `givensRotor_in_phase_mul` for equal pulse angles `ϑ₁ = ϑ₂ = ϑ`. -/
theorem givensRotor_in_phase_double (ϑ : ℝ) :
    givensRotor ϑ 1 * givensRotor ϑ 1 = givensRotor (2 * ϑ) 1 := by
  rw [two_mul]
  exact givensRotor_in_phase_mul ϑ ϑ

/-- **Anti-Phase Coherent Clock-Mass Annihilation / Logical Negation (`¬`) Gate (`Δϕ = π`, `ζ = -1`)**:
    When a second shock arrives with relative rotor phase `Δϕ = π` (`ζ = e^{iπ} = -1`),
    `givensRotor ϑ (-1) * givensRotor ϑ 1 = 1`, discharging 100% of the clock state back into
    the spatial mode and realizing the non-monotone negation (`¬`) gate required to evade MPCVP. -/
theorem givensRotor_anti_phase_mul_eq_one (ϑ : ℝ) :
    givensRotor ϑ (-1) * givensRotor ϑ 1 = 1 := by
  have htrig := ofReal_cos_sq_add_sin_sq ϑ
  unfold givensRotor
  rw [Matrix.fromBlocks_multiply]
  simp only [star_neg, star_one, neg_neg, one_mul, neg_one_mul, scalarMat1_mul, scalarMat1_add]
  have h11 : (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) + (Real.sin ϑ : ℂ) * (Real.sin ϑ : ℂ) = 1 := htrig
  have h12 : (Real.cos ϑ : ℂ) * -(Real.sin ϑ : ℂ) + (Real.sin ϑ : ℂ) * (Real.cos ϑ : ℂ) = 0 := by ring
  have h21 : -(Real.sin ϑ : ℂ) * (Real.cos ϑ : ℂ) + (Real.cos ϑ : ℂ) * (Real.sin ϑ : ℂ) = 0 := by ring
  have h22 : -(Real.sin ϑ : ℂ) * -(Real.sin ϑ : ℂ) + (Real.cos ϑ : ℂ) * (Real.cos ϑ : ℂ) = 1 := by
    linear_combination htrig
  rw [h11, h12, h21, h22, scalarMat1_one, scalarMat1_zero, Matrix.fromBlocks_one]

/-- **Shock Mach–Zehnder / Ramsey Interference Fringe Formula**:
    For any relative phase factor `ζ : ℂ`, the net UV-to-clock off-diagonal matrix element
    `(Sum.inr 0, Sum.inl 0)` of the two-pulse collision operator `givensRotor ϑ ζ * givensRotor ϑ 1`
    is `(1 + ζ) * sin ϑ * cos ϑ`. -/
theorem givensRotor_ramsey_fringe_entry (ϑ : ℝ) (ζ : ℂ) :
    (givensRotor ϑ ζ * givensRotor ϑ 1) (Sum.inr 0) (Sum.inl 0) =
      (1 + ζ) * (Real.sin ϑ : ℂ) * (Real.cos ϑ : ℂ) := by
  unfold givensRotor
  rw [Matrix.fromBlocks_multiply]
  simp only [star_one, scalarMat1_mul, scalarMat1_add, Matrix.fromBlocks_apply₂₁, scalarMat1_apply]
  ring

/-! ### 2. Pauli Exclusion Blockade on 1D Clock Wires -/

/-- **Pauli Saturation Reflection at `ϑ = π / 2`**:
    When a 1D clock wire is driven to full saturation by a `ϑ = π / 2` rotation, applying a
    second in-phase `π / 2` rotation (`givensRotor (π/2) 1 * givensRotor (π/2) 1 = givensRotor π 1`)
    has zero net UV-to-clock transfer (`(Sum.inr 0, Sum.inl 0) = 0`) and reflects the amplitude
    back into the spatial channel with phase `-1`. -/
theorem pauli_saturation_reflection :
    (givensRotor (Real.pi / 2) 1 * givensRotor (Real.pi / 2) 1) (Sum.inr 0) (Sum.inl 0) = 0 ∧
    (givensRotor (Real.pi / 2) 1 * givensRotor (Real.pi / 2) 1) (Sum.inl 0) (Sum.inl 0) = -1 := by
  rw [givensRotor_in_phase_mul]
  have hpi : Real.pi / 2 + Real.pi / 2 = Real.pi := add_halves Real.pi
  rw [hpi]
  unfold givensRotor
  simp [Real.sin_pi, Real.cos_pi]

/-- **Pauli Exclusion Blockade via Rank-Nullity on `BlockIdx d m`**:
    For any elementary 1D clock gate `W` satisfying `IsElementaryClockGate d m c W`,
    the forward leakage operator `forwardLeakage (bandlimitProj d m) W` has rank at most `1`,
    so its kernel has dimension at least `(d + m) - 1`:
    `d + m ≤ Module.finrank ℂ (LinearMap.ker (forwardLeakage (bandlimitProj d m) W).mulVecLin) + 1`.
    Consequently, at most a single 1D spatial channel can compress into wire `c` per step. -/
theorem pauli_exclusion_blockade_ker_finrank
    (d m : ℕ) (c : Fin m) (W : Matrix (BlockIdx d m) (BlockIdx d m) ℂ)
    (hGate : IsElementaryClockGate d m c W) :
    d + m ≤
      Module.finrank ℂ (LinearMap.ker (forwardLeakage (bandlimitProj d m) W).mulVecLin) + 1 := by
  have hRank : (forwardLeakage (bandlimitProj d m) W).rank ≤ 1 :=
    rank_forwardLeakage_le_one_of_isElementaryClockGate d m c W hGate
  have hRN := LinearMap.finrank_range_add_finrank_ker
    (forwardLeakage (bandlimitProj d m) W).mulVecLin
  have hDim : Module.finrank ℂ (BlockIdx d m → ℂ) = d + m := by
    simp [BlockIdx, Fintype.card_sum, Fintype.card_fin]
  dsimp [Matrix.rank] at hRank
  omega

/-! ### 3. Milestone 4 Tripartite 3-Mode Haar Router `(ū_b, u_{b,UV}, c_slot)` -/

/-- Index type for the 3-mode Milestone 4 subspace:
    - `Sum.inl 0`: smooth local Haar mean mode `ū_b = (u_{b_0} + u_{b_1}) / √2`
    - `Sum.inr (Sum.inl 0)`: antisymmetric Nyquist difference mode `u_{b,UV} = (u_{b_0} - u_{b_1}) / √2`
    - `Sum.inr (Sum.inr 0)`: co-moving worldline clock slot `c_slot` -/
abbrev TripartiteIdx := Fin 1 ⊕ BlockIdx 1 1

/-- Stage 1 UV Stabilization gate on `TripartiteIdx` (Monograph Eq. 16),
    rotating `(u_{b,UV}, c_slot)` via `givensRotor ϑ₁ 1` while acting as the identity on `ū_b`. -/
def stage1UVStabilizer (ϑ₁ : ℝ) : Matrix TripartiteIdx TripartiteIdx ℂ :=
  Matrix.fromBlocks 1 0 0 (givensRotor ϑ₁ 1)

/-- `stage1UVStabilizer ϑ₁` is strictly unitary (`∈ Matrix.unitaryGroup TripartiteIdx ℂ`). -/
theorem stage1UVStabilizer_mem_unitaryGroup (ϑ₁ : ℝ) :
    stage1UVStabilizer ϑ₁ ∈ Matrix.unitaryGroup TripartiteIdx ℂ := by
  rw [Matrix.mem_unitaryGroup_iff]
  change stage1UVStabilizer ϑ₁ * (stage1UVStabilizer ϑ₁)ᴴ = 1
  have hG : givensRotor ϑ₁ 1 * (givensRotor ϑ₁ 1)ᴴ = 1 :=
    givensRotor_mul_conjTranspose_self ϑ₁ 1 (by simp)
  unfold stage1UVStabilizer
  rw [Matrix.fromBlocks_conjTranspose, Matrix.fromBlocks_multiply]
  simp [hG, Matrix.fromBlocks_one]

/-- Stage 2 Coherent Collision Readback gate on `TripartiteIdx` (Monograph Eq. 17),
    rotating stored clock condensate `c_slot` (`Sum.inr (Sum.inr 0)`) into the smooth local mean
    mode `ū_b` (`Sum.inl 0`) by `θ_rec` while leaving `u_{b,UV}` (`Sum.inr (Sum.inl 0)`) untouched. -/
def stage2MeanReadback (θ_rec : ℝ) : Matrix TripartiteIdx TripartiteIdx ℂ
  | Sum.inl _, Sum.inl _ => (Real.cos θ_rec : ℂ)
  | Sum.inl _, Sum.inr (Sum.inr _) => (Real.sin θ_rec : ℂ)
  | Sum.inr (Sum.inl _), Sum.inr (Sum.inl _) => 1
  | Sum.inr (Sum.inr _), Sum.inl _ => -(Real.sin θ_rec : ℂ)
  | Sum.inr (Sum.inr _), Sum.inr (Sum.inr _) => (Real.cos θ_rec : ℂ)
  | _, _ => 0

/-- **Zero UV Spike Excitation During Stage 2 Readback**:
    Applying `stage2MeanReadback θ_rec` to any 3-mode state `v : TripartiteIdx → ℂ` leaves the
    antisymmetric Nyquist difference component `v (Sum.inr (Sum.inl 0))` (`u_{b,UV}`) completely
    unchanged, eliminating the negative-viscosity UV spike avalanche. -/
theorem stage2MeanReadback_preserves_uv_mode (θ_rec : ℝ) (v : TripartiteIdx → ℂ) :
    (stage2MeanReadback θ_rec).mulVec v (Sum.inr (Sum.inl 0)) = v (Sum.inr (Sum.inl 0)) := by
  simp [stage2MeanReadback, Matrix.mulVec, dotProduct, Fintype.sum_sum_type]

/-- `stage2MeanReadback θ_rec` is strictly unitary (`∈ Matrix.unitaryGroup TripartiteIdx ℂ`). -/
theorem stage2MeanReadback_mem_unitaryGroup (θ_rec : ℝ) :
    stage2MeanReadback θ_rec ∈ Matrix.unitaryGroup TripartiteIdx ℂ := by
  have htrig := ofReal_cos_sq_add_sin_sq θ_rec
  have hcos_star : star (Real.cos θ_rec : ℂ) = (Real.cos θ_rec : ℂ) :=
    Complex.conj_ofReal (Real.cos θ_rec)
  have hsin_star : star (Real.sin θ_rec : ℂ) = (Real.sin θ_rec : ℂ) :=
    Complex.conj_ofReal (Real.sin θ_rec)
  rw [Matrix.mem_unitaryGroup_iff]
  change stage2MeanReadback θ_rec * (stage2MeanReadback θ_rec)ᴴ = 1
  ext (i | (i | i)) (j | (j | j)) <;> fin_cases i <;> fin_cases j <;>
    simp only [Matrix.mul_apply, Matrix.conjTranspose_apply, Fintype.sum_sum_type,
      Fin.sum_univ_one, stage2MeanReadback, star_neg, star_zero, star_one,
      hcos_star, hsin_star, Matrix.one_apply, reduceCtorEq,
      Sum.inr.injEq, ite_true, ite_false, mul_zero, zero_mul,
      add_zero, zero_add, mul_neg, neg_mul, neg_neg] <;>
    first | rfl | linear_combination htrig | ring

/-- **Full Milestone 4 Tripartite Collision Router** (`Stage 2 ∘ Stage 1`) on `(ū_b, u_{b,UV}, c_slot)`:
    First applies `stage1UVStabilizer ϑ₁` to rotate overflowing Nyquist difference modes into the
    worldline clock slot, then applies `stage2MeanReadback θ_rec` to rotate stored clock amplitude
    back into the smooth local Haar mean mode `ū_b`. -/
def milestone4TripartiteRouter (ϑ₁ θ_rec : ℝ) : Matrix TripartiteIdx TripartiteIdx ℂ :=
  stage2MeanReadback θ_rec * stage1UVStabilizer ϑ₁

/-- **Strict Unitarity of the Milestone 4 Tripartite Collision Router**:
    The composite 2-stage router `milestone4TripartiteRouter ϑ₁ θ_rec` belongs to
    `Matrix.unitaryGroup TripartiteIdx ℂ`, guaranteeing exact norm preservation
    `|ū_out|² + |u_{UV,out}|² + |c_out|² = |ū_in|² + |u_{UV,in}|² + |c_in|²`. -/
theorem milestone4TripartiteRouter_mem_unitaryGroup (ϑ₁ θ_rec : ℝ) :
    milestone4TripartiteRouter ϑ₁ θ_rec ∈ Matrix.unitaryGroup TripartiteIdx ℂ :=
  Submonoid.mul_mem _ (stage2MeanReadback_mem_unitaryGroup θ_rec)
    (stage1UVStabilizer_mem_unitaryGroup ϑ₁)

/-! ### 4. Direct-Sum Spatial Norm Conservation & Apparent Shock Entropy Defect (Extension 3A) -/

/-- **Spatial Block-Compression + Clock-Leakage Gram Partition of Unity**:
    For any unitary `U` and Hermitian idempotent bandlimit projector `P`,
    `(P * U * P)ᴴ * (P * U * P) + (forwardLeakage P U)ᴴ * (forwardLeakage P U) = P`.
    This proves that any norm lost from the projected spatial flow `P U P` is stored
    without loss in the orthogonal worldline clock fiber `forwardLeakage P U = (1 - P) U P`. -/
theorem spatial_compression_add_clock_leakage_eq_proj
    {n : Type*} [Fintype n] [DecidableEq n]
    (P U : Matrix n n ℂ) (hP : P * P = P) (hP_star : Pᴴ = P) (hU : Uᴴ * U = 1) :
    (P * U * P)ᴴ * (P * U * P) + (forwardLeakage P U)ᴴ * (forwardLeakage P U) = P := by
  rw [forwardLeakage_conjTranspose_mul_self P U hP hP_star hU]
  abel

end BekensteinNyquist
