import UnitaryBandwidthDepth
import Mathlib.LinearAlgebra.UnitaryGroup

/-!
# Bekenstein–Nyquist Gauge-Covariant Clock Transport & Cayley Unitarity (Phase A — Test A3)

This module formalizes **Test A3** of the Lean 4 Physics Extension Plan (`lean_physics_extension_plan.md`),
connecting the discrete 1+1D Unitary Nyquist Rotor Lattice (`toy_model_2_nyquist_rotor_pde.py` and
`toy_model_4_shock_collision_gate.py`) and Sections 2.1 and 4.1 of the monograph to formal
matrix algebra in Lean 4:

1. **Unconditional Unitarity of the Cayley / Crank–Nicolson Integrator (`cayleyTransform_mem_unitaryGroup`)**:
   Proves that for any skew-Hermitian operator `D` (`Dᴴ = -D`) and any left resolvent `R` of
   `1 + D` (`R * (1 + D) = 1`), the Cayley transform `U = R * (1 - D)`
   satisfies `U * Uᴴ = 1` and belongs to `Matrix.unitaryGroup n ℂ`.
2. **Skew-Hermiticity of Split-Form Burgers & Gauge-Covariant Clock Advection
   (`splitFormAdvection_skewHermitian`, `covariantClockAdvection_skewHermitian`)**:
   Proves that symmetric bond weights `w i j = w j i`, antisymmetric difference stencils
   `σ j i = -σ i j`, and Hermitian-oriented complex gauge links `U_link j i = star (U_link i j)`
   yield an identically skew-Hermitian generator `Dᴴ = -D`.
3. **Gauge Invariance of Worldline Wilson Holonomy & Two-Shock Ramsey Interference
   (`twoPathHolonomy_gauge_invariant`, `wilsonLoop3_gauge_invariant`,
   `covariantClockAdvection_gauge_equivariant`)**:
   Proves that under any local `U(1)` site gauge transformation `ψ : n → ℂ` with `ψ i * star (ψ i) = 1`,
   the covariant advection operator transforms equivariantly (`G D Gᴴ`) while the closed 2-branch
   Ramsey collision holonomy `U₁(i,k) * star (U₂(i,k))` and 3-site Wilson loop are strictly
   gauge-invariant.
-/

namespace BekensteinNyquist

open Matrix

/-- Complex conjugation fixes real scalars embedded in `ℂ`. -/
lemma star_ofReal_complex (r : ℝ) : star (r : ℂ) = (r : ℂ) :=
  Complex.conj_ofReal r

/-! ### 1. Cayley Transform Unitarity for Skew-Hermitian Generators -/

/-- The Cayley transform `U = (1 + D)⁻¹ (1 - D)` represented via an explicit resolvent
    matrix `R = (1 + D)⁻¹`, i.e., `cayleyTransform D R = R * (1 - D)`. -/
def cayleyTransform {n : Type*} [Fintype n] [DecidableEq n] (D R : Matrix n n ℂ) : Matrix n n ℂ :=
  R * (1 - D)

/-- For any matrix `D`, `1 + D` and `1 - D` commute: `(1 - D) * (1 + D) = (1 + D) * (1 - D)`. -/
lemma one_sub_mul_one_add_comm {n : Type*} [Fintype n] [DecidableEq n] (D : Matrix n n ℂ) :
    (1 - D) * (1 + D) = (1 + D) * (1 - D) := by
  noncomm_ring

/-- For a skew-Hermitian matrix `D` (`Dᴴ = -D`), `(1 - D)ᴴ = 1 + D`. -/
lemma conjTranspose_one_sub_of_skewHermitian {n : Type*} [DecidableEq n]
    (D : Matrix n n ℂ) (hD : Dᴴ = -D) :
    (1 - D)ᴴ = 1 + D := by
  rw [conjTranspose_sub, conjTranspose_one, hD, sub_neg_eq_add]

/-- For a skew-Hermitian matrix `D` (`Dᴴ = -D`), `(1 + D)ᴴ = 1 - D`. -/
lemma conjTranspose_one_add_of_skewHermitian {n : Type*} [DecidableEq n]
    (D : Matrix n n ℂ) (hD : Dᴴ = -D) :
    (1 + D)ᴴ = 1 - D := by
  rw [conjTranspose_add, conjTranspose_one, hD, sub_eq_add_neg]

/-- If `R` is a left inverse of `1 + D` (`R * (1 + D) = 1`) and `D` is skew-Hermitian (`Dᴴ = -D`),
    then `Rᴴ` is a right inverse of `1 - D`: `(1 - D) * Rᴴ = 1`. -/
lemma one_sub_mul_conjTranspose_resolvent {n : Type*} [Fintype n] [DecidableEq n]
    (D R : Matrix n n ℂ) (hD : Dᴴ = -D) (hR_left : R * (1 + D) = 1) :
    (1 - D) * Rᴴ = 1 := by
  have h := congrArg conjTranspose hR_left
  rw [conjTranspose_mul, conjTranspose_one_add_of_skewHermitian D hD, conjTranspose_one] at h
  exact h

/-- **Exact Unitarity of the Cayley Transform (`U * Uᴴ = 1`)**:
    For any skew-Hermitian operator `D` (`Dᴴ = -D`) and resolvent `R` satisfying
    `R * (1 + D) = 1`, `cayleyTransform D R * (cayleyTransform D R)ᴴ = 1`. -/
theorem cayleyTransform_mul_conjTranspose_self {n : Type*} [Fintype n] [DecidableEq n]
    (D R : Matrix n n ℂ) (hD : Dᴴ = -D) (hR_left : R * (1 + D) = 1) :
    cayleyTransform D R * (cayleyTransform D R)ᴴ = 1 := by
  have hR_adj : (1 - D) * Rᴴ = 1 := one_sub_mul_conjTranspose_resolvent D R hD hR_left
  unfold cayleyTransform
  rw [conjTranspose_mul, conjTranspose_one_sub_of_skewHermitian D hD]
  calc
    R * (1 - D) * ((1 + D) * Rᴴ)
      = R * ((1 - D) * (1 + D)) * Rᴴ := by noncomm_ring
    _ = R * ((1 + D) * (1 - D)) * Rᴴ := by rw [one_sub_mul_one_add_comm D]
    _ = (R * (1 + D)) * ((1 - D) * Rᴴ) := by noncomm_ring
    _ = 1 * 1 := by rw [hR_left, hR_adj]
    _ = 1 := by noncomm_ring

/-- **Membership of the Cayley Transform in `Matrix.unitaryGroup n ℂ`**:
    Guarantees that the Crank–Nicolson / Cayley time-stepping operator
    `U(Δt) = (I + (Δt/2) D_w(u))⁻¹ (I - (Δt/2) D_w(u))` in Milestones 2 and 4 preserves the
    discrete L² norm identically for every velocity field `u` and step size `Δt`. -/
theorem cayleyTransform_mem_unitaryGroup {n : Type*} [Fintype n] [DecidableEq n]
    (D R : Matrix n n ℂ) (hD : Dᴴ = -D) (hR_left : R * (1 + D) = 1) :
    cayleyTransform D R ∈ Matrix.unitaryGroup n ℂ := by
  rw [Matrix.mem_unitaryGroup_iff]
  exact cayleyTransform_mul_conjTranspose_self D R hD hR_left

/-! ### 2. Skew-Hermiticity of Split-Form Burgers & Gauge-Covariant Clock Advection -/

/-- The discrete split-form Burgers advection matrix `D_{i,j} = w_{i,j} * σ_{i,j}`,
    where `w i j = (u_i + u_j)/3` is a symmetric bond weight and `σ i j` is an antisymmetric
    derivative stencil (`σ j i = -σ i j`). -/
def splitFormAdvection {n : Type*} (w σ : n → n → ℝ) : Matrix n n ℂ :=
  Matrix.of (fun i j => ((w i j * σ i j : ℝ) : ℂ))

/-- **Skew-Hermiticity of the Split-Form Burgers Operator**:
    Whenever the velocity bond weight `w` is symmetric (`w i j = w j i`) and the finite-difference
    or spectral derivative kernel `σ` is antisymmetric (`σ j i = -σ i j`), the split-form
    advection matrix `splitFormAdvection w σ` satisfies `Dᴴ = -D`. -/
theorem splitFormAdvection_skewHermitian {n : Type*} (w σ : n → n → ℝ)
    (hw : ∀ i j, w i j = w j i) (hσ : ∀ i j, σ j i = -σ i j) :
    (splitFormAdvection w σ)ᴴ = -splitFormAdvection w σ := by
  ext i j
  rw [Matrix.conjTranspose_apply, Matrix.neg_apply]
  simp only [splitFormAdvection, of_apply, star_ofReal_complex]
  rw [hw j i, hσ i j]
  push_cast
  ring

/-- The gauge-covariant clock parallel-transport advection matrix (Monograph Section 2.1):
    `D^cov_{i,j} = (w_{i,j} * σ_{i,j} : ℂ) * U_link i j`,
    where `U_link i j = exp(i θ_{i → j})` is the complex parallel-transport link phase along
    bond `(j → i)`. -/
def covariantClockAdvection {n : Type*} (w σ : n → n → ℝ) (U_link : n → n → ℂ) : Matrix n n ℂ :=
  Matrix.of (fun i j => ((w i j * σ i j : ℝ) : ℂ) * U_link i j)

/-- **Skew-Hermiticity of Gauge-Covariant Clock Advection**:
    For symmetric bond weights (`w i j = w j i`), antisymmetric derivative stencil (`σ j i = -σ i j`),
    and Hermitian-oriented gauge links (`U_link j i = star (U_link i j)`), the covariant
    clock transport generator satisfies `(covariantClockAdvection w σ U_link)ᴴ = -covariantClockAdvection w σ U_link`. -/
theorem covariantClockAdvection_skewHermitian {n : Type*} (w σ : n → n → ℝ) (U_link : n → n → ℂ)
    (hw : ∀ i j, w i j = w j i) (hσ : ∀ i j, σ j i = -σ i j)
    (hLink : ∀ i j, U_link j i = star (U_link i j)) :
    (covariantClockAdvection w σ U_link)ᴴ = -covariantClockAdvection w σ U_link := by
  ext i j
  rw [Matrix.conjTranspose_apply, Matrix.neg_apply]
  simp only [covariantClockAdvection, of_apply, star_mul', star_ofReal_complex]
  rw [hw j i, hσ i j, hLink i j, star_star]
  push_cast
  ring

/-! ### 3. Gauge Invariance of Worldline Wilson Holonomy & Two-Shock Ramsey Interference -/

/-- Local `U(1)` gauge transformation of a lattice link variable `U_link i j` by site phases
    `ψ : n → ℂ`: `U'_{i,j} = ψ_i * U_{i,j} * star (ψ_j)`. -/
def gaugeTransformLink {n : Type*} (ψ : n → ℂ) (U_link : n → n → ℂ) (i j : n) : ℂ :=
  ψ i * U_link i j * star (ψ j)

/-- **Two-Shock Collision / Ramsey Interference Holonomy Gauge Invariance**:
    When two colliding shock worldlines propagate from source site `i` to collision site `k`
    along two branches with holonomies `U₁ i k` and `U₂ i k`, their Ramsey interference cross-term
    `U₁ i k * star (U₂ i k)` is strictly invariant under any local `U(1)` gauge transformation
    `ψ : n → ℂ` satisfying `ψ x * star (ψ x) = 1`. -/
theorem twoPathHolonomy_gauge_invariant {n : Type*} (ψ : n → ℂ) (U₁ U₂ : n → n → ℂ)
    (hψ : ∀ x, ψ x * star (ψ x) = 1) (i k : n) :
    gaugeTransformLink ψ U₁ i k * star (gaugeTransformLink ψ U₂ i k) =
      U₁ i k * star (U₂ i k) := by
  have hi : ψ i * star (ψ i) = 1 := hψ i
  have hk : ψ k * star (ψ k) = 1 := hψ k
  unfold gaugeTransformLink
  simp only [star_mul', star_star]
  calc
    ψ i * U₁ i k * star (ψ k) * (star (ψ i) * star (U₂ i k) * ψ k)
      = (ψ i * star (ψ i)) * (ψ k * star (ψ k)) * (U₁ i k * star (U₂ i k)) := by ring
    _ = 1 * 1 * (U₁ i k * star (U₂ i k)) := by rw [hi, hk]
    _ = U₁ i k * star (U₂ i k) := by ring

/-- **Closed 3-Site Wilson Loop Gauge Invariance**:
    For any closed 3-site plaquette `(i → j → k → i)`, the Wilson loop holonomy
    `U_{i,j} * U_{j,k} * U_{k,i}` is strictly invariant under any local `U(1)` gauge
    transformation `ψ : n → ℂ` satisfying `ψ x * star (ψ x) = 1`. -/
theorem wilsonLoop3_gauge_invariant {n : Type*} (ψ : n → ℂ) (U_link : n → n → ℂ)
    (hψ : ∀ x, ψ x * star (ψ x) = 1) (i j k : n) :
    gaugeTransformLink ψ U_link i j *
      gaugeTransformLink ψ U_link j k *
      gaugeTransformLink ψ U_link k i =
      U_link i j * U_link j k * U_link k i := by
  have hi : ψ i * star (ψ i) = 1 := hψ i
  have hj : ψ j * star (ψ j) = 1 := hψ j
  have hk : ψ k * star (ψ k) = 1 := hψ k
  unfold gaugeTransformLink
  calc
    ψ i * U_link i j * star (ψ j) * (ψ j * U_link j k * star (ψ k)) *
        (ψ k * U_link k i * star (ψ i))
      = (ψ i * star (ψ i)) * (ψ j * star (ψ j)) * (ψ k * star (ψ k)) *
          (U_link i j * U_link j k * U_link k i) := by ring
    _ = 1 * 1 * 1 * (U_link i j * U_link j k * U_link k i) := by rw [hi, hj, hk]
    _ = U_link i j * U_link j k * U_link k i := by ring

/-- Conjugate transpose of a diagonal complex matrix. -/
lemma conjTranspose_diagonal_complex {n : Type*} [DecidableEq n] (ψ : n → ℂ) :
    (Matrix.diagonal ψ)ᴴ = Matrix.diagonal (fun i => star (ψ i)) := by
  ext i j
  simp only [conjTranspose_apply, diagonal_apply]
  by_cases h : i = j
  · subst h; simp
  · have hji : j ≠ i := Ne.symm h
    simp [h, hji]

/-- **Gauge Equivariance of the Covariant Clock Advection Matrix**:
    Under a local site gauge transformation `U_link ↦ gaugeTransformLink ψ U_link`, the matrix
    `covariantClockAdvection w σ` transforms by unitary conjugation with `diagonal ψ`:
    `covariantClockAdvection w σ (gaugeTransformLink ψ U_link) = diagonal ψ * covariantClockAdvection w σ U_link * (diagonal ψ)ᴴ`. -/
theorem covariantClockAdvection_gauge_equivariant {n : Type*} [Fintype n] [DecidableEq n]
    (w σ : n → n → ℝ) (U_link : n → n → ℂ) (ψ : n → ℂ) :
    covariantClockAdvection w σ (gaugeTransformLink ψ U_link) =
      Matrix.diagonal ψ * covariantClockAdvection w σ U_link * (Matrix.diagonal ψ)ᴴ := by
  ext i j
  rw [conjTranspose_diagonal_complex]
  simp only [Matrix.mul_diagonal, Matrix.diagonal_mul, covariantClockAdvection,
    of_apply, gaugeTransformLink]
  ring

end BekensteinNyquist
