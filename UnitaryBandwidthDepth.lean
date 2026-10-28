import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.UnitaryGroup
import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.RCLike.Basic

/-!
# Phase 5: Finite-Dimensional Unitary Bandwidth-Depth Tradeoff (`UnitaryBandwidthDepth.lean`)

Formalizes the finite-dimensional kinematic core of the Bekenstein–Nyquist Shock Worldline
Conjecture (Phase 1 & Phase 5):
1. Canonical block-partitioned index space `BlockIdx d (N - d) := Fin d ⊕ Fin (N - d)` and equivalence
   with `Fin N` via `finSumFinEquiv` and `Matrix.reindex`.
2. Orthogonal bandlimit projector `P_Lambda = Matrix.fromBlocks 1 0 0 0` onto the `d`-dimensional
   low-frequency Bekenstein subspace `V_Lambda`, with ultraviolet complement `I - P_Lambda = Matrix.fromBlocks 0 0 0 1`.
3. `IsElementaryClockGate`: formal predicate for an elementary unitary gate leaking amplitude from `V_Lambda`
   into at most a single 1D clock wire `c ∈ Fin (N - d)`, and proof that `IsElementaryClockGate → rank ≤ 1`.
4. Unitary Gram identities, forward/reverse leakage rank equality `rank(A) = rank(B) = r`, and exact
   commutator rank identity `rank([P_Lambda, U]) = rank(A) + rank(B) = 2r`.
5. Complete inductive proofs of `unitary_bandwidth_depth_tradeoff` (isometric agreement) and
   `unitary_bandwidth_depth_tradeoff_compression` (block-compression dilation agreement).
-/

noncomputable section

open Matrix Module
open scoped ComplexOrder

namespace BekensteinNyquist

/-- Canonical block-partitioned index type separating `d` low-frequency Bekenstein modes
    from `m = N - d` ultraviolet orthogonal modes. -/
abbrev BlockIdx (d m : ℕ) := Fin d ⊕ Fin m

/-- Equivalence reindexing a block-partitioned matrix on `Fin d ⊕ Fin (N - d)` to `Fin (d + (N - d))`
    via `finSumFinEquiv`. -/
def reindexToFin (d m : ℕ) (M : Matrix (BlockIdx d m) (BlockIdx d m) ℂ) :
    Matrix (Fin (d + m)) (Fin (d + m)) ℂ :=
  Matrix.reindex finSumFinEquiv finSumFinEquiv M

/-- Orthogonal bandlimit projector `P_Lambda = diag(I_d, 0_m)` constructed via `Matrix.fromBlocks`. -/
def bandlimitProj (d m : ℕ) : Matrix (BlockIdx d m) (BlockIdx d m) ℂ :=
  Matrix.fromBlocks (1 : Matrix (Fin d) (Fin d) ℂ) 0 0 (0 : Matrix (Fin m) (Fin m) ℂ)

/-- Ultraviolet complement projector `I - P_Lambda = diag(0_d, I_m)` via `Matrix.fromBlocks`. -/
def uvProj (d m : ℕ) : Matrix (BlockIdx d m) (BlockIdx d m) ℂ :=
  Matrix.fromBlocks (0 : Matrix (Fin d) (Fin d) ℂ) 0 0 (1 : Matrix (Fin m) (Fin m) ℂ)

/-- `bandlimitProj` and `uvProj` sum to the identity matrix. -/
theorem bandlimitProj_add_uvProj (d m : ℕ) :
    bandlimitProj d m + uvProj d m = (1 : Matrix (BlockIdx d m) (BlockIdx d m) ℂ) := by
  unfold bandlimitProj uvProj
  rw [← Matrix.fromBlocks_one]
  simp [Matrix.fromBlocks_add]

/-- `uvProj` is literally `1 - bandlimitProj`. -/
theorem uvProj_eq_one_sub_bandlimitProj (d m : ℕ) :
    uvProj d m = 1 - bandlimitProj d m := by
  rw [← bandlimitProj_add_uvProj d m, add_sub_cancel_left]

/-- `bandlimitProj` is idempotent (`P^2 = P`). -/
theorem bandlimitProj_mul_self (d m : ℕ) :
    bandlimitProj d m * bandlimitProj d m = bandlimitProj d m := by
  unfold bandlimitProj
  simp [Matrix.fromBlocks_multiply]

/-- `bandlimitProj` is Hermitian (`Pᴴ = P`). -/
theorem bandlimitProj_conjTranspose (d m : ℕ) :
    (bandlimitProj d m)ᴴ = bandlimitProj d m := by
  unfold bandlimitProj
  simp [Matrix.fromBlocks_conjTranspose]

section GenericIndex

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Forward Nyquist leakage matrix `A = (1 - P) * U * P`. -/
def forwardLeakage (P U : Matrix n n ℂ) : Matrix n n ℂ :=
  (1 - P) * U * P

/-- Reverse Nyquist leakage matrix `B = P * U * (1 - P)`. -/
def reverseLeakage (P U : Matrix n n ℂ) : Matrix n n ℂ :=
  P * U * (1 - P)

/-- Matrix commutator `[P, U] = P * U - U * P`. -/
def commutator (P U : Matrix n n ℂ) : Matrix n n ℂ :=
  P * U - U * P

/-- The commutator decomposes into `[P, U] = P * U * (1 - P) - (1 - P) * U * P = B - A`. -/
theorem commutator_eq_reverse_sub_forward (P U : Matrix n n ℂ) :
    commutator P U = reverseLeakage P U - forwardLeakage P U := by
  unfold commutator reverseLeakage forwardLeakage
  noncomm_ring

omit [DecidableEq n] in
/-- Subadditivity of matrix rank over `ℂ`: `rank(A + B) ≤ rank(A) + rank(B)`. -/
theorem matrix_rank_add_le (A B : Matrix n n ℂ) :
    (A + B).rank ≤ A.rank + B.rank := by
  dsimp [Matrix.rank]
  rw [Matrix.mulVecLin_add]
  have h1 := Submodule.finrank_mono (LinearMap.range_add_le A.mulVecLin B.mulVecLin)
  have h2 := Submodule.finrank_sup_add_finrank_inf_eq
    (LinearMap.range A.mulVecLin) (LinearMap.range B.mulVecLin)
  omega

/-- Helper: `(1 - P) * (1 - P) = 1 - P` when `P * P = P`. -/
theorem one_sub_idem {P : Matrix n n ℂ} (hP : P * P = P) :
    (1 - P) * (1 - P) = 1 - P := by
  calc
    (1 - P) * (1 - P) = 1 - P - P + P * P := by noncomm_ring
    _ = 1 - P - P + P := by rw [hP]
    _ = 1 - P := by abel

/-- One-sided leakage decomposition identity:
    `(1 - P) * (W * U) * P = ((1 - P) * W * P) * (P * U * P) + ((1 - P) * W * (1 - P)) * ((1 - P) * U * P)`. -/
theorem one_sided_leakage_decomposition
    (P W U : Matrix n n ℂ) (hP : P * P = P) :
    forwardLeakage P (W * U) =
      (forwardLeakage P W) * (P * U * P) +
      ((1 - P) * W * (1 - P)) * (forwardLeakage P U) := by
  have hQ := one_sub_idem hP
  unfold forwardLeakage
  calc
    (1 - P) * (W * U) * P
      = (1 - P) * W * 1 * U * P := by simp [Matrix.mul_assoc]
    _ = (1 - P) * W * (P * P + (1 - P) * (1 - P)) * U * P := by
      rw [hP, hQ, add_sub_cancel]
    _ = ((1 - P) * W * P) * (P * U * P) + ((1 - P) * W * (1 - P)) * ((1 - P) * U * P) := by
      noncomm_ring

/-- Single-step rank subadditivity for the one-sided leakage decomposition. -/
theorem forwardLeakage_mul_rank_le
    (P W U : Matrix n n ℂ) (hP : P * P = P) :
    (forwardLeakage P (W * U)).rank ≤
      (forwardLeakage P W).rank + (forwardLeakage P U).rank := by
  rw [one_sided_leakage_decomposition P W U hP]
  refine le_trans (matrix_rank_add_le _ _) ?_
  exact add_le_add (Matrix.rank_mul_le_left _ _) (Matrix.rank_mul_le_right _ _)

/-! ### Commutator Rank Additivity and Unitary Gram / Dilation Identities -/

/-- Exact range decomposition of the commutator `[P, U] = B - A`:
    `range([P, U]) = range(A) ⊔ range(B)`. -/
theorem commutator_range_eq_sup (P U : Matrix n n ℂ) (hP : P * P = P) :
    LinearMap.range (commutator P U).mulVecLin =
      LinearMap.range (forwardLeakage P U).mulVecLin ⊔
      LinearMap.range (reverseLeakage P U).mulVecLin := by
  apply le_antisymm
  · rintro y ⟨x, rfl⟩
    have hsplit : (commutator P U).mulVecLin x =
        (forwardLeakage P U).mulVecLin (-x) + (reverseLeakage P U).mulVecLin x := by
      ext i
      rw [commutator_eq_reverse_sub_forward]
      simp [Matrix.mulVec_neg, sub_eq_neg_add]
    rw [hsplit]
    refine Submodule.add_mem_sup ?_ ?_
    · exact ⟨-x, rfl⟩
    · exact ⟨x, rfl⟩
  · refine sup_le ?_ ?_
    · rintro y ⟨x, rfl⟩
      have hCP : (commutator P U) * (-P) = forwardLeakage P U := by
        unfold commutator forwardLeakage
        calc
          (P * U - U * P) * (-P) = -P * U * P + U * (P * P) := by noncomm_ring
          _ = -P * U * P + U * P := by rw [hP]
          _ = (1 - P) * U * P := by noncomm_ring
      refine ⟨(-P).mulVecLin x, ?_⟩
      rw [← LinearMap.comp_apply, ← Matrix.mulVecLin_mul, hCP]
    · rintro y ⟨x, rfl⟩
      have hCQ : (commutator P U) * (1 - P) = reverseLeakage P U := by
        unfold commutator reverseLeakage
        calc
          (P * U - U * P) * (1 - P) = P * U * (1 - P) - U * (P - P * P) := by noncomm_ring
          _ = P * U * (1 - P) - U * (P - P) := by rw [hP]
          _ = P * U * (1 - P) := by noncomm_ring
      refine ⟨(1 - P).mulVecLin x, ?_⟩
      rw [← LinearMap.comp_apply, ← Matrix.mulVecLin_mul, hCQ]

/-- Orthogonal image intersection: `range(A) ⊓ range(B) = ⊥` when `P * P = P`. -/
theorem forwardLeakage_inf_reverseLeakage_range_eq_bot (P U : Matrix n n ℂ) (hP : P * P = P) :
    LinearMap.range (forwardLeakage P U).mulVecLin ⊓
      LinearMap.range (reverseLeakage P U).mulVecLin = ⊥ := by
  rw [eq_bot_iff]
  rintro y ⟨⟨xA, hxA⟩, ⟨xB, hxB⟩⟩
  rw [Submodule.mem_bot]
  have hPA : P * forwardLeakage P U = 0 := by
    unfold forwardLeakage
    calc
      P * ((1 - P) * U * P) = (P - P * P) * U * P := by noncomm_ring
      _ = (P - P) * U * P := by rw [hP]
      _ = 0 := by noncomm_ring
  have hPB : P * reverseLeakage P U = reverseLeakage P U := by
    unfold reverseLeakage
    calc
      P * (P * U * (1 - P)) = (P * P) * U * (1 - P) := by noncomm_ring
      _ = P * U * (1 - P) := by rw [hP]
  calc
    y = (reverseLeakage P U).mulVecLin xB := hxB.symm
    _ = (P * reverseLeakage P U).mulVecLin xB := by rw [hPB]
    _ = P.mulVecLin ((reverseLeakage P U).mulVecLin xB) := by
      rw [Matrix.mulVecLin_mul, LinearMap.comp_apply]
    _ = P.mulVecLin y := by rw [hxB]
    _ = P.mulVecLin ((forwardLeakage P U).mulVecLin xA) := by rw [← hxA]
    _ = (P * forwardLeakage P U).mulVecLin xA := by
      rw [Matrix.mulVecLin_mul, LinearMap.comp_apply]
    _ = 0 := by rw [hPA, Matrix.mulVecLin_zero, LinearMap.zero_apply]

/-- For any idempotent projector `P * P = P` and any matrix `U`, the rank of the commutator
    equals the sum of the forward and reverse leakage ranks:
    `rank([P, U]) = rank(A) + rank(B)`. -/
theorem commutator_rank_eq_add (P U : Matrix n n ℂ) (hP : P * P = P) :
    (commutator P U).rank = (forwardLeakage P U).rank + (reverseLeakage P U).rank := by
  dsimp [Matrix.rank]
  rw [commutator_range_eq_sup P U hP]
  have hsup := Submodule.finrank_sup_add_finrank_inf_eq
    (LinearMap.range (forwardLeakage P U).mulVecLin)
    (LinearMap.range (reverseLeakage P U).mulVecLin)
  rw [forwardLeakage_inf_reverseLeakage_range_eq_bot P U hP, finrank_bot, add_zero] at hsup
  exact hsup

/-- Unitary off-diagonal Gram identity in projector form:
    `Aᴴ * A = P - (P * U * P)ᴴ * (P * U * P)` for any unitary `U` and Hermitian projector `P`. -/
theorem forwardLeakage_conjTranspose_mul_self
    (P U : Matrix n n ℂ) (hP : P * P = P) (hP_star : Pᴴ = P) (hU : Uᴴ * U = 1) :
    (forwardLeakage P U)ᴴ * (forwardLeakage P U) = P - (P * U * P)ᴴ * (P * U * P) := by
  have hQ := one_sub_idem hP
  have hQ_star : (1 - P)ᴴ = 1 - P := by rw [conjTranspose_sub, conjTranspose_one, hP_star]
  have hA_star : (forwardLeakage P U)ᴴ = P * Uᴴ * (1 - P) := by
    unfold forwardLeakage
    simp only [conjTranspose_mul, hP_star, hQ_star, Matrix.mul_assoc]
  have hPUP_star : (P * U * P)ᴴ = P * Uᴴ * P := by
    simp only [conjTranspose_mul, hP_star, Matrix.mul_assoc]
  rw [hA_star, hPUP_star]
  unfold forwardLeakage
  set W := Uᴴ
  calc
    P * W * (1 - P) * ((1 - P) * U * P)
      = P * W * ((1 - P) * (1 - P)) * U * P := by noncomm_ring
    _ = P * W * (1 - P) * U * P := by rw [hQ]
    _ = P * (W * U) * P - P * W * P * U * P := by noncomm_ring
    _ = P * 1 * P - P * W * (P * P) * U * P := by rw [hU, hP]
    _ = P * P - P * W * P * (P * U * P) := by noncomm_ring
    _ = P - P * W * P * (P * U * P) := by rw [hP]

/-- Kernel dimension inequality between `(1 - P) U P` and `(1 - P) Uᴴ P` when `Uᴴ U = 1`. -/
theorem finrank_ker_forwardLeakage_le
    (P U : Matrix n n ℂ) (hP : P * P = P) (hU : Uᴴ * U = 1) :
    Module.finrank ℂ (LinearMap.ker (forwardLeakage P U).mulVecLin) ≤
      Module.finrank ℂ (LinearMap.ker (forwardLeakage P Uᴴ).mulVecLin) := by
  have hQ := one_sub_idem hP
  let M : Matrix n n ℂ := U * P + (1 - P)
  let N : Matrix n n ℂ := Uᴴ * P + (1 - P)
  have hBM : ∀ x ∈ LinearMap.ker (forwardLeakage P U).mulVecLin,
      M.mulVecLin x ∈ LinearMap.ker (forwardLeakage P Uᴴ).mulVecLin := by
    intro x hx
    rw [LinearMap.mem_ker] at hx ⊢
    have hSplit : U * P = P * U * P + forwardLeakage P U := by
      unfold forwardLeakage
      noncomm_ring
    have hUP : (P * U * P).mulVecLin x = (U * P).mulVecLin x := by
      rw [hSplit, Matrix.mulVecLin_add, LinearMap.add_apply, hx, add_zero]
    have hProd : forwardLeakage P Uᴴ * M = (1 - P) * Uᴴ * (P * U * P) := by
      unfold forwardLeakage M
      calc
        (1 - P) * Uᴴ * P * (U * P + (1 - P))
          = (1 - P) * Uᴴ * (P * U * P) + (1 - P) * Uᴴ * (P - P * P) := by noncomm_ring
        _ = (1 - P) * Uᴴ * (P * U * P) + (1 - P) * Uᴴ * (P - P) := by rw [hP]
        _ = (1 - P) * Uᴴ * (P * U * P) := by noncomm_ring
    calc
      (forwardLeakage P Uᴴ).mulVecLin (M.mulVecLin x)
        = (forwardLeakage P Uᴴ * M).mulVecLin x := by
          rw [Matrix.mulVecLin_mul, LinearMap.comp_apply]
      _ = ((1 - P) * Uᴴ * (P * U * P)).mulVecLin x := by rw [hProd]
      _ = ((1 - P) * Uᴴ).mulVecLin ((P * U * P).mulVecLin x) := by
        rw [Matrix.mulVecLin_mul, LinearMap.comp_apply]
      _ = ((1 - P) * Uᴴ).mulVecLin ((U * P).mulVecLin x) := by rw [hUP]
      _ = ((1 - P) * (Uᴴ * U) * P).mulVecLin x := by
        rw [← LinearMap.comp_apply, ← Matrix.mulVecLin_mul]
        congr 1
        noncomm_ring
      _ = ((1 - P) * P).mulVecLin x := by rw [hU, Matrix.mul_one]
      _ = 0 := by
        have hQP : (1 - P) * P = 0 := by
          rw [Matrix.sub_mul, Matrix.one_mul, hP, sub_self]
        rw [hQP, Matrix.mulVecLin_zero, LinearMap.zero_apply]
  have hNM : ∀ x ∈ LinearMap.ker (forwardLeakage P U).mulVecLin,
      N.mulVecLin (M.mulVecLin x) = x := by
    intro x hx
    rw [LinearMap.mem_ker] at hx
    have hQUP : ((1 - P) * U * P).mulVecLin x = 0 := hx
    have hId : N * M = 1 + (1 - Uᴴ) * ((1 - P) * U * P) := by
      dsimp [M, N]
      calc
        (Uᴴ * P + (1 - P)) * (U * P + (1 - P))
          = Uᴴ * (P * U * P) + Uᴴ * (P - P * P) + (1 - P) * U * P + (1 - P) * (1 - P) := by
            noncomm_ring
        _ = Uᴴ * (U * P - (1 - P) * U * P) + Uᴴ * (P - P) + (1 - P) * U * P + (1 - P) := by
          rw [hP, hQ]
          congr 3
          noncomm_ring
        _ = (Uᴴ * U) * P + (1 - P) + (1 - Uᴴ) * ((1 - P) * U * P) := by noncomm_ring
        _ = 1 + (1 - Uᴴ) * ((1 - P) * U * P) := by
          rw [hU, Matrix.one_mul, add_sub_cancel]
    calc
      N.mulVecLin (M.mulVecLin x)
        = (N * M).mulVecLin x := by rw [Matrix.mulVecLin_mul, LinearMap.comp_apply]
      _ = (1 + (1 - Uᴴ) * ((1 - P) * U * P)).mulVecLin x := by rw [hId]
      _ = x + (1 - Uᴴ).mulVecLin (((1 - P) * U * P).mulVecLin x) := by
        rw [Matrix.mulVecLin_add, LinearMap.add_apply, Matrix.mulVecLin_one, LinearMap.id_apply,
          Matrix.mulVecLin_mul, LinearMap.comp_apply]
      _ = x := by rw [hQUP, LinearMap.map_zero, add_zero]
  let f : LinearMap.ker (forwardLeakage P U).mulVecLin →ₗ[ℂ]
      LinearMap.ker (forwardLeakage P Uᴴ).mulVecLin :=
    { toFun := fun x => ⟨M.mulVecLin x.1, hBM x.1 x.2⟩
      map_add' := fun x y => Subtype.ext (LinearMap.map_add M.mulVecLin x.1 y.1)
      map_smul' := fun c x => Subtype.ext (LinearMap.map_smul M.mulVecLin c x.1) }
  have hf_inj : Function.Injective f := by
    intro x y hxy
    apply Subtype.ext
    have hval : M.mulVecLin x.1 = M.mulVecLin y.1 := congrArg Subtype.val hxy
    calc
      x.1 = N.mulVecLin (M.mulVecLin x.1) := (hNM x.1 x.2).symm
      _ = N.mulVecLin (M.mulVecLin y.1) := by rw [hval]
      _ = y.1 := hNM y.1 y.2
  exact LinearMap.finrank_le_finrank_of_injective hf_inj

/-- For any unitary matrix `U ∈ unitaryGroup n ℂ` and any Hermitian projector `P`,
    the forward and reverse Nyquist leakage matrices have equal rank: `rank(A) = rank(B)`. -/
theorem forwardLeakage_rank_eq_reverseLeakage_rank
    (P : Matrix n n ℂ) (hP : P * P = P) (hP_star : Pᴴ = P)
    (U : Matrix.unitaryGroup n ℂ) :
    (forwardLeakage P U.val).rank = (reverseLeakage P U.val).rank := by
  have hU_left : U.valᴴ * U.val = 1 := Matrix.mem_unitaryGroup_iff'.mp U.prop
  have hU_right : U.val * U.valᴴ = 1 := Matrix.mem_unitaryGroup_iff.mp U.prop
  have hU_right' : (U.valᴴ)ᴴ * U.valᴴ = 1 := by rwa [conjTranspose_conjTranspose]
  have hKer1 := finrank_ker_forwardLeakage_le P U.val hP hU_left
  have hKer2 := finrank_ker_forwardLeakage_le P U.valᴴ hP hU_right'
  rw [conjTranspose_conjTranspose] at hKer2
  have hKerEq : Module.finrank ℂ (LinearMap.ker (forwardLeakage P U.val).mulVecLin) =
      Module.finrank ℂ (LinearMap.ker (forwardLeakage P U.valᴴ).mulVecLin) :=
    le_antisymm hKer1 hKer2
  have hRN1 := LinearMap.finrank_range_add_finrank_ker (forwardLeakage P U.val).mulVecLin
  have hRN2 := LinearMap.finrank_range_add_finrank_ker (forwardLeakage P U.valᴴ).mulVecLin
  have hRankAdj : (forwardLeakage P U.val).rank = (forwardLeakage P U.valᴴ).rank := by
    dsimp [Matrix.rank]
    omega
  have hRevAdj : (reverseLeakage P U.val)ᴴ = forwardLeakage P U.valᴴ := by
    unfold reverseLeakage forwardLeakage
    have hQ_star : (1 - P)ᴴ = 1 - P := by rw [conjTranspose_sub, conjTranspose_one, hP_star]
    simp only [conjTranspose_mul, hP_star, hQ_star, Matrix.mul_assoc]
  rw [← Matrix.rank_conjTranspose (reverseLeakage P U.val), hRevAdj, hRankAdj]

/-- **Commutator Rank Identity (`rank([P, U]) = 2r`)**:
    For any unitary `U ∈ unitaryGroup n ℂ` and Hermitian projector `P`,
    `rank([P, U]) = 2 * rank((1 - P) U P)`. -/
theorem commutator_rank_eq_two_mul_forwardLeakage_rank
    (P : Matrix n n ℂ) (hP : P * P = P) (hP_star : Pᴴ = P)
    (U : Matrix.unitaryGroup n ℂ) :
    (commutator P U.val).rank = 2 * (forwardLeakage P U.val).rank := by
  rw [commutator_rank_eq_add P U.val hP,
    ← forwardLeakage_rank_eq_reverseLeakage_rank P hP hP_star U, two_mul]

/-- **Dilation Rank Invariance under Block Compression**:
    If two unitary matrices `U₁, U₂ ∈ unitaryGroup n ℂ` have the same low-frequency block
    compression `P * U₁ * P = P * U₂ * P`, then their forward Nyquist leakage matrices have
    identical rank: `rank((1 - P) U₁ P) = rank((1 - P) U₂ P)`. -/
theorem forwardLeakage_rank_eq_of_compression_eq
    (P : Matrix n n ℂ) (hP : P * P = P) (hP_star : Pᴴ = P)
    (U₁ U₂ : Matrix.unitaryGroup n ℂ)
    (hComp : P * U₁.val * P = P * U₂.val * P) :
    (forwardLeakage P U₁.val).rank = (forwardLeakage P U₂.val).rank := by
  have hU₁ : U₁.valᴴ * U₁.val = 1 := Matrix.mem_unitaryGroup_iff'.mp U₁.prop
  have hU₂ : U₂.valᴴ * U₂.val = 1 := Matrix.mem_unitaryGroup_iff'.mp U₂.prop
  rw [← Matrix.rank_conjTranspose_mul_self (forwardLeakage P U₁.val),
    ← Matrix.rank_conjTranspose_mul_self (forwardLeakage P U₂.val),
    forwardLeakage_conjTranspose_mul_self P U₁.val hP hP_star hU₁,
    forwardLeakage_conjTranspose_mul_self P U₂.val hP hP_star hU₂,
    hComp]

/-! ### Sequential Unitary Circuits and Inductive Bandwidth-Depth Tradeoff -/

/-- Sequential product of a list of unitary matrices `L : List (Matrix.unitaryGroup n ℂ)`. -/
def circuitProduct (L : List (Matrix.unitaryGroup n ℂ)) : Matrix n n ℂ :=
  (L.map Subtype.val).prod

@[simp]
theorem circuitProduct_nil :
    circuitProduct ([] : List (Matrix.unitaryGroup n ℂ)) = 1 := rfl

@[simp]
theorem circuitProduct_cons (W : Matrix.unitaryGroup n ℂ) (L : List (Matrix.unitaryGroup n ℂ)) :
    circuitProduct (W :: L) = W.val * circuitProduct L := rfl

/-- `circuitProduct L` is itself a member of `Matrix.unitaryGroup n ℂ`. -/
theorem circuitProduct_mem_unitaryGroup (L : List (Matrix.unitaryGroup n ℂ)) :
    circuitProduct L ∈ Matrix.unitaryGroup n ℂ := by
  induction L with
  | nil =>
    rw [circuitProduct_nil]
    exact Submonoid.one_mem (Matrix.unitaryGroup n ℂ)
  | cons W tail ih =>
    rw [circuitProduct_cons]
    exact Submonoid.mul_mem (Matrix.unitaryGroup n ℂ) W.prop ih

/-- Base case: the identity unitary has zero forward leakage across an idempotent projector `P`. -/
theorem forwardLeakage_one_rank_eq_zero (P : Matrix n n ℂ) (hP : P * P = P) :
    (forwardLeakage P 1).rank = 0 := by
  have hzero : forwardLeakage P 1 = 0 := by
    unfold forwardLeakage
    calc
      (1 - P) * 1 * P = (1 - P) * P := by simp
      _ = P - P * P := by rw [Matrix.sub_mul, Matrix.one_mul]
      _ = P - P := by rw [hP]
      _ = 0 := sub_self P
  rw [hzero, Matrix.rank_zero]

/-- Inductive leakage rank bound over any unitary circuit `L : List (Matrix.unitaryGroup n ℂ)`:
    if every elementary gate `W ∈ L` has `rank((1 - P) W P) ≤ 1`, then
    `rank((1 - P) (circuitProduct L) P) ≤ L.length`. -/
theorem circuitProduct_forwardLeakage_rank_le_length
    (P : Matrix n n ℂ) (hP : P * P = P)
    (L : List (Matrix.unitaryGroup n ℂ))
    (hGates : ∀ W ∈ L, (forwardLeakage P W.val).rank ≤ 1) :
    (forwardLeakage P (circuitProduct L)).rank ≤ L.length := by
  induction L with
  | nil =>
    simp [forwardLeakage_one_rank_eq_zero P hP]
  | cons W tail ih =>
    rw [circuitProduct_cons, List.length_cons]
    have hW : (forwardLeakage P W.val).rank ≤ 1 := hGates W (List.mem_cons_self)
    have hTail : ∀ W' ∈ tail, (forwardLeakage P W'.val).rank ≤ 1 :=
      fun W' hW' => hGates W' (List.mem_cons_of_mem W hW')
    calc
      (forwardLeakage P (W.val * circuitProduct tail)).rank
        ≤ (forwardLeakage P W.val).rank + (forwardLeakage P (circuitProduct tail)).rank :=
          forwardLeakage_mul_rank_le P W.val (circuitProduct tail) hP
      _ ≤ 1 + tail.length := add_le_add hW (ih hTail)
      _ = tail.length + 1 := Nat.add_comm 1 tail.length

/-- **Phase 5 Main Theorem (Isometric Form — `unitary_bandwidth_depth_tradeoff`)**:
    If `circuitProduct L * P = U_nl.val * P` and every gate `W ∈ L` has `rank((1 - P) W P) ≤ 1`,
    then `D = L.length ≥ r = rank((1 - P) U_nl P)`. -/
theorem unitary_bandwidth_depth_tradeoff
    (P : Matrix n n ℂ) (hP : P * P = P)
    (U_nl : Matrix.unitaryGroup n ℂ)
    (L : List (Matrix.unitaryGroup n ℂ))
    (hGates : ∀ W ∈ L, (forwardLeakage P W.val).rank ≤ 1)
    (hAgree : circuitProduct L * P = U_nl.val * P) :
    (forwardLeakage P U_nl.val).rank ≤ L.length := by
  have hLeakEq : forwardLeakage P (circuitProduct L) = forwardLeakage P U_nl.val := by
    unfold forwardLeakage
    rw [Matrix.mul_assoc, hAgree, ← Matrix.mul_assoc]
  rw [← hLeakEq]
  exact circuitProduct_forwardLeakage_rank_le_length P hP L hGates

/-- **Phase 5 Stronger Block-Compression Dilation Theorem (`unitary_bandwidth_depth_tradeoff_compression`)**:
    Even if `circuitProduct L` only matches the compressed low-frequency block
    `P * (circuitProduct L) * P = P * U_nl.val * P`, unitarity forces
    `rank((1 - P) (circuitProduct L) P) = rank((1 - P) U_nl P)`, so `L.length ≥ r` still holds! -/
theorem unitary_bandwidth_depth_tradeoff_compression
    (P : Matrix n n ℂ) (hP : P * P = P) (hP_star : Pᴴ = P)
    (U_nl : Matrix.unitaryGroup n ℂ)
    (L : List (Matrix.unitaryGroup n ℂ))
    (hGates : ∀ W ∈ L, (forwardLeakage P W.val).rank ≤ 1)
    (hComp : P * circuitProduct L * P = P * U_nl.val * P) :
    (forwardLeakage P U_nl.val).rank ≤ L.length := by
  let U_D : Matrix.unitaryGroup n ℂ := ⟨circuitProduct L, circuitProduct_mem_unitaryGroup L⟩
  have hRankEq : (forwardLeakage P U_D.val).rank = (forwardLeakage P U_nl.val).rank :=
    forwardLeakage_rank_eq_of_compression_eq P hP hP_star U_D U_nl hComp
  rw [← hRankEq]
  exact circuitProduct_forwardLeakage_rank_le_length P hP L hGates

end GenericIndex

/-! ### 1D Clock Wire Support Predicate & Rank-1 Bound on `BlockIdx d m` -/

/-- Predicate stating that `W` only leaks amplitude from `V_Lambda` into a single 1D ultraviolet
    clock register wire `c ∈ Fin m`: for all UV indices `i ≠ c`, row `Sum.inr i` of
    `forwardLeakage (bandlimitProj d m) W` vanishes. -/
def IsElementaryClockGate (d m : ℕ) (c : Fin m)
    (W : Matrix (BlockIdx d m) (BlockIdx d m) ℂ) : Prop :=
  ∀ i : Fin m, i ≠ c → ∀ col : BlockIdx d m,
    forwardLeakage (bandlimitProj d m) W (Sum.inr i) col = 0

/-- For any matrix `W`, every low-frequency row `Sum.inl j` of `forwardLeakage (bandlimitProj d m) W`
    is identically zero because `1 - bandlimitProj d m = uvProj d m = fromBlocks 0 0 0 1`. -/
theorem forwardLeakage_bandlimitProj_inl_eq_zero
    (d m : ℕ) (W : Matrix (BlockIdx d m) (BlockIdx d m) ℂ) (j : Fin d) (col : BlockIdx d m) :
    forwardLeakage (bandlimitProj d m) W (Sum.inl j) col = 0 := by
  unfold forwardLeakage
  rw [← uvProj_eq_one_sub_bandlimitProj d m, Matrix.mul_assoc]
  unfold uvProj
  simp [Matrix.mul_apply, Matrix.fromBlocks_apply₁₁, Matrix.fromBlocks_apply₁₂]

/-- If `W` satisfies `IsElementaryClockGate d m c W`, then `forwardLeakage (bandlimitProj d m) W`
    factors as a rank-1 outer product `Matrix.vecMulVec` supported on wire `Sum.inr c`. -/
theorem forwardLeakage_eq_vecMulVec_of_isElementaryClockGate
    (d m : ℕ) (c : Fin m) (W : Matrix (BlockIdx d m) (BlockIdx d m) ℂ)
    (hGate : IsElementaryClockGate d m c W) :
    forwardLeakage (bandlimitProj d m) W =
      Matrix.vecMulVec
        (fun r : BlockIdx d m => if r = Sum.inr c then (1 : ℂ) else 0)
        (fun col : BlockIdx d m => forwardLeakage (bandlimitProj d m) W (Sum.inr c) col) := by
  ext r col
  rw [Matrix.vecMulVec_apply]
  cases r with
  | inl j =>
    have hne : (Sum.inl j : BlockIdx d m) ≠ Sum.inr c := Sum.inl_ne_inr
    rw [if_neg hne, zero_mul, forwardLeakage_bandlimitProj_inl_eq_zero]
  | inr i =>
    by_cases hic : i = c
    · subst hic
      rw [if_pos rfl, one_mul]
    · have hne : (Sum.inr i : BlockIdx d m) ≠ Sum.inr c := fun h => hic (Sum.inr.inj h)
      rw [if_neg hne, zero_mul, hGate i hic col]

/-- **1D Clock Wire Rank-1 Theorem**:
    Any matrix `W` satisfying `IsElementaryClockGate d m c W` has forward Nyquist leakage rank `≤ 1`. -/
theorem rank_forwardLeakage_le_one_of_isElementaryClockGate
    (d m : ℕ) (c : Fin m) (W : Matrix (BlockIdx d m) (BlockIdx d m) ℂ)
    (hGate : IsElementaryClockGate d m c W) :
    (forwardLeakage (bandlimitProj d m) W).rank ≤ 1 := by
  rw [forwardLeakage_eq_vecMulVec_of_isElementaryClockGate d m c W hGate]
  exact Matrix.rank_vecMulVec_le _ _

/-- Specialization of `unitary_bandwidth_depth_tradeoff` to `BlockIdx d (N - d)` where each gate
    satisfies the structural 1D wire predicate `IsElementaryClockGate`. -/
theorem unitary_bandwidth_depth_tradeoff_clock_wires
    (N d : ℕ)
    (U_nl : Matrix.unitaryGroup (BlockIdx d (N - d)) ℂ)
    (L : List (Matrix.unitaryGroup (BlockIdx d (N - d)) ℂ))
    (hWires : ∀ W ∈ L, ∃ c : Fin (N - d), IsElementaryClockGate d (N - d) c W.val)
    (hComp : bandlimitProj d (N - d) * circuitProduct L * bandlimitProj d (N - d) =
             bandlimitProj d (N - d) * U_nl.val * bandlimitProj d (N - d)) :
    (forwardLeakage (bandlimitProj d (N - d)) U_nl.val).rank ≤ L.length := by
  refine unitary_bandwidth_depth_tradeoff_compression
    (bandlimitProj d (N - d))
    (bandlimitProj_mul_self d (N - d))
    (bandlimitProj_conjTranspose d (N - d))
    U_nl L ?_ hComp
  intro W hW
  rcases hWires W hW with ⟨c, hc⟩
  exact rank_forwardLeakage_le_one_of_isElementaryClockGate d (N - d) c W.val hc

/-- Rank invariance under `reindexToFin` (`Matrix.reindex finSumFinEquiv finSumFinEquiv`). -/
theorem reindexToFin_rank_eq (d m : ℕ) (M : Matrix (BlockIdx d m) (BlockIdx d m) ℂ) :
    (reindexToFin d m M).rank = M.rank :=
  Matrix.rank_reindex finSumFinEquiv finSumFinEquiv M

end BekensteinNyquist
