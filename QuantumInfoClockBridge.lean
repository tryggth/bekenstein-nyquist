import QuantumInfo.Channels.CPTP
import QuantumInfo.Entropy.VonNeumann
import UnitaryBandwidthDepth
import NyquistRotorGates

/-!
# Phase B Bridge 2 — Direct Upstream `QuantumInfo` CPTP & Entropy Integration (`QuantumInfoClockBridge.lean`)

This module imports `QuantumInfo.Channels.CPTP` and `QuantumInfo.Entropy.VonNeumann` directly from
`leanprover-community/physlib` and bridges them with our Bekenstein–Nyquist formalization
(`UnitaryBandwidthDepth.lean` and `NyquistRotorGates.lean`):

1. **Bipartite Unitary Rotor Evolution as an Upstream `CPTPMap` (`bipartiteUnitaryChannel`)**:
   Maps any unitary operator `U : 𝐔[d_spatial × d_clock]` (and specifically our `givensRotor` and
   `milestone4TripartiteRouter` gates) into an upstream `CPTPMap` via `CPTPMap.ofUnitary`,
   certifying `IsCompletelyPositive` and `IsTracePreserving`.
2. **Reduced Spatial Fluid Channel via Partial Trace (`reducedSpatialFluidChannel`)**:
   Constructs the open-system spatial fluid evolution obtained by preparing an ancilla clock state
   `ρ_clock : MState d_clock`, applying the joint unitary `U : 𝐔[d_spatial × d_clock]`, and tracing
   out the clock fiber via `CPTPMap.traceRight`.
3. **Stinespring / Unitary Purification Duality (`spatialChannel_admits_unitary_purification`)**:
   Directly invokes upstream `CPTPMap.purify_IsUnitary` and `CPTPMap.purify_trace` to prove that
   every completely positive trace-preserving channel on the bandlimited spatial fluid arises as the
   partial trace of a strictly unitary evolution on a dilated Hilbert space.
4. **Zero Global Landauer Entropy vs. Apparent Lax Shock Entropy (Monograph Section 3.3)**:
   Directly invokes upstream `Sᵥₙ_of_pure_zero`, `Sᵥₙ_of_partial_eq`, and `Sᵥₙ_nonneg` to prove that
   for any pure joint fluid+clock state `Ψ : Ket (d_spatial × d_clock)`:
   - Global von Neumann entropy vanishes identically (`Sᵥₙ (MState.pure Ψ) = 0` — zero Landauer erasure).
   - Apparent spatial shock entropy `Sᵥₙ (MState.pure Ψ).traceRight` equals the worldline clock
     fiber entropy `Sᵥₙ (MState.pure Ψ).traceLeft` (`≥ 0`), and the quantum mutual information
     `qMutualInfo (MState.pure Ψ)` equals `2 * Sᵥₙ (MState.pure Ψ).traceRight`.
-/

namespace BekensteinNyquist

noncomputable section

variable {d_spatial d_clock : Type*}
variable [Fintype d_spatial] [DecidableEq d_spatial]
variable [Fintype d_clock] [DecidableEq d_clock]

/-! ### 1. Bipartite Unitary Evolution & Concrete Givens / Tripartite Channels in `CPTPMap` -/

/-- The upstream `CPTPMap` induced by a bipartite unitary `U : 𝐔[d_spatial × d_clock]`. -/
def bipartiteUnitaryChannel (U : 𝐔[d_spatial × d_clock]) :
    CPTPMap (d_spatial × d_clock) (d_spatial × d_clock) :=
  CPTPMap.ofUnitary U

/-- `bipartiteUnitaryChannel U` satisfies the upstream `CPTPMap.IsUnitary` predicate. -/
theorem bipartiteUnitaryChannel_isUnitary (U : 𝐔[d_spatial × d_clock]) :
    (bipartiteUnitaryChannel U).IsUnitary :=
  ⟨U, rfl⟩

/-- **Concrete `SU(2)` Givens Rotor as an Upstream `CPTPMap`**:
    Every impedance-matched Givens rotation `givensRotor ϑ ζ` (`‖ζ‖ = 1`) from `NyquistRotorGates.lean`
    defines a unitary `CPTPMap (BlockIdx 1 1) (BlockIdx 1 1)` via `CPTPMap.ofUnitary`. -/
def givensRotorCPTP (ϑ : ℝ) (ζ : ℂ) (hζ : star ζ * ζ = 1) :
    CPTPMap (BlockIdx 1 1) (BlockIdx 1 1) :=
  CPTPMap.ofUnitary ⟨givensRotor ϑ ζ, givensRotor_mem_unitaryGroup ϑ ζ hζ⟩

/-- `givensRotorCPTP ϑ ζ hζ` is a unitary CPTP channel. -/
theorem givensRotorCPTP_isUnitary (ϑ : ℝ) (ζ : ℂ) (hζ : star ζ * ζ = 1) :
    (givensRotorCPTP ϑ ζ hζ).IsUnitary :=
  ⟨⟨givensRotor ϑ ζ, givensRotor_mem_unitaryGroup ϑ ζ hζ⟩, rfl⟩

/-- **Milestone 4 Tripartite 3-Mode Haar Router as an Upstream `CPTPMap`**:
    The composite 2-stage collision router `milestone4TripartiteRouter ϑ₁ θ_rec` on
    `TripartiteIdx = (ū_b, u_{b,UV}, c_slot)` defines a unitary `CPTPMap TripartiteIdx TripartiteIdx`. -/
def milestone4TripartiteCPTP (ϑ₁ θ_rec : ℝ) :
    CPTPMap TripartiteIdx TripartiteIdx :=
  CPTPMap.ofUnitary ⟨milestone4TripartiteRouter ϑ₁ θ_rec,
    milestone4TripartiteRouter_mem_unitaryGroup ϑ₁ θ_rec⟩

/-- `milestone4TripartiteCPTP ϑ₁ θ_rec` is a unitary CPTP channel. -/
theorem milestone4TripartiteCPTP_isUnitary (ϑ₁ θ_rec : ℝ) :
    (milestone4TripartiteCPTP ϑ₁ θ_rec).IsUnitary :=
  ⟨⟨milestone4TripartiteRouter ϑ₁ θ_rec,
    milestone4TripartiteRouter_mem_unitaryGroup ϑ₁ θ_rec⟩, rfl⟩

/-! ### 2. Reduced Spatial Fluid Channel via Partial Trace (`traceRight`) -/

/-- Given a bipartite unitary `U : 𝐔[d_spatial × d_clock]` and an initial clock state
    `ρ_clock : MState d_clock`, the reduced spatial fluid channel is formed by:
    1. appending `Unit` (`ofEquiv (Equiv.prodPUnit d_spatial).symm`),
    2. preparing `ρ_clock` on the right factor (`id ⊗ᶜᵖ replacement ρ_clock`),
    3. evolving jointly under `ofUnitary U`, and
    4. tracing out the clock fiber (`CPTPMap.traceRight`). -/
def reducedSpatialFluidChannel (U : 𝐔[d_spatial × d_clock]) (ρ_clock : MState d_clock) :
    CPTPMap d_spatial d_spatial :=
  let append : CPTPMap d_spatial (d_spatial × Unit) :=
    CPTPMap.ofEquiv (Equiv.prodPUnit d_spatial).symm
  let prep : CPTPMap (d_spatial × Unit) (d_spatial × d_clock) :=
    CPTPMap.id ⊗ᶜᵖ CPTPMap.replacement ρ_clock
  CPTPMap.traceRight ∘ₘ CPTPMap.ofUnitary U ∘ₘ prep ∘ₘ append

/-- **Exact Trace Preservation of the Reduced Spatial Fluid Channel**:
    Tracing out the worldline clock fiber preserves the total trace of any spatial operator
    `X : Matrix d_spatial d_spatial ℂ` (`(reducedSpatialFluidChannel U ρ_clock).map X` has trace `X.trace`). -/
theorem reducedSpatialFluidChannel_tracePreserving
    (U : 𝐔[d_spatial × d_clock]) (ρ_clock : MState d_clock)
    (X : Matrix d_spatial d_spatial ℂ) :
    ((reducedSpatialFluidChannel U ρ_clock).map X).trace = X.trace :=
  (reducedSpatialFluidChannel U ρ_clock).TP X

/-- **Unitary Purification / Stinespring Dilation of Any Spatial CPTP Channel**:
    By upstream `CPTPMap.purify_IsUnitary` and `CPTPMap.purify_trace`, every CPTP channel
    `Λ : CPTPMap d_spatial d_spatial` on the bandlimited spatial subspace admits a unitary channel
    `Λ.purify` on `d_spatial × d_spatial × d_spatial` whose partial trace recovers `Λ`. -/
theorem spatialChannel_admits_unitary_purification [Inhabited d_spatial]
    (Λ : CPTPMap d_spatial d_spatial) :
    Λ.purify.IsUnitary ∧
    Λ = (
      let zero_prep : CPTPMap Unit (d_spatial × d_spatial) :=
        CPTPMap.replacement (MState.pure (Ket.basis default))
      let prep := CPTPMap.id ⊗ᶜᵖ zero_prep
      let append : CPTPMap d_spatial (d_spatial × Unit) :=
        CPTPMap.ofEquiv (Equiv.prodPUnit d_spatial).symm
      CPTPMap.traceLeft ∘ₘ CPTPMap.traceLeft ∘ₘ Λ.purify ∘ₘ prep ∘ₘ append
    ) :=
  ⟨CPTPMap.purify_IsUnitary Λ, CPTPMap.purify_trace Λ⟩

/-! ### 3. Zero Global Landauer Entropy vs. Apparent Lax Shock Entropy (Section 3.3) -/

/-- **Zero Global Landauer Entropy Production**:
    For any pure bipartite fluid+clock state `Ψ : Ket (d_spatial × d_clock)`, the global
    von Neumann entropy vanishes identically (`Sᵥₙ (MState.pure Ψ) = 0`). -/
theorem bipartite_pure_global_entropy_zero (Ψ : Ket (d_spatial × d_clock)) :
    Sᵥₙ (MState.pure Ψ) = 0 :=
  Sᵥₙ_of_pure_zero Ψ

/-- **Apparent Lax Shock Entropy Equals Worldline Clock Entanglement Entropy**:
    When a pure bipartite state `Ψ : Ket (d_spatial × d_clock)` is projected onto the spatial fluid
    by tracing out the clock register (`(MState.pure Ψ).traceRight`), its apparent macroscopic
    von Neumann entropy equals the von Neumann entropy stored in the clock register
    (`(MState.pure Ψ).traceLeft`), and both are non-negative. -/
theorem spatial_shock_entropy_eq_clock_entropy (Ψ : Ket (d_spatial × d_clock)) :
    Sᵥₙ (MState.pure Ψ).traceRight = Sᵥₙ (MState.pure Ψ).traceLeft ∧
    0 ≤ Sᵥₙ (MState.pure Ψ).traceRight :=
  ⟨(Sᵥₙ_of_partial_eq Ψ).symm, Sᵥₙ_nonneg (MState.pure Ψ).traceRight⟩

/-- **Spatial–Clock Quantum Mutual Information Identity**:
    For any pure bipartite fluid+clock state `Ψ : Ket (d_spatial × d_clock)`, the quantum mutual
    information `qMutualInfo (MState.pure Ψ)` between the spatial fluid and the worldline clock
    register is exactly twice the apparent spatial shock entropy:
    `qMutualInfo (MState.pure Ψ) = 2 * Sᵥₙ (MState.pure Ψ).traceRight`. -/
theorem bipartite_pure_mutual_info_eq_two_mul_spatial_entropy
    (Ψ : Ket (d_spatial × d_clock)) :
    qMutualInfo (MState.pure Ψ) = 2 * Sᵥₙ (MState.pure Ψ).traceRight := by
  unfold qMutualInfo
  rw [Sᵥₙ_of_pure_zero Ψ, Sᵥₙ_of_partial_eq Ψ]
  ring

end

end BekensteinNyquist
