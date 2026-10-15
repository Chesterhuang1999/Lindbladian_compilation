import ChannelFormalization.PauliCompiler
import ChannelFormalization.GuardedRewrites

/-! The current paper's complete guarded rewrite relation, including zero-Kraus
padding, composed with the actual Pauli placement and channel-LCU circuits. -/
noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix Matrix.Norms.L2Operator
namespace ChannelFormalization.GuardedCompiler
open Matrix LCU InnerLCU Placement PauliPlacement ChannelIR EndToEnd ChannelLCU PauliCompiler
variable {T : Type*} [Fintype T] [DecidableEq T] [PartialOrder T] [DecidableLE T]
  {n : ℕ}

/-- Theorem 6.2 for the paper's six guarded rules and optional zero-Kraus padding.
The only rewriting input is a finite path in that relation. Its proved soundness
automatically supplies denotational equality; Proposition 5.1 supplies each
actual phase-restored monotone block-encoding with the shared PREPARE. -/
theorem theorem_6_2_guarded {O : Type*} [Fintype O] [DecidableEq O]
    (C D : Guarded.Channel n) (hrewrite : Guarded.PaddedRewrites C D)
    (E : MatrixMap (QState n) (QState n) ℂ) (εfront : ℝ) (hεfront : 0 ≤ εfront)
    (hfront : diamondNorm (Guarded.denote C - E) ≤ ENNReal.ofReal εfront)
    (J : Fin (Guarded.toChannel D).length → Type*) [∀ j, Fintype (J j)]
    (placement : ∀ j, J j ↪ T) (order : Placement.LinearExtension T) (t₀ : T)
    (β : ∀ j, J j → ℂ) (p : ∀ j, J j → PauliLabel n)
    (active : Fin (Guarded.toChannel D).length → Finset T) (q : Fin (Guarded.toChannel D).length → T → PauliLabel n)
    (hactive : ∀ j t, placement j t ∈ active j)
    (hq : ∀ j t, q j (placement j t) = p j t)
    (hrep : ∀ j, (Guarded.toChannel D).krausFamily j = ∑ t, β j t • labelMatrix (p j t))
    (hnonzero : ∀ j, (Guarded.toChannel D).krausFamily j ≠ 0)
    (P : Fin (Guarded.toChannel D).length → Matrix T T ℂ)
    (hP : ∀ j, (P j).conjTranspose * P j = 1)
    (hinnerPrep : ∀ j t, P j t t₀ =
      (Real.sqrt (‖addressTable (placement j) (β j) t‖ / normalizer (β j)) : ℂ))
    (e : Fin (Guarded.toChannel D).length ↪ O) (o₀ : O) (Q : Matrix O O ℂ)
    (hQ : Q.conjTranspose * Q = 1)
    (hΛ : 0 < ∑ j, normalizer (β j) ^ 2)
    (houterPrep : ∀ o, Q o o₀ =
      ((addressTable e (fun j => normalizer (β j)) o /
        Real.sqrt (∑ j, normalizer (β j) ^ 2) : ℝ) : ℂ)) :
    let Λ := ∑ j, normalizer (β j) ^ 2
    let W := embeddedCircuit e Q (fun j =>
      compiledInner order (active j) (P j) (P j)
        (addressTable (placement j) (β j)) (q j))
    let I_succ := successfulMap o₀ (t₀, ()) W
    WellTyped (Guarded.toChannel D).erase n ∧ Guarded.denote D = Guarded.denote C ∧
    W.conjTranspose * W = 1 ∧
    I_succ = (Λ : ℂ)⁻¹ • Guarded.denote C ∧
    diamondNorm ((Λ : ℂ) • I_succ - E) ≤ ENNReal.ofReal εfront ∧
    ∀ ρ, successProbability o₀ (t₀, ()) W ρ = (Guarded.denote C ρ).trace.re / Λ := by
  exact PauliCompiler.theorem_6_2_withPlacement_of_denote
    (Guarded.toChannel C) (Guarded.toChannel D) hrewrite.sound.symm
    E εfront hεfront hfront J placement order t₀ β p active q hactive hq hrep hnonzero
    P hP hinnerPrep e o₀ Q hQ hΛ houterPrep

end ChannelFormalization.GuardedCompiler
