import ChannelFormalization.InnerLCU
import ChannelFormalization.Placement

/-! Lower the selected Pauli/unitary table to the actual product of monotone
controlled gates. Unoccupied coefficient addresses impose no target constraint.
The factors exist by the proved residual construction of Lemma C.2. -/
noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix Matrix.Norms.L2Operator
namespace ChannelFormalization.OptimizedLCU
open Matrix LCU InnerLCU Placement
variable {T S : Type*} [Fintype T] [Fintype S] [DecidableEq T] [DecidableEq S]
  [PartialOrder T] [DecidableLE T]

def phaseTarget (β : T → ℂ) (Q : T → Matrix S S ℂ)
    (hQ : ∀ t, (Q t).conjTranspose * Q t = 1) (t : T) :
    Matrix.unitaryGroup (Unit × S) ℂ :=
  ⟨trivialAncilla (phased (β t) (Q t)), Matrix.mem_unitaryGroup_iff'.mpr
    (trivialAncilla_unitary _ (phased_unitary _ _ (hQ t)))⟩

def factors (order : Placement.LinearExtension T) (β : T → ℂ) (Q : T → Matrix S S ℂ)
    (hQ : ∀ t, (Q t).conjTranspose * Q t = 1) :
    T → Matrix.unitaryGroup (Unit × S) ℂ := by
  classical
  exact Classical.choose (lemma_C_2_unitary order (fun t => β t ≠ 0) (phaseTarget β Q hQ))

theorem factors_correct (order : Placement.LinearExtension T) (β : T → ℂ)
    (Q : T → Matrix S S ℂ) (hQ : ∀ t, (Q t).conjTranspose * Q t = 1)
    (t : T) (ht : β t ≠ 0) :
    orderedAction order.order (factors order β Q hQ) t = phaseTarget β Q hQ t := by
  classical
  exact ((Classical.choose_spec
    (lemma_C_2_unitary order (fun t => β t ≠ 0) (phaseTarget β Q hQ))).1 t ht).symm

/-- This SELECT is a product of the monotone controlled gates, not the target
table used as a substitute for that product. -/
def compiledInner (order : Placement.LinearExtension T) (L R : Matrix T T ℂ)
    (β : T → ℂ) (Q : T → Matrix S S ℂ)
    (hQ : ∀ t, (Q t).conjTranspose * Q t = 1) :
    Matrix ((T × Unit) × S) ((T × Unit) × S) ℂ :=
  ((liftRegister L).conjTranspose *
    optimizedSelect order.order (factors order β Q hQ) * liftRegister R).submatrix
      (Equiv.prodAssoc T Unit S) (Equiv.prodAssoc T Unit S)

theorem compiledInner_blockEncoding [Nonempty S]
    (order : Placement.LinearExtension T) (t₀ : T) (β : T → ℂ)
    (Q : T → Matrix S S ℂ) (L R : Matrix T T ℂ)
    (hQ : ∀ t, (Q t).conjTranspose * Q t = 1)
    (hL : L.conjTranspose * L = 1) (hR : R.conjTranspose * R = 1)
    (hα : 0 < normalizer β)
    (hprep : ∀ t, (normalizer β : ℂ) * (star (L t t₀) * R t t₀) = (‖β t‖ : ℂ)) :
    BlockEncoding (t₀, ()) (normalizer β) 0 (∑ t, β t • Q t)
      (compiledInner order L R β Q hQ) := by
  let V := fun t => (orderedAction order.order (factors order β Q hQ) t).val
  have hv : ∀ t, BlockEncoding () 1 0 (block () (V t)) (V t) := by
    intro t
    refine ⟨by norm_num, le_rfl,
      Matrix.mem_unitaryGroup_iff'.mp (orderedAction order.order (factors order β Q hQ) t).property, ?_⟩
    simp
  have hy : ∑ t, ‖(‖β t‖ : ℂ)‖ ≤ normalizer β := by simp [normalizer]
  have hc : ∑ t, ‖(normalizer β : ℂ) * (star (L t t₀) * R t t₀) -
      (‖β t‖ : ℂ)‖ ≤ 0 := by
    simp only [hprep, sub_self, norm_zero, Finset.sum_const_zero, le_refl]
  have hs : ∑ t, (‖β t‖ : ℂ) • block () (V t) = ∑ t, β t • Q t := by
    apply Finset.sum_congr rfl
    intro t _
    by_cases ht : β t = 0
    · simp [ht]
    · have hf := factors_correct order β Q hQ t ht
      change (‖β t‖ : ℂ) • block () (orderedAction order.order (factors order β Q hQ) t).val = _
      rw [hf]
      exact coefficient_phase (β t) (Q t)
  have h := theorem_2_2 t₀ () L R V (fun t => block () (V t))
    (fun t => (‖β t‖ : ℂ)) 1 (normalizer β) 0 0
    (by norm_num) hα le_rfl le_rfl hL hR hy hc hv
  simp only [one_mul, mul_zero, add_zero, hs] at h
  simpa only [compiledInner, optimizedSelect_eq_select, groupedCircuit, circuit, V] using h

end ChannelFormalization.OptimizedLCU
