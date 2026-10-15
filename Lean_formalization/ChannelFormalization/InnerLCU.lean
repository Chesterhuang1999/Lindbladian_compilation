import ChannelFormalization.LCU
import Mathlib.Analysis.SpecialFunctions.Complex.Arg

/-! Exact inner LCU lowering, with coefficient phases explicitly included in
the selected unitary. The result applies to Pauli strings and to any unitary basis. -/
noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix Matrix.Norms.L2Operator
namespace ChannelFormalization.InnerLCU
open Matrix LCU
variable {T S : Type*} [Fintype T] [Fintype S] [DecidableEq T] [DecidableEq S]

def normalizer (β : T → ℂ) : ℝ := ∑ t, ‖β t‖

theorem normalizer_pos (β : T → ℂ) (Q : T → Matrix S S ℂ)
    (hK : ∑ t, β t • Q t ≠ 0) : 0 < normalizer β := by
  have hn : 0 ≤ normalizer β := Finset.sum_nonneg (fun _ _ => norm_nonneg _)
  by_contra h
  have hz : normalizer β = 0 := le_antisymm (le_of_not_gt h) hn
  have hall : ∀ t, β t = 0 := by
    intro t
    apply norm_eq_zero.mp
    exact (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => norm_nonneg (β i))).mp hz t
      (Finset.mem_univ _)
  exact hK (by simp [hall])

def phased (β : ℂ) (Q : Matrix S S ℂ) : Matrix S S ℂ :=
  Complex.exp (β.arg * Complex.I) • Q

theorem coefficient_phase (β : ℂ) (Q : Matrix S S ℂ) :
    (‖β‖ : ℂ) • phased β Q = β • Q := by
  rw [phased, smul_smul, Complex.norm_mul_exp_arg_mul_I]

theorem phased_unitary (β : ℂ) (Q : Matrix S S ℂ) (hQ : Q.conjTranspose * Q = 1) :
    (phased β Q).conjTranspose * phased β Q = 1 := by
  rw [phased, Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul,
    smul_smul, hQ]
  have hc : Complex.exp (β.arg * Complex.I) * star (Complex.exp (β.arg * Complex.I)) = 1 := by
    rw [Complex.star_def, Complex.mul_conj, Complex.normSq_eq_norm_sq,
      Complex.norm_exp_ofReal_mul_I]
    norm_num
  rw [mul_comm (star (Complex.exp (β.arg * Complex.I))) _, hc, one_smul]

/-- Regard a system gate as a zero-ancilla block encoding. -/
def trivialAncilla (Q : Matrix S S ℂ) : Matrix (Unit × S) (Unit × S) ℂ :=
  Q.submatrix (Equiv.uniqueProd S Unit) (Equiv.uniqueProd S Unit)

@[simp] theorem trivialAncilla_block (Q : Matrix S S ℂ) :
    block () (trivialAncilla Q) = Q := rfl

theorem trivialAncilla_unitary (Q : Matrix S S ℂ) (hQ : Q.conjTranspose * Q = 1) :
    (trivialAncilla Q).conjTranspose * trivialAncilla Q = 1 := by
  rw [trivialAncilla, Matrix.conjTranspose_submatrix,
    Matrix.submatrix_mul_equiv _ _ _ (Equiv.uniqueProd S Unit), hQ,
    Matrix.submatrix_one_equiv]

def innerCircuit (L R : Matrix T T ℂ) (β : T → ℂ) (Q : T → Matrix S S ℂ) :
    Matrix ((T × Unit) × S) ((T × Unit) × S) ℂ :=
  groupedCircuit L R (fun t => trivialAncilla (phased (β t) (Q t)))

/-- Exact PREPARE and phase-restored SELECT construct the normalized Kraus block.
The preparation premise is a statement about state amplitudes, not about the
resulting block or channel. -/
theorem inner_blockEncoding [Nonempty S] (t₀ : T) (β : T → ℂ)
    (Q : T → Matrix S S ℂ) (L R : Matrix T T ℂ)
    (hQ : ∀ t, (Q t).conjTranspose * Q t = 1)
    (hL : L.conjTranspose * L = 1) (hR : R.conjTranspose * R = 1)
    (hα : 0 < normalizer β)
    (hprep : ∀ t, (normalizer β : ℂ) * (star (L t t₀) * R t t₀) = (‖β t‖ : ℂ)) :
    BlockEncoding (t₀, ()) (normalizer β) 0 (∑ t, β t • Q t)
      (innerCircuit L R β Q) := by
  have hb : ∀ t, BlockEncoding () 1 0 (phased (β t) (Q t))
      (trivialAncilla (phased (β t) (Q t))) := by
    intro t
    refine ⟨by norm_num, le_rfl, trivialAncilla_unitary _ (phased_unitary _ _ (hQ t)), ?_⟩
    simp
  have hy : ∑ t, ‖(‖β t‖ : ℂ)‖ ≤ normalizer β := by simp [normalizer]
  have hc : ∑ t, ‖(normalizer β : ℂ) * (star (L t t₀) * R t t₀) -
      (‖β t‖ : ℂ)‖ ≤ 0 := by
    simp only [hprep, sub_self, norm_zero, Finset.sum_const_zero, le_refl]
  have h := theorem_2_2 t₀ () L R (fun t => trivialAncilla (phased (β t) (Q t)))
    (fun t => phased (β t) (Q t)) (fun t => (‖β t‖ : ℂ))
    1 (normalizer β) 0 0 (by norm_num) hα le_rfl le_rfl hL hR hy hc hb
  simpa only [one_mul, mul_zero, add_zero, coefficient_phase, innerCircuit] using h

end ChannelFormalization.InnerLCU
