import ChannelFormalization.LCU

/-! The one-sided outer channel-LCU circuit, with explicit initial density
matrix, ancillary postselection, and partial trace of the Kraus register. -/
noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix Matrix.Norms.L2Operator
open Matrix
namespace ChannelFormalization.ChannelLCU
open LCU

variable {J A S : Type*} [Fintype J] [Fintype A] [Fintype S]
  [DecidableEq J] [DecidableEq A] [DecidableEq S]

/-- Outer PREPARE is applied once, followed by SELECT. -/
def outerCircuit (P : Matrix J J ℂ) (U : J → Matrix (A × S) (A × S) ℂ) :
    Matrix (J × (A × S)) (J × (A × S)) ℂ := select U * liftRegister P

/-- Initialize both ancillary registers at their chosen zero addresses. -/
def initialDensity (j₀ : J) (a₀ : A) (ρ : Matrix S S ℂ) :
    Matrix (J × (A × S)) (J × (A × S)) ℂ :=
  fun x y => if x.1 = j₀ then if x.2.1 = a₀ then if y.1 = j₀ then
    if y.2.1 = a₀ then ρ x.2.2 y.2.2 else 0 else 0 else 0 else 0

/-- Apply the circuit, project the block register onto zero, and trace out
the Kraus index. The input/output are unnormalized density operators. -/
def successfulOutput (j₀ : J) (a₀ : A)
    (W : Matrix (J × (A × S)) (J × (A × S)) ℂ) (ρ : Matrix S S ℂ) :
    Matrix S S ℂ :=
  Matrix.traceLeft ((W * initialDensity j₀ a₀ ρ * W.conjTranspose).submatrix
    (fun x : J × S => (x.1, (a₀, x.2))) (fun x : J × S => (x.1, (a₀, x.2))))

/-- The surviving system block in branch `j`. -/
def branch (j₀ : J) (a₀ : A) (W : Matrix (J × (A × S)) (J × (A × S)) ℂ)
    (j : J) : Matrix S S ℂ :=
  W.submatrix (fun x => (j, (a₀, x))) (fun y => (j₀, (a₀, y)))

theorem successfulOutput_eq_kraus (j₀ : J) (a₀ : A)
    (W : Matrix (J × (A × S)) (J × (A × S)) ℂ) (ρ : Matrix S S ℂ) :
    successfulOutput j₀ a₀ W ρ =
      MatrixMap.of_kraus (branch j₀ a₀ W) (branch j₀ a₀ W) ρ := by
  ext x y
  simp [successfulOutput, initialDensity, branch, Matrix.traceLeft,
    MatrixMap.of_kraus, Matrix.mul_apply, Fintype.sum_prod_type,
    Matrix.conjTranspose_apply, Finset.sum_mul, LinearMap.sum_apply,
    Matrix.sum_apply]

theorem outer_branch (j₀ : J) (a₀ : A) (P : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ) (j : J) :
    branch j₀ a₀ (outerCircuit P U) j = P j j₀ • block a₀ (U j) := by
  ext x y
  simp [branch, outerCircuit, select, liftRegister, block, Matrix.mul_apply,
    Fintype.sum_prod_type, mul_comm]

theorem outerCircuit_unitary (P : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ)
    (hP : P.conjTranspose * P = 1)
    (hU : ∀ j, (U j).conjTranspose * U j = 1) :
    (outerCircuit P U).conjTranspose * outerCircuit P U = 1 :=
  unitary_mul _ _ (select_unitary U hU) (liftRegister_unitary P hP)

theorem exact_block (a₀ : A) (α : ℝ) (K : Matrix S S ℂ)
    (U : Matrix (A × S) (A × S) ℂ) (h : BlockEncoding a₀ α 0 K U) :
    (α : ℂ) • block a₀ U = K := by
  exact sub_eq_zero.mp (norm_eq_zero.mp (le_antisymm h.error_le (norm_nonneg _)))

theorem kraus_common_scale (K : J → Matrix S S ℂ) (c : ℂ) (ρ : Matrix S S ℂ) :
    MatrixMap.of_kraus (fun j => c • K j) (fun j => c • K j) ρ =
      (c * star c) • MatrixMap.of_kraus K K ρ := by
  simp only [MatrixMap.of_kraus, LinearMap.sum_apply, LinearMap.coe_mk, AddHom.coe_mk,
    Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul, smul_smul]
  rw [Finset.smul_sum]
  apply Finset.sum_congr rfl
  intro j _
  rw [mul_comm (star c) c]

theorem sqrt_inverse_square (Λ : ℝ) (hΛ : 0 < Λ) :
    ((Real.sqrt Λ : ℂ)⁻¹ * star (Real.sqrt Λ : ℂ)⁻¹) = (Λ : ℂ)⁻¹ := by
  simp only [Complex.star_def, map_inv₀, Complex.conj_ofReal]
  rw [← mul_inv]
  congr 1
  exact_mod_cast Real.mul_self_sqrt hΛ.le

/-- Proposition 4.5: exact normalized Kraus blocks in the actual outer
circuit yield the original CP map divided by the squared normalization sum. -/
theorem proposition_4_5 (j₀ : J) (a₀ : A)
    (K : J → Matrix S S ℂ) (α : J → ℝ) (P : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ)
    (hblocks : ∀ j, BlockEncoding a₀ (α j) 0 (K j) (U j))
    (hP : P.conjTranspose * P = 1)
    (hΛ : 0 < ∑ j, α j ^ 2)
    (hprep : ∀ j, P j j₀ = ((α j / Real.sqrt (∑ i, α i ^ 2) : ℝ) : ℂ)) :
    (outerCircuit P U).conjTranspose * outerCircuit P U = 1 ∧
    ∀ ρ, successfulOutput j₀ a₀ (outerCircuit P U) ρ =
      ((∑ j, α j ^ 2 : ℝ) : ℂ)⁻¹ • MatrixMap.of_kraus K K ρ := by
  refine ⟨outerCircuit_unitary P U hP (fun j => (hblocks j).unitary), ?_⟩
  intro ρ
  have hb : branch j₀ a₀ (outerCircuit P U) =
      fun j => (Real.sqrt (∑ i, α i ^ 2) : ℂ)⁻¹ • K j := by
    funext j
    rw [outer_branch, hprep j, Complex.ofReal_div, div_eq_mul_inv, mul_comm,
      ← smul_smul, exact_block a₀ (α j) (K j) (U j) (hblocks j)]
  rw [successfulOutput_eq_kraus, hb, kraus_common_scale, sqrt_inverse_square _ hΛ]

/-- Probability is the real trace of the unnormalized successful output. -/
def successProbability (j₀ : J) (a₀ : A)
    (W : Matrix (J × (A × S)) (J × (A × S)) ℂ) (ρ : Matrix S S ℂ) : ℝ :=
  (successfulOutput j₀ a₀ W ρ).trace.re

theorem success_probability (j₀ : J) (a₀ : A)
    (K : J → Matrix S S ℂ) (α : J → ℝ) (P : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ)
    (hblocks : ∀ j, BlockEncoding a₀ (α j) 0 (K j) (U j))
    (hP : P.conjTranspose * P = 1) (hΛ : 0 < ∑ j, α j ^ 2)
    (hprep : ∀ j, P j j₀ = ((α j / Real.sqrt (∑ i, α i ^ 2) : ℝ) : ℂ))
    (ρ : Matrix S S ℂ) :
    successProbability j₀ a₀ (outerCircuit P U) ρ =
      (MatrixMap.of_kraus K K ρ).trace.re / (∑ j, α j ^ 2) := by
  unfold successProbability
  rw [(proposition_4_5 j₀ a₀ K α P U hblocks hP hΛ hprep).2 ρ,
    Matrix.trace_smul, smul_eq_mul, ← Complex.ofReal_inv]
  rw [Complex.mul_re]
  simp only [Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero]
  exact mul_comm _ _

theorem success_probability_TP (j₀ : J) (a₀ : A)
    (K : J → Matrix S S ℂ) (α : J → ℝ) (P : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ)
    (hblocks : ∀ j, BlockEncoding a₀ (α j) 0 (K j) (U j))
    (hP : P.conjTranspose * P = 1) (hΛ : 0 < ∑ j, α j ^ 2)
    (hprep : ∀ j, P j j₀ = ((α j / Real.sqrt (∑ i, α i ^ 2) : ℝ) : ℂ))
    (hTP : (MatrixMap.of_kraus K K).IsTracePreserving)
    (ρ : Matrix S S ℂ) (hρ : ρ.trace = 1) :
    successProbability j₀ a₀ (outerCircuit P U) ρ = 1 / (∑ j, α j ^ 2) := by
  rw [success_probability j₀ a₀ K α P U hblocks hP hΛ hprep ρ,
    hTP ρ, hρ, Complex.one_re]

/-- Proposition 4.5 with a larger outer address register: inactive branches
apply the identity but receive exactly zero preparation amplitude. -/
theorem proposition_4_5_with_unused_addresses (j₀ : J) (a₀ : A)
    (used : Finset J) (K : J → Matrix S S ℂ) (α : J → ℝ) (P : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ)
    (hblocks : ∀ j ∈ used, BlockEncoding a₀ (α j) 0 (K j) (U j))
    (hP : P.conjTranspose * P = 1)
    (hΛ : 0 < used.sum (fun j => α j ^ 2))
    (hprep : ∀ j ∈ used,
      P j j₀ = ((α j / Real.sqrt (used.sum fun i => α i ^ 2) : ℝ) : ℂ))
    (hzero : ∀ j ∉ used, P j j₀ = 0) :
    let W := outerCircuit P (completeSelect used U)
    let K' := fun j => if j ∈ used then K j else 0
    W.conjTranspose * W = 1 ∧
    ∀ ρ, successfulOutput j₀ a₀ W ρ =
      ((used.sum (fun j => α j ^ 2) : ℝ) : ℂ)⁻¹ • MatrixMap.of_kraus K' K' ρ := by
  dsimp only
  refine ⟨outerCircuit_unitary P _ hP ?_, ?_⟩
  · intro j
    by_cases hj : j ∈ used
    · simpa [completeSelect, hj] using (hblocks j hj).unitary
    · simp [completeSelect, hj]
  · intro ρ
    have hb : branch j₀ a₀ (outerCircuit P (completeSelect used U)) =
        fun j => (Real.sqrt (used.sum fun i => α i ^ 2) : ℂ)⁻¹ •
          (if j ∈ used then K j else 0) := by
      funext j
      rw [outer_branch]
      by_cases hj : j ∈ used
      · simp only [completeSelect, hj, ↓reduceIte]
        rw [hprep j hj, Complex.ofReal_div, div_eq_mul_inv, mul_comm, ← smul_smul,
          exact_block a₀ (α j) (K j) (U j) (hblocks j hj)]
      · simp [hj, hzero j hj]
    rw [successfulOutput_eq_kraus, hb, kraus_common_scale, sqrt_inverse_square _ hΛ]

end ChannelFormalization.ChannelLCU
