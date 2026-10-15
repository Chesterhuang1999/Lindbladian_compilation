import QuantumInfo.Channels.Unbundled
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith

/-! Concrete finite-dimensional PREPARE/SELECT circuits and their selected blocks.
The system and ancillary index types may be any finite types; qubit registers are
obtained by taking `Fin (2 ^ n)`. Norms below are the spectral/operator norm. -/

noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix Matrix.Norms.L2Operator
open Matrix

namespace ChannelFormalization.LCU

variable {J A S : Type*} [Fintype J] [Fintype A] [Fintype S]
  [DecidableEq J] [DecidableEq A] [DecidableEq S]

/-- The ancillary all-zero block, with a chosen zero basis address. -/
def block (a₀ : A) (U : Matrix (A × S) (A × S) ℂ) : Matrix S S ℂ :=
  U.submatrix (fun x => (a₀, x)) (fun x => (a₀, x))

@[simp] theorem block_one (a₀ : A) : block (S := S) a₀ 1 = 1 := by
  ext x y
  simp [block, Matrix.one_apply]

/-- SELECT applies the unitary supplied for each address. -/
def select (U : J → Matrix S S ℂ) : Matrix (J × S) (J × S) ℂ :=
  fun x y => if x.1 = y.1 then U x.1 x.2 y.2 else 0

/-- A register operation tensored with the identity on the system. -/
def liftRegister (P : Matrix J J ℂ) : Matrix (J × S) (J × S) ℂ :=
  fun x y => P x.1 y.1 * if x.2 = y.2 then 1 else 0

@[simp] theorem select_conjTranspose (U : J → Matrix S S ℂ) :
    (select U).conjTranspose = select (fun j => (U j).conjTranspose) := by
  ext ⟨j, x⟩ ⟨k, y⟩
  by_cases h : j = k
  · subst k; simp [select, Matrix.conjTranspose_apply]
  · simp [select, Matrix.conjTranspose_apply, h, Ne.symm h]

@[simp] theorem select_mul (U V : J → Matrix S S ℂ) :
    select U * select V = select (fun j => U j * V j) := by
  ext ⟨j, x⟩ ⟨k, y⟩
  simp only [Matrix.mul_apply, Fintype.sum_prod_type, select]
  by_cases h : j = k
  · subst k; simp
  · simp [h]

@[simp] theorem select_one : select (fun _ : J => (1 : Matrix S S ℂ)) = 1 := by
  ext ⟨j, x⟩ ⟨k, y⟩
  simp [select, Matrix.one_apply, Prod.ext_iff, ite_and]

theorem select_unitary (U : J → Matrix S S ℂ)
    (hU : ∀ j, (U j).conjTranspose * U j = 1) :
    (select U).conjTranspose * select U = 1 := by
  simp only [select_conjTranspose, select_mul, hU, select_one]

@[simp] theorem liftRegister_conjTranspose (P : Matrix J J ℂ) :
    (liftRegister (S := S) P).conjTranspose = liftRegister P.conjTranspose := by
  ext ⟨j, x⟩ ⟨k, y⟩
  by_cases h : x = y
  · subst y; simp [liftRegister, Matrix.conjTranspose_apply]
  · simp [liftRegister, Matrix.conjTranspose_apply, h, Ne.symm h]

@[simp] theorem liftRegister_mul (P Q : Matrix J J ℂ) :
    liftRegister (S := S) P * liftRegister Q = liftRegister (P * Q) := by
  ext ⟨j, x⟩ ⟨k, y⟩
  simp only [Matrix.mul_apply, Fintype.sum_prod_type, liftRegister]
  by_cases h : x = y
  · subst y; simp
  · simp [h]

@[simp] theorem liftRegister_one : liftRegister (S := S) (1 : Matrix J J ℂ) = 1 := by
  ext ⟨j, x⟩ ⟨k, y⟩
  by_cases h : j = k <;> by_cases g : x = y <;>
    simp [liftRegister, Matrix.one_apply, Prod.ext_iff, h, g]

theorem liftRegister_unitary (P : Matrix J J ℂ) (hP : P.conjTranspose * P = 1) :
    (liftRegister (S := S) P).conjTranspose * liftRegister P = 1 := by
  simp only [liftRegister_conjTranspose, liftRegister_mul, hP, liftRegister_one]

/-- The ordinary (two-sided PREPARE) LCU circuit. -/
def circuit (L R : Matrix J J ℂ) (U : J → Matrix (A × S) (A × S) ℂ) :
    Matrix (J × (A × S)) (J × (A × S)) ℂ :=
  (liftRegister L).conjTranspose * select U * liftRegister R

/-- The selected block of the actual circuit, not a definition by the desired sum. -/
def selectedBlock (j₀ : J) (a₀ : A) (W : Matrix (J × (A × S)) (J × (A × S)) ℂ) :
    Matrix S S ℂ := W.submatrix (fun x => (j₀, (a₀, x))) (fun x => (j₀, (a₀, x)))

theorem circuit_block (j₀ : J) (a₀ : A) (L R : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ) :
    selectedBlock j₀ a₀ (circuit L R U) =
      ∑ j, (star (L j j₀) * R j j₀) • block a₀ (U j) := by
  ext x y
  simp [selectedBlock, circuit, liftRegister, select, block, Matrix.mul_apply,
    Fintype.sum_prod_type, Matrix.conjTranspose_apply, Matrix.sum_apply,
    Matrix.smul_apply, mul_assoc, mul_comm, mul_left_comm]

theorem unitary_mul (X Y : Matrix S S ℂ)
    (hX : X.conjTranspose * X = 1) (hY : Y.conjTranspose * Y = 1) :
    (X * Y).conjTranspose * (X * Y) = 1 := by
  calc
    (X * Y).conjTranspose * (X * Y) =
        Y.conjTranspose * (X.conjTranspose * X) * Y := by
      simp only [Matrix.conjTranspose_mul, Matrix.mul_assoc]
    _ = 1 := by rw [hX, Matrix.mul_one, hY]

theorem circuit_unitary (L R : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ)
    (hL : L.conjTranspose * L = 1) (hR : R.conjTranspose * R = 1)
    (hU : ∀ j, (U j).conjTranspose * U j = 1) :
    (circuit L R U).conjTranspose * circuit L R U = 1 := by
  apply unitary_mul
  · apply unitary_mul
    · rw [liftRegister_conjTranspose]
      apply liftRegister_unitary
      simpa using (mul_eq_one_comm.mp hL)
    · exact select_unitary U hU
  · exact liftRegister_unitary R hR

/-- Inclusion of the system at the selected ancillary address. -/
def inclusion (a₀ : A) : Matrix (A × S) S ℂ :=
  (1 : Matrix (A × S) (A × S) ℂ).submatrix id (fun x => (a₀, x))

theorem inclusion_isometry (a₀ : A) :
    (inclusion (S := S) a₀).conjTranspose * inclusion (S := S) a₀ = 1 := by
  exact Matrix.submatrix_one_isometry (R := ℂ)
    (e := (id : A × S → A × S)) (f := fun x : S => (a₀, x))
    Function.bijective_id (fun _ _ h => congrArg Prod.snd h)

theorem block_eq_compression (a₀ : A) (U : Matrix (A × S) (A × S) ℂ) :
    block a₀ U = (inclusion (S := S) a₀).conjTranspose * U * inclusion (S := S) a₀ := by
  ext x y
  simp [block, inclusion, Matrix.mul_apply, Fintype.sum_prod_type,
    Matrix.one_apply, Prod.ext_iff, ite_and]

theorem inclusion_norm [Nonempty S] (a₀ : A) : ‖inclusion (S := S) a₀‖ = 1 := by
  have h : ‖inclusion (S := S) a₀‖ * ‖inclusion (S := S) a₀‖ = 1 := by
    rw [← Matrix.l2_opNorm_conjTranspose_mul_self, inclusion_isometry, norm_one]
  nlinarith [norm_nonneg (inclusion (S := S) a₀)]

theorem block_norm_le [Nonempty S] (a₀ : A) (U : Matrix (A × S) (A × S) ℂ) :
    ‖block a₀ U‖ ≤ ‖U‖ := by
  rw [block_eq_compression]
  calc
    ‖(inclusion (S := S) a₀).conjTranspose * U * inclusion (S := S) a₀‖ ≤
        ‖(inclusion (S := S) a₀).conjTranspose * U‖ * ‖inclusion (S := S) a₀‖ :=
      Matrix.l2_opNorm_mul _ _
    _ = ‖(inclusion (S := S) a₀).conjTranspose * U‖ := by rw [inclusion_norm, mul_one]
    _ ≤ ‖(inclusion (S := S) a₀).conjTranspose‖ * ‖U‖ := Matrix.l2_opNorm_mul _ _
    _ = ‖U‖ := by rw [Matrix.l2_opNorm_conjTranspose, inclusion_norm, one_mul]

theorem block_norm_le_one [Nonempty S] (a₀ : A)
    (U : Matrix (A × S) (A × S) ℂ) (hU : U.conjTranspose * U = 1) :
    ‖block a₀ U‖ ≤ 1 := by
  let : Nonempty A := ⟨a₀⟩
  have hnorm : ‖U‖ = 1 :=
    CStarRing.norm_of_mem_unitary (Matrix.mem_unitaryGroup_iff'.mpr hU)
  simpa [hnorm] using block_norm_le a₀ U

/-- Definition 2.1, with spectral norm and an actual unitary matrix. -/
structure BlockEncoding (a₀ : A) (α ε : ℝ) (T : Matrix S S ℂ)
    (U : Matrix (A × S) (A × S) ℂ) : Prop where
  normalization_pos : 0 < α
  error_nonneg : 0 ≤ ε
  unitary : U.conjTranspose * U = 1
  error_le : ‖(α : ℂ) • block a₀ U - T‖ ≤ ε

/-- Reassociate the two ancilla registers as one register. -/
def groupedCircuit (L R : Matrix J J ℂ) (U : J → Matrix (A × S) (A × S) ℂ) :
    Matrix ((J × A) × S) ((J × A) × S) ℂ :=
  (circuit L R U).submatrix (Equiv.prodAssoc J A S) (Equiv.prodAssoc J A S)

@[simp] theorem groupedCircuit_block (j₀ : J) (a₀ : A) (L R : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ) :
    block (j₀, a₀) (groupedCircuit L R U) = selectedBlock j₀ a₀ (circuit L R U) := rfl

theorem groupedCircuit_unitary (L R : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ)
    (hL : L.conjTranspose * L = 1) (hR : R.conjTranspose * R = 1)
    (hU : ∀ j, (U j).conjTranspose * U j = 1) :
    (groupedCircuit L R U).conjTranspose * groupedCircuit L R U = 1 := by
  simp only [groupedCircuit, Matrix.conjTranspose_submatrix]
  rw [Matrix.submatrix_mul_equiv _ _ _ (Equiv.prodAssoc J A S)]
  rw [circuit_unitary L R U hL hR hU, Matrix.submatrix_one_equiv]

theorem lcu_error_bound [Nonempty S] (j₀ : J) (a₀ : A)
    (L R : Matrix J J ℂ) (U : J → Matrix (A × S) (A × S) ℂ)
    (T : J → Matrix S S ℂ) (y : J → ℂ) (α β ε₁ ε₂ : ℝ)
    (hα : 0 < α) (hε₂ : 0 ≤ ε₂)
    (hy : ∑ j, ‖y j‖ ≤ β)
    (hc : ∑ j, ‖(β : ℂ) * (star (L j j₀) * R j j₀) - y j‖ ≤ ε₁)
    (hU : ∀ j, (U j).conjTranspose * U j = 1)
    (herror : ∀ j, ‖(α : ℂ) • block a₀ (U j) - T j‖ ≤ ε₂) :
    ‖((α * β : ℝ) : ℂ) • selectedBlock j₀ a₀ (circuit L R U) -
      ∑ j, y j • T j‖ ≤ α * ε₁ + β * ε₂ := by
  rw [circuit_block]
  let z : J → ℂ := fun j => star (L j j₀) * R j j₀
  have heq : ((α * β : ℝ) : ℂ) • (∑ j, z j • block a₀ (U j)) - ∑ j, y j • T j =
      ∑ j, ((α : ℂ) • (((β : ℂ) * z j - y j) • block a₀ (U j)) +
        y j • ((α : ℂ) • block a₀ (U j) - T j)) := by
    rw [Finset.smul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ext x y
    simp only [Matrix.smul_apply, Matrix.sub_apply, Matrix.add_apply, smul_eq_mul,
      Complex.ofReal_mul]
    ring
  change ‖((α * β : ℝ) : ℂ) • (∑ j, z j • block a₀ (U j)) - ∑ j, y j • T j‖ ≤ _
  rw [heq]
  calc
    ‖∑ j, ((α : ℂ) • (((β : ℂ) * z j - y j) • block a₀ (U j)) +
        y j • ((α : ℂ) • block a₀ (U j) - T j))‖ ≤
        ∑ j, ‖(α : ℂ) • (((β : ℂ) * z j - y j) • block a₀ (U j)) +
          y j • ((α : ℂ) • block a₀ (U j) - T j)‖ := norm_sum_le _ _
    _ ≤ ∑ j, (α * ‖(β : ℂ) * z j - y j‖ + ‖y j‖ * ε₂) := by
      apply Finset.sum_le_sum
      intro j _
      calc
        _ ≤ ‖(α : ℂ) • (((β : ℂ) * z j - y j) • block a₀ (U j))‖ +
            ‖y j • ((α : ℂ) • block a₀ (U j) - T j)‖ := norm_add_le _ _
        _ = α * (‖(β : ℂ) * z j - y j‖ * ‖block a₀ (U j)‖) +
            ‖y j‖ * ‖(α : ℂ) • block a₀ (U j) - T j‖ := by
          simp only [norm_smul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hα]
        _ ≤ α * ‖(β : ℂ) * z j - y j‖ + ‖y j‖ * ε₂ := by
          apply add_le_add
          · apply mul_le_mul_of_nonneg_left _ hα.le
            simpa using (mul_le_mul_of_nonneg_left
              (block_norm_le_one a₀ (U j) (hU j))
              (norm_nonneg ((β : ℂ) * z j - y j)))
          · exact mul_le_mul_of_nonneg_left (herror j) (norm_nonneg (y j))
    _ = α * (∑ j, ‖(β : ℂ) * z j - y j‖) + (∑ j, ‖y j‖) * ε₂ := by
      rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_mul]
    _ ≤ α * ε₁ + β * ε₂ :=
      add_le_add (mul_le_mul_of_nonneg_left hc hα.le)
        (mul_le_mul_of_nonneg_right hy hε₂)

/-- General approximate LCU theorem, numbered 2.2 in the earlier manuscript.
The current manuscript uses its exact specialization as background. The name
is retained for compatibility. Zero coefficient/overlap on unused addresses
permits identity completion without contributing to either sum. -/
theorem theorem_2_2 [Nonempty S] (j₀ : J) (a₀ : A)
    (L R : Matrix J J ℂ) (U : J → Matrix (A × S) (A × S) ℂ)
    (T : J → Matrix S S ℂ) (y : J → ℂ) (α β ε₁ ε₂ : ℝ)
    (hα : 0 < α) (hβ : 0 < β) (hε₁ : 0 ≤ ε₁) (hε₂ : 0 ≤ ε₂)
    (hL : L.conjTranspose * L = 1) (hR : R.conjTranspose * R = 1)
    (hy : ∑ j, ‖y j‖ ≤ β)
    (hc : ∑ j, ‖(β : ℂ) * (star (L j j₀) * R j j₀) - y j‖ ≤ ε₁)
    (hblocks : ∀ j, BlockEncoding a₀ α ε₂ (T j) (U j)) :
    BlockEncoding (j₀, a₀) (α * β) (α * ε₁ + β * ε₂)
      (∑ j, y j • T j) (groupedCircuit L R U) where
  normalization_pos := mul_pos hα hβ
  error_nonneg := add_nonneg (mul_nonneg hα.le hε₁) (mul_nonneg hβ.le hε₂)
  unitary := groupedCircuit_unitary L R U hL hR (fun j => (hblocks j).unitary)
  error_le := by
    rw [groupedCircuit_block]
    exact lcu_error_bound j₀ a₀ L R U T y α β ε₁ ε₂ hα hε₂ hy hc
      (fun j => (hblocks j).unitary) (fun j => (hblocks j).error_le)

/-- Identity branches fill every address absent from the original list. -/
def completeSelect (used : Finset J) (U : J → Matrix (A × S) (A × S) ℂ) :
    J → Matrix (A × S) (A × S) ℂ := fun j => if j ∈ used then U j else 1

/-- The original theorem's finite list, embedded into a larger address register.
Only occupied addresses require block-encoding hypotheses. -/
theorem theorem_2_2_with_unused_addresses [Nonempty S] (j₀ : J) (a₀ : A)
    (used : Finset J) (L R : Matrix J J ℂ)
    (U : J → Matrix (A × S) (A × S) ℂ) (T : J → Matrix S S ℂ)
    (y : J → ℂ) (α β ε₁ ε₂ : ℝ)
    (hα : 0 < α) (hβ : 0 < β) (hε₁ : 0 ≤ ε₁) (hε₂ : 0 ≤ ε₂)
    (hL : L.conjTranspose * L = 1) (hR : R.conjTranspose * R = 1)
    (hy : used.sum (fun j => ‖y j‖) ≤ β)
    (hc : used.sum (fun j => ‖(β : ℂ) * (star (L j j₀) * R j j₀) - y j‖) ≤ ε₁)
    (hunused : ∀ j ∉ used, star (L j j₀) * R j j₀ = 0)
    (hblocks : ∀ j ∈ used, BlockEncoding a₀ α ε₂ (T j) (U j)) :
    BlockEncoding (j₀, a₀) (α * β) (α * ε₁ + β * ε₂)
      (used.sum fun j => y j • T j) (groupedCircuit L R (completeSelect used U)) := by
  let T' : J → Matrix S S ℂ := fun j => if j ∈ used then T j else (α : ℂ) • 1
  let y' : J → ℂ := fun j => if j ∈ used then y j else 0
  have hblocks' : ∀ j, BlockEncoding a₀ α ε₂ (T' j) (completeSelect used U j) := by
    intro j
    by_cases h : j ∈ used
    · simpa [T', completeSelect, h] using hblocks j h
    · refine ⟨hα, hε₂, ?_, ?_⟩
      · simp [completeSelect, h]
      · simpa [completeSelect, T', h] using hε₂
  have hy' : ∑ j, ‖y' j‖ ≤ β := by
    simpa [y', apply_ite norm, Finset.sum_ite] using hy
  have hc' : ∑ j, ‖(β : ℂ) * (star (L j j₀) * R j j₀) - y' j‖ ≤ ε₁ := by
    have heq : (∑ j, ‖(β : ℂ) * (star (L j j₀) * R j j₀) - y' j‖) =
        used.sum (fun j => ‖(β : ℂ) * (star (L j j₀) * R j j₀) - y j‖) := by
      rw [← Finset.sum_subset (Finset.subset_univ used)]
      · apply Finset.sum_congr rfl
        intro j hj; simp [y', hj]
      · intro j _ hj
        simp only [y', ite_eq_right hj, hunused j hj, mul_zero, sub_zero, norm_zero]
    rwa [heq]
  have hsum : (∑ j, y' j • T' j) = used.sum (fun j => y j • T j) := by
    rw [← Finset.sum_subset (Finset.subset_univ used)]
    · apply Finset.sum_congr rfl
      intro j hj; simp [y', T', hj]
    · intro j _ hj; simp [y', hj]
  simpa only [hsum] using theorem_2_2 j₀ a₀ L R (completeSelect used U) T' y'
    α β ε₁ ε₂ hα hβ hε₁ hε₂ hL hR hy' hc' hblocks'

end ChannelFormalization.LCU
