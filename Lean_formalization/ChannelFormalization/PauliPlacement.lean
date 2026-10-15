import ChannelFormalization.Placement
import Physlib.Relativity.PauliMatrices.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum

/-! Concrete Pauli matrices, their binary labels and global phases. The matrix
multiplication law below supplies the missing semantic connection for C.4. -/
noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix
open Matrix
namespace ChannelFormalization.PauliPlacement
open Placement

abbrev BitLabel := ZMod 2 × ZMod 2
abbrev State (n : ℕ) := Fin n → Fin 2
abbrev PMatrix (n : ℕ) := Matrix (State n) (State n) ℂ

/-- The `(x,z)` labels are I=(0,0), X=(1,0), Y=(1,1), Z=(0,1). -/
def atomIndex (p : BitLabel) : Fin 1 ⊕ Fin 3 :=
  if p.1 = 0 then (if p.2 = 0 then .inl 0 else .inr 2)
  else (if p.2 = 0 then .inr 0 else .inr 1)

def atom (p : BitLabel) : Matrix (Fin 2) (Fin 2) ℂ :=
  PauliMatrix.pauliMatrix (atomIndex p)

/-- The exact phase in the 16-entry single-qubit Pauli multiplication table. -/
def atomPhase (p q : BitLabel) : ℂ :=
  if p = 0 ∨ q = 0 ∨ p = q then 1
  else if (p = (1, 0) ∧ q = (1, 1)) ∨ (p = (1, 1) ∧ q = (0, 1)) ∨
      (p = (0, 1) ∧ q = (1, 0)) then Complex.I else -Complex.I

theorem atomPhase_norm (p q : BitLabel) : ‖atomPhase p q‖ = 1 := by
  unfold atomPhase
  split_ifs <;> simp

private theorem bit_cases (x : ZMod 2) : x = 0 ∨ x = 1 := by
  fin_cases x
  · exact Or.inl rfl
  · exact Or.inr rfl

private theorem bit_one_add : (1 : ZMod 2) + 1 = 0 := rfl

/-- Actual Physlib Pauli multiplication induces XOR of the binary labels. -/
theorem atom_mul (p q : BitLabel) : atom p * atom q = atomPhase p q • atom (p + q) := by
  rcases p with ⟨x, z⟩
  rcases q with ⟨y, w⟩
  rcases bit_cases x with rfl | rfl <;>
    rcases bit_cases z with rfl | rfl <;>
    rcases bit_cases y with rfl | rfl <;>
    rcases bit_cases w with rfl | rfl <;>
    simp only [Prod.mk_add_mk, zero_add, add_zero, bit_one_add] <;>
    ext i j <;> fin_cases i <;> fin_cases j <;>
    norm_num [atom, atomIndex, atomPhase, PauliMatrix.pauliMatrix,
      Matrix.mul_apply, Fin.sum_univ_two, Prod.ext_iff]

/-- Tensor product indexed directly by all computational-basis bit strings. -/
def tensor {n} (M : Fin n → Matrix (Fin 2) (Fin 2) ℂ) : PMatrix n :=
  fun x y => ∏ i, M i (x i) (y i)

theorem tensor_mul {n} (M N : Fin n → Matrix (Fin 2) (Fin 2) ℂ) :
    tensor M * tensor N = tensor (fun i => M i * N i) := by
  ext x y
  simp only [tensor, Matrix.mul_apply]
  simp_rw [← Finset.prod_mul_distrib]
  exact (Fintype.prod_sum (fun i z => M i (x i) z * N i z (y i))).symm

theorem tensor_smul {n} (c : Fin n → ℂ) (M : Fin n → Matrix (Fin 2) (Fin 2) ℂ) :
    tensor (fun i => c i • M i) = (∏ i, c i) • tensor M := by
  ext x y
  simp [tensor, Matrix.smul_apply, Finset.prod_mul_distrib]

theorem tensor_one (n : ℕ) : tensor (fun _ : Fin n => (1 : Matrix (Fin 2) (Fin 2) ℂ)) = 1 := by
  ext x y
  by_cases hxy : x = y
  · subst y
    simp [tensor]
  · have he : ∃ i, x i ≠ y i := by
      by_contra! h
      exact hxy (funext h)
    obtain ⟨i, hi⟩ := he
    simp only [tensor, Matrix.one_apply, ite_eq_right_iff.mpr (fun h => False.elim (hxy h))]
    exact Finset.prod_eq_zero (Finset.mem_univ i) (by simp [hi])

def labelMatrix {n} (p : PauliLabel n) : PMatrix n := tensor (fun i => atom (p i))
def multiplicationPhase {n} (p q : PauliLabel n) : ℂ := ∏ i, atomPhase (p i) (q i)

theorem multiplicationPhase_norm {n} (p q : PauliLabel n) :
    ‖multiplicationPhase p q‖ = 1 := by
  simp [multiplicationPhase, norm_prod, atomPhase_norm]

theorem labelMatrix_mul {n} (p q : PauliLabel n) :
    labelMatrix p * labelMatrix q = multiplicationPhase p q • labelMatrix (p + q) := by
  rw [labelMatrix, labelMatrix, tensor_mul]
  simp_rw [atom_mul]
  exact tensor_smul _ _

@[simp] theorem labelMatrix_zero (n : ℕ) : labelMatrix (0 : PauliLabel n) = 1 := by
  simpa [labelMatrix, atom, atomIndex, PauliMatrix.pauliMatrix] using tensor_one n

theorem tensor_conjTranspose {n} (M : Fin n → Matrix (Fin 2) (Fin 2) ℂ) :
    (tensor M).conjTranspose = tensor (fun i => (M i).conjTranspose) := by
  ext x y
  simp [tensor, Matrix.conjTranspose_apply]

theorem labelMatrix_unitary {n} (p : PauliLabel n) :
    (labelMatrix p).conjTranspose * labelMatrix p = 1 := by
  rw [labelMatrix, tensor_conjTranspose, tensor_mul]
  simp only [atom, PauliMatrix.pauliMatrix_selfAdjoint, PauliMatrix.pauliMatrix_mul_self]
  exact tensor_one n

/-- A concrete phased Pauli, including arbitrary unit-modulus global phases. -/
structure Factor (n : ℕ) where
  phase : ℂ
  phase_norm : ‖phase‖ = 1
  label : PauliLabel n

def Factor.denote {n} (p : Factor n) : PMatrix n := p.phase • labelMatrix p.label

def Factor.one (n : ℕ) : Factor n := ⟨1, norm_one, 0⟩

@[simp] theorem Factor.denote_one (n : ℕ) : (Factor.one n).denote = 1 := by
  simp [Factor.one, Factor.denote]

def Factor.multiply {n} (p q : Factor n) : Factor n where
  phase := p.phase * q.phase * multiplicationPhase p.label q.label
  phase_norm := by simp [p.phase_norm, q.phase_norm, multiplicationPhase_norm]
  label := p.label + q.label

theorem Factor.denote_multiply {n} (p q : Factor n) :
    (p.multiply q).denote = p.denote * q.denote := by
  simp only [Factor.denote, Factor.multiply, Matrix.smul_mul, Matrix.mul_smul,
    smul_smul, labelMatrix_mul]
  congr 1
  ring

theorem Factor.phase_ne_zero {n} (p : Factor n) : p.phase ≠ 0 := by
  intro h
  simpa [h] using p.phase_norm

theorem Factor.denote_unitary {n} (p : Factor n) : p.denote.conjTranspose * p.denote = 1 := by
  simp only [Factor.denote, Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul,
    smul_smul, labelMatrix_unitary]
  have h : p.phase * star p.phase = 1 := by
    rw [← starRingEnd_apply, Complex.mul_conj, Complex.normSq_eq_norm_sq,
      p.phase_norm]
    norm_num
  rw [h, one_smul]

def Factor.unitary {n} (p : Factor n) : Matrix.unitaryGroup (State n) ℂ :=
  ⟨p.denote, Matrix.mem_unitaryGroup_iff'.mpr p.denote_unitary⟩

@[simp] theorem Factor.unitary_one (n : ℕ) : (Factor.one n).unitary = 1 := by
  apply Subtype.ext
  exact Factor.denote_one n

@[simp] theorem Factor.unitary_multiply {n} (p q : Factor n) :
    (p.multiply q).unitary = p.unitary * q.unitary := by
  apply Subtype.ext
  exact Factor.denote_multiply p q

/-- The residual label is precisely the XOR recurrence; its phase compensates
for both the accumulated phase and the concrete Pauli multiplication phase. -/
def Factor.residual {n} (target lower : Factor n) : Factor n where
  label := target.label + lower.label
  phase := target.phase / (lower.phase * multiplicationPhase (target.label + lower.label) lower.label)
  phase_norm := by
    simp [target.phase_norm, lower.phase_norm, multiplicationPhase_norm]

@[simp] theorem Factor.residual_label {n} (target lower : Factor n) :
    (target.residual lower).label = pauliResidual target.label lower.label := rfl

/-- C.4 plus phase recovery as an exact identity of actual n-qubit matrices. -/
theorem Factor.residual_exact {n} (target lower : Factor n) :
    (target.residual lower).denote * lower.denote = target.denote := by
  rw [← Factor.denote_multiply]
  unfold Factor.denote Factor.multiply Factor.residual
  dsimp only
  rw [add_assoc, label_add_self, add_zero]
  have hm : multiplicationPhase (target.label + lower.label) lower.label ≠ 0 := by
    intro h
    simpa [h] using multiplicationPhase_norm (target.label + lower.label) lower.label
  congr 1
  field_simp [lower.phase_ne_zero, hm]

@[simp] theorem Factor.residual_unitary {n} (target lower : Factor n) :
    (target.residual lower).unitary * lower.unitary = target.unitary := by
  apply Subtype.ext
  exact Factor.residual_exact target lower

/-- Product of concrete phased Paulis, retaining its derived phase and XOR label. -/
def factorProduct {n} : List (Factor n) → Factor n
  | [] => Factor.one n
  | p :: ps => p.multiply (factorProduct ps)

theorem factorProduct_unitary {n} (ps : List (Factor n)) :
    (factorProduct ps).unitary = (ps.map Factor.unitary).prod := by
  induction ps with
  | nil => exact Factor.unitary_one n
  | cons p ps ih => simp [factorProduct, ih]

theorem factorProduct_label {n} (ps : List (Factor n)) :
    (factorProduct ps).label = (ps.map Factor.label).sum := by
  induction ps with
  | nil => rfl
  | cons p ps ih => simp [factorProduct, Factor.multiply, ih]

section Reconstruction
variable {A : Type*} [PartialOrder A] [DecidableLE A] [DecidableEq A] {n : ℕ}

def prefixFactor (order : List A) (U : A → Factor n) (b : A) : Factor n :=
  factorProduct ((order.map fun a => if a ≤ b then U a else Factor.one n).reverse)

theorem prefixFactor_unitary (order : List A) (U : A → Factor n) (b : A) :
    (prefixFactor order U b).unitary = orderedAction order (fun a => (U a).unitary) b := by
  rw [prefixFactor, factorProduct_unitary]
  simp only [List.map_reverse, List.map_map, Function.comp_def, orderedAction]
  congr 2
  apply List.map_congr_left
  intro a _
  split_ifs <;> simp

theorem prefixFactor_label (order : List A) (U : A → Factor n) (b : A) :
    (prefixFactor order U b).label =
      (order.map fun a => if a ≤ b then (U a).label else 0).sum := by
  rw [prefixFactor, factorProduct_label]
  simp only [List.map_reverse, List.sum_reverse, List.map_map, Function.comp_def]
  congr 1
  apply List.map_congr_left
  intro a _
  split_ifs <;> rfl

/-- Construct the phase-free residuals and recover their phases as addresses
are processed. Every branch equation is derived from `Factor.residual_exact`. -/
theorem exists_phase_reconstruction_on_list (l : List A)
    (hl : l.Pairwise (fun a b => ¬ b ≤ a)) (S : A → Prop) [DecidablePred S]
    (V : A → Factor n) :
    ∃ U : A → Factor n,
      (∀ b ∈ l, S b → orderedAction l (fun a => (U a).unitary) b = (V b).unitary) ∧
      (∀ b, b ∉ l ∨ ¬ S b → U b = Factor.one n) := by
  induction l using List.reverseRecOn with
  | nil =>
      refine ⟨fun _ => Factor.one n, ?_, ?_⟩
      · simp
      · simp
  | append_singleton l a ih =>
      have hpair := List.pairwise_append.mp hl
      have hlast : ∀ b ∈ l, ¬ a ≤ b := by
        intro b hb
        exact hpair.2.2 b hb a (by simp)
      have ha : a ∉ l := fun h => hlast a h le_rfl
      obtain ⟨U, hU, hzero⟩ := ih hpair.1
      let r : Factor n := if S a then (V a).residual (prefixFactor l U a) else Factor.one n
      let U' : A → Factor n := Function.update U a r
      have hprefix (b : A) :
          orderedAction l (fun c => (U' c).unitary) b =
            orderedAction l (fun c => (U c).unitary) b := by
        apply orderedAction_congr
        intro c hc
        have hca : c ≠ a := by intro h; exact ha (h ▸ hc)
        simp [U', Function.update_of_ne hca]
      refine ⟨U', ?_, ?_⟩
      · intro b hb hS
        rcases List.mem_append.mp hb with hb | hb
        · rw [orderedAction_append]
          simp only [hlast b hb, ↓reduceIte, one_mul]
          rw [hprefix]
          exact hU b hb hS
        · have hba : b = a := by simpa using hb
          subst b
          rw [orderedAction_append]
          simp only [le_refl, ↓reduceIte]
          rw [hprefix]
          change (Function.update U a r a).unitary *
            orderedAction l (fun c => (U c).unitary) a = (V a).unitary
          rw [Function.update_self, ← prefixFactor_unitary]
          simp only [r, hS, ↓reduceIte]
          exact Factor.residual_unitary _ _
      · intro b hb
        by_cases hba : b = a
        · subst b
          have hnS : ¬ S a := by
            rcases hb with hb | hb
            · exact False.elim (hb (by simp))
            · exact hb
          simp [U', r, hnS]
        · have hb' : b ∉ l ∨ ¬ S b := by
            rcases hb with hb | hb
            · exact Or.inl (fun h => hb (List.mem_append_left _ h))
            · exact Or.inr hb
          simpa [U', Function.update_of_ne hba] using hzero b hb'

/-- Full phase-restored Pauli placement on the constrained addresses. -/
theorem exists_phase_reconstruction (order : Placement.LinearExtension A)
    (S : A → Prop) [DecidablePred S] (V : A → Factor n) :
    ∃ U : A → Factor n,
      (∀ b, S b → orderedAction order.order (fun a => (U a).unitary) b = (V b).unitary) ∧
      (∀ b, ¬ S b → U b = Factor.one n) := by
  obtain ⟨U, hU, hzero⟩ := exists_phase_reconstruction_on_list order.order order.sorted S V
  exact ⟨U, fun b hb => hU b (order.complete b) hb, fun b hb => hzero b (Or.inr hb)⟩

/-- The real product of controlled, phase-restored Pauli gates implements the
original SELECT exactly. The witness is produced by the XOR residual algorithm. -/
theorem exists_exact_pauli_select [Fintype A] (order : Placement.LinearExtension A) (V : A → Factor n) :
    ∃ U : A → Factor n,
      optimizedSelect order.order (fun a => (U a).unitary) = LCU.select (fun b => (V b).denote) := by
  obtain ⟨U, hU, _⟩ := exists_phase_reconstruction order (fun _ => True) V
  refine ⟨U, ?_⟩
  exact optimizedSelect_exact order.order (fun a => (U a).unitary)
    (fun b => (V b).unitary) (fun b => hU b trivial)

end Reconstruction

end ChannelFormalization.PauliPlacement
