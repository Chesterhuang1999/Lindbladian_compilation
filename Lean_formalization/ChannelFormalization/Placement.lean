import ChannelFormalization.LCU
import Mathlib.Data.ZMod.Basic
import Mathlib.Order.Extension.Linear
import Mathlib.Data.Finset.Sort
import Mathlib.Algebra.BigOperators.Group.List.Basic

/-! Address relabeling and the constructive group-theoretic monotone-control
factorization. The group theorem specializes to arbitrary unitary matrices.
Phase-free Pauli labels are binary symplectic vectors. Their recurrence and
phase restoration are separate interfaces: the matrix-to-label quotient theorem
is not assumed or claimed by the binary-vector results. -/
noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix
namespace ChannelFormalization.Placement

section Relabel
variable {J A V : Type*} [Fintype J] [Fintype A] [DecidableEq A]

/-- Place values at injectively chosen addresses, and use zero elsewhere. -/
def addressTable [AddCommMonoid V] (e : J ↪ A) (v : J → V) (a : A) : V :=
  ∑ j, if e j = a then v j else 0

@[simp] theorem addressTable_at [AddCommMonoid V] (e : J ↪ A) (v : J → V) (j : J) :
    addressTable e v (e j) = v j := by
  classical
  simp [addressTable, e.injective.eq_iff]

theorem addressTable_unused [AddCommMonoid V] (e : J ↪ A) (v : J → V) (a : A)
    (ha : a ∉ Set.range e) : addressTable e v a = 0 := by
  apply Finset.sum_eq_zero
  intro j _
  simp only [ite_eq_right_iff]
  intro hj
  exact False.elim (ha ⟨j, hj⟩)

/-- Padding and injective relabeling preserve every finite additive sum. -/
theorem sum_addressTable [AddCommMonoid V] (e : J ↪ A) (v : J → V) :
    ∑ a, addressTable e v a = ∑ j, v j := by
  simp only [addressTable]
  rw [Finset.sum_comm]
  simp

theorem map_addressTable {W : Type*} [AddCommMonoid V] [AddCommMonoid W]
    (e : J ↪ A) (v : J → V) (f : V → W) (hf : f 0 = 0) (a : A) :
    f (addressTable e v a) = addressTable e (fun j => f (v j)) a := by
  by_cases ha : a ∈ Set.range e
  · obtain ⟨j, rfl⟩ := ha
    simp
  · rw [addressTable_unused e v a ha,
      addressTable_unused e (fun j => f (v j)) a ha, hf]

/-- In particular, the LCU normalizer (sum of coefficient magnitudes) is unchanged. -/
theorem sum_norm_addressTable (e : J ↪ A) (β : J → ℂ) :
    ∑ a, ‖addressTable e β a‖ = ∑ j, ‖β j‖ := by
  simp_rw [map_addressTable e β norm norm_zero]
  exact sum_addressTable e (fun j => ‖β j‖)

/-- The target on an unused address may be chosen freely; identity is convenient. -/
def targetTable {G : Type*} [One G] (e : J ↪ A) (P : J → G) : A → G :=
  Function.extend e P (fun _ => 1)

@[simp] theorem targetTable_at {G : Type*} [One G] (e : J ↪ A) (P : J → G) (j : J) :
    targetTable e P (e j) = P j := e.injective.extend_apply _ _ _

theorem targetTable_unused {G : Type*} [One G] (e : J ↪ A) (P : J → G)
    (a : A) (ha : a ∉ Set.range e) : targetTable e P a = 1 := by
  exact Function.extend_apply' P (fun _ => 1) a ha

theorem targetTable_unitary {T : Type*} [Fintype T] [DecidableEq T]
    (e : J ↪ A) (P : J → Matrix T T ℂ)
    (hP : ∀ j, (P j).conjTranspose * P j = 1) (a : A) :
    (targetTable e P a).conjTranspose * targetTable e P a = 1 := by
  by_cases ha : a ∈ Set.range e
  · obtain ⟨j, rfl⟩ := ha
    simpa using hP j
  · rw [targetTable_unused e P a ha]
    simp

/-- Lemma C.1's operator identity, with exact zero amplitudes on unused addresses. -/
theorem sum_smul_addressTable {S : Type*} [Fintype S] [DecidableEq S]
    (e : J ↪ A) (β : J → ℂ) (P : J → Matrix S S ℂ) :
    ∑ a, addressTable e β a • targetTable e P a = ∑ j, β j • P j := by
  have h (a : A) : addressTable e β a • targetTable e P a =
      addressTable e (fun j => β j • P j) a := by
    by_cases ha : a ∈ Set.range e
    · obtain ⟨j, rfl⟩ := ha
      simp
    · simp [addressTable_unused e β a ha,
        addressTable_unused e (fun j => β j • P j) a ha]
  simp_rw [h]
  exact sum_addressTable e _

/-- The normalized algebraic block is invariant; `LCU.circuit_block` connects
this equality to the selected block of a constructed PREPARE/SELECT circuit. -/
theorem lemma_C_1_algebraic {S : Type*} [Fintype S] [DecidableEq S]
    (e : J ↪ A) (β : J → ℂ) (P : J → Matrix S S ℂ) :
    (((∑ a, ‖addressTable e β a‖)⁻¹ : ℝ) : ℂ) •
        (∑ a, addressTable e β a • targetTable e P a) =
      (((∑ j, ‖β j‖)⁻¹ : ℝ) : ℂ) • (∑ j, β j • P j) := by
  rw [sum_norm_addressTable, sum_smul_addressTable]

end Relabel

section Monotone
variable {A G : Type*} [PartialOrder A] [DecidableLE A] [DecidableEq A] [Group G]

/-- A chosen linear extension, explicitly ordered from first-applied to last-applied.
The pairwise condition says a later address cannot be below an earlier one. -/
structure LinearExtension (A : Type*) [PartialOrder A] where
  order : List A
  sorted : order.Pairwise (fun a b => ¬ b ≤ a)
  complete : ∀ a, a ∈ order

/-- Ordered system action; gates appearing later in the list multiply on the left. -/
def orderedAction (order : List A) (U : A → G) (b : A) : G :=
  ((order.map fun a => if a ≤ b then U a else 1).reverse).prod

@[simp] theorem orderedAction_nil (U : A → G) (b : A) : orderedAction [] U b = 1 := rfl

@[simp] theorem orderedAction_append (l : List A) (a : A) (U : A → G) (b : A) :
    orderedAction (l ++ [a]) U b = (if a ≤ b then U a else 1) * orderedAction l U b := by
  simp [orderedAction, List.reverse_append]

theorem orderedAction_congr (l : List A) (U V : A → G) (b : A)
    (h : ∀ a ∈ l, U a = V a) : orderedAction l U b = orderedAction l V b := by
  unfold orderedAction
  congr 2
  apply List.map_congr_left
  intro a ha
  rw [h a ha]

theorem orderedAction_update (l : List A) (U : A → G) (a b : A) (u : G)
    (ha : a ∉ l) : orderedAction l (Function.update U a u) b = orderedAction l U b := by
  apply orderedAction_congr
  intro x hx
  apply Function.update_of_ne
  intro h
  exact ha (h ▸ hx)

/-- Construction by the fixed gate order. At each new constrained address `a`,
the actual new factor is `V a * (orderedAction l U a)⁻¹`. Earlier equations are
proved invariant, not taken as an assumption on the output family. -/
theorem exists_factors_on_list (l : List A)
    (hl : l.Pairwise (fun a b => ¬ b ≤ a)) (S : A → Prop) [DecidablePred S]
    (V : A → G) :
    ∃ U : A → G,
      (∀ b ∈ l, S b → orderedAction l U b = V b) ∧
      (∀ b, b ∉ l ∨ ¬ S b → U b = 1) := by
  induction l using List.reverseRecOn with
  | nil =>
      refine ⟨fun _ => 1, ?_, ?_⟩
      · simp
      · simp
  | append_singleton l a ih =>
      have hpair := List.pairwise_append.mp hl
      have hlast : ∀ b ∈ l, ¬ a ≤ b := by
        intro b hb
        exact hpair.2.2 b hb a (by simp)
      have ha : a ∉ l := by
        intro ha
        exact hlast a ha le_rfl
      obtain ⟨U, hU, hzero⟩ := ih hpair.1
      let u : G := if S a then V a * (orderedAction l U a)⁻¹ else 1
      let U' : A → G := Function.update U a u
      refine ⟨U', ?_, ?_⟩
      · intro b hb hS
        rcases List.mem_append.mp hb with hb | hb
        · have hne : b ≠ a := by intro h; exact ha (h ▸ hb)
          rw [orderedAction_append]
          simp only [hlast b hb, ↓reduceIte, one_mul]
          rw [show orderedAction l U' b = orderedAction l U b from
            orderedAction_update l U a b u ha]
          exact hU b hb hS
        · have hba : b = a := by simpa using hb
          subst b
          rw [orderedAction_append]
          simp only [le_refl, ↓reduceIte]
          change Function.update U a u a * orderedAction l (Function.update U a u) a = V a
          rw [Function.update_self, orderedAction_update l U a a u ha]
          simp [u, hS, mul_assoc]
      · intro b hb
        by_cases hba : b = a
        · subst b
          have hnS : ¬ S a := by
            rcases hb with hb | hb
            · exact False.elim (hb (by simp))
            · exact hb
          simp [U', u, hnS]
        · have hb' : b ∉ l ∨ ¬ S b := by
            rcases hb with hb | hb
            · exact Or.inl (fun h => hb (List.mem_append_left _ h))
            · exact Or.inr hb
          simpa [U', Function.update_of_ne hba] using hzero b hb'

/-- Lemma C.2 for any finite partially ordered address set and any group.
Taking `G = Matrix.unitaryGroup S ℂ` gives exactly arbitrary unitary branches. -/
theorem lemma_C_2 (order : LinearExtension A) (S : A → Prop) [DecidablePred S]
    (V : A → G) :
    ∃ U : A → G, (∀ b, S b → V b = orderedAction order.order U b) ∧
      (∀ b, ¬ S b → U b = 1) := by
  obtain ⟨U, hU, hzero⟩ := exists_factors_on_list order.order order.sorted S V
  exact ⟨U, fun b hb => (hU b (order.complete b) hb).symm,
    fun b hb => hzero b (Or.inr hb)⟩

/-- The explicit arbitrary-unitary specialization of Lemma C.2. -/
theorem lemma_C_2_unitary {T : Type*} [Fintype T] [DecidableEq T]
    (order : LinearExtension A) (S : A → Prop) [DecidablePred S]
    (V : A → Matrix.unitaryGroup T ℂ) :
    ∃ U : A → Matrix.unitaryGroup T ℂ,
      (∀ b, S b → V b = orderedAction order.order U b) ∧
      (∀ b, ¬ S b → U b = 1) :=
  lemma_C_2 order S V

end Monotone

/-- Every finite address poset admits the required gate-application order.
This includes bitwise-ordered Boolean control registers of every width. -/
def finiteLinearExtension (A : Type*) [PartialOrder A] [Fintype A] : LinearExtension A := by
  classical
  letI : Fintype (_root_.LinearExtension A) := inferInstanceAs (Fintype A)
  let l := (Finset.univ : Finset (_root_.LinearExtension A)).sort (· ≤ ·)
  refine ⟨l.map (fun a => (show A from a)), ?_, ?_⟩
  · apply List.pairwise_map.mpr
    apply ((Finset.sortedLT_sort (Finset.univ : Finset (_root_.LinearExtension A))).pairwise).imp
    intro a b hab hba
    exact (not_le_of_gt hab) ((_root_.toLinearExtension (α := A)).monotone hba)
  · intro a
    apply List.mem_map.mpr
    refine ⟨_root_.toLinearExtension (α := A) a, ?_, rfl⟩
    exact (Finset.mem_sort (α := _root_.LinearExtension A)
      (s := Finset.univ) (r := (· ≤ ·))).mpr (Finset.mem_univ _)

section PauliLabels

/-- Binary symplectic Pauli labels `(x,z)`, one pair for each system qubit.
Pointwise addition is the bitwise XOR used in Appendix C. -/
abbrev PauliLabel (n : ℕ) := Fin n → ZMod 2 × ZMod 2

theorem label_neg_eq_self {n} (p : PauliLabel n) : -p = p := by
  funext q
  apply Prod.ext <;> apply ZMod.neg_eq_self_mod_two

theorem label_add_self {n} (p : PauliLabel n) : p + p = 0 := by
  calc
    p + p = p + -p := congrArg (fun q => p + q) (label_neg_eq_self p).symm
    _ = 0 := add_neg_cancel p

/-- The unique residual label prescribed by the recurrence. -/
def pauliResidual {n} (target lower : PauliLabel n) : PauliLabel n := target + lower

/-- Lemma C.4's XOR recurrence reconstructs the target label. -/
theorem lemma_C_4_reconstruct {n} (target lower : PauliLabel n) :
    pauliResidual target lower + lower = target := by
  simp [pauliResidual, add_assoc, label_add_self]

theorem lemma_C_4_unique {n} (target lower residual : PauliLabel n)
    (h : residual + lower = target) : residual = pauliResidual target lower := by
  rw [pauliResidual, ← h, add_assoc, label_add_self, add_zero]

/-- Accumulated strict-lower contributions after a finite set of gates is fixed. -/
def lowerContribution {A : Type*} [PartialOrder A] [DecidableLT A] {n}
    (fixed : Finset A) (g : A → PauliLabel n) (b : A) : PauliLabel n :=
  ∑ a ∈ fixed, if a < b then g a else 0

/-- Fixing gate `a` updates every strict superset `b` by XOR with its residual. -/
theorem lemma_C_4_update {A : Type*} [PartialOrder A] [DecidableLT A] [DecidableEq A]
    {n} (fixed : Finset A) (g : A → PauliLabel n) (a b : A)
    (ha : a ∉ fixed) (hab : a < b) :
    lowerContribution (insert a fixed) g b = lowerContribution fixed g b + g a := by
  simp [lowerContribution, Finset.sum_insert ha, hab, add_comm]

/-- An address outside a gate's strict upper set needs no accumulator update. -/
theorem lowerContribution_insert_unaffected {A : Type*} [PartialOrder A]
    [DecidableLT A] [DecidableEq A] {n} (fixed : Finset A) (g : A → PauliLabel n)
    (a b : A) (ha : a ∉ fixed) (hab : ¬ a < b) :
    lowerContribution (insert a fixed) g b = lowerContribution fixed g b := by
  simp [lowerContribution, Finset.sum_insert ha, hab]

/-- The actual phase-restoration matrix identity. The premise is explicitly a
phase-equivalence witness; the binary label theorems above do not establish it. -/
theorem phase_restore {S : Type*} [Fintype S] (u : ℂ) (hu : u ≠ 0)
    (bare lower target : Matrix S S ℂ) (h : bare * lower = u • target) :
    (u⁻¹ • bare) * lower = target := by
  rw [Matrix.smul_mul, h, smul_smul, inv_mul_cancel₀ hu, one_smul]

end PauliLabels

section SelectCircuit
variable {A T : Type*} [PartialOrder A] [DecidableLE A] [DecidableEq A]
  [Fintype A] [Fintype T] [DecidableEq T]

def monotoneGate (a : A) (U : Matrix.unitaryGroup T ℂ) :
    Matrix (A × T) (A × T) ℂ :=
  LCU.select (fun b => if a ≤ b then U.val else 1)

def optimizedSelect (order : List A) (U : A → Matrix.unitaryGroup T ℂ) :
    Matrix (A × T) (A × T) ℂ :=
  ((order.map fun a => monotoneGate a (U a)).reverse).prod

/-- The actual product of monotone controlled gates implements the ordered
branch action, with later-applied gates multiplying on the left. -/
theorem optimizedSelect_eq_select (order : List A) (U : A → Matrix.unitaryGroup T ℂ) :
    optimizedSelect order U = LCU.select (fun b => (orderedAction order U b).val) := by
  induction order with
  | nil => simp [optimizedSelect, orderedAction, LCU.select_one]
  | cons a l ih =>
      have hstep : optimizedSelect (a :: l) U = optimizedSelect l U * monotoneGate a (U a) := by
        simp [optimizedSelect, List.reverse_cons]
      rw [hstep, ih, monotoneGate, LCU.select_mul]
      congr 1
      funext b
      simp only [orderedAction, List.map_cons, List.reverse_cons, List.prod_append,
        List.prod_cons, List.prod_nil, mul_one]
      split_ifs <;> simp

/-- Complete reconstruction implies exact equality of SELECT matrices, and the
proof goes through the product of controlled factors above. -/
theorem optimizedSelect_exact (order : List A) (U V : A → Matrix.unitaryGroup T ℂ)
    (h : ∀ b, orderedAction order U b = V b) :
    optimizedSelect order U = LCU.select (fun b => (V b).val) := by
  rw [optimizedSelect_eq_select]
  simp_rw [h]

end SelectCircuit

end ChannelFormalization.Placement
