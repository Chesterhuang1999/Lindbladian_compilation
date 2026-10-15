import ChannelFormalization.PauliCost
import Mathlib.LinearAlgebra.Dimension.Constructions

/-! The lower bound in Example 5.3, with the zero address treated correctly.
The zero-address residual is an affine offset and has zero control cost. Full
Pauli coverage forces the residuals at nonzero addresses to span all 2n binary
coordinates. This proves the lower bound for the paper's weighted Pauli cost;
it does not count the additional gates implementing pure residual phases. -/
noncomputable section
open scoped BigOperators
namespace ChannelFormalization.PauliCost
open Placement PauliPlacement Submodule Module

/-- The phase-free label seen at a branch of the monotone-control circuit. -/
def branchLabel {b n : ℕ} (g : (Fin b → Fin 2) → PauliLabel n)
    (a : Fin b → Fin 2) : PauliLabel n :=
  ∑ t : Fin b → Fin 2, if t ≤ a then g t else 0

/-- These branch labels are those of the actual ordered phased-Pauli product,
whose matrix semantics is `prefixFactor_unitary`. -/
theorem branchLabel_eq_prefixFactor {b n : ℕ}
    (order : Placement.LinearExtension (Fin b → Fin 2))
    (g : (Fin b → Fin 2) → Factor n) (a : Fin b → Fin 2) :
    branchLabel (fun t => (g t).label) a =
      (prefixFactor order.order g a).label := by
  classical
  rw [prefixFactor_label]
  have hn : order.order.Nodup := order.sorted.imp (by
    intro x y h hxy
    subst y
    exact h le_rfl)
  have he : order.order.toFinset = Finset.univ := by
    ext x
    simp [order.complete]
  rw [← List.sum_toFinset _ hn, he]
  rfl

def positiveResiduals {b n : ℕ} (g : (Fin b → Fin 2) → PauliLabel n) :
    Finset (Fin b → Fin 2) := by
  classical
  exact Finset.univ.filter (fun t => t ≠ 0 ∧ g t ≠ 0)

theorem positiveResiduals_span {b n : ℕ} (g : (Fin b → Fin 2) → PauliLabel n)
    (hfull : Function.Surjective (branchLabel g)) :
    Submodule.span (ZMod 2) (g '' (positiveResiduals g : Set (Fin b → Fin 2))) = ⊤ := by
  classical
  apply top_unique
  intro q _
  obtain ⟨a, ha⟩ := hfull (q + g 0)
  let s : Finset (Fin b → Fin 2) := Finset.univ.filter (fun t => t ≤ a)
  have hz : (0 : Fin b → Fin 2) ∈ s := by
    simp only [s, Finset.mem_filter, Finset.mem_univ, true_and]
    intro i
    exact Fin.zero_le _
  have hs : (∑ t ∈ s.erase 0, g t) = q := by
    apply add_right_cancel (b := g 0)
    rw [Finset.sum_erase_add _ _ hz]
    simpa [branchLabel, s, Finset.sum_filter] using ha
  rw [← hs]
  apply Submodule.sum_mem
  intro t ht
  by_cases hg : g t = 0
  · simp [hg]
  · apply Submodule.subset_span
    refine ⟨t, ?_, rfl⟩
    simp only [positiveResiduals, Finset.mem_coe, Finset.mem_filter,
      Finset.mem_univ, true_and]
    exact ⟨(Finset.mem_erase.mp ht).1, hg⟩

theorem pauliLabel_finrank (n : ℕ) : finrank (ZMod 2) (PauliLabel n) = 2 * n := by
  simp [PauliLabel, Module.finrank_pi_fintype, Module.finrank_prod, Nat.mul_comm]

theorem positiveResiduals_card_lower_bound {b n : ℕ}
    (g : (Fin b → Fin 2) → PauliLabel n)
    (hfull : Function.Surjective (branchLabel g)) :
    2 * n ≤ (positiveResiduals g).card := by
  classical
  have hspan := positiveResiduals_span g hfull
  have hc := finrank_span_finset_le_card (R := ZMod 2)
    ((positiveResiduals g).image g)
  have he : ((↑((positiveResiduals g).image g) : Set (PauliLabel n))) =
      g '' (positiveResiduals g : Set (Fin b → Fin 2)) := by simp
  change finrank (ZMod 2) (Submodule.span (ZMod 2)
    (↑((positiveResiduals g).image g) : Set (PauliLabel n))) ≤ _ at hc
  rw [he, hspan, finrank_top, pauliLabel_finrank] at hc
  exact hc.trans (Finset.card_image_le)

theorem controlWeight_pos {b : ℕ} (a : Fin b → Fin 2) (ha : a ≠ 0) :
    0 < controlWeight a := by
  rw [controlWeight, Finset.card_pos]
  obtain ⟨i, hi⟩ := Function.ne_iff.mp ha
  exact ⟨i, by simpa using hi⟩

theorem pauliWeight_pos {n : ℕ} (p : PauliLabel n) (hp : p ≠ 0) :
    0 < pauliWeight p := by
  classical
  rw [pauliWeight, Finset.card_pos]
  obtain ⟨i, hi⟩ := Function.ne_iff.mp hp
  exact ⟨i, by simpa using hi⟩

/-- Corrected Example 5.3 lower bound. It is valid even when the zero-address
residual is nonidentity, because that residual contributes only an affine offset. -/
theorem example_5_3_lower_bound {b n : ℕ} (g : (Fin b → Fin 2) → PauliLabel n)
    (hfull : Function.Surjective (branchLabel g)) :
    2 * n ≤ ∑ a, gateCost a (g a) := by
  classical
  apply (positiveResiduals_card_lower_bound g hfull).trans
  calc
    (positiveResiduals g).card = ∑ a ∈ positiveResiduals g, 1 := by simp
    _ ≤ ∑ a ∈ positiveResiduals g, gateCost a (g a) := by
      apply Finset.sum_le_sum
      intro a ha
      have h := (Finset.mem_filter.mp ha).2
      rw [gateCost_eq]
      exact Nat.mul_pos (controlWeight_pos a h.1) (pauliWeight_pos (g a) h.2)
    _ ≤ ∑ a, gateCost a (g a) := by
      apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
      intros
      exact Nat.zero_le _

/-- The same lower bound for full coverage by concrete phased-Pauli products. -/
theorem example_5_3_circuit_lower_bound {b n : ℕ}
    (order : Placement.LinearExtension (Fin b → Fin 2))
    (g : (Fin b → Fin 2) → Factor n)
    (hfull : Function.Surjective (fun a => (prefixFactor order.order g a).label)) :
    2 * n ≤ ∑ a, gateCost a (g a).label := by
  apply example_5_3_lower_bound
  intro q
  obtain ⟨a, ha⟩ := hfull q
  exact ⟨a, (branchLabel_eq_prefixFactor order g a).trans ha⟩

end ChannelFormalization.PauliCost
