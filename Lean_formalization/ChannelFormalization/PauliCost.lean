import ChannelFormalization.PauliPlacement

/-! Appendix C.4's local contribution to the paper's weighted-control metric.
This is the stated cost model, not a claim of optimal elementary-gate synthesis. -/
noncomputable section
open scoped BigOperators
namespace ChannelFormalization.PauliCost
open Placement

def controlWeight {b : ℕ} (address : Fin b → Fin 2) : ℕ :=
  (Finset.univ.filter (fun i => address i ≠ 0)).card

def pauliWeight {n : ℕ} (p : PauliLabel n) : ℕ := by
  classical
  exact (Finset.univ.filter (fun i => p i ≠ 0)).card

/-- Each nonidentity single-qubit factor has one controlled operation, charged
by the number of its positive controls, as in Problem 5.1. -/
def gateCost {b n : ℕ} (address : Fin b → Fin 2) (p : PauliLabel n) : ℕ := by
  classical
  exact ∑ i, if p i ≠ 0 then controlWeight address else 0

theorem gateCost_eq {b n : ℕ} (address : Fin b → Fin 2) (p : PauliLabel n) :
    gateCost address p = controlWeight address * pauliWeight p := by
  classical
  simp [gateCost, pauliWeight, Finset.sum_ite, mul_comm]

/-- The local objective contribution for the uniquely determined XOR residual. -/
theorem lemma_C_4_local_cost {b n : ℕ} (address : Fin b → Fin 2)
    (target lower : PauliLabel n) :
    gateCost address (pauliResidual target lower) =
      controlWeight address * pauliWeight (target + lower) :=
  gateCost_eq address _

/-- Adding a new address adds exactly its local contribution to the objective. -/
theorem weightedCost_insert {b n : ℕ} (fixed : Finset (Fin b → Fin 2))
    (labels : (Fin b → Fin 2) → PauliLabel n) (address : Fin b → Fin 2)
    (h : address ∉ fixed) :
    (∑ a ∈ insert address fixed, gateCost a (labels a)) =
      (∑ a ∈ fixed, gateCost a (labels a)) + controlWeight address * pauliWeight (labels address) := by
  rw [Finset.sum_insert h, gateCost_eq]
  exact Nat.add_comm _ _

end ChannelFormalization.PauliCost
