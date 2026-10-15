import ChannelFormalization.PolynomialRank
import ChannelFormalization.GuardedRewrites

/-! The current Proposition 6.1: actual sampled trajectories are expanded into
Pauli terms and rewritten by the guarded deterministic rules. Their output
rank bound follows from those rules, without an assumed row-cleanup certificate.
The argument applies to every finite rewrite prefix, including any normal form. -/
noncomputable section
open scoped BigOperators Matrix
open Matrix
namespace ChannelFormalization
open PauliPlacement Placement ChannelIR

namespace FrontendCleanup

/-- The paper's phase-free Pauli primitive for a binary Pauli label. -/
def pauliPrimitive {n : ℕ} (p : PauliLabel n) : Primitive n :=
  .pauli 0 (fun i => atomIndex (p i))

@[simp] theorem denote_pauliPrimitive {n : ℕ} (p : PauliLabel n) :
    (pauliPrimitive p).denote = labelMatrix p := by
  ext x y
  simp [pauliPrimitive, Primitive.denote, labelMatrix, tensor, atom]

/-- The full finite Pauli expansion. Zero coefficients are permitted by T-Kraus
and can subsequently be removed by R-ZeroTerm. -/
def pauliTerms {n : ℕ} (A : PMatrix n) : Guarded.Kraus n :=
  Finset.univ.toList.map (fun p => (pauliCoordinates n A p, pauliPrimitive p))

@[simp] theorem termsDenote_pauliTerms {n : ℕ} (A : PMatrix n) :
    termsDenote (pauliTerms A) = A := by
  unfold pauliTerms termsDenote
  simp only [List.map_map, Function.comp_def, denote_pauliPrimitive]
  rw [← List.sum_toFinset _ (Finset.nodup_toList _), Finset.toList_toFinset]
  exact pauliCoordinates_reconstruct A

/-- A fixed jump sequence's concrete sampled family from main-text Eq. (6)
(Appendix B.2 Eq. (8)), with square-root quadrature weights. -/
def sampledPauliChannel (n k p M : ℕ) (J : PMatrix n) (δ : ℝ)
    (L : Fin k → PMatrix n) (samples : Fin M → Fin k → ℝ)
    (weights : Fin M → ℝ) : Guarded.Channel n :=
  List.ofFn (fun a => pauliTerms ((Real.sqrt (weights a) : ℂ) •
    sampledTrajectory k p J (δ : ℂ) L (fun i => (samples a i : ℂ))))

@[simp] theorem denote_sampledPauliChannel (n k p M : ℕ) (J : PMatrix n) (δ : ℝ)
    (L : Fin k → PMatrix n) (samples : Fin M → Fin k → ℝ)
    (weights : Fin M → ℝ) :
    Guarded.denote (sampledPauliChannel n k p M J δ L samples weights) =
      MatrixMap.of_kraus
        (fun a => (Real.sqrt (weights a) : ℂ) •
          sampledTrajectory k p J (δ : ℂ) L (fun i => (samples a i : ℂ)))
        (fun a => (Real.sqrt (weights a) : ℂ) •
          sampledTrajectory k p J (δ : ℂ) L (fun i => (samples a i : ℂ))) := by
  simp [Guarded.denote, Guarded.toChannel, ChannelIR.Channel.denote,
    sampledPauliChannel, List.map_ofFn, List.sum_ofFn, MatrixMap.of_kraus_eq_sum_conj]

/-- Actual Pauli coefficients of a flattened ChannelIR program. -/
def channelPauliMatrix {n : ℕ} (C : Guarded.Channel n) :
    Matrix (Fin C.length) (PauliLabel n) ℂ :=
  pauliCoefficientMatrix (fun i => termsDenote (C.get i))

/-- The new concrete deterministic rules preserve the full Pauli coefficient
rank, a stronger fact than the nonincrease required by Proposition 6.1. -/
theorem rank_eq_of_steps {n : ℕ} {C D : Guarded.Channel n}
    (h : Guarded.Steps C D) :
    (channelPauliMatrix C).rank = (channelPauliMatrix D).rank := by
  apply rank_pauliCoefficientMatrix_eq_of_cp_eq
  simpa only [← Guarded.denote_eq_of_kraus] using h.sound

/-- One guarded rule is already sufficient for the concrete rank bridge. -/
theorem rank_eq_of_step {n : ℕ} {C D : Guarded.Channel n}
    (h : Guarded.Step C D) :
    (channelPauliMatrix C).rank = (channelPauliMatrix D).rank :=
  rank_eq_of_steps (Relation.ReflTransGen.single h)

/-- The explicit syntactic input has exactly the weighted trajectory rank. -/
theorem rank_sampledPauliChannel (n k p M : ℕ) (J : PMatrix n) (δ : ℝ)
    (L : Fin k → PMatrix n) (samples : Fin M → Fin k → ℝ)
    (weights : Fin M → ℝ) :
    (channelPauliMatrix (sampledPauliChannel n k p M J δ L samples weights)).rank =
      (sampledPauliCoefficientMatrix n k p M J δ L samples weights).rank := by
  apply rank_pauliCoefficientMatrix_eq_of_cp_eq
  rw [← Guarded.denote_eq_of_kraus, denote_sampledPauliChannel]

/-- Updated Proposition 6.1 directly from the manuscript's guarded ChannelIR
rules, including R-Phase. `D` is an actual reachable program, not an assumed
matrix cleanup or unspecified linear postprocessing. The column condition
identifies its active Pauli support; the bound also holds for any restriction. -/
theorem proposition_6_1_guarded (n k p M N' : ℕ) (J : PMatrix n) (δ : ℝ)
    (L : Fin k → PMatrix n) (samples : Fin M → Fin k → ℝ)
    (weights : Fin M → ℝ) (D : Guarded.Channel n)
    (hsteps : Guarded.Steps (sampledPauliChannel n k p M J δ L samples weights) D)
    (columns : Fin N' → PauliLabel n)
    (_hcolumns : Function.Injective columns ∧
      ∀ q, (channelPauliMatrix D).col q ≠ 0 ↔ ∃ j, columns j = q) :
    ((channelPauliMatrix D).submatrix id columns).rank ≤
      min D.length (min N' ((((k + 1) * p) + k).choose k)) := by
  apply le_min
  · simpa using Matrix.rank_le_height ((channelPauliMatrix D).submatrix id columns)
  · apply le_min
    · simpa using Matrix.rank_le_width ((channelPauliMatrix D).submatrix id columns)
    · refine (Matrix.rank_submatrix_le (channelPauliMatrix D) id columns).trans ?_
      rw [← rank_eq_of_steps hsteps, rank_sampledPauliChannel]
      exact rank_trajectoryCoefficientMatrix k p J (δ : ℂ) L
        (fun a i => (samples a i : ℂ)) (fun a => (Real.sqrt (weights a) : ℂ))
        (pauliCoordinates n)

/-- Exhausting the new deterministic rules reaches a normal form, and every
active-support coefficient table of that normal form satisfies Proposition 6.1. -/
theorem exists_sampled_normalForm_rank (n k p M : ℕ) (J : PMatrix n) (δ : ℝ)
    (L : Fin k → PMatrix n) (samples : Fin M → Fin k → ℝ)
    (weights : Fin M → ℝ) :
    ∃ D : Guarded.Channel n,
      Guarded.Steps (sampledPauliChannel n k p M J δ L samples weights) D ∧
      Guarded.NormalForm D ∧
      ∀ (N' : ℕ) (columns : Fin N' → PauliLabel n),
        (Function.Injective columns ∧
          ∀ q, (channelPauliMatrix D).col q ≠ 0 ↔ ∃ j, columns j = q) →
        ((channelPauliMatrix D).submatrix id columns).rank ≤
          min D.length (min N' ((((k + 1) * p) + k).choose k)) := by
  obtain ⟨D, hsteps, hnormal⟩ := Guarded.exists_normalForm
    (sampledPauliChannel n k p M J δ L samples weights)
  exact ⟨D, hsteps, hnormal, fun N' columns hcolumns =>
    proposition_6_1_guarded n k p M N' J δ L samples weights D hsteps columns hcolumns⟩

end FrontendCleanup
end ChannelFormalization
