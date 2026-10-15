import ChannelFormalization.KrausEquivalence
import Mathlib.Analysis.InnerProductSpace.GramSchmidtOrtho
import Mathlib.LinearAlgebra.LinearIndependent.Lemmas
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.Algebra.BigOperators.Intervals

/-! The unitary triangular compression and support bound of Proposition 4.6. -/
noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 2000000
set_option synthInstance.maxHeartbeats 200000
open scoped BigOperators Matrix
open Matrix Submodule Module InnerProductSpace

namespace ChannelFormalization

def euclideanColumn {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℂ) (j : Fin n) :
    EuclideanSpace ℂ (Fin m) := WithLp.toLp 2 (A.col j)

theorem finrank_span_euclideanColumn {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℂ) :
    finrank ℂ (span ℂ (Set.range (euclideanColumn A))) = A.rank := by
  let e := (WithLp.linearEquiv 2 ℂ (Fin m → ℂ)).symm
  have hs : span ℂ (Set.range (euclideanColumn A)) =
      (span ℂ (Set.range A.col)).map e.toLinearMap := by
    rw [Submodule.map_span, ← Set.range_comp]
    rfl
  rw [hs, e.finrank_map_eq, ← Matrix.rank_eq_finrank_span_cols]

/-- Select an independent set of actual columns that spans the whole column space. -/
theorem exists_pivot_columns {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℂ) :
    ∃ p : Fin A.rank → Fin n, Function.Injective p ∧
      LinearIndependent ℂ (euclideanColumn A ∘ p) ∧
      span ℂ (Set.range (euclideanColumn A ∘ p)) =
        span ℂ (Set.range (euclideanColumn A)) := by
  obtain ⟨κ, a, ha, hs, hi⟩ := exists_linearIndependent' ℂ (euclideanColumn A)
  let : Fintype κ := Fintype.ofInjective a ha
  have hc : Fintype.card κ = A.rank := by
    rw [← finrank_span_eq_card hi, hs, finrank_span_euclideanColumn]
  let e : Fin A.rank ≃ κ := (Fintype.equivFinOfCardEq hc).symm
  refine ⟨a ∘ e, ha.comp e.injective, ?_, ?_⟩
  · exact hi.comp e e.injective
  · simpa only [Function.comp_assoc, Set.range_comp, e.surjective.range_eq,
      Set.image_univ] using hs

private theorem exists_triangular_orthonormalBasis
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] [FiniteDimensional ℂ E]
    {r : ℕ} (hd : finrank ℂ E = r) (v : Fin r → E) :
    ∃ q : OrthonormalBasis (Fin r) ℂ E,
      ∀ i j : Fin r, j < i → inner ℂ (q i) (v j) = 0 := by
  have hc : finrank ℂ E = Fintype.card (Fin r) := by simpa using hd
  exact ⟨gramSchmidtOrthonormalBasis hc v, fun i j hij =>
    gramSchmidtOrthonormalBasis_inv_triangular hc v hij⟩

/-- Orthonormal coordinates for the column space retain an upper-triangular pivot block. -/
theorem exists_columnSpace_basis {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℂ) :
    ∃ (p : Fin A.rank → Fin n)
      (q : OrthonormalBasis (Fin A.rank) ℂ (span ℂ (Set.range (euclideanColumn A)))),
      Function.Injective p ∧
      ∀ i j : Fin A.rank, j < i →
        inner ℂ (q i : EuclideanSpace ℂ (Fin m)) (euclideanColumn A (p j)) = 0 := by
  obtain ⟨p, hp, hi, hs⟩ := exists_pivot_columns A
  let S := span ℂ (Set.range (euclideanColumn A))
  let : InnerProductSpace ℂ S := Submodule.innerProductSpace S
  let v : Fin A.rank → S := fun j =>
    ⟨euclideanColumn A (p j), subset_span ⟨p j, rfl⟩⟩
  have hd : finrank ℂ S = A.rank := by
    simp [S, finrank_span_euclideanColumn]
  obtain ⟨q, hq⟩ := exists_triangular_orthonormalBasis (E := S) (r := A.rank) hd v
  refine ⟨p, q, hp, ?_⟩
  intro i j hji
  exact hq i j hji

/-- Extend an orthonormal basis of a subspace, keeping its indices as the first coordinates. -/
theorem exists_extend_orthonormalBasis {m r : ℕ}
    (S : Submodule ℂ (EuclideanSpace ℂ (Fin m)))
    (q : OrthonormalBasis (Fin r) ℂ S) (hrm : r ≤ m) :
    ∃ Q : OrthonormalBasis (Fin m) ℂ (EuclideanSpace ℂ (Fin m)),
      ∀ j : Fin r, Q (j.castLE hrm) = (q j : EuclideanSpace ℂ (Fin m)) := by
  let s : Set (Fin m) := {i | i.val < r}
  let f : Fin m → EuclideanSpace ℂ (Fin m) :=
    fun i => if h : i.val < r then (q ⟨i.val, h⟩ : EuclideanSpace ℂ (Fin m)) else 0
  let g : s → Fin r := fun i => ⟨i.val.val, i.property⟩
  have hg : Function.Injective g := by
    intro i j h
    apply Subtype.ext
    apply Fin.ext
    exact congrArg (fun k : Fin r => k.val) h
  have ho : Orthonormal ℂ (s.domRestrict f) := by
    convert (q.orthonormal.comp_linearIsometry S.subtypeₗᵢ).comp g hg using 1
    funext i
    have hi : i.val.val < r := i.property
    simp [f, g, Set.domRestrict, hi]
  obtain ⟨Q, hQ⟩ := ho.exists_orthonormalBasis_extension_of_card_eq (by simp)
  refine ⟨Q, ?_⟩
  intro j
  simpa [f, j.isLt] using hQ (j.castLE hrm) j.isLt

/-- The unitary taking standard coordinates to the coordinates of an orthonormal basis. -/
def basisRowUnitary {m : ℕ}
    (Q : OrthonormalBasis (Fin m) ℂ (EuclideanSpace ℂ (Fin m))) :
    Matrix.unitaryGroup (Fin m) ℂ :=
  ⟨Q.toBasis.toMatrix (EuclideanSpace.basisFun (Fin m) ℂ),
    Q.toMatrix_orthonormalBasis_mem_unitary (EuclideanSpace.basisFun (Fin m) ℂ)⟩

theorem basisRowUnitary_mul_apply {m n : ℕ}
    (Q : OrthonormalBasis (Fin m) ℂ (EuclideanSpace ℂ (Fin m)))
    (A : Matrix (Fin m) (Fin n) ℂ) (i : Fin m) (j : Fin n) :
    ((basisRowUnitary Q).val * A) i j = inner ℂ (Q i) (euclideanColumn A j) := by
  simp [basisRowUnitary, Matrix.mul_apply, Module.Basis.toMatrix_apply,
    OrthonormalBasis.repr_apply_apply, euclideanColumn, EuclideanSpace.inner_eq_star_dotProduct,
    dotProduct, EuclideanSpace.basisFun_apply, Matrix.col, mul_comm]

/-- An actual unitary row transformation simultaneously isolates rank-many nonzero rows
and creates an upper-triangular block on distinct pivot columns. -/
theorem exists_unitary_triangular_compression {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℂ) :
    ∃ (U : Matrix.unitaryGroup (Fin m) ℂ) (p : Fin A.rank → Fin n),
      Function.Injective p ∧
      (∀ i : Fin m, A.rank ≤ i.val → ∀ j, (U.val * A) i j = 0) ∧
      (∀ i : Fin A.rank, ∃ j, (U.val * A) (i.castLE A.rank_le_height) j ≠ 0) ∧
      (∀ i j : Fin A.rank, j < i →
        (U.val * A) (i.castLE A.rank_le_height) (p j) = 0) := by
  obtain ⟨p, q, hp, htri⟩ := exists_columnSpace_basis A
  let S := span ℂ (Set.range (euclideanColumn A))
  obtain ⟨Q, hQ⟩ := exists_extend_orthonormalBasis S q A.rank_le_height
  refine ⟨basisRowUnitary Q, p, hp, ?_, ?_, ?_⟩
  · intro i hi j
    rw [basisRowUnitary_mul_apply]
    have horth (k : Fin A.rank) :
        inner ℂ (Q i) (q k : EuclideanSpace ℂ (Fin m)) = 0 := by
      rw [← hQ k]
      apply Q.orthonormal.2
      intro hik
      have hv := congrArg Fin.val hik
      simp only [Fin.val_castLE] at hv
      omega
    let x : S := ⟨euclideanColumn A j, subset_span ⟨j, rfl⟩⟩
    change inner ℂ (Q i) (x : EuclideanSpace ℂ (Fin m)) = 0
    rw [← q.sum_repr x]
    simp only [Submodule.coe_sum, Submodule.coe_smul, inner_sum, inner_smul_right,
      horth, mul_zero, Finset.sum_const_zero]
  · intro i
    by_contra hz
    push Not at hz
    have hall : ∀ x ∈ S, inner ℂ (q i : EuclideanSpace ℂ (Fin m)) x = 0 := by
      intro x hx
      induction hx using Submodule.span_induction with
      | mem x hx =>
        obtain ⟨j, rfl⟩ := hx
        simpa only [basisRowUnitary_mul_apply, hQ] using hz j
      | zero => simp
      | add x y hx hy hxi hyi => simp only [inner_add_right, hxi, hyi, add_zero]
      | smul c x hx hxi => simp only [inner_smul_right, hxi, mul_zero]
    have hzero : (q i : EuclideanSpace ℂ (Fin m)) = 0 :=
      inner_self_eq_zero.mp (hall _ (q i).property)
    have hzero' : q i = 0 := Subtype.ext hzero
    have hone := q.orthonormal.1 i
    simp [hzero'] at hone
  · intro i j hji
    rw [basisRowUnitary_mul_apply, hQ]
    exact htri i j hji

/-- Number of nonzero coefficients, summing the active Pauli terms in each Kraus row. -/
def pauliCount {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℂ) : ℕ := by
  classical
  exact ∑ i, (Finset.univ.filter fun j => A i j ≠ 0).card

/-- Count the zeros forced by a triangular block on distinct columns. -/
theorem pauliCount_le_of_triangular {r n : ℕ}
    (B : Matrix (Fin r) (Fin n) ℂ) (p : Fin r → Fin n)
    (hp : Function.Injective p)
    (htri : ∀ i j : Fin r, j < i → B i (p j) = 0) :
    pauliCount B ≤ r * n - r * (r - 1) / 2 := by
  classical
  have hrn : r ≤ n := by
    simpa using Fintype.card_le_of_injective p hp
  have hrow (i : Fin r) :
      (Finset.univ.filter fun j => B i j ≠ 0).card ≤ n - i.val := by
    have hs : (Finset.univ.filter fun j => B i j ≠ 0) ⊆
        Finset.univ \ (Finset.Iio i).image p := by
      intro j hj
      refine Finset.mem_sdiff.mpr ⟨Finset.mem_univ _, ?_⟩
      intro hz
      obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp hz
      exact (Finset.mem_filter.mp hj).2 (htri i k (Finset.mem_Iio.mp hk))
    calc
      (Finset.univ.filter fun j => B i j ≠ 0).card ≤
          (Finset.univ \ (Finset.Iio i).image p).card := Finset.card_le_card hs
      _ = n - i.val := by
        rw [Finset.card_sdiff_of_subset (Finset.subset_univ _),
          Finset.card_image_of_injective _ hp]
        simp
  calc
    pauliCount B ≤ ∑ i : Fin r, (n - i.val) := Finset.sum_le_sum (fun i _ => hrow i)
    _ = r * n - r * (r - 1) / 2 := by
      rw [Finset.sum_tsub_distrib _ (fun i _ => le_trans i.isLt.le hrn)]
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
      congr 1
      rw [Finset.sum_fin_eq_sum_range]
      trans ∑ i ∈ Finset.range r, i
      · apply Finset.sum_congr rfl
        intro i hi
        exact ite_eq_left (Finset.mem_range.mp hi)
      · exact Finset.sum_range_id r

/-- Proposition 4.6's complete matrix construction: rank-many nonzero rows, unitary
equivalence after zero padding, and the Pauli-count bound. -/
theorem exists_pauliCount_reduction {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℂ) :
    ∃ (U : Matrix.unitaryGroup (Fin m) ℂ) (B : Matrix (Fin A.rank) (Fin n) ℂ),
      (∀ i : Fin A.rank, ∃ j, B i j ≠ 0) ∧
      (∀ i : Fin m, ∀ j : Fin n,
        (U.val * A) i j = if h : i.val < A.rank then B ⟨i.val, h⟩ j else 0) ∧
      pauliCount B ≤ A.rank * n - A.rank * (A.rank - 1) / 2 := by
  obtain ⟨U, p, hp, hzero, hnonzero, htri⟩ := exists_unitary_triangular_compression A
  let B : Matrix (Fin A.rank) (Fin n) ℂ :=
    fun i j => (U.val * A) (i.castLE A.rank_le_height) j
  refine ⟨U, B, hnonzero, ?_, pauliCount_le_of_triangular B p hp htri⟩
  intro i j
  by_cases hi : i.val < A.rank
  · simp only [dite_eq_left hi, B]
    congr 2
  · simp only [dite_eq_right hi]
    exact hzero i (Nat.le_of_not_gt hi) j

/-- The paper's strict-reduction consequence follows from its explicit sufficient condition. -/
theorem exists_strict_pauliCount_reduction {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℂ)
    (h : A.rank * n - A.rank * (A.rank - 1) / 2 < pauliCount A) :
    ∃ (U : Matrix.unitaryGroup (Fin m) ℂ) (B : Matrix (Fin A.rank) (Fin n) ℂ),
      (∀ i : Fin A.rank, ∃ j, B i j ≠ 0) ∧
      (∀ i : Fin m, ∀ j : Fin n,
        (U.val * A) i j = if h : i.val < A.rank then B ⟨i.val, h⟩ j else 0) ∧
      pauliCount B < pauliCount A := by
  obtain ⟨U, B, hn, he, hc⟩ := exists_pauliCount_reduction A
  exact ⟨U, B, hn, he, hc.trans_lt h⟩

/-- Removing zero rows does not change the dimension of the row space. -/
theorem rank_eq_of_zero_padding {r m n : ℕ} (hrm : r ≤ m)
    (A : Matrix (Fin m) (Fin n) ℂ) (B : Matrix (Fin r) (Fin n) ℂ)
    (h : ∀ i j, A i j = if hi : i.val < r then B ⟨i.val, hi⟩ j else 0) :
    A.rank = B.rank := by
  have hs : span ℂ (Set.range A.row) = span ℂ (Set.range B.row) := by
    apply le_antisymm
    · apply span_le.mpr
      rintro _ ⟨i, rfl⟩
      by_cases hi : i.val < r
      · have he : A.row i = B.row ⟨i.val, hi⟩ := by
          funext j
          change A i j = B ⟨i.val, hi⟩ j
          simpa only [dite_eq_left hi] using h i j
        rw [he]
        exact subset_span ⟨⟨i.val, hi⟩, rfl⟩
      · have he : A.row i = 0 := by
          funext j
          change A i j = 0
          simpa only [dite_eq_right hi] using h i j
        rw [he]
        exact zero_mem _
    · apply span_le.mpr
      rintro _ ⟨i, rfl⟩
      have he : B.row i = A.row (i.castLE hrm) := by
        funext j
        change B i j = A (i.castLE hrm) j
        simpa only [Fin.val_castLE, dite_eq_left i.isLt] using (h (i.castLE hrm) j).symm
      rw [he]
      exact subset_span ⟨i.castLE hrm, rfl⟩
  rw [Matrix.rank_eq_finrank_span_row, Matrix.rank_eq_finrank_span_row, hs]

/-- The compressed matrix represents exactly the same CP map in any common operator family. -/
theorem krausMap_eq_of_unitary_zero_padding {m r n : ℕ} {a b : Type*}
    [Fintype a] [Fintype b] [DecidableEq a]
    (P : Fin n → Matrix b a ℂ) (hrm : r ≤ m)
    (A : Matrix (Fin m) (Fin n) ℂ) (B : Matrix (Fin r) (Fin n) ℂ)
    (U : Matrix.unitaryGroup (Fin m) ℂ)
    (h : ∀ i j, (U.val * A) i j = if hi : i.val < r then B ⟨i.val, hi⟩ j else 0) :
    MatrixMap.of_kraus (krausOfCoefficients P B) (krausOfCoefficients P B) =
      MatrixMap.of_kraus (krausOfCoefficients P A) (krausOfCoefficients P A) := by
  have he : krausOfCoefficients P (U.val * A) = padKraus (krausOfCoefficients P B) := by
    funext i
    by_cases hi : i.val < r
    · simp only [padKraus, dite_eq_left hi, krausOfCoefficients]
      apply Finset.sum_congr rfl
      intro j _
      rw [h, dite_eq_left hi]
    · simp only [padKraus, dite_eq_right hi, krausOfCoefficients]
      apply Finset.sum_eq_zero
      intro j _
      rw [h, dite_eq_right hi, zero_smul]
  calc
    MatrixMap.of_kraus (krausOfCoefficients P B) (krausOfCoefficients P B) =
        MatrixMap.of_kraus (padKraus (m := m) (krausOfCoefficients P B))
          (padKraus (m := m) (krausOfCoefficients P B)) := (krausMap_padKraus hrm _).symm
    _ = MatrixMap.of_kraus (krausMix U.val (krausOfCoefficients P A))
          (krausMix U.val (krausOfCoefficients P A)) := by
      rw [← he, ← krausMix_ofCoefficients]
    _ = MatrixMap.of_kraus (krausOfCoefficients P A) (krausOfCoefficients P A) :=
      krausMap_unitary_mix _ U

/-- Full algebraic statement of Proposition 4.6, including preservation of the CP map.
For a Pauli basis, `pauliCount` is exactly the paper's total Pauli coefficient count. -/
theorem proposition_4_6 {m n : ℕ} {a b : Type*}
    [Fintype a] [Fintype b] [DecidableEq a]
    (P : Fin n → Matrix b a ℂ) (A : Matrix (Fin m) (Fin n) ℂ) :
    ∃ (U : Matrix.unitaryGroup (Fin m) ℂ) (B : Matrix (Fin A.rank) (Fin n) ℂ),
      B.rank = A.rank ∧
      (∀ i : Fin A.rank, ∃ j, B i j ≠ 0) ∧
      (∀ i : Fin m, ∀ j : Fin n,
        (U.val * A) i j = if h : i.val < A.rank then B ⟨i.val, h⟩ j else 0) ∧
      MatrixMap.of_kraus (krausOfCoefficients P B) (krausOfCoefficients P B) =
        MatrixMap.of_kraus (krausOfCoefficients P A) (krausOfCoefficients P A) ∧
      pauliCount B ≤ A.rank * n - A.rank * (A.rank - 1) / 2 ∧
      (A.rank * n - A.rank * (A.rank - 1) / 2 < pauliCount A →
        pauliCount B < pauliCount A) := by
  obtain ⟨U, B, hn, he, hc⟩ := exists_pauliCount_reduction A
  have hr : B.rank = A.rank := by
    rw [← rank_eq_of_zero_padding A.rank_le_height (U.val * A) B he]
    exact Matrix.rank_mul_eq_right_of_isUnit_det U.val A
      (Matrix.UnitaryGroup.det_isUnit U)
  exact ⟨U, B, hr, hn, he,
    krausMap_eq_of_unitary_zero_padding P A.rank_le_height A B U he,
    hc, fun h => hc.trans_lt h⟩

#print axioms exists_unitary_triangular_compression
#print axioms proposition_4_6

end ChannelFormalization
