import QuantumInfo.Channels.Unbundled
import Mathlib.LinearAlgebra.Isomorphisms
import Mathlib.Analysis.Normed.Module.FiniteDimension

/-! Kraus unitary freedom, proved from equality of the completely positive maps. -/
noncomputable section
set_option backward.isDefEq.respectTransparency false
open scoped BigOperators Matrix
open Matrix

namespace ChannelFormalization

/-- Equal output norms induce an isometry between the images of two linear maps. -/
def rangeIsometryOfNormEq
    {E V : Type*} [AddCommGroup E] [Module ℂ E]
    [NormedAddCommGroup V] [NormedSpace ℂ V]
    (f g : E →ₗ[ℂ] V) (h : ∀ x, ‖f x‖ = ‖g x‖) :
    LinearMap.range f →ₗᵢ[ℂ] V := by
  have hk : LinearMap.ker f ≤ LinearMap.ker g := by
    intro x hx
    apply LinearMap.mem_ker.mpr
    apply norm_eq_zero.mp
    rw [← h x, LinearMap.mem_ker.mp hx, norm_zero]
  let l := ((LinearMap.ker f).liftQ g hk).comp f.quotKerEquivRange.symm.toLinearMap
  refine ⟨l, ?_⟩
  rintro ⟨_, x, rfl⟩
  change ‖(f.ker.liftQ g hk) (f.quotKerEquivRange.symm ⟨f x, _⟩)‖ = ‖f x‖
  rw [LinearMap.quotKerEquivRange_symm_apply_image]
  exact (h x).symm

@[simp] theorem rangeIsometryOfNormEq_apply
    {E V : Type*} [AddCommGroup E] [Module ℂ E]
    [NormedAddCommGroup V] [NormedSpace ℂ V]
    (f g : E →ₗ[ℂ] V) (h : ∀ x, ‖f x‖ = ‖g x‖) (x : E) :
    rangeIsometryOfNormEq f g h ⟨f x, LinearMap.mem_range_self f x⟩ = g x := by
  simp only [rangeIsometryOfNormEq, LinearIsometry.coe_mk,
    LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe,
    LinearMap.quotKerEquivRange_symm_apply_image]
  rfl

/-- An equality of output norms is implemented by a unitary linear equivalence. -/
theorem exists_isometryEquiv_comp_eq_of_norm_eq
    {E V : Type*} [AddCommGroup E] [Module ℂ E]
    [NormedAddCommGroup V] [InnerProductSpace ℂ V] [FiniteDimensional ℂ V]
    (f g : E →ₗ[ℂ] V) (h : ∀ x, ‖f x‖ = ‖g x‖) :
    ∃ u : V ≃ₗᵢ[ℂ] V, u.toLinearMap.comp f = g := by
  let l := rangeIsometryOfNormEq f g h
  refine ⟨l.extend.toLinearIsometryEquiv rfl, ?_⟩
  ext x
  change l.extend (f x) = g x
  simpa [l] using l.extend_apply ⟨f x, LinearMap.mem_range_self f x⟩

private theorem inner_mulVec_eq_gram
    {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]
    (A : Matrix m n ℂ) (x y : EuclideanSpace ℂ n) :
    inner ℂ (A.toEuclideanLin x) (A.toEuclideanLin y) =
      star x.ofLp ⬝ᵥ ((Aᴴ * A).mulVec y.ofLp) := by
  simp only [Matrix.toEuclideanLin, Matrix.toLpLin_apply,
    EuclideanSpace.inner_eq_star_dotProduct]
  rw [dotProduct_comm, Matrix.star_mulVec, Matrix.dotProduct_mulVec,
    Matrix.vecMul_vecMul, Matrix.dotProduct_mulVec]

/-- Rectangular matrices with equal Gram matrices differ by a left unitary matrix. -/
theorem exists_unitary_mul_of_gram_eq
    {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
    (A B : Matrix m n ℂ) (h : Aᴴ * A = Bᴴ * B) :
    ∃ U : Matrix.unitaryGroup m ℂ, B = U.val * A := by
  have hn (x : EuclideanSpace ℂ n) : ‖A.toEuclideanLin x‖ = ‖B.toEuclideanLin x‖ := by
    have hi : inner ℂ (A.toEuclideanLin x) (A.toEuclideanLin x) =
        inner ℂ (B.toEuclideanLin x) (B.toEuclideanLin x) := by
      rw [inner_mulVec_eq_gram, inner_mulVec_eq_gram, h]
    have hs : ‖A.toEuclideanLin x‖ ^ 2 = ‖B.toEuclideanLin x‖ ^ 2 := by
      simp only [inner_self_eq_norm_sq_to_K] at hi
      exact_mod_cast hi
    exact (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp hs
  obtain ⟨u, hu⟩ := exists_isometryEquiv_comp_eq_of_norm_eq
    A.toEuclideanLin B.toEuclideanLin hn
  let b := EuclideanSpace.basisFun m ℂ
  let c := EuclideanSpace.basisFun n ℂ
  let U : Matrix.unitaryGroup m ℂ :=
    ⟨u.toMatrix b.toBasis b.toBasis, u.toMatrix_mem_unitaryGroup b b⟩
  refine ⟨U, ?_⟩
  have hm := congrArg (LinearMap.toMatrix c.toBasis b.toBasis) hu
  rw [LinearMap.toMatrix_comp c.toBasis b.toBasis b.toBasis] at hm
  simpa [U, b, c, Matrix.toEuclideanLin_eq_toLin_orthonormal] using hm.symm

/-- Rows are Kraus operators, with the operator's two indices combined into one column. -/
def flattenKraus {ι a b : Type*} (K : ι → Matrix b a ℂ) : Matrix ι (b × a) ℂ :=
  fun i p => K i p.1 p.2

theorem flattenKraus_gram_eq_choi_transpose
    {ι a b : Type*} [Fintype ι] [Fintype a] [Fintype b] [DecidableEq a]
    (K : ι → Matrix b a ℂ) :
    (flattenKraus K)ᴴ * flattenKraus K = (MatrixMap.of_kraus K K).choi_matrixᵀ := by
  rw [MatrixMap.choi_of_kraus]
  ext ⟨p, q⟩ ⟨r, s⟩
  simp [flattenKraus, Matrix.mul_apply, Matrix.sum_apply, Matrix.vecMulVec, mul_comm]

theorem krausMap_eq_iff_gram_eq
    {ι a b : Type*} [Fintype ι] [Fintype a] [Fintype b] [DecidableEq a]
    (K L : ι → Matrix b a ℂ) :
    MatrixMap.of_kraus K K = MatrixMap.of_kraus L L ↔
      (flattenKraus K)ᴴ * flattenKraus K = (flattenKraus L)ᴴ * flattenKraus L := by
  rw [flattenKraus_gram_eq_choi_transpose, flattenKraus_gram_eq_choi_transpose]
  constructor
  · intro h; rw [h]
  · intro h
    apply MatrixMap.choi_matrix_inj
    exact Matrix.transpose_injective h

/-- The matrix entries of a unitary act on the Kraus index, leaving operator indices unchanged. -/
def krausMix {ι a b : Type*} [Fintype ι]
    (U : Matrix ι ι ℂ) (K : ι → Matrix b a ℂ) : ι → Matrix b a ℂ :=
  fun i => ∑ j, U i j • K j

theorem flattenKraus_mix {ι a b : Type*} [Fintype ι]
    (U : Matrix ι ι ℂ) (K : ι → Matrix b a ℂ) :
    flattenKraus (krausMix U K) = U * flattenKraus K := by
  ext i ⟨p, q⟩
  simp [flattenKraus, krausMix, Matrix.mul_apply, Matrix.sum_apply]

/-- A unitary change of Kraus operators preserves the actual completely positive map. -/
theorem krausMap_unitary_mix
    {ι a b : Type*} [Fintype ι] [Fintype a] [Fintype b]
    [DecidableEq ι] [DecidableEq a]
    (K : ι → Matrix b a ℂ) (U : Matrix.unitaryGroup ι ℂ) :
    MatrixMap.of_kraus (krausMix U.val K) (krausMix U.val K) =
      MatrixMap.of_kraus K K := by
  apply (krausMap_eq_iff_gram_eq _ _).mpr
  rw [flattenKraus_mix, Matrix.conjTranspose_mul]
  have hu : U.valᴴ * U.val = 1 := (Matrix.mem_unitaryGroup_iff').mp U.prop
  calc
    (flattenKraus K)ᴴ * U.valᴴ * (U.val * flattenKraus K) =
        (flattenKraus K)ᴴ * (U.valᴴ * U.val) * flattenKraus K := by
      simp only [Matrix.mul_assoc]
    _ = (flattenKraus K)ᴴ * flattenKraus K := by rw [hu, Matrix.mul_one]

/-- Equal-length Kraus representations of the same map are related by a unitary mixing. -/
theorem kraus_unitary_freedom
    {ι a b : Type*} [Fintype ι] [Fintype a] [Fintype b]
    [DecidableEq ι] [DecidableEq a] [DecidableEq b]
    (K L : ι → Matrix b a ℂ)
    (h : MatrixMap.of_kraus K K = MatrixMap.of_kraus L L) :
    ∃ U : Matrix.unitaryGroup ι ℂ, L = krausMix U.val K := by
  obtain ⟨U, hU⟩ := exists_unitary_mul_of_gram_eq
    (flattenKraus K) (flattenKraus L) ((krausMap_eq_iff_gram_eq K L).mp h)
  refine ⟨U, ?_⟩
  rw [← flattenKraus_mix] at hU
  funext i p q
  exact congrFun₂ hU i (p, q)

/-- Pad a shorter Kraus list with exact zero operators. -/
def padKraus {r m : ℕ} {a b : Type*} (L : Fin r → Matrix b a ℂ) :
    Fin m → Matrix b a ℂ :=
  fun i => if h : i.val < r then L ⟨i.val, h⟩ else 0

@[simp] theorem padKraus_castLE {r m : ℕ} {a b : Type*}
    (h : r ≤ m) (L : Fin r → Matrix b a ℂ) (i : Fin r) :
    padKraus (m := m) L (i.castLE h) = L i := by
  simp [padKraus, i.isLt]

theorem krausMap_padKraus {r m : ℕ} {a b : Type*}
    [Fintype a] [DecidableEq a]
    (h : r ≤ m) (L : Fin r → Matrix b a ℂ) :
    MatrixMap.of_kraus (padKraus (m := m) L) (padKraus (m := m) L) =
      MatrixMap.of_kraus L L := by
  ext ρ p q
  simp only [MatrixMap.of_kraus, LinearMap.sum_apply, LinearMap.coe_mk,
    AddHom.coe_mk, Matrix.sum_apply]
  let e : Fin r → Fin m := Fin.castLE h
  let s := Finset.univ.image e
  have he : Function.Injective e := Fin.castLE_injective h
  calc
    (∑ i : Fin m, (padKraus L i * ρ * (padKraus L i)ᴴ) p q) =
        ∑ i ∈ s, (padKraus L i * ρ * (padKraus L i)ᴴ) p q := by
      symm
      apply Finset.sum_subset (Finset.subset_univ s)
      intro i _ hi
      have hir : ¬ i.val < r := by
        intro hir
        apply hi
        exact Finset.mem_image.mpr ⟨⟨i.val, hir⟩, Finset.mem_univ _, by ext; rfl⟩
      simp [padKraus, hir]
    _ = ∑ i : Fin r, (L i * ρ * (L i)ᴴ) p q := by
      rw [Finset.sum_image (fun i _ j _ hij => he hij)]
      simp [e]

/-- Proposition 4.4's unitary-freedom assertion, including the stated zero padding. -/
theorem kraus_unitary_freedom_padded {r m : ℕ} {a b : Type*}
    [Fintype a] [Fintype b] [DecidableEq a] [DecidableEq b]
    (hrm : r ≤ m) (K : Fin m → Matrix b a ℂ) (L : Fin r → Matrix b a ℂ)
    (h : MatrixMap.of_kraus K K = MatrixMap.of_kraus L L) :
    ∃ U : Matrix.unitaryGroup (Fin m) ℂ, padKraus L = krausMix U.val K := by
  apply kraus_unitary_freedom K (padKraus L)
  rw [krausMap_padKraus hrm L]
  exact h

/-- Interpret coefficients in a common operator basis. -/
def krausOfCoefficients {ι d a b : Type*} [Fintype d]
    (P : d → Matrix b a ℂ) (A : Matrix ι d ℂ) : ι → Matrix b a ℂ :=
  fun i => ∑ p, A i p • P p

theorem krausMix_ofCoefficients {ι d a b : Type*} [Fintype ι] [Fintype d]
    (P : d → Matrix b a ℂ) (A : Matrix ι d ℂ) (U : Matrix ι ι ℂ) :
    krausMix U (krausOfCoefficients P A) = krausOfCoefficients P (U * A) := by
  funext i
  simp only [krausMix, krausOfCoefficients, Finset.smul_sum, smul_smul,
    Matrix.mul_apply, Finset.sum_smul]
  rw [Finset.sum_comm]

theorem krausOfCoefficients_injective {ι d a b : Type*} [Fintype d]
    (P : d → Matrix b a ℂ) (hP : LinearIndependent ℂ P) :
    Function.Injective (krausOfCoefficients (ι := ι) P) := by
  intro A B h
  funext i
  apply (linearIndependent_iff_injective_fintypeLinearCombination.mp hP)
  exact congrFun h i

/-- Common independent operator coordinates are carried by the same row unitary.
This applies in particular to a common Pauli basis with phases absorbed in coefficients. -/
theorem coefficient_unitary_freedom
    {ι d a b : Type*} [Fintype ι] [Fintype d] [Fintype a] [Fintype b]
    [DecidableEq ι] [DecidableEq a] [DecidableEq b]
    (P : d → Matrix b a ℂ) (hP : LinearIndependent ℂ P)
    (A B : Matrix ι d ℂ)
    (h : MatrixMap.of_kraus (krausOfCoefficients P A) (krausOfCoefficients P A) =
      MatrixMap.of_kraus (krausOfCoefficients P B) (krausOfCoefficients P B)) :
    ∃ U : Matrix.unitaryGroup ι ℂ, B = U.val * A := by
  obtain ⟨U, hU⟩ := kraus_unitary_freedom _ _ h
  refine ⟨U, krausOfCoefficients_injective P hP ?_⟩
  rw [hU, krausMix_ofCoefficients]

#print axioms kraus_unitary_freedom_padded
#print axioms coefficient_unitary_freedom

end ChannelFormalization
