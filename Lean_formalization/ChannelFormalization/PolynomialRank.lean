import Mathlib.RingTheory.MvPolynomial.Basic
import Mathlib.Algebra.MvPolynomial.CommRing
import Mathlib.Data.Finsupp.Multiset
import Mathlib.Data.Sym.Card
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.Basic.Complex.Basic
import Mathlib.Data.Matrix.Basis
import Mathlib.Analysis.Real.Sqrt
import Mathlib.LinearAlgebra.Finsupp.LinearCombination
import ChannelFormalization.PauliPlacement
import ChannelFormalization.KrausEquivalence

noncomputable section

open scoped BigOperators ComplexOrder
open MvPolynomial

namespace ChannelFormalization

/-- Exponents of monomials in `k` variables with total degree at most `d`. -/
def MonomialsLE (k d : ℕ) :=
  {e : Fin k →₀ ℕ // e.sum (fun _ n => n) ≤ d}

/-- Decompose bounded-degree monomials by their exact degree. -/
def monomialsLEEquiv (k d : ℕ) :
    MonomialsLE k d ≃ Σ t : Fin (d + 1), Sym (Fin k) t.val := by
  let e : {t : ℕ // t ≤ d} ≃ Fin (d + 1) :=
    { toFun := fun t => ⟨t.val, Nat.lt_succ_iff.mpr t.property⟩
      invFun := fun t => ⟨t.val, t.is_le⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  exact (Equiv.sigmaSubtypeFiberEquivSubtype
    (fun a : Fin k →₀ ℕ => a.sum (fun _ n => n))
    (p := fun a => a.sum (fun _ n => n) ≤ d) (q := fun t => t ≤ d)
    (fun _ => Iff.rfl)).symm.trans
      (Equiv.sigmaCongr e fun t => (Sym.equivNatSum (Fin k) t.val).symm)

instance (k d : ℕ) : Fintype (MonomialsLE k d) :=
  Fintype.ofEquiv (Σ t : Fin (d + 1), Sym (Fin k) t.val)
    (monomialsLEEquiv k d).symm

/-- The exact stars-and-bars count, including zero variables and zero degree. -/
theorem card_monomialsLE (k d : ℕ) :
    Fintype.card (MonomialsLE k d) = (d + k).choose k := by
  rw [Fintype.card_congr (monomialsLEEquiv k d), Fintype.card_sigma]
  simp_rw [Sym.card_sym_eq_multichoose, Fintype.card_fin]
  rw [Fin.sum_univ_eq_sum_range, Nat.sum_range_multichoose]

/-- The bounded-total-degree polynomial space has the exact expected dimension. -/
theorem finrank_restrictTotalDegree (k d : ℕ) :
    Module.finrank ℂ (MvPolynomial.restrictTotalDegree (Fin k) ℂ d) =
      (d + k).choose k := by
  let b : Module.Basis (MonomialsLE k d) ℂ
      (MvPolynomial.restrictTotalDegree (Fin k) ℂ d) :=
    MvPolynomial.basisRestrictSupport ℂ _
  exact (Module.finrank_eq_card_basis b).trans (card_monomialsLE k d)

/-- Evaluation at an arbitrary finite collection of sample points is linear. -/
def boundedPolynomialEvaluation {k d : ℕ} {ι : Type*}
    (samples : ι → Fin k → ℂ) :
    MvPolynomial.restrictTotalDegree (Fin k) ℂ d →ₗ[ℂ] (ι → ℂ) where
  toFun f a := MvPolynomial.eval (samples a) f.val
  map_add' f g := by ext a; simp
  map_smul' c f := by ext a; simp [MvPolynomial.smul_eval]

/-- A sample matrix factors through the space of bounded-degree polynomials. -/
theorem rank_polynomial_samples {k d : ℕ} {ι ν : Type*}
    [Fintype ι] [Fintype ν] (samples : ι → Fin k → ℂ)
    (f : ν → MvPolynomial (Fin k) ℂ)
    (hf : ∀ j, (f j).totalDegree ≤ d) :
    (Matrix.of fun a j => MvPolynomial.eval (samples a) (f j)).rank ≤
      (d + k).choose k := by
  let ev := boundedPolynomialEvaluation (d := d) samples
  let f' (j : ν) : MvPolynomial.restrictTotalDegree (Fin k) ℂ d :=
    ⟨f j, (MvPolynomial.mem_restrictTotalDegree (Fin k) d (f j)).mpr (hf j)⟩
  rw [Matrix.rank_eq_finrank_span_cols]
  calc
    _ ≤ Module.finrank ℂ (LinearMap.range ev) := by
      apply Submodule.finrank_mono
      apply Submodule.span_le.mpr
      rintro _ ⟨j, rfl⟩
      exact ⟨f' j, rfl⟩
    _ ≤ Module.finrank ℂ (MvPolynomial.restrictTotalDegree (Fin k) ℂ d) :=
      ev.finrank_range_le
    _ = (d + k).choose k := finrank_restrictTotalDegree k d

abbrev TimePolynomial (k : ℕ) := MvPolynomial (Fin k) ℂ

/-- Entrywise total-degree bound; multiplication of these matrices need not commute. -/
def PolynomialMatrixDegreeLE {k : ℕ} {η : Type*}
    (F : Matrix η η (TimePolynomial k)) (d : ℕ) : Prop :=
  ∀ i j, (F i j).totalDegree ≤ d

def constantPolynomialMatrix {η : Type*} (k : ℕ) (A : Matrix η η ℂ) :
    Matrix η η (TimePolynomial k) := A.map MvPolynomial.C

theorem degree_constantPolynomialMatrix {η : Type*} (k : ℕ) (A : Matrix η η ℂ) :
    PolynomialMatrixDegreeLE (constantPolynomialMatrix k A) 0 := by
  intro i j
  simp [constantPolynomialMatrix, Matrix.map]

theorem PolynomialMatrixDegreeLE.mono {k : ℕ} {η : Type*}
    {F : Matrix η η (TimePolynomial k)} {a b : ℕ}
    (h : PolynomialMatrixDegreeLE F a) (hab : a ≤ b) :
    PolynomialMatrixDegreeLE F b := fun i j => (h i j).trans hab

theorem PolynomialMatrixDegreeLE.sum {k : ℕ} {η ι : Type*}
    (s : Finset ι) (F : ι → Matrix η η (TimePolynomial k)) {d : ℕ}
    (h : ∀ a ∈ s, PolynomialMatrixDegreeLE (F a) d) :
    PolynomialMatrixDegreeLE (∑ a ∈ s, F a) d := by
  intro i j
  simp only [Matrix.sum_apply]
  exact MvPolynomial.totalDegree_finsetSum_le fun a ha => h a ha i j

theorem PolynomialMatrixDegreeLE.smul {k : ℕ} {η : Type*}
    {F : Matrix η η (TimePolynomial k)} {T : TimePolynomial k} {a b : ℕ}
    (hT : T.totalDegree ≤ a) (hF : PolynomialMatrixDegreeLE F b) :
    PolynomialMatrixDegreeLE (T • F) (a + b) := by
  intro i j
  exact (MvPolynomial.totalDegree_mul T (F i j)).trans (add_le_add hT (hF i j))

theorem PolynomialMatrixDegreeLE.mul {k : ℕ} {η : Type*} [Fintype η]
    {F G : Matrix η η (TimePolynomial k)} {a b : ℕ}
    (hF : PolynomialMatrixDegreeLE F a) (hG : PolynomialMatrixDegreeLE G b) :
    PolynomialMatrixDegreeLE (F * G) (a + b) := by
  intro i j
  rw [Matrix.mul_apply]
  exact MvPolynomial.totalDegree_finsetSum_le fun l _ =>
    (MvPolynomial.totalDegree_mul (F i l) (G l j)).trans (add_le_add (hF i l) (hG l j))

/-- The degree-`p` drift polynomial evaluated at a scalar time. -/
def driftMatrix {η : Type*} [Fintype η] [DecidableEq η]
    (p : ℕ) (J : Matrix η η ℂ) (t : ℂ) : Matrix η η ℂ :=
  ∑ b ∈ Finset.range (p + 1), ((b.factorial : ℂ)⁻¹ * t ^ b) • J ^ b

/-- The actual drift polynomial with a polynomial-valued time interval. -/
def driftPolynomialMatrix {η : Type*} [Fintype η] [DecidableEq η]
    {k : ℕ} (p : ℕ) (J : Matrix η η ℂ) (T : TimePolynomial k) :
    Matrix η η (TimePolynomial k) :=
  ∑ b ∈ Finset.range (p + 1),
    (((b.factorial : ℂ)⁻¹) • T ^ b) • constantPolynomialMatrix k (J ^ b)

theorem degree_driftPolynomialMatrix {η : Type*} [Fintype η] [DecidableEq η]
    {k : ℕ} (p : ℕ) (J : Matrix η η ℂ) (T : TimePolynomial k)
    (hT : T.totalDegree ≤ 1) :
    PolynomialMatrixDegreeLE (driftPolynomialMatrix p J T) p := by
  apply PolynomialMatrixDegreeLE.sum
  intro b hb
  have hpow : (T ^ b).totalDegree ≤ b :=
    (MvPolynomial.totalDegree_pow T b).trans (by simpa using Nat.mul_le_mul_left b hT)
  have hcoeff : ((((b.factorial : ℂ)⁻¹) • T ^ b) : TimePolynomial k).totalDegree ≤ b :=
    (MvPolynomial.totalDegree_smul_le _ _).trans hpow
  exact (PolynomialMatrixDegreeLE.smul hcoeff
    (degree_constantPolynomialMatrix k (J ^ b))).mono (by
      simpa using Nat.le_of_lt_succ (Finset.mem_range.mp hb))

/-- Entrywise evaluation as a ring homomorphism, hence preserving ordered matrix products. -/
def evaluatePolynomialMatrix {η : Type*} [Fintype η] [DecidableEq η]
    {k : ℕ} (s : Fin k → ℂ) :
    Matrix η η (TimePolynomial k) →+* Matrix η η ℂ :=
  (MvPolynomial.eval s).mapMatrix

@[simp] theorem evaluate_constantPolynomialMatrix {η : Type*}
    [Fintype η] [DecidableEq η] {k : ℕ} (s : Fin k → ℂ) (A : Matrix η η ℂ) :
    evaluatePolynomialMatrix s (constantPolynomialMatrix k A) = A := by
  ext i j
  simp [evaluatePolynomialMatrix, constantPolynomialMatrix, Matrix.map]

@[simp] theorem evaluatePolynomialMatrix_smul {η : Type*}
    [Fintype η] [DecidableEq η] {k : ℕ} (s : Fin k → ℂ)
    (T : TimePolynomial k) (F : Matrix η η (TimePolynomial k)) :
    evaluatePolynomialMatrix s (T • F) = MvPolynomial.eval s T • evaluatePolynomialMatrix s F := by
  ext i j
  simp [evaluatePolynomialMatrix, Matrix.map, Matrix.smul_apply]

@[simp] theorem evaluate_driftPolynomialMatrix {η : Type*}
    [Fintype η] [DecidableEq η] {k : ℕ} (p : ℕ) (J : Matrix η η ℂ)
    (T : TimePolynomial k) (s : Fin k → ℂ) :
    evaluatePolynomialMatrix s (driftPolynomialMatrix p J T) =
      driftMatrix p J (MvPolynomial.eval s T) := by
  unfold driftPolynomialMatrix driftMatrix
  rw [map_sum]
  apply Finset.sum_congr rfl
  intro b hb
  rw [evaluatePolynomialMatrix_smul, evaluate_constantPolynomialMatrix]
  simp [MvPolynomial.smul_eval]

/-- Ordered drift/jump product. The last drift and jump multiply on the left. -/
def trajectoryProduct {R A : Type*} [Mul A] (E : R → A) :
    (q : ℕ) → (Fin (q + 1) → R) → (Fin q → A) → A
  | 0, T, _ => E (T 0)
  | q + 1, T, L => E (T (Fin.last (q + 1))) * L (Fin.last q) *
      trajectoryProduct E q (fun i => T i.castSucc) (fun i => L i.castSucc)

/-- Boundary times `0, s₁, ..., sₖ, δ`, also for an empty jump sequence. -/
def timeBoundaries {R : Type*} [Zero R] {k : ℕ} (δ : R) (s : Fin k → R) :
    Fin (k + 2) → R := Fin.cases 0 (Fin.lastCases δ s)

/-- The `k+1` adjacent differences of the boundary times. -/
def timeIntervals {R : Type*} [Zero R] [Sub R] {k : ℕ} (δ : R) (s : Fin k → R)
    (i : Fin (k + 1)) : R :=
  timeBoundaries δ s i.succ - timeBoundaries δ s i.castSucc

theorem degree_timeBoundaries (k : ℕ) (δ : ℂ) (i : Fin (k + 2)) :
    (timeBoundaries (MvPolynomial.C δ) (MvPolynomial.X : Fin k → TimePolynomial k) i).totalDegree ≤ 1 := by
  refine Fin.cases ?_ (fun j => ?_) i
  · simp [timeBoundaries]
  · refine Fin.lastCases ?_ (fun j => ?_) j
    · simp only [timeBoundaries, Fin.cases_succ, Fin.lastCases_last, totalDegree_C, Nat.zero_le]
    · simp [timeBoundaries]

theorem degree_timeIntervals (k : ℕ) (δ : ℂ) (i : Fin (k + 1)) :
    (timeIntervals (MvPolynomial.C δ) (MvPolynomial.X : Fin k → TimePolynomial k) i).totalDegree ≤ 1 := by
  exact (MvPolynomial.totalDegree_sub _ _).trans
    (max_le (degree_timeBoundaries k δ i.succ) (degree_timeBoundaries k δ i.castSucc))

@[simp] theorem eval_timeBoundaries {k : ℕ} (δ : ℂ) (s : Fin k → ℂ)
    (i : Fin (k + 2)) :
    MvPolynomial.eval s
      (timeBoundaries (MvPolynomial.C δ) (MvPolynomial.X : Fin k → TimePolynomial k) i) =
      timeBoundaries δ s i := by
  refine Fin.cases ?_ (fun j => ?_) i
  · simp [timeBoundaries]
  · refine Fin.lastCases ?_ (fun j => ?_) j
    · simp only [timeBoundaries, Fin.cases_succ, Fin.lastCases_last, eval_C]
    · simp [timeBoundaries]

@[simp] theorem eval_timeIntervals {k : ℕ} (δ : ℂ) (s : Fin k → ℂ)
    (i : Fin (k + 1)) :
    MvPolynomial.eval s
      (timeIntervals (MvPolynomial.C δ) (MvPolynomial.X : Fin k → TimePolynomial k) i) =
      timeIntervals δ s i := by
  simp [timeIntervals]

theorem degree_trajectoryProduct {η : Type*} [Fintype η] [DecidableEq η]
    {k : ℕ} (p : ℕ) (J : Matrix η η ℂ) (q : ℕ)
    (T : Fin (q + 1) → TimePolynomial k)
    (L : Fin q → Matrix η η (TimePolynomial k))
    (hT : ∀ i, (T i).totalDegree ≤ 1)
    (hL : ∀ i, PolynomialMatrixDegreeLE (L i) 0) :
    PolynomialMatrixDegreeLE (trajectoryProduct (driftPolynomialMatrix p J) q T L)
      ((q + 1) * p) := by
  induction q with
  | zero => simpa [trajectoryProduct] using degree_driftPolynomialMatrix p J (T 0) (hT 0)
  | succ q ih =>
    have h := ((degree_driftPolynomialMatrix p J (T (Fin.last (q + 1)))
      (hT _)).mul (hL (Fin.last q))).mul
      (ih (fun i => T i.castSucc) (fun i => L i.castSucc)
        (fun i => hT _) (fun i => hL _))
    simpa [trajectoryProduct, Nat.add_mul, two_mul, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using h

/-- Matrix-valued polynomial for the unweighted trajectory in main-text Equation (6) (Appendix B.2 Equation (8)). -/
def trajectoryPolynomial {η : Type*} [Fintype η] [DecidableEq η]
    (k p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin k → Matrix η η ℂ) :
    Matrix η η (TimePolynomial k) :=
  trajectoryProduct (driftPolynomialMatrix p J) k
    (timeIntervals (MvPolynomial.C δ) MvPolynomial.X)
    (fun i => constantPolynomialMatrix k (L i))

/-- Actual ordered matrix product in main-text Equation (6) (Appendix B.2 Equation (8)), before its quadrature weight. -/
def sampledTrajectory {η : Type*} [Fintype η] [DecidableEq η]
    (k p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin k → Matrix η η ℂ)
    (s : Fin k → ℂ) : Matrix η η ℂ :=
  trajectoryProduct (driftMatrix p J) k (timeIntervals δ s) L

@[simp] theorem sampledTrajectory_zero {η : Type*} [Fintype η] [DecidableEq η]
    (p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin 0 → Matrix η η ℂ)
    (s : Fin 0 → ℂ) : sampledTrajectory 0 p J δ L s = driftMatrix p J δ := by
  change driftMatrix p J (δ - 0) = driftMatrix p J δ
  rw [sub_zero]

theorem sampledTrajectory_one {η : Type*} [Fintype η] [DecidableEq η]
    (p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin 1 → Matrix η η ℂ)
    (s : Fin 1 → ℂ) : sampledTrajectory 1 p J δ L s =
      driftMatrix p J (δ - s 0) * L 0 * driftMatrix p J (s 0) := by
  change driftMatrix p J (δ - s 0) * L 0 * driftMatrix p J (s 0 - 0) = _
  rw [sub_zero]

theorem degree_trajectoryPolynomial {η : Type*} [Fintype η] [DecidableEq η]
    (k p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin k → Matrix η η ℂ) :
    PolynomialMatrixDegreeLE (trajectoryPolynomial k p J δ L) ((k + 1) * p) :=
  degree_trajectoryProduct p J k _ _ (degree_timeIntervals k δ)
    (fun i => degree_constantPolynomialMatrix k (L i))

theorem evaluate_trajectoryProduct {η : Type*} [Fintype η] [DecidableEq η]
    {k : ℕ} (p : ℕ) (J : Matrix η η ℂ) (q : ℕ)
    (T : Fin (q + 1) → TimePolynomial k)
    (L : Fin q → Matrix η η (TimePolynomial k)) (s : Fin k → ℂ) :
    evaluatePolynomialMatrix s (trajectoryProduct (driftPolynomialMatrix p J) q T L) =
      trajectoryProduct (driftMatrix p J) q (fun i => MvPolynomial.eval s (T i))
        (fun i => evaluatePolynomialMatrix s (L i)) := by
  induction q with
  | zero => simp [trajectoryProduct]
  | succ q ih => simp only [trajectoryProduct, map_mul, evaluate_driftPolynomialMatrix, ih]

/-- Polynomial evaluation is exactly the noncommutative matrix trajectory, not an assumption. -/
@[simp] theorem evaluate_trajectoryPolynomial {η : Type*} [Fintype η] [DecidableEq η]
    (k p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin k → Matrix η η ℂ)
    (s : Fin k → ℂ) :
    evaluatePolynomialMatrix s (trajectoryPolynomial k p J δ L) =
      sampledTrajectory k p J δ L s := by
  simp [trajectoryPolynomial, sampledTrajectory, evaluate_trajectoryProduct]

/-- Extend any linear operator-coordinate map to polynomial matrix entries. -/
def coordinatePolynomial {η ν : Type*} [Fintype η] [DecidableEq η] {k : ℕ}
    (coord : Matrix η η ℂ →ₗ[ℂ] (ν → ℂ))
    (F : Matrix η η (TimePolynomial k)) (j : ν) : TimePolynomial k :=
  ∑ i : η, ∑ l : η, (coord (Matrix.single i l 1) j) • F i l

theorem degree_coordinatePolynomial {η ν : Type*} [Fintype η] [DecidableEq η]
    {k d : ℕ} (coord : Matrix η η ℂ →ₗ[ℂ] (ν → ℂ))
    (F : Matrix η η (TimePolynomial k)) (hF : PolynomialMatrixDegreeLE F d) (j : ν) :
    (coordinatePolynomial coord F j).totalDegree ≤ d := by
  apply MvPolynomial.totalDegree_finsetSum_le
  intro i hi
  apply MvPolynomial.totalDegree_finsetSum_le
  intro l hl
  exact (MvPolynomial.totalDegree_smul_le _ _).trans (hF i l)

theorem linear_coordinates_eq_sum {η ν : Type*} [Fintype η] [DecidableEq η]
    (coord : Matrix η η ℂ →ₗ[ℂ] (ν → ℂ)) (A : Matrix η η ℂ) (j : ν) :
    coord A j = ∑ i : η, ∑ l : η, A i l * coord (Matrix.single i l 1) j := by
  calc
    coord A j = coord (∑ i : η, ∑ l : η, Matrix.single i l (A i l)) j :=
      congrArg (fun B => coord B j) (Matrix.matrix_eq_sum_single A)
    _ = _ := by
      simp only [map_sum, Finset.sum_apply]
      apply Finset.sum_congr rfl
      intro i hi
      apply Finset.sum_congr rfl
      intro l hl
      have h : Matrix.single i l (A i l) = A i l • Matrix.single i l (1 : ℂ) := by simp
      rw [h, map_smul]
      rfl

@[simp] theorem eval_coordinatePolynomial {η ν : Type*} [Fintype η] [DecidableEq η]
    {k : ℕ} (coord : Matrix η η ℂ →ₗ[ℂ] (ν → ℂ))
    (F : Matrix η η (TimePolynomial k)) (j : ν) (s : Fin k → ℂ) :
    MvPolynomial.eval s (coordinatePolynomial coord F j) =
      coord (evaluatePolynomialMatrix s F) j := by
  rw [linear_coordinates_eq_sum]
  simp [coordinatePolynomial, MvPolynomial.smul_eval, evaluatePolynomialMatrix,
    Matrix.map, mul_comm]

/-- The coefficient matrix of the actual weighted Kraus trajectories from main-text Equation (6)
(Appendix B.2 Equation (8)). -/
def trajectoryCoefficientMatrix {η ι ν : Type*} [Fintype η] [DecidableEq η]
    (k p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin k → Matrix η η ℂ)
    (samples : ι → Fin k → ℂ) (weights : ι → ℂ)
    (coord : Matrix η η ℂ →ₗ[ℂ] (ν → ℂ)) : Matrix ι ν ℂ :=
  Matrix.of fun a j => coord (weights a • sampledTrajectory k p J δ L (samples a)) j

theorem trajectoryCoefficientMatrix_eq_diagonal_mul {η ι ν : Type*}
    [Fintype η] [DecidableEq η] [Fintype ι] [DecidableEq ι]
    (k p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin k → Matrix η η ℂ)
    (samples : ι → Fin k → ℂ) (weights : ι → ℂ)
    (coord : Matrix η η ℂ →ₗ[ℂ] (ν → ℂ)) :
    trajectoryCoefficientMatrix k p J δ L samples weights coord =
      Matrix.diagonal weights * trajectoryCoefficientMatrix k p J δ L samples (fun _ => 1) coord := by
  ext a j
  simp [trajectoryCoefficientMatrix, Matrix.diagonal_mul, map_smul]

/-- The sampled Kraus rank bound, deriving the polynomial structure from the drift/jump product. -/
theorem rank_trajectoryCoefficientMatrix {η ι ν : Type*}
    [Fintype η] [DecidableEq η] [Fintype ι] [Fintype ν]
    (k p : ℕ) (J : Matrix η η ℂ) (δ : ℂ) (L : Fin k → Matrix η η ℂ)
    (samples : ι → Fin k → ℂ) (weights : ι → ℂ)
    (coord : Matrix η η ℂ →ₗ[ℂ] (ν → ℂ)) :
    (trajectoryCoefficientMatrix k p J δ L samples weights coord).rank ≤
      (((k + 1) * p) + k).choose k := by
  classical
  have hpoly := rank_polynomial_samples samples
    (coordinatePolynomial coord (trajectoryPolynomial k p J δ L))
    (degree_coordinatePolynomial coord _ (degree_trajectoryPolynomial k p J δ L))
  have hunweighted :
      (trajectoryCoefficientMatrix k p J δ L samples (fun _ => 1) coord).rank ≤
        (((k + 1) * p) + k).choose k := by
    simpa only [eval_coordinatePolynomial, evaluate_trajectoryPolynomial,
      trajectoryCoefficientMatrix, one_smul] using hpoly
  rw [trajectoryCoefficientMatrix_eq_diagonal_mul]
  exact (Matrix.rank_mul_le_right _ _).trans hunweighted

/-- Row recombination and column restriction include deletion and proportional merging. -/
def coefficientPostprocess {ι ι' ν ν' : Type*} [Fintype ι]
    (R : Matrix ι' ι ℂ) (columns : ν' → ν) (A : Matrix ι ν ℂ) : Matrix ι' ν' ℂ :=
  (R * A).submatrix id columns

theorem rank_coefficientPostprocess_le {ι ι' ν ν' : Type*}
    [Fintype ι] [Fintype ν] [Fintype ν']
    (R : Matrix ι' ι ℂ) (columns : ν' → ν) (A : Matrix ι ν ℂ) :
    (coefficientPostprocess R columns A).rank ≤ A.rank :=
  (Matrix.rank_submatrix_le (R * A) id columns).trans (Matrix.rank_mul_le_right R A)

/-- Deleting any rows or columns cannot increase coefficient rank. -/
theorem rank_delete_rows_columns_le {ι ι' ν ν' : Type*}
    [Fintype ν] [Fintype ν'] (A : Matrix ι ν ℂ)
    (rows : ι' → ι) (columns : ν' → ν) :
    (A.submatrix rows columns).rank ≤ A.rank :=
  Matrix.rank_submatrix_le A rows columns

/-- The exact-zero cleanup used by the paper, with nonzero rows and columns as indices. -/
def removeExactZeros {ι ν : Type*} (A : Matrix ι ν ℂ) :
    Matrix {i // A.row i ≠ 0} {j // A.col j ≠ 0} ℂ :=
  A.submatrix Subtype.val Subtype.val

theorem rank_removeExactZeros_le {ι ν : Type*} [Fintype ν] (A : Matrix ι ν ℂ) :
    (letI : Fintype {j // A.col j ≠ 0} := Fintype.ofFinite _;
      (removeExactZeros A).rank) ≤ A.rank := by
  classical
  let _ : Fintype {j // A.col j ≠ 0} := Fintype.ofFinite _
  exact Matrix.rank_submatrix_le A Subtype.val Subtype.val

/-- Turn row-span membership into an actual row-recombination matrix. -/
theorem exists_row_recombination {ι ι' ν : Type*} [Fintype ι]
    (A : Matrix ι ν ℂ) (B : Matrix ι' ν ℂ)
    (h : ∀ i, B.row i ∈ Submodule.span ℂ (Set.range A.row)) :
    ∃ R : Matrix ι' ι ℂ, B = R * A := by
  classical
  have hc (i : ι') : ∃ c : ι → ℂ, ∑ a, c a • A.row a = B.row i :=
    (Submodule.mem_span_range_iff_exists_fun ℂ).mp (h i)
  refine ⟨Matrix.of (fun i => Classical.choose (hc i)), ?_⟩
  ext i j
  have hi := congrArg (fun v : ν → ℂ => v j) (Classical.choose_spec (hc i))
  simpa [Matrix.mul_apply, Matrix.row, Finset.sum_apply] using hi.symm

/-- Normalization in the proportional-Kraus merge rule. -/
def proportionalMergeFactor {σ : Type*} [Fintype σ] (a : σ → ℂ) : ℂ :=
  (Real.sqrt (∑ i, Complex.normSq (a i)) : ℝ)

/-- A group of proportional rows, together with untouched rows. -/
def proportionalRows {σ τ ν : Type*} (a : σ → ℂ) (v : ν → ℂ)
    (B : Matrix τ ν ℂ) : Matrix (σ ⊕ τ) ν ℂ :=
  Matrix.of (Sum.elim (fun i => a i • v) B.row)

/-- Replace all proportional rows by their norm-preserving aggregate. -/
def mergedProportionalRows {σ τ ν : Type*} [Fintype σ]
    (a : σ → ℂ) (v : ν → ℂ) (B : Matrix τ ν ℂ) : Matrix (Option τ) ν ℂ :=
  Matrix.of (fun i => i.elim (proportionalMergeFactor a • v) B.row)

theorem mergedProportionalRows_mem_span {σ τ ν : Type*} [Fintype σ]
    (a : σ → ℂ) (v : ν → ℂ) (B : Matrix τ ν ℂ) (i : Option τ) :
    (mergedProportionalRows a v B).row i ∈
      Submodule.span ℂ (Set.range (proportionalRows a v B).row) := by
  classical
  cases i with
  | some i => exact Submodule.subset_span ⟨Sum.inr i, rfl⟩
  | none =>
    change proportionalMergeFactor a • v ∈
      Submodule.span ℂ (Set.range (proportionalRows a v B).row)
    by_cases h : ∃ t, a t ≠ 0
    · obtain ⟨t, ht⟩ := h
      have hv : a t • v ∈ Submodule.span ℂ (Set.range (proportionalRows a v B).row) :=
        Submodule.subset_span ⟨Sum.inl t, rfl⟩
      have hscaled := Submodule.smul_mem
        (Submodule.span ℂ (Set.range (proportionalRows a v B).row))
        (proportionalMergeFactor a / a t) hv
      simpa [smul_smul, div_mul_cancel₀ _ ht] using hscaled
    · have ha : ∀ t, a t = 0 := by
        intro t
        by_contra ht
        exact h ⟨t, ht⟩
      simp [proportionalMergeFactor, ha]

/-- The actual normalized proportional merge has a row-linear certificate. -/
theorem exists_proportional_merge_matrix {σ τ ν : Type*}
    [Fintype σ] [Fintype τ] (a : σ → ℂ) (v : ν → ℂ) (B : Matrix τ ν ℂ) :
    ∃ R : Matrix (Option τ) (σ ⊕ τ) ℂ,
      mergedProportionalRows a v B = R * proportionalRows a v B :=
  exists_row_recombination _ _ (mergedProportionalRows_mem_span a v B)

theorem rank_proportional_merge_le {σ τ ν : Type*}
    [Fintype σ] [Fintype τ] [Fintype ν]
    (a : σ → ℂ) (v : ν → ℂ) (B : Matrix τ ν ℂ) :
    (mergedProportionalRows a v B).rank ≤ (proportionalRows a v B).rank := by
  obtain ⟨R, hR⟩ := exists_proportional_merge_matrix a v B
  rw [hR]
  exact Matrix.rank_mul_le_right _ _

/-- Proposition 6.1, including weights and all linear row cleanup/column-removal steps.
No polynomial-representation, degree, or rank bound is assumed. -/
theorem rank_trajectory_postprocess {η : Type*} [Fintype η] [DecidableEq η]
    (k p M N M' N' : ℕ) (J : Matrix η η ℂ) (δ : ℂ)
    (L : Fin k → Matrix η η ℂ) (samples : Fin M → Fin k → ℂ)
    (weights : Fin M → ℂ) (coord : Matrix η η ℂ →ₗ[ℂ] (Fin N → ℂ))
    (R : Matrix (Fin M') (Fin M) ℂ) (columns : Fin N' → Fin N) :
    (coefficientPostprocess R columns
      (trajectoryCoefficientMatrix k p J δ L samples weights coord)).rank ≤
      min M' (min N' ((((k + 1) * p) + k).choose k)) := by
  apply le_min
  · exact Matrix.rank_le_height _
  · apply le_min
    · exact Matrix.rank_le_width _
    · exact (rank_coefficientPostprocess_le R columns _).trans
        (rank_trajectoryCoefficientMatrix k p J δ L samples weights coord)

open PauliPlacement Placement

private theorem pauli_bit_cases (x : ZMod 2) : x = 0 ∨ x = 1 := by
  fin_cases x
  · exact Or.inl rfl
  · exact Or.inr rfl

/-- Hilbert--Schmidt orthogonality for the four actual one-qubit Pauli matrices. -/
theorem trace_pauli_atom_product (p q : BitLabel) :
    Matrix.trace ((atom p).conjTranspose * atom q) = if p = q then 2 else 0 := by
  rcases p with ⟨x, z⟩
  rcases q with ⟨y, w⟩
  rcases pauli_bit_cases x with rfl | rfl <;>
    rcases pauli_bit_cases z with rfl | rfl <;>
    rcases pauli_bit_cases y with rfl | rfl <;>
    rcases pauli_bit_cases w with rfl | rfl <;>
    norm_num [atom, atomIndex, PauliMatrix.pauliMatrix, Matrix.trace,
      Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_two, Prod.ext_iff]

theorem trace_pauli_tensor {n : ℕ} (M : Fin n → Matrix (Fin 2) (Fin 2) ℂ) :
    Matrix.trace (tensor M) = ∏ i, Matrix.trace (M i) := by
  simp only [Matrix.trace, Matrix.diag, tensor]
  exact (Fintype.prod_sum (fun i z => M i z z)).symm

/-- Distinct tensor Pauli strings are orthogonal; their squared norm is `2^n`. -/
theorem trace_pauli_product {n : ℕ} (p q : PauliLabel n) :
    Matrix.trace ((labelMatrix p).conjTranspose * labelMatrix q) =
      if p = q then (2 : ℂ) ^ n else 0 := by
  rw [labelMatrix, labelMatrix, tensor_conjTranspose, tensor_mul, trace_pauli_tensor]
  simp_rw [trace_pauli_atom_product]
  by_cases hpq : p = q
  · subst q
    simp
  · simp only [hpq, ↓reduceIte]
    have h : ∃ i, p i ≠ q i := by
      by_contra! h
      exact hpq (funext h)
    obtain ⟨i, hi⟩ := h
    exact Finset.prod_eq_zero (Finset.mem_univ i) (by simp [hi])

/-- Actual Pauli coefficients: `trace(P† A) / 2^n`, with fixed phase-free strings. -/
def pauliCoordinates (n : ℕ) : PMatrix n →ₗ[ℂ] (PauliLabel n → ℂ) where
  toFun A p := ((2 : ℂ) ^ n)⁻¹ * Matrix.trace ((labelMatrix p).conjTranspose * A)
  map_add' A B := by
    ext p
    simp [Matrix.trace_add, mul_add]
  map_smul' c A := by
    ext p
    simp [Matrix.trace_smul, mul_left_comm]

@[simp] theorem pauliCoordinates_labelMatrix {n : ℕ} (p q : PauliLabel n) :
    pauliCoordinates n (labelMatrix q) p = if p = q then 1 else 0 := by
  simp only [pauliCoordinates, LinearMap.coe_mk, AddHom.coe_mk, trace_pauli_product]
  split_ifs <;> simp

/-- The trace map recovers the collected coefficient of every finite Pauli expansion. -/
theorem pauliCoordinates_expansion {n : ℕ} (c : PauliLabel n → ℂ) :
    pauliCoordinates n (∑ q, c q • labelMatrix q) = c := by
  ext p
  simp [map_sum, map_smul, pauliCoordinates_labelMatrix]

/-- Orthogonality proves independence of the complete actual tensor-Pauli family. -/
theorem pauli_labelMatrix_linearIndependent (n : ℕ) :
    LinearIndependent ℂ (labelMatrix (n := n)) := by
  apply Fintype.linearIndependent_iff.mpr
  intro c hc p
  have h := congrArg (fun A => pauliCoordinates n A p) hc
  simpa only [pauliCoordinates_expansion, map_zero, Pi.zero_apply] using h

/-- Every `n`-qubit matrix has exactly the coefficients computed by the trace map. -/
theorem pauliCoordinates_reconstruct {n : ℕ} (A : PMatrix n) :
    ∑ p, pauliCoordinates n A p • labelMatrix p = A := by
  have hcard : Fintype.card (PauliLabel n) = Module.finrank ℂ (PMatrix n) := by
    simp [PauliLabel, PMatrix, State, Module.finrank_matrix, ← mul_pow]
  have hspan := (pauli_labelMatrix_linearIndependent n).span_eq_top_of_card_eq_finrank hcard
  have hmem : A ∈ Submodule.span ℂ (Set.range (labelMatrix (n := n))) := by
    rw [hspan]
    trivial
  obtain ⟨c, hc⟩ := (Submodule.mem_span_range_iff_exists_fun ℂ).mp hmem
  have hcoeff := congrArg (pauliCoordinates n) hc
  rw [pauliCoordinates_expansion] at hcoeff
  simpa only [hcoeff] using hc

/-- The actual Pauli coefficient table of any finite Kraus family. -/
def pauliCoefficientMatrix {n : ℕ} {ι : Type*} (K : ι → PMatrix n) :
    Matrix ι (PauliLabel n) ℂ := fun i => pauliCoordinates n (K i)

/-- Passing between matrix entries and the complete Pauli basis preserves rank.
Both changes of coordinates are constructed from the trace formula and Pauli
reconstruction, rather than postulated as a rank-preservation hypothesis. -/
theorem rank_pauliCoefficientMatrix {n : ℕ} {ι : Type*} [Fintype ι]
    (K : ι → PMatrix n) :
    (pauliCoefficientMatrix K).rank = (flattenKraus K).rank := by
  classical
  let E : Matrix (State n × State n) (PauliLabel n) ℂ :=
    fun x p => pauliCoordinates n (Matrix.single x.1 x.2 1) p
  let P : Matrix (PauliLabel n) (State n × State n) ℂ :=
    fun p x => labelMatrix p x.1 x.2
  have hE : pauliCoefficientMatrix K = flattenKraus K * E := by
    ext i p
    change pauliCoordinates n (K i) p =
      ∑ x : State n × State n, K i x.1 x.2 *
        pauliCoordinates n (Matrix.single x.1 x.2 1) p
    rw [Fintype.sum_prod_type]
    exact linear_coordinates_eq_sum (pauliCoordinates n) (K i) p
  have hP : flattenKraus K = pauliCoefficientMatrix K * P := by
    ext i x
    have h := congrArg (fun A : PMatrix n => A x.1 x.2)
      (pauliCoordinates_reconstruct (K i))
    change K i x.1 x.2 = ∑ p, pauliCoordinates n (K i) p * labelMatrix p x.1 x.2
    simpa [Matrix.sum_apply, Matrix.smul_apply] using h.symm
  apply le_antisymm
  · rw [hE]
    exact Matrix.rank_mul_le_left _ _
  · rw [hP]
    exact Matrix.rank_mul_le_left _ _

/-- Equal CP maps have equal actual Pauli coefficient ranks, even when their
Kraus lists have different lengths. This is derived from Choi Gram equality. -/
theorem rank_pauliCoefficientMatrix_eq_of_cp_eq {n : ℕ} {ι κ : Type*}
    [Fintype ι] [Fintype κ] (K : ι → PMatrix n) (L : κ → PMatrix n)
    (h : MatrixMap.of_kraus K K = MatrixMap.of_kraus L L) :
    (pauliCoefficientMatrix K).rank = (pauliCoefficientMatrix L).rank := by
  rw [rank_pauliCoefficientMatrix, rank_pauliCoefficientMatrix]
  have hgram : (flattenKraus K).conjTranspose * flattenKraus K =
      (flattenKraus L).conjTranspose * flattenKraus L := by
    rw [flattenKraus_gram_eq_choi_transpose, flattenKraus_gram_eq_choi_transpose, h]
  have hrank := congrArg Matrix.rank hgram
  simpa only [Matrix.rank_conjTranspose_mul_self] using hrank

/-- Finite concrete row cleanup: delete zero rows, merge a proportional group,
normalize branch phases, reindex rows, and compose these operations. -/
inductive CoefficientRowCleanup {ν : Type} :
    {ι : Type} → [Fintype ι] → Matrix ι ν ℂ →
    {ι' : Type} → [Fintype ι'] → Matrix ι' ν ℂ → Prop
  | refl {ι : Type} [Fintype ι] (A : Matrix ι ν ℂ) : CoefficientRowCleanup A A
  | deleteZeroRows {ι ι' : Type} [Fintype ι] [Fintype ι']
      (A : Matrix ι ν ℂ) (rows : ι' → ι) (hinj : Function.Injective rows)
      (hkeep : ∀ i, A.row i ≠ 0 → ∃ j, rows j = i) :
      CoefficientRowCleanup A (A.submatrix rows id)
  | mergeProportional {σ τ : Type} [Fintype σ] [Fintype τ]
      (a : σ → ℂ) (v : ν → ℂ) (B : Matrix τ ν ℂ) :
      CoefficientRowCleanup (proportionalRows a v B) (mergedProportionalRows a v B)
  | phaseRows {ι : Type} [Fintype ι] [DecidableEq ι]
      (A : Matrix ι ν ℂ) (c : ι → ℂ) (hc : ∀ i, ‖c i‖ = 1) :
      CoefficientRowCleanup A (Matrix.diagonal c * A)
  | trans {ι ι' ι'' : Type} [Fintype ι] [Fintype ι'] [Fintype ι'']
      {A : Matrix ι ν ℂ} {B : Matrix ι' ν ℂ} {C : Matrix ι'' ν ℂ} :
      CoefficientRowCleanup A B → CoefficientRowCleanup B C → CoefficientRowCleanup A C

theorem CoefficientRowCleanup.reindex {ν ι ι' : Type} [Fintype ι] [Fintype ι']
    (A : Matrix ι ν ℂ) (rows : ι' ≃ ι) :
    CoefficientRowCleanup A (A.submatrix rows id) :=
  .deleteZeroRows A rows rows.injective
    (fun i _ => ⟨rows.symm i, rows.apply_symm_apply i⟩)

/-- R-Phase at a selected branch is an explicit cleanup step. The other rows
are multiplied by one, and the selected row by the stated unit complex phase. -/
theorem CoefficientRowCleanup.phaseAt {ν ι : Type} [Fintype ι] [DecidableEq ι]
    (A : Matrix ι ν ℂ) (i : ι) (θ : ℝ) :
    CoefficientRowCleanup A
      (Matrix.diagonal (Function.update (fun _ => (1 : ℂ)) i
        (Complex.exp (θ * Complex.I))) * A) := by
  apply CoefficientRowCleanup.phaseRows
  intro j
  by_cases h : j = i
  · subst j
    simp [Complex.norm_exp_ofReal_mul_I]
  · simp [Function.update_of_ne h]

/-- The exact zero-row filter is an available concrete cleanup step. -/
theorem CoefficientRowCleanup.removeZeroRows {ν ι : Type} [Fintype ι]
    (A : Matrix ι ν ℂ) :
    (letI : Fintype {i // A.row i ≠ 0} := Fintype.ofFinite _;
      CoefficientRowCleanup A
        (A.submatrix (Subtype.val : {i // A.row i ≠ 0} → ι) id)) := by
  classical
  let _ : Fintype {i // A.row i ≠ 0} := Fintype.ofFinite _
  exact .deleteZeroRows A Subtype.val Subtype.val_injective (fun i hi => ⟨⟨i, hi⟩, rfl⟩)

/-- Every finite concrete cleanup sequence decreases or preserves coefficient rank. -/
theorem CoefficientRowCleanup.rank_le {ν ι ι' : Type}
    [Fintype ν] [Fintype ι] [Fintype ι'] {A : Matrix ι ν ℂ} {B : Matrix ι' ν ℂ}
    (h : CoefficientRowCleanup A B) : B.rank ≤ A.rank := by
  induction h with
  | refl => exact le_rfl
  | deleteZeroRows A rows _ _ => exact Matrix.rank_submatrix_le A rows id
  | mergeProportional a v B => exact rank_proportional_merge_le a v B
  | phaseRows A c _ => exact Matrix.rank_mul_le_right (Matrix.diagonal c) A
  | trans _ _ hAB hBC => exact hBC.trans hAB

/-- Weighted Pauli coefficient table of the actual frontend drift/jump trajectories. -/
def sampledPauliCoefficientMatrix (n k p M : ℕ)
    (J : PMatrix n) (δ : ℝ) (L : Fin k → PMatrix n)
    (samples : Fin M → Fin k → ℝ) (weights : Fin M → ℝ) :
    Matrix (Fin M) (PauliLabel n) ℂ :=
  trajectoryCoefficientMatrix k p J (δ : ℂ) L
    (fun a i => (samples a i : ℂ)) (fun a => (Real.sqrt (weights a) : ℂ))
    (pauliCoordinates n)

/-- Proposition 6.1 for the actual Pauli table, real sampled times, square-root
quadrature weights, a finite concrete cleanup sequence, and its active columns.
The polynomial representation, degree bound and rank bound are all derived. -/
theorem proposition_6_1 (n k p M M' N' : ℕ)
    (J : PMatrix n) (δ : ℝ) (L : Fin k → PMatrix n)
    (samples : Fin M → Fin k → ℝ) (weights : Fin M → ℝ)
    (B : Matrix (Fin M') (PauliLabel n) ℂ)
    (hcleanup : CoefficientRowCleanup
      (sampledPauliCoefficientMatrix n k p M J δ L samples weights) B)
    (columns : Fin N' → PauliLabel n)
    (_hcolumns : Function.Injective columns ∧
      ∀ q, B.col q ≠ 0 ↔ ∃ j, columns j = q) :
    (B.submatrix id columns).rank ≤
      min M' (min N' ((((k + 1) * p) + k).choose k)) := by
  apply le_min
  · exact Matrix.rank_le_height _
  · apply le_min
    · exact Matrix.rank_le_width _
    · exact (Matrix.rank_submatrix_le B id columns).trans
        (hcleanup.rank_le.trans
          (rank_trajectoryCoefficientMatrix k p J (δ : ℂ) L
            (fun a i => (samples a i : ℂ)) (fun a => (Real.sqrt (weights a) : ℂ))
            (pauliCoordinates n)))

end ChannelFormalization
