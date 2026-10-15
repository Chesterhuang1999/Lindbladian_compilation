import ChannelFormalization.ChannelIR

/-! Proposition 4.4: finite rewriting reaches an equivalent collected common-basis
Kraus representation. The common basis is fixed in order and has phases already
absorbed in its coefficients. -/
noncomputable section
set_option backward.isDefEq.respectTransparency false
open scoped BigOperators Matrix
open Matrix

namespace ChannelFormalization.ChannelIR

def BasisSum.sum {d m} (f : Fin m → BasisSum d) : BasisSum d :=
  (List.ofFn f).foldr BasisSum.add .zero

theorem BasisSum.toOp_sum {n d m} (P : Fin d → Primitive n) (f : Fin m → BasisSum d) :
    (BasisSum.sum f).toOp P = Op.sum (fun i => (f i).toOp P) := by
  have h (xs : List (BasisSum d)) :
      (xs.foldr BasisSum.add .zero).toOp P =
        (xs.map (BasisSum.toOp P)).foldr Op.add .zero := by
    induction xs with
    | nil => rfl
    | cons x xs ih => simp [BasisSum.toOp, ih]
  simpa [BasisSum.sum, Op.sum, List.map_ofFn, Function.comp_def] using h (List.ofFn f)

theorem BasisSum.eval_sum {d m V} [AddCommGroup V] [Module ℂ V]
    (P : Fin d → V) (f : Fin m → BasisSum d) :
    (BasisSum.sum f).eval P = ∑ i, (f i).eval P := by
  have h (xs : List (BasisSum d)) :
      (xs.foldr BasisSum.add .zero).eval P = (xs.map (BasisSum.eval P)).sum := by
    induction xs with
    | nil => rfl
    | cons x xs ih => simp [BasisSum.eval, ih]
  simpa [BasisSum.sum, List.map_ofFn, List.sum_ofFn] using h (List.ofFn f)

/-- Normalize each branch using explicit symbolic coefficient collection. -/
theorem rewrites_collect_list {n d} (P : Fin d → Primitive n)
    (pre : Channel n) (es : List (BasisSum d)) :
    Rewrites (pre ++ es.map (BasisSum.toOp P))
      (pre ++ es.map (fun e => canonicalBasisOp P e.coefficients)) := by
  induction es generalizing pre with
  | nil => exact .refl
  | cons e es ih =>
    have step : Rewrite (pre ++ e.toOp P :: es.map (BasisSum.toOp P))
        (pre ++ canonicalBasisOp P e.coefficients :: es.map (BasisSum.toOp P)) :=
      .term pre _ (.collectFinite P e)
    apply Relation.ReflTransGen.trans (Relation.ReflTransGen.single step)
    simpa only [List.map_cons, List.append_assoc, List.singleton_append] using
      ih (pre ++ [canonicalBasisOp P e.coefficients])

theorem rewrites_collect_family {n d m} (P : Fin d → Primitive n)
    (e : Fin m → BasisSum d) :
    Rewrites (ofFamily (fun i => (e i).toOp P))
      (ofFamily (fun i => canonicalBasisOp P (e i).coefficients)) := by
  simpa [ofFamily, List.map_ofFn, Function.comp_def] using rewrites_collect_list P [] (List.ofFn e)

/-- Repeated R-ZeroKraus steps delete a trailing list of syntactic zeros. -/
theorem rewrites_remove_zero_tail {n} (C : Channel n) (k : ℕ) :
    Rewrites (C ++ List.replicate k Op.zero) C := by
  induction k with
  | zero => simpa using (Relation.ReflTransGen.refl : Rewrites C C)
  | succ k ih =>
    have step : Rewrite (C ++ Op.zero :: List.replicate k Op.zero)
        (C ++ List.replicate k Op.zero) := .zeroKraus C _
    simpa only [List.replicate_succ] using (Relation.ReflTransGen.single step).trans ih

def padCoefficients {m r d : ℕ} (B : Matrix (Fin r) (Fin d) ℂ) :
    Matrix (Fin m) (Fin d) ℂ :=
  fun i => if hi : i.val < r then B ⟨i.val, hi⟩ else 0

@[simp] theorem canonical_padCoefficients {n d m r}
    (P : Fin d → Primitive n) (B : Matrix (Fin r) (Fin d) ℂ) (i : Fin m) :
    canonicalBasisOp P (padCoefficients (m := m) B i) =
      if hi : i.val < r then canonicalBasisOp P (B ⟨i.val, hi⟩) else .zero := by
  by_cases hi : i.val < r <;> simp [padCoefficients, hi]

theorem family_padCoefficients {n d m r} (hrm : r ≤ m)
    (P : Fin d → Primitive n) (B : Matrix (Fin r) (Fin d) ℂ) :
    ofFamily (fun i : Fin m => canonicalBasisOp P (padCoefficients B i)) =
      ofFamily (fun i : Fin r => canonicalBasisOp P (B i)) ++
        List.replicate (m - r) Op.zero := by
  apply List.ext_getElem
  · simp [ofFamily]
    omega
  · intro i hi₁ hi₂
    simp only [ofFamily, List.getElem_ofFn, canonical_padCoefficients]
    by_cases hir : i < r
    · rw [List.getElem_append_left (by simpa [ofFamily] using hir)]
      simp [hir, List.getElem_ofFn]
    · rw [List.getElem_append_right (by simpa [ofFamily] using Nat.le_of_not_gt hir)]
      simp [hir]

theorem denote_canonical_family {n d m} (P : Fin d → Primitive n)
    (A : Matrix (Fin m) (Fin d) ℂ) :
    (ofFamily (fun i => canonicalBasisOp P (A i))).denote =
      MatrixMap.of_kraus (krausOfCoefficients (fun j => (P j).denote) A)
        (krausOfCoefficients (fun j => (P j).denote) A) := by
  simp only [denote_ofFamily, canonicalBasisOp_denote]
  rfl

theorem krausOfCoefficients_pad {n d m r} (P : Fin d → Primitive n)
    (B : Matrix (Fin r) (Fin d) ℂ) :
    krausOfCoefficients (fun j => (P j).denote) (padCoefficients (m := m) B) =
      padKraus (krausOfCoefficients (fun j => (P j).denote) B) := by
  funext i
  by_cases hi : i.val < r <;> simp [krausOfCoefficients, padCoefficients, padKraus, hi]

theorem BasisSum.coefficients_mix {d m}
    (A : Matrix (Fin m) (Fin d) ℂ) (U : Matrix (Fin m) (Fin m) ℂ) (i : Fin m) :
    (BasisSum.sum (fun j => .scale (U i j) (BasisSum.canonical (A j)))).coefficients =
      (U * A) i := by
  change (BasisSum.sum _).eval (fun j => Pi.single j 1) = _
  rw [BasisSum.eval_sum]
  change (∑ j, U i j • (BasisSum.canonical (A j)).coefficients) = _
  simp only [BasisSum.coefficients_canonical]
  funext k
  simp [Matrix.mul_apply, Finset.sum_apply]

/-- Proposition 4.4 for canonical, collected common-basis Kraus syntax.
The witness is a real finite rewrite sequence: one unitary mixing, symbolic
coefficient collection in each branch, and deletion of the padded zero branches.
The input and target use one fixed basis order; changing that order is merely
choosing another presentation of the same collected common-basis object. -/
theorem proposition_4_4 {n d m r}
    (P : Fin d → Primitive n) (hP : LinearIndependent ℂ (fun j => (P j).denote))
    (A : Matrix (Fin m) (Fin d) ℂ) (B : Matrix (Fin r) (Fin d) ℂ)
    (hrm : r ≤ m)
    (h : (ofFamily (fun i => canonicalBasisOp P (A i))).denote =
      (ofFamily (fun i => canonicalBasisOp P (B i))).denote) :
    Rewrites (ofFamily (fun i => canonicalBasisOp P (A i)))
      (ofFamily (fun i => canonicalBasisOp P (B i))) := by
  have heq :
      MatrixMap.of_kraus (krausOfCoefficients (fun j => (P j).denote) A)
          (krausOfCoefficients (fun j => (P j).denote) A) =
        MatrixMap.of_kraus (krausOfCoefficients (fun j => (P j).denote)
          (padCoefficients (m := m) B))
          (krausOfCoefficients (fun j => (P j).denote) (padCoefficients (m := m) B)) := by
    rw [krausOfCoefficients_pad, krausMap_padKraus hrm]
    simpa only [denote_canonical_family] using h
  obtain ⟨U, hU⟩ := coefficient_unitary_freedom
    (fun j => (P j).denote) hP A (padCoefficients (m := m) B) heq
  let e : Fin m → BasisSum d := fun i =>
    BasisSum.sum (fun j => .scale (U.val i j) (BasisSum.canonical (A j)))
  have he (i : Fin m) : (e i).coefficients = padCoefficients (m := m) B i := by
    rw [hU]
    exact BasisSum.coefficients_mix A U.val i
  have hin : (fun i => (e i).toOp P) =
      (fun i => Op.sum (fun j => .scale (U.val i j) (canonicalBasisOp P (A j)))) := by
    funext i
    dsimp only [e]
    rw [BasisSum.toOp_sum]
    rfl
  have hnorm := rewrites_collect_family P e
  rw [hin] at hnorm
  simp only [he] at hnorm
  have hmix := Relation.ReflTransGen.single
    (Rewrite.unitaryMix (fun i => canonicalBasisOp P (A i)) U)
  refine hmix.trans (hnorm.trans ?_)
  rw [family_padCoefficients hrm]
  exact rewrites_remove_zero_tail _ _

#print axioms proposition_4_4

end ChannelFormalization.ChannelIR
