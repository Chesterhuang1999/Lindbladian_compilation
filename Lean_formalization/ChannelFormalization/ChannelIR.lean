import ChannelFormalization.KrausEquivalence
import Physlib.Relativity.PauliMatrices.Basic
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Logic.Relation

/-! ChannelIR syntax, its erasure to an independently checked arity language, and
the algebraic identities underlying Figure 5's six rewrites. `Op` is the
additive-expression presentation of a Kraus sum. As in T-Kraus and T-Zero,
zero coefficients and empty sums are well typed; removing zeros is a separate
canonicalization operation. The general `Rewrite` relation below also admits
unguarded phase changes and singleton proportional merges. Paper-specific guards
and deterministic termination are supplied in `GuardedRewrites`.
Qubit states are bit strings; their cardinality is exactly `2^n`. -/
noncomputable section
open scoped BigOperators Matrix
open Matrix
namespace ChannelFormalization
namespace ChannelIR

abbrev QState (n : ℕ) := Fin n → Fin 2
abbrev QMatrix (n : ℕ) := Matrix (QState n) (QState n) ℂ

@[simp] theorem qstate_card (n : ℕ) : Fintype.card (QState n) = 2 ^ n := by
  simp [QState]

inductive Primitive (n : ℕ) where
  | pauli (θ : ℝ) (word : Fin n → Fin 1 ⊕ Fin 3)
  | block (matrix : QMatrix n)

def Primitive.denote {n} : Primitive n → QMatrix n
  | .pauli θ word => fun x y => Complex.exp (θ * Complex.I) *
      ∏ i, PauliMatrix.pauliMatrix (word i) (x i) (y i)
  | .block q => q

inductive Op (n : ℕ) where
  | zero
  | primitive (p : Primitive n)
  | add (a b : Op n)
  | scale (c : ℂ) (a : Op n)

def Op.denote {n} : Op n → QMatrix n
  | .zero => 0
  | .primitive p => p.denote
  | .add a b => a.denote + b.denote
  | .scale c a => c • a.denote

/-- The untyped erasure retains the declared arity at every primitive and zero. -/
inductive RawOp where
  | zero (n : ℕ)
  | primitive (n : ℕ)
  | add (a b : RawOp)
  | scale (c : ℂ) (a : RawOp)

/-- An independent arity judgment. In particular, addition has no mixed-arity rule. -/
inductive RawOp.HasArity : RawOp → ℕ → Prop where
  | zero (n) : HasArity (.zero n) n
  | primitive (n) : HasArity (.primitive n) n
  | add {a b n} : HasArity a n → HasArity b n → HasArity (.add a b) n
  | scale {a n} (c) : HasArity a n → HasArity (.scale c a) n

def Op.erase {n} : Op n → RawOp
  | .zero => .zero n
  | .primitive _ => .primitive n
  | .add a b => .add a.erase b.erase
  | .scale c a => .scale c a.erase

theorem Op.erase_hasArity {n} (a : Op n) : a.erase.HasArity n := by
  induction a with
  | zero => exact .zero n
  | primitive p => exact .primitive n
  | add a b ha hb => exact .add ha hb
  | scale c a ha => exact .scale c ha

/-- Arity is determined by the raw expression, rather than assumed by a rewrite. -/
theorem RawOp.HasArity.unique {a n m} (h : HasArity a n) (g : HasArity a m) : n = m := by
  induction h with
  | zero n => cases g; rfl
  | primitive n => cases g; rfl
  | add h₁ h₂ ih₁ ih₂ => cases g with | add g₁ g₂ => exact ih₁ g₁
  | scale c h ih => cases g with | scale c g => exact ih g

abbrev Channel (n : ℕ) := List (Op n)
def Channel.erase {n} (C : Channel n) : List RawOp := C.map Op.erase
def WellTyped (C : List RawOp) (n : ℕ) : Prop := ∀ a ∈ C, a.HasArity n

theorem Channel.erase_wellTyped {n} (C : Channel n) : WellTyped C.erase n := by
  intro a ha
  obtain ⟨b, _, rfl⟩ := List.mem_map.mp ha
  exact b.erase_hasArity

/-- The finite Kraus sum, as an actual Physlib linear matrix map. -/
def Channel.denote {n} (C : Channel n) : MatrixMap (QState n) (QState n) ℂ :=
  (C.map fun a => MatrixMap.conj a.denote).sum

@[simp] theorem Channel.denote_nil {n} : (Channel.denote ([] : Channel n)) = 0 := rfl
@[simp] theorem Channel.denote_cons {n} (a : Op n) (C : Channel n) :
    Channel.denote (a :: C) = MatrixMap.conj a.denote + C.denote := rfl
@[simp] theorem Channel.denote_append {n} (C D : Channel n) :
    (C ++ D).denote = C.denote + D.denote := by simp [Channel.denote]

/-- Theorem 4.1: the denotation is linear by its type and completely positive. -/
theorem theorem_4_1 {n} (C : Channel n) : C.denote.IsCompletelyPositive := by
  induction C with
  | nil => exact MatrixMap.IsCompletelyPositive.zero _ _
  | cons a C ih => exact (MatrixMap.conj_isCompletelyPositive a.denote).add ih

/-- A fixed-length family becomes a channel in its given branch order. -/
def ofFamily {n m} (K : Fin m → Op n) : Channel n := List.ofFn K

theorem denote_ofFamily {n m} (K : Fin m → Op n) :
    (ofFamily K).denote = MatrixMap.of_kraus (fun i => (K i).denote)
      (fun i => (K i).denote) := by
  simp [ofFamily, Channel.denote, List.map_ofFn, List.sum_ofFn,
    MatrixMap.of_kraus_eq_sum_conj]

def Channel.krausFamily {n} (C : Channel n) (i : Fin C.length) : QMatrix n :=
  (C.get i).denote

theorem Channel.denote_eq_of_kraus {n} (C : Channel n) :
    C.denote = MatrixMap.of_kraus C.krausFamily C.krausFamily := by
  change C.denote = MatrixMap.of_kraus (fun i : Fin C.length => (C.get i).denote)
    (fun i : Fin C.length => (C.get i).denote)
  have h := denote_ofFamily (fun i : Fin C.length => C.get i)
  simpa [ofFamily, List.ofFn_get] using h

/-- Flatten the additive AST into the syntax's finite list of primitive terms. -/
def Op.terms {n} : Op n → List (ℂ × Primitive n)
  | .zero => []
  | .primitive p => [(1, p)]
  | .add a b => a.terms ++ b.terms
  | .scale c a => a.terms.map (fun t => (c * t.1, t.2))

def termsDenote {n} (ts : List (ℂ × Primitive n)) : QMatrix n :=
  (ts.map fun t => t.1 • t.2.denote).sum

theorem termsDenote_append {n} (ts us : List (ℂ × Primitive n)) :
    termsDenote (ts ++ us) = termsDenote ts + termsDenote us := by
  simp [termsDenote]

theorem termsDenote_scale {n} (c : ℂ) (ts : List (ℂ × Primitive n)) :
    termsDenote (ts.map (fun t => (c * t.1, t.2))) = c • termsDenote ts := by
  induction ts with
  | nil => simp [termsDenote]
  | cons t ts ih => simpa [termsDenote, smul_add, smul_smul] using ih

theorem Op.denote_terms {n} (a : Op n) : termsDenote a.terms = a.denote := by
  induction a with
  | zero => rfl
  | primitive p => simp [Op.terms, termsDenote, Op.denote]
  | add a b ha hb => simpa [Op.terms, Op.denote, termsDenote_append] using congrArg₂ (· + ·) ha hb
  | scale c a ha => simpa [Op.terms, termsDenote_scale, Op.denote] using congrArg (c • ·) ha

/-- Remove all exact zero coefficients, including the empty result. This is a
canonical-form operation, not a prerequisite of the paper's typing judgment.
A later equal-primitive merge is represented by `mergeTerm`. -/
def Op.canonicalTerms {n} (a : Op n) : List (ℂ × Primitive n) :=
  a.terms.filter (fun t => decide (t.1 ≠ 0))

theorem canonicalTerms_nonzero {n} (a : Op n) {t : ℂ × Primitive n}
    (ht : t ∈ a.canonicalTerms) : t.1 ≠ 0 := by
  exact of_decide_eq_true (List.mem_filter.mp ht).2

theorem termsDenote_filter {n} (ts : List (ℂ × Primitive n)) :
    termsDenote (ts.filter (fun t => decide (t.1 ≠ 0))) = termsDenote ts := by
  induction ts with
  | nil => rfl
  | cons t ts ih =>
      by_cases ht : t.1 = 0
      · simp [termsDenote, ht] at ih ⊢
        exact ih
      · simpa [termsDenote, ht] using ih

theorem Op.denote_canonicalTerms {n} (a : Op n) :
    termsDenote a.canonicalTerms = a.denote :=
  (termsDenote_filter a.terms).trans a.denote_terms

/-- T-Kraus enforces the common primitive arity and permits every complex
coefficient, including zero. T-Zero types the empty list at its ambient arity. -/
def SurfaceKrausWellTyped (ts : List (ℂ × ℕ)) (n : ℕ) : Prop :=
  ∀ t ∈ ts, t.2 = n

/-- Nonzero coefficients are a canonical-form invariant, separate from typing. -/
def SurfaceKrausCanonical (ts : List (ℂ × ℕ)) (n : ℕ) : Prop :=
  SurfaceKrausWellTyped ts n ∧ ∀ t ∈ ts, t.1 ≠ 0

/-- The unfiltered surface form already satisfies the current typing rules. -/
theorem Op.terms_wellTyped {n} (a : Op n) :
    SurfaceKrausWellTyped (a.terms.map (fun t => (t.1, n))) n := by
  intro t ht
  obtain ⟨s, _, rfl⟩ := List.mem_map.mp ht
  rfl

def Op.surfaceErase {n} (a : Op n) : List (ℂ × ℕ) :=
  a.canonicalTerms.map (fun t => (t.1, n))

theorem Op.surfaceErase_wellTyped {n} (a : Op n) : SurfaceKrausWellTyped a.surfaceErase n := by
  intro t ht
  obtain ⟨s, hs, rfl⟩ := List.mem_map.mp ht
  rfl

theorem Op.surfaceErase_canonical {n} (a : Op n) :
    SurfaceKrausCanonical a.surfaceErase n := by
  refine ⟨a.surfaceErase_wellTyped, ?_⟩
  intro t ht
  obtain ⟨s, hs, rfl⟩ := List.mem_map.mp ht
  exact canonicalTerms_nonzero a hs

/-! A finite common-basis sum has purely syntactic coordinates. The following
normalizer implements distributivity, scalar multiplication, finite-sum
interchange, equal-term collection, and deletion of zero coefficients. Its
correctness holds in every complex module, independently of quantum semantics. -/
inductive BasisSum (d : ℕ) where
  | zero
  | atom (j : Fin d)
  | add (a b : BasisSum d)
  | scale (c : ℂ) (a : BasisSum d)

def BasisSum.eval {d V} [AddCommGroup V] [Module ℂ V]
    (P : Fin d → V) : BasisSum d → V
  | .zero => 0
  | .atom j => P j
  | .add a b => a.eval P + b.eval P
  | .scale c a => c • a.eval P

def BasisSum.coefficients {d} (e : BasisSum d) : Fin d → ℂ :=
  e.eval (fun j => Pi.single j 1)

def BasisSum.toOp {n d} (P : Fin d → Primitive n) : BasisSum d → Op n
  | .zero => .zero
  | .atom j => .primitive (P j)
  | .add a b => .add (a.toOp P) (b.toOp P)
  | .scale c a => .scale c (a.toOp P)

theorem BasisSum.denote_toOp {n d} (P : Fin d → Primitive n) (e : BasisSum d) :
    (e.toOp P).denote = e.eval (fun j => (P j).denote) := by
  induction e with
  | zero => rfl
  | atom j => rfl
  | add a b ha hb => simp [toOp, Op.denote, eval, ha, hb]
  | scale c a ha => simp [toOp, Op.denote, eval, ha]

theorem BasisSum.eval_eq_sum_coefficients {d V} [AddCommGroup V] [Module ℂ V]
    (P : Fin d → V) (e : BasisSum d) :
    e.eval P = ∑ j, e.coefficients j • P j := by
  classical
  induction e with
  | zero => simp [eval, coefficients]
  | atom k => simp [eval, coefficients, Pi.single_apply]
  | add a b ha hb =>
    simpa [eval, coefficients, add_smul, Finset.sum_add_distrib] using congrArg₂ (· + ·) ha hb
  | scale c a ha =>
    simpa [eval, coefficients, smul_smul, Finset.smul_sum] using congrArg (c • ·) ha

def BasisSum.canonicalList {d} (c : Fin d → ℂ) : List (Fin d) → BasisSum d
  | [] => .zero
  | j :: js => if c j = 0 then canonicalList c js
      else .add (.scale (c j) (.atom j)) (canonicalList c js)

def BasisSum.canonical {d} (c : Fin d → ℂ) : BasisSum d :=
  canonicalList c (List.ofFn id)

theorem BasisSum.eval_canonicalList {d V} [AddCommGroup V] [Module ℂ V]
    (P : Fin d → V) (c : Fin d → ℂ) (js : List (Fin d)) :
    (canonicalList c js).eval P = (js.map fun j => c j • P j).sum := by
  classical
  induction js with
  | nil => rfl
  | cons j js ih =>
    by_cases h : c j = 0
    · simp [canonicalList, h, ih]
    · simp [canonicalList, h, eval, ih]

theorem BasisSum.eval_canonical {d V} [AddCommGroup V] [Module ℂ V]
    (P : Fin d → V) (c : Fin d → ℂ) :
    (canonical c).eval P = ∑ j, c j • P j := by
  simpa [canonical, List.map_ofFn, List.sum_ofFn] using
    eval_canonicalList P c (List.ofFn id)

@[simp] theorem BasisSum.coefficients_canonical {d} (c : Fin d → ℂ) :
    (canonical c).coefficients = c := by
  funext j
  simp [coefficients, eval_canonical, Finset.sum_apply, Pi.single_apply]

@[simp] theorem BasisSum.canonical_zero {d} : canonical (0 : Fin d → ℂ) = .zero := by
  have h (js : List (Fin d)) : canonicalList (0 : Fin d → ℂ) js = .zero := by
    induction js with
    | nil => rfl
    | cons j js ih => simp [canonicalList, ih]
  exact h _

/-- Collected, fixed-order common-basis syntax with every exact zero coefficient removed. -/
def canonicalBasisOp {n d} (P : Fin d → Primitive n) (c : Fin d → ℂ) : Op n :=
  (BasisSum.canonical c).toOp P

theorem canonicalBasisOp_denote {n d} (P : Fin d → Primitive n) (c : Fin d → ℂ) :
    (canonicalBasisOp P c).denote = ∑ j, c j • (P j).denote := by
  rw [canonicalBasisOp, BasisSum.denote_toOp, BasisSum.eval_canonical]

@[simp] theorem canonicalBasisOp_zero {n d} (P : Fin d → Primitive n) :
    canonicalBasisOp P 0 = .zero := by simp [canonicalBasisOp, BasisSum.toOp]

/-- Both term simplifications and their additive/scalar contexts. -/
inductive OpRewrite {n} : Op n → Op n → Prop where
  | mergeTerm (a b : ℂ) (p : Primitive n) :
      OpRewrite (.add (.scale a (.primitive p)) (.scale b (.primitive p)))
        (.scale (a + b) (.primitive p))
  | zeroTerm (p : Primitive n) : OpRewrite (.scale 0 (.primitive p)) .zero
  /-- R-ZeroTerm's empty-sum bookkeeping in the additive AST. -/
  | emptyLeft (a : Op n) : OpRewrite (.add .zero a) a
  | emptyRight (a : Op n) : OpRewrite (.add a .zero) a
  | zeroScale (a : Op n) : OpRewrite (.scale 0 a) .zero
  | addLeft {a b} (c) : OpRewrite a b → OpRewrite (.add a c) (.add b c)
  | addRight {a b} (c) : OpRewrite a b → OpRewrite (.add c a) (.add c b)
  | scale {a b} (c) : OpRewrite a b → OpRewrite (.scale c a) (.scale c b)
  /-- Structural finite-sum bookkeeping: this rule computes symbolic coefficients;
  it has no matrix-equality premise. It bundles distribution, collection, and zero deletion. -/
  | collectFinite {d} (P : Fin d → Primitive n) (e : BasisSum d) :
      OpRewrite (e.toOp P) (canonicalBasisOp P e.coefficients)

theorem OpRewrite.sound {n} {a b : Op n} (h : OpRewrite a b) : a.denote = b.denote := by
  induction h with
  | mergeTerm a b p => simp [Op.denote, add_smul]
  | zeroTerm p => simp [Op.denote]
  | emptyLeft a => simp [Op.denote]
  | emptyRight a => simp [Op.denote]
  | zeroScale a => simp [Op.denote]
  | addLeft c h ih => simp [Op.denote, ih]
  | addRight c h ih => simp [Op.denote, ih]
  | scale c h ih => simp [Op.denote, ih]
  | collectFinite P e =>
      rw [BasisSum.denote_toOp, canonicalBasisOp_denote]
      exact BasisSum.eval_eq_sum_coefficients _ e

/-- An explicit finite sum in the operator syntax, used for unitary mixing. -/
def Op.sum {n m} (f : Fin m → Op n) : Op n :=
  (List.ofFn f).foldr Op.add .zero

theorem Op.denote_sum {n m} (f : Fin m → Op n) :
    (Op.sum f).denote = ∑ i, (f i).denote := by
  have h (l : List (Op n)) : (l.foldr Op.add .zero).denote = (l.map Op.denote).sum := by
    induction l with
    | nil => rfl
    | cons a l ih => simp [Op.denote, ih]
  simpa [Op.sum, List.map_ofFn, List.sum_ofFn] using h (List.ofFn f)

/-- Scaling a Kraus branch multiplies its CP contribution by the squared norm. -/
theorem conj_scale {S : Type*} [Fintype S] (c : ℂ) (K : Matrix S S ℂ) :
    MatrixMap.conj (c • K) = ((‖c‖ ^ 2 : ℝ) : ℂ) • MatrixMap.conj K := by
  ext ρ i j
  simp only [LinearMap.smul_apply, MatrixMap.conj_apply, Matrix.conjTranspose_smul,
    Matrix.smul_mul, Matrix.mul_smul, smul_smul]
  rw [mul_comm (star c) c]
  have hc : c * star c = ((‖c‖ ^ 2 : ℝ) : ℂ) := by
    rw [← Complex.normSq_eq_norm_sq]
    exact Complex.mul_conj c
  rw [hc]

theorem conj_phase {n} (θ : ℝ) (a : Op n) :
    MatrixMap.conj (Op.denote (.scale (Complex.exp (θ * Complex.I)) a)) =
      MatrixMap.conj a.denote := by
  simp [Op.denote, conj_scale, Complex.norm_exp_ofReal_mul_I]

def mergeWeight {m} (a : Fin m → ℂ) : ℂ :=
  (Real.sqrt (∑ i, ‖a i‖ ^ 2) : ℝ)

theorem mergeWeight_norm_sq {m} (a : Fin m → ℂ) :
    ‖mergeWeight a‖ ^ 2 = ∑ i, ‖a i‖ ^ 2 := by
  simp only [mergeWeight, Complex.norm_real, Real.norm_eq_abs,
    abs_of_nonneg (Real.sqrt_nonneg _)]
  exact Real.sq_sqrt (Finset.sum_nonneg fun i _ => sq_nonneg ‖a i‖)

theorem denote_mergeKraus {n m} (a : Fin m → ℂ) (K : Op n) :
    (ofFamily (fun i => Op.scale (a i) K)).denote =
      MatrixMap.conj (Op.denote (.scale (mergeWeight a) K)) := by
  rw [denote_ofFamily, MatrixMap.of_kraus_eq_sum_conj]
  simp only [Op.denote, conj_scale]
  rw [mergeWeight_norm_sq, Complex.ofReal_sum, Finset.sum_smul]

/-- General algebraic rewrites underlying Figure 5. Operator rewrites may occur
in any branch and under additive contexts. Phase and proportional merging here
are deliberately more general than the guarded deterministic paper rules. -/
inductive Rewrite {n} : Channel n → Channel n → Prop where
  | term (pre post : Channel n) {a b : Op n} : OpRewrite a b →
      Rewrite (pre ++ a :: post) (pre ++ b :: post)
  | phase (pre post : Channel n) (θ : ℝ) (a : Op n) :
      Rewrite (pre ++ .scale (Complex.exp (θ * Complex.I)) a :: post)
        (pre ++ a :: post)
  | zeroKraus (pre post : Channel n) : Rewrite (pre ++ .zero :: post) (pre ++ post)
  | mergeKraus {m} (pre post : Channel n) (a : Fin m → ℂ) (K : Op n) :
      Rewrite (pre ++ ofFamily (fun i => .scale (a i) K) ++ post)
        (pre ++ .scale (mergeWeight a) K :: post)
  | unitaryMix {m} (K : Fin m → Op n) (U : Matrix.unitaryGroup (Fin m) ℂ) :
      Rewrite (ofFamily K)
        (ofFamily (fun i => Op.sum (fun j => .scale (U.val i j) (K j))))

/-- Theorem 4.3's algebraic foundation: even the more general single-step
relation preserves semantics. -/
theorem theorem_4_3 {n} {C D : Channel n} (h : Rewrite C D) : C.denote = D.denote := by
  cases h with
  | term pre post h => simp [h.sound]
  | phase pre post θ a => simp [conj_phase]
  | zeroKraus pre post =>
      have hz : MatrixMap.conj (0 : QMatrix n) = 0 := by
        ext ρ i j
        simp [MatrixMap.conj_apply]
      simp [Op.denote, hz]
  | mergeKraus pre post a K => simp [denote_mergeKraus]
  | unitaryMix K U =>
      rw [denote_ofFamily, denote_ofFamily]
      simp only [Op.denote_sum, Op.denote]
      exact (krausMap_unitary_mix (fun i => (K i).denote) U).symm

/-- Lemma 4.2, one step: the erased target has the same arity. This uses the
structural erasure theorem; `RawOp.HasArity` has no T-Rewrite rule. -/
theorem lemma_4_2 {n} {C D : Channel n} (_h : Rewrite C D)
    (_hc : WellTyped C.erase n) : WellTyped D.erase n := D.erase_wellTyped

/-- The target's canonical surface representation satisfies the typing judgment.
Its additional nonzero-coefficient invariant is `surfaceErase_canonical`. -/
theorem lemma_4_2_surface {n} {C D : Channel n} (_h : Rewrite C D) :
    ∀ a ∈ D, SurfaceKrausWellTyped a.surfaceErase n := by
  intro a _
  exact a.surfaceErase_wellTyped

abbrev Rewrites {n} := Relation.ReflTransGen (@Rewrite n)

/-- The finite-sequence form of Theorem 4.3. -/
theorem rewrites_sound {n} {C D : Channel n} (h : Rewrites C D) : C.denote = D.denote := by
  induction h with
  | refl => rfl
  | tail h step ih => exact ih.trans (theorem_4_3 step)

/-- The finite-sequence form of Lemma 4.2. -/
theorem lemma_4_2_sequence {n} {C D : Channel n} (_h : Rewrites C D)
    (_hc : WellTyped C.erase n) : WellTyped D.erase n := D.erase_wellTyped

/-- Explicit zero padding is separate from the six directed simplification rules. -/
inductive PaddedRewrite {n} : Channel n → Channel n → Prop where
  | rewrite {C D} : Rewrite C D → PaddedRewrite C D
  | pad (pre post : Channel n) : PaddedRewrite (pre ++ post) (pre ++ .zero :: post)

abbrev PaddedRewrites {n} := Relation.ReflTransGen (@PaddedRewrite n)

theorem paddedRewrite_sound {n} {C D : Channel n} (h : PaddedRewrite C D) :
    C.denote = D.denote := by
  cases h with
  | rewrite h => exact theorem_4_3 h
  | pad pre post => exact (theorem_4_3 (Rewrite.zeroKraus pre post)).symm

theorem paddedRewrites_sound {n} {C D : Channel n} (h : PaddedRewrites C D) :
    C.denote = D.denote := by
  induction h with
  | refl => rfl
  | tail h step ih => exact ih.trans (paddedRewrite_sound step)

end ChannelIR
end ChannelFormalization
