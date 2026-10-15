import ChannelFormalization.ChannelIR
import Mathlib.Order.WellFounded

/-! The deterministic rules of Figure 5 of the current manuscript, on its
flattened finite sums. Permutation constructors only transport a genuine step;
permutations themselves are not rewrite steps. Thus the presentation respects
unordered sums without introducing administrative rewriting loops. -/
noncomputable section
open scoped BigOperators Matrix
open Matrix
namespace ChannelFormalization.ChannelIR.Guarded

abbrev Kraus (n : ℕ) := List (ℂ × Primitive n)
abbrev Channel (n : ℕ) := List (Kraus n)

/-- Primitive Pauli phases participate in the effective coefficient. A supplied
matrix block has no separately stored primitive phase. -/
def primitivePhase {n} : Primitive n → ℂ
  | .pauli θ _ => Complex.exp (θ * Complex.I)
  | .block _ => 1

def PositiveReal (z : ℂ) : Prop := z.im = 0 ∧ 0 < z.re

def zeroPhase {n} (t : ℂ × Primitive n) : Prop :=
  PositiveReal (t.1 * primitivePhase t.2)

/-- `zp` counts strictly positive real effective coefficients, not zero terms. -/
def zp {n} (K : Kraus n) : ℕ := by
  classical
  exact (K.filter (fun t => decide (zeroPhase t))).length

def scale {n} (c : ℂ) (K : Kraus n) : Kraus n :=
  K.map (fun t => (c * t.1, t.2))

def toOp {n} (K : Kraus n) : Op n :=
  K.foldr (fun t a => .add (.scale t.1 (.primitive t.2)) a) .zero

def toChannel {n} (C : Channel n) : ChannelIR.Channel n := C.map toOp

def denote {n} (C : Channel n) := (toChannel C).denote

@[simp] theorem denote_toOp {n} (K : Kraus n) :
    (toOp K).denote = termsDenote K := by
  induction K with
  | nil => rfl
  | cons t K ih =>
      change t.1 • t.2.denote + (toOp K).denote = t.1 • t.2.denote + termsDenote K
      rw [ih]

@[simp] theorem length_scale {n} (c : ℂ) (K : Kraus n) :
    (scale c K).length = K.length := List.length_map _

@[simp] theorem denote_scale {n} (c : ℂ) (K : Kraus n) :
    (toOp (scale c K)).denote = c • (toOp K).denote := by
  simpa [scale] using termsDenote_scale c K

theorem zp_le_length {n} (K : Kraus n) : zp K ≤ K.length := by
  classical
  exact List.length_filter_le _ _

theorem zp_perm {n} {K L : Kraus n} (h : K.Perm L) : zp K = zp L := by
  classical
  exact (h.filter _).length_eq

theorem denote_perm {n} {K L : Kraus n} (h : K.Perm L) :
    (toOp K).denote = (toOp L).denote := by
  simp only [denote_toOp, termsDenote]
  exact (h.map _).sum_eq

/-- The two local sum rules, closed under term order and additive context. -/
inductive TermStep {n} : Kraus n → Kraus n → Prop where
  | merge (pre post : Kraus n) (a b : ℂ) (p : Primitive n) :
      TermStep (pre ++ (a, p) :: (b, p) :: post)
        (pre ++ (a + b, p) :: post)
  | zero (pre post : Kraus n) (p : Primitive n) :
      TermStep (pre ++ (0, p) :: post) (pre ++ post)
  | perm {K L K' L' : Kraus n} :
      K.Perm K' → L.Perm L' → TermStep K L → TermStep K' L'

theorem TermStep.sound {n} {K L : Kraus n} (h : TermStep K L) :
    (toOp K).denote = (toOp L).denote := by
  induction h with
  | merge pre post a b p =>
      simp [termsDenote, add_smul, add_assoc]
  | zero pre post p => simp [termsDenote]
  | perm hK hL _ ih => exact (denote_perm hK).symm.trans (ih.trans (denote_perm hL))

theorem TermStep.length_lt {n} {K L : Kraus n} (h : TermStep K L) :
    L.length < K.length := by
  induction h with
  | merge => simp
  | zero => simp
  | perm hK hL _ ih => simpa [← hK.length_eq, ← hL.length_eq] using ih

/-- Exactly the five deterministic rule schemes. The phase and merge guards
are the manuscript's guards, not assumed decreases of a termination measure. -/
inductive Step {n} : Channel n → Channel n → Prop where
  | term (pre post : Channel n) {K L : Kraus n} : TermStep K L →
      Step (pre ++ K :: post) (pre ++ L :: post)
  | phase (pre post : Channel n) (θ : ℝ) (K : Kraus n)
      (guard : zp (scale (Complex.exp (θ * Complex.I)) K) < zp K) :
      Step (pre ++ scale (Complex.exp (θ * Complex.I)) K :: post)
        (pre ++ K :: post)
  | zeroKraus (pre post : Channel n) :
      Step (pre ++ [] :: post) (pre ++ post)
  | mergeKraus {m : ℕ} (pre post : Channel n) (a : Fin m → ℂ)
      (K : Kraus n) (guard : 2 ≤ m) :
      Step (pre ++ List.ofFn (fun i => scale (a i) K) ++ post)
        (pre ++ scale (mergeWeight a) K :: post)
  | perm {C D C' D' : Channel n} :
      C.Perm C' → D.Perm D' → Step C D → Step C' D'
  | termOrder {C D C' D' : Channel n} :
      List.Forall₂ List.Perm C C' → List.Forall₂ List.Perm D D' →
      Step C D → Step C' D'

@[simp] theorem denote_nil {n} : denote ([] : Channel n) = 0 := rfl
@[simp] theorem denote_cons {n} (K : Kraus n) (C : Channel n) :
    denote (K :: C) = MatrixMap.conj (toOp K).denote + denote C := rfl
@[simp] theorem denote_append {n} (C D : Channel n) :
    denote (C ++ D) = denote C + denote D := by
  simp [denote, toChannel]

theorem channelDenote_perm {n} {C D : Channel n} (h : C.Perm D) :
    denote C = denote D := by
  unfold denote toChannel ChannelIR.Channel.denote
  exact ((h.map toOp).map (fun a : Op n => MatrixMap.conj a.denote)).sum_eq

theorem channelDenote_termOrder {n} {C D : Channel n}
    (h : List.Forall₂ List.Perm C D) : denote C = denote D := by
  induction h with
  | nil => rfl
  | cons hK _ ih => simp only [denote_cons, denote_perm hK, ih]

/-- Connection to the ordinary finite matrix Kraus family, with precisely one
matrix for each flattened branch. -/
theorem denote_eq_of_kraus {n} (C : Channel n) :
    denote C = MatrixMap.of_kraus
      (fun i : Fin C.length => termsDenote (C.get i))
      (fun i : Fin C.length => termsDenote (C.get i)) := by
  rw [MatrixMap.of_kraus_eq_sum_conj]
  have h : C = List.ofFn (fun i : Fin C.length => C.get i) := (List.ofFn_get C).symm
  conv_lhs => rw [h]
  simp only [denote, toChannel, ChannelIR.Channel.denote, List.map_ofFn,
    List.sum_ofFn, Function.comp_apply, denote_toOp]

/-- The new guarded relation preserves the same Physlib CP-map semantics. -/
theorem Step.sound {n} {C D : Channel n} (h : Step C D) : denote C = denote D := by
  induction h with
  | term pre post h => simp [h.sound]
  | phase pre post θ K guard =>
      simp only [denote_append, denote_cons, denote_scale, conj_scale,
        Complex.norm_exp_ofReal_mul_I, one_pow, Complex.ofReal_one, one_smul]
  | zeroKraus pre post =>
      have hz : MatrixMap.conj (0 : QMatrix n) = 0 := by
        ext ρ i j
        simp [MatrixMap.conj_apply]
      simp [toOp, Op.denote, hz]
  | mergeKraus pre post a K guard =>
      have hgroup : denote (List.ofFn (fun i => scale (a i) K)) =
          MatrixMap.conj (toOp (scale (mergeWeight a) K)).denote := by
        simp only [denote, toChannel, List.map_ofFn, ChannelIR.Channel.denote,
          List.map_ofFn, List.sum_ofFn, Function.comp_apply, denote_scale, conj_scale]
        rw [mergeWeight_norm_sq, Complex.ofReal_sum, Finset.sum_smul]
      simp [hgroup]
  | perm hC hD _ ih =>
      exact (channelDenote_perm hC).symm.trans (ih.trans (channelDenote_perm hD))
  | termOrder hC hD _ ih =>
      exact (channelDenote_termOrder hC).symm.trans (ih.trans (channelDenote_termOrder hD))

/-- Every flattened term retains the declared qubit arity through rewriting. -/
theorem Step.wellTyped {n} {C D : Channel n} (_h : Step C D) :
    WellTyped (toChannel D).erase n := ChannelIR.Channel.erase_wellTyped _

def termCount {n} (C : Channel n) : ℕ := (C.map List.length).sum

def phaseCount {n} (C : Channel n) : ℕ := (C.map zp).sum

def measure {n} (C : Channel n) : ℕ × ℕ × ℕ :=
  (C.length, termCount C, termCount C - phaseCount C)

def MeasureLT : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop :=
  Prod.Lex (· < ·) (Prod.Lex (· < ·) (· < ·))

@[simp] theorem termCount_nil {n} : termCount ([] : Channel n) = 0 := rfl
@[simp] theorem termCount_cons {n} (K : Kraus n) (C : Channel n) :
    termCount (K :: C) = K.length + termCount C := rfl
@[simp] theorem termCount_append {n} (C D : Channel n) :
    termCount (C ++ D) = termCount C + termCount D := by simp [termCount]
@[simp] theorem phaseCount_nil {n} : phaseCount ([] : Channel n) = 0 := rfl
@[simp] theorem phaseCount_cons {n} (K : Kraus n) (C : Channel n) :
    phaseCount (K :: C) = zp K + phaseCount C := rfl
@[simp] theorem phaseCount_append {n} (C D : Channel n) :
    phaseCount (C ++ D) = phaseCount C + phaseCount D := by simp [phaseCount]

theorem phaseCount_le_termCount {n} (C : Channel n) : phaseCount C ≤ termCount C := by
  induction C with
  | nil => rfl
  | cons K C ih => exact Nat.add_le_add (zp_le_length K) ih

theorem measure_perm {n} {C D : Channel n} (h : C.Perm D) : measure C = measure D := by
  have ht : termCount C = termCount D := (h.map List.length).sum_eq
  have hp : phaseCount C = phaseCount D := (h.map zp).sum_eq
  simp [measure, h.length_eq, ht, hp]

theorem measure_termOrder {n} {C D : Channel n}
    (h : List.Forall₂ List.Perm C D) : measure C = measure D := by
  have ht : termCount C = termCount D := by
    induction h with
    | nil => rfl
    | cons hK _ ih => simp only [termCount_cons, hK.length_eq, ih]
  have hp : phaseCount C = phaseCount D := by
    clear ht
    induction h with
    | nil => rfl
    | cons hK _ ih => simp only [phaseCount_cons, zp_perm hK, ih]
  simp [measure, h.length_eq, ht, hp]

/-- Branch count first, then term count, then the number of non-positive-real
terms. Each actual rule strictly decreases this lexicographic measure. -/
theorem Step.decreases {n} {C D : Channel n} (h : Step C D) :
    MeasureLT (measure D) (measure C) := by
  induction h with
  | term pre post h =>
      have hlt := h.length_lt
      simp only [measure, List.length_append, List.length_cons, termCount_append,
        termCount_cons]
      apply Prod.Lex.right
      apply Prod.Lex.left
      omega
  | phase pre post θ K guard =>
      have hb := phaseCount_le_termCount (pre ++ K :: post)
      simp only [phaseCount_append, phaseCount_cons, termCount_append, termCount_cons] at hb
      simp only [measure, List.length_append, List.length_cons, length_scale,
        termCount_append, termCount_cons, phaseCount_append, phaseCount_cons]
      apply Prod.Lex.right
      apply Prod.Lex.right
      omega
  | zeroKraus pre post =>
      apply Prod.Lex.left
      simp
  | mergeKraus pre post a K guard =>
      apply Prod.Lex.left
      simp only [List.length_append, List.length_cons, List.length_ofFn]
      omega
  | perm hC hD _ ih => simpa [← measure_perm hC, ← measure_perm hD] using ih
  | termOrder hC hD _ ih =>
      simpa [← measure_termOrder hC, ← measure_termOrder hD] using ih

/-- Rewrite direction is reversed in `WellFounded`: successors are smaller. -/
theorem deterministic_terminates (n : ℕ) :
    WellFounded (fun D C : Channel n => Step C D) := by
  have hw : WellFounded MeasureLT :=
    Nat.lt_wfRel.wf.prod_lex (Nat.lt_wfRel.wf.prod_lex Nat.lt_wfRel.wf)
  exact (InvImage.wf measure hw).mono (fun _ _ h => h.decreases)

/-- No infinite sequence can consist entirely of deterministic rewrites. -/
theorem no_infinite_sequence {n} (f : ℕ → Channel n) :
    ¬ (∀ k, Step (f k) (f (k + 1))) := by
  intro h
  exact (wellFounded_iff_isEmpty_descending_chain.mp
    (deterministic_terminates n)).false ⟨f, h⟩

/-- The manuscript's definition: a program is normal precisely when no
further deterministic rule applies. This does not assert uniqueness. -/
def NormalForm {n} (C : Channel n) : Prop := ∀ D, ¬ Step C D

abbrev Steps {n} := Relation.ReflTransGen (@Step n)

theorem Steps.sound {n} {C D : Channel n} (h : Steps C D) : denote C = denote D := by
  induction h with
  | refl => rfl
  | tail _ h ih => exact ih.trans h.sound

/-- Every finite ChannelIR program admits a reachable deterministic normal form. -/
theorem exists_normalForm {n} (C : Channel n) :
    ∃ D, Steps C D ∧ NormalForm D := by
  classical
  induction C using (deterministic_terminates n).induction with
  | h C ih =>
      by_cases hn : NormalForm C
      · exact ⟨C, .refl, hn⟩
      · simp only [NormalForm, not_forall, not_not] at hn
        obtain ⟨D, hD⟩ := hn
        obtain ⟨E, hE, hnE⟩ := ih D hD
        exact ⟨E, (Relation.ReflTransGen.single hD).trans hE, hnE⟩

/-- The flattened sum used by R-UnitaryMix; zero terms are retained until the
separate R-ZeroTerm cleanup, exactly as in Figure 5. -/
def mixRow {n m} (K : Fin m → Kraus n) (a : Fin m → ℂ) : Kraus n :=
  (List.ofFn (fun j => scale (a j) (K j))).flatten

theorem denote_flatten {n} (Ks : List (Kraus n)) :
    termsDenote Ks.flatten = (Ks.map termsDenote).sum := by
  induction Ks with
  | nil => rfl
  | cons K Ks ih => simp [termsDenote_append, ih]

theorem denote_mixRow {n m} (K : Fin m → Kraus n) (a : Fin m → ℂ) :
    termsDenote (mixRow K a) = ∑ j, a j • termsDenote (K j) := by
  simp only [mixRow, denote_flatten, List.map_ofFn, List.sum_ofFn,
    Function.comp_apply, scale, termsDenote_scale]

theorem denote_ofFn {n m} (K : Fin m → Kraus n) :
    denote (List.ofFn K) = MatrixMap.of_kraus (fun i => termsDenote (K i))
      (fun i => termsDenote (K i)) := by
  simp only [denote, toChannel, ChannelIR.Channel.denote, List.map_ofFn,
    List.sum_ofFn, Function.comp_apply, denote_toOp, MatrixMap.of_kraus_eq_sum_conj]

/-- All six current Figure 5 rules. Only `Step`, the five deterministic rules,
is asserted to terminate; unrestricted unitary mixing is intentionally excluded
from that claim. -/
inductive Rewrite {n} : Channel n → Channel n → Prop where
  | deterministic {C D : Channel n} : Step C D → Rewrite C D
  | unitaryMix {m} (K : Fin m → Kraus n) (U : Matrix.unitaryGroup (Fin m) ℂ) :
      Rewrite (List.ofFn K) (List.ofFn (fun i => mixRow K (fun j => U.val i j)))
  | perm {C D C' D' : Channel n} :
      C.Perm C' → D.Perm D' → Rewrite C D → Rewrite C' D'
  | termOrder {C D C' D' : Channel n} :
      List.Forall₂ List.Perm C C' → List.Forall₂ List.Perm D D' →
      Rewrite C D → Rewrite C' D'

/-- Theorem 4.3 for the current guarded six-rule relation. -/
theorem Rewrite.sound {n} {C D : Channel n} (h : Rewrite C D) :
    denote C = denote D := by
  induction h with
  | deterministic h => exact h.sound
  | unitaryMix K U =>
      simp only [denote_ofFn, denote_mixRow]
      exact (krausMap_unitary_mix (fun i => termsDenote (K i)) U).symm
  | perm hC hD _ ih =>
      exact (channelDenote_perm hC).symm.trans (ih.trans (channelDenote_perm hD))
  | termOrder hC hD _ ih =>
      exact (channelDenote_termOrder hC).symm.trans (ih.trans (channelDenote_termOrder hD))

/-- Lemma 4.2 for the current guarded six-rule relation. -/
theorem Rewrite.wellTyped {n} {C D : Channel n} (_h : Rewrite C D) :
    WellTyped (toChannel D).erase n := ChannelIR.Channel.erase_wellTyped _

abbrev Rewrites {n} := Relation.ReflTransGen (@Rewrite n)

theorem Rewrites.sound {n} {C D : Channel n} (h : Rewrites C D) : denote C = denote D := by
  induction h with
  | refl => rfl
  | tail _ h ih => exact ih.trans h.sound

theorem Rewrites.wellTyped {n} {C D : Channel n} (_h : Rewrites C D) :
    WellTyped (toChannel D).erase n := ChannelIR.Channel.erase_wellTyped _


/-- Explicit empty-branch padding is separate from the six rewrite schemes. -/
inductive PaddedRewrite {n} : Channel n → Channel n → Prop where
  | rewrite {C D : Channel n} : Rewrite C D → PaddedRewrite C D
  | pad (pre post : Channel n) :
      PaddedRewrite (pre ++ post) (pre ++ [] :: post)

abbrev PaddedRewrites {n} := Relation.ReflTransGen (@PaddedRewrite n)

theorem PaddedRewrite.sound {n} {C D : Channel n} (h : PaddedRewrite C D) :
    denote C = denote D := by
  cases h with
  | rewrite h => exact h.sound
  | pad pre post => exact (Step.sound (Step.zeroKraus pre post)).symm

theorem PaddedRewrites.sound {n} {C D : Channel n} (h : PaddedRewrites C D) :
    denote C = denote D := by
  induction h with
  | refl => rfl
  | tail _ h ih => exact ih.trans h.sound

theorem PaddedRewrite.wellTyped {n} {C D : Channel n} (_h : PaddedRewrite C D) :
    WellTyped (toChannel D).erase n := ChannelIR.Channel.erase_wellTyped _

theorem PaddedRewrites.wellTyped {n} {C D : Channel n} (_h : PaddedRewrites C D) :
    WellTyped (toChannel D).erase n := ChannelIR.Channel.erase_wellTyped _


end ChannelFormalization.ChannelIR.Guarded
