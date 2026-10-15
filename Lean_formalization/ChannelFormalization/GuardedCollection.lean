import ChannelFormalization.GuardedRewrites
import ChannelFormalization.KrausCompleteness

/-! Expansion of symbolic finite-sum collection into the actual merge-term and
zero-term rules. Auxiliary term permutations are bookkeeping only; the final
channel lemma absorbs them into genuine preceding rewrite steps. -/
noncomputable section
open scoped BigOperators Matrix
open Matrix
namespace ChannelFormalization.ChannelIR.Guarded

inductive TermMove {n} : Kraus n → Kraus n → Prop where
  | step {K L} : TermStep K L → TermMove K L
  | perm {K L} : K.Perm L → TermMove K L

abbrev TermMoves {n} := Relation.ReflTransGen (@TermMove n)

theorem TermStep.append {n} {K L : Kraus n} (h : TermStep K L)
    (pre post : Kraus n) : TermStep (pre ++ K ++ post) (pre ++ L ++ post) := by
  induction h with
  | merge a b c d p => simpa [List.append_assoc] using TermStep.merge (pre ++ a) (b ++ post) c d p
  | zero a b p => simpa [List.append_assoc] using TermStep.zero (pre ++ a) (b ++ post) p
  | perm hK hL _ ih =>
      exact .perm ((hK.append_left pre).append_right post)
        ((hL.append_left pre).append_right post) ih

theorem TermMove.append {n} {K L : Kraus n} (h : TermMove K L)
    (pre post : Kraus n) : TermMove (pre ++ K ++ post) (pre ++ L ++ post) := by
  cases h with
  | step h => exact .step (h.append pre post)
  | perm h => exact .perm ((h.append_left pre).append_right post)

theorem TermMoves.append {n} {K L : Kraus n} (h : TermMoves K L)
    (pre post : Kraus n) : TermMoves (pre ++ K ++ post) (pre ++ L ++ post) := by
  induction h with
  | refl => exact .refl
  | tail _ h ih => exact ih.tail (h.append pre post)

theorem TermMoves.cons {n} {K L : Kraus n} (h : TermMoves K L) (t) :
    TermMoves (t :: K) (t :: L) := by
  simpa using h.append [t] []

def oneTerm {n} (p : Primitive n) (a : ℂ) : Kraus n :=
  if a = 0 then [] else [(a,p)]

theorem clean_single {n} (p : Primitive n) (a : ℂ) :
    TermMoves [(a,p)] (oneTerm p a) := by
  by_cases h : a = 0
  · subst a
    simpa [oneTerm] using
      (Relation.ReflTransGen.single (.step (TermStep.zero [] [] p)) : TermMoves [(0,p)] [])
  · simpa [oneTerm, h] using (Relation.ReflTransGen.refl : TermMoves [(a,p)] [(a,p)])

theorem merge_nonempty {n} (p : Primitive n) (a : ℂ) (as : List ℂ) :
    TermMoves ((a :: as).map (fun c => (c,p))) [(a + as.sum,p)] := by
  induction as generalizing a with
  | nil => simp only [List.map_cons, List.map_nil, List.sum_nil, add_zero]; exact .refl
  | cons b bs ih =>
      have h := (ih b).cons (a,p)
      have hm := Relation.ReflTransGen.single
        (TermMove.step (TermStep.merge [] [] a (b + bs.sum) p))
      exact h.trans hm

theorem collect_same {n} (p : Primitive n) (as : List ℂ) :
    TermMoves (as.map (fun c => (c,p))) (oneTerm p as.sum) := by
  cases as with
  | nil => simp only [List.map_nil, List.sum_nil, oneTerm, ite_true]; exact .refl
  | cons a as => exact (merge_nonempty p a as).trans (clean_single p (a + as.sum))

abbrev IndexedTerms (d : ℕ) := List (ℂ × Fin d)

def encode {n d} (P : Fin d → Primitive n) (ts : IndexedTerms d) : Kraus n :=
  ts.map (fun t => (t.1, P t.2))

def coefficients {d} (ts : IndexedTerms d) (j : Fin d) : ℂ :=
  ((ts.filter (fun t => t.2 = j)).map Prod.fst).sum

def canonicalListTerms {n d} (P : Fin d → Primitive n) (c : Fin d → ℂ) :
    List (Fin d) → Kraus n
  | [] => []
  | j :: js => oneTerm (P j) (c j) ++ canonicalListTerms P c js

def canonicalTerms {n d} (P : Fin d → Primitive n) (c : Fin d → ℂ) : Kraus n :=
  (canonicalBasisOp P c).terms

theorem canonicalListTerms_eq {n d} (P : Fin d → Primitive n) (c : Fin d → ℂ)
    (js : List (Fin d)) :
    canonicalListTerms P c js = ((BasisSum.canonicalList c js).toOp P).terms := by
  induction js with
  | nil => rfl
  | cons j js ih =>
      by_cases h : c j = 0 <;>
        simp [canonicalListTerms, oneTerm, BasisSum.canonicalList, BasisSum.toOp,
          Op.terms, h, ih]

theorem canonicalListTerms_congr {n d} (P : Fin d → Primitive n) {c e : Fin d → ℂ}
    (js : List (Fin d)) (h : ∀ j ∈ js, c j = e j) :
    canonicalListTerms P c js = canonicalListTerms P e js := by
  induction js with
  | nil => rfl
  | cons j js ih =>
      simp only [canonicalListTerms, h j (by simp), ih (fun k hk => h k (by simp [hk]))]

theorem coefficients_filter_ne {d} (ts : IndexedTerms d) (j k : Fin d) (h : k ≠ j) :
    coefficients (ts.filter (fun t => t.2 ≠ j)) k = coefficients ts k := by
  induction ts with
  | nil => simp [coefficients]
  | cons t ts ih =>
      by_cases ht : t.2 = j <;> by_cases hk : t.2 = k <;>
        simp_all [coefficients]

/-- Process one basis key at a time. Every collected coefficient is obtained by
actual pairwise merges; an exact zero is removed by an actual zero-term step. -/
theorem collect_keys {n d} (P : Fin d → Primitive n) (js : List (Fin d))
    (hn : js.Nodup) (ts : IndexedTerms d)
    (hs : ∀ t ∈ ts, t.2 ∈ js) :
    TermMoves (encode P ts) (canonicalListTerms P (coefficients ts) js) := by
  induction js generalizing ts with
  | nil =>
      have ht : ts = [] := by
        apply List.eq_nil_iff_forall_not_mem.mpr
        intro t ht
        exact List.not_mem_nil (hs t ht)
      subst ts
      exact .refl
  | cons j js ih =>
      let hit := ts.filter (fun t => t.2 = j)
      let rest := ts.filter (fun t => t.2 ≠ j)
      have hp : (encode P ts).Perm (encode P hit ++ encode P rest) := by
        simpa [encode, hit, rest] using
          (List.filter_append_perm (fun t : ℂ × Fin d => decide (t.2 = j)) ts).symm.map
            (fun t => (t.1, P t.2))
      have he : encode P hit = (hit.map Prod.fst).map (fun c => (c,P j)) := by
        unfold encode
        rw [List.map_map]
        apply List.map_congr_left
        intro t ht
        have hj : t.2 = j := (List.mem_filter.mp ht).2 |> of_decide_eq_true
        simp [hj]
      have hg : TermMoves (encode P hit) (oneTerm (P j) (coefficients ts j)) := by
        rw [he]
        exact collect_same (P j) (hit.map Prod.fst)
      have hsrest : ∀ t ∈ rest, t.2 ∈ js := by
        intro t ht
        have hm := List.mem_filter.mp ht
        have hmem := hs t hm.1
        have hne : t.2 ≠ j := of_decide_eq_true hm.2
        simpa [hne] using hmem
      have hr := ih hn.of_cons rest hsrest
      have hc : canonicalListTerms P (coefficients rest) js =
          canonicalListTerms P (coefficients ts) js := by
        apply canonicalListTerms_congr
        intro k hk
        exact coefficients_filter_ne ts j k (fun h => (List.nodup_cons.mp hn).1 (h ▸ hk))
      rw [hc] at hr
      have h₁ : TermMoves (encode P hit ++ encode P rest)
          (oneTerm (P j) (coefficients ts j) ++ encode P rest) := by
        simpa using hg.append [] (encode P rest)
      have h₂ : TermMoves (oneTerm (P j) (coefficients ts j) ++ encode P rest)
          (canonicalListTerms P (coefficients ts) (j :: js)) := by
        simpa [canonicalListTerms] using hr.append (oneTerm (P j) (coefficients ts j)) []
      exact (Relation.ReflTransGen.single (.perm hp)).trans (h₁.trans h₂)

/-- The actual flattened symbolic terms, before primitive labels are substituted. -/
def labelTerms {d} : BasisSum d → IndexedTerms d
  | .zero => []
  | .atom j => [(1,j)]
  | .add a b => labelTerms a ++ labelTerms b
  | .scale c a => (labelTerms a).map (fun t => (c * t.1, t.2))

theorem encode_labelTerms {n d} (P : Fin d → Primitive n) (e : BasisSum d) :
    encode P (labelTerms e) = (e.toOp P).terms := by
  induction e with
  | zero => rfl
  | atom j => rfl
  | add a b ha hb => simpa [labelTerms, encode, BasisSum.toOp, Op.terms] using congrArg₂ (· ++ ·) ha hb
  | scale c a ha =>
      simpa [encode, labelTerms, BasisSum.toOp, Op.terms, List.map_map, Function.comp_def] using
        congrArg (List.map (fun t : ℂ × Primitive n => (c * t.1, t.2))) ha

theorem coefficients_append {d} (ts us : IndexedTerms d) :
    coefficients (ts ++ us) = coefficients ts + coefficients us := by
  funext j
  simp [coefficients]

theorem coefficients_scale {d} (c : ℂ) (ts : IndexedTerms d) :
    coefficients (ts.map (fun t => (c * t.1, t.2))) = c • coefficients ts := by
  funext j
  induction ts with
  | nil => simp [coefficients]
  | cons t ts ih =>
      by_cases h : t.2 = j <;> simp_all [coefficients, mul_add]

theorem coefficients_labelTerms {d} (e : BasisSum d) :
    coefficients (labelTerms e) = e.coefficients := by
  induction e with
  | zero => ext j; simp [labelTerms, coefficients, BasisSum.coefficients, BasisSum.eval]
  | atom k =>
      ext j
      by_cases h : k = j <;>
        simp [labelTerms, coefficients, BasisSum.coefficients, BasisSum.eval,
          h, Ne.symm]
  | add a b ha hb =>
      rw [labelTerms, coefficients_append, ha, hb]
      rfl
  | scale c a ha =>
      rw [labelTerms, coefficients_scale, ha]
      rfl

/-- A fully expanded proof of symbolic collection: only merge-term,
zero-term and harmless term-order changes are used. -/
theorem collect_basisSum {n d} (P : Fin d → Primitive n) (e : BasisSum d) :
    TermMoves ((e.toOp P).terms) (canonicalTerms P e.coefficients) := by
  have h := collect_keys P (List.ofFn id) (List.nodup_ofFn.mpr Function.injective_id)
    (labelTerms e) (fun t _ => List.mem_ofFn.mpr ⟨t.2, rfl⟩)
  rw [encode_labelTerms, coefficients_labelTerms, canonicalListTerms_eq] at h
  exact h


/-- A term-order change can be absorbed into the last genuine channel rewrite.
This is why permutations are not added as independent rewrite rules. -/
theorem transGen_termOrder {n} {C D E : Channel n}
    (h : Relation.TransGen Rewrite C D) (hp : List.Forall₂ List.Perm D E) :
    Relation.TransGen Rewrite C E := by
  obtain ⟨B, hCB, hBD⟩ := Relation.TransGen.tail'_iff.mp h
  exact Relation.TransGen.tail' hCB (.termOrder (List.forall₂_refl B) hp hBD)

theorem termMoves_after {n} {C : Channel n} (pre post : Channel n)
    {K L : Kraus n} (h : Relation.TransGen Rewrite C (pre ++ K :: post))
    (hm : TermMoves K L) : Relation.TransGen Rewrite C (pre ++ L :: post) := by
  induction hm with
  | refl => exact h
  | @tail M L _ hm ih =>
      cases hm with
      | step hm => exact ih.tail (.deterministic (.term pre post hm))
      | perm hp =>
          exact transGen_termOrder ih
            (List.rel_append (List.forall₂_refl pre) (.cons hp (List.forall₂_refl post)))

theorem collect_list_after {n d} (P : Fin d → Primitive n)
    (es : List (BasisSum d)) (pre : Channel n) {C : Channel n}
    (h : Relation.TransGen Rewrite C (pre ++ es.map (fun e => (e.toOp P).terms))) :
    Relation.TransGen Rewrite C (pre ++ es.map (fun e => canonicalTerms P e.coefficients)) := by
  induction es generalizing pre with
  | nil => exact h
  | cons e es ih =>
      have hh := termMoves_after pre (es.map (fun e => (e.toOp P).terms)) h
        (collect_basisSum P e)
      have hnext : Relation.TransGen Rewrite C
          ((pre ++ [canonicalTerms P e.coefficients]) ++ es.map (fun e => (e.toOp P).terms)) := by
        simpa only [List.append_assoc, List.singleton_append] using hh
      simpa only [List.map_cons, List.append_assoc, List.singleton_append] using
        ih (pre ++ [canonicalTerms P e.coefficients]) hnext

/-- The supplied first rewrite is genuine (in Proposition 4.4 it is unitary
mixing). All following collection is expanded into real Figure 5 rules, with
term-order changes absorbed into those steps. No collection macro remains. -/
theorem collect_family_after {n d m} (P : Fin d → Primitive n)
    (es : Fin m → BasisSum d) {C : Channel n}
    (h : Rewrite C (List.ofFn (fun i => ((es i).toOp P).terms))) :
    Rewrites C (List.ofFn (fun i => canonicalTerms P (es i).coefficients)) := by
  have hh : Relation.TransGen Rewrite C
      ([] ++ (List.ofFn es).map (fun e => (e.toOp P).terms)) := by
    simpa only [List.nil_append, List.map_ofFn, Function.comp_def] using
      Relation.TransGen.single h
  simpa only [List.nil_append, List.map_ofFn, Function.comp_def] using
    (collect_list_after P (List.ofFn es) [] hh).to_reflTransGen


end ChannelFormalization.ChannelIR.Guarded
