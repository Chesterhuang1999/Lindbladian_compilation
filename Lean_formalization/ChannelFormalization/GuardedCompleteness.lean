import ChannelFormalization.KrausCompleteness
import ChannelFormalization.GuardedRewrites
import ChannelFormalization.GuardedCollection

/-! Proposition 4.4 in the current guarded relation. The common basis has its
phases absorbed in the coefficients. Collection is expanded into actual
R-MergeTerm/R-ZeroTerm steps, followed by deletion of empty Kraus branches. -/
noncomputable section
set_option backward.isDefEq.respectTransparency false
open scoped BigOperators Matrix
namespace ChannelFormalization.ChannelIR.Guarded
open Matrix

theorem terms_Op_sum {n m} (f : Fin m → Op n) :
    (Op.sum f).terms = (List.ofFn (fun i => (f i).terms)).flatten := by
  have h (l : List (Op n)) :
      (l.foldr Op.add Op.zero).terms = (l.map Op.terms).flatten := by
    induction l with
    | nil => rfl
    | cons a l ih => simp [Op.terms, ih]
  simpa [Op.sum, List.map_ofFn, Function.comp_def] using h (List.ofFn f)

theorem mixRow_canonical_terms {n d m} (P : Fin d → Primitive n)
    (A : Matrix (Fin m) (Fin d) ℂ) (a : Fin m → ℂ) :
    mixRow (fun j => (canonicalBasisOp P (A j)).terms) a =
      ((BasisSum.sum (fun j => .scale (a j) (BasisSum.canonical (A j)))).toOp P).terms := by
  rw [BasisSum.toOp_sum, terms_Op_sum]
  rfl

theorem denote_canonical_terms_family {n d m} (P : Fin d → Primitive n)
    (A : Matrix (Fin m) (Fin d) ℂ) :
    denote (List.ofFn (fun i => (canonicalBasisOp P (A i)).terms)) =
      MatrixMap.of_kraus (krausOfCoefficients (fun j => (P j).denote) A)
        (krausOfCoefficients (fun j => (P j).denote) A) := by
  rw [denote_ofFn]
  simp only [Op.denote_terms, canonicalBasisOp_denote]
  rfl

theorem canonical_terms_family_pad {n d m r} (hrm : r ≤ m)
    (P : Fin d → Primitive n) (B : Matrix (Fin r) (Fin d) ℂ) :
    List.ofFn (fun i : Fin m => (canonicalBasisOp P (padCoefficients B i)).terms) =
      List.ofFn (fun i : Fin r => (canonicalBasisOp P (B i)).terms) ++
        List.replicate (m - r) [] := by
  have h := congrArg (List.map Op.terms) (family_padCoefficients hrm P B)
  simpa only [ofFamily, List.map_ofFn, List.map_append, List.map_replicate,
    Op.terms, Function.comp_def] using h

theorem rewrites_remove_empty_tail {n} (C : Channel n) (k : ℕ) :
    Rewrites (C ++ List.replicate k []) C := by
  induction k with
  | zero => simpa using (Relation.ReflTransGen.refl : Rewrites C C)
  | succ k ih =>
      have h : Rewrite (C ++ [] :: List.replicate k [])
          (C ++ List.replicate k []) := .deterministic (.zeroKraus C _)
      simpa only [List.replicate_succ] using (Relation.ReflTransGen.single h).trans ih

/-- Proposition 4.4 for the current six-rule relation. The actual path consists
of a unitary mixing, finitely many R-MergeTerm/R-ZeroTerm steps and empty-branch
deletions. No atomic coefficient-collection rule is added to that relation. -/
theorem proposition_4_4_guarded {n d m r}
    (P : Fin d → Primitive n) (hP : LinearIndependent ℂ (fun j => (P j).denote))
    (A : Matrix (Fin m) (Fin d) ℂ) (B : Matrix (Fin r) (Fin d) ℂ)
    (hrm : r ≤ m)
    (h : denote (List.ofFn (fun i => canonicalTerms P (A i))) =
      denote (List.ofFn (fun i => canonicalTerms P (B i)))) :
    Rewrites (List.ofFn (fun i => canonicalTerms P (A i)))
      (List.ofFn (fun i => canonicalTerms P (B i))) := by
  have heq :
      MatrixMap.of_kraus (krausOfCoefficients (fun j => (P j).denote) A)
          (krausOfCoefficients (fun j => (P j).denote) A) =
        MatrixMap.of_kraus (krausOfCoefficients (fun j => (P j).denote)
          (padCoefficients (m := m) B))
          (krausOfCoefficients (fun j => (P j).denote) (padCoefficients (m := m) B)) := by
    rw [krausOfCoefficients_pad, krausMap_padKraus hrm]
    simpa only [canonicalTerms, denote_canonical_terms_family] using h
  obtain ⟨U, hU⟩ := coefficient_unitary_freedom
    (fun j => (P j).denote) hP A (padCoefficients (m := m) B) heq
  let es : Fin m → BasisSum d := fun i =>
    BasisSum.sum (fun j => .scale (U.val i j) (BasisSum.canonical (A j)))
  have he (i : Fin m) : (es i).coefficients = padCoefficients (m := m) B i := by
    rw [hU]
    exact BasisSum.coefficients_mix A U.val i
  have hin : (fun i => ((es i).toOp P).terms) =
      (fun i => mixRow (fun j => canonicalTerms P (A j)) (fun j => U.val i j)) := by
    funext i
    exact (mixRow_canonical_terms P A (fun j => U.val i j)).symm
  have hmix : Rewrite (List.ofFn (fun i => canonicalTerms P (A i)))
      (List.ofFn (fun i => ((es i).toOp P).terms)) := by
    rw [hin]
    exact .unitaryMix _ U
  have hnorm := collect_family_after P es hmix
  simp only [he] at hnorm
  apply hnorm.trans
  change Rewrites (List.ofFn (fun i => (canonicalBasisOp P (padCoefficients B i)).terms))
    (List.ofFn (fun i => (canonicalBasisOp P (B i)).terms))
  rw [canonical_terms_family_pad hrm]
  exact rewrites_remove_empty_tail _ _

end ChannelFormalization.ChannelIR.Guarded
