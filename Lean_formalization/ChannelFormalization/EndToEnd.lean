import ChannelFormalization.ChannelIR
import ChannelFormalization.ChannelLCU
import ChannelFormalization.InnerLCU
import ChannelFormalization.OptimizedLCU
import ChannelFormalization.Placement
import ChannelFormalization.Diamond

/-! End-to-end composition of exact rewriting, inner LCU construction, and
outer channel-LCU, with an injective placement into a larger outer register.
The frontend diamond-error guarantee is the hypothesis of Theorem 6.2. -/
noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix Matrix.Norms.L2Operator
open Matrix
namespace ChannelFormalization.EndToEnd
open ChannelIR LCU ChannelLCU InnerLCU Placement OptimizedLCU

variable {I O A S : Type*} [Fintype I] [Fintype O] [Fintype A] [Fintype S]
  [DecidableEq I] [DecidableEq O] [DecidableEq A] [DecidableEq S]

theorem kraus_addressTable (e : I ↪ O) (K : I → Matrix S S ℂ) :
    MatrixMap.of_kraus (addressTable e K) (addressTable e K) =
      MatrixMap.of_kraus K K := by
  have hz : MatrixMap.conj (0 : Matrix S S ℂ) = 0 := by
    ext ρ x y
    simp [MatrixMap.conj_apply]
  rw [MatrixMap.of_kraus_eq_sum_conj, MatrixMap.of_kraus_eq_sum_conj]
  simp_rw [map_addressTable e K MatrixMap.conj hz]
  exact sum_addressTable e _

/-- The prepared outer oracle is completed with identities off the injective
image of the Kraus family. -/
def embeddedCircuit (e : I ↪ O) (P : Matrix O O ℂ)
    (U : I → Matrix (A × S) (A × S) ℂ) :
    Matrix (O × (A × S)) (O × (A × S)) ℂ :=
  outerCircuit P (targetTable e U)

theorem embedded_output (e : I ↪ O) (o₀ : O) (a₀ : A)
    (K : I → Matrix S S ℂ) (α : I → ℝ) (P : Matrix O O ℂ)
    (U : I → Matrix (A × S) (A × S) ℂ)
    (hblock : ∀ j, BlockEncoding a₀ (α j) 0 (K j) (U j))
    (hP : P.conjTranspose * P = 1) (hΛ : 0 < ∑ j, α j ^ 2)
    (hprep : ∀ o, P o o₀ =
      ((addressTable e α o / Real.sqrt (∑ j, α j ^ 2) : ℝ) : ℂ)) :
    (embeddedCircuit e P U).conjTranspose * embeddedCircuit e P U = 1 ∧
    ∀ ρ, successfulOutput o₀ a₀ (embeddedCircuit e P U) ρ =
      ((∑ j, α j ^ 2 : ℝ) : ℂ)⁻¹ • MatrixMap.of_kraus K K ρ := by
  refine ⟨outerCircuit_unitary P _ hP
    (targetTable_unitary e U (fun j => (hblock j).unitary)), ?_⟩
  intro ρ
  have hb : branch o₀ a₀ (embeddedCircuit e P U) = fun o =>
      (Real.sqrt (∑ j, α j ^ 2) : ℂ)⁻¹ • addressTable e K o := by
    funext o
    rw [embeddedCircuit, outer_branch, hprep]
    by_cases ho : o ∈ Set.range e
    · obtain ⟨j, rfl⟩ := ho
      rw [addressTable_at, addressTable_at, targetTable_at,
        Complex.ofReal_div, div_eq_mul_inv, mul_comm, ← smul_smul,
        exact_block a₀ (α j) (K j) (U j) (hblock j)]
    · rw [addressTable_unused e α o ho, addressTable_unused e K o ho]
      simp
  rw [successfulOutput_eq_kraus, hb, kraus_common_scale,
    sqrt_inverse_square _ hΛ, kraus_addressTable]

/-- The actual successful circuit operation, bundled as a linear map using
its independently proved Kraus expression. -/
def successfulMap (o₀ : O) (a₀ : A)
    (W : Matrix (O × (A × S)) (O × (A × S)) ℂ) : MatrixMap S S ℂ :=
  MatrixMap.of_kraus (branch o₀ a₀ W) (branch o₀ a₀ W)

theorem successfulMap_apply (o₀ : O) (a₀ : A)
    (W : Matrix (O × (A × S)) (O × (A × S)) ℂ) (ρ : Matrix S S ℂ) :
    successfulMap o₀ a₀ W ρ = successfulOutput o₀ a₀ W ρ :=
  (successfulOutput_eq_kraus o₀ a₀ W ρ).symm

section Compiler
variable {n : ℕ} {T : Type*} [Fintype T] [DecidableEq T]
  [PartialOrder T] [DecidableLE T]

/-- Internal semantic composition principle for independently proved exact inner
encodings. Public compiler entry points derive `hdenote` from their rewrite-path
certificates; concrete lowering theorems discharge `hblocks`. -/
theorem compiler_correct_of_blocks_of_denote (C D : Channel n) (hdenote : D.denote = C.denote)
    (E : MatrixMap (QState n) (QState n) ℂ) (εfront : ℝ) (_hεfront : 0 ≤ εfront)
    (hfront : diamondNorm (C.denote - E) ≤ ENNReal.ofReal εfront)
    (a₀ : A) (α : Fin D.length → ℝ)
    (U : Fin D.length → Matrix (A × QState n) (A × QState n) ℂ)
    (hblocks : ∀ j, BlockEncoding a₀ (α j) 0 (D.krausFamily j) (U j))
    (e : Fin D.length ↪ O) (o₀ : O) (P : Matrix O O ℂ)
    (hP : P.conjTranspose * P = 1) (hΛ : 0 < ∑ j, α j ^ 2)
    (hprep : ∀ o, P o o₀ = ((addressTable e α o / Real.sqrt (∑ j, α j ^ 2) : ℝ) : ℂ)) :
    let Λ := ∑ j, α j ^ 2
    let W := embeddedCircuit e P U
    let I_succ := successfulMap o₀ a₀ W
    WellTyped D.erase n ∧ D.denote = C.denote ∧
    W.conjTranspose * W = 1 ∧ I_succ = (Λ : ℂ)⁻¹ • C.denote ∧
    diamondNorm ((Λ : ℂ) • I_succ - E) ≤ ENNReal.ofReal εfront ∧
    ∀ ρ, successProbability o₀ a₀ W ρ = (C.denote ρ).trace.re / Λ := by
  dsimp only
  obtain ⟨hunit, hout⟩ := embedded_output e o₀ a₀ D.krausFamily α P U
    hblocks hP hΛ hprep
  have hmap : successfulMap o₀ a₀ (embeddedCircuit e P U) =
      ((∑ j, α j ^ 2 : ℝ) : ℂ)⁻¹ • C.denote := by
    ext ρ x y
    rw [successfulMap_apply, hout, ← D.denote_eq_of_kraus, hdenote]
    rfl
  refine ⟨D.erase_wellTyped, hdenote, hunit, hmap, ?_, ?_⟩
  · rw [hmap, smul_smul, mul_inv_cancel₀ (by exact_mod_cast ne_of_gt hΛ), one_smul]
    exact hfront
  · intro ρ
    unfold successProbability
    rw [hout, ← D.denote_eq_of_kraus, hdenote, Matrix.trace_smul,
      smul_eq_mul, ← Complex.ofReal_inv, Complex.mul_re]
    simp only [Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero]
    exact mul_comm _ _

/-- Composition principle for the original broad rewrite relation. The semantic
helper also supports the paper's guarded relation, without requiring a second
certificate in this older syntax. -/
theorem compiler_correct_of_blocks (C D : Channel n) (hrewrite : PaddedRewrites C D)
    (E : MatrixMap (QState n) (QState n) ℂ) (εfront : ℝ) (hεfront : 0 ≤ εfront)
    (hfront : diamondNorm (C.denote - E) ≤ ENNReal.ofReal εfront)
    (a₀ : A) (α : Fin D.length → ℝ)
    (U : Fin D.length → Matrix (A × QState n) (A × QState n) ℂ)
    (hblocks : ∀ j, BlockEncoding a₀ (α j) 0 (D.krausFamily j) (U j))
    (e : Fin D.length ↪ O) (o₀ : O) (P : Matrix O O ℂ)
    (hP : P.conjTranspose * P = 1) (hΛ : 0 < ∑ j, α j ^ 2)
    (hprep : ∀ o, P o o₀ = ((addressTable e α o / Real.sqrt (∑ j, α j ^ 2) : ℝ) : ℂ)) :
    let Λ := ∑ j, α j ^ 2
    let W := embeddedCircuit e P U
    let I_succ := successfulMap o₀ a₀ W
    WellTyped D.erase n ∧ D.denote = C.denote ∧
    W.conjTranspose * W = 1 ∧ I_succ = (Λ : ℂ)⁻¹ • C.denote ∧
    diamondNorm ((Λ : ℂ) • I_succ - E) ≤ ENNReal.ofReal εfront ∧
    ∀ ρ, successProbability o₀ a₀ W ρ = (C.denote ρ).trace.re / Λ := by
  exact compiler_correct_of_blocks_of_denote C D (paddedRewrites_sound hrewrite).symm
    E εfront hεfront hfront a₀ α U hblocks e o₀ P hP hΛ hprep

/-- Theorem 6.2. The final Kraus coefficient table is lowered by actual inner
PREPARE/phase-restored SELECT circuits; the outer circuit initializes, postselects,
and traces registers exactly as in Proposition 4.5. Any finite unitary dictionary
is allowed, hence in particular the paper's Pauli strings. -/
theorem theorem_6_2 (C D : Channel n) (hrewrite : PaddedRewrites C D)
    (E : MatrixMap (QState n) (QState n) ℂ) (εfront : ℝ) (_hεfront : 0 ≤ εfront)
    (hfront : diamondNorm (C.denote - E) ≤ ENNReal.ofReal εfront)
    (order : Placement.LinearExtension T) (t₀ : T) (β : Fin D.length → T → ℂ)
    (Q : Fin D.length → T → QMatrix n)
    (hQ : ∀ j t, (Q j t).conjTranspose * Q j t = 1)
    (hrep : ∀ j, D.krausFamily j = ∑ t, β j t • Q j t)
    (hnonzero : ∀ j, D.krausFamily j ≠ 0)
    (L R : Fin D.length → Matrix T T ℂ)
    (hL : ∀ j, (L j).conjTranspose * L j = 1)
    (hR : ∀ j, (R j).conjTranspose * R j = 1)
    (hinnerPrep : ∀ j t, (normalizer (β j) : ℂ) *
      (star (L j t t₀) * R j t t₀) = (‖β j t‖ : ℂ))
    (e : Fin D.length ↪ O) (o₀ : O) (P : Matrix O O ℂ)
    (hP : P.conjTranspose * P = 1)
    (hΛ : 0 < ∑ j, normalizer (β j) ^ 2)
    (houterPrep : ∀ o, P o o₀ =
      ((addressTable e (fun j => normalizer (β j)) o /
        Real.sqrt (∑ j, normalizer (β j) ^ 2) : ℝ) : ℂ)) :
    let Λ := ∑ j, normalizer (β j) ^ 2
    let W := embeddedCircuit e P (fun j => compiledInner order (L j) (R j) (β j) (Q j) (hQ j))
    let I_succ := successfulMap o₀ (t₀, ()) W
    WellTyped D.erase n ∧ D.denote = C.denote ∧
    W.conjTranspose * W = 1 ∧
    I_succ = (Λ : ℂ)⁻¹ • C.denote ∧
    diamondNorm ((Λ : ℂ) • I_succ - E) ≤ ENNReal.ofReal εfront ∧
    ∀ ρ, successProbability o₀ (t₀, ()) W ρ = (C.denote ρ).trace.re / Λ := by
  dsimp only
  have hdenote : D.denote = C.denote := (paddedRewrites_sound hrewrite).symm
  have hblocks : ∀ j, BlockEncoding (t₀, ()) (normalizer (β j)) 0
      (D.krausFamily j) (compiledInner order (L j) (R j) (β j) (Q j) (hQ j)) := by
    intro j
    rw [hrep j]
    exact compiledInner_blockEncoding order t₀ (β j) (Q j) (L j) (R j) (hQ j) (hL j) (hR j)
      (normalizer_pos (β j) (Q j) (by rw [← hrep j]; exact hnonzero j)) (hinnerPrep j)
  obtain ⟨hunit, hout⟩ := embedded_output e o₀ (t₀, ()) D.krausFamily
    (fun j => normalizer (β j)) P _ hblocks hP hΛ houterPrep
  have hmap : successfulMap o₀ (t₀, ())
      (embeddedCircuit e P (fun j => compiledInner order (L j) (R j) (β j) (Q j) (hQ j))) =
        ((∑ j, normalizer (β j) ^ 2 : ℝ) : ℂ)⁻¹ • C.denote := by
    ext ρ x y
    rw [successfulMap_apply, hout, ← D.denote_eq_of_kraus, hdenote]
    rfl
  refine ⟨D.erase_wellTyped, hdenote, hunit, hmap, ?_, ?_⟩
  · rw [hmap, smul_smul, mul_inv_cancel₀ (by exact_mod_cast ne_of_gt hΛ), one_smul]
    exact hfront
  · intro ρ
    unfold successProbability
    rw [hout, ← D.denote_eq_of_kraus, hdenote, Matrix.trace_smul,
      smul_eq_mul, ← Complex.ofReal_inv, Complex.mul_re]
    simp only [Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero]
    exact mul_comm _ _

end Compiler
end ChannelFormalization.EndToEnd
