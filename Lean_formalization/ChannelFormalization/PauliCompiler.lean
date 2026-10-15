import ChannelFormalization.EndToEnd
import ChannelFormalization.PauliPlacement

/-! The paper-specific Pauli compiler: binary residuals, concrete phase
reconstruction, monotone controlled gate products, and channel-LCU. -/
noncomputable section
set_option linter.unusedSectionVars false
open scoped BigOperators Matrix Matrix.Norms.L2Operator
namespace ChannelFormalization.PauliCompiler
open Matrix LCU InnerLCU Placement PauliPlacement ChannelIR EndToEnd ChannelLCU
variable {T : Type*} [Fintype T] [DecidableEq T] [PartialOrder T] [DecidableLE T]
  {n : ℕ}

def target (β : T → ℂ) (p : T → PauliLabel n) (t : T) : Factor n where
  phase := Complex.exp ((β t).arg * Complex.I)
  phase_norm := Complex.norm_exp_ofReal_mul_I _
  label := p t

def residualFactors (order : Placement.LinearExtension T) (active : Finset T) (β : T → ℂ) (p : T → PauliLabel n) :
    T → Factor n := by
  classical
  exact Classical.choose (exists_phase_reconstruction order (fun t => t ∈ active) (target β p))

theorem residualFactors_correct (order : Placement.LinearExtension T) (active : Finset T) (β : T → ℂ)
    (p : T → PauliLabel n) (t : T) (ht : t ∈ active) :
    orderedAction order.order (fun a => (residualFactors order active β p a).unitary) t =
      (target β p t).unitary := by
  classical
  exact (Classical.choose_spec
    (exists_phase_reconstruction order (fun t => t ∈ active) (target β p))).1 t ht

/-- Tensoring with a trivial register is a multiplicative map on actual unitaries. -/
def trivialUnitaryHom {S : Type*} [Fintype S] [DecidableEq S] :
    Matrix.unitaryGroup S ℂ →* Matrix.unitaryGroup (Unit × S) ℂ where
  toFun U := ⟨trivialAncilla U.val, Matrix.mem_unitaryGroup_iff'.mpr
    (trivialAncilla_unitary _ (Matrix.mem_unitaryGroup_iff'.mp U.property))⟩
  map_one' := by
    apply Subtype.ext
    exact Matrix.submatrix_one_equiv _
  map_mul' U V := by
    apply Subtype.ext
    exact (Matrix.submatrix_mul_equiv U.val V.val (Equiv.uniqueProd S Unit)
      (Equiv.uniqueProd S Unit) (Equiv.uniqueProd S Unit)).symm

theorem map_orderedAction {G H : Type*} [Group G] [Group H]
    (f : G →* H) (l : List T) (U : T → G) (t : T) :
    f (orderedAction l U t) = orderedAction l (fun a => f (U a)) t := by
  simp only [orderedAction, map_list_prod, List.map_reverse, List.map_map, Function.comp_def]
  congr 2
  apply List.map_congr_left
  intro a _
  split_ifs <;> simp

def liftedFactors (order : Placement.LinearExtension T) (active : Finset T) (β : T → ℂ) (p : T → PauliLabel n) :
    T → Matrix.unitaryGroup (Unit × QState n) ℂ :=
  fun t => trivialUnitaryHom (residualFactors order active β p t).unitary

def compiledInner (order : Placement.LinearExtension T) (active : Finset T) (L R : Matrix T T ℂ)
    (β : T → ℂ) (p : T → PauliLabel n) :
    Matrix ((T × Unit) × QState n) ((T × Unit) × QState n) ℂ :=
  ((liftRegister L).conjTranspose *
    optimizedSelect order.order (liftedFactors order active β p) * liftRegister R).submatrix
      (Equiv.prodAssoc T Unit (QState n)) (Equiv.prodAssoc T Unit (QState n))

theorem compiledInner_blockEncoding
    (order : Placement.LinearExtension T) (active : Finset T) (t₀ : T) (β : T → ℂ) (p : T → PauliLabel n)
    (L R : Matrix T T ℂ)
    (hL : L.conjTranspose * L = 1) (hR : R.conjTranspose * R = 1)
    (hα : 0 < normalizer β)
    (hsupport : ∀ t, β t ≠ 0 → t ∈ active)
    (hprep : ∀ t, (normalizer β : ℂ) * (star (L t t₀) * R t t₀) = (‖β t‖ : ℂ)) :
    BlockEncoding (t₀, ()) (normalizer β) 0 (∑ t, β t • labelMatrix (p t))
      (compiledInner order active L R β p) := by
  let V := fun t => (orderedAction order.order (liftedFactors order active β p) t).val
  have hv : ∀ t, BlockEncoding () 1 0 (block () (V t)) (V t) := by
    intro t
    refine ⟨by norm_num, le_rfl,
      Matrix.mem_unitaryGroup_iff'.mp (orderedAction order.order (liftedFactors order active β p) t).property, ?_⟩
    simp
  have hy : ∑ t, ‖(‖β t‖ : ℂ)‖ ≤ normalizer β := by simp [normalizer]
  have hc : ∑ t, ‖(normalizer β : ℂ) * (star (L t t₀) * R t t₀) -
      (‖β t‖ : ℂ)‖ ≤ 0 := by
    simp only [hprep, sub_self, norm_zero, Finset.sum_const_zero, le_refl]
  have hs : ∑ t, (‖β t‖ : ℂ) • block () (V t) = ∑ t, β t • labelMatrix (p t) := by
    apply Finset.sum_congr rfl
    intro t _
    by_cases ht : β t = 0
    · simp [ht]
    · have hf := residualFactors_correct order active β p t (hsupport t ht)
      change (‖β t‖ : ℂ) • block ()
        (orderedAction order.order (fun a => trivialUnitaryHom (residualFactors order active β p a).unitary) t).val = _
      rw [← map_orderedAction, hf]
      exact coefficient_phase (β t) (labelMatrix (p t))
  have h := theorem_2_2 t₀ () L R V (fun t => block () (V t))
    (fun t => (‖β t‖ : ℂ)) 1 (normalizer β) 0 0
    (by norm_num) hα le_rfl le_rfl hL hR hy hc hv
  simp only [one_mul, mul_zero, add_zero, hs] at h
  simpa only [compiledInner, optimizedSelect_eq_select, groupedCircuit, circuit, V] using h

/-- The paper's positive square-root amplitudes satisfy the general left/right
preparation interface when the same PREPARE is used on both sides. -/
theorem sqrt_prepare_overlap (α : ℝ) (hα : 0 < α) (z : ℂ) :
    (α : ℂ) * (star (Real.sqrt (‖z‖ / α) : ℂ) *
      (Real.sqrt (‖z‖ / α) : ℂ)) = (‖z‖ : ℂ) := by
  rw [Complex.star_def, Complex.conj_ofReal, ← Complex.ofReal_mul,
    Real.mul_self_sqrt (div_nonneg (norm_nonneg _) hα.le), ← Complex.ofReal_mul]
  congr 1
  exact mul_div_cancel₀ _ (ne_of_gt hα)

/-- Exact lowering with the shared PREPARE and explicit amplitudes used in the
paper. Zero coefficients force zero individual preparation amplitudes. -/
theorem compiledInner_sharedPrepare
    (order : Placement.LinearExtension T) (active : Finset T) (t₀ : T)
    (β : T → ℂ) (p : T → PauliLabel n) (P : Matrix T T ℂ)
    (hP : P.conjTranspose * P = 1) (hα : 0 < normalizer β)
    (hsupport : ∀ t, β t ≠ 0 → t ∈ active)
    (hprep : ∀ t, P t t₀ = (Real.sqrt (‖β t‖ / normalizer β) : ℂ)) :
    BlockEncoding (t₀, ()) (normalizer β) 0 (∑ t, β t • labelMatrix (p t))
      (compiledInner order active P P β p) := by
  apply compiledInner_blockEncoding order active t₀ β p P P hP hP hα hsupport
  intro t
  rw [hprep t]
  exact sqrt_prepare_overlap _ hα _

/-- Placing a Pauli label at an unused address gives the identity Pauli matrix.
On assigned addresses it keeps the original coefficient--operator pairing. -/
theorem labelMatrix_addressTable {J : Type*} [Fintype J]
    (e : J ↪ T) (p : J → PauliLabel n) (t : T) :
    labelMatrix (addressTable e p t) = targetTable e (fun j => labelMatrix (p j)) t := by
  by_cases ht : t ∈ Set.range e
  · obtain ⟨j, rfl⟩ := ht
    simp
  · rw [addressTable_unused e p t ht,
      targetTable_unused e (fun j => labelMatrix (p j)) t ht, labelMatrix_zero]

/-- Arbitrary zero-coefficient padding labels leave the original Pauli sum
unchanged; only the label at each original term's assigned address matters. -/
theorem sum_placed_paulis {J : Type*} [Fintype J]
    (e : J ↪ T) (β : J → ℂ) (p : J → PauliLabel n) (q : T → PauliLabel n)
    (hq : ∀ j, q (e j) = p j) :
    ∑ t, addressTable e β t • labelMatrix (q t) = ∑ j, β j • labelMatrix (p j) := by
  have hs (t : T) : addressTable e β t • labelMatrix (q t) =
      addressTable e (fun j => β j • labelMatrix (p j)) t := by
    by_cases ht : t ∈ Set.range e
    · obtain ⟨j, rfl⟩ := ht
      simp [hq j]
    · rw [addressTable_unused e β t ht,
        addressTable_unused e (fun j => β j • labelMatrix (p j)) t ht, zero_smul]
  simp_rw [hs]
  exact sum_addressTable e _

/-- Proposition 5.1. Every injective placement, and hence every valid placement
returned by the search, preserves the original Pauli sum and its normalizer.
The circuit is the actual monotone controlled-gate product with reconstructed
phases. It uses one shared PREPARE with the relabeled square-root amplitudes;
all addresses outside the original placement have zero amplitude. The active
set may additionally contain arbitrary zero-coefficient padding labels. -/
theorem proposition_5_1 {J : Type*} [Fintype J]
    (e : J ↪ T) (order : Placement.LinearExtension T) (t₀ : T)
    (β : J → ℂ) (p : J → PauliLabel n) (active : Finset T) (q : T → PauliLabel n)
    (hactive : ∀ j, e j ∈ active) (hq : ∀ j, q (e j) = p j)
    (P : Matrix T T ℂ)
    (hP : P.conjTranspose * P = 1) (hα : 0 < normalizer β)
    (hprep : ∀ t, P t t₀ =
      (Real.sqrt (‖addressTable e β t‖ / normalizer β) : ℂ)) :
    normalizer (addressTable e β) = normalizer β ∧
    (∀ t, t ∉ Set.range e → P t t₀ = 0) ∧
    BlockEncoding (t₀, ()) (normalizer β) 0 (∑ j, β j • labelMatrix (p j))
      (compiledInner order active P P (addressTable e β) q) := by
  have hn : normalizer (addressTable e β) = normalizer β := sum_norm_addressTable e β
  refine ⟨hn, ?_, ?_⟩
  · intro t ht
    rw [hprep t, addressTable_unused e β t ht]
    simp
  · have hs : ∀ t, addressTable e β t ≠ 0 → t ∈ active := by
      intro t ht
      by_cases hr : t ∈ Set.range e
      · obtain ⟨j, rfl⟩ := hr
        exact hactive j
      · exact (ht (addressTable_unused e β t hr)).elim
    have h := compiledInner_sharedPrepare order active t₀
      (addressTable e β) q P hP (by rwa [hn]) hs
      (by simpa only [hn] using hprep)
    rwa [hn, sum_placed_paulis e β p q hq] at h

/-- Theorem 6.2 specialized to the actual n-qubit Pauli matrices and the
binary-residual, phase-reconstructed implementation of Section 5. This general
interface also allows distinct left and right preparations; the paper's shared
PREPARE and explicit term placement appear in `theorem_6_2_withPlacement`. -/
theorem theorem_6_2 {O : Type*} [Fintype O] [DecidableEq O]
    (C D : Channel n) (hrewrite : PaddedRewrites C D)
    (E : MatrixMap (QState n) (QState n) ℂ) (εfront : ℝ) (hεfront : 0 ≤ εfront)
    (hfront : diamondNorm (C.denote - E) ≤ ENNReal.ofReal εfront)
    (order : Placement.LinearExtension T) (t₀ : T)
    (β : Fin D.length → T → ℂ) (p : Fin D.length → T → PauliLabel n)
    (active : Fin D.length → Finset T)
    (hsupport : ∀ j t, β j t ≠ 0 → t ∈ active j)
    (hrep : ∀ j, D.krausFamily j = ∑ t, β j t • labelMatrix (p j t))
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
    let W := embeddedCircuit e P (fun j => compiledInner order (active j) (L j) (R j) (β j) (p j))
    let I_succ := successfulMap o₀ (t₀, ()) W
    WellTyped D.erase n ∧ D.denote = C.denote ∧
    W.conjTranspose * W = 1 ∧
    I_succ = (Λ : ℂ)⁻¹ • C.denote ∧
    diamondNorm ((Λ : ℂ) • I_succ - E) ≤ ENNReal.ofReal εfront ∧
    ∀ ρ, successProbability o₀ (t₀, ()) W ρ = (C.denote ρ).trace.re / Λ := by
  apply compiler_correct_of_blocks C D hrewrite E εfront hεfront hfront (t₀, ())
    (fun j => normalizer (β j)) _ ?_ e o₀ P hP hΛ houterPrep
  intro j
  rw [hrep j]
  exact compiledInner_blockEncoding order (active j) t₀ (β j) (p j) (L j) (R j) (hL j) (hR j)
    (normalizer_pos (β j) (fun t => labelMatrix (p j t))
      (by rw [← hrep j]; exact hnonzero j)) (hsupport j) (hinnerPrep j)

/-- Internal semantic composition with explicit placement. Public entry points
derive `hdenote` from proved rewrite soundness; inner correctness is exactly
Proposition 5.1 with shared PREPARE and zero padding amplitudes. -/
theorem theorem_6_2_withPlacement_of_denote {O : Type*} [Fintype O] [DecidableEq O]
    (C D : Channel n) (hdenote : D.denote = C.denote)
    (E : MatrixMap (QState n) (QState n) ℂ) (εfront : ℝ) (hεfront : 0 ≤ εfront)
    (hfront : diamondNorm (C.denote - E) ≤ ENNReal.ofReal εfront)
    (J : Fin D.length → Type*) [∀ j, Fintype (J j)]
    (placement : ∀ j, J j ↪ T) (order : Placement.LinearExtension T) (t₀ : T)
    (β : ∀ j, J j → ℂ) (p : ∀ j, J j → PauliLabel n)
    (active : Fin D.length → Finset T) (q : Fin D.length → T → PauliLabel n)
    (hactive : ∀ j t, placement j t ∈ active j)
    (hq : ∀ j t, q j (placement j t) = p j t)
    (hrep : ∀ j, D.krausFamily j = ∑ t, β j t • labelMatrix (p j t))
    (hnonzero : ∀ j, D.krausFamily j ≠ 0)
    (P : Fin D.length → Matrix T T ℂ)
    (hP : ∀ j, (P j).conjTranspose * P j = 1)
    (hinnerPrep : ∀ j t, P j t t₀ =
      (Real.sqrt (‖addressTable (placement j) (β j) t‖ / normalizer (β j)) : ℂ))
    (e : Fin D.length ↪ O) (o₀ : O) (Q : Matrix O O ℂ)
    (hQ : Q.conjTranspose * Q = 1)
    (hΛ : 0 < ∑ j, normalizer (β j) ^ 2)
    (houterPrep : ∀ o, Q o o₀ =
      ((addressTable e (fun j => normalizer (β j)) o /
        Real.sqrt (∑ j, normalizer (β j) ^ 2) : ℝ) : ℂ)) :
    let Λ := ∑ j, normalizer (β j) ^ 2
    let W := embeddedCircuit e Q (fun j =>
      compiledInner order (active j) (P j) (P j)
        (addressTable (placement j) (β j)) (q j))
    let I_succ := successfulMap o₀ (t₀, ()) W
    WellTyped D.erase n ∧ D.denote = C.denote ∧
    W.conjTranspose * W = 1 ∧
    I_succ = (Λ : ℂ)⁻¹ • C.denote ∧
    diamondNorm ((Λ : ℂ) • I_succ - E) ≤ ENNReal.ofReal εfront ∧
    ∀ ρ, successProbability o₀ (t₀, ()) W ρ = (C.denote ρ).trace.re / Λ := by
  classical
  apply compiler_correct_of_blocks_of_denote C D hdenote E εfront hεfront hfront (t₀, ())
    (fun j => normalizer (β j)) _ ?_ e o₀ Q hQ hΛ houterPrep
  intro j
  rw [hrep j]
  exact (proposition_5_1 (placement j) order t₀ (β j) (p j)
    (active j) (q j) (hactive j) (hq j) (P j) (hP j)
    (normalizer_pos (β j) (fun t => labelMatrix (p j t))
      (by rw [← hrep j]; exact hnonzero j)) (hinnerPrep j)).2.2


/-- Theorem 6.2 with explicit placement for the original broad rewrite relation.
`GuardedCompiler.theorem_6_2_guarded` accepts the current paper's complete guarded
rewrite path directly. -/
theorem theorem_6_2_withPlacement {O : Type*} [Fintype O] [DecidableEq O]
    (C D : Channel n) (hrewrite : PaddedRewrites C D)
    (E : MatrixMap (QState n) (QState n) ℂ) (εfront : ℝ) (hεfront : 0 ≤ εfront)
    (hfront : diamondNorm (C.denote - E) ≤ ENNReal.ofReal εfront)
    (J : Fin D.length → Type*) [∀ j, Fintype (J j)]
    (placement : ∀ j, J j ↪ T) (order : Placement.LinearExtension T) (t₀ : T)
    (β : ∀ j, J j → ℂ) (p : ∀ j, J j → PauliLabel n)
    (active : Fin D.length → Finset T) (q : Fin D.length → T → PauliLabel n)
    (hactive : ∀ j t, placement j t ∈ active j)
    (hq : ∀ j t, q j (placement j t) = p j t)
    (hrep : ∀ j, D.krausFamily j = ∑ t, β j t • labelMatrix (p j t))
    (hnonzero : ∀ j, D.krausFamily j ≠ 0)
    (P : Fin D.length → Matrix T T ℂ)
    (hP : ∀ j, (P j).conjTranspose * P j = 1)
    (hinnerPrep : ∀ j t, P j t t₀ =
      (Real.sqrt (‖addressTable (placement j) (β j) t‖ / normalizer (β j)) : ℂ))
    (e : Fin D.length ↪ O) (o₀ : O) (Q : Matrix O O ℂ)
    (hQ : Q.conjTranspose * Q = 1)
    (hΛ : 0 < ∑ j, normalizer (β j) ^ 2)
    (houterPrep : ∀ o, Q o o₀ =
      ((addressTable e (fun j => normalizer (β j)) o /
        Real.sqrt (∑ j, normalizer (β j) ^ 2) : ℝ) : ℂ)) :
    let Λ := ∑ j, normalizer (β j) ^ 2
    let W := embeddedCircuit e Q (fun j =>
      compiledInner order (active j) (P j) (P j)
        (addressTable (placement j) (β j)) (q j))
    let I_succ := successfulMap o₀ (t₀, ()) W
    WellTyped D.erase n ∧ D.denote = C.denote ∧
    W.conjTranspose * W = 1 ∧
    I_succ = (Λ : ℂ)⁻¹ • C.denote ∧
    diamondNorm ((Λ : ℂ) • I_succ - E) ≤ ENNReal.ofReal εfront ∧
    ∀ ρ, successProbability o₀ (t₀, ()) W ρ = (C.denote ρ).trace.re / Λ := by
  exact theorem_6_2_withPlacement_of_denote C D (paddedRewrites_sound hrewrite).symm
    E εfront hεfront hfront J placement order t₀ β p active q hactive hq hrep hnonzero
    P hP hinnerPrep e o₀ Q hQ hΛ houterPrep

end ChannelFormalization.PauliCompiler
