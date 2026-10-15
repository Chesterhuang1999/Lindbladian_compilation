import ChannelFormalization.InnerLCU
import ChannelFormalization.ChannelLCU
import ChannelFormalization.Placement

/-! Lemma C.1 for the selected blocks of actual LCU circuits, including an
injective term-address relabeling and exact zero coefficients off its image. -/
noncomputable section
open scoped BigOperators Matrix Matrix.Norms.L2Operator
namespace ChannelFormalization.AddressPlacement
open Matrix LCU InnerLCU ChannelLCU Placement

theorem lemma_C_1 {J A S : Type*} [Fintype J] [Fintype A] [Fintype S]
    [DecidableEq J] [DecidableEq A] [DecidableEq S] [Nonempty S]
    (e : J ↪ A) (j₀ : J) (a₀ : A) (β : J → ℂ) (Q : J → Matrix S S ℂ)
    (hQ : ∀ j, (Q j).conjTranspose * Q j = 1)
    (hα : 0 < normalizer β)
    (L R : Matrix J J ℂ) (L' R' : Matrix A A ℂ)
    (hL : L.conjTranspose * L = 1) (hR : R.conjTranspose * R = 1)
    (hL' : L'.conjTranspose * L' = 1) (hR' : R'.conjTranspose * R' = 1)
    (hprep : ∀ j, (normalizer β : ℂ) * (star (L j j₀) * R j j₀) = (‖β j‖ : ℂ))
    (hprep' : ∀ a, (normalizer β : ℂ) * (star (L' a a₀) * R' a a₀) =
      (‖addressTable e β a‖ : ℂ)) :
    block (a₀, ()) (innerCircuit L' R' (addressTable e β) (targetTable e Q)) =
      block (j₀, ()) (innerCircuit L R β Q) := by
  have hn : normalizer (addressTable e β) = normalizer β := sum_norm_addressTable e β
  have hnew := inner_blockEncoding a₀ (addressTable e β) (targetTable e Q) L' R'
    (targetTable_unitary e Q hQ) hL' hR' (by rwa [hn]) (by simpa only [hn] using hprep')
  have hold := inner_blockEncoding j₀ β Q L R hQ hL hR hα hprep
  have h1 := exact_block (a₀, ()) _ _ _ hnew
  have h2 := exact_block (j₀, ()) _ _ _ hold
  rw [hn, sum_smul_addressTable] at h1
  apply (smul_right_injective (M := Matrix S S ℂ)
    (show (normalizer β : ℂ) ≠ 0 by exact_mod_cast ne_of_gt hα))
  exact h1.trans h2.symm

end ChannelFormalization.AddressPlacement
