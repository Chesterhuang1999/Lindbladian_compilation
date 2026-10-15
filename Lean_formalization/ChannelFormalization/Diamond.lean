import QuantumInfo.Channels.Unbundled
import QuantumInfo.ForMathlib.MatrixNorm.TraceNorm
import Mathlib.Basic.ENNReal.Real

/-! The diamond norm as the supremum of the trace-norm induced norms over
all finite auxiliary registers. We use extended nonnegative reals so that the
definition does not rely on an implicit boundedness hypothesis for a real supremum.
A finite error bound has exactly the usual trace-norm meaning. -/
noncomputable section
open scoped MatrixMap
namespace ChannelFormalization
variable {S : Type*} [Fintype S] [DecidableEq S]

/-- Stabilized trace-norm supremum, including every finite auxiliary dimension. -/
def diamondNorm (Φ : MatrixMap S S ℂ) : ENNReal :=
  ⨆ k : ℕ, ⨆ X : Matrix (S × Fin k) (S × Fin k) ℂ,
    ⨆ (_hX : X.traceNorm ≤ 1),
      ENNReal.ofReal (((Φ ⊗ₖₘ MatrixMap.id (Fin k) ℂ) X).traceNorm)

/-- A finite diamond bound is equivalent to the expected bound on every
trace-norm unit ball, in every auxiliary dimension. -/
theorem diamondNorm_le_iff (Φ : MatrixMap S S ℂ) (ε : ℝ) (hε : 0 ≤ ε) :
    diamondNorm Φ ≤ ENNReal.ofReal ε ↔
      ∀ k (X : Matrix (S × Fin k) (S × Fin k) ℂ), X.traceNorm ≤ 1 →
        ((Φ ⊗ₖₘ MatrixMap.id (Fin k) ℂ) X).traceNorm ≤ ε := by
  simp only [diamondNorm, iSup_le_iff, ENNReal.ofReal_le_ofReal_iff hε]

end ChannelFormalization
