import QuantumInfo.Channels.Unbundled
import Physlib.Relativity.PauliMatrices.Basic

/-! Smoke tests exercise actual quantum-information and Pauli-matrix theorems. -/
namespace ChannelFormalization.Smoke

 theorem kraus_completely_positive {ι A B : Type*}
    [Fintype ι] [Fintype A] [Fintype B] [DecidableEq A]
    (K : ι → Matrix B A ℂ) :
    (MatrixMap.of_kraus K K).IsCompletelyPositive :=
  MatrixMap.of_kraus_isCompletelyPositive K

 theorem pauli_unitary (μ : Fin 1 ⊕ Fin 3) :
    (PauliMatrix.pauliMatrix μ).conjTranspose * PauliMatrix.pauliMatrix μ = 1 := by
  rw [PauliMatrix.pauliMatrix_selfAdjoint, PauliMatrix.pauliMatrix_mul_self]

#print axioms kraus_completely_positive
#print axioms pauli_unitary
end ChannelFormalization.Smoke
