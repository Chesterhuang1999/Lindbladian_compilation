import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli
from abc import ABC, abstractmethod
from collections import defaultdict 
from copy import deepcopy
class OperatorAtom(ABC):
    def __init__(self, phase: complex = 1.0):
        self.phase = phase
    
    @abstractmethod 
    def bare_op(self):
        pass
    
    def eff_op(self):
        return self.phase * self.bare_op() # type: ignore 
    @abstractmethod
    def adjoint(self):
        pass
    
    @abstractmethod
    def multiply(self, other):
        pass    
    
    @abstractmethod
    def to_operator(self) -> Operator:
        pass

class PauliAtom(OperatorAtom):
    def __init__(self, expr, phase: complex = 1.0):
        super().__init__(phase)
        self.expr = str(expr)
        self.size = len(self.expr)
    def bare_op(self):
        return SparsePauliOp.from_list([(self.expr, 1.0)])
    def to_operator(self) -> Operator:
        return Operator(self.eff_op())
    def adjoint(self):
        return PauliAtom(self.expr, np.conj(self.phase))
    
    def multiply(self, other):
        if not isinstance(other, PauliAtom):
            raise ValueError("Can only multiply with another PauliAtom")
        p = Pauli(self.expr) @ Pauli(other.expr)
        new_phase = self.phase * other.phase * ((-1j)**p.phase)
        p.phase = 0
        return PauliAtom(p.to_label(), new_phase)

    def __repr__(self):
        return f"PauliAtom: {self.phase} * {self.expr}"
    
class MatrixAtom(OperatorAtom):
    def __init__(self, mat: np.ndarray, phase: complex = 1.0):
        self.mat = mat
        self.size = mat.shape[0]

    def adjoint(self):
        return MatrixAtom(self.mat.conj().T, np.conj(self.phase))
    
    def bare_op(self):
        return self.mat
    
    def multiply(self, other):
        if isinstance(other, PauliAtom):
            new_mat = self.mat @ other.bare_op().to_matrix()
            new_phase = self.phase * other.phase
            return MatrixAtom(new_mat, new_phase)
        elif isinstance(other, MatrixAtom):
            new_mat = self.mat @ other.mat
            new_phase = self.phase * other.phase
            return MatrixAtom(new_mat, new_phase)
        else:
            raise ValueError("Can only multiply with other OperatorAtom")
        
    def to_operator(self) -> Operator:
        return Operator(self.eff_op())
    
    
class Matrixsum:
    """
    Linear combination of OperatorAtom matrices.
    """

    def __init__(self, instances = None):
        self.instances = instances if instances is not None else []
        for inst, coeff in self.instances: 
            if not np.isreal(coeff) or coeff < 0: 
                phase = coeff / abs(coeff)
                inst.phase *= phase
                coeff = abs(coeff)  
        self.size = max([inst.size for inst, _ in self.instances]) if self.instances else 0
        self.length = len(self.instances)
        self.ctrl_size = int(np.ceil(np.log2(self.length))) if self.length > 0 else 0

    def mul_coeffs(self, factor: complex):
        if not np.isreal(factor) or (np.isreal(factor) and np.real(factor) < 0):
            phase = factor / abs(factor)
        else: 
            phase = 1.0
        for i in range(len(self.instances)):
            inst, coeff = self.instances[i]
            inst.phase *= phase
            self.instances[i] = (inst, coeff * abs(factor))

    def eff_op(self):
        ops = None 
        for inst, coeff in self.instances:
            if ops == None:
                ops = inst.eff_op().to_operator() * coeff
            else:
                ops += inst.eff_op().to_operator() * coeff
        return ops        
    
    def add(self, other):
        return Matrixsum(self.instances + other.instances).simplify()
    
    def mul(self, other):
        out = []
        for a, coeff1 in self.instances:
            for b, coeff2 in other.instances:
                out.append((a.multiply(b), round(coeff1 * coeff2, 8)))
    
        return Matrixsum(out).simplify()
    

    def adj(self):
        new_instances = []
        for inst, c in self.instances:
            new_instances.append((inst.adjoint(), np.conj(c)))
        return Matrixsum(new_instances)
    
    def operator_norm(self):
        total_op = None
        for inst, c in self.instances:
            op = inst.eff_op() * c
            if total_op is None:
                total_op = op
            else:
                total_op += op
        if total_op is None:
            return 0
        else:
            eigs = np.linalg.eigvals(total_op)
            return max(abs(eigs))
    def pauli_norm(self):
        ### sum up the coefficients
        total = 0.0 
        for inst, c in self.instances:
            total += c
        return total
    
    def identity(self, size: int):
        iden = Pauli('I' * size)
        return Matrixsum([(PauliAtom(iden, phase = 1.0), 1.0)])
    
    def simplify(self):
        # Combine same OperatorAtom instances
        matrix_dict = defaultdict(complex)
        for inst, coeff in self.instances:
            if not isinstance(inst, PauliAtom):
                matrix_dict[inst.bare_op()] += coeff * inst.phase 
            else:
                matrix_dict[inst.expr] += coeff * inst.phase
        
        new_instances = []
        for key, total_coeff in matrix_dict.items():
            if total_coeff != 0:
                if isinstance(key, str):
                    new_instances.append((PauliAtom(key, phase = total_coeff / abs(total_coeff)), abs(total_coeff)))
                else:
                    new_instances.append((MatrixAtom(key, phase = total_coeff / abs(total_coeff)), abs(total_coeff)))

        return Matrixsum(new_instances)
    
    def __repr__(self):
        repr_str = "Matrixsum:\n"
        for inst, coeff in self.instances:
            repr_str += f" Coeff: {coeff}, Pauli: {inst.phase}*{inst.expr}" if isinstance(inst, PauliAtom) else f" Coeff: {coeff}, Matrix with phase {inst.phase}\n"
        return repr_str

def matsum_mul(A: Matrixsum, B: Matrixsum) -> Matrixsum:
    out = []
    for a, coeff1 in A.instances:
        for b, coeff2 in B.instances:
            out.append((a.multiply(b), round(coeff1 * coeff2, 8)))
    
    return Matrixsum(out).simplify()
#### An isomorphism from Matrixsum(PauliAtom) to SparsePauliOp
def paulisum_to_sp(A: Matrixsum) -> SparsePauliOp:
    paulis, coeffs = [], []
    for inst, c in A.instances:
        if not isinstance(inst, PauliAtom):
            raise ValueError("All instances must be PauliAtom for conversion to SparsePauliOp")
        paulis.append(inst.expr)
        coeffs.append(c * inst.phase)
    return SparsePauliOp.from_list(zip(paulis, coeffs))

def list2matsum(ops: list) -> Matrixsum:
    instances = []
    for i in range(len(ops)):
        mat, coeff = ops[i]
        if isinstance(mat, str) or isinstance(mat, Pauli):
            instances.append((PauliAtom(mat, phase = coeff / abs(coeff)), abs(coeff)))
        elif isinstance(mat, np.ndarray):
            instances.append((MatrixAtom(mat, phase = coeff / abs(coeff)), abs(coeff)))
    return Matrixsum(instances)

class Lindbladian:
    def __init__(self, H, L_list: list):
        ### H is either a matrix, or a list of unitaries
        self.H = self.input2matsum(H)
        
        ### L_list is a list of Lindblad operators, each either a matrix or a list of unitaries
        self.L_list = []
        if L_list is not None:
            for L in L_list:
                self.L_list.append(self.input2matsum(L))
    
    def input2matsum(self, ops):
        if isinstance(ops, np.ndarray):
            H_pl = SparsePauliOp.from_operator(Operator(ops))
            H_pl = H_pl.simplify(atol=1e-8)
            new_instances = [(PauliAtom(p.to_label(), phase = c/abs(c)), abs(c)) for p, c in zip(H_pl.paulis, H_pl.coeffs)] #type: ignore
        elif isinstance(ops, list):
            new_instances = []
            for i in range(len(ops)):
                
                mats, coeff = ops[i][0], ops[i][1]
                new_coeff = abs(coeff)
                # If input is Pauli
                if isinstance(mats, str) or isinstance(mats, Pauli):
                    new_instances.append((PauliAtom(mats, phase = coeff / new_coeff), new_coeff))
                elif isinstance(mats, np.ndarray):
                    new_instances.append((MatrixAtom(mats, phase = coeff / new_coeff), new_coeff))

        return Matrixsum(new_instances)
    
    def pauli_norm(self) -> float:
        total = self.H.pauli_norm()
        for L in self.L_list:
            total += L.pauli_norm()**2

        return total
    
    def effective_H(self) -> Matrixsum:
        """
        Return the effective Hamiltonian H_eff = H - i/2 sum L^dag L
        """
        H_eff = deepcopy(self.H)
        for L in self.L_list:
            L_dag = L.adj()
            L_dag_L = matsum_mul(L_dag, L)
            L_dag_L.mul_coeffs(-0.5j)
            H_eff = H_eff.add(L_dag_L)
        return H_eff
    
class channel_ensemble:
    """
    A intermediate representation for the (probabilistic ensemble) of quantum channels.
    The channels are of the form: [(s_j, E_j)], where s_j is the probability weight and 
    E_j = sum_l A_jk\rho A_jk^dagger is the j-th quantum channel in the ensemble. 
    """

    def __init__(self, channels: list, probs = None):
        """
        Initialize the channel_IR object.
        
        Args:
            channels (list of Matrixsum): A list of quantum channels, where each channel is represented as a list of Kraus operators (In matrix sum).
            probs (list, optional): A list of probabilities corresponding to each channel. If None, equal probabilities are assumed.
        """
        self.channels = []
        self.length = []
        if probs is None:
            probs = [1/len(channels)] * len(channels)  
        assert len(channels) == len(probs), "Length of channels and probs must match." 
        for i, channel in enumerate(channels):
            self.length.append(max([inst.length for inst in channel]))
            for inst in channel:
                assert isinstance(inst, Matrixsum), "Each Kraus operator must be a Matrixsum."
                
                self.size = inst.size
            self.channels.append((probs[i], channel))

if __name__ == "__main__":
    A = -1j * np.array([[np.exp(-1j * np.pi/4),0],[0, np.exp(1j * np.pi/4)]])
    pa1 = PauliAtom('XIZ', phase=1.0)
    pa2 = PauliAtom('YIZ', phase=1.0j)
    pa3 = PauliAtom('XZZ', phase=np.exp(1j * np.pi / 4))
    pa4 = PauliAtom('XIZ', phase = 1.0j)
    ms1 = Matrixsum([(pa1, 0.5), (pa2, 0.3), (pa4, 0.2)])
    ms2 = Matrixsum([(pa3, 0.2)])
    ms_mul = ms1.mul(ms2)
    print(ms_mul.size)
    print(ms_mul.operator_norm())
    for inst, c in ms_mul.instances:
        print(inst, c)
    
