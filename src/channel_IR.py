import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli
from abc import ABC, abstractmethod
from collections import defaultdict 
from copy import deepcopy
from itertools import combinations
class OperatorAtom(ABC):
    def __init__(self, phase: complex = 1.0):
        self.phase = round(phase.real, 15) + 1j * round(phase.imag, 15)
        # self.phase = phase
    
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
        super().__init__(phase)
        self.mat = mat
        self.size = int(np.ceil(np.log2(mat.shape[0])))

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
        ## Assert that all instances have the same size
        if self.instances:
            self.size = self.instances[0][0].size
            for inst, _ in self.instances:
                assert inst.size == self.size, "All instances must have the same size"
        else:
            self.size = 0
        # self.size = max([inst.size for inst, _ in self.instances]) if self.instances else 0

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
    
    def poly(self, deg):
        ## Compute the polynomial expansion of the Matrixsum up to degree deg
        if deg == 0:
            return self.identity(self.size)
        elif deg == 1:
            return self
        else: 
            result = self
            for _ in range(deg - 1):
                result = result.mul(self)
            return result.simplify()
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

    def eliminate_global_phase(self, tol=1e-10):
        """Shift all instance phases by a shared angle that maximizes zero-phase terms."""
        if len(self.instances) == 0:
            return Matrixsum([])

        angles = [np.angle(inst.phase) for inst, _ in self.instances]

        best_theta = angles[0]
        best_count = -1
        for theta in angles:
            zero_count = 0
            for angle in angles:
                wrapped_diff = angle - theta
                if abs(wrapped_diff) <= tol:
                    zero_count += 1
            if zero_count > best_count:
                best_count = zero_count
                best_theta = theta

        global_factor = np.exp(-1j * best_theta)
        adjusted_instances = []
        for inst, coeff in self.instances:
            new_phase = inst.phase * global_factor
            if isinstance(inst, PauliAtom):
                adjusted_instances.append((PauliAtom(inst.expr, phase=new_phase), coeff))
            else:
                adjusted_instances.append((MatrixAtom(inst.bare_op(), phase=new_phase), coeff))

        return Matrixsum(adjusted_instances)
    
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

        return Matrixsum(new_instances).eliminate_global_phase()
    def is_proportional(self, other, tol=1e-10):
        """
        Check if self = c * other for some complex number c.

        Returns:
            (True, c) if proportional, otherwise (False, 0.0).
        """
        if not isinstance(other, Matrixsum):
            raise TypeError("Can only compare with another Matrixsum")

        def _dense_matrix(ms):
            total = None
            for inst, coeff in ms.instances:
                op = inst.to_operator().data * coeff
                if total is None:
                    total = np.array(op, dtype=complex)
                else:
                    total = total + op
            return total

        def _is_pauli_only(ms):
            return all(isinstance(inst, PauliAtom) for inst, _ in ms.instances)

        def _compare_dense(A, B):
            if A is None and B is None:
                return True, 1.0
            if A is None or B is None:
                return False, 0.0
            if A.shape != B.shape:
                return False, 0.0

            if np.allclose(A, 0, atol=tol) and np.allclose(B, 0, atol=tol):
                return True, 1.0
            if np.allclose(B, 0, atol=tol):
                return False, 0.0

            idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
            ref = B[idx]
            if abs(ref) <= tol:
                return False, 0.0

            scale = A[idx] / ref
            if np.allclose(A, scale * B, atol=tol, rtol=tol):
                return True, scale
            return False, 0.0

        def _compare_pauli(ms1, ms2):
            def _signature(ms):
                sig = defaultdict(complex)
                for inst, coeff in ms.instances:
                    sig[inst.expr] += coeff * inst.phase
                return sig

            sig1 = _signature(ms1)
            sig2 = _signature(ms2)

            if set(sig1.keys()) != set(sig2.keys()):
                return False, 0.0

            if len(sig2) == 0:
                return True, 1.0

            ref_key = None
            for key, value in sig2.items():
                if abs(value) > tol:
                    ref_key = key
                    break

            if ref_key is None:
                if all(abs(value) <= tol for value in sig1.values()):
                    return True, 1.0
                return False, 0.0

            scale = sig1[ref_key] / sig2[ref_key]
            for key in sig1.keys():
                if not np.isclose(sig1[key], scale * sig2[key], atol=tol, rtol=tol):
                    return False, 0.0
            return True, scale

        if self.length == 0 and other.length == 0:
            return True, 1.0
        elif self.length == 0 or other.length == 0:
            return False, 0.0

        if not (_is_pauli_only(self) and _is_pauli_only(other)):
            return _compare_dense(_dense_matrix(self), _dense_matrix(other))

        return _compare_pauli(self, other)

    def remove_iden(self):
        iden_coeff = 0.0
        for inst, c in self.instances:
            if isinstance(inst, PauliAtom) and inst.expr == 'I' * inst.size:
                self.instances.remove((inst, c))
                iden_coeff += c * inst.phase
        return iden_coeff
    def __repr__(self):
        repr_str = "Matrixsum:\n"
        for inst, coeff in self.instances:
            repr_str += f" Coeff: {np.round(coeff, 6)}, Pauli: {inst.phase}*{inst.expr}" if isinstance(inst, PauliAtom) else f" Coeff: {np.round(coeff, 6)}, Matrix with phase {inst.phase}\n"
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
    return Matrixsum(instances).simplify()

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
            H_pl = H_pl.simplify(atol=1e-10)
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
            total += 0.5 * L.pauli_norm()**2

        return total
    def operator_norm(self):
        total = np.linalg.norm(self.H.eff_op().data, ord=2) #type: ignore
        for L in self.L_list:
            total += np.linalg.norm(L.eff_op().data, ord=2)**2
    
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

        H_eff.mul_coeffs(-1j)
        H_eff.simplify()
        return H_eff
    def __size__(self):
        if self.H.size > 0:
            return self.H.size
        elif len(self.L_list) > 0:
            return max([L.size for L in self.L_list])
        else:
            return 0
        
    def __repr__(self):
        repr_str = "Lindbladian:\n"
        repr_str += f"Hamiltonian:\n{self.H}\n"
        repr_str += "Lindblad operators:\n"
        for L in self.L_list:
            repr_str += f"{L}\n"
        return repr_str
    
class channel:
    """ 
    An intermediate representation for a quantum channel, represented as a list of Kraus operators in Matrixsum form. 
    The Matrixsum form naturally supports syntax-level rewrites for Kraus operators, such as zero elim, merging and unitary transform. 
    """
    def __init__(self, kraus_ops: list):
        self.kraus_ops = []
        self.size = None
        for op in kraus_ops:
            if isinstance(op, Matrixsum):
            
                self.kraus_ops.append(op)
                ## Type checking: checking dimensional consistency of Kraus operators. All Kraus operators must have the same size.
                ## If op.size = 0, it is a zero operator, which can be ignored for check.
                if op.size != 0: 
                    if self.size is None:
                        self.size = op.size
                    else: 
                        assert self.size == op.size, "All Kraus operators must have the same size"
            else:
                raise ValueError("Kraus operators must be in Matrixsum form")


    ### Channel-IR rewrite rules  
     
    def zero_elim(self):
        """Eliminate zero Kraus operators and update the size of kraus_ops."""
        self.kraus_ops = [op for op in self.kraus_ops if op.length > 0]
        if len(self.kraus_ops) == 0:
            self.size = None
        return self

     
    def glob_phase_elim(self):
        """
        Global phase elimination: eliminate global phase for each Kraus operator, minimize the number of non-zero phases."""
        self.kraus_ops = [op.eliminate_global_phase() for op in self.kraus_ops]
        return self

    def merging(self, tol=1e-10):
        """Merge Kraus operators that are proportional to each other, i.e. K1 = c * K2. """
        # First eliminate zero operators
        self.zero_elim()

        merged_kraus_ops = []
        visited = [False] * len(self.kraus_ops)

        for i, base_op in enumerate(self.kraus_ops):
            if visited[i]:
                continue

            visited[i] = True
            norm_sq_sum = 1.0

            for j in range(i + 1, len(self.kraus_ops)):
                if visited[j]:
                    continue

                is_prop, scale = self.kraus_ops[j].is_proportional(base_op, tol=tol)
                if is_prop:
                    norm_sq_sum += abs(scale) ** 2
                    visited[j] = True

            merged_op = deepcopy(base_op).eliminate_global_phase()
            merged_op.mul_coeffs(np.sqrt(norm_sq_sum))
            merged_kraus_ops.append(merged_op)

        self.kraus_ops = merged_kraus_ops
        self.size = None if len(self.kraus_ops) == 0 else self.kraus_ops[0].size
        return self

    def permutation(self, perm: list):
        """
        Permute the order of Kraus operators.
        """
        assert len(perm) == len(self.kraus_ops), "Permutation length must match number of Kraus operators"
        self.kraus_ops = [self.kraus_ops[i] for i in perm]

    def _matrixsum_to_dense(self, op: Matrixsum, dim: int | None = None) -> np.ndarray:
        """Convert one Kraus operator in Matrixsum form to a dense matrix."""
        if dim is None:
            if op.size > 0:
                dim = 2 ** op.size
            elif self.size is not None:
                dim = 2 ** self.size
            else:
                raise ValueError("Cannot infer channel dimension from zero Kraus operators only")

        dense = np.zeros((dim, dim), dtype=complex)
        for inst, coeff in op.instances:
            dense += coeff * inst.to_operator().data
        return dense

    def choi_matrix(self, normalized: bool = False) -> np.ndarray:
        """
        Construct the Choi matrix of the channel E(rho)=sum_i K_i rho K_i^dagger.

        We use the column-stacking convention:
            J(E) = sum_i |K_i>><<K_i|,  |K_i>> = vec(K_i)

        Args:
            normalized: If True, return J / d where d is system dimension.

        Returns:
            The Choi matrix as a dense complex ndarray of shape (d^2, d^2).
        """
        if self.size is None:
            self.zero_elim()
            if self.size is None and len(self.kraus_ops) > 0:
                self.size = max(op.size for op in self.kraus_ops)
            if self.size is None:
                raise ValueError("Cannot construct Choi matrix for an empty channel")

        dim = 2 ** self.size
        choi = np.zeros((dim * dim, dim * dim), dtype=complex)

        for op in self.kraus_ops:
            K = self._matrixsum_to_dense(op, dim=dim)
            vec_K = K.reshape(dim * dim, order='F')
            choi += np.outer(vec_K, np.conj(vec_K))

        if normalized:
            choi = choi / dim

        return choi

    def choi_rank(self, tol: float = 1e-10, normalized: bool = False) -> int:
        """
        Compute the rank of the Choi matrix of this channel.

        Args:
            tol: Eigenvalue threshold for numerical rank.
            normalized: Whether to compute rank from normalized Choi matrix J/d.

        Returns:
            Numerical rank as an integer.
        """
        choi = self.choi_matrix(normalized=normalized)
        hermitian_choi = 0.5 * (choi + choi.conj().T)
        eigvals = np.linalg.eigvalsh(hermitian_choi)
        return int(np.sum(np.abs(eigvals) > tol))

    def two_kraus_unitary_transform(self):
        """
        Reshape the two Kraus operators K1, K2 into K1' = aK1 + bK2, K2' = -b^* K1 + a^*K2, where |a|^2 + |b|^2 = 1. 
        """
        pass

    # ================================================================
    # Heuristic rewrite search framework for unitary rewrite of Kraus operators
    # ================================================================

    def _coeff_matrix(self, tol=1e-12):
        """
        Extract the coefficient matrix A and the ordered Pauli label list.

        Returns:
            A: np.ndarray of shape (m, n), where m = #Kraus ops, n = #distinct Pauli labels.
               Entry A[i,j] = signed coefficient of the j-th Pauli in the i-th Kraus op.
            labels: list of str, the n distinct Pauli labels (column ordering).
        """
        label_set = {}
        for op in self.kraus_ops:
            for inst, coeff in op.instances:
                if not isinstance(inst, PauliAtom):
                    raise ValueError("Coefficient matrix extraction requires all PauliAtom instances")
                key = inst.expr
                if key not in label_set:
                    label_set[key] = len(label_set)

        labels = [''] * len(label_set)
        for k, v in label_set.items():
            labels[v] = k

        m = len(self.kraus_ops)
        n = len(labels)
        
        A = np.zeros((m, n), dtype=complex)
        for i, op in enumerate(self.kraus_ops):
            for inst, coeff in op.instances:
                j = label_set[inst.expr]
                # Keep full complex phase information. eliminate_global_phase only
                # removes a shared phase and does not force each term to be real.
                A[i, j] = coeff * inst.phase

        # Zero out tiny entries  
        A[np.abs(A) < tol] = 0.0
        return A, labels

    @staticmethod
    def _support(A, tol=1e-12):
        """Total number of non-zero entries in coefficient matrix."""
        return int(np.sum(np.abs(A) > tol))

    @staticmethod
    def _row_supports(A, tol=1e-12):
        """Number of non-zero entries per row."""
        return np.array([int(np.sum(np.abs(A[i]) > tol)) for i in range(A.shape[0])])

    @staticmethod
    def _apply_unitary(A, i, j, U):
        """
        Apply a 2x2 unitary U to rows i, j of matrix A (on a copy).

        The row update is:
            [A_i^new]   [U[0,0]  U[0,1]] [A_i]
            [A_j^new] = [U[1,0]  U[1,1]] [A_j]

        Returns the new matrix.
        """
        B = A.copy()
        B[i] = U[0, 0] * A[i] + U[0, 1] * A[j]
        B[j] = U[1, 0] * A[i] + U[1, 1] * A[j]
        return B

    @staticmethod
    def _candidate_unitaries_for_pair(A, i, j, tol=1e-12):
        """
        Generate candidate 2x2 unitaries for row pair (i, j).

        Parameterization (global-phase gauge a >= 0 real):
            U = [[a,  -conj(b)],
                 [b,   a      ]],   a in R_{>=0}, b in C, a^2 + |b|^2 = 1.

        For each column k, two candidates are generated:
          - U^{(i,k)} that zeros A_i^new[k] when |A[j,k]| > tol:
                r  = |A[i,k]/A[j,k]|,
                a  = 1/sqrt(1+r^2),
                b  = a * conj(A[i,k]/A[j,k]).
          - U^{(j,k)} that zeros A_j^new[k] when |A[i,k]| > tol:
                r' = |A[j,k]/A[i,k]|,
                a  = 1/sqrt(1+r'^2),
                b  = -a * (A[j,k]/A[i,k]).

        The identity is always included as a no-op candidate.

        Returns:
            List of 2x2 complex ndarrays.
        """
        unitaries = [np.eye(2, dtype=complex)]
        ri, rj = A[i], A[j]

        for k in range(A.shape[1]):
            ai, aj = ri[k], rj[k]

            # Zero A_i^new[k]: bar(b)/a = ai/aj  =>  b = a * conj(ai/aj)
            if abs(aj) > tol:
                ratio = ai / aj
                r = abs(ratio)
                a = 1.0 / np.sqrt(1.0 + r * r)
                b = a * np.conj(ratio)
                U = np.array([[a, -np.conj(b)],
                              [b,  a         ]], dtype=complex)
                unitaries.append(U)

            # Zero A_j^new[k]: b = -a * aj/ai
            if abs(ai) > tol:
                ratio = aj / ai
                r = abs(ratio)
                a = 1.0 / np.sqrt(1.0 + r * r)
                b = -a * ratio
                U = np.array([[a, -np.conj(b)],
                              [b,  a         ]], dtype=complex)
                unitaries.append(U)

        return unitaries

    @staticmethod
    def _best_unitary_for_pair(A, i, j, tol=1e-12):
        """
        Find the 2x2 unitary among the generated candidates that minimizes
        the support on rows i and j.

        Returns:
            (best_U, best_support, best_A)
        """
        row_supports = channel._row_supports(A, tol)
        old_pair_support = int(row_supports[i] + row_supports[j])
        total_support = int(np.sum(row_supports))
        unaffected_support = total_support - old_pair_support
        candidates = channel._candidate_unitaries_for_pair(A, i, j, tol)

        best_U = np.eye(2, dtype=complex)
        best_pair_support = old_pair_support
        best_support = total_support
        best_A = A

        for U in candidates:
            B = channel._apply_unitary(A, i, j, U)
            B[np.abs(B) < tol] = 0.0
            new_pair_support = int(np.sum(np.abs(B[i]) > tol) + np.sum(np.abs(B[j]) > tol))
            if new_pair_support < best_pair_support:
                best_pair_support = new_pair_support
                best_support = unaffected_support + new_pair_support
                best_U = U
                best_A = B

        return best_U, best_support, best_A

    def rewrite_search(self, strategy='greedy', beam_width=3, max_steps=50,
                       tol=1e-12, verbose=False, strict_beam=False):
        """
        Heuristic search for 2-row unitary rewrites that minimize total Pauli term count.

        Args:
            strategy: 'greedy' — always pick the best single-step improvement.
                      'beam'   — keep top beam_width states and explore all pair unitaries.
            beam_width: Number of states to keep at each step (only for 'beam').
            max_steps: Maximum number of rewrite steps.
            tol: Numerical tolerance.
            verbose: Print progress info.
            strict_beam: If True, beam only accepts strictly improving candidates.
                         If False, beam accepts no-worse candidates.

        Returns:
            dict with keys:
                'initial_support': int
                'final_support': int
                'steps': list of ((i,j), U, support_after) tuples
                'A_final': final coefficient matrix
                'labels': Pauli labels
        """
        A, labels = self._coeff_matrix(tol)
        m = A.shape[0]
        initial_support = self._support(A, tol)

        if strategy == 'greedy':
            return self._greedy_search(A, labels, m, initial_support, max_steps, tol, verbose)
        elif strategy == 'beam':
            return self._beam_search(
                A,
                labels,
                m,
                initial_support,
                beam_width,
                max_steps,
                tol,
                verbose,
                strict_beam,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _greedy_search(self, A, labels, m, initial_support, max_steps, tol, verbose):
        """
        Greedy: always apply the single best 2-row unitary.

        Optimization: only compute support change for the two affected rows,
        rather than recomputing the entire matrix support each time.
        """
        steps = []
        current_A = A.copy()
        current_support = initial_support
        support_trajectory = [initial_support]
        stop_reason = "max_steps_reached"

        for step in range(max_steps):
            best_pair = None
            best_U = np.eye(2, dtype=complex)
            best_delta = 0  # Change in support (negative means improvement)

            row_supports = self._row_supports(current_A, tol)

            for i, j in combinations(range(m), 2):
                old_pair_support = int(row_supports[i] + row_supports[j])

                candidates = self._candidate_unitaries_for_pair(current_A, i, j, tol)

                for U in candidates:
                    B_rotated = self._apply_unitary(current_A, i, j, U)
                    B_rotated[np.abs(B_rotated) < tol] = 0.0

                    new_support_i = int(np.sum(np.abs(B_rotated[i]) > tol))
                    new_support_j = int(np.sum(np.abs(B_rotated[j]) > tol))
                    new_pair_support = new_support_i + new_support_j

                    delta = new_pair_support - old_pair_support

                    if delta < best_delta:
                        best_delta = delta
                        best_U = U
                        best_pair = (i, j)

            if best_pair is None or best_delta >= 0:
                stop_reason = "no_improvement"
                if verbose:
                    print(f"  Step {step}: no improvement found, stopping.")
                break

            i, j = best_pair
            current_A = self._apply_unitary(current_A, i, j, best_U)
            current_A[np.abs(current_A) < tol] = 0.0
            current_support += best_delta
            support_trajectory.append(current_support)

            steps.append((best_pair, best_U, current_support))
            if verbose:
                print(f"  Step {step}: apply U on pair {best_pair}, "
                      f"delta={best_delta}, support={current_support}")

        return {
            'initial_support': initial_support,
            'final_support': current_support,
            'steps': steps,
            'A_final': current_A,
            'labels': labels,
            'termination': {
                'stop_reason': stop_reason,
                'iterations': len(steps),
                'max_steps': max_steps,
                'support_trajectory': support_trajectory,
            },
        }

    def _beam_search(self, A, labels, m, initial_support, beam_width, max_steps, tol, verbose, strict_beam):
        """
        Beam search: maintain top beam_width states, expand all pair unitaries.
        Allows zero-gain steps to escape local minima.
        """
        # State: (support, A_matrix, steps_list)
        beam = [(initial_support, A.copy(), [])]
        global_best = (initial_support, A.copy(), [])
        best_support_trajectory = [initial_support]
        stop_reason = "max_steps_reached"
        iterations = 0

        for step in range(max_steps):
            iterations += 1
            candidates = []

            for sup, state_A, state_steps in beam:
                for i, j in combinations(range(m), 2):
                    U, new_sup, new_A = self._best_unitary_for_pair(state_A, i, j, tol)
                    # Accept policy for beam expansion.
                    if (strict_beam and new_sup < sup) or ((not strict_beam) and new_sup <= sup):
                        new_steps = state_steps + [((i, j), U, new_sup)]
                        candidates.append((new_sup, new_A, new_steps))

                        if new_sup < global_best[0]:
                            global_best = (new_sup, new_A.copy(), list(new_steps))

            if not candidates:
                stop_reason = "no_candidates"
                if verbose:
                    print(f"  Step {step}: no candidates, stopping.")
                break

            # Deduplicate by support signature (row supports as tuple)
            seen = {}
            for sup, cA, csteps in candidates:
                key = tuple(sorted(self._row_supports(cA, tol)))
                if key not in seen or sup < seen[key][0]:
                    seen[key] = (sup, cA, csteps)

            unique = sorted(seen.values(), key=lambda x: x[0])
            beam = unique[:beam_width]

            if verbose:
                supports = [s for s, _, _ in beam]
                print(f"  Step {step}: beam supports = {supports}, global best = {global_best[0]}")

            best_support_trajectory.append(global_best[0])

            # Early termination: if global best hasn't improved in this step
            if beam[0][0] >= global_best[0] and step > 0:
                # Check if all beam states are at or above global best
                if all(s >= global_best[0] for s, _, _ in beam):
                    stop_reason = "converged"
                    if verbose:
                        print(f"  Step {step}: beam converged at support={global_best[0]}")
                    break

        return {
            'initial_support': initial_support,
            'final_support': global_best[0],
            'steps': global_best[2],
            'A_final': global_best[1],
            'labels': labels,
            'termination': {
                'stop_reason': stop_reason,
                'iterations': iterations,
                'max_steps': max_steps,
                'support_trajectory': best_support_trajectory,
                'strict_beam': strict_beam,
            },
        }

    def apply_rewrite_result(self, result, tol=1e-12):
        """
        Apply a rewrite search result back to the channel's Kraus operators.
        Reconstructs Matrixsum objects from the final coefficient matrix.
        
        Args:
            result: dict returned by rewrite_search().
        Returns:
            self (mutated in-place).
        """
        A_final = result['A_final']
        labels = result['labels']
        m, n = A_final.shape

        new_kraus_ops = []
        for i in range(m):
            instances = []
            for j in range(n):
                val = A_final[i, j]
                if abs(val) > tol:
                    phase = val / abs(val)
                    instances.append((PauliAtom(labels[j], phase=phase), abs(val)))
            new_kraus_ops.append(Matrixsum(instances))

        self.kraus_ops = new_kraus_ops
        return self

    # def unitary_transform(self, U: np.ndarray, indices: list):
    #     """
    #     Perform a general unitary transform over Kraus operator {K1, ... Kn}.
    #     """
    #     assert U.conj().T @ U == np.eye(U.shape[0]), "U must be unitary"
    #     assert U.shape[0] == U.shape[1] == len(indices), "U must be square and match the number of indices"
    #     assert max(indices) < len(self.kraus_ops), "Indices must be within the range of Kraus operators"

    #     new_kraus_ops = [deepcopy(k) for k in self.kraus_ops]
    #     for i in range(len(indices)):
    #         new_op = Matrixsum([])
    #         for j in range(len(indices)):
    #             new_op = new_op.add(self.kraus_ops[indices[j]].mul_coeffs(U[i, j]))
    #         new_kraus_ops[indices[i]] = new_op

    #     self.kraus_ops = new_kraus_ops


class channel_ensemble:

    """
    An intermediate representation for the (probabilistic ensemble) of quantum channels.
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
    # A = -1j * np.array([[np.exp(-1j * np.pi/4),0],[0, np.exp(1j * np.pi/4)]])
    # pa1 = PauliAtom('XIZ', phase=1.0)
    # pa2 = PauliAtom('YIZ', phase=1.0j)
    # pa3 = PauliAtom('XZZ', phase=np.exp(1j * np.pi / 4))
    # pa4 = PauliAtom('XIZ', phase = 1.0j)
    # ms1 = Matrixsum([(pa1, 0.5), (pa2, 0.3), (pa4, 0.2)])
    # ms2 = Matrixsum([(pa3, 0.2)])
    # ms_mul = ms1.mul(ms2)
    # print(ms_mul.size)
    # print(ms_mul.operator_norm())
    # for inst, c in ms_mul.instances:
    #     print(inst, c)
    k1 = Matrixsum([(PauliAtom("XX", 1.0), 1.0), (PauliAtom("YY", 1.0), 1.0), (PauliAtom("ZZ", -1.0), 1.0)])
    k2 = Matrixsum([(PauliAtom("XX", 1.0j), 1.0), (PauliAtom("YY", -1.0), 1.0), (PauliAtom("IZ", -1.0), 1.0)])
    k3 = Matrixsum([(PauliAtom("XX", 1.0), 0.5), (PauliAtom("YY", 1.0), 1.3), (PauliAtom("ZZ", -1.0), 0.8)])
    k4 = Matrixsum([(PauliAtom("XX", -1.0j), 1.0), (PauliAtom("YY", 1.0), 1.0), (PauliAtom("IZ", 1.0), 0.4)])
    ch = channel([k1, k2, k3, k4])
    result = ch.rewrite_search(strategy='beam')
    ch.apply_rewrite_result(result)
    print(result)
    print(ch.kraus_ops)
 
