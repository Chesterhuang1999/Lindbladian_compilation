"""Reference primitives for a TFIM two-unitary-decomposition experiment.

This is NOT the user's ChannelIR backend and NOT the complete benchmark runner.
The NumPy algebra can run without Qiskit. Circuit builders import Qiskit lazily.
Qiskit builders must pass the supplied local tests before benchmark execution.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]

PAULI: dict[str, ComplexMatrix] = {
    'I': np.eye(2, dtype=np.complex128),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.diag([1, -1]).astype(np.complex128),
}


def _matrix(a: Any) -> ComplexMatrix:
    out = np.asarray(a, dtype=np.complex128)
    if out.ndim != 2 or out.shape[0] != out.shape[1] or out.shape[0] < 2:
        raise ValueError('Expected a square matrix of dimension >= 2.')
    d = out.shape[0]
    if d & (d - 1):
        raise ValueError('Matrix dimension must be a power of two.')
    if not np.isfinite(out).all():
        raise ValueError('Matrix entries must be finite.')
    return out


def pauli_matrix(label: str) -> ComplexMatrix:
    """Qiskit label convention: the rightmost character acts on qubit 0."""
    if not label or any(c not in PAULI for c in label):
        raise ValueError('A nonempty I/X/Y/Z label is required.')
    result = np.ones((1, 1), dtype=np.complex128)
    for char in label:
        result = np.kron(result, PAULI[char])
    return result


def _label(n: int, ops: dict[int, str]) -> str:
    s = ['I'] * n
    for q, p in ops.items():
        if not 0 <= q < n:
            raise ValueError('Qubit index out of range.')
        s[n - 1 - q] = p
    return ''.join(s)


def tfim_k0_terms(n: int, J: float = 1.0, h: float = 1.0,
                  gamma: float = 0.1, delta: float = 0.1
                  ) -> list[tuple[str, complex]]:
    """Example 5.1 convention, periodic chain n>=3; L=sqrt(gamma)(X-iY)/2.

    delta=0.1 is a proposed benchmark default, not a verified repository setting.
    """
    if n < 3:
        raise ValueError('TFIM chain requires n>=3 to avoid ambiguous periodic bonds.')
    if not all(math.isfinite(x) for x in [J, h, gamma, delta]):
        raise ValueError('Parameters must be finite.')
    if gamma <= 0 or delta <= 0:
        raise ValueError('gamma and delta must be positive.')
    pairs: list[tuple[str, complex]] = [('I' * n, 1 - n * gamma * delta / 4)]
    for q in range(n):
        pairs.extend([
            (_label(n, {q: 'Z', (q + 1) % n: 'Z'}), 1j * delta * J),
            (_label(n, {q: 'X'}), 1j * delta * h),
            (_label(n, {q: 'Z'}), -gamma * delta / 4),
        ])
    # Only exact zeros are removed. No approximate coefficient truncation.
    return [(p, complex(c)) for p, c in pairs if c != 0]


def dense_from_terms(terms: Sequence[tuple[str, complex]]) -> ComplexMatrix:
    if not terms:
        raise ValueError('A nonempty Pauli sum is required.')
    n = len(terms[0][0])
    if any(len(p) != n for p, _ in terms):
        raise ValueError('All Pauli labels must have the same length.')
    out = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
    for label, coeff in terms:
        out += coeff * pauli_matrix(label)
    return out


def tfim_k0(n: int, **kwargs: float) -> tuple[ComplexMatrix, float]:
    terms = tfim_k0_terms(n, **kwargs)
    return dense_from_terms(terms), float(sum(abs(c) for _, c in terms))


@dataclass(frozen=True)
class TUDFactors:
    W: ComplexMatrix
    singular_values: NDArray[np.float64]
    Vh: ComplexMatrix
    beta: float
    d_plus: NDArray[np.complex128]
    d_minus: NDArray[np.complex128]
    clipped_entries: int
    max_overshoot: float

    @property
    def n(self) -> int:
        return self.W.shape[0].bit_length() - 1

    def branch_unitaries(self) -> tuple[ComplexMatrix, ComplexMatrix]:
        # Multiplication by a diagonal uses column-wise scaling, not a dense diag.
        return ((self.W * self.d_plus) @ self.Vh,
                (self.W * self.d_minus) @ self.Vh)

    def reconstructed(self) -> ComplexMatrix:
        up, um = self.branch_unitaries()
        return (self.beta / 2) * (up + um)


def factor_tud(K: Any, beta: float | None = None, *, clip_tol: float = 1e-12
               ) -> TUDFactors:
    """Classical SVD-based TUD. beta=None uses the numerical spectral norm.

    For a nonzero matrix, NumPy's returned Vh is V-dagger; DO NOT conjugate it
    again. No rank truncation, SU projection, random perturbation, or phase reset.
    """
    k = _matrix(K)
    if not math.isfinite(clip_tol) or not 0 < clip_tol <= 1e-12:
        raise ValueError('clip_tol must be finite, positive, and at most 1e-12.')
    w, s, vh = np.linalg.svd(k, full_matrices=False)
    sigma_max = float(s[0])
    if sigma_max == 0:
        raise ValueError('Zero operator: eliminate it rather than invent beta=0.')
    b = sigma_max if beta is None else float(beta)
    if not math.isfinite(b) or b <= 0:
        raise ValueError('beta must be finite and positive.')
    x = s / b
    overshoot = float(max(0.0, np.max(x) - 1.0))
    if overshoot > clip_tol:
        raise ValueError(f'beta is too small: max singular-value ratio={x.max()}.')
    clipped = int(np.count_nonzero((x < 0) | (x > 1)))
    x = np.clip(x, 0.0, 1.0)
    # Stable relative to 1-x**2 near x=1; any permitted clipping is recorded.
    y = np.sqrt(np.maximum(0.0, (1 - x) * (1 + x)))
    dp = np.asarray(x + 1j * y, dtype=np.complex128)
    dm = np.asarray(x - 1j * y, dtype=np.complex128)
    return TUDFactors(w, s, vh, b, dp, dm, clipped, overshoot)


def factor_tud_branch(K: Any, beta: float | None = None, *, clip_tol: float = 1e-12
                      ) -> TUDFactors | None:
    """Return None for an exact zero branch, before SVD or normalization.

    A caller must skip this branch, not synthesize a circuit with beta=0.
    Nonzero inputs use factor_tud without thresholding any matrix entries.
    """
    k = _matrix(K)
    if not np.any(k != 0):
        return None
    return factor_tud(k, beta, clip_tol=clip_tol)


def algebra_metrics(K: Any, f: TUDFactors) -> dict[str, float | int]:
    k = _matrix(K)
    up, um = f.branch_unitaries()
    identity = np.eye(k.shape[0], dtype=np.complex128)
    return {
        'svd_relative_fro_error': float(np.linalg.norm(
            (f.W * f.singular_values) @ f.Vh - k, 'fro') / np.linalg.norm(k, 'fro')),
        'tud_relative_fro_error': float(np.linalg.norm(
            f.reconstructed() - k, 'fro') / np.linalg.norm(k, 'fro')),
        'U_plus_unitarity_fro': float(np.linalg.norm(up.conj().T @ up - identity, 'fro')),
        'U_minus_unitarity_fro': float(np.linalg.norm(um.conj().T @ um - identity, 'fro')),
        'U_plus_unitarity_normalized_fro': float(np.linalg.norm(
            up.conj().T @ up - identity, 'fro') / np.sqrt(k.shape[0])),
        'U_minus_unitarity_normalized_fro': float(np.linalg.norm(
            um.conj().T @ um - identity, 'fro') / np.sqrt(k.shape[0])),
        'clipped_entries': f.clipped_entries,
        'max_overshoot': f.max_overshoot,
    }


def dense_reference_block(f: TUDFactors) -> ComplexMatrix:
    """Small-test oracle only. Ancilla is most significant; data are low bits."""
    up, um = f.branch_unitaries()
    return np.block([[(up + um) / 2, (up - um) / 2],
                     [(up - um) / 2, (up + um) / 2]])


def synth_unitary_qsd(U: Any):
    """Explicit public Qiskit synthesis entry point, not a one-gate wrapper.

    Do not pass deprecated opt_a1/opt_a2 switches. Record the installed signature
    and Qiskit version before using this routine for measured experiments.
    """
    from qiskit.synthesis import qs_decomposition
    return qs_decomposition(_matrix(U))


def lower_to_u_cx(circuit: Any, *, optimization_level: int, seed: int = 646):
    from qiskit import transpile
    if optimization_level not in (0, 3):
        raise ValueError('This protocol permits optimization levels 0 and 3.')
    if 'qubits_initially_zero' not in inspect.signature(transpile).parameters:
        raise RuntimeError(
            'This Qiskit version lacks qubits_initially_zero. Use an isolated '
            'compatible environment or explicitly audit an alternative; do not '
            'silently assume the arbitrary input state is |0...0>.')
    out = transpile(
        circuit,
        basis_gates=['u', 'cx'],
        coupling_map=None,
        optimization_level=optimization_level,
        approximation_degree=1.0,
        seed_transpiler=seed,
        unitary_synthesis_method='default',
        num_processes=1,
        qubits_initially_zero=False,
    )
    bad = sorted(set(out.count_ops()) - {'u', 'cx'})
    if bad:
        raise RuntimeError(f'Output contains non-native operations: {bad}')
    return out


def build_factored_tud(f: TUDFactors,
                       middle: Literal['diagonal', 'ucrz'] = 'diagonal'):
    """Return a raw full TUD block encoding (including BOTH Hadamards).

    The main baseline uses DiagonalGate. UCRZ is a prespecified optional
    optimization; it must pass exact matrix tests before results are accepted.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate
    n = f.n
    data = list(range(n))
    ancilla = n
    qc = QuantumCircuit(n + 1, name=f'tud_factored_{middle}')
    qc.h(ancilla)
    qc.compose(synth_unitary_qsd(f.Vh), qubits=data, inplace=True)
    if middle == 'diagonal':
        # qargs are [data low bits..., ancilla high bit]. Thus concat, not interleave.
        diagonal = np.concatenate([f.d_plus, f.d_minus])
        qc.append(DiagonalGate(diagonal.tolist()), data + [ancilla])
    elif middle == 'ucrz':
        from qiskit.circuit.library import UCRZGate
        theta = np.angle(f.d_plus)
        # RZ(-2 theta) = diag(exp(+i theta), exp(-i theta)).
        # UCRZ's first qarg is the target; data index selects the angle.
        qc.append(UCRZGate((-2 * theta).tolist()), [ancilla] + data)
    else:
        raise ValueError('Unknown middle synthesis choice.')
    qc.compose(synth_unitary_qsd(f.W), qubits=data, inplace=True)
    qc.h(ancilla)
    return qc


def build_separate_tud(f: TUDFactors):
    """Diagnostic baseline: synthesize both branches, then coherently control.

    Keep each branch's global phase. Discarding it changes the LCU interference.
    """
    from qiskit import QuantumCircuit
    n = f.n
    data = list(range(n))
    ancilla = n
    up, um = f.branch_unitaries()
    plus = synth_unitary_qsd(up).to_gate()
    minus = synth_unitary_qsd(um).to_gate()
    qc = QuantumCircuit(n + 1, name='tud_separate')
    qc.h(ancilla)
    qc.x(ancilla)
    qc.append(plus.control(1), [ancilla] + data)
    qc.x(ancilla)
    qc.append(minus.control(1), [ancilla] + data)
    qc.h(ancilla)
    return qc


def build_isometry_reference(K: Any, f: TUDFactors):
    """Optional single-Kraus isometry comparator, NOT a TUD implementation.

    Upper block K/beta; lower block sqrt(I-Kdag K/beta**2). One output flag.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry
    k = _matrix(K)
    v = f.Vh.conj().T
    root_diag = np.sqrt(np.maximum(0.0, 1 - (f.singular_values / f.beta) ** 2))
    complement = (v * root_diag) @ f.Vh
    iso = np.vstack([k / f.beta, complement])
    qc = QuantumCircuit(f.n + 1, name='single_kraus_isometry')
    qc.append(Isometry(iso, num_ancillas_zero=0, num_ancillas_dirty=0),
              list(range(f.n + 1)))
    return qc


def small_circuit_metrics(K: Any, f: TUDFactors, circuit: Any,
                          *, max_system_qubits: int = 4) -> dict[str, float]:
    """Full operator checks for small TUD circuits. No per-branch phase alignment."""
    from qiskit.quantum_info import Operator
    k = _matrix(K)
    if f.n > max_system_qubits:
        raise ValueError('Use batched statevector probes, not a large circuit matrix.')
    matrix = Operator(circuit).data
    reference = dense_reference_block(f)
    d = k.shape[0]
    expected = k / f.beta
    return {
        'whole_tud_fro_error': float(np.linalg.norm(matrix - reference, 'fro')),
        'block_fro_error': float(np.linalg.norm(matrix[:d, :d] - expected, 'fro')),
        'block_operator_error': float(np.linalg.norm(matrix[:d, :d] - expected, 2)),
    }


def native_metrics(circuit: Any) -> dict[str, Any]:
    counts = {str(k): int(v) for k, v in circuit.count_ops().items()}
    if set(counts) - {'u', 'cx'}:
        raise ValueError('Count only completely lowered U/CX circuits.')
    return {'counts': counts, 'gates_total': sum(counts.values()),
            'gates_cx': counts.get('cx', 0), 'gates_u': counts.get('u', 0),
            'depth': int(circuit.depth()), 'width': int(circuit.num_qubits)}
