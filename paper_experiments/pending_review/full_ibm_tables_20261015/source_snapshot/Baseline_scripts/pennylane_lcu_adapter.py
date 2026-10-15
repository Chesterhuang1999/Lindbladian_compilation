"""PennyLane-native Pauli LCU construction and phase-preserving Qiskit export.

Input labels use Qiskit's convention: the leftmost character acts on the most
significant system qubit. Coefficients must already include any phase stored in
the source Pauli representation. Term order and zero-weight SELECT branches are
preserved. No dense matrix is used for circuit synthesis.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pennylane as qml


LEAF_GATE_SET = frozenset(
    {
        "Identity", "PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "SX",
        "RX", "RY", "RZ", "PhaseShift", "CNOT", "CY", "CZ", "SWAP", "GlobalPhase",
    }
)
DEFAULT_PIPELINE = (
    "commute_controlled", "cancel_inverses", "merge_rotations", "remove_barrier"
)


@contextmanager
def _decomposition_mode(mode: str):
    if mode not in {"legacy", "graph"}:
        raise ValueError("decomposition must be 'legacy' or 'graph'")
    enabled = qml.decomposition.enabled_graph()
    (qml.decomposition.enable_graph if mode == "graph" else qml.decomposition.disable_graph)()
    try:
        yield
    finally:
        (qml.decomposition.enable_graph if enabled else qml.decomposition.disable_graph)()


def _pauli_operator(label: str):
    ops = [
        {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[char](wire)
        for wire, char in enumerate(reversed(label)) if char != "I"
    ]
    if not ops:
        return qml.Identity(0)
    return ops[0] if len(ops) == 1 else qml.prod(*ops)


def build_lcu_tape(
    terms: Iterable[tuple[str, complex]],
    num_system_qubits: int,
    scope: str = "block",
    work_wires: int = 0,
) -> tuple[qml.tape.QuantumScript, list[int], dict[str, Any]]:
    """Build ``PrepSelPrep`` or full ``Select`` from ordered Pauli terms.

    Returned ``wire_order`` is the PennyLane matrix order (MSB first), not
    Qiskit's argument order. The system is the least-significant register, then
    workspace, then selector. All ancillas start in zero for block projection;
    SELECT's unused addresses implement identity. Zero coefficients retain their
    address with phase one. A completely zero LCU has no normalization and fails.
    """
    started = time.perf_counter()
    if scope not in {"block", "select"}:
        raise ValueError("scope must be 'block' or 'select'")
    if not isinstance(num_system_qubits, int) or num_system_qubits < 1:
        raise ValueError("num_system_qubits must be a positive integer")
    if not isinstance(work_wires, int) or work_wires < 0:
        raise ValueError("work_wires must be a nonnegative integer")
    terms = [(str(label), complex(coeff)) for label, coeff in terms]
    if not terms:
        raise ValueError("LCU must contain at least one term")
    for label, coeff in terms:
        if len(label) != num_system_qubits or set(label) - set("IXYZ"):
            raise ValueError(f"Invalid {num_system_qubits}-qubit Pauli label: {label!r}")
        if not math.isfinite(coeff.real) or not math.isfinite(coeff.imag):
            raise ValueError("LCU coefficients must be finite")
    alpha = math.fsum(abs(coeff) for _, coeff in terms)
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("LCU normalization alpha must be positive and finite")
    n_control = (len(terms) - 1).bit_length()
    system = list(range(num_system_qubits))
    work = list(range(num_system_qubits, num_system_qubits + work_wires))
    control = list(reversed(range(num_system_qubits + work_wires,
                                  num_system_qubits + work_wires + n_control)))
    wire_order = control + list(reversed(work)) + list(reversed(system))
    with qml.QueuingManager.stop_recording():
        ops = [_pauli_operator(label) for label, _ in terms]
        coeffs = [coeff for _, coeff in terms]
        if scope == "block":
            # qml.dot produces Sum, whose terms() may combine duplicate labels.
            lcu = qml.ops.LinearCombination(coeffs, ops)
            circuit_ops = [qml.PrepSelPrep(lcu, control=control)]
        else:
            phase_ops = [
                qml.prod(op, qml.GlobalPhase(-float(np.angle(coeff)) if coeff else 0.0,
                                           wires=op.wires))
                for op, coeff in zip(ops, coeffs)
            ]
            # PennyLane 0.44's no-control Select decomposition raises IndexError.
            circuit_ops = ([phase_ops[0]] if not control else
                           [qml.Select(phase_ops, control=control, work_wires=work, partial=False)])
    tape = qml.tape.QuantumScript(circuit_ops)
    metadata = {
        "pennylane_version": qml.__version__,
        "scope": scope,
        "template": "PrepSelPrep" if scope == "block" else "Select",
        "alpha": alpha,
        "num_terms": len(terms),
        "num_system_qubits": num_system_qubits,
        "num_control_qubits": n_control,
        "num_work_qubits": work_wires,
        "system_wires": system,
        "control_wires": control,
        "work_wires": work,
        "wire_order": wire_order,
        "label_order": "leftmost Pauli character acts on most-significant system qubit",
        "partial_select": scope == "block",
        "single_term_select_bypass": scope == "select" and not control,
        "construction_seconds": time.perf_counter() - started,
    }
    # QuantumScript has no wire-register metadata API; retain the adapter's
    # explicit register contract for compilation, which otherwise loses idles.
    tape._lcu_adapter_metadata = metadata
    return tape, wire_order, dict(metadata)


def _inject_select_workspace(tape, work):
    if not work:
        return tape
    stop = LEAF_GATE_SET | {"Select", "AmplitudeEmbedding", "Adjoint(AmplitudeEmbedding)"}
    with _decomposition_mode("legacy"):
        [expanded], _ = qml.transforms.decompose(tape, gate_set=stop)
    with qml.QueuingManager.stop_recording():
        operations = [
            qml.Select(op.ops, control=op.control, work_wires=work, partial=op.partial)
            if isinstance(op, qml.Select) else op
            for op in expanded.operations
        ]
    return expanded.copy(operations=operations)


def _audit_tape(tape, wire_order):
    if tape.measurements:
        raise ValueError("LCU export requires a measurement-free unitary tape")
    unknown = sorted({op.name for op in tape.operations} - LEAF_GATE_SET)
    if unknown:
        raise ValueError(f"Circuit has undecomposed or non-unitary operations: {unknown}")
    if len(wire_order) != len(set(wire_order)):
        raise ValueError("wire_order must contain each wire exactly once")
    extra = set(tape.wires) - set(wire_order)
    if extra:
        raise ValueError(f"Circuit uses unreserved wires: {extra}")


def compile_lcu_tape(
    tape: qml.tape.QuantumScript,
    wire_order: Sequence[int],
    optimize: bool = True,
    decomposition: str = "legacy",
) -> tuple[qml.tape.QuantumScript, dict[str, Any]]:
    """Decompose natively, then optionally run PennyLane's default compile passes.

    Graph mode is opt-in and restored after the call. Explicit workspace is
    inserted into the native Select inside PrepSelPrep before decomposition,
    allowing PennyLane's native unary-iterator alternatives. No additional
    dynamic workspace, reset, measurement or matrix synthesis is permitted.
    """
    started = time.perf_counter()
    source_meta = getattr(tape, "_lcu_adapter_metadata", {})
    work = list(source_meta.get("work_wires", ()))
    input_counts = dict(Counter(op.name for op in tape.operations))
    single_block_legacy = source_meta.get("num_terms") == 1 and source_meta.get("scope") == "block"
    with _decomposition_mode(decomposition):
        if single_block_legacy:
            # Native graph Select requires a control even for one term; the
            # legacy PrepSelPrep expansion correctly eliminates that register.
            with _decomposition_mode("legacy"):
                [expanded], _ = qml.transforms.decompose(tape, gate_set=LEAF_GATE_SET)
        else:
            expanded = _inject_select_workspace(tape, work)
        [expanded], _ = qml.transforms.decompose(
            expanded, gate_set=LEAF_GATE_SET, num_work_wires=0
        )
        _audit_tape(expanded, wire_order)
        decomposed_counts = dict(Counter(op.name for op in expanded.operations))
        decomposition_seconds = time.perf_counter() - started
        optimization_started = time.perf_counter()
        if optimize:
            [expanded], _ = qml.compile(expanded, basis_set=sorted(LEAF_GATE_SET))
        optimization_seconds = time.perf_counter() - optimization_started
        _audit_tape(expanded, wire_order)
    expanded._lcu_adapter_metadata = dict(source_meta)
    metadata = {
        "decomposition": decomposition,
        "optimize": optimize,
        "pipeline": list(DEFAULT_PIPELINE) if optimize else [],
        "num_passes": 1 if optimize else 0,
        "leaf_gate_set": sorted(LEAF_GATE_SET),
        "input_operation_counts": input_counts,
        "decomposed_operation_counts": decomposed_counts,
        "output_operation_counts": dict(Counter(op.name for op in expanded.operations)),
        "workspace_injected_into_native_select": bool(work) and not single_block_legacy,
        "single_term_block_legacy_decomposition": single_block_legacy,
        "explicit_work_wires": work,
        "dynamic_work_wire_budget": 0,
        "decomposition_seconds": decomposition_seconds,
        "optimization_seconds": optimization_seconds,
        "compile_seconds": time.perf_counter() - started,
    }
    return expanded, metadata


def tape_to_qiskit(tape: qml.tape.QuantumScript, wire_order: Sequence[int]):
    """Export audited leaf operations, preserving the exact global phase.

    Qiskit qubits are reversed relative to ``wire_order`` so ``Operator(qc)``
    equals ``qml.matrix(tape, wire_order=wire_order)`` without bit permutation.
    The returned circuit is logical and must be lowered to NAM/IBM at opt0 for
    basis counts. OpenQASM 2 cannot encode its global phase; record that metadata
    separately if exporting QASM for counting.
    """
    from qiskit import QuantumCircuit

    _audit_tape(tape, wire_order)
    circuit = QuantumCircuit(len(wire_order))
    positions = {wire: idx for idx, wire in enumerate(reversed(wire_order))}
    methods = {
        "Identity": "id", "PauliX": "x", "PauliY": "y", "PauliZ": "z",
        "Hadamard": "h", "S": "s", "T": "t", "SX": "sx", "RX": "rx",
        "RY": "ry", "RZ": "rz", "PhaseShift": "p", "CNOT": "cx",
        "CY": "cy", "CZ": "cz", "SWAP": "swap",
    }
    for op in tape.operations:
        params = [float(value) for value in op.parameters]
        if op.name == "GlobalPhase":
            # PennyLane exp(-i phi) versus Qiskit exp(+i global_phase).
            circuit.global_phase -= params[0]
        elif op.name == "Identity":
            continue
        else:
            getattr(circuit, methods[op.name])(*params, *(positions[w] for w in op.wires))
    circuit.metadata = {
        "source": "PennyLane native LCU decomposition",
        "pennylane_version": qml.__version__,
        "pennylane_wire_order_msb_first": list(wire_order),
        "qasm2_global_phase_requires_sidecar": float(circuit.global_phase),
        "omitted_identity_operations": sum(op.name == "Identity" for op in tape.operations),
    }
    return circuit
