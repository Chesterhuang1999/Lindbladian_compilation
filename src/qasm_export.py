from __future__ import annotations

from pathlib import Path
from typing import Iterable

from qiskit import QuantumCircuit, qasm2, qasm3, transpile


DEFAULT_BASELINE_BASIS_GATES = ("u1", "u2", "u3", "cx", "reset")
DEFAULT_BASELINE_OPTIMIZATION_LEVEL = 0
DEFAULT_QUARTZ_BASIS_GATES = ("u1", "u2", "u3", "cx")


def _default_quartz_out_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_quartz_input{out_path.suffix}")


def _single_qreg_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Return an equivalent circuit with one flat q register for Quartz."""
    flat = QuantumCircuit(qc.num_qubits, name=qc.name)
    flat.global_phase = qc.global_phase
    qubit_indices = {qubit: index for index, qubit in enumerate(qc.qubits)}

    for instruction in qc.data:
        if instruction.clbits:
            raise ValueError("Quartz QASM export does not support classical bits.")
        flat.append(
            instruction.operation.copy(),
            [qubit_indices[qubit] for qubit in instruction.qubits],
        )

    return flat


def export_openqasm2_quartz_input(
    qc: QuantumCircuit,
    out_path: str | Path,
    basis_gates: Iterable[str] = DEFAULT_QUARTZ_BASIS_GATES,
) -> dict[str, object]:
    """Export a Quartz-readable OpenQASM 2 file with a single qreg q[N]."""
    basis_gate_list = list(basis_gates)
    unsupported_ops = sorted(set(qc.count_ops()) - set(basis_gate_list))
    if unsupported_ops:
        raise ValueError(
            "Quartz QASM export only supports "
            f"{basis_gate_list}; found unsupported ops: {unsupported_ops}."
        )

    flat = _single_qreg_circuit(qc)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        qasm2.dump(flat, out_file)

    ops = {str(gate): int(count) for gate, count in flat.count_ops().items()}
    return {
        "path": str(out_path),
        "basis_gates": basis_gate_list,
        "num_qubits": int(flat.num_qubits),
        "depth": int(flat.depth()),
        "size": int(flat.size()),
        "cx": int(ops.get("cx", 0)),
        "u1": int(ops.get("u1", 0)),
        "u2": int(ops.get("u2", 0)),
        "u3": int(ops.get("u3", 0)),
        "ops": ops,
    }


def export_openqasm3_baseline(
    qc: QuantumCircuit,
    out_path: str | Path,
    basis_gates: Iterable[str] = DEFAULT_BASELINE_BASIS_GATES,
    optimization_level: int = DEFAULT_BASELINE_OPTIMIZATION_LEVEL,
    export_quartz_input: bool = True,
    quartz_out_path: str | Path | None = None,
    quartz_basis_gates: Iterable[str] = DEFAULT_QUARTZ_BASIS_GATES,
) -> dict[str, object]:
    """Transpile a baseline circuit and export it as OpenQASM 3.

    When possible, also writes a Quartz-readable OpenQASM 2 companion file with
    a single qreg q[N]. Quartz's parser is more reliable on that flattened form
    than on multi-register OpenQASM 3 such as ctrl/sys declarations.
    """
    basis_gate_list = list(basis_gates)
    tqc = transpile(
        qc,
        basis_gates=basis_gate_list,
        optimization_level=optimization_level,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        qasm3.dump(tqc, out_file)

    ops = {str(gate): int(count) for gate, count in tqc.count_ops().items()}
    quartz_input: dict[str, object] | None = None
    if export_quartz_input:
        quartz_gate_list = list(quartz_basis_gates)
        unsupported_quartz_ops = sorted(set(ops) - set(quartz_gate_list))
        if unsupported_quartz_ops:
            quartz_input = {
                "written": False,
                "unsupported_ops": unsupported_quartz_ops,
                "basis_gates": quartz_gate_list,
            }
        else:
            quartz_path = (
                Path(quartz_out_path)
                if quartz_out_path is not None
                else _default_quartz_out_path(out_path)
            )
            quartz_input = export_openqasm2_quartz_input(
                tqc,
                quartz_path,
                basis_gates=quartz_gate_list,
            )
            quartz_input["written"] = True

    return {
        "path": str(out_path),
        "basis_gates": basis_gate_list,
        "optimization_level": int(optimization_level),
        "num_qubits": int(tqc.num_qubits),
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx": int(ops.get("cx", 0)),
        "u1": int(ops.get("u1", 0)),
        "u2": int(ops.get("u2", 0)),
        "u3": int(ops.get("u3", 0)),
        "reset": int(ops.get("reset", 0)),
        "ops": ops,
        "quartz_input": quartz_input,
    }
