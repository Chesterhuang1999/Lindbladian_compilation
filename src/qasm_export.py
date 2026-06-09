from __future__ import annotations

from pathlib import Path
from typing import Iterable

from qiskit import QuantumCircuit, qasm3, transpile


DEFAULT_BASELINE_BASIS_GATES = ("cx", "u3", "u2", "u1")
DEFAULT_BASELINE_OPTIMIZATION_LEVEL = 0


def export_openqasm3_baseline(
    qc: QuantumCircuit,
    out_path: str | Path,
    basis_gates: Iterable[str] = DEFAULT_BASELINE_BASIS_GATES,
    optimization_level: int = DEFAULT_BASELINE_OPTIMIZATION_LEVEL,
) -> dict[str, object]:
    """Transpile a baseline circuit and export it as OpenQASM 3."""
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
    return {
        "path": str(out_path),
        "basis_gates": basis_gate_list,
        "optimization_level": int(optimization_level),
        "num_qubits": int(tqc.num_qubits),
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx": int(ops.get("cx", 0)),
        "u3": int(ops.get("u3", 0)),
        "ops": ops,
    }
