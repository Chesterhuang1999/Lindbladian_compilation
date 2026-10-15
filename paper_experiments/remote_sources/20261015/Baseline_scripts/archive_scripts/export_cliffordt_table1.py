#!/usr/bin/env python3
"""Export table1 SELECT circuits to approximate Clifford+T QASM2."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qasm2
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SolovayKitaev


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_INPUT = ROOT / "circuits" / "select_only" / "test_table1_m_no_n4_nam_qasm2.qasm"
DEFAULT_DIRECT_OUTPUT = (
    ROOT / "circuits" / "select_only" / "test_table1_m_no_n4_cliffordt_direct_qasm2_sk1.qasm"
)
DEFAULT_DIRECT_SUMMARY = (
    ROOT
    / "Baseline_results"
    / "quizx"
    / "cliffordt_direct"
    / "TFIM_4"
    / "test_table1_TFIM_cliffordt_direct_export_summary.json"
)
DEFAULT_NAM_OUTPUT = ROOT / "circuits" / "select_only" / "test_table1_m_no_n4_cliffordt_qasm2_sk1.qasm"
DEFAULT_NAM_SUMMARY = (
    ROOT
    / "Baseline_results"
    / "quizx"
    / "cliffordt"
    / "TFIM_4"
    / "test_table1_TFIM_cliffordt_export_summary.json"
)
CLIFFORD_T_GATES = {"h", "x", "z", "s", "sdg", "t", "tdg", "cx"}


def gate_counts(qc: QuantumCircuit) -> dict[str, Any]:
    ops = {str(gate): int(count) for gate, count in qc.count_ops().items()}
    return {
        "num_qubits": int(qc.num_qubits),
        "depth": int(qc.depth()),
        "total": int(qc.size()),
        "ops": dict(sorted(ops.items())),
    }


def build_direct_original_select_circuit(num_qubits: int, metrics: str, delta_t: float) -> QuantumCircuit:
    import sys

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from test_table1 import build_tfim_lcu_select_circuit

    return build_tfim_lcu_select_circuit(num_qubits, metrics=metrics, delta_t=delta_t)


def synthesize_decomposed_circuit_to_cliffordt(
    input_qc: QuantumCircuit,
    recursion_degree: int,
    sk_depth: int,
    decompose_reps: int,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    decomposed_qc = input_qc.decompose(reps=decompose_reps)
    pass_manager = PassManager(
        [
            SolovayKitaev(
                recursion_degree=recursion_degree,
                basis_gates=["h", "t", "tdg"],
                depth=sk_depth,
            )
        ]
    )
    output_qc = pass_manager.run(decomposed_qc)
    unsupported = sorted(set(output_qc.count_ops()) - {"h", "t", "tdg", "cx"})
    if unsupported:
        raise ValueError(f"Direct Solovay-Kitaev output contains unsupported gates: {unsupported}")
    return output_qc, {
        "decompose_reps": int(decompose_reps),
        "decomposed": gate_counts(decomposed_qc),
        "unsupported_output_gates": unsupported,
    }


def append_phase_mod_pi_over_4(qc: QuantumCircuit, qubit: int, k: int) -> int:
    """Append P(k*pi/4), equivalent to Rz(k*pi/4) up to global phase."""
    k %= 8
    if k == 0:
        return 0
    if k == 1:
        qc.t(qubit)
        return 1
    if k == 2:
        qc.s(qubit)
        return 1
    if k == 3:
        qc.s(qubit)
        qc.t(qubit)
        return 2
    if k == 4:
        qc.z(qubit)
        return 1
    if k == 5:
        qc.z(qubit)
        qc.t(qubit)
        return 2
    if k == 6:
        qc.sdg(qubit)
        return 1
    qc.tdg(qubit)
    return 1


def append_one_qubit_gate(qc: QuantumCircuit, gate_name: str, qubit: int) -> None:
    if gate_name == "h":
        qc.h(qubit)
    elif gate_name == "x":
        qc.x(qubit)
    elif gate_name == "z":
        qc.z(qubit)
    elif gate_name == "s":
        qc.s(qubit)
    elif gate_name == "sdg":
        qc.sdg(qubit)
    elif gate_name == "t":
        qc.t(qubit)
    elif gate_name == "tdg":
        qc.tdg(qubit)
    else:
        raise ValueError(f"Unsupported Clifford+T 1q gate from synthesis: {gate_name}")


def synthesize_rz_to_cliffordt(
    theta: float,
    pass_manager: PassManager,
    cache: dict[float, list[str]],
) -> list[str]:
    key = round(float(theta), 15)
    if key in cache:
        return cache[key]

    circuit = QuantumCircuit(1)
    circuit.rz(theta, 0)
    synthesized = pass_manager.run(circuit)
    sequence = [inst.operation.name.lower() for inst in synthesized.data]
    unsupported = sorted(set(sequence) - (CLIFFORD_T_GATES - {"cx"}))
    if unsupported:
        raise ValueError(f"Solovay-Kitaev produced unsupported gates: {unsupported}")
    cache[key] = sequence
    return sequence


def convert_to_cliffordt(
    input_qc: QuantumCircuit,
    recursion_degree: int,
    sk_depth: int,
    exact_tol: float,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    output_qc = QuantumCircuit(input_qc.num_qubits, name=f"{input_qc.name}_cliffordt")
    pass_manager = PassManager(
        [
            SolovayKitaev(
                recursion_degree=recursion_degree,
                basis_gates=["h", "t", "tdg"],
                depth=sk_depth,
            )
        ]
    )
    sk_cache: dict[float, list[str]] = {}
    stats: dict[str, Any] = {
        "exact_rz_pi_over_4": 0,
        "exact_rz_removed": 0,
        "approximated_rz": 0,
        "sk_unique_angles": {},
        "sk_inserted_gates": 0,
    }

    for instruction in input_qc.data:
        op = instruction.operation
        name = op.name.lower()
        qubits = [input_qc.find_bit(qubit).index for qubit in instruction.qubits]

        if name in {"h", "x", "z", "s", "sdg", "t", "tdg"}:
            append_one_qubit_gate(output_qc, name, qubits[0])
        elif name == "cx":
            output_qc.cx(qubits[0], qubits[1])
        elif name == "rz":
            theta = float(op.params[0])
            ratio = theta / (math.pi / 4)
            nearest = round(ratio)
            if math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=exact_tol):
                inserted = append_phase_mod_pi_over_4(output_qc, qubits[0], nearest)
                stats["exact_rz_pi_over_4"] += 1
                if inserted == 0:
                    stats["exact_rz_removed"] += 1
            else:
                sequence = synthesize_rz_to_cliffordt(theta, pass_manager, sk_cache)
                for gate_name in sequence:
                    append_one_qubit_gate(output_qc, gate_name, qubits[0])
                stats["approximated_rz"] += 1
                stats["sk_inserted_gates"] += len(sequence)
        else:
            raise ValueError(f"Unsupported input gate for Clifford+T export: {name}")

    stats["sk_unique_angles"] = {
        f"{angle:.15g}": len(sequence) for angle, sequence in sorted(sk_cache.items())
    }
    return output_qc, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Approximate table1 SELECT circuits as Clifford+T QASM2."
    )
    parser.add_argument(
        "--source",
        choices=["direct-original", "nam-qasm"],
        default="direct-original",
        help=(
            "direct-original builds the original SELECT circuit from scripts/test_table1.py "
            "and decomposes its multi-control structure before Solovay-Kitaev. nam-qasm "
            "converts the already-Nam-transpiled QASM input."
        ),
    )
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--metrics", default="m_no")
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--recursion-degree", type=int, default=1)
    parser.add_argument("--sk-depth", type=int, default=12)
    parser.add_argument("--decompose-reps", type=int, default=10)
    parser.add_argument("--exact-tol", type=float, default=1e-10)
    args = parser.parse_args()

    if args.output is None:
        args.output = DEFAULT_DIRECT_OUTPUT if args.source == "direct-original" else DEFAULT_NAM_OUTPUT
    if args.summary is None:
        args.summary = DEFAULT_DIRECT_SUMMARY if args.source == "direct-original" else DEFAULT_NAM_SUMMARY

    if args.source == "direct-original":
        input_qc = build_direct_original_select_circuit(
            num_qubits=args.num_qubits,
            metrics=args.metrics,
            delta_t=args.delta_t,
        )
        output_qc, conversion_stats = synthesize_decomposed_circuit_to_cliffordt(
            input_qc,
            recursion_degree=args.recursion_degree,
            sk_depth=args.sk_depth,
            decompose_reps=args.decompose_reps,
        )
        input_label = f"test_table1.py::{args.metrics}, n={args.num_qubits}, SELECT"
        note = (
            "This export starts from the original SELECT circuit with multi-control "
            "structure, decomposes its custom gates, and applies Qiskit's Solovay-Kitaev "
            "pass to synthesize one-qubit gates into Clifford+T."
        )
    else:
        input_qc = QuantumCircuit.from_qasm_file(str(args.input))
        output_qc, conversion_stats = convert_to_cliffordt(
            input_qc,
            recursion_degree=args.recursion_degree,
            sk_depth=args.sk_depth,
            exact_tol=args.exact_tol,
        )
        input_label = str(args.input)
        note = (
            "This export starts from already-Nam-transpiled QASM. Rz(k*pi/4) gates are "
            "converted exactly up to global phase. Other Rz angles are approximated by "
            "Qiskit's Solovay-Kitaev pass."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(qasm2.dumps(output_qc))

    summary = {
        "source": args.source,
        "input": input_label,
        "output": str(args.output),
        "num_qubits": int(args.num_qubits),
        "metrics": args.metrics,
        "delta_t": float(args.delta_t),
        "recursion_degree": int(args.recursion_degree),
        "sk_depth": int(args.sk_depth),
        "decompose_reps": int(args.decompose_reps),
        "exact_tol": float(args.exact_tol),
        "before": gate_counts(input_qc),
        "after": gate_counts(output_qc),
        "conversion": conversion_stats,
        "note": note,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
