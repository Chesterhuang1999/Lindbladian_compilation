#!/usr/bin/env python3
"""Run Qiskit IBM opt0/opt3 baselines for hypercube channel-LCU circuits.

The raw channel-LCU circuit is built in memory.  QASM files are written only
after transpilation, for counting and reproducibility; they are not used as
inputs to the Qiskit opt3 transpilation path.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from qiskit import transpile
from qiskit.circuit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import compare_gate_stats, count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import channel_ensemble  # type: ignore  # noqa: E402
from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore  # noqa: E402
import channel_LCU  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


IBM_BASIS = ("u1", "u2", "u3", "cx")
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "hypercube_random_walk_qiskit_ibm_opt3"


def build_hypercube_channel(n: int) -> list:
    ensemble = build_hypercube_section52_channel(n)
    return ensemble.channels[0][1]


def with_new_block_encoding(func):
    old_block_encoding = channel_LCU.BlockEncoding
    channel_LCU.BlockEncoding = NewBlockEncoding
    try:
        return func()
    finally:
        channel_LCU.BlockEncoding = old_block_encoding


def build_raw_channel_lcu_circuit(kraus_ops: list) -> tuple[QuantumCircuit, Any, float]:
    started = time.perf_counter()

    def _build():
        return channel_LCU.channel_to_LCU(
            channel_ensemble([copy.deepcopy(kraus_ops)]),
            structure="basic",
            opt="No",
        )

    qc, qubit_indexes = with_new_block_encoding(_build)
    return qc, qubit_indexes, time.perf_counter() - started


def channel_metrics(kraus_ops: list) -> dict[str, int]:
    nonzero = [op for op in kraus_ops if len(op.instances) > 0]
    return {
        "kraus_count": int(len(nonzero)),
        "pauli_terms_total": int(sum(len(op.instances) for op in nonzero)),
        "max_terms_per_kraus": int(max((len(op.instances) for op in nonzero), default=0)),
        "system_qubits": int(max((op.size for op in nonzero), default=0)),
    }


def transpile_export_count(
    qc: QuantumCircuit,
    *,
    optimization_level: int,
    qasm_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    tqc = transpile(
        qc,
        basis_gates=list(IBM_BASIS),
        optimization_level=optimization_level,
    )
    transpile_time = time.perf_counter() - started

    export = export_openqasm2_quartz_input(tqc, qasm_path, basis_gates=IBM_BASIS)
    stats = count_qasm_file(qasm_path, gate_set="ibm")
    return {
        "optimization_level": int(optimization_level),
        "transpile_time_s": float(transpile_time),
        "qasm": str(qasm_path),
        "export": export,
        "stats": stats,
    }


def compare_with_canonical_opt0(n: int, opt0_stats: dict[str, Any]) -> dict[str, Any] | None:
    canonical_path = (
        ROOT
        / "circuits"
        / "Table1"
        / f"hypercube_random_walk_n{n}"
        / f"hypercube_random_walk_n{n}_basic_raw_ibm_opt0.qasm"
    )
    if not canonical_path.exists():
        return None

    canonical_stats = count_qasm_file(canonical_path, gate_set="ibm")
    fields_equal = {
        "num_qubits": int(opt0_stats["num_qubits"]) == int(canonical_stats["num_qubits"]),
        "metric_total": int(opt0_stats["metric_total"]) == int(canonical_stats["metric_total"]),
        "non_clifford": int(opt0_stats["clifford"]["non_clifford"])
        == int(canonical_stats["clifford"]["non_clifford"]),
        "ops": opt0_stats["ops"] == canonical_stats["ops"],
    }
    all_ops = sorted(set(opt0_stats["ops"]) | set(canonical_stats["ops"]))
    return {
        "canonical_qasm": str(canonical_path),
        "equal": bool(all(fields_equal.values())),
        "fields_equal": fields_equal,
        "diff_memory_minus_canonical": {
            "num_qubits": int(opt0_stats["num_qubits"]) - int(canonical_stats["num_qubits"]),
            "metric_total": int(opt0_stats["metric_total"]) - int(canonical_stats["metric_total"]),
            "non_clifford": int(opt0_stats["clifford"]["non_clifford"])
            - int(canonical_stats["clifford"]["non_clifford"]),
            "ops": {
                gate: int(opt0_stats["ops"].get(gate, 0))
                - int(canonical_stats["ops"].get(gate, 0))
                for gate in all_ops
                if int(opt0_stats["ops"].get(gate, 0))
                != int(canonical_stats["ops"].get(gate, 0))
            },
        },
    }


def run_one(n: int, out_dir: Path) -> dict[str, Any]:
    example_name = f"hypercube_random_walk_n{n}"
    example_dir = out_dir / example_name
    example_dir.mkdir(parents=True, exist_ok=True)

    kraus_ops = build_hypercube_channel(n)
    raw_qc, qubit_indexes, build_time = build_raw_channel_lcu_circuit(kraus_ops)

    opt0 = transpile_export_count(
        raw_qc,
        optimization_level=0,
        qasm_path=example_dir / f"{example_name}_basic_raw_ibm_opt0.qasm",
    )
    opt3 = transpile_export_count(
        raw_qc,
        optimization_level=3,
        qasm_path=example_dir / f"{example_name}_basic_raw_ibm_opt3_qiskit.qasm",
    )

    opt0_total = int(opt0["stats"]["metric_total"])
    opt3_total = int(opt3["stats"]["metric_total"])
    ratio = None if opt0_total == 0 else opt3_total / opt0_total

    return {
        "example": example_name,
        "num_system_qubits": int(n),
        "built_from_memory": True,
        "qasm_used_as_transpile_input": False,
        "basis": list(IBM_BASIS),
        "channel_metrics": channel_metrics(kraus_ops),
        "raw_circuit": {
            "num_qubits": int(raw_qc.num_qubits),
            "depth": int(raw_qc.depth()),
            "size": int(raw_qc.size()),
            "ops": {str(gate): int(count) for gate, count in raw_qc.count_ops().items()},
            "qubit_indexes": qubit_indexes,
            "build_time_s": float(build_time),
        },
        "qiskit_opt0_ibm": opt0,
        "qiskit_opt3_ibm": opt3,
        "comparison_vs_opt0": compare_gate_stats(
            opt0["stats"],
            opt3["stats"],
            metric_gate_set="ibm",
        ),
        "opt3_over_opt0_ratio": ratio,
        "canonical_opt0_match": compare_with_canonical_opt0(n, opt0["stats"]),
    }


def format_int(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Example | Qubits | IBM opt0 gates | IBM opt3 gates | opt3 / opt0 | Non-Clifford opt0 | Non-Clifford opt3 | opt3 time (s) | canonical opt0 match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        opt0 = result["qiskit_opt0_ibm"]
        opt3 = result["qiskit_opt3_ibm"]
        match = result.get("canonical_opt0_match")
        match_cell = "-" if match is None else str(bool(match["equal"]))
        ratio = result["opt3_over_opt0_ratio"]
        lines.append(
            f"| HRW n={result['num_system_qubits']} "
            f"| {opt0['stats']['num_qubits']} "
            f"| {format_int(int(opt0['stats']['metric_total']))} "
            f"| {format_int(int(opt3['stats']['metric_total']))} "
            f"| {'-' if ratio is None else f'{100 * ratio:.3f}%'} "
            f"| {format_int(int(opt0['stats']['clifford']['non_clifford']))} "
            f"| {format_int(int(opt3['stats']['clifford']['non_clifford']))} "
            f"| {opt3['transpile_time_s']:.3f} "
            f"| {match_cell} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run in-memory Qiskit IBM opt0/opt3 baselines for hypercube channel-LCU."
    )
    parser.add_argument("--num-qubits", nargs="+", type=int, default=[4, 8, 12, 20, 28])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n in args.num_qubits:
        print(f"[qiskit-ibm] building/running hypercube_random_walk_n{n}", flush=True)
        result = run_one(n, out_dir)
        results.append(result)
        print(
            f"[qiskit-ibm] n={n}: opt0={result['qiskit_opt0_ibm']['stats']['metric_total']:,}, "
            f"opt3={result['qiskit_opt3_ibm']['stats']['metric_total']:,}, "
            f"ratio={100 * result['opt3_over_opt0_ratio']:.3f}%",
            flush=True,
        )

    summary = {
        "experiment": "hypercube_random_walk_qiskit_ibm_opt3",
        "basis": list(IBM_BASIS),
        "built_from_memory": True,
        "qasm_used_as_transpile_input": False,
        "results": results,
        "markdown_table": markdown_table(results),
    }

    summary_path = out_dir / "hypercube_random_walk_qiskit_ibm_opt3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print()
    print(summary["markdown_table"])
    print()
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
