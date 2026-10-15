#!/usr/bin/env python3
"""Run the hypercube_random_walk_n4 channel-LCU experiment in the NAM basis."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import qasm2, transpile
from qiskit.circuit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import channel as ChannelIR, channel_ensemble  # type: ignore  # noqa: E402
from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore  # noqa: E402
import channel_LCU  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


NAM_BASIS = ("h", "x", "rz", "cx")
NAM_TRANSPILER_BASIS = ("h", "x", "rz", "cx", "reset")
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "hypercube_random_walk_n4_ours_nam"
DEFAULT_TABLE1_RAW_QASM = (
    ROOT
    / "circuits"
    / "Table1"
    / "hypercube_random_walk_n4"
    / "hypercube_random_walk_n4_basic_raw.qasm"
)


def build_hypercube_channel(n: int) -> list:
    ensemble = build_hypercube_section52_channel(n)
    return ensemble.channels[0][1]


def channel_metrics(kraus_ops: list) -> dict[str, int]:
    nonzero = [op for op in kraus_ops if len(op.instances) > 0]
    return {
        "kraus_count": int(len(nonzero)),
        "pauli_terms_total": int(sum(len(op.instances) for op in nonzero)),
        "max_terms_per_kraus": int(max((len(op.instances) for op in nonzero), default=0)),
        "system_qubits": int(max((op.size for op in nonzero), default=0)),
    }


def rewrite_channel(
    kraus_ops: list,
    *,
    strategy: str,
    beam_width: int,
    max_steps: int,
    tol: float,
) -> tuple[list, dict[str, Any], float]:
    started = time.perf_counter()
    rewrite_source = ChannelIR(copy.deepcopy(kraus_ops))
    rewrite_source.zero_elim()
    result = rewrite_source.rewrite_search(
        strategy=strategy,
        beam_width=beam_width,
        max_steps=max_steps,
        tol=tol,
    )

    rewritten = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten.apply_rewrite_result(result, tol=tol)
    rewritten.zero_elim()
    elapsed = time.perf_counter() - started
    return rewritten.kraus_ops, result, elapsed


def with_new_block_encoding(func):
    old_block_encoding = channel_LCU.BlockEncoding
    channel_LCU.BlockEncoding = NewBlockEncoding
    try:
        return func()
    finally:
        channel_LCU.BlockEncoding = old_block_encoding


def build_channel_lcu_circuit(kraus_ops: list, *, structure: str, opt: str) -> tuple[QuantumCircuit, Any, float]:
    started = time.perf_counter()

    def _build():
        return channel_LCU.channel_to_LCU(
            channel_ensemble([copy.deepcopy(kraus_ops)]),
            structure=structure,
            opt=opt,
        )

    qc, qubit_indexes = with_new_block_encoding(_build)
    elapsed = time.perf_counter() - started
    return qc, qubit_indexes, elapsed


def build_rewrite_opt_matrix_order_circuit(kraus_ops: list) -> tuple[QuantumCircuit, Any, float]:
    """Build channel-level ctrl-line + per-Kraus Matrix-order circuit."""
    started = time.perf_counter()
    sys_size = max(ms.size for ms in kraus_ops)
    select_size = int(np.ceil(np.log2(len(kraus_ops))))
    coeff_sums = [ms.pauli_norm() for ms in kraus_ops]

    lcu_ini = channel_LCU.prep_sup_state(coeff_sums)
    circuits = [
        NewBlockEncoding(copy.deepcopy(ms)).circuit(opt="Matrix-order")
        for ms in kraus_ops
    ]
    ctrl_size = max(circ.num_qubits - sys_size for circ in circuits)

    qc = QuantumCircuit(2 * select_size + ctrl_size + sys_size)
    select_indexes = [2 * j for j in range(select_size)]
    anc_indexes = [2 * j + 1 for j in range(select_size)]
    ctrl_indexes = list(range(2 * select_size, 2 * select_size + ctrl_size))
    sys_indexes = list(range(2 * select_size + ctrl_size, 2 * select_size + ctrl_size + sys_size))

    qc.compose(lcu_ini, qubits=select_indexes, inplace=True)
    qc, _tcount, _mccount, _cxcount = channel_LCU.mulplex_U_opt(
        qc,
        circuits,
        select_size,
        ctrl_size,
        sys_size,
    )
    elapsed = time.perf_counter() - started
    return qc, [select_indexes, anc_indexes, ctrl_indexes, sys_indexes], elapsed


def dump_untranspiled_qasm(qc: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        qasm2.dump(qc, out_file)


def export_nam_count_qasm(
    qc: QuantumCircuit,
    path: Path,
    *,
    optimization_level: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    tqc = transpile(
        qc,
        basis_gates=list(NAM_TRANSPILER_BASIS),
        optimization_level=optimization_level,
    )
    elapsed = time.perf_counter() - started
    export_openqasm2_quartz_input(
        tqc,
        path,
        basis_gates=NAM_TRANSPILER_BASIS,
    )
    return count_qasm_file(path, gate_set="nam"), elapsed


def export_nam_count_qasm_from_file(
    source_qasm: Path,
    path: Path,
    *,
    optimization_level: int,
) -> tuple[dict[str, Any], float]:
    source_text = source_qasm.read_text(encoding="utf-8")
    # Qiskit's legacy qelib parser treats "s" as a built-in gate name.  The
    # Table1 raw file uses qreg s[3], so rename that register in-memory only.
    source_text = source_text.replace("qreg s[", "qreg sel[")
    source_text = source_text.replace(" s[", " sel[")
    source_text = source_text.replace(",s[", ",sel[")
    qc = qasm2.loads(
        source_text,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    return export_nam_count_qasm(qc, path, optimization_level=optimization_level)


def row_from_stats(name: str, stats: dict[str, Any], compile_time: float | None) -> dict[str, Any]:
    return {
        "name": name,
        "qubits": int(stats["num_qubits"]),
        "nam_total": int(stats["nam"]["rz_clifford_total"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "reset": int(stats["ops"].get("reset", 0)),
        "compile_time_s": None if compile_time is None else float(compile_time),
        "ops": stats["ops"],
        "clifford": stats["clifford"],
    }


def compare_stats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    fields_equal = {
        "num_qubits": int(a["num_qubits"]) == int(b["num_qubits"]),
        "nam_total": int(a["nam"]["rz_clifford_total"]) == int(b["nam"]["rz_clifford_total"]),
        "non_clifford": int(a["clifford"]["non_clifford"]) == int(b["clifford"]["non_clifford"]),
        "ops": a["ops"] == b["ops"],
    }
    all_ops = sorted(set(a["ops"]) | set(b["ops"]))
    return {
        "equal": all(fields_equal.values()),
        "fields_equal": fields_equal,
        "diff": {
            "num_qubits": int(a["num_qubits"]) - int(b["num_qubits"]),
            "nam_total": int(a["nam"]["rz_clifford_total"]) - int(b["nam"]["rz_clifford_total"]),
            "non_clifford": int(a["clifford"]["non_clifford"]) - int(b["clifford"]["non_clifford"]),
            "ops": {
                gate: int(a["ops"].get(gate, 0)) - int(b["ops"].get(gate, 0))
                for gate in all_ops
                if int(a["ops"].get(gate, 0)) != int(b["ops"].get(gate, 0))
            },
        },
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Circuit | Qubits | NAM total | Non-Clifford | Reset | Compilation time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        time_cell = "-" if row["compile_time_s"] is None else f"{row['compile_time_s']:.3f}"
        lines.append(
            f"| {row['name']} | {row['qubits']} | {row['nam_total']} | "
            f"{row['non_clifford']} | {row['reset']} | {time_cell} |"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_kraus_ops = build_hypercube_channel(args.num_qubits)
    raw_channel_metrics = channel_metrics(raw_kraus_ops)

    raw_qc, raw_qubit_indexes, raw_build_time = build_channel_lcu_circuit(
        raw_kraus_ops,
        structure="basic",
        opt="No",
    )
    raw_qasm = out_dir / f"hypercube_random_walk_n{args.num_qubits}_basic_raw.qasm"
    raw_nam_qasm = out_dir / f"hypercube_random_walk_n{args.num_qubits}_basic_raw_nam_from_memory.qasm"
    dump_untranspiled_qasm(raw_qc, raw_qasm)
    raw_nam_stats, raw_nam_time = export_nam_count_qasm(
        raw_qc,
        raw_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )

    table1_raw_nam_qasm = out_dir / f"hypercube_random_walk_n{args.num_qubits}_basic_raw_nam_from_table1_qasm.qasm"
    table1_raw_nam_stats, table1_raw_nam_time = export_nam_count_qasm_from_file(
        Path(args.table1_raw_qasm),
        table1_raw_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )
    raw_path_comparison = compare_stats(raw_nam_stats, table1_raw_nam_stats)

    rewritten_kraus_ops, rewrite_result, rewrite_time = rewrite_channel(
        raw_kraus_ops,
        strategy=args.rewrite_strategy,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        tol=args.tol,
    )
    rewrite_channel_metrics = channel_metrics(rewritten_kraus_ops)

    rewrite_qc, rewrite_qubit_indexes, rewrite_build_time = build_channel_lcu_circuit(
        rewritten_kraus_ops,
        structure="basic",
        opt="No",
    )
    rewrite_qasm = out_dir / f"hypercube_random_walk_n{args.num_qubits}_basic_rewrite.qasm"
    rewrite_nam_qasm = out_dir / f"hypercube_random_walk_n{args.num_qubits}_basic_rewrite_nam.qasm"
    dump_untranspiled_qasm(rewrite_qc, rewrite_qasm)
    rewrite_nam_stats, rewrite_nam_time = export_nam_count_qasm(
        rewrite_qc,
        rewrite_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )

    ours_qc, ours_qubit_indexes, ours_build_time = build_rewrite_opt_matrix_order_circuit(
        rewritten_kraus_ops
    )
    ours_qasm = out_dir / f"hypercube_random_walk_n{args.num_qubits}_rewrite_opt_matrix_order.qasm"
    ours_nam_qasm = out_dir / f"hypercube_random_walk_n{args.num_qubits}_rewrite_opt_matrix_order_nam.qasm"
    dump_untranspiled_qasm(ours_qc, ours_qasm)
    ours_nam_stats, ours_nam_time = export_nam_count_qasm(
        ours_qc,
        ours_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )
    ours_compile_time = rewrite_time + ours_build_time + ours_nam_time

    rows = [
        row_from_stats("Raw channel-LCU (memory -> NAM)", raw_nam_stats, None),
        row_from_stats("Rewrite channel-LCU (basic/no)", rewrite_nam_stats, rewrite_time + rewrite_build_time + rewrite_nam_time),
        row_from_stats("Ours (rewrite + opt + matrix-order)", ours_nam_stats, ours_compile_time),
    ]

    summary = {
        "experiment": "hypercube_random_walk_n4_ours_nam",
        "num_qubits": int(args.num_qubits),
        "nam_basis_counted": list(NAM_BASIS),
        "nam_transpiler_basis": list(NAM_TRANSPILER_BASIS),
        "nam_optimization_level": int(args.nam_optimization_level),
        "raw": {
            "channel_metrics": raw_channel_metrics,
            "build_time_s": float(raw_build_time),
            "nam_transpile_time_s": float(raw_nam_time),
            "qasm": str(raw_qasm),
            "nam_qasm": str(raw_nam_qasm),
            "qubit_indexes": raw_qubit_indexes,
            "stats": raw_nam_stats,
        },
        "raw_table1_qasm_path": {
            "source_qasm": str(Path(args.table1_raw_qasm)),
            "nam_transpile_time_s": float(table1_raw_nam_time),
            "nam_qasm": str(table1_raw_nam_qasm),
            "stats": table1_raw_nam_stats,
        },
        "raw_path_comparison": raw_path_comparison,
        "rewrite": {
            "channel_metrics": rewrite_channel_metrics,
            "time_s": float(rewrite_time),
            "initial_support": int(rewrite_result["initial_support"]),
            "final_support": int(rewrite_result["final_support"]),
            "steps": int(len(rewrite_result["steps"])),
            "termination": rewrite_result.get("termination", {}),
            "build_time_s": float(rewrite_build_time),
            "nam_transpile_time_s": float(rewrite_nam_time),
            "qasm": str(rewrite_qasm),
            "nam_qasm": str(rewrite_nam_qasm),
            "qubit_indexes": rewrite_qubit_indexes,
            "stats": rewrite_nam_stats,
        },
        "ours": {
            "structure": "opt",
            "opt": "Matrix-order",
            "build_time_s": float(ours_build_time),
            "nam_transpile_time_s": float(ours_nam_time),
            "compile_time_s": float(ours_compile_time),
            "qasm": str(ours_qasm),
            "nam_qasm": str(ours_nam_qasm),
            "qubit_indexes": ours_qubit_indexes,
            "stats": ours_nam_stats,
        },
        "table_rows": rows,
        "markdown_table": markdown_table(rows),
    }

    summary_path = out_dir / f"hypercube_random_walk_n{args.num_qubits}_ours_nam_summary.json"
    with summary_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, default=str)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hypercube_random_walk_n4 channel-LCU rewrite+opt+matrix-order NAM statistics."
    )
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--table1-raw-qasm", type=Path, default=DEFAULT_TABLE1_RAW_QASM)
    parser.add_argument("--rewrite-strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--nam-optimization-level", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(summary["markdown_table"])
    print()
    print("Raw path comparison:")
    print(json.dumps(summary["raw_path_comparison"], indent=2))
    print()
    print(f"summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
