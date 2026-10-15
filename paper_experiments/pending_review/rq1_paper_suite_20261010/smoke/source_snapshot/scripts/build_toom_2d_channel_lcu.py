#!/usr/bin/env python3
"""Build Toom-rule dissipative preparation channel-LCU circuits.

The model is the local engineered-dissipation rule

    L_v = sqrt(gamma) Z_v (I - X_v X_e) / 2 (I - X_v X_n) / 2,

on an L x L periodic square lattice, where e and n are the east and north
neighbours of v.  In Pauli-sum form each jump has four terms:

    L_v = sqrt(gamma)/4 * Z_v
        * (I - X_v X_e - X_v X_n + X_e X_n).

The Hamiltonian is H = 0.  The first-order Lindblad-to-channel conversion then
produces K0 = I - dt/2 sum_v L_v^dag L_v and one jump Kraus sqrt(dt) L_v per
site.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import Lindbladian, channel as ChannelIR, channel_ensemble  # type: ignore  # noqa: E402
import channel_LCU  # type: ignore  # noqa: E402
from channel_LCU import Lindblad_to_channel  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402

from experiment_protocol import transpile_channel_circuit as transpile


NAM_BASIS = ("h", "x", "rz", "cx")
NAM_TRANSPILER_BASIS = ("h", "x", "rz", "cx", "reset")
DEFAULT_OUT_ROOT = ROOT / "circuits" / "Table1"
DEFAULT_SUMMARY_ROOT = ROOT / "Baseline_results" / "toom_2d_channel_lcu"


@dataclass(frozen=True)
class ToomSpec:
    name: str
    lattice_L: int
    num_qubits: int
    rule: str
    periodic: bool


def _float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _site(L: int, row: int, col: int) -> int:
    return row * L + col


def _east_north_neighbours(L: int, vertex: int) -> tuple[int, int]:
    row, col = divmod(vertex, L)
    east = _site(L, row, (col + 1) % L)
    north = _site(L, (row - 1) % L, col)
    return east, north


def _pauli_label(num_qubits: int, positions: dict[int, str]) -> str:
    return "".join(positions.get(i, "I") for i in range(num_qubits))


def _toom_pauli_terms(num_qubits: int, vertex: int, east: int, north: int, amp: float):
    z_v = _pauli_label(num_qubits, {vertex: "Z"})
    y_v_x_e = _pauli_label(num_qubits, {vertex: "Y", east: "X"})
    y_v_x_n = _pauli_label(num_qubits, {vertex: "Y", north: "X"})
    z_v_x_e_x_n = _pauli_label(num_qubits, {vertex: "Z", east: "X", north: "X"})
    return [
        (z_v, amp),
        (y_v_x_e, -1j * amp),
        (y_v_x_n, -1j * amp),
        (z_v_x_e_x_n, amp),
    ]


def build_toom_lindbladian(lattice_L: int, gamma: float) -> tuple[Lindbladian, dict[str, Any]]:
    if lattice_L < 2:
        raise ValueError("lattice_L must be at least 2.")

    num_qubits = lattice_L * lattice_L
    amp = float(np.sqrt(gamma) / 4)
    h_terms: list[tuple[str, complex]] = []
    l_terms: list[list[tuple[str, complex]]] = []
    neighbour_table: list[dict[str, int]] = []

    for vertex in range(num_qubits):
        east, north = _east_north_neighbours(lattice_L, vertex)
        l_terms.append(_toom_pauli_terms(num_qubits, vertex, east, north, amp))
        neighbour_table.append({"v": int(vertex), "east": int(east), "north": int(north)})

    lind = Lindbladian(h_terms, l_terms)
    info = {
        "model": "toom_2d_periodic_engineered_dissipation",
        "num_qubits": int(num_qubits),
        "lattice_L": int(lattice_L),
        "periodic": True,
        "hamiltonian": "H = 0",
        "jump_operator": (
            "L_v = sqrt(gamma) Z_v (I - X_v X_e)/2 (I - X_v X_n)/2"
        ),
        "jump_operator_pauli_sum": (
            "sqrt(gamma)/4 * (Z_v - i Y_v X_e - i Y_v X_n + Z_v X_e X_n)"
        ),
        "rule": "east+north Toom majority rule",
        "gamma": float(gamma),
        "jump_pauli_terms_per_site": 4,
        "neighbours": neighbour_table,
    }
    return lind, info


def build_toom_channel(lattice_L: int, gamma: float, delta_t: float) -> tuple[list, dict[str, Any]]:
    lind, model_info = build_toom_lindbladian(lattice_L, gamma)
    channel_lind, success_prob, coeff_sums = Lindblad_to_channel(lind, delta_t)
    model_info.update(
        {
            "delta_t": float(delta_t),
            "success_prob": float(success_prob),
            "coeff_sums": [float(x) for x in coeff_sums],
        }
    )
    return channel_lind.channels[0][1], model_info


def channel_metrics(kraus_ops: list) -> dict[str, Any]:
    nonzero = nonzero_kraus_ops(kraus_ops)
    row_supports = [int(len(op.instances)) for op in nonzero]
    return {
        "kraus_count": int(len(nonzero)),
        "pauli_terms_total": int(sum(row_supports)),
        "max_terms_per_kraus": int(max(row_supports, default=0)),
        "min_terms_per_kraus": int(min(row_supports, default=0)),
        "row_supports": row_supports,
        "system_qubits": int(max((op.size for op in nonzero), default=0)),
    }


def nonzero_kraus_ops(kraus_ops: list) -> list:
    return [op for op in kraus_ops if len(op.instances) > 0]


def gatecost_rewrite_channel(
    kraus_ops: list,
    *,
    strategy: str,
    beam_width: int,
    max_steps: int,
    tol: float,
    keep_neutral: bool,
    max_neutral_candidates: int | None,
    skip_disjoint_pairs: bool,
    verbose: bool,
) -> tuple[list, dict[str, Any], float]:
    started = time.perf_counter()
    rewrite_source = ChannelIR(copy.deepcopy(kraus_ops))
    rewrite_source.zero_elim()
    result = rewrite_source.gatecost_rewrite_search(
        strategy=strategy,
        beam_width=beam_width,
        max_steps=max_steps,
        tol=tol,
        verbose=verbose,
        keep_neutral=keep_neutral,
        max_neutral_candidates=max_neutral_candidates,
        skip_disjoint_pairs=skip_disjoint_pairs,
    )

    rewritten = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten.apply_rewrite_result(result, tol=tol)
    return nonzero_kraus_ops(rewritten.kraus_ops), result, time.perf_counter() - started


def with_new_block_encoding(func):
    old_block_encoding = channel_LCU.BlockEncoding
    channel_LCU.BlockEncoding = NewBlockEncoding
    try:
        return func()
    finally:
        channel_LCU.BlockEncoding = old_block_encoding


def build_basic_channel_lcu_circuit(kraus_ops: list) -> tuple[QuantumCircuit, Any, float]:
    started = time.perf_counter()

    def _build():
        return channel_LCU.channel_to_LCU(
            channel_ensemble([copy.deepcopy(kraus_ops)]),
            structure="basic",
            opt="No",
        )

    qc, qubit_indexes = with_new_block_encoding(_build)
    return qc, qubit_indexes, time.perf_counter() - started


def build_rewrite_opt_matrix_order_circuit(kraus_ops: list) -> tuple[QuantumCircuit, Any, float]:
    """Use the shared corrected backend and explicit tree outer selection."""
    from channel_IR import channel_ensemble
    import channel_LCU

    started = time.perf_counter()
    qc, qubit_indexes = channel_LCU.channel_to_LCU(
        channel_ensemble([copy.deepcopy(kraus_ops)]), structure="tree", opt="Matrix-order",
    )
    return qc, qubit_indexes, time.perf_counter() - started


def dump_qasm2(qc: QuantumCircuit, path: Path) -> None:
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
    export_openqasm2_quartz_input(tqc, path, basis_gates=NAM_TRANSPILER_BASIS)
    return count_qasm_file(path, gate_set="nam"), elapsed


def circuit_stats_from_qc(qc: QuantumCircuit) -> dict[str, Any]:
    op_counts = {str(gate): int(count) for gate, count in qc.count_ops().items()}
    return {
        "num_qubits": int(qc.num_qubits),
        "num_clbits": int(qc.num_clbits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "ops": dict(sorted(op_counts.items())),
    }


def qasm_file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
    }


def row_from_stats(
    name: str,
    qc_stats: dict[str, Any],
    nam_stats: dict[str, Any],
    compile_time: float | None,
    *,
    reference_nam_total: int,
) -> dict[str, Any]:
    nam_total = int(nam_stats["nam"]["rz_clifford_total"])
    return {
        "name": name,
        "qubits": int(nam_stats["num_qubits"]),
        "native_size": int(qc_stats["size"]),
        "native_depth": int(qc_stats["depth"]),
        "nam_total": nam_total,
        "ratio_to_raw": float(nam_total / reference_nam_total),
        "non_clifford": int(nam_stats["clifford"]["non_clifford"]),
        "reset": int(nam_stats["ops"].get("reset", 0)),
        "compile_time_s": None if compile_time is None else float(compile_time),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Circuit | Qubits | Native size | Native depth | NAM total | G/G0 | Non-Clifford | Reset | Compilation time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        time_cell = "-" if row["compile_time_s"] is None else f"{row['compile_time_s']:.3f}"
        lines.append(
            f"| {row['name']} | {row['qubits']} | {row['native_size']} | "
            f"{row['native_depth']} | {row['nam_total']} | "
            f"{row['ratio_to_raw']:.6f} | {row['non_clifford']} | "
            f"{row['reset']} | {time_cell} |"
        )
    return "\n".join(lines)


def process(args: argparse.Namespace) -> dict[str, Any]:
    example_name = f"toom_2d_periodic_L{args.lattice_L}_dt{_float_tag(args.delta_t)}"
    out_dir = Path(args.out_root) / example_name
    summary_dir = Path(args.summary_root) / example_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_kraus_ops, model_info = build_toom_channel(args.lattice_L, args.gamma, args.delta_t)
    raw_channel_metrics = channel_metrics(raw_kraus_ops)

    raw_qc, raw_qubit_indexes, raw_build_time = build_basic_channel_lcu_circuit(raw_kraus_ops)
    raw_qasm = out_dir / f"{example_name}_basic_raw.qasm"
    raw_nam_qasm = out_dir / f"{example_name}_basic_raw_nam_opt{args.nam_optimization_level}.qasm"
    dump_qasm2(raw_qc, raw_qasm)
    raw_nam_stats, raw_nam_time = export_nam_count_qasm(
        raw_qc,
        raw_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )
    raw_qc_stats = circuit_stats_from_qc(raw_qc)

    rewritten_kraus_ops, rewrite_result, rewrite_time = gatecost_rewrite_channel(
        raw_kraus_ops,
        strategy=args.rewrite_strategy,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        tol=args.tol,
        keep_neutral=args.keep_neutral,
        max_neutral_candidates=args.max_neutral_candidates,
        skip_disjoint_pairs=not args.no_skip_disjoint_pairs,
        verbose=args.verbose_rewrite,
    )
    rewrite_channel_metrics = channel_metrics(rewritten_kraus_ops)

    rewrite_qc, rewrite_qubit_indexes, rewrite_build_time = build_basic_channel_lcu_circuit(
        rewritten_kraus_ops
    )
    rewrite_qasm = out_dir / f"{example_name}_basic_rewrite.qasm"
    rewrite_nam_qasm = out_dir / f"{example_name}_basic_rewrite_nam_opt{args.nam_optimization_level}.qasm"
    dump_qasm2(rewrite_qc, rewrite_qasm)
    rewrite_nam_stats, rewrite_nam_time = export_nam_count_qasm(
        rewrite_qc,
        rewrite_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )
    rewrite_qc_stats = circuit_stats_from_qc(rewrite_qc)

    ours_qc, ours_qubit_indexes, ours_build_time = build_rewrite_opt_matrix_order_circuit(
        rewritten_kraus_ops
    )
    ours_qasm = out_dir / f"{example_name}_rewrite_opt_matrix_order.qasm"
    ours_nam_qasm = out_dir / f"{example_name}_rewrite_opt_matrix_order_nam_opt{args.nam_optimization_level}.qasm"
    dump_qasm2(ours_qc, ours_qasm)
    ours_nam_stats, ours_nam_time = export_nam_count_qasm(
        ours_qc,
        ours_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )
    ours_qc_stats = circuit_stats_from_qc(ours_qc)

    raw_nam_total = int(raw_nam_stats["nam"]["rz_clifford_total"])
    ours_compile_time = rewrite_time + ours_build_time + ours_nam_time
    rows = [
        row_from_stats(
            "Raw channel-LCU (basic/no)",
            raw_qc_stats,
            raw_nam_stats,
            None,
            reference_nam_total=raw_nam_total,
        ),
        row_from_stats(
            "Gatecost rewrite channel-LCU (basic/no)",
            rewrite_qc_stats,
            rewrite_nam_stats,
            rewrite_time + rewrite_build_time + rewrite_nam_time,
            reference_nam_total=raw_nam_total,
        ),
        row_from_stats(
            "Ours (gatecost rewrite + tree + matrix-order)",
            ours_qc_stats,
            ours_nam_stats,
            ours_compile_time,
            reference_nam_total=raw_nam_total,
        ),
    ]

    summary = {
        "experiment": "toom_2d_channel_lcu",
        "name": example_name,
        "model": model_info,
        "nam_basis_counted": list(NAM_BASIS),
        "nam_transpiler_basis": list(NAM_TRANSPILER_BASIS),
        "nam_optimization_level": int(args.nam_optimization_level),
        "raw": {
            "channel_metrics": raw_channel_metrics,
            "build_time_s": float(raw_build_time),
            "nam_transpile_time_s": float(raw_nam_time),
            "qasm": qasm_file_info(raw_qasm),
            "nam_qasm": qasm_file_info(raw_nam_qasm),
            "qubit_indexes": raw_qubit_indexes,
            "circuit_stats": raw_qc_stats,
            "stats": raw_nam_stats,
        },
        "rewrite": {
            "method": "gatecost_rewrite_search",
            "channel_metrics": rewrite_channel_metrics,
            "time_s": float(rewrite_time),
            "initial_support": int(rewrite_result["initial_support"]),
            "final_support": int(rewrite_result["final_support"]),
            "initial_cost": int(rewrite_result["initial_cost"]),
            "final_cost": int(rewrite_result["final_cost"]),
            "initial_cost_components": rewrite_result["initial_cost_components"],
            "final_cost_components": rewrite_result["final_cost_components"],
            "initial_row_supports": rewrite_result["initial_row_supports"],
            "final_row_supports": rewrite_result["final_row_supports"],
            "steps": int(len(rewrite_result["steps"])),
            "step_metadata": rewrite_result.get("step_metadata", []),
            "termination": rewrite_result.get("termination", {}),
            "basic_build_time_s": float(rewrite_build_time),
            "basic_nam_transpile_time_s": float(rewrite_nam_time),
            "basic_qasm": qasm_file_info(rewrite_qasm),
            "basic_nam_qasm": qasm_file_info(rewrite_nam_qasm),
            "basic_qubit_indexes": rewrite_qubit_indexes,
            "basic_circuit_stats": rewrite_qc_stats,
            "basic_stats": rewrite_nam_stats,
        },
        "ours": {
            "structure": "tree",
            "block_backend": "block_encoding.BlockEncoding",
            "qubits_initially_zero": False,
            "opt": "Matrix-order",
            "build_time_s": float(ours_build_time),
            "nam_transpile_time_s": float(ours_nam_time),
            "compile_time_s": float(ours_compile_time),
            "qasm": qasm_file_info(ours_qasm),
            "nam_qasm": qasm_file_info(ours_nam_qasm),
            "qubit_indexes": ours_qubit_indexes,
            "circuit_stats": ours_qc_stats,
            "stats": ours_nam_stats,
        },
        "comparison": {
            "basis": "nam",
            "raw_metric_total": raw_nam_total,
            "rewrite_metric_total": int(rewrite_nam_stats["nam"]["rz_clifford_total"]),
            "ours_metric_total": int(ours_nam_stats["nam"]["rz_clifford_total"]),
            "rewrite_to_raw_ratio": float(
                int(rewrite_nam_stats["nam"]["rz_clifford_total"]) / raw_nam_total
            ),
            "ours_to_raw_ratio": float(
                int(ours_nam_stats["nam"]["rz_clifford_total"]) / raw_nam_total
            ),
        },
        "table_rows": rows,
        "markdown_table": markdown_table(rows),
    }

    summary_path = summary_dir / f"{example_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, default=str)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Toom-rule dissipative channel-LCU QASM2 circuits."
    )
    parser.add_argument("--lattice-L", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--rewrite-strategy", choices=("greedy", "beam"), default="beam")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--keep-neutral", action="store_true")
    parser.add_argument("--max-neutral-candidates", type=int, default=3)
    parser.add_argument("--no-skip-disjoint-pairs", action="store_true")
    parser.add_argument("--verbose-rewrite", action="store_true")
    parser.add_argument("--nam-optimization-level", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    summary = process(parse_args())
    raw = summary["raw"]["channel_metrics"]
    rewrite = summary["rewrite"]["channel_metrics"]
    print(
        f"{summary['name']}: kraus {raw['kraus_count']} -> {rewrite['kraus_count']}, "
        f"pauli terms {raw['pauli_terms_total']} -> {rewrite['pauli_terms_total']}, "
        f"H_length {summary['rewrite']['initial_cost']} -> {summary['rewrite']['final_cost']}"
    )
    print(summary["markdown_table"])
    print()
    print(f"summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
