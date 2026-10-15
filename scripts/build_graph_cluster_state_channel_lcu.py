#!/usr/bin/env python3
"""Build dissipative graph/cluster-state preparation channel-LCU circuits.

The Lindblad model follows the graph-state dissipative preparation rule

    L_v = sqrt(gamma) Z_v (I - K_v) / 2,
    K_v = X_v prod_{u in N(v)} Z_u.

In Pauli form this becomes

    L_v = sqrt(gamma)/2 (Z_v - i Y_v prod_{u in N(v)} Z_u),

which gives a compact, graph-structured benchmark with one jump per vertex.
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
DEFAULT_SUMMARY_ROOT = ROOT / "Baseline_results" / "graph_cluster_state_channel_lcu"


@dataclass(frozen=True)
class GraphSpec:
    name: str
    num_qubits: int
    edges: tuple[tuple[int, int], ...]
    description: str


def _float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _pauli_label(num_qubits: int, positions: dict[int, str]) -> str:
    return "".join(positions.get(i, "I") for i in range(num_qubits))


def _neighbors(num_qubits: int, edges: tuple[tuple[int, int], ...]) -> list[list[int]]:
    out = [[] for _ in range(num_qubits)]
    for u, v in edges:
        out[u].append(v)
        out[v].append(u)
    return [sorted(items) for items in out]


def grid_2d_cluster_graph(rows: int, cols: int) -> GraphSpec:
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return GraphSpec(
        name=f"graph_cluster_2d_grid_{rows}x{cols}",
        num_qubits=rows * cols,
        edges=tuple(edges),
        description="2D grid cluster-state dissipative preparation graph",
    )


def random_bounded_degree_graph(num_qubits: int, degree: int, seed: int) -> GraphSpec:
    """Generate a reproducible simple graph with maximum degree <= degree.

    A ring is used as a connected backbone, then random admissible edges are
    added until no further edge can be added without exceeding the degree cap.
    """
    if num_qubits < 3:
        raise ValueError("num_qubits must be at least 3.")
    if degree < 2:
        raise ValueError("degree must be at least 2 for the ring backbone.")

    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = {
        (min(i, (i + 1) % num_qubits), max(i, (i + 1) % num_qubits))
        for i in range(num_qubits)
    }
    deg = [2 for _ in range(num_qubits)]

    candidates = [
        (u, v)
        for u in range(num_qubits)
        for v in range(u + 1, num_qubits)
        if (u, v) not in edges
    ]
    rng.shuffle(candidates)
    for u, v in candidates:
        if deg[u] >= degree or deg[v] >= degree:
            continue
        edges.add((u, v))
        deg[u] += 1
        deg[v] += 1

    return GraphSpec(
        name=f"graph_cluster_random_bd{degree}_n{num_qubits}_seed{seed}",
        num_qubits=num_qubits,
        edges=tuple(sorted(edges)),
        description=f"random connected bounded-degree graph, max degree {degree}",
    )


def build_graph_cluster_lindbladian(spec: GraphSpec, gamma: float) -> Lindbladian:
    neighbors = _neighbors(spec.num_qubits, spec.edges)
    amp = np.sqrt(gamma) / 2
    h_terms: list[tuple[str, complex]] = []
    l_terms: list[list[tuple[str, complex]]] = []

    for v, nbs in enumerate(neighbors):
        z_v = _pauli_label(spec.num_qubits, {v: "Z"})
        y_v_z_neighbors = _pauli_label(
            spec.num_qubits,
            {**{u: "Z" for u in nbs}, v: "Y"},
        )
        l_terms.append([(z_v, amp), (y_v_z_neighbors, -1j * amp)])

    return Lindbladian(h_terms, l_terms)


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
    return rewritten.kraus_ops, result, time.perf_counter() - started


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


def build_channel_for_spec(spec: GraphSpec, gamma: float, delta_t: float) -> tuple[list, dict[str, Any]]:
    lind = build_graph_cluster_lindbladian(spec, gamma)
    channel_lind, success_prob, coeff_sums = Lindblad_to_channel(lind, delta_t)
    neighbors = _neighbors(spec.num_qubits, spec.edges)
    model_info = {
        "model": "dissipative_graph_cluster_state_preparation",
        "description": spec.description,
        "num_qubits": int(spec.num_qubits),
        "num_edges": int(len(spec.edges)),
        "edges": [list(edge) for edge in spec.edges],
        "degrees": [int(len(items)) for items in neighbors],
        "gamma": float(gamma),
        "delta_t": float(delta_t),
        "jump_operator": "L_v = sqrt(gamma)/2 * (Z_v - i Y_v prod_{u in N(v)} Z_u)",
        "success_prob": float(success_prob),
        "coeff_sums": [float(x) for x in coeff_sums],
    }
    return channel_lind.channels[0][1], model_info


def process_spec(args: argparse.Namespace, spec: GraphSpec) -> dict[str, Any]:
    out_dir = Path(args.out_root) / spec.name
    summary_dir = Path(args.summary_root) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_kraus_ops, model_info = build_channel_for_spec(spec, args.gamma, args.delta_t)
    raw_channel_metrics = channel_metrics(raw_kraus_ops)

    raw_qc, raw_qubit_indexes, raw_build_time = build_channel_lcu_circuit(
        raw_kraus_ops,
        structure="basic",
        opt="No",
    )
    raw_qasm = out_dir / f"{spec.name}_basic_raw.qasm"
    raw_nam_qasm = out_dir / f"{spec.name}_basic_raw_nam_opt{args.nam_optimization_level}.qasm"
    dump_qasm2(raw_qc, raw_qasm)
    raw_nam_stats, raw_nam_time = export_nam_count_qasm(
        raw_qc,
        raw_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )

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
    rewrite_qasm = out_dir / f"{spec.name}_basic_rewrite.qasm"
    rewrite_nam_qasm = out_dir / f"{spec.name}_basic_rewrite_nam_opt{args.nam_optimization_level}.qasm"
    dump_qasm2(rewrite_qc, rewrite_qasm)
    rewrite_nam_stats, rewrite_nam_time = export_nam_count_qasm(
        rewrite_qc,
        rewrite_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )

    opt_qc, opt_qubit_indexes, opt_build_time = build_rewrite_opt_matrix_order_circuit(
        rewritten_kraus_ops
    )
    opt_qasm = out_dir / f"{spec.name}_rewrite_opt_matrix_order.qasm"
    opt_nam_qasm = out_dir / f"{spec.name}_rewrite_opt_matrix_order_nam_opt{args.nam_optimization_level}.qasm"
    dump_qasm2(opt_qc, opt_qasm)
    opt_nam_stats, opt_nam_time = export_nam_count_qasm(
        opt_qc,
        opt_nam_qasm,
        optimization_level=args.nam_optimization_level,
    )

    summary = {
        "experiment": "graph_cluster_state_channel_lcu",
        "name": spec.name,
        "model": model_info,
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
        "rewrite": {
            "channel_metrics": rewrite_channel_metrics,
            "time_s": float(rewrite_time),
            "initial_support": int(rewrite_result["initial_support"]),
            "final_support": int(rewrite_result["final_support"]),
            "steps": int(len(rewrite_result["steps"])),
            "termination": rewrite_result.get("termination", {}),
            "basic_build_time_s": float(rewrite_build_time),
            "basic_nam_transpile_time_s": float(rewrite_nam_time),
            "basic_qasm": str(rewrite_qasm),
            "basic_nam_qasm": str(rewrite_nam_qasm),
            "basic_qubit_indexes": rewrite_qubit_indexes,
            "basic_stats": rewrite_nam_stats,
        },
        "ours": {
            "structure": "tree",
            "block_backend": "block_encoding.BlockEncoding",
            "qubits_initially_zero": False,
            "opt": "Matrix-order",
            "build_time_s": float(opt_build_time),
            "nam_transpile_time_s": float(opt_nam_time),
            "compile_time_s": float(rewrite_time + opt_build_time + opt_nam_time),
            "qasm": str(opt_qasm),
            "nam_qasm": str(opt_nam_qasm),
            "qubit_indexes": opt_qubit_indexes,
            "stats": opt_nam_stats,
        },
    }
    summary_path = summary_dir / f"{spec.name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, default=str)
    summary["summary_path"] = str(summary_path)
    return summary


def selected_specs(args: argparse.Namespace) -> list[GraphSpec]:
    specs: list[GraphSpec] = []
    selected = set(args.examples)
    if "all" in selected or "grid" in selected:
        specs.append(grid_2d_cluster_graph(args.grid_rows, args.grid_cols))
    if "all" in selected or "random" in selected:
        specs.append(random_bounded_degree_graph(args.random_n, args.random_degree, args.random_seed))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graph/cluster-state dissipative channel-LCU QASM2 circuits."
    )
    parser.add_argument("--examples", nargs="+", choices=("all", "grid", "random"), default=["all"])
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--random-n", type=int, default=12)
    parser.add_argument("--random-degree", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--rewrite-strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--nam-optimization-level", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [process_spec(args, spec) for spec in selected_specs(args)]
    print("| Example | Qubits | Edges | Kraus raw->rewrite | Pauli terms raw->rewrite | Raw NAM | Ours NAM | Summary |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for summary in summaries:
        model = summary["model"]
        raw_metrics = summary["raw"]["channel_metrics"]
        rewrite_metrics = summary["rewrite"]["channel_metrics"]
        raw_nam = summary["raw"]["stats"]["nam"]["rz_clifford_total"]
        ours_nam = summary["ours"]["stats"]["nam"]["rz_clifford_total"]
        print(
            f"| {summary['name']} | {model['num_qubits']} | {model['num_edges']} | "
            f"{raw_metrics['kraus_count']}->{rewrite_metrics['kraus_count']} | "
            f"{raw_metrics['pauli_terms_total']}->{rewrite_metrics['pauli_terms_total']} | "
            f"{raw_nam} | {ours_nam} | {summary['summary_path']} |"
        )


if __name__ == "__main__":
    main()
