#!/usr/bin/env python3
"""Explore sparse coefficient PREPARE for TFIM_8 A0 matrix-order block encoding.

This is an experiment-only harness.  It compares the current dense Qiskit
StatePreparation used by ``BlockEncoding.mulplex_B_opt_order`` with a sparse
binary-tree PREPARE that prunes zero-probability branches and treats
deterministic branches as direct bit fixes plus unconditional lower-level
preparation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, StatePreparation, XGate
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiment_tfim8_a0_full_matrix_order_forced_w import (  # type: ignore  # noqa: E402
    ForcedWBlockEncoding,
    summarize_basis,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_sparse_stateprep_flat_v1"


def dense_stateprep_from_weights(weights: list[float] | np.ndarray) -> QuantumCircuit:
    weights_arr = np.asarray(weights, dtype=float)
    n = int(np.log2(len(weights_arr)))
    if 2**n != len(weights_arr):
        raise ValueError("weights length must be a power of two")
    total = float(np.sum(weights_arr))
    if total <= 0:
        raise ValueError("weights must contain positive mass")
    amps = np.sqrt(weights_arr / total)
    qc = QuantumCircuit(n, name="dense_prepare")
    qc.append(StatePreparation(Statevector(amps)), range(n))
    return qc


def sparse_lcu_prepare_tree(weights: list[float] | np.ndarray, *, tol: float = 1e-15) -> QuantumCircuit:
    """Prepare ``sum_i sqrt(weights[i]/sum(weights)) |i>`` using a sparse tree.

    The implementation is deliberately flat: each nontrivial probability split
    becomes one prefix-controlled ``RY`` and deterministic branches are path
    compressed.  This avoids nested controlled-subcircuit synthesis, which can
    overwhelm the savings from pruning zero branches.

    Qubit ordering matches ``StatePreparation(Statevector(amps))`` and the
    existing ``subroutine.lcu_prepare_tree`` helper: the current split bit at
    recursion level ``level`` is qubit ``n - level - 1``.
    """

    probs = np.asarray(weights, dtype=float)
    n = int(np.log2(len(probs)))
    if 2**n != len(probs):
        raise ValueError("weights length must be a power of two")
    total = float(np.sum(probs))
    if total <= 0:
        raise ValueError("weights must contain positive mass")
    probs = probs / total

    qc = QuantumCircuit(n, name="sparse_prepare")
    stats = {
        "binary_nodes": 0,
        "deterministic_zero_nodes": 0,
        "deterministic_one_nodes": 0,
        "zero_nodes": 0,
        "ry_nodes": 0,
        "x_nodes": 0,
        "controlled_ry_nodes": 0,
        "controlled_x_nodes": 0,
        "prefix_x_wrappers": 0,
        "max_active_prefix": 0,
        "max_level": 0,
    }

    def with_prefix(gate: Any, target: int, active_prefix: list[tuple[int, int]]) -> None:
        stats["max_active_prefix"] = max(stats["max_active_prefix"], len(active_prefix))
        if len(active_prefix) == 0:
            qc.append(gate, [target])
            return

        controls = [qubit for qubit, _bit in active_prefix]
        for qubit, bit in active_prefix:
            if bit == 0:
                qc.x(qubit)
                stats["prefix_x_wrappers"] += 1
        qc.append(gate.control(len(controls)), controls + [target])
        for qubit, bit in reversed(active_prefix):
            if bit == 0:
                qc.x(qubit)
                stats["prefix_x_wrappers"] += 1

    def recurse(level: int, branch_probs: np.ndarray, active_prefix: list[tuple[int, int]]) -> None:
        if level == n:
            return

        half = len(branch_probs) // 2
        p0 = float(np.sum(branch_probs[:half]))
        p1 = float(np.sum(branch_probs[half:]))
        mass = p0 + p1
        if mass <= tol:
            stats["zero_nodes"] += 1
            return

        t = n - level - 1
        stats["max_level"] = max(stats["max_level"], level)

        if p1 <= tol:
            stats["deterministic_zero_nodes"] += 1
            recurse(level + 1, branch_probs[:half], active_prefix)
            return

        if p0 <= tol:
            stats["deterministic_one_nodes"] += 1
            stats["x_nodes"] += 1
            if active_prefix:
                stats["controlled_x_nodes"] += 1
            with_prefix(XGate(), t, active_prefix)
            recurse(level + 1, branch_probs[half:], active_prefix)
            return

        stats["binary_nodes"] += 1
        stats["ry_nodes"] += 1
        if active_prefix:
            stats["controlled_ry_nodes"] += 1
        theta = 2 * np.arccos(np.sqrt(p0 / mass))
        with_prefix(RYGate(theta), t, active_prefix)

        recurse(level + 1, branch_probs[:half], active_prefix + [(t, 0)])
        recurse(level + 1, branch_probs[half:], active_prefix + [(t, 1)])

    recurse(0, probs, [])
    qc.sparse_prepare_stats = stats  # type: ignore[attr-defined]
    return qc


class SparsePrepareForcedWBlockEncoding(ForcedWBlockEncoding):
    """Forced-w BlockEncoding that replaces dense coefficient PREPARE."""

    def mulplex_B_opt_order(self, ctrl_size):  # type: ignore[override]
        coeff_list_ordered = self.coeff_list_ordered
        weights = np.array([abs(c) for c in coeff_list_ordered], dtype=float)
        qc = sparse_lcu_prepare_tree(weights)
        self.sparse_prepare_stats = getattr(qc, "sparse_prepare_stats", {})
        return qc


def select_kraus_zero(kraus_ops: list[Any]) -> Any:
    if not kraus_ops or len(kraus_ops[0].instances) == 0:
        raise ValueError("TFIM_8 Kraus index 0 is missing or empty")
    return kraus_ops[0]


def recount_bases(qc: QuantumCircuit, out_dir: Path, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for basis in ("nam", "ibm"):
        status, result, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=out_dir / f"{prefix}_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        rows[basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": result,
            "error": error,
        }
    return rows


def build_full_block(ms: Any, forced_w: int, prepare: str) -> tuple[QuantumCircuit, Any, float]:
    cls = SparsePrepareForcedWBlockEncoding if prepare == "sparse" else ForcedWBlockEncoding
    started = time.perf_counter()
    be = cls(copy.deepcopy(ms), forced_w=forced_w)
    qc = be.circuit(opt="Matrix-order")
    return qc, be, time.perf_counter() - started


def run_full_case(ms: Any, forced_w: int, prepare: str, out_dir: Path) -> dict[str, Any]:
    qc, be, build_seconds = build_full_block(ms, forced_w, prepare)
    prefix = f"tfim8_a0_full_matrix_order_w{forced_w}_{prepare}_prepare"
    write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    basis_recounts = recount_bases(qc, out_dir, prefix)
    return {
        "case": "full_block_encoding",
        "forced_w": int(forced_w),
        "prepare": prepare,
        "status": "ok",
        "build_seconds": float(build_seconds),
        "width": int(qc.num_qubits),
        "logical_metrics": qiskit_metrics(qc),
        "raw_ctrl_size": int(be.ctrl_size),
        "matrix_order_ctrl_size": int(forced_w),
        "coeff_slots": int(2**forced_w),
        "nonzero_coeff_slots": int(sum(abs(complex(c)) > 0 for c in getattr(be, "coeff_list_ordered", []))),
        "selection_metadata": getattr(be, "forced_w_selection_metadata", {}),
        "sparse_prepare_stats": getattr(be, "sparse_prepare_stats", None),
        "basis_recounts": basis_recounts,
        "basis_summary": {basis: summarize_basis(entry) for basis, entry in basis_recounts.items()},
    }


def run_prepare_only(weights: np.ndarray, name: str, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for prepare, builder in (
        ("dense", dense_stateprep_from_weights),
        ("sparse", sparse_lcu_prepare_tree),
    ):
        qc = builder(weights)
        prefix = f"prepare_only_{name}_{prepare}"
        write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
        basis_recounts = recount_bases(qc, out_dir, prefix)
        rows.append(
            {
                "case": "prepare_only",
                "name": name,
                "prepare": prepare,
                "width": int(qc.num_qubits),
                "support": int(np.count_nonzero(np.abs(weights) > 0)),
                "slots": int(len(weights)),
                "sparse_prepare_stats": getattr(qc, "sparse_prepare_stats", None),
                "logical_metrics": qiskit_metrics(qc),
                "basis_recounts": basis_recounts,
                "basis_summary": {basis: summarize_basis(entry) for basis, entry in basis_recounts.items()},
            }
        )
    return rows


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Sparse StatePreparation Exploration",
        "",
        "Sparse PREPARE uses a pruned probability tree.  It skips zero subtrees and replaces deterministic branches by direct bit fixes plus unconditional preparation of the remaining suffix.",
        "",
        "## Full TFIM_8 A0 BlockEncoding",
        "",
        "| w | prepare | width | support/slots | NAM gates | NAM non-Clifford | IBM gates | IBM non-Clifford | build time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["full_block_results"]:
        nam = row["basis_summary"]["nam"]
        ibm = row["basis_summary"]["ibm"]
        lines.append(
            f"| {row['forced_w']} | {row['prepare']} | {row['width']} | "
            f"{row['nonzero_coeff_slots']}/{row['coeff_slots']} | "
            f"{nam['gate_count']} | {nam['non_clifford']} | "
            f"{ibm['gate_count']} | {ibm['non_clifford']} | {row['build_seconds']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## PREPARE Only",
            "",
            "| vector | prepare | width | support/slots | NAM gates | NAM non-Clifford | IBM gates | IBM non-Clifford |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["prepare_only_results"]:
        nam = row["basis_summary"]["nam"]
        ibm = row["basis_summary"]["ibm"]
        lines.append(
            f"| {row['name']} | {row['prepare']} | {row['width']} | {row['support']}/{row['slots']} | "
            f"{nam['gate_count']} | {nam['non_clifford']} | {ibm['gate_count']} | {ibm['non_clifford']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--w", type=int, nargs="+", default=[5, 8])
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    ms = select_kraus_zero(kraus_ops)

    full_results = []
    for forced_w in args.w:
        for prepare in ("dense", "sparse"):
            full_results.append(run_full_case(ms, forced_w, prepare, out_dir))

    prepare_only_results: list[dict[str, Any]] = []
    prepare_only_results.extend(
        run_prepare_only(
            np.array([0.5, 0.5, 0.0, 0.0], dtype=float),
            "toy_adjacent_2q",
            out_dir,
        )
    )
    prepare_only_results.extend(
        run_prepare_only(
            np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=float),
            "toy_far_3q",
            out_dir,
        )
    )

    summary = {
        "script": str(Path(__file__).resolve()),
        "result_dir": str(out_dir),
        "model_info": model_info,
        "benchmark": "TFIM_8",
        "kraus_index": 0,
        "kraus_metrics": {
            "system_qubits": int(ms.size),
            "pauli_terms": int(len(ms.instances)),
            "pauli_norm": float(ms.pauli_norm()),
        },
        "full_block_results": full_results,
        "prepare_only_results": prepare_only_results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(out_dir / "report.md", summary)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
