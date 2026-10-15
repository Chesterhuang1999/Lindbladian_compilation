#!/usr/bin/env python3
"""Sparse trie PREPARE probe for TFIM_8 A0 matrix-order coefficients.

This standalone prototype compares dense Qiskit ``StatePreparation`` with a
prefix-controlled rotation trie over the nonzero matrix-order coefficient
addresses.  It is intentionally experimental and does not modify the main
block-encoding implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
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

from experiment_tfim8_a0_full_matrix_order_forced_w_new import (  # type: ignore  # noqa: E402
    ForcedWNewBlockEncoding,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)


RESULT_ROOT = ROOT / "Baseline_results" / "sparse_trie_stateprep_probe_v1"
TOL = 1e-14


def json_default(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def extract_coefficients(ms: Any, width: int) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    be = ForcedWNewBlockEncoding(copy.deepcopy(ms), forced_w=width)
    coeff_mode_dict, ctrl_size = be.find_optimal_order_matrices()
    coeffs = np.zeros(1 << ctrl_size, dtype=complex)
    for ctrl_value, coeff in coeff_mode_dict.items():
        coeffs[int(str(ctrl_value), 2)] = complex(coeff)
    weights = np.abs(coeffs)
    nonzero = [
        {
            "index": int(index),
            "ctrl": format(index, f"0{ctrl_size}b"),
            "weight": float(weight),
            "popcount": int(index.bit_count()),
        }
        for index, weight in enumerate(weights)
        if weight > TOL
    ]
    debug = getattr(be, "placement_search_debug", {})
    meta = {
        "w": int(width),
        "ctrl_size": int(ctrl_size),
        "support": int(len(nonzero)),
        "slots": int(len(coeffs)),
        "extract_seconds": float(time.perf_counter() - started),
        "placement_shortcut": debug.get("shortcut"),
        "placement_best_cost": debug.get("best_cost"),
        "nonzero_coefficients": nonzero,
        "popcount_histogram": dict(sorted(Counter(row["popcount"] for row in nonzero).items())),
    }
    return coeffs, meta


def dense_prepare(weights: np.ndarray) -> QuantumCircuit:
    n = int(round(math.log2(len(weights))))
    probs = weights / float(np.sum(weights))
    qc = QuantumCircuit(n, name="dense_stateprep")
    qc.append(StatePreparation(Statevector(np.sqrt(probs))), range(n))
    return qc


def sparse_trie_prepare(weights: np.ndarray, *, tol: float = TOL) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Prepare real nonnegative amplitudes using a pruned binary tree.

    This is a direct prototype.  Each nontrivial split emits a prefix-controlled
    ``Ry`` and deterministic one-branches emit a prefix-controlled ``X``.  Qiskit
    will decompose these high-control rotations, which is precisely the cost we
    want to expose in this probe.
    """

    weights = np.asarray(weights, dtype=float)
    n = int(round(math.log2(len(weights))))
    probs = weights / float(np.sum(weights))
    qc = QuantumCircuit(n, name="sparse_trie_prepare")
    stats = {
        "ry_nodes": 0,
        "x_nodes": 0,
        "controlled_ry_nodes": 0,
        "controlled_x_nodes": 0,
        "zero_nodes": 0,
        "deterministic_zero_nodes": 0,
        "deterministic_one_nodes": 0,
        "prefix_x_wrappers": 0,
        "max_controls": 0,
    }

    def append_with_prefix(gate: Any, target: int, prefix: list[tuple[int, int]]) -> None:
        stats["max_controls"] = max(stats["max_controls"], len(prefix))
        if not prefix:
            qc.append(gate, [target])
            return
        controls = [qubit for qubit, _bit in prefix]
        for qubit, bit in prefix:
            if bit == 0:
                qc.x(qubit)
                stats["prefix_x_wrappers"] += 1
        qc.append(gate.control(len(controls)), controls + [target])
        for qubit, bit in reversed(prefix):
            if bit == 0:
                qc.x(qubit)
                stats["prefix_x_wrappers"] += 1

    def rec(level: int, branch: np.ndarray, prefix: list[tuple[int, int]]) -> None:
        if level == n:
            return
        half = len(branch) // 2
        p0 = float(np.sum(branch[:half]))
        p1 = float(np.sum(branch[half:]))
        mass = p0 + p1
        if mass <= tol:
            stats["zero_nodes"] += 1
            return
        target = n - level - 1
        if p1 <= tol:
            stats["deterministic_zero_nodes"] += 1
            rec(level + 1, branch[:half], prefix)
            return
        if p0 <= tol:
            stats["deterministic_one_nodes"] += 1
            stats["x_nodes"] += 1
            if prefix:
                stats["controlled_x_nodes"] += 1
            append_with_prefix(XGate(), target, prefix)
            rec(level + 1, branch[half:], prefix + [(target, 1)])
            return
        theta = 2.0 * math.acos(math.sqrt(p0 / mass))
        stats["ry_nodes"] += 1
        if prefix:
            stats["controlled_ry_nodes"] += 1
        append_with_prefix(RYGate(theta), target, prefix)
        rec(level + 1, branch[:half], prefix + [(target, 0)])
        rec(level + 1, branch[half:], prefix + [(target, 1)])

    rec(0, probs, [])
    return qc, stats


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
        if status == "ok" and result is not None:
            stats = result["stats"]
            summary = {
                "gate_count": int(stats["metric_total"]),
                "width": int(stats["num_qubits"]),
                "non_clifford": int(stats["clifford"]["non_clifford"]),
                "qasm": result["qasm"],
            }
        else:
            summary = None
        rows[basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "summary": summary,
            "error": error,
        }
    return rows


def run_prepare_case(weights: np.ndarray, width: int, kind: str, out_dir: Path) -> dict[str, Any]:
    if kind == "dense":
        qc = dense_prepare(weights)
        stats = None
    elif kind == "sparse_trie":
        qc, stats = sparse_trie_prepare(weights)
    else:
        raise ValueError(kind)
    prefix = f"tfim8_a0_prepare_w{width}_{kind}"
    write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    return {
        "scope": "prepare_only",
        "w": int(width),
        "kind": kind,
        "width": int(qc.num_qubits),
        "stats": stats,
        "basis": recount_bases(qc, out_dir, prefix),
    }


def write_report(path: Path, rows: list[dict[str, Any]], coeff_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Sparse Trie State Preparation Probe",
        "",
        "This reports a direct prefix-controlled rotation trie.  It is correct but intentionally exposes the high-control decomposition cost.",
        "",
        "## Coefficient Support",
        "",
        "| w | support | slots | shortcut | extract time (s) |",
        "| ---: | ---: | ---: | --- | ---: |",
    ]
    for row in coeff_rows:
        lines.append(
            f"| {row['w']} | {row['support']} | {row['slots']} | "
            f"`{row.get('placement_shortcut')}` | {row['extract_seconds']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## PREPARE-only Counts",
            "",
            "| w | prepare | width | NAM gates | NAM non-Clifford | IBM gates | IBM non-Clifford |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        nam = row["basis"]["nam"]["summary"]
        ibm = row["basis"]["ibm"]["summary"]
        lines.append(
            f"| {row['w']} | {row['kind']} | {row['width']} | "
            f"{None if nam is None else nam['gate_count']} | "
            f"{None if nam is None else nam['non_clifford']} | "
            f"{None if ibm is None else ibm['gate_count']} | "
            f"{None if ibm is None else ibm['non_clifford']} |"
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
    ms = kraus_ops[0]

    rows = []
    coeff_rows = []
    for width in args.w:
        coeffs, meta = extract_coefficients(ms, width)
        coeff_rows.append(meta)
        weights = np.abs(coeffs)
        rows.append(run_prepare_case(weights, width, "dense", out_dir))
        rows.append(run_prepare_case(weights, width, "sparse_trie", out_dir))

    summary = {
        "script": str(Path(__file__).resolve()),
        "model_info": model_info,
        "coefficients": coeff_rows,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(out_dir / "report.md", rows, coeff_rows)


if __name__ == "__main__":
    main()
