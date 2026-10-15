#!/usr/bin/env python3
"""Compare row-increasing isometry representations in channel-LCU NAM counts.

This experiment uses a 7-cycle incidence channel.  One representation uses a
minimal 6-row Cholesky-like Kraus factor; the other uses the sparse 7-row
incidence factor.  They have the same Gram matrix A^dagger A and therefore
represent the same channel, but the sparse factor increases the Kraus count.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from channel_IR import Matrixsum, PauliAtom, channel_ensemble  # type: ignore  # noqa: E402
import channel_LCU  # type: ignore  # noqa: E402


NAM_TRANSPILER_BASIS = ("h", "x", "rz", "cx", "reset")
NAM_METRIC_GATES = ("rz", "h", "x", "cx")
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "isometry_row_increase_ctrl_line"


def ceil_log2(value: int) -> int:
    return 0 if value <= 1 else int(math.ceil(math.log2(value)))


def control_proxy_cost(kraus_ops: list[Matrixsum]) -> int:
    nonzero = [op for op in kraus_ops if len(op.instances) > 0]
    m = len(nonzero)
    return m * ceil_log2(m) + sum(ceil_log2(len(op.instances)) for op in nonzero)


def row_supports(kraus_ops: list[Matrixsum]) -> list[int]:
    return [len(op.instances) for op in kraus_ops if len(op.instances) > 0]


def random_pauli_labels(num_qubits: int, count: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    labels: set[str] = set()
    alphabet = np.array(list("IXYZ"))

    while len(labels) < count:
        label = "".join(rng.choice(alphabet, size=num_qubits).tolist())
        if label != "I" * num_qubits:
            labels.add(label)

    return sorted(labels)


def cycle_incidence_matrix(cycle_size: int) -> np.ndarray:
    matrix = np.zeros((cycle_size, cycle_size), dtype=float)
    for row in range(cycle_size):
        matrix[row, row] = 1.0
        matrix[row, (row + 1) % cycle_size] = -1.0
    return matrix


def cholesky_like_minimal_factor(incidence: np.ndarray) -> np.ndarray:
    """Return a full-row-rank factor B with B^T B = incidence^T incidence."""
    gram = incidence.T @ incidence
    minor = gram[:-1, :-1]
    upper = np.linalg.cholesky(minor).T
    factor = np.zeros((incidence.shape[1] - 1, incidence.shape[1]), dtype=float)
    factor[:, :-1] = upper
    factor[:, -1] = -np.sum(factor[:, :-1], axis=1)
    return factor


def matrix_to_kraus_ops(matrix: np.ndarray, labels: list[str], tol: float) -> list[Matrixsum]:
    kraus_ops: list[Matrixsum] = []
    for row in matrix:
        instances = []
        for value, label in zip(row, labels):
            if abs(value) <= tol:
                continue
            phase = value / abs(value)
            instances.append((PauliAtom(label, phase=phase), float(abs(value))))
        kraus_ops.append(Matrixsum(instances))
    return kraus_ops


def channel_metrics(kraus_ops: list[Matrixsum]) -> dict[str, Any]:
    return {
        "kraus_count": len([op for op in kraus_ops if len(op.instances) > 0]),
        "row_supports": row_supports(kraus_ops),
        "pauli_terms_total": sum(row_supports(kraus_ops)),
        "proxy_control_cost": control_proxy_cost(kraus_ops),
        "coeff_sums": [float(op.pauli_norm()) for op in kraus_ops],
    }


def build_channel_lcu_circuit(
    kraus_ops: list[Matrixsum],
    *,
    structure: str,
    opt: str,
) -> tuple[Any, Any, float]:
    started = time.perf_counter()
    qc, qubit_indexes = channel_LCU.channel_to_LCU(
        channel_ensemble([kraus_ops]),
        structure=structure,
        opt=opt,
    )
    return qc, qubit_indexes, time.perf_counter() - started


def nam_transpile_metrics(qc, optimization_level: int) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    tqc = transpile(
        qc,
        basis_gates=list(NAM_TRANSPILER_BASIS),
        optimization_level=optimization_level,
    )
    elapsed = time.perf_counter() - started
    ops = {str(name): int(count) for name, count in tqc.count_ops().items()}
    return {
        "num_qubits": int(tqc.num_qubits),
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "nam_total": int(sum(ops.get(gate, 0) for gate in NAM_METRIC_GATES)),
        "ops": ops,
    }, elapsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = random_pauli_labels(args.num_qubits, args.cycle_size, args.seed)
    sparse_matrix = cycle_incidence_matrix(args.cycle_size)
    minimal_matrix = cholesky_like_minimal_factor(sparse_matrix)

    gram_error = float(
        np.max(np.abs(minimal_matrix.T @ minimal_matrix - sparse_matrix.T @ sparse_matrix))
    )
    if gram_error > args.tol:
        raise ValueError(f"Gram equality failed: {gram_error}")

    cases = [
        ("minimal_B", minimal_matrix),
        ("sparse_E", sparse_matrix),
    ]

    rows = []
    details: dict[str, Any] = {}
    for name, matrix in cases:
        kraus_ops = matrix_to_kraus_ops(matrix, labels, args.tol)
        metrics = channel_metrics(kraus_ops)
        qc, qubit_indexes, build_time = build_channel_lcu_circuit(
            kraus_ops,
            structure="opt",
            opt=args.block_opt,
        )
        nam_metrics, transpile_time = nam_transpile_metrics(
            qc,
            optimization_level=args.nam_optimization_level,
        )

        row = {
            "representation": name,
            **metrics,
            **nam_metrics,
            "build_time_s": float(build_time),
            "transpile_time_s": float(transpile_time),
        }
        rows.append(row)
        details[name] = {
            "coefficient_matrix": matrix.tolist(),
            "qubit_indexes": qubit_indexes,
            "channel_metrics": metrics,
            "circuit_metrics": nam_metrics,
            "build_time_s": float(build_time),
            "transpile_time_s": float(transpile_time),
        }

    minimal = rows[0]
    sparse = rows[1]
    comparison = {
        "sparse_minus_minimal_nam_total": int(sparse["nam_total"] - minimal["nam_total"]),
        "sparse_to_minimal_nam_ratio": float(sparse["nam_total"] / minimal["nam_total"]),
        "sparse_minus_minimal_proxy_cost": int(
            sparse["proxy_control_cost"] - minimal["proxy_control_cost"]
        ),
    }

    summary = {
        "experiment": "isometry_row_increase_channel_lcu_nam",
        "num_qubits": int(args.num_qubits),
        "cycle_size": int(args.cycle_size),
        "seed": int(args.seed),
        "pauli_labels": labels,
        "structure": "opt",
        "logical_structure": "ctrl-line",
        "block_opt": args.block_opt,
        "nam_transpiler_basis": list(NAM_TRANSPILER_BASIS),
        "nam_metric_gates": list(NAM_METRIC_GATES),
        "nam_optimization_level": int(args.nam_optimization_level),
        "gram_error": gram_error,
        "rows": rows,
        "details": details,
        "comparison": comparison,
    }

    summary_path = out_dir / "isometry_row_increase_ctrl_line_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def format_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Representation | Kraus | Supports | Proxy C | Qubits | Depth | NAM total | rz | h | x | cx | reset | Build s | Transpile s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ops = row["ops"]
        lines.append(
            f"| {row['representation']} | {row['kraus_count']} | {row['row_supports']} | "
            f"{row['proxy_control_cost']} | {row['num_qubits']} | {row['depth']} | "
            f"{row['nam_total']} | {ops.get('rz', 0)} | {ops.get('h', 0)} | "
            f"{ops.get('x', 0)} | {ops.get('cx', 0)} | {ops.get('reset', 0)} | "
            f"{row['build_time_s']:.3f} | {row['transpile_time_s']:.3f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 6-row minimal and 7-row sparse isometry-equivalent channel-LCU circuits."
    )
    parser.add_argument("--num-qubits", type=int, default=3)
    parser.add_argument("--cycle-size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument(
        "--block-opt",
        choices=("No", "Ctrl-line", "Matrix-order"),
        default="Ctrl-line",
        help=(
            "Per-Kraus BlockEncoding option. The channel-level structure is always "
            "'opt' (logical ctrl-line)."
        ),
    )
    parser.add_argument("--nam-optimization-level", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(format_table(summary["rows"]))
    print()
    print(f"Pauli labels: {summary['pauli_labels']}")
    print(f"Gram error: {summary['gram_error']:.3e}")
    print(f"Comparison: {summary['comparison']}")
    print(f"Summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
