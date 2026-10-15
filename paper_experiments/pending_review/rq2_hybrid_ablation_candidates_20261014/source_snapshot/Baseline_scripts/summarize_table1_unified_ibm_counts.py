#!/usr/bin/env python3
"""Collect compact table1 unified IBM-count results across n values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def elapsed_seconds(summary: dict[str, Any] | None) -> float | None:
    if summary is None:
        return None
    process = summary.get("process")
    if isinstance(process, dict) and process.get("elapsed_seconds") is not None:
        return float(process["elapsed_seconds"])
    pytket_stats = summary.get("pytket_stats")
    if isinstance(pytket_stats, dict) and pytket_stats.get("elapsed_seconds") is not None:
        return float(pytket_stats["elapsed_seconds"])
    return None


def count_ibm_qasm(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    counts = {"u1": 0, "u2": 0, "u3": 0, "cx": 0}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        name = line.split("(", 1)[0].split()[0]
        if name in counts:
            counts[name] += 1
    counts["total"] = sum(counts.values())
    return counts


def rows_for_n(root: Path, n: int) -> list[dict[str, Any]]:
    n_dir = root / f"n{n}"
    summary = load_json(n_dir / f"test_table1_n{n}_unified_ibm_count_summary.json")
    rows_by_method = {}
    if summary is not None:
        for row in summary.get("rows", []):
            rows_by_method[str(row["method"])] = dict(row)

    rows: list[dict[str, Any]] = []
    for method in ("qiskit_opt0_ibm", "qiskit_opt3_ibm", "feynman", "voqc", "pytket", "bqskit"):
        base_row = rows_by_method.get(
            method,
            {
                "benchmark": f"test_table1_select_n{n}",
                "method": method,
                "gate_counts": None,
            },
        )
        native_summary = None
        if method in {"feynman", "voqc", "pytket", "bqskit"}:
            native_summary = load_json(n_dir / method / f"{method}_native_summary.json")
            counted = count_ibm_qasm(n_dir / method / f"{method}_qiskit_ibm_counted.qasm")
            if counted is not None:
                base_row["gate_counts"] = counted
        base_row["runtime_seconds"] = elapsed_seconds(native_summary)
        if method == "bqskit" and native_summary is None and n in {12, 20}:
            base_row["status"] = "not_completed"
            base_row["note"] = (
                "n=12 exceeded the 1h limit without writing output; "
                "n=20 was not started under the same configuration."
            )
        rows.append(base_row)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.result_dir).resolve()
    all_rows: list[dict[str, Any]] = []
    for n in args.num_qubits:
        all_rows.extend(rows_for_n(root, int(n)))
    summary = {
        "benchmark": "test_table1_select",
        "final_count_basis": "ibm",
        "rows": all_rows,
    }
    output_path = root / "table1_unified_ibm_count_check_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="Baseline_results/table1_unified_ibm_count_check",
    )
    parser.add_argument("--num-qubits", nargs="+", type=int, default=[8, 12, 20])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
