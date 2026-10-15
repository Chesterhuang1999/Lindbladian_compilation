#!/usr/bin/env python3
"""Fill missing random-channel qiskit-opt3 IBM results.

This script only targets the random-channel seed rows used by the appendix
mean, i.e. seeds 0--4 for the N5 and N9 parameter sets. It writes results to a
fresh generic_tools_* directory so previous tool outputs are preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "Baseline_scripts"
SRC_DIR = ROOT / "src"
for path in (BASELINE_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


OUT_ROOT = ROOT / "Baseline_results" / "eval_redesign" / "random_channel_benchmarks"
RESULTS_CSV = ROOT / "Results" / "random_channel_benchmarks_summary.csv"
IBM_BASIS = ("u1", "u2", "u3", "cx")
PARAMETERS = {
    "N5_dim4_N1_10_N2_2": (5, "random_channel_N5_dim4_N1_10_N2_2_seed"),
    "N9_dim8_N1_20_N2_4": (9, "random_channel_N9_dim8_N1_20_N2_4_seed"),
}
SEEDS = set(range(5))
FILL_DIR_NAME = "generic_tools_qiskit_opt3_fill_20260706"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_total(stats: dict[str, Any]) -> int:
    if stats.get("metric_total") is not None:
        return int(stats["metric_total"])
    return int(stats["ibm"]["cx_u_total"])


def ibm_gate_counts(stats: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(stats, dict):
        return None
    ops = stats.get("ops", {}) or {}
    clifford = stats.get("clifford", {}) or {}
    return {
        "total": metric_total(stats),
        "cx": int(ops.get("cx", 0)),
        "u1": int(ops.get("u1", 0)),
        "u2": int(ops.get("u2", 0)),
        "u3": int(ops.get("u3", 0)),
        "clifford": int(clifford.get("clifford", 0)),
        "non_clifford": int(clifford.get("non_clifford", 0)),
    }


def comparison(before: dict[str, Any], after: dict[str, Any] | None) -> dict[str, Any] | None:
    if after is None:
        return None
    before_total = metric_total(before)
    after_total = metric_total(after)
    return {
        "total_ops_before": int(before_total),
        "total_ops_after": int(after_total),
        "total_ops_delta": int(after_total - before_total),
        "total_ops_reduction": int(before_total - after_total),
        "reduction_ratio": float((before_total - after_total) / before_total)
        if before_total
        else None,
    }


def status_rows(results_csv: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    if not results_csv.exists():
        return statuses
    for row in read_csv(results_csv):
        if (
            row.get("category") == "generic"
            and row.get("method") == "qiskit_opt3"
            and row.get("basis") == "ibm"
        ):
            statuses[row.get("benchmark", "")] = row.get("status", "")
    return statuses


def selected_benchmarks(args: argparse.Namespace) -> list[str]:
    statuses = status_rows(Path(args.results_csv))
    requested: list[str] = []
    parameter_labels = args.parameter or list(PARAMETERS)
    for label in parameter_labels:
        if label not in PARAMETERS:
            raise ValueError(f"Unknown parameter label: {label}")
        _n, prefix = PARAMETERS[label]
        for seed in args.seeds:
            if seed not in SEEDS:
                continue
            benchmark = f"{prefix}{seed}"
            if args.force or statuses.get(benchmark) != "ok":
                requested.append(benchmark)
    return requested


def build_summary_path(out_root: Path, benchmark: str) -> Path:
    return out_root / "cases" / benchmark / f"{benchmark}_build_summary.json"


def run_one(out_root: Path, benchmark: str, *, force: bool) -> dict[str, Any]:
    build_path = build_summary_path(out_root, benchmark)
    build_summary = load_json(build_path)
    case_dir = build_path.parent
    tool_dir = case_dir / FILL_DIR_NAME
    inputs_dir = tool_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    raw_source = Path(build_summary["raw"]["basis_outputs"]["ibm"]["qasm"])
    raw_input = inputs_dir / f"{benchmark}_ibm_opt0.qasm"
    output_qasm = inputs_dir / f"{benchmark}_ibm_opt3_qiskit.qasm"
    summary_path = tool_dir / f"{benchmark}_qiskit_opt3_fill_summary.json"
    native_summary_path = tool_dir / f"{benchmark}_native_basis_summary.json"
    command_result_path = tool_dir / "workflow_command_result.json"

    shutil.copyfile(raw_source, raw_input)
    raw_stats = count_qasm_file(raw_input, gate_set="ibm")

    started = time.perf_counter()
    status = "ok"
    error = None
    opt_stats = None
    existing_outputs = sorted(
        path
        for path in case_dir.glob("generic_tools*/inputs/*_ibm_opt3_qiskit.qasm")
        if path.resolve() != output_qasm.resolve()
    )
    if output_qasm.exists() and not force:
        opt_stats = count_qasm_file(output_qasm, gate_set="ibm")
        elapsed = None
        status = "ok_recount_only"
    elif existing_outputs and not force:
        shutil.copyfile(existing_outputs[0], output_qasm)
        opt_stats = count_qasm_file(output_qasm, gate_set="ibm")
        elapsed = None
        status = "ok_recount_only"
    else:
        try:
            qc = QuantumCircuit.from_qasm_file(str(raw_input))
            tqc = transpile(qc, basis_gates=list(IBM_BASIS), optimization_level=3)
            export_openqasm2_quartz_input(tqc, output_qasm, basis_gates=IBM_BASIS)
            opt_stats = count_qasm_file(output_qasm, gate_set="ibm")
            elapsed = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            status = "failed"
            error = repr(exc) + "\n" + traceback.format_exc()

    qiskit_summary = {
        "benchmark": benchmark,
        "tool": "qiskit_opt3",
        "basis": "ibm",
        "status": status,
        "input_qasm": str(raw_input),
        "output_qasm": str(output_qasm) if output_qasm.exists() else None,
        "elapsed_seconds": elapsed,
        "raw_stats": raw_stats,
        "output_stats": opt_stats,
        "error": error,
    }
    summary_path.write_text(
        json.dumps(qiskit_summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    native_row = {
        "basis": "ibm",
        "benchmark": benchmark,
        "comparison_vs_opt0": comparison(raw_stats, opt_stats),
        "elapsed_seconds": elapsed,
        "gate_counts": ibm_gate_counts(opt_stats),
        "method": "qiskit_opt3_ibm",
        "status": "ok" if opt_stats is not None else status,
    }
    native_summary = {
        "benchmark": benchmark,
        "comparison_mode": "native_basis",
        "rows": [
            native_row,
            {
                "basis": "nam",
                "benchmark": benchmark,
                "comparison_vs_opt0": None,
                "elapsed_seconds": None,
                "gate_counts": None,
                "method": "qiskit_opt3_nam",
                "status": "skipped",
            },
        ],
        "tool_results": [
            {
                "tool": "qiskit_opt3",
                "basis": "ibm",
                "status": native_row["status"],
                "input": str(raw_input),
                "native_output": str(output_qasm) if output_qasm.exists() else None,
                "summary": str(summary_path),
                "elapsed_seconds": elapsed,
            }
        ],
    }
    native_summary_path.write_text(
        json.dumps(native_summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    command_result = {
        "benchmark": benchmark,
        "tool": "qiskit_opt3",
        "returncode": 0 if opt_stats is not None else 1,
        "timed_out": False,
        "elapsed_seconds": elapsed,
        "native_summary": str(native_summary_path),
        "summary": str(summary_path),
        "error": error,
    }
    command_result_path.write_text(
        json.dumps(command_result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "benchmark": benchmark,
        "status": native_row["status"],
        "raw_gates": metric_total(raw_stats),
        "opt_gates": metric_total(opt_stats) if opt_stats is not None else None,
        "elapsed_seconds": elapsed,
        "native_summary": str(native_summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--parameter", action="append", choices=sorted(PARAMETERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=sorted(SEEDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    benchmarks = selected_benchmarks(args)
    if not benchmarks:
        print("no missing qiskit_opt3 random-channel IBM rows")
        return 0

    print("selected benchmarks:")
    for benchmark in benchmarks:
        print(f"  {benchmark}")

    results = []
    for benchmark in benchmarks:
        print(f"[qiskit-opt3] {benchmark}", flush=True)
        result = run_one(out_root, benchmark, force=args.force)
        results.append(result)
        print(
            f"[done] {benchmark} status={result['status']} "
            f"raw={result['raw_gates']} opt={result['opt_gates']} "
            f"time={result['elapsed_seconds']}",
            flush=True,
        )

    print(json.dumps(results, indent=2, sort_keys=True, default=str))
    return 0 if all(result["opt_gates"] is not None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
