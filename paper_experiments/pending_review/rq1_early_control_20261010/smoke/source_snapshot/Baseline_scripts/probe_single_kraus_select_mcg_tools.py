#!/usr/bin/env python3
"""Probe MCG-tool compatibility on single-Kraus SELECT/multiplexed-U circuits.

This runner intentionally excludes the coefficient PREPARE/UNPREPARE layers.
For each benchmark it selects one nontrivial Kraus operator and builds:

* raw SELECT: ``BlockEncoding(ms).mulplex_U(...)`` from ``src/block_encoding.py``;
* matrix-order SELECT: ``BlockEncoding(ms).mulplex_U_opt_order()`` from the same file.

Only the raw SELECT circuit is passed to Qrisp/qclib.  The matrix-order SELECT
is a direct in-house comparison row and is only lowered by Qiskit opt0 to the
common NAM/IBM counting bases.
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


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding import BlockEncoding  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BENCHMARKS,
    BenchmarkSpec,
    build_kraus_ops,
    extract_stats,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    run_qclib_mcg_adapter,
    run_qrisp,
    run_qrisp_workspace_auto,
    timed,
    write_qasm2,
)


RESULT_ROOT = ROOT / "Baseline_results" / "single_kraus_select_mcg_tools_v1"


@dataclass(frozen=True)
class SelectedKraus:
    index: int
    operator: Any


def select_kraus_operator(kraus_ops: list[Any]) -> SelectedKraus:
    nonempty = [(index, op) for index, op in enumerate(kraus_ops) if len(op.instances) > 0]
    if not nonempty:
        raise ValueError("benchmark has no nonempty Kraus operators")

    for index, op in nonempty:
        if len(op.instances) > 1:
            return SelectedKraus(index, op)
    return SelectedKraus(nonempty[0][0], nonempty[0][1])


def kraus_metrics(ms: Any) -> dict[str, Any]:
    return {
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "instances_sample": [
            {"pauli": repr(instance), "coeff": repr(coeff)}
            for instance, coeff in list(ms.instances)[:8]
        ],
    }


def build_raw_select(ms: Any) -> tuple[Any, dict[str, Any]]:
    be = BlockEncoding(copy.deepcopy(ms))
    qc, tcount, mccount, cxcount = be.mulplex_U(be.mat_list, be.ctrl_size, be.sys_size)
    return qc, {
        "builder": "BlockEncoding.mulplex_U(mat_list, ctrl_size, sys_size)",
        "ctrl_size": int(be.ctrl_size),
        "sys_size": int(be.sys_size),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
    }


def build_matrix_order_select(ms: Any) -> tuple[Any, dict[str, Any]]:
    be = BlockEncoding(copy.deepcopy(ms))
    qc, tcount, mccount, cxcount, ctrl_size = be.mulplex_U_opt_order()
    return qc, {
        "builder": "BlockEncoding.mulplex_U_opt_order()",
        "ctrl_size": int(ctrl_size),
        "sys_size": int(be.sys_size),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
        "additional_modes_count": int(len(getattr(be, "additional_modes", []))),
        "g_modes_count": int(len(getattr(be, "mobius_phase_result", {}).get("g_modes_with_phase", []))),
    }


def recount_bases(qc: Any, bench_dir: Path, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for basis in ("nam", "ibm"):
        status, recount, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=bench_dir / f"{prefix}_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        rows[basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": recount,
            "error": error,
        }
    return rows


def comparison_rows(benchmark: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for basis in ("nam", "ibm"):
        raw_stats = extract_stats(summary.get("raw_select", {}).get("basis_recounts", {}).get(basis))
        raw_total = int(raw_stats["metric_total"]) if raw_stats else None
        sources = [
            (
                "raw_select_qiskit_opt0",
                "Raw SELECT/mulplex_U + Qiskit opt0",
                raw_stats,
                summary.get("raw_select", {}).get("build_and_recount_time_s"),
                "raw",
            ),
            (
                "qrisp",
                "Qrisp direct import default(gray) + Qrisp transpile + Qiskit opt0",
                extract_stats(summary.get("tools", {}).get("qrisp", {}).get("basis_recounts", {}).get(basis)),
                summary.get("tools", {}).get("qrisp", {}).get("elapsed_seconds_total"),
                summary.get("tools", {}).get("qrisp", {}).get("status"),
            ),
            (
                "qrisp_workspace_auto",
                "Qrisp direct import + controlled-op auto + QuantumSession.compile(workspace) + Qiskit opt0",
                extract_stats(
                    summary.get("tools", {}).get("qrisp_workspace_auto", {}).get("basis_recounts", {}).get(basis)
                ),
                summary.get("tools", {}).get("qrisp_workspace_auto", {}).get("elapsed_seconds_total"),
                summary.get("tools", {}).get("qrisp_workspace_auto", {}).get("status"),
            ),
            (
                "qclib_mcg_adapter",
                "qclib recursive controlled-gate adapter + Mcg + Qiskit opt0",
                extract_stats(
                    summary.get("tools", {}).get("qclib_mcg_adapter", {}).get("basis_recounts", {}).get(basis)
                ),
                summary.get("tools", {}).get("qclib_mcg_adapter", {}).get("elapsed_seconds_total"),
                summary.get("tools", {}).get("qclib_mcg_adapter", {}).get("status"),
            ),
            (
                "matrix_order_select",
                "Matrix-order SELECT/mulplex_U_opt_order + Qiskit opt0",
                extract_stats(summary.get("matrix_order_select", {}).get("basis_recounts", {}).get(basis)),
                summary.get("matrix_order_select", {}).get("build_and_recount_time_s"),
                summary.get("matrix_order_select", {}).get("status"),
            ),
        ]

        for method, label, stats, elapsed, status in sources:
            if not stats:
                rows.append(
                    {
                        "benchmark": benchmark,
                        "basis": basis,
                        "method": method,
                        "label": label,
                        "status": status or "no_stats",
                        "metric_total": None,
                        "ratio_vs_raw": None,
                        "non_clifford": None,
                        "qubits": None,
                        "elapsed_seconds": elapsed,
                    }
                )
                continue

            total = int(stats["metric_total"])
            rows.append(
                {
                    "benchmark": benchmark,
                    "basis": basis,
                    "method": method,
                    "label": label,
                    "status": status or "ok",
                    "metric_total": total,
                    "ratio_vs_raw": None if raw_total in (None, 0) else float(total / raw_total),
                    "non_clifford": int(stats["clifford"]["non_clifford"]),
                    "qubits": int(stats["num_qubits"]),
                    "elapsed_seconds": elapsed,
                }
            )
    return rows


def run_benchmark(spec: BenchmarkSpec, out_dir: Path, timeout_s: float) -> dict[str, Any]:
    bench_dir = out_dir / "benchmarks" / spec.name
    bench_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "benchmark": spec.name,
        "family": spec.family,
        "size": int(spec.size),
        "out_dir": str(bench_dir),
        "policy": {
            "input": "single selected Kraus operator SELECT/multiplexed-U only",
            "raw_select": "BlockEncoding(...).mulplex_U(...) from src/block_encoding.py",
            "matrix_order_select": "BlockEncoding(...).mulplex_U_opt_order(); comparison only",
            "external_tools_receive_project_optimized_circuit": False,
            "coefficient_prepare_unprepare_included": False,
        },
    }

    status, value, elapsed, error = timed(lambda: build_kraus_ops(spec))
    summary["channel_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        write_summary(summary, bench_dir / "summary.json")
        return summary

    kraus_ops, model_info = value
    selected = select_kraus_operator(kraus_ops)
    summary["model_info"] = model_info
    summary["selected_kraus"] = {
        "index": int(selected.index),
        "metrics": kraus_metrics(selected.operator),
    }

    status, raw_value, elapsed, error = timed(lambda: build_raw_select(selected.operator))
    summary["raw_select"] = {
        "status": status,
        "build_time_s": elapsed,
        "error": error,
        "basis_recounts": {},
    }
    if status != "ok":
        write_summary(summary, bench_dir / "summary.json")
        return summary

    raw_select, raw_builder_metrics = raw_value
    summary["raw_select"]["builder_metrics"] = raw_builder_metrics
    summary["raw_select"]["qiskit_metrics"] = qiskit_metrics(raw_select)
    summary["raw_select"]["qasm"] = write_qasm2(raw_select, bench_dir / "raw_single_kraus_select.qasm")
    recounted = recount_bases(raw_select, bench_dir, "raw_select")
    summary["raw_select"]["basis_recounts"] = recounted
    summary["raw_select"]["build_and_recount_time_s"] = float(
        elapsed + sum(row["elapsed_seconds"] for row in recounted.values() if row["status"] == "ok")
    )

    status, matrix_value, elapsed, error = timed(lambda: build_matrix_order_select(selected.operator))
    summary["matrix_order_select"] = {
        "status": status,
        "build_time_s": elapsed,
        "error": error,
        "basis_recounts": {},
    }
    if status == "ok":
        matrix_select, matrix_builder_metrics = matrix_value
        summary["matrix_order_select"]["builder_metrics"] = matrix_builder_metrics
        summary["matrix_order_select"]["qiskit_metrics"] = qiskit_metrics(matrix_select)
        summary["matrix_order_select"]["qasm"] = write_qasm2(
            matrix_select,
            bench_dir / "matrix_order_single_kraus_select.qasm",
        )
        recounted = recount_bases(matrix_select, bench_dir, "matrix_order_select")
        summary["matrix_order_select"]["basis_recounts"] = recounted
        summary["matrix_order_select"]["build_and_recount_time_s"] = float(
            elapsed + sum(row["elapsed_seconds"] for row in recounted.values() if row["status"] == "ok")
        )

    raw_width = int(summary["raw_select"]["qiskit_metrics"]["num_qubits"])
    matrix_width = None
    if summary["matrix_order_select"].get("status") == "ok":
        matrix_width = int(summary["matrix_order_select"]["qiskit_metrics"]["num_qubits"])
    workspace_budget = max(0, int(matrix_width - raw_width)) if matrix_width is not None else 0
    summary["workspace_budget_policy"] = {
        "raw_select_width": raw_width,
        "matrix_order_select_width": matrix_width,
        "workspace_for_qrisp_compile": int(workspace_budget),
        "notes": "Qrisp workspace uses only the extra width of the matrix-order SELECT as a resource budget.",
    }

    summary["tools"] = {
        "qrisp": run_qrisp(raw_select, bench_dir, timeout_s),
        "qrisp_workspace_auto": run_qrisp_workspace_auto(raw_select, bench_dir, timeout_s, workspace_budget),
        "qclib_mcg_adapter": run_qclib_mcg_adapter(raw_select, bench_dir, timeout_s),
    }
    summary["comparison_rows"] = comparison_rows(spec.name, summary)
    write_summary(summary, bench_dir / "summary.json")
    return summary


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Benchmark | Basis | Method | Status | Qubits | Gates | G/G_raw | Non-Clifford | Time (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gates = "-" if row["metric_total"] is None else str(row["metric_total"])
        ratio = "-" if row["ratio_vs_raw"] is None else f"{row['ratio_vs_raw']:.6f}"
        non_clifford = "-" if row["non_clifford"] is None else str(row["non_clifford"])
        qubits = "-" if row["qubits"] is None else str(row["qubits"])
        elapsed = row.get("elapsed_seconds")
        elapsed_cell = "-" if elapsed is None else f"{float(elapsed):.3f}"
        lines.append(
            f"| {row['benchmark']} | {row['basis'].upper()} | {row['label']} | {row['status']} | "
            f"{qubits} | {gates} | {ratio} | {non_clifford} | {elapsed_cell} |"
        )
    return "\n".join(lines)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Single-Kraus SELECT MCG Tool Probe",
        "",
        "## Policy",
        "",
        "- Each benchmark contributes one selected nontrivial Kraus operator.",
        "- External tools receive only the raw `BlockEncoding(...).mulplex_U(...)` SELECT/multiplexed-U circuit.",
        "- The coefficient `StatePreparation` and inverse are intentionally excluded.",
        "- The matrix-order row is `BlockEncoding(...).mulplex_U_opt_order()` and is not fed to Qrisp/qclib.",
        "",
        "## Selected Kraus Operators",
        "",
        "| Benchmark | Kraus index | System qubits | Pauli terms | Raw width | Matrix-order width | Qrisp workspace |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bench in summary["benchmarks"]:
        selected = bench.get("selected_kraus", {})
        metrics = selected.get("metrics", {})
        policy = bench.get("workspace_budget_policy", {})
        lines.append(
            f"| {bench['benchmark']} | {selected.get('index', '-')} | "
            f"{metrics.get('system_qubits', '-')} | {metrics.get('pauli_terms', '-')} | "
            f"{policy.get('raw_select_width', '-')} | {policy.get('matrix_order_select_width', '-')} | "
            f"{policy.get('workspace_for_qrisp_compile', '-')} |"
        )

    lines.extend(
        [
            "",
            "## Performance Recount",
            "",
            markdown_table(summary["comparison_rows"]),
            "",
            "## Runtime Summary",
            "",
            "| Benchmark | Raw SELECT build+recount(s) | Qrisp default(s) | Qrisp workspace-auto(s) | qclib adapter(s) | Matrix-order SELECT(s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bench in summary["benchmarks"]:
        tools = bench.get("tools", {})
        raw = bench.get("raw_select", {})
        matrix = bench.get("matrix_order_select", {})

        def cell(value: Any) -> str:
            return "-" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {bench['benchmark']} | {cell(raw.get('build_and_recount_time_s'))} | "
            f"{cell(tools.get('qrisp', {}).get('elapsed_seconds_total'))} | "
            f"{cell(tools.get('qrisp_workspace_auto', {}).get('elapsed_seconds_total'))} | "
            f"{cell(tools.get('qclib_mcg_adapter', {}).get('elapsed_seconds_total'))} | "
            f"{cell(matrix.get('build_and_recount_time_s'))} |"
        )

    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- Machine-readable summary: `{summary['summary_path']}`")
    for bench in summary["benchmarks"]:
        lines.append(f"- `{bench['benchmark']}` summary: `{Path(bench['out_dir']) / 'summary.json'}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--benchmarks", nargs="+", default=[spec.name for spec in BENCHMARKS])
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {name.strip() for name in args.benchmarks}
    specs = [spec for spec in BENCHMARKS if spec.name in selected]
    summary: dict[str, Any] = {
        "experiment": "single_kraus_select_mcg_tools",
        "out_dir": str(out_dir),
        "benchmarks": [],
        "comparison_rows": [],
    }
    for spec in specs:
        started = time.perf_counter()
        bench_summary = run_benchmark(spec, out_dir, args.timeout_sec)
        bench_summary["elapsed_seconds_total"] = float(time.perf_counter() - started)
        summary["benchmarks"].append(bench_summary)
        summary["comparison_rows"].extend(bench_summary.get("comparison_rows", []))
        write_summary(summary, out_dir / "summary.partial.json")

    summary_path = out_dir / "summary.json"
    report_path = out_dir / "single_kraus_select_mcg_tools_report.md"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    write_summary(summary, summary_path)
    write_report(summary, report_path)
    print(f"summary: {summary_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
