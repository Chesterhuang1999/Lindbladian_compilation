#!/usr/bin/env python3
"""Probe MCG-tool compatibility on single-Kraus block-encoding circuits.

This is intentionally separate from the full channel-LCU MCG-tool runner.  It
extracts one nontrivial Kraus operator from each benchmark, builds the raw
legacy block-encoding circuit with ``BlockEncoding(...).circuit(opt="No")``,
and then tests the same direct/adapted MCG-tool paths used in the full-channel
experiment.
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

from block_encoding import BlockEncoding as LegacyBlockEncoding  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
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


RESULT_ROOT = ROOT / "Baseline_results" / "single_kraus_block_encoding_mcg_tools_v1"


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


def build_raw_block_encoding(ms: Any) -> Any:
    return LegacyBlockEncoding(copy.deepcopy(ms)).circuit(opt="No")


def build_matrix_order_block_encoding(ms: Any) -> Any:
    return NewBlockEncoding(copy.deepcopy(ms)).circuit(opt="Matrix-order")


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
        raw_stats = extract_stats(summary.get("raw_block", {}).get("basis_recounts", {}).get(basis))
        raw_total = int(raw_stats["metric_total"]) if raw_stats else None
        sources = [
            (
                "raw_block_qiskit_opt0",
                "Raw single-Kraus block-encoding + Qiskit opt0",
                raw_stats,
                summary.get("raw_block", {}).get("build_time_s"),
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
                "matrix_order_block",
                "Matrix-order block-encoding + Qiskit opt0 (comparison only)",
                extract_stats(summary.get("matrix_order_block", {}).get("basis_recounts", {}).get(basis)),
                summary.get("matrix_order_block", {}).get("build_and_recount_time_s"),
                summary.get("matrix_order_block", {}).get("status"),
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
            "input": "single selected Kraus operator block-encoding",
            "raw_block_encoding": 'Legacy BlockEncoding(...).circuit(opt="No")',
            "matrix_order_block": "comparison only; not fed to external tools",
            "external_tools_receive_project_optimized_circuit": False,
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

    status, raw_block, elapsed, error = timed(lambda: build_raw_block_encoding(selected.operator))
    summary["raw_block"] = {
        "status": status,
        "build_time_s": elapsed,
        "error": error,
        "basis_recounts": {},
    }
    if status != "ok":
        write_summary(summary, bench_dir / "summary.json")
        return summary

    summary["raw_block"]["qiskit_metrics"] = qiskit_metrics(raw_block)
    summary["raw_block"]["qasm"] = write_qasm2(raw_block, bench_dir / "raw_single_kraus_block_encoding.qasm")
    summary["raw_block"]["basis_recounts"] = recount_bases(raw_block, bench_dir, "raw_block")

    status, matrix_block, elapsed, error = timed(lambda: build_matrix_order_block_encoding(selected.operator))
    summary["matrix_order_block"] = {
        "status": status,
        "build_time_s": elapsed,
        "error": error,
        "basis_recounts": {},
    }
    if status == "ok":
        summary["matrix_order_block"]["qiskit_metrics"] = qiskit_metrics(matrix_block)
        summary["matrix_order_block"]["qasm"] = write_qasm2(
            matrix_block,
            bench_dir / "matrix_order_single_kraus_block_encoding.qasm",
        )
        recounted = recount_bases(matrix_block, bench_dir, "matrix_order_block")
        summary["matrix_order_block"]["basis_recounts"] = recounted
        summary["matrix_order_block"]["build_and_recount_time_s"] = float(
            elapsed + sum(row["elapsed_seconds"] for row in recounted.values() if row["status"] == "ok")
        )

    raw_width = int(summary["raw_block"]["qiskit_metrics"]["num_qubits"])
    matrix_width = None
    if summary["matrix_order_block"].get("status") == "ok":
        matrix_width = int(summary["matrix_order_block"]["qiskit_metrics"]["num_qubits"])
    workspace_budget = max(0, int(matrix_width - raw_width)) if matrix_width is not None else 0
    summary["workspace_budget_policy"] = {
        "raw_block_width": raw_width,
        "matrix_order_block_width": matrix_width,
        "workspace_for_qrisp_compile": int(workspace_budget),
        "notes": "Qrisp workspace uses the extra width of the matrix-order block-encoding as a resource budget only.",
    }

    summary["tools"] = {
        "qrisp": run_qrisp(raw_block, bench_dir, timeout_s),
        "qrisp_workspace_auto": run_qrisp_workspace_auto(raw_block, bench_dir, timeout_s, workspace_budget),
        "qclib_mcg_adapter": run_qclib_mcg_adapter(raw_block, bench_dir, timeout_s),
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
        "# Single-Kraus Block-Encoding MCG Tool Probe",
        "",
        "## Policy",
        "",
        "- Each benchmark contributes one selected nontrivial Kraus operator.",
        "- External tools receive the raw legacy `BlockEncoding(...).circuit(opt=\"No\")` circuit.",
        "- The matrix-order block-encoding is reported only as a comparison row and is not fed to external tools.",
        "- qclib uses the recursive `ControlledGate`/definition adapter to `qclib.gates.Mcg`; Qrisp uses the same direct import paths as the full-channel runner.",
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
            f"{policy.get('raw_block_width', '-')} | {policy.get('matrix_order_block_width', '-')} | "
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
            "| Benchmark | Qrisp default(s) | Qrisp workspace-auto(s) | qclib adapter(s) | Matrix-order block(s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bench in summary["benchmarks"]:
        tools = bench.get("tools", {})
        matrix = bench.get("matrix_order_block", {})

        def cell(value: Any) -> str:
            return "-" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {bench['benchmark']} | {cell(tools.get('qrisp', {}).get('elapsed_seconds_total'))} | "
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
        "experiment": "single_kraus_block_encoding_mcg_tools",
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
    report_path = out_dir / "single_kraus_block_encoding_mcg_tools_report.md"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    write_summary(summary, summary_path)
    write_report(summary, report_path)
    print(f"summary: {summary_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
