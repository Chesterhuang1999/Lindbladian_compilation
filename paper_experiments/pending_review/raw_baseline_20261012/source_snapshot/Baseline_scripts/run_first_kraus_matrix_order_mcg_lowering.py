#!/usr/bin/env python3
"""Lower first-Kraus matrix-order block encodings through Qrisp and qclib.

For each benchmark, this runner:

1. selects the first nonempty Kraus operator;
2. records its Pauli-sum expression;
3. builds ``NewBlockEncoding(kraus).circuit(opt="Matrix-order")``;
4. sends that matrix-order circuit through Qrisp and qclib lowering paths;
5. recounts all outputs after Qiskit opt0 lowering to NAM and IBM bases.

This is a post-matrix-order lowering experiment, not a raw full-channel
baseline.
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

from block_encoding import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
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


RESULT_ROOT = ROOT / "Baseline_results" / "first_kraus_matrix_order_mcg_lowering_v1"


@dataclass(frozen=True)
class SelectedKraus:
    index: int
    operator: Any


def select_first_nonempty_kraus(kraus_ops: list[Any]) -> SelectedKraus:
    for index, op in enumerate(kraus_ops):
        if len(op.instances) > 0:
            return SelectedKraus(index, op)
    raise ValueError("benchmark has no nonempty Kraus operators")


def scalar_to_string(value: Any) -> str:
    text = repr(value)
    for prefix in ("np.float64(", "np.complex128("):
        if text.startswith(prefix) and text.endswith(")"):
            return text[len(prefix) : -1]
    return text


def pauli_label(instance: Any) -> str:
    for attr in ("to_label",):
        method = getattr(instance, attr, None)
        if callable(method):
            try:
                return str(method())
            except Exception:
                pass
    for attr in ("label", "pauli", "name"):
        value = getattr(instance, attr, None)
        if value is not None:
            return str(value)
    text = repr(instance)
    marker = " * "
    if marker in text:
        return text.rsplit(marker, 1)[-1]
    return text


def pauli_phase(instance: Any) -> str | None:
    text = repr(instance)
    marker = " * "
    if marker not in text:
        return None
    prefix = text.split(marker, 1)[0]
    if ": " in prefix:
        return prefix.split(": ", 1)[1]
    return prefix


def kraus_pauli_sum(ms: Any) -> dict[str, Any]:
    terms = []
    expression_terms = []
    for instance, coeff in ms.instances:
        label = pauli_label(instance)
        phase = pauli_phase(instance)
        coeff_text = scalar_to_string(coeff)
        if phase and phase not in {"(1+0j)", "1", "1.0"}:
            term_coeff = f"({coeff_text})*({phase})"
        else:
            term_coeff = coeff_text
        terms.append(
            {
                "coefficient": coeff_text,
                "pauli": label,
                "instance_phase": phase,
                "repr": repr(instance),
            }
        )
        expression_terms.append(f"({term_coeff}) {label}")
    return {
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "terms": terms,
        "expression": " + ".join(expression_terms),
    }


def build_matrix_order_block(ms: Any) -> Any:
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
        matrix_stats = extract_stats(summary.get("matrix_order_block", {}).get("basis_recounts", {}).get(basis))
        matrix_total = int(matrix_stats["metric_total"]) if matrix_stats else None
        sources = [
            (
                "matrix_order_block",
                "Matrix-order block-encoding + Qiskit opt0",
                matrix_stats,
                summary.get("matrix_order_block", {}).get("build_and_recount_time_s"),
                summary.get("matrix_order_block", {}).get("status"),
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
                        "ratio_vs_matrix_order": None,
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
                    "ratio_vs_matrix_order": None if matrix_total in (None, 0) else float(total / matrix_total),
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
            "input": "first nonempty Kraus operator",
            "encoding": 'NewBlockEncoding(...).circuit(opt="Matrix-order")',
            "experiment_type": "post-matrix-order lowering through MCG tools",
        },
    }

    status, value, elapsed, error = timed(lambda: build_kraus_ops(spec))
    summary["channel_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        write_summary(summary, bench_dir / "summary.json")
        return summary

    kraus_ops, model_info = value
    selected = select_first_nonempty_kraus(kraus_ops)
    summary["model_info"] = model_info
    summary["selected_kraus"] = {
        "index": int(selected.index),
        "pauli_sum": kraus_pauli_sum(selected.operator),
    }

    status, matrix_block, elapsed, error = timed(lambda: build_matrix_order_block(selected.operator))
    summary["matrix_order_block"] = {
        "status": status,
        "build_time_s": elapsed,
        "error": error,
        "basis_recounts": {},
    }
    if status != "ok":
        write_summary(summary, bench_dir / "summary.json")
        return summary

    summary["matrix_order_block"]["qiskit_metrics"] = qiskit_metrics(matrix_block)
    summary["matrix_order_block"]["qasm"] = write_qasm2(
        matrix_block,
        bench_dir / "first_kraus_matrix_order_block_encoding.qasm",
    )
    recounted = recount_bases(matrix_block, bench_dir, "matrix_order_block")
    summary["matrix_order_block"]["basis_recounts"] = recounted
    summary["matrix_order_block"]["build_and_recount_time_s"] = float(
        elapsed + sum(row["elapsed_seconds"] for row in recounted.values() if row["status"] == "ok")
    )

    matrix_width = int(summary["matrix_order_block"]["qiskit_metrics"]["num_qubits"])
    summary["workspace_budget_policy"] = {
        "matrix_order_width": matrix_width,
        "workspace_for_qrisp_compile": 0,
        "notes": "No extra workspace is granted beyond the matrix-order circuit width in this lowering experiment.",
    }

    summary["tools"] = {
        "qrisp": run_qrisp(matrix_block, bench_dir, timeout_s),
        "qrisp_workspace_auto": run_qrisp_workspace_auto(matrix_block, bench_dir, timeout_s, 0),
        "qclib_mcg_adapter": run_qclib_mcg_adapter(matrix_block, bench_dir, timeout_s),
    }
    summary["comparison_rows"] = comparison_rows(spec.name, summary)
    write_summary(summary, bench_dir / "summary.json")
    return summary


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Benchmark | Basis | Method | Status | Qubits | Gates | G/G_matrix | Non-Clifford | Time (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gates = "-" if row["metric_total"] is None else str(row["metric_total"])
        ratio = "-" if row["ratio_vs_matrix_order"] is None else f"{row['ratio_vs_matrix_order']:.6f}"
        non_cliff = "-" if row["non_clifford"] is None else str(row["non_clifford"])
        qubits = "-" if row["qubits"] is None else str(row["qubits"])
        elapsed = row.get("elapsed_seconds")
        elapsed_cell = "-" if elapsed is None else f"{float(elapsed):.3f}"
        lines.append(
            f"| {row['benchmark']} | {row['basis'].upper()} | {row['label']} | {row['status']} | "
            f"{qubits} | {gates} | {ratio} | {non_cliff} | {elapsed_cell} |"
        )
    return "\n".join(lines)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# First-Kraus Matrix-Order MCG Lowering",
        "",
        "## Policy",
        "",
        "- Each benchmark uses the first nonempty Kraus operator.",
        "- The input circuit for Qrisp and qclib is `NewBlockEncoding(kraus).circuit(opt=\"Matrix-order\")`.",
        "- This is a post-matrix-order lowering experiment, not a raw external-tool baseline.",
        "",
        "## Selected Kraus Pauli Sums",
        "",
    ]
    for bench in summary["benchmarks"]:
        selected = bench.get("selected_kraus", {})
        pauli_sum = selected.get("pauli_sum", {})
        lines.extend(
            [
                f"### {bench['benchmark']}",
                "",
                f"- Kraus index: `{selected.get('index', '-')}`",
                f"- System qubits: `{pauli_sum.get('system_qubits', '-')}`",
                f"- Pauli terms: `{pauli_sum.get('pauli_terms', '-')}`",
                f"- Pauli norm: `{pauli_sum.get('pauli_norm', '-')}`",
                "",
                "```text",
                str(pauli_sum.get("expression", "")),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Performance Recount",
            "",
            markdown_table(summary["comparison_rows"]),
            "",
            "## Runtime Summary",
            "",
            "| Benchmark | Matrix-order build+recount(s) | Qrisp default(s) | Qrisp auto(s) | qclib adapter(s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bench in summary["benchmarks"]:
        matrix = bench.get("matrix_order_block", {})
        tools = bench.get("tools", {})

        def cell(value: Any) -> str:
            return "-" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {bench['benchmark']} | {cell(matrix.get('build_and_recount_time_s'))} | "
            f"{cell(tools.get('qrisp', {}).get('elapsed_seconds_total'))} | "
            f"{cell(tools.get('qrisp_workspace_auto', {}).get('elapsed_seconds_total'))} | "
            f"{cell(tools.get('qclib_mcg_adapter', {}).get('elapsed_seconds_total'))} |"
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
        "experiment": "first_kraus_matrix_order_mcg_lowering",
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
    report_path = out_dir / "first_kraus_matrix_order_mcg_lowering_report.md"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    write_summary(summary, summary_path)
    write_report(summary, report_path)
    print(f"summary: {summary_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
