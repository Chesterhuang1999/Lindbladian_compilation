#!/usr/bin/env python3
"""TFIM-4 A0 SELECT-only Qrisp comparison.

This keeps the same scope as the qclib McxVchainDirty Mode A probe: only the
SELECT/multiplexed-U block of the first TFIM-4 Kraus operator A0 is tested.
Coefficient PREPARE/UNPREPARE layers are intentionally excluded.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from compare_tfim4_a0_select_qclib_mcxvchain_mode_a import (  # type: ignore  # noqa: E402
    build_matrix_order_select,
    build_raw_select,
    build_tfim4_a0,
    pauli_expression_from_rows,
    pauli_sum_rows,
    verify_a0_global_phase,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    extract_stats,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    run_qrisp,
    run_qrisp_workspace_auto,
    timed,
    write_qasm2,
)


OUT_DIR = ROOT / "Baseline_results" / "tfim4_a0_select_qrisp_v1"


def recount_bases(qc: Any, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for basis in ("ibm", "nam"):
        status, recount, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=OUT_DIR / f"{prefix}_{basis}_opt0.qasm",
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


def count_local_select(qc: Any, key: str, label: str, metadata: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = {
        "key": key,
        "label": label,
        "status": "ok",
        "metadata": metadata,
        "qiskit_metrics": qiskit_metrics(qc),
        "qasm": write_qasm2(qc, OUT_DIR / f"{key}_source.qasm"),
    }
    result["basis_recounts"] = recount_bases(qc, key)
    result["elapsed_seconds_total"] = float(time.perf_counter() - started)
    return result


def summarize_basis(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    stats = extract_stats(entry)
    if not stats:
        return None
    return {
        "width": int(stats["num_qubits"]),
        "gate_count": int(stats["metric_total"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "ops": stats["ops"],
    }


def comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    raw_by_basis = {}
    raw = summary["circuits"]["raw_select"]
    for basis in ("ibm", "nam"):
        raw_stats = summarize_basis(raw.get("basis_recounts", {}).get(basis))
        if raw_stats:
            raw_by_basis[basis] = int(raw_stats["gate_count"])

    entries = [
        ("raw_select", "Qiskit raw mulplex_U SELECT"),
        ("qrisp_default", "Qrisp direct import default(gray)"),
        ("qrisp_workspace_auto", "Qrisp workspace-auto compile"),
        ("ours_matrix_order_select", "ours matrix-order SELECT expanded"),
    ]
    for key, label in entries:
        item = summary["circuits"][key]
        for basis in ("ibm", "nam"):
            stats = summarize_basis(item.get("basis_recounts", {}).get(basis))
            rows.append(
                {
                    "variant": label,
                    "key": key,
                    "basis": basis,
                    "status": item.get("status"),
                    "width": None if stats is None else stats["width"],
                    "gate_count": None if stats is None else stats["gate_count"],
                    "non_clifford": None if stats is None else stats["non_clifford"],
                    "ratio_vs_raw": (
                        None
                        if stats is None or basis not in raw_by_basis or raw_by_basis[basis] == 0
                        else float(stats["gate_count"] / raw_by_basis[basis])
                    ),
                    "ops": None if stats is None else stats["ops"],
                    "elapsed_seconds": item.get("elapsed_seconds_total"),
                }
            )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TFIM_4 A0 SELECT: Qrisp",
        "",
        f"- A0 global-phase verification passed: `{summary['a0_global_phase_check']['passed_up_to_global_phase']}`",
        f"- A0 phase vs canonical: `{summary['a0_global_phase_check']['phase_actual_over_canonical']}`",
        f"- Qrisp workspace-auto budget: `{summary['workspace_budget']}` extra qubit",
        "",
        "| Variant | Basis | Status | Width | Gates | Non-Clifford | Ratio vs raw | Time (s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison_rows"]:
        width = "-" if row["width"] is None else str(row["width"])
        gates = "-" if row["gate_count"] is None else str(row["gate_count"])
        nc = "-" if row["non_clifford"] is None else str(row["non_clifford"])
        ratio = "-" if row["ratio_vs_raw"] is None else f"{row['ratio_vs_raw']:.3f}"
        elapsed = "-" if row["elapsed_seconds"] is None else f"{float(row['elapsed_seconds']):.3f}"
        lines.append(
            f"| {row['variant']} | {row['basis'].upper()} | `{row['status']}` | "
            f"{width} | {gates} | {nc} | {ratio} | {elapsed} |"
        )
    lines.extend(
        [
            "",
            "## Pauli Sum",
            "",
            "```text",
            summary["pauli_sum_expression"],
            "```",
            "",
            "## Artifacts",
            "",
            f"- JSON summary: `{summary['summary_path']}`",
            f"- Output directory: `{summary['out_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ms, metadata = build_tfim4_a0()
    rows = pauli_sum_rows(ms)
    expression = pauli_expression_from_rows(rows)
    a0_check = verify_a0_global_phase(ms)

    raw_select, raw_meta = build_raw_select(ms)
    matrix_select, matrix_meta = build_matrix_order_select(ms)
    workspace_budget = max(0, int(matrix_select.num_qubits - raw_select.num_qubits))

    summary: dict[str, Any] = {
        "experiment": "tfim4_a0_select_qrisp",
        "out_dir": str(OUT_DIR.resolve()),
        "metadata": metadata,
        "pauli_sum_expression": expression,
        "pauli_sum_rows": rows,
        "a0_global_phase_check": a0_check,
        "workspace_budget": int(workspace_budget),
        "policy": {
            "scope": "TFIM_4 A0 SELECT/mulplex_U only",
            "qrisp_input": "raw BlockEncoding.mulplex_U Qiskit QuantumCircuit via Qrisp QuantumCircuit.from_qiskit",
            "prepare_unprepare_included": False,
        },
        "circuits": {},
    }
    summary["circuits"]["raw_select"] = count_local_select(
        raw_select,
        "raw_select",
        "Qiskit raw mulplex_U SELECT",
        raw_meta,
    )
    summary["circuits"]["ours_matrix_order_select"] = count_local_select(
        matrix_select,
        "ours_matrix_order_select",
        "ours matrix-order SELECT expanded",
        matrix_meta,
    )
    summary["circuits"]["qrisp_default"] = run_qrisp(copy.deepcopy(raw_select), OUT_DIR, timeout_s=1800.0)
    summary["circuits"]["qrisp_workspace_auto"] = run_qrisp_workspace_auto(
        copy.deepcopy(raw_select),
        OUT_DIR,
        timeout_s=1800.0,
        workspace_budget=workspace_budget,
    )
    summary["comparison_rows"] = comparison_rows(summary)

    summary_path = OUT_DIR / "summary.json"
    report_path = OUT_DIR / "summary.md"
    summary["summary_path"] = str(summary_path.resolve())
    summary["report_path"] = str(report_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    report_path.write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
