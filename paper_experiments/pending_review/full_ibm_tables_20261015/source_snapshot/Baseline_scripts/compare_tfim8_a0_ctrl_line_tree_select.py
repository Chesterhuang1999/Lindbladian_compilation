#!/usr/bin/env python3
"""Compare legacy ctrl-line SELECT with balanced-tree SELECT on TFIM_8 A0."""

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

from block_encoding import BlockEncoding  # type: ignore  # noqa: E402
from probe_single_kraus_select_mcg_tools import (  # type: ignore  # noqa: E402
    kraus_metrics,
    select_kraus_operator,
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


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_ctrl_line_vs_tree_select_v1"


def build_ctrl_line_select(ms: Any) -> tuple[Any, dict[str, Any]]:
    be = BlockEncoding(copy.deepcopy(ms))
    started = time.perf_counter()
    qc, tcount, mccount, cxcount = be.mulplex_U_opt(
        be.mat_list,
        be.ctrl_size,
        be.sys_size,
    )
    return qc, {
        "builder": "BlockEncoding.mulplex_U_opt(mat_list, ctrl_size, sys_size)",
        "build_elapsed_seconds": time.perf_counter() - started,
        "ctrl_size": int(be.ctrl_size),
        "sys_size": int(be.sys_size),
        "width": int(qc.num_qubits),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
    }


def build_tree_select(ms: Any) -> tuple[Any, dict[str, Any]]:
    be = BlockEncoding(copy.deepcopy(ms))
    started = time.perf_counter()
    qc, tcount, mccount, cxcount = be.mulplex_U_tree(
        be.mat_list,
        be.ctrl_size,
        be.sys_size,
    )
    return qc, {
        "builder": "BlockEncoding.mulplex_U_tree(mat_list, ctrl_size, sys_size)",
        "build_elapsed_seconds": time.perf_counter() - started,
        "ctrl_size": int(be.ctrl_size),
        "sys_size": int(be.sys_size),
        "width": int(qc.num_qubits),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
    }


def count_method(name: str, qc: Any, metadata: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": name,
        "metadata": metadata,
        "qiskit_metrics": qiskit_metrics(qc),
        "qasm": write_qasm2(qc, out_dir / f"{name}.qasm"),
        "basis_recounts": {},
    }
    for basis in ("nam", "ibm"):
        status, recount, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=out_dir / f"{name}_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        result["basis_recounts"][basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": recount,
            "error": error,
        }
    return result


def extract_basis_summary(row: dict[str, Any], basis: str) -> dict[str, Any]:
    entry = row["basis_recounts"][basis]
    if entry["status"] != "ok":
        return {"status": entry["status"], "error": entry["error"]}
    stats = entry["result"]["stats"]
    return {
        "width": int(stats["num_qubits"]),
        "gate_count": int(stats["metric_total"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "ops": stats["ops"],
        "transpile_elapsed_seconds": float(entry["elapsed_seconds"]),
    }


def markdown_table(summary: dict[str, Any]) -> str:
    rows = []
    for method in ("ctrl_line", "tree"):
        row = summary["results"][method]
        for basis in ("nam", "ibm"):
            basis_row = row["basis_summary"][basis]
            rows.append(
                [
                    method,
                    basis.upper(),
                    row["metadata"]["width"],
                    row["metadata"]["estimated_mccount"],
                    row["metadata"]["estimated_cxcount"],
                    basis_row.get("width", "-"),
                    basis_row.get("gate_count", "-"),
                    basis_row.get("non_clifford", "-"),
                    basis_row.get("ops", "-"),
                ]
            )

    lines = [
        "| method | basis | high-level width | est MCX-like | est CX | lowered width | gates | non-Clifford | ops |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> None:
    out_dir = RESULT_ROOT.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    selected = select_kraus_operator(kraus_ops)

    ctrl_line_qc, ctrl_line_meta = build_ctrl_line_select(selected.operator)
    tree_qc, tree_meta = build_tree_select(selected.operator)

    results = {
        "ctrl_line": count_method("ctrl_line", ctrl_line_qc, ctrl_line_meta, out_dir),
        "tree": count_method("tree", tree_qc, tree_meta, out_dir),
    }
    for row in results.values():
        row["basis_summary"] = {
            basis: extract_basis_summary(row, basis)
            for basis in ("nam", "ibm")
        }

    summary = {
        "experiment": "tfim8_a0_ctrl_line_vs_tree_select",
        "scope": "single Kraus SELECT only; no coefficient PREPARE/UNPREPARE",
        "result_dir": str(out_dir),
        "benchmark": "TFIM_8",
        "selected_kraus_index": int(selected.index),
        "model_info": model_info,
        "kraus_metrics": kraus_metrics(selected.operator),
        "results": results,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TFIM_8 A0 Ctrl-Line vs Tree SELECT",
        "",
        f"- Scope: {summary['scope']}",
        f"- Selected Kraus index: `{summary['selected_kraus_index']}`",
        f"- Pauli terms: `{summary['kraus_metrics']['pauli_terms']}`",
        "",
        markdown_table(summary),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
