#!/usr/bin/env python3
"""Compare TFIM_4 A0 block-encoding opt entry points."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding import BlockEncoding as LegacyBlockEncoding  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from probe_spire_tfim4_a0_select_adapter import (  # type: ignore  # noqa: E402
    build_tfim4_a0_matrixsum,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    write_qasm2,
)


RESULT_ROOT = ROOT / "Baseline_results" / "tfim4_a0_block_opt_entries_v1"


def recount(qc: Any, out_dir: Path, name: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for basis in ("nam", "ibm"):
        try:
            result = recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=out_dir / f"{name}_{basis}_opt0.qasm",
                optimization_level=0,
            )
            stats = result["stats"]
            rows[basis] = {
                "status": "ok",
                "metric_total": int(stats["metric_total"]),
                "non_clifford": int(stats["clifford"]["non_clifford"]),
                "ops": {str(k): int(v) for k, v in stats["ops"].items()},
            }
        except Exception as exc:
            rows[basis] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return rows


def build_entries(matrixsum: Any) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for scope in ("select", "block"):
        for opt in ("No", "Ctrl-line", "Tree"):
            block = LegacyBlockEncoding(copy.deepcopy(matrixsum))
            qc = block.select_circuit(opt=opt) if scope == "select" else block.circuit(opt=opt)
            entries[f"{scope}_{opt.lower().replace('-', '_')}"] = {
                "scope": scope,
                "backend": "src/block_encoding.py",
                "opt": opt,
                "qc": qc,
                "metadata": {
                    "ctrl_size": int(block.ctrl_size),
                    "sys_size": int(block.sys_size),
                    "width": int(qc.num_qubits),
                    "estimated_tcount": getattr(block, "tcount", None),
                    "estimated_mccount": getattr(block, "mccount", None),
                    "estimated_cxcount": getattr(block, "cxcount", None),
                },
            }

    block = NewBlockEncoding(copy.deepcopy(matrixsum))
    qc = block.select_circuit(opt="Matrix-order")
    entries["select_matrix_order"] = {
        "scope": "select",
        "backend": "src/block_encoding_new.py",
        "opt": "Matrix-order",
        "qc": qc,
        "metadata": {
            "ctrl_size": int(getattr(block, "ctrl_size", qc.num_qubits - matrixsum.size)),
            "sys_size": int(block.sys_size),
            "width": int(qc.num_qubits),
            "estimated_tcount": None,
            "estimated_mccount": None,
            "estimated_cxcount": None,
        },
    }

    block = NewBlockEncoding(copy.deepcopy(matrixsum))
    qc = block.circuit(opt="Matrix-order")
    entries["block_matrix_order"] = {
        "scope": "block",
        "backend": "src/block_encoding_new.py",
        "opt": "Matrix-order",
        "qc": qc,
        "metadata": {
            "ctrl_size": int(getattr(block, "ctrl_size", qc.num_qubits - matrixsum.size)),
            "sys_size": int(block.sys_size),
            "width": int(qc.num_qubits),
            "estimated_tcount": getattr(block, "tcount", None),
            "estimated_mccount": getattr(block, "mccount", None),
            "estimated_cxcount": getattr(block, "cxcount", None),
        },
    }
    return entries


def main() -> None:
    out_dir = RESULT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    matrixsum, kraus_index, model_info = build_tfim4_a0_matrixsum()
    entries = build_entries(matrixsum)

    summary: dict[str, Any] = {
        "experiment": "tfim4_a0_block_opt_entries",
        "benchmark": "TFIM_4_A0",
        "kraus_index": int(kraus_index),
        "model_info": model_info,
        "entries": {},
    }
    for name, row in entries.items():
        qc = row.pop("qc")
        summary["entries"][name] = {
            **row,
            "qiskit_metrics": qiskit_metrics(qc),
            "qasm": write_qasm2(qc, out_dir / f"{name}.qasm"),
            "basis_summary": recount(qc, out_dir, name),
        }

    report_lines = [
        "# TFIM_4 A0 BlockEncoding opt Entries",
        "",
        "| entry | scope | backend | opt | qubits | NAM gates | NAM non-Clifford | IBM gates | IBM non-Clifford |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["entries"].items():
        nam = row["basis_summary"].get("nam", {})
        ibm = row["basis_summary"].get("ibm", {})
        report_lines.append(
            f"| {name} | {row['scope']} | `{row['backend']}` | `{row['opt']}` | "
            f"{row['qiskit_metrics']['num_qubits']} | "
            f"{nam.get('metric_total', '-')} | {nam.get('non_clifford', '-')} | "
            f"{ibm.get('metric_total', '-')} | {ibm.get('non_clifford', '-')} |"
        )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "summary": str(out_dir / "summary.json"),
            "report": str(out_dir / "report.md"),
            "entries": {
                name: {
                    "qubits": row["qiskit_metrics"]["num_qubits"],
                    "nam": row["basis_summary"]["nam"],
                    "ibm": row["basis_summary"]["ibm"],
                }
                for name, row in summary["entries"].items()
            },
        },
        indent=2,
        sort_keys=True,
        default=json_default,
    ))


if __name__ == "__main__":
    main()
