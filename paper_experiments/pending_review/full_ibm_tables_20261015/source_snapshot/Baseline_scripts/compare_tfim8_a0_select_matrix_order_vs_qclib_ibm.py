#!/usr/bin/env python3
"""IBM-basis SELECT-only comparison for TFIM-8 A0.

Rows:
1. raw ``mulplex_U`` SELECT, for context;
2. raw SELECT adapted to qclib ``Mcg`` U(2), using the same circuit adapter
   as ``probe_qclib_tfim8_a0_raw_select_mcg.py``;
3. current block_encoding_new matrix-order SELECT, expanded with the same
   lowering used by ``BlockEncoding.circuit(opt="Matrix-order")``.
"""

from __future__ import annotations

import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

QCLIB_SITE = Path(
    "/Users/chester_hqf/miniconda3/envs/ChannelIR_test/lib/python3.12/site-packages"
)
if str(QCLIB_SITE) not in sys.path:
    sys.path.append(str(QCLIB_SITE))

from block_encoding_new import (  # type: ignore  # noqa: E402
    BlockEncoding,
    expand_large_controlled_gates,
)
from probe_qclib_tfim8_a0_raw_select_mcg import (  # type: ignore  # noqa: E402
    controlled_pauli_to_qclib_mcg,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    qiskit_metrics,
    recount_qiskit_circuit,
    write_qasm2,
)
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore  # noqa: E402


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_select_matrix_order_vs_qclib_ibm_v1"


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    start = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - start, None
    except Exception:
        return "error", None, time.perf_counter() - start, traceback.format_exc()


def write_qasm(qc: QuantumCircuit, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)
    return {"path": str(path), "bytes": int(path.stat().st_size)}


def recount_ibm(qc: QuantumCircuit, out_dir: Path, stem: str) -> dict[str, Any]:
    return recount_qiskit_circuit(
        qc,
        basis_name="ibm",
        out_path=out_dir / f"{stem}_ibm_opt0.qasm",
        optimization_level=0,
    )


def build_tfim8_a0() -> tuple[Any, dict[str, Any]]:
    kraus_ops, model_info = build_periodic_tfim_channel(8, 0.1)
    ms = kraus_ops[0]
    if len(ms.instances) == 0:
        raise ValueError("TFIM_8 A0 is empty")
    metadata = {
        "benchmark": "TFIM_8_A0",
        "scope": "SELECT only",
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "model_info": model_info,
    }
    return ms, metadata


def build_raw_select(ms: Any) -> tuple[QuantumCircuit, dict[str, Any]]:
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


def build_matrix_order_select(ms: Any) -> tuple[QuantumCircuit, QuantumCircuit, dict[str, Any]]:
    be = BlockEncoding(copy.deepcopy(ms))
    logical, tcount, mccount, cxcount, ctrl_size = be.mulplex_U_opt_order()
    expanded = expand_large_controlled_gates(logical, if_optimize=True)
    metadata = {
        "builder": "BlockEncoding.mulplex_U_opt_order() + expand_large_controlled_gates(if_optimize=True)",
        "ctrl_size": int(ctrl_size),
        "sys_size": int(be.sys_size),
        "logical_width": int(logical.num_qubits),
        "expanded_width": int(expanded.num_qubits),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
        "selected_k": int(getattr(be, "selected_k", ctrl_size)),
        "selected_crit": str(getattr(be, "selected_crit", "")),
        "candidate_costs": getattr(be, "candidate_costs", []),
        "width_candidate_costs": getattr(be, "width_candidate_costs", None),
        "placement_search_debug": getattr(be, "placement_search_debug", {}),
        "placement_width_analysis": getattr(be, "placement_width_analysis", None),
        "g_modes_count": int(len(getattr(be, "mobius_phase_result", {}).get("g_modes_with_phase", []))),
    }
    return logical, expanded, metadata


def spot_equivalence(
    reference: QuantumCircuit,
    candidate: QuantumCircuit,
    *,
    samples: int = 8,
    seed: int = 20260705,
) -> dict[str, Any]:
    if reference.num_qubits != candidate.num_qubits:
        return {
            "method": "statevector_random_spotcheck",
            "passed": None,
            "note": f"skipped: width mismatch {reference.num_qubits} vs {candidate.num_qubits}",
        }
    rng = __import__("numpy").random.default_rng(seed)
    dim = 1 << reference.num_qubits
    max_phase_diff = 0.0
    for _ in range(samples):
        state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        state = state / __import__("numpy").linalg.norm(state)
        ref_out = Statevector(state).evolve(reference).data
        cand_out = Statevector(state).evolve(candidate).data
        inner = __import__("numpy").vdot(ref_out, cand_out)
        phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
        max_phase_diff = max(
            max_phase_diff,
            float(__import__("numpy").linalg.norm(ref_out * phase - cand_out)),
        )
    return {
        "method": "statevector_random_spotcheck",
        "samples": int(samples),
        "seed": int(seed),
        "passed": bool(max_phase_diff < 1e-8),
        "max_phase_corrected_diff_norm": max_phase_diff,
    }


def summarize_row(
    *,
    key: str,
    label: str,
    qc: QuantumCircuit,
    metadata: dict[str, Any],
    out_dir: Path,
    compare_to: QuantumCircuit | None = None,
) -> dict[str, Any]:
    stem = key
    write_qasm2(qc, out_dir / f"{stem}_logical.qasm")
    status, recount, elapsed, error = timed(lambda: recount_ibm(qc, out_dir, stem))
    verification = None
    if compare_to is not None:
        verification = spot_equivalence(compare_to, qc)
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "metadata": metadata,
        "logical_metrics": qiskit_metrics(qc),
        "ibm_recount": {
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
            "result": recount,
        },
        "verification_vs_raw_select": verification,
    }
    if status == "ok":
        stats = recount["stats"]
        row["summary"] = {
            "basis": "IBM",
            "width": int(stats["num_qubits"]),
            "gate_count": int(stats["metric_total"]),
            "non_clifford": int(stats["clifford"]["non_clifford"]),
            "ops": stats["ops"],
            "qasm": recount["qasm"],
        }
    return row


def comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    raw_count = None
    for row in summary["rows"]:
        if row["key"] == "raw_select" and "summary" in row:
            raw_count = int(row["summary"]["gate_count"])
            break
    for row in summary["rows"]:
        item = row.get("summary")
        if not item:
            continue
        gate_count = int(item["gate_count"])
        rows.append(
            {
                "key": row["key"],
                "label": row["label"],
                "width": item["width"],
                "gate_count": gate_count,
                "non_clifford": item["non_clifford"],
                "ratio_vs_raw": None if raw_count in (None, 0) else float(gate_count / raw_count),
                "ops": item["ops"],
                "qasm": item["qasm"],
                "verification": row.get("verification_vs_raw_select"),
            }
        )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TFIM-8 A0 SELECT: Matrix-Order vs qclib Mcg (IBM)",
        "",
        f"- Scope: `{summary['metadata']['scope']}`",
        f"- Terms/system qubits: `{summary['metadata']['pauli_terms']}` / `{summary['metadata']['system_qubits']}`",
        "- qclib row uses raw `mulplex_U` SELECT adapted into serial qclib `Mcg` U(2) single-target gates.",
        "- matrix-order row uses current `block_encoding_new` SELECT lowering only; no PREPARE.",
        "",
        "| Variant | Width | IBM gates | IBM non-Clifford | Ratio vs raw | Verification vs raw |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["comparison_rows"]:
        ratio = row["ratio_vs_raw"]
        ratio_cell = "-" if ratio is None else f"{ratio:.3f}"
        verification = row.get("verification")
        if verification is None:
            vcell = "-"
        else:
            passed = verification.get("passed")
            vcell = "pass" if passed is True else ("skipped" if passed is None else "fail")
        lines.append(
            f"| {row['label']} | {row['width']} | {row['gate_count']} | "
            f"{row['non_clifford']} | {ratio_cell} | {vcell} |"
        )
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- JSON summary: `{summary['summary_path']}`")
    lines.append(f"- Output directory: `{summary['out_dir']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    out_dir = RESULT_ROOT.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ms, metadata = build_tfim8_a0()
    raw_select, raw_meta = build_raw_select(ms)
    status, adapted_value, adapter_elapsed, adapter_error = timed(
        lambda: controlled_pauli_to_qclib_mcg(raw_select, use_su2_unitary=False)
    )
    if status != "ok":
        raise RuntimeError(f"qclib adapter failed:\n{adapter_error}")
    qclib_select, qclib_adapter_meta = adapted_value

    matrix_logical, matrix_expanded, matrix_meta = build_matrix_order_select(ms)

    rows = [
        summarize_row(
            key="raw_select",
            label="raw mulplex_U SELECT",
            qc=raw_select,
            metadata=raw_meta,
            out_dir=out_dir,
        ),
        summarize_row(
            key="qclib_mcg_u2",
            label="qclib Mcg U(2) adapter",
            qc=qclib_select,
            metadata={
                **qclib_adapter_meta,
                "adapter_elapsed_seconds": adapter_elapsed,
            },
            out_dir=out_dir,
            compare_to=raw_select,
        ),
        summarize_row(
            key="matrix_order_logical",
            label="matrix-order SELECT logical",
            qc=matrix_logical,
            metadata={**matrix_meta, "note": "before expand_large_controlled_gates"},
            out_dir=out_dir,
        ),
        summarize_row(
            key="matrix_order_expanded",
            label="matrix-order SELECT expanded",
            qc=matrix_expanded,
            metadata=matrix_meta,
            out_dir=out_dir,
        ),
    ]

    summary: dict[str, Any] = {
        "experiment": "tfim8_a0_select_matrix_order_vs_qclib_ibm",
        "out_dir": str(out_dir),
        "metadata": metadata,
        "rows": rows,
    }
    summary["comparison_rows"] = comparison_rows(summary)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "summary.md"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=repr) + "\n", encoding="utf-8")
    report_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}, indent=2))
    print(markdown(summary))


if __name__ == "__main__":
    main()
