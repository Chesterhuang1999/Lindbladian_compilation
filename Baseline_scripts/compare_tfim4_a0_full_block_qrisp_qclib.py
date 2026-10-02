#!/usr/bin/env python3
"""TFIM-4 A0 full block-encoding comparison.

Rows:
* raw Qiskit opt0: legacy BlockEncoding(...).circuit(opt="No");
* ours: current block_encoding_new.BlockEncoding(...).circuit(opt="Matrix-order");
* Qrisp: legal from_lcu input with nonnegative magnitudes and phase-bearing
  Pauli unitaries;
* qclib shortcut: qclib McxVchainDirty SELECT cost plus Qrisp PREPARE pair
  cost, because both routes use the same coefficient state preparation.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding import BlockEncoding as LegacyBlockEncoding  # type: ignore  # noqa: E402
from compare_tfim4_a0_select_qclib_mcxvchain_mode_a import (  # type: ignore  # noqa: E402
    build_mode_a_select,
    build_tfim4_a0,
    pauli_expression_from_rows,
    pauli_sum_rows,
    verify_a0_global_phase,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from run_table3_extra_block_encoding_cases import (  # type: ignore  # noqa: E402
    count_ours_block_encoding_new,
    count_qrisp_from_lcu_phase_unitaries,
    matrixsum_metrics,
    qrisp_pauli_unitary_with_phase,
    summarize_basis,
    term_to_label_phase,
)


OUT_DIR = ROOT / "Baseline_results" / "tfim4_a0_full_block_qrisp_qclib_v1"


def recount_bases(qc: Any, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for basis in ("nam", "ibm"):
        status, result, elapsed, error = timed(
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
            "result": result,
            "error": error,
        }
    return rows


def count_qiskit_circuit(qc: Any, prefix: str, method: str, metadata: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = {
        "method": method,
        "status": "ok",
        "metadata": metadata,
        "qiskit_metrics": qiskit_metrics(qc),
        "qasm": write_qasm2(qc, OUT_DIR / f"{prefix}_logical.qasm"),
    }
    result["basis_recounts"] = recount_bases(qc, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def count_raw_qiskit_block(ms: Any) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        block = LegacyBlockEncoding(copy.deepcopy(ms))
        qc = block.circuit(opt="No")
    except Exception:
        return {
            "method": "raw_qiskit_legacy_block_opt_no",
            "status": "build_error",
            "elapsed_seconds": float(time.perf_counter() - started),
            "error": traceback.format_exc(),
        }
    result = count_qiskit_circuit(
        qc,
        "raw_qiskit_block_opt_no",
        "raw_qiskit_legacy_block_opt_no",
        {
            "builder": 'src/block_encoding.BlockEncoding.circuit(opt="No")',
            "ctrl_size": int(block.ctrl_size),
            "sys_size": int(block.sys_size),
        },
    )
    result["elapsed_seconds"] += float(time.perf_counter() - started)
    return result


def effective_terms(ms: Any) -> list[tuple[str, complex, float, complex]]:
    terms = []
    for instance, coeff in copy.deepcopy(ms).instances:
        label, atom_phase = term_to_label_phase(instance)
        effective = complex(coeff) * atom_phase
        magnitude = abs(effective)
        phase = 1.0 + 0.0j if magnitude == 0 else effective / magnitude
        terms.append((label, effective, magnitude, phase))
    return terms


def count_qrisp_component_sessions(ms: Any, workspace: int = 0) -> dict[str, Any]:
    from qrisp import QuantumFloat, QuantumVariable
    from qrisp.alg_primitives.state_preparation import prepare
    from qrisp.block_encodings import BlockEncoding
    from qrisp.environments import invert
    from qrisp.jasp import q_switch

    terms = effective_terms(ms)
    coeffs = [magnitude for _label, _effective, magnitude, _phase in terms]
    unitaries = [
        qrisp_pauli_unitary_with_phase(label, phase)
        for label, _effective, _magnitude, phase in terms
    ]
    m = len(coeffs)
    n = (m - 1).bit_length()
    padded = np.pad(np.asarray(coeffs, dtype=float), (0, (1 << n) - m))
    alpha = float(np.sum(padded))
    amps = np.sqrt(padded / alpha)

    def count_session(qs: Any, prefix: str) -> dict[str, Any]:
        result: dict[str, Any] = {"workspace": int(workspace)}
        started = time.perf_counter()
        status, compiled, elapsed, error = timed(lambda: qs.compile(workspace=workspace))
        result["compile"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "compile_error"
            result["elapsed_seconds"] = float(time.perf_counter() - started)
            return result
        status, qiskit_qc, elapsed, error = timed(compiled.to_qiskit)
        result["to_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "to_qiskit_error"
            result["elapsed_seconds"] = float(time.perf_counter() - started)
            return result
        result["qiskit_metrics"] = qiskit_metrics(qiskit_qc)
        result["qasm"] = write_qasm2(qiskit_qc, OUT_DIR / f"{prefix}_qiskit.qasm")
        result["basis_recounts"] = recount_bases(qiskit_qc, prefix)
        result["basis_summary"] = {
            basis: summarize_basis(entry)
            for basis, entry in result["basis_recounts"].items()
        }
        result["status"] = "ok"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        return result

    components: dict[str, Any] = {
        "num_terms": int(m),
        "index_qubits": int(n),
        "padded_slots": int(1 << n),
        "alpha": alpha,
        "results": {},
    }

    idx = QuantumFloat(n, name="idx")
    prepare(idx, amps)
    components["results"]["prep_r"] = count_session(idx.qs, "qrisp_component_prep_r")

    idx = QuantumFloat(n, name="idx")
    with invert():
        prepare(idx, amps)
    components["results"]["prep_inverse"] = count_session(idx.qs, "qrisp_component_prep_inverse")

    idx = QuantumFloat(n, name="idx")
    prepare(idx, amps)
    with invert():
        prepare(idx, amps)
    components["results"]["prep_pair"] = count_session(idx.qs, "qrisp_component_prep_pair")

    idx = QuantumFloat(n, name="idx")
    sys_qv = QuantumVariable(ms.size, name="sys")
    q_switch(idx, unitaries, sys_qv)
    components["results"]["select_only"] = count_session(sys_qv.qs, "qrisp_component_select_only")

    sys_qv = QuantumVariable(ms.size, name="sys")
    be = BlockEncoding.from_lcu(coeffs, unitaries, is_hermitian=False)
    be.apply(sys_qv)
    components["block_encoding"] = {
        "alpha": repr(be.alpha),
        "num_ancs": int(be.num_ancs),
        "num_ops": int(be.num_ops),
        "is_hermitian": bool(be.is_hermitian),
    }
    components["results"]["full_from_lcu"] = count_session(sys_qv.qs, "qrisp_component_full_from_lcu")
    return components


def qclib_shortcut_summary(qclib_select: dict[str, Any], qrisp_components: dict[str, Any]) -> dict[str, Any]:
    prep_pair = qrisp_components["results"]["prep_pair"]
    out: dict[str, Any] = {
        "method": "qclib_mcxvchain_select_plus_qrisp_prepare_pair_shortcut",
        "status": "ok",
        "assumption": "Qrisp and qclib use the same coefficient PREPARE/PREPARE^dag; add Qrisp prep_pair cost to qclib SELECT cost.",
        "basis_summary": {},
    }
    for basis in ("nam", "ibm"):
        prep = prep_pair.get("basis_summary", {}).get(basis)
        select = qclib_select.get("basis_summary", {}).get(basis)
        if not prep or not select:
            out["basis_summary"][basis] = None
            out["status"] = "partial"
            continue
        out["basis_summary"][basis] = {
            "gate_count": int(prep["gate_count"]) + int(select["gate_count"]),
            "non_clifford": int(prep["non_clifford"]) + int(select["non_clifford"]),
            "width": max(int(prep["width"]), int(select["width"])),
            "components": {
                "qrisp_prep_pair": prep,
                "qclib_select": select,
            },
        }
    return out


def extract_table_row(summary: dict[str, Any], method: str, basis: str) -> dict[str, Any]:
    result = summary["methods"].get(method, {})
    basis_summary = result.get("basis_summary", {}).get(basis)
    if not basis_summary:
        return {
            "method": method,
            "basis": basis,
            "status": result.get("status", "missing"),
            "gate_count": None,
            "non_clifford": None,
            "width": None,
            "ratio_vs_raw": None,
        }
    raw = summary["methods"]["raw_qiskit_opt0"]["basis_summary"][basis]
    return {
        "method": method,
        "basis": basis,
        "status": result.get("status", "ok"),
        "gate_count": int(basis_summary["gate_count"]),
        "non_clifford": int(basis_summary["non_clifford"]),
        "width": int(basis_summary["width"]),
        "ratio_vs_raw": float(basis_summary["gate_count"] / raw["gate_count"]),
    }


def write_report(summary: dict[str, Any]) -> None:
    labels = {
        "raw_qiskit_opt0": "Qiskit raw block opt0",
        "qrisp_from_lcu": "Qrisp from_lcu",
        "qclib_shortcut": "qclib SELECT + Qrisp PREPARE",
        "ours_block_encoding_new": "Ours block_encoding_new",
    }
    lines = [
        "# TFIM_4 A0 Full Block-Encoding: Qrisp/qclib/Ours",
        "",
        f"- A0 phase check passed: `{summary['a0_global_phase_check']['passed_up_to_global_phase']}`",
        f"- A0 phase vs canonical: `{summary['a0_global_phase_check']['phase_actual_over_canonical']}`",
        f"- Qrisp PREPARE pair is reused for qclib shortcut: `{summary['qclib_shortcut_assumption']}`",
        "",
        "| Method | Basis | Status | Width | Gates | Non-Clifford | Ratio vs raw |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for basis in ("nam", "ibm"):
        for method in ("raw_qiskit_opt0", "qrisp_from_lcu", "qclib_shortcut", "ours_block_encoding_new"):
            row = extract_table_row(summary, method, basis)
            gates = "-" if row["gate_count"] is None else str(row["gate_count"])
            nc = "-" if row["non_clifford"] is None else str(row["non_clifford"])
            width = "-" if row["width"] is None else str(row["width"])
            ratio = "-" if row["ratio_vs_raw"] is None else f"{row['ratio_vs_raw']:.3f}"
            lines.append(
                f"| {labels[method]} | {basis.upper()} | `{row['status']}` | "
                f"{width} | {gates} | {nc} | {ratio} |"
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
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ms, metadata = build_tfim4_a0()
    rows = pauli_sum_rows(ms)
    a0_check = verify_a0_global_phase(ms)

    raw = count_raw_qiskit_block(ms)
    ours = count_ours_block_encoding_new(copy.deepcopy(ms), OUT_DIR, "ours_block_encoding_new")
    qrisp = count_qrisp_from_lcu_phase_unitaries(copy.deepcopy(ms), OUT_DIR, "qrisp_from_lcu", workspace=0)

    qclib_select_qc, qclib_select_meta, _mode_rows = build_mode_a_select(copy.deepcopy(ms))
    qclib_select = count_qiskit_circuit(
        qclib_select_qc,
        "qclib_mcxvchain_select",
        "qclib_mcxvchain_mode_a_select_only",
        qclib_select_meta,
    )
    qrisp_components = count_qrisp_component_sessions(copy.deepcopy(ms), workspace=0)
    qclib_shortcut = qclib_shortcut_summary(qclib_select, qrisp_components)

    summary: dict[str, Any] = {
        "experiment": "tfim4_a0_full_block_qrisp_qclib",
        "out_dir": str(OUT_DIR.resolve()),
        "metadata": metadata,
        "metrics": matrixsum_metrics(ms),
        "pauli_sum_expression": pauli_expression_from_rows(rows),
        "pauli_sum_rows": rows,
        "a0_global_phase_check": a0_check,
        "qrisp_components": qrisp_components,
        "qclib_select_metadata": qclib_select_meta,
        "qclib_shortcut_assumption": qclib_shortcut["assumption"],
        "methods": {
            "raw_qiskit_opt0": raw,
            "qrisp_from_lcu": qrisp,
            "qclib_select_only": qclib_select,
            "qclib_shortcut": qclib_shortcut,
            "ours_block_encoding_new": ours,
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    write_report(summary)
    print((OUT_DIR / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
