#!/usr/bin/env python3
"""Split block_encoding_new matrix-order TFIM_8 A0 into PREPARE and SELECT costs."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, QuantumRegister


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding import (  # type: ignore  # noqa: E402
    BlockEncoding as NewBlockEncoding,
    expand_larger_controlled_gates,
)
from experiment_tfim8_a0_full_matrix_order_forced_w_new import (  # type: ignore  # noqa: E402
    ForcedWNewBlockEncoding,
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


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_block_encoding_new_split_v1"


def select_kraus_zero(kraus_ops: list[Any]) -> Any:
    if not kraus_ops:
        raise ValueError("TFIM_8 builder returned no Kraus operators")
    if len(kraus_ops[0].instances) == 0:
        raise ValueError("TFIM_8 Kraus index 0 is empty")
    return kraus_ops[0]


def make_block_encoding(ms: Any, width_mode: str, forced_w: int | None) -> NewBlockEncoding:
    if width_mode == "auto":
        return NewBlockEncoding(copy.deepcopy(ms))
    if width_mode == "forced":
        if forced_w is None:
            raise ValueError("forced width mode requires forced_w")
        return ForcedWNewBlockEncoding(copy.deepcopy(ms), forced_w=forced_w)
    raise ValueError(f"unknown width mode: {width_mode}")


def build_components(ms: Any, width_mode: str, forced_w: int | None) -> tuple[dict[str, QuantumCircuit], Any, dict[str, Any]]:
    be = make_block_encoding(ms, width_mode, forced_w)

    qc_u_raw, tcount, mccount, cxcount, ctrl_size = be.mulplex_U_opt_order()
    qc_u_expanded = expand_larger_controlled_gates(qc_u_raw)
    qc_prep = be.mulplex_B_opt_order(ctrl_size)

    ctrl = QuantumRegister(ctrl_size, "ctrl")
    sys = QuantumRegister(be.sys_size, "sys")
    ancilla_size = qc_u_expanded.num_qubits - ctrl_size - be.sys_size
    if ancilla_size > 0:
        anc = QuantumRegister(ancilla_size, "anc")
        full = QuantumCircuit(ctrl, sys, anc, name="BlockEncoding")
        select_qargs = list(ctrl) + list(sys) + list(anc)
    else:
        full = QuantumCircuit(ctrl, sys, name="BlockEncoding")
        select_qargs = list(ctrl) + list(sys)
    full.compose(qc_prep, qubits=ctrl, inplace=True)
    full.compose(qc_u_expanded, qubits=select_qargs, inplace=True)
    full.compose(qc_prep.inverse(), qubits=ctrl, inplace=True)

    prep_pair = QuantumCircuit(ctrl_size, name="MatrixOrderPreparePair")
    prep_pair.compose(qc_prep, qubits=range(ctrl_size), inplace=True)
    prep_pair.compose(qc_prep.inverse(), qubits=range(ctrl_size), inplace=True)

    be.circuit_width = full.num_qubits
    meta = {
        "width_mode": width_mode,
        "forced_w": forced_w,
        "ctrl_size": int(ctrl_size),
        "sys_size": int(be.sys_size),
        "full_width": int(full.num_qubits),
        "select_raw_width": int(qc_u_raw.num_qubits),
        "select_expanded_width": int(qc_u_expanded.num_qubits),
        "select_ancilla_size": int(ancilla_size),
        "raw_ctrl_size": int(be.ctrl_size),
        "selected_k": int(getattr(be, "selected_k", ctrl_size)),
        "selected_crit": str(getattr(be, "selected_crit", "")),
        "candidate_costs": getattr(be, "candidate_costs", []),
        "width_candidate_costs": getattr(be, "width_candidate_costs", None),
        "placement_search_debug": getattr(be, "placement_search_debug", {}),
        "placement_width_analysis": getattr(be, "placement_width_analysis", None),
        "coeff_slots": int(1 << ctrl_size),
        "nonzero_coeff_slots": int(sum(abs(complex(v)) > 0 for v in getattr(be, "coeff_list_ordered", []))),
        "g_modes_count": int(len(getattr(be, "mobius_phase_result", {}).get("g_modes_with_phase", []))),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
        "select_raw_metrics": qiskit_metrics(qc_u_raw),
        "select_expanded_metrics": qiskit_metrics(qc_u_expanded),
        "prep_metrics": qiskit_metrics(qc_prep),
    }
    return {
        "prep_r": qc_prep,
        "prep_l_inverse": qc_prep.inverse(),
        "prep_pair": prep_pair,
        "select_expanded": qc_u_expanded,
        "full_matrix_order": full,
    }, be, meta


def recount_bases(qc: QuantumCircuit, out_dir: Path, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    for basis in ("nam", "ibm"):
        status, result, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=out_dir / f"{prefix}_{basis}_opt0.qasm",
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


def summarize_basis(entry: dict[str, Any]) -> dict[str, Any] | None:
    if entry.get("status") != "ok":
        return None
    stats = entry["result"]["stats"]
    return {
        "width": int(stats["num_qubits"]),
        "gate_count": int(stats["metric_total"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "ops": stats["ops"],
    }


def run_case(ms: Any, out_dir: Path, width_mode: str, forced_w: int | None) -> dict[str, Any]:
    started = time.perf_counter()
    components, _be, meta = build_components(ms, width_mode, forced_w)
    build_seconds = time.perf_counter() - started

    case_label = "auto" if width_mode == "auto" else f"w{forced_w}"
    results: dict[str, Any] = {}
    for name, qc in components.items():
        basis_recounts = recount_bases(qc, out_dir, f"tfim8_a0_new_{case_label}_{name}")
        results[name] = {
            "component": name,
            "logical_metrics": qiskit_metrics(qc),
            "basis_recounts": basis_recounts,
            "basis_summary": {basis: summarize_basis(entry) for basis, entry in basis_recounts.items()},
        }

    return {
        "case": case_label,
        "width_mode": width_mode,
        "forced_w": forced_w,
        "build_seconds": float(build_seconds),
        "metadata": meta,
        "components": results,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for case in summary["cases"]:
        meta = case["metadata"]
        shortcut = meta.get("placement_search_debug", {}).get("shortcut")
        for component in summary["component_order"]:
            row = case["components"][component]
            nam = row["basis_summary"]["nam"]
            ibm = row["basis_summary"]["ibm"]
            rows.append(
                [
                    case["case"],
                    meta["ctrl_size"],
                    meta["full_width"],
                    shortcut,
                    component,
                    nam["width"],
                    nam["gate_count"],
                    nam["non_clifford"],
                    ibm["width"],
                    ibm["gate_count"],
                    ibm["non_clifford"],
                ]
            )

    lines = [
        "# TFIM_8 A0 block_encoding_new Matrix-Order Split",
        "",
        "Components use the same placement and coefficient order as `block_encoding.BlockEncoding.circuit(opt=\"Matrix-order\")`.",
        "",
        markdown_table(
            [
                "case",
                "ctrl w",
                "full width",
                "shortcut",
                "component",
                "NAM width",
                "NAM gates",
                "NAM non-Clifford",
                "IBM width",
                "IBM gates",
                "IBM non-Clifford",
            ],
            rows,
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = RESULT_ROOT.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    ms = select_kraus_zero(kraus_ops)

    summary: dict[str, Any] = {
        "experiment": "tfim8_a0_block_encoding_new_matrix_order_split",
        "result_dir": str(out_dir),
        "model_info": model_info,
        "benchmark": "TFIM_8",
        "kraus_index": 0,
        "kraus_metrics": {
            "system_qubits": int(ms.size),
            "pauli_terms": int(len(ms.instances)),
            "pauli_norm": float(ms.pauli_norm()),
        },
        "component_order": [
            "prep_r",
            "prep_l_inverse",
            "prep_pair",
            "select_expanded",
            "full_matrix_order",
        ],
        "cases": [],
    }

    for width_mode, forced_w in [("auto", None), ("forced", 5), ("forced", 8)]:
        summary["cases"].append(run_case(ms, out_dir, width_mode, forced_w))

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
