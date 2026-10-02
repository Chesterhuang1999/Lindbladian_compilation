#!/usr/bin/env python3
"""Compare pytket CnX-pairwise raw SELECT against TFIM_8 A0 raw SELECT.

This driver uses the first Kraus operator (A0) of the periodic TFIM n=8
channel.  The pytket input is the raw address-table SELECT, not the in-house
Matrix-order SELECT.  Final counts are in the IBM gate set ``{u1, u2, u3, cx}``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from baseline_common import count_qasm_text  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from run_pytket_cnx_pairwise_channelir import (  # type: ignore  # noqa: E402
    Mode,
    append_controlled_phase_as_cnx,
    append_controlled_pauli_as_cnx,
    clean_pauli_label,
    is_zero_phase_units,
    jsonable,
    run_pytket_passes,
    write_qasm as write_pytket_qasm,
)


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_raw_pytket_cnx_pairwise_select_ibm_v1"


def select_kraus_zero(kraus_ops: list[Any]) -> Any:
    if not kraus_ops:
        raise ValueError("TFIM_8 builder returned no Kraus operators")
    if len(kraus_ops[0].instances) == 0:
        raise ValueError("TFIM_8 Kraus index 0 is empty")
    return kraus_ops[0]


def coefficient_variant(ms: Any, mode: str) -> Any:
    """Return a Matrixsum with modified Pauli phases for coefficient ablations."""
    if mode == "original":
        return copy.deepcopy(ms)

    instances = []
    for atom, coeff in ms.instances:
        label = getattr(atom, "expr", None)
        if label is None:
            raise TypeError(f"Only PauliAtom Matrixsum terms are supported, got {atom!r}")
        magnitude = float(abs(coeff))
        if mode == "positive-real":
            phase = 1.0
        elif mode == "signed-real":
            original_phase = complex(getattr(atom, "phase", 1.0))
            phase = 1.0 if np.real(original_phase) >= 0 else -1.0
        else:
            raise ValueError(f"Unsupported coefficient mode: {mode}")
        instances.append((PauliAtom(str(label), phase=phase), magnitude))
    return Matrixsum(instances)


def export_matrixsum_json(ms: Any, path: Path) -> None:
    terms = []
    for index, (atom, coeff) in enumerate(ms.instances):
        label = getattr(atom, "expr", None)
        if label is None:
            raise TypeError(f"Only PauliAtom Matrixsum terms are supported, got {atom!r}")
        phase = complex(getattr(atom, "phase", 1.0))
        terms.append(
            {
                "term_index": int(index),
                "label": str(label),
                "coeff": float(coeff),
                "phase_real": float(np.real(phase)),
                "phase_imag": float(np.imag(phase)),
                "phase_pi_over_2": float(2 * np.angle(phase) / np.pi),
                "effective_coeff_real": float(np.real(phase * coeff)),
                "effective_coeff_imag": float(np.imag(phase * coeff)),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"index": 0, "terms": terms}], indent=2, sort_keys=True), encoding="utf-8")


def raw_mode_rows_from_block_encoding(be: NewBlockEncoding) -> list[dict[str, Any]]:
    rows = []
    for index, mat_item in enumerate(be.mat_list):
        if not isinstance(mat_item, tuple) and not isinstance(mat_item, list):
            raise TypeError("Raw pytket CnX-pairwise SELECT currently supports PauliAtom terms only.")
        pauli_label, phase = mat_item
        rows.append(
            {
                "term_index": int(index),
                "pauli": clean_pauli_label(pauli_label),
                "ctrl_value": format(index, f"0{be.ctrl_size}b"),
                "phase_pi_over_2": float(2 * np.angle(phase) / np.pi),
            }
        )
    return rows


def summarize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "width": int(stats["num_qubits"]),
        "gate_count": int(stats["metric_total"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "unknown_clifford": int(stats["clifford"]["unknown"]),
        "ops": stats["ops"],
    }


def recount_raw_qiskit_ibm(qc: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
    logical_qasm = write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    status, result, elapsed, error = timed(
        lambda: recount_qiskit_circuit(
            qc,
            basis_name="ibm",
            out_path=out_dir / f"{prefix}_ibm_opt0.qasm",
            optimization_level=0,
        )
    )
    summary = None
    if status == "ok":
        summary = summarize_stats(result["stats"])
    return {
        "status": status,
        "logical_qasm": logical_qasm,
        "elapsed_seconds": float(elapsed),
        "error": error,
        "result": result,
        "summary": summary,
    }


def build_raw_qiskit_select(ms: Any, out_dir: Path) -> tuple[dict[str, Any], NewBlockEncoding, int, int]:
    be = NewBlockEncoding(copy.deepcopy(ms))
    qc_raw, tcount, mccount, cxcount = be.mulplex_U(be.mat_list, be.ctrl_size, be.sys_size)
    recount = recount_raw_qiskit_ibm(qc_raw, out_dir, "tfim8_a0_raw_select_qiskit")
    metadata = {
        "ctrl_size": int(be.ctrl_size),
        "sys_size": int(be.sys_size),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
        "raw_logical_metrics": qiskit_metrics(qc_raw),
    }
    return {
        "metadata": metadata,
        "methods": {
            "raw_select_qiskit_ibm_opt0": recount,
        },
    }, be, be.ctrl_size, be.sys_size


def build_raw_pytket_circuit(
    raw_modes: list[dict[str, Any]],
    *,
    ctrl_size: int,
    sys_size: int,
    phase_policy: str,
) -> tuple[Any, dict[str, Any]]:
    from pytket import Circuit  # type: ignore  # noqa: PLC0415

    needs_phase_ancilla = (
        phase_policy == "synthesize"
        and any(not is_zero_phase_units(float(row["phase_pi_over_2"])) for row in raw_modes)
        and ctrl_size >= 2
    )
    phase_ancilla_count = 1 if needs_phase_ancilla else 0
    phase_ancilla = ctrl_size + sys_size if needs_phase_ancilla else None
    circuit = Circuit(ctrl_size + sys_size + phase_ancilla_count)
    all_controls = list(range(ctrl_size))
    all_ones_mode_ctrl = "1" * ctrl_size

    stats = {
        "input_select": "raw_address_table",
        "modes": len(raw_modes),
        "ctrl_size": int(ctrl_size),
        "sys_size": int(sys_size),
        "num_qubits": int(ctrl_size + sys_size + phase_ancilla_count),
        "phase_ancilla_count": int(phase_ancilla_count),
        "nonzero_controlled_phase_modes": 0,
        "synthesized_controlled_phase_modes": 0,
        "ignored_controlled_phase_modes": 0,
        "phase_u1_gates_before_pairwise": 0,
        "phase_cnx_gates_before_pairwise": 0,
        "pauli_components": 0,
        "cnx_gates_before_pairwise": 0,
        "cx_gates_before_pairwise": 0,
        "x_gates_before_pairwise": 0,
        "open_control_x_gates": 0,
        "identity_modes": 0,
    }

    for row in raw_modes:
        pauli = clean_pauli_label(row["pauli"])
        ctrl_value = str(row["ctrl_value"])
        phase = float(row["phase_pi_over_2"])
        open_controls = [index for index, bit in enumerate(ctrl_value[::-1]) if bit == "0"]

        for control in open_controls:
            circuit.X(control)
        stats["open_control_x_gates"] += 2 * len(open_controls)

        if not is_zero_phase_units(phase):
            stats["nonzero_controlled_phase_modes"] += 1
            if phase_policy == "reject":
                raise ValueError(
                    "Raw SELECT term has nonzero controlled global phase "
                    f"(term={row.get('term_index')}, pauli={pauli}, ctrl={ctrl_value}, "
                    f"phase={phase} * pi/2). Use phase_policy=synthesize for an exact "
                    "phase-ancilla implementation."
                )
            if phase_policy == "ignore":
                stats["ignored_controlled_phase_modes"] += 1
            elif phase_policy == "synthesize":
                phase_added = append_controlled_phase_as_cnx(
                    circuit,
                    all_controls,
                    phase,
                    phase_ancilla,
                )
                stats["synthesized_controlled_phase_modes"] += 1
                stats["phase_u1_gates_before_pairwise"] += phase_added["u1"]
                stats["phase_cnx_gates_before_pairwise"] += phase_added["cnx"]
            else:
                raise ValueError(f"Unsupported phase policy: {phase_policy}")

        controlled_pauli_mode = Mode(pauli=pauli, ctrl=all_ones_mode_ctrl, phase=0.0)
        added = append_controlled_pauli_as_cnx(circuit, controlled_pauli_mode, ctrl_size, sys_size)
        stats["pauli_components"] += added["pauli_weight"]
        stats["cnx_gates_before_pairwise"] += added["cnx"]
        stats["cx_gates_before_pairwise"] += added["cx"]
        stats["x_gates_before_pairwise"] += added["x"]
        if added["pauli_weight"] == 0:
            stats["identity_modes"] += 1

        for control in reversed(open_controls):
            circuit.X(control)

    return circuit, stats


def run_raw_pairwise_select(
    raw_modes: list[dict[str, Any]],
    *,
    ctrl_size: int,
    sys_size: int,
    phase_policy: str,
    out_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        initial_circuit, component_stats = build_raw_pytket_circuit(
            raw_modes,
            ctrl_size=ctrl_size,
            sys_size=sys_size,
            phase_policy=phase_policy,
        )
        optimized_circuit, pass_results = run_pytket_passes(initial_circuit, gate_set="ibm")
        qasm_path = out_dir / f"{prefix}.qasm"
        qasm_text = write_pytket_qasm(optimized_circuit, qasm_path)
        stats = count_qasm_text(qasm_text, path=str(qasm_path), gate_set="ibm")
        return {
            "status": "ok",
            "equivalent_to_raw_select": phase_policy in {"reject", "synthesize"},
            "phase_policy": phase_policy,
            "gate_set": "ibm",
            "component_stats": component_stats,
            "pytket_passes": pass_results,
            "qasm": str(qasm_path),
            "stats": stats,
            "summary": summarize_stats(stats),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:
        return {
            "status": "error",
            "equivalent_to_raw_select": False,
            "phase_policy": phase_policy,
            "gate_set": "ibm",
            "error": repr(exc),
            "elapsed_seconds": float(time.perf_counter() - started),
        }


def method_row(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("status") != "ok" or entry.get("summary") is None:
        return {
            "method": name,
            "status": entry.get("status", "error"),
            "equivalent": entry.get("equivalent_to_raw_select", True),
            "width": None,
            "ibm_gates": None,
            "non_clifford": None,
            "ops": None,
            "error": entry.get("error"),
        }
    summary = entry["summary"]
    return {
        "method": name,
        "status": "ok",
        "equivalent": entry.get("equivalent_to_raw_select", True),
        "width": summary["width"],
        "ibm_gates": summary["gate_count"],
        "non_clifford": summary["non_clifford"],
        "ops": summary["ops"],
        "error": None,
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["method", "status", "equiv", "width", "IBM gates", "non-Cliff", "ops"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        ops = "" if row["ops"] is None else "`" + json.dumps(row["ops"], sort_keys=True) + "`"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["status"]),
                    str(row["equivalent"]),
                    "" if row["width"] is None else str(row["width"]),
                    "" if row["ibm_gates"] is None else str(row["ibm_gates"]),
                    "" if row["non_clifford"] is None else str(row["non_clifford"]),
                    ops,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--coefficient-mode",
        choices=("original", "positive-real", "signed-real"),
        default="original",
        help=(
            "Modify TFIM_8 A0 coefficients before building raw SELECT. "
            "positive-real keeps magnitudes and sets all Pauli phases to +1; "
            "signed-real keeps magnitudes and maps phases to +/-1 by real-part sign."
        ),
    )
    parser.add_argument(
        "--skip-phase-ignore-ablation",
        action="store_true",
        help="Do not run the non-equivalent phase-ignored pairwise ablation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    original_ms = select_kraus_zero(kraus_ops)
    ms = coefficient_variant(original_ms, args.coefficient_mode)
    matrixsum_json = out_dir / "tfim8_a0_matrixsum_terms.json"
    export_matrixsum_json(ms, matrixsum_json)

    raw_qiskit, be, ctrl_size, sys_size = build_raw_qiskit_select(ms, out_dir)
    raw_modes = raw_mode_rows_from_block_encoding(be)
    raw_modes_json = out_dir / "tfim8_a0_raw_select_modes.json"
    raw_modes_json.write_text(json.dumps(raw_modes, indent=2, sort_keys=True), encoding="utf-8")

    pairwise_reject = run_raw_pairwise_select(
        raw_modes,
        ctrl_size=ctrl_size,
        sys_size=sys_size,
        phase_policy="reject",
        out_dir=out_dir,
        prefix="tfim8_a0_raw_select_pytket_cnx_pairwise_phase_reject_ibm",
    )
    pairwise_synthesized = run_raw_pairwise_select(
        raw_modes,
        ctrl_size=ctrl_size,
        sys_size=sys_size,
        phase_policy="synthesize",
        out_dir=out_dir,
        prefix="tfim8_a0_raw_select_pytket_cnx_pairwise_phase_synthesized_ibm",
    )
    pairwise_methods = {
        "raw_select_pytket_cnx_pairwise_phase_reject_check": pairwise_reject,
        "raw_select_pytket_cnx_pairwise_phase_synthesized_ibm": pairwise_synthesized,
    }
    if not args.skip_phase_ignore_ablation:
        pairwise_methods["raw_select_pytket_cnx_pairwise_phase_ignored_ablation_ibm"] = run_raw_pairwise_select(
            raw_modes,
            ctrl_size=ctrl_size,
            sys_size=sys_size,
            phase_policy="ignore",
            out_dir=out_dir,
            prefix="tfim8_a0_raw_select_pytket_cnx_pairwise_phase_ignored_ibm",
        )

    methods = {
        **raw_qiskit["methods"],
        **pairwise_methods,
    }
    rows = [method_row(name, entry) for name, entry in methods.items()]

    baseline = next(
        (row for row in rows if row["method"] == "raw_select_qiskit_ibm_opt0" and row["ibm_gates"] is not None),
        None,
    )
    if baseline is not None:
        base_gates = int(baseline["ibm_gates"])
        for row in rows:
            if row["ibm_gates"] is not None:
                row["ratio_vs_raw_qiskit_ibm_opt0"] = float(int(row["ibm_gates"]) / base_gates)

    nonzero_phase_modes = [
        row
        for row in raw_modes
        if not is_zero_phase_units(float(row["phase_pi_over_2"]))
    ]
    summary = {
        "experiment": "tfim8_a0_raw_pytket_cnx_pairwise_select_ibm",
        "benchmark": "TFIM_8",
        "kraus_index": 0,
        "model_info": model_info,
        "result_dir": str(out_dir),
        "input_select": "raw_address_table",
        "gate_set": "ibm",
        "coefficient_mode": args.coefficient_mode,
        "matrixsum_json": str(matrixsum_json),
        "raw_modes_json": str(raw_modes_json),
        "kraus_metrics": {
            "system_qubits": int(ms.size),
            "pauli_terms": int(len(ms.instances)),
            "pauli_norm": float(ms.pauli_norm()),
        },
        "raw_select": {
            **raw_qiskit["metadata"],
            "raw_modes_count": int(len(raw_modes)),
            "nonzero_phase_modes": int(len(nonzero_phase_modes)),
            "first_nonzero_phase_mode": nonzero_phase_modes[0] if nonzero_phase_modes else None,
        },
        "methods": methods,
        "comparison_rows": rows,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )

    report = [
        "# TFIM_8 A0 Raw SELECT: pytket CnX Pairwise (IBM)",
        "",
        f"- Matrixsum JSON: `{matrixsum_json}`",
        f"- Raw SELECT modes JSON: `{raw_modes_json}`",
        f"- input SELECT: `raw_address_table`",
        f"- gate set: `ibm` (`u1`, `u2`, `u3`, `cx`)",
        f"- coefficient mode: `{args.coefficient_mode}`",
        f"- ctrl size: `{ctrl_size}`; sys size: `{sys_size}`",
        f"- raw modes: `{len(raw_modes)}`; nonzero phase modes: `{len(nonzero_phase_modes)}`",
        "",
        markdown_table(rows),
        "",
        "Notes:",
        "- `raw_select_qiskit_ibm_opt0` is Qiskit's direct raw SELECT lowered to IBM basis with optimization level 0.",
        "- `raw_select_pytket_cnx_pairwise_phase_synthesized_ibm` uses the same raw address controls, open-control X conjugations, and one reusable clean phase ancilla for controlled global phases.",
        "- `phase_ignored_ablation` drops controlled global phases and is not equivalent.",
        "",
        f"Summary JSON: `{summary_path}`",
        "",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(jsonable({"summary": str(summary_path), "rows": rows}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
