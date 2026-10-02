#!/usr/bin/env python3
"""TFIM-4 A0 SELECT-only qclib MultiTargetMCSU2 comparison.

Artifacts:
  * raw TFIM_4 A0 Pauli-sum expression;
  * per-address MultiTargetMCSU2 payload metadata, including unitaries lists;
  * IBM-basis counts for raw Qiskit opt0/opt3, qclib opt0/opt3, and current
    matrix-order SELECT.
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
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import PhaseGate, XGate, YGate, ZGate
from qiskit.quantum_info import Operator


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
BASELINE_DIR = ROOT / "Baseline_scripts"
QCLIB_SITE = Path(
    "/Users/chester_hqf/miniconda3/envs/ChannelIR_test/lib/python3.12/site-packages"
)

for candidate in (SRC_DIR, SCRIPTS_DIR, BASELINE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
if str(QCLIB_SITE) not in sys.path:
    sys.path.append(str(QCLIB_SITE))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding, expand_large_controlled_gates  # type: ignore  # noqa: E402
from channel_IR import PauliAtom  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    qiskit_metrics,
    recount_qiskit_circuit,
    write_qasm2,
)
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore  # noqa: E402


OUT_DIR = ROOT / "Baseline_results" / "tfim4_a0_select_qclib_multitarget_mcsu2_v1"
IBM_BASIS = ("u1", "u2", "u3", "cx")
PAULI_GATES = {
    "X": XGate(),
    "Y": YGate(),
    "Z": ZGate(),
}


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    started = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - started, None
    except Exception:
        return "error", None, time.perf_counter() - started, traceback.format_exc()


def complex_record(value: complex, *, tol: float = 1e-12) -> dict[str, Any]:
    z = complex(value)
    real = 0.0 if abs(z.real) < tol else float(z.real)
    imag = 0.0 if abs(z.imag) < tol else float(z.imag)
    return {
        "real": real,
        "imag": imag,
        "repr": repr(complex(real, imag)),
        "angle_pi": float(np.angle(z) / np.pi),
    }


def complex_cell(value: complex, *, digits: int = 10) -> str:
    z = complex(value)
    real = 0.0 if abs(z.real) < 1e-12 else z.real
    imag = 0.0 if abs(z.imag) < 1e-12 else z.imag
    if abs(imag) < 1e-12:
        return f"{real:.{digits}g}"
    if abs(real) < 1e-12:
        return f"{imag:.{digits}g}j"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.{digits}g}{sign}{abs(imag):.{digits}g}j"


def matrix_as_string_rows(matrix: np.ndarray) -> list[list[str]]:
    return [[complex_cell(value, digits=6) for value in row] for row in matrix]


def write_qasm(qc: QuantumCircuit, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)
    return {"path": str(path), "bytes": int(path.stat().st_size)}


def pauli_targets(label: str, ctrl_size: int) -> list[dict[str, Any]]:
    """Return target metadata in Qiskit's Pauli-label convention.

    For a Pauli label of length n, label[0] acts on qubit n-1 and label[-1]
    acts on qubit 0 when appended to qargs ``range(n)``.
    """

    targets: list[dict[str, Any]] = []
    sys_size = len(label)
    for label_index, char in enumerate(label):
        if char == "I":
            continue
        sys_qubit = sys_size - 1 - label_index
        targets.append(
            {
                "pauli": char,
                "pauli_label_index": int(label_index),
                "sys_qubit": int(sys_qubit),
                "absolute_qubit": int(ctrl_size + sys_qubit),
            }
        )
    return targets


def su2_pauli_matrix(pauli: str) -> np.ndarray:
    return 1j * np.asarray(PAULI_GATES[pauli].to_matrix(), dtype=complex)


def branch_phase_for_pauli_string(term_phase: complex, weight: int) -> complex:
    return complex(term_phase) / ((1j) ** weight)


def ctrl_bit_for_qiskit_control_index(ctrl_state: str, control_index: int) -> str:
    return ctrl_state[len(ctrl_state) - 1 - control_index]


def append_controlled_branch_phase(
    qc: QuantumCircuit,
    controls: list[int],
    ctrl_state: str,
    phase: complex,
    stats: dict[str, int],
) -> None:
    from qclib.gates.mcg import Mcg

    angle = float(np.angle(phase))
    if abs(angle) < 1e-12:
        return
    if not controls:
        qc.global_phase += angle
        stats["global_phase_only"] += 1
        return

    phase_target_index = len(controls) - 1
    phase_target = controls[phase_target_index]
    target_state = ctrl_bit_for_qiskit_control_index(ctrl_state, phase_target_index)
    remaining_controls = controls[:-1]
    remaining_state = ctrl_state[1:]

    if target_state == "0":
        qc.x(phase_target)
        stats["open_control_x_conjugations"] += 2

    if remaining_controls:
        qc.append(
            Mcg(PhaseGate(angle).to_matrix(), len(remaining_controls), ctrl_state=remaining_state),
            remaining_controls + [phase_target],
        )
        stats["controlled_branch_phases"] += 1
        stats["qclib_mcg_phase_gates"] += 1
    else:
        qc.p(angle, phase_target)
        stats["single_control_branch_phases"] += 1

    if target_state == "0":
        qc.x(phase_target)


def build_qclib_multitarget_select(ms: Any) -> tuple[QuantumCircuit, list[dict[str, Any]], dict[str, Any]]:
    from qclib.gates.multitargetmcsu2 import MultiTargetMCSU2

    be = BlockEncoding(copy.deepcopy(ms))
    ctrl_size = int(be.ctrl_size)
    sys_size = int(be.sys_size)
    controls = list(range(ctrl_size))
    qc = QuantumCircuit(ctrl_size + sys_size, name="tfim4_a0_qclib_multitarget_select")
    mcgate_rows: list[dict[str, Any]] = []
    stats = {
        "mcgates": 0,
        "identity_terms": 0,
        "controlled_branch_phases": 0,
        "single_control_branch_phases": 0,
        "global_phase_only": 0,
        "open_control_x_conjugations": 0,
        "qclib_mcg_phase_gates": 0,
        "su2_unitaries": 0,
    }

    for address, (atom, coeff) in enumerate(ms.instances):
        if not isinstance(atom, PauliAtom):
            raise TypeError("This adapter expects TFIM A0 to contain only PauliAtom terms")

        ctrl_state = format(address, f"0{ctrl_size}b")
        targets = pauli_targets(atom.expr, ctrl_size)
        weight = len(targets)
        term_phase = complex(atom.phase)
        branch_phase = branch_phase_for_pauli_string(term_phase, weight)
        append_controlled_branch_phase(qc, controls, ctrl_state, branch_phase, stats)

        unitaries: list[np.ndarray] = [su2_pauli_matrix(str(target["pauli"])) for target in targets]
        if unitaries:
            mt_gate = MultiTargetMCSU2(
                unitaries,
                len(controls),
                num_target=len(unitaries),
                ctrl_state=ctrl_state,
            )
            qc.append(mt_gate.definition, controls + [int(t["absolute_qubit"]) for t in targets])
            stats["mcgates"] += 1
            stats["su2_unitaries"] += len(unitaries)
        else:
            stats["identity_terms"] += 1

        mcgate_rows.append(
            {
                "address": int(address),
                "ctrl_state": ctrl_state,
                "pauli": atom.expr,
                "coefficient_abs": float(coeff),
                "term_phase": complex_record(term_phase),
                "term_complex_coefficient": complex_record(complex(coeff) * term_phase),
                "pauli_weight": int(weight),
                "branch_phase": complex_record(branch_phase),
                "branch_phase_formula": "term_phase / (i ** pauli_weight)",
                "targets": targets,
                "unitaries": [
                    {
                        "target_sys_qubit": int(target["sys_qubit"]),
                        "target_absolute_qubit": int(target["absolute_qubit"]),
                        "pauli": str(target["pauli"]),
                        "unitary_label": f"i{target['pauli']}",
                        "matrix": matrix_as_string_rows(unitary),
                    }
                    for target, unitary in zip(targets, unitaries)
                ],
            }
        )

    return qc, mcgate_rows, {
        "adapter": "qclib MultiTargetMCSU2 with SU(2) payload iP per non-identity target",
        "ctrl_size": ctrl_size,
        "sys_size": sys_size,
        "controls": controls,
        "stats": stats,
    }


def build_tfim4_a0() -> tuple[Any, dict[str, Any]]:
    kraus_ops, model_info = build_periodic_tfim_channel(4, 0.1)
    ms = kraus_ops[0]
    return ms, {
        "benchmark": "TFIM_4_A0",
        "scope": "SELECT only",
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "model_info": model_info,
    }


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
    return logical, expanded, {
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
        "mobius_phase_result": getattr(be, "mobius_phase_result", {}),
    }


def pauli_sum_rows(ms: Any) -> list[dict[str, Any]]:
    rows = []
    for address, (atom, coeff) in enumerate(ms.instances):
        if not isinstance(atom, PauliAtom):
            raise TypeError("TFIM_4 A0 contains a non-PauliAtom term")
        complex_coeff = complex(coeff) * complex(atom.phase)
        rows.append(
            {
                "address": int(address),
                "pauli": atom.expr,
                "coefficient_abs": float(coeff),
                "phase": complex_record(complex(atom.phase)),
                "complex_coefficient": complex_record(complex_coeff),
                "select_ctrl_state": "",
            }
        )
    ctrl_size = int(math.ceil(math.log2(len(rows)))) if rows else 0
    for row in rows:
        row["select_ctrl_state"] = format(int(row["address"]), f"0{ctrl_size}b")
    return rows


def pauli_sum_expression(rows: list[dict[str, Any]]) -> str:
    pieces = []
    for row in rows:
        coeff = complex(row["complex_coefficient"]["real"], row["complex_coefficient"]["imag"])
        pieces.append(f"({complex_cell(coeff)}) {row['pauli']}")
    return " + ".join(pieces)


def recount_ibm(
    qc: QuantumCircuit,
    *,
    key: str,
    optimization_level: int,
) -> dict[str, Any]:
    return recount_qiskit_circuit(
        qc,
        basis_name="ibm",
        out_path=OUT_DIR / f"{key}_ibm_opt{optimization_level}.qasm",
        optimization_level=optimization_level,
    )


def summarize_circuit(
    *,
    key: str,
    label: str,
    qc: QuantumCircuit,
    metadata: dict[str, Any],
    optimization_levels: tuple[int, ...],
) -> dict[str, Any]:
    source_path = OUT_DIR / f"{key}_source.qasm"
    source_write = write_qasm(qc, source_path)
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "metadata": metadata,
        "source_qasm": source_write,
        "source_metrics": qiskit_metrics(qc),
        "ibm": {},
    }
    for opt_level in optimization_levels:
        status, recount, elapsed, error = timed(
            lambda opt_level=opt_level: recount_ibm(
                qc,
                key=key,
                optimization_level=opt_level,
            )
        )
        opt_key = f"opt{opt_level}"
        row["ibm"][opt_key] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
            "result": recount,
        }
        if status == "ok":
            stats = recount["stats"]
            row["ibm"][opt_key]["summary"] = {
                "width": int(stats["num_qubits"]),
                "gate_count": int(stats["metric_total"]),
                "non_clifford": int(stats["clifford"]["non_clifford"]),
                "ops": stats["ops"],
                "qasm": recount["qasm"],
                "transpile_time_s": float(recount["transpile_time_s"]),
            }
    return row


def exact_equivalence(reference: QuantumCircuit, candidate: QuantumCircuit) -> dict[str, Any]:
    if reference.num_qubits != candidate.num_qubits:
        return {
            "method": "operator_norm",
            "passed": None,
            "note": f"skipped width mismatch {reference.num_qubits} vs {candidate.num_qubits}",
        }
    ref = Operator(reference).data
    cand = Operator(candidate).data
    diff_norm = float(np.linalg.norm(ref - cand))
    inner = np.vdot(ref.reshape(-1), cand.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
    phase_diff_norm = float(np.linalg.norm(ref * phase - cand))
    return {
        "method": "operator_norm",
        "strict_passed": bool(diff_norm < 1e-8),
        "global_phase_passed": bool(phase_diff_norm < 1e-8),
        "diff_norm": diff_norm,
        "phase_corrected_diff_norm": phase_diff_norm,
        "global_phase": repr(phase),
    }


def comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    raw_opt0 = None
    for circuit in summary["circuits"]:
        if circuit["key"] == "raw_select":
            raw_opt0 = circuit.get("ibm", {}).get("opt0", {}).get("summary", {}).get("gate_count")
            break
    for circuit in summary["circuits"]:
        for opt_key, data in circuit.get("ibm", {}).items():
            item = data.get("summary")
            if not item:
                continue
            gate_count = int(item["gate_count"])
            rows.append(
                {
                    "variant": circuit["label"],
                    "key": circuit["key"],
                    "qiskit_opt": opt_key,
                    "width": int(item["width"]),
                    "gate_count": gate_count,
                    "non_clifford": int(item["non_clifford"]),
                    "ratio_vs_raw_opt0": None
                    if raw_opt0 in (None, 0)
                    else float(gate_count / raw_opt0),
                    "transpile_time_s": float(item["transpile_time_s"]),
                    "ops": item["ops"],
                    "qasm": item["qasm"],
                }
            )
    return rows


def write_intermediate_markdown(
    path: Path,
    pauli_rows: list[dict[str, Any]],
    expression: str,
    mcgate_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# TFIM_4 A0 SELECT qclib MultiTargetMCSU2 Intermediates",
        "",
        "## Pauli Sum",
        "",
        "```text",
        expression,
        "```",
        "",
        "| Address | Ctrl state | Complex coefficient | Abs coeff | Phase | Pauli |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in pauli_rows:
        coeff = complex(row["complex_coefficient"]["real"], row["complex_coefficient"]["imag"])
        phase = complex(row["phase"]["real"], row["phase"]["imag"])
        lines.append(
            f"| {row['address']} | `{row['select_ctrl_state']}` | `{complex_cell(coeff)}` | "
            f"{row['coefficient_abs']:.12g} | `{complex_cell(phase)}` | `{row['pauli']}` |"
        )

    lines.extend(
        [
            "",
            "## MultiTargetMCSU2 Payloads",
            "",
            "Each non-identity Pauli factor is passed as an SU(2) matrix `iP`; "
            "the controlled branch phase is `term_phase / i^weight`.",
            "",
        ]
    )
    for row in mcgate_rows:
        branch = complex(row["branch_phase"]["real"], row["branch_phase"]["imag"])
        term_phase = complex(row["term_phase"]["real"], row["term_phase"]["imag"])
        lines.extend(
            [
                f"### Address {row['address']} / ctrl_state `{row['ctrl_state']}`",
                "",
                f"- Pauli: `{row['pauli']}`",
                f"- term phase: `{complex_cell(term_phase)}`",
                f"- branch phase: `{complex_cell(branch)}`",
                f"- targets: `{row['targets']}`",
                "",
                "| Target sys | Target abs | Pauli | Unitary label | Matrix |",
                "|---:|---:|---:|---|---|",
            ]
        )
        if row["unitaries"]:
            for unitary in row["unitaries"]:
                matrix = "; ".join("[" + ", ".join(mat_row) + "]" for mat_row in unitary["matrix"])
                lines.append(
                    f"| {unitary['target_sys_qubit']} | {unitary['target_absolute_qubit']} | "
                    f"{unitary['pauli']} | `{unitary['unitary_label']}` | `{matrix}` |"
                )
        else:
            lines.append("| - | - | I | identity | no MultiTargetMCSU2 gate |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TFIM_4 A0 SELECT: qclib MultiTargetMCSU2 vs Qiskit/Matrix-Order",
        "",
        f"- Scope: `{summary['metadata']['scope']}`",
        f"- System qubits / Pauli terms: `{summary['metadata']['system_qubits']}` / `{summary['metadata']['pauli_terms']}`",
        f"- Raw SELECT ctrl size: `{summary['raw_metadata']['ctrl_size']}`",
        f"- qclib equivalence vs raw SELECT: `{summary['qclib_equivalence_to_raw'].get('strict_passed')}`",
        "",
        "| Variant | Qiskit opt | Width | IBM gates | Non-Clifford | Ratio vs raw opt0 | Transpile time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison_rows"]:
        ratio = row["ratio_vs_raw_opt0"]
        ratio_cell = "-" if ratio is None else f"{ratio:.3f}"
        lines.append(
            f"| {row['variant']} | {row['qiskit_opt']} | {row['width']} | "
            f"{row['gate_count']} | {row['non_clifford']} | {ratio_cell} | "
            f"{row['transpile_time_s']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Intermediate Artifacts",
            "",
            f"- Pauli sum and MultiTargetMCSU2 payloads: `{summary['intermediate_markdown']}`",
            f"- JSON summary: `{summary['summary_path']}`",
            f"- Output directory: `{summary['out_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ms, metadata = build_tfim4_a0()
    pauli_rows = pauli_sum_rows(ms)
    expression = pauli_sum_expression(pauli_rows)

    raw_select, raw_meta = build_raw_select(ms)
    qclib_select, mcgate_rows, qclib_meta = build_qclib_multitarget_select(ms)
    matrix_logical, matrix_expanded, matrix_meta = build_matrix_order_select(ms)

    intermediate_path = OUT_DIR / "pauli_sum_and_multitarget_unitaries.md"
    write_intermediate_markdown(intermediate_path, pauli_rows, expression, mcgate_rows)

    circuits = [
        summarize_circuit(
            key="raw_select",
            label="Qiskit raw mulplex_U SELECT",
            qc=raw_select,
            metadata=raw_meta,
            optimization_levels=(0, 3),
        ),
        summarize_circuit(
            key="qclib_multitarget_select",
            label="qclib MultiTargetMCSU2 SELECT",
            qc=qclib_select,
            metadata=qclib_meta,
            optimization_levels=(0, 3),
        ),
        summarize_circuit(
            key="ours_matrix_order_select_logical",
            label="ours matrix-order SELECT logical",
            qc=matrix_logical,
            metadata={**matrix_meta, "note": "before expand_large_controlled_gates"},
            optimization_levels=(0,),
        ),
        summarize_circuit(
            key="ours_matrix_order_select_expanded",
            label="ours matrix-order SELECT expanded",
            qc=matrix_expanded,
            metadata=matrix_meta,
            optimization_levels=(0,),
        ),
    ]

    summary: dict[str, Any] = {
        "experiment": "tfim4_a0_select_qclib_multitarget_mcsu2_ibm",
        "out_dir": str(OUT_DIR.resolve()),
        "metadata": metadata,
        "raw_metadata": raw_meta,
        "qclib_metadata": qclib_meta,
        "matrix_order_metadata": matrix_meta,
        "pauli_sum_expression": expression,
        "pauli_sum_rows": pauli_rows,
        "multitarget_mcsu2_rows": mcgate_rows,
        "intermediate_markdown": str(intermediate_path.resolve()),
        "qclib_equivalence_to_raw": exact_equivalence(raw_select, qclib_select),
        "matrix_logical_equivalence_to_raw": exact_equivalence(raw_select, matrix_logical),
        "matrix_expanded_equivalence_to_raw": exact_equivalence(raw_select, matrix_expanded),
        "circuits": circuits,
    }
    summary["comparison_rows"] = comparison_rows(summary)

    summary_path = OUT_DIR / "summary.json"
    report_path = OUT_DIR / "summary.md"
    summary["summary_path"] = str(summary_path.resolve())
    summary["report_path"] = str(report_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=repr) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": summary["summary_path"], "report": summary["report_path"]}, indent=2))
    print(markdown(summary))


if __name__ == "__main__":
    main()
