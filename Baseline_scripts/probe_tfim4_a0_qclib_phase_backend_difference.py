#!/usr/bin/env python3
"""Compare qclib-vs-Qiskit phase-correction backends for TFIM-4 A0 SELECT."""

from __future__ import annotations

import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate
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

from block_encoding_new import BlockEncoding  # type: ignore  # noqa: E402
from channel_IR import PauliAtom  # type: ignore  # noqa: E402
from compare_tfim4_a0_select_qclib_multitarget_ibm import (  # type: ignore  # noqa: E402
    append_controlled_branch_phase,
    branch_phase_for_pauli_string,
    build_qclib_multitarget_select,
    build_raw_select,
    build_tfim4_a0,
    pauli_targets,
    su2_pauli_matrix,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    qiskit_metrics,
    recount_qiskit_circuit,
    write_qasm2,
)


OUT_DIR = ROOT / "Baseline_results" / "tfim4_a0_qclib_phase_backend_difference_v1"


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    started = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - started, None
    except Exception:
        return "error", None, time.perf_counter() - started, traceback.format_exc()


def append_controlled_branch_phase_qiskit(
    qc: QuantumCircuit,
    controls: list[int],
    ctrl_state: str,
    phase: complex,
    stats: dict[str, int],
) -> None:
    angle = float(np.angle(phase))
    if abs(angle) < 1e-12:
        return
    if not controls:
        qc.global_phase += angle
        stats["global_phase_only"] += 1
        return

    phase_target = controls[-1]
    target_state = ctrl_state[0]
    remaining_controls = controls[:-1]
    remaining_state = ctrl_state[1:]

    if target_state == "0":
        qc.x(phase_target)
        stats["open_control_x_conjugations"] += 2

    if remaining_controls:
        qc.append(
            PhaseGate(angle).control(
                num_ctrl_qubits=len(remaining_controls),
                ctrl_state=remaining_state,
            ),
            remaining_controls + [phase_target],
        )
        stats["controlled_branch_phases_qiskit"] += 1
    else:
        qc.p(angle, phase_target)
        stats["single_control_branch_phases"] += 1

    if target_state == "0":
        qc.x(phase_target)


def build_qclib_select_with_qiskit_phase(ms: Any) -> tuple[QuantumCircuit, dict[str, Any]]:
    from qclib.gates.multitargetmcsu2 import MultiTargetMCSU2

    be = BlockEncoding(copy.deepcopy(ms))
    ctrl_size = int(be.ctrl_size)
    sys_size = int(be.sys_size)
    controls = list(range(ctrl_size))
    qc = QuantumCircuit(ctrl_size + sys_size, name="tfim4_a0_qclib_mtmcsu2_qiskit_phase")
    stats = {
        "mcgates": 0,
        "identity_terms": 0,
        "controlled_branch_phases_qiskit": 0,
        "single_control_branch_phases": 0,
        "global_phase_only": 0,
        "open_control_x_conjugations": 0,
        "su2_unitaries": 0,
    }

    for address, (atom, _coeff) in enumerate(ms.instances):
        if not isinstance(atom, PauliAtom):
            raise TypeError("This probe expects PauliAtom terms")
        ctrl_state = format(address, f"0{ctrl_size}b")
        targets = pauli_targets(atom.expr, ctrl_size)
        branch_phase = branch_phase_for_pauli_string(complex(atom.phase), len(targets))
        append_controlled_branch_phase_qiskit(qc, controls, ctrl_state, branch_phase, stats)

        unitaries = [su2_pauli_matrix(str(target["pauli"])) for target in targets]
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

    return qc, {
        "phase_backend": "Qiskit PhaseGate(theta).control(n-1)",
        "ctrl_size": ctrl_size,
        "sys_size": sys_size,
        "stats": stats,
    }


def exact_equivalence(reference: QuantumCircuit, candidate: QuantumCircuit) -> dict[str, Any]:
    ref = Operator(reference).data
    cand = Operator(candidate).data
    diff_norm = float(np.linalg.norm(ref - cand))
    inner = np.vdot(ref.reshape(-1), cand.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
    phase_diff_norm = float(np.linalg.norm(ref * phase - cand))
    return {
        "strict_passed": bool(diff_norm < 1e-8),
        "global_phase_passed": bool(phase_diff_norm < 1e-8),
        "diff_norm": diff_norm,
        "phase_corrected_diff_norm": phase_diff_norm,
        "global_phase": repr(phase),
    }


def recount(qc: QuantumCircuit, key: str, opt: int) -> dict[str, Any]:
    return recount_qiskit_circuit(
        qc,
        basis_name="ibm",
        out_path=OUT_DIR / f"{key}_ibm_opt{opt}.qasm",
        optimization_level=opt,
    )


def summarize_variant(
    key: str,
    label: str,
    qc: QuantumCircuit,
    metadata: dict[str, Any],
    raw_select: QuantumCircuit,
) -> dict[str, Any]:
    write_qasm2(qc, OUT_DIR / f"{key}_source.qasm")
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "metadata": metadata,
        "source_metrics": qiskit_metrics(qc),
        "equivalence_to_raw": exact_equivalence(raw_select, qc),
        "ibm": {},
    }
    for opt in (0, 3):
        status, result, elapsed, error = timed(lambda opt=opt: recount(qc, key, opt))
        opt_key = f"opt{opt}"
        row["ibm"][opt_key] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
            "result": result,
        }
        if status == "ok":
            stats = result["stats"]
            row["ibm"][opt_key]["summary"] = {
                "width": int(stats["num_qubits"]),
                "gate_count": int(stats["metric_total"]),
                "non_clifford": int(stats["clifford"]["non_clifford"]),
                "ops": stats["ops"],
                "qasm": result["qasm"],
                "transpile_time_s": float(result["transpile_time_s"]),
            }
    return row


def comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in summary["variants"]:
        for opt_key, opt_data in variant["ibm"].items():
            item = opt_data.get("summary")
            if not item:
                continue
            rows.append(
                {
                    "variant": variant["label"],
                    "qiskit_opt": opt_key,
                    "width": item["width"],
                    "gate_count": item["gate_count"],
                    "non_clifford": item["non_clifford"],
                    "ops": item["ops"],
                    "transpile_time_s": item["transpile_time_s"],
                }
            )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TFIM_4 A0 qclib MultiTargetMCSU2 Phase Backend Probe",
        "",
        "| Phase backend | Qiskit opt | Width | IBM gates | Non-Clifford |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["comparison_rows"]:
        lines.append(
            f"| {row['variant']} | {row['qiskit_opt']} | {row['width']} | "
            f"{row['gate_count']} | {row['non_clifford']} |"
        )
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- JSON summary: `{summary['summary_path']}`")
    lines.append(f"- Output directory: `{summary['out_dir']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ms, metadata = build_tfim4_a0()
    raw_select, raw_meta = build_raw_select(ms)
    qclib_phase_select, _mcgate_rows, qclib_phase_meta = build_qclib_multitarget_select(ms)
    qiskit_phase_select, qiskit_phase_meta = build_qclib_select_with_qiskit_phase(ms)

    variants = [
        summarize_variant(
            "qclib_mcg_phase",
            "qclib Mcg(Phase)",
            qclib_phase_select,
            qclib_phase_meta,
            raw_select,
        ),
        summarize_variant(
            "qiskit_controlled_phase",
            "Qiskit controlled Phase",
            qiskit_phase_select,
            qiskit_phase_meta,
            raw_select,
        ),
    ]
    summary: dict[str, Any] = {
        "experiment": "tfim4_a0_qclib_multitarget_phase_backend_difference",
        "metadata": metadata,
        "raw_metadata": raw_meta,
        "out_dir": str(OUT_DIR.resolve()),
        "variants": variants,
    }
    summary["comparison_rows"] = comparison_rows(summary)
    summary_path = OUT_DIR / "summary.json"
    report_path = OUT_DIR / "summary.md"
    summary["summary_path"] = str(summary_path.resolve())
    summary["report_path"] = str(report_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=repr) + "\n", encoding="utf-8")
    report_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": summary["summary_path"], "report": summary["report_path"]}, indent=2))
    print(markdown(summary))


if __name__ == "__main__":
    main()
