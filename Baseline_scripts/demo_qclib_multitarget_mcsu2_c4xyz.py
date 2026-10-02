#!/usr/bin/env python3
"""Demo qclib MultiTargetMCSU2 on a C^4(XYZ) gate.

``MultiTargetMCSU2`` accepts a list of SU(2) matrices, one per target qubit.
Bare Pauli X/Y/Z have determinant -1, so the qclib-native payload uses
``iX, iY, iZ``.  Their product contributes a branch phase ``-i``; the exact
``C^4(XYZ)`` demo adds one four-control branch phase of ``+i``.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import PhaseGate, XGate, YGate, ZGate
from qiskit.quantum_info import Operator


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "Baseline_scripts"
QCLIB_SITE = Path(
    "/Users/chester_hqf/miniconda3/envs/ChannelIR_test/lib/python3.12/site-packages"
)

for candidate in (BASELINE_DIR, QCLIB_SITE):
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402


OUT_DIR = ROOT / "Baseline_results" / "qclib_multitarget_mcsu2_c4xyz_demo_v1"
IBM_BASIS = ("u1", "u2", "u3", "cx")
NUM_CONTROLS = 4
NUM_TARGETS = 3
CONTROLS = list(range(NUM_CONTROLS))
TARGETS = list(range(NUM_CONTROLS, NUM_CONTROLS + NUM_TARGETS))


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    start = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - start, None
    except Exception:
        return "error", None, time.perf_counter() - start, traceback.format_exc()


def qiskit_metrics(qc: QuantumCircuit) -> dict[str, Any]:
    return {
        "num_qubits": int(qc.num_qubits),
        "num_clbits": int(qc.num_clbits),
        "size": int(qc.size()),
        "depth": int(qc.depth()),
        "ops": {str(name): int(count) for name, count in qc.count_ops().items()},
    }


def write_qasm(qc: QuantumCircuit, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)
    return {"path": str(path), "bytes": int(path.stat().st_size)}


def su2_pauli_payload() -> list[np.ndarray]:
    return [
        1j * np.asarray(XGate().to_matrix(), dtype=complex),
        1j * np.asarray(YGate().to_matrix(), dtype=complex),
        1j * np.asarray(ZGate().to_matrix(), dtype=complex),
    ]


def determinant_report(unitaries: list[np.ndarray]) -> list[dict[str, Any]]:
    labels = ("iX", "iY", "iZ")
    return [
        {
            "label": label,
            "determinant": repr(complex(np.linalg.det(unitary))),
        }
        for label, unitary in zip(labels, unitaries)
    ]


def build_qiskit_reference() -> QuantumCircuit:
    qc = QuantumCircuit(NUM_CONTROLS + NUM_TARGETS, name="qiskit_c4xyz_reference")
    for gate, target in ((XGate(), TARGETS[0]), (YGate(), TARGETS[1]), (ZGate(), TARGETS[2])):
        qc.append(gate.control(NUM_CONTROLS, ctrl_state="1" * NUM_CONTROLS), CONTROLS + [target])
    return qc


def build_qclib_su2_branch() -> QuantumCircuit:
    from qclib.gates.multitargetmcsu2 import MultiTargetMCSU2

    qc = QuantumCircuit(NUM_CONTROLS + NUM_TARGETS, name="qclib_mtmcsu2_iXiYiZ")
    MultiTargetMCSU2.multi_target_mcsu2(qc, su2_pauli_payload(), CONTROLS, TARGETS)
    return qc


def build_qclib_exact_c4xyz() -> QuantumCircuit:
    qc = QuantumCircuit(NUM_CONTROLS + NUM_TARGETS, name="qclib_mtmcsu2_exact_c4xyz")
    # Correct the native C^4(iX, iY, iZ) branch phase: i^3 = -i.
    qc.append(PhaseGate(np.pi / 2).control(NUM_CONTROLS - 1), CONTROLS)
    qc.compose(build_qclib_su2_branch(), inplace=True)
    return qc


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


def transpile_ibm(qc: QuantumCircuit, stem: str) -> dict[str, Any]:
    start = time.perf_counter()
    tqc = transpile(qc, basis_gates=list(IBM_BASIS), optimization_level=0)
    elapsed = time.perf_counter() - start
    qasm_path = OUT_DIR / f"{stem}_ibm_opt0.qasm"
    write_qasm(tqc, qasm_path)
    return {
        "basis": "ibm",
        "basis_gates": list(IBM_BASIS),
        "optimization_level": 0,
        "transpile_time_s": float(elapsed),
        "qasm": str(qasm_path),
        "qiskit_metrics": qiskit_metrics(tqc),
        "stats": count_qasm_file(qasm_path, gate_set="ibm"),
    }


def summarize_variant(
    key: str,
    label: str,
    builder: Callable[[], QuantumCircuit],
    reference: QuantumCircuit,
) -> dict[str, Any]:
    status, qc, build_time, error = timed(builder)
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "status": status,
        "build_time_s": build_time,
        "error": error,
    }
    if status != "ok":
        return row

    source_qasm = OUT_DIR / f"{key}_source.qasm"
    row["source_qasm"] = write_qasm(qc, source_qasm)
    row["source_metrics"] = qiskit_metrics(qc)
    row["equivalence_to_c4xyz_reference"] = exact_equivalence(reference, qc)
    t_status, ibm, t_time, t_error = timed(lambda: transpile_ibm(qc, key))
    row["ibm_recount"] = {
        "status": t_status,
        "elapsed_seconds": t_time,
        "error": t_error,
        "result": ibm,
    }
    if t_status == "ok":
        stats = ibm["stats"]
        row["summary"] = {
            "width": int(stats["num_qubits"]),
            "gate_count": int(stats["metric_total"]),
            "non_clifford": int(stats["clifford"]["non_clifford"]),
            "ops": stats["ops"],
            "qasm": ibm["qasm"],
        }
    return row


def comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in summary["variants"]:
        if "summary" not in variant:
            continue
        item = variant["summary"]
        rows.append(
            {
                "key": variant["key"],
                "label": variant["label"],
                "width": item["width"],
                "gate_count": item["gate_count"],
                "non_clifford": item["non_clifford"],
                "ops": item["ops"],
                "strict_equiv": variant["equivalence_to_c4xyz_reference"]["strict_passed"],
                "global_phase_equiv": variant["equivalence_to_c4xyz_reference"][
                    "global_phase_passed"
                ],
                "qasm": item["qasm"],
            }
        )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# qclib MultiTargetMCSU2 C^4(XYZ) Demo",
        "",
        "## Interface Notes",
        "",
        "- `MultiTargetMCSU2.multi_target_mcsu2(circuit, unitary, controls, target)` accepts either one SU(2) matrix for one target, or a `list` of SU(2) matrices for multiple targets.",
        "- In the list form, `unitary[i]` is applied to `target[i]`; all targets share the same control list.",
        "- Bare Pauli `X/Y/Z` are U(2), not SU(2). This demo passes `iX/iY/iZ` and adds a four-control branch phase for an exact `C^4(XYZ)` variant.",
        "- For multi-target use, the static helper appends the generated `.definition`; direct append of the `Gate` object is not the reliable interface.",
        "",
        "## IBM Opt0 Counts",
        "",
        "| Variant | Width | IBM gates | IBM non-Clifford | Strict equiv. to C^4(XYZ) | Global-phase equiv. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison_rows"]:
        lines.append(
            f"| {row['label']} | {row['width']} | {row['gate_count']} | "
            f"{row['non_clifford']} | {str(row['strict_equiv']).lower()} | "
            f"{str(row['global_phase_equiv']).lower()} |"
        )
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- JSON summary: `{summary['summary_path']}`")
    lines.append(f"- Output directory: `{summary['out_dir']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = build_qiskit_reference()
    summary: dict[str, Any] = {
        "experiment": "qclib_multitarget_mcsu2_c4xyz_demo",
        "out_dir": str(OUT_DIR.resolve()),
        "controls": NUM_CONTROLS,
        "targets": NUM_TARGETS,
        "target_payload": ["X", "Y", "Z"],
        "su2_payload": determinant_report(su2_pauli_payload()),
        "variants": [
            summarize_variant(
                "qiskit_reference",
                "Qiskit serial C^4(X), C^4(Y), C^4(Z)",
                build_qiskit_reference,
                reference,
            ),
            summarize_variant(
                "qclib_mtmcsu2_su2_branch",
                "qclib MultiTargetMCSU2 C^4(iX,iY,iZ)",
                build_qclib_su2_branch,
                reference,
            ),
            summarize_variant(
                "qclib_mtmcsu2_exact",
                "qclib MultiTargetMCSU2 exact C^4(XYZ)",
                build_qclib_exact_c4xyz,
                reference,
            ),
        ],
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
