#!/usr/bin/env python3
"""Small C^5(X)/C^5(Z) demo comparing Qiskit and qclib Mcg."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import PhaseGate, XGate, ZGate
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


OUT_DIR = ROOT / "Baseline_results" / "qclib_mcg_c5x_c5z_demo_v1"
NAM_BASIS = ("h", "x", "rz", "cx", "reset")
IBM_BASIS = ("u1", "u2", "u3", "cx")


def timed(fn):
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


def qiskit_metrics(qc: QuantumCircuit) -> dict[str, Any]:
    return {
        "num_qubits": int(qc.num_qubits),
        "num_clbits": int(qc.num_clbits),
        "size": int(qc.size()),
        "depth": int(qc.depth()),
        "ops": {str(name): int(count) for name, count in qc.count_ops().items()},
    }


def recount(qc: QuantumCircuit, stem: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for basis_name, basis, gate_set in (
        ("nam", NAM_BASIS, "nam"),
        ("ibm", IBM_BASIS, "ibm"),
    ):
        start = time.perf_counter()
        tqc = transpile(qc, basis_gates=list(basis), optimization_level=0)
        elapsed = time.perf_counter() - start
        path = OUT_DIR / f"{stem}_{basis_name}_opt0.qasm"
        write_qasm(tqc, path)
        rows[basis_name] = {
            "basis": basis_name,
            "basis_gates": list(basis),
            "optimization_level": 0,
            "transpile_time_s": float(elapsed),
            "qasm": str(path),
            "qiskit_metrics": qiskit_metrics(tqc),
            "stats": count_qasm_file(path, gate_set=gate_set),
        }
    return rows


def qiskit_controlled(base: str) -> QuantumCircuit:
    gate = XGate() if base == "x" else ZGate()
    qc = QuantumCircuit(6, name=f"qiskit_c5{base}")
    qc.append(gate.control(5, ctrl_state="11111"), list(range(6)))
    return qc


def qclib_mcg(base: str) -> QuantumCircuit:
    from qclib.gates.mcg import Mcg

    gate = XGate() if base == "x" else ZGate()
    qc = QuantumCircuit(6, name=f"qclib_mcg_u2_c5{base}")
    qc.append(Mcg(gate.to_matrix(), 5, ctrl_state="11111"), list(range(6)))
    return qc


def qclib_mcg_su2_phase(base: str) -> QuantumCircuit:
    """Exact U(2) as controlled global phase plus qclib Mcg on SU(2)."""

    from qclib.gates.mcg import Mcg
    from qclib.gates.util import u2_to_su2

    gate = XGate() if base == "x" else ZGate()
    su2, phase = u2_to_su2(np.asarray(gate.to_matrix(), dtype=complex))
    qc = QuantumCircuit(6, name=f"qclib_mcg_su2_phase_c5{base}")

    # A controlled global phase on a C^5(U) branch can be represented as a
    # C^4 phase on one of the controls, conditioned on the other controls.
    qc.append(PhaseGate(float(phase)).control(4, ctrl_state="1111"), list(range(5)))
    qc.append(Mcg(su2, 5, ctrl_state="11111"), list(range(6)))
    return qc


def build_variant(case: str, variant: str) -> QuantumCircuit:
    base = case[-1].lower()
    if variant == "qiskit_control":
        return qiskit_controlled(base)
    if variant == "qclib_mcg_u2":
        return qclib_mcg(base)
    if variant == "qclib_mcg_su2_phase":
        return qclib_mcg_su2_phase(base)
    raise ValueError(f"unknown variant: {variant}")


def exact_equivalence(reference: QuantumCircuit, candidate: QuantumCircuit) -> dict[str, Any]:
    ref = Operator(reference).data
    cand = Operator(candidate).data
    diff_norm = float(np.linalg.norm(ref - cand))
    inner = np.vdot(ref.reshape(-1), cand.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
    phase_diff_norm = float(np.linalg.norm(ref * phase - cand))
    return {
        "passed": bool(phase_diff_norm < 1e-8),
        "strict_passed": bool(diff_norm < 1e-8),
        "diff_norm": diff_norm,
        "phase_corrected_diff_norm": phase_diff_norm,
        "global_phase": repr(phase),
    }


def run_case(case: str) -> dict[str, Any]:
    reference = qiskit_controlled(case[-1].lower())
    result: dict[str, Any] = {"case": case, "variants": {}}
    for variant in ("qiskit_control", "qclib_mcg_u2", "qclib_mcg_su2_phase"):
        status, qc, elapsed, error = timed(lambda variant=variant: build_variant(case, variant))
        row: dict[str, Any] = {
            "status": status,
            "build_time_s": elapsed,
            "error": error,
        }
        if status == "ok":
            row["source_metrics"] = qiskit_metrics(qc)
            row["source_qasm"] = write_qasm(qc, OUT_DIR / f"{case}_{variant}.qasm")
            row["exact_equivalence_to_qiskit_control"] = exact_equivalence(reference, qc)
            status2, recounts, elapsed2, error2 = timed(lambda qc=qc, case=case, variant=variant: recount(qc, f"{case}_{variant}"))
            row["basis_recounts"] = {
                "status": status2,
                "elapsed_seconds": elapsed2,
                "error": error2,
                "result": recounts,
            }
        result["variants"][variant] = row
    return result


def comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case_result in summary["cases"]:
        case = case_result["case"]
        for variant, data in case_result["variants"].items():
            recounts = data.get("basis_recounts", {}).get("result", {})
            for basis in ("nam", "ibm"):
                recount_row = recounts.get(basis)
                if not recount_row:
                    continue
                stats = recount_row["stats"]
                rows.append(
                    {
                        "case": case,
                        "variant": variant,
                        "equivalent": data.get("exact_equivalence_to_qiskit_control", {}).get("passed"),
                        "basis": basis,
                        "width": stats["num_qubits"],
                        "metric_total": stats["metric_total"],
                        "non_clifford": stats["clifford"]["non_clifford"],
                        "ops": stats["ops"],
                        "qasm": recount_row["qasm"],
                    }
                )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    labels = {
        "qiskit_control": "Qiskit C^5 gate",
        "qclib_mcg_u2": "qclib Mcg U(2)",
        "qclib_mcg_su2_phase": "qclib Mcg SU(2)+phase",
    }
    lines = [
        "# qclib Mcg C^5(X)/C^5(Z) Demo",
        "",
        "| Case | Variant | Basis | Width | Gates | Non-Clifford | Exact equiv. |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison_rows"]:
        lines.append(
            f"| {row['case']} | {labels.get(row['variant'], row['variant'])} | "
            f"{row['basis'].upper()} | {row['width']} | {row['metric_total']} | "
            f"{row['non_clifford']} | {str(row['equivalent']).lower()} |"
        )
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- JSON summary: `{summary['summary_path']}`")
    lines.append(f"- Output directory: `{summary['out_dir']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": "qclib_mcg_c5x_c5z_demo",
        "out_dir": str(OUT_DIR.resolve()),
        "cases": [run_case("C5X"), run_case("C5Z")],
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
