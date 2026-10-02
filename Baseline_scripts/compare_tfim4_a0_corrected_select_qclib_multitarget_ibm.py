#!/usr/bin/env python3
"""Corrected TFIM-4 A0 SELECT-only qclib MultiTargetMCSU2 comparison.

This script avoids the current intermediate ``Matrixsum.add().simplify()``
global-phase issue in ``Lindblad_to_channel`` by directly constructing

    A0 = I - i dt H - dt/2 sum_j L_j^\dagger L_j

for the periodic TFIM model used by ``run_tfim4_channel_lcu_ours_nam``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
for candidate in (SRC_DIR, BASELINE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: E402
import compare_tfim4_a0_select_qclib_multitarget_ibm as base  # type: ignore  # noqa: E402
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore  # noqa: E402


OUT_DIR = ROOT / "Baseline_results" / "tfim4_a0_corrected_select_qclib_multitarget_mcsu2_v1"


def build_corrected_tfim_a0(num_qubits: int = 4, delta_t: float = 0.1) -> tuple[Matrixsum, dict[str, Any]]:
    gamma_sq = 0.1 / 4
    instances = [(PauliAtom("I" * num_qubits, phase=1.0), 1.0)]

    # -i dt H with H = -sum_i Z_i Z_{i+1} - sum_i X_i.
    for i in range(num_qubits):
        z_ind = [i, (i + 1) % num_qubits]
        z_str = "".join("Z" if j in z_ind else "I" for j in range(num_qubits))
        x_str = "".join("X" if j == i else "I" for j in range(num_qubits))
        instances.append((PauliAtom(z_str, phase=1.0j), delta_t))
        instances.append((PauliAtom(x_str, phase=1.0j), delta_t))

    # L_i = gamma (X_i - i Y_i), so L_i^\dagger L_i = 2 gamma^2 (I + Z_i).
    dissipative_coeff = delta_t * gamma_sq
    identity_shift = num_qubits * dissipative_coeff
    instances[0] = (instances[0][0], 1.0 - identity_shift)
    for i in range(num_qubits):
        z_str = "".join("Z" if j == i else "I" for j in range(num_qubits))
        instances.append((PauliAtom(z_str, phase=-1.0), dissipative_coeff))

    ms = Matrixsum(instances)
    return ms, {
        "benchmark": "TFIM_4_A0",
        "scope": "SELECT only",
        "construction": "canonical A0 = I - i dt H - dt/2 sum LdagL; avoids intermediate Matrixsum global-phase elimination",
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "delta_t": float(delta_t),
        "gamma_squared": float(gamma_sq),
        "dissipative_single_z_abs_coeff": float(dissipative_coeff),
        "identity_abs_coeff": float(1.0 - identity_shift),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.OUT_DIR = OUT_DIR

    corrected_ms, metadata = build_corrected_tfim_a0()
    current_ms, current_model = build_periodic_tfim_channel(4, 0.1)
    current_a0 = current_ms[0]

    pauli_rows = base.pauli_sum_rows(corrected_ms)
    expression = base.pauli_sum_expression(pauli_rows)
    raw_select, raw_meta = base.build_raw_select(corrected_ms)
    qclib_select, mcgate_rows, qclib_meta = base.build_qclib_multitarget_select(corrected_ms)
    matrix_logical, matrix_expanded, matrix_meta = base.build_matrix_order_select(corrected_ms)

    intermediate_path = OUT_DIR / "pauli_sum_and_multitarget_unitaries.md"
    base.write_intermediate_markdown(intermediate_path, pauli_rows, expression, mcgate_rows)

    circuits = [
        base.summarize_circuit(
            key="raw_select",
            label="Qiskit raw mulplex_U SELECT",
            qc=raw_select,
            metadata=raw_meta,
            optimization_levels=(0, 3),
        ),
        base.summarize_circuit(
            key="qclib_multitarget_select",
            label="qclib MultiTargetMCSU2 SELECT",
            qc=qclib_select,
            metadata=qclib_meta,
            optimization_levels=(0, 3),
        ),
        base.summarize_circuit(
            key="ours_matrix_order_select_logical",
            label="ours matrix-order SELECT logical",
            qc=matrix_logical,
            metadata={**matrix_meta, "note": "before expand_large_controlled_gates"},
            optimization_levels=(0,),
        ),
        base.summarize_circuit(
            key="ours_matrix_order_select_expanded",
            label="ours matrix-order SELECT expanded",
            qc=matrix_expanded,
            metadata=matrix_meta,
            optimization_levels=(0,),
        ),
    ]

    summary: dict[str, Any] = {
        "experiment": "tfim4_a0_corrected_select_qclib_multitarget_mcsu2_ibm",
        "out_dir": str(OUT_DIR.resolve()),
        "metadata": metadata,
        "current_project_a0_expression": base.pauli_sum_expression(base.pauli_sum_rows(current_a0)),
        "current_project_model_info": current_model,
        "corrected_pauli_sum_expression": expression,
        "pauli_sum_rows": pauli_rows,
        "multitarget_mcsu2_rows": mcgate_rows,
        "raw_metadata": raw_meta,
        "qclib_metadata": qclib_meta,
        "matrix_order_metadata": matrix_meta,
        "intermediate_markdown": str(intermediate_path.resolve()),
        "qclib_equivalence_to_raw": base.exact_equivalence(raw_select, qclib_select),
        "matrix_logical_equivalence_to_raw": base.exact_equivalence(raw_select, matrix_logical),
        "matrix_expanded_equivalence_to_raw": base.exact_equivalence(raw_select, matrix_expanded),
        "circuits": circuits,
    }
    summary["comparison_rows"] = base.comparison_rows(summary)

    summary_path = OUT_DIR / "summary.json"
    report_path = OUT_DIR / "summary.md"
    summary["summary_path"] = str(summary_path.resolve())
    summary["report_path"] = str(report_path.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=repr) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(base.markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": summary["summary_path"], "report": summary["report_path"]}, indent=2))
    print(base.markdown(summary))


if __name__ == "__main__":
    main()
