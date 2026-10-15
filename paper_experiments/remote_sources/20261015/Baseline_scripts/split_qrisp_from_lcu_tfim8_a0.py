#!/usr/bin/env python3
"""Split Qrisp.from_lcu TFIM_8 A0 counts into PREPARE and SELECT pieces."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from probe_qrisp_tfim8_kraus_block_encoding import (  # type: ignore  # noqa: E402
    qrisp_pauli_unitary,
    term_to_label_phase,
)
from probe_single_kraus_select_mcg_tools import select_kraus_operator  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)


RESULT_ROOT = ROOT / "Baseline_results" / "qrisp_tfim8_a0_from_lcu_split_v1"


def build_tfim8_a0_terms() -> tuple[list[tuple[str, complex]], dict[str, Any]]:
    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    selected = select_kraus_operator(kraus_ops)
    terms: list[tuple[str, complex]] = []
    for instance, coeff in copy.deepcopy(selected.operator).instances:
        label, phase = term_to_label_phase(instance)
        terms.append((label, complex(coeff) * phase))
    return terms, {
        "model_info": model_info,
        "selected_kraus_index": int(selected.index),
        "system_qubits": int(selected.operator.size),
        "pauli_terms": int(len(selected.operator.instances)),
    }


def qrisp_count_session(qs: Any, out_dir: Path, prefix: str, workspace: int) -> dict[str, Any]:
    result: dict[str, Any] = {"workspace": int(workspace)}
    status, compiled, elapsed, error = timed(lambda: qs.compile(workspace=workspace))
    result["compile"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        return result

    result["compiled_qrisp_metrics"] = {
        "num_qubits": int(compiled.num_qubits()),
        "depth": int(compiled.depth()),
        "ops": dict(compiled.count_ops()),
    }

    status, qiskit_qc, elapsed, error = timed(compiled.to_qiskit)
    result["to_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        return result

    result["qiskit_metrics"] = qiskit_metrics(qiskit_qc)
    result["qasm"] = write_qasm2(qiskit_qc, out_dir / f"{prefix}_qiskit.qasm")

    basis_recounts = {}
    for basis in ("nam", "ibm"):
        status, recount, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qiskit_qc,
                basis_name=basis,
                out_path=out_dir / f"{prefix}_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        basis_recounts[basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": recount,
            "error": error,
        }
    result["basis_recounts"] = basis_recounts
    return result


def summarize_count(result: dict[str, Any], basis: str) -> dict[str, Any] | None:
    entry = result.get("basis_recounts", {}).get(basis, {})
    if entry.get("status") != "ok":
        return None
    stats = entry["result"]["stats"]
    return {
        "width": int(stats["num_qubits"]),
        "gate_count": int(stats["metric_total"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "ops": stats["ops"],
    }


def build_component(
    component: str,
    coeffs: list[complex],
    unitaries: list[Callable[..., None]],
) -> tuple[Any, dict[str, Any]]:
    import qrisp
    from qrisp import QuantumFloat, QuantumVariable
    from qrisp.alg_primitives.state_preparation import prepare
    from qrisp.block_encodings import BlockEncoding
    from qrisp.environments import invert
    from qrisp.jasp import q_switch

    m = len(coeffs)
    n = (m - 1).bit_length()
    padded = np.pad(np.asarray(coeffs, dtype=complex), (0, (1 << n) - m))
    alpha = np.sum(np.abs(padded))
    amps_r = np.sqrt(padded / alpha)
    amps_l = np.conjugate(amps_r)

    meta = {
        "qrisp_version": getattr(qrisp, "__version__", "unknown"),
        "qrisp_file": getattr(qrisp, "__file__", None),
        "component": component,
        "num_terms": int(m),
        "index_qubits": int(n),
        "padded_slots": int(1 << n),
        "alpha": repr(alpha),
    }

    if component == "prep_r":
        idx = QuantumFloat(n, name="idx")
        prepare(idx, amps_r)
        return idx.qs, meta

    if component == "prep_l_inverse":
        idx = QuantumFloat(n, name="idx")
        with invert():
            prepare(idx, amps_l)
        return idx.qs, meta

    if component == "prep_pair":
        idx = QuantumFloat(n, name="idx")
        prepare(idx, amps_r)
        with invert():
            prepare(idx, amps_l)
        return idx.qs, meta

    if component == "select_only":
        idx = QuantumFloat(n, name="idx")
        sys_qv = QuantumVariable(8, name="sys")
        q_switch(idx, unitaries, sys_qv)
        return sys_qv.qs, meta

    if component == "full_from_lcu":
        sys_qv = QuantumVariable(8, name="sys")
        be = BlockEncoding.from_lcu(coeffs, unitaries, is_hermitian=False)
        be.apply(sys_qv)
        meta.update(
            {
                "block_encoding_alpha": repr(be.alpha),
                "block_encoding_num_ancs": int(be.num_ancs),
                "block_encoding_num_ops": int(be.num_ops),
                "block_encoding_is_hermitian": bool(be.is_hermitian),
            }
        )
        return sys_qv.qs, meta

    raise ValueError(f"unknown component: {component}")


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
    for workspace in summary["workspaces"]:
        for component in summary["component_order"]:
            result = summary["results"][str(workspace)][component]
            nam = result["summary"]["nam"]
            ibm = result["summary"]["ibm"]
            rows.append(
                [
                    workspace,
                    component,
                    nam["width"] if nam else "-",
                    nam["gate_count"] if nam else "-",
                    nam["non_clifford"] if nam else "-",
                    ibm["width"] if ibm else "-",
                    ibm["gate_count"] if ibm else "-",
                    ibm["non_clifford"] if ibm else "-",
                ]
            )

    lines = [
        "# Qrisp.from_lcu TFIM_8 A0 Split Counts",
        "",
        f"- terms: `{summary['tfim8_a0']['pauli_terms']}`",
        f"- selected Kraus index: `{summary['tfim8_a0']['selected_kraus_index']}`",
        f"- decomposition: `PREP_R -> q_switch SELECT -> PREP_L^dag`",
        "",
        markdown_table(
            [
                "workspace",
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
        "",
        "Note: component counts are compiled independently, so they expose source-region cost but do not include cross-region cancellations or scheduling changes in the full compiled circuit.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = RESULT_ROOT.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    terms, tfim_meta = build_tfim8_a0_terms()
    coeffs = [coeff for _label, coeff in terms]
    unitaries = [qrisp_pauli_unitary(label, 1.0 + 0.0j) for label, _coeff in terms]

    summary: dict[str, Any] = {
        "experiment": "qrisp_from_lcu_tfim8_a0_split_counts",
        "result_dir": str(out_dir),
        "tfim8_a0": tfim_meta,
        "terms": [{"pauli": label, "coeff": repr(coeff)} for label, coeff in terms],
        "component_order": [
            "prep_r",
            "prep_l_inverse",
            "prep_pair",
            "select_only",
            "full_from_lcu",
        ],
        "workspaces": [0, 8],
        "results": {},
    }

    for workspace in summary["workspaces"]:
        summary["results"][str(workspace)] = {}
        for component in summary["component_order"]:
            started = time.perf_counter()
            status, built, build_elapsed, build_error = timed(
                lambda component=component: build_component(component, coeffs, unitaries)
            )
            row: dict[str, Any] = {
                "component": component,
                "workspace": int(workspace),
                "build": {
                    "status": status,
                    "elapsed_seconds": build_elapsed,
                    "error": build_error,
                },
            }
            if status == "ok":
                qs, meta = built
                row["metadata"] = meta
                row["count"] = qrisp_count_session(
                    qs,
                    out_dir,
                    f"qrisp_from_lcu_{component}_workspace{workspace}",
                    workspace,
                )
                row["summary"] = {
                    basis: summarize_count(row["count"], basis)
                    for basis in ("nam", "ibm")
                }
            else:
                row["summary"] = {"nam": None, "ibm": None}
            row["elapsed_seconds_total"] = float(time.perf_counter() - started)
            summary["results"][str(workspace)][component] = row

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
