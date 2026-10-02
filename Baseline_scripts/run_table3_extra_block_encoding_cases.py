#!/usr/bin/env python3
"""Extra single-Kraus block-encoding cases for Table 3.

Cases:
* TFIM_3 A0 and TFIM_8 A0.
* Full n-qubit Pauli sums for n=6,8 with uniform positive coefficients.

The Qrisp path uses ``BlockEncoding.from_lcu`` with nonnegative coefficients
and phase-bearing Pauli unitaries, which is the native LCU form supported by
the installed Qrisp API.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding_new import BlockEncoding as CurrentBlockEncoding  # type: ignore  # noqa: E402
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
from sum_rewritten_kraus_block_counts_qrisp_vs_ours import (  # type: ignore  # noqa: E402
    kraus_effective_terms,
    natural_width,
)


RESULT_ROOT = ROOT / "Baseline_results" / "table3_extra_block_encoding_cases_v1"


def term_to_label_phase(instance: Any) -> tuple[str, complex]:
    if hasattr(instance, "expr") and hasattr(instance, "phase"):
        return str(instance.expr), complex(instance.phase)
    label_method = getattr(instance, "to_label", None)
    if callable(label_method):
        return str(label_method()), 1.0 + 0.0j
    return repr(instance), 1.0 + 0.0j


def qrisp_pauli_unitary_with_phase(label: str, phase: complex) -> Callable[..., None]:
    angle = math.atan2(phase.imag, phase.real)

    def unitary(qv: Any) -> None:
        from qrisp import gphase, x, y, z

        if abs(angle) > 1e-14:
            gphase(angle, qv[0])
        for idx, char in enumerate(label):
            if char == "I":
                continue
            if char == "X":
                x(qv[idx])
            elif char == "Y":
                y(qv[idx])
            elif char == "Z":
                z(qv[idx])
            else:
                raise ValueError(f"Unsupported Pauli character {char!r} in {label!r}")

    unitary.__name__ = f"phase_pauli_{label}"
    return unitary


def recount_bases(qc: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
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


def summarize_basis(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry or entry.get("status") != "ok" or entry.get("result") is None:
        return None
    stats = entry["result"]["stats"]
    return {
        "gate_count": int(stats["metric_total"]),
        "width": int(stats["num_qubits"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "ops": stats["ops"],
        "qasm": entry["result"]["qasm"],
        "elapsed_seconds": float(entry["elapsed_seconds"]),
        "transpile_time_s": float(entry["result"]["transpile_time_s"]),
    }


def count_qrisp_from_lcu_phase_unitaries(ms: Matrixsum, out_dir: Path, prefix: str, workspace: int) -> dict[str, Any]:
    import qrisp
    from qrisp import QuantumVariable
    from qrisp.block_encodings import BlockEncoding

    terms = []
    for instance, coeff in copy.deepcopy(ms).instances:
        label, atom_phase = term_to_label_phase(instance)
        effective = complex(coeff) * atom_phase
        magnitude = abs(effective)
        phase = 1.0 + 0.0j if magnitude == 0 else effective / magnitude
        terms.append((label, magnitude, phase))

    result: dict[str, Any] = {
        "method": "qrisp_from_lcu_phase_unitaries",
        "workspace": int(workspace),
        "term_count": int(len(ms.instances)),
        "index_width": int(natural_width(ms)),
        "qrisp_version": getattr(qrisp, "__version__", None),
        "qrisp_file": getattr(qrisp, "__file__", None),
    }

    coeffs = [magnitude for _label, magnitude, _phase in terms]
    unitaries = [
        qrisp_pauli_unitary_with_phase(label, phase)
        for label, _magnitude, phase in terms
    ]

    started_total = time.perf_counter()
    status, be, elapsed, error = timed(lambda: BlockEncoding.from_lcu(coeffs, unitaries, is_hermitian=False))
    result["from_lcu_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "from_lcu_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started_total)
        return result

    result["block_encoding"] = {
        "alpha": repr(be.alpha),
        "num_ancs": int(be.num_ancs),
        "num_ops": int(be.num_ops),
        "is_hermitian": bool(be.is_hermitian),
    }

    qv = QuantumVariable(ms.size, name="sys")
    status, _unused, elapsed, error = timed(lambda: be.apply(qv))
    result["apply"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "apply_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started_total)
        return result

    status, compiled, elapsed, error = timed(lambda: qv.qs.compile(workspace=workspace))
    result["compile"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "compile_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started_total)
        return result

    result["compiled_qrisp_metrics"] = {
        "num_qubits": int(compiled.num_qubits()),
        "depth": int(compiled.depth()),
        "ops": {str(k): int(v) for k, v in compiled.count_ops().items()},
    }
    status, qiskit_qc, elapsed, error = timed(compiled.to_qiskit)
    result["to_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "to_qiskit_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started_total)
        return result

    result["qiskit_metrics"] = qiskit_metrics(qiskit_qc)
    result["qasm"] = write_qasm2(qiskit_qc, out_dir / f"{prefix}_qiskit.qasm")
    result["basis_recounts"] = recount_bases(qiskit_qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["status"] = "ok"
    result["elapsed_seconds"] = float(time.perf_counter() - started_total)
    return result


def count_ours_block_encoding_new(ms: Matrixsum, out_dir: Path, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "ours_block_encoding_new_current",
        "class": "block_encoding_new.BlockEncoding",
        "term_count": int(len(ms.instances)),
        "natural_width": int(natural_width(ms)),
        "rank_width_heuristic_env": os.environ.get("CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC"),
        "force_original_width_env": os.environ.get("CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH"),
    }
    started_total = time.perf_counter()
    try:
        be = CurrentBlockEncoding(copy.deepcopy(ms))
        qc = be.circuit(opt="Matrix-order")
    except Exception:
        result["status"] = "build_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started_total)
        result["error"] = traceback.format_exc()
        return result

    result["status"] = "ok"
    result["build_seconds"] = float(time.perf_counter() - started_total)
    result["selected_k"] = int(getattr(be, "selected_k", result["natural_width"]))
    result["selected_crit"] = str(getattr(be, "selected_crit", ""))
    result["placement_search_debug"] = getattr(be, "placement_search_debug", {})
    result["placement_width_analysis"] = getattr(be, "placement_width_analysis", {})
    result["width_candidate_costs"] = getattr(be, "width_candidate_costs", [])
    result["qiskit_metrics"] = qiskit_metrics(qc)
    result["qasm"] = write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    result["basis_recounts"] = recount_bases(qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started_total)
    return result


def first_nonempty_kraus(kraus_ops: list[Any]) -> tuple[int, Matrixsum]:
    for index, op in enumerate(kraus_ops):
        if len(op.instances) > 0:
            return index, op
    raise ValueError("no nonempty Kraus operator")


def build_tfim_a0(num_qubits: int) -> tuple[str, Matrixsum, dict[str, Any]]:
    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec(f"TFIM_{num_qubits}", "tfim", num_qubits))
    index, op = first_nonempty_kraus(kraus_ops)
    return f"TFIM_{num_qubits}_A0", op, {
        "family": "periodic_tfim_lindbladian",
        "scope": "single first nonempty Kraus A0",
        "kraus_index": int(index),
        "model_info": model_info,
    }


def full_pauli_matrixsum(num_qubits: int) -> Matrixsum:
    labels = ["".join(chars) for chars in itertools.product("IXYZ", repeat=num_qubits)]
    coeff = 1.0 / float(len(labels))
    return Matrixsum([(PauliAtom(label, phase=1.0), coeff) for label in labels])


def build_full_pauli(num_qubits: int) -> tuple[str, Matrixsum, dict[str, Any]]:
    return f"full_pauli_n{num_qubits}", full_pauli_matrixsum(num_qubits), {
        "family": "full_pauli_group_uniform",
        "scope": "single Pauli-sum over all n-qubit Paulis including identity",
        "num_qubits": int(num_qubits),
    }


def matrixsum_metrics(ms: Matrixsum) -> dict[str, Any]:
    terms = kraus_effective_terms(ms)
    labels = [label for label, _coeff in terms]
    return {
        "system_qubits": int(ms.size),
        "term_count": int(len(terms)),
        "unique_terms": int(len(set(labels))),
        "has_identity": ("I" * ms.size) in labels,
        "l1_norm": float(sum(abs(coeff) for _label, coeff in terms)),
        "natural_width": int(natural_width(ms)),
        "terms_preview": [
            {"pauli": label, "coeff": repr(coeff)}
            for label, coeff in terms[: min(8, len(terms))]
        ],
    }


def run_case(name: str, ms: Matrixsum, metadata: dict[str, Any], out_dir: Path, workspace: int) -> dict[str, Any]:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "case": name,
        "metadata": metadata,
        "metrics": matrixsum_metrics(ms),
        "ours_block_encoding_new_current": None,
        "qrisp_from_lcu": None,
    }
    try:
        summary["ours_block_encoding_new_current"] = count_ours_block_encoding_new(
            copy.deepcopy(ms),
            case_dir,
            f"{name}_ours",
        )
    except Exception:
        summary["ours_matrix_order_forced_natural_w"] = {
            "status": "runner_exception",
            "error": traceback.format_exc(),
        }

    try:
        summary["qrisp_from_lcu"] = count_qrisp_from_lcu_phase_unitaries(
            copy.deepcopy(ms),
            case_dir,
            f"{name}_qrisp",
            workspace=workspace,
        )
    except Exception:
        summary["qrisp_from_lcu"] = {
            "status": "runner_exception",
            "error": traceback.format_exc(),
        }
    return summary


def extract_row(case_summary: dict[str, Any], basis: str, method: str) -> dict[str, Any]:
    result = case_summary.get(method) or {}
    basis_summary = result.get("basis_summary", {}).get(basis)
    if result.get("status") != "ok" or basis_summary is None:
        return {
            "case": case_summary["case"],
            "basis": basis,
            "method": method,
            "status": result.get("status", "missing"),
            "gate_count": None,
            "non_clifford": None,
            "width": None,
            "elapsed_seconds": result.get("elapsed_seconds"),
            "error": result.get("error"),
        }
    return {
        "case": case_summary["case"],
        "basis": basis,
        "method": method,
        "status": "ok",
        "gate_count": int(basis_summary["gate_count"]),
        "non_clifford": int(basis_summary["non_clifford"]),
        "width": int(basis_summary["width"]),
        "elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
        "error": None,
    }


def comparison_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case_summary in cases:
        for basis in ("nam", "ibm"):
            ours = extract_row(case_summary, basis, "ours_block_encoding_new_current")
            qrisp = extract_row(case_summary, basis, "qrisp_from_lcu")
            if ours["gate_count"] and qrisp["gate_count"]:
                qrisp["qrisp_over_ours"] = float(qrisp["gate_count"] / ours["gate_count"])
            else:
                qrisp["qrisp_over_ours"] = None
            ours["qrisp_over_ours"] = None
            rows.extend([ours, qrisp])
    return rows


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
    for row in summary["comparison_rows"]:
        rows.append(
            [
                row["case"],
                row["basis"],
                row["method"],
                row["status"],
                "-" if row["gate_count"] is None else row["gate_count"],
                "-" if row["non_clifford"] is None else row["non_clifford"],
                "-" if row["width"] is None else row["width"],
                "-" if row["elapsed_seconds"] is None else f"{row['elapsed_seconds']:.3f}",
                "" if row.get("qrisp_over_ours") is None else f"{row['qrisp_over_ours']:.6g}",
            ]
        )
    validation_rows = [
        [
            case["case"],
            case["metrics"]["system_qubits"],
            case["metrics"]["term_count"],
            case["metrics"]["unique_terms"],
            case["metrics"]["has_identity"],
            f"{case['metrics']['l1_norm']:.6g}",
            case["metrics"]["natural_width"],
        ]
        for case in summary["cases"]
    ]
    text = "\n\n".join(
        [
            "# Table 3 Extra Block-Encoding Cases",
            markdown_table(
                ["case", "basis", "method", "status", "gates", "non-Clifford", "width", "time (s)", "qrisp/ours"],
                rows,
            ),
            "## Validation",
            markdown_table(
                ["case", "qubits", "terms", "unique", "has identity", "L1 norm", "natural width"],
                validation_rows,
            ),
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["TFIM_3_A0", "TFIM_8_A0", "full_pauli_n6", "full_pauli_n8"],
    )
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--workspace", type=int, default=0)
    args = parser.parse_args()

    builders = {
        "TFIM_3_A0": lambda: build_tfim_a0(3),
        "TFIM_4_A0": lambda: build_tfim_a0(4),
        "TFIM_8_A0": lambda: build_tfim_a0(8),
        "full_pauli_n4": lambda: build_full_pauli(4),
        "full_pauli_n6": lambda: build_full_pauli(6),
        "full_pauli_n8": lambda: build_full_pauli(8),
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": "table3_extra_block_encoding_cases",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "cases": [],
        "comparison_rows": [],
    }

    for case_key in args.cases:
        name, ms, metadata = builders[case_key]()
        case_summary = run_case(name, ms, metadata, out_dir, workspace=args.workspace)
        summary["cases"].append(case_summary)
        summary["comparison_rows"] = comparison_rows(summary["cases"])
        (out_dir / "summary.partial.json").write_text(
            json.dumps(summary, indent=2, default=json_default),
            encoding="utf-8",
        )
        write_report(summary, out_dir / "report.partial.md")

    summary["comparison_rows"] = comparison_rows(summary["cases"])
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary["comparison_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
