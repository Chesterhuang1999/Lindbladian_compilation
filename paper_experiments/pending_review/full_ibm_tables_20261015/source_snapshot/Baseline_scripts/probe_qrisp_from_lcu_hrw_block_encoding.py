#!/usr/bin/env python3
"""Compare Qrisp.from_lcu and matrix-order block encodings for HRW Kraus-0.

This runner uses the first nonempty Kraus operator from the hypercube random
walk channel for each requested system size.  HRW Sec. 5.2 Kraus operators are
structurally symmetric, so this is a representative single-Kraus block-encoding
comparison rather than a full channel-LCU experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def install_aer_stub() -> None:
    """Avoid optional qiskit-aer import/version failures in HRW construction."""

    if "qiskit_aer" in sys.modules:
        return

    aer = types.ModuleType("qiskit_aer")
    library = types.ModuleType("qiskit_aer.library")

    class AerSimulator:  # pragma: no cover - not used by this runner
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("AerSimulator is not used in this HRW from_lcu probe")

    def _unused(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise RuntimeError("qiskit_aer helpers are not used in this HRW from_lcu probe")

    aer.AerSimulator = AerSimulator
    aer.noise = types.SimpleNamespace()
    library.save_statevector = _unused
    library.set_statevector = _unused
    sys.modules["qiskit_aer"] = aer
    sys.modules["qiskit_aer.library"] = library


install_aer_stub()

from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)


RESULT_ROOT = ROOT / "Baseline_results" / "qrisp_from_lcu_hrw_block_encoding_v1"


def term_to_label_phase(instance: Any) -> tuple[str, complex]:
    if hasattr(instance, "expr") and hasattr(instance, "phase"):
        return str(instance.expr), complex(instance.phase)
    label_method = getattr(instance, "to_label", None)
    if callable(label_method):
        return str(label_method()), 1.0 + 0.0j
    return repr(instance), 1.0 + 0.0j


def qrisp_pauli_unitary(label: str) -> Callable[..., None]:
    def unitary(qv: Any) -> None:
        from qrisp import x, y, z

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

    unitary.__name__ = f"pauli_{label}"
    return unitary


def build_hrw_first_kraus(n: int) -> Any:
    ensemble = build_hypercube_section52_channel(n)
    kraus_ops = ensemble.channels[0][1]
    for op in kraus_ops:
        if len(op.instances) > 0:
            return op
    raise ValueError(f"HRW_{n} has no nonempty Kraus operators")


def kraus_terms(ms: Any) -> list[dict[str, Any]]:
    rows = []
    for instance, coeff in copy.deepcopy(ms).instances:
        label, phase = term_to_label_phase(instance)
        effective_coeff = complex(coeff) * phase
        rows.append(
            {
                "pauli": label,
                "matrixsum_coeff": repr(coeff),
                "phase": repr(phase),
                "effective_coeff": repr(effective_coeff),
            }
        )
    return rows


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


def summarize_recount(entry: dict[str, Any] | None) -> dict[str, Any] | None:
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


def build_qrisp_from_lcu(ms: Any, out_dir: Path, prefix: str, workspace: int) -> dict[str, Any]:
    import qrisp
    from qrisp import QuantumVariable
    from qrisp.block_encodings import BlockEncoding

    terms = kraus_terms(ms)
    coeffs = [complex(row["effective_coeff"].strip("()")) for row in terms]
    unitaries = [qrisp_pauli_unitary(row["pauli"]) for row in terms]

    result: dict[str, Any] = {
        "status": "ok",
        "qrisp_version": getattr(qrisp, "__version__", None),
        "qrisp_file": getattr(qrisp, "__file__", None),
        "workspace": int(workspace),
    }
    status, be, elapsed, error = timed(lambda: BlockEncoding.from_lcu(coeffs, unitaries, is_hermitian=False))
    result["from_lcu_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "from_lcu_error"
        return result

    result["block_encoding"] = {
        "alpha": repr(be.alpha),
        "num_ancs": int(be.num_ancs),
        "num_ops": int(be.num_ops),
        "is_hermitian": bool(be.is_hermitian),
    }

    started = time.perf_counter()
    qv = QuantumVariable(ms.size, name="sys")
    be.apply(qv)
    result["apply_seconds"] = float(time.perf_counter() - started)
    result["session_precompile"] = str(qv.qs)

    status, compiled, elapsed, error = timed(lambda: qv.qs.compile(workspace=workspace))
    result["compile"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "compile_error"
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
        return result

    result["qiskit_metrics"] = qiskit_metrics(qiskit_qc)
    result["qasm"] = write_qasm2(qiskit_qc, out_dir / f"{prefix}_qiskit.qasm")
    result["basis_recounts"] = recount_bases(qiskit_qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_recount(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    return result


def build_ours_matrix_order(ms: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok"}
    started = time.perf_counter()
    be = NewBlockEncoding(copy.deepcopy(ms))
    try:
        qc = be.circuit(opt="Matrix-order")
    except Exception:
        result["status"] = "build_error"
        result["error"] = traceback.format_exc()
        result["build_seconds"] = float(time.perf_counter() - started)
        return result
    result["build_seconds"] = float(time.perf_counter() - started)
    result["ctrl_size_raw"] = int(be.ctrl_size)
    result["selected_k"] = int(getattr(be, "selected_k", qc.num_qubits - ms.size))
    result["selected_crit"] = str(getattr(be, "selected_crit", ""))
    result["placement_search_debug"] = getattr(be, "placement_search_debug", {})
    result["circuit_width"] = int(qc.num_qubits)
    result["qiskit_metrics"] = qiskit_metrics(qc)
    result["qasm"] = write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    result["basis_recounts"] = recount_bases(qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_recount(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    return result


def comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for bench in summary["benchmarks"]:
        ours = bench["ours_matrix_order"].get("basis_summary", {})
        qrisp = bench["qrisp_from_lcu"].get("basis_summary", {})
        for basis in ("nam", "ibm"):
            ours_row = ours.get(basis)
            qrisp_row = qrisp.get(basis)
            ours_count = None if ours_row is None else int(ours_row["gate_count"])
            qrisp_count = None if qrisp_row is None else int(qrisp_row["gate_count"])
            rows.append(
                {
                    "benchmark": bench["benchmark"],
                    "basis": basis,
                    "method": "ours_matrix_order",
                    "status": bench["ours_matrix_order"].get("status"),
                    "gate_count": ours_count,
                    "non_clifford": None if ours_row is None else int(ours_row["non_clifford"]),
                    "width": None if ours_row is None else int(ours_row["width"]),
                    "ratio_qrisp_over_ours": None,
                    "elapsed_seconds": bench["ours_matrix_order"].get("build_seconds"),
                }
            )
            rows.append(
                {
                    "benchmark": bench["benchmark"],
                    "basis": basis,
                    "method": "qrisp_from_lcu",
                    "status": bench["qrisp_from_lcu"].get("status"),
                    "gate_count": qrisp_count,
                    "non_clifford": None if qrisp_row is None else int(qrisp_row["non_clifford"]),
                    "width": None if qrisp_row is None else int(qrisp_row["width"]),
                    "ratio_qrisp_over_ours": (
                        None
                        if qrisp_count is None or ours_count in (None, 0)
                        else float(qrisp_count / ours_count)
                    ),
                    "elapsed_seconds": bench["qrisp_from_lcu"].get("compile", {}).get("elapsed_seconds"),
                }
            )
    return rows


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# HRW Qrisp.from_lcu Block-Encoding Comparison",
        "",
        "Scope: first nonempty HRW Kraus operator (`K_{0,0}`), representative single-Kraus block encoding.",
        "",
        "| benchmark | basis | method | width | gates | non-Clifford | qrisp/ours | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["comparison_rows"]:
        ratio = row["ratio_qrisp_over_ours"]
        lines.append(
            f"| {row['benchmark']} | {row['basis']} | {row['method']} | "
            f"{row['width']} | {row['gate_count']} | {row['non_clifford']} | "
            f"{'' if ratio is None else f'{ratio:.6g}'} | {row['status']} |"
        )

    lines.extend(["", "## Kraus Terms", ""])
    for bench in summary["benchmarks"]:
        lines.extend(
            [
                f"### {bench['benchmark']}",
                "",
                f"- system qubits: `{bench['system_qubits']}`",
                f"- pauli terms: `{bench['pauli_terms']}`",
                f"- pauli norm: `{bench['pauli_norm']}`",
                f"- Qrisp alpha: `{bench['qrisp_from_lcu'].get('block_encoding', {}).get('alpha')}`",
                f"- ours selected_k: `{bench['ours_matrix_order'].get('selected_k')}`",
                "",
                "| pauli | effective coeff |",
                "| --- | ---: |",
            ]
        )
        for term in bench["terms"]:
            lines.append(f"| `{term['pauli']}` | `{term['effective_coeff']}` |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_case(n: int, out_dir: Path, workspace: int) -> dict[str, Any]:
    bench_dir = out_dir / f"HRW_{n}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    ms = build_hrw_first_kraus(n)
    terms = kraus_terms(ms)
    return {
        "benchmark": f"HRW_{n}",
        "system_qubits": int(ms.size),
        "kraus_index": 0,
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "terms": terms,
        "qrisp_from_lcu": build_qrisp_from_lcu(ms, bench_dir, f"hrw{n}_qrisp_from_lcu_wksp{workspace}", workspace),
        "ours_matrix_order": build_ours_matrix_order(ms, bench_dir, f"hrw{n}_ours_matrix_order"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 20, 28])
    parser.add_argument("--workspace", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "qrisp_from_lcu_hrw_block_encoding",
        "scope": "first nonempty HRW Kraus operator representative block encoding",
        "workspace": int(args.workspace),
        "out_dir": str(out_dir),
        "benchmarks": [],
    }
    for n in args.sizes:
        summary["benchmarks"].append(run_case(int(n), out_dir, int(args.workspace)))
        partial_rows = comparison_rows(summary)
        partial = {**summary, "comparison_rows": partial_rows}
        (out_dir / "summary.partial.json").write_text(
            json.dumps(partial, indent=2, default=json_default),
            encoding="utf-8",
        )

    summary["comparison_rows"] = comparison_rows(summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary["comparison_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
