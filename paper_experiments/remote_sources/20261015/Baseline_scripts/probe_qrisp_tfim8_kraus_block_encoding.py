#!/usr/bin/env python3
"""Build TFIM_8 Kraus-0 block encodings with Qrisp built-in APIs."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from probe_single_kraus_select_mcg_tools import select_kraus_operator  # type: ignore  # noqa: E402


RESULT_ROOT = ROOT / "Baseline_results" / "qrisp_tfim8_kraus_block_encoding_v1"


def term_to_label_phase(instance: Any) -> tuple[str, complex]:
    if hasattr(instance, "expr") and hasattr(instance, "phase"):
        return str(instance.expr), complex(instance.phase)
    text = repr(instance)
    raise TypeError(f"Unsupported Pauli instance: {text}")


def qrisp_pauli_unitary(label: str, phase: complex) -> Callable[..., None]:
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

    unitary.__name__ = f"pauli_{label}"
    return unitary


def build_qrisp_operator(terms: list[tuple[str, complex]]) -> Any:
    from qrisp.operators import QubitOperator, X, Y, Z

    op = QubitOperator()
    for label, coeff in terms:
        term_op: Any = None
        for idx, char in enumerate(label):
            if char == "I":
                continue
            factor = {"X": X, "Y": Y, "Z": Z}[char](idx)
            term_op = factor if term_op is None else term_op * factor
        if term_op is None:
            op = op + coeff
        else:
            op = op + coeff * term_op
    return op


def compile_block_encoding(be: Any, out_dir: Path, prefix: str, workspace: int = 0) -> dict[str, Any]:
    from qrisp import QuantumVariable

    result: dict[str, Any] = {"workspace": int(workspace)}
    started = time.perf_counter()
    qv = QuantumVariable(8, name="sys")
    be.apply(qv)
    result["apply_time_s"] = float(time.perf_counter() - started)
    result["session_precompile_str"] = str(qv.qs)

    status, compiled, elapsed, error = timed(lambda: qv.qs.compile(workspace=workspace))
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


def main() -> None:
    out_dir = RESULT_ROOT.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"experiment": "qrisp_tfim8_kraus_block_encoding", "out_dir": str(out_dir)}

    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    selected = select_kraus_operator(kraus_ops)
    summary["model_info"] = model_info
    summary["selected_kraus_index"] = int(selected.index)

    terms: list[tuple[str, complex]] = []
    raw_terms = []
    for instance, coeff in copy.deepcopy(selected.operator).instances:
        label, phase = term_to_label_phase(instance)
        effective_coeff = complex(coeff) * phase
        terms.append((label, effective_coeff))
        raw_terms.append({"pauli": label, "coeff": repr(effective_coeff)})
    summary["terms"] = raw_terms

    try:
        import qrisp
        from qrisp.block_encodings import BlockEncoding

        summary["qrisp_version"] = getattr(qrisp, "__version__", "unknown")
        summary["qrisp_file"] = getattr(qrisp, "__file__", None)
        summary["block_encoding_api"] = {
            "from_lcu": True,
            "from_operator": True,
            "from_foqcs_lcu_operator": hasattr(BlockEncoding, "from_foqcs_lcu_operator"),
        }

        coeffs = [coeff for _label, coeff in terms]
        unitaries = [qrisp_pauli_unitary(label, 1.0 + 0.0j) for label, _coeff in terms]
        # Put Pauli phases into the coefficients, so each unitary is a plain Pauli string.
        status, be_lcu, elapsed, error = timed(lambda: BlockEncoding.from_lcu(coeffs, unitaries, is_hermitian=False))
        summary["from_lcu_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status == "ok":
            summary["from_lcu_block_encoding"] = {
                "alpha": repr(be_lcu.alpha),
                "num_ancs": int(be_lcu.num_ancs),
                "num_ops": int(be_lcu.num_ops),
                "is_hermitian": bool(be_lcu.is_hermitian),
            }
            summary["from_lcu_compile_workspace0"] = compile_block_encoding(be_lcu, out_dir, "qrisp_from_lcu_workspace0", 0)
            summary["from_lcu_compile_workspace8"] = compile_block_encoding(be_lcu, out_dir, "qrisp_from_lcu_workspace8", 8)

        op = build_qrisp_operator(terms)
        summary["qrisp_operator_repr"] = repr(op)
        status, be_op, elapsed, error = timed(lambda: BlockEncoding.from_operator(op))
        summary["from_operator_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status == "ok":
            summary["from_operator_block_encoding"] = {
                "alpha": repr(be_op.alpha),
                "num_ancs": int(be_op.num_ancs),
                "num_ops": int(be_op.num_ops),
                "is_hermitian": bool(be_op.is_hermitian),
                "semantic_note": "Qrisp from_operator encodes the Hermitian part according to installed documentation.",
            }
            summary["from_operator_compile_workspace0"] = compile_block_encoding(be_op, out_dir, "qrisp_from_operator_workspace0", 0)

        status, be_foqcs, elapsed, error = timed(lambda: BlockEncoding.from_foqcs_lcu_operator(op))
        summary["from_foqcs_lcu_operator_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status == "ok":
            summary["from_foqcs_lcu_operator_block_encoding"] = {
                "alpha": repr(be_foqcs.alpha),
                "num_ancs": int(be_foqcs.num_ancs),
                "num_ops": int(be_foqcs.num_ops),
                "is_hermitian": bool(be_foqcs.is_hermitian),
            }
            summary["from_foqcs_lcu_operator_compile_workspace0"] = compile_block_encoding(
                be_foqcs,
                out_dir,
                "qrisp_from_foqcs_lcu_operator_workspace0",
                0,
            )

    except Exception:
        summary["qrisp_probe_error"] = traceback.format_exc()

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(f"summary: {out_path}")


if __name__ == "__main__":
    main()
