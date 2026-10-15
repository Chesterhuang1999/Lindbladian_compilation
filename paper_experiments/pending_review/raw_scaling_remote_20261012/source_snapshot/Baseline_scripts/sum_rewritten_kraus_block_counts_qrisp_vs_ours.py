#!/usr/bin/env python3
"""Sum per-Kraus block-encoding counts after channel rewrite.

Benchmarks:
* HRW_8, HRW_20, HRW_28
* random coloring n=4, q=4 on linear and cycle graphs
* repetition-code DQE d=3

For every benchmark, this script first runs the project's channel
``rewrite_search``.  It then builds a single-Kraus block encoding for every
nonempty rewritten Kraus operator and sums the resulting NAM/IBM opt0 counts.

Both methods receive the same rewritten Kraus list.  The in-house route forces
the matrix-order control width to the natural value ``ceil(log2(term_count))``
so that this experiment does not trade depth for extra LCU control bits.
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
    """Avoid optional qiskit-aer import/version failures in builders."""

    if "qiskit_aer" in sys.modules:
        return

    aer = types.ModuleType("qiskit_aer")
    library = types.ModuleType("qiskit_aer.library")

    class AerSimulator:  # pragma: no cover - not used by this runner
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("AerSimulator is not used in this block-count runner")

    def _unused(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise RuntimeError("qiskit_aer helpers are not used in this block-count runner")

    aer.AerSimulator = AerSimulator
    aer.noise = types.SimpleNamespace()
    library.save_statevector = _unused
    library.set_statevector = _unused
    sys.modules["qiskit_aer"] = aer
    sys.modules["qiskit_aer.library"] = library


install_aer_stub()

from channel_IR import channel as ChannelIR  # type: ignore  # noqa: E402
from experiment_tfim8_a0_full_matrix_order_forced_w_new import (  # type: ignore  # noqa: E402
    ForcedWNewBlockEncoding,
)
from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from run_random_coloring_channel_lcu_ours_nam import (  # type: ignore  # noqa: E402
    build_random_coloring_channel,
)
from build_repetition_code_dqe_channel import (  # type: ignore  # noqa: E402
    DqeSpec,
    build_repetition_dqe_channel,
    pauli_dict_to_matrixsum,
)


RESULT_ROOT = ROOT / "Baseline_results" / "rewritten_kraus_block_sum_qrisp_vs_ours_v1"


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


def kraus_terms(ms: Any) -> list[dict[str, Any]]:
    terms = []
    for instance, coeff in copy.deepcopy(ms).instances:
        label, phase = term_to_label_phase(instance)
        effective = complex(coeff) * phase
        terms.append(
            {
                "pauli": label,
                "matrixsum_coeff": repr(coeff),
                "phase": repr(phase),
                "effective_coeff": repr(effective),
            }
        )
    return terms


def kraus_effective_terms(ms: Any) -> list[tuple[str, complex]]:
    terms = []
    for instance, coeff in copy.deepcopy(ms).instances:
        label, phase = term_to_label_phase(instance)
        terms.append((label, complex(coeff) * phase))
    return terms


def kraus_cache_key(ms: Any) -> str:
    return json.dumps(
        {
            "size": int(ms.size),
            "terms": kraus_terms(ms),
        },
        sort_keys=True,
    )


def natural_width(ms: Any) -> int:
    term_count = max(1, len(ms.instances))
    return int(math.ceil(math.log2(term_count)))


def build_benchmark(name: str) -> tuple[list[Any], dict[str, Any]]:
    if name.startswith("HRW_"):
        n = int(name.split("_", 1)[1])
        ensemble = build_hypercube_section52_channel(n)
        return ensemble.channels[0][1], {
            "family": "hypercube_random_walk",
            "system_qubits": n,
        }

    if name == "random_coloring_linear_n4_q4":
        kraus_ops, model_info = build_random_coloring_channel(
            4,
            "linear",
            q=4,
            tol=1e-12,
            max_total_pauli_terms=None,
            max_terms_per_kraus=None,
            mix_colors=False,
        )
        return kraus_ops, model_info

    if name == "random_coloring_cycle_n4_q4":
        kraus_ops, model_info = build_random_coloring_channel(
            4,
            "cycle",
            q=4,
            tol=1e-12,
            max_total_pauli_terms=None,
            max_terms_per_kraus=None,
            mix_colors=False,
        )
        return kraus_ops, model_info

    if name == "repetition_dqe_d3":
        channel, metadata = build_repetition_dqe_channel(DqeSpec(d=3, epsilon=0.25))
        return [pauli_dict_to_matrixsum(kraus) for kraus in channel], metadata

    raise ValueError(f"Unsupported benchmark {name!r}")


def channel_metrics(kraus_ops: list[Any]) -> dict[str, int]:
    nonzero = [op for op in kraus_ops if len(op.instances) > 0]
    return {
        "kraus_count": int(len(kraus_ops)),
        "nonzero_kraus_count": int(len(nonzero)),
        "pauli_terms_total": int(sum(len(op.instances) for op in nonzero)),
        "min_terms_per_kraus": int(min((len(op.instances) for op in nonzero), default=0)),
        "max_terms_per_kraus": int(max((len(op.instances) for op in nonzero), default=0)),
        "system_qubits": int(max((op.size for op in nonzero), default=0)),
    }


def rewrite_channel(kraus_ops: list[Any], *, max_steps: int, beam_width: int, tol: float) -> tuple[list[Any], dict[str, Any], float]:
    started = time.perf_counter()
    source = ChannelIR(copy.deepcopy(kraus_ops))
    source.zero_elim()
    result = source.rewrite_search(strategy="greedy", beam_width=beam_width, max_steps=max_steps, tol=tol)
    rewritten = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten.apply_rewrite_result(result, tol=tol)
    rewritten.zero_elim()
    return rewritten.kraus_ops, result, time.perf_counter() - started


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


def count_ours_block(ms: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
    width = natural_width(ms)
    result: dict[str, Any] = {
        "method": "ours_matrix_order_forced_natural_w",
        "forced_w": int(width),
        "term_count": int(len(ms.instances)),
    }
    started_total = time.perf_counter()
    try:
        be = ForcedWNewBlockEncoding(copy.deepcopy(ms), forced_w=width)
        qc = be.circuit(opt="Matrix-order")
    except Exception:
        result["status"] = "build_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started_total)
        result["error"] = traceback.format_exc()
        return result

    result["status"] = "ok"
    result["build_seconds"] = float(time.perf_counter() - started_total)
    result["selected_k"] = int(getattr(be, "selected_k", width))
    result["selected_crit"] = str(getattr(be, "selected_crit", ""))
    result["placement_search_debug"] = getattr(be, "placement_search_debug", {})
    result["qiskit_metrics"] = qiskit_metrics(qc)
    result["qasm"] = write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    result["basis_recounts"] = recount_bases(qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started_total)
    return result


def count_qrisp_from_lcu(ms: Any, out_dir: Path, prefix: str, workspace: int) -> dict[str, Any]:
    import qrisp
    from qrisp import QuantumVariable
    from qrisp.block_encodings import BlockEncoding

    terms = kraus_effective_terms(ms)
    coeffs = [coeff for _label, coeff in terms]
    unitaries = [qrisp_pauli_unitary(label) for label, _coeff in terms]
    result: dict[str, Any] = {
        "method": "qrisp_from_lcu",
        "workspace": int(workspace),
        "term_count": int(len(ms.instances)),
        "index_width": int(natural_width(ms)),
        "qrisp_version": getattr(qrisp, "__version__", None),
        "qrisp_file": getattr(qrisp, "__file__", None),
    }

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


def add_basis_totals(totals: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
    basis_summary = result.get("basis_summary", {})
    for basis in ("nam", "ibm"):
        summary = basis_summary.get(basis)
        elapsed = float(result.get("elapsed_seconds", 0.0))
        totals[basis]["elapsed_seconds_per_occurrence_sum"] += elapsed
        if not result.get("cache_hit"):
            totals[basis]["elapsed_seconds_unique"] += elapsed
        if summary is None:
            totals[basis]["missing"] += 1
            continue
        totals[basis]["gate_count"] += int(summary["gate_count"])
        totals[basis]["non_clifford"] += int(summary["non_clifford"])
        totals[basis]["max_width"] = max(totals[basis]["max_width"], int(summary["width"]))
        totals[basis]["completed"] += 1


def empty_totals() -> dict[str, dict[str, Any]]:
    return {
        basis: {
            "gate_count": 0,
            "non_clifford": 0,
            "max_width": 0,
            "completed": 0,
            "missing": 0,
            "elapsed_seconds_unique": 0.0,
            "elapsed_seconds_per_occurrence_sum": 0.0,
        }
        for basis in ("nam", "ibm")
    }


def run_benchmark(
    name: str,
    out_dir: Path,
    *,
    rewrite_max_steps: int,
    rewrite_beam_width: int,
    rewrite_tol: float,
    workspace: int,
    write_qasm_limit: int,
) -> dict[str, Any]:
    bench_dir = out_dir / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    raw_kraus_ops, model_info = build_benchmark(name)
    rewritten_kraus_ops, rewrite_result, rewrite_seconds = rewrite_channel(
        raw_kraus_ops,
        max_steps=rewrite_max_steps,
        beam_width=rewrite_beam_width,
        tol=rewrite_tol,
    )
    rewritten_nonzero = [
        (index, op)
        for index, op in enumerate(rewritten_kraus_ops)
        if len(op.instances) > 0
    ]

    summary: dict[str, Any] = {
        "benchmark": name,
        "model_info": model_info,
        "raw_channel_metrics": channel_metrics(raw_kraus_ops),
        "rewritten_channel_metrics": channel_metrics(rewritten_kraus_ops),
        "rewrite": {
            "elapsed_seconds": float(rewrite_seconds),
            "result": rewrite_result,
        },
        "policy": {
            "qrisp_input": "rewritten Kraus operator terms via BlockEncoding.from_lcu",
            "ours_input": "same rewritten Kraus operator terms",
            "ours_control_width": "forced natural width ceil(log2(term_count))",
            "sum_scope": "sum of per-Kraus block-encoding counts; no outer channel-LCU select/prepare",
        },
        "kraus_results": [],
        "totals": {
            "ours_matrix_order_forced_natural_w": empty_totals(),
            "qrisp_from_lcu": empty_totals(),
        },
    }

    ours_cache: dict[str, dict[str, Any]] = {}
    qrisp_cache: dict[str, dict[str, Any]] = {}
    for ordinal, (kraus_index, ms) in enumerate(rewritten_nonzero):
        key = kraus_cache_key(ms)
        should_write_qasm = ordinal < write_qasm_limit
        qasm_prefix_base = f"{name}_kraus{kraus_index:04d}"
        if key in ours_cache:
            ours_result = copy.deepcopy(ours_cache[key])
            ours_result["cache_hit"] = True
        else:
            prefix = qasm_prefix_base + "_ours" if should_write_qasm else qasm_prefix_base + "_ours"
            ours_result = count_ours_block(ms, bench_dir, prefix)
            ours_result["cache_hit"] = False
            ours_cache[key] = copy.deepcopy(ours_result)

        if key in qrisp_cache:
            qrisp_result = copy.deepcopy(qrisp_cache[key])
            qrisp_result["cache_hit"] = True
        else:
            prefix = qasm_prefix_base + "_qrisp" if should_write_qasm else qasm_prefix_base + "_qrisp"
            qrisp_result = count_qrisp_from_lcu(ms, bench_dir, prefix, workspace=workspace)
            qrisp_result["cache_hit"] = False
            qrisp_cache[key] = copy.deepcopy(qrisp_result)

        add_basis_totals(summary["totals"]["ours_matrix_order_forced_natural_w"], ours_result)
        add_basis_totals(summary["totals"]["qrisp_from_lcu"], qrisp_result)

        summary["kraus_results"].append(
            {
                "ordinal": int(ordinal),
                "kraus_index": int(kraus_index),
                "term_count": int(len(ms.instances)),
                "natural_width": int(natural_width(ms)),
                "pauli_norm": float(ms.pauli_norm()),
                "terms_preview": kraus_terms(ms)[:8],
                "ours_matrix_order_forced_natural_w": ours_result,
                "qrisp_from_lcu": qrisp_result,
            }
        )

        (bench_dir / "summary.partial.json").write_text(
            json.dumps(summary, indent=2, default=json_default),
            encoding="utf-8",
        )

    return summary


def comparison_rows(benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for bench in benchmarks:
        for basis in ("nam", "ibm"):
            ours = bench["totals"]["ours_matrix_order_forced_natural_w"][basis]
            qrisp = bench["totals"]["qrisp_from_lcu"][basis]
            for method, total in (
                ("ours_matrix_order_forced_natural_w", ours),
                ("qrisp_from_lcu", qrisp),
            ):
                ratio = None
                if method == "qrisp_from_lcu" and ours["gate_count"] > 0:
                    ratio = float(qrisp["gate_count"] / ours["gate_count"])
                rows.append(
                    {
                        "benchmark": bench["benchmark"],
                        "basis": basis,
                        "method": method,
                        "gate_count_sum": int(total["gate_count"]),
                        "non_clifford_sum": int(total["non_clifford"]),
                        "max_width": int(total["max_width"]),
                        "completed_kraus": int(total["completed"]),
                        "missing_kraus": int(total["missing"]),
                        "elapsed_seconds_unique": float(total["elapsed_seconds_unique"]),
                        "elapsed_seconds_per_occurrence_sum": float(total["elapsed_seconds_per_occurrence_sum"]),
                        "qrisp_over_ours": ratio,
                    }
                )
    return rows


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Rewritten-Kraus Block-Encoding Sum: Qrisp vs Ours",
        "",
        "All rows use rewrite_search output. Totals sum per-Kraus block-encoding counts only; outer channel-LCU select/prepare is excluded.",
        "",
        "| benchmark | basis | method | sum gates | sum non-Clifford | max width | completed/missing | unique time (s) | per-K sum time (s) | qrisp/ours |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["comparison_rows"]:
        ratio = row["qrisp_over_ours"]
        lines.append(
            f"| {row['benchmark']} | {row['basis']} | {row['method']} | "
            f"{row['gate_count_sum']} | {row['non_clifford_sum']} | {row['max_width']} | "
            f"{row['completed_kraus']}/{row['missing_kraus']} | "
            f"{row['elapsed_seconds_unique']:.6f} | "
            f"{row['elapsed_seconds_per_occurrence_sum']:.6f} | "
            f"{'' if ratio is None else f'{ratio:.6g}'} |"
        )

    lines.extend(["", "## Rewrite Metrics", "", "| benchmark | raw terms | rewritten terms | raw max/K | rewritten max/K | rewrite time (s) |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for bench in summary["benchmarks"]:
        raw = bench["raw_channel_metrics"]
        rew = bench["rewritten_channel_metrics"]
        lines.append(
            f"| {bench['benchmark']} | {raw['pauli_terms_total']} | {rew['pauli_terms_total']} | "
            f"{raw['max_terms_per_kraus']} | {rew['max_terms_per_kraus']} | "
            f"{bench['rewrite']['elapsed_seconds']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "HRW_8",
            "HRW_20",
            "HRW_28",
            "random_coloring_linear_n4_q4",
            "random_coloring_cycle_n4_q4",
            "repetition_dqe_d3",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--rewrite-max-steps", type=int, default=50)
    parser.add_argument("--rewrite-beam-width", type=int, default=3)
    parser.add_argument("--rewrite-tol", type=float, default=1e-12)
    parser.add_argument("--workspace", type=int, default=0)
    parser.add_argument("--write-qasm-limit", type=int, default=999999)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": "sum_rewritten_kraus_block_counts_qrisp_vs_ours",
        "result_dir": str(out_dir),
        "benchmarks": [],
        "arguments": vars(args),
    }

    for benchmark in args.benchmarks:
        bench_summary = run_benchmark(
            benchmark,
            out_dir,
            rewrite_max_steps=args.rewrite_max_steps,
            rewrite_beam_width=args.rewrite_beam_width,
            rewrite_tol=args.rewrite_tol,
            workspace=args.workspace,
            write_qasm_limit=args.write_qasm_limit,
        )
        summary["benchmarks"].append(bench_summary)
        summary["comparison_rows"] = comparison_rows(summary["benchmarks"])
        (out_dir / "summary.partial.json").write_text(
            json.dumps(summary, indent=2, default=json_default),
            encoding="utf-8",
        )
        write_report(summary, out_dir / "report.partial.md")

    summary["comparison_rows"] = comparison_rows(summary["benchmarks"])
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary["comparison_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
