#!/usr/bin/env python3
"""Rerun Qrisp comparisons with the current ``block_encoding_new.BlockEncoding``.

This runner intentionally avoids ``ForcedWNewBlockEncoding``.  The in-house
path directly instantiates the current ``block_encoding_new.BlockEncoding`` and
builds the complete ``circuit(opt="Matrix-order")`` block encoding:

    PREPARE(coefficients) -> matrix-order SELECT -> PREPARE(coefficients)^dag
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding_new import BlockEncoding as CurrentBlockEncoding  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from run_random_pauli_kraus_block_encoding import (  # type: ignore  # noqa: E402
    CASES as RANDOM_PAULI_CASES,
    random_pauli_matrixsum,
    validate_matrixsum,
)
from run_random_coloring_channel_lcu_ours_nam import (  # type: ignore  # noqa: E402
    build_random_coloring_channel,
)
from run_table3_extra_block_encoding_cases import (  # type: ignore  # noqa: E402
    build_tfim_a0,
    count_qrisp_from_lcu_phase_unitaries,
    matrixsum_metrics,
)
from sum_rewritten_kraus_block_counts_qrisp_vs_ours import (  # type: ignore  # noqa: E402
    add_basis_totals,
    channel_metrics,
    count_qrisp_from_lcu,
    empty_totals,
    kraus_cache_key,
    kraus_terms,
    natural_width,
    rewrite_channel,
)
from build_repetition_code_dqe_channel import (  # type: ignore  # noqa: E402
    DqeSpec,
    build_repetition_dqe_channel,
    pauli_dict_to_matrixsum,
)
from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore  # noqa: E402


RESULT_ROOT = ROOT / "Baseline_results" / "block_encoding_new_qrisp_comparison_v1"
TABLE_MD = ROOT / "Results" / "Table_block_encoding_new_ours_Qrisp.md"
TABLE_TEX = ROOT / "Results" / "Table_block_encoding_new_ours_Qrisp.tex"


def build_benchmark_current(name: str) -> tuple[list[Any], dict[str, Any]]:
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
            color_mixing_preprocess=False,
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
            color_mixing_preprocess=False,
        )
        return kraus_ops, model_info

    if name == "repetition_dqe_d3":
        channel, metadata = build_repetition_dqe_channel(DqeSpec(d=3, epsilon=0.25))
        return [pauli_dict_to_matrixsum(kraus) for kraus in channel], metadata

    raise ValueError(f"Unsupported benchmark {name!r}")


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


def count_ours_block_encoding_new(ms: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
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


def comparison_rows_sum(benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for bench in benchmarks:
        for basis in ("nam", "ibm"):
            ours = bench["totals"]["ours_block_encoding_new_current"][basis]
            qrisp = bench["totals"]["qrisp_from_lcu"][basis]
            for method, total in (
                ("ours_block_encoding_new_current", ours),
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
                        "gate_count_sum": (
                            int(total["gate_count"]) if int(total["completed"]) > 0 else None
                        ),
                        "non_clifford_sum": (
                            int(total["non_clifford"]) if int(total["completed"]) > 0 else None
                        ),
                        "max_width": (
                            int(total["max_width"]) if int(total["completed"]) > 0 else None
                        ),
                        "completed_kraus": int(total["completed"]),
                        "missing_kraus": int(total["missing"]),
                        "elapsed_seconds_unique": float(total["elapsed_seconds_unique"]),
                        "elapsed_seconds_per_occurrence_sum": float(total["elapsed_seconds_per_occurrence_sum"]),
                        "qrisp_over_ours": ratio,
                    }
                )
    return rows


def run_rewritten_benchmark(
    name: str,
    out_dir: Path,
    *,
    rewrite_max_steps: int,
    rewrite_beam_width: int,
    rewrite_tol: float,
    workspace: int,
    write_qasm_limit: int,
) -> dict[str, Any]:
    bench_dir = out_dir / "rewritten_kraus" / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    raw_kraus_ops, model_info = build_benchmark_current(name)
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
            "ours_class": "block_encoding_new.BlockEncoding",
            "sum_scope": "sum of per-Kraus block-encoding counts; no outer channel-LCU select/prepare",
        },
        "kraus_results": [],
        "totals": {
            "ours_block_encoding_new_current": empty_totals(),
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
            ours_result = count_ours_block_encoding_new(ms, bench_dir, prefix)
            ours_result["cache_hit"] = False
            ours_cache[key] = copy.deepcopy(ours_result)

        if key in qrisp_cache:
            qrisp_result = copy.deepcopy(qrisp_cache[key])
            qrisp_result["cache_hit"] = True
        else:
            prefix = qasm_prefix_base + "_qrisp" if should_write_qasm else qasm_prefix_base + "_qrisp"
            qrisp_result = count_qrisp_from_lcu_phase_unitaries(
                ms,
                bench_dir,
                prefix,
                workspace=workspace,
            )
            qrisp_result["cache_hit"] = False
            qrisp_cache[key] = copy.deepcopy(qrisp_result)

        add_basis_totals(summary["totals"]["ours_block_encoding_new_current"], ours_result)
        add_basis_totals(summary["totals"]["qrisp_from_lcu"], qrisp_result)
        summary["kraus_results"].append(
            {
                "ordinal": int(ordinal),
                "kraus_index": int(kraus_index),
                "term_count": int(len(ms.instances)),
                "natural_width": int(natural_width(ms)),
                "pauli_norm": float(ms.pauli_norm()),
                "terms_preview": kraus_terms(ms)[:8],
                "ours_block_encoding_new_current": ours_result,
                "qrisp_from_lcu": qrisp_result,
            }
        )
    return summary


def basis_summary(result: dict[str, Any], basis: str) -> dict[str, Any] | None:
    if result.get("status") != "ok":
        return None
    return result.get("basis_summary", {}).get(basis)


def aggregate_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(statistics.fmean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_random_case(case_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for basis in ("nam", "ibm"):
        values: dict[str, dict[str, list[float]]] = {
            method: {"gates": [], "non_clifford": [], "width": [], "time": []}
            for method in ("ours_block_encoding_new_current", "qrisp_from_lcu")
        }
        for sample in case_summary["samples"]:
            for method in values:
                result = sample[method]
                summary = basis_summary(result, basis)
                if summary is None:
                    continue
                values[method]["gates"].append(float(summary["gate_count"]))
                values[method]["non_clifford"].append(float(summary["non_clifford"]))
                values[method]["width"].append(float(summary["width"]))
                values[method]["time"].append(float(result.get("elapsed_seconds", 0.0)))

        for method, buckets in values.items():
            rows.append(
                {
                    "case": case_summary["case"]["name"],
                    "basis": basis,
                    "method": method,
                    "gate_count": aggregate_values(buckets["gates"]),
                    "non_clifford": aggregate_values(buckets["non_clifford"]),
                    "width": aggregate_values(buckets["width"]),
                    "elapsed_seconds": aggregate_values(buckets["time"]),
                    "completed_samples": int(len(buckets["gates"])),
                    "missing_samples": int(case_summary["num_samples"] - len(buckets["gates"])),
                }
            )
    for row in rows:
        if row["method"] != "qrisp_from_lcu":
            continue
        ours_row = next(
            candidate
            for candidate in rows
            if candidate["case"] == row["case"]
            and candidate["basis"] == row["basis"]
            and candidate["method"] == "ours_block_encoding_new_current"
        )
        ours_mean = ours_row["gate_count"]["mean"]
        qrisp_mean = row["gate_count"]["mean"]
        row["qrisp_over_ours_mean"] = (
            None if not ours_mean or qrisp_mean is None else float(qrisp_mean / ours_mean)
        )
    return rows


def run_random_pauli_case(
    case: dict[str, Any],
    out_dir: Path,
    *,
    samples: int,
    seed_base: int,
    case_offset: int,
    workspace: int,
) -> dict[str, Any]:
    case_summary = {
        "case": case,
        "num_samples": int(samples),
        "samples": [],
    }
    for sample_index in range(samples):
        seed = int(seed_base + 100000 * case_offset + sample_index)
        ms = random_pauli_matrixsum(case["num_qubits"], case["num_terms"], seed)
        validation = validate_matrixsum(ms)
        if validation["unique_terms"] != validation["term_count"] or validation["has_identity"]:
            raise RuntimeError(f"invalid random Pauli sample: {validation}")
        sample_prefix = f"{case['name']}_sample{sample_index:02d}_seed{seed}"
        sample_dir = out_dir / "random_pauli" / case["name"] / f"sample{sample_index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        ours = count_ours_block_encoding_new(copy.deepcopy(ms), sample_dir, sample_prefix + "_ours")
        qrisp = count_qrisp_from_lcu(copy.deepcopy(ms), sample_dir, sample_prefix + "_qrisp", workspace=workspace)
        case_summary["samples"].append(
            {
                "case": case["name"],
                "sample_index": int(sample_index),
                "seed": int(seed),
                "validation": validation,
                "ours_block_encoding_new_current": ours,
                "qrisp_from_lcu": qrisp,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    return case_summary


def extract_single_row(case_summary: dict[str, Any], basis: str, method: str) -> dict[str, Any]:
    result = case_summary.get(method) or {}
    summary = result.get("basis_summary", {}).get(basis)
    if result.get("status") != "ok" or summary is None:
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
        "gate_count": int(summary["gate_count"]),
        "non_clifford": int(summary["non_clifford"]),
        "width": int(summary["width"]),
        "elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
        "error": None,
    }


def comparison_rows_single(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case_summary in cases:
        for basis in ("nam", "ibm"):
            ours = extract_single_row(case_summary, basis, "ours_block_encoding_new_current")
            qrisp = extract_single_row(case_summary, basis, "qrisp_from_lcu")
            qrisp["qrisp_over_ours"] = (
                float(qrisp["gate_count"] / ours["gate_count"])
                if ours["gate_count"] and qrisp["gate_count"]
                else None
            )
            ours["qrisp_over_ours"] = None
            rows.extend([ours, qrisp])
    return rows


def run_tfim_case(name: str, ms: Any, metadata: dict[str, Any], out_dir: Path, workspace: int) -> dict[str, Any]:
    case_dir = out_dir / "tfim" / name
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
        summary["ours_block_encoding_new_current"] = {
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


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{int(round(float(value))):,}"


def fmt_stat(mean: float | None, stdev: float | None, digits: int = 1) -> str:
    if mean is None:
        return "-"
    if stdev is None:
        stdev = 0.0
    return f"{mean:.{digits}f} +/- {stdev:.{digits}f}"


def cell(gates: Any, non_clifford: Any, width: Any, *, stat: bool = False) -> str:
    if gates is None:
        return "-"
    if stat:
        return f"{gates} / {non_clifford} / {width}"
    return f"{fmt_int(gates)} / {fmt_int(non_clifford)} / {fmt_int(width)}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def tex_num_cell(text: str) -> str:
    if text in {"-", "DNF"}:
        return text
    return tex_escape(text).replace("+/-", "$\\pm$")


def build_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rewritten_labels = {
        "HRW_8": "HRW_8",
        "HRW_20": "HRW_20",
        "HRW_28": "HRW_28",
        "random_coloring_linear_n4_q4": "random coloring linear",
        "random_coloring_cycle_n4_q4": "random coloring cycle",
        "repetition_dqe_d3": "repetition DQE d=3",
    }
    grouped_sum: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary["rewritten_kraus"]["comparison_rows"]:
        key = (row["benchmark"], row["basis"].upper())
        grouped_sum.setdefault(
            key,
            {"example": rewritten_labels[row["benchmark"]], "basis": row["basis"].upper()},
        )
        target = "ours" if row["method"].startswith("ours") else "qrisp"
        grouped_sum[key][target] = {
            "gates": row["gate_count_sum"],
            "non_clifford": row["non_clifford_sum"],
            "width": row["max_width"],
        }
        if row["gate_count_sum"] is None:
            grouped_sum[key][f"{target}_status"] = (
                f"{row['completed_kraus']}/{row['missing_kraus']}"
            )
        if row.get("qrisp_over_ours") is not None:
            grouped_sum[key]["ratio"] = row["qrisp_over_ours"]
    present_rewritten = [
        bench["benchmark"]
        for bench in summary["rewritten_kraus"]["benchmarks"]
    ]
    for case in present_rewritten:
        for basis in ("NAM", "IBM"):
            key = (case, basis)
            if key in grouped_sum:
                rows.append(grouped_sum[key])

    tfim_grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary["tfim"]["comparison_rows"]:
        key = (row["case"], row["basis"].upper())
        tfim_grouped.setdefault(
            key,
            {"example": row["case"].replace("_A0", " A0"), "basis": row["basis"].upper()},
        )
        target = "ours" if row["method"].startswith("ours") else "qrisp"
        tfim_grouped[key][target] = {
            "gates": row["gate_count"],
            "non_clifford": row["non_clifford"],
            "width": row["width"],
        }
        if row.get("qrisp_over_ours") is not None:
            tfim_grouped[key]["ratio"] = row["qrisp_over_ours"]
    present_tfim = [
        case_summary["case"]
        for case_summary in summary["tfim"]["cases"]
    ]
    for case in present_tfim:
        for basis in ("NAM", "IBM"):
            key = (case, basis)
            if key in tfim_grouped:
                rows.append(tfim_grouped[key])

    random_grouped: dict[tuple[str, str], dict[str, Any]] = {}
    random_labels = {
        "random_pauli_4q_20t": "4q random Pauli, 20 terms",
        "random_pauli_8q_80t": "8q random Pauli, 80 terms",
    }
    for row in summary["random_pauli"]["aggregate_rows"]:
        key = (row["case"], row["basis"].upper())
        random_grouped.setdefault(
            key,
            {"example": random_labels[row["case"]], "basis": row["basis"].upper(), "stat": True},
        )
        target = "ours" if row["method"].startswith("ours") else "qrisp"
        random_grouped[key][target] = {
            "gates": fmt_stat(row["gate_count"]["mean"], row["gate_count"]["stdev"], 1),
            "non_clifford": fmt_stat(row["non_clifford"]["mean"], row["non_clifford"]["stdev"], 1),
            "width": fmt_stat(row["width"]["mean"], row["width"]["stdev"], 1),
        }
        if row.get("qrisp_over_ours_mean") is not None:
            random_grouped[key]["ratio"] = row["qrisp_over_ours_mean"]
    present_random = [
        case_summary["case"]["name"]
        for case_summary in summary["random_pauli"]["cases"]
    ]
    for case in present_random:
        for basis in ("NAM", "IBM"):
            key = (case, basis)
            if key in random_grouped:
                rows.append(random_grouped[key])
    return rows


def method_cell(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return row.get(f"{key}_status", "-")
    return cell(value["gates"], value["non_clifford"], value["width"], stat=bool(row.get("stat")))


def write_table(summary: dict[str, Any], md_path: Path, tex_path: Path) -> None:
    rows = build_table_rows(summary)
    md_lines = [
        "# Current block_encoding_new Ours vs Qrisp",
        "",
        "Cells report `gates / non-Clifford / width` after Qiskit opt0 lowering.",
        "Ours directly instantiates `block_encoding_new.BlockEncoding` and uses `circuit(opt=\"Matrix-order\")`.",
        "",
        "| example | basis | ours block_encoding_new | Qrisp | Qrisp/ours |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        ratio = row.get("ratio")
        md_lines.append(
            f"| {row['example']} | {row['basis']} | {method_cell(row, 'ours')} | "
            f"{method_cell(row, 'qrisp')} | {'' if ratio is None else f'{ratio:.3g}'} |"
        )
    md_lines.extend(
        [
            "",
            "Notes:",
            "- HRW, random coloring, and repetition DQE rows sum over all nonempty rewritten Kraus operators.",
            "- TFIM rows use the first nonempty Kraus operator A0.",
            "- Random-Pauli rows report 10-sample means using the same seeds as the previous run.",
        ]
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Current block\\_encoding\\_new vs. Qrisp block-encoding counts. Cells are gates / non-Clifford / width after Qiskit opt0 lowering.}",
        "\\label{tab:block-encoding-new-qrisp}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Example & Basis & Ours & Qrisp & Qrisp/Ours \\\\",
        "\\midrule",
    ]
    for row in rows:
        ratio = row.get("ratio")
        tex_lines.append(
            f"{tex_escape(row['example'])} & {row['basis']} & "
            f"{tex_num_cell(method_cell(row, 'ours'))} & "
            f"{tex_num_cell(method_cell(row, 'qrisp'))} & "
            f"{'' if ratio is None else f'{ratio:.3g}'} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.5ex}",
            "\\begin{minipage}{0.98\\linewidth}",
            "\\footnotesize\\emph{Notes.} HRW, random coloring, and repetition DQE sum over nonempty rewritten Kraus operators. TFIM rows use A0. Random-Pauli rows report 10-sample means.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_partial(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.partial.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rewritten-benchmarks",
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
    parser.add_argument("--tfim-cases", nargs="+", default=["TFIM_3_A0", "TFIM_8_A0"])
    parser.add_argument("--random-pauli-samples", type=int, default=10)
    parser.add_argument("--random-pauli-seed", type=int, default=20260703)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--workspace", type=int, default=0)
    parser.add_argument("--rewrite-max-steps", type=int, default=50)
    parser.add_argument("--rewrite-beam-width", type=int, default=3)
    parser.add_argument("--rewrite-tol", type=float, default=1e-12)
    parser.add_argument("--write-qasm-limit", type=int, default=999999)
    parser.add_argument("--skip-qrisp", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    summary: dict[str, Any] = {
        "experiment": "block_encoding_new_qrisp_comparison",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "environment": {
            "CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC": os.environ.get("CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC"),
            "CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH": os.environ.get("CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH"),
        },
        "rewritten_kraus": {"benchmarks": [], "comparison_rows": []},
        "tfim": {"cases": [], "comparison_rows": []},
        "random_pauli": {"cases": [], "aggregate_rows": []},
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in args.rewritten_benchmarks:
        bench_summary = run_rewritten_benchmark(
            benchmark,
            out_dir,
            rewrite_max_steps=args.rewrite_max_steps,
            rewrite_beam_width=args.rewrite_beam_width,
            rewrite_tol=args.rewrite_tol,
            workspace=args.workspace,
            write_qasm_limit=args.write_qasm_limit,
        )
        summary["rewritten_kraus"]["benchmarks"].append(bench_summary)
        summary["rewritten_kraus"]["comparison_rows"] = comparison_rows_sum(
            summary["rewritten_kraus"]["benchmarks"]
        )
        write_partial(summary, out_dir)

    tfim_builders = {
        "TFIM_3_A0": lambda: build_tfim_a0(3),
        "TFIM_8_A0": lambda: build_tfim_a0(8),
    }
    for case_key in args.tfim_cases:
        name, ms, metadata = tfim_builders[case_key]()
        case_summary = run_tfim_case(name, ms, metadata, out_dir, args.workspace)
        summary["tfim"]["cases"].append(case_summary)
        summary["tfim"]["comparison_rows"] = comparison_rows_single(summary["tfim"]["cases"])
        write_partial(summary, out_dir)

    for case_offset, case in enumerate(RANDOM_PAULI_CASES):
        case_summary = run_random_pauli_case(
            case,
            out_dir,
            samples=args.random_pauli_samples,
            seed_base=args.random_pauli_seed,
            case_offset=case_offset,
            workspace=args.workspace,
        )
        summary["random_pauli"]["cases"].append(case_summary)
        rows = []
        for random_case in summary["random_pauli"]["cases"]:
            rows.extend(aggregate_random_case(random_case))
        summary["random_pauli"]["aggregate_rows"] = rows
        write_partial(summary, out_dir)

    summary["rewritten_kraus"]["comparison_rows"] = comparison_rows_sum(
        summary["rewritten_kraus"]["benchmarks"]
    )
    summary["tfim"]["comparison_rows"] = comparison_rows_single(summary["tfim"]["cases"])
    rows = []
    for random_case in summary["random_pauli"]["cases"]:
        rows.extend(aggregate_random_case(random_case))
    summary["random_pauli"]["aggregate_rows"] = rows

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_table(summary, TABLE_MD, TABLE_TEX)
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {TABLE_MD}")
    print(f"wrote {TABLE_TEX}")
    print(json.dumps(build_table_rows(summary), indent=2, default=json_default))


if __name__ == "__main__":
    main()
