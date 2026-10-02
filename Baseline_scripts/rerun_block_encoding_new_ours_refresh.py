#!/usr/bin/env python3
"""Refresh in-house block_encoding_new counts while reusing fixed Qrisp data.

This script is for rerunning the in-house Matrix-order path after local
``block_encoding_new.py`` changes.  It rebuilds the same benchmark inputs as
``run_block_encoding_new_qrisp_comparison.py``, recomputes only the in-house
``block_encoding_new.BlockEncoding(...).circuit(opt="Matrix-order")`` counts,
and copies the previous Qrisp results into the new summary/table.
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
from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: E402
from run_block_encoding_new_qrisp_comparison import (  # type: ignore  # noqa: E402
    build_benchmark_current,
    build_table_rows,
    comparison_rows_single,
    comparison_rows_sum,
    matrixsum_metrics,
    recount_bases,
    summarize_basis,
    write_table,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    write_qasm2,
)
from run_random_pauli_kraus_block_encoding import (  # type: ignore  # noqa: E402
    CASES as RANDOM_PAULI_CASES,
    random_pauli_matrixsum,
    validate_matrixsum,
)
from run_table3_extra_block_encoding_cases import build_tfim_a0  # type: ignore  # noqa: E402
from sum_rewritten_kraus_block_counts_qrisp_vs_ours import (  # type: ignore  # noqa: E402
    add_basis_totals,
    channel_metrics,
    empty_totals,
    kraus_cache_key,
    kraus_terms,
    natural_width,
    rewrite_channel,
)


DEFAULT_PREVIOUS = ROOT / "Baseline_results" / "block_encoding_new_qrisp_comparison_v1" / "summary.json"
DEFAULT_FULL_PREVIOUS = ROOT / "Baseline_results" / "full_pauli_new_block_encoding_n4_n6_v1" / "summary.json"
RESULT_ROOT = ROOT / "Baseline_results" / "block_encoding_new_ours_refresh_v1"
TABLE_MD = ROOT / "Results" / "Table_block_encoding_new_ours_refresh.md"
TABLE_TEX = ROOT / "Results" / "Table_block_encoding_new_ours_refresh.tex"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_ours_current(ms: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
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


def qrisp_rewritten_lookup(previous: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for bench in previous.get("rewritten_kraus", {}).get("benchmarks", []):
        name = bench["benchmark"]
        for row in bench.get("kraus_results", []):
            lookup[(name, int(row["kraus_index"]))] = row.get("qrisp_from_lcu", {})
    return lookup


def qrisp_tfim_lookup(previous: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        case["case"]: case.get("qrisp_from_lcu", {})
        for case in previous.get("tfim", {}).get("cases", [])
    }


def qrisp_random_lookup(previous: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for case in previous.get("random_pauli", {}).get("cases", []):
        name = case["case"]["name"]
        for sample in case.get("samples", []):
            lookup[(name, int(sample["sample_index"]))] = sample.get("qrisp_from_lcu", {})
    return lookup


def qrisp_full_lookup(previous_full: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for case in previous_full.get("cases", []):
        lookup[case["case"]] = case.get("qrisp_from_lcu", {})
    return lookup


def run_rewritten_benchmark_refresh(
    name: str,
    out_dir: Path,
    qrisp_lookup: dict[tuple[str, int], dict[str, Any]],
    *,
    rewrite_max_steps: int,
    rewrite_beam_width: int,
    rewrite_tol: float,
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
        "rewrite": {"elapsed_seconds": float(rewrite_seconds), "result": rewrite_result},
        "policy": {
            "qrisp_input": "copied from previous fixed Qrisp summary",
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
    for ordinal, (kraus_index, ms) in enumerate(rewritten_nonzero):
        key = kraus_cache_key(ms)
        prefix = f"{name}_kraus{kraus_index:04d}_ours"
        if key in ours_cache:
            ours_result = copy.deepcopy(ours_cache[key])
            ours_result["cache_hit"] = True
        else:
            ours_result = count_ours_current(ms, bench_dir, prefix)
            ours_result["cache_hit"] = False
            ours_cache[key] = copy.deepcopy(ours_result)

        qrisp_result = copy.deepcopy(qrisp_lookup.get((name, int(kraus_index)), {}))
        if not qrisp_result:
            qrisp_result = {
                "status": "missing_previous_qrisp",
                "elapsed_seconds": 0.0,
            }
        qrisp_result["cache_hit"] = False
        qrisp_result["copied_from_previous_summary"] = True

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


def comparison_rows_full(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def run_tfim_case_refresh(
    name: str,
    ms: Any,
    metadata: dict[str, Any],
    out_dir: Path,
    qrisp_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_dir = out_dir / "tfim" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    return {
        "case": name,
        "metadata": metadata,
        "metrics": matrixsum_metrics(ms),
        "ours_block_encoding_new_current": count_ours_current(ms, case_dir, f"{name}_ours"),
        "qrisp_from_lcu": copy.deepcopy(qrisp_lookup.get(name, {"status": "missing_previous_qrisp"})),
    }


def run_random_pauli_case_refresh(
    case: dict[str, Any],
    out_dir: Path,
    qrisp_lookup: dict[tuple[str, int], dict[str, Any]],
    *,
    samples: int,
    seed_base: int,
    case_offset: int,
) -> dict[str, Any]:
    case_summary = {"case": case, "num_samples": int(samples), "samples": []}
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
        ours = count_ours_current(copy.deepcopy(ms), sample_dir, sample_prefix + "_ours")
        qrisp = copy.deepcopy(qrisp_lookup.get((case["name"], sample_index), {"status": "missing_previous_qrisp"}))
        qrisp["copied_from_previous_summary"] = True
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


def full_pauli_matrixsum(num_qubits: int) -> Matrixsum:
    import itertools

    labels = ["".join(chars) for chars in itertools.product("IXYZ", repeat=num_qubits)]
    coeff = 1.0 / float(len(labels))
    return Matrixsum([(PauliAtom(label, phase=1.0), coeff) for label in labels])


def run_full_pauli_case_refresh(
    num_qubits: int,
    out_dir: Path,
    qrisp_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    name = f"full_pauli_n{num_qubits}"
    ms = full_pauli_matrixsum(num_qubits)
    case_dir = out_dir / "full_pauli" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    qrisp = copy.deepcopy(qrisp_lookup.get(name, {"status": "missing_previous_qrisp"}))
    qrisp["copied_from_previous_summary"] = True
    return {
        "case": name,
        "metadata": {
            "family": "full_pauli_group_uniform",
            "scope": "single Pauli-sum over all n-qubit Paulis including identity",
            "num_qubits": int(num_qubits),
        },
        "metrics": matrixsum_metrics(ms),
        "ours_block_encoding_new_current": count_ours_current(ms, case_dir, f"{name}_ours"),
        "qrisp_from_lcu": qrisp,
    }


def build_refresh_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = build_table_rows(summary)
    full_grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary["full_pauli"]["comparison_rows"]:
        key = (row["case"], row["basis"].upper())
        full_grouped.setdefault(
            key,
            {"example": row["case"].replace("_", " "), "basis": row["basis"].upper()},
        )
        target = "ours" if row["method"].startswith("ours") else "qrisp"
        full_grouped[key][target] = {
            "gates": row["gate_count"],
            "non_clifford": row["non_clifford"],
            "width": row["width"],
        }
        if row.get("qrisp_over_ours") is not None:
            full_grouped[key]["ratio"] = row["qrisp_over_ours"]
    for case in [case_summary["case"] for case_summary in summary["full_pauli"]["cases"]]:
        for basis in ("NAM", "IBM"):
            key = (case, basis)
            if key in full_grouped:
                rows.append(full_grouped[key])
    return rows


def write_refresh_table(summary: dict[str, Any], md_path: Path, tex_path: Path) -> None:
    rows = build_refresh_table_rows(summary)

    def fmt_int(value: int | float | None) -> str:
        if value is None:
            return "-"
        return f"{int(round(float(value))):,}"

    def method_cell(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        if value is None:
            return row.get(f"{key}_status", "-")
        if row.get("stat"):
            return f"{value['gates']} / {value['non_clifford']} / {value['width']}"
        return f"{fmt_int(value['gates'])} / {fmt_int(value['non_clifford'])} / {fmt_int(value['width'])}"

    md_lines = [
        "# Refreshed block_encoding_new Ours vs Qrisp",
        "",
        "Cells report `gates / non-Clifford / width` after Qiskit opt0 lowering.",
        "Ours was rerun with the current `block_encoding_new.BlockEncoding` and `circuit(opt=\"Matrix-order\")`; Qrisp entries are copied from the fixed previous baseline summary.",
        "",
        "| example | basis | refreshed ours block_encoding_new | Qrisp | Qrisp/ours |",
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
            "- Full Pauli n=6 uses the generic `block_encoding_new.BlockEncoding` interface, not the closed-form structured shortcut.",
        ]
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    def tex_escape(text: str) -> str:
        return (
            text.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
        )

    def tex_cell(text: str) -> str:
        return tex_escape(text).replace("+/-", "$\\pm$")

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Refreshed current block\\_encoding\\_new vs. Qrisp block-encoding counts. Cells are gates / non-Clifford / width after Qiskit opt0 lowering.}",
        "\\label{tab:block-encoding-new-qrisp-refresh}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Example & Basis & Ours & Qrisp & Qrisp/Ours \\\\",
        "\\midrule",
    ]
    for row in rows:
        ratio = row.get("ratio")
        tex_lines.append(
            f"{tex_escape(row['example'])} & {row['basis']} & "
            f"{tex_cell(method_cell(row, 'ours'))} & "
            f"{tex_cell(method_cell(row, 'qrisp'))} & "
            f"{'' if ratio is None else f'{ratio:.3g}'} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.5ex}",
            "\\begin{minipage}{0.98\\linewidth}",
            "\\footnotesize\\emph{Notes.} Ours was rerun with the current block\\_encoding\\_new implementation; Qrisp rows reuse the fixed previous Qrisp baseline. HRW, random coloring, and repetition DQE sum over nonempty rewritten Kraus operators. TFIM rows use A0. Random-Pauli rows report 10-sample means. Full Pauli n=6 uses the generic interface.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_partial(summary: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "summary.partial.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-summary", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--previous-full-summary", type=Path, default=DEFAULT_FULL_PREVIOUS)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
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
    parser.add_argument("--full-pauli-n", type=int, nargs="+", default=[6])
    parser.add_argument("--rewrite-max-steps", type=int, default=50)
    parser.add_argument("--rewrite-beam-width", type=int, default=3)
    parser.add_argument("--rewrite-tol", type=float, default=1e-12)
    args = parser.parse_args()

    previous = load_json(args.previous_summary.resolve())
    previous_full = load_json(args.previous_full_summary.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    qrisp_rewritten = qrisp_rewritten_lookup(previous)
    qrisp_tfim = qrisp_tfim_lookup(previous)
    qrisp_random = qrisp_random_lookup(previous)
    qrisp_full = qrisp_full_lookup(previous_full)

    summary: dict[str, Any] = {
        "experiment": "block_encoding_new_ours_refresh",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "environment": {
            "CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC": os.environ.get("CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC"),
            "CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH": os.environ.get("CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH"),
        },
        "previous_qrisp_summary": str(args.previous_summary.resolve()),
        "previous_full_qrisp_summary": str(args.previous_full_summary.resolve()),
        "rewritten_kraus": {"benchmarks": [], "comparison_rows": []},
        "tfim": {"cases": [], "comparison_rows": []},
        "random_pauli": {"cases": [], "aggregate_rows": []},
        "full_pauli": {"cases": [], "comparison_rows": []},
    }

    for benchmark in args.rewritten_benchmarks:
        bench_summary = run_rewritten_benchmark_refresh(
            benchmark,
            out_dir,
            qrisp_rewritten,
            rewrite_max_steps=args.rewrite_max_steps,
            rewrite_beam_width=args.rewrite_beam_width,
            rewrite_tol=args.rewrite_tol,
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
        summary["tfim"]["cases"].append(
            run_tfim_case_refresh(name, ms, metadata, out_dir, qrisp_tfim)
        )
        summary["tfim"]["comparison_rows"] = comparison_rows_single(summary["tfim"]["cases"])
        write_partial(summary, out_dir)

    for case_offset, case in enumerate(RANDOM_PAULI_CASES):
        case_summary = run_random_pauli_case_refresh(
            case,
            out_dir,
            qrisp_random,
            samples=args.random_pauli_samples,
            seed_base=args.random_pauli_seed,
            case_offset=case_offset,
        )
        summary["random_pauli"]["cases"].append(case_summary)
        rows = []
        for random_case in summary["random_pauli"]["cases"]:
            rows.extend(aggregate_random_case(random_case))
        summary["random_pauli"]["aggregate_rows"] = rows
        write_partial(summary, out_dir)

    for n in args.full_pauli_n:
        summary["full_pauli"]["cases"].append(
            run_full_pauli_case_refresh(n, out_dir, qrisp_full)
        )
        summary["full_pauli"]["comparison_rows"] = comparison_rows_full(
            summary["full_pauli"]["cases"]
        )
        write_partial(summary, out_dir)

    summary["rewritten_kraus"]["comparison_rows"] = comparison_rows_sum(
        summary["rewritten_kraus"]["benchmarks"]
    )
    summary["tfim"]["comparison_rows"] = comparison_rows_single(summary["tfim"]["cases"])
    random_rows = []
    for random_case in summary["random_pauli"]["cases"]:
        random_rows.extend(aggregate_random_case(random_case))
    summary["random_pauli"]["aggregate_rows"] = random_rows
    summary["full_pauli"]["comparison_rows"] = comparison_rows_full(summary["full_pauli"]["cases"])

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_refresh_table(summary, TABLE_MD, TABLE_TEX)
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {TABLE_MD}")
    print(f"wrote {TABLE_TEX}")
    print(json.dumps(build_refresh_table_rows(summary), indent=2, default=json_default))


if __name__ == "__main__":
    main()
