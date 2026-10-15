#!/usr/bin/env python3
"""Render the RQ2 main-body full-factor ablation table.

The table follows Section 5 of codex_multiagent_evaluation_plan_v2.md:
cells are normalized by the Basic configuration unless an explicit reference
configuration is listed, and F is labeled as the adopted prior
flattening/unary-iteration pass.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Results" / "eval_redesign" / "tables"

CONFIGS = [
    ("R", "Rewrite only"),
    ("O", "Order only"),
    ("F*", "Flat only"),
    ("R+O", "Rewrite + Order"),
    ("R+F*", "Rewrite + Flat"),
    ("O+F*", "Flat + Order"),
    ("All", "Rewrite + Flat + Order"),
]

SOURCES = [
    {
        "family": "Lindblad",
        "benchmark": "TFIM-8",
        "summary": ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "cp2_rq2_main_ablation_tfim8_20260706_matrixsum_rerun"
        / "summary.csv",
        "reference_config": "Basic",
        "note": "R neutral (41->41); O+F best.",
    },
    {
        "family": "Algorithm",
        "benchmark": "HRW-20",
        "summary": ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "cp2_rq2_main_ablation_hrw20_20260705_stage_timing"
        / "summary.csv",
        "reference_config": "Basic",
        "note": "R support 160->80; All best.",
    },
    {
        "family": "Algorithm",
        "benchmark": "Rand-color-n4q4",
        "summary": ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "cp2_rq2_main_ablation_random_20260706_matrixsum_rerun"
        / "random_coloring_n4_q4_linear"
        / "summary.csv",
        "reference_config": "Flat + Order",
        "note": "Basic, R-only, and F-only timeout; denominator is O+F.",
    },
    {
        "family": "Synthetic",
        "benchmark": "Rand-Paulis-(5,4,10,2)",
        "summary": ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "cp2_rq2_main_ablation_random_20260705_stage_timing"
        / "random_pauli_kraus_N5_dim4_N1_10_N2_2"
        / "summary.csv",
        "reference_config": "Basic",
        "note": "5-seed mean; R nearly neutral; O+F dominates.",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt_gates(value: Any, status: str = "") -> str:
    val = to_float(value)
    if val is None:
        return "T/O" if status == "timeout" else "--"
    if abs(val - round(val)) < 1e-9:
        return f"{int(round(val)):,}"
    return f"{val:,.1f}"


def fmt_ratio(value: Any) -> str:
    val = to_float(value)
    if val is None:
        return "n/a"
    return f"{val * 100:.2f}%"


def fmt_metric(value: Any) -> str:
    val = to_float(value)
    if val is None:
        return ""
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.12g}"


def config_cell(row: dict[str, str] | None, reference_gates: float | None) -> str:
    if row is None:
        return "--"
    status = row.get("status", "")
    if status == "timeout":
        return "T/O"
    gates = to_float(row.get("total_gates"))
    if reference_gates is None or gates is None:
        return "n/a"
    return fmt_ratio(gates / reference_gates)


def validate_source(meta: dict[str, Any], rows: list[dict[str, str]]) -> None:
    if len(rows) != 8:
        raise ValueError(f"{meta['benchmark']} has {len(rows)} rows, expected 8")
    by_config = {row["config"]: row for row in rows}
    missing = {"Basic", *(name for _, name in CONFIGS)} - set(by_config)
    if missing:
        raise ValueError(f"{meta['benchmark']} missing configs: {sorted(missing)}")
    for row in rows:
        if row.get("basis") != "ibm":
            raise ValueError(f"{meta['benchmark']} has non-IBM row: {row['config']}")
        if row.get("rewrite_search") == "True":
            if row.get("rewrite_method") != "gatecost_rewrite_search":
                raise ValueError(f"{meta['benchmark']} bad rewrite method: {row['config']}")
            if row.get("rewrite_strategy") != "beam":
                raise ValueError(f"{meta['benchmark']} bad rewrite strategy: {row['config']}")
        if row.get("order_enabled") == "True" and "if_optimize=True" not in row.get("notes", ""):
            raise ValueError(f"{meta['benchmark']} missing matrix-order guardrail: {row['config']}")
    if "Rand-Paulis" in meta["benchmark"]:
        for row in rows:
            if row.get("completed_seed_count") != "5":
                raise ValueError(f"{meta['benchmark']} is not a 5-seed mean: {row['config']}")


def build_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for meta in SOURCES:
        summary_path = Path(meta["summary"])
        rows = read_csv(summary_path)
        validate_source(meta, rows)
        by_config = {row["config"]: row for row in rows}
        basic = by_config["Basic"]
        rewrite_row = by_config["Rewrite only"]
        reference_config = meta.get("reference_config", "Basic")
        reference = by_config[reference_config]
        reference_gates = to_float(reference.get("total_gates")) if reference.get("status") == "ok" else None
        rewrite_change = (
            f"{fmt_metric(rewrite_row.get('rewrite_initial_support'))}"
            f"->{fmt_metric(rewrite_row.get('rewrite_final_support'))}"
        )

        main_row: dict[str, Any] = {
            "Family": meta["family"],
            "Benchmark": meta["benchmark"],
            "Basis": basic.get("basis", "ibm"),
            "Denominator": "Basic" if reference_config == "Basic" else "O+F*",
            "Rewrite": rewrite_change,
            "Reference gates": fmt_gates(reference.get("total_gates"), reference.get("status", "")),
            "R": config_cell(by_config.get("Rewrite only"), reference_gates),
            "O": config_cell(by_config.get("Order only"), reference_gates),
            "F*": config_cell(by_config.get("Flat only"), reference_gates),
            "R+O": config_cell(by_config.get("Rewrite + Order"), reference_gates),
            "R+F*": config_cell(by_config.get("Rewrite + Flat"), reference_gates),
            "O+F*": config_cell(by_config.get("Flat + Order"), reference_gates),
            "All": config_cell(by_config.get("Rewrite + Flat + Order"), reference_gates),
            "Note": meta["note"],
        }
        main_rows.append(main_row)
        source_rows.append(
            {
                "Family": meta["family"],
                "Benchmark": meta["benchmark"],
                "summary_csv": str(summary_path.relative_to(ROOT)),
                "basic_status": basic.get("status", ""),
                "reference_config": reference_config,
                "reference_status": reference.get("status", ""),
                "row_count": len(rows),
            }
        )

        for symbol, config in [("Basic", "Basic"), *CONFIGS]:
            row = by_config[config]
            gates = to_float(row.get("total_gates"))
            full_rows.append(
                {
                    "family": meta["family"],
                    "benchmark": meta["benchmark"],
                    "basis": row.get("basis", ""),
                    "reference_config": reference_config,
                    "config_symbol": symbol,
                    "config": config,
                    "status": row.get("status", ""),
                    "timeout": row.get("timeout", ""),
                    "total_gates": fmt_metric(row.get("total_gates")),
                    "ratio_vs_basic": fmt_metric(row.get("ratio_vs_basic")),
                    "ratio_vs_reference": fmt_metric(gates / reference_gates if gates is not None and reference_gates else ""),
                    "non_clifford_count": fmt_metric(row.get("non_clifford_count")),
                    "cx_count": fmt_metric(row.get("cx_count") or row.get("cx")),
                    "depth": fmt_metric(row.get("depth")),
                    "width": fmt_metric(row.get("width")),
                    "total_time_sec": fmt_metric(row.get("total_time_sec")),
                    "execution_time_s": fmt_metric(row.get("execution_time_s")),
                    "construction_time_sec": fmt_metric(row.get("construction_time_sec")),
                    "rewrite_time_sec": fmt_metric(row.get("rewrite_time_sec")),
                    "structure_time_sec": fmt_metric(row.get("structure_time_sec")),
                    "ordering_time_sec": fmt_metric(row.get("ordering_time_sec")),
                    "channel_lcu_total_time_sec": fmt_metric(row.get("channel_lcu_total_time_sec")),
                    "block_encoding_time_sec": fmt_metric(row.get("block_encoding_time_sec")),
                    "block_encoding_total_time_sec": fmt_metric(row.get("block_encoding_total_time_sec")),
                    "block_encoding_mean_time_sec": fmt_metric(row.get("block_encoding_mean_time_sec")),
                    "block_encoding_max_time_sec": fmt_metric(row.get("block_encoding_max_time_sec")),
                    "block_encoding_kraus_count": fmt_metric(row.get("block_encoding_kraus_count")),
                    "channel_lcu_without_block_encoding_time_sec": fmt_metric(row.get("channel_lcu_without_block_encoding_time_sec")),
                    "per_kraus_block_encoding_times_sec": row.get("per_kraus_block_encoding_times_sec", ""),
                    "per_kraus_block_encoding_details_json": row.get("per_kraus_block_encoding_details_json", ""),
                    "transpile_time_sec": fmt_metric(row.get("transpile_time_sec")),
                    "count_time_sec": fmt_metric(row.get("count_time_sec")),
                    "rewrite_method": row.get("rewrite_method", ""),
                    "rewrite_strategy": row.get("rewrite_strategy", ""),
                    "rewrite_initial_support": fmt_metric(row.get("rewrite_initial_support")),
                    "rewrite_final_support": fmt_metric(row.get("rewrite_final_support")),
                    "rewrite_steps": fmt_metric(row.get("rewrite_steps")),
                    "completed_seed_count": row.get("completed_seed_count", ""),
                    "notes": row.get("notes", ""),
                    "error_msg": row.get("error_msg", ""),
                }
            )

    return main_rows, full_rows, source_rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Family",
        "Benchmark",
        "Denominator",
        "Rewrite",
        "Reference gates",
        "R",
        "O",
        "F*",
        "R+O",
        "R+F*",
        "O+F*",
        "All",
    ]
    lines = [
        "# RQ2 Main Ablation Table",
        "",
        "Cells report total gate-count ratios in the IBM gate set, normalized by the listed denominator. `T/O` means timeout.",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    lines.extend(
        [
            "",
            "All ratios are normalized by Basic except RandomColoring, where Basic/R/F timed out and O+F* (opt+Matrix-order) is used as the 100% reference. F* denotes adopted prior conditional flattening / unary iteration, not a new optimization introduced by this work.",
            "",
            "The companion `rq2_main_ablation_full_metrics.csv` records gates, CX, non-Clifford counts, width/depth, status, and execution time for every configuration.",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_escape(text: Any) -> str:
    escaped = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def latex_cell(value: Any) -> str:
    text = str(value)
    if text == "O+F*":
        return r"O+F$^\ast$"
    return latex_escape(text)


def render_latex(rows: list[dict[str, Any]]) -> str:
    header = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{RQ2 full-factor ablation of ChannelIR rewrite (R), Pauli-sum ordering (O), and the adopted flattening pass (F). Cells report total gate-count ratios in the IBM gate set, normalized by the listed denominator.}",
        r"\label{tab:rq2-main-ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrrrrrr}",
        r"\toprule",
        r"Family & Benchmark & Denom. & Rewrite & Ref. gates & R & O & F$^\ast$ & R+O & R+F$^\ast$ & O+F$^\ast$ & All \\",
        r"\midrule",
    ]
    body: list[str] = []
    columns = [
        "Family",
        "Benchmark",
        "Denominator",
        "Rewrite",
        "Reference gates",
        "R",
        "O",
        "F*",
        "R+O",
        "R+F*",
        "O+F*",
        "All",
    ]
    for row in rows:
        body.append(" & ".join(latex_cell(row[col]) for col in columns) + r" \\")
    footer = [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{0.5ex}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize\emph{Notes.} All ratios are normalized by Basic except RandomColoring, where Basic/R/F timed out and O+F$^\ast$ (opt+Matrix-order) is used as the 100\% reference. F$^\ast$ denotes adopted prior conditional flattening / unary iteration, not a new optimization introduced by this work. T/O means timeout. Random Pauli-Kraus uses the mean over seeds 0--4.",
        r"\end{minipage}",
        r"\end{table*}",
    ]
    return "\n".join(header + body + footer) + "\n"


def main() -> None:
    main_rows, full_rows, source_rows = build_tables()

    main_fields = [
        "Family",
        "Benchmark",
        "Basis",
        "Denominator",
        "Rewrite",
        "Reference gates",
        "R",
        "O",
        "F*",
        "R+O",
        "R+F*",
        "O+F*",
        "All",
    ]
    full_fields = [
        "family",
        "benchmark",
        "basis",
        "reference_config",
        "config_symbol",
        "config",
        "status",
        "timeout",
        "total_gates",
        "ratio_vs_basic",
        "ratio_vs_reference",
        "non_clifford_count",
        "cx_count",
        "depth",
        "width",
        "total_time_sec",
        "execution_time_s",
        "construction_time_sec",
        "rewrite_time_sec",
        "structure_time_sec",
        "ordering_time_sec",
        "channel_lcu_total_time_sec",
        "block_encoding_time_sec",
        "block_encoding_total_time_sec",
        "block_encoding_mean_time_sec",
        "block_encoding_max_time_sec",
        "block_encoding_kraus_count",
        "channel_lcu_without_block_encoding_time_sec",
        "per_kraus_block_encoding_times_sec",
        "per_kraus_block_encoding_details_json",
        "transpile_time_sec",
        "count_time_sec",
        "rewrite_method",
        "rewrite_strategy",
        "rewrite_initial_support",
        "rewrite_final_support",
        "rewrite_steps",
        "completed_seed_count",
        "notes",
        "error_msg",
    ]
    source_fields = [
        "Family",
        "Benchmark",
        "summary_csv",
        "basic_status",
        "reference_config",
        "reference_status",
        "row_count",
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "rq2_main_ablation_table.csv", main_rows, main_fields)
    write_csv(RESULTS / "rq2_main_ablation_full_metrics.csv", full_rows, full_fields)
    write_csv(RESULTS / "rq2_main_ablation_sources.csv", source_rows, source_fields)
    (RESULTS / "rq2_main_ablation_table.md").write_text(render_markdown(main_rows))
    (RESULTS / "rq2_main_ablation_table.tex").write_text(render_latex(main_rows))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "basis": "ibm",
        "basis_gates": ["u1", "u2", "u3", "cx"],
        "normalization": "optimized_total_gates / listed_denominator_total_gates; denominator is Basic except RandomColoring, where it is Flat + Order because Basic/R/F timed out",
        "rewrite_method": "gatecost_rewrite_search",
        "rewrite_strategy": "beam",
        "matrix_order_guardrail": "BlockEncoding.circuit(opt='Matrix-order') uses expand_large_controlled_gates(..., if_optimize=True).",
        "random_pauli_seed_policy": "mean over seeds 0,1,2,3,4",
        "outputs": [
            "rq2_main_ablation_table.csv",
            "rq2_main_ablation_table.md",
            "rq2_main_ablation_table.tex",
            "rq2_main_ablation_full_metrics.csv",
            "rq2_main_ablation_sources.csv",
        ],
    }
    (RESULTS / "rq2_main_ablation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
