#!/usr/bin/env python3
"""Build selected paper-facing channel-LCU tables.

The tables are intentionally smaller than the full appendix tables:

* width+time: representative examples, with one raw-width column and selected
  tool width/time side by side;
* scalability: hard cases where raw exists but generic optimizers mostly do
  not finish within one hour, plus cases where the raw denominator is missing.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import build_channel_lcu_appendix_derived_tables as derived  # noqa: E402
import generate_channel_lcu_appendix_full_table as appendix  # noqa: E402


OUT_WIDTH_TIME_CSV = ROOT / "Results" / "channel_lcu_selected_width_time.csv"
OUT_WIDTH_TIME_MD = ROOT / "Results" / "channel_lcu_selected_width_time.md"
OUT_WIDTH_TIME_TEX = ROOT / "Results" / "channel_lcu_selected_width_time.tex"
OUT_SCALABILITY_CSV = ROOT / "Results" / "channel_lcu_selected_scalability_cases.csv"
OUT_SCALABILITY_MD = ROOT / "Results" / "channel_lcu_selected_scalability_cases.md"
OUT_SCALABILITY_TEX = ROOT / "Results" / "channel_lcu_selected_scalability_cases.tex"

WIDTH_TIME_BENCHMARKS = [
    "periodic_tfim_n4_dt0p1",
    "random_channel_N5_dim4_N1_10_N2_2_mean",
    "graph_cluster_2d_grid_2x4",
    "hypercube_random_walk_n12",
    "repetition_dqe_d3_eps0p25",
]
SCALABILITY_BENCHMARKS = [
    "periodic_tfim_n12_dt0p1",
    "graph_cluster_random_bd3_n12_seed7",
    "random_channel_N9_dim8_N1_20_N2_4_mean",
    "random_coloring_n4_q4_linear",
    "repetition_dqe_d5_eps0p25",
]
NAM_SELECTED_TOOLS = ["ours", "voqc", "wisq", "quartz"]
IBM_SELECTED_TOOLS = ["ours", "qiskit_opt3", "pytket", "bqskit"]
ALL_SELECTED_TOOLS = sorted(set(NAM_SELECTED_TOOLS + IBM_SELECTED_TOOLS))
DISPLAY_TOOLS = ["ours", "voqc", "wisq", "quartz", "qiskit_opt3", "pytket", "bqskit"]
TOOL_BASIS = {
    "voqc": "NAM",
    "wisq": "NAM",
    "quartz": "NAM",
    "qiskit_opt3": "IBM",
    "pytket": "IBM",
    "bqskit": "IBM",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(round(number))


def fmt_width(value: float | None) -> str:
    if value is None:
        return "--"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def fmt_time_cell(value: str) -> str:
    return value or "--"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "--"
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def time_cell_to_float(value: str) -> float | None:
    if not value or value in {"--", "n/a"}:
        return None
    stripped = value.rstrip("*")
    if stripped.startswith(("TO", "ERR", "PI", "UG")):
        return None
    return as_float(stripped)


def tex_escape(text: str) -> str:
    text = text.replace("\\_", "_")
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def label_for(benchmark: str) -> str:
    return appendix.LABELS.get(benchmark, benchmark).replace("\\_", "_")


def markdown_table(fields: list[str], rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") or "--" for field in fields) + " |")
    return "\n".join(lines)


def load_runtime_cells() -> dict[tuple[str, str, str], str]:
    cells: dict[tuple[str, str, str], str] = {}
    for row in read_csv(ROOT / "Results" / "channel_lcu_appendix_compile_times.csv"):
        benchmark = row["benchmark_name"]
        basis = row["basis"].upper()
        for tool in ALL_SELECTED_TOOLS:
            if row.get(tool):
                cells[(benchmark, basis, tool)] = row[tool]
    return cells


def load_generic_widths() -> dict[tuple[str, str, str], float]:
    widths: dict[tuple[str, str, str], float] = {}
    for row in read_csv(derived.SOURCE_CSV):
        benchmark = row.get("benchmark_name", "")
        basis = row.get("basis_group", "").upper()
        if basis not in {"NAM", "IBM"}:
            continue
        method = row.get("method", "")
        if method == "raw_denominator":
            tool = "raw"
            width = as_float(row.get("raw_width") or row.get("opt_width"))
        elif method == "generic_optimizer":
            tool = row.get("optimizer", "").lower()
            width = as_float(row.get("opt_width") or row.get("raw_width"))
        else:
            continue
        if tool and width is not None:
            widths[(benchmark, basis, tool)] = width

    # Random-channel generic summary rows do not always carry qubit width. The
    # native tools are compiled from the raw native-basis circuit and preserve
    # the input width in these runs, so use the mean raw width when a numeric
    # tool output exists but no per-tool width was recorded.
    entries, _family, _benchmarks = appendix.load_entries()
    raw_widths = derived.load_raw_widths()
    for benchmark in appendix.RANDOM_CHANNEL_PARAMS.values():
        for basis, tools in (("NAM", ("voqc", "wisq", "quartz")), ("IBM", ("qiskit_opt3", "pytket", "bqskit"))):
            raw_width = raw_widths.get((benchmark, basis))
            if raw_width is None:
                continue
            for tool in tools:
                entry = entries.get((benchmark, basis, tool))
                if entry is not None and as_int(entry.opt_gates) is not None:
                    widths.setdefault((benchmark, basis, tool), raw_width)
    return widths


def tool_width_cell(
    *,
    benchmark: str,
    basis: str,
    tool: str,
    entries: dict[tuple[str, str, str], appendix.Entry],
    raw_widths: dict[tuple[str, str], float],
    ours_widths: dict[tuple[str, str], float],
    generic_widths: dict[tuple[str, str, str], float],
) -> str:
    entry = entries.get((benchmark, basis, tool))
    has_gates = entry is not None and as_int(entry.opt_gates) is not None
    if tool == "ours":
        return fmt_width(ours_widths.get((benchmark, basis))) if has_gates else "--"
    width = generic_widths.get((benchmark, basis, tool))
    if width is None and has_gates:
        width = raw_widths.get((benchmark, basis))
    return fmt_width(width) if has_gates else "--"


def collapsed_width(widths: dict[tuple[str, str], float], benchmark: str) -> str:
    values = [
        widths.get((benchmark, "NAM")),
        widths.get((benchmark, "IBM")),
    ]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return "--"
    return fmt_width(max(numeric))


def collapsed_ours_width(
    widths: dict[tuple[str, str], float],
    benchmark: str,
) -> str:
    values = [
        widths.get((benchmark, "NAM")),
        widths.get((benchmark, "IBM")),
    ]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return "--"
    return fmt_width(max(numeric))


def collapsed_ours_time(
    runtime_cells: dict[tuple[str, str, str], str],
    benchmark: str,
) -> str:
    values = [
        runtime_cells.get((benchmark, "NAM", "ours"), ""),
        runtime_cells.get((benchmark, "IBM", "ours"), ""),
    ]
    numeric = [time_cell_to_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if numeric:
        return fmt_seconds(max(numeric))
    for value in values:
        if value:
            return value
    return "--"


def build_width_time_rows() -> list[dict[str, str]]:
    entries, family_map, _benchmarks = appendix.load_entries()
    raw_widths = derived.load_raw_widths()
    raw_widths.update(derived.load_refresh_raw_widths())
    ours_widths = derived.load_latest_ours_widths()
    generic_widths = load_generic_widths()
    runtime_cells = load_runtime_cells()

    rows: list[dict[str, str]] = []
    for benchmark in WIDTH_TIME_BENCHMARKS:
        row = {
            "example": label_for(benchmark),
            "benchmark_name": benchmark,
            "family": family_map.get((benchmark, "IBM"))
            or family_map.get((benchmark, "NAM"))
            or "",
        }
        row["raw_width"] = collapsed_width(raw_widths, benchmark)

        for tool in DISPLAY_TOOLS:
            if tool == "ours":
                row["ours_width"] = collapsed_ours_width(ours_widths, benchmark)
                row["ours_time_s"] = collapsed_ours_time(runtime_cells, benchmark)
                nam_entry = entries.get((benchmark, "NAM", "ours"))
                ibm_entry = entries.get((benchmark, "IBM", "ours"))
                row["ours_status"] = (
                    nam_entry.status
                    if nam_entry and ibm_entry and nam_entry.status == ibm_entry.status
                    else f"NAM:{nam_entry.status if nam_entry else 'missing'};IBM:{ibm_entry.status if ibm_entry else 'missing'}"
                )
                continue
            basis = TOOL_BASIS[tool]
            row[f"{tool}_width"] = tool_width_cell(
                benchmark=benchmark,
                basis=basis,
                tool=tool,
                entries=entries,
                raw_widths=raw_widths,
                ours_widths=ours_widths,
                generic_widths=generic_widths,
            )
            row[f"{tool}_time_s"] = fmt_time_cell(
                runtime_cells.get((benchmark, basis, tool), "")
            )
            entry = entries.get((benchmark, basis, tool))
            row[f"{tool}_status"] = entry.status if entry else "missing"
        rows.append(row)
    return rows


def write_width_time_md(rows: list[dict[str, str]]) -> None:
    fields = [
        "example",
        "raw_width",
        "ours_width",
        "ours_time_s",
        "voqc_time_s",
        "wisq_time_s",
        "quartz_time_s",
        "qiskit_opt3_time_s",
        "pytket_time_s",
        "bqskit_time_s",
    ]
    text = [
        "# Selected Channel-LCU Width And Compile-Time Comparison",
        "",
        "Representative instances. Raw reports width only; Ours reports width and compile time. External optimizer columns report compile/optimizer time in seconds. For these native raw-input runs, external optimizer widths equal the raw width whenever a numeric output is available.",
        "",
        markdown_table(fields, rows),
        "",
    ]
    OUT_WIDTH_TIME_MD.write_text("\n".join(text), encoding="utf-8")


def write_width_time_tex(rows: list[dict[str, str]]) -> None:
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\caption{Selected circuit width and compile/optimization time comparison. Raw reports width only; Ours reports width and compile time. External optimizer columns report time in seconds; their output widths equal the raw width whenever a numeric output is available. Raw-circuit generation is excluded.}",
        "\\label{tab:channel-lcu-selected-width-time}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        "Example & Raw W & Ours W & Ours T & VOQC T & WISQ T & Quartz T & Qiskit opt3 T & pytket T & BQSKit T \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [
            row["example"],
            row["raw_width"],
            row["ours_width"],
            row["ours_time_s"],
            row["voqc_time_s"],
            row["wisq_time_s"],
            row["quartz_time_s"],
            row["qiskit_opt3_time_s"],
            row["pytket_time_s"],
            row["bqskit_time_s"],
        ]
        lines.append(" & ".join(tex_escape(value) for value in values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{2pt}",
            "\\begin{minipage}{\\textwidth}",
            "\\scriptsize",
            "\\emph{Notes.} W: circuit width. T: compile/optimizer time in seconds. Generic optimizers are reported on their native inputs; their widths are the corresponding raw width for numeric outputs. TO: timeout/no output; ERR: runner/environment error; n/a: numeric result exists but runtime was not recorded; *: partial/checkpoint or non-standard-status result.",
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    OUT_WIDTH_TIME_TEX.write_text("\n".join(lines), encoding="utf-8")


def build_scalability_rows() -> list[dict[str, str]]:
    source_rows = {
        row["benchmark_name"]: row
        for row in read_csv(ROOT / "Results" / "channel_lcu_appendix_scalability_cases.csv")
    }
    category = {
        "periodic_tfim_n12_dt0p1": "external timeout",
        "graph_cluster_random_bd3_n12_seed7": "external timeout",
        "random_channel_N9_dim8_N1_20_N2_4_mean": "external timeout",
        "random_coloring_n4_q4_linear": "raw unavailable",
        "repetition_dqe_d5_eps0p25": "raw unavailable",
    }
    rows: list[dict[str, str]] = []
    for benchmark in SCALABILITY_BENCHMARKS:
        row = source_rows[benchmark]
        case_type = category[benchmark]
        raw_availability = "No" if case_type == "raw unavailable" else row["raw_availability"]
        rows.append(
            {
                "example": label_for(benchmark),
                "benchmark_name": benchmark,
                "family": row["family"],
                "case_type": case_type,
                "raw_availability": raw_availability,
                "ours_result": row["ours_result"],
                "external_under_1h": row["external_under_1h"],
            }
        )
    return rows


def write_scalability_md(rows: list[dict[str, str]]) -> None:
    fields = [
        "example",
        "case_type",
        "raw_availability",
        "ours_result",
        "external_under_1h",
    ]
    text = [
        "# Selected Channel-LCU Scalability Cases",
        "",
        "Representative cases selected from the full CP1A/CP1B appendix table.",
        "",
        markdown_table(fields, rows),
        "",
    ]
    OUT_SCALABILITY_MD.write_text("\n".join(text), encoding="utf-8")


def write_scalability_tex(rows: list[dict[str, str]]) -> None:
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\caption{Selected scalability cases from the CP1A/CP1B channel-LCU appendix.}",
        "\\label{tab:channel-lcu-selected-scalability}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Example & Case type & Raw availability & Ours & External $<1$h \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [
            row["example"],
            row["case_type"],
            row["raw_availability"],
            row["ours_result"],
            row["external_under_1h"],
        ]
        lines.append(" & ".join(tex_escape(value) for value in values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    OUT_SCALABILITY_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    width_rows = build_width_time_rows()
    width_fields = [
        "example",
        "benchmark_name",
        "family",
        "raw_width",
        "ours_width",
        "ours_time_s",
        "voqc_time_s",
        "wisq_time_s",
        "quartz_time_s",
        "qiskit_opt3_time_s",
        "pytket_time_s",
        "bqskit_time_s",
    ]
    write_csv(OUT_WIDTH_TIME_CSV, width_fields, width_rows)
    write_width_time_md(width_rows)
    write_width_time_tex(width_rows)

    scalability_rows = build_scalability_rows()
    scalability_fields = [
        "example",
        "benchmark_name",
        "family",
        "case_type",
        "raw_availability",
        "ours_result",
        "external_under_1h",
    ]
    write_csv(OUT_SCALABILITY_CSV, scalability_fields, scalability_rows)
    write_scalability_md(scalability_rows)
    write_scalability_tex(scalability_rows)

    print(f"wrote {len(width_rows)} selected width/time rows to {OUT_WIDTH_TIME_CSV}")
    print(f"wrote {len(scalability_rows)} selected scalability rows to {OUT_SCALABILITY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
