#!/usr/bin/env python3
"""Build SELECT-only random trend artifacts and compact main-table variants."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_SUMMARY = ROOT / "Baseline_results" / "table_main_select_only_ours_qrisp_v1" / "summary.json"
DEFAULT_EXTRA_SUMMARY = ROOT / "Baseline_results" / "table_main_select_only_random4q100_v1" / "summary.json"
DEFAULT_EXTRA_SUMMARIES = (
    DEFAULT_EXTRA_SUMMARY,
    ROOT / "Baseline_results" / "table_main_select_only_random4q164_v1" / "summary.json",
)
TREND_CSV = ROOT / "Results" / "random_pauli_4q_select_only_trend.csv"
TREND_MD = ROOT / "Results" / "random_pauli_4q_select_only_trend.md"
TREND_SVG = ROOT / "Results" / "random_pauli_4q_select_only_trend.svg"
TREND_PDF = ROOT / "Results" / "random_pauli_4q_select_only_trend.pdf"
COMPACT_MD = ROOT / "Results" / "Table_main_select_only_ours_qrisp_compact.md"
COMPACT_TEX = ROOT / "Results" / "Table_main_select_only_ours_qrisp_compact.tex"

COMPACT_CASES = (
    "HRW_8",
    "HRW_28",
    "random_coloring_linear_n4_q4",
    "random_coloring_cycle_n4_q4",
    "repetition_dqe_d3",
    "TFIM_3_A0",
    "TFIM_8_A0",
    "random_pauli_4q_20t",
    "random_pauli_8q_80t",
    "full_pauli_n6",
)

TREND_CASES = (
    "random_pauli_4q_20t",
    "random_pauli_4q_64t",
    "random_pauli_4q_100t",
    "random_pauli_4q_129t",
    "random_pauli_4q_164t",
    "random_pauli_4q_200t",
    "full_pauli_n4",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("table_rows", []):
        rows.append(row)
    for section in ("rewritten_kraus", "single", "random"):
        for case in summary.get(section, {}).get("cases", []):
            for row in case.get("rows", []):
                rows.append(row)
    return rows


def merge_rows(summaries: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for summary in summaries:
        for row in iter_rows(summary):
            merged[(row["case"], row["basis"])] = row
    return merged


def metric_value(row: dict[str, Any], method: str, metric: str) -> float | None:
    value = row[method][metric]
    if isinstance(value, dict):
        return value.get("mean")
    return value


def fmt_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):,}"
    return f"{numeric:,.{digits}f}"


def fmt_ratio(value: Any) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if numeric >= 100:
        return f"{numeric:.3g}"
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def trend_rows(merged: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in TREND_CASES:
        row = merged[(case, "nam")]
        rows.append(
            {
                "case": case,
                "label": row["label"],
                "terms": int(row["term_count"]),
                "ours_gate_count": float(metric_value(row, "ours", "gate_count")),
                "ours_non_clifford": float(metric_value(row, "ours", "non_clifford")),
                "ours_width": float(metric_value(row, "ours", "width")),
                "qrisp_gate_count": float(metric_value(row, "qrisp", "gate_count")),
                "qrisp_non_clifford": float(metric_value(row, "qrisp", "non_clifford")),
                "qrisp_width": float(metric_value(row, "qrisp", "width")),
                "ratio_gate": float(row["ratio_gate"]),
                "ratio_non_clifford": float(row["ratio_non_clifford"]),
                "samples": int(row.get("samples", 1)),
            }
        )
    return rows


def write_trend_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_trend_md(rows: list[dict[str, Any]], path: Path) -> None:
    body = []
    for row in rows:
        body.append(
            [
                row["label"],
                row["terms"],
                fmt_number(row["ours_gate_count"], 1),
                fmt_number(row["qrisp_gate_count"], 1),
                fmt_ratio(row["ratio_gate"]),
                fmt_number(row["ours_non_clifford"], 1),
                fmt_number(row["qrisp_non_clifford"], 1),
                fmt_ratio(row["ratio_non_clifford"]),
                row["samples"],
            ]
        )
    lines = [
        "# Random Pauli 4q SELECT-Only Trend",
        "",
        "NAM-basis SELECT-only counts. Ratio is Qrisp/Ours.",
        "",
        markdown_table(
            [
                "Example",
                "Terms",
                "Ours gates",
                "Qrisp gates",
                "Gate ratio",
                "Ours NC",
                "Qrisp NC",
                "NC ratio",
                "Samples",
            ],
            body,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_trend_plot(rows: list[dict[str, Any]], svg_path: Path, pdf_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib.pyplot as plt

    terms = [row["terms"] for row in rows]
    ours = [row["ours_gate_count"] for row in rows]
    qrisp = [row["qrisp_gate_count"] for row in rows]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(3.8, 2.6), constrained_layout=True)

    ax.plot(terms, ours, marker="o", linewidth=1.8, label="Ours")
    ax.plot(terms, qrisp, marker="s", linewidth=1.8, label="Qrisp")
    ax.set_xlabel("Number of Pauli terms")
    ax.set_ylabel("NAM SELECT gate count")
    ax.set_yscale("log")
    ax.set_xticks(terms)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def highlight_md(text: str, *, bold: bool = False, blue: bool = False) -> str:
    if bold:
        text = f"**{text}**"
    if blue:
        text = f'<span style="color:blue">{text}</span>'
    return text


def highlight_tex(text: str, *, bold: bool = False, blue: bool = False) -> str:
    if bold:
        text = f"\\textbf{{{text}}}"
    if blue:
        text = f"\\textcolor{{blue}}{{{text}}}"
    return text


def is_min(value: float | None, other: float | None) -> bool:
    if value is None:
        return False
    if other is None:
        return True
    return value <= other


def write_compact_table(merged: dict[tuple[str, str], dict[str, Any]], md_path: Path, tex_path: Path) -> None:
    rows = [merged[(case, "nam")] for case in COMPACT_CASES]
    md_body = []
    for row in rows:
        ours_total = metric_value(row, "ours", "gate_count")
        qrisp_total = metric_value(row, "qrisp", "gate_count")
        ours_nc = metric_value(row, "ours", "non_clifford")
        qrisp_nc = metric_value(row, "qrisp", "non_clifford")
        md_body.append(
            [
                row["label"],
                row["num_qubits"],
                f"{int(row['term_count']):,}",
                highlight_md(
                    fmt_number(ours_total),
                    bold=is_min(ours_total, qrisp_total),
                ),
                highlight_md(
                    fmt_number(ours_nc),
                    blue=is_min(ours_nc, qrisp_nc),
                ),
                fmt_number(metric_value(row, "ours", "width")),
                highlight_md(
                    fmt_number(qrisp_total),
                    bold=is_min(qrisp_total, ours_total),
                ),
                highlight_md(
                    fmt_number(qrisp_nc),
                    blue=is_min(qrisp_nc, ours_nc),
                ),
                fmt_number(metric_value(row, "qrisp", "width")),
                fmt_ratio(row["ratio_gate"]),
                fmt_ratio(row["ratio_non_clifford"]),
            ]
        )
    md_lines = [
        "# Compact NAM SELECT-Only Main Table Candidate",
        "",
        "Random rows are restricted to 4q/20t and 8q/80t; dense 4q trend rows are moved to the figure data.",
        "",
        markdown_table(
            [
                "Example",
                "n",
                "Terms",
                "Ours Total",
                "Ours NC",
                "Ours Width",
                "Qrisp Total",
                "Qrisp NC",
                "Qrisp Width",
                "Ratio Total",
                "Ratio NC",
            ],
            md_body,
        ),
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    tex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{NAM-basis SELECT-only cost comparison. Dense random-Pauli scaling is reported separately in Fig.~\\ref{fig:random-pauli-select-trend}. Ratio is Qrisp/Ours. NC denotes non-Clifford NAM gates.}",
        "\\label{tab:nam-select-only-ours-qrisp-main-compact}",
        "\\begin{tabular}{lrrrrrrrrrr}",
        "\\toprule",
        "& & & \\multicolumn{3}{c}{Ours} & \\multicolumn{3}{c}{Qrisp} & \\multicolumn{2}{c}{Ratio} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}",
        "Example & $n$ & Terms & Total & NC & Width & Total & NC & Width & Total & NC \\\\",
        "\\midrule",
    ]
    for row in rows:
        ours_total = metric_value(row, "ours", "gate_count")
        qrisp_total = metric_value(row, "qrisp", "gate_count")
        ours_nc = metric_value(row, "ours", "non_clifford")
        qrisp_nc = metric_value(row, "qrisp", "non_clifford")
        tex_lines.append(
            f"{tex_escape(row['label'])} & {row['num_qubits']} & {int(row['term_count']):,} & "
            f"{highlight_tex(fmt_number(ours_total), bold=is_min(ours_total, qrisp_total))} & "
            f"{highlight_tex(fmt_number(ours_nc), blue=is_min(ours_nc, qrisp_nc))} & "
            f"{fmt_number(metric_value(row, 'ours', 'width'))} & "
            f"{highlight_tex(fmt_number(qrisp_total), bold=is_min(qrisp_total, ours_total))} & "
            f"{highlight_tex(fmt_number(qrisp_nc), blue=is_min(qrisp_nc, ours_nc))} & "
            f"{fmt_number(metric_value(row, 'qrisp', 'width'))} & "
            f"{fmt_ratio(row['ratio_gate'])} & "
            f"{fmt_ratio(row['ratio_non_clifford'])} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_BASE_SUMMARY)
    parser.add_argument(
        "--extra-summary",
        type=Path,
        action="append",
        default=[],
        help="Additional summary.json files to merge. May be passed more than once.",
    )
    args = parser.parse_args()

    summaries = [load_json(args.base_summary)]
    for extra_summary in (*DEFAULT_EXTRA_SUMMARIES, *args.extra_summary):
        if extra_summary.exists():
            summaries.append(load_json(extra_summary))
    merged = merge_rows(summaries)
    rows = trend_rows(merged)
    write_trend_csv(rows, TREND_CSV)
    write_trend_md(rows, TREND_MD)
    write_trend_plot(rows, TREND_SVG, TREND_PDF)
    write_compact_table(merged, COMPACT_MD, COMPACT_TEX)
    print(f"trend csv: {TREND_CSV}")
    print(f"trend md: {TREND_MD}")
    print(f"trend svg: {TREND_SVG}")
    print(f"trend pdf: {TREND_PDF}")
    print(f"compact md: {COMPACT_MD}")
    print(f"compact tex: {COMPACT_TEX}")


if __name__ == "__main__":
    main()
