#!/usr/bin/env python3
"""Build a combined raw-input vs ours-input external-optimizer ratio table."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
APPENDIX_CSV = ROOT / "Results" / "Table1_channelLCU_appendix_full.csv"
SECOND_STAGE_CSV = ROOT / "Results" / "ours_second_stage_external_optimizers_merged.csv"
OUT_MD = ROOT / "Results" / "external_optimizer_reduction_two_inputs.md"
OUT_TEX = ROOT / "Results" / "external_optimizer_reduction_two_inputs.tex"

SELECTED = [
    ("TFIM\\_4", "periodic_tfim_n4_dt0p1"),
    ("HRW\\_8", "hypercube_random_walk_n8"),
    ("RandCh\\_N5", "random_channel_N5_dim4_N1_10_N2_2_mean"),
    ("RepDQE\\_d3", "repetition_dqe_d3_eps0p25"),
]

RAW_TOOLS = [
    ("VOQC", "NAM", "nam_voqc"),
    ("Feynman", "NAM", "nam_feynman"),
    ("WISQ", "NAM", "nam_wisq"),
    ("Quartz", "NAM", "nam_quartz"),
    ("QuiZX", "NAM", "nam_quizx"),
    ("Qiskit opt3", "IBM", "ibm_qiskit_opt3"),
    ("pytket", "IBM", "ibm_pytket"),
    ("BQSKit", "IBM", "ibm_bqskit"),
]

OURS_TOOLS = [
    ("VOQC", "NAM", "voqc"),
    ("WISQ", "NAM", "wisq"),
    ("Quartz", "NAM", "quartz"),
    ("pytket", "IBM", "pytket"),
    ("BQSKit", "IBM", "bqskit"),
]

TOOL_CANDIDATES = [tool for tool, _basis, _column in RAW_TOOLS]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def parse_int(value: str) -> int | None:
    if not value:
        return None
    match = re.match(r"\s*(-?[\d,]+)", value)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def has_dagger(value: str) -> bool:
    return "dagger" in (value or "")


def reduction(input_gates: float, output_gates: float) -> tuple[float, float]:
    delta = input_gates - output_gates
    return delta, delta / input_gates if input_gates else float("nan")


def fmt_number(value: float | None, *, latex: bool = False, dagger: bool = False) -> str:
    if value is None:
        return "--"
    if abs(value - round(value)) < 1e-9:
        text = f"{int(round(value)):,}"
    else:
        text = f"{value:,.1f}"
    if latex and value < 0:
        text = f"${text}$"
    if dagger:
        text += "$^\\dagger$" if latex else "†"
    return text


def fmt_ratio(value: float | None, *, latex: bool = False, dagger: bool = False) -> str:
    if value is None:
        return "--"
    percent = r"\%" if latex else "%"
    text = f"{100 * value:.2f}{percent}"
    if latex and value < 0:
        text = f"${text}$"
    if dagger:
        text += "$^\\dagger$" if latex else "†"
    return text


def md_label(label: str) -> str:
    return label.replace("\\_", "_")


def latex_escape_tool(tool: str) -> str:
    return tool.replace("_", "\\_")


def raw_results() -> dict[tuple[str, str], tuple[float, float, bool]]:
    rows = {row["benchmark_name"]: row for row in read_csv(APPENDIX_CSV)}
    out: dict[tuple[str, str], tuple[float, float, bool]] = {}
    for _label, benchmark in SELECTED:
        row = rows[benchmark]
        for tool, basis, column in RAW_TOOLS:
            input_column = "nam_raw" if basis == "NAM" else "ibm_raw"
            input_gates = parse_int(row[input_column])
            output_gates = parse_int(row[column])
            if input_gates is None or output_gates is None:
                continue
            delta, ratio = reduction(float(input_gates), float(output_gates))
            out[(tool, benchmark)] = (delta, ratio, has_dagger(row[column]))
    return out


def normalize_random_mean(benchmark: str) -> str:
    return benchmark.replace("_mean", "_seed")


def second_stage_results() -> dict[tuple[str, str], tuple[float, float, bool]]:
    rows = read_csv(SECOND_STAGE_CSV)
    exact: dict[tuple[str, str], tuple[float, float, bool]] = {}
    random_groups: dict[tuple[str, str], list[tuple[float, float, bool]]] = defaultdict(list)

    for row in rows:
        input_gates = parse_int(row.get("input_gate_count", ""))
        output_gates = parse_int(row.get("output_gate_count", ""))
        if input_gates is None or output_gates is None:
            continue
        delta, ratio = reduction(float(input_gates), float(output_gates))
        tool = {
            "voqc": "VOQC",
            "wisq": "WISQ",
            "quartz": "Quartz",
            "qiskit_opt3": "Qiskit opt3",
            "pytket": "pytket",
            "bqskit": "BQSKit",
        }.get(row["tool"])
        if tool is None:
            continue
        dagger = row.get("status") == "timeout_checkpoint"
        benchmark = row["benchmark"]
        exact[(tool, benchmark)] = (delta, ratio, dagger)
        if benchmark.startswith("random_channel_N5_dim4_N1_10_N2_2_seed"):
            random_groups[(tool, "random_channel_N5_dim4_N1_10_N2_2_mean")].append(
                (delta, ratio, dagger)
            )

    for key, values in random_groups.items():
        exact[key] = (
            mean(value[0] for value in values),
            mean(value[1] for value in values),
            any(value[2] for value in values),
        )
    return exact


def paired_tools(
    raw: dict[tuple[str, str], tuple[float, float, bool]],
    ours: dict[tuple[str, str], tuple[float, float, bool]],
) -> list[str]:
    tools: list[str] = []
    for tool in TOOL_CANDIDATES:
        has_paired_output = all(
            (tool, benchmark) in raw and (tool, benchmark) in ours
            for _label, benchmark in SELECTED
        )
        if has_paired_output:
            tools.append(tool)
    return tools


def write_markdown(
    raw: dict[tuple[str, str], tuple[float, float, bool]],
    ours: dict[tuple[str, str], tuple[float, float, bool]],
    tools: list[str],
) -> None:
    lines = [
        "# External Optimizer Gate Reduction: Raw Input vs Ours Input",
        "",
        "Selected instances are those with at least four external-optimizer outputs for both raw-circuit input and ours-circuit input.",
        "Cells report reduction ratio, defined as `(input gates - output gates) / input gates`.",
        "",
    ]

    headers = ["Example"]
    for tool in tools:
        headers.extend([f"{tool} raw", f"{tool} ours"])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for label, benchmark in SELECTED:
        row = [md_label(label)]
        for tool in tools:
            raw_value = raw.get((tool, benchmark))
            ours_value = ours.get((tool, benchmark))
            row.append("--" if raw_value is None else fmt_ratio(raw_value[1], dagger=raw_value[2]))
            row.append("--" if ours_value is None else fmt_ratio(ours_value[1], dagger=ours_value[2]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.extend(
        [
            "Notes:",
            "- Columns retain only tools with both raw-input and ours-input outputs for every displayed instance.",
            "- Raw-input denominators use each retained tool's native raw basis.",
            "- Ours-input denominators use the corresponding ours NAM or IBM circuit from the second-stage optimizer table.",
            "- RandCh_N5 values are seed averages. For ours-input results, WISQ is averaged over the four seeds with outputs.",
            "- `†` marks checkpoint/partial/reused results inherited from the source tables.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_latex(
    raw: dict[tuple[str, str], tuple[float, float, bool]],
    ours: dict[tuple[str, str], tuple[float, float, bool]],
    tools: list[str],
) -> None:
    lines = [
        "%% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\caption{Reduction ratios achieved by external optimizers when given either raw channel-LCU circuits or the in-house optimized circuits as input. Each instance has at least four external-optimizer outputs in both settings.}",
        "\\label{tab:external-optimizer-reduction-two-inputs}",
    ]

    colspec = "l" + "r" * (2 * len(tools))
    lines.extend(
        [
            "\\resizebox{\\textwidth}{!}{%",
            f"\\begin{{tabular}}{{{colspec}}}",
            "\\toprule",
        ]
    )
    header = ["Example"]
    cmidrules = []
    col = 2
    for tool in tools:
        header.append(f"\\multicolumn{{2}}{{c}}{{{latex_escape_tool(tool)}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col}-{col + 1}}}")
        col += 2
    lines.append(" & ".join(header) + r" \\")
    lines.append(" ".join(cmidrules))
    subheader = [""]
    for _tool in tools:
        subheader.extend(["Raw input", "Ours input"])
    lines.append(" & ".join(subheader) + r" \\")
    lines.append("\\midrule")
    for label, benchmark in SELECTED:
        row = [label]
        for tool in tools:
            raw_value = raw.get((tool, benchmark))
            ours_value = ours.get((tool, benchmark))
            row.append("--" if raw_value is None else fmt_ratio(raw_value[1], latex=True, dagger=raw_value[2]))
            row.append("--" if ours_value is None else fmt_ratio(ours_value[1], latex=True, dagger=ours_value[2]))
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{2pt}",
            "\\begin{minipage}{\\textwidth}",
            "\\scriptsize",
            "\\emph{Notes.} Cells report $(G_{\\mathrm{in}}-G_{\\mathrm{out}})/G_{\\mathrm{in}}$. Columns retain only tools with both raw-input and ours-input outputs for every displayed instance. Raw-input denominators use each retained tool's native raw basis, while ours-input denominators use the corresponding in-house NAM or IBM circuit. RandCh\\_N5 values are seed averages; the ours-input WISQ average uses the four seeds with outputs. $^\\dagger$ marks checkpoint/partial/reused results inherited from the source tables.",
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    raw = raw_results()
    ours = second_stage_results()
    tools = paired_tools(raw, ours)
    write_markdown(raw, ours, tools)
    write_latex(raw, ours, tools)
    print(OUT_MD)
    print(OUT_TEX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
