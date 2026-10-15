#!/usr/bin/env python3
"""Draft ratio bar chart for representative channel-LCU cases."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPTS = ROOT / "Baseline_scripts"
if str(BASELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BASELINE_SCRIPTS))

import generate_channel_lcu_appendix_full_table as appendix  # type: ignore  # noqa: E402


OUT_SVG = ROOT / "Results" / "channel_lcu_ratio_bars_draft.svg"
OUT_PNG = ROOT / "Results" / "channel_lcu_ratio_bars_draft.png"
OUT_PDF = ROOT / "Results" / "channel_lcu_ratio_bars_draft.pdf"
OUT_CSV = ROOT / "Results" / "channel_lcu_ratio_bars_draft_values.csv"
QUARTZ_RERUN_CSV = ROOT / "Results" / "quartz_rerun_hrw12_randch_n5_20260705_summary.csv"

CASES = [
    ("TFIM_4", "periodic_tfim_n4_dt0p1"),
    ("HRW_12", "hypercube_random_walk_n12"),
    ("RandCh_N5", "random_channel_N5_dim4_N1_10_N2_2_mean"),
    ("GC_2x4", "graph_cluster_2d_grid_2x4"),
    ("RepDQE_d3", "repetition_dqe_d3_eps0p25"),
]

TOOLS = [
    ("ours", "ours", "NAM"),
    ("VOQC", "voqc", "NAM"),
    ("Feynman", "feynman", "NAM"),
    ("WISQ", "wisq", "NAM"),
    ("Quartz", "quartz", "NAM"),
    ("Qiskit O3", "qiskit_opt3", "IBM"),
    ("pytket", "pytket", "IBM"),
    ("BQSKit", "bqskit", "IBM"),
]

COLORS = {
    "ours": "#7300D6",
    "VOQC": "#6F8FAE",
    "Feynman": "#87B8B3",
    "WISQ": "#8DB27F",
    "Quartz": "#A79282",
    "Qiskit O3": "#D79B55",
    "pytket": "#D07378",
    "BQSKit": "#D3BD63",
}

REDUCTION_COLOR = "#D9D9D9"

EXTERNAL_CALLOUTS = {
    "RandCh_N5": (0.20, 0.88),
    "GC_2x4": (0.20, 0.91),
    "RepDQE_d3": (0.20, 0.88),
}

NO_RETURN_STATUSES = {
    "timeout",
    "runner_outer_timeout",
    "no_output",
    "stopped_after_smaller_scale_3600_timeout_or_no_output",
}

NO_NUMERIC_NO_RETURN_STATUSES = NO_RETURN_STATUSES | {
    "reused_existing_3600",
}

ERROR_STATUSES = {
    "failed",
    "environment_permission_error",
    "unavailable",
    "unknown",
}

PENDING_STATUSES = {
    "pending_input_qasm",
    "missing_existing_basis_qasm",
    "unable_to_generate_raw_circuit",
    "blocked_unable_to_generate_raw_circuit",
}


@dataclass(frozen=True)
class PlotDatum:
    case_label: str
    benchmark: str
    tool_label: str
    slot: str
    basis: str
    raw_gates: int | None
    opt_gates: int | None
    ratio: float
    status: str
    display_status: str
    shaded: bool
    source_path: str


def optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percent_formatter(value: float, _pos: int) -> str:
    return f"{100 * value:g}%"


def raw_gate_count(entries: dict[tuple[str, str, str], appendix.Entry], benchmark: str, basis: str) -> int | None:
    raw = entries.get((benchmark, basis, "raw"))
    if raw is None:
        return None
    return appendix.as_int(raw.raw_gates) or appendix.as_int(raw.opt_gates)


def display_status(status: str | None, entry_missing: bool, *, no_numeric_result: bool) -> str:
    if entry_missing:
        return "TO"
    status = status or ""
    if status in NO_RETURN_STATUSES or (no_numeric_result and status in NO_NUMERIC_NO_RETURN_STATUSES):
        return "TO"
    if status in ERROR_STATUSES:
        return "TO"
    if status in PENDING_STATUSES:
        return "TO"
    return ""


def should_shade(status: str | None, entry_missing: bool, *, no_numeric_result: bool) -> bool:
    if entry_missing:
        return True
    if no_numeric_result and (status or "") in NO_NUMERIC_NO_RETURN_STATUSES:
        return True
    return (status or "") in (NO_RETURN_STATUSES | ERROR_STATUSES | PENDING_STATUSES)


def collect_plot_data() -> list[PlotDatum]:
    entries, _, _ = appendix.load_entries()
    data: list[PlotDatum] = []
    for case_label, benchmark in CASES:
        for tool_label, slot, basis in TOOLS:
            entry = entries.get((benchmark, basis, slot))
            missing = entry is None
            raw_gates = raw_gate_count(entries, benchmark, basis)
            opt_gates = appendix.as_int(entry.opt_gates) if entry is not None else None
            no_numeric_result = opt_gates is None
            ratio = appendix.as_float(entry.ratio_vs_raw) if entry is not None else None
            if ratio is None and opt_gates is not None and raw_gates:
                ratio = opt_gates / raw_gates
            status = entry.status if entry is not None else "missing"
            shaded = should_shade(status, missing, no_numeric_result=no_numeric_result) or ratio is None
            data.append(
                PlotDatum(
                    case_label=case_label,
                    benchmark=benchmark,
                    tool_label=tool_label,
                    slot=slot,
                    basis=basis,
                    raw_gates=raw_gates,
                    opt_gates=opt_gates,
                    ratio=ratio if ratio is not None else 1.0,
                    status=status,
                    display_status=display_status(status, missing, no_numeric_result=no_numeric_result),
                    shaded=shaded,
                    source_path=entry.source_path if entry is not None else "",
                )
            )
    return apply_quartz_rerun_overrides(data)


def load_quartz_rerun_overrides() -> dict[tuple[str, str, str], dict[str, str]]:
    if not QUARTZ_RERUN_CSV.exists():
        return {}
    with QUARTZ_RERUN_CSV.open(encoding="utf-8", newline="") as in_file:
        rows = [
            row for row in csv.DictReader(in_file)
            if row.get("row_type") == "plot"
        ]
    return {
        (row["benchmark"], row["tool"], row["basis"]): row
        for row in rows
    }


def apply_quartz_rerun_overrides(data: list[PlotDatum]) -> list[PlotDatum]:
    overrides = load_quartz_rerun_overrides()
    if not overrides:
        return data
    updated: list[PlotDatum] = []
    for row in data:
        override = overrides.get((row.benchmark, row.tool_label, row.basis))
        if override is None:
            updated.append(row)
            continue
        raw_gates = optional_int(override.get("raw_gates")) or row.raw_gates
        opt_gates = optional_int(override.get("opt_gates"))
        ratio = optional_float(override.get("ratio"))
        if ratio is None and raw_gates and opt_gates is not None:
            ratio = opt_gates / raw_gates
        status = override.get("status") or row.status
        has_numeric = opt_gates is not None and ratio is not None
        updated.append(
            PlotDatum(
                case_label=row.case_label,
                benchmark=row.benchmark,
                tool_label=row.tool_label,
                slot=row.slot,
                basis=row.basis,
                raw_gates=raw_gates,
                opt_gates=opt_gates,
                ratio=ratio if ratio is not None else row.ratio,
                status=status,
                display_status="" if has_numeric else row.display_status,
                shaded=False if has_numeric else row.shaded,
                source_path=override.get("source_path") or row.source_path,
            )
        )
    return updated


def write_values_csv(data: list[PlotDatum]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=[
                "case",
                "benchmark",
                "tool",
                "basis",
                "raw_gates",
                "opt_gates",
                "ratio",
                "status",
                "display_status",
                "plotted_at_100_percent",
                "source_path",
            ],
        )
        writer.writeheader()
        for row in data:
            writer.writerow(
                {
                    "case": row.case_label,
                    "benchmark": row.benchmark,
                    "tool": row.tool_label,
                    "basis": row.basis,
                    "raw_gates": row.raw_gates if row.raw_gates is not None else "",
                    "opt_gates": row.opt_gates if row.opt_gates is not None else "",
                    "ratio": f"{row.ratio:.10g}",
                    "status": row.status,
                    "display_status": row.display_status,
                    "plotted_at_100_percent": "yes" if row.shaded and row.opt_gates is None else "no",
                    "source_path": row.source_path,
                }
            )


def baseline_label(entries: dict[tuple[str, str, str], appendix.Entry], benchmark: str) -> str:
    nam = raw_gate_count(entries, benchmark, "NAM")
    ibm = raw_gate_count(entries, benchmark, "IBM")
    def compact(value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 10_000:
            return f"{value / 1_000:.0f}k"
        return f"{value:,}"

    if nam is not None and ibm is not None and nam == ibm:
        return f"100% = {compact(nam)}"
    if nam is not None and ibm is not None:
        return f"100%: N {compact(nam)} / I {compact(ibm)}"
    if nam is not None:
        return f"100%: N {compact(nam)}"
    if ibm is not None:
        return f"100%: I {compact(ibm)}"
    return "100% baseline"


def best_generic_by_case(data: list[PlotDatum]) -> dict[str, PlotDatum]:
    best: dict[str, PlotDatum] = {}
    for row in data:
        if row.slot == "ours" or row.shaded or row.opt_gates is None:
            continue
        old = best.get(row.case_label)
        if old is None or row.ratio < old.ratio:
            best[row.case_label] = row
    return best


def plot(data: list[PlotDatum]) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 17.5,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "legend.fontsize": 10.5,
        }
    )
    entries, _, _ = appendix.load_entries()
    tool_count = len(TOOLS)
    group_gap = 0.42
    bar_width = 0.13
    group_width = tool_count * bar_width
    centers = [idx * (group_width + group_gap) for idx, _ in enumerate(CASES)]

    fig, ax = plt.subplots(figsize=(10.8, 5.2), constrained_layout=True)
    ax.set_ylim(0.0, 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.8, alpha=0.55)

    by_case_tool = {(row.case_label, row.tool_label): row for row in data}
    best_generic = best_generic_by_case(data)
    for case_idx, (case_label, benchmark) in enumerate(CASES):
        center = centers[case_idx]
        left = center - group_width / 2
        ax.hlines(1.0, left - 0.08, left + group_width + 0.08, colors="#222222", linewidth=1.0)
        ax.text(
            center,
            1.12,
            baseline_label(entries, benchmark),
            ha="center",
            va="top",
            fontsize=11.2,
            color="#333333",
        )

        for tool_idx, (tool_label, _, _) in enumerate(TOOLS):
            row = by_case_tool[(case_label, tool_label)]
            x = left + tool_idx * bar_width + bar_width / 2
            width = bar_width * 0.86
            remaining = min(max(row.ratio, 0.0), 1.0)
            reduced = 1.0 - remaining
            if row.shaded:
                ax.bar(
                    x,
                    1.0,
                    width=width,
                    color=COLORS[tool_label],
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.18,
                )
                ax.bar(
                    x,
                    1.0,
                    width=width,
                    color="none",
                    edgecolor=COLORS[tool_label],
                    linewidth=0.0,
                    alpha=0.55,
                    hatch="////",
                )
            else:
                ax.bar(
                    x,
                    remaining,
                    width=width,
                    color=COLORS[tool_label],
                    edgecolor="white",
                    linewidth=0.4,
                    alpha=0.92,
                )
                if reduced > 0:
                    ax.bar(
                        x,
                        reduced,
                        bottom=remaining,
                        width=width,
                        color=REDUCTION_COLOR,
                        edgecolor="white",
                        linewidth=0.4,
                        alpha=0.92,
                    )
            if row.shaded:
                ax.text(
                    x,
                    1.015,
                    row.display_status or "TO",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=9,
                    color="#333333",
                )
            elif row.slot == "ours" or row == best_generic.get(case_label):
                label = f"{100 * row.ratio:.1f}%"
                if row.slot != "ours" and case_label in EXTERNAL_CALLOUTS:
                    dx, label_y = EXTERNAL_CALLOUTS[case_label]
                    ax.annotate(
                        label,
                        xy=(x, row.ratio),
                        xytext=(x + dx, label_y),
                        ha="left",
                        va="center",
                        fontsize=11.6,
                        color=COLORS[tool_label],
                        arrowprops={
                            "arrowstyle": "->",
                            "color": COLORS[tool_label],
                            "lw": 1.0,
                            "shrinkA": 2,
                            "shrinkB": 2,
                        },
                    )
                else:
                    y = min(row.ratio + 0.025, 1.08)
                    ax.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=11.6,
                        color=COLORS[tool_label],
                    )

    ax.set_xticks(centers)
    ax.set_xticklabels([label for label, _ in CASES])
    ax.set_ylabel("Gate count ratio to raw baseline")

    handles = [
        Patch(facecolor=COLORS[tool_label], edgecolor="white", label=f"{tool_label} ({basis})")
        for tool_label, _, basis in TOOLS
    ]
    handles.append(
        Patch(
            facecolor=REDUCTION_COLOR,
            edgecolor="white",
            alpha=0.92,
            label="Reduced portion",
        )
    )
    handles.append(
        Patch(
            facecolor="#BBBBBB",
            edgecolor="#888888",
            hatch="///",
            linewidth=0.0,
            alpha=0.35,
            label="No 3600s return / unavailable, plotted at 100%",
        )
    )
    ax.legend(
        handles=handles,
        ncols=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        columnspacing=0.55,
        handlelength=1.25,
        handletextpad=0.3,
        labelspacing=0.2,
        borderaxespad=0.05,
    )

    xmin = centers[0] - group_width / 2 - 0.18
    xmax = centers[-1] + group_width / 2 + 0.18
    ax.set_xlim(xmin, xmax)
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)


def main() -> None:
    data = collect_plot_data()
    write_values_csv(data)
    plot(data)
    print(f"Wrote {OUT_SVG.relative_to(ROOT)}")
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
