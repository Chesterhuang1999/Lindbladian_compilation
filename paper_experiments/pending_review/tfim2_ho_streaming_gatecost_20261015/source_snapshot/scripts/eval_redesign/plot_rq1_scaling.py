"""Plot measured RQ1 resources, separating compilation platforms and failures."""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-rq1-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Results"


def main():
    rows = list(csv.DictReader((OUTPUT / "rq1_scaling_20261014.csv").open()))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(6.9, 5.5), sharex="col")
    series = [("Ours", "local", "#0072B2", "o", "-", "Ours (M4)"),
              ("Raw", "local", "#D55E00", "s", "--", "Raw (M4)"),
              ("Raw", "remote", "#8A4F9E", "D", "None", "Raw (EPYC)")]
    metrics = [("G", "NAM gates $G$"), ("Lambda_G", "$\\Lambda G$"),
               ("reported_compile_s", "Compilation time (s)")]
    handles, labels = [], []
    for col, family in enumerate(["TFIM", "HRW"]):
        for row_index, (key, ylabel) in enumerate(metrics):
            ax = axes[row_index, col]
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", axis="y", color="#d7d7d7", linewidth=.6)
            for method, host, color, marker, style, label in series:
                points = sorted([r for r in rows if r["family"] == family and
                    r["method"] == method and r["host"] == host and r["status"] == "ok"],
                    key=lambda r: int(r["n"]))
                assert all(r[key] and float(r[key]) > 0 for r in points)
                line, = ax.plot([int(r["n"]) for r in points], [float(r[key]) for r in points],
                    color=color, marker=marker, linestyle=style, linewidth=1.3,
                    markersize=3.8, markeredgewidth=.7, label=label)
                if (col, row_index) == (0, 0):
                    handles.append(line)
                    labels.append(label)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(family, pad=6, fontweight="semibold")
            if family == "HRW":
                ax.set_xscale("log", base=2)
                ax.set_xlim(3.5, 1700)
                ticks = [4, 16, 64, 256, 1440]
            else:
                ax.set_xlim(1, 64)
                ticks = [4, 12, 20, 40, 60]
            ax.set_xticks(ticks, labels=[str(t) for t in ticks])
            ax.xaxis.set_minor_formatter(NullFormatter())
            if row_index == 2:
                ax.set_xlabel("System qubits $n$")
                endpoint = max([r for r in rows if r["family"] == family and r["method"] == "Ours"],
                               key=lambda r: int(r["n"]))
                ax.annotate(f"{float(endpoint[key]) / 3600:.2f} h",
                    xy=(int(endpoint["n"]), float(endpoint[key])), xytext=(-30, -16),
                    textcoords="offset points", color="#0072B2", fontsize=8,
                    ha="right", arrowprops={"arrowstyle": "-", "color": "#0072B2", "lw": .6})
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.52, .993), handlelength=2.6, columnspacing=1.5)
    fig.subplots_adjust(left=.115, right=.985, bottom=.175, top=.885, hspace=.24, wspace=.25)
    fig.text(.115, .062, "Raw (M4): memory limit at 54; timeout at 60.\nRaw (EPYC): 60 deferred.",
             fontsize=7.4, va="top", linespacing=1.6)
    fig.text(.585, .062, "Raw (M4): timeout at 512, 1024, 1440.\nRaw (EPYC): 1024, 1440 deferred.",
             fontsize=7.4, va="top", linespacing=1.6)
    for suffix in ["pdf", "png"]:
        path = OUTPUT / ("rq1_scaling_20261014." + suffix)
        fig.savefig(path, dpi=220, metadata={"Title": "RQ1 TFIM and HRW scaling"})
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
