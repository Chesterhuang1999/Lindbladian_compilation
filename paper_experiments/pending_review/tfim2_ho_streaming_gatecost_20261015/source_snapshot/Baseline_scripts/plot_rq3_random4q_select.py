#!/usr/bin/env python3
"""Plot the measured current-default four-qubit SELECT density sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "Baseline_results/rq3_random4q_20261014"


def plot_log(directory):
    """Show total gates on a log axis and width on a linear right axis."""
    data = json.loads((directory / "summary.json").read_text())
    assert data["passed"] and len(data["rows"]) == 60
    groups = data["aggregate"]
    assert [g["terms"] for g in groups] == [20, 64, 100, 129, 200, 256]
    assert all(g["samples"] == 10 for g in groups)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.linewidth": 1.1, "xtick.major.width": 1.1,
                         "ytick.major.width": 1.1,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, left = plt.subplots(figsize=(6.4, 4.9))
    right = left.twinx()
    colors = {"G": {"qrisp": "#7961a9", "ours": "#187f7f"},
              "W": {"qrisp": "#d68527", "ours": "#2878b5"}}
    metrics = {"G": {"linestyle": "-", "marker": "o", "label": "gates"},
               "W": {"linestyle": ":", "marker": "s", "label": "width"}}
    terms = [g["terms"] for g in groups]
    for method in ("qrisp", "ours"):
        for metric, style in metrics.items():
            color = colors[metric][method]
            stats = [g["basis"]["nam"][method][metric] for g in groups]
            axis = right if metric == "W" else left
            mean = [stat["mean"] for stat in stats]
            std = [stat["population_std"] for stat in stats]
            assert all(value > 0 for value in mean)
            axis.errorbar(terms, mean, yerr=std, color=color,
                linestyle=style["linestyle"], marker=style["marker"],
                markerfacecolor="white" if metric == "W" else color,
                linewidth=2.5 if metric == "W" else 2.8,
                markersize=7, markeredgewidth=1.5,
                capsize=3.5, capthick=1.4, elinewidth=1.3,
                zorder=3 if metric == "W" else 2)
    left.set_yscale("log")
    left.set_ylim(6, 16000)
    left.yaxis.set_major_locator(LogLocator(base=10))
    left.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    left.set_ylabel("NAM gate count (log scale)")
    left.set_xlabel("Number of Pauli terms")
    left.set_xticks(terms)
    left.set_xlim(10, 266)
    left.grid(axis="y", which="major", color="#dedede", linewidth=.65)
    left.set_axisbelow(True)
    right.set_ylabel("Circuit width")
    right.set_ylim(0, 30)
    right.yaxis.set_major_locator(MultipleLocator(6))
    left.spines["top"].set_visible(True)
    right.spines["right"].set_visible(True)
    right.patch.set_visible(False)
    handles = [Line2D([0], [0], color=colors[metric][method],
                linestyle=style["linestyle"], marker=style["marker"],
                markerfacecolor="white" if metric == "W" else colors[metric][method],
                linewidth=2.5 if metric == "W" else 2.8,
                markersize=7, markeredgewidth=1.5,
                label=f"{method.capitalize()} {style['label']}")
               for metric, style in metrics.items() for method in ("qrisp", "ours")]
    left.set_title("Random 4q SELECT", fontsize=17, pad=10)
    right.legend(handles=handles, loc="lower left", bbox_to_anchor=(.02, .025),
                 ncol=2, frameon=False, fontsize=11.5, columnspacing=1.0,
                 handlelength=1.8, labelspacing=.5, borderaxespad=.2)
    fig.tight_layout(pad=.6)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(directory / f"random4q_select_density_log.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(directory / "random4q_select_density_log.pdf")


def main(directory):
    data = json.loads((directory / "summary.json").read_text())
    assert data["passed"] and len(data["rows"]) == 60
    groups = data["aggregate"]
    assert [g["terms"] for g in groups] == [20, 64, 100, 129, 200, 256]
    assert all(g["samples"] == 10 for g in groups)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    terms = [g["terms"] for g in groups]
    styles = {"ours": {"color": "#187f7f", "marker": "o", "label": "Ours"},
              "qrisp": {"color": "#7961a9", "marker": "s", "label": "Qrisp"}}
    for method, style in styles.items():
        stats = [g["basis"]["nam"][method]["G"] for g in groups]
        ax.errorbar(terms, [v["mean"] for v in stats],
                    yerr=[v["population_std"] for v in stats], capsize=3,
                    linewidth=1.8, markersize=5, **style)
    ax.set_xlabel("Number of Pauli terms")
    ax.set_ylabel("NAM gate count")
    ax.set_xticks(terms)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.text(0.5, 0.015, "SELECT only · 4 system qubits · 10 samples per support size\nError bars: population standard deviation",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.095, 1, 1))
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(directory / f"random4q_select_density.{suffix}", dpi=220, bbox_inches="tight")
    fields = ["terms", "samples", "basis", "method", "G_mean", "G_std", "NC_mean", "NC_std", "W_mean", "W_std", "qrisp_over_ours_G"]
    with (directory / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            for basis in ("nam", "ibm"):
                for method in ("qrisp", "ours"):
                    metrics = group["basis"][basis][method]
                    row = {"terms": group["terms"], "samples": group["samples"], "basis": basis, "method": method,
                           "qrisp_over_ours_G": group["basis"][basis]["qrisp_over_ours_ratio_of_means"]["G"]}
                    for metric in ("G", "NC", "W"):
                        row[metric + "_mean"] = metrics[metric]["mean"]
                        row[metric + "_std"] = metrics[metric]["population_std"]
                    writer.writerow(row)
    with (directory / "samples.csv").open("w", newline="") as handle:
        fields = ["terms", "sample", "seed", "basis", "method", "G", "NC", "W", "CX"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in data["rows"]:
            for basis in ("nam", "ibm"):
                for method in ("qrisp", "ours"):
                    writer.writerow({**{k: sample[k] for k in ("terms", "sample", "seed")},
                        "basis": basis, "method": method,
                        **{k: sample[method]["basis"][basis][k] for k in ("G", "NC", "W", "CX")}})
    print(directory / "random4q_select_density.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--log", action="store_true", help="Log total gates plus linear width, keeping the original plot")
    args = parser.parse_args()
    (plot_log if args.log else main)(args.directory)
