#!/usr/bin/env python3
"""Summarize TFIM/HRW compilation results without running benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


CASE_LIMIT_S = 18000
FAMILIES = ("tfim", "hrw")
TIMING_DEFINITIONS = {
    "ibm_compile_time_s": "rewrite + circuit build + IBM basis lowering",
    "nam_compile_time_s": "rewrite + circuit build + NAM basis lowering",
    "compile_exclusions": "model construction, imports, provenance, exports, counting, and validation",
    "logical_compile_time_s": "model construction + rewrite + circuit build",
    "program_generation_time_s": "logical compile + IBM lowering + QPY/program export",
    "case_wall_time_s": "model start through all exports/counts/validation; excludes imports/provenance",
    "case_wall_limit_s": "configured limit on case wall time; not a measured compile time",
    "peak_rss_mib": "worker peak resident memory in MiB (2**20 bytes)",
}
PLOT_SERIES = (
    ("ibm_compile_time_s", "IBM compile total", "#222222", "o"),
    ("rewrite_time_s", "Rewrite", "#e69f00", "s"),
    ("circuit_build_time_s", "Circuit build", "#d6604d", "D"),
)


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def result_row(summary, source):
    family = str(summary.get("family", "")).lower()
    n = summary.get("n")
    if family not in FAMILIES or not isinstance(n, int) or n < 1:
        raise ValueError(f"Missing or invalid TFIM/HRW family/n in {source}")
    config = summary.get("configuration", {})
    rss = summary.get("peak_rss_bytes")
    row = {
        "case": f"{family.upper()}-{n}", "family": family,
        "status": summary.get("status", "unknown"),
        "max_steps": config.get("max_steps"),
        "case_wall_limit_s": config.get("case_wall_limit_s"),
        "system_qubits": n, "total_qubits": summary.get("width"),
        "model_time_s": summary.get("model_time_s"),
        "rewrite_time_s": summary.get("rewrite_time_s"),
        "circuit_build_time_s": summary.get("circuit_build_time_s"),
        "logical_compile_time_s": summary.get("logical_compile_time_s"),
        "program_generation_time_s": summary.get("program_generation_time_s"),
        "case_wall_time_s": summary.get("case_wall_time_s"),
        "peak_rss_mib": None if rss is None else rss / 2**20,
        "source_path": str(source),
    }
    for basis in ("ibm", "nam"):
        entry = summary.get("basis_results", {}).get(basis, {})
        stats = entry.get("stats", {})
        row[f"{basis}_compile_time_s"] = entry.get("compile_time_s")
        row[f"{basis}_lowering_time_s"] = entry.get("lowering_time_s")
        row[f"{basis}_gates"] = stats.get("metric_total")
        row[f"{basis}_non_clifford"] = stats.get("clifford", {}).get("non_clifford")
    return row


def collect(roots):
    selected = {}
    inputs = []
    overrides = []
    for root_index, root in enumerate(roots):
        if not root.is_dir():
            raise ValueError(f"Result root is not a directory: {root}")
        for source in sorted(root.glob("*/summary.json")):
            raw = source.read_bytes()
            summary = json.loads(raw)
            if not isinstance(summary, dict):
                raise ValueError(f"Expected a per-case summary object: {source}")
            row = result_row(summary, source)
            key = (row["family"], row["system_qubits"])
            if key in selected:
                previous = selected[key]
                overrides.append({
                    "case": row["case"], "previous_source": previous["source_path"],
                    "previous_status": previous["status"], "selected_source": str(source),
                    "selected_status": row["status"], "selected_root_index": root_index,
                })
            selected[key] = row
            inputs.append({
                "root_index": root_index, "source_path": str(source),
                "sha256": hashlib.sha256(raw).hexdigest(), "case": row["case"],
                "status": row["status"], "configuration": summary.get("configuration", {}),
                "timing_definitions": summary.get("timing_definitions", {}),
                "source_sha256": summary.get("source_sha256", {}),
            })
    rows = sorted(selected.values(), key=lambda row: (FAMILIES.index(row["family"]), row["system_qubits"]))
    if not rows:
        raise ValueError("No direct case subdirectories containing summary.json were found")
    return rows, inputs, overrides


def plot_rows(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    plotted_values = [CASE_LIMIT_S]
    exclusions = []
    for ax, family in zip(axes, FAMILIES):
        family_rows = [row for row in rows if row["family"] == family]
        xs = [row["system_qubits"] for row in family_rows]
        for field, label, color, marker in PLOT_SERIES:
            ys = [row[field] if row["status"] == "ok" and positive(row[field]) else math.nan
                  for row in family_rows]
            plotted_values.extend(value for value in ys if positive(value))
            is_total = field == "ibm_compile_time_s"
            ax.plot(xs, ys, color=color, marker=marker,
                    markersize=8 if is_total else 4,
                    markerfacecolor="none" if is_total else color,
                    markeredgewidth=1.6 if is_total else 0.8,
                    linestyle="--" if is_total else "-",
                    linewidth=1.8 if is_total else 1.6,
                    zorder=4 if is_total else 3, label=label)
        for row in family_rows:
            if row["status"] == "timeout" and positive(row["case_wall_limit_s"]):
                limit = row["case_wall_limit_s"]
                ax.scatter(row["system_qubits"], limit, marker="^", s=65,
                           color="#b2182b", zorder=5)
                plotted_values.append(limit)
            elif row["status"] != "ok":
                exclusions.append({"case": row["case"], "status": row["status"],
                                   "reason": "No completed compile time or recorded timeout limit"})
        ax.axhline(CASE_LIMIT_S, color="#666666", linestyle="--", linewidth=1)
        ax.set(title=family.upper(), xlabel="System qubits", yscale="log")
        ax.set_xticks(xs)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if len(xs) == 1:
            ax.set_xlim(xs[0] - 1, xs[0] + 1)
    axes[0].set_ylabel("Time (seconds, logarithmic scale)")
    axes[0].set_ylim(min(plotted_values) * 0.65, max(plotted_values) * 1.8)
    handles = axes[0].get_legend_handles_labels()[0]
    handles.extend([
        Line2D([], [], color="#b2182b", marker="^", linestyle="none", label="Timeout at case wall limit"),
        Line2D([], [], color="#666666", linestyle="--", label="Configured case limit: 5 h"),
    ])
    fig.suptitle("Full-channel compilation scaling", fontsize=15, y=0.97)
    fig.text(0.5, 0.90, "Single runs; no simulation; no fitted complexity model", ha="center", fontsize=10)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=9)
    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.09, right=0.98, wspace=0.13)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return exclusions


def markdown(rows, outputs, overrides):
    def cell(value):
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = [
        "# TFIM / HRW Compilation Timing Trend", "",
        "Full-channel compilation only; no simulation. Each selected case is one run,",
        "without repeats, error bars, or a fitted complexity model. Later CLI roots",
        "override duplicate family/size cases; every override is recorded in the manifest.", "",
        "IBM/NAM compile time = rewrite + circuit build + lowering to that basis.",
        "Model construction, imports, provenance, exports, counting, and validation are excluded.",
        "Logical compile = model + rewrite + build; program generation additionally includes",
        "IBM lowering and QPY/program export. The CSV preserves these separate timings.", "",
        "The plot uses positive measured times from completed (`ok`) cases only.",
        "Timeout triangles mark recorded case wall limits, not completed compile times.",
        "The dashed 5 h line is the configured case limit; older cases may have no limit.",
        "Missing/failed results are not zero-filled. Per-case rewrite budgets, limits,",
        "gate counts, non-Clifford counts, peak RSS, and source paths are in the CSV.", "",
        "| Case | Status | Total qubits | IBM (s) | Rewrite (s) | Build (s) | NAM (s) | RSS (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    fields = ("case", "status", "total_qubits", "ibm_compile_time_s", "rewrite_time_s",
              "circuit_build_time_s", "nam_compile_time_s", "peak_rss_mib")
    lines.extend("| " + " | ".join(cell(row[field]) for field in fields) + " |" for row in rows)
    lines.extend(["", f"Duplicate overrides: {len(overrides)}.", "",
                  f"![Compilation timing trend]({outputs['png'].name})", "",
                  f"[CSV]({outputs['csv'].name}) | [Manifest]({outputs['manifest'].name})", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True,
                        help="Result roots in priority order; later roots override duplicate cases")
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    roots = [root.resolve() for root in args.roots]
    try:
        rows, inputs, overrides = collect(roots)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    prefix = args.out_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {kind: Path(f"{prefix}.{suffix}") for kind, suffix in
               (("csv", "csv"), ("markdown", "md"), ("png", "png"), ("manifest", "manifest.json"))}
    with outputs["csv"].open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    exclusions = plot_rows(rows, outputs["png"])
    outputs["markdown"].write_text(markdown(rows, outputs, overrides), encoding="utf-8")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full-channel compilation only; no simulation",
        "measurement_policy": "one selected run per family/size; no fitted complexity or error bars",
        "root_precedence": "later CLI roots override earlier roots, regardless of status or timestamp",
        "roots": [str(root) for root in roots], "selected_case_count": len(rows),
        "selected_sources": [row["source_path"] for row in rows],
        "duplicate_overrides": overrides, "inputs": inputs,
        "timing_definitions": TIMING_DEFINITIONS, "configured_case_limit_line_s": CASE_LIMIT_S,
        "plot_policy": "only positive finite times from status=ok; timeout at its recorded case wall limit; gaps otherwise",
        "plot_exclusions": exclusions, "outputs": {kind: str(path) for kind, path in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "duplicate_overrides": len(overrides),
                      "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
