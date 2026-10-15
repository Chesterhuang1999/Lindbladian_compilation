#!/usr/bin/env python3
"""Generate the appendix native-basis benchmark table.

The extraction policy follows Benchmark_guide/appendix_table_extraction_workflow.md.
It writes a machine-readable row dump plus a LaTeX sidewaystable suitable for an
ACM appendix.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"


@dataclass(frozen=True)
class ToolSource:
    tool: str
    path: str
    status_override: str | None = None
    output_override: dict[str, int] | None = None
    time_override: float | None = None


EXAMPLES: dict[str, dict[str, Any]] = {
    "HRW_4": {
        "n": 4,
        "ours": "Baseline_results/hypercube_random_walk_n4_ours_nam/hypercube_random_walk_n4_ours_nam_summary.json",
        "tools": [
            ToolSource("VOQC", "Baseline_results/hypercube_random_walk_n4_canonical_phase1_300s/voqc/voqc_native_summary.json"),
            ToolSource("Feynman", "Baseline_results/hypercube_random_walk_n4_canonical_phase1_300s/feynman/feynman_native_summary.json"),
            ToolSource("WISQ", "Baseline_results/hypercube_random_walk_n4_canonical_phase2_wisq_quartz_300s/wisq/wisq_native_summary.json"),
            ToolSource("Quartz", "Baseline_results/hypercube_random_walk_n4_canonical_phase2_wisq_quartz_300s/quartz/quartz_native_summary.json"),
            ToolSource("pytket", "Baseline_results/hypercube_random_walk_n4_canonical_phase1_300s/pytket/pytket_native_summary.json"),
            ToolSource("BQSKit", "Baseline_results/hypercube_random_walk_n4_canonical_phase1_300s/bqskit/bqskit_native_summary.json"),
        ],
    },
    "HRW_8": {
        "n": 8,
        "ours": "Baseline_results/hypercube_random_walk_n8_ours_nam/hypercube_random_walk_n8_ours_nam_summary.json",
        "tools": [
            ToolSource("VOQC", "Baseline_results/hypercube_random_walk_n8_baselines_3600/voqc/voqc_native_summary.json"),
            ToolSource("Feynman", "Baseline_results/hypercube_random_walk_n8_baselines_3600/feynman/feynman_native_summary.json"),
            ToolSource("WISQ", "Baseline_results/hypercube_random_walk_n8_baselines_3600/wisq/wisq_native_summary.json"),
            ToolSource("Quartz", "Baseline_results/hypercube_random_walk_n8_baselines_3600/quartz/quartz_native_summary.json"),
            ToolSource("pytket", "Baseline_results/hypercube_random_walk_n8_baselines_3600/pytket/pytket_native_summary.json"),
            ToolSource("BQSKit", "Baseline_results/hypercube_random_walk_n8_baselines_3600/bqskit/bqskit_native_summary.json"),
        ],
    },
    "HRW_12": {
        "n": 12,
        "ours": "Baseline_results/hypercube_random_walk_n12_ours_nam/hypercube_random_walk_n12_ours_nam_summary.json",
        "tools": [
            ToolSource("VOQC", "Baseline_results/hypercube_random_walk_n12_baselines_no_quartz_3600/voqc/voqc_native_summary.json"),
            ToolSource("Feynman", "Baseline_results/hypercube_random_walk_n12_baselines_no_quartz_3600/feynman/feynman_native_summary.json"),
            ToolSource("WISQ", "Baseline_results/hypercube_random_walk_n12_baselines_no_quartz_3600/wisq/wisq_native_summary.json"),
            ToolSource("Quartz", "", status_override="not run"),
            ToolSource("pytket", "Baseline_results/hypercube_random_walk_n12_baselines_no_quartz_3600/pytket/pytket_native_summary.json"),
            ToolSource("BQSKit", "Baseline_results/hypercube_random_walk_n12_baselines_no_quartz_3600/bqskit/bqskit_native_summary.json"),
        ],
    },
    "HRW_20": {
        "n": 20,
        "ours": "Baseline_results/hypercube_random_walk_n20_ours_nam/hypercube_random_walk_n20_ours_nam_summary.json",
        "tools": [
            ToolSource("VOQC", "Baseline_results/hypercube_random_walk_n20_baselines_no_quartz_3600/voqc/voqc_native_summary.json"),
            ToolSource("Feynman", "Baseline_results/hypercube_random_walk_n20_baselines_no_quartz_3600/feynman/feynman_native_summary.json"),
            ToolSource("WISQ", "Baseline_results/hypercube_random_walk_n20_baselines_no_quartz_3600/wisq/wisq_native_summary.json"),
            ToolSource("Quartz", "", status_override="not run"),
            ToolSource("pytket", "Baseline_results/hypercube_random_walk_n20_baselines_no_quartz_3600/pytket/pytket_native_summary.json"),
            ToolSource("BQSKit", "Baseline_results/hypercube_random_walk_n20_baselines_no_quartz_3600/bqskit/bqskit_native_summary.json"),
        ],
    },
    "HRW_28": {
        "n": 28,
        "ours": "Baseline_results/hypercube_random_walk_n28_ours_nam/hypercube_random_walk_n28_ours_nam_summary.json",
        "tools": [
            ToolSource("VOQC", "", status_override="not run"),
            ToolSource("Feynman", "", status_override="not run"),
            ToolSource("WISQ", "", status_override="not run"),
            ToolSource("Quartz", "", status_override="not run"),
            ToolSource("pytket", "", status_override="not run"),
            ToolSource("BQSKit", "", status_override="not run"),
        ],
    },
    "TFIM_4": {
        "n": None,
        "ours": "Baseline_results/tfim4_channel_lcu_ours_nam/tfim_n4_channel_lcu_ours_nam_summary.json",
        "unified": "Baseline_results/periodic_tfim_n4_dt0p1_channel_lcu_raw/periodic_tfim_n4_dt0p1_channel_lcu_raw_unified_nam_count_summary.json",
        "tools": [
            ToolSource(
                "VOQC",
                "Baseline_results/periodic_tfim_n4_dt0p1_channel_lcu_raw_1h/voqc/voqc_native_output_nam_lowered.qasm",
                status_override="ok; exact recount",
                output_override={"total": 79109, "non_clifford": 32759},
            ),
            ToolSource("Feynman", "Baseline_results/periodic_tfim_n4_dt0p1_channel_lcu_raw_1h/feynman/feynman_native_summary.json"),
            ToolSource("WISQ", "Baseline_results/periodic_tfim_n4_dt0p1_channel_lcu_raw_1h/wisq/wisq_native_summary.json"),
            ToolSource("Quartz", "Baseline_results/periodic_tfim_n4_dt0p1_channel_lcu_raw_1h/quartz/quartz_native_summary.json"),
            ToolSource("pytket", "Baseline_results/periodic_tfim_n4_dt0p1_channel_lcu_raw/pytket/pytket_native_summary.json"),
            ToolSource("BQSKit", "Baseline_results/periodic_tfim_n4_dt0p1_channel_lcu_raw/bqskit/bqskit_native_summary.json"),
        ],
    },
}


TOOL_BASIS = {
    "VOQC": "NAM",
    "Feynman": "NAM",
    "WISQ": "NAM",
    "Quartz": "NAM",
    "pytket": "IBM",
    "BQSKit": "IBM",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "basis": str(stats["metric_gate_set"]).upper(),
        "total": int(stats["metric_total"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "width": int(stats["num_qubits"]),
    }


def raw_counts_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    data = metric_stats(stats)
    return {"G0": data["total"], "NC0": data["non_clifford"], "width": data["width"]}


def row(
    *,
    example: str,
    tool: str,
    basis: str,
    raw_stats: dict[str, Any],
    out_stats: dict[str, Any] | None,
    time_s: float | None,
    status: str,
    source: str,
) -> dict[str, Any]:
    raw = raw_counts_from_stats(raw_stats)
    out = metric_stats(out_stats) if out_stats is not None else None
    g = out["total"] if out else None
    nc = out["non_clifford"] if out else None
    return {
        "Example": example,
        "Tool": tool,
        "Basis": basis,
        "G0": raw["G0"],
        "G": g,
        "G/G0": None if g is None else g / raw["G0"],
        "NC0": raw["NC0"],
        "NC": nc,
        "NC/NC0": None if nc is None else nc / raw["NC0"],
        "Time (s)": time_s,
        "Status": status,
        "Source": source,
    }


def raw_row(example: str, tool: str, basis: str, stats: dict[str, Any], source: str) -> dict[str, Any]:
    return row(
        example=example,
        tool=tool,
        basis=basis,
        raw_stats=stats,
        out_stats=stats,
        time_s=None,
        status="ok",
        source=source,
    )


def elapsed_seconds(summary: dict[str, Any]) -> float | None:
    if summary.get("process", {}).get("elapsed_seconds") is not None:
        return float(summary["process"]["elapsed_seconds"])
    if summary.get("pytket_stats", {}).get("elapsed_seconds") is not None:
        return float(summary["pytket_stats"]["elapsed_seconds"])
    if summary.get("quartz_stats", {}).get("search", {}).get("elapsed_seconds") is not None:
        return float(summary["quartz_stats"]["search"]["elapsed_seconds"])
    if summary.get("comparison_vs_opt0", {}).get("elapsed_seconds") is not None:
        return float(summary["comparison_vs_opt0"]["elapsed_seconds"])
    if summary.get("comparison", {}).get("elapsed_seconds") is not None:
        return float(summary["comparison"]["elapsed_seconds"])
    return None


def status(summary: dict[str, Any]) -> str:
    if summary.get("status"):
        return str(summary["status"])
    if summary.get("process", {}).get("timed_out"):
        return "timeout"
    if summary.get("quartz_stats", {}).get("search", {}).get("timed_out"):
        return "timeout; checkpoint kept"
    if summary.get("kept_output") is False:
        return "no output"
    if summary.get("output_stats") is not None:
        return "ok"
    return "unknown"


def synthetic_stats(basis: str, total: int, non_clifford: int, width: int) -> dict[str, Any]:
    return {
        "metric_gate_set": basis.lower(),
        "metric_total": total,
        "num_qubits": width,
        "clifford": {"non_clifford": non_clifford},
    }


def qiskit_summary_for(example_info: dict[str, Any], n: int | None) -> tuple[dict[str, Any], str]:
    if n is not None:
        summary = load_json("Baseline_results/hypercube_random_walk_qiskit_ibm_opt3/hypercube_random_walk_qiskit_ibm_opt3_summary.json")
        for result in summary["results"]:
            if result["num_system_qubits"] == n:
                return result, "Baseline_results/hypercube_random_walk_qiskit_ibm_opt3/hypercube_random_walk_qiskit_ibm_opt3_summary.json"
        raise KeyError(f"missing hypercube qiskit result for n={n}")
    summary = load_json(example_info["unified"])
    return {
        "qiskit_opt0_ibm": {"stats": summary["inputs"]["ibm_opt0_stats"]},
        "qiskit_opt3_ibm": {
            "stats": summary["inputs"]["ibm_opt3_stats"],
            "transpile_time_s": None,
        },
    }, example_info["unified"]


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for example, info in EXAMPLES.items():
        ours_summary = load_json(info["ours"])
        raw_nam = ours_summary["raw"]["stats"]
        ours = ours_summary["ours"]["stats"]
        rows.append(raw_row(example, "Raw NAM", "NAM", raw_nam, info["ours"]))
        rows.append(
            row(
                example=example,
                tool="Ours",
                basis="NAM",
                raw_stats=raw_nam,
                out_stats=ours,
                time_s=ours_summary["ours"].get("compile_time_s"),
                status="ok",
                source=info["ours"],
            )
        )

        qiskit_result, qiskit_source = qiskit_summary_for(info, info["n"])
        raw_ibm = qiskit_result["qiskit_opt0_ibm"]["stats"]
        opt3 = qiskit_result["qiskit_opt3_ibm"]["stats"]
        rows.append(raw_row(example, "Raw IBM", "IBM", raw_ibm, qiskit_source))
        rows.append(
            row(
                example=example,
                tool="Qiskit opt3",
                basis="IBM",
                raw_stats=raw_ibm,
                out_stats=opt3,
                time_s=qiskit_result["qiskit_opt3_ibm"].get("transpile_time_s"),
                status="ok",
                source=qiskit_source,
            )
        )

        for tool_source in info["tools"]:
            basis = TOOL_BASIS[tool_source.tool]
            raw_stats = raw_nam if basis == "NAM" else raw_ibm
            if not tool_source.path:
                rows.append(
                    row(
                        example=example,
                        tool=tool_source.tool,
                        basis=basis,
                        raw_stats=raw_stats,
                        out_stats=None,
                        time_s=None,
                        status=tool_source.status_override or "not run",
                        source="--",
                    )
                )
                continue

            if tool_source.output_override is not None:
                raw = metric_stats(raw_stats)
                out_stats = synthetic_stats(
                    basis,
                    tool_source.output_override["total"],
                    tool_source.output_override["non_clifford"],
                    raw["width"],
                )
                rows.append(
                    row(
                        example=example,
                        tool=tool_source.tool,
                        basis=basis,
                        raw_stats=raw_stats,
                        out_stats=out_stats,
                        time_s=tool_source.time_override,
                        status=tool_source.status_override or "ok",
                        source=tool_source.path,
                    )
                )
                continue

            summary = load_json(tool_source.path)
            rows.append(
                row(
                    example=example,
                    tool=tool_source.tool,
                    basis=basis,
                    raw_stats=raw_stats,
                    out_stats=summary.get("output_stats"),
                    time_s=elapsed_seconds(summary),
                    status=tool_source.status_override or status(summary),
                    source=tool_source.path,
                )
            )

    return rows


def fmt_int(value: Any) -> str:
    if value is None:
        return "--"
    return f"{int(value):,}"


def fmt_ratio(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def fmt_percent(value: Any) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):.2f}\\%"


def fmt_tex_int(value: Any) -> str:
    return fmt_int(value).replace(",", r"{,}")


def fmt_count_ratio(count: Any, ratio_value: Any) -> str:
    if count is None or ratio_value is None:
        return "--"
    return f"{fmt_tex_int(count)} ({fmt_percent(ratio_value)})"


def fmt_time(value: Any) -> str:
    if value is None:
        return "--"
    value = float(value)
    if value < 10:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def tex_escape(text: str) -> str:
    repl = {
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
    return "".join(repl.get(ch, ch) for ch in text)


def source_key(path: str, source_map: dict[str, str]) -> str:
    if path == "--":
        return "--"
    if path not in source_map:
        source_map[path] = f"S{len(source_map) + 1}"
    return source_map[path]


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    examples = list(EXAMPLES.keys())
    by_example_tool = {
        (str(item["Example"]), str(item["Tool"])): item
        for item in rows
    }
    lines: list[str] = [r"% Requires: \usepackage{booktabs,graphicx}"]

    def add_table(
        *,
        caption: str,
        label: str,
        raw_tool: str,
        tool_names: list[str],
        tabular_spec: str,
        cmidrules: str,
    ) -> None:
        lines.extend(
            [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\resizebox{\textwidth}{!}{%",
                rf"\begin{{tabular}}{{{tabular_spec}}}",
        r"\toprule",
            ]
        )

        main_header = [r"Example", rf"\multicolumn{{2}}{{c}}{{{raw_tool}}}"]
        main_header.extend(rf"\multicolumn{{3}}{{c}}{{{tool}}}" for tool in tool_names)
        lines.append(" & ".join(main_header) + r" \\")
        lines.append(cmidrules)

        sub_header = ["", "$G$", "$NC$"]
        for _ in tool_names:
            sub_header.extend([r"$G$ (ratio)", r"$NC$ (ratio)", "Time"])
        lines.extend(
            [
                " & ".join(sub_header) + r" \\",
        r"\midrule",
            ]
        )

        for example in examples:
            raw = by_example_tool[(example, raw_tool)]
            line = [tex_escape(example), fmt_tex_int(raw["G"]), fmt_tex_int(raw["NC"])]
            for tool in tool_names:
                item = by_example_tool[(example, tool)]
                line.extend(
                    [
                        fmt_count_ratio(item["G"], item["G/G0"]),
                        fmt_count_ratio(item["NC"], item["NC/NC0"]),
                        fmt_time(item["Time (s)"]),
                    ]
                )
            expected_cols = 3 + 3 * len(tool_names)
            if len(line) != expected_cols:
                raise AssertionError(f"expected {expected_cols} LaTeX columns for {example}, got {len(line)}")
            lines.append(" & ".join(line) + r" \\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}%",
                r"}",
                r"\end{table*}",
                "",
            ]
        )

    add_table(
        caption="Native-basis appendix benchmark results for NAM-basis channel-LCU circuit optimization.",
        label="tab:appendix-native-basis-nam-results",
        raw_tool="Raw NAM",
        tool_names=["Ours", "VOQC", "Feynman", "WISQ", "Quartz"],
        tabular_spec="lrr*{5}{rrr}",
        cmidrules=r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}\cmidrule(lr){16-18}",
    )
    add_table(
        caption="Native-basis appendix benchmark results for IBM-basis channel-LCU circuit optimization.",
        label="tab:appendix-native-basis-ibm-results",
        raw_tool="Raw IBM",
        tool_names=["Qiskit opt3", "pytket", "BQSKit"],
        tabular_spec="lrr*{3}{rrr}",
        cmidrules=r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate(rows: list[dict[str, Any]]) -> None:
    expected = len(EXAMPLES) * 10
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} rows, got {len(rows)}")
    for item in rows:
        if item["Tool"] in {"Raw NAM", "Ours", "VOQC", "Feynman", "WISQ", "Quartz"}:
            assert item["Basis"] == "NAM", item
        if item["Tool"] in {"Raw IBM", "Qiskit opt3", "pytket", "BQSKit"}:
            assert item["Basis"] == "IBM", item
        if item["G"] is None:
            assert item["G/G0"] is None and item["NC"] is None and item["NC/NC0"] is None, item
        else:
            assert item["G/G0"] is not None and item["NC/NC0"] is not None, item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default=RESULTS / "appendix_native_basis_table.tex", type=Path)
    parser.add_argument("--json", default=RESULTS / "appendix_native_basis_table.json", type=Path)
    parser.add_argument("--csv", default=RESULTS / "appendix_native_basis_table.csv", type=Path)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows = collect_rows()
    validate(rows)
    write_json(rows, args.json)
    write_csv(rows, args.csv)
    write_latex(rows, args.tex)
    print(f"wrote {len(rows)} rows")
    print(args.tex)
    print(args.json)
    print(args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
