#!/usr/bin/env python3
"""Build selected scalability, width, and runtime tables from appendix data."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Baseline_scripts"))

import generate_channel_lcu_appendix_full_table as appendix  # noqa: E402


SOURCE_CSV = ROOT / "Baseline_results" / "eval_redesign" / "cp5_aggregation" / "rq1_appendix_full_per_tool.csv"
RANDOM_CHANNEL_CSV = ROOT / "Results" / "random_channel_benchmarks_summary.csv"
MATRIXSUM_REFRESH_CSV = ROOT / "Results" / "channel_lcu_matrixsum_phase_refresh_all.csv"

OUT_SCALABILITY_CSV = ROOT / "Results" / "channel_lcu_appendix_scalability_cases.csv"
OUT_SCALABILITY_MD = ROOT / "Results" / "channel_lcu_appendix_scalability_cases.md"
OUT_SCALABILITY_TEX = ROOT / "Results" / "channel_lcu_appendix_scalability_cases.tex"
OUT_WIDTH_CSV = ROOT / "Results" / "channel_lcu_appendix_width_comparison.csv"
OUT_WIDTH_MD = ROOT / "Results" / "channel_lcu_appendix_width_comparison.md"
OUT_WIDTH_TEX = ROOT / "Results" / "channel_lcu_appendix_width_comparison.tex"
OUT_RUNTIME_CSV = ROOT / "Results" / "channel_lcu_appendix_compile_times.csv"
OUT_RUNTIME_MD = ROOT / "Results" / "channel_lcu_appendix_compile_times.md"
OUT_RUNTIME_TEX = ROOT / "Results" / "channel_lcu_appendix_compile_times.tex"

ALL_TOOL_SLOTS = {
    "NAM": ["ours", *appendix.NAM_TOOLS],
    "IBM": ["ours", *appendix.IBM_TOOLS],
}
EXTERNAL_TOOL_SLOTS = {
    "NAM": appendix.NAM_TOOLS,
    "IBM": appendix.IBM_TOOLS,
}
RANDOM_MEAN_NAMES = set(appendix.RANDOM_CHANNEL_PARAMS.values())
RANDOM_PARAMS_BY_MEAN = {value: key for key, value in appendix.RANDOM_CHANNEL_PARAMS.items()}
RANDOM_SEEDS = {0, 1, 2, 3, 4}

WIDTH_BENCHMARKS = [
    "toom_2d_periodic_L2_dt0p1",
    "toom_2d_periodic_L3_dt0p1",
    "hypercube_random_walk_n28",
    "periodic_tfim_n12_dt0p1",
    "graph_cluster_random_bd3_n12_seed7",
    "decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K1_4",
    "repetition_dqe_d3_eps0p25",
    "random_channel_N5_dim4_N1_10_N2_2_mean",
    "random_channel_N9_dim8_N1_20_N2_4_mean",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
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


def mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def fmt_int(value: Any) -> str:
    number = as_int(value)
    if number is None:
        return ""
    return f"{number:,}"


def fmt_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def fmt_time(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    if number >= 1000:
        return f"{number:.0f}"
    if number >= 100:
        return f"{number:.1f}"
    if number >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def fmt_ratio(gates: Any, raw: Any) -> str:
    gate_value = as_float(gates)
    raw_value = as_float(raw)
    if gate_value is None or raw_value in (None, 0):
        return ""
    return f"{100.0 * gate_value / raw_value:.2f}%"


def tex_escape(text: str) -> str:
    text = text.replace("\\_", "_")
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def markdown_table(fields: list[str], rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") or "--" for field in fields) + " |")
    return "\n".join(lines)


def source_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


def load_json(path_text: str) -> dict[str, Any] | None:
    path = source_path(path_text)
    if path is None or path.suffix != ".json":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def width_from_latest_ours_json(data: dict[str, Any], basis: str) -> float | None:
    basis_key = basis.lower()
    if isinstance(data.get("basis_results"), dict):
        stats = data["basis_results"].get(basis_key, {}).get("stats", {})
        return as_float(stats.get("num_qubits"))
    ours = data.get("ours")
    if isinstance(ours, dict):
        stats = ours.get("basis_outputs", {}).get(basis_key, {}).get("stats", {})
        return as_float(stats.get("num_qubits"))
    return None


def raw_width_from_refresh_json(data: dict[str, Any], basis: str) -> float | None:
    raw = data.get("raw")
    if not isinstance(raw, dict):
        return None
    stats = raw.get("basis_outputs", {}).get(basis.lower(), {}).get("stats", {})
    return as_float(stats.get("num_qubits"))


def load_per_tool_rows() -> list[dict[str, str]]:
    return read_csv(SOURCE_CSV) if SOURCE_CSV.exists() else []


def slot_for_row(row: dict[str, str]) -> str:
    method = row.get("method", "")
    if method == "raw_denominator":
        return "raw"
    if method == "ours_full":
        return "ours"
    if method == "generic_optimizer":
        return row.get("optimizer", "").lower()
    return ""


def load_raw_widths() -> dict[tuple[str, str], float]:
    widths: dict[tuple[str, str], float] = {}
    for row in load_per_tool_rows():
        if row.get("method") != "raw_denominator":
            continue
        benchmark = row.get("benchmark_name", "")
        basis = row.get("basis_group", "").upper()
        width = as_float(row.get("raw_width") or row.get("opt_width"))
        if benchmark and basis in {"NAM", "IBM"} and width is not None:
            widths[(benchmark, basis)] = width

    if RANDOM_CHANNEL_CSV.exists():
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in read_csv(RANDOM_CHANNEL_CSV):
            if row.get("category") != "raw":
                continue
            param = row.get("parameter_set", "")
            benchmark = appendix.RANDOM_CHANNEL_PARAMS.get(param)
            basis = row.get("basis", "").upper()
            seed = appendix.seed_from_benchmark(row.get("benchmark", "")) or as_int(row.get("seed"))
            width = as_float(row.get("qubits"))
            if benchmark and basis in {"NAM", "IBM"} and seed in RANDOM_SEEDS and width is not None:
                grouped[(benchmark, basis)].append(width)
        for key, values in grouped.items():
            value = mean(values)
            if value is not None:
                widths[key] = value
    return widths


def load_latest_ours_widths() -> dict[tuple[str, str], float]:
    widths: dict[tuple[str, str], float] = {}
    random_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    if not MATRIXSUM_REFRESH_CSV.exists():
        return widths
    for row in read_csv(MATRIXSUM_REFRESH_CSV):
        benchmark = row.get("benchmark", "")
        basis = row.get("basis", "").upper()
        if basis not in {"NAM", "IBM"}:
            continue
        data = load_json(row.get("source_after", ""))
        if data is None:
            continue
        width = width_from_latest_ours_json(data, basis)
        if width is None:
            continue
        random_mean = ""
        for param, mean_name in appendix.RANDOM_CHANNEL_PARAMS.items():
            if param in benchmark:
                seed = appendix.seed_from_benchmark(benchmark)
                if seed in RANDOM_SEEDS:
                    random_mean = mean_name
                break
        if random_mean:
            random_grouped[(random_mean, basis)].append(width)
        else:
            widths[(benchmark, basis)] = width
    for key, values in random_grouped.items():
        value = mean(values)
        if value is not None:
            widths[key] = value
    return widths


def load_refresh_raw_widths() -> dict[tuple[str, str], float]:
    widths: dict[tuple[str, str], float] = {}
    random_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    if not MATRIXSUM_REFRESH_CSV.exists():
        return widths
    for row in read_csv(MATRIXSUM_REFRESH_CSV):
        benchmark = row.get("benchmark", "")
        basis = row.get("basis", "").upper()
        data = load_json(row.get("source_after", ""))
        if basis not in {"NAM", "IBM"} or data is None:
            continue
        width = raw_width_from_refresh_json(data, basis)
        if width is None:
            continue
        for param, mean_name in appendix.RANDOM_CHANNEL_PARAMS.items():
            if param in benchmark and appendix.seed_from_benchmark(benchmark) in RANDOM_SEEDS:
                random_grouped[(mean_name, basis)].append(width)
                break
    for key, values in random_grouped.items():
        value = mean(values)
        if value is not None:
            widths[key] = value
    return widths


def runtime_for_entry(
    benchmark: str,
    basis: str,
    slot: str,
    entry: appendix.Entry | None,
    runtime_overrides: dict[tuple[str, str, str], str],
) -> str:
    if entry is None:
        return ""
    if slot == "ours":
        return entry.total_time_sec or runtime_overrides.get((benchmark, basis, slot), "")
    return (
        runtime_overrides.get((benchmark, basis, slot), "")
        or entry.optimizer_time_sec
        or entry.total_time_sec
    )


def is_numeric_result(entry: appendix.Entry | None) -> bool:
    return entry is not None and as_int(entry.opt_gates) is not None


def is_strict_under_1h(
    benchmark: str,
    basis: str,
    slot: str,
    entry: appendix.Entry | None,
    runtime_overrides: dict[tuple[str, str, str], str],
) -> bool:
    if not is_numeric_result(entry):
        return False
    if entry.status not in {"ok"}:
        return False
    runtime = as_float(runtime_for_entry(benchmark, basis, slot, entry, runtime_overrides))
    return runtime is not None and runtime <= 3600.0


def compact_cell(
    benchmark: str,
    basis: str,
    slot: str,
    entry: appendix.Entry | None,
    runtime_overrides: dict[tuple[str, str, str], str],
) -> str:
    if entry is None:
        return "--"
    runtime = runtime_for_entry(benchmark, basis, slot, entry, runtime_overrides)
    time_text = fmt_time(runtime)
    if is_numeric_result(entry):
        marker = ""
        if entry.status not in {"ok", "ours_ok"}:
            marker = "*"
        return (time_text or "n/a") + marker
    status = entry.status or "missing"
    status_map = {
        "timeout": "TO",
        "runner_outer_timeout": "TO",
        "no_output": "NO",
        "stopped_after_smaller_scale_3600_timeout_or_no_output": "TO",
        "reused_existing_3600": "TO",
        "skipped_larger_due_to_timeout": "TO",
        "pending_input_qasm": "PI",
        "missing_existing_basis_qasm": "PI",
        "unable_to_generate_raw_circuit": "UG",
        "blocked_unable_to_generate_raw_circuit": "UG",
        "environment_permission_error": "ERR",
        "failed": "ERR",
        "unavailable": "--",
        "unknown": "--",
    }
    label = status_map.get(status, "ERR")
    if time_text and label not in {"--", "PI", "UG"}:
        return f"{label}({time_text})"
    return label


def build_runtime_rows(
    entries: dict[tuple[str, str, str], appendix.Entry],
    family_map: dict[tuple[str, str], str],
    benchmarks: set[str],
) -> list[dict[str, str]]:
    runtime_overrides = appendix.load_runtime_overrides()
    rows: list[dict[str, str]] = []
    for benchmark in sorted(benchmarks, key=appendix.benchmark_sort_key):
        if benchmark in appendix.DROP_BENCHMARKS:
            continue
        for basis in ("NAM", "IBM"):
            slots = ALL_TOOL_SLOTS[basis]
            if not any(entries.get((benchmark, basis, slot)) is not None for slot in slots):
                continue
            row = {
                "example": appendix.LABELS.get(benchmark, appendix.tex_escape(benchmark)),
                "benchmark_name": benchmark,
                "family": family_map.get((benchmark, basis)) or family_map.get((benchmark, "NAM")) or family_map.get((benchmark, "IBM")) or "",
                "basis": basis,
                "ours": "",
                "voqc": "",
                "feynman": "",
                "wisq": "",
                "quartz": "",
                "quizx": "",
                "qiskit_opt3": "",
                "pytket": "",
                "bqskit": "",
            }
            for slot in slots:
                row[slot] = compact_cell(
                    benchmark,
                    basis,
                    slot,
                    entries.get((benchmark, basis, slot)),
                    runtime_overrides,
                )
            rows.append(row)
    return rows


def build_runtime_long_rows(
    entries: dict[tuple[str, str, str], appendix.Entry],
    family_map: dict[tuple[str, str], str],
    benchmarks: set[str],
) -> list[dict[str, str]]:
    runtime_overrides = appendix.load_runtime_overrides()
    rows: list[dict[str, str]] = []
    for benchmark in sorted(benchmarks, key=appendix.benchmark_sort_key):
        if benchmark in appendix.DROP_BENCHMARKS:
            continue
        for basis, slots in ALL_TOOL_SLOTS.items():
            for slot in slots:
                entry = entries.get((benchmark, basis, slot))
                runtime = runtime_for_entry(benchmark, basis, slot, entry, runtime_overrides)
                rows.append(
                    {
                        "example": appendix.LABELS.get(benchmark, appendix.tex_escape(benchmark)),
                        "benchmark_name": benchmark,
                        "family": family_map.get((benchmark, basis)) or family_map.get((benchmark, "NAM")) or family_map.get((benchmark, "IBM")) or "",
                        "basis": basis,
                        "tool": slot,
                        "status": entry.status if entry else "not_run",
                        "time_sec": runtime,
                        "time_cell": compact_cell(benchmark, basis, slot, entry, runtime_overrides),
                        "gates": entry.opt_gates if entry else "",
                        "source_path": entry.source_path if entry else "",
                    }
                )
    return rows


def build_scalability_rows(
    entries: dict[tuple[str, str, str], appendix.Entry],
    family_map: dict[tuple[str, str], str],
    benchmarks: set[str],
) -> list[dict[str, str]]:
    runtime_overrides = appendix.load_runtime_overrides()
    rows: list[dict[str, str]] = []
    for benchmark in sorted(benchmarks, key=appendix.benchmark_sort_key):
        if benchmark in appendix.DROP_BENCHMARKS:
            continue
        ours_entries = [entries.get((benchmark, basis, "ours")) for basis in ("NAM", "IBM")]
        if not any(is_numeric_result(entry) for entry in ours_entries):
            continue
        raw_entries = [entries.get((benchmark, basis, "raw")) for basis in ("NAM", "IBM")]
        raw_numeric = [entry for entry in raw_entries if is_numeric_result(entry)]
        raw_available = bool(raw_numeric)

        successes: list[str] = []
        blocked: list[str] = []
        partial: list[str] = []
        for basis, tools in EXTERNAL_TOOL_SLOTS.items():
            for tool in tools:
                entry = entries.get((benchmark, basis, tool))
                if is_strict_under_1h(benchmark, basis, tool, entry, runtime_overrides):
                    successes.append(f"{basis}:{tool} {fmt_time(runtime_for_entry(benchmark, basis, tool, entry, runtime_overrides))}s")
                elif is_numeric_result(entry):
                    runtime = fmt_time(runtime_for_entry(benchmark, basis, tool, entry, runtime_overrides))
                    label = f"{basis}:{tool}"
                    partial.append(f"{label} {runtime}s/{entry.status}" if runtime else f"{label}/{entry.status}")
                elif entry is not None and entry.status not in {"unknown", "unavailable", ""}:
                    runtime = fmt_time(runtime_for_entry(benchmark, basis, tool, entry, runtime_overrides))
                    label = f"{basis}:{tool}"
                    blocked.append(f"{label} {runtime}s/{entry.status}" if runtime else f"{label}/{entry.status}")

        missing_raw = not raw_available
        near_no_external = raw_available and len(successes) <= 1
        only_ours_basis = []
        for basis in ("NAM", "IBM"):
            ours = entries.get((benchmark, basis, "ours"))
            raw = entries.get((benchmark, basis, "raw"))
            if is_numeric_result(ours) and not is_numeric_result(raw):
                only_ours_basis.append(basis)
        if not (missing_raw or only_ours_basis or near_no_external):
            continue

        raw_text = []
        for basis in ("NAM", "IBM"):
            raw = entries.get((benchmark, basis, "raw"))
            if is_numeric_result(raw):
                raw_text.append(f"{basis} {fmt_int(raw.opt_gates or raw.raw_gates)}")
            elif raw is not None and raw.status not in {"unknown", "unavailable", ""}:
                raw_text.append(f"{basis} {raw.status}")
        ours_text = []
        for basis in ("NAM", "IBM"):
            ours = entries.get((benchmark, basis, "ours"))
            if is_numeric_result(ours):
                raw = entries.get((benchmark, basis, "raw"))
                raw_gates = raw.raw_gates if raw else ours.raw_gates
                ratio = fmt_ratio(ours.opt_gates, raw_gates)
                ratio_text = f" ({ratio})" if ratio else ""
                ours_text.append(f"{basis} {fmt_int(ours.opt_gates)}{ratio_text}")

        if missing_raw:
            interpretation = "ours-only circuit generation path"
        elif only_ours_basis:
            interpretation = "ours covers a basis with missing raw denominator and no external output"
        elif successes:
            interpretation = "raw exists, but at most one external optimizer finishes within 1h"
        else:
            interpretation = "raw exists, but no external optimizer finishes within 1h"

        rows.append(
            {
                "example": appendix.LABELS.get(benchmark, appendix.tex_escape(benchmark)),
                "benchmark_name": benchmark,
                "family": family_map.get((benchmark, "NAM")) or family_map.get((benchmark, "IBM")) or "",
                "raw_availability": "; ".join(raw_text) or "no raw circuit in appendix table",
                "ours_result": "; ".join(ours_text),
                "external_under_1h": "; ".join(successes) or "none",
                "external_other_statuses": "; ".join([*partial, *blocked]) or "--",
                "interpretation": interpretation,
            }
        )
    return rows


def build_width_rows(
    entries: dict[tuple[str, str, str], appendix.Entry],
    family_map: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    raw_widths = load_raw_widths()
    raw_widths.update(load_refresh_raw_widths())
    ours_widths = load_latest_ours_widths()
    rows: list[dict[str, str]] = []
    for benchmark in WIDTH_BENCHMARKS:
        for basis in ("NAM", "IBM"):
            ours = entries.get((benchmark, basis, "ours"))
            raw = entries.get((benchmark, basis, "raw"))
            ours_width = ours_widths.get((benchmark, basis))
            raw_width = raw_widths.get((benchmark, basis))
            if not is_numeric_result(ours) and ours_width is None:
                continue
            if raw_width is None and (raw is None or not raw.status):
                continue
            delta = None
            if raw_width is not None and ours_width is not None:
                delta = ours_width - raw_width
            note = ""
            if raw_width is None:
                note = "raw width unavailable"
            elif raw is not None and not is_numeric_result(raw):
                note = raw.status
            rows.append(
                {
                    "example": appendix.LABELS.get(benchmark, appendix.tex_escape(benchmark)),
                    "benchmark_name": benchmark,
                    "family": family_map.get((benchmark, basis)) or family_map.get((benchmark, "NAM")) or family_map.get((benchmark, "IBM")) or "",
                    "basis": basis,
                    "raw_width": fmt_number(raw_width, 1) or "--",
                    "ours_width": fmt_number(ours_width, 1) or "--",
                    "delta_width": fmt_number(delta, 1) if delta is not None else "--",
                    "raw_gates": fmt_int(raw.opt_gates or raw.raw_gates) if raw and is_numeric_result(raw) else "--",
                    "ours_gates": fmt_int(ours.opt_gates) if ours and is_numeric_result(ours) else "--",
                    "note": note,
                }
            )
    return rows


def write_scalability_md(rows: list[dict[str, str]]) -> None:
    fields = [
        "example",
        "family",
        "raw_availability",
        "ours_result",
        "external_under_1h",
        "interpretation",
    ]
    text = [
        "# Channel-LCU Appendix Scalability Cases",
        "",
        "Selected from `Results/Table1_channelLCU_appendix_full.csv`. External success is counted only when a generic optimizer returns a numeric circuit with status `ok` and measured runtime at or below 3600 s.",
        "",
        markdown_table(fields, rows),
        "",
        "Detailed non-success and partial/checkpoint statuses are retained in the CSV.",
        "",
    ]
    OUT_SCALABILITY_MD.write_text("\n".join(text), encoding="utf-8")


def write_width_md(rows: list[dict[str, str]]) -> None:
    fields = [
        "example",
        "basis",
        "raw_width",
        "ours_width",
        "delta_width",
        "raw_gates",
        "ours_gates",
        "note",
    ]
    text = [
        "# Channel-LCU Appendix Width Comparison",
        "",
        "Widths are qubit counts. Ours widths use the latest Matrixsum phase refresh summaries; random-channel widths are averaged over seeds 0--4.",
        "",
        markdown_table(fields, rows),
        "",
    ]
    OUT_WIDTH_MD.write_text("\n".join(text), encoding="utf-8")


def write_runtime_md(rows: list[dict[str, str]]) -> None:
    nam_fields = ["example", "ours", "voqc", "feynman", "wisq", "quartz", "quizx"]
    ibm_fields = ["example", "ours", "qiskit_opt3", "pytket", "bqskit"]
    nam_rows = [row for row in rows if row["basis"] == "NAM"]
    ibm_rows = [row for row in rows if row["basis"] == "IBM"]
    text = [
        "# Channel-LCU Appendix Compile/Optimizer Times",
        "",
        "Cells are seconds. `TO`, `PI`, `UG`, and `ERR` denote timeout/no-output, pending input, unable to generate raw circuit, and runner/environment error. A trailing `*` marks a numeric partial/checkpoint/non-standard-status result. Raw circuit generation time is not included when the source separates it.",
        "`n/a` means a numeric circuit result exists, but the current artifact set does not record the optimizer runtime.",
        "",
        "## NAM Basis",
        "",
        markdown_table(nam_fields, nam_rows),
        "",
        "## IBM Basis",
        "",
        markdown_table(ibm_fields, ibm_rows),
        "",
    ]
    OUT_RUNTIME_MD.write_text("\n".join(text), encoding="utf-8")


def write_simple_tex(path: Path, caption: str, label: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    align = "l" * len(fields)
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[p]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        f"\\caption{{{tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(tex_escape(field.replace("_", " ")) for field in fields) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row.get(field, "") or "--") for field in fields) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_runtime_tex(rows: list[dict[str, str]]) -> None:
    nam_fields = ["example", "ours", "voqc", "feynman", "wisq", "quartz", "quizx"]
    ibm_fields = ["example", "ours", "qiskit_opt3", "pytket", "bqskit"]
    nam_rows = [row for row in rows if row["basis"] == "NAM"]
    ibm_rows = [row for row in rows if row["basis"] == "IBM"]
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[p]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\caption{Compile/optimizer time for CP1A/CP1B appendix instances. Times are in seconds; raw circuit generation is excluded when the source separates it.}",
        "\\label{tab:channel-lcu-appendix-compile-times}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "\\multicolumn{7}{c}{NAM basis} \\\\",
        "\\midrule",
        " & ".join(tex_escape(field.replace("_", " ")) for field in nam_fields) + " \\\\",
        "\\midrule",
    ]
    for row in nam_rows:
        lines.append(" & ".join(tex_escape(row.get(field, "") or "--") for field in nam_fields) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{4pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{lllll}",
            "\\toprule",
            "\\multicolumn{5}{c}{IBM basis} \\\\",
            "\\midrule",
            " & ".join(tex_escape(field.replace("_", " ")) for field in ibm_fields) + " \\\\",
            "\\midrule",
        ]
    )
    for row in ibm_rows:
        lines.append(" & ".join(tex_escape(row.get(field, "") or "--") for field in ibm_fields) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{2pt}",
            "\\begin{minipage}{\\textwidth}",
            "\\scriptsize",
            "\\emph{Notes.} TO: timeout/no output or skipped after a smaller timeout; PI: pending/missing input; UG: unable to generate raw circuit; ERR: runner/environment error. A trailing * marks a numeric partial/checkpoint/non-standard-status result.",
            "A cell marked n/a has a numeric circuit result but no recorded optimizer runtime in the current artifacts.",
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    OUT_RUNTIME_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    entries, family_map, benchmarks = appendix.load_entries()

    scalability_rows = build_scalability_rows(entries, family_map, benchmarks)
    scalability_fields = [
        "example",
        "benchmark_name",
        "family",
        "raw_availability",
        "ours_result",
        "external_under_1h",
        "external_other_statuses",
        "interpretation",
    ]
    write_csv(OUT_SCALABILITY_CSV, scalability_fields, scalability_rows)
    write_scalability_md(scalability_rows)
    write_simple_tex(
        OUT_SCALABILITY_TEX,
        "Selected scalability cases from the CP1A/CP1B appendix table.",
        "tab:channel-lcu-appendix-scalability",
        ["example", "family", "raw_availability", "ours_result", "external_under_1h", "interpretation"],
        scalability_rows,
    )

    width_rows = build_width_rows(entries, family_map)
    width_fields = [
        "example",
        "benchmark_name",
        "family",
        "basis",
        "raw_width",
        "ours_width",
        "delta_width",
        "raw_gates",
        "ours_gates",
        "note",
    ]
    write_csv(OUT_WIDTH_CSV, width_fields, width_rows)
    write_width_md(width_rows)
    write_simple_tex(
        OUT_WIDTH_TEX,
        "Selected raw-vs-ours circuit width comparison for CP1A/CP1B appendix instances.",
        "tab:channel-lcu-appendix-width",
        ["example", "basis", "raw_width", "ours_width", "delta_width", "raw_gates", "ours_gates", "note"],
        width_rows,
    )

    runtime_rows = build_runtime_rows(entries, family_map, benchmarks)
    runtime_fields = [
        "example",
        "benchmark_name",
        "family",
        "basis",
        "ours",
        "voqc",
        "feynman",
        "wisq",
        "quartz",
        "quizx",
        "qiskit_opt3",
        "pytket",
        "bqskit",
    ]
    write_csv(OUT_RUNTIME_CSV, runtime_fields, runtime_rows)
    write_runtime_md(runtime_rows)
    write_runtime_tex(runtime_rows)

    runtime_long_rows = build_runtime_long_rows(entries, family_map, benchmarks)
    write_csv(
        ROOT / "Results" / "channel_lcu_appendix_compile_times_long.csv",
        ["example", "benchmark_name", "family", "basis", "tool", "status", "time_sec", "time_cell", "gates", "source_path"],
        runtime_long_rows,
    )

    print(f"wrote {len(scalability_rows)} scalability rows to {OUT_SCALABILITY_CSV}")
    print(f"wrote {len(width_rows)} width rows to {OUT_WIDTH_CSV}")
    print(f"wrote {len(runtime_rows)} runtime rows to {OUT_RUNTIME_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
