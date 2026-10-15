#!/usr/bin/env python3
"""Build a before/after table for the Matrixsum phase refresh.

``before`` is the previous 2026-07-05 structure=opt refresh overlay.
``after`` is the 2026-07-06 Matrixsum phase refresh overlay.
Rows follow the current appendix Table 1 benchmark set; random-channel rows are
reported as seed-0..4 means, matching the appendix table policy.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BEFORE_CSV = ROOT / "Results" / "channel_lcu_structure_opt_refresh_all.csv"
AFTER_CSV = ROOT / "Results" / "channel_lcu_matrixsum_phase_refresh_all.csv"
APPENDIX_CSV = ROOT / "Results" / "Table1_channelLCU_appendix_full.csv"
OUT_CSV = ROOT / "Results" / "channel_lcu_matrixsum_phase_before_after.csv"
OUT_MD = ROOT / "Results" / "channel_lcu_matrixsum_phase_before_after.md"
OUT_TEX = ROOT / "Results" / "channel_lcu_matrixsum_phase_before_after.tex"

RANDOM_CHANNEL_PARAMS = {
    "random_channel_N5_dim4_N1_10_N2_2_mean": "random_channel_N5_dim4_N1_10_N2_2",
    "random_channel_N9_dim8_N1_20_N2_4_mean": "random_channel_N9_dim8_N1_20_N2_4",
}
RANDOM_CHANNEL_SEEDS = (0, 1, 2, 3, 4)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    value_f = as_float(value)
    if value_f is None:
        return None
    return int(round(value_f))


def fmt_int(value: Any) -> str:
    value_i = as_int(value)
    return "" if value_i is None else f"{value_i:,}"


def fmt_signed_int(value: Any) -> str:
    value_i = as_int(value)
    if value_i is None:
        return ""
    return f"{value_i:+,}"


def fmt_pct(value: Any) -> str:
    value_f = as_float(value)
    return "" if value_f is None else f"{100.0 * value_f:+.2f}%"


def fmt_time(value: Any) -> str:
    value_f = as_float(value)
    if value_f is None:
        return ""
    if value_f >= 100:
        return f"{value_f:.1f}"
    if value_f >= 10:
        return f"{value_f:.2f}"
    return f"{value_f:.3f}"


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        benchmark = row.get("benchmark", "")
        basis = row.get("basis", "").upper()
        if benchmark and basis in {"NAM", "IBM"}:
            indexed[(benchmark, basis)] = row
    return indexed


def numeric_row(row: dict[str, str], prefix: str) -> dict[str, float | None]:
    return {
        "gate": as_float(row.get(f"{prefix}_gate_count")),
        "nc": as_float(row.get(f"{prefix}_non_clifford")),
        "time": as_float(row.get(f"{prefix}_compile_time_s")),
    }


def mean(values: list[float | None]) -> float | None:
    actual = [value for value in values if value is not None]
    if not actual:
        return None
    return sum(actual) / len(actual)


def aggregate_random(
    indexed: dict[tuple[str, str], dict[str, str]],
    param_prefix: str,
    basis: str,
    prefix: str,
) -> tuple[dict[str, float | None], int]:
    values: dict[str, list[float | None]] = {"gate": [], "nc": [], "time": []}
    count = 0
    for seed in RANDOM_CHANNEL_SEEDS:
        row = indexed.get((f"{param_prefix}_seed{seed}", basis))
        if row is None:
            continue
        count += 1
        item = numeric_row(row, prefix)
        for key in values:
            values[key].append(item[key])
    return {key: mean(bucket) for key, bucket in values.items()}, count


def appendix_examples() -> list[dict[str, str]]:
    rows = read_csv(APPENDIX_CSV)
    return [
        {
            "example": row["example"],
            "benchmark": row["benchmark_name"],
            "family": row["family"],
        }
        for row in rows
    ]


def build_rows() -> list[dict[str, Any]]:
    before_index = index_rows(read_csv(BEFORE_CSV))
    after_index = index_rows(read_csv(AFTER_CSV))
    rows: list[dict[str, Any]] = []
    for example in appendix_examples():
        benchmark = example["benchmark"]
        for basis in ("NAM", "IBM"):
            seed_count_before = ""
            seed_count_after = ""
            if benchmark in RANDOM_CHANNEL_PARAMS:
                prefix = RANDOM_CHANNEL_PARAMS[benchmark]
                before, before_count = aggregate_random(before_index, prefix, basis, "after")
                after, after_count = aggregate_random(after_index, prefix, basis, "after")
                seed_count_before = str(before_count)
                seed_count_after = str(after_count)
            else:
                before_row = before_index.get((benchmark, basis))
                after_row = after_index.get((benchmark, basis))
                before = numeric_row(before_row, "after") if before_row else {"gate": None, "nc": None, "time": None}
                after = numeric_row(after_row, "after") if after_row else {"gate": None, "nc": None, "time": None}

            before_gate = before["gate"]
            after_gate = after["gate"]
            before_nc = before["nc"]
            after_nc = after["nc"]
            before_time = before["time"]
            after_time = after["time"]

            delta_gate = None if before_gate is None or after_gate is None else after_gate - before_gate
            delta_gate_pct = None if before_gate in (None, 0) or delta_gate is None else delta_gate / before_gate
            delta_nc = None if before_nc is None or after_nc is None else after_nc - before_nc
            delta_nc_pct = None if before_nc in (None, 0) or delta_nc is None else delta_nc / before_nc
            delta_time = None if before_time is None or after_time is None else after_time - before_time

            rows.append(
                {
                    "example": example["example"],
                    "benchmark": benchmark,
                    "family": example["family"],
                    "basis": basis,
                    "before_gate_count": before_gate,
                    "after_gate_count": after_gate,
                    "delta_gate_count": delta_gate,
                    "delta_gate_pct": delta_gate_pct,
                    "before_non_clifford": before_nc,
                    "after_non_clifford": after_nc,
                    "delta_non_clifford": delta_nc,
                    "delta_non_clifford_pct": delta_nc_pct,
                    "before_compile_time_s": before_time,
                    "after_compile_time_s": after_time,
                    "delta_compile_time_s": delta_time,
                    "seed_count_before": seed_count_before,
                    "seed_count_after": seed_count_after,
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Matrixsum Phase Refresh: Before/After Ours Counts",
        "",
        "Before = 2026-07-05 structure=opt refresh. After = 2026-07-06 Matrixsum phase refresh. Random-channel rows report seed-0..4 means.",
        "",
        "| Example | Basis | Gates before -> after | Delta | Non-Clifford before -> after | Delta | Time before -> after (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gate_delta = fmt_signed_int(row["delta_gate_count"])
        gate_pct = fmt_pct(row["delta_gate_pct"])
        if gate_delta and gate_pct:
            gate_delta = f"{gate_delta} ({gate_pct})"
        nc_delta = fmt_signed_int(row["delta_non_clifford"])
        nc_pct = fmt_pct(row["delta_non_clifford_pct"])
        if nc_delta and nc_pct:
            nc_delta = f"{nc_delta} ({nc_pct})"
        time_delta = fmt_time(row["delta_compile_time_s"])
        if time_delta:
            sign = "+" if as_float(row["delta_compile_time_s"]) and as_float(row["delta_compile_time_s"]) > 0 else ""
            time_delta = f" ({sign}{time_delta})"
        lines.append(
            f"| {row['example']} | {row['basis']} | "
            f"{fmt_int(row['before_gate_count'])} -> {fmt_int(row['after_gate_count'])} | "
            f"{gate_delta} | "
            f"{fmt_int(row['before_non_clifford'])} -> {fmt_int(row['after_non_clifford'])} | "
            f"{nc_delta} | "
            f"{fmt_time(row['before_compile_time_s'])} -> {fmt_time(row['after_compile_time_s'])}{time_delta} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.1pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\caption{Before/after comparison for the 2026-07-06 Matrixsum phase refresh. Before is the 2026-07-05 structure\\_opt refresh; after is the refreshed ours result.}",
        "\\label{tab:channel-lcu-matrixsum-phase-before-after}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Example & Basis & $G_{\\mathrm{before}}$ & $G_{\\mathrm{after}}$ & $\\Delta G$ & $NC_{\\mathrm{before}}$ & $NC_{\\mathrm{after}}$ & $\\Delta NC$ & Time after \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["example"],
                    row["basis"],
                    fmt_int(row["before_gate_count"]).replace(",", r"{,}") or "--",
                    fmt_int(row["after_gate_count"]).replace(",", r"{,}") or "--",
                    fmt_signed_int(row["delta_gate_count"]).replace(",", r"{,}") or "--",
                    fmt_int(row["before_non_clifford"]).replace(",", r"{,}") or "--",
                    fmt_int(row["after_non_clifford"]).replace(",", r"{,}") or "--",
                    fmt_signed_int(row["delta_non_clifford"]).replace(",", r"{,}") or "--",
                    fmt_time(row["after_compile_time_s"]) or "--",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_csv(rows, OUT_CSV)
    write_md(rows, OUT_MD)
    write_tex(rows, OUT_TEX)
    print(f"wrote {len(rows)} rows")
    print(OUT_CSV)
    print(OUT_MD)
    print(OUT_TEX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
