#!/usr/bin/env python3
"""Merge sharded second-stage optimizer results into compact tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "Results" / "ours_second_stage_external_optimizers.csv",
    *sorted((ROOT / "Results" / "ours_second_stage_shards").glob("*.csv")),
]
DEFAULT_OUT_CSV = ROOT / "Results" / "ours_second_stage_external_optimizers_merged.csv"
DEFAULT_OUT_MD = ROOT / "Results" / "ours_second_stage_external_optimizers_merged.md"

ORDER = [
    "hypercube_random_walk_n8",
    "hypercube_random_walk_n28",
    "periodic_tfim_n4_dt0p1",
    "periodic_tfim_n8_dt0p1",
    "periodic_tfim_n12_dt0p1",
    "repetition_dqe_d3_eps0p25",
    "random_channel_N5_dim4_N1_10_N2_2_seed0",
    "random_channel_N5_dim4_N1_10_N2_2_seed1",
    "random_channel_N5_dim4_N1_10_N2_2_seed2",
    "random_channel_N5_dim4_N1_10_N2_2_seed3",
    "random_channel_N5_dim4_N1_10_N2_2_seed4",
]
TOOL_ORDER = ["voqc", "wisq", "quartz", "qiskit_opt3", "pytket", "bqskit"]
DROP_CSV_FIELDS = {"command", "stdout_tail", "stderr_tail"}


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as in_file:
            rows.extend(csv.DictReader(in_file))
    return rows


def rank_row(row: dict[str, str]) -> tuple[int, float]:
    status = row.get("status", "")
    has_count = bool(row.get("output_gate_count"))
    status_rank = {
        "ok": 0,
        "timeout_checkpoint": 1,
        "timeout_fallback_input": 2,
        "failed_fallback_input": 3,
        "timeout": 4,
        "failed": 5,
    }.get(status, 6)
    elapsed = as_float(row.get("optimizer_time_s"))
    return (status_rank - (1 if has_count else 0), elapsed if elapsed is not None else 1e99)


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["benchmark"], row["basis"], row["tool"])
        if key not in best or rank_row(row) < rank_row(best[key]):
            best[key] = row
    return sorted(
        best.values(),
        key=lambda row: (
            ORDER.index(row["benchmark"]) if row["benchmark"] in ORDER else 999,
            row["basis"],
            TOOL_ORDER.index(row["tool"]) if row["tool"] in TOOL_ORDER else 999,
        ),
    )


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_int(value: Any) -> str:
    if value in (None, ""):
        return "-"
    return f"{int(float(value)):,}"


def fmt_time(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    if number >= 100:
        return f"{number:.1f}"
    return f"{number:.2f}"


def fmt_cell(row: dict[str, str] | None) -> str:
    if row is None:
        return "-"
    status = row.get("status") or "-"
    gates = fmt_int(row.get("output_gate_count"))
    ratio = as_float(row.get("ratio_vs_ours_input"))
    ratio_text = "-" if ratio is None else f"{100 * ratio:.1f}%"
    time_text = fmt_time(row.get("optimizer_time_s"))
    if gates == "-":
        return f"{status}, {time_text}s"
    if status == "ok":
        return f"{gates} ({ratio_text}), {time_text}s"
    return f"{gates} ({ratio_text}), {time_text}s, {status}"


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field for field in rows[0].keys() if field not in DROP_CSV_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(rows: list[dict[str, str]], path: Path, csv_path: Path) -> None:
    by_case: dict[tuple[str, str, str], dict[str, str]] = {}
    inputs: dict[tuple[str, str], dict[str, str]] = {}
    families: dict[str, str] = {}
    for row in rows:
        by_case[(row["benchmark"], row["basis"], row["tool"])] = row
        inputs[(row["benchmark"], row["basis"])] = row
        families[row["benchmark"]] = row["family"]

    lines = [
        "# Ours Circuits Followed by External Optimizers",
        "",
        f"CSV: `{csv_path}`",
        "",
        "Cells show `output gates (output/input), optimizer time`; non-`ok` statuses are appended. NAM tools receive the in-house NAM circuit; IBM tools receive the in-house IBM circuit.",
        "",
        "| Family | Benchmark | NAM input | VOQC | WISQ | Quartz | IBM input | pytket | BQSKit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ORDER:
        family = families.get(benchmark)
        if family is None:
            continue
        nam_input = inputs.get((benchmark, "NAM"))
        ibm_input = inputs.get((benchmark, "IBM"))
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    benchmark,
                    fmt_int(nam_input.get("input_gate_count") if nam_input else None),
                    fmt_cell(by_case.get((benchmark, "NAM", "voqc"))),
                    fmt_cell(by_case.get((benchmark, "NAM", "wisq"))),
                    fmt_cell(by_case.get((benchmark, "NAM", "quartz"))),
                    fmt_int(ibm_input.get("input_gate_count") if ibm_input else None),
                    fmt_cell(by_case.get((benchmark, "IBM", "pytket"))),
                    fmt_cell(by_case.get((benchmark, "IBM", "bqskit"))),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = dedupe(read_rows(DEFAULT_INPUTS))
    write_csv(rows, DEFAULT_OUT_CSV)
    write_markdown(rows, DEFAULT_OUT_MD, DEFAULT_OUT_CSV)
    print(f"merged {len(rows)} rows")
    print(DEFAULT_OUT_CSV)
    print(DEFAULT_OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
