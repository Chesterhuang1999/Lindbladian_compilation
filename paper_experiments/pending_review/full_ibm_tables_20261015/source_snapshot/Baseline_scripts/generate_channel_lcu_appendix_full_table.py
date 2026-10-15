#!/usr/bin/env python3
"""Generate a compact appendix table for all CP1A/CP1B channel-LCU rows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "Baseline_results" / "eval_redesign" / "cp5_aggregation" / "rq1_appendix_full_per_tool.csv"
RUNTIME_CSV = ROOT / "Baseline_results" / "eval_redesign" / "cp5_aggregation" / "cp1_cp1b_tool_runtime_summary.csv"
OURS_RERUN_CSV = ROOT / "Results" / "block_encoding_new_ours_rerun_all_families.csv"
STRUCTURE_OPT_REFRESH_MAIN_CSV = ROOT / "Results" / "channel_lcu_structure_opt_refresh_main.csv"
STRUCTURE_OPT_REFRESH_STRUCTURED_CSV = ROOT / "Results" / "channel_lcu_structure_opt_refresh_structured.csv"
MATRIXSUM_PHASE_REFRESH_CSV = ROOT / "Results" / "channel_lcu_matrixsum_phase_refresh_all.csv"
STRUCTURE_OPT_REFRESH_CSVS = [
    STRUCTURE_OPT_REFRESH_MAIN_CSV,
    STRUCTURE_OPT_REFRESH_STRUCTURED_CSV,
    MATRIXSUM_PHASE_REFRESH_CSV,
]
RANDOM_CHANNEL_CSV = ROOT / "Results" / "random_channel_benchmarks_summary.csv"
RANDOM_CHANNEL_VOQC_FIXED_CSV = ROOT / "Results" / "random_channel_n5_voqc_fixed_summary.csv"
QUARTZ_RERUN_CSV = ROOT / "Results" / "quartz_rerun_hrw12_randch_n5_20260705_summary.csv"
HRW28_BQSKIT_RERUN_CSV = ROOT / "Results" / "hrw28_bqskit_rerun_20260708_summary.csv"
TFIM_CURRENT_IBM_OVERRIDES = {
    "periodic_tfim_n8_dt0p1": {
        "raw": 775_106,
        "qiskit_opt3": 642_659,
        "qiskit_time_s": 20.79832416700083,
        "source": "Baseline_results/eval_redesign/tfim_qiskit_opt3_current_raw_rerun_20260710/tfim8/qiskit_opt3_summary.json",
    },
    "periodic_tfim_n12_dt0p1": {
        "raw": 1_896_052,
        "qiskit_opt3": 1_574_808,
        "qiskit_time_s": 53.29036745800113,
        "source": "Baseline_results/eval_redesign/tfim_qiskit_opt3_current_raw_rerun_20260710/tfim12/qiskit_opt3_summary.json",
    },
}
OUT_CSV = ROOT / "Results" / "Table1_channelLCU_appendix_full.csv"
OUT_TEX = ROOT / "Results" / "Table1_channelLCU_appendix_full.tex"
OUT_NAM_TEX = ROOT / "Results" / "Table1_channelLCU_appendix_full_nam.tex"
OUT_IBM_TEX = ROOT / "Results" / "Table1_channelLCU_appendix_full_ibm.tex"
OUT_TIME_CSV = ROOT / "Results" / "Table1_channelLCU_tool_runtime_complete_cases.csv"
OUT_TIME_TEX = ROOT / "Results" / "Table1_channelLCU_tool_runtime_complete_cases.tex"

NAM_TOOLS = ["voqc", "feynman", "wisq", "quartz"]
IBM_TOOLS = ["qiskit_opt3", "pytket", "bqskit"]

BENCHMARK_ORDER = [
    "toom_2d_periodic_L2_dt0p1",
    "toom_2d_periodic_L3_dt0p1",
    "hypercube_random_walk_n4",
    "hypercube_random_walk_n8",
    "hypercube_random_walk_n12",
    "hypercube_random_walk_n20",
    "hypercube_random_walk_n28",
    "periodic_tfim_n4_dt0p1",
    "periodic_tfim_n8_dt0p1",
    "periodic_tfim_n12_dt0p1",
    "graph_cluster_2d_grid_2x4",
    "graph_cluster_random_bd3_n12_seed7",
    "decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K1_4",
    "periodic_tfim_n2_dt0p1_higher_order_K2_q4_K1_4",
    "random_coloring_n4_cycle",
    "random_coloring_n4_complete",
    "random_coloring_n4_q4_linear",
    "random_coloring_n4_q4_cycle",
    "random_coloring_n4_q4_complete",
    "random_coloring_n8_q8_cycle",
    "random_coloring_n8_q8_complete",
    "repetition_dqe_d3_eps0p25",
    "repetition_dqe_d5_eps0p25",
    "repetition_dqe_d7_eps0p25",
    "surface_code_dqe_d3",
    "random_channel_N5_dim4_N1_10_N2_2_mean",
    "random_channel_N9_dim8_N1_20_N2_4_mean",
]

DROP_BENCHMARKS = {
    "periodic_tfim_n2_dt0p1_higher_order_K2_q4_K1_4",
    "random_coloring_n4_cycle",
    "random_coloring_n4_complete",
    "random_coloring_n4_q4_cycle",
    "random_coloring_n4_q4_complete",
    "random_coloring_n8_q8_cycle",
    "random_coloring_n8_q8_complete",
    "repetition_dqe_d7_eps0p25",
    "surface_code_dqe_d3",
}

LABELS = {
    "toom_2d_periodic_L2_dt0p1": "Toom\\_L2",
    "toom_2d_periodic_L3_dt0p1": "Toom\\_L3",
    "hypercube_random_walk_n4": "HRW\\_4",
    "hypercube_random_walk_n8": "HRW\\_8",
    "hypercube_random_walk_n12": "HRW\\_12",
    "hypercube_random_walk_n20": "HRW\\_20",
    "hypercube_random_walk_n28": "HRW\\_28",
    "periodic_tfim_n4_dt0p1": "TFIM\\_4",
    "periodic_tfim_n8_dt0p1": "TFIM\\_8",
    "periodic_tfim_n12_dt0p1": "TFIM\\_12",
    "graph_cluster_2d_grid_2x4": "GC\\_2x4",
    "graph_cluster_random_bd3_n12_seed7": "GC\\_bd3\\_12",
    "decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K1_4": "Decay\\_HO",
    "periodic_tfim_n2_dt0p1_higher_order_K2_q4_K1_4": "TFIM\\_HO",
    "random_coloring_n4_cycle": "Color\\_n4\\_cyc",
    "random_coloring_n4_complete": "Color\\_n4\\_comp",
    "random_coloring_n4_q4_linear": "Color\\_n4q4\\_lin",
    "random_coloring_n4_q4_cycle": "Color\\_n4q4\\_cyc",
    "random_coloring_n4_q4_complete": "Color\\_n4q4\\_comp",
    "random_coloring_n8_q8_cycle": "Color\\_n8q8\\_cyc",
    "random_coloring_n8_q8_complete": "Color\\_n8q8\\_comp",
    "repetition_dqe_d3_eps0p25": "RepDQE\\_d3",
    "repetition_dqe_d5_eps0p25": "RepDQE\\_d5",
    "repetition_dqe_d7_eps0p25": "RepDQE\\_d7",
    "surface_code_dqe_d3": "SurfDQE\\_d3",
    "random_channel_N5_dim4_N1_10_N2_2_mean": "RandCh\\_N5",
    "random_channel_N9_dim8_N1_20_N2_4_mean": "RandCh\\_N9",
}

APPENDIX_GROUPS = [
    (
        "Lindbladian",
        [
            "periodic_tfim_n4_dt0p1",
            "periodic_tfim_n8_dt0p1",
            "periodic_tfim_n12_dt0p1",
            "toom_2d_periodic_L2_dt0p1",
            "toom_2d_periodic_L3_dt0p1",
            "graph_cluster_2d_grid_2x4",
            "graph_cluster_random_bd3_n12_seed7",
            "decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K1_4",
        ],
    ),
    (
        "Algorithm-derived",
        [
            "hypercube_random_walk_n4",
            "hypercube_random_walk_n8",
            "hypercube_random_walk_n12",
            "hypercube_random_walk_n20",
            "hypercube_random_walk_n28",
            "random_coloring_n4_q4_linear",
            "repetition_dqe_d3_eps0p25",
            "repetition_dqe_d5_eps0p25",
        ],
    ),
    (
        "Synthetic",
        [
            "random_channel_N5_dim4_N1_10_N2_2_mean",
            "random_channel_N9_dim8_N1_20_N2_4_mean",
        ],
    ),
]

APPENDIX_GROUP_BY_BENCHMARK = {
    benchmark: group_name
    for group_name, benchmarks in APPENDIX_GROUPS
    for benchmark in benchmarks
}
APPENDIX_GROUP_RANK = {
    benchmark: (group_index, benchmark_index)
    for group_index, (_group_name, benchmarks) in enumerate(APPENDIX_GROUPS)
    for benchmark_index, benchmark in enumerate(benchmarks)
}

TO_MARK = "\\textsc{to}"
UG_MARK = "\\textsc{ug}"

UNABLE_INPUT_COLUMNS = {
    "toom_2d_periodic_L3_dt0p1": {
        "ibm_raw",
        "ibm_qiskit_opt3",
        "ibm_pytket",
        "ibm_bqskit",
    },
    "random_coloring_n4_q4_linear": {
        "nam_raw",
        "nam_voqc",
        "nam_feynman",
        "nam_wisq",
        "nam_quartz",
        "ibm_raw",
        "ibm_qiskit_opt3",
        "ibm_pytket",
        "ibm_bqskit",
    },
    "repetition_dqe_d5_eps0p25": {
        "nam_raw",
        "nam_voqc",
        "nam_feynman",
        "nam_wisq",
        "nam_quartz",
        "ibm_raw",
        "ibm_qiskit_opt3",
        "ibm_pytket",
        "ibm_bqskit",
    },
}

RANDOM_CHANNEL_PARAMS = {
    "N5_dim4_N1_10_N2_2": "random_channel_N5_dim4_N1_10_N2_2_mean",
    "N9_dim8_N1_20_N2_4": "random_channel_N9_dim8_N1_20_N2_4_mean",
}
RANDOM_CHANNEL_SEEDS = {
    "N5_dim4_N1_10_N2_2": {0, 1, 2, 3, 4},
    "N9_dim8_N1_20_N2_4": {0, 1, 2, 3, 4},
}


@dataclass
class Entry:
    benchmark_family: str
    benchmark_name: str
    basis_group: str
    method: str
    optimizer: str
    opt_gates: str = ""
    raw_gates: str = ""
    ratio_vs_raw: str = ""
    optimizer_time_sec: str = ""
    total_time_sec: str = ""
    status: str = ""
    note: str = ""
    source_path: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rounded_mean_string(values: list[float]) -> str:
    number = mean(values)
    if number is None:
        return ""
    return str(int(round(number)))


def seed_from_benchmark(benchmark: str) -> int | None:
    if "_seed" not in benchmark:
        return None
    try:
        return int(benchmark.rsplit("_seed", 1)[1])
    except ValueError:
        return None


def entry_from_row(row: dict[str, str]) -> Entry:
    return Entry(
        benchmark_family=row.get("benchmark_family", ""),
        benchmark_name=row.get("benchmark_name", ""),
        basis_group=row.get("basis_group", ""),
        method=row.get("method", ""),
        optimizer=row.get("optimizer", ""),
        opt_gates=row.get("opt_gates", ""),
        raw_gates=row.get("raw_gates", ""),
        ratio_vs_raw=row.get("ratio_vs_raw", ""),
        optimizer_time_sec=row.get("optimizer_time_sec", ""),
        total_time_sec=row.get("total_time_sec", ""),
        status=row.get("status", ""),
        note=row.get("note", ""),
        source_path=row.get("source_path", ""),
    )


def status_rank(status: str, has_gates: bool) -> tuple[int, int]:
    if status == "ok":
        return (0, 0 if has_gates else 1)
    if has_gates and status in {
        "reused_existing_3600",
        "checkpoint_best_after_timeout",
        "killed_checkpoint",
        "timeout_partial",
        "external_killed",
        "ok_non_native_basis_output",
        "ok_fixed_recount",
        "ok_recount",
    }:
        return (1, 0)
    if has_gates:
        return (2, 0)
    return {
        "timeout": (3, 0),
        "runner_outer_timeout": (3, 1),
        "no_output": (3, 2),
        "unable_to_generate_raw_circuit": (4, 0),
        "blocked_unable_to_generate_raw_circuit": (4, 1),
        "pending_input_qasm": (5, 0),
        "missing_existing_basis_qasm": (5, 1),
        "environment_permission_error": (6, 0),
        "skipped_larger_due_to_timeout": (6, 1),
        "stopped_after_smaller_scale_3600_timeout_or_no_output": (6, 2),
        "unavailable": (7, 0),
        "unknown": (7, 1),
    }.get(status, (8, 0))


def choose_better(old: Entry | None, new: Entry) -> Entry:
    if old is None:
        return new
    old_key = status_rank(old.status, as_int(old.opt_gates) is not None)
    new_key = status_rank(new.status, as_int(new.opt_gates) is not None)
    if new_key < old_key:
        return new
    if new_key == old_key:
        old_g = as_int(old.opt_gates)
        new_g = as_int(new.opt_gates)
        if old_g is None:
            return new
        if new_g is not None and new_g < old_g:
            return new
    return old


def load_entries() -> tuple[dict[tuple[str, str, str], Entry], dict[tuple[str, str], str], set[str]]:
    entries: dict[tuple[str, str, str], Entry] = {}
    family: dict[tuple[str, str], str] = {}
    benchmarks: set[str] = set()
    for row in read_csv(SOURCE_CSV):
        entry = entry_from_row(row)
        benchmark = entry.benchmark_name
        basis = entry.basis_group.upper()
        if not benchmark or basis not in {"NAM", "IBM"}:
            continue
        benchmarks.add(benchmark)
        family[(benchmark, basis)] = entry.benchmark_family
        if entry.method == "raw_denominator":
            slot = "raw"
        elif entry.method == "ours_full":
            slot = "ours"
        elif entry.method == "generic_optimizer":
            slot = entry.optimizer.lower()
        else:
            continue
        if slot in {"all_nam_tools", "all_ibm_tools"}:
            continue
        key = (benchmark, basis, slot)
        entries[key] = choose_better(entries.get(key), entry)

    raw_by_basis = {
        (benchmark, basis): entries.get((benchmark, basis, "raw"))
        for benchmark in benchmarks
        for basis in ("NAM", "IBM")
    }
    apply_ours_override(entries, family, benchmarks, raw_by_basis, OURS_RERUN_CSV, "block_encoding_new rerun override")
    for refresh_csv in STRUCTURE_OPT_REFRESH_CSVS:
        apply_ours_override(
            entries,
            family,
            benchmarks,
            raw_by_basis,
            refresh_csv,
            (
                "2026-07-06 Matrixsum phase refresh override"
                if refresh_csv == MATRIXSUM_PHASE_REFRESH_CSV
                else "2026-07-05 structure=opt refresh override"
            ),
        )
    load_random_channel_entries(entries, family, benchmarks)
    apply_generic_override_csv(entries, family, benchmarks, QUARTZ_RERUN_CSV, "Quartz rerun override")
    apply_generic_override_csv(entries, family, benchmarks, HRW28_BQSKIT_RERUN_CSV, "HRW28 BQSKit rerun override")
    apply_random_channel_voqc_fixed_override(entries, family, benchmarks)
    apply_tfim_current_ibm_override(entries, family, benchmarks)
    return entries, family, benchmarks


def apply_ours_override(
    entries: dict[tuple[str, str, str], Entry],
    family: dict[tuple[str, str], str],
    benchmarks: set[str],
    raw_by_basis: dict[tuple[str, str], Entry | None],
    path: Path,
    note: str,
) -> None:
    if not path.exists():
        return
    for row in read_csv(path):
        benchmark = row.get("benchmark", "")
        basis = row.get("basis", "").upper()
        if benchmark not in benchmarks or basis not in {"NAM", "IBM"}:
            continue
        gates = row.get("after_gate_count", "")
        raw_entry = raw_by_basis.get((benchmark, basis))
        raw_gates = raw_entry.raw_gates if raw_entry else ""
        ratio = ""
        raw_int = as_int(raw_gates)
        gate_int = as_int(gates)
        if raw_int and gate_int is not None:
            ratio = str(gate_int / raw_int)
        family[(benchmark, basis)] = row.get("family", family.get((benchmark, basis), ""))
        entries[(benchmark, basis, "ours")] = Entry(
            benchmark_family=family.get((benchmark, basis), ""),
            benchmark_name=benchmark,
            basis_group=basis,
            method="ours_full",
            optimizer="channel_first_rewrite_opt_matrix_order",
            opt_gates=gates,
            raw_gates=raw_gates,
            ratio_vs_raw=ratio,
            optimizer_time_sec="",
            total_time_sec=row.get("after_compile_time_s", ""),
            status=row.get("after_status", "ok"),
            note=note,
            source_path=row.get("source_after", ""),
        )


def random_channel_slot(row: dict[str, str]) -> str:
    method = row.get("method", "")
    category = row.get("category", "")
    if category == "raw" or method == "raw":
        return "raw"
    if category == "ours" or method == "ours":
        return "ours"
    if category == "generic":
        return method.lower()
    return method.lower()


def load_random_channel_entries(
    entries: dict[tuple[str, str, str], Entry],
    family: dict[tuple[str, str], str],
    benchmarks: set[str],
) -> None:
    if not RANDOM_CHANNEL_CSV.exists():
        return

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in read_csv(RANDOM_CHANNEL_CSV):
        param = row.get("parameter_set", "")
        benchmark = RANDOM_CHANNEL_PARAMS.get(param)
        basis = row.get("basis", "").upper()
        seed = seed_from_benchmark(row.get("benchmark", "")) or as_int(row.get("seed"))
        if benchmark is None or basis not in {"NAM", "IBM"} or seed is None:
            continue
        if seed not in RANDOM_CHANNEL_SEEDS[param]:
            continue
        slot = random_channel_slot(row)
        if slot not in ({"raw", "ours", "qiskit_opt3"} | set(NAM_TOOLS) | set(IBM_TOOLS)):
            continue
        grouped.setdefault((param, basis, slot), []).append(row)

    latest_ours: dict[tuple[str, str], list[dict[str, str]]] = {}
    latest_ours_source: dict[tuple[str, str], Path] = {}
    for source_path in [OURS_RERUN_CSV, *STRUCTURE_OPT_REFRESH_CSVS]:
        if not source_path.exists():
            continue
        source_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in read_csv(source_path):
            benchmark = row.get("benchmark", "")
            basis = row.get("basis", "").upper()
            for param in RANDOM_CHANNEL_PARAMS:
                if param in benchmark and basis in {"NAM", "IBM"}:
                    seed = seed_from_benchmark(benchmark)
                    if seed is not None and seed in RANDOM_CHANNEL_SEEDS[param]:
                        source_rows.setdefault((param, basis), []).append(row)
        for key, rows_for_key in source_rows.items():
            latest_ours[key] = rows_for_key
            latest_ours_source[key] = source_path

    for param, benchmark in RANDOM_CHANNEL_PARAMS.items():
        for basis in ("NAM", "IBM"):
            benchmarks.add(benchmark)
            family[(benchmark, basis)] = "random channel"
            raw_rows = grouped.get((param, basis, "raw"), [])
            raw_values = [as_float(row.get("metric_total")) for row in raw_rows]
            raw_values = [value for value in raw_values if value is not None]
            raw_gates = rounded_mean_string(raw_values)
            raw_status = "ok" if len(raw_values) == len(RANDOM_CHANNEL_SEEDS[param]) else "ok_partial_seed_mean"
            if raw_gates:
                entries[(benchmark, basis, "raw")] = Entry(
                    benchmark_family="random channel",
                    benchmark_name=benchmark,
                    basis_group=basis,
                    method="raw_denominator",
                    optimizer="raw",
                    opt_gates=raw_gates,
                    raw_gates=raw_gates,
                    ratio_vs_raw="1",
                    optimizer_time_sec="",
                    total_time_sec=str(mean([as_float(row.get("compile_time_s")) for row in raw_rows if as_float(row.get("compile_time_s")) is not None]) or ""),
                    status=raw_status,
                    note=f"mean over {len(raw_values)}/{len(RANDOM_CHANNEL_SEEDS[param])} seeds",
                    source_path=str(RANDOM_CHANNEL_CSV.relative_to(ROOT)),
                )

            ours_rows = latest_ours.get((param, basis), [])
            if ours_rows:
                gate_values = [as_float(row.get("after_gate_count")) for row in ours_rows]
                time_values = [as_float(row.get("after_compile_time_s")) for row in ours_rows]
            else:
                ours_source_rows = grouped.get((param, basis, "ours"), [])
                gate_values = [as_float(row.get("metric_total")) for row in ours_source_rows]
                time_values = [as_float(row.get("compile_time_s")) for row in ours_source_rows]
            gate_values = [value for value in gate_values if value is not None]
            time_values = [value for value in time_values if value is not None]
            gates = rounded_mean_string(gate_values)
            if gates:
                raw_int = as_int(raw_gates)
                gate_int = as_int(gates)
                entries[(benchmark, basis, "ours")] = Entry(
                    benchmark_family="random channel",
                    benchmark_name=benchmark,
                    basis_group=basis,
                    method="ours_full",
                    optimizer="channel_first_rewrite_opt_matrix_order",
                    opt_gates=gates,
                    raw_gates=raw_gates,
                    ratio_vs_raw=(str(gate_int / raw_int) if raw_int and gate_int is not None else ""),
                    optimizer_time_sec="",
                    total_time_sec=(str(mean(time_values)) if time_values else ""),
                    status="ok" if len(gate_values) == len(RANDOM_CHANNEL_SEEDS[param]) else "ok_partial_seed_mean",
                    note=f"mean over {len(gate_values)}/{len(RANDOM_CHANNEL_SEEDS[param])} seeds",
                    source_path=str(latest_ours_source.get((param, basis), RANDOM_CHANNEL_CSV).relative_to(ROOT) if ours_rows else RANDOM_CHANNEL_CSV.relative_to(ROOT)),
                )

            tools = NAM_TOOLS if basis == "NAM" else IBM_TOOLS
            for slot in tools:
                tool_rows = grouped.get((param, basis, slot), [])
                gate_values = [as_float(row.get("metric_total")) for row in tool_rows]
                gate_values = [value for value in gate_values if value is not None]
                time_values = [as_float(row.get("elapsed_seconds") or row.get("compile_time_s")) for row in tool_rows]
                time_values = [value for value in time_values if value is not None]
                if gate_values:
                    gates = rounded_mean_string(gate_values)
                    raw_int = as_int(raw_gates)
                    gate_int = as_int(gates)
                    statuses = {row.get("status", "") for row in tool_rows if row.get("metric_total")}
                    complete = len(gate_values) == len(RANDOM_CHANNEL_SEEDS[param]) and statuses <= {"ok"}
                    status = "ok" if complete else ("timeout_partial_seed_mean" if "timeout_partial" in statuses else "ok_partial_seed_mean")
                    entries[(benchmark, basis, slot)] = Entry(
                        benchmark_family="random channel",
                        benchmark_name=benchmark,
                        basis_group=basis,
                        method="generic_optimizer",
                        optimizer=slot,
                        opt_gates=gates,
                        raw_gates=raw_gates,
                        ratio_vs_raw=(str(gate_int / raw_int) if raw_int and gate_int is not None else ""),
                        optimizer_time_sec=(str(mean(time_values)) if time_values else ""),
                        total_time_sec="",
                        status=status,
                        note=f"mean over {len(gate_values)}/{len(RANDOM_CHANNEL_SEEDS[param])} seeds",
                        source_path=str(RANDOM_CHANNEL_CSV.relative_to(ROOT)),
                    )
                elif tool_rows:
                    statuses = {row.get("status", "") for row in tool_rows}
                    if "timeout" in statuses:
                        status = "timeout"
                    elif "failed" in statuses:
                        status = "failed"
                    else:
                        status = "unavailable"
                    entries[(benchmark, basis, slot)] = Entry(
                        benchmark_family="random channel",
                        benchmark_name=benchmark,
                        basis_group=basis,
                        method="generic_optimizer",
                        optimizer=slot,
                        opt_gates="",
                        raw_gates=raw_gates,
                        ratio_vs_raw="",
                        optimizer_time_sec=(str(mean(time_values)) if time_values else ""),
                        total_time_sec="",
                        status=status,
                        note=f"no numeric output over {len(tool_rows)}/{len(RANDOM_CHANNEL_SEEDS[param])} seed rows",
                        source_path=str(RANDOM_CHANNEL_CSV.relative_to(ROOT)),
                    )


def apply_generic_override_csv(
    entries: dict[tuple[str, str, str], Entry],
    family: dict[tuple[str, str], str],
    benchmarks: set[str],
    path: Path,
    note: str,
) -> None:
    if not path.exists():
        return
    for row in read_csv(path):
        if row.get("row_type") not in {"", "plot"}:
            continue
        benchmark = row.get("benchmark", "")
        basis = row.get("basis", "").upper()
        tool = row.get("tool", "").lower()
        opt_gates = row.get("opt_gates", "")
        raw_gates = row.get("raw_gates", "")
        if not benchmark or basis not in {"NAM", "IBM"} or not tool or not opt_gates:
            continue
        benchmarks.add(benchmark)
        family[(benchmark, basis)] = family.get((benchmark, basis), row.get("family", ""))
        raw_entry = entries.get((benchmark, basis, "raw"))
        table_raw_gates = ""
        if raw_entry is not None:
            table_raw_gates = raw_entry.raw_gates or raw_entry.opt_gates
        display_raw_gates = table_raw_gates or raw_gates
        raw_int = as_int(display_raw_gates)
        opt_int = as_int(opt_gates)
        ratio = row.get("ratio", "")
        if raw_int and opt_int is not None:
            ratio = str(opt_int / raw_int)
        entries[(benchmark, basis, tool)] = Entry(
            benchmark_family=family.get((benchmark, basis), row.get("family", "")),
            benchmark_name=benchmark,
            basis_group=basis,
            method="generic_optimizer",
            optimizer=tool,
            opt_gates=opt_gates,
            raw_gates=display_raw_gates,
            ratio_vs_raw=ratio,
            optimizer_time_sec=row.get("elapsed_sec", ""),
            total_time_sec="",
            status=row.get("status", "ok"),
            note=note + (f"; {row.get('note', '')}" if row.get("note") else ""),
            source_path=row.get("source_path", str(path.relative_to(ROOT))),
        )


def apply_random_channel_voqc_fixed_override(
    entries: dict[tuple[str, str, str], Entry],
    family: dict[tuple[str, str], str],
    benchmarks: set[str],
) -> None:
    if not RANDOM_CHANNEL_VOQC_FIXED_CSV.exists():
        return
    for row in read_csv(RANDOM_CHANNEL_VOQC_FIXED_CSV):
        if row.get("benchmark") != "random_channel_N5_dim4_N1_10_N2_2_mean":
            continue
        benchmark = row["benchmark"]
        basis = row.get("basis", "").upper()
        if basis != "NAM":
            continue
        raw_gates = row.get("raw_gates", "")
        opt_gates = row.get("opt_gates", "")
        raw_int = as_int(raw_gates)
        opt_int = as_int(opt_gates)
        benchmarks.add(benchmark)
        family[(benchmark, basis)] = "random channel"
        entries[(benchmark, basis, "voqc")] = Entry(
            benchmark_family="random channel",
            benchmark_name=benchmark,
            basis_group=basis,
            method="generic_optimizer",
            optimizer="voqc",
            opt_gates=opt_gates,
            raw_gates=raw_gates,
            ratio_vs_raw=(str(opt_int / raw_int) if raw_int and opt_int is not None else row.get("ratio", "")),
            optimizer_time_sec=row.get("elapsed_sec", ""),
            total_time_sec="",
            status=row.get("status", "ok_fixed_recount"),
            note=row.get("note", "VOQC recount override after exact RZ angle folding."),
            source_path=row.get("source_path", str(RANDOM_CHANNEL_VOQC_FIXED_CSV.relative_to(ROOT))),
        )


def apply_tfim_current_ibm_override(
    entries: dict[tuple[str, str, str], Entry],
    family: dict[tuple[str, str], str],
    benchmarks: set[str],
) -> None:
    for benchmark, override in TFIM_CURRENT_IBM_OVERRIDES.items():
        if benchmark not in benchmarks:
            continue
        basis = "IBM"
        raw_gates = str(override["raw"])
        benchmarks.add(benchmark)
        family[(benchmark, basis)] = family.get((benchmark, basis), "TFIM")
        entries[(benchmark, basis, "raw")] = Entry(
            benchmark_family=family[(benchmark, basis)],
            benchmark_name=benchmark,
            basis_group=basis,
            method="raw_denominator",
            optimizer="raw",
            opt_gates=raw_gates,
            raw_gates=raw_gates,
            ratio_vs_raw="1",
            status="ok",
            note="2026-07-10 regenerated IBM opt0 raw denominator",
            source_path=str(override["source"]),
        )
        ours = entries.get((benchmark, basis, "ours"))
        if ours is not None:
            ours.raw_gates = raw_gates
            ours.ratio_vs_raw = str(as_int(ours.opt_gates) / int(raw_gates)) if as_int(ours.opt_gates) is not None else ""
        qiskit_gates = str(override["qiskit_opt3"])
        entries[(benchmark, basis, "qiskit_opt3")] = Entry(
            benchmark_family=family[(benchmark, basis)],
            benchmark_name=benchmark,
            basis_group=basis,
            method="generic_optimizer",
            optimizer="qiskit_opt3",
            opt_gates=qiskit_gates,
            raw_gates=raw_gates,
            ratio_vs_raw=str(int(qiskit_gates) / int(raw_gates)),
            optimizer_time_sec=str(override["qiskit_time_s"]),
            total_time_sec="",
            status="ok",
            note="2026-07-10 rerun from regenerated IBM opt0 raw input",
            source_path=str(override["source"]),
        )


def load_runtime_overrides() -> dict[tuple[str, str, str], str]:
    runtimes: dict[tuple[str, str, str], str] = {}
    if not RUNTIME_CSV.exists():
        return runtimes
    for row in read_csv(RUNTIME_CSV):
        benchmark = row.get("benchmark_name", "")
        basis = row.get("basis_group", "").upper()
        method = row.get("method", "")
        optimizer = row.get("optimizer", "").lower()
        runtime = row.get("runtime_sec", "")
        if not benchmark or basis not in {"NAM", "IBM"} or not runtime:
            continue
        if method == "ours_full":
            slot = "ours"
        elif method == "generic_optimizer":
            slot = optimizer
        else:
            continue
        runtimes[(benchmark, basis, slot)] = runtime
    return runtimes


def fmt_int(value: Any) -> str:
    number = as_int(value)
    if number is None:
        return ""
    return f"{number:,}"


def fmt_percent(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{100 * number:.2f}\\%"


def fmt_time(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    if number >= 100:
        return f"{number:.1f}"
    if number >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def compact_status(status: str) -> str:
    if status in {"timeout", "runner_outer_timeout", "no_output", "stopped_after_smaller_scale_3600_timeout_or_no_output"}:
        return TO_MARK
    if status in {"unable_to_generate_raw_circuit", "blocked_unable_to_generate_raw_circuit"}:
        return UG_MARK
    if status in {"pending_input_qasm", "missing_existing_basis_qasm"}:
        return UG_MARK
    if status in {"environment_permission_error"}:
        return TO_MARK
    if status in {"failed"}:
        return TO_MARK
    return TO_MARK


def has_warning_marker(status: str) -> bool:
    return status in {
        "reused_existing_3600",
        "checkpoint_best_after_timeout",
        "killed_checkpoint",
        "timeout_partial",
        "external_killed",
        "ok_non_native_basis_output",
        "ok_partial_seed_mean",
        "timeout_partial_seed_mean",
        "ok_fixed_recount",
        "ok_recount",
    }


def result_cell(entry: Entry | None, *, raw: Entry | None = None, is_raw: bool = False) -> str:
    if entry is None:
        return TO_MARK
    if is_raw:
        gates = as_int(entry.raw_gates) or as_int(entry.opt_gates)
        if gates is not None and entry.status == "ok":
            return fmt_int(gates)
        return compact_status(entry.status)

    gates = as_int(entry.opt_gates)
    if gates is not None:
        ratio = as_float(entry.ratio_vs_raw)
        if ratio is None and raw is not None:
            raw_g = as_int(raw.raw_gates) or as_int(raw.opt_gates)
            if raw_g:
                ratio = gates / raw_g
        marker = "$^{\\dagger}$" if has_warning_marker(entry.status) else ""
        if ratio is None:
            return f"{fmt_int(gates)}{marker}"
        return f"{fmt_int(gates)} ({fmt_percent(ratio)}){marker}"
    return compact_status(entry.status)


def normalize_appendix_row(row: dict[str, str]) -> dict[str, str]:
    benchmark = row["benchmark_name"]
    unable_columns = UNABLE_INPUT_COLUMNS.get(benchmark, set())
    for key, value in list(row.items()):
        if key in {"example", "benchmark_name", "family"}:
            continue
        if key in unable_columns:
            row[key] = UG_MARK
        elif value in {"--", "\\textsc{pi}", "\\textsc{err}"}:
            row[key] = TO_MARK
    return row


def cell_for(
    entries: dict[tuple[str, str, str], Entry],
    benchmark: str,
    basis: str,
    slot: str,
) -> str:
    raw = entries.get((benchmark, basis, "raw"))
    return result_cell(entries.get((benchmark, basis, slot)), raw=raw, is_raw=(slot == "raw"))


def has_numeric_result(entry: Entry | None) -> bool:
    return entry is not None and as_int(entry.opt_gates) is not None


def has_complete_numeric_result(entry: Entry | None) -> bool:
    return has_numeric_result(entry) and entry.status not in {
        "ok_partial_seed_mean",
        "timeout_partial_seed_mean",
    }


def has_basis_content(entries: dict[tuple[str, str, str], Entry], benchmark: str, basis: str) -> bool:
    slots = ["raw", "ours"] + (NAM_TOOLS if basis == "NAM" else IBM_TOOLS)
    return any(
        has_numeric_result(entries.get((benchmark, basis, slot)))
        or as_int(entries.get((benchmark, basis, slot)).raw_gates if entries.get((benchmark, basis, slot)) else None) is not None
        for slot in slots
    )


def tex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def benchmark_sort_key(benchmark: str) -> tuple[int, str]:
    if benchmark in BENCHMARK_ORDER:
        return (BENCHMARK_ORDER.index(benchmark), benchmark)
    return (10_000, benchmark)


def appendix_group_sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    benchmark = row["benchmark_name"]
    group_index, benchmark_index = APPENDIX_GROUP_RANK.get(
        benchmark,
        (len(APPENDIX_GROUPS), 10_000),
    )
    return (group_index, benchmark_index, benchmark)


def grouped_appendix_rows(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    by_benchmark = {row["benchmark_name"]: row for row in rows}
    grouped: list[tuple[str, list[dict[str, str]]]] = []
    used: set[str] = set()
    for group_name, benchmarks in APPENDIX_GROUPS:
        group_rows = [by_benchmark[benchmark] for benchmark in benchmarks if benchmark in by_benchmark]
        if group_rows:
            grouped.append((group_name, group_rows))
            used.update(row["benchmark_name"] for row in group_rows)
    extras = sorted(
        [row for row in rows if row["benchmark_name"] not in used],
        key=appendix_group_sort_key,
    )
    if extras:
        grouped.append(("Other", extras))
    return grouped


def appendix_group_header(group_name: str, width: int) -> str:
    return (
        f"\\addlinespace[3pt]\n"
        f"\\multicolumn{{{width}}}{{c}}{{\\rule{{0pt}}{{2.4ex}}"
        f"\\raisebox{{0.45ex}}{{\\rule{{0.18\\linewidth}}{{0.35pt}}}}"
        f"\\quad\\textbf{{\\footnotesize {group_name}}}\\quad"
        f"\\raisebox{{0.45ex}}{{\\rule{{0.18\\linewidth}}{{0.35pt}}}}}}\\\\[2pt]"
    )


def build_rows() -> list[dict[str, str]]:
    entries, family_map, benchmarks = load_entries()
    rows: list[dict[str, str]] = []
    for benchmark in sorted(benchmarks, key=benchmark_sort_key):
        if benchmark in DROP_BENCHMARKS:
            continue
        if not (
            has_basis_content(entries, benchmark, "NAM")
            or has_basis_content(entries, benchmark, "IBM")
        ):
            continue
        label = LABELS.get(benchmark, tex_escape(benchmark))
        row = {
            "example": label,
            "benchmark_name": benchmark,
            "family": family_map.get((benchmark, "NAM")) or family_map.get((benchmark, "IBM")) or "",
            "nam_raw": cell_for(entries, benchmark, "NAM", "raw"),
            "nam_ours": cell_for(entries, benchmark, "NAM", "ours"),
            "nam_voqc": cell_for(entries, benchmark, "NAM", "voqc"),
            "nam_feynman": cell_for(entries, benchmark, "NAM", "feynman"),
            "nam_wisq": cell_for(entries, benchmark, "NAM", "wisq"),
            "nam_quartz": cell_for(entries, benchmark, "NAM", "quartz"),
            "nam_quizx": cell_for(entries, benchmark, "NAM", "quizx"),
            "ibm_raw": cell_for(entries, benchmark, "IBM", "raw"),
            "ibm_ours": cell_for(entries, benchmark, "IBM", "ours"),
            "ibm_qiskit_opt3": cell_for(entries, benchmark, "IBM", "qiskit_opt3"),
            "ibm_pytket": cell_for(entries, benchmark, "IBM", "pytket"),
            "ibm_bqskit": cell_for(entries, benchmark, "IBM", "bqskit"),
        }
        rows.append(normalize_appendix_row(row))
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "example",
        "benchmark_name",
        "family",
        "nam_raw",
        "nam_ours",
        "nam_voqc",
        "nam_feynman",
        "nam_wisq",
        "nam_quartz",
        "nam_quizx",
        "ibm_raw",
        "ibm_ours",
        "ibm_qiskit_opt3",
        "ibm_pytket",
        "ibm_bqskit",
    ]
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[p]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.45pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\caption{Appendix CP1A/CP1B channel-LCU optimization results. Cells report total native-basis gate count with optimized/raw ratio in parentheses; raw columns report the raw denominator. NAM tools use the $\\{R_z,H,X,\\mathrm{CNOT}\\}$ basis and IBM tools use the $\\{U_1,U_2,U_3,\\mathrm{CNOT}\\}$ basis. Ours uses the 2026-07-06 Matrixsum phase refresh where available.}",
        "\\label{tab:channel-lcu-appendix-full}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llllllllllll}",
        "\\toprule",
        " & \\multicolumn{6}{c}{NAM basis} & \\multicolumn{5}{c}{IBM basis} \\\\",
        "\\cmidrule(lr){2-7}\\cmidrule(lr){8-12}",
        "Example & Raw & Ours & VOQC & Feynman & WISQ & Quartz & Raw & Ours & Qiskit opt3 & pytket & BQSKit \\\\",
        "\\midrule",
    ]
    for group_name, group_rows in grouped_appendix_rows(rows):
        lines.append(appendix_group_header(group_name, 12))
        for row in group_rows:
            lines.append(
                " & ".join(
                    [
                        row["example"],
                        row["nam_raw"],
                        row["nam_ours"],
                        row["nam_voqc"],
                        row["nam_feynman"],
                        row["nam_wisq"],
                        row["nam_quartz"],
                        row["ibm_raw"],
                        row["ibm_ours"],
                        row["ibm_qiskit_opt3"],
                        row["ibm_pytket"],
                        row["ibm_bqskit"],
                    ]
                )
                + " \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{2pt}",
            "\\begin{minipage}{\\textwidth}",
            "\\scriptsize",
            "\\emph{Notes.} $^{\\dagger}$ marks a retained partial/checkpoint, killed, non-native-output, or reused-3600s result. \\textsc{to}: timeout/no effective optimizer output within the run budget; \\textsc{ug}: unable to generate the corresponding raw/input circuit. Ratios are always within the same basis; NAM and IBM absolute counts should not be compared directly.",
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_basis_tex(rows: list[dict[str, str]], path: Path, basis: str) -> None:
    basis_l = basis.lower()
    if basis == "NAM":
        columns = [
            ("Raw", "nam_raw"),
            ("Ours", "nam_ours"),
            ("VOQC", "nam_voqc"),
            ("Feynman", "nam_feynman"),
            ("WISQ", "nam_wisq"),
            ("Quartz", "nam_quartz"),
        ]
        tools_note = "$\\{R_z,H,X,\\mathrm{CNOT}\\}$"
        label = "tab:channel-lcu-appendix-full-nam"
    else:
        columns = [
            ("Raw", "ibm_raw"),
            ("Ours", "ibm_ours"),
            ("Qiskit opt3", "ibm_qiskit_opt3"),
            ("pytket", "ibm_pytket"),
            ("BQSKit", "ibm_bqskit"),
        ]
        tools_note = "$\\{U_1,U_2,U_3,\\mathrm{CNOT}\\}$"
        label = "tab:channel-lcu-appendix-full-ibm"

    filtered = [row for row in rows if any(row[key] != "--" for _, key in columns)]
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[p]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        f"\\caption{{Appendix CP1A/CP1B channel-LCU optimization results in the {basis} native basis. Cells report total {basis} gate count with optimized/raw ratio in parentheses; the raw column reports the denominator. {basis} circuits use the {tools_note} basis. Ours uses the 2026-07-06 Matrixsum phase refresh where available.}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * (1 + len(columns)) + "}",
        "\\toprule",
        "Example & " + " & ".join(title for title, _ in columns) + " \\\\",
        "\\midrule",
    ]
    for group_name, group_rows in grouped_appendix_rows(filtered):
        lines.append(appendix_group_header(group_name, 1 + len(columns)))
        for row in group_rows:
            lines.append(
                " & ".join([row["example"], *[row[key] for _, key in columns]]) + " \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{2pt}",
            "\\begin{minipage}{\\textwidth}",
            "\\scriptsize",
            "\\emph{Notes.} $^{\\dagger}$ marks a retained partial/checkpoint, killed, non-native-output, or reused-3600s result. \\textsc{to}: timeout/no effective optimizer output within the run budget; \\textsc{ug}: unable to generate the corresponding raw/input circuit. Ratios are always within the same basis.",
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def complete_time_rows() -> list[dict[str, str]]:
    entries, _family_map, benchmarks = load_entries()
    runtime_overrides = load_runtime_overrides()
    rows: list[dict[str, str]] = []
    for benchmark in sorted(benchmarks, key=benchmark_sort_key):
        if benchmark in DROP_BENCHMARKS:
            continue
        nam_slots = ["ours", *NAM_TOOLS]
        ibm_slots = ["ours", *IBM_TOOLS]
        nam_ok = all(has_complete_numeric_result(entries.get((benchmark, "NAM", slot))) for slot in nam_slots)
        ibm_ok = all(has_complete_numeric_result(entries.get((benchmark, "IBM", slot))) for slot in ibm_slots)
        label = LABELS.get(benchmark, tex_escape(benchmark))
        if nam_ok:
            row = {"example": label, "benchmark_name": benchmark, "basis": "NAM"}
            for slot in nam_slots:
                entry = entries[(benchmark, "NAM", slot)]
                row[slot] = fmt_time(
                    ("" if slot == "ours" else runtime_overrides.get((benchmark, "NAM", slot), ""))
                    or entry.optimizer_time_sec
                    or entry.total_time_sec
                    or (runtime_overrides.get((benchmark, "NAM", slot), "") if slot == "ours" else "")
                )
            if all(row.get(slot) for slot in nam_slots):
                rows.append(row)
        if ibm_ok:
            row = {"example": label, "benchmark_name": benchmark, "basis": "IBM"}
            for slot in ibm_slots:
                entry = entries[(benchmark, "IBM", slot)]
                row[slot] = fmt_time(
                    ("" if slot == "ours" else runtime_overrides.get((benchmark, "IBM", slot), ""))
                    or entry.optimizer_time_sec
                    or entry.total_time_sec
                    or (runtime_overrides.get((benchmark, "IBM", slot), "") if slot == "ours" else "")
                )
            if all(row.get(slot) for slot in ibm_slots):
                rows.append(row)
    return rows


def write_time_csv(rows: list[dict[str, str]], path: Path) -> None:
    fields = [
        "example",
        "benchmark_name",
        "basis",
        "ours",
        "voqc",
        "feynman",
        "wisq",
        "quartz",
        "qiskit_opt3",
        "pytket",
        "bqskit",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_time_tex(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\caption{Compilation/optimization time for instances where all tools in a native-basis group produced an output. Times are in seconds and exclude raw-circuit generation when the source separates it.}",
        "\\label{tab:channel-lcu-tool-runtime-complete}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\multicolumn{6}{c}{NAM basis} \\\\",
        "\\midrule",
        "Example & Ours & VOQC & Feynman & WISQ & Quartz \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["basis"] != "NAM":
            continue
        lines.append(
            " & ".join(
                [
                    row["example"],
                    row.get("ours", "--") or "--",
                    row.get("voqc", "--") or "--",
                    row.get("feynman", "--") or "--",
                    row.get("wisq", "--") or "--",
                    row.get("quartz", "--") or "--",
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{3pt}",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "\\multicolumn{5}{c}{IBM basis} \\\\",
            "\\midrule",
            "Example & Ours & Qiskit opt3 & pytket & BQSKit \\\\",
            "\\midrule",
        ]
    )
    for row in rows:
        if row["basis"] != "IBM":
            continue
        lines.append(
            " & ".join(
                [
                    row["example"],
                    row.get("ours", "--") or "--",
                    row.get("qiskit_opt3", "--") or "--",
                    row.get("pytket", "--") or "--",
                    row.get("bqskit", "--") or "--",
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_csv(rows, OUT_CSV)
    write_basis_tex(rows, OUT_NAM_TEX, "NAM")
    write_basis_tex(rows, OUT_IBM_TEX, "IBM")
    OUT_TEX.write_text(
        OUT_NAM_TEX.read_text(encoding="utf-8")
        + "\n\n"
        + OUT_IBM_TEX.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    time_rows = complete_time_rows()
    write_time_csv(time_rows, OUT_TIME_CSV)
    write_time_tex(time_rows, OUT_TIME_TEX)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")
    print(f"wrote split LaTeX tables to {OUT_TEX}, {OUT_NAM_TEX}, and {OUT_IBM_TEX}")
    print(f"wrote {len(time_rows)} complete runtime rows to {OUT_TIME_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
