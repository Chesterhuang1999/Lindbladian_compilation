#!/usr/bin/env python3
"""Run the CP2 TFIM_8 internal ablation in the IBM gate set.

This mirrors the HRW_20 RQ2 ablation runner, but builds the periodic TFIM
local-decay channel at n=8.  R-enabled configurations use
gatecost_rewrite_search(strategy="beam"), and all metric QASM is lowered and
counted in the IBM {u1,u2,u3,cx} gate set.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, BASELINE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from channel_IR import Lindbladian, channel as ChannelIR  # type: ignore  # noqa: E402
from experiment_protocol import rewrite_cache_paths
import channel_LCU  # type: ignore  # noqa: E402
from run_cp2_hrw20_ablation import (  # type: ignore  # noqa: E402
    CONFIGS,
    CONFIG_BY_NAME,
    CONFIG_BY_SLUG,
    IBM_BASIS,
    IBM_TRANSPILER_BASIS,
    ConfigSpec,
    add_ratios,
    build_config_circuit,
    channel_metrics,
    count_ibm_circuit,
    dump_pickle,
    json_default,
    load_pickle,
    read_json,
    run_git,
    write_csv,
    write_json,
)


DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "eval_redesign" / "cp2_rq2_main_ablation_tfim8"


def benchmark_dir_name(n: int) -> str:
    return f"tfim_{n}"


def benchmark_name(n: int) -> str:
    return f"TFIM_{n}"


def float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_periodic_tfim_channel(num_qubits: int, delta_t: float) -> tuple[list[Any], dict[str, Any]]:
    h_terms: list[tuple[str, complex]] = []
    l_terms: list[list[tuple[str, complex]]] = []
    gamma = np.sqrt(0.1) / 2

    for i in range(num_qubits):
        z_ind = [i, (i + 1) % num_qubits]
        z_str = "".join("Z" if j in z_ind else "I" for j in range(num_qubits))
        x_str = "".join("X" if j == i else "I" for j in range(num_qubits))
        y_str = "".join("Y" if j == i else "I" for j in range(num_qubits))
        h_terms.append((z_str, -1.0))
        h_terms.append((x_str, -1.0))
        l_terms.append([(x_str, gamma), (y_str, -1j * gamma)])

    tfim_lind = Lindbladian(h_terms, l_terms)
    channel_lind, success_prob, coeff_sums = channel_LCU.Lindblad_to_channel(tfim_lind, delta_t)
    return channel_lind.channels[0][1], {
        "gamma": float(gamma),
        "delta_t": float(delta_t),
        "success_prob": float(success_prob),
        "coeff_sums": [float(x) for x in coeff_sums],
    }


def cache_paths(out_dir: Path, n: int, delta_t: float) -> dict[str, Path]:
    bench_dir = out_dir / benchmark_dir_name(n)
    cache_dir = bench_dir / "cache"
    prefix = f"tfim_n{n}_dt{float_tag(delta_t)}"
    return {
        "cache_dir": cache_dir,
        "raw_kraus": cache_dir / f"{prefix}_raw_kraus.pkl",
        "raw_metrics": cache_dir / f"{prefix}_raw_metrics.json",
        "rewrite_kraus": cache_dir / f"{prefix}_gatecost_beam_rewritten_kraus.pkl",
        "rewrite_summary": cache_dir / f"{prefix}_gatecost_beam_rewrite_summary.json",
    }


def ensure_raw_cache(out_dir: Path, n: int, delta_t: float) -> tuple[list[Any], dict[str, Any]]:
    paths = cache_paths(out_dir, n, delta_t)
    if paths["raw_kraus"].exists() and paths["raw_metrics"].exists():
        return load_pickle(paths["raw_kraus"]), read_json(paths["raw_metrics"])

    started = time.perf_counter()
    kraus_ops, model_info = build_periodic_tfim_channel(n, delta_t)
    metrics: dict[str, Any] = channel_metrics(kraus_ops)
    metrics["build_channel_time_s"] = float(time.perf_counter() - started)
    metrics["benchmark_name"] = benchmark_name(n)
    metrics["generator"] = "periodic_tfim_local_decay_lindblad_to_channel"
    metrics["model"] = model_info
    metrics["delta_t"] = float(delta_t)
    metrics["gamma"] = float(model_info["gamma"])
    metrics["success_prob"] = float(model_info["success_prob"])
    metrics["coeff_sums"] = model_info["coeff_sums"]
    dump_pickle(paths["raw_kraus"], kraus_ops)
    write_json(paths["raw_metrics"], metrics)
    return kraus_ops, metrics


def ensure_rewrite_cache(
    out_dir: Path,
    n: int,
    delta_t: float,
    *,
    beam_width: int,
    max_steps: int,
    tol: float,
    structure: str = "tree",
) -> tuple[list[Any], dict[str, Any]]:
    paths = rewrite_cache_paths(cache_paths(out_dir, n, delta_t), structure=structure,
                               beam_width=beam_width, max_steps=max_steps, tol=tol)
    if paths["rewrite_kraus"].exists() and paths["rewrite_summary"].exists():
        return load_pickle(paths["rewrite_kraus"]), read_json(paths["rewrite_summary"])

    raw_kraus_ops, raw_metrics = ensure_raw_cache(out_dir, n, delta_t)
    started = time.perf_counter()
    rewrite_source = ChannelIR(copy.deepcopy(raw_kraus_ops))
    rewrite_source.zero_elim()
    result = rewrite_source.gatecost_rewrite_search(
        strategy="beam",
        beam_width=beam_width,
        max_steps=max_steps,
        tol=tol,
        structure=structure,
    )
    rewritten = ChannelIR(copy.deepcopy(raw_kraus_ops))
    rewritten.apply_rewrite_result(result, tol=tol)
    rewritten.zero_elim()
    rewrite_time = time.perf_counter() - started
    rewrite_metrics = channel_metrics(rewritten.kraus_ops)
    summary = {
        "method": "gatecost_rewrite_search",
        "strategy": "beam",
        "beam_width": int(beam_width),
        "max_steps": int(max_steps),
        "tol": float(tol),
        "time_s": float(rewrite_time),
        "raw_channel_metrics": raw_metrics,
        "rewritten_channel_metrics": rewrite_metrics,
        "initial_support": int(result.get("initial_support", 0)),
        "final_support": int(result.get("final_support", 0)),
        "cost_model": result["cost_model"],
        "cost_settings": result["cost_settings"],
        "initial_cost": float(result.get("initial_cost", 0)),
        "final_cost": float(result.get("final_cost", 0)),
        "steps": int(len(result.get("steps", []))),
        "termination": result.get("termination", {}),
        "result": result,
    }
    dump_pickle(paths["rewrite_kraus"], rewritten.kraus_ops)
    write_json(paths["rewrite_summary"], summary)
    return rewritten.kraus_ops, summary


def metric_row(
    *,
    cfg: ConfigSpec,
    status: str,
    error_msg: str,
    n: int,
    delta_t: float,
    raw_metrics: dict[str, Any],
    rewrite_summary: dict[str, Any] | None,
    build_time_s: float,
    transpile_time_s: float,
    count_time_s: float,
    total_time_s: float,
    block_timing: dict[str, Any] | None,
    qasm_path: Path | None,
    stats: dict[str, Any] | None,
    export_meta: dict[str, Any] | None,
    qubit_indexes: Any,
) -> dict[str, Any]:
    rewrite_time_s = 0.0
    rewrite_initial_support = raw_metrics.get("pauli_terms_total")
    rewrite_final_support = raw_metrics.get("pauli_terms_total")
    rewrite_steps = 0
    rewrite_initial_cost = None
    rewrite_final_cost = None
    if cfg.rewrite and rewrite_summary is not None:
        rewrite_time_s = float(rewrite_summary.get("time_s", 0.0))
        rewrite_initial_support = rewrite_summary.get("initial_support")
        rewrite_final_support = rewrite_summary.get("final_support")
        rewrite_steps = int(rewrite_summary.get("steps", 0))
        rewrite_initial_cost = rewrite_summary.get("initial_cost")
        rewrite_final_cost = rewrite_summary.get("final_cost")

    from run_cp2_hrw20_ablation import summarize_block_timings  # type: ignore

    timing = block_timing or summarize_block_timings([], build_time_s)
    ops = {} if stats is None else stats.get("ops", {})
    row = {
        "checkpoint": "cp2_rq2_main_ablation",
        "benchmark_family": "Algorithm",
        "benchmark_name": benchmark_name(n),
        "benchmark_scope": "main",
        "n_or_size": int(n),
        "seed": "",
        "delta_t": float(delta_t),
        "basis": "ibm",
        "selected_basis_reason": "fixed_ibm_gate_set_no_external_optimizers",
        "config": cfg.name,
        "rewrite_enabled": bool(cfg.rewrite),
        "flat_enabled": bool(cfg.flat),
        "order_enabled": bool(cfg.order),
        "rewrite_search": bool(cfg.rewrite),
        "rewrite_method": "gatecost_rewrite_search" if cfg.rewrite else "none",
        "rewrite_strategy": "beam" if cfg.rewrite else "none",
        "structure": cfg.structure,
        "opt": cfg.opt,
        "status": status,
        "error_msg": error_msg,
        "total_gates": None if stats is None else int(stats["metric_total"]),
        "gate_total": None if stats is None else int(stats["metric_total"]),
        "cx": None if stats is None else int(stats["ibm"]["cx"]),
        "cx_count": None if stats is None else int(stats["ibm"]["cx"]),
        "u1": int(ops.get("u1", 0)),
        "u2": int(ops.get("u2", 0)),
        "u3": int(ops.get("u3", 0)),
        "depth": None if export_meta is None else int(export_meta["depth"]),
        "width": None if stats is None else int(stats["num_qubits"]),
        "ancilla_count": "",
        "non_clifford_count": None if stats is None else int(stats["clifford"]["non_clifford"]),
        "reset_count": int(ops.get("reset", 0)),
        "lambda": "",
        "ratio_vs_basic": "",
        "raw_kraus_count": int(raw_metrics.get("kraus_count", 0)),
        "raw_pauli_terms_total": int(raw_metrics.get("pauli_terms_total", 0)),
        "raw_max_terms_per_kraus": int(raw_metrics.get("max_terms_per_kraus", 0)),
        "raw_success_prob": raw_metrics.get("success_prob", ""),
        "raw_gamma": raw_metrics.get("gamma", ""),
        "rewrite_initial_support": rewrite_initial_support,
        "rewrite_final_support": rewrite_final_support,
        "rewrite_initial_cost": rewrite_initial_cost,
        "rewrite_final_cost": rewrite_final_cost,
        "rewrite_steps": int(rewrite_steps),
        "construction_time_sec": float(build_time_s),
        "rewrite_time_sec": float(rewrite_time_s),
        "structure_time_sec": float(timing["channel_lcu_without_block_encoding_time_sec"]) if cfg.flat else 0.0,
        "ordering_time_sec": float(timing["block_encoding_time_sec"]) if cfg.order else 0.0,
        "channel_lcu_total_time_sec": float(timing["channel_lcu_total_time_sec"]),
        "block_encoding_time_sec": float(timing["block_encoding_time_sec"]),
        "block_encoding_total_time_sec": float(timing["block_encoding_total_time_sec"]),
        "block_encoding_mean_time_sec": float(timing["block_encoding_mean_time_sec"]),
        "block_encoding_max_time_sec": float(timing["block_encoding_max_time_sec"]),
        "block_encoding_kraus_count": int(timing["block_encoding_kraus_count"]),
        "channel_lcu_without_block_encoding_time_sec": float(timing["channel_lcu_without_block_encoding_time_sec"]),
        "per_kraus_block_encoding_times_sec": timing["per_kraus_block_encoding_times_sec"],
        "per_kraus_block_encoding_details_json": timing["per_kraus_block_encoding_details_json"],
        "transpile_time_sec": float(transpile_time_s),
        "count_time_sec": float(count_time_s),
        "total_time_sec": float(total_time_s),
        "build_time_s": float(build_time_s),
        "transpile_time_s": float(transpile_time_s),
        "count_time_s": float(count_time_s),
        "total_time_s": float(total_time_s),
        "execution_time_s": float(total_time_s),
        "timeout": False,
        "qasm_path": "" if qasm_path is None else str(qasm_path),
        "qubit_indexes": qubit_indexes,
        "notes": "Matrix-order uses block_encoding.BlockEncoding.circuit(opt='Matrix-order') with expand_large_controlled_gates(..., if_optimize=True)." if cfg.order else "",
    }
    return row


def run_config(args: argparse.Namespace, cfg: ConfigSpec) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    bench_dir = out_dir / benchmark_dir_name(args.num_qubits)
    rows_dir = bench_dir / "rows"
    raw_runs_dir = bench_dir / "raw_runs"
    qasm_dir = bench_dir / "qasm"
    rows_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_dir.mkdir(parents=True, exist_ok=True)
    qasm_dir.mkdir(parents=True, exist_ok=True)

    attempt_started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    attempt_started = time.perf_counter()
    command = " ".join(sys.argv)
    row: dict[str, Any]
    try:
        raw_kraus_ops, raw_metrics = ensure_raw_cache(out_dir, args.num_qubits, args.delta_t)
        rewrite_summary = None
        if cfg.rewrite:
            kraus_ops, rewrite_summary = ensure_rewrite_cache(
                out_dir,
                args.num_qubits,
                args.delta_t,
                beam_width=args.beam_width,
                max_steps=args.max_steps,
                tol=args.tol,
                structure=cfg.structure,
            )
        else:
            kraus_ops = raw_kraus_ops

        build_started = time.perf_counter()
        qc, qubit_indexes, build_time_s, block_timing = build_config_circuit(kraus_ops, cfg)
        build_elapsed = time.perf_counter() - build_started
        if abs(build_elapsed - build_time_s) > 1e-6:
            build_time_s = build_elapsed
            block_timing = dict(block_timing)
            block_timing["channel_lcu_total_time_sec"] = float(build_time_s)
            block_timing["channel_lcu_without_block_encoding_time_sec"] = float(
                max(0.0, build_time_s - float(block_timing.get("block_encoding_time_sec", 0.0)))
            )

        qasm_path = qasm_dir / (
            f"tfim_{args.num_qubits}_{cfg.slug}_ibm_opt{args.transpile_optimization_level}.qasm"
        )
        stats, export_meta, transpile_time_s, count_time_s = count_ibm_circuit(
            qc,
            qasm_path,
            optimization_level=args.transpile_optimization_level,
        )
        if cfg.rewrite and rewrite_summary is not None:
            total_time_s = float(rewrite_summary.get("time_s", 0.0)) + build_time_s + transpile_time_s + count_time_s
        else:
            total_time_s = build_time_s + transpile_time_s + count_time_s

        row = metric_row(
            cfg=cfg,
            status="ok",
            error_msg="",
            n=args.num_qubits,
            delta_t=args.delta_t,
            raw_metrics=raw_metrics,
            rewrite_summary=rewrite_summary,
            build_time_s=build_time_s,
            transpile_time_s=transpile_time_s,
            count_time_s=count_time_s,
            total_time_s=total_time_s,
            block_timing=block_timing,
            qasm_path=qasm_path,
            stats=stats,
            export_meta=export_meta,
            qubit_indexes=qubit_indexes,
        )
    except Exception as exc:  # Keep failed rows visible for aggregation.
        try:
            _, raw_metrics = ensure_raw_cache(out_dir, args.num_qubits, args.delta_t)
        except Exception:
            raw_metrics = {"kraus_count": 0, "pauli_terms_total": 0, "max_terms_per_kraus": 0}
        row = metric_row(
            cfg=cfg,
            status="failed",
            error_msg=f"{type(exc).__name__}: {exc}",
            n=args.num_qubits,
            delta_t=args.delta_t,
            raw_metrics=raw_metrics,
            rewrite_summary=None,
            build_time_s=0.0,
            transpile_time_s=0.0,
            count_time_s=0.0,
            total_time_s=time.perf_counter() - attempt_started,
            block_timing=None,
            qasm_path=None,
            stats=None,
            export_meta=None,
            qubit_indexes=None,
        )

    row_path = rows_dir / f"{cfg.slug}.json"
    write_json(row_path, row)
    raw_run = {
        "benchmark": benchmark_name(args.num_qubits),
        "configuration": cfg.name,
        "command": command,
        "start_time": attempt_started_wall,
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": row["status"],
        "timeout": row["timeout"],
        "exception_or_stderr_summary": row["error_msg"],
        "output_files_produced": [str(row_path)] + ([row["qasm_path"]] if row.get("qasm_path") else []),
        "row": row,
    }
    write_json(raw_runs_dir / f"{cfg.slug}.json", raw_run)
    print(json.dumps(row, indent=2, default=json_default))
    return row


def load_rows(out_dir: Path, n: int, delta_t: float) -> list[dict[str, Any]]:
    rows_dir = out_dir / benchmark_dir_name(n) / "rows"
    rows = []
    for cfg in CONFIGS:
        path = rows_dir / f"{cfg.slug}.json"
        if path.exists():
            rows.append(read_json(path))
        else:
            rows.append(
                {
                    "checkpoint": "cp2_rq2_main_ablation",
                    "benchmark_family": "Algorithm",
                    "benchmark_name": benchmark_name(n),
                    "benchmark_scope": "main",
                    "n_or_size": int(n),
                    "seed": "",
                    "delta_t": float(delta_t),
                    "basis": "ibm",
                    "selected_basis_reason": "fixed_ibm_gate_set_no_external_optimizers",
                    "config": cfg.name,
                    "rewrite_enabled": cfg.rewrite,
                    "flat_enabled": cfg.flat,
                    "order_enabled": cfg.order,
                    "rewrite_search": cfg.rewrite,
                    "rewrite_method": "gatecost_rewrite_search" if cfg.rewrite else "none",
                    "rewrite_strategy": "beam" if cfg.rewrite else "none",
                    "structure": cfg.structure,
                    "opt": cfg.opt,
                    "status": "not_run",
                    "error_msg": "row file missing",
                    "total_gates": "",
                    "gate_total": "",
                    "cx": "",
                    "cx_count": "",
                    "u1": "",
                    "u2": "",
                    "u3": "",
                    "depth": "",
                    "width": "",
                    "ancilla_count": "",
                    "non_clifford_count": "",
                    "reset_count": "",
                    "lambda": "",
                    "ratio_vs_basic": "",
                    "raw_kraus_count": "",
                    "raw_pauli_terms_total": "",
                    "raw_max_terms_per_kraus": "",
                    "raw_success_prob": "",
                    "raw_gamma": "",
                    "construction_time_sec": "",
                    "rewrite_time_sec": "",
                    "structure_time_sec": "",
                    "ordering_time_sec": "",
                    "channel_lcu_total_time_sec": "",
                    "block_encoding_time_sec": "",
                    "block_encoding_total_time_sec": "",
                    "block_encoding_mean_time_sec": "",
                    "block_encoding_max_time_sec": "",
                    "block_encoding_kraus_count": "",
                    "channel_lcu_without_block_encoding_time_sec": "",
                    "per_kraus_block_encoding_times_sec": "",
                    "per_kraus_block_encoding_details_json": "",
                    "transpile_time_sec": "",
                    "count_time_sec": "",
                    "total_time_sec": "",
                    "execution_time_s": "",
                    "timeout": False,
                    "qasm_path": "",
                    "notes": "",
                }
            )
    return rows


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows = load_rows(out_dir, args.num_qubits, args.delta_t)
    add_ratios(rows)

    summary_columns = [
        "checkpoint",
        "benchmark_family",
        "benchmark_name",
        "benchmark_scope",
        "n_or_size",
        "seed",
        "delta_t",
        "basis",
        "selected_basis_reason",
        "config",
        "rewrite_enabled",
        "flat_enabled",
        "order_enabled",
        "rewrite_search",
        "rewrite_method",
        "rewrite_strategy",
        "structure",
        "opt",
        "total_gates",
        "gate_total",
        "cx",
        "cx_count",
        "u1",
        "u2",
        "u3",
        "depth",
        "width",
        "ancilla_count",
        "non_clifford_count",
        "lambda",
        "ratio_vs_basic",
        "raw_kraus_count",
        "raw_pauli_terms_total",
        "raw_max_terms_per_kraus",
        "raw_success_prob",
        "raw_gamma",
        "rewrite_initial_support",
        "rewrite_final_support",
        "rewrite_initial_cost",
        "rewrite_final_cost",
        "rewrite_steps",
        "construction_time_sec",
        "rewrite_time_sec",
        "structure_time_sec",
        "ordering_time_sec",
        "channel_lcu_total_time_sec",
        "block_encoding_time_sec",
        "block_encoding_total_time_sec",
        "block_encoding_mean_time_sec",
        "block_encoding_max_time_sec",
        "block_encoding_kraus_count",
        "channel_lcu_without_block_encoding_time_sec",
        "per_kraus_block_encoding_times_sec",
        "per_kraus_block_encoding_details_json",
        "transpile_time_sec",
        "count_time_sec",
        "total_time_sec",
        "execution_time_s",
        "timeout",
        "status",
        "error_msg",
        "qasm_path",
        "notes",
    ]
    write_csv(out_dir / "summary.csv", rows, summary_columns)
    write_csv(out_dir / "rq2a_internal_ablation.csv", rows, summary_columns)

    raw_runs_dir = out_dir / benchmark_dir_name(args.num_qubits) / "raw_runs"
    with (out_dir / "raw_runs.jsonl").open("w", encoding="utf-8") as out_file:
        for cfg in CONFIGS:
            path = raw_runs_dir / f"{cfg.slug}.json"
            if path.exists():
                out_file.write(json.dumps(read_json(path), default=json_default) + "\n")

    row_by_config = {row["config"]: row for row in rows}
    basic = row_by_config.get("Basic", {})
    all_row = row_by_config.get("Rewrite + Flat + Order", {})
    main_row = {
        "benchmark_family": "Algorithm",
        "benchmark_name": benchmark_name(args.num_qubits),
        "basis": "ibm",
        "delta_t": float(args.delta_t),
        "basic_gates": basic.get("total_gates", ""),
        "basic_non_clifford_count": basic.get("non_clifford_count", ""),
        "basic_total_time_s": basic.get("total_time_sec", ""),
        "Rewrite_only_ratio": row_by_config.get("Rewrite only", {}).get("ratio_vs_basic", ""),
        "Flat_only_ratio": row_by_config.get("Flat only", {}).get("ratio_vs_basic", ""),
        "Order_only_ratio": row_by_config.get("Order only", {}).get("ratio_vs_basic", ""),
        "Rewrite_Flat_ratio": row_by_config.get("Rewrite + Flat", {}).get("ratio_vs_basic", ""),
        "Rewrite_Order_ratio": row_by_config.get("Rewrite + Order", {}).get("ratio_vs_basic", ""),
        "Flat_Order_ratio": row_by_config.get("Flat + Order", {}).get("ratio_vs_basic", ""),
        "Rewrite_Flat_Order_ratio": all_row.get("ratio_vs_basic", ""),
        "best_config": "",
        "best_ratio": "",
        "All_non_clifford_count": all_row.get("non_clifford_count", ""),
        "All_total_time_s": all_row.get("total_time_sec", ""),
        "status": "ok" if all(row.get("status") == "ok" for row in rows) else "partial",
        "note": f"{benchmark_name(args.num_qubits)} running example; IBM gate set only; R uses gatecost_rewrite_search(strategy='beam').",
    }
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("ratio_vs_basic") not in ("", None)
    ]
    if ok_rows:
        best = min(ok_rows, key=lambda row: float(row["ratio_vs_basic"]))
        main_row["best_config"] = best["config"]
        main_row["best_ratio"] = best["ratio_vs_basic"]

    main_columns = [
        "benchmark_family",
        "benchmark_name",
        "basis",
        "delta_t",
        "basic_gates",
        "basic_non_clifford_count",
        "basic_total_time_s",
        "Rewrite_only_ratio",
        "Flat_only_ratio",
        "Order_only_ratio",
        "Rewrite_Flat_ratio",
        "Rewrite_Order_ratio",
        "Flat_Order_ratio",
        "Rewrite_Flat_Order_ratio",
        "best_config",
        "best_ratio",
        "All_non_clifford_count",
        "All_total_time_s",
        "status",
        "note",
    ]
    write_csv(out_dir / "main_table.csv", [main_row], main_columns)
    write_csv(out_dir / "rq2a_main_table.csv", [main_row], main_columns)

    known_failures = [
        {"config": row["config"], "status": row["status"], "error_msg": row.get("error_msg", "")}
        for row in rows
        if row.get("status") != "ok"
    ]
    manifest = {
        "checkpoint": "cp2_rq2_main_ablation",
        "agent": "CP2-RQ2-Main-Ablation-TFIM8",
        "repo_commit": run_git(["rev-parse", "HEAD"]),
        "dirty_status": run_git(["status", "--short"]),
        "start_time": "",
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "basis": "ibm",
        "basis_gates": list(IBM_BASIS),
        "transpiler_basis": list(IBM_TRANSPILER_BASIS),
        "transpile_optimization_level": int(args.transpile_optimization_level),
        "rewrite_method": "gatecost_rewrite_search",
        "rewrite_strategy": "beam",
        "delta_t": float(args.delta_t),
        "matrix_order_guardrail": "block_encoding.BlockEncoding.circuit(opt='Matrix-order') calls expand_large_controlled_gates(..., if_optimize=True)",
        "success": not known_failures,
        "known_failures": known_failures,
    }
    write_json(out_dir / "manifest.json", manifest)

    notes = [
        f"# CP2 {benchmark_name(args.num_qubits)} Running Example Notes",
        "",
        f"- Benchmark: {benchmark_name(args.num_qubits)} / periodic TFIM local decay, delta_t={args.delta_t}.",
        "- Target/reporting basis: IBM `{u1,u2,u3,cx}` only; no external optimizer rows are included.",
        "- R uses `ChannelIR.gatecost_rewrite_search(strategy=\"beam\")`; this runner does not call `rewrite_search()`.",
        "- Matrix-order uses `block_encoding.BlockEncoding.circuit(opt=\"Matrix-order\")`, whose SELECT path calls `expand_large_controlled_gates(..., if_optimize=True)`.",
        "- F denotes the adopted channel-level optimized outer SELECT / flattening-style pass, not a new contribution.",
        "- Every completed row records total gates, CX, U1/U2/U3 counts, non-Clifford count, width, depth, and total execution time.",
        "",
        "## Status",
        "",
    ]
    for row in rows:
        notes.append(
            f"- {row['config']}: {row['status']}; gates={row.get('total_gates', '-')}; "
            f"ratio={row.get('ratio_vs_basic', '-')}; non_clifford={row.get('non_clifford_count', '-')}; "
            f"time_s={row.get('total_time_sec', '-')}"
        )
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    findings = [
        f"# RQ2 {benchmark_name(args.num_qubits)} Findings Draft",
        "",
        "## Setup",
        "",
        f"This {benchmark_name(args.num_qubits)} running example evaluates Basic, R, F, O, R+F, R+O, F+O, and All in the fixed IBM gate set. R is gate-cost-guided Kraus rewrite search, F is the adopted optimized outer SELECT / flattening-style pass, and O is per-Kraus Matrix-order.",
        "",
        "## Finding",
        "",
        "Use `rq2a_main_table.csv` after all rows complete to state which pass combination gives the smallest Basic-normalized gate ratio.",
        "",
        "## Caution",
        "",
        "R is structure-dependent, F is adopted prior/backend structure, and neutral or negative rows are preserved.",
        "",
    ]
    (out_dir / "rq2_findings_draft.md").write_text("\n".join(findings), encoding="utf-8")

    reproduce_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation",
        "",
        "CONDA=/Users/chester_hqf/miniconda3/bin/conda",
        "SCRIPT=scripts/eval_redesign/run_cp2_tfim8_ablation.py",
        "OUT=Baseline_results/eval_redesign/cp2_rq2_main_ablation_tfim8",
        "",
        "$CONDA run -n qiskit_simulate python $SCRIPT --out-dir $OUT run --config Basic",
        "$CONDA run -n qiskit_simulate python $SCRIPT --out-dir $OUT prepare-rewrite",
    ]
    for cfg in CONFIGS:
        if cfg.name == "Basic":
            continue
        reproduce_lines.append(
            f"$CONDA run -n qiskit_simulate python $SCRIPT --out-dir $OUT run --config '{cfg.name}'"
        )
    reproduce_lines.append("$CONDA run -n qiskit_simulate python $SCRIPT --out-dir $OUT aggregate")
    (out_dir / "reproduce.sh").write_text("\n".join(reproduce_lines) + "\n", encoding="utf-8")
    os.chmod(out_dir / "reproduce.sh", 0o755)

    print(f"Wrote aggregate outputs under {out_dir}")


def parse_config(value: str) -> ConfigSpec:
    if value in CONFIG_BY_NAME:
        return CONFIG_BY_NAME[value]
    if value in CONFIG_BY_SLUG:
        return CONFIG_BY_SLUG[value]
    choices = sorted(set(CONFIG_BY_NAME) | set(CONFIG_BY_SLUG))
    raise argparse.ArgumentTypeError(f"Unknown config {value!r}. Choices: {choices}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-qubits", type=int, default=8)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--transpile-optimization-level", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one ablation configuration.")
    run_parser.add_argument("--config", type=parse_config, required=True)

    subparsers.add_parser("prepare-raw", help="Build and cache the raw TFIM channel.")
    subparsers.add_parser("prepare-rewrite", help="Build and cache gatecost beam rewrite output.")
    subparsers.add_parser("aggregate", help="Aggregate per-config row files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare-raw":
        kraus_ops, metrics = ensure_raw_cache(out_dir, args.num_qubits, args.delta_t)
        print(json.dumps({"raw_metrics": metrics, "kraus_count": len(kraus_ops)}, indent=2))
    elif args.command == "prepare-rewrite":
        _, summary = ensure_rewrite_cache(
            out_dir,
            args.num_qubits,
            args.delta_t,
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            tol=args.tol,
        )
        print(json.dumps(summary, indent=2, default=json_default))
    elif args.command == "run":
        run_config(args, args.config)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
