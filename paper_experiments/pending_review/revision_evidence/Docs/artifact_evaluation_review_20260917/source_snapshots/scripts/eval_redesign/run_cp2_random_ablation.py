#!/usr/bin/env python3
"""Run CP2 internal ablations for RandomColoring and random Pauli-Kraus cases.

The runner is intentionally IBM-only and does not dispatch external
optimizers.  Every R-enabled configuration uses
ChannelIR.gatecost_rewrite_search(strategy="beam").  Random Pauli-Kraus
aggregates all requested seeds into a mean row per configuration while keeping
the per-seed rows.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import platform
import statistics
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
LEGACY_SCRIPT_DIR = ROOT / "scripts"
for path in (SRC_DIR, BASELINE_DIR, SCRIPT_DIR, LEGACY_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_random_channel_lcu_benchmark import (  # type: ignore  # noqa: E402
    RandomChannelCase,
    RandomChannelSpec,
    build_random_kraus_ops,
)
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import channel as ChannelIR  # type: ignore  # noqa: E402
from channel_IR import channel_ensemble  # type: ignore  # noqa: E402
import channel_LCU  # type: ignore  # noqa: E402
from run_cp2_hrw20_ablation import (  # type: ignore  # noqa: E402
    CONFIGS,
    CONFIG_BY_NAME,
    CONFIG_BY_SLUG,
    IBM_BASIS,
    IBM_TRANSPILER_BASIS,
    ConfigSpec,
    channel_metrics,
    count_ibm_circuit,
    json_default,
    read_json,
    summarize_block_timings,
    timed_block_encoding_class,
    write_csv,
    write_json,
)
from run_random_coloring_channel_lcu_ours_nam import (  # type: ignore  # noqa: E402
    build_random_coloring_channel,
)


DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "eval_redesign" / "cp2_rq2_main_ablation_random_examples"
MATRIX_ORDER_NOTE = (
    "Matrix-order uses block_encoding_new.BlockEncoding.circuit(opt='Matrix-order') "
    "with expand_large_controlled_gates(..., if_optimize=True); single-term "
    "Kraus blocks use opt='No' because no Matrix-order placement is needed."
)


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def parse_config(value: str) -> ConfigSpec:
    if value in CONFIG_BY_NAME:
        return CONFIG_BY_NAME[value]
    if value in CONFIG_BY_SLUG:
        return CONFIG_BY_SLUG[value]
    choices = sorted(set(CONFIG_BY_NAME) | set(CONFIG_BY_SLUG))
    raise argparse.ArgumentTypeError(f"Unknown config {value!r}. Choices: {choices}")


def example_slug(args: argparse.Namespace) -> str:
    if args.example == "random-coloring":
        return f"random_coloring_n{args.num_vertices}_q{args.q}_{args.graph}"
    spec = RandomChannelSpec(
        args.kraus_count,
        args.hilbert_dim,
        args.first_kraus_terms,
        args.other_kraus_terms,
    )
    return f"random_pauli_kraus_{spec.label}"


def benchmark_name(args: argparse.Namespace) -> str:
    if args.example == "random-coloring":
        return f"RandomColoring_{args.graph}_n{args.num_vertices}_q{args.q}"
    return (
        "RandomPauliKraus_"
        f"N{args.kraus_count}_dim{args.hilbert_dim}_"
        f"N1_{args.first_kraus_terms}_N2_{args.other_kraus_terms}"
    )


def seed_label(seed: int | None) -> str:
    return "" if seed is None else f"seed_{seed}"


def case_dir(args: argparse.Namespace, seed: int | None = None) -> Path:
    base = Path(args.out_dir) / example_slug(args)
    if args.example == "random-pauli":
        if seed is None:
            raise ValueError("random-pauli case_dir requires a seed")
        return base / seed_label(seed)
    return base


def cache_paths(args: argparse.Namespace, seed: int | None = None) -> dict[str, Path]:
    cache_dir = case_dir(args, seed) / "cache"
    prefix = example_slug(args)
    if seed is not None:
        prefix += f"_seed{seed}"
    return {
        "cache_dir": cache_dir,
        "raw_kraus": cache_dir / f"{prefix}_raw_kraus.pkl",
        "raw_metrics": cache_dir / f"{prefix}_raw_metrics.json",
        "rewrite_kraus": cache_dir / f"{prefix}_gatecost_beam_rewritten_kraus.pkl",
        "rewrite_summary": cache_dir / f"{prefix}_gatecost_beam_rewrite_summary.json",
    }


def load_pickle(path: Path) -> Any:
    with path.open("rb") as in_file:
        return pickle.load(in_file)


def dump_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out_file:
        pickle.dump(value, out_file)


def build_raw_channel(args: argparse.Namespace, seed: int | None = None) -> tuple[list[Any], dict[str, Any]]:
    if args.example == "random-coloring":
        kraus_ops, model = build_random_coloring_channel(
            args.num_vertices,
            args.graph,
            q=args.q,
            tol=args.tol,
            max_total_pauli_terms=None,
            max_terms_per_kraus=None,
            color_mixing_preprocess=False,
        )
    else:
        if seed is None:
            raise ValueError("random-pauli build requires a seed")
        spec = RandomChannelSpec(
            args.kraus_count,
            args.hilbert_dim,
            args.first_kraus_terms,
            args.other_kraus_terms,
        )
        kraus_ops, model = build_random_kraus_ops(RandomChannelCase(spec=spec, seed=seed))

    metrics: dict[str, Any] = channel_metrics(kraus_ops)
    metrics["benchmark_name"] = benchmark_name(args)
    metrics["generator"] = args.example
    metrics["model"] = model
    if seed is not None:
        metrics["seed"] = int(seed)
    return kraus_ops, metrics


def ensure_raw_cache(args: argparse.Namespace, seed: int | None = None) -> tuple[list[Any], dict[str, Any]]:
    paths = cache_paths(args, seed)
    if paths["raw_kraus"].exists() and paths["raw_metrics"].exists():
        return load_pickle(paths["raw_kraus"]), read_json(paths["raw_metrics"])

    started = time.perf_counter()
    kraus_ops, metrics = build_raw_channel(args, seed)
    metrics["build_channel_time_s"] = float(time.perf_counter() - started)
    dump_pickle(paths["raw_kraus"], kraus_ops)
    write_json(paths["raw_metrics"], metrics)
    return kraus_ops, metrics


def ensure_rewrite_cache(args: argparse.Namespace, seed: int | None = None) -> tuple[list[Any], dict[str, Any]]:
    paths = cache_paths(args, seed)
    if paths["rewrite_kraus"].exists() and paths["rewrite_summary"].exists():
        return load_pickle(paths["rewrite_kraus"]), read_json(paths["rewrite_summary"])

    raw_kraus_ops, raw_metrics = ensure_raw_cache(args, seed)
    started = time.perf_counter()
    rewrite_source = ChannelIR(copy.deepcopy(raw_kraus_ops))
    rewrite_source.zero_elim()
    result = rewrite_source.gatecost_rewrite_search(
        strategy="beam",
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        tol=args.tol,
    )
    rewritten = ChannelIR(copy.deepcopy(raw_kraus_ops))
    rewritten.apply_rewrite_result(result, tol=args.tol)
    rewritten.zero_elim()
    rewrite_time = time.perf_counter() - started
    rewrite_metrics = channel_metrics(rewritten.kraus_ops)
    summary = {
        "method": "gatecost_rewrite_search",
        "strategy": "beam",
        "beam_width": int(args.beam_width),
        "max_steps": int(args.max_steps),
        "tol": float(args.tol),
        "time_s": float(rewrite_time),
        "raw_channel_metrics": raw_metrics,
        "rewritten_channel_metrics": rewrite_metrics,
        "initial_support": int(result.get("initial_support", 0)),
        "final_support": int(result.get("final_support", 0)),
        "initial_cost": int(result.get("initial_cost", 0)),
        "final_cost": int(result.get("final_cost", 0)),
        "steps": int(len(result.get("steps", []))),
        "termination": result.get("termination", {}),
        "result": result,
    }
    dump_pickle(paths["rewrite_kraus"], rewritten.kraus_ops)
    write_json(paths["rewrite_summary"], summary)
    return rewritten.kraus_ops, summary


def with_new_block_encoding(func):
    old_block_encoding = channel_LCU.BlockEncoding
    channel_LCU.BlockEncoding = NewBlockEncoding
    try:
        return func()
    finally:
        channel_LCU.BlockEncoding = old_block_encoding


def build_config_circuit_random(kraus_ops: list[Any], cfg: ConfigSpec) -> tuple[Any, Any, float, dict[str, Any]]:
    started = time.perf_counter()
    block_timings: list[dict[str, Any]] = []
    if not cfg.order:
        def _build():
            return channel_LCU.channel_to_LCU(
                channel_ensemble([copy.deepcopy(kraus_ops)]),
                structure=cfg.structure,
                opt=cfg.opt,
            )

        old_block_encoding = channel_LCU.BlockEncoding
        channel_LCU.BlockEncoding = timed_block_encoding_class(block_timings)
        try:
            qc, qubit_indexes = _build()
        finally:
            channel_LCU.BlockEncoding = old_block_encoding
        build_time_s = time.perf_counter() - started
        return qc, qubit_indexes, build_time_s, summarize_block_timings(block_timings, build_time_s)

    sys_size = max(ms.size for ms in kraus_ops)
    select_size = int(np.ceil(np.log2(len(kraus_ops))))
    coeff_sums = [ms.pauli_norm() for ms in kraus_ops]
    lcu_ini = channel_LCU.prep_sup_state(coeff_sums)
    block_circuits = []
    for ms in kraus_ops:
        opt = "No" if len(ms.instances) <= 1 else "Matrix-order"
        block_started = time.perf_counter()
        block = NewBlockEncoding(copy.deepcopy(ms)).circuit(opt=opt)
        block_elapsed = time.perf_counter() - block_started
        block_timings.append(
            {
                "kraus_index": int(len(block_timings)),
                "opt": opt,
                "pauli_terms": int(len(ms.instances)),
                "system_qubits": int(ms.size),
                "time_sec": float(block_elapsed),
                "output_qubits": int(block.num_qubits),
            }
        )
        block_circuits.append(block)
    ctrl_size = max(circ.num_qubits - sys_size for circ in block_circuits)

    if cfg.structure == "basic":
        from qiskit import QuantumCircuit, QuantumRegister

        ctrl = QuantumRegister(ctrl_size, "c")
        select = QuantumRegister(select_size, "s")
        sys_reg = QuantumRegister(sys_size, "sys")
        qc = QuantumCircuit(select, ctrl, sys_reg)
        qc.compose(lcu_ini, qubits=list(range(select_size)), inplace=True)
        for index, block in enumerate(block_circuits):
            ctrl_value = bin(index)[2:].zfill(select_size)[::-1]
            ctrl_size_ms = block.num_qubits - sys_size
            controlled = block.to_gate().control(num_ctrl_qubits=select_size, ctrl_state=ctrl_value)
            qc.append(controlled, qargs=list(select) + list(ctrl[:ctrl_size_ms]) + list(sys_reg))
        qubit_indexes = [
            list(range(ctrl_size)),
            list(range(ctrl_size, ctrl_size + select_size)),
            list(range(ctrl_size + select_size, ctrl_size + select_size + sys_size)),
        ]
    elif cfg.structure == "opt":
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2 * select_size + ctrl_size + sys_size)
        select_indexes = [2 * j for j in range(select_size)]
        anc_indexes = [2 * j + 1 for j in range(select_size)]
        ctrl_indexes = list(range(2 * select_size, 2 * select_size + ctrl_size))
        sys_indexes = list(range(2 * select_size + ctrl_size, 2 * select_size + ctrl_size + sys_size))
        qc.compose(lcu_ini, qubits=select_indexes, inplace=True)
        qc, _tcount, _mccount, _cxcount = channel_LCU.mulplex_U_opt(
            qc,
            block_circuits,
            select_size,
            ctrl_size,
            sys_size,
        )
        qubit_indexes = [select_indexes, anc_indexes, ctrl_indexes, sys_indexes]
    else:
        raise ValueError(f"Unsupported structure: {cfg.structure}")
    build_time_s = time.perf_counter() - started
    return qc, qubit_indexes, build_time_s, summarize_block_timings(block_timings, build_time_s)


def metric_row(
    *,
    args: argparse.Namespace,
    cfg: ConfigSpec,
    seed: int | None,
    status: str,
    error_msg: str,
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

    timing = block_timing or summarize_block_timings([], build_time_s)
    ops = {} if stats is None else stats.get("ops", {})
    model = raw_metrics.get("model", {})
    return {
        "checkpoint": "cp2_rq2_main_ablation",
        "benchmark_family": "Algorithm" if args.example == "random-coloring" else "Synthetic",
        "benchmark_name": benchmark_name(args),
        "benchmark_scope": "main",
        "example": args.example,
        "n_or_size": int(args.num_vertices if args.example == "random-coloring" else args.kraus_count),
        "graph": "" if args.example != "random-coloring" else args.graph,
        "q": "" if args.example != "random-coloring" else int(args.q),
        "kraus_tuple": ""
        if args.example != "random-pauli"
        else f"({args.kraus_count},{args.hilbert_dim},{args.first_kraus_terms},{args.other_kraus_terms})",
        "seed": "" if seed is None else int(seed),
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
        "system_qubits": int(raw_metrics.get("system_qubits", 0)),
        "model": model.get("model", ""),
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
        "notes": MATRIX_ORDER_NOTE if cfg.order else "",
    }


def run_config(args: argparse.Namespace, cfg: ConfigSpec, seed: int | None = None) -> dict[str, Any]:
    work_dir = case_dir(args, seed)
    rows_dir = work_dir / "rows"
    raw_runs_dir = work_dir / "raw_runs"
    qasm_dir = work_dir / "qasm"
    rows_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_dir.mkdir(parents=True, exist_ok=True)
    qasm_dir.mkdir(parents=True, exist_ok=True)

    attempt_started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    attempt_started = time.perf_counter()
    command = " ".join(sys.argv)
    try:
        raw_kraus_ops, raw_metrics = ensure_raw_cache(args, seed)
        rewrite_summary = None
        if cfg.rewrite:
            kraus_ops, rewrite_summary = ensure_rewrite_cache(args, seed)
        else:
            kraus_ops = raw_kraus_ops

        qc, qubit_indexes, build_time_s, block_timing = build_config_circuit_random(kraus_ops, cfg)
        qasm_path = qasm_dir / f"{example_slug(args)}_{seed_label(seed)}_{cfg.slug}_ibm_opt{args.transpile_optimization_level}.qasm"
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
            args=args,
            cfg=cfg,
            seed=seed,
            status="ok",
            error_msg="",
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
    except Exception as exc:
        try:
            _, raw_metrics = ensure_raw_cache(args, seed)
        except Exception:
            raw_metrics = {"kraus_count": 0, "pauli_terms_total": 0, "max_terms_per_kraus": 0}
        row = metric_row(
            args=args,
            cfg=cfg,
            seed=seed,
            status="failed",
            error_msg=f"{type(exc).__name__}: {exc}",
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
        "benchmark": benchmark_name(args),
        "configuration": cfg.name,
        "seed": seed,
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


def record_timeout_row(args: argparse.Namespace, cfg: ConfigSpec, seed: int | None = None) -> dict[str, Any]:
    work_dir = case_dir(args, seed)
    rows_dir = work_dir / "rows"
    raw_runs_dir = work_dir / "raw_runs"
    rows_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_dir.mkdir(parents=True, exist_ok=True)
    try:
        _, raw_metrics = ensure_raw_cache(args, seed)
    except Exception:
        raw_metrics = {"kraus_count": 0, "pauli_terms_total": 0, "max_terms_per_kraus": 0}
    rewrite_summary = None
    if cfg.rewrite:
        try:
            _, rewrite_summary = ensure_rewrite_cache(args, seed)
        except Exception:
            rewrite_summary = None
    row = metric_row(
        args=args,
        cfg=cfg,
        seed=seed,
        status="timeout",
        error_msg=f"manual timeout after {args.record_timeout_sec:g}s; {cfg.name} synthesis/count did not finish",
        raw_metrics=raw_metrics,
        rewrite_summary=rewrite_summary,
        build_time_s=0.0,
        transpile_time_s=0.0,
        count_time_s=0.0,
        total_time_s=float(args.record_timeout_sec),
        block_timing=None,
        qasm_path=None,
        stats=None,
        export_meta=None,
        qubit_indexes=None,
    )
    row["timeout"] = True
    row_path = rows_dir / f"{cfg.slug}.json"
    write_json(row_path, row)
    raw_run = {
        "benchmark": benchmark_name(args),
        "configuration": cfg.name,
        "seed": seed,
        "command": " ".join(sys.argv),
        "start_time": "",
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": row["status"],
        "timeout": True,
        "exception_or_stderr_summary": row["error_msg"],
        "output_files_produced": [str(row_path)],
        "row": row,
    }
    write_json(raw_runs_dir / f"{cfg.slug}.json", raw_run)
    print(json.dumps(row, indent=2, default=json_default))
    return row


def missing_row(args: argparse.Namespace, cfg: ConfigSpec, seed: int | None = None) -> dict[str, Any]:
    return {
        "checkpoint": "cp2_rq2_main_ablation",
        "benchmark_family": "Algorithm" if args.example == "random-coloring" else "Synthetic",
        "benchmark_name": benchmark_name(args),
        "benchmark_scope": "main",
        "example": args.example,
        "n_or_size": int(args.num_vertices if args.example == "random-coloring" else args.kraus_count),
        "graph": "" if args.example != "random-coloring" else args.graph,
        "q": "" if args.example != "random-coloring" else int(args.q),
        "kraus_tuple": ""
        if args.example != "random-pauli"
        else f"({args.kraus_count},{args.hilbert_dim},{args.first_kraus_terms},{args.other_kraus_terms})",
        "seed": "" if seed is None else int(seed),
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
        "timeout": False,
        "qasm_path": "",
        "notes": MATRIX_ORDER_NOTE if cfg.order else "",
    }


def load_rows_for_case(args: argparse.Namespace, seed: int | None = None) -> list[dict[str, Any]]:
    rows = []
    rows_dir = case_dir(args, seed) / "rows"
    for cfg in CONFIGS:
        path = rows_dir / f"{cfg.slug}.json"
        rows.append(read_json(path) if path.exists() else missing_row(args, cfg, seed))
    return rows


def add_ratios_for_rows(rows: list[dict[str, Any]]) -> None:
    basic = next((row for row in rows if row["config"] == "Basic" and row["status"] == "ok"), None)
    if basic is None or basic.get("total_gates") in ("", None, 0):
        return
    basic_gates = float(basic["total_gates"])
    for row in rows:
        if row.get("status") == "ok" and row.get("total_gates") not in ("", None):
            row["ratio_vs_basic"] = float(float(row["total_gates"]) / basic_gates)


SUMMARY_COLUMNS = [
    "checkpoint",
    "benchmark_family",
    "benchmark_name",
    "benchmark_scope",
    "example",
    "n_or_size",
    "graph",
    "q",
    "kraus_tuple",
    "seed",
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
    "system_qubits",
    "model",
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
    "completed_seed_count",
    "total_gates_stdev",
    "ratio_vs_basic_stdev",
    "non_clifford_count_stdev",
    "total_time_sec_stdev",
]


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in ("", None):
            values.append(float(value))
    return values


def mean_or_blank(rows: list[dict[str, Any]], key: str) -> float | str:
    values = numeric_values(rows, key)
    return "" if not values else float(statistics.fmean(values))


def stdev_or_blank(rows: list[dict[str, Any]], key: str) -> float | str:
    values = numeric_values(rows, key)
    if not values:
        return ""
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def aggregate_seed_rows(args: argparse.Namespace, per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cfg in CONFIGS:
        cfg_rows = [
            row for row in per_seed_rows if row["config"] == cfg.name and row.get("status") == "ok"
        ]
        template = cfg_rows[0] if cfg_rows else missing_row(args, cfg, None)
        avg = dict(template)
        avg["seed"] = "mean_" + "_".join(str(seed) for seed in args.seeds)
        avg["status"] = "ok" if len(cfg_rows) == len(args.seeds) else "partial"
        avg["error_msg"] = "" if avg["status"] == "ok" else f"{len(cfg_rows)}/{len(args.seeds)} seeds completed"
        avg["completed_seed_count"] = int(len(cfg_rows))
        avg["qasm_path"] = ""
        avg["qubit_indexes"] = ""
        for key in (
            "total_gates",
            "gate_total",
            "cx",
            "cx_count",
            "u1",
            "u2",
            "u3",
            "depth",
            "width",
            "non_clifford_count",
            "reset_count",
            "ratio_vs_basic",
            "raw_kraus_count",
            "raw_pauli_terms_total",
            "raw_max_terms_per_kraus",
            "system_qubits",
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
            "transpile_time_sec",
            "count_time_sec",
            "total_time_sec",
            "execution_time_s",
        ):
            avg[key] = mean_or_blank(cfg_rows, key)
        avg["total_gates_stdev"] = stdev_or_blank(cfg_rows, "total_gates")
        avg["ratio_vs_basic_stdev"] = stdev_or_blank(cfg_rows, "ratio_vs_basic")
        avg["non_clifford_count_stdev"] = stdev_or_blank(cfg_rows, "non_clifford_count")
        avg["total_time_sec_stdev"] = stdev_or_blank(cfg_rows, "total_time_sec")
        rows.append(avg)
    return rows


def write_raw_runs_jsonl(args: argparse.Namespace, seeds: list[int | None]) -> None:
    out_path = Path(args.out_dir) / example_slug(args) / "raw_runs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        for seed in seeds:
            raw_runs_dir = case_dir(args, seed) / "raw_runs"
            for cfg in CONFIGS:
                path = raw_runs_dir / f"{cfg.slug}.json"
                if path.exists():
                    out_file.write(json.dumps(read_json(path), default=json_default) + "\n")


def main_table_row(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_by_config = {row["config"]: row for row in rows}
    basic = row_by_config.get("Basic", {})
    all_row = row_by_config.get("Rewrite + Flat + Order", {})
    out = {
        "benchmark_family": "Algorithm" if args.example == "random-coloring" else "Synthetic",
        "benchmark_name": benchmark_name(args),
        "basis": "ibm",
        "seed": "mean_" + "_".join(str(seed) for seed in args.seeds)
        if args.example == "random-pauli"
        else "",
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
        "note": f"{benchmark_name(args)}; IBM gate set only; R uses gatecost_rewrite_search(strategy='beam').",
    }
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("ratio_vs_basic") not in ("", None)
    ]
    if ok_rows:
        best = min(ok_rows, key=lambda row: float(row["ratio_vs_basic"]))
        out["best_config"] = best["config"]
        out["best_ratio"] = best["ratio_vs_basic"]
    return out


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / example_slug(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.example == "random-pauli":
        per_seed_rows: list[dict[str, Any]] = []
        for seed in args.seeds:
            seed_rows = load_rows_for_case(args, seed)
            add_ratios_for_rows(seed_rows)
            per_seed_rows.extend(seed_rows)
        rows = aggregate_seed_rows(args, per_seed_rows)
        write_csv(out_dir / "per_seed_summary.csv", per_seed_rows, SUMMARY_COLUMNS)
        write_raw_runs_jsonl(args, [int(seed) for seed in args.seeds])
    else:
        rows = load_rows_for_case(args, None)
        add_ratios_for_rows(rows)
        write_raw_runs_jsonl(args, [None])

    write_csv(out_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    write_csv(out_dir / "rq2a_internal_ablation.csv", rows, SUMMARY_COLUMNS)
    main_row = main_table_row(args, rows)
    main_columns = [
        "benchmark_family",
        "benchmark_name",
        "basis",
        "seed",
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
        {"config": row["config"], "seed": row.get("seed", ""), "status": row["status"], "error_msg": row.get("error_msg", "")}
        for row in rows
        if row.get("status") != "ok"
    ]
    manifest = {
        "checkpoint": "cp2_rq2_main_ablation",
        "agent": "CP2-RQ2-Main-Ablation-RandomExamples",
        "repo_commit": run_git(["rev-parse", "HEAD"]),
        "dirty_status": run_git(["status", "--short"]),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "example": args.example,
        "benchmark_name": benchmark_name(args),
        "basis": "ibm",
        "basis_gates": list(IBM_BASIS),
        "transpiler_basis": list(IBM_TRANSPILER_BASIS),
        "transpile_optimization_level": int(args.transpile_optimization_level),
        "rewrite_method": "gatecost_rewrite_search",
        "rewrite_strategy": "beam",
        "matrix_order_guardrail": MATRIX_ORDER_NOTE,
        "seeds": [int(seed) for seed in args.seeds] if args.example == "random-pauli" else [],
        "success": not known_failures,
        "known_failures": known_failures,
    }
    write_json(out_dir / "manifest.json", manifest)

    notes = [
        f"# CP2 {benchmark_name(args)} Notes",
        "",
        "- Target/reporting basis: IBM `{u1,u2,u3,cx}` only; no external optimizer rows are included.",
        "- R uses `ChannelIR.gatecost_rewrite_search(strategy=\"beam\")`; this runner does not call `rewrite_search()`.",
        f"- Matrix-order guardrail: {MATRIX_ORDER_NOTE}",
        "- Every completed row records total gates, CX, U1/U2/U3, non-Clifford count, width, depth, and total execution time.",
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

    reproduce_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /Users/chester_hqf/Desktop/Research/2025/Lindbladian/Lindbladian_compilation",
        "",
        "CONDA=/Users/chester_hqf/miniconda3/bin/conda",
        "SCRIPT=scripts/eval_redesign/run_cp2_random_ablation.py",
        f"OUT={args.out_dir}",
        f"COMMON=(--example {args.example} --out-dir $OUT)",
    ]
    if args.example == "random-coloring":
        reproduce_lines.append(
            f"COMMON+=(--num-vertices {args.num_vertices} --q {args.q} --graph {args.graph})"
        )
        for cfg in CONFIGS:
            reproduce_lines.append(
                f"$CONDA run -n qiskit_simulate python $SCRIPT \"${{COMMON[@]}}\" run --config '{cfg.name}'"
            )
    else:
        reproduce_lines.append(
            f"COMMON+=(--kraus-count {args.kraus_count} --hilbert-dim {args.hilbert_dim} "
            f"--first-kraus-terms {args.first_kraus_terms} --other-kraus-terms {args.other_kraus_terms} "
            f"--seeds {' '.join(str(seed) for seed in args.seeds)})"
        )
        for seed in args.seeds:
            for cfg in CONFIGS:
                reproduce_lines.append(
                    f"$CONDA run -n qiskit_simulate python $SCRIPT \"${{COMMON[@]}}\" --seed {seed} run --config '{cfg.name}'"
                )
    reproduce_lines.append("$CONDA run -n qiskit_simulate python $SCRIPT \"${COMMON[@]}\" aggregate")
    (out_dir / "reproduce.sh").write_text("\n".join(reproduce_lines) + "\n", encoding="utf-8")
    os.chmod(out_dir / "reproduce.sh", 0o755)

    print(f"Wrote aggregate outputs under {out_dir}")


def require_seed(args: argparse.Namespace) -> int | None:
    if args.example == "random-pauli":
        if args.seed is None:
            raise SystemExit("--seed is required for random-pauli run/prepare commands")
        return int(args.seed)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("random-coloring", "random-pauli"), required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-vertices", type=int, default=4)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--graph", choices=("linear", "path", "cycle", "complete"), default="linear")
    parser.add_argument("--kraus-count", type=int, default=5)
    parser.add_argument("--hilbert-dim", type=int, default=4)
    parser.add_argument("--first-kraus-terms", type=int, default=10)
    parser.add_argument("--other-kraus-terms", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--transpile-optimization-level", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one ablation configuration.")
    run_parser.add_argument("--config", type=parse_config, required=True)
    timeout_parser = subparsers.add_parser("record-timeout", help="Record one timed-out configuration row.")
    timeout_parser.add_argument("--config", type=parse_config, required=True)
    timeout_parser.add_argument("--record-timeout-sec", type=float, required=True)
    subparsers.add_parser("prepare-raw", help="Build and cache the raw channel.")
    subparsers.add_parser("prepare-rewrite", help="Build and cache the gatecost beam rewrite output.")
    subparsers.add_parser("run-all", help="Run every ablation configuration for this example.")
    subparsers.add_parser("aggregate", help="Aggregate per-config row files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-raw":
        seed = require_seed(args)
        kraus_ops, metrics = ensure_raw_cache(args, seed)
        print(json.dumps({"raw_metrics": metrics, "kraus_count": len(kraus_ops)}, indent=2, default=json_default))
    elif args.command == "prepare-rewrite":
        seed = require_seed(args)
        _, summary = ensure_rewrite_cache(args, seed)
        print(json.dumps(summary, indent=2, default=json_default))
    elif args.command == "run":
        seed = require_seed(args)
        run_config(args, args.config, seed)
    elif args.command == "record-timeout":
        seed = require_seed(args)
        record_timeout_row(args, args.config, seed)
    elif args.command == "run-all":
        if args.example == "random-pauli":
            for seed in args.seeds:
                args.seed = int(seed)
                for cfg in CONFIGS:
                    run_config(args, cfg, int(seed))
        else:
            for cfg in CONFIGS:
                run_config(args, cfg, None)
        aggregate(args)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
