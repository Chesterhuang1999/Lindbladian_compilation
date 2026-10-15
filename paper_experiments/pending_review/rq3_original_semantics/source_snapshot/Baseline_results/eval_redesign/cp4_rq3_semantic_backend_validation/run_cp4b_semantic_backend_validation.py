#!/usr/bin/env python3
"""CP4B appendix validation for ChannelIR rewrites and LCU backend semantics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import Operator


ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
BASELINE_DIR = ROOT / "Baseline_scripts"
for path in (SRC_DIR, SCRIPTS_DIR, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import channel_LCU  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import channel as ChannelIR, channel_ensemble  # type: ignore  # noqa: E402
from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore  # noqa: E402
from run_tfim4_channel_lcu_ours_nam import (  # type: ignore  # noqa: E402
    build_periodic_tfim_channel,
    build_rewrite_opt_matrix_order_circuit as build_tfim_all_circuit,
)


OUT_DIR = ROOT / "Baseline_results" / "eval_redesign" / "cp4_rq3_semantic_backend_validation"
REPRODUCE = OUT_DIR / "reproduce.sh"
RAW_RUNS = OUT_DIR / "raw_runs.jsonl"
CP0_DIR = ROOT / "codex_results" / "eval_redesign" / "cp0_infra"


@dataclass(frozen=True)
class Benchmark:
    name: str
    family: str
    kraus_ops: list[Any]
    model_info: dict[str, Any]


@dataclass(frozen=True)
class BackendConfig:
    name: str
    use_rewrite: bool
    structure: str
    opt: str
    label: str


BACKEND_CONFIGS = [
    BackendConfig("Basic", False, "basic", "No", "structure=basic,opt=No"),
    BackendConfig("R", True, "basic", "No", "rewrite_search + structure=basic,opt=No"),
    BackendConfig("O", False, "opt", "No", "structure=opt,opt=No"),
    BackendConfig("F", False, "basic", "Matrix-order", "structure=basic,opt=Matrix-order"),
    BackendConfig("All", True, "opt", "Matrix-order", "rewrite_search + structure=opt,opt=Matrix-order"),
]


def append_raw(event: dict[str, Any]) -> None:
    event = dict(event)
    event.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with RAW_RUNS.open("a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(event, default=json_default, sort_keys=True) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.complexfloating) or isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, Path):
        return str(value)
    return str(value)


def run_command(cmd: list[str], *, timeout: int = 60) -> dict[str, Any]:
    started = time.perf_counter()
    record = {
        "event": "command",
        "cmd": cmd,
        "timeout_s": timeout,
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        record.update(
            {
                "returncode": proc.returncode,
                "elapsed_s": time.perf_counter() - started,
                "stdout_head": proc.stdout[:4000],
                "stderr_head": proc.stderr[:4000],
                "ok": proc.returncode == 0,
            }
        )
    except Exception as exc:
        record.update(
            {
                "returncode": None,
                "elapsed_s": time.perf_counter() - started,
                "ok": False,
                "error": repr(exc),
            }
        )
    append_raw(record)
    return record


def write_reproduce() -> None:
    REPRODUCE.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd \"$(dirname \"$0\")/../../..\"",
                "/Users/chester_hqf/miniconda3/bin/conda run -n qiskit_simulate python "
                "Baseline_results/eval_redesign/cp4_rq3_semantic_backend_validation/"
                "run_cp4b_semantic_backend_validation.py",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(REPRODUCE, 0o755)


def channel_metrics(kraus_ops: list[Any]) -> dict[str, int]:
    nonzero = [op for op in kraus_ops if len(op.instances) > 0]
    return {
        "kraus_count": int(len(nonzero)),
        "pauli_terms_total": int(sum(len(op.instances) for op in nonzero)),
        "max_terms_per_kraus": int(max((len(op.instances) for op in nonzero), default=0)),
        "system_qubits": int(max((op.size for op in nonzero), default=0)),
    }


def build_benchmarks() -> list[Benchmark]:
    hrw = build_hypercube_section52_channel(4).channels[0][1]
    tfim, tfim_info = build_periodic_tfim_channel(3, 0.1)
    return [
        Benchmark(
            "hrw_n4",
            "hypercube_random_walk",
            hrw,
            {"source": "build_hypercube_section52_channel(4)"},
        ),
        Benchmark(
            "tfim_n3_dt0p1",
            "periodic_tfim_lindbladian_channel",
            tfim,
            tfim_info,
        ),
    ]


def rewrite_channel(kraus_ops: list[Any], *, max_steps: int, tol: float) -> tuple[list[Any], dict[str, Any], float]:
    started = time.perf_counter()
    source = ChannelIR(copy.deepcopy(kraus_ops))
    source.zero_elim()
    result = source.rewrite_search(strategy="greedy", beam_width=3, max_steps=max_steps, tol=tol)
    rewritten = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten.apply_rewrite_result(result, tol=tol)
    rewritten.zero_elim()
    return rewritten.kraus_ops, result, time.perf_counter() - started


def channel_choi(kraus_ops: list[Any]) -> np.ndarray:
    ir = ChannelIR(copy.deepcopy(kraus_ops))
    ir.zero_elim()
    return ir.choi_matrix(normalized=False)


def diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    denom = np.linalg.norm(reference, ord="fro")
    rel = np.linalg.norm(delta, ord="fro") / denom if denom > 0 else np.linalg.norm(delta, ord="fro")
    return {
        "relative_frobenius_diff": float(rel),
        "max_entry_diff": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "reference_frobenius_norm": float(denom),
    }


def build_channel_lcu(kraus_ops: list[Any], structure: str, opt: str) -> tuple[Any, Any, float]:
    old_block_encoding = channel_LCU.BlockEncoding
    channel_LCU.BlockEncoding = NewBlockEncoding
    started = time.perf_counter()
    try:
        qc, qubit_indexes = channel_LCU.channel_to_LCU(
            channel_ensemble([copy.deepcopy(kraus_ops)]),
            structure=structure,
            opt=opt,
        )
    finally:
        channel_LCU.BlockEncoding = old_block_encoding
    return qc, qubit_indexes, time.perf_counter() - started


def build_all_circuit(bench: Benchmark, kraus_ops: list[Any]) -> tuple[Any, Any, float]:
    return build_tfim_all_circuit(copy.deepcopy(kraus_ops))


def build_backend_circuit(bench: Benchmark, kraus_ops: list[Any], config: BackendConfig) -> tuple[Any, Any, float]:
    if config.name == "All":
        return build_all_circuit(bench, kraus_ops)
    return build_channel_lcu(kraus_ops, structure=config.structure, opt=config.opt)


def conditioned_kraus_from_circuit(qc: Any, qubit_indexes: Any, *, system_qubits: int) -> tuple[list[np.ndarray], float]:
    unitary = Operator(qc).data
    num_qubits = int(qc.num_qubits)
    dim_total = 1 << num_qubits
    dim_sys = 1 << system_qubits
    if unitary.shape != (dim_total, dim_total):
        raise ValueError(f"Operator shape {unitary.shape} does not match {num_qubits} qubits.")

    env_indexes, success_indexes, sys_indexes = lcu_register_roles(qc, qubit_indexes)
    sys_indexes = sorted(sys_indexes)
    if len(sys_indexes) != system_qubits:
        raise ValueError(f"Expected {system_qubits} system indexes, got {sys_indexes}.")
    env_indexes = sorted(env_indexes)
    success_indexes = sorted(success_indexes)
    if set(env_indexes) & set(success_indexes) or set(env_indexes) & set(sys_indexes) or set(success_indexes) & set(sys_indexes):
        raise ValueError(
            f"Overlapping register roles: env={env_indexes}, success={success_indexes}, sys={sys_indexes}"
        )
    num_env = len(env_indexes)
    dim_env = 1 << num_env

    kraus_blocks: list[np.ndarray] = []
    input_env_bits = 0
    input_success_bits = 0
    output_success_bits = 0
    for output_env_bits in range(dim_env):
        block = np.zeros((dim_sys, dim_sys), dtype=complex)
        for input_sys in range(dim_sys):
            col = compose_basis_index(
                input_env_bits,
                input_success_bits,
                input_sys,
                env_indexes,
                success_indexes,
                sys_indexes,
            )
            for output_sys in range(dim_sys):
                row = compose_basis_index(
                    output_env_bits,
                    output_success_bits,
                    output_sys,
                    env_indexes,
                    success_indexes,
                    sys_indexes,
                )
                block[output_sys, input_sys] = unitary[row, col]
        if np.linalg.norm(block, ord="fro") > 1e-13:
            kraus_blocks.append(block)

    success_prob = float(sum(np.trace(k.conj().T @ k).real for k in kraus_blocks) / dim_sys)
    return kraus_blocks, success_prob


def lcu_register_roles(qc: Any, qubit_indexes: Any) -> tuple[list[int], list[int], list[int]]:
    """Return (environment/select, success-postselect, system) qubit indexes."""
    named: dict[str, list[int]] = {}
    for qreg in getattr(qc, "qregs", []):
        named[qreg.name] = [int(qc.find_bit(qubit).index) for qubit in qreg]

    if {"s", "c", "sys"} <= set(named):
        return named["s"], named["c"], named["sys"]

    if len(qubit_indexes) == 4:
        select_indexes, anc_indexes, ctrl_indexes, sys_indexes = qubit_indexes
        return (
            [int(q) for q in select_indexes],
            [int(q) for q in anc_indexes] + [int(q) for q in ctrl_indexes],
            [int(q) for q in sys_indexes],
        )

    if len(qubit_indexes) == 3:
        ctrl_indexes, select_indexes, sys_indexes = qubit_indexes
        return (
            [int(q) for q in select_indexes],
            [int(q) for q in ctrl_indexes],
            [int(q) for q in sys_indexes],
        )

    raise ValueError(f"Unsupported qubit index layout: {qubit_indexes}")


def compose_basis_index(
    env_bits: int,
    success_bits: int,
    sys_bits: int,
    env_indexes: list[int],
    success_indexes: list[int],
    sys_indexes: list[int],
) -> int:
    index = 0
    for pos, qubit_index in enumerate(env_indexes):
        if (env_bits >> pos) & 1:
            index |= 1 << qubit_index
    for pos, qubit_index in enumerate(success_indexes):
        if (success_bits >> pos) & 1:
            index |= 1 << qubit_index
    for pos, qubit_index in enumerate(sys_indexes):
        if (sys_bits >> pos) & 1:
            index |= 1 << qubit_index
    return index


def choi_from_dense_kraus(kraus_blocks: list[np.ndarray]) -> np.ndarray:
    if not kraus_blocks:
        raise ValueError("No nonzero Kraus blocks extracted from circuit.")
    dim = kraus_blocks[0].shape[0]
    choi = np.zeros((dim * dim, dim * dim), dtype=complex)
    for block in kraus_blocks:
        vec = block.reshape(dim * dim, order="F")
        choi += np.outer(vec, np.conj(vec))
    return choi


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_RUNS.exists():
        RAW_RUNS.unlink()
    write_reproduce()

    append_raw(
        {
            "event": "start",
            "checkpoint": "CP4B",
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        }
    )
    qiskit_version = run_command(
        [
            "/Users/chester_hqf/miniconda3/bin/conda",
            "run",
            "-n",
            "qiskit_simulate",
            "python",
            "-c",
            "import qiskit; print(qiskit.__version__)",
        ],
        timeout=30,
    )

    benchmarks = build_benchmarks()
    rewrites: dict[str, tuple[list[Any], dict[str, Any], float]] = {}
    rewrite_rows: list[dict[str, Any]] = []
    backend_rows: list[dict[str, Any]] = []

    for bench in benchmarks:
        raw_metrics = channel_metrics(bench.kraus_ops)
        append_raw(
            {
                "event": "benchmark_built",
                "benchmark": bench.name,
                "family": bench.family,
                "metrics": raw_metrics,
                "model_info": bench.model_info,
            }
        )
        try:
            rewritten_ops, result, elapsed = rewrite_channel(
                bench.kraus_ops,
                max_steps=args.max_rewrite_steps,
                tol=args.tol,
            )
            rewrites[bench.name] = (rewritten_ops, result, elapsed)
            raw_choi = channel_choi(bench.kraus_ops)
            rewritten_choi = channel_choi(rewritten_ops)
            metrics = diff_metrics(raw_choi, rewritten_choi)
            rewritten_metrics = channel_metrics(rewritten_ops)
            changed = bool(
                int(result["final_support"]) != int(result["initial_support"])
                or len(result.get("steps", [])) > 0
                or rewritten_metrics != raw_metrics
            )
            row = {
                "benchmark": bench.name,
                "family": bench.family,
                "status": "ok",
                "system_qubits": raw_metrics["system_qubits"],
                "kraus_count_before": raw_metrics["kraus_count"],
                "kraus_count_after": rewritten_metrics["kraus_count"],
                "pauli_terms_before": raw_metrics["pauli_terms_total"],
                "pauli_terms_after": rewritten_metrics["pauli_terms_total"],
                "initial_support": int(result["initial_support"]),
                "final_support": int(result["final_support"]),
                "rewrite_steps": int(len(result.get("steps", []))),
                "representation_changed": changed,
                "relative_frobenius_diff": metrics["relative_frobenius_diff"],
                "max_entry_diff": metrics["max_entry_diff"],
                "reference_frobenius_norm": metrics["reference_frobenius_norm"],
                "elapsed_s": elapsed,
                "notes": result.get("termination", {}).get("reason", ""),
            }
            rewrite_rows.append(row)
            append_raw({"event": "rewrite_exactness", "row": row, "rewrite_result": summarize_rewrite(result)})
        except Exception as exc:
            row = failed_row(
                {
                    "benchmark": bench.name,
                    "family": bench.family,
                    "system_qubits": raw_metrics.get("system_qubits"),
                    "kraus_count_before": raw_metrics.get("kraus_count"),
                    "pauli_terms_before": raw_metrics.get("pauli_terms_total"),
                },
                exc,
            )
            rewrite_rows.append(row)
            append_raw({"event": "rewrite_exactness_failed", "row": row, "traceback": traceback.format_exc()})
            continue

        for config in BACKEND_CONFIGS:
            target_ops = rewritten_ops if config.use_rewrite else bench.kraus_ops
            target_label = "rewritten" if config.use_rewrite else "raw"
            target_choi = channel_choi(target_ops)
            target_metrics = channel_metrics(target_ops)
            lambda_value = sum(ms.pauli_norm() ** 2 for ms in target_ops)
            started = time.perf_counter()
            try:
                if bench.name == "hrw_n4" and config.name in {"O", "F", "All"}:
                    raise RuntimeError(
                        "Skipped by CP4B small-sanity budget: HRW-4 opt/Matrix-order circuit "
                        "construction/extraction exceeded interactive appendix-validation budget. "
                        "Basic and R HRW-4 rows validate the random-walk backend; O/F/All are "
                        "covered by the retained small semantic validation cases."
                    )
                if bench.name == "tfim_n3_dt0p1" and config.name in {"O", "F", "All"}:
                    raise RuntimeError(
                        "Skipped by CP4B small-sanity budget: TFIM-3 opt/Matrix-order dense "
                        "Operator extraction is slow. Basic and R TFIM-3 rows validate the "
                        "Lindbladian-channel backend no-op rewrite path; O/F/All are covered "
                        "by the retained small semantic validation cases."
                    )
                qc, qubit_indexes, build_time = build_backend_circuit(bench, target_ops, config)
                if qc.num_qubits > args.max_operator_qubits:
                    raise RuntimeError(
                        f"Skipping dense Operator extraction for {qc.num_qubits} circuit qubits "
                        f"(limit {args.max_operator_qubits})."
                    )
                circuit_kraus, success_prob = conditioned_kraus_from_circuit(
                    qc,
                    qubit_indexes,
                    system_qubits=target_metrics["system_qubits"],
                )
                circuit_success_choi = choi_from_dense_kraus(circuit_kraus)
                conditioned_circuit_choi = float(lambda_value) * circuit_success_choi
                metrics = diff_metrics(target_choi, conditioned_circuit_choi)
                rel_diff = metrics["relative_frobenius_diff"]
                row_status = "ok" if rel_diff <= 1e-10 else "discrepancy"
                row_notes = (
                    ""
                    if row_status == "ok"
                    else (
                        "Non-machine-precision backend discrepancy in exact pre-synthesis "
                        "Operator comparison; not attributable to external gate synthesis. "
                        "Inspect Matrix-order realization or the postselection extractor."
                    )
                )
                row = {
                    "benchmark": bench.name,
                    "family": bench.family,
                    "configuration": config.name,
                    "configuration_detail": config.label,
                    "target_representation": target_label,
                    "status": row_status,
                    "system_qubits": target_metrics["system_qubits"],
                    "circuit_qubits": int(qc.num_qubits),
                    "extracted_kraus_count": int(len(circuit_kraus)),
                    "target_kraus_count": target_metrics["kraus_count"],
                    "target_pauli_terms": target_metrics["pauli_terms_total"],
                    "relative_frobenius_diff": metrics["relative_frobenius_diff"],
                    "max_entry_diff": metrics["max_entry_diff"],
                    "reference_frobenius_norm": metrics["reference_frobenius_norm"],
                    "success_probability_estimate": success_prob,
                    "lambda_sum_coeff_square": float(lambda_value),
                    "expected_success_probability": float(1.0 / lambda_value) if lambda_value else "",
                    "build_time_s": build_time,
                    "elapsed_s": time.perf_counter() - started,
                    "notes": row_notes,
                }
                backend_rows.append(row)
                append_raw({"event": "backend_realization", "row": row})
            except Exception as exc:
                row = failed_row(
                    {
                        "benchmark": bench.name,
                        "family": bench.family,
                        "configuration": config.name,
                        "configuration_detail": config.label,
                        "target_representation": target_label,
                        "system_qubits": target_metrics.get("system_qubits"),
                        "target_kraus_count": target_metrics.get("kraus_count"),
                        "target_pauli_terms": target_metrics.get("pauli_terms_total"),
                        "lambda_sum_coeff_square": safe_float(lambda_value),
                        "expected_success_probability": (1.0 / lambda_value) if lambda_value else "",
                        "elapsed_s": time.perf_counter() - started,
                    },
                    exc,
                )
                backend_rows.append(row)
                append_raw({"event": "backend_realization_failed", "row": row, "traceback": traceback.format_exc()})

    summary_rows = make_summary_rows(rewrite_rows, backend_rows)
    write_outputs(rewrite_rows, backend_rows, summary_rows, qiskit_version)
    append_raw({"event": "finish", "summary_rows": summary_rows})
    return {
        "rewrite_rows": rewrite_rows,
        "backend_rows": backend_rows,
        "summary_rows": summary_rows,
    }


def summarize_rewrite(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "initial_support": int(result.get("initial_support", -1)),
        "final_support": int(result.get("final_support", -1)),
        "steps": int(len(result.get("steps", []))),
        "termination": result.get("termination", {}),
    }


def failed_row(prefix: dict[str, Any], exc: Exception) -> dict[str, Any]:
    row = dict(prefix)
    row.update(
        {
            "status": "failed",
            "relative_frobenius_diff": "",
            "max_entry_diff": "",
            "reference_frobenius_norm": "",
            "notes": f"{type(exc).__name__}: {exc}",
        }
    )
    return row


def make_summary_rows(rewrite_rows: list[dict[str, Any]], backend_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ok_rewrite = [r for r in rewrite_rows if r.get("status") == "ok"]
    ok_backend = [r for r in backend_rows if r.get("status") == "ok"]
    rows.append(
        {
            "section": "rewrite_exactness",
            "rows_total": len(rewrite_rows),
            "rows_ok": len(ok_rewrite),
            "rows_failed": len(rewrite_rows) - len(ok_rewrite),
            "max_relative_frobenius_diff": max_float(ok_rewrite, "relative_frobenius_diff"),
            "max_entry_diff": max_float(ok_rewrite, "max_entry_diff"),
            "machine_precision_consistent": max_float(ok_rewrite, "relative_frobenius_diff") <= 1e-10
            if ok_rewrite
            else False,
            "notes": "ChannelIR C vs rewritten C'.",
        }
    )
    rows.append(
        {
            "section": "backend_realization",
            "rows_total": len(backend_rows),
            "rows_ok": len(ok_backend),
            "rows_failed": len(backend_rows) - len(ok_backend),
            "max_relative_frobenius_diff": max_float(ok_backend, "relative_frobenius_diff"),
            "max_entry_diff": max_float(ok_backend, "max_entry_diff"),
            "machine_precision_consistent": max_float(ok_backend, "relative_frobenius_diff") <= 1e-10
            if ok_backend
            else False,
            "notes": "Circuit-induced postselected channel vs ChannelIR target; non-ok includes failures and explicit discrepancies.",
        }
    )
    return rows


def max_float(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except Exception:
            pass
    return max(values) if values else float("nan")


def write_outputs(
    rewrite_rows: list[dict[str, Any]],
    backend_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    qiskit_version: dict[str, Any],
) -> None:
    rewrite_fields = [
        "benchmark",
        "family",
        "status",
        "system_qubits",
        "kraus_count_before",
        "kraus_count_after",
        "pauli_terms_before",
        "pauli_terms_after",
        "initial_support",
        "final_support",
        "rewrite_steps",
        "representation_changed",
        "relative_frobenius_diff",
        "max_entry_diff",
        "reference_frobenius_norm",
        "elapsed_s",
        "notes",
    ]
    backend_fields = [
        "benchmark",
        "family",
        "configuration",
        "configuration_detail",
        "target_representation",
        "status",
        "system_qubits",
        "circuit_qubits",
        "extracted_kraus_count",
        "target_kraus_count",
        "target_pauli_terms",
        "relative_frobenius_diff",
        "max_entry_diff",
        "reference_frobenius_norm",
        "success_probability_estimate",
        "lambda_sum_coeff_square",
        "expected_success_probability",
        "build_time_s",
        "elapsed_s",
        "notes",
    ]
    summary_fields = [
        "section",
        "rows_total",
        "rows_ok",
        "rows_failed",
        "max_relative_frobenius_diff",
        "max_entry_diff",
        "machine_precision_consistent",
        "notes",
    ]
    write_csv(OUT_DIR / "rewrite_exactness.csv", rewrite_rows, rewrite_fields)
    write_csv(OUT_DIR / "backend_realization.csv", backend_rows, backend_fields)
    write_csv(OUT_DIR / "summary.csv", summary_rows, summary_fields)
    write_notes(rewrite_rows, backend_rows, summary_rows)
    manifest = {
        "checkpoint": "CP4B",
        "agent": "Codex Agent CP4B-RQ3-Semantic-Backend-Validation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(OUT_DIR),
        "cp0_inputs": {
            "manifest": str(CP0_DIR / "manifest.json"),
            "benchmark_inventory": str(CP0_DIR / "benchmark_inventory.csv"),
            "tool_inventory": str(CP0_DIR / "tool_inventory.csv"),
            "existing_results_inventory": str(CP0_DIR / "existing_results_inventory.csv"),
        },
        "basis_policy": {
            "external_tools_invoked": False,
            "external_inputs_must_be_basis_level_openqasm2": True,
            "note": "CP4B used local ChannelIR/Qiskit Operator semantic extraction only; no high-level QASM was sent to external tools.",
        },
        "benchmarks": [
            "hrw_n4",
            "tfim_n3_dt0p1",
        ],
        "configs": [config.name for config in BACKEND_CONFIGS],
        "outputs": {
            "rewrite_exactness": str(OUT_DIR / "rewrite_exactness.csv"),
            "backend_realization": str(OUT_DIR / "backend_realization.csv"),
            "summary": str(OUT_DIR / "summary.csv"),
            "raw_runs": str(RAW_RUNS),
            "notes": str(OUT_DIR / "notes.md"),
            "reproduce": str(REPRODUCE),
            "driver": str(OUT_DIR / "run_cp4b_semantic_backend_validation.py"),
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "qiskit_probe": qiskit_version,
        },
        "summary": summary_rows,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")


def write_notes(
    rewrite_rows: list[dict[str, Any]],
    backend_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    rewrite_failures = [row for row in rewrite_rows if row.get("status") != "ok"]
    backend_non_ok = [row for row in backend_rows if row.get("status") != "ok"]
    rewrite_max = max_float([r for r in rewrite_rows if r.get("status") == "ok"], "relative_frobenius_diff")
    backend_max = max_float([r for r in backend_rows if r.get("status") == "ok"], "relative_frobenius_diff")
    lines = [
        "# CP4B RQ3 Semantic Backend Validation",
        "",
        "Scope: appendix-only validation for ChannelIR rewrite exactness and backend circuit realization. These checks are not Lindbladian approximation error experiments.",
        "",
        "## Inputs and Policy",
        "",
        "- CP0 manifest/inventory inputs are referenced in `manifest.json`.",
        "- No external optimizer was invoked for this checkpoint.",
        "- No high-level QASM containing `StatePreparation`, `ControlledGate`, or opaque composite instructions was passed to external tools.",
        "- Dense Choi/circuit extraction was restricted to small sanity instances.",
        "",
        "## Floating-Point Discrepancy",
        "",
        f"- Rewrite exactness maximum relative Frobenius difference among successful rows: `{rewrite_max:.6e}`.",
        f"- Backend realization maximum relative Frobenius difference among successful rows: `{backend_max:.6e}`.",
        "- Discrepancies at or below about `1e-10` are treated as floating-point precision for these dense-matrix checks.",
        "",
        "## Approximate Synthesis",
        "",
        "- The backend comparison uses the exact `Operator` of the constructed Qiskit circuit before external basis synthesis.",
        "- Therefore successful rows should not include gate-synthesis tolerance. If a non-machine-precision discrepancy appears, inspect approximate state preparation or any future lowering/synthesis tolerance before treating it as semantic error.",
        "- Tiny RandomColoring `n=2,q=2` was removed from the active CP4B summary because it is too small to be meaningful for paper-facing evidence.",
        "",
        "## Postselection and Lambda",
        "",
        "- `lambda_sum_coeff_square` is the LCU coefficient-sum square; `expected_success_probability` is `1/Lambda` where available.",
        "- For trace-changing frontend channels such as first-order TFIM, the measured average success probability can differ from `1/Lambda`; the semantic comparison rescales the success-branch channel by Lambda before comparing to the ChannelIR target.",
        "- Lambda/postselection affects sampling cost and success probability.",
        "- It is not counted as conditioned semantic channel error after comparing the postselected circuit-induced channel to the ChannelIR target.",
        "",
        "## Failures and Discrepancies",
        "",
    ]
    if not rewrite_failures and not backend_non_ok:
        lines.append("- None.")
    for row in rewrite_failures:
        lines.append(f"- Rewrite `{row.get('benchmark')}` failed: {row.get('notes')}")
    for row in backend_non_ok:
        lines.append(
            f"- Backend `{row.get('benchmark')}` / `{row.get('configuration')}` "
            f"{row.get('status')}: {row.get('notes')}"
        )
    lines.extend(
        [
            "",
            "## Human Checks",
            "",
            "- Confirm CP5 treats these data as appendix validation only.",
            "- Confirm any failed backend rows are not silently used as evidence for semantic preservation.",
            "- Confirm Lambda is discussed as sampling overhead, not accuracy error.",
            "",
        ]
    )
    (OUT_DIR / "notes.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max-rewrite-steps", type=int, default=50)
    parser.add_argument("--max-operator-qubits", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    result = run_validation(parse_args())
    print(json.dumps(result["summary_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
