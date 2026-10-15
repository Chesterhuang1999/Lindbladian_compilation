#!/usr/bin/env python3
"""State-level CP4B probe for skipped HRW-4 / TFIM-3 backend rows.

This verifies the same backend circuits as CP4B but avoids constructing a full
dense circuit Operator.  It follows the simulation/post-measurement convention
from src/channel_LCU.py: simulate the all-zero input state, extract the
postselected system density with get_postmeas_density, normalize density
operators, and compare against the direct dense Kraus channel baseline.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector


ROOT = Path(__file__).resolve().parents[3]
CP4B_DIR = ROOT / "Baseline_results" / "eval_redesign" / "cp4_rq3_semantic_backend_validation"
if str(CP4B_DIR) not in sys.path:
    sys.path.insert(0, str(CP4B_DIR))

import run_cp4b_semantic_backend_validation as cp4b  # type: ignore  # noqa: E402


OUT_CSV = CP4B_DIR / "state_probe_hrw_tfim_backend.csv"
OUT_JSONL = CP4B_DIR / "state_probe_hrw_tfim_backend_raw.jsonl"
TARGET_BENCHMARKS = {"hrw_n4", "tfim_n3_dt0p1"}
TARGET_CONFIGS = {"O", "F", "All"}


def append_raw(event: dict[str, Any]) -> None:
    event = dict(event)
    event.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with OUT_JSONL.open("a", encoding="utf-8") as out_file:
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


def matrixsum_dense(op: Any, dim: int) -> np.ndarray:
    dense = op.eff_op().to_matrix()
    if dense.shape != (dim, dim):
        raise ValueError(f"Kraus matrix shape {dense.shape} does not match dim {dim}.")
    return np.asarray(dense, dtype=complex)


def normalize_density(rho: np.ndarray) -> tuple[np.ndarray, float]:
    trace_val = np.trace(rho)
    trace_real = float(np.real(trace_val))
    if abs(trace_val) <= 1e-14:
        raise ValueError("Cannot normalize near-zero-trace density operator.")
    return np.asarray(rho / trace_val, dtype=complex), trace_real


def apply_direct_channel(kraus_ops: list[Any], rho: np.ndarray, system_qubits: int) -> tuple[np.ndarray, float]:
    dim = 1 << system_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for kms in kraus_ops:
        K = matrixsum_dense(kms, dim)
        out += K @ rho @ K.conj().T
    return normalize_density(out)


def simulate_postselected_density(qc: Any, qubit_indexes: Any) -> tuple[np.ndarray, str]:
    """Return normalized postselected system density using channel_LCU helpers."""
    total_qubits = int(qc.num_qubits)
    ini_state = Statevector.from_label("0" * total_qubits)
    try:
        density = cp4b.channel_LCU.simulate_circuit(qc, ini_state=ini_state, qubit_regs=qubit_indexes)
        return np.asarray(density.data, dtype=complex), "channel_LCU.simulate_circuit"
    except Exception as exc:
        final_sv = ini_state.evolve(qc)
        final_sv = cp4b.channel_LCU.approx_state(final_sv, tol=1e-6)
        density = cp4b.channel_LCU.get_postmeas_density(final_sv, qubit_indexes)
        append_raw(
            {
                "event": "simulate_circuit_fallback",
                "reason": f"{type(exc).__name__}: {exc}",
                "fallback": "Statevector.evolve + channel_LCU.get_postmeas_density",
            }
        )
        return np.asarray(density.data, dtype=complex), "Statevector.evolve+channel_LCU.get_postmeas_density"


def diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    denom = np.linalg.norm(reference, ord="fro")
    rel_fro = np.linalg.norm(delta, ord="fro") / denom if denom > 0 else np.linalg.norm(delta, ord="fro")
    trace_distance = 0.5 * float(np.sum(np.linalg.svd(delta, compute_uv=False)))
    return {
        "relative_frobenius_diff": float(rel_fro),
        "trace_distance": trace_distance,
        "max_entry_diff": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "reference_trace": float(np.trace(reference).real),
        "candidate_trace": float(np.trace(candidate).real),
    }


def run_probe() -> list[dict[str, Any]]:
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()

    append_raw(
        {
            "event": "start",
                    "note": "rho=|0...0><0...0| normalized-density backend validation for skipped HRW/TFIM rows",
        }
    )

    rewrites: dict[str, tuple[list[Any], dict[str, Any], float]] = {}
    rows: list[dict[str, Any]] = []
    for bench in cp4b.build_benchmarks():
        if bench.name not in TARGET_BENCHMARKS:
            continue
        raw_metrics = cp4b.channel_metrics(bench.kraus_ops)
        rewritten_ops, rewrite_result, rewrite_elapsed = cp4b.rewrite_channel(
            bench.kraus_ops,
            max_steps=50,
            tol=1e-12,
        )
        rewrites[bench.name] = (rewritten_ops, rewrite_result, rewrite_elapsed)
        append_raw(
            {
                "event": "benchmark_ready",
                "benchmark": bench.name,
                "family": bench.family,
                "raw_metrics": raw_metrics,
                "rewrite_result": cp4b.summarize_rewrite(rewrite_result),
            }
        )

        for config in cp4b.BACKEND_CONFIGS:
            if config.name not in TARGET_CONFIGS:
                continue
            target_ops = rewritten_ops if config.use_rewrite else bench.kraus_ops
            target_metrics = cp4b.channel_metrics(target_ops)
            system_qubits = int(target_metrics["system_qubits"])
            dim_sys = 1 << system_qubits
            rho0 = np.zeros((dim_sys, dim_sys), dtype=complex)
            rho0[0, 0] = 1.0
            lambda_value = float(sum(ms.pauli_norm() ** 2 for ms in target_ops))
            started = time.perf_counter()
            try:
                reference, reference_trace_before_normalization = apply_direct_channel(target_ops, rho0, system_qubits)
                qc, qubit_indexes, build_time = cp4b.build_backend_circuit(bench, target_ops, config)
                candidate, simulation_method = simulate_postselected_density(qc, qubit_indexes)
                candidate, candidate_trace_before_normalization = normalize_density(candidate)
                metrics = diff_metrics(reference, candidate)
                row_status = "ok" if metrics["trace_distance"] <= 1e-10 else "discrepancy"
                row = {
                    "benchmark": bench.name,
                    "family": bench.family,
                    "configuration": config.name,
                    "configuration_detail": config.label,
                    "target_representation": "rewritten" if config.use_rewrite else "raw",
                    "status": row_status,
                    "input_state": "zero_density",
                    "system_qubits": system_qubits,
                    "circuit_qubits": int(qc.num_qubits),
                    "simulation_method": simulation_method,
                    "target_kraus_count": int(target_metrics["kraus_count"]),
                    "target_pauli_terms": int(target_metrics["pauli_terms_total"]),
                    "lambda_sum_coeff_square": lambda_value,
                    "reference_trace_before_normalization": reference_trace_before_normalization,
                    "candidate_trace_before_normalization": candidate_trace_before_normalization,
                    "relative_frobenius_diff": metrics["relative_frobenius_diff"],
                    "trace_distance": metrics["trace_distance"],
                    "max_entry_diff": metrics["max_entry_diff"],
                    "reference_trace_after_normalization": metrics["reference_trace"],
                    "candidate_trace_after_normalization": metrics["candidate_trace"],
                    "build_time_s": build_time,
                    "elapsed_s": time.perf_counter() - started,
                    "notes": "" if row_status == "ok" else "state-level zero-input discrepancy",
                }
                rows.append(row)
                append_raw({"event": "state_probe", "row": row})
            except Exception as exc:
                row = {
                    "benchmark": bench.name,
                    "family": bench.family,
                    "configuration": config.name,
                    "configuration_detail": config.label,
                    "target_representation": "rewritten" if config.use_rewrite else "raw",
                    "status": "failed",
                    "input_state": "zero_density",
                    "system_qubits": system_qubits,
                    "circuit_qubits": "",
                    "simulation_method": "",
                    "target_kraus_count": int(target_metrics["kraus_count"]),
                    "target_pauli_terms": int(target_metrics["pauli_terms_total"]),
                    "lambda_sum_coeff_square": lambda_value,
                    "reference_trace_before_normalization": "",
                    "candidate_trace_before_normalization": "",
                    "relative_frobenius_diff": "",
                    "trace_distance": "",
                    "max_entry_diff": "",
                    "reference_trace_after_normalization": "",
                    "candidate_trace_after_normalization": "",
                    "build_time_s": "",
                    "elapsed_s": time.perf_counter() - started,
                    "notes": f"{type(exc).__name__}: {exc}",
                }
                rows.append(row)
                append_raw({"event": "state_probe_failed", "row": row, "traceback": traceback.format_exc()})

    fieldnames = [
        "benchmark",
        "family",
        "configuration",
        "configuration_detail",
        "target_representation",
        "status",
        "input_state",
        "system_qubits",
        "circuit_qubits",
        "simulation_method",
        "target_kraus_count",
        "target_pauli_terms",
        "lambda_sum_coeff_square",
        "reference_trace_before_normalization",
        "candidate_trace_before_normalization",
        "relative_frobenius_diff",
        "trace_distance",
        "max_entry_diff",
        "reference_trace_after_normalization",
        "candidate_trace_after_normalization",
        "build_time_s",
        "elapsed_s",
        "notes",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    append_raw({"event": "finish", "rows": rows})
    return rows


def main() -> None:
    rows = run_probe()
    print(json.dumps(rows, indent=2, default=json_default))


if __name__ == "__main__":
    main()
