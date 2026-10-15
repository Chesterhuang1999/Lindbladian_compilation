#!/usr/bin/env python3
"""Diagnose RandomColoring full-Operator extractor vs channel_LCU simulation."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[3]
CP4B_DIR = ROOT / "Baseline_results" / "eval_redesign" / "cp4_rq3_semantic_backend_validation"
if str(CP4B_DIR) not in sys.path:
    sys.path.insert(0, str(CP4B_DIR))

import run_cp4b_semantic_backend_validation as cp4b  # type: ignore  # noqa: E402


OUT_CSV = CP4B_DIR / "random_coloring_operator_layout_diagnosis.csv"
OUT_JSONL = CP4B_DIR / "random_coloring_operator_layout_diagnosis_raw.jsonl"


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


def append_raw(event: dict[str, Any]) -> None:
    event = dict(event)
    event.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with OUT_JSONL.open("a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(event, default=json_default, sort_keys=True) + "\n")


def normalize_density(rho: np.ndarray) -> tuple[np.ndarray, float]:
    trace_val = np.trace(rho)
    if abs(trace_val) <= 1e-14:
        raise ValueError("Cannot normalize near-zero trace density.")
    return np.asarray(rho / trace_val, dtype=complex), float(np.real(trace_val))


def direct_density(kraus_ops: list[Any], system_qubits: int) -> tuple[np.ndarray, float]:
    dim = 1 << system_qubits
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[0, 0] = 1.0
    out = np.zeros_like(rho0)
    for ms in kraus_ops:
        k = np.asarray(ms.eff_op().to_matrix(), dtype=complex)
        out += k @ rho0 @ k.conj().T
    return normalize_density(out)


def sim_density(qc: Any, qubit_indexes: Any) -> tuple[np.ndarray, float]:
    ini_state = Statevector.from_label("0" * int(qc.num_qubits))
    density = cp4b.channel_LCU.simulate_circuit(qc, ini_state=ini_state, qubit_regs=qubit_indexes)
    return normalize_density(np.asarray(density.data, dtype=complex))


def operator_extractor_density(qc: Any, qubit_indexes: Any, system_qubits: int) -> tuple[np.ndarray, float, int]:
    circuit_kraus, _success_prob = cp4b.conditioned_kraus_from_circuit(
        qc,
        qubit_indexes,
        system_qubits=system_qubits,
    )
    dim = 1 << system_qubits
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[0, 0] = 1.0
    out = np.zeros_like(rho0)
    for k in circuit_kraus:
        out += k @ rho0 @ k.conj().T
    rho, trace_before = normalize_density(out)
    return rho, trace_before, len(circuit_kraus)


def diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    denom = np.linalg.norm(reference, ord="fro")
    return {
        "relative_frobenius_diff": float(np.linalg.norm(delta, ord="fro") / denom if denom else np.linalg.norm(delta, ord="fro")),
        "trace_distance": 0.5 * float(np.sum(np.linalg.svd(delta, compute_uv=False))),
        "max_entry_diff": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def config_by_name(name: str) -> Any:
    for config in cp4b.BACKEND_CONFIGS:
        if config.name == name:
            return config
    raise KeyError(name)


def run() -> list[dict[str, Any]]:
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()
    append_raw({"event": "start", "note": "RandomColoring layout diagnosis"})

    bench = next(b for b in cp4b.build_benchmarks() if b.name == "random_coloring_n2_q2_linear")
    raw_ops = bench.kraus_ops
    rewritten_ops, rewrite_result, _rewrite_elapsed = cp4b.rewrite_channel(raw_ops, max_steps=50, tol=1e-12)
    rows: list[dict[str, Any]] = []

    for name in ["Basic", "O", "F", "All"]:
        config = config_by_name(name)
        target_ops = rewritten_ops if config.use_rewrite else raw_ops
        system_qubits = int(cp4b.channel_metrics(target_ops)["system_qubits"])
        qc, qubit_indexes, build_time = cp4b.build_backend_circuit(bench, target_ops, config)

        direct, direct_trace = direct_density(target_ops, system_qubits)
        sim, sim_trace = sim_density(qc, qubit_indexes)
        op, op_trace, op_kraus_count = operator_extractor_density(qc, qubit_indexes, system_qubits)

        sim_vs_direct = diff_metrics(direct, sim)
        op_vs_direct = diff_metrics(direct, op)
        op_vs_sim = diff_metrics(sim, op)

        env_indexes, success_indexes, sys_indexes = cp4b.lcu_register_roles(qc, qubit_indexes)
        row = {
            "benchmark": bench.name,
            "configuration": name,
            "configuration_detail": config.label,
            "target_representation": "rewritten" if config.use_rewrite else "raw",
            "circuit_qubits": int(qc.num_qubits),
            "qubit_indexes": repr(qubit_indexes),
            "env_indexes_by_cp4b": repr(env_indexes),
            "success_indexes_by_cp4b": repr(success_indexes),
            "sys_indexes_by_cp4b": repr(sys_indexes),
            "operator_extracted_kraus_count": op_kraus_count,
            "direct_trace_before_norm": direct_trace,
            "simulate_trace_before_norm": sim_trace,
            "operator_trace_before_norm": op_trace,
            "simulate_vs_direct_trace_distance": sim_vs_direct["trace_distance"],
            "simulate_vs_direct_rel_fro": sim_vs_direct["relative_frobenius_diff"],
            "operator_vs_direct_trace_distance": op_vs_direct["trace_distance"],
            "operator_vs_direct_rel_fro": op_vs_direct["relative_frobenius_diff"],
            "operator_vs_simulate_trace_distance": op_vs_sim["trace_distance"],
            "operator_vs_simulate_rel_fro": op_vs_sim["relative_frobenius_diff"],
            "build_time_s": build_time,
            "notes": "",
        }
        if op_vs_sim["trace_distance"] > 1e-10 and sim_vs_direct["trace_distance"] <= 1e-10:
            row["notes"] = "full-Operator extractor disagrees with channel_LCU simulation; likely layout/bit-order issue"
        rows.append(row)
        append_raw({"event": "diagnosis_row", "row": row, "rewrite_result": cp4b.summarize_rewrite(rewrite_result)})

    fieldnames = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    append_raw({"event": "finish", "rows": rows})
    return rows


def main() -> None:
    rows = run()
    print(json.dumps(rows, indent=2, default=json_default))


if __name__ == "__main__":
    main()
