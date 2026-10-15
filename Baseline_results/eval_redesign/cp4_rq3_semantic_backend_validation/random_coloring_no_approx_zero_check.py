#!/usr/bin/env python3
"""Check RandomColoring zero-state simulation without channel_LCU.approx_state."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[3]
CP4B_DIR = ROOT / "Baseline_results" / "eval_redesign" / "cp4_rq3_semantic_backend_validation"
if str(CP4B_DIR) not in sys.path:
    sys.path.insert(0, str(CP4B_DIR))

import run_cp4b_semantic_backend_validation as cp4b  # type: ignore  # noqa: E402


def normalize(rho: np.ndarray) -> np.ndarray:
    trace = np.trace(rho)
    if abs(trace) <= 1e-14:
        raise ValueError("near-zero trace")
    return np.asarray(rho / trace, dtype=complex)


def direct_density(kraus_ops: list, system_qubits: int) -> np.ndarray:
    dim = 1 << system_qubits
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[0, 0] = 1.0
    out = np.zeros_like(rho0)
    for ms in kraus_ops:
        k = np.asarray(ms.eff_op().to_matrix(), dtype=complex)
        out += k @ rho0 @ k.conj().T
    return normalize(out)


def sim_no_approx(qc, qubit_indexes) -> np.ndarray:
    final_sv = Statevector.from_label("0" * int(qc.num_qubits)).evolve(qc)
    rho = cp4b.channel_LCU.get_postmeas_density(final_sv, qubit_indexes).data
    return normalize(np.asarray(rho, dtype=complex))


def metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    delta = candidate - reference
    return (
        0.5 * float(np.sum(np.linalg.svd(delta, compute_uv=False))),
        float(np.linalg.norm(delta, ord="fro")),
        float(np.max(np.abs(delta))) if delta.size else 0.0,
    )


def main() -> None:
    bench = next(b for b in cp4b.build_benchmarks() if b.name == "random_coloring_n2_q2_linear")
    raw_ops = bench.kraus_ops
    rewritten_ops, _result, _elapsed = cp4b.rewrite_channel(raw_ops, max_steps=50, tol=1e-12)
    for config in cp4b.BACKEND_CONFIGS:
        if config.name not in {"Basic", "O", "F", "All"}:
            continue
        target_ops = rewritten_ops if config.use_rewrite else raw_ops
        system_qubits = int(cp4b.channel_metrics(target_ops)["system_qubits"])
        qc, qubit_indexes, _build_time = cp4b.build_backend_circuit(bench, target_ops, config)
        reference = direct_density(target_ops, system_qubits)
        candidate = sim_no_approx(qc, qubit_indexes)
        trace_distance, frobenius, max_entry = metrics(reference, candidate)
        print(
            config.name,
            "trace_distance=", repr(trace_distance),
            "frobenius=", repr(frobenius),
            "max_entry=", repr(max_entry),
        )


if __name__ == "__main__":
    main()
