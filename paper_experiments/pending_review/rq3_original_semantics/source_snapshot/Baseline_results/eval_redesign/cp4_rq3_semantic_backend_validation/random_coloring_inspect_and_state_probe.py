#!/usr/bin/env python3
"""Inspect CP4B RandomColoring-small and compare O vs F by normalized simulation."""

from __future__ import annotations

import copy
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


OUT_TXT = CP4B_DIR / "random_coloring_inspection.txt"
OUT_CSV = CP4B_DIR / "random_coloring_state_probe.csv"
OUT_JSON = CP4B_DIR / "random_coloring_inspection.json"
OUT_JSONL = CP4B_DIR / "random_coloring_state_probe_raw.jsonl"


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


def matrixsum_terms(ms: Any) -> list[dict[str, Any]]:
    terms = []
    for inst, coeff in ms.instances:
        terms.append(
            {
                "pauli": str(inst),
                "coeff": {
                    "real": float(np.real(coeff)),
                    "imag": float(np.imag(coeff)),
                },
            }
        )
    return terms


def matrixsum_summary(kraus_ops: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for idx, ms in enumerate(kraus_ops):
        rows.append(
            {
                "kraus_index": idx,
                "size": int(ms.size),
                "term_count": int(len(ms.instances)),
                "pauli_norm": float(ms.pauli_norm()),
                "repr": str(ms),
                "terms": matrixsum_terms(ms),
            }
        )
    return rows


def dense_kraus(ms: Any, dim: int) -> np.ndarray:
    dense = np.asarray(ms.eff_op().to_matrix(), dtype=complex)
    if dense.shape != (dim, dim):
        raise ValueError(f"Kraus shape {dense.shape} does not match {dim}.")
    return dense


def normalize_density(rho: np.ndarray) -> tuple[np.ndarray, float]:
    trace_val = np.trace(rho)
    if abs(trace_val) <= 1e-14:
        raise ValueError("Cannot normalize near-zero-trace density operator.")
    return np.asarray(rho / trace_val, dtype=complex), float(np.real(trace_val))


def apply_direct_channel(kraus_ops: list[Any], rho: np.ndarray, system_qubits: int) -> tuple[np.ndarray, float]:
    dim = 1 << system_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for ms in kraus_ops:
        k = dense_kraus(ms, dim)
        out += k @ rho @ k.conj().T
    return normalize_density(out)


def simulate_config(bench: Any, target_ops: list[Any], config: Any) -> tuple[np.ndarray, float, Any, float]:
    qc, qubit_indexes, build_time = cp4b.build_backend_circuit(bench, target_ops, config)
    ini_state = Statevector.from_label("0" * int(qc.num_qubits))
    density = cp4b.channel_LCU.simulate_circuit(qc, ini_state=ini_state, qubit_regs=qubit_indexes)
    rho_norm, trace_before_norm = normalize_density(np.asarray(density.data, dtype=complex))
    return rho_norm, trace_before_norm, qc, build_time


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


def format_kraus_section(title: str, kraus_ops: list[Any]) -> list[str]:
    lines = [title]
    for row in matrixsum_summary(kraus_ops):
        lines.append(
            f"K{row['kraus_index']}: size={row['size']}, terms={row['term_count']}, "
            f"pauli_norm={row['pauli_norm']:.16g}"
        )
        for term in row["terms"]:
            coeff = term["coeff"]
            lines.append(
                f"  {coeff['real']:+.16g}{coeff['imag']:+.16g}j * {term['pauli']}"
            )
    return lines


def run() -> dict[str, Any]:
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()

    append_raw({"event": "start", "checkpoint": "CP4B-random-coloring-state-probe"})

    random_bench = None
    for bench in cp4b.build_benchmarks():
        if bench.name == "random_coloring_n2_q2_linear":
            random_bench = bench
            break
    if random_bench is None:
        raise RuntimeError("random_coloring_n2_q2_linear benchmark not found.")

    raw_ops = random_bench.kraus_ops
    rewritten_ops, rewrite_result, rewrite_elapsed = cp4b.rewrite_channel(raw_ops, max_steps=50, tol=1e-12)
    raw_metrics = cp4b.channel_metrics(raw_ops)
    rewritten_metrics = cp4b.channel_metrics(rewritten_ops)

    # F uses the same raw target Kraus representation as Basic/O; only the
    # per-Kraus block encoding changes to Matrix-order. All uses the rewritten
    # target Kraus representation plus structure=opt and Matrix-order.
    configs = {
        "Basic": (config_by_name("Basic"), raw_ops),
        "O": (config_by_name("O"), raw_ops),
        "F": (config_by_name("F"), raw_ops),
        "All": (config_by_name("All"), rewritten_ops),
    }

    system_qubits = int(raw_metrics["system_qubits"])
    dim = 1 << system_qubits
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[0, 0] = 1.0
    direct_reference, direct_trace_before_norm = apply_direct_channel(raw_ops, rho0, system_qubits)

    rows: list[dict[str, Any]] = []
    densities: dict[str, np.ndarray] = {"direct": direct_reference}
    for name, (config, target_ops) in configs.items():
        started = time.perf_counter()
        rho, trace_before_norm, qc, build_time = simulate_config(random_bench, target_ops, config)
        densities[name] = rho
        reference_ops = raw_ops if name != "All" else rewritten_ops
        reference, reference_trace_before_norm = apply_direct_channel(reference_ops, rho0, system_qubits)
        vs_direct = diff_metrics(reference, rho)
        row = {
            "benchmark": random_bench.name,
            "configuration": name,
            "configuration_detail": config.label,
            "input_state": "zero_density",
            "status_vs_direct": "ok" if vs_direct["trace_distance"] <= 1e-10 else "discrepancy",
            "system_qubits": system_qubits,
            "circuit_qubits": int(qc.num_qubits),
            "target_kraus_representation": "rewritten" if name == "All" else "raw",
            "target_kraus_count": int(cp4b.channel_metrics(target_ops)["kraus_count"]),
            "target_pauli_terms": int(cp4b.channel_metrics(target_ops)["pauli_terms_total"]),
            "density_trace_before_normalization": trace_before_norm,
            "direct_trace_before_normalization": reference_trace_before_norm,
            "trace_distance_vs_direct": vs_direct["trace_distance"],
            "relative_frobenius_diff_vs_direct": vs_direct["relative_frobenius_diff"],
            "max_entry_diff_vs_direct": vs_direct["max_entry_diff"],
            "build_time_s": build_time,
            "elapsed_s": time.perf_counter() - started,
        }
        rows.append(row)
        append_raw({"event": "config_probe", "row": row})

    direct_ob = diff_metrics(densities["Basic"], densities["O"])
    direct_of = diff_metrics(densities["O"], densities["F"])
    direct_oa = diff_metrics(densities["O"], densities["All"])
    direct_bf = diff_metrics(densities["Basic"], densities["F"])
    direct_ba = diff_metrics(densities["Basic"], densities["All"])
    pairwise = [
        {"lhs": "Basic", "rhs": "O", **direct_ob},
        {"lhs": "O", "rhs": "F", **direct_of},
        {"lhs": "O", "rhs": "All", **direct_oa},
        {"lhs": "Basic", "rhs": "F", **direct_bf},
        {"lhs": "Basic", "rhs": "All", **direct_ba},
    ]
    append_raw({"event": "pairwise", "rows": pairwise})

    inspection = {
        "benchmark": random_bench.name,
        "family": random_bench.family,
        "construction_parameters": random_bench.model_info,
        "raw_metrics": raw_metrics,
        "rewrite_result": cp4b.summarize_rewrite(rewrite_result),
        "rewrite_elapsed_s": rewrite_elapsed,
        "rewritten_metrics": rewritten_metrics,
        "raw_kraus": matrixsum_summary(raw_ops),
        "F_target_kraus_note": "F uses raw Kraus operators; it changes per-Kraus block encoding to Matrix-order.",
        "F_target_kraus": matrixsum_summary(raw_ops),
        "All_target_kraus_note": "All uses rewritten Kraus operators plus structure=opt and Matrix-order.",
        "All_target_kraus": matrixsum_summary(rewritten_ops),
        "rewritten_kraus_for_R_or_All_reference": matrixsum_summary(rewritten_ops),
        "state_probe_rows": rows,
        "pairwise_normalized_density_comparisons": pairwise,
    }

    OUT_JSON.write_text(json.dumps(inspection, indent=2, default=json_default, sort_keys=True), encoding="utf-8")

    text_lines = [
        "# RandomColoring-small Inspection",
        "",
        "## Construction Parameters",
        json.dumps(random_bench.model_info, indent=2, default=json_default, sort_keys=True),
        "",
        "## Metrics",
        f"raw_metrics = {json.dumps(raw_metrics, sort_keys=True)}",
        f"rewritten_metrics = {json.dumps(rewritten_metrics, sort_keys=True)}",
        f"rewrite_result = {json.dumps(cp4b.summarize_rewrite(rewrite_result), sort_keys=True)}",
        "",
        "## Raw Kraus Operators",
        *format_kraus_section("", raw_ops),
        "",
        "## F Target Kraus Operators",
        "F uses the same raw target Kraus operators as Basic/O; only the per-Kraus block encoding uses Matrix-order.",
        *format_kraus_section("", raw_ops),
        "",
        "## Rewritten Kraus Operators For R/All Reference",
        *format_kraus_section("", rewritten_ops),
        "",
        "## All Target Kraus Operators",
        "All uses the rewritten Kraus operators plus structure=opt and Matrix-order.",
        *format_kraus_section("", rewritten_ops),
        "",
        "## Normalized Density Probe",
    ]
    for row in rows:
        text_lines.append(
            f"{row['configuration']}: status_vs_direct={row['status_vs_direct']}, "
            f"trace_distance_vs_direct={row['trace_distance_vs_direct']:.16g}, "
            f"rel_fro_vs_direct={row['relative_frobenius_diff_vs_direct']:.16g}, "
            f"circuit_qubits={row['circuit_qubits']}"
        )
    text_lines.extend(["", "## Pairwise Normalized Density Comparisons"])
    for row in pairwise:
        text_lines.append(
            f"{row['lhs']} vs {row['rhs']}: trace_distance={row['trace_distance']:.16g}, "
            f"rel_fro={row['relative_frobenius_diff']:.16g}, max_entry={row['max_entry_diff']:.16g}"
        )
    OUT_TXT.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    fieldnames = [
        "benchmark",
        "configuration",
        "configuration_detail",
        "input_state",
        "status_vs_direct",
        "system_qubits",
        "circuit_qubits",
        "target_kraus_representation",
        "target_kraus_count",
        "target_pauli_terms",
        "density_trace_before_normalization",
        "direct_trace_before_normalization",
        "trace_distance_vs_direct",
        "relative_frobenius_diff_vs_direct",
        "max_entry_diff_vs_direct",
        "build_time_s",
        "elapsed_s",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    append_raw({"event": "finish", "inspection": inspection})
    return inspection


def main() -> None:
    result = run()
    print(OUT_TXT.read_text(encoding="utf-8"))
    print(json.dumps(result["pairwise_normalized_density_comparisons"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
