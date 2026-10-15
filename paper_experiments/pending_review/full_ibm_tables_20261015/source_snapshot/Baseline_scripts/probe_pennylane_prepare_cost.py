#!/usr/bin/env python3
"""Audit additive PREP estimates without synthesizing large dense matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pennylane as qml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Baseline_scripts"))

from pennylane_lcu_adapter import (  # noqa: E402
    LEAF_GATE_SET,
    _decomposition_mode,
    _inject_select_workspace,
    build_lcu_tape,
    compile_lcu_tape,
    tape_to_qiskit,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import recount_qiskit_circuit  # noqa: E402


def counts(tape, wires, directory):
    lowered, _ = compile_lcu_tape(tape, wires, optimize=True, decomposition="graph")
    circuit = tape_to_qiskit(lowered, wires)
    result = {}
    for basis in ("nam", "ibm"):
        stats = recount_qiskit_circuit(
            circuit, basis_name=basis, out_path=directory / f"{basis}.qasm",
            optimization_level=0,
        )["stats"]
        result[basis] = {
            "gate_count": int(stats["metric_total"]),
            "non_clifford": int(stats["clifford"]["non_clifford"]),
            "reserved_width": int(stats["num_qubits"]),
        }
    return result


def native_components(block, metadata):
    """Keep the adapter's workspace route and stop at its native outer pieces."""
    stop = LEAF_GATE_SET | {
        "Select", "AmplitudeEmbedding", "Adjoint(AmplitudeEmbedding)",
        "StatePrep", "Adjoint(StatePrep)",
    }
    with _decomposition_mode("graph"):
        expanded = _inject_select_workspace(block, metadata["work_wires"])
        [expanded], _ = qml.transforms.decompose(expanded, gate_set=stop, num_work_wires=0)
    if len(expanded.operations) != 3 or not isinstance(expanded.operations[1], qml.Select):
        raise RuntimeError(f"Unexpected native PREP/SELECT/adjoint structure: {expanded.operations}")
    prep, select, unprep = expanded.operations
    if not select.partial:
        raise RuntimeError("Expected native PrepSelPrep to contain partial=True Select")
    return prep, select, unprep


def preparation_counts(amplitudes, num_control, directory):
    """Count separate forward and inverse state preparations on selector wires."""
    if num_control == 0:
        metrics = {basis: {"gate_count": 0, "non_clifford": 0, "reserved_width": 0}
                   for basis in ("nam", "ibm")}
        return {
            "forward": metrics,
            "adjoint": metrics,
            "pair": metrics,
            "timings_seconds": {"total": 0.0, "forward": 0.0, "adjoint": 0.0},
        }
    wires = list(range(num_control))
    with qml.QueuingManager.stop_recording():
        prep = qml.AmplitudeEmbedding(amplitudes, normalize=True, pad_with=0, wires=wires)
        pieces = {
            "forward": qml.tape.QuantumScript([prep]),
            "adjoint": qml.tape.QuantumScript([qml.adjoint(prep)]),
        }
    metrics, timings = {}, {}
    started = time.perf_counter()
    for name, tape in pieces.items():
        piece_started = time.perf_counter()
        metrics[name] = counts(tape, wires, directory / name)
        timings[name] = time.perf_counter() - piece_started
    metrics["pair"] = {
        basis: {
            "gate_count": sum(metrics[direction][basis]["gate_count"] for direction in pieces),
            "non_clifford": sum(metrics[direction][basis]["non_clifford"] for direction in pieces),
            "reserved_width": num_control,
        }
        for basis in ("nam", "ibm")
    }
    timings["total"] = time.perf_counter() - started
    metrics["timings_seconds"] = timings
    return metrics


def all_preparations(args, manifest):
    started = time.perf_counter()
    cache = {}
    result = {
        "schema_version": 1,
        "pennylane_version": qml.__version__,
        "input_manifest": str(args.inputs),
        "input_manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "status": "running",
        "policy": {
            "preparation": "AmplitudeEmbedding(sqrt(abs(coefficients)), normalize=True, pad_with=0); normalized before construction for exact-vector cache reuse",
            "compilation": "Native graph decomposition, default qml.compile, Qiskit opt0 NAM/IBM recount",
            "counting": "Forward and adjoint compiled separately and gate/non-Clifford counts added; no cross-boundary optimization",
            "width": "Selector wires only. Pair width is shared selector width, not the sum of forward and inverse widths",
            "cache": "Exact IEEE-754 bytes of padded normalized real amplitude vector; no approximate rounding",
            "aggregation": "sum_kraus sums, sample_mean uses arithmetic mean and sample standard deviation (ddof=1), single preserves its only input",
            "limitation": "Measured common preparation cost does not prove composability with another tool's optimized SELECT: selector address permutations and any removed branches must be reconciled",
        },
        "cases": [],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for case in manifest["cases"]:
        entry = {"case": case["name"], "aggregation": case["aggregation"], "inputs": []}
        for item in case["inputs"]:
            input_started = time.perf_counter()
            weights = np.array([abs(complex(term["real"], term["imag"])) for term in item["terms"]], dtype=np.float64)
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError(f"Invalid coefficient weights: {case['name']}/{item['id']}")
            num_control = (len(weights) - 1).bit_length()
            amplitudes = np.sqrt(weights)
            amplitudes /= np.linalg.norm(amplitudes)
            amplitudes = np.pad(amplitudes, (0, 2**num_control - len(amplitudes)))
            amplitude_sha = hashlib.sha256(amplitudes.astype("<f8").tobytes()).hexdigest()
            cache_hit = amplitude_sha in cache
            if not cache_hit:
                cache[amplitude_sha] = preparation_counts(amplitudes, num_control, args.out_dir / "preparations" / amplitude_sha)
            measured = cache[amplitude_sha]
            entry["inputs"].append({
                "id": item["id"], "seed": item.get("seed"), "input_sha256": item["input_sha256"],
                "term_count": len(weights), "num_control_qubits": num_control,
                "normalized_amplitude_sha256": amplitude_sha, "cache_hit": cache_hit,
                "status": "ok", "metrics": {key: value for key, value in measured.items() if key != "timings_seconds"},
                "compile_timings_seconds": None if cache_hit else measured["timings_seconds"],
                "elapsed_seconds": time.perf_counter() - input_started,
            })
        entry["input_count"] = len(entry["inputs"])
        entry["basis_summary"] = {}
        for basis in ("nam", "ibm"):
            metrics = [item["metrics"]["pair"][basis] for item in entry["inputs"]]
            summary = {"width": max(value["reserved_width"] for value in metrics)}
            for metric in ("gate_count", "non_clifford"):
                values = [value[metric] for value in metrics]
                summary[metric] = (statistics.mean(values) if case["aggregation"] == "sample_mean" else sum(values))
                if case["aggregation"] == "sample_mean":
                    summary[metric + "_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
            entry["basis_summary"][basis] = summary
        result["cases"].append(entry)
        (args.out_dir / "summary.partial.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"case": case["name"], "input_count": entry["input_count"], "basis_summary": entry["basis_summary"]}), flush=True)
    result.update({
        "status": "ok", "total_inputs": sum(case["input_count"] for case in result["cases"]),
        "unique_preparations": len(cache), "total_seconds": time.perf_counter() - started,
        "unique_compile_seconds": math.fsum(item["timings_seconds"]["total"] for item in cache.values()),
    })
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "Baseline_results/pennylane_lcu_v1/inputs.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Baseline_results/pennylane_prepare_cost_probe")
    parser.add_argument("--all-preparations", action="store_true", help="Count only coefficient PREP and PREP adjoint for every manifest input")
    args = parser.parse_args()
    manifest = json.loads(args.inputs.read_text())
    if args.all_preparations:
        all_preparations(args, manifest)
        return
    selected = ("TFIM_3_A0", "random_pauli_4q_20t", "HRW_8")
    result = {
        "pennylane_version": qml.__version__,
        "input_manifest": str(args.inputs),
        "scope": "Three representative inputs, not whole-case aggregate benchmark results",
        "policy": "Native graph decomposition, default qml.compile, Qiskit opt0 NAM/IBM recount. PREP and its adjoint are compiled separately; concatenating them alone would cancel.",
        "interpretation": "block - full_address_select is a residual accounting estimate. It need not equal isolated PREP plus adjoint cost, due to partial SELECT and cross-boundary optimization. Delta NC is likewise a derived count, not a physical PREP count. No additive depth/time/width inference is justified.",
        "cases": [],
    }
    for name in selected:
        case = next(case for case in manifest["cases"] if case["name"] == name)
        item = case["inputs"][0]
        terms = [(term["pauli"], complex(term["real"], term["imag"])) for term in item["terms"]]
        entry = {"case": name, "input_id": item["id"], "input_sha256": item["input_sha256"], "term_count": len(terms), "profiles": {}}
        for work in (0, max(0, (len(terms) - 1).bit_length() - 1)):
            profile = f"graph_work_{work}"
            if profile in entry["profiles"]:
                continue
            directory = args.out_dir / name / profile
            block, wires, metadata = build_lcu_tape(terms, item["num_system_qubits"], scope="block", work_wires=work)
            standalone, _, _ = build_lcu_tape(terms, item["num_system_qubits"], scope="select", work_wires=work)
            prep, partial, unprep = native_components(block, metadata)
            pieces = {
                "block": block,
                "standalone_full_select": standalone,
                "internal_partial_select": qml.tape.QuantumScript([partial]),
                "prep": qml.tape.QuantumScript([prep]),
                "unprep": qml.tape.QuantumScript([unprep]),
            }
            metrics = {piece: counts(tape, wires, directory / piece) for piece, tape in pieces.items()}
            residuals = {}
            for basis in ("nam", "ibm"):
                residuals[basis] = {}
                for metric in ("gate_count", "non_clifford"):
                    value = {piece: counts_[basis][metric] for piece, counts_ in metrics.items()}
                    pair = value["prep"] + value["unprep"]
                    residuals[basis][metric] = {
                        "block_minus_full_select": value["block"] - value["standalone_full_select"],
                        "block_minus_partial_select": value["block"] - value["internal_partial_select"],
                        "isolated_prep_pair": pair,
                        "partial_minus_full_select": value["internal_partial_select"] - value["standalone_full_select"],
                        "block_minus_isolated_piece_sum": value["block"] - value["internal_partial_select"] - pair,
                    }
            entry["profiles"][profile] = {"work_wires": work, "native_prep_name": prep.name, "metrics": metrics, "residuals": residuals}
            print(json.dumps({"case": name, "input_id": item["id"], "profile": profile, "residuals": residuals}), flush=True)
        result["cases"].append(entry)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
