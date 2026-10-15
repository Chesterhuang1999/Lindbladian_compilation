#!/usr/bin/env python3
"""Load the sibling circuit_ibm.qpy; inspect by default, simulate explicitly.

This template is copied into each benchmark artifact directory alongside the
compiled QPY and summary.json. It does not import the source compiler.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent


def build_circuit():
    """Return the compiled circuit stored beside this executable program."""
    from qiskit import qpy

    with (ARTIFACT_DIR / "circuit_ibm.qpy").open("rb") as source:
        circuits = qpy.load(source)
    if len(circuits) != 1:
        raise ValueError(f"Expected one compiled circuit, found {len(circuits)}")
    return circuits[0]


def peak_rss_bytes() -> int | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if platform.system() == "Darwin" else peak * 1024)
    except (ImportError, AttributeError, OSError):
        return None


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Inspect the compiled circuit (default).")
    mode.add_argument("--simulate", action="store_true", help="Run the full statevector simulation.")
    parser.add_argument("--max-memory-mb", type=positive_int, default=8192)
    parser.add_argument("--shots", type=positive_int, default=1)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "status": "ok",
        "action": "simulate" if args.simulate else "check",
        "artifact_directory": str(ARTIFACT_DIR),
    }
    simulation_started: float | None = None
    exit_code = 0
    try:
        circuit = build_circuit()
        report.update({
            "num_qubits": circuit.num_qubits,
            "num_clbits": circuit.num_clbits,
            "depth": circuit.depth(),
            "size": circuit.size(),
            "operations": dict(circuit.count_ops()),
        })
        summary_path = ARTIFACT_DIR / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "qubit_indexes" in summary:
                report["qubit_indexes"] = summary["qubit_indexes"]
        if args.simulate:
            from qiskit_aer import AerSimulator

            report["simulation_options"] = {
                "method": "statevector",
                "precision": "double",
                "max_memory_mb": args.max_memory_mb,
                "max_parallel_threads": 1,
                "enable_truncation": False,
                "seed_simulator": 7,
                "shots": args.shots,
                "input_state": "all_zero",
            }
            run_circuit = circuit.copy()
            run_circuit.measure_all()
            simulation_started = time.perf_counter()
            simulator = AerSimulator(
                method="statevector",
                precision="double",
                max_memory_mb=args.max_memory_mb,
                max_parallel_threads=1,
                enable_truncation=False,
            )
            result = simulator.run(
                run_circuit, shots=args.shots, seed_simulator=7,
            ).result()
            report["simulation_time_s"] = time.perf_counter() - simulation_started
            report["aer_status"] = result.status
            report["aer_success"] = bool(result.success)
            report["aer_metadata"] = getattr(result, "metadata", {})
            report["experiment_status"] = [item.status for item in result.results]
            report["experiment_metadata"] = [item.metadata for item in result.results]
            if result.success:
                report["counts"] = dict(result.get_counts(0))
                report["measurement_qubits"] = list(range(circuit.num_qubits))
            else:
                report["status"] = "simulation_failed"
                exit_code = 1
    except Exception as exc:
        report["status"] = "memory_error" if isinstance(exc, MemoryError) else "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        if simulation_started is not None:
            report["simulation_time_s"] = time.perf_counter() - simulation_started
        exit_code = 1
    report["peak_rss_bytes"] = peak_rss_bytes()
    print(json.dumps(report, indent=2, default=str), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
