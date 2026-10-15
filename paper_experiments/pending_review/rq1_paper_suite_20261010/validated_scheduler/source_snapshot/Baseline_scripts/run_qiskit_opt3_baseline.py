#!/usr/bin/env python3
"""Run Qiskit transpile optimization_level=3 on basis-level QASM input."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "Baseline_scripts"
SRC_DIR = ROOT / "src"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


BASIS_GATES = {
    "ibm": ("u1", "u2", "u3", "cx"),
    "nam": ("rz", "h", "x", "cx"),
}


def write_summary(
    *,
    path: Path,
    input_qasm: Path,
    output_qasm: Path,
    basis: str,
    returncode: int,
    elapsed: float,
    output_stats: dict[str, Any] | None,
    error: str | None = None,
) -> None:
    summary = {
        "tool": "qiskit_opt3",
        "basis": basis,
        "input_qasm": str(input_qasm),
        "output_qasm": str(output_qasm) if output_qasm.exists() else None,
        "output_stats": output_stats,
        "process": {
            "returncode": returncode,
            "elapsed_seconds": float(elapsed),
            "timed_out": False,
        },
        "error": error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--basis", choices=("ibm", "nam"), default="ibm")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        qc = QuantumCircuit.from_qasm_file(str(args.input))
        tqc = transpile(
            qc,
            basis_gates=list(BASIS_GATES[args.basis]),
            optimization_level=3,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_openqasm2_quartz_input(tqc, args.output, basis_gates=BASIS_GATES[args.basis])
        output_stats = count_qasm_file(args.output, gate_set=args.basis)
        elapsed = time.perf_counter() - started
        write_summary(
            path=args.summary,
            input_qasm=args.input,
            output_qasm=args.output,
            basis=args.basis,
            returncode=0,
            elapsed=elapsed,
            output_stats=output_stats,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        write_summary(
            path=args.summary,
            input_qasm=args.input,
            output_qasm=args.output,
            basis=args.basis,
            returncode=1,
            elapsed=elapsed,
            output_stats=None,
            error=repr(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
