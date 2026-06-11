#!/usr/bin/env python3
"""Unified pytket baseline runner for IBM/NAM QASM inputs."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_common import (  # noqa: E402
    compare_gate_stats,
    count_qasm_file,
    require_input_qasm,
    write_summary,
)


def build_pytket_pass(pass_name: str):
    from pytket.passes import FullPeepholeOptimise, RemoveRedundancies, SynthesiseTket  # noqa: PLC0415

    if pass_name == "full-peephole":
        return FullPeepholeOptimise()
    if pass_name == "remove-redundancies":
        return RemoveRedundancies()
    if pass_name == "synthesise-tket":
        return SynthesiseTket()
    raise ValueError(f"Unsupported pytket pass: {pass_name}")


def rebase_to_gate_set(circuit, gate_set: str) -> bool:
    from pytket.circuit import OpType  # noqa: PLC0415
    from pytket.passes import AutoRebase  # noqa: PLC0415

    if gate_set == "ibm":
        target = {OpType.U1, OpType.U2, OpType.U3, OpType.CX}
    elif gate_set == "nam":
        target = {OpType.Rz, OpType.H, OpType.X, OpType.CX}
    else:
        raise ValueError(f"Unsupported gate set for rebase: {gate_set}")
    return bool(AutoRebase(target).apply(circuit))


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from pytket.qasm import circuit_from_qasm, circuit_to_qasm  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Could not import pytket. Run with qiskit_simulate or ChannelIR_test."
        ) from exc

    input_path = require_input_qasm(Path(args.input))
    input_stats = count_qasm_file(input_path)
    gate_set = str(input_stats["detected_gate_set"])
    output_path = Path(args.output).resolve() if args.output else None

    start = time.perf_counter()
    circuit = circuit_from_qasm(str(input_path))
    before_tket_gates = int(circuit.n_gates)
    opt_pass = build_pytket_pass(args.pass_name)
    applied = bool(opt_pass.apply(circuit))
    rebase_applied = False
    if not args.no_rebase_to_input_gateset:
        rebase_applied = rebase_to_gate_set(circuit, gate_set)
    after_tket_gates = int(circuit.n_gates)
    elapsed = time.perf_counter() - start

    with tempfile.TemporaryDirectory(prefix="pytket_baseline_") as tmp_dir:
        actual_output_path = output_path or (Path(tmp_dir) / "optimized.qasm")
        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        circuit_to_qasm(circuit, str(actual_output_path))
        output_stats = count_qasm_file(actual_output_path, gate_set=gate_set)
        if output_path is None:
            output_stats["path"] = None

    summary: dict[str, Any] = {
        "tool": "pytket",
        "input": str(input_path),
        "detected_gate_set": gate_set,
        "pass": args.pass_name,
        "kept_output": output_path is not None,
        "output_qasm": str(output_path) if output_path is not None else None,
        "input_stats": input_stats,
        "output_stats": output_stats,
        "comparison": compare_gate_stats(input_stats, output_stats, gate_set),
        "pytket_stats": {
            "pass_applied": applied,
            "rebase_to_input_gateset": not bool(args.no_rebase_to_input_gateset),
            "rebase_applied": rebase_applied,
            "n_gates_before": before_tket_gates,
            "n_gates_after": after_tket_gates,
            "elapsed_seconds": elapsed,
        },
        "note": "pytket may rebase to a different QASM basis; output_stats.ops counts the gates actually emitted.",
    }

    write_summary(summary, Path(args.summary).resolve() if args.summary else None)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a pytket optimization pass on one QASM input and report gate counts."
    )
    parser.add_argument("--input", required=True, help="Input OpenQASM2 file.")
    parser.add_argument("--output", default=None, help="Optional path to keep optimized QASM.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument(
        "--pass-name",
        choices=["full-peephole", "remove-redundancies", "synthesise-tket"],
        default="full-peephole",
    )
    parser.add_argument(
        "--no-rebase-to-input-gateset",
        action="store_true",
        help="Export pytket's natural output basis instead of rebasing back to detected IBM or NAM gates.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
