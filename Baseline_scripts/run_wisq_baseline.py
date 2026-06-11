#!/usr/bin/env python3
"""Unified WISQ/GUOQ baseline runner for IBM/NAM QASM inputs."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_common import (  # noqa: E402
    compare_gate_stats,
    count_qasm_file,
    require_input_qasm,
    run_command,
    tail_text,
    write_summary,
)


WISQ_GATESETS = {
    "ibm": "IBMO",
    "nam": "NAM",
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = require_input_qasm(Path(args.input))
    input_stats = count_qasm_file(input_path)
    gate_set = str(input_stats["detected_gate_set"])
    target_gateset = args.target_gateset or WISQ_GATESETS[gate_set]
    output_path = Path(args.output).resolve() if args.output else None

    with tempfile.TemporaryDirectory(prefix="wisq_baseline_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_input = tmp_dir_path / input_path.name
        actual_output_path = output_path or (tmp_dir_path / "optimized.qasm")
        shutil.copy2(input_path, tmp_input)

        command = [
            args.wisq,
            str(tmp_input),
            "--mode",
            "opt",
            "--target_gateset",
            target_gateset,
            "--optimization_objective",
            args.optimization_objective,
            "--opt_timeout",
            str(int(args.timeout)),
            "--output_path",
            str(actual_output_path),
        ]
        if args.approx_epsilon is not None:
            command.extend(["--approx_epsilon", str(args.approx_epsilon)])
        if int(args.opt_threads) > 1:
            command.extend(["--opt_threads", str(int(args.opt_threads))])
        if args.advanced_args:
            command.extend(["--advanced_args", str(Path(args.advanced_args).resolve())])
        if args.verbose:
            command.append("--verbose")

        proc = run_command(command)
        output_exists = actual_output_path.exists()
        output_stats = (
            count_qasm_file(actual_output_path, gate_set=gate_set)
            if output_exists
            else None
        )
        if output_stats is not None and output_path is None:
            output_stats["path"] = None

        summary: dict[str, Any] = {
            "tool": "wisq",
            "input": str(input_path),
            "detected_gate_set": gate_set,
            "wisq_target_gateset": target_gateset,
            "optimization_objective": args.optimization_objective,
            "timeout_seconds": int(args.timeout),
            "kept_output": output_path is not None,
            "output_qasm": str(actual_output_path) if output_path is not None else None,
            "input_stats": input_stats,
            "output_stats": output_stats,
            "comparison": (
                compare_gate_stats(input_stats, output_stats, gate_set)
                if output_stats is not None
                else None
            ),
            "process": {
                "command": command,
                "returncode": proc["returncode"],
                "elapsed_seconds": proc["elapsed_seconds"],
                "timed_out": proc["timed_out"],
                "stdout_tail": tail_text(proc["stdout"]),
                "stderr_tail": tail_text(proc["stderr"]),
                "output_written": output_exists,
            },
        }

    write_summary(summary, Path(args.summary).resolve() if args.summary else None)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WISQ/GUOQ on one QASM input, infer IBM/NAM, and report gate counts."
    )
    parser.add_argument("--input", required=True, help="Input OpenQASM2 file.")
    parser.add_argument("--output", default=None, help="Optional path to keep optimized QASM.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument("--wisq", default="wisq", help="WISQ executable.")
    parser.add_argument("--target-gateset", default=None, choices=["NAM", "CLIFFORDT", "IBMO", "IBMN", "ION"])
    parser.add_argument(
        "--optimization-objective",
        default="TOTAL",
        choices=["TWO_Q", "FIDELITY", "FT", "TOTAL", "T"],
    )
    parser.add_argument("--timeout", type=int, default=3600, help="WISQ --opt_timeout in seconds.")
    parser.add_argument("--approx-epsilon", type=float, default=None)
    parser.add_argument("--opt-threads", type=int, default=1)
    parser.add_argument("--advanced-args", default=None)
    parser.add_argument("--verbose", action="store_true")
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
