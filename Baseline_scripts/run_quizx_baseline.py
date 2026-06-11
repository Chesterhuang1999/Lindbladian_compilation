#!/usr/bin/env python3
"""Unified QuiZX baseline runner for NAM QASM inputs."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
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


DEFAULT_QUIZX = ROOT / "external_tools" / "quizx" / "target" / "release" / "quizx"


def build_quizx_command(quizx: Path, method: str, input_path: Path, output_path: Path) -> list[str]:
    command = [str(quizx), "opt", str(input_path), "--out", str(output_path)]
    if method != "full":
        command.append(f"--{method}")
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = require_input_qasm(Path(args.input))
    input_stats = count_qasm_file(input_path)
    gate_set = str(input_stats["detected_gate_set"])
    if gate_set != "nam":
        raise ValueError("QuiZX baseline is NAM-only in this workflow; IBM input is not supported.")

    quizx = Path(args.quizx).resolve()
    if not quizx.exists():
        raise FileNotFoundError(f"QuiZX executable not found: {quizx}")

    output_path = Path(args.output).resolve() if args.output else None
    with tempfile.TemporaryDirectory(prefix="quizx_baseline_") as tmp_dir:
        actual_output_path = output_path or (Path(tmp_dir) / "optimized.qasm")
        command = build_quizx_command(quizx, args.method, input_path, actual_output_path)
        proc = run_command(command, timeout=args.timeout)
        output_exists = actual_output_path.exists()
        output_stats = (
            count_qasm_file(actual_output_path, gate_set=gate_set)
            if output_exists
            else None
        )
        if output_stats is not None and output_path is None:
            output_stats["path"] = None

        summary: dict[str, Any] = {
            "tool": "quizx",
            "input": str(input_path),
            "detected_gate_set": gate_set,
            "method": args.method,
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
            "note": "QuiZX may output a different basis; output_stats.ops counts the gates actually emitted.",
        }

    write_summary(summary, Path(args.summary).resolve() if args.summary else None)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run QuiZX on one NAM QASM input and report gate counts."
    )
    parser.add_argument("--input", required=True, help="Input OpenQASM2 file.")
    parser.add_argument("--output", default=None, help="Optional path to keep optimized QASM.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument("--quizx", default=str(DEFAULT_QUIZX), help="QuiZX executable.")
    parser.add_argument("--method", choices=["full", "flow", "clifford"], default="full")
    parser.add_argument("--timeout", type=float, default=None, help="Optional outer timeout in seconds.")
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
