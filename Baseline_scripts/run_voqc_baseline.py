#!/usr/bin/env python3
"""Unified VOQC baseline runner for NAM QASM inputs."""

from __future__ import annotations

import argparse
import re
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
    load_qasm_stats_or_count,
    require_input_qasm,
    run_command,
    tail_text,
    write_summary,
)
from lower_voqc_qasm_to_nam import lower_voqc_qasm_file_to_nam  # noqa: E402
from normalize_qasm_rz_for_voqc import normalize_qasm_file  # noqa: E402


DEFAULT_VOQC = ROOT / "external_tools" / "VOQC" / "VOQC" / "_build" / "default" / "voqc.exe"
COUNT_RE = re.compile(
    r"^(Original|Final):\s*Total\s+(\d+),\s*Rz\s+(\d+),\s*Clifford\s+(\d+),"
    r"\s*T\s+(\d+|N/A),\s*H\s+(\d+),\s*X\s+(\d+),\s*CNOT\s+(\d+)"
)


def parse_voqc_counts(text: str) -> dict[str, dict[str, int | None]]:
    counts: dict[str, dict[str, int | None]] = {}
    for line in text.splitlines():
        match = COUNT_RE.match(line.strip())
        if not match:
            continue
        label = "input" if match.group(1) == "Original" else "output"
        counts[label] = {
            "total": int(match.group(2)),
            "rz": int(match.group(3)),
            "clifford": int(match.group(4)),
            "t": None if match.group(5) == "N/A" else int(match.group(5)),
            "h": int(match.group(6)),
            "x": int(match.group(7)),
            "cx": int(match.group(8)),
        }
    return counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = require_input_qasm(Path(args.input))
    input_stats = load_qasm_stats_or_count(
        input_path,
        stats_path=Path(args.input_stats).resolve() if args.input_stats else None,
    )
    gate_set = str(input_stats["detected_gate_set"])
    if gate_set != "nam":
        raise ValueError("VOQC baseline is NAM-only in this workflow; IBM input is not supported.")

    voqc = Path(args.voqc).resolve()
    if not voqc.exists():
        raise FileNotFoundError(f"VOQC executable not found: {voqc}")

    output_path = Path(args.output).resolve() if args.output else None
    with tempfile.TemporaryDirectory(prefix="voqc_baseline_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        normalized_input = tmp_dir_path / "input_voqc_normalized.qasm"
        actual_output_path = output_path or (tmp_dir_path / "optimized.qasm")
        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        replacements = normalize_qasm_file(input_path, normalized_input)
        normalized_stats = count_qasm_file(normalized_input, gate_set=gate_set)

        command = [str(voqc), "-i", str(normalized_input), "-o", str(actual_output_path)]
        if args.iterations is not None:
            command.extend(["-n", str(int(args.iterations))])

        proc = run_command(command, timeout=args.timeout)
        output_exists = actual_output_path.exists()
        lowered_output_path = actual_output_path.with_name(
            f"{actual_output_path.stem}_nam_lowered.qasm"
        )
        lowering_info = None
        if output_exists:
            lowering_info = lower_voqc_qasm_file_to_nam(
                actual_output_path,
                lowered_output_path,
            )
        output_stats = (
            count_qasm_file(lowered_output_path, gate_set=gate_set)
            if output_exists
            else None
        )
        if output_stats is not None and output_path is None:
            output_stats["path"] = None
            normalized_stats["path"] = None
        combined_log = proc["stdout"]
        if proc["stderr"]:
            combined_log += "\n[stderr]\n" + proc["stderr"]

        summary: dict[str, Any] = {
            "tool": "voqc",
            "input": str(input_path),
            "detected_gate_set": gate_set,
            "kept_output": output_path is not None,
            "output_qasm": str(actual_output_path) if output_path is not None else None,
            "lowered_output_qasm": str(lowered_output_path)
            if output_path is not None and output_exists
            else None,
            "output_basis_lowering": lowering_info,
            "rz_angle_normalization": {
                "replacements": replacements,
                "stats": normalized_stats,
            },
            "input_stats": input_stats,
            "output_stats": output_stats,
            "comparison": (
                compare_gate_stats(input_stats, output_stats, gate_set)
                if output_stats is not None
                else None
            ),
            "voqc_reported_counts": parse_voqc_counts(combined_log),
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
        description="Normalize RZ angles, run VOQC on one NAM QASM input, and report gate counts."
    )
    parser.add_argument("--input", required=True, help="Input OpenQASM2 file.")
    parser.add_argument("--input-stats", default=None, help="Optional cached input stats JSON.")
    parser.add_argument("--output", default=None, help="Optional path to keep optimized QASM.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument("--voqc", default=str(DEFAULT_VOQC), help="VOQC executable.")
    parser.add_argument("--iterations", type=int, default=None, help="Optional VOQC -n value.")
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
