#!/usr/bin/env python3
"""Unified Feynman baseline runner for QASM inputs."""

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
    count_qasm_text,
    load_qasm_stats_or_count,
    require_input_qasm,
    run_command,
    tail_text,
    write_summary,
)


DEFAULT_FEYNOPT = (
    ROOT
    / "external_tools"
    / "feynman"
    / "dist-newstyle"
    / "build"
    / "aarch64-osx"
    / "ghc-9.14.1"
    / "Feynman-0.1.0.0"
    / "x"
    / "feynopt"
    / "build"
    / "feynopt"
    / "feynopt"
)
COUNT_LINE_RE = re.compile(r"^//\s+([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?):\s+(\d+)\s*$")


def split_feynman_stdout(stdout: str) -> tuple[str, str]:
    marker = "OPENQASM 2.0;"
    index = stdout.find(marker)
    if index < 0:
        return stdout, ""
    return stdout[:index], stdout[index:]


def parse_comment_counts(section: str) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    current: str | None = None

    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("// Original"):
            current = "input"
            counts[current] = {"ops": {}}
            continue
        if stripped.startswith("// Result"):
            current = "output"
            counts[current] = {"ops": {}}
            continue
        if current is None:
            continue
        match = COUNT_LINE_RE.match(stripped)
        if not match:
            continue
        key, value = match.group(1).lower(), int(match.group(2))
        if key in {"qubits", "cbits"}:
            counts[current][key] = value
        else:
            counts[current]["ops"][key] = value

    for data in counts.values():
        ops = data.get("ops", {})
        data["total"] = int(sum(ops.values()))
    return counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = require_input_qasm(Path(args.input))
    input_stats = load_qasm_stats_or_count(
        input_path,
        stats_path=Path(args.input_stats).resolve() if args.input_stats else None,
    )
    gate_set = str(input_stats["detected_gate_set"])

    feynopt = Path(args.feynopt).resolve()
    if not feynopt.exists():
        raise FileNotFoundError(f"Feynman feynopt executable not found: {feynopt}")

    output_path = Path(args.output).resolve() if args.output else None
    command = [str(feynopt), *args.passes, str(input_path)]
    proc = run_command(command, timeout=args.timeout)
    stats_section, qasm_section = split_feynman_stdout(proc["stdout"])
    output_stats = count_qasm_text(qasm_section, gate_set=gate_set) if qasm_section else None

    with tempfile.TemporaryDirectory(prefix="feynman_baseline_") as tmp_dir:
        if qasm_section and output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(qasm_section, encoding="utf-8")
            output_stats = count_qasm_file(output_path, gate_set=gate_set)
        elif qasm_section:
            tmp_output = Path(tmp_dir) / "optimized.qasm"
            tmp_output.write_text(qasm_section, encoding="utf-8")
            output_stats = count_qasm_file(tmp_output, gate_set=gate_set)
            output_stats["path"] = None

    summary: dict[str, Any] = {
        "tool": "feynman",
        "input": str(input_path),
        "detected_gate_set": gate_set,
        "passes": list(args.passes),
        "kept_output": output_path is not None,
        "output_qasm": str(output_path) if output_path is not None and qasm_section else None,
        "input_stats": input_stats,
        "output_stats": output_stats,
        "comparison": (
            compare_gate_stats(input_stats, output_stats, gate_set)
            if output_stats is not None
            else None
        ),
        "feynman_reported_counts": parse_comment_counts(stats_section),
        "process": {
            "command": command,
            "returncode": proc["returncode"],
            "elapsed_seconds": proc["elapsed_seconds"],
            "timed_out": proc["timed_out"],
            "stats_stdout_tail": tail_text(stats_section),
            "stderr_tail": tail_text(proc["stderr"]),
            "output_written": bool(qasm_section),
        },
        "note": "Feynman emits stats and QASM on stdout; the runner splits at OPENQASM 2.0.",
    }

    write_summary(summary, Path(args.summary).resolve() if args.summary else None)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Feynman feynopt on one QASM input and report gate counts."
    )
    parser.add_argument("--input", required=True, help="Input OpenQASM2 file.")
    parser.add_argument("--input-stats", default=None, help="Optional cached input stats JSON.")
    parser.add_argument("--output", default=None, help="Optional path to keep optimized QASM.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument("--feynopt", default=str(DEFAULT_FEYNOPT), help="Feynman feynopt executable.")
    parser.add_argument(
        "--passes",
        nargs="+",
        default=["-O2"],
        help="Feynman optimization passes, e.g. --passes -O2.",
    )
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
