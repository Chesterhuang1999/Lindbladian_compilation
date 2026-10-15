#!/usr/bin/env python3
"""Run Feynman on the table1 Nam QASM2 circuit and save gate-count results."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "circuits" / "select_only" / "test_table1_m_no_n4_nam_qasm2.qasm"
DEFAULT_RESULT_DIR = ROOT / "Baseline_results" / "feynman" / "nam" / "TFIM_4"
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


def parse_feynman_stdout(stdout: str) -> tuple[str, str]:
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
            current = "original"
            counts[current] = {"ops": {}}
            continue
        if stripped.startswith("// Result"):
            current = "final"
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


def qasm_gate_counts(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return {"total": 0, "ops": {}}

    for raw_line in path.read_text().splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or not line.endswith(";"):
            continue
        head = line.split(None, 1)[0]
        name = head.split("(", 1)[0].lower()
        if name in {"openqasm", "include", "qreg", "creg", "gate", "opaque", "barrier", "measure"}:
            continue
        counts[name] += 1
    return {"total": int(sum(counts.values())), "ops": dict(sorted(counts.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize table1 Nam QASM2 with Feynman.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--feynopt", type=Path, default=DEFAULT_FEYNOPT)
    parser.add_argument(
        "--passes",
        nargs="+",
        default=["-O2"],
        help="Feynman optimization passes, e.g. --passes -O2 or --passes -phasefold -simplify.",
    )
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    output_qasm = args.result_dir / "test_table1_TFIM_feynman_output.qasm"
    stat_txt = args.result_dir / "test_table1_TFIM_feynman_stat.txt"
    raw_stdout = args.result_dir / "test_table1_TFIM_feynman_stdout.txt"
    summary_json = args.result_dir / "test_table1_TFIM_feynman_summary.json"

    command = [str(args.feynopt), *args.passes, str(args.input)]
    start = time.perf_counter()
    proc = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - start

    stats_section, qasm_section = parse_feynman_stdout(proc.stdout)
    raw_stdout.write_text(proc.stdout)
    stat_text = stats_section
    if proc.stderr:
        stat_text += "\n[stderr]\n" + proc.stderr
    stat_txt.write_text(stat_text)
    if qasm_section:
        output_qasm.write_text(qasm_section)

    parsed_counts = parse_comment_counts(stats_section)
    output_counts = qasm_gate_counts(output_qasm)
    if "final" in parsed_counts and output_counts["total"]:
        parsed_counts["final_from_output_qasm"] = output_counts

    summary = {
        "tool": "feynman",
        "input": str(args.input),
        "output_qasm": str(output_qasm),
        "stat_txt": str(stat_txt),
        "raw_stdout": str(raw_stdout),
        "summary_json": str(summary_json),
        "command": command,
        "returncode": int(proc.returncode),
        "elapsed_seconds": elapsed,
        "counts": parsed_counts,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(stat_text, end="" if stat_text.endswith("\n") else "\n")
    print(f"summary: {summary_json}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
