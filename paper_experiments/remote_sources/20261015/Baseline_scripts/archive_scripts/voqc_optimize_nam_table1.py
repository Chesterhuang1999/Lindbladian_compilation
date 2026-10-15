#!/usr/bin/env python3
"""Run a minimal VOQC workflow on the table1 Nam QASM2 circuit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Baseline_scripts.normalize_qasm_rz_for_voqc import normalize_qasm_file


COUNT_RE = re.compile(
    r"^(Original|Final):\s*Total\s+(\d+),\s*Rz\s+(\d+),\s*Clifford\s+(\d+),"
    r"\s*T\s+(\d+|N/A),\s*H\s+(\d+),\s*X\s+(\d+),\s*CNOT\s+(\d+)"
)


def parse_voqc_counts(text: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        match = COUNT_RE.match(line.strip())
        if not match:
            continue
        label = match.group(1).lower()
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize and optimize Nam QASM2 with VOQC.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("circuits/select_only/test_table1_m_no_n4_nam_qasm2.qasm"),
        help="Original Nam QASM2 input.",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=Path("circuits/test_table1_m_no_n4_nam_qasm2_normal.qasm"),
        help="VOQC-friendly normalized QASM2 path.",
    )
    parser.add_argument(
        "--voqc-exe",
        type=Path,
        default=Path("external_tools/VOQC/VOQC/_build/default/voqc.exe"),
        help="Path to the VOQC executable.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("Baseline_results/voqc/nam/TFIM_4"),
        help="Directory for VOQC output and stats.",
    )
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    output_qasm = args.result_dir / "test_table1_TFIM_voqc_output.qasm"
    stat_txt = args.result_dir / "test_table1_TFIM_voqc_stat.txt"
    summary_json = args.result_dir / "test_table1_TFIM_voqc_summary.json"

    replacements = normalize_qasm_file(args.input, args.normalized_output)

    command = [
        str(args.voqc_exe),
        "-i",
        str(args.normalized_output),
        "-o",
        str(output_qasm),
    ]

    start = time.perf_counter()
    proc = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - start

    combined_log = proc.stdout
    if proc.stderr:
        combined_log += "\n[stderr]\n" + proc.stderr
    stat_txt.write_text(combined_log)

    summary = {
        "input": str(args.input),
        "normalized_input": str(args.normalized_output),
        "output_qasm": str(output_qasm),
        "stat_txt": str(stat_txt),
        "rz_replacements": replacements,
        "command": command,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "counts": parse_voqc_counts(combined_log),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(combined_log, end="" if combined_log.endswith("\n") else "\n")
    print(f"summary: {summary_json}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
