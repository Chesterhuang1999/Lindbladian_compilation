#!/usr/bin/env python3
"""Run QuiZX on the table1 Nam QASM2 circuit and save gate-count results."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "circuits" / "select_only" / "test_table1_m_no_n4_nam_qasm2.qasm"
DEFAULT_RESULT_DIR = ROOT / "Baseline_results" / "quizx" / "nam" / "TFIM_4"
DEFAULT_QUIZX = ROOT / "external_tools" / "quizx" / "target" / "release" / "quizx"
SKIP_QASM_HEADS = {
    "openqasm",
    "include",
    "qreg",
    "creg",
    "gate",
    "opaque",
    "barrier",
    "measure",
}


def qasm_gate_counts(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or not line.endswith(";"):
            continue
        head = line.split(None, 1)[0]
        name = head.split("(", 1)[0].lower()
        if name in SKIP_QASM_HEADS:
            continue
        counts[name] += 1
    return {"total": int(sum(counts.values())), "ops": dict(sorted(counts.items()))}


def build_command(quizx: Path, method: str, input_path: Path, output_path: Path) -> list[str]:
    command = [str(quizx), "opt", str(input_path), "--out", str(output_path)]
    if method != "full":
        command.append(f"--{method}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize table1 Nam QASM2 with QuiZX.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--quizx", type=Path, default=DEFAULT_QUIZX)
    parser.add_argument("--method", choices=["full", "flow", "clifford"], default="full")
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    output_qasm = args.result_dir / "test_table1_TFIM_quizx_output.qasm"
    stat_txt = args.result_dir / "test_table1_TFIM_quizx_stat.txt"
    summary_json = args.result_dir / "test_table1_TFIM_quizx_summary.json"

    command = build_command(args.quizx, args.method, args.input, output_qasm)
    before = qasm_gate_counts(args.input)

    start = time.perf_counter()
    proc = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - start

    stat_text = proc.stdout
    if proc.stderr:
        stat_text += "\n[stderr]\n" + proc.stderr
    stat_txt.write_text(stat_text)

    after = qasm_gate_counts(output_qasm) if output_qasm.exists() else None
    comparison = None
    if after is not None:
        all_gates = sorted(set(before["ops"]) | set(after["ops"]))
        comparison = {
            "total_delta": int(after["total"] - before["total"]),
            "total_reduction": int(before["total"] - after["total"]),
            "total_reduction_ratio": (before["total"] - after["total"]) / before["total"],
            "ops_delta": {
                gate: int(after["ops"].get(gate, 0) - before["ops"].get(gate, 0))
                for gate in all_gates
            },
        }

    summary = {
        "tool": "quizx",
        "method": args.method,
        "input": str(args.input),
        "output_qasm": str(output_qasm),
        "stat_txt": str(stat_txt),
        "summary_json": str(summary_json),
        "command": command,
        "returncode": int(proc.returncode),
        "elapsed_seconds": elapsed,
        "before": before,
        "after": after,
        "comparison": comparison,
        "note": "QuiZX does not emit a gate-count summary; counts were computed from QASM operation lines.",
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
