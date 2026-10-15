#!/usr/bin/env python3
"""Detach GC 2x4 VOQC and pytket no-timeout reruns."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    ROOT
    / "Baseline_results/eval_redesign/gc_external_3600_20260705"
    / "graph_cluster_2d_grid_2x4/inputs"
)
OUT_DIR = (
    ROOT
    / "Baseline_results/eval_redesign"
    / "gc_external_voqc_pytket_rerun_no_timeout_detached_20260706"
    / "graph_cluster_2d_grid_2x4"
)


def launch(name: str, command: list[str]) -> dict[str, object]:
    run_dir = OUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
    info: dict[str, object] = {
        "tool": name,
        "pid": proc.pid,
        "command": command,
        "cwd": str(ROOT),
        "started_at_epoch": time.time(),
        "timeout": None,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (run_dir / "pid.txt").write_text(f"{proc.pid}\n", encoding="utf-8")
    (run_dir / "launch_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return info


def main() -> int:
    py = sys.executable
    jobs = {
        "voqc": [
            py,
            "Baseline_scripts/run_voqc_baseline.py",
            "--input",
            str(INPUT_DIR / "graph_cluster_2d_grid_2x4_nam_opt0.qasm"),
            "--input-stats",
            str(INPUT_DIR / "nam_opt0_stats.json"),
            "--output",
            str(OUT_DIR / "voqc/voqc_native_output.qasm"),
            "--summary",
            str(OUT_DIR / "voqc/voqc_native_summary.json"),
        ],
        "pytket": [
            py,
            "Baseline_scripts/run_pytket_baseline.py",
            "--input",
            str(INPUT_DIR / "graph_cluster_2d_grid_2x4_ibm_opt0.qasm"),
            "--input-stats",
            str(INPUT_DIR / "ibm_opt0_stats.json"),
            "--output",
            str(OUT_DIR / "pytket/pytket_native.qasm"),
            "--summary",
            str(OUT_DIR / "pytket/pytket_native_summary.json"),
            "--strategy",
            "best-of",
            "--best-of-objective",
            "metric_total",
        ],
    }
    launched = {name: launch(name, command) for name, command in jobs.items()}
    print(json.dumps(launched, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
