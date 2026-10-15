#!/usr/bin/env python3
"""Batch runner for hypercubeW Nam QASM baseline optimizers."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_N_POINTS = [4, 8, 12, 20]
DEFAULT_TOOLS = ["quartz", "wisq", "voqc", "quizx", "feynman", "pytket"]
INPUT_DIR = ROOT / "circuits" / "hypercubeW"
RESULT_DIR = ROOT / "Baseline_results" / "hypercubeW"


@dataclass(frozen=True)
class Task:
    tool: str
    n: int
    input_path: Path
    summary_path: Path
    log_path: Path
    command: list[str]
    outer_timeout: float


def input_path_for_n(n: int, input_dir: Path, gate_set: str = "nam") -> Path:
    return input_dir / f"hypercubeW_channel_lcu_basic_no_n{n}_{gate_set}_qasm2.qasm"


def input_gate_set_for_tool(args: argparse.Namespace, tool: str) -> str:
    if tool == "pytket":
        return str(args.pytket_input_gate_set)
    return "nam"


def generate_inputs(n_points: list[int], input_dir: Path) -> list[dict[str, Any]]:
    from test_hypercube_channel_lcu import export_hypercube_channel_lcu_nam_qasm2_batch

    return export_hypercube_channel_lcu_nam_qasm2_batch(
        n_points=n_points,
        out_dir=input_dir,
    )


def generate_pytket_ibm_inputs(n_points: list[int], input_dir: Path) -> list[dict[str, Any]]:
    from test_hypercube_channel_lcu import export_hypercube_channel_lcu_ibm_qasm2_batch

    return export_hypercube_channel_lcu_ibm_qasm2_batch(
        n_points=n_points,
        out_dir=input_dir,
    )


def python_command_for_tool(args: argparse.Namespace, tool: str) -> list[str]:
    if tool in {"quartz", "wisq"}:
        return ["conda", "run", "-n", args.channel_env, "python"]
    if tool == "pytket":
        return ["conda", "run", "-n", args.qiskit_env, "python"]
    return [sys.executable]


def build_command(args: argparse.Namespace, tool: str, input_path: Path, summary_path: Path) -> list[str]:
    python_prefix = python_command_for_tool(args, tool)
    script = SCRIPT_DIR / f"run_{tool}_baseline.py"
    command = [
        *python_prefix,
        str(script),
        "--input",
        str(input_path),
        "--summary",
        str(summary_path),
    ]
    if tool in {"quartz", "wisq", "voqc", "quizx", "feynman"}:
        timeout_value = str(int(args.timeout)) if tool == "wisq" else str(float(args.timeout))
        command.extend(["--timeout", timeout_value])
    if tool == "quartz":
        command.extend(
            [
                "--max-candidates",
                str(int(args.quartz_max_candidates)),
                "--upper-limit",
                str(float(args.quartz_upper_limit)),
            ]
        )
        if args.quartz_progress_every > 0:
            command.extend(["--progress-every", str(int(args.quartz_progress_every))])
    if tool == "wisq":
        command.extend(["--optimization-objective", args.wisq_objective])
    if tool == "pytket":
        command.extend(["--strategy", args.pytket_strategy])
        command.extend(["--best-of-objective", args.pytket_best_of_objective])
        if args.pytket_strategy == "single":
            command.extend(["--pass-name", args.pytket_pass])
        if args.pytket_strategy == "single" and args.pytket_no_rebase:
            command.append("--no-rebase-to-input-gateset")
    return command


def build_tasks(args: argparse.Namespace) -> list[Task]:
    input_dir = Path(args.input_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    tasks: list[Task] = []
    outer_timeout = float(args.outer_timeout or (float(args.timeout) + 600.0))

    for n in args.n:
        for tool in args.tools:
            if tool not in DEFAULT_TOOLS:
                raise ValueError(f"Unknown tool: {tool}")
            gate_set = input_gate_set_for_tool(args, tool)
            input_path = input_path_for_n(int(n), input_dir, gate_set=gate_set).resolve()
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing {gate_set.upper()} input QASM for tool={tool}, n={n}: {input_path}. "
                    "Run with --generate-inputs first."
                )
            tool_dir = result_dir / tool / f"n{n}"
            summary_path = tool_dir / f"hypercubeW_n{n}_{tool}_summary.json"
            log_path = tool_dir / f"hypercubeW_n{n}_{tool}.log"
            tasks.append(
                Task(
                    tool=tool,
                    n=int(n),
                    input_path=input_path,
                    summary_path=summary_path,
                    log_path=log_path,
                    command=build_command(args, tool, input_path, summary_path),
                    outer_timeout=outer_timeout,
                )
            )
    return tasks


def run_task(task: Task, skip_existing: bool) -> dict[str, Any]:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and task.summary_path.exists():
        return {
            "tool": task.tool,
            "n": task.n,
            "status": "skipped_existing",
            "summary": str(task.summary_path),
            "log": str(task.log_path),
        }

    started = time.time()
    status: dict[str, Any] = {
        "tool": task.tool,
        "n": task.n,
        "status": "running",
        "input": str(task.input_path),
        "summary": str(task.summary_path),
        "log": str(task.log_path),
        "command": task.command,
        "started_at": started,
    }
    status_path = task.log_path.with_name(f"hypercubeW_n{task.n}_{task.tool}_status.json")
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    with task.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(task.command) + "\n\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                task.command,
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=task.outer_timeout,
                check=False,
            )
            returncode = int(completed.returncode)
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = None
            timed_out = True
            log_file.write(
                f"\n[driver] outer timeout expired after {task.outer_timeout:.1f}s\n"
            )

    finished = time.time()
    status.update(
        {
            "status": "timed_out" if timed_out else ("ok" if returncode == 0 else "failed"),
            "returncode": returncode,
            "timed_out": timed_out,
            "finished_at": finished,
            "elapsed_seconds": finished - started,
            "summary_exists": task.summary_path.exists(),
        }
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def run(args: argparse.Namespace) -> dict[str, Any]:
    n_points = [int(n) for n in args.n]
    input_dir = Path(args.input_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    generated = None
    if args.generate_inputs:
        generated = {"nam": generate_inputs(n_points, input_dir)}
        if "pytket" in args.tools and args.pytket_input_gate_set == "ibm":
            generated["ibm"] = generate_pytket_ibm_inputs(n_points, input_dir)

    tasks = build_tasks(args)
    if args.dry_run:
        summary = {
            "benchmark": "hypercubeW",
            "dry_run": True,
            "generated": generated,
            "tasks": [
                {
                    "tool": task.tool,
                    "n": task.n,
                    "input": str(task.input_path),
                    "summary": str(task.summary_path),
                    "log": str(task.log_path),
                    "command": task.command,
                }
                for task in tasks
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    started = time.time()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.parallelism)) as pool:
        future_to_task = {
            pool.submit(run_task, task, bool(args.skip_existing)): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "benchmark": "hypercubeW",
        "n": n_points,
        "tools": list(args.tools),
        "timeout_seconds": float(args.timeout),
        "parallelism": int(args.parallelism),
        "generated": generated,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    batch_summary = result_dir / "hypercubeW_batch_summary.json"
    batch_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate hypercubeW Nam QASM inputs and run the unified baseline "
            "optimizers with logs under Baseline_results/hypercubeW."
        )
    )
    parser.add_argument("--n", nargs="+", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--tools", nargs="+", default=DEFAULT_TOOLS, choices=DEFAULT_TOOLS)
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--result-dir", default=str(RESULT_DIR))
    parser.add_argument("--generate-inputs", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--outer-timeout",
        type=float,
        default=None,
        help="Driver subprocess timeout. Defaults to --timeout + 600 seconds.",
    )
    parser.add_argument("--qiskit-env", default="qiskit_simulate")
    parser.add_argument("--channel-env", default="ChannelIR_test")
    parser.add_argument("--quartz-max-candidates", type=int, default=10000)
    parser.add_argument("--quartz-upper-limit", type=float, default=1.05)
    parser.add_argument("--quartz-progress-every", type=int, default=0)
    parser.add_argument(
        "--wisq-objective",
        default="TOTAL",
        choices=["TWO_Q", "FIDELITY", "FT", "TOTAL", "T"],
    )
    parser.add_argument(
        "--pytket-pass",
        default="full-peephole",
        choices=["full-peephole", "remove-redundancies", "synthesise-tket"],
        help="pytket pass used only when --pytket-strategy single is selected.",
    )
    parser.add_argument(
        "--pytket-strategy",
        default="best-of",
        choices=["single", "best-of"],
        help=(
            "best-of compares FullPeephole without rebasing against "
            "RemoveRedundancies rebased to the input gate set."
        ),
    )
    parser.add_argument(
        "--pytket-best-of-objective",
        default="total_ops",
        choices=["total_ops", "metric_total"],
    )
    parser.add_argument(
        "--pytket-input-gate-set",
        default="ibm",
        choices=["ibm", "nam"],
        help=(
            "HypercubeW exception: pytket defaults to IBM input/output basis; "
            "other tools keep using Nam inputs."
        ),
    )
    parser.add_argument("--pytket-no-rebase", action="store_true")
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
