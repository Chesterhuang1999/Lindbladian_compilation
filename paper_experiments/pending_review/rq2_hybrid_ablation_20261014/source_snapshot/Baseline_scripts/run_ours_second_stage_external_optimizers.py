#!/usr/bin/env python3
"""Run external optimizers on already-optimized in-house QASM circuits.

This experiment treats the current ``ours`` NAM/IBM basis-level circuits as
the input to generic optimizers:

- NAM inputs -> VOQC and Feynman by default
- IBM inputs -> pytket, BQSKit

It does not regenerate raw channel-LCU circuits.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(ROOT / "src"))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from experiment_protocol import EXTERNAL_NAM_OPTIMIZERS


DEFAULT_INPUT_TABLE = ROOT / "Results" / "block_encoding_new_ours_rerun_all_families.csv"
DEFAULT_OUT_ROOT = (
    ROOT
    / "Baseline_results"
    / "eval_redesign"
    / "ours_second_stage_external_optimizers_20260704"
)
DEFAULT_RESULTS_CSV = ROOT / "Results" / "ours_second_stage_external_optimizers.csv"
DEFAULT_RESULTS_MD = ROOT / "Results" / "ours_second_stage_external_optimizers.md"

NAM_TOOLS = ("voqc", "feynman", "wisq", "quartz")
IBM_TOOLS = ("qiskit_opt3", "pytket", "bqskit")


@dataclass(frozen=True)
class CaseInput:
    family: str
    benchmark: str
    basis: str
    input_qasm: Path
    input_gate_count: int
    input_non_clifford: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def qasm_from_summary(source: Path, basis: str) -> Path:
    data = load_json(source)
    basis_l = basis.lower()
    candidates: list[str] = []
    if isinstance(data.get("ours"), dict):
        ours = data["ours"]
        if basis_l == "nam" and ours.get("nam_qasm"):
            candidates.append(str(ours["nam_qasm"]))
        if isinstance(ours.get("basis_outputs"), dict):
            entry = ours["basis_outputs"].get(basis_l)
            if isinstance(entry, dict) and entry.get("qasm"):
                candidates.append(str(entry["qasm"]))
    if isinstance(data.get("basis_results"), dict):
        entry = data["basis_results"].get(basis_l)
        if isinstance(entry, dict):
            for key in ("qasm", "qasm_path", "path"):
                if entry.get(key):
                    candidates.append(str(entry[key]))
    if isinstance(data.get("circuit"), dict):
        circuit = data["circuit"]
        for key in (
            f"{basis_l}_qasm",
            f"{basis_l}_qasm_path",
            f"basis_{basis_l}_qasm",
        ):
            if circuit.get(key):
                candidates.append(str(circuit[key]))

    # Structured subagent summaries consistently keep basis files in qasm/.
    stem = source.stem
    if stem.endswith("_after_ours_summary"):
        prefix = stem[: -len("_after_ours_summary")]
        candidates.append(
            str(
                source.parent
                / "qasm"
                / f"{prefix}_after_ours_rewrite_opt_matrix_order_{basis_l}_opt0.qasm"
            )
        )

    for candidate in candidates:
        path = resolve_path(candidate)
        if path.exists() and path.suffix == ".qasm":
            return path
    raise FileNotFoundError(f"Could not resolve {basis} QASM from summary: {source}")


def selected_benchmarks() -> set[str]:
    base = {
        "periodic_tfim_n4_dt0p1",
        "periodic_tfim_n8_dt0p1",
        "periodic_tfim_n12_dt0p1",
        "hypercube_random_walk_n8",
        "hypercube_random_walk_n28",
        "repetition_dqe_d3_eps0p25",
    }
    base.update(f"random_channel_N5_dim4_N1_10_N2_2_seed{seed}" for seed in range(5))
    return base


def collect_inputs(input_table: Path) -> list[CaseInput]:
    selected = selected_benchmarks()
    rows = []
    for row in read_csv(input_table):
        benchmark = row["benchmark"]
        basis = row["basis"].upper()
        if benchmark not in selected or basis not in {"NAM", "IBM"}:
            continue
        source = resolve_path(row["source_after"])
        if source.suffix == ".json":
            qasm = qasm_from_summary(source, basis)
        else:
            qasm = source
        if not qasm.exists():
            raise FileNotFoundError(qasm)
        rows.append(
            CaseInput(
                family=row["family"],
                benchmark=benchmark,
                basis=basis,
                input_qasm=qasm,
                input_gate_count=int(float(row["after_gate_count"])),
                input_non_clifford=int(float(row["after_non_clifford"])),
            )
        )
    rows.sort(key=lambda item: (item.family, item.benchmark, item.basis))
    return rows


def tool_list_for_basis(basis: str) -> tuple[str, ...]:
    if basis == "NAM":
        return NAM_TOOLS
    if basis == "IBM":
        return IBM_TOOLS
    raise ValueError(basis)


def build_command(
    *,
    tool: str,
    case: CaseInput,
    out_dir: Path,
    timeout: float,
    python_exe: str,
    quartz_max_candidates: int,
    quartz_checkpoint_interval: float,
    bqskit_optimization_level: int,
    bqskit_workflow_mode: str,
) -> tuple[list[str], Path, Path]:
    tool_dir = out_dir / case.benchmark / case.basis.lower() / tool
    tool_dir.mkdir(parents=True, exist_ok=True)
    output = tool_dir / f"{case.benchmark}_ours_{case.basis.lower()}_{tool}.qasm"
    summary = tool_dir / f"{tool}_summary.json"
    if tool in ("voqc", "feynman"):
        command = [
            python_exe,
            str(BASELINE_DIR / f"run_{tool}_baseline.py"),
            "--input",
            str(case.input_qasm),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--timeout",
            str(float(timeout)),
        ]
    elif tool == "wisq":
        command = [
            python_exe,
            str(BASELINE_DIR / "run_wisq_baseline.py"),
            "--input",
            str(case.input_qasm),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--timeout",
            str(int(timeout)),
            "--optimization-objective",
            "TOTAL",
        ]
    elif tool == "quartz":
        checkpoint = tool_dir / f"{case.benchmark}_ours_{case.basis.lower()}_quartz_checkpoint.qasm"
        command = [
            python_exe,
            str(BASELINE_DIR / "run_quartz_baseline.py"),
            "--input",
            str(case.input_qasm),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--timeout",
            str(float(timeout)),
            "--max-candidates",
            str(int(quartz_max_candidates)),
            "--checkpoint-interval",
            str(float(quartz_checkpoint_interval)),
            "--checkpoint-output",
            str(checkpoint),
            "--no-increase",
            "--no-nop",
            "--use-available-xfers",
        ]
    elif tool == "pytket":
        command = [
            python_exe,
            str(BASELINE_DIR / "run_pytket_baseline.py"),
            "--input",
            str(case.input_qasm),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--strategy",
            "best-of",
            "--best-of-objective",
            "metric_total",
        ]
    elif tool == "qiskit_opt3":
        command = [
            python_exe,
            str(BASELINE_DIR / "run_qiskit_opt3_baseline.py"),
            "--input",
            str(case.input_qasm),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--basis",
            case.basis.lower(),
        ]
    elif tool == "bqskit":
        command = [
            python_exe,
            str(BASELINE_DIR / "run_bqskit_baseline.py"),
            "--input",
            str(case.input_qasm),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--timeout",
            str(float(timeout)),
            "--optimization-level",
            str(int(bqskit_optimization_level)),
            "--workflow-mode",
            str(bqskit_workflow_mode),
            "--checkpoint",
        ]
    else:
        raise ValueError(tool)
    return command, summary, output


def status_from_summary(tool: str, summary: dict[str, Any]) -> tuple[str, float | None, bool]:
    if tool in {"voqc", "feynman", "wisq"}:
        proc = summary.get("process", {})
        if proc.get("timed_out"):
            return "timeout", as_float(proc.get("elapsed_seconds")), True
        if proc.get("returncode") == 0 and summary.get("output_stats") is not None:
            return "ok", as_float(proc.get("elapsed_seconds")), False
        return "failed", as_float(proc.get("elapsed_seconds")), False
    if tool == "quartz":
        search = summary.get("quartz_stats", {}).get("search", {})
        timed_out = bool(search.get("timed_out"))
        if summary.get("output_stats") is not None:
            return ("timeout_checkpoint" if timed_out else "ok"), as_float(search.get("elapsed_seconds")), timed_out
        return ("timeout" if timed_out else "failed"), as_float(search.get("elapsed_seconds")), timed_out
    if tool == "pytket":
        stats = summary.get("pytket_stats", {})
        return ("ok" if summary.get("output_stats") is not None else "failed"), as_float(stats.get("elapsed_seconds")), False
    if tool == "qiskit_opt3":
        proc = summary.get("process", {})
        if proc.get("timed_out"):
            return "timeout", as_float(proc.get("elapsed_seconds")), True
        if proc.get("returncode") == 0 and summary.get("output_stats") is not None:
            return "ok", as_float(proc.get("elapsed_seconds")), False
        return "failed", as_float(proc.get("elapsed_seconds")), False
    if tool == "bqskit":
        proc = summary.get("process", {})
        timed_out = bool(proc.get("timed_out"))
        fallback = bool(summary.get("fallback_output_used"))
        checkpoint = bool(summary.get("checkpoint_output_used"))
        if summary.get("output_stats") is not None and not fallback:
            if timed_out and checkpoint:
                return "timeout_checkpoint", as_float(proc.get("elapsed_seconds")), True
            if bool(proc.get("ok")):
                return "ok", as_float(proc.get("elapsed_seconds")), False
        if fallback:
            return "timeout_fallback_input" if timed_out else "failed_fallback_input", as_float(proc.get("elapsed_seconds")), timed_out
        return ("timeout" if timed_out else "failed"), as_float(proc.get("elapsed_seconds")), timed_out
    return "unknown", None, False


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_total(stats: dict[str, Any] | None) -> int | None:
    if not stats:
        return None
    return int(stats["metric_total"])


def non_clifford(stats: dict[str, Any] | None) -> int | None:
    if not stats:
        return None
    return int(stats["clifford"]["non_clifford"])


def row_from_tool_result(
    *,
    case: CaseInput,
    tool: str,
    summary_path: Path,
    command: list[str],
    command_elapsed: float,
    returncode: int | None,
    stdout_tail: str,
    stderr_tail: str,
) -> dict[str, Any]:
    summary: dict[str, Any] | None = None
    if summary_path.exists():
        try:
            summary = load_json(summary_path)
        except json.JSONDecodeError:
            summary = None

    if summary is None:
        status = "failed"
        elapsed = command_elapsed
        timed_out = False
        output_stats = None
        output_qasm = None
    else:
        status, elapsed, timed_out = status_from_summary(tool, summary)
        output_stats = summary.get("output_stats")
        output_qasm = summary.get("output_qasm")

    out_gate_count = metric_total(output_stats)
    out_non_clifford = non_clifford(output_stats)
    return {
        "family": case.family,
        "benchmark": case.benchmark,
        "basis": case.basis,
        "input_qasm": str(case.input_qasm),
        "input_gate_count": case.input_gate_count,
        "input_non_clifford": case.input_non_clifford,
        "tool": tool,
        "output_gate_count": out_gate_count,
        "output_non_clifford": out_non_clifford,
        "ratio_vs_ours_input": (
            None if out_gate_count is None else float(out_gate_count / case.input_gate_count)
        ),
        "gate_delta": None if out_gate_count is None else int(out_gate_count - case.input_gate_count),
        "non_clifford_delta": (
            None if out_non_clifford is None else int(out_non_clifford - case.input_non_clifford)
        ),
        "optimizer_time_s": elapsed,
        "command_elapsed_s": command_elapsed,
        "timeout": timed_out,
        "status": status,
        "returncode": returncode,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "output_qasm": output_qasm,
    }


def run_one(
    *,
    case: CaseInput,
    tool: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command, summary_path, _output_path = build_command(
        tool=tool,
        case=case,
        out_dir=Path(args.out_root),
        timeout=float(args.timeout),
        python_exe=str(args.runner_python),
        quartz_max_candidates=int(args.quartz_max_candidates),
        quartz_checkpoint_interval=float(args.quartz_checkpoint_interval),
        bqskit_optimization_level=int(args.bqskit_optimization_level),
        bqskit_workflow_mode=str(args.bqskit_workflow_mode),
    )
    if summary_path.exists() and not args.force:
        return row_from_tool_result(
            case=case,
            tool=tool,
            summary_path=summary_path,
            command=command,
            command_elapsed=0.0,
            returncode=0,
            stdout_tail="skipped existing summary",
            stderr_tail="",
        )

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(args.outer_timeout) if float(args.outer_timeout) > 0 else None,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

    elapsed = time.perf_counter() - started
    log_dir = summary_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_text(stdout, encoding="utf-8", errors="replace")
    (log_dir / "stderr.log").write_text(stderr, encoding="utf-8", errors="replace")
    return row_from_tool_result(
        case=case,
        tool=tool,
        summary_path=summary_path,
        command=command,
        command_elapsed=elapsed,
        returncode=returncode,
        stdout_tail=stdout,
        stderr_tail=stderr,
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "benchmark",
        "basis",
        "input_gate_count",
        "input_non_clifford",
        "tool",
        "output_gate_count",
        "output_non_clifford",
        "ratio_vs_ours_input",
        "gate_delta",
        "non_clifford_delta",
        "optimizer_time_s",
        "command_elapsed_s",
        "timeout",
        "status",
        "returncode",
        "input_qasm",
        "output_qasm",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def fmt_int(value: Any) -> str:
    if value in (None, ""):
        return "-"
    return f"{int(float(value)):,}"


def fmt_float(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_ratio(value: Any) -> str:
    if value in (None, ""):
        return "-"
    return f"{100 * float(value):.2f}%"


def write_markdown(rows: list[dict[str, Any]], path: Path, csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ours second-stage external optimizer results",
        "",
        f"CSV: `{csv_path}`",
        "",
        "Input circuits are the current in-house `ours` basis-level circuits. NAM inputs use VOQC/WISQ/Quartz; IBM inputs use pytket/BQSKit.",
        "",
        "| Family | Benchmark | Basis | Ours input gates | Tool | Output gates | Ratio | Non-Clifford out | Time (s) | Status |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['benchmark']} | {row['basis']} | "
            f"{fmt_int(row['input_gate_count'])} | {row['tool']} | "
            f"{fmt_int(row.get('output_gate_count'))} | {fmt_ratio(row.get('ratio_vs_ours_input'))} | "
            f"{fmt_int(row.get('output_non_clifford'))} | {fmt_float(row.get('optimizer_time_s'))} | "
            f"{row.get('status') or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, default=DEFAULT_INPUT_TABLE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS_MD)
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--outer-timeout", type=float, default=3900.0)
    parser.add_argument("--quartz-max-candidates", type=int, default=64)
    parser.add_argument("--quartz-checkpoint-interval", type=float, default=300.0)
    parser.add_argument("--bqskit-optimization-level", type=int, default=2)
    parser.add_argument(
        "--bqskit-workflow-mode",
        choices=("standard", "skip-retarget-mapping"),
        default="skip-retarget-mapping",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Benchmark name to include. May be repeated.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Family name to include. May be repeated.",
    )
    parser.add_argument(
        "--basis",
        action="append",
        choices=("NAM", "IBM", "nam", "ibm"),
        default=[],
        help="Basis to include. May be repeated.",
    )
    parser.add_argument(
        "--tool",
        action="append",
        choices=NAM_TOOLS + IBM_TOOLS,
        default=[],
        help="Tool to include. May be repeated.",
    )
    args = parser.parse_args()
    if not args.basis:
        args.basis = ["NAM"]
    if not args.tool:
        args.tool = list(EXTERNAL_NAM_OPTIMIZERS)
    return args


def main() -> int:
    args = parse_args()
    cases = collect_inputs(Path(args.input_table))
    if args.benchmark:
        wanted = set(args.benchmark)
        cases = [case for case in cases if case.benchmark in wanted]
    if args.family:
        wanted = set(args.family)
        cases = [case for case in cases if case.family in wanted]
    if args.basis:
        wanted = {basis.upper() for basis in args.basis}
        cases = [case for case in cases if case.basis in wanted]
    requested_tools = set(args.tool)
    if args.list_only:
        for case in cases:
            tools_for_case = tuple(
                tool for tool in tool_list_for_basis(case.basis)
                if not requested_tools or tool in requested_tools
            )
            if not tools_for_case:
                continue
            tools = ",".join(tools_for_case)
            print(
                f"{case.family}\t{case.benchmark}\t{case.basis}\t"
                f"{case.input_gate_count}\t{tools}\t{case.input_qasm}"
            )
        task_count = sum(
            len(
                [
                    tool for tool in tool_list_for_basis(case.basis)
                    if not requested_tools or tool in requested_tools
                ]
            )
            for case in cases
        )
        print(f"cases={len(cases)} tasks={task_count}")
        return 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        for tool in tool_list_for_basis(case.basis):
            if requested_tools and tool not in requested_tools:
                continue
            print(f"[run] {case.benchmark} {case.basis} {tool}", flush=True)
            row = run_one(case=case, tool=tool, args=args)
            rows.append(row)
            print(
                f"[done] {case.benchmark} {case.basis} {tool}: "
                f"{row['status']} {row.get('output_gate_count')}",
                flush=True,
            )
            write_csv(rows, Path(args.results_csv))
            write_markdown(rows, Path(args.results_md), Path(args.results_csv))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rows.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    write_csv(rows, Path(args.results_csv))
    write_markdown(rows, Path(args.results_md), Path(args.results_csv))
    print(f"wrote {len(rows)} rows to {args.results_csv}")
    print(f"wrote markdown to {args.results_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
