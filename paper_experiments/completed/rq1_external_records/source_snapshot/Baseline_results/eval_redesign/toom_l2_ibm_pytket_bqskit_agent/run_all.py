#!/usr/bin/env python3
"""Run Toom L2 IBM-native pytket and BQSKit baselines.

All generated artifacts are kept inside this directory.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
IBM_INPUT_DIR = OUT_DIR / "ibm_input"
PYTKET_DIR = OUT_DIR / "pytket"
BQSKIT_DIR = OUT_DIR / "bqskit"
NAM_INPUT = (
    ROOT
    / "circuits"
    / "Table1"
    / "toom_2d_periodic_L2_dt0p1"
    / "toom_2d_periodic_L2_dt0p1_basic_raw_nam_opt0.qasm"
)
IBM_INPUT = IBM_INPUT_DIR / "toom_2d_periodic_L2_dt0p1_basic_raw_ibm_opt0.qasm"
IBM_STATS = IBM_INPUT_DIR / "toom_2d_periodic_L2_dt0p1_basic_raw_ibm_opt0_stats.json"
GENERATION_SUMMARY = IBM_INPUT_DIR / "generation_summary.json"
PYTKET_SUMMARY = PYTKET_DIR / "summary.json"
PYTKET_OUTPUT = PYTKET_DIR / "toom_2d_periodic_L2_dt0p1_pytket_best_of_metric_total.qasm"
BQSKIT_SUMMARY = BQSKIT_DIR / "summary.json"
BQSKIT_OUTPUT = BQSKIT_DIR / "toom_2d_periodic_L2_dt0p1_bqskit_level2_skip_retarget_mapping.qasm"
SUMMARY_CSV = OUT_DIR / "summary.csv"
NOTES_MD = OUT_DIR / "notes.md"
REPRODUCE_SH = OUT_DIR / "reproduce.sh"

IBM_BASIS = ("u1", "u2", "u3", "cx")
BENCHMARK_NAME = "toom_2d_periodic_L2_dt0p1"
CONDA = Path("/Users/chester_hqf/miniconda3/bin/conda")


def add_project_imports() -> None:
    for rel in ("Baseline_scripts", "src"):
        path = str(ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def ensure_dirs() -> None:
    IBM_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PYTKET_DIR.mkdir(parents=True, exist_ok=True)
    BQSKIT_DIR.mkdir(parents=True, exist_ok=True)


def tail_text(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_generation_summary(generation: dict[str, Any]) -> dict[str, Any]:
    if "nam_stats" not in generation and "input_nam_stats" in generation:
        generation = dict(generation)
        generation["nam_stats"] = generation["input_nam_stats"]
    if "qiskit_load_time_s" not in generation:
        generation = dict(generation)
        generation["qiskit_load_time_s"] = generation.get("load_time_sec")
    if "qiskit_transpile_time_s" not in generation:
        generation = dict(generation)
        generation["qiskit_transpile_time_s"] = generation.get("transpile_time_sec")
    generation.setdefault("reused_existing_ibm_input", True)
    return generation


def run_command(
    name: str,
    command: list[str],
    cwd: Path,
    timeout: float,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "timeout_seconds": float(timeout),
        "timed_out": False,
        "returncode": None,
        "elapsed_seconds": None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    stdout_file = stdout_path.open("w", encoding="utf-8")
    stderr_file = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
        start_new_session=True,
    )
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        try:
            os.killpg(process.pid, 15)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            process.wait()
    finally:
        stdout_file.close()
        stderr_file.close()

    result["returncode"] = process.returncode
    result["elapsed_seconds"] = time.perf_counter() - started
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    result["stdout_tail"] = tail_text(stdout_text)
    result["stderr_tail"] = tail_text(stderr_text)
    return result


def ensure_ibm_input() -> dict[str, Any]:
    add_project_imports()
    from baseline_common import count_qasm_file  # type: ignore

    if not NAM_INPUT.exists():
        raise FileNotFoundError(f"Missing NAM input: {NAM_INPUT}")

    nam_stats = count_qasm_file(NAM_INPUT, gate_set="nam")
    reused_existing_ibm_input = IBM_INPUT.exists()
    load_time: float | None = None
    transpile_time: float | None = None
    export: dict[str, Any] | None = None

    if reused_existing_ibm_input:
        ibm_stats = count_qasm_file(IBM_INPUT, gate_set="ibm")
    else:
        from qasm_export import export_openqasm2_quartz_input  # type: ignore
        from qiskit import QuantumCircuit, transpile

        started = time.perf_counter()
        raw_qc = QuantumCircuit.from_qasm_file(str(NAM_INPUT))
        load_time = time.perf_counter() - started

        started = time.perf_counter()
        ibm_qc = transpile(raw_qc, basis_gates=list(IBM_BASIS), optimization_level=0)
        transpile_time = time.perf_counter() - started
        export = export_openqasm2_quartz_input(ibm_qc, IBM_INPUT, basis_gates=IBM_BASIS)
        ibm_stats = count_qasm_file(IBM_INPUT, gate_set="ibm")

    if ibm_stats["detected_gate_set"] != "ibm" or ibm_stats["metric_gate_set"] != "ibm":
        raise ValueError(
            "IBM input did not count as IBM basis: "
            f"detected={ibm_stats['detected_gate_set']}, "
            f"metric={ibm_stats['metric_gate_set']}"
        )

    generation = {
        "benchmark_name": BENCHMARK_NAME,
        "source_nam_qasm": str(NAM_INPUT),
        "source_nam_expected_raw_gates": 61726,
        "ibm_qasm": str(IBM_INPUT),
        "reused_existing_ibm_input": reused_existing_ibm_input,
        "basis_gates": list(IBM_BASIS),
        "qiskit_transpile_optimization_level": 0,
        "qiskit_load_time_s": None if load_time is None else float(load_time),
        "qiskit_transpile_time_s": None if transpile_time is None else float(transpile_time),
        "ibm_export": export,
        "nam_stats": nam_stats,
        "ibm_stats": ibm_stats,
    }
    GENERATION_SUMMARY.write_text(
        json.dumps(generation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    IBM_STATS.write_text(
        json.dumps({"stats": ibm_stats}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return generation


def summarize_tool(
    *,
    method: str,
    optimizer: str,
    raw_gates: int,
    source_path: Path,
    command_record: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    summary = load_json_if_exists(summary_path)
    status = "failed"
    error_msg = ""
    opt_gates: int | None = None
    optimizer_time: float | None = None
    total_time: float | None = float(command_record.get("elapsed_seconds") or 0.0)
    timed_out = bool(command_record.get("timed_out"))

    if timed_out:
        status = "timeout"
        error_msg = "runner timed out"
    elif command_record.get("returncode") != 0:
        status = "failed"
        error_msg = str(command_record.get("stderr_tail") or command_record.get("stdout_tail") or "")
    elif summary is None:
        status = "failed"
        error_msg = f"missing summary JSON: {summary_path}"
    else:
        if method == "pytket":
            output_stats = summary.get("output_stats") or {}
            metric_total = output_stats.get("metric_total")
            opt_gates = None if metric_total in (None, "") else int(metric_total)
            optimizer_time = float(
                (summary.get("pytket_stats") or {}).get("elapsed_seconds", total_time)
            )
            candidate_times = [
                float((candidate.get("pytket_stats") or {}).get("elapsed_seconds", 0.0) or 0.0)
                for candidate in summary.get("candidates", [])
            ]
            if candidate_times:
                total_time = sum(candidate_times) + float(optimizer_time or 0.0)
            elif not total_time:
                total_time = optimizer_time
            status = "success" if opt_gates is not None else "failed"
            if opt_gates is None:
                error_msg = "missing output_stats.metric_total in pytket summary"
        elif method == "bqskit":
            process = summary.get("process") or {}
            output_stats = summary.get("output_stats") or {}
            metric_total = output_stats.get("metric_total")
            opt_gates = None if metric_total in (None, "") else int(metric_total)
            optimizer_time = float(process.get("elapsed_seconds", total_time))
            if bool(process.get("timed_out")):
                status = "timeout"
                timed_out = True
            elif bool(process.get("ok")):
                status = "success" if opt_gates is not None else "failed"
                if opt_gates is None:
                    error_msg = "missing output_stats.metric_total in BQSKit summary"
            else:
                status = "failed"
                error_msg = str(process.get("error") or process.get("traceback") or "")
        else:
            raise ValueError(f"Unknown method: {method}")

    ratio = None if opt_gates is None else float(opt_gates) / float(raw_gates)
    return {
        "benchmark_name": BENCHMARK_NAME,
        "basis_group": "ibm",
        "method": method,
        "optimizer": optimizer,
        "raw_gates": raw_gates,
        "opt_gates": "" if opt_gates is None else opt_gates,
        "ratio_vs_raw": "" if ratio is None else f"{ratio:.12g}",
        "optimizer_time_sec": "" if optimizer_time is None else f"{optimizer_time:.6f}",
        "total_time_sec": "" if total_time is None else f"{total_time:.6f}",
        "timeout": str(bool(timed_out)).lower(),
        "status": status,
        "error_msg": tail_text(error_msg.replace("\n", " "), max_chars=500),
        "source_path": str(source_path),
    }


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark_name",
        "basis_group",
        "method",
        "optimizer",
        "raw_gates",
        "opt_gates",
        "ratio_vs_raw",
        "optimizer_time_sec",
        "total_time_sec",
        "timeout",
        "status",
        "error_msg",
        "source_path",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_notes(
    generation: dict[str, Any],
    pytket_cmd: dict[str, Any],
    bqskit_cmd: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    ibm_stats = generation["ibm_stats"]
    nam_stats = generation["nam_stats"]
    lines = [
        "# Toom L2 IBM pytket/BQSKit Baseline Notes",
        "",
        "## IBM input generation",
        "",
        f"- Source NAM QASM: `{NAM_INPUT}`",
        f"- IBM QASM: `{IBM_INPUT}`",
        "- Reused the existing IBM QASM when present; otherwise loaded the basis-level NAM "
        "OpenQASM2 with Qiskit and transpiled with "
        "`basis_gates=['u1','u2','u3','cx']`, `optimization_level=0`.",
        f"- Reused existing IBM input: {generation['reused_existing_ibm_input']}",
        f"- NAM raw metric gates: {nam_stats['metric_total']}",
        f"- IBM raw metric gates: {ibm_stats['metric_total']}",
        f"- Detected gate set: {ibm_stats['detected_gate_set']}",
        f"- Metric gate set: {ibm_stats['metric_gate_set']}",
        f"- IBM op breakdown: {json.dumps(ibm_stats['ops'], sort_keys=True)}",
        f"- Qiskit load time: {generation['qiskit_load_time_s']}",
        f"- Qiskit IBM transpile time: {generation['qiskit_transpile_time_s']}",
        "",
        "## Tool results",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['method']}: status={row['status']}, "
            f"gates={row['opt_gates']}, ratio={row['ratio_vs_raw']}, "
            f"optimizer_time={row['optimizer_time_sec']}s, "
            f"total_time={row['total_time_sec']}s, timeout={row['timeout']}"
        )
        if row["error_msg"]:
            lines.append(f"  error tail: `{row['error_msg']}`")
    lines.extend(
        [
            "",
            "## Logs",
            "",
            f"- pytket stdout: `{pytket_cmd['stdout_path']}`",
            f"- pytket stderr: `{pytket_cmd['stderr_path']}`",
            f"- BQSKit stdout: `{bqskit_cmd['stdout_path']}`",
            f"- BQSKit stderr: `{bqskit_cmd['stderr_path']}`",
        ]
    )
    NOTES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproduce() -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

cd {ROOT}
/Users/chester_hqf/miniconda3/bin/conda run -n qiskit_simulate \\
  python {OUT_DIR / 'run_all.py'} --only-generate

/Users/chester_hqf/miniconda3/bin/conda run -n ChannelIR_test \\
  python Baseline_scripts/run_pytket_baseline.py \\
  --input {IBM_INPUT} \\
  --output {PYTKET_OUTPUT} \\
  --summary {PYTKET_SUMMARY} \\
  --strategy best-of \\
  --best-of-objective metric_total

/Users/chester_hqf/miniconda3/bin/conda run -n ChannelIR_test \\
  python Baseline_scripts/run_bqskit_baseline.py \\
  --input {IBM_INPUT} \\
  --output {BQSKIT_OUTPUT} \\
  --summary {BQSKIT_SUMMARY} \\
  --gate-set ibm \\
  --optimization-level 2 \\
  --workflow-mode skip-retarget-mapping \\
  --timeout 3600

/Users/chester_hqf/miniconda3/bin/conda run -n qiskit_simulate \\
  python {OUT_DIR / 'run_all.py'} --only-summarize
"""
    REPRODUCE_SH.write_text(text, encoding="utf-8")
    REPRODUCE_SH.chmod(0o755)


def run_all() -> None:
    ensure_dirs()
    generation = ensure_ibm_input()
    write_reproduce()
    raw_gates = int(generation["ibm_stats"]["metric_total"])

    pytket_cmd = run_command(
        "pytket",
        [
            str(CONDA),
            "run",
            "-n",
            "ChannelIR_test",
            "python",
            "Baseline_scripts/run_pytket_baseline.py",
            "--input",
            str(IBM_INPUT),
            "--output",
            str(PYTKET_OUTPUT),
            "--summary",
            str(PYTKET_SUMMARY),
            "--strategy",
            "best-of",
            "--best-of-objective",
            "metric_total",
        ],
        cwd=ROOT,
        timeout=3600.0,
        stdout_path=PYTKET_DIR / "stdout.log",
        stderr_path=PYTKET_DIR / "stderr.log",
    )
    (PYTKET_DIR / "command_result.json").write_text(
        json.dumps(pytket_cmd, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bqskit_cmd = run_command(
        "bqskit",
        [
            str(CONDA),
            "run",
            "-n",
            "ChannelIR_test",
            "python",
            "Baseline_scripts/run_bqskit_baseline.py",
            "--input",
            str(IBM_INPUT),
            "--output",
            str(BQSKIT_OUTPUT),
            "--summary",
            str(BQSKIT_SUMMARY),
            "--gate-set",
            "ibm",
            "--optimization-level",
            "2",
            "--workflow-mode",
            "skip-retarget-mapping",
            "--timeout",
            "3600",
        ],
        cwd=ROOT,
        timeout=3660.0,
        stdout_path=BQSKIT_DIR / "stdout.log",
        stderr_path=BQSKIT_DIR / "stderr.log",
    )
    (BQSKIT_DIR / "command_result.json").write_text(
        json.dumps(bqskit_cmd, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = [
        summarize_tool(
            method="pytket",
            optimizer="best-of_metric-total",
            raw_gates=raw_gates,
            source_path=IBM_INPUT,
            command_record=pytket_cmd,
            summary_path=PYTKET_SUMMARY,
        ),
        summarize_tool(
            method="bqskit",
            optimizer="optimization-level-2_skip-retarget-mapping",
            raw_gates=raw_gates,
            source_path=IBM_INPUT,
            command_record=bqskit_cmd,
            summary_path=BQSKIT_SUMMARY,
        ),
    ]
    write_summary_csv(rows)
    write_notes(generation, pytket_cmd, bqskit_cmd, rows)
    print(json.dumps({"generation": generation, "rows": rows}, indent=2, sort_keys=True))


def run_bqskit_only() -> None:
    ensure_dirs()
    generation = normalize_generation_summary(ensure_ibm_input())
    write_reproduce()
    raw_gates = int(generation["ibm_stats"]["metric_total"])
    bqskit_cmd = run_command(
        "bqskit",
        [
            str(CONDA),
            "run",
            "-n",
            "ChannelIR_test",
            "python",
            "Baseline_scripts/run_bqskit_baseline.py",
            "--input",
            str(IBM_INPUT),
            "--output",
            str(BQSKIT_OUTPUT),
            "--summary",
            str(BQSKIT_SUMMARY),
            "--gate-set",
            "ibm",
            "--optimization-level",
            "2",
            "--workflow-mode",
            "skip-retarget-mapping",
            "--timeout",
            "3600",
        ],
        cwd=ROOT,
        timeout=3660.0,
        stdout_path=BQSKIT_DIR / "stdout.log",
        stderr_path=BQSKIT_DIR / "stderr.log",
    )
    (BQSKIT_DIR / "command_result.json").write_text(
        json.dumps(bqskit_cmd, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summarize_existing()
    print(json.dumps({"generation": generation, "bqskit_command": bqskit_cmd}, indent=2, sort_keys=True))


def summarize_existing() -> None:
    add_project_imports()
    generation = load_json_if_exists(GENERATION_SUMMARY)
    if generation is None:
        generation = ensure_ibm_input()
    generation = normalize_generation_summary(generation)
    raw_gates = int(generation["ibm_stats"]["metric_total"])
    pytket_cmd = load_json_if_exists(PYTKET_DIR / "command_result.json") or {
        "elapsed_seconds": None,
        "timed_out": False,
        "returncode": 0 if PYTKET_SUMMARY.exists() else 1,
        "stdout_tail": "",
        "stderr_tail": "",
        "stdout_path": str(PYTKET_DIR / "stdout.log"),
        "stderr_path": str(PYTKET_DIR / "stderr.log"),
    }
    bqskit_cmd = load_json_if_exists(BQSKIT_DIR / "command_result.json") or {
        "elapsed_seconds": None,
        "timed_out": False,
        "returncode": 0 if BQSKIT_SUMMARY.exists() else 1,
        "stdout_tail": "",
        "stderr_tail": "",
        "stdout_path": str(BQSKIT_DIR / "stdout.log"),
        "stderr_path": str(BQSKIT_DIR / "stderr.log"),
    }
    rows = [
        summarize_tool(
            method="pytket",
            optimizer="best-of_metric-total",
            raw_gates=raw_gates,
            source_path=IBM_INPUT,
            command_record=pytket_cmd,
            summary_path=PYTKET_SUMMARY,
        ),
        summarize_tool(
            method="bqskit",
            optimizer="optimization-level-2_skip-retarget-mapping",
            raw_gates=raw_gates,
            source_path=IBM_INPUT,
            command_record=bqskit_cmd,
            summary_path=BQSKIT_SUMMARY,
        ),
    ]
    write_summary_csv(rows)
    write_notes(generation, pytket_cmd, bqskit_cmd, rows)
    write_reproduce()
    print(json.dumps({"generation": generation, "rows": rows}, indent=2, sort_keys=True))


def main() -> int:
    args = set(sys.argv[1:])
    ensure_dirs()
    if "--only-generate" in args:
        generation = ensure_ibm_input()
        write_reproduce()
        print(json.dumps(generation, indent=2, sort_keys=True))
        return 0
    if "--only-summarize" in args:
        summarize_existing()
        return 0
    if "--only-bqskit" in args:
        run_bqskit_only()
        return 0
    run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
