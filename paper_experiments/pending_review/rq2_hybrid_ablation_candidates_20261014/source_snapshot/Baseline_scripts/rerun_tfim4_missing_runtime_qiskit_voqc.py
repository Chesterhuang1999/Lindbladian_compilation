#!/usr/bin/env python3
"""Rerun TFIM_4 Qiskit opt3 and VOQC to recover missing runtimes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import compare_gate_stats, count_qasm_file, run_command, tail_text  # noqa: E402
from fix_voqc_large_rz_angles import fix_text  # noqa: E402
from lower_voqc_qasm_to_nam import lower_voqc_qasm_file_to_nam  # noqa: E402
from normalize_qasm_rz_for_voqc import normalize_qasm_file  # noqa: E402


BENCHMARK = "periodic_tfim_n4_dt0p1"
FAMILY = "Lindblad"
N_OR_SIZE = "4"
IBM_INPUT = ROOT / "circuits" / "Table1" / BENCHMARK / f"{BENCHMARK}_basic_raw_ibm_opt0.qasm"
NAM_INPUT = ROOT / "circuits" / "Table1" / BENCHMARK / f"{BENCHMARK}_basic_raw_nam_opt0.qasm"
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "eval_redesign" / "tfim4_qiskit_voqc_runtime_rerun_20260707"
DEFAULT_RUNTIME_CSV = (
    ROOT
    / "Baseline_results"
    / "eval_redesign"
    / "cp5_aggregation"
    / "cp1_cp1b_tool_runtime_summary.csv"
)
DEFAULT_VOQC = ROOT / "external_tools" / "VOQC" / "VOQC" / "_build" / "default" / "voqc.exe"
RUNTIME_FIELDS = [
    "benchmark_name",
    "basis_group",
    "tool/method",
    "method",
    "optimizer",
    "optimizer_variant",
    "status",
    "timeout",
    "runtime_sec",
    "runtime_source",
    "source_path",
    "note",
    "benchmark_family",
    "n_or_size",
    "collection_source_file",
    "source_type",
    "construction_time_sec",
    "optimizer_time_sec",
    "transpile_time_sec",
    "total_time_sec",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def conda_python(env: str) -> str:
    candidate = ROOT.parent.parent.parent.parent / "miniconda3" / "envs" / env / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    fallback = Path.home() / "miniconda3" / "envs" / env / "bin" / "python"
    return str(fallback if fallback.exists() else Path(sys.executable))


def run_qiskit_opt3(out_dir: Path, timeout: float | None) -> dict[str, Any]:
    tool_dir = out_dir / "qiskit_opt3"
    summary_path = tool_dir / "qiskit_opt3_summary.json"
    output_path = tool_dir / f"{BENCHMARK}_raw_ibm_qiskit_opt3.qasm"
    command = [
        conda_python("qiskit_simulate"),
        str(BASELINE_DIR / "run_qiskit_opt3_baseline.py"),
        "--input",
        str(IBM_INPUT),
        "--output",
        str(output_path),
        "--summary",
        str(summary_path),
        "--basis",
        "ibm",
    ]
    proc = run_command(command, timeout=timeout)
    summary = read_json(summary_path) if summary_path.exists() else {}
    output_stats = summary.get("output_stats")
    input_stats = count_qasm_file(IBM_INPUT, gate_set="ibm")
    comparison = (
        compare_gate_stats(input_stats, output_stats, "ibm")
        if isinstance(output_stats, dict)
        else None
    )
    result = {
        "benchmark": BENCHMARK,
        "basis": "IBM",
        "tool": "qiskit_opt3",
        "input_qasm": rel(IBM_INPUT),
        "output_qasm": rel(output_path) if output_path.exists() else "",
        "summary_path": rel(summary_path),
        "status": "ok" if proc["returncode"] == 0 and isinstance(output_stats, dict) else "failed",
        "timed_out": bool(proc["timed_out"]),
        "process": {
            "command": command,
            "returncode": proc["returncode"],
            "elapsed_seconds": proc["elapsed_seconds"],
            "timed_out": proc["timed_out"],
            "stdout_tail": tail_text(proc["stdout"]),
            "stderr_tail": tail_text(proc["stderr"]),
        },
        "input_stats": input_stats,
        "output_stats": output_stats,
        "comparison": comparison,
        "runner_summary": summary,
    }
    write_json(tool_dir / "tfim4_qiskit_opt3_runtime_result.json", result)
    return result


def run_voqc(out_dir: Path, timeout: float | None) -> dict[str, Any]:
    tool_dir = out_dir / "voqc"
    tool_dir.mkdir(parents=True, exist_ok=True)
    normalized_input = tool_dir / f"{BENCHMARK}_raw_nam_voqc_normalized.qasm"
    output_path = tool_dir / "voqc_native_output.qasm"
    lowered_path = tool_dir / "voqc_native_output_nam_lowered.qasm"
    fixed_path = tool_dir / "voqc_native_output_nam_lowered_fixed.qasm"
    fix_summary_path = tool_dir / "voqc_large_rz_fix_summary.json"
    input_stats = count_qasm_file(NAM_INPUT, gate_set="nam")
    replacements = normalize_qasm_file(NAM_INPUT, normalized_input)
    command = [str(DEFAULT_VOQC), "-i", str(normalized_input), "-o", str(output_path)]
    proc = run_command(command, timeout=timeout)
    output_stats = None
    comparison = None
    lowering_info = None
    fix_summary: dict[str, Any] | None = None
    status = "timeout" if proc["timed_out"] else "failed"
    if output_path.exists():
        lowering_info = lower_voqc_qasm_file_to_nam(output_path, lowered_path)
        fixed_text, fix_stats = fix_text(lowered_path.read_text(encoding="utf-8", errors="ignore"))
        fixed_path.write_text(fixed_text, encoding="utf-8")
        fix_summary = {"input": rel(lowered_path), "output": rel(fixed_path), "stats": fix_stats}
        write_json(fix_summary_path, fix_summary)
        output_stats = count_qasm_file(fixed_path, gate_set="nam")
        comparison = compare_gate_stats(input_stats, output_stats, "nam")
        if proc["returncode"] == 0 and not proc["timed_out"]:
            status = "ok_fixed_recount" if fix_stats.get("rz_fraction_changed", 0) else "ok"
        else:
            status = "timeout_partial" if proc["timed_out"] else "ok_fixed_recount_process_nonzero"
    result = {
        "benchmark": BENCHMARK,
        "basis": "NAM",
        "tool": "voqc",
        "input_qasm": rel(NAM_INPUT),
        "normalized_input_qasm": rel(normalized_input),
        "output_qasm": rel(output_path) if output_path.exists() else "",
        "lowered_output_qasm": rel(lowered_path) if lowered_path.exists() else "",
        "fixed_output_qasm": rel(fixed_path) if fixed_path.exists() else "",
        "summary_path": rel(tool_dir / "tfim4_voqc_runtime_result.json"),
        "status": status,
        "timed_out": bool(proc["timed_out"]),
        "rz_angle_normalization_replacements": replacements,
        "output_basis_lowering": lowering_info,
        "large_rz_fix": fix_summary,
        "process": {
            "command": command,
            "returncode": proc["returncode"],
            "elapsed_seconds": proc["elapsed_seconds"],
            "timed_out": proc["timed_out"],
            "stdout_tail": tail_text(proc["stdout"]),
            "stderr_tail": tail_text(proc["stderr"]),
            "output_written": output_path.exists(),
        },
        "input_stats": input_stats,
        "output_stats": output_stats,
        "comparison": comparison,
    }
    write_json(tool_dir / "tfim4_voqc_runtime_result.json", result)
    return result


def runtime_row(result: dict[str, Any], runtime_source_file: Path) -> dict[str, str]:
    tool = str(result["tool"])
    basis = str(result["basis"])
    status = str(result["status"])
    runtime = runtime_seconds_for_table(result)
    output_qasm = result.get("fixed_output_qasm") or result.get("output_qasm") or result.get("summary_path")
    note = (
        "2026-07-07 rerun to fill missing TFIM_4 runtime; raw circuit generation excluded."
    )
    if tool == "voqc":
        note += " VOQC output recounted after folding huge exact RZ fractions modulo 2pi."
    return {
        "benchmark_name": BENCHMARK,
        "basis_group": basis,
        "tool/method": tool,
        "method": "generic_optimizer",
        "optimizer": tool,
        "optimizer_variant": (
            "qiskit_optimization_level_3_rerun_20260707"
            if tool == "qiskit_opt3"
            else "voqc_default_fixed_recount_rerun_20260707"
        ),
        "status": status,
        "timeout": str(bool(result.get("timed_out"))).lower(),
        "runtime_sec": str(runtime),
        "runtime_source": f"{rel(Path(result['summary_path']))}:process.elapsed_seconds",
        "source_path": str(output_qasm),
        "note": note,
        "benchmark_family": FAMILY,
        "n_or_size": N_OR_SIZE,
        "collection_source_file": rel(runtime_source_file),
        "source_type": "tfim4_missing_runtime_rerun_20260707",
        "construction_time_sec": "",
        "optimizer_time_sec": str(runtime),
        "transpile_time_sec": "",
        "total_time_sec": str(runtime),
    }


def runtime_seconds_for_table(result: dict[str, Any]) -> float | str:
    if result.get("tool") == "qiskit_opt3":
        runner_summary = result.get("runner_summary") or {}
        runner_process = runner_summary.get("process") or {}
        if runner_process.get("elapsed_seconds") not in (None, ""):
            return runner_process["elapsed_seconds"]
    process = result.get("process") or {}
    return process.get("elapsed_seconds", "")


def update_runtime_csv(path: Path, new_rows: list[dict[str, str]], backup: bool = True) -> None:
    rows: list[dict[str, str]]
    if path.exists():
        with path.open(encoding="utf-8", newline="") as in_file:
            reader = csv.DictReader(in_file)
            rows = list(reader)
            fields = list(reader.fieldnames or RUNTIME_FIELDS)
    else:
        rows = []
        fields = list(RUNTIME_FIELDS)
    for field in RUNTIME_FIELDS:
        if field not in fields:
            fields.append(field)

    replacements = {
        (row["benchmark_name"], row["basis_group"], row["tool/method"]): row
        for row in new_rows
    }
    updated: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.get("benchmark_name", ""), row.get("basis_group", ""), row.get("tool/method", ""))
        replacement = replacements.get(key)
        if replacement is not None:
            updated.append({field: replacement.get(field, "") for field in fields})
            seen.add(key)
        else:
            updated.append({field: row.get(field, "") for field in fields})
    for key, row in replacements.items():
        if key not in seen:
            updated.append({field: row.get(field, "") for field in fields})

    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".pre_tfim4_runtime_rerun_20260707.bak")
        if not backup_path.exists():
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(updated)


def write_combined_summary(out_dir: Path, results: list[dict[str, Any]], runtime_rows: list[dict[str, str]]) -> Path:
    summary_path = out_dir / "tfim4_qiskit_voqc_runtime_rerun_summary.json"
    payload = {
        "benchmark": BENCHMARK,
        "created_for": "Fill missing TFIM_4 qiskit_opt3 and VOQC runtimes.",
        "raw_generation_excluded": True,
        "results": results,
        "runtime_rows": runtime_rows,
    }
    write_json(summary_path, payload)
    csv_path = out_dir / "tfim4_qiskit_voqc_runtime_rerun_summary.csv"
    fields = [
        "benchmark",
        "basis",
        "tool",
        "status",
        "timeout",
        "runtime_sec",
        "raw_gates",
        "opt_gates",
        "ratio_vs_raw",
        "non_clifford",
        "output_qasm",
        "summary_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            comparison = result.get("comparison") or {}
            output_stats = result.get("output_stats") or {}
            raw_gates = comparison.get("metric_total_before")
            opt_gates = comparison.get("metric_total_after")
            ratio = ""
            if raw_gates:
                ratio = str(float(opt_gates) / float(raw_gates))
            writer.writerow(
                {
                    "benchmark": BENCHMARK,
                    "basis": result["basis"],
                    "tool": result["tool"],
                    "status": result["status"],
                    "timeout": str(bool(result.get("timed_out"))).lower(),
                    "runtime_sec": runtime_seconds_for_table(result),
                    "raw_gates": raw_gates or "",
                    "opt_gates": opt_gates or "",
                    "ratio_vs_raw": ratio,
                    "non_clifford": (output_stats.get("clifford") or {}).get("non_clifford", ""),
                    "output_qasm": result.get("fixed_output_qasm") or result.get("output_qasm", ""),
                    "summary_path": result.get("summary_path", ""),
                }
            )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--runtime-csv", type=Path, default=DEFAULT_RUNTIME_CSV)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--skip-qiskit", action="store_true")
    parser.add_argument("--skip-voqc", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--no-update-runtime-csv", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout = None if args.timeout <= 0 else float(args.timeout)
    results: list[dict[str, Any]] = []
    if args.reuse_existing:
        if not args.skip_qiskit:
            results.append(read_json(out_dir / "qiskit_opt3" / "tfim4_qiskit_opt3_runtime_result.json"))
        if not args.skip_voqc:
            results.append(read_json(out_dir / "voqc" / "tfim4_voqc_runtime_result.json"))
    elif not args.skip_qiskit:
        results.append(run_qiskit_opt3(out_dir, timeout))
    if not args.reuse_existing and not args.skip_voqc:
        results.append(run_voqc(out_dir, timeout))
    runtime_summary_path = out_dir / "tfim4_qiskit_voqc_runtime_rerun_summary.json"
    rows = [runtime_row(result, runtime_summary_path) for result in results]
    summary_path = write_combined_summary(out_dir, results, rows)
    rows = [runtime_row(result, summary_path) for result in results]
    write_combined_summary(out_dir, results, rows)
    if not args.no_update_runtime_csv:
        update_runtime_csv(args.runtime_csv.resolve(), rows)
    print(f"wrote rerun summary to {summary_path}")
    if not args.no_update_runtime_csv:
        print(f"updated runtime CSV {args.runtime_csv}")
    failed = [result for result in results if not str(result.get("status", "")).startswith("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
