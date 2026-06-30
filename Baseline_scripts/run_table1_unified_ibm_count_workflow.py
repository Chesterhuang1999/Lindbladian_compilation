#!/usr/bin/env python3
"""Unified-count workflow for Table1 full-channel and SELECT benchmarks."""

from __future__ import annotations

import argparse
import math
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for path in (SCRIPT_DIR, SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_common import count_qasm_file, require_input_qasm, write_summary  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # noqa: E402
from test_table1 import build_tfim_lcu_select_circuit  # noqa: E402


CIRCUITS_ROOT = ROOT / "circuits"
IBM_BASIS = ("u1", "u2", "u3", "cx")
NAM_BASIS = ("h", "x", "rz", "cx")
DEFAULT_TOOLS = ("voqc", "feynman", "wisq", "quartz", "pytket", "bqskit")
CONDA_EXE = os.environ.get("CONDA_EXE", str(Path.home() / "miniconda3" / "bin" / "conda"))
TOOL_BASIS = {
    "voqc": "nam",
    "feynman": "nam",
    "wisq": "nam",
    "quartz": "nam",
    "pytket": "ibm",
    "bqskit": "ibm",
}
RZQ_RE = re.compile(r"\brzq\(\s*([^,\s]+)\s*,\s*([^)]+?)\s*\)")
QASM_REG_DECL_RE = re.compile(r"\b(?P<kind>[cq]reg)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[")
QASM2_RESERVED_REG_NAMES = {
    "barrier",
    "ccx",
    "ch",
    "cp",
    "crz",
    "cswap",
    "cu",
    "cu1",
    "cu3",
    "cx",
    "cy",
    "cz",
    "gate",
    "h",
    "id",
    "if",
    "include",
    "measure",
    "opaque",
    "openqasm",
    "p",
    "qreg",
    "creg",
    "rxx",
    "rx",
    "ry",
    "rz",
    "rzz",
    "s",
    "sdg",
    "swap",
    "t",
    "tdg",
    "u",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z",
}


def benchmark_name(args: argparse.Namespace) -> str:
    if getattr(args, "benchmark_name", None):
        return str(args.benchmark_name)
    nam_qasm_arg = getattr(args, "nam_qasm", None)
    if nam_qasm_arg is not None:
        stem = Path(str(nam_qasm_arg)).expanduser().stem
        return re.sub(r"_(nam|ibm)(_opt[0-9]+)?$", "", stem)
    ibm_qasm_arg = getattr(args, "ibm_qasm", None)
    if ibm_qasm_arg is not None:
        stem = Path(str(ibm_qasm_arg)).expanduser().stem
        return re.sub(r"_(nam|ibm)(_opt[0-9]+)?$", "", stem)
    return f"test_table1_select_n{int(args.num_qubits)}"


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._") or "benchmark"


def qiskit_stats(qc: QuantumCircuit) -> dict[str, int]:
    ops = {str(gate): int(count) for gate, count in qc.count_ops().items()}
    return {
        "num_qubits": int(qc.num_qubits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "total_ops": int(sum(ops.values())),
        "cx": int(ops.get("cx", 0)),
        "u1": int(ops.get("u1", 0)),
        "u2": int(ops.get("u2", 0)),
        "u3": int(ops.get("u3", 0)),
        "ops": ops,
    }


def basis_gate_counts(stats: dict[str, Any] | None, basis: str) -> dict[str, int] | None:
    if stats is None:
        return None
    ops = stats.get("ops", {})
    if basis == "ibm":
        clifford = stats.get("clifford", {})
        return {
            "clifford": int(clifford.get("clifford", 0)),
            "non_clifford": int(clifford.get("non_clifford", 0)),
            "u1": int(ops.get("u1", 0)),
            "u2": int(ops.get("u2", 0)),
            "u3": int(ops.get("u3", 0)),
            "cx": int(ops.get("cx", 0)),
            "total": int(stats.get("total_ops", 0)),
        }
    if basis == "nam":
        rz_breakdown = stats.get("rz_angle_breakdown", {})
        rz_non_clifford = int(rz_breakdown.get("rz_non_clifford", 0))
        rz_clifford = int(rz_breakdown.get("rz_clifford_k_pi_over_2", 0))
        h = int(ops.get("h", 0))
        x = int(ops.get("x", 0))
        cx = int(ops.get("cx", 0))
        return {
            "clifford": h + x + cx + rz_clifford,
            "non_clifford": rz_non_clifford,
            "h": h,
            "x": x,
            "cx": cx,
            "rz_clifford": rz_clifford,
            "rz_non_clifford": rz_non_clifford,
            "rz_total": int(rz_breakdown.get("rz_total", ops.get("rz", 0))),
            "total": int(stats.get("total_ops", 0)),
        }
    raise ValueError(f"Unsupported basis: {basis}")


def gate_count_total(stats: dict[str, Any] | None) -> int | None:
    if stats is None:
        return None
    return int(stats.get("metric_total", stats.get("total_ops", 0)))


def reduction_summary(before: int | None, after: int | None) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    reduction = before - after
    return {
        "total_ops_before": before,
        "total_ops_after": after,
        "total_ops_delta": after - before,
        "total_ops_reduction": reduction,
        "reduction_ratio": None if before == 0 else reduction / before,
    }


def native_elapsed_seconds(result: dict[str, Any]) -> float | None:
    process = result.get("native_process")
    if isinstance(process, dict) and process.get("elapsed_seconds") is not None:
        return float(process["elapsed_seconds"])
    if result.get("elapsed_seconds") is not None:
        return float(result["elapsed_seconds"])
    return None


def tool_status(result: dict[str, Any]) -> str:
    process = result.get("native_process")
    process_timed_out = isinstance(process, dict) and bool(process.get("timed_out"))
    if result.get("timed_out") or process_timed_out:
        if result.get("native_comparison") is not None:
            return "timeout_partial"
        return "timeout"
    if bool(result.get("native_fallback_output_used")):
        return "failed_fallback"
    if isinstance(process, dict) and process.get("ok") is False:
        return "failed"
    if result.get("launch_error"):
        return "launch_error"
    if result.get("returncode") not in (0, None):
        return "failed"
    if result.get("native_comparison") is not None:
        return "ok"
    if result.get("qiskit_final_stats") is not None:
        return "ok_recount_only"
    return "no_output"


def parse_angle_expr(expr: str) -> float | None:
    try:
        return float(eval(expr.strip(), {"__builtins__": {}}, {"pi": math.pi}))
    except Exception:
        return None


def extract_rz_angle_expr(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.lower().startswith("rz("):
        return None
    close = stripped.rfind(")")
    if close < len("rz("):
        return None
    rest = stripped[close + 1 :].strip()
    if not rest or not rest.endswith(";"):
        return None
    return stripped[len("rz(") : close]


def rz_angle_breakdown(qasm_path: Path) -> dict[str, int]:
    total = 0
    clifford = 0
    non_clifford = 0
    unknown = 0
    zero = 0
    for raw_line in qasm_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        expr = extract_rz_angle_expr(line)
        if expr is None:
            continue
        total += 1
        value = parse_angle_expr(expr)
        if value is None:
            unknown += 1
            continue
        units = value / (math.pi / 2)
        if abs(value) < 1e-10:
            zero += 1
            clifford += 1
        elif abs(units - round(units)) < 1e-8:
            clifford += 1
        else:
            non_clifford += 1
    return {
        "rz_total": total,
        "rz_clifford_k_pi_over_2": clifford,
        "rz_non_clifford": non_clifford,
        "rz_zero": zero,
        "rz_unknown": unknown,
    }


def resolve_circuits_qasm_input(qasm_arg: str | None, *, label: str) -> Path | None:
    if qasm_arg is None:
        return None

    qasm_path = Path(qasm_arg).expanduser()
    if not qasm_path.is_absolute():
        qasm_path = CIRCUITS_ROOT / qasm_path
    resolved = require_input_qasm(qasm_path.resolve())
    circuits_root = CIRCUITS_ROOT.resolve()
    if not resolved.is_relative_to(circuits_root):
        raise ValueError(f"{label} QASM must be under {circuits_root}: {resolved}")
    return resolved


def qasm2_register_renames(text: str) -> dict[str, str]:
    names = {match.group("name") for match in QASM_REG_DECL_RE.finditer(text)}
    renames: dict[str, str] = {}
    occupied = set(names) | QASM2_RESERVED_REG_NAMES

    for name in sorted(names):
        if name.lower() not in QASM2_RESERVED_REG_NAMES:
            continue
        index = 0
        candidate = f"{name}_reg"
        while candidate.lower() in occupied or candidate in renames.values():
            index += 1
            candidate = f"{name}_reg{index}"
        renames[name] = candidate
        occupied.add(candidate)
    return renames


def write_workflow_input_qasm(input_qasm: Path, output_qasm: Path) -> dict[str, str]:
    text = input_qasm.read_text(encoding="utf-8")
    renames = qasm2_register_renames(text)
    for old_name, new_name in renames.items():
        text = re.sub(rf"\b{re.escape(old_name)}(?=\[)", new_name, text)
    output_qasm.parent.mkdir(parents=True, exist_ok=True)
    output_qasm.write_text(text, encoding="utf-8")
    return renames


def cached_external_input_stats(source_qasm: Path, copied_qasm: Path, basis: str) -> dict[str, Any] | None:
    cache_path = ROOT / "Baseline_results" / "tfim_external_baseline_inputs_summary.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    source_resolved = str(source_qasm.resolve())
    for result in data.get("results", []):
        if not isinstance(result, dict):
            continue
        key = "canonical_ibm_qasm" if basis == "ibm" else "canonical_nam_qasm"
        if result.get(key) != source_resolved:
            continue
        stats_key = "ibm_stats" if basis == "ibm" else "nam_stats"
        stats = result.get(stats_key)
        if not isinstance(stats, dict):
            return None
        copied_stats = dict(stats)
        copied_stats["path"] = str(copied_qasm)
        return copied_stats
    return None


def build_inputs(args: argparse.Namespace, work_dir: Path) -> dict[str, Any]:
    bench_name = benchmark_name(args)
    prefix = sanitize_filename(bench_name)
    ibm_opt0_qasm = work_dir / f"{prefix}_ibm_opt0.qasm"
    ibm_opt3_qasm = work_dir / f"{prefix}_ibm_opt3_qiskit.qasm"
    nam_opt0_qasm = work_dir / f"{prefix}_nam_opt0.qasm"
    nam_opt3_qasm = work_dir / f"{prefix}_nam_opt3_qiskit.qasm"

    external_nam_qasm = resolve_circuits_qasm_input(
        getattr(args, "nam_qasm", None),
        label="External NAM",
    )
    external_ibm_qasm = resolve_circuits_qasm_input(
        getattr(args, "ibm_qasm", None),
        label="External IBM",
    )
    external_lowered_inputs = external_nam_qasm is not None or external_ibm_qasm is not None
    if external_lowered_inputs and (external_nam_qasm is None or external_ibm_qasm is None):
        raise ValueError(
            "External-tool workflow now expects both canonical lowered inputs: "
            "--nam-qasm for NAM-native tools and --ibm-qasm for IBM-native tools."
        )

    logical_stats = None
    input_source_qasm: dict[str, str | None] = {"nam": None, "ibm": None}
    register_renames: dict[str, dict[str, str]] = {"nam": {}, "ibm": {}}
    cached_opt0_stats: dict[str, dict[str, Any] | None] = {"nam": None, "ibm": None}

    if not external_lowered_inputs:
        logical_qc = build_tfim_lcu_select_circuit(
            int(args.num_qubits),
            metrics=args.metrics,
            delta_t=float(args.delta_t),
        )
        input_source = "generated_in_memory"
        logical_stats = qiskit_stats(logical_qc)

        ibm_opt0 = transpile(logical_qc, basis_gates=list(IBM_BASIS), optimization_level=0)
        ibm_opt3 = transpile(logical_qc, basis_gates=list(IBM_BASIS), optimization_level=3)
        nam_opt0 = transpile(logical_qc, basis_gates=list(NAM_BASIS), optimization_level=0)
        nam_opt3 = transpile(logical_qc, basis_gates=list(NAM_BASIS), optimization_level=3)
    else:
        assert external_nam_qasm is not None
        assert external_ibm_qasm is not None
        input_source = "external_lowered_qasm"
        input_source_qasm = {
            "nam": str(external_nam_qasm),
            "ibm": str(external_ibm_qasm),
        }
        register_renames = {
            "nam": write_workflow_input_qasm(external_nam_qasm, nam_opt0_qasm),
            "ibm": write_workflow_input_qasm(external_ibm_qasm, ibm_opt0_qasm),
        }
        cached_opt0_stats = {
            "nam": cached_external_input_stats(external_nam_qasm, nam_opt0_qasm, "nam"),
            "ibm": cached_external_input_stats(external_ibm_qasm, ibm_opt0_qasm, "ibm"),
        }

        if bool(args.skip_qiskit_opt3):
            ibm_opt3 = None
            nam_opt3 = None
        else:
            # These inputs are already in the requested basis.  The QASM readback is
            # only used to run the Qiskit opt3 baseline on the same canonical input.
            ibm_opt0 = QuantumCircuit.from_qasm_file(str(ibm_opt0_qasm))
            nam_opt0 = QuantumCircuit.from_qasm_file(str(nam_opt0_qasm))
            ibm_opt3 = transpile(ibm_opt0, basis_gates=list(IBM_BASIS), optimization_level=3)
            nam_opt3 = transpile(nam_opt0, basis_gates=list(NAM_BASIS), optimization_level=3)

    if not external_lowered_inputs:
        export_openqasm2_quartz_input(ibm_opt0, ibm_opt0_qasm, basis_gates=IBM_BASIS)
        export_openqasm2_quartz_input(nam_opt0, nam_opt0_qasm, basis_gates=NAM_BASIS)
    if ibm_opt3 is not None:
        export_openqasm2_quartz_input(ibm_opt3, ibm_opt3_qasm, basis_gates=IBM_BASIS)
    if nam_opt3 is not None:
        export_openqasm2_quartz_input(nam_opt3, nam_opt3_qasm, basis_gates=NAM_BASIS)

    nam_opt0_stats = cached_opt0_stats["nam"] or count_qasm_file(nam_opt0_qasm, gate_set="nam")
    if "rz_angle_breakdown" not in nam_opt0_stats:
        nam_opt0_stats["rz_angle_breakdown"] = rz_angle_breakdown(nam_opt0_qasm)
    if nam_opt3 is None:
        nam_opt3_stats = None
    else:
        nam_opt3_stats = count_qasm_file(nam_opt3_qasm, gate_set="nam")
        nam_opt3_stats["rz_angle_breakdown"] = rz_angle_breakdown(nam_opt3_qasm)

    return {
        "benchmark": bench_name,
        "input_source": input_source,
        "source_qasm": input_source_qasm,
        "qasm_register_renames": register_renames,
        "canonical_input_policy": (
            "External tools receive basis-level OpenQASM2 only: NAM tools use "
            "nam_opt0_qasm and IBM tools use ibm_opt0_qasm. High-level raw "
            "channel-LCU QASM is intentionally not used at the tool boundary."
        ),
        "ibm_opt0_qasm": str(ibm_opt0_qasm),
        "ibm_opt3_qasm": str(ibm_opt3_qasm) if ibm_opt3 is not None else None,
        "nam_opt0_qasm": str(nam_opt0_qasm),
        "nam_opt3_qasm": str(nam_opt3_qasm) if nam_opt3 is not None else None,
        "logical_circuit_stats": logical_stats,
        "ibm_opt0_stats": cached_opt0_stats["ibm"]
        or count_qasm_file(ibm_opt0_qasm, gate_set="ibm"),
        "ibm_opt3_stats": None
        if ibm_opt3 is None
        else count_qasm_file(ibm_opt3_qasm, gate_set="ibm"),
        "nam_opt0_stats": nam_opt0_stats,
        "nam_opt3_stats": nam_opt3_stats,
    }


def runner_command(
    tool: str,
    input_path: Path,
    input_stats_path: Path | None,
    output_path: Path,
    summary_path: Path,
    timeout: float,
    args: argparse.Namespace,
) -> list[str]:
    base = [
        CONDA_EXE,
        "run",
        "-n",
        "ChannelIR_test",
        "python",
        str(SCRIPT_DIR / f"run_{tool}_baseline.py"),
    ]
    command = [
        *base,
        "--input",
        str(input_path),
        "--summary",
        str(summary_path),
        "--output",
        str(output_path),
    ]
    if input_stats_path is not None:
        command.extend(["--input-stats", str(input_stats_path)])
    if tool in {"voqc", "feynman"}:
        command.extend(["--timeout", str(float(timeout))])
    elif tool == "wisq":
        command = [
            CONDA_EXE,
            "run",
            "-n",
            "ChannelIR_test",
            "python",
            str(SCRIPT_DIR / "run_wisq_baseline.py"),
            "--input",
            str(input_path),
            "--summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--timeout",
            str(int(timeout)),
        ]
        if input_stats_path is not None:
            command.extend(["--input-stats", str(input_stats_path)])
        command.extend(["--optimization-objective", str(args.wisq_optimization_objective)])
        if args.wisq_approx_epsilon is not None:
            command.extend(["--approx-epsilon", str(float(args.wisq_approx_epsilon))])
        if int(args.wisq_opt_threads) > 1:
            command.extend(["--opt-threads", str(int(args.wisq_opt_threads))])
        if args.wisq_advanced_args:
            command.extend(["--advanced-args", str(args.wisq_advanced_args)])
    elif tool == "quartz":
        command = [
            CONDA_EXE,
            "run",
            "-n",
            "ChannelIR_test",
            "python",
            str(SCRIPT_DIR / "run_quartz_baseline.py"),
            "--input",
            str(input_path),
            "--summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--timeout",
            str(float(timeout)),
            "--max-candidates",
            str(int(args.quartz_max_candidates)),
            "--upper-limit",
            str(float(args.quartz_upper_limit)),
        ]
        if input_stats_path is not None:
            command.extend(["--input-stats", str(input_stats_path)])
        if args.quartz_ecc:
            command.extend(["--ecc", str(args.quartz_ecc)])
        if int(args.quartz_progress_every) > 0:
            command.extend(["--progress-every", str(int(args.quartz_progress_every))])
        if float(args.quartz_checkpoint_interval) > 0:
            command.extend(["--checkpoint-interval", str(float(args.quartz_checkpoint_interval))])
            command.extend(
                [
                    "--checkpoint-output",
                    str(output_path.with_name(f"{tool}_native_checkpoint.qasm")),
                ]
            )
            command.extend(
                [
                    "--checkpoint-summary",
                    str(output_path.with_name(f"{tool}_native_checkpoint.json")),
                ]
            )
        if bool(args.quartz_allow_trivial_xfer_set):
            command.append("--allow-trivial-xfer-set")
        if bool(args.quartz_no_increase):
            command.append("--no-increase")
        if bool(args.quartz_no_nop):
            command.append("--no-nop")
        if bool(args.quartz_use_available_xfers):
            command.append("--use-available-xfers")
        if bool(args.quartz_prune_relative_to_best):
            command.append("--prune-relative-to-best")
        if bool(args.quartz_no_eliminate_rotation):
            command.append("--no-eliminate-rotation")
    elif tool == "pytket":
        command.extend(["--strategy", str(args.pytket_strategy)])
        if args.pytket_strategy == "best-of":
            command.extend(["--best-of-objective", str(args.pytket_best_of_objective)])
        else:
            command.extend(["--pass-name", str(args.pytket_pass)])
    elif tool == "bqskit":
        command = [
            CONDA_EXE,
            "run",
            "-n",
            "ChannelIR_test",
            "python",
            str(SCRIPT_DIR / "run_bqskit_baseline.py"),
            "--input",
            str(input_path),
            "--summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--gate-set",
            "ibm",
            "--optimization-level",
            str(int(args.bqskit_optimization_level)),
            "--max-synthesis-size",
            str(int(args.bqskit_max_synthesis_size)),
            "--workflow-mode",
            str(args.bqskit_workflow_mode),
            "--timeout",
            str(float(timeout)),
        ]
        if input_stats_path is not None:
            command.extend(["--input-stats", str(input_stats_path)])
        if bool(args.bqskit_checkpoint):
            command.append("--checkpoint")
    else:
        raise ValueError(f"Unsupported tool: {tool}")
    return command


def make_qiskit_readable_qasm(input_path: Path, output_path: Path) -> Path:
    text = input_path.read_text(encoding="utf-8", errors="ignore")

    def replace_rzq(match: re.Match[str]) -> str:
        numerator = match.group(1).strip()
        denominator = match.group(2).strip()
        return f"rz(({numerator})*pi/({denominator}))"

    text = RZQ_RE.sub(replace_rzq, text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def qiskit_recount_output(
    output_path: Path,
    counted_path: Path,
    final_count_basis: str,
) -> dict[str, Any] | None:
    if not output_path.exists():
        return None
    readable_path = output_path.with_name(f"{output_path.stem}_qiskit_readable.qasm")
    make_qiskit_readable_qasm(output_path, readable_path)
    qc = QuantumCircuit.from_qasm_file(str(readable_path))
    if final_count_basis == "ibm":
        basis_gates = IBM_BASIS
    elif final_count_basis == "nam":
        basis_gates = NAM_BASIS
    else:
        raise ValueError(f"Unsupported final count basis: {final_count_basis}")

    tqc = transpile(qc, basis_gates=list(basis_gates), optimization_level=0)
    export_openqasm2_quartz_input(tqc, counted_path, basis_gates=basis_gates)
    stats = count_qasm_file(counted_path, gate_set=final_count_basis)
    stats[f"qiskit_transpile_to_{final_count_basis}_opt_level"] = 0
    stats["qiskit_readable_input"] = str(readable_path)
    if final_count_basis == "nam":
        stats["rz_angle_breakdown"] = rz_angle_breakdown(counted_path)
    return stats


def safe_qiskit_recount_output(
    output_path: Path,
    counted_path: Path,
    final_count_basis: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return qiskit_recount_output(output_path, counted_path, final_count_basis), None
    except Exception as exc:
        return None, str(exc)


def run_tool(
    tool: str,
    input_path: Path,
    input_stats_path: Path | None,
    tool_dir: Path,
    timeout: float,
    keep_intermediate: bool,
    final_count_basis: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = tool_dir / f"{tool}_native_output.qasm"
    summary_path = tool_dir / f"{tool}_native_summary.json"
    counted_path = tool_dir / f"{tool}_qiskit_{final_count_basis}_counted.qasm"
    tool_dir.mkdir(parents=True, exist_ok=True)
    command = runner_command(tool, input_path, input_stats_path, output_path, summary_path, timeout, args)

    start = time.perf_counter()
    timed_out = False
    launch_error = None
    proc = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=float(timeout) + 120.0)
        returncode = proc.returncode
        stdout_tail = stdout[-4000:]
        stderr_tail = stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
                stdout, stderr = proc.communicate()
            returncode = proc.returncode
        else:
            returncode = None
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stdout_tail = stdout[-4000:]
        stderr_tail = stderr[-4000:]
    except OSError as exc:
        returncode = None
        stdout_tail = ""
        stderr_tail = str(exc)
        launch_error = str(exc)
    elapsed = time.perf_counter() - start

    native_summary = None
    if summary_path.exists():
        native_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        native_output_stats = native_summary.get("output_stats")
        if (
            TOOL_BASIS[tool] == "nam"
            and isinstance(native_output_stats, dict)
            and "rz_angle_breakdown" not in native_output_stats
        ):
            stats_path = native_output_stats.get("path")
            breakdown_path = Path(str(stats_path)) if stats_path else output_path
            if breakdown_path.exists():
                native_output_stats["rz_angle_breakdown"] = rz_angle_breakdown(breakdown_path)
    if bool(args.skip_final_recount):
        qiskit_final_stats = None
        qiskit_final_error = "skipped"
    else:
        qiskit_final_stats, qiskit_final_error = safe_qiskit_recount_output(
            output_path,
            counted_path,
            final_count_basis,
        )

    if not keep_intermediate and output_path.exists():
        output_path.unlink()
    if not keep_intermediate and counted_path.exists():
        counted_path.unlink()

    return {
        "tool": tool,
        "native_basis": TOOL_BASIS[tool],
        "input": str(input_path),
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "elapsed_seconds": elapsed,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "native_output": str(output_path) if output_path.exists() else None,
        "native_summary": str(summary_path) if summary_path.exists() else None,
        "native_comparison": None if native_summary is None else native_summary.get("comparison"),
        "native_process": None if native_summary is None else native_summary.get("process"),
        "native_fallback_output_used": (
            None if native_summary is None else native_summary.get("fallback_output_used")
        ),
        "native_checkpoint_output_used": (
            None if native_summary is None else native_summary.get("checkpoint_output_used")
        ),
        "native_input_stats": None if native_summary is None else native_summary.get("input_stats"),
        "native_output_stats": None if native_summary is None else native_summary.get("output_stats"),
        "qiskit_final_count_basis": final_count_basis,
        "qiskit_final_counted_qasm": str(counted_path) if counted_path.exists() else None,
        "qiskit_final_stats": qiskit_final_stats,
        "qiskit_final_error": qiskit_final_error,
        # Backward-compatible aliases for older IBM-count experiments.
        "qiskit_ibm_counted_qasm": (
            str(counted_path)
            if final_count_basis == "ibm" and counted_path.exists()
            else None
        ),
        "qiskit_ibm_stats": qiskit_final_stats if final_count_basis == "ibm" else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_dir = Path(args.result_dir).resolve()
    work_dir = result_dir / "inputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    bench_name = benchmark_name(args)

    inputs = build_inputs(args, work_dir)
    ibm_input = require_input_qasm(Path(inputs["ibm_opt0_qasm"]))
    nam_input = require_input_qasm(Path(inputs["nam_opt0_qasm"]))
    ibm_input_stats_path = work_dir / "ibm_opt0_stats.json"
    nam_input_stats_path = work_dir / "nam_opt0_stats.json"
    ibm_input_stats_path.write_text(
        json.dumps({"stats": inputs["ibm_opt0_stats"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    nam_input_stats_path.write_text(
        json.dumps({"stats": inputs["nam_opt0_stats"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tools = list(args.tools)

    tool_results = []
    for tool in tools:
        basis = TOOL_BASIS[tool]
        tool_input = ibm_input if basis == "ibm" else nam_input
        tool_input_stats = ibm_input_stats_path if basis == "ibm" else nam_input_stats_path
        tool_results.append(
            run_tool(
                tool=tool,
                input_path=tool_input,
                input_stats_path=tool_input_stats,
                tool_dir=result_dir / tool,
                timeout=float(args.timeout),
                keep_intermediate=bool(args.keep_intermediate),
                final_count_basis=args.final_count_basis,
                args=args,
            )
        )

    if args.final_count_basis == "ibm":
        baseline_stats = inputs["ibm_opt0_stats"]
    else:
        baseline_stats = inputs["nam_opt0_stats"]
    baseline_total = int(baseline_stats["total_ops"])
    for result in tool_results:
        stats = result.get("qiskit_final_stats")
        if stats is None:
            result["qiskit_final_comparison_vs_tool_input_opt0"] = None
        else:
            after = int(stats["total_ops"])
            result["qiskit_final_comparison_vs_tool_input_opt0"] = {
                "basis": args.final_count_basis,
                "total_ops_before": baseline_total,
                "total_ops_after": after,
                "total_ops_delta": after - baseline_total,
                "total_ops_reduction": baseline_total - after,
                "reduction_ratio": None
                if baseline_total == 0
                else (baseline_total - after) / baseline_total,
            }
        result["qiskit_ibm_comparison_vs_ibm_opt0"] = (
            result.get("qiskit_final_comparison_vs_tool_input_opt0")
            if args.final_count_basis == "ibm"
            else None
        )
        result["status"] = tool_status(result)

        native_comparison = result.get("native_comparison")
        native_process = result.get("native_process")
        if native_comparison is not None:
            native_before = native_comparison.get("metric_total_before")
            native_reduction = native_comparison.get("metric_total_reduction")
            result["native_basis_comparison_vs_tool_input_opt0"] = {
                **native_comparison,
                "basis": result["native_basis"],
                "reduction_ratio": (
                    None
                    if not native_before
                    else float(native_reduction) / float(native_before)
                ),
                "elapsed_seconds": native_elapsed_seconds(result),
                "output_covers_metric_basis": native_comparison.get(
                    "metric_covers_all_output_ops",
                ),
                "status": result["status"],
            }
        else:
            result["native_basis_comparison_vs_tool_input_opt0"] = {
                "basis": result["native_basis"],
                "elapsed_seconds": native_elapsed_seconds(result),
                "status": result["status"],
                "process": native_process,
            }

    if args.final_count_basis == "ibm":
        qiskit_opt0_stats = inputs["ibm_opt0_stats"]
        qiskit_opt3_stats = inputs["ibm_opt3_stats"]
    else:
        qiskit_opt0_stats = inputs["nam_opt0_stats"]
        qiskit_opt3_stats = inputs["nam_opt3_stats"]

    rows: list[dict[str, Any]] = [
        {
            "benchmark": bench_name,
            "method": f"qiskit_opt0_{args.final_count_basis}",
            "gate_counts": basis_gate_counts(qiskit_opt0_stats, args.final_count_basis),
        },
        {
            "benchmark": bench_name,
            "method": f"qiskit_opt3_{args.final_count_basis}",
            "gate_counts": basis_gate_counts(qiskit_opt3_stats, args.final_count_basis),
        },
    ]
    for result in tool_results:
        final_stats = result.get("qiskit_final_stats")
        rows.append(
            {
                "benchmark": bench_name,
                "method": result["tool"],
                "gate_counts": basis_gate_counts(final_stats, args.final_count_basis),
            }
        )

    native_rows: list[dict[str, Any]] = []
    native_qiskit_rows: list[dict[str, Any]] = []
    for basis in ("ibm", "nam"):
        opt0_stats = inputs[f"{basis}_opt0_stats"]
        opt3_stats = inputs[f"{basis}_opt3_stats"]
        before = gate_count_total(opt0_stats)
        after = gate_count_total(opt3_stats)
        native_rows.append(
            {
                "benchmark": bench_name,
                "method": f"qiskit_opt3_{basis}",
                "basis": basis,
                "status": "skipped" if opt3_stats is None else "ok",
                "gate_counts": basis_gate_counts(opt3_stats, basis),
                "comparison_vs_opt0": reduction_summary(before, after),
                "elapsed_seconds": None,
            }
        )

    for result in tool_results:
        basis = str(result["native_basis"])
        native_rows.append(
            {
                "benchmark": bench_name,
                "method": result["tool"],
                "basis": basis,
                "status": result["status"],
                "gate_counts": (
                    basis_gate_counts(result.get("native_output_stats"), basis)
                    if result.get("native_output_stats") is not None
                    else None
                ),
                "comparison_vs_opt0": result.get(
                    "native_basis_comparison_vs_tool_input_opt0",
                ),
                "elapsed_seconds": native_elapsed_seconds(result),
            }
        )
        final_stats = result.get("qiskit_final_stats")
        native_qiskit_rows.append(
            {
                "benchmark": bench_name,
                "method": result["tool"],
                "native_basis": basis,
                "final_count_basis": args.final_count_basis,
                "status": result["status"],
                "gate_counts": basis_gate_counts(final_stats, args.final_count_basis),
                "comparison_vs_final_basis_opt0": result.get(
                    "qiskit_final_comparison_vs_tool_input_opt0",
                ),
                "elapsed_seconds": native_elapsed_seconds(result),
            }
        )

    summary = {
        "benchmark": bench_name,
        "inputs": inputs,
        "tool_results": tool_results,
        "rows": rows,
        "native_basis_rows": native_rows,
        "final_recount_rows": native_qiskit_rows,
    }
    summary_path = (
        result_dir
        / f"{sanitize_filename(bench_name)}_unified_{args.final_count_basis}_count_summary.json"
    )
    write_summary(summary, summary_path)

    native_summary = {
        "benchmark": bench_name,
        "comparison_mode": "native_basis",
        "rows": native_rows,
        "tool_results": tool_results,
    }
    native_summary_path = (
        result_dir
        / f"{sanitize_filename(bench_name)}_native_basis_summary.json"
    )
    write_summary(native_summary, native_summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype a unified workflow: canonical basis-level QASM inputs -> "
            "tool optimization -> native-basis and Qiskit recount summaries."
        )
    )
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--metrics", default="m_no")
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Optional benchmark label used in summaries and generated input filenames.",
    )
    parser.add_argument(
        "--nam-qasm",
        default=None,
        help=(
            "Canonical NAM-basis OpenQASM2 input for NAM-native tools. "
            "Relative paths are resolved under "
            f"{CIRCUITS_ROOT}."
        ),
    )
    parser.add_argument(
        "--ibm-qasm",
        default=None,
        help=(
            "Canonical IBM-basis OpenQASM2 input for IBM-native tools. "
            "Relative paths are resolved under "
            f"{CIRCUITS_ROOT}."
        ),
    )
    parser.add_argument("--result-dir", default=str(ROOT / "Baseline_results" / "table1_unified"))
    parser.add_argument("--tools", nargs="+", choices=sorted(TOOL_BASIS), default=list(DEFAULT_TOOLS))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--final-count-basis", choices=["ibm", "nam"], default="ibm")
    parser.add_argument(
        "--skip-qiskit-opt3",
        action="store_true",
        help="Skip Qiskit opt3 baseline generation for large external lowered inputs.",
    )
    parser.add_argument(
        "--skip-final-recount",
        action="store_true",
        help="Skip Qiskit recount of tool outputs and keep native-basis tool metrics only.",
    )
    parser.add_argument(
        "--wisq-optimization-objective",
        choices=["TWO_Q", "FIDELITY", "FT", "TOTAL", "T"],
        default="TOTAL",
    )
    parser.add_argument("--wisq-approx-epsilon", type=float, default=None)
    parser.add_argument("--wisq-opt-threads", type=int, default=1)
    parser.add_argument(
        "--wisq-advanced-args",
        default=None,
        help="Optional WISQ/GUOQ advanced-args JSON path.",
    )
    parser.add_argument("--quartz-ecc", default=None, help="Optional Quartz ECC JSON override.")
    parser.add_argument("--quartz-max-candidates", type=int, default=1000)
    parser.add_argument("--quartz-upper-limit", type=float, default=1.0)
    parser.add_argument("--quartz-progress-every", type=int, default=0)
    parser.add_argument("--quartz-checkpoint-interval", type=float, default=0.0)
    parser.add_argument("--quartz-allow-trivial-xfer-set", action="store_true")
    parser.add_argument("--quartz-no-increase", action="store_true")
    parser.add_argument("--quartz-no-nop", action="store_true")
    parser.add_argument("--quartz-use-available-xfers", action="store_true")
    parser.add_argument("--quartz-prune-relative-to-best", action="store_true")
    parser.add_argument("--quartz-no-eliminate-rotation", action="store_true")
    parser.add_argument(
        "--pytket-strategy",
        choices=["single", "best-of"],
        default="best-of",
    )
    parser.add_argument(
        "--pytket-pass",
        choices=["full-peephole", "remove-redundancies", "synthesise-tket"],
        default="remove-redundancies",
    )
    parser.add_argument(
        "--pytket-best-of-objective",
        choices=["total_ops", "metric_total"],
        default="total_ops",
    )
    parser.add_argument("--bqskit-optimization-level", type=int, default=3)
    parser.add_argument("--bqskit-max-synthesis-size", type=int, default=3)
    parser.add_argument(
        "--bqskit-workflow-mode",
        choices=["standard", "skip-retarget-mapping"],
        default="standard",
    )
    parser.add_argument("--bqskit-checkpoint", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
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
