#!/usr/bin/env python3
"""Shared helpers for external-tool baseline workflows."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

IBM_U_GATES = ("u1", "u2", "u3")
IBM_METRIC_GATES = ("u1", "u2", "u3", "cx")
NAM_CLIFFORD_GATES = ("h", "x", "cx")
NAM_METRIC_GATES = ("rz", "h", "x", "cx")
HEADER_NAMES = {
    "openqasm",
    "include",
    "qreg",
    "creg",
    "qubit",
    "bit",
    "gate",
    "opaque",
    "barrier",
    "measure",
}

QREG2_RE = re.compile(r"\bqreg\s+[A-Za-z_][A-Za-z0-9_]*\[(\d+)\]\s*;")
QREG3_RE = re.compile(r"\bqubit\s*\[(\d+)\]\s+[A-Za-z_][A-Za-z0-9_]*\s*;")


def strip_qasm_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def iter_qasm_operation_names(text: str) -> list[str]:
    names: list[str] = []
    gate_definition_depth = 0

    for raw_line in text.splitlines():
        line = strip_qasm_comment(raw_line)
        if not line:
            continue

        if gate_definition_depth:
            gate_definition_depth += line.count("{") - line.count("}")
            continue

        lowered = line.lower()
        if lowered.startswith("gate ") and "{" in line:
            gate_definition_depth += line.count("{") - line.count("}")
            continue

        if lowered.startswith("if"):
            close = line.find(")")
            if close >= 0:
                line = line[close + 1 :].strip()
                lowered = line.lower()

        if not line.endswith(";"):
            continue

        head = line.split(None, 1)[0].rstrip(";").lower()
        name = head.split("(", 1)[0]
        if name.startswith("qubit[") or name.startswith("bit["):
            continue
        if name in HEADER_NAMES:
            continue
        if not name:
            continue
        names.append(name)

    return names


def count_qasm_text(text: str, path: str | None = None, gate_set: str | None = None) -> dict[str, Any]:
    names = iter_qasm_operation_names(text)
    ops = Counter(names)
    detected = detect_gate_set_from_ops(ops)
    selected_gate_set = gate_set or detected

    num_qubits = 0
    for match in QREG2_RE.finditer(text):
        num_qubits += int(match.group(1))
    for match in QREG3_RE.finditer(text):
        num_qubits += int(match.group(1))

    ibm_u_total = sum(ops.get(gate, 0) for gate in IBM_U_GATES)
    ibm_metric_total = ibm_u_total + ops.get("cx", 0)
    nam_clifford_total = sum(ops.get(gate, 0) for gate in NAM_CLIFFORD_GATES)
    nam_metric_total = ops.get("rz", 0) + nam_clifford_total
    metric_total = ibm_metric_total if selected_gate_set == "ibm" else nam_metric_total

    return {
        "path": path,
        "num_qubits": int(num_qubits),
        "detected_gate_set": detected,
        "metric_gate_set": selected_gate_set,
        "total_ops": int(sum(ops.values())),
        "ops": dict(sorted((gate, int(count)) for gate, count in ops.items())),
        "metric_total": int(metric_total),
        "ibm": {
            "cx": int(ops.get("cx", 0)),
            "u1": int(ops.get("u1", 0)),
            "u2": int(ops.get("u2", 0)),
            "u3": int(ops.get("u3", 0)),
            "u_total": int(ibm_u_total),
            "cx_u_total": int(ibm_metric_total),
        },
        "nam": {
            "rz": int(ops.get("rz", 0)),
            "h": int(ops.get("h", 0)),
            "x": int(ops.get("x", 0)),
            "cx": int(ops.get("cx", 0)),
            "clifford_total": int(nam_clifford_total),
            "rz_clifford_total": int(nam_metric_total),
        },
    }


def count_qasm_file(path: Path, gate_set: str | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return count_qasm_text(text, path=str(path), gate_set=gate_set)


def detect_gate_set_from_ops(ops: Counter[str] | dict[str, int]) -> str:
    return "ibm" if any(int(ops.get(gate, 0)) > 0 for gate in IBM_U_GATES) else "nam"


def detect_gate_set(path: Path) -> str:
    return str(count_qasm_file(path)["detected_gate_set"])


def compare_gate_stats(before: dict[str, Any], after: dict[str, Any], metric_gate_set: str) -> dict[str, Any]:
    before_ops = before.get("ops", {})
    after_ops = after.get("ops", {})
    all_gates = sorted(set(before_ops) | set(after_ops))

    if metric_gate_set == "ibm":
        metric_gates = set(IBM_METRIC_GATES)
        before_metric = int(before["ibm"]["cx_u_total"])
        after_metric = int(after["ibm"]["cx_u_total"])
    else:
        metric_gates = set(NAM_METRIC_GATES)
        before_metric = int(before["nam"]["rz_clifford_total"])
        after_metric = int(after["nam"]["rz_clifford_total"])

    untracked_output_ops = {
        gate: int(count)
        for gate, count in sorted(after_ops.items())
        if gate not in metric_gates and int(count) > 0
    }

    return {
        "metric_gate_set": metric_gate_set,
        "input_detected_gate_set": before.get("detected_gate_set"),
        "output_detected_gate_set": after.get("detected_gate_set"),
        "total_ops_before": int(before["total_ops"]),
        "total_ops_after": int(after["total_ops"]),
        "total_ops_delta": int(after["total_ops"] - before["total_ops"]),
        "metric_total_before": before_metric,
        "metric_total_after": after_metric,
        "metric_total_delta": int(after_metric - before_metric),
        "metric_total_reduction": int(before_metric - after_metric),
        "metric_covers_all_output_ops": not untracked_output_ops,
        "untracked_output_ops_for_metric": untracked_output_ops,
        "ops_delta": {
            gate: int(after_ops.get(gate, 0) - before_ops.get(gate, 0))
            for gate in all_gates
        },
    }


def run_command(
    command: list[str],
    timeout: float | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": int(proc.returncode),
            "elapsed_seconds": time.perf_counter() - start,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "elapsed_seconds": time.perf_counter() - start,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def relativize(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_summary(summary: dict[str, Any], summary_path: Path | None) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def require_input_qasm(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input QASM not found: {resolved}")
    return resolved
