#!/usr/bin/env python3
"""Shared helpers for external-tool baseline workflows."""

from __future__ import annotations

import json
import ast
import math
import os
import re
import signal
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class QasmOperation:
    name: str
    params: tuple[str, ...]


def strip_qasm_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def split_qasm_params(params: str) -> tuple[str, ...]:
    values: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(params):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            values.append(params[start:index].strip())
            start = index + 1
    tail = params[start:].strip()
    if tail:
        values.append(tail)
    return tuple(values)


def parse_qasm_operation(line: str) -> QasmOperation | None:
    if not line.endswith(";"):
        return None

    stripped = line.strip()
    match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)", stripped)
    if match is None:
        return None
    name = match.group("name")
    cursor = match.end()
    params: tuple[str, ...] = ()

    if cursor < len(stripped) and stripped[cursor] == "(":
        depth = 0
        close = None
        for index in range(cursor, len(stripped)):
            char = stripped[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close = index
                    break
        if close is None:
            return None
        params = split_qasm_params(stripped[cursor + 1 : close])
        rest = stripped[close + 1 :].strip()
    else:
        rest = stripped[cursor:].strip()

    if rest and not rest.endswith(";"):
        return None

    name = name.lower()
    if name.startswith("qubit[") or name.startswith("bit["):
        return None
    if name in HEADER_NAMES:
        return None
    if not name:
        return None
    return QasmOperation(name=name, params=params)


def iter_qasm_operations(text: str) -> list[QasmOperation]:
    operations: list[QasmOperation] = []
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
        operation = parse_qasm_operation(line)
        if operation is not None:
            operations.append(operation)

    return operations


def iter_qasm_operation_names(text: str) -> list[str]:
    return [operation.name for operation in iter_qasm_operations(text)]


class AngleEvalError(ValueError):
    pass


def eval_angle_expr(expr: str) -> float:
    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise AngleEvalError(f"unsupported angle expression: {expr!r}")

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise AngleEvalError(f"invalid angle expression: {expr!r}") from exc
    return visit(tree)


def is_multiple_of_pi_over_2(value: float, tol: float = 1e-8) -> bool:
    units = value / (math.pi / 2)
    return abs(units - round(units)) <= tol


def angle_value(params: tuple[str, ...], index: int) -> float | None:
    if index >= len(params):
        return None
    try:
        return eval_angle_expr(params[index])
    except AngleEvalError:
        return None


def is_ibm_operation_clifford(operation: QasmOperation) -> bool | None:
    name = operation.name
    if name == "cx":
        return True
    if name == "u1":
        lam = angle_value(operation.params, 0)
        return None if lam is None else is_multiple_of_pi_over_2(lam)
    if name == "u2":
        phi = angle_value(operation.params, 0)
        lam = angle_value(operation.params, 1)
        if phi is None or lam is None:
            return None
        return is_multiple_of_pi_over_2(phi) and is_multiple_of_pi_over_2(lam)
    if name == "u3":
        theta = angle_value(operation.params, 0)
        phi = angle_value(operation.params, 1)
        lam = angle_value(operation.params, 2)
        if theta is None or phi is None or lam is None:
            return None

        theta_units = theta / (math.pi / 2)
        nearest = round(theta_units)
        if abs(theta_units - nearest) > 1e-8:
            return False
        theta_mod = nearest % 4
        if theta_mod == 0:
            return is_multiple_of_pi_over_2(phi + lam)
        if theta_mod == 2:
            return is_multiple_of_pi_over_2(phi - lam)
        return is_multiple_of_pi_over_2(phi) and is_multiple_of_pi_over_2(lam)

    return None


def is_nam_operation_clifford(operation: QasmOperation) -> bool | None:
    if operation.name in NAM_CLIFFORD_GATES:
        return True
    if operation.name == "rz":
        theta = angle_value(operation.params, 0)
        return None if theta is None else is_multiple_of_pi_over_2(theta)
    return None


def clifford_breakdown(operations: list[QasmOperation], gate_set: str) -> dict[str, Any]:
    if gate_set == "ibm":
        metric_gates = set(IBM_METRIC_GATES)
        classifier = is_ibm_operation_clifford
    elif gate_set == "nam":
        metric_gates = set(NAM_METRIC_GATES)
        classifier = is_nam_operation_clifford
    else:
        raise ValueError(f"Unsupported gate set: {gate_set}")

    clifford_ops = 0
    non_clifford_ops = 0
    unknown_ops = 0
    ignored_ops = 0
    by_gate: dict[str, dict[str, int]] = {}

    for operation in operations:
        if operation.name not in metric_gates:
            ignored_ops += 1
            continue
        gate_counts = by_gate.setdefault(
            operation.name,
            {"clifford": 0, "non_clifford": 0, "unknown": 0},
        )
        is_clifford = classifier(operation)
        if is_clifford is True:
            clifford_ops += 1
            gate_counts["clifford"] += 1
        elif is_clifford is False:
            non_clifford_ops += 1
            gate_counts["non_clifford"] += 1
        else:
            unknown_ops += 1
            gate_counts["unknown"] += 1

    return {
        "gate_set": gate_set,
        "clifford": int(clifford_ops),
        "non_clifford": int(non_clifford_ops),
        "unknown": int(unknown_ops),
        "ignored": int(ignored_ops),
        "metric_total_classified": int(clifford_ops + non_clifford_ops + unknown_ops),
        "by_gate": {gate: counts for gate, counts in sorted(by_gate.items())},
    }


def count_qasm_text(text: str, path: str | None = None, gate_set: str | None = None) -> dict[str, Any]:
    operations = iter_qasm_operations(text)
    names = [operation.name for operation in operations]
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
        "clifford": clifford_breakdown(operations, selected_gate_set),
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


def load_qasm_stats_or_count(
    path: Path,
    gate_set: str | None = None,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    if stats_path is None:
        return count_qasm_file(path, gate_set=gate_set)

    data = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = data.get("stats", data)
    if not isinstance(stats, dict):
        raise ValueError(f"Input stats JSON does not contain a stats object: {stats_path}")
    loaded = dict(stats)
    loaded["path"] = str(path)
    if gate_set is not None:
        loaded["metric_gate_set"] = gate_set
    return loaded


def detect_gate_set_from_ops(ops: Counter[str] | dict[str, int]) -> str:
    return "ibm" if any(int(ops.get(gate, 0)) > 0 for gate in IBM_U_GATES) else "nam"


def detect_gate_set(path: Path) -> str:
    return str(count_qasm_file(path)["detected_gate_set"])


def compare_gate_stats(before: dict[str, Any], after: dict[str, Any], metric_gate_set: str) -> dict[str, Any]:
    before_ops = before.get("ops", {})
    after_ops = after.get("ops", {})
    all_gates = sorted(set(before_ops) | set(after_ops))
    before_clifford = before.get("clifford", {})
    after_clifford = after.get("clifford", {})

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
        "clifford_before": int(before_clifford.get("clifford", 0)),
        "clifford_after": int(after_clifford.get("clifford", 0)),
        "clifford_delta": int(
            after_clifford.get("clifford", 0) - before_clifford.get("clifford", 0)
        ),
        "non_clifford_before": int(before_clifford.get("non_clifford", 0)),
        "non_clifford_after": int(after_clifford.get("non_clifford", 0)),
        "non_clifford_delta": int(
            after_clifford.get("non_clifford", 0)
            - before_clifford.get("non_clifford", 0)
        ),
        "unknown_clifford_before": int(before_clifford.get("unknown", 0)),
        "unknown_clifford_after": int(after_clifford.get("unknown", 0)),
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
    def to_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "command": command,
            "returncode": int(proc.returncode),
            "elapsed_seconds": time.perf_counter() - start,
            "stdout": to_text(stdout),
            "stderr": to_text(stderr),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
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
        return {
            "command": command,
            "returncode": proc.returncode,
            "elapsed_seconds": time.perf_counter() - start,
            "stdout": to_text(stdout or exc.stdout),
            "stderr": to_text(stderr or exc.stderr),
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
