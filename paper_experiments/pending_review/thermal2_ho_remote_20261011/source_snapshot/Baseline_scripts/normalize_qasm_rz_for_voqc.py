#!/usr/bin/env python3
"""Normalize OpenQASM2 rz angles to VOQC-friendly rz(f*pi) form."""

from __future__ import annotations

import argparse
import ast
import math
import re
from fractions import Fraction
from pathlib import Path


RZ_CALL_RE = re.compile(r"(?<![A-Za-z0-9_])rz\s*\(([^()]*)\)")


class AngleEvalError(ValueError):
    pass


def _eval_angle_expr(expr: str) -> float:
    """Evaluate a small arithmetic expression containing pi."""

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
        raise AngleEvalError(f"unsupported rz angle expression: {expr!r}")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise AngleEvalError(f"invalid rz angle expression: {expr!r}") from exc
    return visit(tree)


def _format_real(value: float) -> str:
    """Format as a token VOQC's parser treats as a real number."""
    if not math.isfinite(value):
        raise AngleEvalError(f"non-finite rz angle coefficient: {value!r}")

    frac = Fraction(value).limit_denominator(1_048_576)
    if math.isclose(value, float(frac), rel_tol=0.0, abs_tol=1e-12):
        value = float(frac)

    text = f"{value:.17g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def normalize_rz_angle(expr: str) -> str:
    coeff = _eval_angle_expr(expr.strip()) / math.pi
    return f"{_format_real(coeff)}*pi"


def normalize_qasm_text(text: str) -> tuple[str, int]:
    replacements = 0
    out_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        code, sep, comment = line.partition("//")

        def replace(match: re.Match[str]) -> str:
            nonlocal replacements
            replacements += 1
            return f"rz({normalize_rz_angle(match.group(1))})"

        out_lines.append(RZ_CALL_RE.sub(replace, code) + sep + comment)

    return "".join(out_lines), replacements


def normalize_qasm_file(input_path: Path, output_path: Path) -> int:
    text = input_path.read_text()
    normalized, replacements = normalize_qasm_text(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(normalized)
    return replacements


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize OpenQASM2 rz angles to VOQC-friendly rz(f*pi)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("circuits/select_only/test_table1_m_no_n4_nam_qasm2.qasm"),
        help="Input OpenQASM2 file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("circuits/test_table1_m_no_n4_nam_qasm2_normal.qasm"),
        help="Output normalized OpenQASM2 file.",
    )
    args = parser.parse_args()

    replacements = normalize_qasm_file(args.input, args.output)
    print(f"normalized {replacements} rz gates")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
