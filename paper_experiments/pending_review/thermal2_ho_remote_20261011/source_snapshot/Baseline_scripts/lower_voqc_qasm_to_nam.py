#!/usr/bin/env python3
"""Lower VOQC's phase-gate QASM output to the NAM basis {rz, h, x, cx}."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from math import gcd
from pathlib import Path


RZQ_RE = re.compile(
    r"^(?P<indent>\s*)rzq\s*\(\s*(?P<num>[^,\s]+)\s*,\s*(?P<den>[^)\s]+)\s*\)"
    r"\s+(?P<arg>[^;]+?)\s*;\s*$",
    re.IGNORECASE,
)
PHASE_GATE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<gate>s|sdg|t|tdg|z)\s+(?P<arg>[^;]+?)\s*;\s*$",
    re.IGNORECASE,
)
PHASE_GATE_TO_RZ = {
    "s": "pi/2",
    "sdg": "-pi/2",
    "t": "pi/4",
    "tdg": "-pi/4",
    "z": "pi",
}


def normalize_rzq_fraction(numerator: str, denominator: str) -> tuple[str, str]:
    """Fold VOQC rzq(num, den) coefficients into a small exact pi multiple."""
    num = int(numerator)
    den = int(denominator)
    if den == 0:
        return numerator, denominator
    if den < 0:
        num = -num
        den = -den

    period = 2 * den
    num = ((num + den) % period) - den
    divisor = gcd(abs(num), den)
    if divisor:
        num //= divisor
        den //= divisor
    return str(num), str(den)


def split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def lower_voqc_code_line_to_nam(code: str, converted_ops: Counter[str]) -> str:
    rzq_match = RZQ_RE.match(code)
    if rzq_match is not None:
        converted_ops["rzq"] += 1
        indent = rzq_match.group("indent")
        numerator = rzq_match.group("num")
        denominator = rzq_match.group("den")
        try:
            numerator, denominator = normalize_rzq_fraction(numerator, denominator)
            converted_ops["rzq_normalized"] += 1
        except ValueError:
            pass
        arg = rzq_match.group("arg")
        return f"{indent}rz(({numerator})*pi/({denominator})) {arg};"

    phase_match = PHASE_GATE_RE.match(code)
    if phase_match is not None:
        gate = phase_match.group("gate").lower()
        converted_ops[gate] += 1
        indent = phase_match.group("indent")
        arg = phase_match.group("arg")
        return f"{indent}rz({PHASE_GATE_TO_RZ[gate]}) {arg};"

    return code


def lower_voqc_qasm_text_to_nam(text: str) -> tuple[str, dict[str, int]]:
    converted_ops: Counter[str] = Counter()
    lowered_lines: list[str] = []

    for raw_line in text.splitlines(keepends=True):
        body, line_ending = split_line_ending(raw_line)
        code, separator, comment = body.partition("//")
        lowered_code = lower_voqc_code_line_to_nam(code, converted_ops)
        lowered_lines.append(lowered_code + separator + comment + line_ending)

    return "".join(lowered_lines), dict(sorted(converted_ops.items()))


def lower_voqc_qasm_file_to_nam(input_path: Path, output_path: Path) -> dict[str, object]:
    lowered_text, converted_ops = lower_voqc_qasm_text_to_nam(
        input_path.read_text(encoding="utf-8", errors="ignore")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(lowered_text, encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "converted_ops": converted_ops,
        "total_converted": int(sum(converted_ops.values())),
        "target_basis": ["rz", "h", "x", "cx"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    summary = lower_voqc_qasm_file_to_nam(args.input, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
