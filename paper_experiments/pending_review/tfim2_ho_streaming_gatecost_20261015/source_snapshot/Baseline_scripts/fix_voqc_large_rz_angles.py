#!/usr/bin/env python3
"""Fold huge VOQC RZ angle fractions in lowered NAM QASM."""

from __future__ import annotations

import argparse
import json
import re
from math import gcd
from pathlib import Path


RZ_PI_FRAC_RE = re.compile(
    r"rz\(\((?P<num>[-+]?\d+)\)\*pi/\((?P<den>[-+]?\d+)\)\)"
)


def fold_fraction(num: int, den: int) -> tuple[int, int]:
    if den == 0:
        return num, den
    if den < 0:
        num = -num
        den = -den
    period = 2 * den
    num = ((num + den) % period) - den
    divisor = gcd(abs(num), den)
    if divisor:
        num //= divisor
        den //= divisor
    return num, den


def fix_text(text: str) -> tuple[str, dict[str, int]]:
    stats = {
        "rz_fraction_seen": 0,
        "rz_fraction_changed": 0,
        "rz_fraction_huge_seen": 0,
    }

    def repl(match: re.Match[str]) -> str:
        stats["rz_fraction_seen"] += 1
        old_num = match.group("num")
        old_den = match.group("den")
        if len(old_num.lstrip("+-")) > 18 or len(old_den.lstrip("+-")) > 18:
            stats["rz_fraction_huge_seen"] += 1
        num, den = fold_fraction(int(old_num), int(old_den))
        if str(num) != old_num or str(den) != old_den:
            stats["rz_fraction_changed"] += 1
        return f"rz(({num})*pi/({den}))"

    return RZ_PI_FRAC_RE.sub(repl, text), stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    fixed, stats = fix_text(args.input.read_text(encoding="utf-8", errors="ignore"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fixed, encoding="utf-8")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "stats": stats,
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
