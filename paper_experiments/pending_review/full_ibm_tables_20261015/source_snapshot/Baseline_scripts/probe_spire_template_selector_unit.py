#!/usr/bin/env python3
"""Generate and run the canonical Tower template_selector_unit controlled-PX probe."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "spire_template_selector_unit_v1"
DEFAULT_TOWER = Path("/private/tmp/spire-src-local/spire-src/tower")


def emit_and_tree(lines: list[str], literals: list[str], indent: str) -> str | None:
    if not literals:
        return None
    if len(literals) == 1:
        return literals[0]

    level = 0
    current = list(literals)
    while len(current) > 1:
        nxt: list[str] = []
        pairs = [(idx, current[idx], current[idx + 1]) for idx in range(0, len(current) - 1, 2)]
        has_carry = len(current) % 2 == 1
        final_pair = len(pairs) == 1 and not has_carry
        for pair_idx, lhs, rhs in pairs:
            out = "sel" if final_pair else f"s_{level}_{pair_idx // 2}"
            lines.append(f"{indent}let {out} <- {lhs} && {rhs};")
            nxt.append(out)
        if has_carry:
            nxt.append(current[-1])
        current = nxt
        level += 1
    return current[0]


def template_selector_unit_source(
    *,
    function_name: str,
    ctrl_state: str,
    px_string: str,
) -> str:
    if not ctrl_state:
        raise ValueError("ctrl_state must not be empty")
    if set(ctrl_state) - {"0", "1"}:
        raise ValueError("ctrl_state must contain only 0/1")
    if set(px_string) - {"X", "I"}:
        raise ValueError("px_string must contain only X/I")
    if "X" not in px_string:
        raise ValueError("px_string must contain at least one X target")

    controls = [f"c{i}" for i in range(len(ctrl_state))]
    targets = [f"q{i}" for i in range(len(px_string))]
    params = ", ".join(f"{name}: bool" for name in [*controls, *targets])
    x_targets = [(target, f"n{target}") for target, pauli in zip(targets, px_string, strict=True) if pauli == "X"]

    lines = [
        "/*",
        "  Canonical template_selector_unit controlled-PX probe.",
        "  Computes selector and target complements in a with-do block,",
        "  applies controlled swaps in the body, and returns unit.",
        f"  ctrl_state = {ctrl_state}, P_X = {px_string}",
        "*/",
        f"fun {function_name}({params}) -> unit {{",
        "",
        "  with {",
    ]

    literals: list[str] = []
    for control, bit in zip(controls, ctrl_state, strict=True):
        if bit == "1":
            literals.append(control)
        else:
            neg = f"n{control}"
            lines.append(f"    let {neg} <- not {control};")
            literals.append(neg)

    selector = emit_and_tree(lines, literals, "    ")
    lines.append("")

    for target, complement in x_targets:
        lines.append(f"    let {complement} <- not {target};")

    lines.extend(["  } do {"])
    if selector is None:
        for target, complement in x_targets:
            lines.append(f"    {target} <-> {complement};")
    else:
        lines.append(f"    if {selector} {{")
        for target, complement in x_targets:
            lines.append(f"      {target} <-> {complement};")
        lines.append("    }")
    lines.extend(["  }", "", "  return ();", "}", ""])
    return "\n".join(lines)


def run_tower(tower_bin: Path, source: Path, function_name: str, *, optimize_lir: bool) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(tower_bin),
        "-c",
        "--word_size",
        "1",
        "--mem_size",
        "0",
        "-b",
        f"{function_name}:1",
        "--dotQC",
    ]
    if optimize_lir:
        cmd.append("--optimize_lir")
    cmd.append(str(source))
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def dotqc_stats(text: str) -> dict[str, Any]:
    wires: list[str] = []
    gates: list[str] = []
    in_body = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(".v "):
            wires = line.split()[1:]
        elif line == "BEGIN":
            in_body = True
        elif line == "END":
            in_body = False
        elif in_body:
            gates.append(line)

    hist: dict[str, int] = {}
    for gate in gates:
        parts = gate.split()
        if parts[0] == "X":
            key = "X/1"
        elif parts[0] == "tof":
            key = f"tof/{len(parts) - 1}"
        else:
            key = parts[0]
        hist[key] = hist.get(key, 0) + 1
    return {"wire_count": len(wires), "gate_count": len(gates), "gate_histogram": hist}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tower-bin", type=Path, default=DEFAULT_TOWER)
    parser.add_argument("--ctrl-state", default="1111")
    parser.add_argument("--px-string", default="XIXX")
    parser.add_argument("--function-name", default=None)
    args = parser.parse_args()

    function_name = args.function_name or f"c{len(args.ctrl_state)}_{args.px_string.lower()}_template_selector_unit"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = out_dir / f"{function_name}.twr"
    source_path.write_text(
        template_selector_unit_source(
            function_name=function_name,
            ctrl_state=str(args.ctrl_state),
            px_string=str(args.px_string),
        ),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "template": "template_selector_unit",
        "tower_bin": str(args.tower_bin),
        "source": str(source_path),
        "ctrl_state": str(args.ctrl_state),
        "px_string": str(args.px_string),
        "variants": {},
    }
    for label, optimize in (("orig", False), ("optimized_lir", True)):
        proc = run_tower(Path(args.tower_bin), source_path, function_name, optimize_lir=optimize)
        qc_path = out_dir / f"{function_name}.{label}.qc"
        stderr_path = out_dir / f"{function_name}.{label}.stderr.txt"
        qc_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        summary["variants"][label] = {
            "returncode": int(proc.returncode),
            "qc": str(qc_path),
            "stderr": str(stderr_path),
            "dotqc": dotqc_stats(proc.stdout) if proc.returncode == 0 else None,
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
