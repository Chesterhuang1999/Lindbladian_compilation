#!/usr/bin/env python3
"""Generate canonical IBM-basis Toom L2 input for pytket/BQSKit baselines."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile


ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


IBM_BASIS = ("u1", "u2", "u3", "cx")
NAM_INPUT = (
    ROOT
    / "circuits"
    / "Table1"
    / "toom_2d_periodic_L2_dt0p1"
    / "toom_2d_periodic_L2_dt0p1_basic_raw_nam_opt0.qasm"
)
OUT_DIR = ROOT / "Baseline_results" / "eval_redesign" / "toom_l2_ibm_pytket_bqskit_agent"
IBM_QASM = OUT_DIR / "ibm_input" / "toom_2d_periodic_L2_dt0p1_basic_raw_ibm_opt0.qasm"
SUMMARY_JSON = OUT_DIR / "ibm_input" / "toom_2d_periodic_L2_dt0p1_basic_raw_ibm_opt0_stats.json"
NOTES_MD = OUT_DIR / "notes.md"
PROGRESS_MD = OUT_DIR / "progress.md"

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


def load_qasm2_with_safe_registers(input_qasm: Path) -> tuple[QuantumCircuit, dict[str, str]]:
    text = input_qasm.read_text(encoding="utf-8")
    renames = qasm2_register_renames(text)
    for old_name, new_name in renames.items():
        text = re.sub(rf"\b{re.escape(old_name)}(?=\[)", new_name, text)
    return QuantumCircuit.from_qasm_str(text), renames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IBM_QASM.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    qc, register_renames = load_qasm2_with_safe_registers(NAM_INPUT)
    input_stats = count_qasm_file(NAM_INPUT, gate_set="nam")
    transpile_started = time.perf_counter()
    ibm_qc = transpile(qc, basis_gates=list(IBM_BASIS), optimization_level=0)
    transpile_time = time.perf_counter() - transpile_started
    export = export_openqasm2_quartz_input(ibm_qc, IBM_QASM, basis_gates=IBM_BASIS)
    ibm_stats = count_qasm_file(IBM_QASM, gate_set="ibm")
    total_time = time.perf_counter() - started

    summary = {
        "benchmark_name": "toom_2d_periodic_L2_dt0p1",
        "source_nam_qasm": str(NAM_INPUT),
        "ibm_qasm": str(IBM_QASM),
        "basis_gates": list(IBM_BASIS),
        "qasm_register_renames": register_renames,
        "transpile_optimization_level": 0,
        "transpile_time_sec": transpile_time,
        "total_time_sec": total_time,
        "input_nam_stats": input_stats,
        "ibm_export": export,
        "ibm_stats": ibm_stats,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    raw_ibm = ibm_stats["metric_total"]
    notes = [
        "# Toom L2 IBM-native Baseline Notes",
        "",
        "Current step: IBM basis input generated; pytket/BQSKit are not complete in this directory yet.",
        "",
        "## IBM Input",
        "",
        f"- Source NAM QASM: `{NAM_INPUT}`",
        f"- IBM basis QASM: `{IBM_QASM}`",
        "- Basis: `{u1,u2,u3,cx}`",
        "- Qiskit transpile optimization level: `0`",
        f"- Raw NAM metric gates: `{input_stats['metric_total']}`",
        f"- Raw IBM metric gates: `{raw_ibm}`",
        f"- Raw IBM width: `{ibm_stats['num_qubits']}`",
        f"- Raw IBM CX: `{ibm_stats['ibm']['cx']}`",
        f"- Raw IBM non-Clifford count: `{ibm_stats['clifford']['non_clifford']}`",
        f"- IBM transpile time sec: `{transpile_time:.6f}`",
        "",
        "## Tool Status",
        "",
        "- pytket: pending. Use `run_pytket_baseline.py` directly; it has no `--timeout` argument. Suggested options: `--strategy best-of --best-of-objective metric_total`.",
        "- BQSKit: pending. Use `--timeout 3600 --optimization-level 2 --workflow-mode skip-retarget-mapping`.",
        "",
        "The generated IBM QASM is the canonical input for both IBM-native generic optimizer runs.",
    ]
    NOTES_MD.write_text("\n".join(notes) + "\n", encoding="utf-8")

    progress = [
        "# Toom L2 IBM-native Baseline Progress",
        "",
        "Status as of 2026-07-03:",
        "",
        "- Current step: IBM basis QASM generated and counted.",
        "- pytket: pending, not yet complete in this directory.",
        "- BQSKit: pending, not yet complete in this directory.",
        f"- Canonical IBM input: `{IBM_QASM}`",
        f"- Raw IBM metric gates: `{raw_ibm}`",
        "",
        "Next action: run pytket on the canonical IBM QASM, then BQSKit on the same canonical IBM QASM.",
    ]
    PROGRESS_MD.write_text("\n".join(progress) + "\n", encoding="utf-8")
    print(json.dumps({"ibm_qasm": str(IBM_QASM), "raw_ibm_metric_gates": raw_ibm}, sort_keys=True))


if __name__ == "__main__":
    main()
