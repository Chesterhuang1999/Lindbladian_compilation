#!/usr/bin/env python3
"""Prepare basis-level TFIM channel-LCU QASM inputs for external baselines."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


IBM_BASIS = ("u1", "u2", "u3", "cx")
NAM_BASIS = ("h", "x", "rz", "cx")
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


def source_dir(n: int) -> Path:
    return ROOT / "Baseline_results" / f"tfim{n}_channel_lcu_ours_nam"


def target_dir(n: int) -> Path:
    return ROOT / "circuits" / "Table1" / f"tfim_n{n}_channel_lcu"


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


def prepare_one(n: int) -> dict[str, Any]:
    src_dir = source_dir(n)
    raw_qasm = src_dir / f"tfim_n{n}_channel_lcu_basic_raw.qasm"
    raw_nam_qasm = src_dir / f"tfim_n{n}_channel_lcu_basic_raw_nam.qasm"
    if not raw_qasm.exists():
        raise FileNotFoundError(f"Missing raw high-level QASM: {raw_qasm}")
    if not raw_nam_qasm.exists():
        raise FileNotFoundError(f"Missing raw NAM QASM: {raw_nam_qasm}")

    out_dir = target_dir(n)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_nam = out_dir / f"tfim_n{n}_channel_lcu_basic_raw_nam_opt0.qasm"
    canonical_ibm = out_dir / f"tfim_n{n}_channel_lcu_basic_raw_ibm_opt0.qasm"

    shutil.copy2(raw_nam_qasm, canonical_nam)
    started = time.perf_counter()
    raw_qc, register_renames = load_qasm2_with_safe_registers(raw_qasm)
    ibm_qc = transpile(raw_qc, basis_gates=list(IBM_BASIS), optimization_level=0)
    ibm_transpile_time = time.perf_counter() - started
    ibm_export = export_openqasm2_quartz_input(
        ibm_qc,
        canonical_ibm,
        basis_gates=IBM_BASIS,
    )

    nam_stats = count_qasm_file(canonical_nam, gate_set="nam")
    ibm_stats = count_qasm_file(canonical_ibm, gate_set="ibm")
    return {
        "num_qubits": int(n),
        "source_raw_qasm": str(raw_qasm),
        "source_nam_qasm": str(raw_nam_qasm),
        "canonical_nam_qasm": str(canonical_nam),
        "canonical_ibm_qasm": str(canonical_ibm),
        "qasm_register_renames": register_renames,
        "ibm_transpile_time_s": float(ibm_transpile_time),
        "ibm_export": ibm_export,
        "nam_stats": nam_stats,
        "ibm_stats": ibm_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare canonical NAM/IBM opt0 QASM inputs for TFIM external baselines."
    )
    parser.add_argument("--num-qubits", nargs="+", type=int, default=[8, 12])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for n in args.num_qubits:
        print(f"[prepare-tfim] n={n}", flush=True)
        result = prepare_one(n)
        results.append(result)
        print(
            f"[prepare-tfim] n={n}: NAM={result['nam_stats']['metric_total']:,}, "
            f"IBM={result['ibm_stats']['metric_total']:,}, "
            f"IBM transpile={result['ibm_transpile_time_s']:.3f}s",
            flush=True,
        )

    summary_path = ROOT / "Baseline_results" / "tfim_external_baseline_inputs_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
