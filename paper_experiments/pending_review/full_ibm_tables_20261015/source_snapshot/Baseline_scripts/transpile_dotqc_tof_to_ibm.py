#!/usr/bin/env python3
"""Parse dotQC X/tof skeletons and lower them to IBM {u1,u2,u3,cx} QASM."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import XGate


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402


IBM_BASIS = ("u1", "u2", "u3", "cx")


@dataclass(frozen=True)
class DotQCGate:
    name: str
    target: int
    controls: tuple[int, ...] = ()


def parse_dotqc(path: Path) -> tuple[list[str], list[DotQCGate]]:
    wires: list[str] = []
    gates: list[DotQCGate] = []
    in_body = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(".v "):
            wires = line.split()[1:]
            continue
        if line == "BEGIN":
            in_body = True
            continue
        if line == "END":
            in_body = False
            continue
        if not in_body:
            continue

        parts = line.split()
        if parts[0] == "X" and len(parts) == 2:
            gates.append(DotQCGate("X", int(parts[1][1:])))
        elif parts[0] == "tof" and len(parts) >= 3:
            wire_ids = tuple(int(token[1:]) for token in parts[1:])
            gates.append(DotQCGate("tof", wire_ids[-1], wire_ids[:-1]))
        else:
            raise ValueError(f"unsupported dotQC line in {path}: {line}")
    return wires, gates


def dotqc_to_qiskit(wires: list[str], gates: list[DotQCGate], name: str) -> QuantumCircuit:
    max_wire = -1
    for wire in wires:
        if wire.startswith("q"):
            max_wire = max(max_wire, int(wire[1:]))
    for gate in gates:
        max_wire = max(max_wire, gate.target, *(gate.controls or (-1,)))

    qc = QuantumCircuit(max_wire + 1, name=name)
    for gate in gates:
        if gate.name == "X":
            qc.x(gate.target)
        elif len(gate.controls) == 1:
            qc.cx(gate.controls[0], gate.target)
        else:
            qc.append(XGate().control(len(gate.controls)), list(gate.controls) + [gate.target])
    return qc


def qiskit_metrics(qc: QuantumCircuit) -> dict[str, Any]:
    return {
        "num_qubits": int(qc.num_qubits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "ops": {str(name): int(count) for name, count in qc.count_ops().items()},
    }


def dotqc_stats(wires: list[str], gates: list[DotQCGate]) -> dict[str, Any]:
    hist: dict[str, int] = {}
    for gate in gates:
        key = "X/1" if gate.name == "X" else f"tof/{len(gate.controls) + 1}"
        hist[key] = hist.get(key, 0) + 1
    return {
        "wire_count": int(len(wires)),
        "gate_count": int(len(gates)),
        "gate_histogram": hist,
    }


def write_qasm(qc: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)


def lower_one(path: Path, out_dir: Path, opt_level: int) -> dict[str, Any]:
    wires, gates = parse_dotqc(path)
    logical = dotqc_to_qiskit(wires, gates, path.stem)
    started = time.perf_counter()
    lowered = transpile(logical, basis_gates=list(IBM_BASIS), optimization_level=opt_level)
    elapsed = time.perf_counter() - started

    qasm_path = out_dir / f"{path.stem}.ibm_opt{opt_level}.qasm"
    write_qasm(lowered, qasm_path)
    return {
        "input": str(path),
        "dotqc": dotqc_stats(wires, gates),
        "logical_qiskit": qiskit_metrics(logical),
        "ibm_basis": list(IBM_BASIS),
        "optimization_level": int(opt_level),
        "transpile_time_s": float(elapsed),
        "ibm_qasm": str(qasm_path),
        "ibm_qiskit": qiskit_metrics(lowered),
        "ibm_count": count_qasm_file(qasm_path, gate_set="ibm"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--optimization-level", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "out_dir": str(out_dir),
        "ibm_basis": list(IBM_BASIS),
        "optimization_level": int(args.optimization_level),
        "cases": [lower_one(path, out_dir, args.optimization_level) for path in args.inputs],
    }
    summary_path = out_dir / f"summary_ibm_opt{args.optimization_level}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
