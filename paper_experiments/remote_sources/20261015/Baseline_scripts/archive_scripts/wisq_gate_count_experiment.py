from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "circuits" / "select_only"
DEFAULT_OUTPUT = ROOT / "Baseline_results" / "wisq" / "test_table1_wisq_gate_counts_10min.json"
PHYSICAL_GATES = {
    "ibm": ("u1", "u2", "u3", "cx"),
    "nam": ("h", "x", "rz", "cx"),
}
WISQ_GATESETS = {
    "ibm": "IBMO",
    "nam": "NAM",
}


def infer_gateset(path: Path) -> str:
    name = path.name.lower()
    if "_ibm_" in name:
        return "ibm"
    if "_nam_" in name:
        return "nam"
    raise ValueError(f"Cannot infer WISQ gate set from input filename: {path}")


def gate_stats(path: Path, gateset: str) -> dict[str, Any]:
    circuit = QuantumCircuit.from_qasm_file(str(path))
    ops = {str(gate): int(count) for gate, count in circuit.count_ops().items()}
    physical_gates = PHYSICAL_GATES[gateset]
    return {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "physical_gate_count": int(sum(ops.get(gate, 0) for gate in physical_gates)),
        "ops": {gate: int(ops.get(gate, 0)) for gate in physical_gates},
        "all_ops": ops,
    }


def run_wisq(input_path: Path, output_path: Path, gateset: str, timeout: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="wisq_gate_count_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_input = tmp_dir_path / input_path.name
        tmp_output = tmp_dir_path / output_path.name
        shutil.copy2(input_path, tmp_input)

        command = [
            "wisq",
            str(tmp_input),
            "--mode",
            "opt",
            "--target_gateset",
            WISQ_GATESETS[gateset],
            "--optimization_objective",
            "TOTAL",
            "--opt_timeout",
            str(timeout),
            "--output_path",
            str(tmp_output),
        ]
        start = time.time()
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.time() - start

        if completed.returncode != 0:
            return {
                "returncode": int(completed.returncode),
                "elapsed_sec": elapsed,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "output_written": False,
            }
        if not tmp_output.exists():
            return {
                "returncode": int(completed.returncode),
                "elapsed_sec": elapsed,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "output_written": False,
                "error": "WISQ completed without writing the output QASM.",
            }

        return {
            "returncode": int(completed.returncode),
            "elapsed_sec": elapsed,
            "output_written": True,
            "gate_counts": gate_stats(tmp_output, gateset),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir).resolve()
    input_paths = sorted(input_dir.glob("*_qasm2.qasm"))
    if not input_paths:
        raise FileNotFoundError(f"No QASM2 inputs found in {input_dir}")

    results = []
    for input_path in input_paths:
        gateset = infer_gateset(input_path)
        before = gate_stats(input_path, gateset)
        wisq_result = run_wisq(
            input_path=input_path,
            output_path=Path(f"{input_path.stem}_wisq_optimized.qasm"),
            gateset=gateset,
            timeout=int(args.timeout),
        )
        after = wisq_result.get("gate_counts")
        comparison = None
        if isinstance(after, dict):
            comparison = {
                "physical_gate_count_delta": int(
                    after["physical_gate_count"] - before["physical_gate_count"]
                ),
                "size_delta": int(after["size"] - before["size"]),
                "depth_delta": int(after["depth"] - before["depth"]),
            }
        results.append(
            {
                "input": str(input_path.relative_to(ROOT)),
                "gateset": gateset,
                "wisq_target_gateset": WISQ_GATESETS[gateset],
                "optimization_objective": "TOTAL",
                "timeout_sec": int(args.timeout),
                "before": before,
                "after": after,
                "comparison": comparison,
                "wisq": {
                    key: value
                    for key, value in wisq_result.items()
                    if key != "gate_counts"
                },
            }
        )

    summary = {
        "tool": "wisq",
        "mode": "opt",
        "inputs": [str(path.relative_to(ROOT)) for path in input_paths],
        "results": results,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WISQ on SELECT-only QASM2 inputs and save gate-count-only results."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout", type=int, default=600)
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
