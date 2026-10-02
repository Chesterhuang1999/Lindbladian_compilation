#!/usr/bin/env python3
"""Build decay-only Lindbladian higher-order expansion circuits.

The default model is the one-qubit decay/thermalization example used by
``evaluate_test_iii_2_higher_order_series.py``:

    H = 0,
    L_0 = sqrt(lambda * (v + 1)) / scaling * (X + iY),
    L_1 = sqrt(lambda * v) / scaling * (X - iY).

Default expansion parameters are K=2, q=4, K1=4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import qasm2, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
BASELINE_DIR = ROOT / "Baseline_scripts"
for path in (SRC_DIR, SCRIPT_DIR, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from build_periodic_tfim_higher_order_expansion import (  # type: ignore  # noqa: E402
    NAM_BASIS,
    NAM_TRANSPILER_BASIS,
    _float_tag,
    add_selection_state_preparation,
    construct_higher_order_circuit_dynamic,
    dump_qasm2,
    export_nam_count_qasm,
    higher_order_logical_kraus_count,
)
from channel_IR import Lindbladian  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


DEFAULT_OUT_ROOT = ROOT / "circuits" / "Table1"
DEFAULT_SUMMARY_ROOT = ROOT / "Baseline_results" / "decay_higher_order_expansion"
IBM_BASIS = ("u1", "u2", "u3", "cx")


def build_decay_lindbladian(lambda_value: float, v: float, scaling: float) -> Lindbladian:
    h_terms: list[tuple[str, complex]] = []
    l_terms = [
        [
            ("X", np.sqrt(lambda_value * (v + 1)) / scaling),
            ("Y", 1j * np.sqrt(lambda_value * (v + 1)) / scaling),
        ],
        [
            ("X", np.sqrt(lambda_value * v) / scaling),
            ("Y", -1j * np.sqrt(lambda_value * v) / scaling),
        ],
    ]
    return Lindbladian(h_terms, l_terms)


def build_decay_higher_order_circuit(args: argparse.Namespace) -> tuple[QuantumCircuit, dict[str, Any], float]:
    lind = build_decay_lindbladian(args.lambda_value, args.v, args.scaling)
    started = time.perf_counter()
    select_qc, reg_sizes, coeff_sum_sq, sel_state, expansion_metadata = construct_higher_order_circuit_dynamic(
        lind,
        args.k_order,
        args.sampling_points,
        args.total_time,
        args.coherent_order,
        coherent_opt=args.coherent_opt,
        jump_opt=args.jump_opt,
        structure=args.structure,
    )
    qc, qubit_indexes = add_selection_state_preparation(
        select_qc,
        np.asarray(sel_state, dtype=complex),
        list(reg_sizes),
        structure=args.structure,
    )
    elapsed = time.perf_counter() - started

    model_info = {
        "model": "decay_only_lindbladian_higher_order_expansion",
        "num_qubits": 1,
        "lambda_value": float(args.lambda_value),
        "v": float(args.v),
        "scaling": float(args.scaling),
        "total_time": float(args.total_time),
        "duhamel_order_K": int(args.k_order),
        "sampling_points_q": int(args.sampling_points),
        "coherent_taylor_order_K1": int(args.coherent_order),
        "num_jump_operators": int(len(lind.L_list)),
        "logical_kraus_count": int(
            higher_order_logical_kraus_count(
                len(lind.L_list),
                args.k_order,
                args.sampling_points,
            )
        ),
        "selection_state_nonzero": int(np.count_nonzero(np.abs(sel_state) > args.tol)),
        "coeff_sum_square": float(coeff_sum_sq),
        "select_register_state_norm": float(np.linalg.norm(sel_state)),
        "structure": args.structure,
        "coherent_opt": args.coherent_opt,
        "jump_opt": args.jump_opt,
        "reg_sizes_from_select": [int(x) for x in reg_sizes],
        "qubit_indexes": qubit_indexes,
        "expansion_metadata": expansion_metadata,
    }
    return qc, model_info, elapsed


def export_ibm_count_qasm(
    qc: QuantumCircuit,
    path: Path,
    *,
    optimization_level: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    tqc = transpile(
        qc,
        basis_gates=list(IBM_BASIS),
        optimization_level=optimization_level,
    )
    elapsed = time.perf_counter() - started
    export_openqasm2_quartz_input(tqc, path, basis_gates=IBM_BASIS)
    return count_qasm_file(path, gate_set="ibm"), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate decay-only higher-order expansion OpenQASM2 circuit."
    )
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--scaling", type=float, default=4.01)
    parser.add_argument("--total-time", type=float, default=0.1)
    parser.add_argument("--k-order", type=int, default=2)
    parser.add_argument("--sampling-points", type=int, default=4)
    parser.add_argument("--coherent-order", type=int, default=4)
    parser.add_argument("--coherent-opt", choices=("No", "Matrix-order"), default="No")
    parser.add_argument("--jump-opt", choices=("No", "Matrix-order"), default="No")
    parser.add_argument("--structure", choices=("basic", "optimized"), default="basic")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--nam-optimization-level", type=int, default=0)
    parser.add_argument("--ibm-optimization-level", type=int, default=0)
    parser.add_argument("--skip-qasm", action="store_true")
    parser.add_argument("--skip-nam", action="store_true")
    parser.add_argument("--skip-ibm", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc, model_info, build_time = build_decay_higher_order_circuit(args)

    name = (
        f"decay_only_lambda{_float_tag(args.lambda_value)}_v{_float_tag(args.v)}"
        f"_scale{_float_tag(args.scaling)}_t{_float_tag(args.total_time)}"
        f"_higher_order_K{args.k_order}_q{args.sampling_points}_K1{args.coherent_order}"
    )
    if args.structure != "basic":
        name += f"_{args.structure}"
    if args.coherent_opt != "No":
        name += f"_coh_{args.coherent_opt.lower()}"
    if args.jump_opt != "No":
        name += f"_jump_{args.jump_opt.lower()}"

    out_dir = Path(args.out_root) / name
    summary_dir = Path(args.summary_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_qasm = out_dir / f"{name}.qasm"
    nam_qasm = out_dir / f"{name}_nam_opt{args.nam_optimization_level}.qasm"
    ibm_qasm = out_dir / f"{name}_ibm_opt{args.ibm_optimization_level}.qasm"
    if not args.skip_qasm:
        dump_qasm2(qc, raw_qasm)
    if args.skip_nam:
        nam_stats = None
        nam_time = None
    else:
        if args.skip_qasm:
            raise ValueError("--skip-nam is required when --skip-qasm is used.")
        nam_stats, nam_time = export_nam_count_qasm(
            qc,
            nam_qasm,
            optimization_level=args.nam_optimization_level,
        )
    if args.skip_ibm:
        ibm_stats = None
        ibm_time = None
    else:
        ibm_stats, ibm_time = export_ibm_count_qasm(
            qc,
            ibm_qasm,
            optimization_level=args.ibm_optimization_level,
        )

    existing_summary: dict[str, Any] = {}
    summary_path = summary_dir / f"{name}_summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as existing_file:
            existing_summary = json.load(existing_file)

    restored_raw_qasm = existing_summary.get("raw_qasm") or (str(raw_qasm) if raw_qasm.exists() else None)
    restored_nam_qasm = existing_summary.get("nam_qasm") or (str(nam_qasm) if nam_qasm.exists() else None)
    restored_ibm_qasm = existing_summary.get("ibm_qasm") or (str(ibm_qasm) if ibm_qasm.exists() else None)
    restored_nam_stats = existing_summary.get("nam_stats")
    if restored_nam_stats is None and args.skip_nam and nam_qasm.exists():
        restored_nam_stats = count_qasm_file(nam_qasm, gate_set="nam")
    restored_ibm_stats = existing_summary.get("ibm_stats")
    if restored_ibm_stats is None and args.skip_ibm and ibm_qasm.exists():
        restored_ibm_stats = count_qasm_file(ibm_qasm, gate_set="ibm")

    summary = {
        "experiment": "decay_higher_order_expansion",
        "name": name,
        "model": model_info,
        "raw_qasm": restored_raw_qasm if args.skip_qasm else str(raw_qasm),
        "nam_qasm": restored_nam_qasm if args.skip_nam else str(nam_qasm),
        "ibm_qasm": restored_ibm_qasm if args.skip_ibm else str(ibm_qasm),
        "build_time_s": float(build_time),
        "nam_transpile_time_s": existing_summary.get("nam_transpile_time_s")
        if nam_time is None
        else float(nam_time),
        "ibm_transpile_time_s": existing_summary.get("ibm_transpile_time_s")
        if ibm_time is None
        else float(ibm_time),
        "nam_basis_counted": list(NAM_BASIS),
        "nam_transpiler_basis": list(NAM_TRANSPILER_BASIS),
        "nam_optimization_level": int(args.nam_optimization_level),
        "ibm_basis_counted": list(IBM_BASIS),
        "ibm_transpiler_basis": list(IBM_BASIS),
        "ibm_optimization_level": int(args.ibm_optimization_level),
        "raw_circuit": {
            "num_qubits": int(qc.num_qubits),
            "depth": int(qc.depth()),
            "size": int(qc.size()),
            "ops": {str(gate): int(count) for gate, count in qc.count_ops().items()},
        },
        "nam_stats": restored_nam_stats if nam_stats is None else nam_stats,
        "ibm_stats": restored_ibm_stats if ibm_stats is None else ibm_stats,
    }

    with summary_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, default=str)

    nam_stats_for_print = summary.get("nam_stats")
    ibm_stats_for_print = summary.get("ibm_stats")
    nam_total = "-" if nam_stats_for_print is None else nam_stats_for_print["metric_total"]
    non_clifford = "-" if nam_stats_for_print is None else nam_stats_for_print["clifford"]["non_clifford"]
    ibm_total = "-" if ibm_stats_for_print is None else ibm_stats_for_print["metric_total"]
    nam_time_cell = "-" if nam_time is None else f"{nam_time:.3f}"
    ibm_time_cell = "-" if ibm_time is None else f"{ibm_time:.3f}"
    print("| Circuit | Qubits | Logical Kraus | Raw ops | NAM total | IBM total | Non-Clifford | Build time (s) | NAM transpile (s) | IBM transpile (s) | Summary |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    print(
        f"| {name} | {qc.num_qubits} | {model_info['logical_kraus_count']} | "
        f"{qc.size()} | {nam_total} | {ibm_total} | {non_clifford} | {build_time:.3f} | "
        f"{nam_time_cell} | {ibm_time_cell} | {summary_path} |"
    )


if __name__ == "__main__":
    main()
