#!/usr/bin/env python3
"""Build periodic TFIM Lindbladian higher-order expansion circuits.

Default parameters match the requested setting:

    K = 2, q = 4, K1 = 4

Here K is the Duhamel expansion order, q is the number of sampling points, and
K1 is the Taylor truncation order for the coherent J term.
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
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import Lindbladian, Matrixsum, matsum_mul  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402
from series_expansion import (  # type: ignore  # noqa: E402
    create_index_set,
    quadrature_points_N_weights,
)
from subroutine import mulplex_U_opt  # type: ignore  # noqa: E402


NAM_BASIS = ("h", "x", "rz", "cx")
NAM_TRANSPILER_BASIS = ("h", "x", "rz", "cx", "reset")
DEFAULT_OUT_ROOT = ROOT / "circuits" / "Table1"
DEFAULT_SUMMARY_ROOT = ROOT / "Baseline_results" / "periodic_tfim_higher_order_expansion"


def _float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_periodic_tfim_lindbladian(num_qubits: int, gamma: float) -> Lindbladian:
    h_terms: list[tuple[str, complex]] = []
    l_terms: list[list[tuple[str, complex]]] = []

    for i in range(num_qubits):
        z_neighbors = [i, (i + 1) % num_qubits]
        zz_str = "".join("Z" if j in z_neighbors else "I" for j in range(num_qubits))
        x_str = "".join("X" if j == i else "I" for j in range(num_qubits))
        y_str = "".join("Y" if j == i else "I" for j in range(num_qubits))
        h_terms.append((zz_str, -1.0))
        h_terms.append((x_str, -1.0))
        l_terms.append([(x_str, gamma), (y_str, -1j * gamma)])

    return Lindbladian(h_terms, l_terms)


def higher_order_logical_kraus_count(num_jumps: int, k_order: int, sampling_points: int) -> int:
    return 1 + sum(len(create_index_set(k, sampling_points, num_jumps)) for k in range(1, k_order + 1))


def construct_circuit_coherent_dynamic(
    generator: Matrixsum,
    coherent_order: int,
    interval: float,
    *,
    opt: str,
) -> tuple[QuantumCircuit, float]:
    """Construct the Taylor LCU block for exp(generator * interval).

    This mirrors ``series_expansion.construct_circuit_coherent`` but uses the
    newer placement-aware block encoding and returns the actual block width for
    each interval.
    """
    sum_of_generator = generator.identity(generator.size)
    product = None
    for order in range(1, coherent_order + 1):
        if order == 1:
            product = copy_matrixsum(generator)
            product.mul_coeffs(interval)
        else:
            assert product is not None
            product = matsum_mul(product, generator)
            product.mul_coeffs(interval / order)
        sum_of_generator = sum_of_generator.add(product)

    qc = NewBlockEncoding(sum_of_generator).circuit(opt=opt)
    return qc, sum_of_generator.pauli_norm()


def copy_matrixsum(matrixsum: Matrixsum) -> Matrixsum:
    # Avoid importing deepcopy into the global namespace only for this small use.
    import copy

    return copy.deepcopy(matrixsum)


def build_single_duhamel_kraus_circuit(
    k_order: int,
    index: list[int],
    jump_ops: list[Matrixsum],
    coherent_order: int,
    total_time: float,
    sampling_points: int,
    sys_size: int,
    effective_generator: Matrixsum,
    *,
    coherent_opt: str,
    jump_opt: str,
    coherent_cache: dict[tuple[str, str, int], tuple[QuantumCircuit, float]],
    jump_cache: dict[tuple[int, str], QuantumCircuit],
) -> tuple[QuantumCircuit, float, int]:
    l_list = index[1 : k_order + 1]
    j_list = index[k_order + 1 : 2 * k_order + 1]

    points, weights = quadrature_points_N_weights(k_order, sampling_points, total_time, j_list)
    points.append(0)

    coeff_sum = float(np.sqrt(np.prod(weights)))
    blocks: list[QuantumCircuit] = []

    for segment in range(k_order + 1):
        if segment == k_order:
            interval = total_time - points[0]
        else:
            interval = points[k_order - segment - 1] - points[k_order - segment]

        interval_key = f"{float(interval):.17g}"
        coherent_key = (interval_key, coherent_opt, coherent_order)
        if coherent_key not in coherent_cache:
            coherent_cache[coherent_key] = construct_circuit_coherent_dynamic(
                effective_generator,
                coherent_order,
                float(interval),
                opt=coherent_opt,
            )
        coherent_block, coherent_coeff = coherent_cache[coherent_key]
        blocks.append(coherent_block)
        coeff_sum *= float(coherent_coeff)

        if segment < k_order:
            jump = jump_ops[l_list[k_order - segment - 1]]
            coeff_sum *= float(jump.pauli_norm())
            jump_index = l_list[k_order - segment - 1]
            jump_key = (jump_index, jump_opt)
            if jump_key not in jump_cache:
                jump_cache[jump_key] = NewBlockEncoding(jump).circuit(opt=jump_opt)
            blocks.append(jump_cache[jump_key])

    ctrl_widths = [block.num_qubits - sys_size for block in blocks]
    total_ctrl_size = int(sum(ctrl_widths))
    ctrl = QuantumRegister(total_ctrl_size, "ctrl")
    sys = QuantumRegister(sys_size, "sys")
    qc = QuantumCircuit(ctrl, sys)

    ctrl_offset = 0
    for block, ctrl_width in zip(blocks, ctrl_widths):
        qargs = list(ctrl[ctrl_offset : ctrl_offset + ctrl_width]) + list(sys)
        qc.compose(block, qubits=qargs, inplace=True)
        ctrl_offset += ctrl_width

    return qc, coeff_sum, total_ctrl_size


def construct_higher_order_circuit_dynamic(
    lind: Lindbladian,
    k_order: int,
    sampling_points: int,
    total_time: float,
    coherent_order: int,
    *,
    coherent_opt: str,
    jump_opt: str,
    structure: str,
) -> tuple[QuantumCircuit, list[int], float, np.ndarray, dict[str, Any]]:
    effective_generator = lind.effective_H()
    sys_size = effective_generator.size
    num_jumps = len(lind.L_list)

    coherent_qc, coherent_coeff = construct_circuit_coherent_dynamic(
        effective_generator,
        coherent_order,
        total_time,
        opt=coherent_opt,
    )
    circuits = [coherent_qc]
    coeff_sums = [float(coherent_coeff)]
    ctrl_sizes = [int(coherent_qc.num_qubits - sys_size)]
    coherent_cache: dict[tuple[str, str, int], tuple[QuantumCircuit, float]] = {
        (f"{float(total_time):.17g}", coherent_opt, coherent_order): (coherent_qc, coherent_coeff),
    }
    jump_cache: dict[tuple[int, str], QuantumCircuit] = {}

    order_counts: dict[str, int] = {"0": 1}
    for order in range(1, k_order + 1):
        index_set = create_index_set(order, sampling_points, num_jumps)
        order_counts[str(order)] = len(index_set)
        for index in index_set:
            qc_elem, coeff_sum, ctrl_size = build_single_duhamel_kraus_circuit(
                order,
                index,
                lind.L_list,
                coherent_order,
                total_time,
                sampling_points,
                sys_size,
                effective_generator,
                coherent_opt=coherent_opt,
                jump_opt=jump_opt,
                coherent_cache=coherent_cache,
                jump_cache=jump_cache,
            )
            circuits.append(qc_elem)
            coeff_sums.append(float(coeff_sum))
            ctrl_sizes.append(int(ctrl_size))

    kraus_count = len(circuits)
    sel_size = int(np.ceil(np.log2(kraus_count)))
    ctrl_size_max = int(max(ctrl_sizes, default=0))
    coeff_sum_square = float(sum(abs(c) ** 2 for c in coeff_sums))
    sel_state = np.zeros(2**sel_size, dtype=complex)
    for i, coeff in enumerate(coeff_sums):
        sel_state[i] = np.sqrt(abs(coeff) ** 2 / coeff_sum_square)

    if structure == "basic":
        select = QuantumRegister(sel_size, "sel")
        ctrl = QuantumRegister(ctrl_size_max, "ctrl")
        sys = QuantumRegister(sys_size, "sys")
        qc = QuantumCircuit(select, ctrl, sys)
        for i, subcircuit in enumerate(circuits):
            ctrl_size = ctrl_sizes[i]
            ctrl_state = bin(i)[2:].zfill(sel_size)
            controlled = subcircuit.to_gate().control(num_ctrl_qubits=sel_size, ctrl_state=ctrl_state)
            qc.append(controlled, qargs=list(select) + list(ctrl[:ctrl_size]) + list(sys))
        reg_sizes = [sel_size, ctrl_size_max, sys_size]
    elif structure == "optimized":
        qc = QuantumCircuit(2 * sel_size + ctrl_size_max + sys_size)
        qc, _tcount, _mccount, _cxcount = mulplex_U_opt(
            qc,
            circuits,
            sel_size,
            ctrl_size_max,
            sys_size,
        )
        reg_sizes = [sel_size, sel_size, ctrl_size_max, sys_size]
    else:
        raise ValueError(f"Unsupported structure: {structure}")

    metadata = {
        "order_counts": order_counts,
        "ctrl_sizes": ctrl_sizes,
        "ctrl_size_max": ctrl_size_max,
        "coeff_sums": coeff_sums,
        "coherent_cache_size": len(coherent_cache),
        "jump_cache_size": len(jump_cache),
    }
    return qc, reg_sizes, coeff_sum_square, sel_state, metadata


def add_selection_state_preparation(
    select_qc: QuantumCircuit,
    sel_state: np.ndarray,
    reg_sizes: list[int],
    *,
    structure: str,
) -> tuple[QuantumCircuit, list[list[int]]]:
    if structure == "basic":
        sel_size, ctrl_size, sys_size = reg_sizes
        qc = QuantumCircuit(sel_size + ctrl_size + sys_size)
        select_indexes = list(range(sel_size))
        ctrl_indexes = list(range(sel_size, sel_size + ctrl_size))
        sys_indexes = list(range(sel_size + ctrl_size, sel_size + ctrl_size + sys_size))
        qc.append(StatePreparation(Statevector(sel_state)), select_indexes)
        qc.compose(select_qc, qubits=qc.qubits, inplace=True)
        return qc, [select_indexes, ctrl_indexes, sys_indexes]

    if structure == "optimized":
        sel_size, anc_size, ctrl_size, sys_size = reg_sizes
        qc = QuantumCircuit(2 * sel_size + ctrl_size + sys_size)
        select_indexes = [2 * j for j in range(sel_size)]
        anc_indexes = [2 * j + 1 for j in range(sel_size)]
        ctrl_indexes = list(range(2 * sel_size, 2 * sel_size + ctrl_size))
        sys_indexes = list(range(2 * sel_size + ctrl_size, 2 * sel_size + ctrl_size + sys_size))
        qc.append(StatePreparation(Statevector(sel_state)), select_indexes)
        qc.compose(select_qc, qubits=qc.qubits, inplace=True)
        return qc, [select_indexes, anc_indexes, ctrl_indexes, sys_indexes]

    raise ValueError(f"Unsupported structure: {structure}")


def build_higher_order_circuit(args: argparse.Namespace) -> tuple[QuantumCircuit, dict[str, Any], float]:
    gamma = args.gamma if args.gamma is not None else np.sqrt(0.1) / 2
    lind = build_periodic_tfim_lindbladian(args.num_qubits, gamma)

    started = time.perf_counter()
    select_qc, reg_sizes, coeff_sum_sq, sel_state, expansion_metadata = construct_higher_order_circuit_dynamic(
        lind,
        args.k_order,
        args.sampling_points,
        args.delta_t,
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
        "model": "periodic_tfim_lindbladian_higher_order_expansion",
        "num_qubits": int(args.num_qubits),
        "gamma": float(gamma),
        "delta_t": float(args.delta_t),
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


def dump_qasm2(qc: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        qasm2.dump(qc, out_file)


def export_nam_count_qasm(
    qc: QuantumCircuit,
    path: Path,
    *,
    optimization_level: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    tqc = transpile(
        qc,
        basis_gates=list(NAM_TRANSPILER_BASIS),
        optimization_level=optimization_level,
    )
    elapsed = time.perf_counter() - started
    export_openqasm2_quartz_input(tqc, path, basis_gates=NAM_TRANSPILER_BASIS)
    return count_qasm_file(path, gate_set="nam"), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate periodic TFIM higher-order expansion OpenQASM2 circuit."
    )
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--k-order", type=int, default=2)
    parser.add_argument("--sampling-points", type=int, default=4)
    parser.add_argument("--coherent-order", type=int, default=4)
    parser.add_argument("--coherent-opt", choices=("No", "Matrix-order"), default="No")
    parser.add_argument("--jump-opt", choices=("No", "Matrix-order"), default="No")
    parser.add_argument("--structure", choices=("basic", "optimized"), default="basic")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--nam-optimization-level", type=int, default=0)
    parser.add_argument("--skip-nam", action="store_true")
    parser.add_argument("--skip-qasm", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc, model_info, build_time = build_higher_order_circuit(args)

    name = (
        f"periodic_tfim_n{args.num_qubits}_dt{_float_tag(args.delta_t)}"
        f"_higher_order_K{args.k_order}_q{args.sampling_points}_K1{args.coherent_order}"
    )
    if args.structure != "basic":
        name += f"_{args.structure}"
    if args.coherent_opt != "Matrix-order":
        name += f"_coh_{args.coherent_opt.lower()}"
    if args.jump_opt != "No":
        name += f"_jump_{args.jump_opt.lower()}"

    out_dir = Path(args.out_root) / name
    summary_dir = Path(args.summary_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_qasm = out_dir / f"{name}.qasm"
    nam_qasm = out_dir / f"{name}_nam_opt{args.nam_optimization_level}.qasm"
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

    summary = {
        "experiment": "periodic_tfim_higher_order_expansion",
        "name": name,
        "model": model_info,
        "raw_qasm": None if args.skip_qasm else str(raw_qasm),
        "nam_qasm": None if args.skip_nam else str(nam_qasm),
        "build_time_s": float(build_time),
        "nam_transpile_time_s": None if nam_time is None else float(nam_time),
        "nam_basis_counted": list(NAM_BASIS),
        "nam_transpiler_basis": list(NAM_TRANSPILER_BASIS),
        "nam_optimization_level": int(args.nam_optimization_level),
        "raw_circuit": {
            "num_qubits": int(qc.num_qubits),
            "depth": int(qc.depth()),
            "size": int(qc.size()),
            "ops": {str(gate): int(count) for gate, count in qc.count_ops().items()},
        },
        "nam_stats": nam_stats,
    }

    summary_path = summary_dir / f"{name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, default=str)

    print("| Circuit | Qubits | Logical Kraus | Raw ops | NAM total | Non-Clifford | Build time (s) | NAM transpile (s) | Summary |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    nam_total = "-" if nam_stats is None else nam_stats["nam"]["rz_clifford_total"]
    non_clifford = "-" if nam_stats is None else nam_stats["clifford"]["non_clifford"]
    nam_time_cell = "-" if nam_time is None else f"{nam_time:.3f}"
    print(
        f"| {name} | {qc.num_qubits} | {model_info['logical_kraus_count']} | "
        f"{qc.size()} | {nam_total} | {non_clifford} | {build_time:.3f} | "
        f"{nam_time_cell} | {summary_path} |"
    )


if __name__ == "__main__":
    main()
