#!/usr/bin/env python3
"""Run q-color random-coloring channel-LCU experiments in NAM and IBM bases."""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import qasm2, transpile
from qiskit.circuit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import Matrixsum, PauliAtom, channel as ChannelIR, channel_ensemble  # type: ignore  # noqa: E402
import channel_LCU  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


NAM_BASIS = ("h", "x", "rz", "cx")
NAM_TRANSPILER_BASIS = ("h", "x", "rz", "cx", "reset")
IBM_BASIS = ("u1", "u2", "u3", "cx")
IBM_TRANSPILER_BASIS = ("u1", "u2", "u3", "cx", "reset")
DEFAULT_OUT_ROOT = ROOT / "Baseline_results"
PAULI_MUL_TABLE = {
    ("I", "I"): ("I", 1.0),
    ("I", "X"): ("X", 1.0),
    ("I", "Y"): ("Y", 1.0),
    ("I", "Z"): ("Z", 1.0),
    ("X", "I"): ("X", 1.0),
    ("Y", "I"): ("Y", 1.0),
    ("Z", "I"): ("Z", 1.0),
    ("X", "X"): ("I", 1.0),
    ("Y", "Y"): ("I", 1.0),
    ("Z", "Z"): ("I", 1.0),
    ("X", "Y"): ("Z", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "X"): ("Y", 1j),
    ("X", "Z"): ("Y", -1j),
}
PauliDict = dict[str, complex]


class SynthesisTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class CaseSpec:
    n: int
    q: int
    graph: str

    @property
    def color_bits(self) -> int:
        return int(math.ceil(math.log2(self.q)))

    @property
    def system_qubits(self) -> int:
        return self.n * self.color_bits

    @property
    def name(self) -> str:
        return f"random_coloring_n{self.n}_q{self.q}_{self.graph}"


def cycle_edges(num_vertices: int) -> list[tuple[int, int]]:
    return sorted({tuple(sorted((i, (i + 1) % num_vertices))) for i in range(num_vertices)})


def linear_edges(num_vertices: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(num_vertices - 1)]


def complete_edges(num_vertices: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)]


def graph_edges(graph: str, num_vertices: int) -> list[tuple[int, int]]:
    if graph in {"linear", "path"}:
        return linear_edges(num_vertices)
    if graph == "cycle":
        return cycle_edges(num_vertices)
    if graph == "complete":
        return complete_edges(num_vertices)
    raise ValueError(f"Unsupported graph family: {graph}")


def neighbor_sets(num_vertices: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    neighbors = [set() for _ in range(num_vertices)]
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors


def int_to_bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def bits_to_int(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def valid_color_values(q: int) -> set[int]:
    return set(range(q))


def state_to_colors(state: int, spec: CaseSpec) -> tuple[int, ...]:
    bits = int_to_bits(state, spec.system_qubits)
    return tuple(
        bits_to_int(bits[v * spec.color_bits : (v + 1) * spec.color_bits])
        for v in range(spec.n)
    )


def colors_to_state(colors: tuple[int, ...], spec: CaseSpec) -> int:
    bits: list[int] = []
    for color in colors:
        bits.extend(int_to_bits(color, spec.color_bits))
    return bits_to_int(tuple(bits))


def is_valid_coloring(colors: tuple[int, ...], q: int) -> bool:
    return all(0 <= color < q for color in colors)


def is_proper_coloring(colors: tuple[int, ...], edges: list[tuple[int, int]], q: int) -> bool:
    return is_valid_coloring(colors, q) and all(colors[left] != colors[right] for left, right in edges)


def vertex_bit_positions(vertex: int, color_bits: int) -> list[int]:
    return list(range(vertex * color_bits, (vertex + 1) * color_bits))


def clean_pauli_dict(terms: PauliDict, tol: float = 1e-15) -> PauliDict:
    return {label: coeff for label, coeff in terms.items() if abs(coeff) > tol}


def pd_add(left: PauliDict, right: PauliDict) -> PauliDict:
    out = dict(left)
    for label, coeff in right.items():
        out[label] = out.get(label, 0.0) + coeff
    return clean_pauli_dict(out)


def pd_scaled(terms: PauliDict, factor: complex) -> PauliDict:
    if factor == 0:
        return {}
    return clean_pauli_dict({label: coeff * factor for label, coeff in terms.items()})


@lru_cache(maxsize=None)
def multiply_labels(left: str, right: str) -> tuple[str, complex]:
    label: list[str] = []
    phase: complex = 1.0
    for a, b in zip(left, right):
        out, local_phase = PAULI_MUL_TABLE[(a, b)]
        label.append(out)
        phase *= local_phase
    return "".join(label), phase


def pd_mul(left: PauliDict, right: PauliDict) -> PauliDict:
    out: PauliDict = {}
    for left_label, left_coeff in left.items():
        for right_label, right_coeff in right.items():
            label, phase = multiply_labels(left_label, right_label)
            out[label] = out.get(label, 0.0) + left_coeff * right_coeff * phase
    return clean_pauli_dict(out)


def matrixsum_from_pauli_dict(terms: PauliDict) -> Matrixsum:
    instances = []
    for label, coeff in terms.items():
        if abs(coeff) <= 1e-15:
            continue
        instances.append((PauliAtom(label, phase=coeff / abs(coeff)), float(abs(coeff))))
    return Matrixsum(instances)


def matrixsum_to_pauli_dict(ms: Matrixsum) -> PauliDict:
    terms: PauliDict = {}
    for inst, coeff in ms.instances:
        if not isinstance(inst, PauliAtom):
            raise TypeError("Expected Pauli-expanded Matrixsum.")
        terms[inst.expr] = terms.get(inst.expr, 0.0) + coeff * inst.phase
    return clean_pauli_dict(terms)


def hadamard_sign(row: int, col: int) -> int:
    return -1 if (row & col).bit_count() % 2 else 1


def color_hadamard_mix_kraus(kraus_ops: list[Matrixsum], spec: CaseSpec, tol: float) -> list[Matrixsum]:
    if spec.q <= 1 or spec.q & (spec.q - 1):
        raise ValueError("Walsh-Hadamard color mixing requires q to be a power of two.")
    if len(kraus_ops) != spec.n * spec.q:
        raise ValueError(f"Expected {spec.n * spec.q} Kraus operators, got {len(kraus_ops)}.")

    inv_sqrt_q = 1.0 / math.sqrt(spec.q)
    mixed: list[Matrixsum] = []
    for vertex in range(spec.n):
        group = [matrixsum_to_pauli_dict(kraus_ops[vertex * spec.q + color]) for color in range(spec.q)]
        for row in range(spec.q):
            terms: PauliDict = {}
            for color, color_terms in enumerate(group):
                terms = pd_add(terms, pd_scaled(color_terms, inv_sqrt_q * hadamard_sign(row, color)))
            if tol > 0:
                terms = {label: coeff for label, coeff in terms.items() if abs(coeff) > tol}
            mixed.append(matrixsum_from_pauli_dict(terms))
    return mixed


def single_bit_projector_dict(bit: int, index: int, total: int) -> PauliDict:
    sign = 1.0 if bit == 0 else -1.0
    z_label = ["I"] * total
    z_label[index] = "Z"
    return {"I" * total: 0.5, "".join(z_label): 0.5 * sign}


def single_bit_transition_dict(out_bit: int, in_bit: int, index: int, total: int) -> PauliDict:
    x_label = ["I"] * total
    x_label[index] = "X"
    y_label = ["I"] * total
    y_label[index] = "Y"
    if out_bit == 0 and in_bit == 0:
        return single_bit_projector_dict(0, index, total)
    if out_bit == 1 and in_bit == 1:
        return single_bit_projector_dict(1, index, total)
    if out_bit == 0 and in_bit == 1:
        # |0><1| = (X + iY) / 2
        return {"".join(x_label): 0.5, "".join(y_label): 0.5j}
    # |1><0| = (X - iY) / 2
    return {"".join(x_label): 0.5, "".join(y_label): -0.5j}


def color_projector_dict(vertex: int, color: int, spec: CaseSpec) -> PauliDict:
    result = {"I" * spec.system_qubits: 1.0}
    bits = int_to_bits(color, spec.color_bits)
    for bit, index in zip(bits, vertex_bit_positions(vertex, spec.color_bits)):
        result = pd_mul(result, single_bit_projector_dict(bit, index, spec.system_qubits))
    return result


def color_transition_dict(vertex: int, out_color: int, in_color: int, spec: CaseSpec) -> PauliDict:
    result = {"I" * spec.system_qubits: 1.0}
    out_bits = int_to_bits(out_color, spec.color_bits)
    in_bits = int_to_bits(in_color, spec.color_bits)
    for out_bit, in_bit, index in zip(out_bits, in_bits, vertex_bit_positions(vertex, spec.color_bits)):
        result = pd_mul(result, single_bit_transition_dict(out_bit, in_bit, index, spec.system_qubits))
    return result


def valid_register_projector_dict(vertex: int, spec: CaseSpec) -> PauliDict:
    total: PauliDict = {}
    for color in range(spec.q):
        total = pd_add(total, color_projector_dict(vertex, color, spec))
    return total


def valid_space_projector_dict(spec: CaseSpec) -> PauliDict:
    result = {"I" * spec.system_qubits: 1.0}
    for vertex in range(spec.n):
        result = pd_mul(result, valid_register_projector_dict(vertex, spec))
    return result


def edge_equal_projector_dict(left: int, right: int, spec: CaseSpec) -> PauliDict:
    total: PauliDict = {}
    for color in range(spec.q):
        total = pd_add(total, pd_mul(color_projector_dict(left, color, spec), color_projector_dict(right, color, spec)))
    return total


def proper_coloring_projector_dict(spec: CaseSpec, edges: list[tuple[int, int]]) -> PauliDict:
    result = valid_space_projector_dict(spec)
    for left, right in edges:
        not_equal = pd_add({"I" * spec.system_qubits: 1.0}, pd_scaled(edge_equal_projector_dict(left, right, spec), -1.0))
        result = pd_mul(result, not_equal)
    return result


def neighbor_conflict_projector_dict(
    vertex: int,
    proposed_color: int,
    spec: CaseSpec,
    neighbors: list[set[int]],
) -> PauliDict:
    no_conflict = {"I" * spec.system_qubits: 1.0}
    for neighbor in sorted(neighbors[vertex]):
        no_conflict = pd_mul(
            no_conflict,
            pd_add({"I" * spec.system_qubits: 1.0}, pd_scaled(color_projector_dict(neighbor, proposed_color, spec), -1.0)),
        )
    return pd_add({"I" * spec.system_qubits: 1.0}, pd_scaled(no_conflict, -1.0))


def target_reset_to_color_dict(vertex: int, proposed_color: int, spec: CaseSpec) -> PauliDict:
    total: PauliDict = {}
    for old_color in range(spec.q):
        total = pd_add(total, color_transition_dict(vertex, proposed_color, old_color, spec))
    return total


def forward_update_paulisum(spec: CaseSpec, neighbors: list[set[int]], vertex: int, proposed_color: int) -> Matrixsum:
    """Build a scalable full-space A_{v,k} using only local conflict checks.

    On the proper-coloring subspace this matches the usual random-coloring
    update: recolor vertex v to k if no neighbor currently has k, otherwise
    leave the state fixed.  Since the requested q=n cases have q a power of two,
    every bit pattern is a valid color.  We intentionally avoid the older global
    "freeze every improper state" extension because its Pauli expansion contains
    a product over all graph edges and is already impractical for n=4,q=4.
    """
    conflict = neighbor_conflict_projector_dict(vertex, proposed_color, spec, neighbors)
    no_conflict = pd_add({"I" * spec.system_qubits: 1.0}, pd_scaled(conflict, -1.0))
    reset = target_reset_to_color_dict(vertex, proposed_color, spec)
    # A = conflict + reset_to_k * no_conflict
    return matrixsum_from_pauli_dict(pd_add(conflict, pd_mul(reset, no_conflict)))


def estimate_local_terms(spec: CaseSpec, degree: int) -> int:
    # reset_to_k has q^2 Pauli terms; no_conflict is product of degree factors,
    # each with at most q+1 terms before simplification.
    return (spec.q**2 + 1) * ((spec.q + 1) ** degree)


def proper_coloring_count_formula(spec: CaseSpec, edges: list[tuple[int, int]]) -> int | None:
    complete = set(complete_edges(spec.n))
    edge_set = set(edges)
    if edge_set == complete:
        if spec.q < spec.n:
            return 0
        value = 1
        for k in range(spec.n):
            value *= spec.q - k
        return int(value)
    if edge_set == set(cycle_edges(spec.n)):
        return int((spec.q - 1) ** spec.n + ((-1) ** spec.n) * (spec.q - 1))
    if edge_set == set(linear_edges(spec.n)):
        return int(spec.q * ((spec.q - 1) ** (spec.n - 1)))
    return None


def random_coloring_model_info(spec: CaseSpec, edges: list[tuple[int, int]], max_degree: int) -> dict[str, Any]:
    estimated_per_kraus = estimate_local_terms(spec, max_degree)
    estimated_total = spec.n * spec.q * estimated_per_kraus
    valid_states = spec.q**spec.n
    formula_count = proper_coloring_count_formula(spec, edges)
    if formula_count is not None:
        proper_states = formula_count
    elif valid_states <= 1_000_000:
        proper_states = 0
        for colors_index in range(valid_states):
            rem = colors_index
            colors = []
            for _ in range(spec.n):
                colors.append(rem % spec.q)
                rem //= spec.q
            color_tuple = tuple(reversed(colors))
            if is_proper_coloring(color_tuple, edges, spec.q):
                proper_states += 1
    else:
        proper_states = -1

    return {
        "model": "q_color_random_coloring_local_full_space_extension",
        "graph": spec.graph,
        "num_vertices": int(spec.n),
        "q": int(spec.q),
        "color_bits": int(spec.color_bits),
        "system_qubits": int(spec.system_qubits),
        "edges": [[int(i), int(j)] for i, j in edges],
        "valid_coloring_count": int(valid_states),
        "proper_coloring_count": int(proper_states),
        "kraus_normalization": f"1/sqrt({spec.n * spec.q})",
        "estimated_pauli_terms_per_kraus": int(estimated_per_kraus),
        "estimated_pauli_terms_total": int(estimated_total),
        "construction": "A=Pconflict+ResetToK*(I-Pconflict); K=A^dagger/sqrt(nq)",
    }


def build_random_coloring_channel(
    n: int,
    graph: str,
    *,
    q: int | None,
    tol: float,
    max_total_pauli_terms: int | None,
    max_terms_per_kraus: int | None,
    mix_colors: bool,
) -> tuple[list[Matrixsum], dict[str, Any]]:
    q = n if q is None else q
    spec = CaseSpec(n=n, q=q, graph=graph)
    edges = graph_edges(graph, n)
    neighbors = neighbor_sets(n, edges)
    denom = math.sqrt(n * q)

    max_degree = max((len(nbrs) for nbrs in neighbors), default=0)
    model_info = random_coloring_model_info(spec, edges, max_degree)
    estimated_per_kraus = model_info["estimated_pauli_terms_per_kraus"]
    estimated_total = model_info["estimated_pauli_terms_total"]
    if max_terms_per_kraus is not None and estimated_per_kraus > max_terms_per_kraus:
        raise RuntimeError(
            f"{spec.name} estimated {estimated_per_kraus} Pauli terms per Kraus "
            f"for degree {max_degree}, exceeding --max-terms-per-kraus={max_terms_per_kraus}."
        )
    if max_total_pauli_terms is not None and estimated_total > max_total_pauli_terms:
        raise RuntimeError(
            f"{spec.name} estimated {estimated_total} total Pauli terms, exceeding "
            f"--max-total-pauli-terms={max_total_pauli_terms}."
        )

    kraus_ops: list[Matrixsum] = []
    for vertex in range(n):
        for proposed_color in range(q):
            forward = forward_update_paulisum(spec, neighbors, vertex, proposed_color)
            # Pauli-expanded A_{v,k} is real/Hermitian for this full-space extension.
            kraus = copy.deepcopy(forward.adj())
            kraus.mul_coeffs(1.0 / denom)
            if tol > 0:
                kraus = Matrixsum([(inst, coeff) for inst, coeff in kraus.instances if coeff > tol])
            if max_terms_per_kraus is not None and len(kraus.instances) > max_terms_per_kraus:
                raise RuntimeError(
                    f"{spec.name} Kraus(v={vertex},k={proposed_color}) has {len(kraus.instances)} "
                    f"terms, exceeding --max-terms-per-kraus={max_terms_per_kraus}."
                )
            kraus_ops.append(kraus)
            total_terms = sum(len(op.instances) for op in kraus_ops)
            if max_total_pauli_terms is not None and total_terms > max_total_pauli_terms:
                raise RuntimeError(
                    f"{spec.name} generated {total_terms} Pauli terms, exceeding "
                    f"--max-total-pauli-terms={max_total_pauli_terms}."
                )

    if mix_colors:
        kraus_ops = color_hadamard_mix_kraus(kraus_ops, spec, tol)
        if max_terms_per_kraus is not None:
            for index, kraus in enumerate(kraus_ops):
                if len(kraus.instances) > max_terms_per_kraus:
                    raise RuntimeError(
                        f"{spec.name} mixed Kraus(index={index}) has {len(kraus.instances)} "
                        f"terms, exceeding --max-terms-per-kraus={max_terms_per_kraus}."
                    )
        total_terms = sum(len(op.instances) for op in kraus_ops)
        if max_total_pauli_terms is not None and total_terms > max_total_pauli_terms:
            raise RuntimeError(
                f"{spec.name} mixed channel has {total_terms} Pauli terms, exceeding "
                f"--max-total-pauli-terms={max_total_pauli_terms}."
            )
        model_info["kraus_color_mixing"] = "walsh_hadamard_by_vertex"
        model_info["pauli_terms_after_color_mixing"] = int(total_terms)
    else:
        model_info["kraus_color_mixing"] = "none"

    return kraus_ops, model_info


def channel_metrics(kraus_ops: list[Matrixsum]) -> dict[str, int]:
    nonzero = [op for op in kraus_ops if len(op.instances) > 0]
    return {
        "kraus_count": int(len(nonzero)),
        "pauli_terms_total": int(sum(len(op.instances) for op in nonzero)),
        "max_terms_per_kraus": int(max((len(op.instances) for op in nonzero), default=0)),
        "system_qubits": int(max((op.size for op in nonzero), default=0)),
    }


def rewrite_channel(
    kraus_ops: list[Matrixsum],
    *,
    strategy: str,
    beam_width: int,
    max_steps: int,
    tol: float,
) -> tuple[list[Matrixsum], dict[str, Any], float]:
    started = time.perf_counter()
    rewrite_source = ChannelIR(copy.deepcopy(kraus_ops))
    rewrite_source.zero_elim()
    result = rewrite_source.rewrite_search(
        strategy=strategy,
        beam_width=beam_width,
        max_steps=max_steps,
        tol=tol,
    )
    rewritten = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten.apply_rewrite_result(result, tol=tol)
    rewritten.zero_elim()
    elapsed = time.perf_counter() - started
    return rewritten.kraus_ops, result, elapsed


def with_new_block_encoding(func):
    old_block_encoding = channel_LCU.BlockEncoding
    channel_LCU.BlockEncoding = NewBlockEncoding
    try:
        return func()
    finally:
        channel_LCU.BlockEncoding = old_block_encoding


def build_channel_lcu_circuit(
    kraus_ops: list[Matrixsum],
    *,
    structure: str,
    opt: str,
) -> tuple[QuantumCircuit, Any, float]:
    started = time.perf_counter()

    def _build():
        return channel_LCU.channel_to_LCU(
            channel_ensemble([copy.deepcopy(kraus_ops)]),
            structure=structure,
            opt=opt,
        )

    qc, qubit_indexes = with_new_block_encoding(_build)
    elapsed = time.perf_counter() - started
    return qc, qubit_indexes, elapsed


def build_single_kraus_matrix_order_circuit(kraus_ops: list[Matrixsum]) -> tuple[QuantumCircuit, Any, float]:
    started = time.perf_counter()
    if len(kraus_ops) != 1:
        raise ValueError("single-Kraus builder requires exactly one Kraus operator.")

    sys_size = kraus_ops[0].size
    if len(kraus_ops[0].instances) == 1:
        inst, coeff = kraus_ops[0].instances[0]
        if isinstance(inst, PauliAtom) and abs(coeff - 1.0) <= 1e-12:
            qc = QuantumCircuit(sys_size)
            elapsed = time.perf_counter() - started
            return qc, [[], [], [], list(range(sys_size))], elapsed

    block = NewBlockEncoding(copy.deepcopy(kraus_ops[0])).circuit(opt="No")
    ctrl_size = block.num_qubits - sys_size
    qc = QuantumCircuit(ctrl_size + sys_size)
    qc.compose(block, qubits=range(ctrl_size + sys_size), inplace=True)
    elapsed = time.perf_counter() - started
    return qc, [[], [], list(range(ctrl_size)), list(range(ctrl_size, ctrl_size + sys_size))], elapsed


def build_rewrite_opt_matrix_order_circuit(
    kraus_ops: list[Matrixsum],
) -> tuple[QuantumCircuit, Any, float]:
    if len(kraus_ops) == 1:
        return build_single_kraus_matrix_order_circuit(kraus_ops)

    started = time.perf_counter()
    sys_size = max(ms.size for ms in kraus_ops)
    select_size = int(np.ceil(np.log2(len(kraus_ops))))
    coeff_sums = [ms.pauli_norm() for ms in kraus_ops]

    lcu_ini = channel_LCU.prep_sup_state(coeff_sums)
    circuits = [
        NewBlockEncoding(copy.deepcopy(ms)).circuit(opt="Matrix-order")
        for ms in kraus_ops
    ]
    ctrl_size = max(circ.num_qubits - sys_size for circ in circuits)

    qc = QuantumCircuit(2 * select_size + ctrl_size + sys_size)
    select_indexes = [2 * j for j in range(select_size)]
    anc_indexes = [2 * j + 1 for j in range(select_size)]
    ctrl_indexes = list(range(2 * select_size, 2 * select_size + ctrl_size))
    sys_indexes = list(range(2 * select_size + ctrl_size, 2 * select_size + ctrl_size + sys_size))

    qc.compose(lcu_ini, qubits=select_indexes, inplace=True)
    qc, _tcount, _mccount, _cxcount = channel_LCU.mulplex_U_opt(
        qc,
        circuits,
        select_size,
        ctrl_size,
        sys_size,
    )
    elapsed = time.perf_counter() - started
    return qc, [select_indexes, anc_indexes, ctrl_indexes, sys_indexes], elapsed


def _raw_circuit_worker(conn: Any, kraus_ops: list[Matrixsum], structure: str, opt: str) -> None:
    try:
        qc, qubit_indexes, elapsed = build_channel_lcu_circuit(kraus_ops, structure=structure, opt=opt)
        conn.send(("ok", qc, qubit_indexes, elapsed))
    except Exception as exc:
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


def _ours_circuit_worker(conn: Any, kraus_ops: list[Matrixsum]) -> None:
    try:
        qc, qubit_indexes, elapsed = build_rewrite_opt_matrix_order_circuit(kraus_ops)
        conn.send(("ok", qc, qubit_indexes, elapsed))
    except Exception as exc:
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


class _InlineConn:
    def __init__(self) -> None:
        self.message: tuple[Any, ...] | None = None

    def send(self, message: tuple[Any, ...]) -> None:
        self.message = message

    def close(self) -> None:
        pass


def run_with_timeout(worker: Any, payload: tuple[Any, ...], timeout_s: float | None, stage: str) -> tuple[QuantumCircuit, Any, float]:
    if timeout_s is None or timeout_s <= 0:
        inline_conn = _InlineConn()
        worker(inline_conn, *payload)
        if inline_conn.message is None:
            raise RuntimeError(f"{stage} did not return a result.")
        message = inline_conn.message
    else:
        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = mp.Process(target=worker, args=(child_conn, *payload))
        proc.start()
        child_conn.close()
        deadline = time.monotonic() + timeout_s
        message = None
        while True:
            if parent_conn.poll(0.05):
                message = parent_conn.recv()
                proc.join(5)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(5)
                break
            if not proc.is_alive():
                proc.join()
                if parent_conn.poll():
                    message = parent_conn.recv()
                    break
                raise RuntimeError(f"{stage} subprocess exited without returning a result.")
            if time.monotonic() >= deadline:
                proc.terminate()
                proc.join(5)
                parent_conn.close()
                raise SynthesisTimeout(f"{stage} exceeded --synthesis-timeout-sec={timeout_s}.")
        parent_conn.close()

    if message[0] == "ok":
        return message[1], message[2], message[3]
    raise RuntimeError(f"{stage} failed: {message[1]}")


def dump_untranspiled_qasm(qc: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        qasm2.dump(qc, out_file)


def export_basis_count_qasm(
    qc: QuantumCircuit,
    path: Path,
    *,
    gate_set: str,
    optimization_level: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if gate_set == "nam":
        basis = NAM_TRANSPILER_BASIS
    elif gate_set == "ibm":
        basis = IBM_TRANSPILER_BASIS
    else:
        raise ValueError(gate_set)
    tqc = transpile(qc, basis_gates=list(basis), optimization_level=optimization_level)
    elapsed = time.perf_counter() - started
    export_openqasm2_quartz_input(tqc, path, basis_gates=basis)
    return count_qasm_file(path, gate_set=gate_set), elapsed


def metric_total(stats: dict[str, Any], gate_set: str) -> int:
    if gate_set == "nam":
        return int(stats["nam"]["rz_clifford_total"])
    if gate_set == "ibm":
        return int(stats["metric_total"])
    raise ValueError(gate_set)


def row_from_stats(
    name: str,
    gate_set: str,
    stats: dict[str, Any],
    compile_time: float | None,
    *,
    reference_total: int | None,
) -> dict[str, Any]:
    total = metric_total(stats, gate_set)
    return {
        "name": name,
        "basis": gate_set,
        "qubits": int(stats["num_qubits"]),
        "metric_total": total,
        "ratio_to_raw": None if reference_total is None else float(total / reference_total),
        "cx": int(stats["ops"].get("cx", 0)),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "reset": int(stats["ops"].get("reset", 0)),
        "compile_time_s": None if compile_time is None else float(compile_time),
        "ops": stats["ops"],
        "clifford": stats["clifford"],
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Circuit / tool | Basis | Qubits | Metric total | G/G0 | CX | Non-Clifford | Reset | Compilation time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        time_cell = "-" if row["compile_time_s"] is None else f"{row['compile_time_s']:.3f}"
        ratio_cell = "-" if row["ratio_to_raw"] is None else f"{row['ratio_to_raw']:.6f}"
        lines.append(
            f"| {row['name']} | {row['basis']} | {row['qubits']} | {row['metric_total']} | "
            f"{ratio_cell} | {row['cx']} | {row['non_clifford']} | {row['reset']} | {time_cell} |"
        )
    return "\n".join(lines)


def unavailable_summary(
    case: CaseSpec,
    out_dir: Path,
    started: float,
    exc: Exception,
    *,
    model_info: dict[str, Any] | None = None,
    status: str = "unavailable",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_info is None:
        edges = graph_edges(case.graph, case.n)
        neighbors = neighbor_sets(case.n, edges)
        max_degree = max((len(nbrs) for nbrs in neighbors), default=0)
        model_info = random_coloring_model_info(case, edges, max_degree)
    summary = {
        "experiment": "random_coloring_channel_lcu_ours_dual_basis",
        "case": case.name,
        "status": status,
        "error_msg": f"{type(exc).__name__}: {exc}",
        "total_time_s": float(time.perf_counter() - started),
        "model": model_info,
        "basis_counted": {
            "nam": list(NAM_BASIS),
            "ibm": list(IBM_BASIS),
        },
    }
    if extra:
        summary.update(extra)
    summary_path = out_dir / f"{case.name}_ours_dual_basis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_case(args: argparse.Namespace, case: CaseSpec) -> dict[str, Any]:
    started_case = time.perf_counter()
    out_dir = Path(args.out_root) / f"{case.name}_ours_dual_basis"
    edges = graph_edges(case.graph, case.n)
    neighbors = neighbor_sets(case.n, edges)
    max_degree = max((len(nbrs) for nbrs in neighbors), default=0)
    model_info = random_coloring_model_info(case, edges, max_degree)
    if args.metadata_only:
        model_info["kraus_color_mixing"] = "none" if args.no_color_mixing else "walsh_hadamard_by_vertex"
        summary = {
            "experiment": "random_coloring_channel_lcu_ours_dual_basis",
            "case": case.name,
            "status": "metadata_only",
            "model": model_info,
            "basis_counted": {
                "nam": list(NAM_BASIS),
                "ibm": list(IBM_BASIS),
            },
            "note": "No channel-LCU circuit synthesis was requested.",
            "total_time_s": float(time.perf_counter() - started_case),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"{case.name}_ours_dual_basis_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary

    try:
        raw_kraus_ops, model_info = build_random_coloring_channel(
            case.n,
            case.graph,
            q=case.q,
            tol=args.tol,
            max_total_pauli_terms=args.max_total_pauli_terms,
            max_terms_per_kraus=args.max_terms_per_kraus,
            mix_colors=not args.no_color_mixing,
        )
    except Exception as exc:
        return unavailable_summary(case, out_dir, started_case, exc, model_info=model_info)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_channel_metrics = channel_metrics(raw_kraus_ops)

    try:
        raw_qc, raw_qubit_indexes, raw_build_time = run_with_timeout(
            _raw_circuit_worker,
            (raw_kraus_ops, "basic", "No"),
            args.synthesis_timeout_sec,
            f"{case.name} raw channel-LCU synthesis",
        )
    except SynthesisTimeout as exc:
        return unavailable_summary(
            case,
            out_dir,
            started_case,
            exc,
            model_info=model_info,
            status="timeout",
            extra={
                "failed_stage": "raw_channel_lcu_synthesis",
                "raw": {"channel_metrics": raw_channel_metrics},
                "synthesis_timeout_sec": args.synthesis_timeout_sec,
            },
        )
    raw_qasm = out_dir / f"{case.name}_basic_raw.qasm"
    dump_untranspiled_qasm(raw_qc, raw_qasm)

    raw_stats: dict[str, Any] = {}
    raw_times: dict[str, float] = {}
    raw_metric_totals: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for gate_set in ("nam", "ibm"):
        path = out_dir / f"{case.name}_basic_raw_{gate_set}.qasm"
        stats, elapsed = export_basis_count_qasm(
            raw_qc,
            path,
            gate_set=gate_set,
            optimization_level=args.transpile_optimization_level,
        )
        raw_stats[gate_set] = {"qasm": str(path), "stats": stats}
        raw_times[gate_set] = elapsed
        raw_metric_totals[gate_set] = metric_total(stats, gate_set)
        rows.append(row_from_stats("Raw channel-LCU (basic/no)", gate_set, stats, None, reference_total=raw_metric_totals[gate_set]))

    if args.raw_only:
        summary = {
            "experiment": "random_coloring_channel_lcu_ours_dual_basis",
            "case": case.name,
            "status": "raw_ok",
            "model": model_info,
            "basis_counted": {
                "nam": list(NAM_BASIS),
                "ibm": list(IBM_BASIS),
            },
            "transpiler_basis": {
                "nam": list(NAM_TRANSPILER_BASIS),
                "ibm": list(IBM_TRANSPILER_BASIS),
            },
            "transpile_optimization_level": int(args.transpile_optimization_level),
            "raw": {
                "channel_metrics": raw_channel_metrics,
                "build_time_s": float(raw_build_time),
                "qasm": str(raw_qasm),
                "basis_outputs": raw_stats,
                "basis_transpile_times_s": raw_times,
                "qubit_indexes": raw_qubit_indexes,
            },
            "table_rows": rows,
            "markdown_table": markdown_table(rows),
            "note": "--raw-only requested; rewrite and optimized channel-LCU synthesis were not run.",
            "total_time_s": float(time.perf_counter() - started_case),
        }
        summary_path = out_dir / f"{case.name}_ours_dual_basis_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary

    rewritten_kraus_ops, rewrite_result, rewrite_time = rewrite_channel(
        raw_kraus_ops,
        strategy=args.rewrite_strategy,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        tol=args.tol,
    )
    rewrite_channel_metrics = channel_metrics(rewritten_kraus_ops)

    try:
        ours_qc, ours_qubit_indexes, ours_build_time = run_with_timeout(
            _ours_circuit_worker,
            (rewritten_kraus_ops,),
            args.synthesis_timeout_sec,
            f"{case.name} optimized channel-LCU synthesis",
        )
    except SynthesisTimeout as exc:
        return unavailable_summary(
            case,
            out_dir,
            started_case,
            exc,
            model_info=model_info,
            status="timeout",
            extra={
                "failed_stage": "optimized_channel_lcu_synthesis",
                "raw": {
                    "channel_metrics": raw_channel_metrics,
                    "qasm": str(raw_qasm),
                    "basis_outputs": raw_stats,
                    "basis_transpile_times_s": raw_times,
                    "qubit_indexes": raw_qubit_indexes,
                },
                "rewrite": {
                    "channel_metrics": rewrite_channel_metrics,
                    "time_s": float(rewrite_time),
                    "initial_support": int(rewrite_result["initial_support"]),
                    "final_support": int(rewrite_result["final_support"]),
                    "steps": int(len(rewrite_result["steps"])),
                    "termination": rewrite_result.get("termination", {}),
                },
                "synthesis_timeout_sec": args.synthesis_timeout_sec,
            },
        )
    ours_qasm = out_dir / f"{case.name}_rewrite_opt_matrix_order.qasm"
    dump_untranspiled_qasm(ours_qc, ours_qasm)
    ours_stats: dict[str, Any] = {}
    ours_times: dict[str, float] = {}
    comparisons: dict[str, Any] = {}
    for gate_set in ("nam", "ibm"):
        path = out_dir / f"{case.name}_rewrite_opt_matrix_order_{gate_set}.qasm"
        stats, elapsed = export_basis_count_qasm(
            ours_qc,
            path,
            gate_set=gate_set,
            optimization_level=args.transpile_optimization_level,
        )
        ours_stats[gate_set] = {"qasm": str(path), "stats": stats}
        ours_times[gate_set] = elapsed
        total = metric_total(stats, gate_set)
        comparisons[gate_set] = {
            "raw_metric_total": raw_metric_totals[gate_set],
            "ours_metric_total": total,
            "ours_to_raw_ratio": float(total / raw_metric_totals[gate_set]),
        }
        rows.append(
            row_from_stats(
                "Ours (rewrite + opt + matrix-order)",
                gate_set,
                stats,
                rewrite_time + ours_build_time + elapsed,
                reference_total=raw_metric_totals[gate_set],
            )
        )

    summary = {
        "experiment": "random_coloring_channel_lcu_ours_dual_basis",
        "case": case.name,
        "status": "ok",
        "model": model_info,
        "basis_counted": {
            "nam": list(NAM_BASIS),
            "ibm": list(IBM_BASIS),
        },
        "transpiler_basis": {
            "nam": list(NAM_TRANSPILER_BASIS),
            "ibm": list(IBM_TRANSPILER_BASIS),
        },
        "transpile_optimization_level": int(args.transpile_optimization_level),
        "raw": {
            "channel_metrics": raw_channel_metrics,
            "build_time_s": float(raw_build_time),
            "qasm": str(raw_qasm),
            "basis_outputs": raw_stats,
            "basis_transpile_times_s": raw_times,
            "qubit_indexes": raw_qubit_indexes,
        },
        "rewrite": {
            "channel_metrics": rewrite_channel_metrics,
            "time_s": float(rewrite_time),
            "initial_support": int(rewrite_result["initial_support"]),
            "final_support": int(rewrite_result["final_support"]),
            "steps": int(len(rewrite_result["steps"])),
            "termination": rewrite_result.get("termination", {}),
        },
        "ours": {
            "structure": "opt",
            "opt": "Matrix-order",
            "build_time_s": float(ours_build_time),
            "qasm": str(ours_qasm),
            "basis_outputs": ours_stats,
            "basis_transpile_times_s": ours_times,
            "qubit_indexes": ours_qubit_indexes,
        },
        "comparison": comparisons,
        "table_rows": rows,
        "markdown_table": markdown_table(rows),
        "total_time_s": float(time.perf_counter() - started_case),
    }
    summary_path = out_dir / f"{case.name}_ours_dual_basis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run q-color random-coloring channel-LCU rewrite+opt+matrix-order statistics in NAM and IBM bases."
    )
    parser.add_argument("--num-vertices", nargs="+", default=["4", "8"], help="Vertex counts; default: 4 8.")
    parser.add_argument("--q", type=int, default=None, help="Number of colors. Default: q=n for each case.")
    parser.add_argument("--graph", nargs="+", choices=("linear", "path", "cycle", "complete", "all"), default=["all"])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--rewrite-strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--transpile-optimization-level", type=int, default=0)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write model/config summaries without building channel-LCU circuits.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Stop after raw channel-LCU synthesis and basis export.",
    )
    parser.add_argument(
        "--no-color-mixing",
        action="store_true",
        help="Disable per-vertex Walsh-Hadamard Kraus mixing over proposed colors.",
    )
    parser.add_argument(
        "--synthesis-timeout-sec",
        type=float,
        default=0.0,
        help="Per-stage timeout for channel-LCU circuit synthesis; <=0 disables the timeout.",
    )
    parser.add_argument(
        "--max-total-pauli-terms",
        type=int,
        default=250_000,
        help="Protective cap; cases exceeding it are recorded as unavailable.",
    )
    parser.add_argument(
        "--max-terms-per-kraus",
        type=int,
        default=50_000,
        help="Protective cap; cases exceeding it are recorded as unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = parse_int_list(args.num_vertices)
    graphs = ["cycle", "complete"] if "all" in args.graph else list(args.graph)
    summaries = []
    for n in ns:
        q = n if args.q is None else int(args.q)
        for graph in graphs:
            case = CaseSpec(n=n, q=q, graph=graph)
            summary = run_case(args, case)
            summaries.append(summary)
            print(f"## {case.name} [{summary['status']}]")
            if summary["status"] == "ok":
                print(summary["markdown_table"])
                print(
                    "rewrite support: "
                    f"{summary['rewrite']['initial_support']} -> {summary['rewrite']['final_support']} "
                    f"({summary['rewrite']['steps']} steps)"
                )
                print(
                    "proper colorings: "
                    f"{summary['model']['proper_coloring_count']}, "
                    f"terms raw/rewrite: "
                    f"{summary['raw']['channel_metrics']['pauli_terms_total']} -> "
                    f"{summary['rewrite']['channel_metrics']['pauli_terms_total']}"
                )
            elif summary["status"] == "metadata_only":
                print(
                    "metadata only: "
                    f"system_qubits={summary['model']['system_qubits']}, "
                    f"proper_colorings={summary['model']['proper_coloring_count']}, "
                    f"estimated_terms={summary['model']['estimated_pauli_terms_total']}"
                )
            else:
                print(summary["error_msg"])
            print(f"summary: {summary['summary_path']}")
            print()

    index_path = Path(args.out_root) / "random_coloring_qcolor_dual_basis_index.json"
    index_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
