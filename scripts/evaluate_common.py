"""Shared helpers for scripts split from ``src/evaluate.ipynb``."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "Results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_CACHE_DIR = Path(tempfile.gettempdir()) / "lindbladian_compilation_cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
from qutip import Qobj, qeye, sigmam, sigmax, sigmaz, spre, spost, tensor, to_super

from channel_IR import Lindbladian, Matrixsum, PauliAtom, channel_ensemble  # noqa: E402
from channel_LCU import (  # noqa: E402
    Lindblad_to_channel,
    channel_to_LCU,
    construct_qobj_lind,
    simulate_circuit,
)
from block_encoding import BlockEncoding  # noqa: E402
from baseline import simulate_lindblad  # noqa: E402
from Hamiltonian import gate_count_summary, qdrift_hamiltonian, qsvt_Hamiltonian  # noqa: E402
from simulator_utils import normalize, simulate_circuit_statevec  # noqa: E402
from series_expansion import (  # noqa: E402
    construct_circuit_coherent,
    construct_higher_order_circuit,
)


def count_metrics(
    qc: QuantumCircuit,
    basis_gates: list[str] | tuple[str, ...] = ("cx", "u3"),
    optimization_level: int = 3,
) -> dict[str, int]:
    tqc = transpile(
        qc,
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
    )
    ops = tqc.count_ops()
    return {
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx": int(ops.get("cx", 0)),
        "u3": int(ops.get("u3", 0)),
        "num_qubits": int(tqc.num_qubits),
    }


def ratio(curr: int | float, base: int | float) -> float:
    if base == 0:
        return float("nan")
    return 1.0 - (float(curr) / float(base))


def compression_ratio(other: int | float, base: int | float) -> float:
    if base == 0:
        return float("nan")
    return float(other) / float(base)


def remove_save_instructions(qc: QuantumCircuit) -> QuantumCircuit:
    qc_clean = QuantumCircuit(*qc.qregs, *qc.cregs, name=f"{qc.name}_nosave")
    for instruction in qc.data:
        op = instruction.operation
        if op.name.startswith("save_"):
            continue
        qc_clean.append(op, instruction.qubits, instruction.clbits)
    return qc_clean


def build_periodic_tfim_lindbladian_pauli(
    num_qubits: int,
    gamma: float,
) -> tuple[list[tuple[str, complex]], list[list[tuple[str, complex]]]]:
    """TFIM-style model used by Test I/Test III in ``evaluate.ipynb``."""
    h_terms: list[tuple[str, complex]] = []
    l_terms: list[list[tuple[str, complex]]] = []

    for i in range(num_qubits):
        z_neighbors = [i, (i + 1) % num_qubits]
        zz_str = "".join("Z" if k in z_neighbors else "I" for k in range(num_qubits))
        x_str = "".join("X" if k == i else "I" for k in range(num_qubits))
        y_str = "".join("Y" if k == i else "I" for k in range(num_qubits))
        h_terms.append((zz_str, -1.0))
        h_terms.append((x_str, -1.0))
        l_terms.append([(x_str, gamma), (y_str, -1j * gamma)])

    return h_terms, l_terms


def build_tfim_decay_lindbladian_pauli(
    num_qubits: int,
    delta: float,
    coupling: float,
    gamma: float,
) -> tuple[list[tuple[str, complex]], list[list[tuple[str, complex]]]]:
    """TFIM + decay model from ``evaluate.ipynb`` Test III.2."""
    h_terms: list[tuple[str, complex]] = []
    l_terms: list[list[tuple[str, complex]]] = []

    for j in range(num_qubits):
        z_str = "".join("Z" if k == j else "I" for k in range(num_qubits))
        h_terms.append((z_str, delta))

    for j in range(num_qubits - 1):
        xx_str = "".join(
            "X" if (k == j or k == j + 1) else "I"
            for k in range(num_qubits)
        )
        h_terms.append((xx_str, -coupling))

    amp = np.sqrt(gamma) / 2
    for j in range(num_qubits):
        x_str = "".join("X" if k == j else "I" for k in range(num_qubits))
        y_str = "".join("Y" if k == j else "I" for k in range(num_qubits))
        l_terms.append([(x_str, amp), (y_str, -1j * amp)])

    return h_terms, l_terms


def build_tfim_decay_qutip_reference(
    num_qubits: int,
    delta: float,
    coupling: float,
    gamma: float,
):
    sz = sigmaz()
    sx = sigmax()
    h_qobj = 0 * tensor([qeye(2) for _ in range(num_qubits)])

    for j in range(1, num_qubits + 1):
        left = [qeye(2) for _ in range(j - 1)]
        right = [qeye(2) for _ in range(num_qubits - j)]
        h_qobj += delta * tensor(left + [sz] + right)

    for j in range(1, num_qubits):
        left = [qeye(2) for _ in range(j - 1)]
        right = [qeye(2) for _ in range(num_qubits - j - 1)]
        h_qobj += (-coupling) * tensor(left + [sx, sx] + right)

    l_qobj = []
    for j in range(num_qubits):
        left = [qeye(2) for _ in range(j)]
        right = [qeye(2) for _ in range(num_qubits - j - 1)]
        l_qobj.append(np.sqrt(gamma) * tensor(left + [sigmam()] + right))

    return h_qobj, l_qobj


def get_superop_qutip(h_qobj: Qobj, l_qobj_list: list[Qobj], time_step: float):
    superop = spre(-1j * h_qobj)
    superop += spost(1j * h_qobj)
    for l_op in l_qobj_list:
        superop += to_super(l_op)
        superop += -0.5 * spre(l_op.dag() @ l_op)
        superop += -0.5 * spost(l_op.dag() @ l_op)
    return (superop * time_step).expm()


def label_to_symplectic_masks(label: str) -> tuple[int, int]:
    x_mask = 0
    z_mask = 0
    for i, ch in enumerate(label):
        bit = 1 << i
        if ch == "X":
            x_mask |= bit
        elif ch == "Y":
            x_mask |= bit
            z_mask |= bit
        elif ch == "Z":
            z_mask |= bit
    return x_mask, z_mask


def symplectic_inner_product_mod2(mask1: tuple[int, int], mask2: tuple[int, int]) -> int:
    x1, z1 = mask1
    x2, z2 = mask2
    return (((x1 & z2) ^ (z1 & x2)).bit_count()) & 1


def extract_single_pauli_terms(h_terms):
    labels, coeffs = [], []
    if hasattr(h_terms, "instances"):
        for atom, coeff in h_terms.instances:
            labels.append(atom.expr if hasattr(atom, "expr") else str(atom))
            coeffs.append(complex(coeff) * complex(getattr(atom, "phase", 1.0)))
    else:
        for label, coeff in h_terms:
            labels.append(label)
            coeffs.append(complex(coeff))
    return labels, coeffs


def nested_commutator(h_terms) -> float:
    labels, coeffs = extract_single_pauli_terms(h_terms)
    if len(labels) < 2:
        raise ValueError("At least two Pauli terms are required.")

    masks = [label_to_symplectic_masks(label) for label in labels]
    norms = 0.0
    for i in range(len(labels)):
        coeff_i = abs(coeffs[i])
        for j in range(i + 1, len(labels)):
            if symplectic_inner_product_mod2(masks[i], masks[j]) == 1:
                norms += 2.0 * coeff_i * abs(coeffs[j])
    return float(norms)


def sample_unique_random_pauli_labels(
    num_qubits: int,
    size: int,
    max_rounds: int = 50,
):
    from qiskit.quantum_info import random_pauli_list

    if size > 4**num_qubits:
        raise ValueError(f"size={size} exceeds total number of Pauli strings 4**{num_qubits}")

    labels = set()
    rounds = 0
    while len(labels) < size and rounds < max_rounds:
        need = size - len(labels)
        batch_size = max(need, 8)
        batch = random_pauli_list(num_qubits=num_qubits, size=batch_size, phase=False)
        labels.update(p.to_label() for p in batch)
        rounds += 1

    if len(labels) < size:
        raise RuntimeError(
            f"Failed to collect {size} unique Pauli strings after {max_rounds} rounds; "
            f"got {len(labels)}."
        )

    return list(labels)[:size]


def count_1q2q_gates(qc: QuantumCircuit):
    gate_details = dict(qc.count_ops())
    count2q = int(gate_details.get("cx", 0))
    count1q = int(sum(count for gate, count in gate_details.items() if gate != "cx"))
    return gate_details, count2q, count1q


def density_trace_distance(left, right) -> float:
    return float(np.linalg.norm(DensityMatrix(left).data - DensityMatrix(right).data, ord="nuc") / 2)
