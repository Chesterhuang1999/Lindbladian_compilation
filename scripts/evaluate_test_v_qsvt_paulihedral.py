"""Test V from ``evaluate.ipynb``: QSVT gate counts vs Paulihedral."""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import Target
from scipy.special import jv

from evaluate_common import (
    BlockEncoding,
    Lindbladian,
    Matrixsum,
    RESULTS_DIR,
    count_1q2q_gates,
    nested_commutator,
    qdrift_hamiltonian,
    qsvt_Hamiltonian,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from subroutine import get_adaptive_qsp_phases


def scaled_matrixsum(ms: Matrixsum, factor: complex) -> Matrixsum:
    out = deepcopy(ms)
    out.mul_coeffs(factor)
    return out.simplify()


def chebyshev_matrixsum(h_op: Matrixsum, degree: int) -> Matrixsum:
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if degree == 0:
        return h_op.identity(h_op.size)
    if degree == 1:
        return deepcopy(h_op)

    t_prev = h_op.identity(h_op.size)
    t_curr = deepcopy(h_op)
    for _ in range(1, degree):
        t_next = scaled_matrixsum(h_op.mul(t_curr), 2.0).add(scaled_matrixsum(t_prev, -1.0))
        t_prev, t_curr = t_curr, t_next.simplify()
    return t_curr


def jacobi_anger_trig_matrixsum(h_op: Matrixsum, degree: int, theta: float = 7.0):
    """Truncated Matrixsum approximant for cos(theta H) - i sin(theta H)."""
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    identity = h_op.identity(h_op.size)
    cos_ms = scaled_matrixsum(identity, float(jv(0, theta)))
    sin_ms = Matrixsum([])
    if degree == 0:
        return cos_ms.simplify()

    t_prev = identity
    t_curr = deepcopy(h_op)
    for order in range(1, degree + 1):
        t_order = t_curr
        coeff = 2.0 * float(jv(order, theta))
        if order % 2 == 0:
            coeff *= (-1.0) ** (order // 2)
            cos_ms = cos_ms.add(scaled_matrixsum(t_order, coeff))
        else:
            coeff *= (-1.0) ** ((order - 1) // 2)
            sin_ms = sin_ms.add(scaled_matrixsum(t_order, coeff))
        if order != degree:
            t_next = scaled_matrixsum(h_op.mul(t_curr), 2.0).add(scaled_matrixsum(t_prev, -1.0))
            t_prev, t_curr = t_curr, t_next.simplify()

    sin_ms.mul_coeffs(-1j)
    return cos_ms.add(sin_ms).simplify()


def run_jacobi_anger_lcu_toy(degrees=(4, 6, 8, 10, 12, 14, 16, 18, 20, 22), theta=7.0):
    cx_count = []
    u3_count = []
    cx_count_lcu = []
    u3_count_lcu = []

    for degree in degrees:
        h_terms = [("X", 1.0), ("Y", 1.0), ("Z", 1.0)]
        h_eff = Lindbladian(h_terms, []).H

        _, qc, _ = qsvt_Hamiltonian(h_eff, theta, degree, opt="No")
        lcu_ms = jacobi_anger_trig_matrixsum(h_eff, degree=degree, theta=theta)
        print(lcu_ms)
        lcu_circ = BlockEncoding(lcu_ms).circuit(opt="No")

        tqc_lcu = transpile(lcu_circ, basis_gates=["cx", "u3"], optimization_level=1)
        tqc_qsvt = transpile(qc, basis_gates=["cx", "u3"], optimization_level=1)
        print(tqc_lcu.count_ops())

        cx_count.append(int(tqc_qsvt.count_ops().get("cx", 0)))
        u3_count.append(int(tqc_qsvt.count_ops().get("u3", 0)))
        cx_count_lcu.append(int(tqc_lcu.count_ops().get("cx", 0)))
        u3_count_lcu.append(int(tqc_lcu.count_ops().get("u3", 0)))

    print("qsvt cx:", cx_count)
    print("lcu cx:", cx_count_lcu)
    return {
        "degrees": list(degrees),
        "qsvt_cx": cx_count,
        "qsvt_u3": u3_count,
        "lcu_cx": cx_count_lcu,
        "lcu_u3": u3_count_lcu,
    }


def candidate_pool_cond_random_oplist(num_qubits: int):
    pool = set()
    paulis = "XYZ"
    for t in range(1, num_qubits // 2 + 1):
        middle_len = num_qubits - 2 * t
        for t0 in range(0, t + 1):
            if middle_len == 0:
                pool.add("I" * num_qubits)
                continue
            for mid in product(paulis, repeat=middle_len):
                label = (2 * t0) * "I" + "".join(mid) + (2 * (t - t0)) * "I"
                pool.add(label)
    return sorted(pool)


def gene_cond_random_oplist(num_qubits: int, num_terms: int, seed=10, strict=False):
    pool = candidate_pool_cond_random_oplist(num_qubits)
    max_unique = len(pool)
    if max_unique == 0:
        return []
    if num_terms > max_unique:
        msg = f"requested num={num_terms} exceeds max unique={max_unique} for qubit={num_qubits}."
        if strict:
            raise ValueError(msg)
        print(f"[gene_cond_random_oplist] {msg} Use num={max_unique} instead.")

    rng = random.Random(seed)
    selected = rng.sample(pool, min(num_terms, max_unique))
    coeff = -0.2
    return [(label, coeff) for label in selected]


def build_trotter(num_qubits, pauli_terms, time=0.1, epsilon=1e-4):
    basis_gates = ["cx", "u3"]
    h_list = SparsePauliOp.from_list(pauli_terms)
    op = PauliEvolutionGate(h_list, time=time)
    qc = QuantumCircuit(num_qubits)
    qc.append(op, qc.qubits)
    qc = transpile(qc, basis_gates=basis_gates, optimization_level=3)
    gate_details, count2q, count1q = count_1q2q_gates(qc)

    h_ms = Lindbladian(pauli_terms, []).H
    commutator_scaling = nested_commutator(h_ms)
    min_steps = int(commutator_scaling * time**2 / epsilon)
    count2q_expected = min_steps * count2q
    count1q_expected = min_steps * count1q

    count2q_paulihedral = count2q_expected * (93885 / 133014)
    count1q_paulihedral = count1q_expected * (53241 / 93019)

    scale_factor_qdrift = sum(abs(coeff) for _, coeff in h_ms.instances) * time
    min_repeat_qdrift = 2 * int(np.ceil((scale_factor_qdrift) ** 2 / epsilon))
    print("qdrift repeats:", min_repeat_qdrift)
    print("trotter steps:", min_steps)
    op_qd = qdrift_hamiltonian(
        h_ms,
        time,
        reps=100,
        basis_gates=basis_gates,
        optimization_level=1,
    )
    _, count2q_qd, count1q_qd = count_1q2q_gates(op_qd)
    count2q_qd = min_repeat_qdrift * count2q_qd / 100
    count1q_qd = min_repeat_qdrift * count1q_qd / 100
    return (
        gate_details,
        count2q_expected,
        count1q_expected,
        count2q_paulihedral,
        count1q_paulihedral,
        count2q_qd,
        count1q_qd,
    )


def build_qsvt_single(h_op: Matrixsum, degree: int):
    basis_gates_basic = ["cx", "u3"]
    be = BlockEncoding(h_op)
    qc_basic = be.circuit(opt="Matrix-order")

    subnorm_fac = sum(abs(coeff) for _, coeff in h_op.instances)
    time = 0.1
    cos_phi_value, _ = get_adaptive_qsp_phases(lambda x: np.cos(subnorm_fac * time * x), degree)
    sin_phi_value, _ = get_adaptive_qsp_phases(lambda x: np.sin(subnorm_fac * time * x), degree - 1)
    print(cos_phi_value)

    cos_phi_value[0] -= np.pi / 4
    for i in range(1, len(cos_phi_value) - 1):
        cos_phi_value[i] += np.pi / 2
    cos_phi_value[-1] += 3 * np.pi / 4

    sin_phi_value[0] -= np.pi / 4
    for i in range(1, len(sin_phi_value)):
        sin_phi_value[i] += np.pi / 2
    sin_phi_value[-1] += 3 * np.pi / 4

    sys_size = h_op.size
    ctrl_size = qc_basic.num_qubits - sys_size
    data_indices = list(range(2, qc_basic.num_qubits + 2))
    ctrl_indices = list(range(2, 2 + ctrl_size))
    qsvt_block = Gate(name="qsvt_block", num_qubits=qc_basic.num_qubits, params=[])
    target_with_block = Target.from_configuration(
        basis_gates=["cx", "u3", "qsvt_block"],
        custom_name_mapping={"qsvt_block": qsvt_block},
    )

    def build_controlled_placeholder(phi_values):
        qc = QuantumCircuit(qc_basic.num_qubits + 2)
        qc.ch(0, 1)
        for i, phi in enumerate(reversed(phi_values)):
            if ctrl_size > 0:
                for ctrl in ctrl_indices:
                    qc.cx(0, ctrl)
                qc.mcx([0] + ctrl_indices, 1)
            qc.crz(2 * phi, 0, 1)
            if ctrl_size > 0:
                qc.mcx([0] + ctrl_indices, 1)
                for ctrl in ctrl_indices:
                    qc.cx(0, ctrl)
            if i != len(phi_values) - 1:
                qc.append(qsvt_block, data_indices)
        qc.ch(0, 1)
        return qc

    tqc_cos_placeholder = transpile(
        build_controlled_placeholder(cos_phi_value),
        target=target_with_block,
        optimization_level=3,
    )
    tqc_sin_placeholder = transpile(
        build_controlled_placeholder(sin_phi_value),
        target=target_with_block,
        optimization_level=3,
    )

    gate_basic_ctrl = qc_basic.to_gate().control(1)
    qc_basic_ctrl = QuantumCircuit(qc_basic.num_qubits + 1)
    qc_basic_ctrl.append(gate_basic_ctrl, qc_basic_ctrl.qubits)
    tqc_basic_ctrl = transpile(qc_basic_ctrl, basis_gates=basis_gates_basic, optimization_level=3)

    total_phase_count = len(cos_phi_value) + len(sin_phi_value)
    basic_ops = tqc_basic_ctrl.count_ops()
    scaled_cx = int(basic_ops.get("cx", 0)) * total_phase_count
    scaled_u3 = int(basic_ops.get("u3", 0)) * total_phase_count

    cos_ops = tqc_cos_placeholder.count_ops()
    sin_ops = tqc_sin_placeholder.count_ops()
    overhead_cx = int(cos_ops.get("cx", 0)) + int(sin_ops.get("cx", 0))
    overhead_u3 = int(cos_ops.get("u3", 0)) + int(sin_ops.get("u3", 0))
    count2q = scaled_cx + overhead_cx
    count1q = scaled_u3 + overhead_u3

    gate_details = {
        "scaled_qc_basic_ctrl_cx": scaled_cx,
        "scaled_qc_basic_ctrl_u3": scaled_u3,
        "placeholder_overhead_cx": overhead_cx,
        "placeholder_overhead_u3": overhead_u3,
        "total_phase_count": int(total_phase_count),
        "count2q_total": int(count2q),
        "count1q_total": int(count1q),
        "placeholder_ops_cos": dict(cos_ops),
        "placeholder_ops_sin": dict(sin_ops),
    }
    return gate_details, count2q, count1q


def paulihedral_gate_counts(nq_list=(4, 6, 8, 10), seed_base=571):
    results = {}
    for nq in nq_list:
        num_terms = min(5 * nq**2, int(4**nq))
        if nq == 4:
            num_terms = 19
        pauli_terms = gene_cond_random_oplist(nq, num_terms, seed=seed_base)
        scale_factor = num_terms * 0.1 * 0.2
        degree = 2
        while True:
            bound = 2 * jv(degree + 2, scale_factor) + 2 * jv(degree + 1, scale_factor)
            if 0 < bound < 3e-4:
                break
            degree += 2
        err = 2 * jv(degree + 2, scale_factor) + 2 * jv(degree + 1, scale_factor)

        print("degree:", degree)
        gate_details, count2q, count1q, count2q_ph, count1q_ph, count2q_qd, count1q_qd = build_trotter(
            nq,
            pauli_terms,
            time=0.1,
            epsilon=err,
        )
        results[nq] = {
            "num_terms": num_terms,
            "count2q": count2q,
            "count1q": count1q,
            "count2q_ph": count2q_ph,
            "count1q_ph": count1q_ph,
            "count2q_qd": count2q_qd,
            "count1q_qd": count1q_qd,
            "gate_details": gate_details,
            "epsilon": err,
            "deg": degree,
        }
    return results


def qsvt_gate_counts(degrees: list[int], nq_list=(4, 6, 8, 10), seed_base=100):
    results = {}
    for idx, nq in enumerate(nq_list):
        num_terms = min(5 * nq**2, int(4**nq))
        pauli_terms = gene_cond_random_oplist(nq, num_terms, seed=seed_base + nq)
        h_ms = Lindbladian(pauli_terms, []).H
        gate_details, count2q, count1q = build_qsvt_single(h_ms, degrees[idx])
        results[nq] = {
            "num_terms": num_terms,
            "count2q": count2q,
            "count1q": count1q,
            "gate_details": gate_details,
        }
        print(f"nq={nq}, num_terms={num_terms}, 2q={count2q}, 1q={count1q}")
    return results


def plot_ratio(paulihedral_counts, qsvt_counts, nq_list=(4, 6, 8, 10)):
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
    })

    rows = []
    for nq in nq_list:
        base_2q = paulihedral_counts[nq]["count2q"]
        base_1q = paulihedral_counts[nq]["count1q"]
        rows.append({"nq": str(nq), "Method": "Paulihedral", "Gate": "2Q", "Ratio": paulihedral_counts[nq]["count2q_ph"] / base_2q})
        rows.append({"nq": str(nq), "Method": "Paulihedral", "Gate": "1Q", "Ratio": paulihedral_counts[nq]["count1q_ph"] / base_1q})
        rows.append({"nq": str(nq), "Method": "QDrift", "Gate": "2Q", "Ratio": paulihedral_counts[nq]["count2q_qd"] / base_2q})
        rows.append({"nq": str(nq), "Method": "QDrift", "Gate": "1Q", "Ratio": paulihedral_counts[nq]["count1q_qd"] / base_1q})
        rows.append({"nq": str(nq), "Method": "QSVT", "Gate": "2Q", "Ratio": qsvt_counts[nq]["count2q"] / base_2q})
        rows.append({"nq": str(nq), "Method": "QSVT", "Gate": "1Q", "Ratio": qsvt_counts[nq]["count1q"] / base_1q})

    df_ratio = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, gate_type in zip(axes, ["2Q", "1Q"]):
        sub = df_ratio[df_ratio["Gate"] == gate_type]
        bp = sns.barplot(data=sub, x="nq", y="Ratio", hue="Method", ax=ax)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=2.0, label="Baseline (Trotter=1)")
        ax.set_title(f"{gate_type} Gate Count Ratio vs Trotter Baseline", fontsize=19)
        ax.set_xlabel("Number of Qubits", fontsize=17)
        ax.set_ylabel("Ratio to Trotter Baseline", fontsize=17)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Method", fontsize=15)
        for patch in bp.patches:
            patch.set_linewidth(1.5)
            patch.set_edgecolor("black")

    plt.tight_layout()
    out_dir = RESULTS_DIR / "comp_qsvt_paulihedral"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comp_qsvt_trotter.svg"
    plt.savefig(out_path)
    print(f"Saved figure to: {out_path}")


def run_paulihedral_comparison(nq_list=(4, 6, 8, 10), make_plot=False):
    paulihedral_counts = paulihedral_gate_counts(nq_list=nq_list)
    degrees = [paulihedral_counts[nq]["deg"] for nq in nq_list]
    qsvt_counts = qsvt_gate_counts(nq_list=nq_list, degrees=degrees)
    if make_plot:
        plot_ratio(paulihedral_counts, qsvt_counts, nq_list=nq_list)
    return paulihedral_counts, qsvt_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy", "paulihedral"], default="paulihedral")
    parser.add_argument("--nq-list", type=int, nargs="*", default=[4, 6, 8, 10])
    parser.add_argument("--degrees", type=int, nargs="*", default=[4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    parser.add_argument("--theta", type=float, default=7.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.mode == "toy":
        return run_jacobi_anger_lcu_toy(degrees=tuple(args.degrees), theta=args.theta)
    return run_paulihedral_comparison(nq_list=tuple(args.nq_list), make_plot=args.plot)


if __name__ == "__main__":
    main()
