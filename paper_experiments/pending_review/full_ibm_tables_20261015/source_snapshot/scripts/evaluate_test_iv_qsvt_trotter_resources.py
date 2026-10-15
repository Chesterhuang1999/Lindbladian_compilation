"""Test IV from ``evaluate.ipynb``: QSVT vs Trotterization/QDrift resources."""

from __future__ import annotations

import argparse

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, Statevector, random_pauli_list
from scipy.linalg import expm
from scipy.special import jv

from evaluate_common import (
    DATA_DIR,
    Lindbladian,
    export_openqasm3_baseline,
    gate_count_summary,
    nested_commutator,
    normalize,
    qdrift_hamiltonian,
    qsvt_Hamiltonian,
    sample_unique_random_pauli_labels,
    simulate_circuit_statevec,
)


def run_fixed_n_random_resource_comparison(
    num_qubits=3,
    set_size=50,
    num_terms=6,
    delta_t=0.1,
    degree=4,
):
    scale_factor = num_terms * delta_t
    epsilon = 2 * jv(degree + 2, scale_factor) + 2 * jv(degree + 1, scale_factor)
    print("epsilon:", epsilon)

    qsvt_gate_count_set = {"cx": [], "u3": []}
    qsvt_gate_count_set_old = {"cx": [], "u3": []}
    qdrift_gate_count_set = {"cx": [], "u3": []}
    trotter_gate_count_set = {"cx": [], "u3": []}
    min_repeat = 1

    for _ in range(set_size):
        random_pauli = random_pauli_list(num_qubits=num_qubits, size=num_terms, phase=False)
        h_terms = [(ms.to_label(), -1.0) for ms in random_pauli]
        random_lind = Lindbladian(h_terms, [])
        h_eff = random_lind.H
        total_norm = nested_commutator(h_eff)

        _, qc, _ = qsvt_Hamiltonian(h_eff, delta_t, deg=degree, opt="No")
        _, qc_opt, _ = qsvt_Hamiltonian(h_eff, delta_t, deg=degree, opt="Matrix-order")
        min_repeat = int(total_norm * delta_t**2 / epsilon)
        sum_coeffs = sum(abs(coeff) for _, coeff in h_eff.instances)
        min_repeat_qdrift = 2 * int(np.ceil((sum_coeffs * delta_t) ** 2 / epsilon))

        op = PauliEvolutionGate(SparsePauliOp.from_list(h_terms), time=delta_t)
        qc_trotter = QuantumCircuit(num_qubits)
        qc_trotter.append(op, range(num_qubits))
        qc_qdrift = qdrift_hamiltonian(h_eff, delta_t, reps=min_repeat_qdrift)

        trotter_gate_count = gate_count_summary(qc_trotter, basis_gates=["cx", "u3"], optimization_level=3)
        qdrift_gate_count = gate_count_summary(qc_qdrift, basis_gates=["cx", "u3"], optimization_level=3)
        qsvt_gate_count_old = gate_count_summary(qc, basis_gates=["cx", "u3"], optimization_level=3)
        qsvt_gate_count = gate_count_summary(qc_opt, basis_gates=["cx", "u3"], optimization_level=3)

        trotter_gate_count_set["cx"].append(trotter_gate_count["cx"])
        trotter_gate_count_set["u3"].append(trotter_gate_count["u3"])
        qdrift_gate_count_set["cx"].append(qdrift_gate_count["cx"])
        qdrift_gate_count_set["u3"].append(qdrift_gate_count["u3"])
        qsvt_gate_count_set["cx"].append(qsvt_gate_count["cx"])
        qsvt_gate_count_set["u3"].append(qsvt_gate_count["u3"])
        qsvt_gate_count_set_old["cx"].append(qsvt_gate_count_old["cx"])
        qsvt_gate_count_set_old["u3"].append(qsvt_gate_count_old["u3"])

    print(
        "Average Trotter gate counts set:",
        "cx:",
        min_repeat * np.mean(trotter_gate_count_set["cx"]),
        "u3:",
        min_repeat * np.mean(trotter_gate_count_set["u3"]),
    )
    print(
        "Average QDrift gate counts set:",
        "cx:",
        np.mean(qdrift_gate_count_set["cx"]),
        "u3:",
        np.mean(qdrift_gate_count_set["u3"]),
    )
    print(
        "qsvt gate counts set with pauli-level optimization:",
        "cx:",
        np.mean(qsvt_gate_count_set["cx"]),
        "u3:",
        np.mean(qsvt_gate_count_set["u3"]),
    )
    print(
        "qsvt gate counts with no optimization:",
        "cx:",
        np.mean(qsvt_gate_count_set_old["cx"]),
        "u3:",
        np.mean(qsvt_gate_count_set_old["u3"]),
    )
    return {
        "trotter": trotter_gate_count_set,
        "qdrift": qdrift_gate_count_set,
        "qsvt_matrix_order": qsvt_gate_count_set,
        "qsvt_no": qsvt_gate_count_set_old,
    }


def build_qsvt_resource_baseline_circuit(
    num_qubits: int = 3,
    num_terms: int = 6,
    delta_t: float = 0.1,
    degree: int = 4,
    seed: int = 20260314,
):
    random_pauli = random_pauli_list(
        num_qubits=num_qubits,
        size=num_terms,
        phase=False,
        seed=seed,
    )
    h_terms = [(ms.to_label(), -1.0) for ms in random_pauli]
    random_lind = Lindbladian(h_terms, [])
    _, qc, _ = qsvt_Hamiltonian(random_lind.H, delta_t, deg=degree, opt="No")
    return qc


def export_qsvt_resource_baseline_openqasm3(
    num_qubits: int = 3,
    num_terms: int = 6,
    delta_t: float = 0.1,
    degree: int = 4,
    seed: int = 20260314,
    out_path=None,
) -> dict[str, object]:
    if out_path is None:
        out_path = DATA_DIR / f"evaluate_test_iv_qsvt_no_n{num_qubits}_baseline.qasm"
    qc = build_qsvt_resource_baseline_circuit(
        num_qubits=num_qubits,
        num_terms=num_terms,
        delta_t=delta_t,
        degree=degree,
        seed=seed,
    )
    return export_openqasm3_baseline(qc, out_path)


def run_averaged_gate_counts(
    n_list=(3, 4),
    num_cases=3,
    delta_t=0.1,
):
    trotter_gate_count_set = {}
    qdrift_gate_count_set = {}
    qsvt_gate_count_set = {}

    for num_qubits in n_list:
        print(f"\n=== Running N = {num_qubits} ===")
        num_terms = min(4**num_qubits, num_qubits * num_qubits - 2)
        degree = 4
        scale_factor = num_terms * delta_t
        while True:
            epsilon = 2 * jv(degree + 2, scale_factor) + 2 * jv(degree + 1, scale_factor)
            if 0 < epsilon < 5e-4:
                break
            degree += 2
        print(f"Chosen degree: {degree}, epsilon: {epsilon}")

        sum_trotter_cx = 0.0
        sum_trotter_u3 = 0.0
        sum_qdrift_cx = 0.0
        sum_qdrift_u3 = 0.0
        sum_qsvt_cx = 0.0
        sum_qsvt_u3 = 0.0
        valid_cases = 0

        for case_id in range(num_cases):
            print(f"Processing case {case_id + 1}/{num_cases}")
            random_pauli = sample_unique_random_pauli_labels(num_qubits=num_qubits, size=num_terms)
            h_terms = [(p, -1.0) for p in random_pauli]
            random_lind = Lindbladian(h_terms, [])
            h_eff = random_lind.H

            coeff_sum = sum(abs(coeff) for _, coeff in h_eff.instances)
            total_norm = nested_commutator(h_eff)
            _, qc, _ = qsvt_Hamiltonian(h_eff, delta_t, deg=degree, opt="No")
            _, qc_opt, _ = qsvt_Hamiltonian(h_eff, delta_t, deg=degree, opt="Matrix-order")

            h_evo = Operator(expm(-1j * h_eff.eff_op() * delta_t))
            initial_qsvt = Statevector.from_label("+" * num_qubits + "0" * (qc_opt.num_qubits - num_qubits))
            initial_baseline = Statevector.from_label("+" * num_qubits)
            final_baseline = normalize(initial_baseline.evolve(h_evo))
            final_sys = simulate_circuit_statevec(
                qc_opt,
                initial_qsvt,
                None,
                reg_sizes=[0, qc.num_qubits - num_qubits, num_qubits],
            )
            err = np.linalg.norm(DensityMatrix(final_sys) - DensityMatrix(final_baseline), ord="nuc") / 2
            print(f" error={err}")

            if err < 1e-3:
                valid_cases += 1
                epsilon = min(epsilon, 1.1 * err)
                min_repeat_t = int(total_norm * delta_t**2 / epsilon)
                min_repeat_qd = 2 * int(np.ceil((coeff_sum * delta_t) ** 2 / epsilon))
                qc_qdrift = qdrift_hamiltonian(h_eff, delta_t, reps=min_repeat_qd)

                op = PauliEvolutionGate(SparsePauliOp.from_list(h_terms), time=delta_t)
                qc_trotter = QuantumCircuit(num_qubits)
                qc_trotter.append(op, range(num_qubits))

                trotter_gate_count = gate_count_summary(qc_trotter, basis_gates=["cx", "u3"], optimization_level=3)
                qdrift_gate_count = gate_count_summary(qc_qdrift, basis_gates=["cx", "u3"], optimization_level=3)
                qsvt_gate_count = gate_count_summary(qc_opt, basis_gates=["cx", "u3"], optimization_level=3)

                sum_trotter_cx += min_repeat_t * trotter_gate_count["cx"]
                sum_trotter_u3 += min_repeat_t * trotter_gate_count["u3"]
                sum_qdrift_cx += qdrift_gate_count["cx"]
                sum_qdrift_u3 += qdrift_gate_count["u3"]
                sum_qsvt_cx += qsvt_gate_count["cx"]
                sum_qsvt_u3 += qsvt_gate_count["u3"]

        denom = valid_cases if valid_cases > 0 else 1
        trotter_gate_count_set[num_qubits] = {
            "cx": sum_trotter_cx / denom,
            "u3": sum_trotter_u3 / denom,
            "valid_cases": valid_cases,
            "num_cases": num_cases,
        }
        qdrift_gate_count_set[num_qubits] = {
            "cx": sum_qdrift_cx / denom,
            "u3": sum_qdrift_u3 / denom,
            "valid_cases": valid_cases,
            "num_cases": num_cases,
        }
        qsvt_gate_count_set[num_qubits] = {
            "cx": sum_qsvt_cx / denom,
            "u3": sum_qsvt_u3 / denom,
            "valid_cases": valid_cases,
            "num_cases": num_cases,
        }

    print("\n=== Averaged gate counts by N ===")
    print("Trotter:", trotter_gate_count_set)
    print("QDrift:", qdrift_gate_count_set)
    print("QSVT (Matrix-order):", qsvt_gate_count_set)
    return trotter_gate_count_set, qdrift_gate_count_set, qsvt_gate_count_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "averaged"], default="averaged")
    parser.add_argument("--n-list", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--set-size", type=int, default=50)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--degree", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "fixed":
        return run_fixed_n_random_resource_comparison(
            num_qubits=args.n_list[0],
            set_size=args.set_size,
            delta_t=args.delta_t,
            degree=args.degree,
        )
    return run_averaged_gate_counts(
        n_list=tuple(args.n_list),
        num_cases=args.num_cases,
        delta_t=args.delta_t,
    )


if __name__ == "__main__":
    main()
