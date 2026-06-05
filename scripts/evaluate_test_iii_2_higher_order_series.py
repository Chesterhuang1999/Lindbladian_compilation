"""Test III.2 from ``evaluate.ipynb``: higher-order series expansion accuracy."""

from __future__ import annotations

import argparse

import numpy as np

from evaluate_common import (
    RESULTS_DIR,
    Lindbladian,
    build_tfim_decay_lindbladian_pauli,
    build_tfim_decay_qutip_reference,
    construct_circuit_coherent,
    construct_higher_order_circuit,
    construct_qobj_lind,
    get_superop_qutip,
    simulate_lindblad,
)
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector, random_statevector
from qiskit_aer import AerSimulator
from qutip import basis
from series_expansion import simulate_circuit_statevec


def run_decay_only_higher_order_accuracy():
    h_terms = []
    total_time = 0.1
    v = 0.5
    nlist = [10, 20, 40, 100]
    lambdas = [1.0, 2.0, 3.0]
    k_values = [1, 2]
    k1 = 3
    err_sets = np.zeros((len(lambdas), len(k_values), len(nlist)))
    rescaling = [np.sqrt(n / 10) for n in nlist]

    for lambda_idx, lambda_value in enumerate(lambdas):
        for n_idx, n_value in enumerate(nlist):
            print(f"Test for evolution time dt = {1 / n_value}:")
            scaling = rescaling[n_idx]
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
            decay_lind = Lindbladian(h_terms, l_terms)
            print(2 * decay_lind.pauli_norm())

            ini_state = Statevector.from_label("0")
            h_qobj, l_qobj_list = construct_qobj_lind(decay_lind, decay_lind.__size__())
            ini_dens_qobj = basis(2, 0) @ basis(2, 0).dag()
            final_dens = get_superop_qutip(h_qobj, l_qobj_list, total_time)(ini_dens_qobj).data_as("ndarray")
            simulator = AerSimulator(method="statevector")

            for k_idx, k_order in enumerate(k_values):
                qc, reg_sizes, coeff_sum_sq, sel_state = construct_higher_order_circuit(
                    decay_lind,
                    k_order,
                    k1,
                    total_time,
                    k1,
                    opt="No",
                )
                qc = transpile(qc, simulator, optimization_level=2)
                final_state_sys = coeff_sum_sq * simulate_circuit_statevec(qc, ini_state, sel_state, reg_sizes)
                final_state_sys = final_state_sys / np.trace(final_state_sys)
                err = np.linalg.norm(final_state_sys - final_dens, ord="nuc") / 2
                err_rescaled = err * 10 * scaling**2
                err_sets[lambda_idx, k_idx, n_idx] = err_rescaled
                print(err_rescaled)

    print(err_sets)
    return err_sets


def run_tfim_decay_higher_order_accuracy(
    n_list=(2, 4),
    piece_list=(10, 20, 50, 100),
    k_list=(1, 2),
    gamma_list=(0.1, 1.0),
    delta=1.0,
    coupling=1.0,
    k1=3,
    q=3,
):
    err_sets = {
        str(num_qubits): np.zeros((len(gamma_list), len(k_list), len(piece_list)))
        for num_qubits in n_list
    }

    for num_qubits in n_list:
        rng_init = np.random.default_rng(1)
        ini_state = random_statevector(2**num_qubits, seed=rng_init)
        from qutip import Qobj

        ini_state_qobj = Qobj(
            np.asarray(ini_state.data, dtype=complex).reshape((2**num_qubits, 1)),
            dims=[[2] * num_qubits, [1] * num_qubits],
        )

        for gamma_idx, gamma in enumerate(gamma_list):
            for piece_idx, piece in enumerate(piece_list):
                time_step = 1.0 / piece
                print(f"\n=== Test for N = {num_qubits}, gamma={gamma}, piece={piece} ===")

                h_terms, l_terms = build_tfim_decay_lindbladian_pauli(
                    num_qubits,
                    delta,
                    coupling,
                    gamma,
                )
                tfim_lind = Lindbladian(h_terms, l_terms)
                h_qobj, l_qobj_list = build_tfim_decay_qutip_reference(
                    num_qubits,
                    delta,
                    coupling,
                    gamma,
                )
                baseline_superop = get_superop_qutip(h_qobj, l_qobj_list, time_step)
                final_dens_baseline = baseline_superop(ini_state_qobj).data_as("ndarray")
                _ = simulate_lindblad(h_qobj, l_qobj_list, ini_state_qobj, time_step, r=1)

                for k_idx, k_order in enumerate(k_list):
                    qc_opt, reg_sizes, coeff_sum_sq, sel_state = construct_higher_order_circuit(
                        tfim_lind,
                        k_order,
                        q,
                        time_step,
                        k1,
                        opt="Matrix-order",
                    )
                    print(qc_opt.num_qubits)
                    final_state_sys = coeff_sum_sq * simulate_circuit_statevec(
                        qc_opt,
                        ini_state,
                        sel_state,
                        reg_sizes,
                    )
                    final_state_sys = final_state_sys / np.trace(final_state_sys)
                    err = np.linalg.norm(final_state_sys - final_dens_baseline, ord="nuc") / 2
                    err_sets[str(num_qubits)][gamma_idx, k_idx, piece_idx] = err * piece
                    print(f"K={k_order}, K1={k1}, q={q} -> err={err * piece}")

    print("\nErr_sets:")
    print(err_sets)
    return err_sets


def run_coherent_opt_sanity_check(
    num_qubits=2,
    delta=1.0,
    coupling=1.0,
    gamma=0.1,
    k1=3,
    time_step=0.1,
):
    h_terms, l_terms = build_tfim_decay_lindbladian_pauli(num_qubits, delta, coupling, gamma)
    tfim_lind = Lindbladian(h_terms, l_terms)
    h_eff = tfim_lind.effective_H()

    qc_no, _ = construct_circuit_coherent(h_eff, k1, time_step, opt="No")
    qc_opt, _ = construct_circuit_coherent(h_eff, k1, time_step, opt="Matrix-order")
    simulator = AerSimulator(method="statevector")
    qc_no = transpile(qc_no, simulator, optimization_level=2)
    qc_opt = transpile(qc_opt, simulator, optimization_level=2)
    print(qc_no.count_ops())
    print(qc_opt.count_ops())

    ini_sys = Statevector.from_label("+" * num_qubits)
    ctrl_size = qc_no.num_qubits - num_qubits
    sel_state = Statevector([1.0])
    final_no = DensityMatrix(
        simulate_circuit_statevec(qc_no, ini_sys, sel_state=sel_state, reg_sizes=[0, ctrl_size, num_qubits])
    )
    final_opt = DensityMatrix(
        simulate_circuit_statevec(qc_opt, ini_sys, sel_state=sel_state, reg_sizes=[0, ctrl_size, num_qubits])
    )
    final_no = final_no / np.trace(final_no)
    final_opt = final_opt / np.trace(final_opt)
    err = np.linalg.norm(final_no - final_opt, ord="nuc") / 2
    print(f"Error between No opt and Matrix-order opt: {err}")
    return float(err)


def plot_decay_only_qmethod():
    new_color_list = [
        ["#B1DDDD", "#428F8F"],
        ["#FFCC6F", "#EA801C"],
        ["#DDA0DD", "#800080"],
    ]
    err_sets = np.array([
        [[0.0382864, 0.01364053, 0.00411233, 0.00073905],
         [0.02300232, 0.00749532, 0.00215805, 0.00037688]],
        [[0.15978667, 0.07657279, 0.02728107, 0.00546979],
         [0.11576498, 0.04600464, 0.01499064, 0.00284344]],
        [[0.29316662, 0.18522529, 0.07689294, 0.01710174],
         [0.26601572, 0.12184389, 0.04418769, 0.0090589]],
    ])
    lambdas = [1.0, 2.0, 3.0]
    nlist = [10, 20, 40, 100]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for lambda_idx, lambda_value in enumerate(lambdas):
        c1, c2 = new_color_list[lambda_idx][0], new_color_list[lambda_idx][1]
        ax.plot(nlist, err_sets[lambda_idx][0], "-o", markersize=6, label=rf"$\lambda_0={lambda_value}, order=1$", color=c1)
        ax.plot(nlist, err_sets[lambda_idx][1], "-s", markersize=6, label=rf"$\lambda_0={lambda_value}, order=2$", color=c2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T/\Delta t$", fontsize=12)
    ax.set_ylabel("Avg. error", fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    out_dir = RESULTS_DIR / "accuracy_HOexp"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / "decay_only_Qmethod.svg"
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved figure to: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-decay-only", action="store_true")
    parser.add_argument("--run-tfim-decay", action="store_true")
    parser.add_argument("--run-coherent-check", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    results = {}
    if args.run_decay_only:
        results["decay_only"] = run_decay_only_higher_order_accuracy()
    if args.run_tfim_decay:
        results["tfim_decay"] = run_tfim_decay_higher_order_accuracy()
    if args.run_coherent_check:
        results["coherent_check"] = run_coherent_opt_sanity_check()
    if args.plot:
        plot_decay_only_qmethod()
    if not any((args.run_decay_only, args.run_tfim_decay, args.run_coherent_check, args.plot)):
        parser.print_help()
    return results


if __name__ == "__main__":
    main()
