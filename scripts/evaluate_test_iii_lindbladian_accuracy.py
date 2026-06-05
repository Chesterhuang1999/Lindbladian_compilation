"""Test III from ``evaluate.ipynb``: Channel-LCU Lindbladian accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluate_common import (
    RESULTS_DIR,
    Lindblad_to_channel,
    Lindbladian,
    build_periodic_tfim_lindbladian_pauli,
    channel_ensemble,
    channel_to_LCU,
    construct_qobj_lind,
    simulate_circuit,
    simulate_lindblad,
)
import matplotlib.pyplot as plt
from qiskit.quantum_info import DensityMatrix, Statevector
from qutip import basis, tensor


def simulate_case_lcu_vs_baseline(num_qubits, delta_t_values, gamma):
    h_terms, l_terms = build_periodic_tfim_lindbladian_pauli(num_qubits, gamma)
    tfim_lind = Lindbladian(h_terms, l_terms)
    print("Block-encoding (Pauli Norm):", round(tfim_lind.pauli_norm(), 6))

    result = {
        "delta_t": [],
        "lcu_err_vs_qutip": [],
        "op_norm_bound": [],
        "sim_time_lcu": [],
        "sim_time_qutip": [],
    }

    h_qobj, l_qobj_list = construct_qobj_lind(tfim_lind, dim_sys=num_qubits)
    ini_state_qutip = tensor([basis(2, 0)] * num_qubits)

    import time

    for delta_t in delta_t_values:
        channel_lind, _, _ = Lindblad_to_channel(tfim_lind, float(delta_t))
        channel_lind = channel_lind.channels[0][1]
        op_norm_bound = 5 * (float(delta_t) * tfim_lind.operator_norm()) ** 2
        lcu, qubit_regs = channel_to_LCU(channel_ensemble([channel_lind]))
        qubit_size = [len(qreg) for qreg in qubit_regs]
        ini_state = Statevector.from_label("0" * sum(qubit_size))

        start = time.time()
        system_density_final = simulate_circuit(lcu, ini_state=ini_state, qubit_regs=qubit_regs)
        time1 = time.time()
        result["sim_time_lcu"].append(time1 - start)

        if num_qubits <= 8:
            baseline_density_qutip = simulate_lindblad(
                h_qobj,
                l_qobj_list,
                ini_state_qutip,
                float(delta_t),
                5,
            )
            time2 = time.time()
            result["sim_time_qutip"].append(time2 - time1)
            lcu_err = (
                np.linalg.norm(system_density_final - DensityMatrix(baseline_density_qutip), ord="nuc")
                / 2
            )
            result["delta_t"].append(float(delta_t))
            result["lcu_err_vs_qutip"].append(float(lcu_err))
            result["op_norm_bound"].append(float(op_norm_bound))
            print(f"delta_t={delta_t:.6f} | LCU err: {lcu_err:.8f} | bound: {op_norm_bound:.8f}")
        else:
            print(
                f"delta_t={delta_t:.6f} | qubit_size = {qubit_size} | "
                f"LCU simulation done; skipping QuTiP comparison for N={num_qubits}."
            )

    return result


def plot_error_rates(n_list, delta_t_rec, error_rate, pauli_bound):
    plt.figure(figsize=(10, 6))
    for num_qubits in n_list:
        if num_qubits not in error_rate:
            continue
        plt.plot(delta_t_rec, error_rate[num_qubits], marker="o", label=f"N={num_qubits} LCU error", linewidth=3)
        plt.plot(
            delta_t_rec,
            pauli_bound[num_qubits],
            marker="s",
            linestyle="--",
            label=f"N={num_qubits} Error Bound",
            linewidth=2,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$1/\Delta t$", fontsize=15)
    plt.ylabel("Error Rate", fontsize=15)
    plt.title("Error Rate vs. Evolution time for Different Qubit Numbers", fontsize=18)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    out_dir = RESULTS_DIR / "accuracy_cLCU"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cLCU_errs.svg"
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


def plot_sim_times(n_list, delta_t_rec, sim_time_lcu, sim_time_qutip):
    plt.figure(figsize=(10, 6))
    for num_qubits in n_list:
        plt.plot(delta_t_rec[: len(sim_time_lcu[num_qubits])], sim_time_lcu[num_qubits], marker="o", label=f"N={num_qubits} LCU time", linewidth=3)
        if sim_time_qutip.get(num_qubits):
            plt.plot(delta_t_rec[: len(sim_time_qutip[num_qubits])], sim_time_qutip[num_qubits], marker="s", linestyle="--", label=f"N={num_qubits} QuTiP time", linewidth=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$1/\Delta t$", fontsize=15)
    plt.ylabel("Simulation Time (s)", fontsize=15)
    plt.title("Simulation Time Comparison: LCU vs QuTiP", fontsize=18)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    out_dir = RESULTS_DIR / "accuracy_cLCU"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sim_time_compare.svg"
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", type=int, nargs="*", default=[8, 9])
    parser.add_argument("--delta-t-rec", type=float, nargs="*", default=list(np.linspace(10, 100, 10)))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    gamma = np.sqrt(0.1) / 2
    delta_t_values = 1.0 / np.asarray(args.delta_t_rec, dtype=float)
    error_rate = {}
    pauli_bound = {}
    sim_time_lcu = {}
    sim_time_qutip = {}

    for num_qubits in args.n_list:
        values = delta_t_values[:1] if num_qubits >= 7 else delta_t_values
        case_res = simulate_case_lcu_vs_baseline(num_qubits, values, gamma)
        if num_qubits < 7:
            error_rate[num_qubits] = case_res["lcu_err_vs_qutip"]
            pauli_bound[num_qubits] = case_res["op_norm_bound"]
        sim_time_lcu[num_qubits] = case_res["sim_time_lcu"]
        sim_time_qutip[num_qubits] = case_res["sim_time_qutip"]

    print("Sim_time_lcu:", sim_time_lcu)
    print("Sim_time_qutip:", sim_time_qutip)
    print("Error_rate:", error_rate)
    print("Pauli_bound:", pauli_bound)

    if args.plot:
        plot_error_rates(args.n_list, args.delta_t_rec, error_rate, pauli_bound)
        plot_sim_times(args.n_list, args.delta_t_rec, sim_time_lcu, sim_time_qutip)

    return {
        "error_rate": error_rate,
        "pauli_bound": pauli_bound,
        "sim_time_lcu": sim_time_lcu,
        "sim_time_qutip": sim_time_qutip,
    }


if __name__ == "__main__":
    main()
