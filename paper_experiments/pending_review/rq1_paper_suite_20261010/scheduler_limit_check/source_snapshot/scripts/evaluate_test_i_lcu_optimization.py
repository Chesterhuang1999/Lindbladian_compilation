"""Test I from ``evaluate.ipynb``: block-encoding optimization gate counts."""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import YGate

from evaluate_common import (
    BlockEncoding,
    DATA_DIR,
    Lindblad_to_channel,
    Lindbladian,
    Matrixsum,
    PauliAtom,
    build_periodic_tfim_lindbladian_pauli,
    count_metrics,
    export_openqasm3_baseline,
    ratio,
)


def test_tfim_lcu_optimization(num_qubits: int, delta_t: float = 0.1):
    gamma = np.sqrt(0.1) / 2
    h_terms, l_terms = build_periodic_tfim_lindbladian_pauli(num_qubits, gamma)
    tfim_lind = Lindbladian(h_terms, l_terms)
    channel_lind, _, _ = Lindblad_to_channel(tfim_lind, delta_t)
    single_channel = channel_lind.channels[0][1]
    ms = single_channel[0]

    print(f"Test for case N = {num_qubits}:")

    metrics = {}
    for opt in ("No", "Ctrl-line", "Matrix-order"):
        be = BlockEncoding(ms)
        metrics[f"({opt})"] = count_metrics(be.circuit(opt=opt))

    qc_pauli = QuantumCircuit(2 * num_qubits + 1)
    for i in range(num_qubits):
        qc_pauli.p(np.pi, i)
        qc_pauli.cz(i, i + num_qubits + 1)
    for i in range(num_qubits):
        qc_pauli.cp(np.pi, i, num_qubits)
        ccy = YGate().control(num_ctrl_qubits=2, ctrl_state="11")
        qc_pauli.append(ccy, [i, num_qubits, i + num_qubits + 1])
    metrics["(Pauli-structure)"] = count_metrics(qc_pauli)

    base = metrics["(No)"]
    print("=== Transpiled metrics (basis_gates=['cx','u3'], opt_level=3) ===")
    for tag, values in metrics.items():
        print(
            f"{tag}: qubits={values['num_qubits']}, depth={values['depth']}, "
            f"size={values['size']}, cx={values['cx']}, u3={values['u3']}, "
            f"depth_ratio={ratio(values['depth'], base['depth']):.4f}, "
            f"size_ratio={ratio(values['size'], base['size']):.4f}, "
            f"cx_ratio={ratio(values['cx'], base['cx']):.4f}, "
            f"u3_ratio={ratio(values['u3'], base['u3']):.4f}"
        )
    print("-----------------------------")
    return metrics


def build_tfim_lcu_baseline_circuit(num_qubits: int, delta_t: float = 0.1):
    gamma = np.sqrt(0.1) / 2
    h_terms, l_terms = build_periodic_tfim_lindbladian_pauli(num_qubits, gamma)
    tfim_lind = Lindbladian(h_terms, l_terms)
    channel_lind, _, _ = Lindblad_to_channel(tfim_lind, delta_t)
    ms = channel_lind.channels[0][1][0]
    return BlockEncoding(ms).circuit(opt="No")


def export_tfim_lcu_baseline_openqasm3(
    num_qubits: int = 4,
    delta_t: float = 0.1,
    out_path=None,
) -> dict[str, object]:
    if out_path is None:
        out_path = DATA_DIR / f"evaluate_test_i_tfim_lcu_no_n{num_qubits}_baseline.qasm"
    qc = build_tfim_lcu_baseline_circuit(num_qubits, delta_t=delta_t)
    return export_openqasm3_baseline(qc, out_path)


def test_tfim_lcu_legacy_counts(num_qubits: int, delta_t: float = 0.1):
    gamma = np.sqrt(0.1) / 2
    h_terms, l_terms = build_periodic_tfim_lindbladian_pauli(num_qubits, gamma)
    tfim_lind = Lindbladian(h_terms, l_terms)
    channel_lind, _, _ = Lindblad_to_channel(tfim_lind, delta_t)
    ms = channel_lind.channels[0][1][0]

    print(f"Legacy count test for case N = {num_qubits}:")
    results = {}
    for opt in ("No", "Ctrl-line", "Matrix-order"):
        be = BlockEncoding(ms)
        _ = be.circuit(opt=opt)
        results[opt] = {
            "tcount": int(be.tcount),
            "mccount": int(be.mccount),
            "cxcount": int(be.cxcount),
        }
        print(
            f"{opt}: T-count={results[opt]['tcount']}, "
            f"Multi-controlled={results[opt]['mccount']}, CX={results[opt]['cxcount']}"
        )

    t_sp, mc_sp, cx_sp = 0, 0, 0
    qc_sp = QuantumCircuit(2 * num_qubits + 1)
    for i in range(num_qubits):
        qc_sp.p(np.pi, i)
        qc_sp.cz(i, i + num_qubits + 1)
        cx_sp += 1
    for i in range(num_qubits):
        qc_sp.cp(np.pi, i, num_qubits)
        ccy = YGate().control(num_ctrl_qubits=2, ctrl_state="11")
        qc_sp.append(ccy, [i, num_qubits, i + num_qubits + 1])
        mc_sp += 1
        t_sp += 4
        cx_sp += 4

    results["Special-legacy"] = {
        "tcount": t_sp,
        "mccount": mc_sp,
        "cxcount": cx_sp,
        "transpiled": count_metrics(qc_sp),
    }
    print("Special-legacy:", results["Special-legacy"])
    print("-----------------------------")
    return results


def build_random_pauli_matrixsum(num_qubits: int, seed: int = 20260314):
    num_terms = max(num_qubits * num_qubits - num_qubits, 2 ** (num_qubits - 1) - num_qubits)
    all_labels = ["".join(p) for p in product("IXYZ", repeat=num_qubits)]
    if num_terms > len(all_labels):
        raise ValueError(
            f"Requested num_terms={num_terms} exceeds total Pauli strings={len(all_labels)}."
        )

    rng = np.random.default_rng(seed + num_qubits)
    idx = rng.choice(len(all_labels), size=num_terms, replace=False)
    instances = [(PauliAtom(all_labels[int(i)], phase=1.0), 1.0) for i in idx]
    return Matrixsum(instances)


def run_random_pauli_matrixsum_metrics(ms: Matrixsum, num_qubits: int):
    metrics = {}
    for opt in ("No", "Ctrl-line", "Matrix-order"):
        be = BlockEncoding(ms)
        metrics[f"({opt})"] = count_metrics(be.circuit(opt=opt))

    base = metrics["(No)"]
    print(f"=== Random-Pauli Matrixsum metrics for N = {num_qubits} ===")
    for tag, values in metrics.items():
        print(
            f"{tag}: qubits={values['num_qubits']}, depth={values['depth']}, "
            f"size={values['size']}, cx={values['cx']}, u3={values['u3']}, "
            f"depth_ratio={ratio(values['depth'], base['depth']):.4f}, "
            f"size_ratio={ratio(values['size'], base['size']):.4f}, "
            f"cx_ratio={ratio(values['cx'], base['cx']):.4f}, "
            f"u3_ratio={ratio(values['u3'], base['u3']):.4f}"
        )
    print("-----------------------------")
    return metrics


def export_random_pauli_matrixsum_baseline_openqasm3(
    num_qubits: int = 4,
    seed: int = 20260314,
    out_path=None,
) -> dict[str, object]:
    if out_path is None:
        out_path = DATA_DIR / f"evaluate_test_i_random_pauli_no_n{num_qubits}_baseline.qasm"
    ms = build_random_pauli_matrixsum(num_qubits, seed=seed)
    qc = BlockEncoding(ms).circuit(opt="No")
    return export_openqasm3_baseline(qc, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfim-n", type=int, nargs="*", default=[4, 8, 12, 16])
    parser.add_argument("--legacy-n", type=int, default=4)
    parser.add_argument("--random-n", type=int, nargs="*", default=[4, 8, 10])
    parser.add_argument("--delta-t", type=float, default=0.1)
    args = parser.parse_args()

    tfim_results = {
        n: test_tfim_lcu_optimization(n, delta_t=args.delta_t)
        for n in args.tfim_n
    }
    legacy_result = test_tfim_lcu_legacy_counts(args.legacy_n, delta_t=args.delta_t)
    random_results = {}
    for n in args.random_n:
        random_results[n] = run_random_pauli_matrixsum_metrics(
            build_random_pauli_matrixsum(n),
            n,
        )

    return {
        "tfim_results": tfim_results,
        "legacy_result": legacy_result,
        "random_results": random_results,
    }


if __name__ == "__main__":
    main()
