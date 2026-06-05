"""Test II from ``evaluate.ipynb``: all-Pauli Ctrl-line benchmark."""

from __future__ import annotations

import argparse
from itertools import product

from qiskit import transpile

from evaluate_common import BlockEncoding, Matrixsum, PauliAtom


def build_all_pauli_matrixsum(num_qubits: int) -> Matrixsum:
    instances = [
        (PauliAtom("".join(p_tuple), phase=1.0), 1.0)
        for p_tuple in product("IXYZ", repeat=num_qubits)
    ]
    return Matrixsum(instances)


def gate_metrics_ctrl_line_from_blockencoding(ms: Matrixsum):
    be = BlockEncoding(ms)
    qc_ctrl, _, _, _ = be.mulplex_U_opt(be.mat_list, be.ctrl_size, be.sys_size)
    tqc = transpile(qc_ctrl, basis_gates=["cx", "h", "s", "t"], optimization_level=2)
    ops = tqc.count_ops()
    return {
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx": int(ops.get("cx", 0)),
        "h": int(ops.get("h", 0)),
        "s": int(ops.get("s", 0)),
        "t": int(ops.get("t", 0)),
    }


def test_all_pauli_matrixsum_ctrl_line(num_qubits: int):
    ms = build_all_pauli_matrixsum(num_qubits)
    metrics = gate_metrics_ctrl_line_from_blockencoding(ms)
    print(
        f"All-Pauli Matrixsum Ctrl-line metrics "
        f"(N={num_qubits}, basis_gates=['cx','h','s','t']):"
    )
    print(
        f"depth={metrics['depth']}, size={metrics['size']}, cx={metrics['cx']}, "
        f"h={metrics['h']}, s={metrics['s']}, t={metrics['t']}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", type=int, nargs="*", default=[3, 4, 6, 8])
    args = parser.parse_args()

    flatten_allpauli_data = {}
    for num_qubits in args.n_list:
        try:
            metrics = test_all_pauli_matrixsum_ctrl_line(num_qubits)
            flatten_allpauli_data[num_qubits] = {
                "depth": int(metrics["depth"]),
                "cx": int(metrics["cx"]),
                "h": int(metrics["h"]),
                "s": int(metrics["s"]),
                "t": int(metrics["t"]),
            }
        except Exception as exc:
            flatten_allpauli_data[num_qubits] = {"error": str(exc)}

    print("flatten_allpauli_data =")
    print(flatten_allpauli_data)
    return flatten_allpauli_data


if __name__ == "__main__":
    main()

