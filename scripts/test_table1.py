from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import YGate

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from block_encoding import BlockEncoding
from channel_IR import Lindbladian
from channel_LCU import Lindblad_to_channel


def _count_metrics(qc: QuantumCircuit) -> dict[str, int]:
    tqc = transpile(qc, basis_gates=["cx", "u3"], optimization_level=3)
    ops = tqc.count_ops()
    return {
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx": int(ops.get("cx", 0)),
        "u3": int(ops.get("u3", 0)),
        "num_qubits": int(tqc.num_qubits),
    }


def _ratio(curr: int, base: int) -> float:
    if base == 0:
        return float("nan")
    return 1.0 - (curr / base)


def test_tfim_lcu_optimization(num_qubits: int) -> dict[str, dict[str, int]]:
    h_terms = []
    l_terms = []
    gamma = np.sqrt(0.1) / 2

    for i in range(num_qubits):
        z_ind = [i, (i + 1) % num_qubits]
        z_str = "".join("Z" if j in z_ind else "I" for j in range(num_qubits))
        x_str = "".join("X" if j == i else "I" for j in range(num_qubits))
        y_str = "".join("Y" if j == i else "I" for j in range(num_qubits))
        h_terms.append((z_str, -1))
        h_terms.append((x_str, -1))
        l_terms.append([(x_str, gamma), (y_str, -1j * gamma)])

    delta_t = 0.1
    tfim_lind = Lindbladian(h_terms, l_terms)
    channel_lind, _, _ = Lindblad_to_channel(tfim_lind, delta_t)
    ms = channel_lind.channels[0][1][0]
    print(ms)
    print(f"Test for case N = {num_qubits}:")

    block_encoding = BlockEncoding(ms)
    qc_no = block_encoding.circuit(opt="No")
    m_no = _count_metrics(qc_no)

    block_encoding = BlockEncoding(ms)
    qc_ctrl_line = block_encoding.circuit(opt="Ctrl-line")
    m_ctrl_line = _count_metrics(qc_ctrl_line)

    block_encoding = BlockEncoding(ms)
    qc_matrix_order = block_encoding.circuit(opt="Matrix-order")
    m_matrix_order = _count_metrics(qc_matrix_order)

    qc_pauli = QuantumCircuit(2 * num_qubits + 1)
    for i in range(num_qubits):
        qc_pauli.p(np.pi, i)
        qc_pauli.cz(i, i + num_qubits + 1)

    for i in range(num_qubits):
        qc_pauli.cp(np.pi, i, num_qubits)
        ccy = YGate().control(num_ctrl_qubits=2, ctrl_state="11")
        qc_pauli.append(ccy, [i, num_qubits, i + num_qubits + 1])

    m_pauli = _count_metrics(qc_pauli)

    all_metrics = {
        "(No)": m_no,
        "(Ctrl-line)": m_ctrl_line,
        "(Matrix-order)": m_matrix_order,
        "(Pauli-structure)": m_pauli,
    }

    base = all_metrics["(No)"]
    print("=== Transpiled metrics (basis_gates=['cx','u3'], opt_level=3) ===")
    for tag, values in all_metrics.items():
        depth_ratio = _ratio(values["depth"], base["depth"])
        size_ratio = _ratio(values["size"], base["size"])
        cx_ratio = _ratio(values["cx"], base["cx"])
        u3_ratio = _ratio(values["u3"], base["u3"])
        print(
            f"{tag}: qubits={values['num_qubits']}, depth={values['depth']}, "
            f"size={values['size']}, cx={values['cx']}, u3={values['u3']}, "
            f"depth_ratio={depth_ratio:.4f}, size_ratio={size_ratio:.4f}, "
            f"cx_ratio={cx_ratio:.4f}, u3_ratio={u3_ratio:.4f}"
        )

    print("-----------------------------")
    return all_metrics


def main() -> None:
    for num_qubits in range(4, 20, 4):
        test_tfim_lcu_optimization(num_qubits)


if __name__ == "__main__":
    main()
