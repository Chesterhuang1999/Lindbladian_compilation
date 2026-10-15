"""Native Qrisp LCU blocks behind the shared channel-LCU outer tree.

Positive real coefficients go to BlockEncoding.from_lcu; all coefficient
phases go into the Pauli unitaries. Qrisp's conjugate environment then leaves
PREPARE uncontrolled when we request a controlled block (the F* contract).

Export fixed logical wires, not a zero-input QuantumSession allocation. A
session allocator can reuse eventual input ports as earlier clean scratch,
which is invalid for a block later embedded on arbitrary system inputs.
Qrisp's native control/conjugation and inner tree construction remain active.
"""

from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
from qiskit import QuantumCircuit
from channel_IR import PauliAtom


@dataclass(frozen=True)
class QrispControlledBlock:
    """Implements the circuit/system_last/controlled interface of F* blocks."""

    circuit: QuantumCircuit
    controlled_block: QuantumCircuit
    system_size: int

    def system_last(self, system_size):
        if system_size != self.system_size:
            raise ValueError("Qrisp block and channel must have the same system width.")
        return self

    def controlled(self, num_ctrl_qubits=1, ctrl_state=None):
        if num_ctrl_qubits != 1 or ctrl_state not in (None, 1, "1"):
            raise ValueError("Qrisp F* adapter supports the single active control of opt/tree.")
        return self.controlled_block.copy()


class QrispBlockEncoding:
    """Native LCU block. Use qrisp_backend_for_channel for a shared workspace.

    Native controlled q_switch assumes its scratch wires are clean. Reserve
    one address region for the entire channel so a shorter block's scratch
    never overlaps a preceding block's possibly nonzero LCU address outputs.
    """

    def __init__(self, matrix_sum, *, shared_address_qubits=None):
        self.matrix_sum = deepcopy(matrix_sum)
        self.system_size = matrix_sum.size
        own_address = (len(matrix_sum.instances) - 1).bit_length()
        self.address_width = own_address if shared_address_qubits is None else shared_address_qubits
        if self.address_width < own_address:
            raise ValueError("Shared address register is too short for this block.")
        if not matrix_sum.instances or matrix_sum.pauli_norm() <= 0:
            raise ValueError("Qrisp LCU requires a nonzero Kraus operator.")
        for atom, coefficient in matrix_sum.instances:
            if not isinstance(atom, PauliAtom):
                raise TypeError("The Qrisp Pauli-LCU adapter requires PauliAtom terms.")
            if coefficient <= 0 or not np.isfinite(coefficient):
                raise ValueError("Canonicalize zero/nonfinite coefficients before Qrisp LCU.")
            if not np.isclose(abs(atom.phase), 1, atol=1e-12, rtol=0):
                raise ValueError("PauliAtom phases must have unit modulus.")
        self._parts = None

    def _native(self, controlled):
        from qrisp import QuantumFloat, QuantumVariable, control, gphase, x, y, z
        from qrisp.block_encodings import BlockEncoding

        def phased_pauli(atom):
            label, angle = atom.expr, float(np.angle(atom.phase))

            def unitary(system):
                if angle != 0:
                    gphase(angle, system[0])
                # Qiskit/PauliAtom labels put qubit zero at the right.
                for bit, letter in enumerate(reversed(label)):
                    if letter != "I":
                        {"X": x, "Y": y, "Z": z}[letter](system[bit])

            return unitary

        weights = np.asarray([float(c) for _, c in self.matrix_sum.instances])
        native = BlockEncoding.from_lcu(weights, [phased_pauli(a) for a, _ in self.matrix_sum.instances])
        system = QuantumVariable(self.system_size, name="system")
        address_size = (len(weights) - 1).bit_length()
        address = QuantumFloat(address_size, name="address", qs=system.qs) if address_size else None
        outer = QuantumVariable(1, name="outer", qs=system.qs) if controlled else None
        operands = (address, system) if address_size else (system,)
        if controlled:
            with control(outer):
                native.unitary(*operands)
        else:
            native.unitary(*operands)

        # copy() resolves environments but keeps the original logical ports.
        logical = system.qs.copy()
        qc = logical.to_qiskit()
        positions = lambda qv: [logical.qubits.index(q) for q in qv.reg] if qv is not None else []
        sys_bits, address_bits, outer_bits = positions(system), positions(address), positions(outer)
        scratch = [q for q in range(qc.num_qubits) if q not in sys_bits + address_bits + outer_bits]
        return qc, outer_bits + address_bits + scratch + sys_bits, len(scratch), float(native.alpha)

    def circuit_components(self, opt="Qrisp"):
        if opt != "Qrisp":
            raise ValueError("QrispBlockEncoding accepts only opt='Qrisp'.")
        if self._parts is not None:
            return self._parts
        plain, controlled = self._native(False), self._native(True)
        address_size = (len(self.matrix_sum.instances) - 1).bit_length()
        workspace_size = self.address_width + max(plain[2], controlled[2])

        def pack(native, controls):
            source, order, _, _ = native
            out = QuantumCircuit(controls + workspace_size + self.system_size)
            destinations = (list(range(controls + address_size))
                            + list(range(controls + self.address_width,
                                         controls + self.address_width + native[2]))
                            + list(range(controls + workspace_size, out.num_qubits)))
            mapping = dict(zip(order, destinations))
            out.compose(source, qubits=[mapping[q] for q in range(source.num_qubits)], inplace=True)
            return out

        circuit = pack(plain, 0)
        controlled_circuit = pack(controlled, 1)
        if not np.isclose(plain[3], self.matrix_sum.pauli_norm(), atol=1e-12, rtol=1e-12):
            raise ValueError("Qrisp and ChannelIR block normalization disagree.")
        circuit.metadata = {
            "block_backend": "qrisp.BlockEncoding.from_lcu",
            "block_opt": "Qrisp", "qrisp_version": version("qrisp"),
            "alpha": plain[3], "single_term": len(self.matrix_sum.instances) == 1,
            "system_qubits": list(range(workspace_size, circuit.num_qubits)),
            "workspace_qubits": list(range(workspace_size)),
            "address_qubits": address_size, "shared_address_qubits": self.address_width,
            "scratch_qubits": workspace_size - self.address_width,
            "scratch_input": "zero; disjoint from every block's address region",
            "coefficients": "positive magnitudes", "phases": "Pauli unitaries",
            "controlled_prepare": False, "inner_select": "q_switch auto (tree in Qrisp 0.9.9)",
            "export": "fixed logical wires; QuantumSession allocator disabled",
        }
        self._parts = QrispControlledBlock(circuit, controlled_circuit, self.system_size)
        return self._parts

    def circuit(self, opt="Qrisp"):
        return self.circuit_components(opt).circuit.copy()


def qrisp_backend_for_channel(kraus):
    """Factory enforcing the native Qrisp scratch precondition across branches."""
    address_width = (max(len(ms.instances) for ms in kraus) - 1).bit_length()

    def backend(ms):
        return QrispBlockEncoding(ms, shared_address_qubits=address_width)

    return backend
