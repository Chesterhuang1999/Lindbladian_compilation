"""Bounded Matrix-order optimizations, retaining phase-aware Boolean ANF SELECT.

The five established optimization switches are independent. Disabling iterative_anf retains the
historical single ANF pass; anf_max_iterations counts the first pass as well.
actual_nam_cost chooses the placement objective (NAM versus residual proxy).
Extra phase-aware address permutations and pure-phase lowering candidates are
opt-in. Partial-address search is the default for incomplete nonidentity inputs;
random_padding=True explicitly enables the historical random table completion.
Original/Gray candidates and deterministic subgroup completion remain enabled.
Control-order search compares forward and reversed address controls, keeping
outer controls first. That search and iterative ANF retain the cheapest NAM
result even when placement ranking uses the proxy.
NAM costs always use opt0 and arbitrary-input-safe lowering. Product PREPARE
preserves its prepared state, not necessarily its full unitary extension.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from collections.abc import Mapping

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import StatePreparation, PhaseGate, ZGate, SGate, SdgGate
from qiskit.quantum_info import Pauli


@dataclass(frozen=True)
class MatrixOrderOptions:
    placement_candidates: bool = True
    actual_nam_cost: bool = True
    control_order_search: bool = True
    iterative_anf: bool = True
    product_prepare: bool = True
    phase_aware_candidates: bool = False
    random_padding: bool = False
    anf_max_iterations: int = 3
    product_prepare_atol: float = 1e-12

    def __post_init__(self):
        for key in ("placement_candidates", "actual_nam_cost", "control_order_search",
                    "iterative_anf", "product_prepare", "phase_aware_candidates",
                    "random_padding"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a bool")
        if type(self.anf_max_iterations) is not int or self.anf_max_iterations < 1:
            raise ValueError("anf_max_iterations must be a positive integer")
        if (isinstance(self.product_prepare_atol, bool)
                or not isfinite(self.product_prepare_atol)
                or not 0 <= self.product_prepare_atol <= 1e-10):
            raise ValueError("product_prepare_atol must be finite and between 0 and 1e-10")


def resolve_options(options=None):
    if isinstance(options, MatrixOrderOptions):
        return options
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("matrix_order_options must be a mapping or MatrixOrderOptions")
    supplied = dict(options or {})
    unknown = set(supplied) - set(MatrixOrderOptions.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unsupported Matrix-order options: {sorted(unknown)}")
    return MatrixOrderOptions(**supplied)


def nam_cost(circuit):
    lowered = transpile(circuit, basis_gates=["rz", "h", "x", "cx"],
                       optimization_level=0, seed_transpiler=20261009,
                       qubits_initially_zero=False)
    return sum(lowered.count_ops().values())


def prepare_coefficients(coefficients, options):
    probabilities = np.abs(np.asarray(coefficients, dtype=complex))
    probabilities /= probabilities.sum()
    amplitudes = np.sqrt(probabilities)
    width = (len(probabilities) - 1).bit_length()
    if len(probabilities) != 1 << width:
        raise ValueError("PREPARE coefficients must be padded to the address space")
    circuit = QuantumCircuit(width)
    if options.product_prepare:
        addresses = np.arange(len(probabilities))
        marginals = [float(probabilities[(addresses & (1 << bit)) != 0].sum())
                     for bit in range(width)]
        # Clip only rounding outside [0,1]. H/X specialization is itself
        # validated by reconstructing the *actual* amplitudes it will emit.
        marginals = [float(np.clip(p, 0, 1)) for p in marginals]
        marginals = [min((0.0, 0.5, 1.0), key=lambda x: abs(x - p))
                     if min(abs(x - p) for x in (0, 0.5, 1)) <= options.product_prepare_atol
                     else p for p in marginals]
        product = np.ones(len(probabilities))
        for bit, p in enumerate(marginals):
            product *= np.where((addresses & (1 << bit)) != 0, np.sqrt(p), np.sqrt(1 - p))
        error = float(np.max(np.abs(product - amplitudes)))
        if error <= options.product_prepare_atol:
            for bit, p in enumerate(marginals):
                if p == 1:
                    circuit.x(bit)
                elif p == 0.5:
                    circuit.h(bit)
                elif p != 0:
                    circuit.ry(2 * np.arctan2(np.sqrt(p), np.sqrt(1 - p)), bit)
            return circuit, {"method": "product", "amplitude_max_error": error,
                             "marginal_probabilities": marginals}
    if width:
        circuit.append(StatePreparation(amplitudes), range(width))
    return circuit, {"method": "StatePreparation"}


def extra_placements(labels, width, system_size):
    """Original/Gray order and a low-weight basis of the affine Pauli span."""
    from placement_search import (_label_to_bits, _bits_to_label, _gf2_rank,
                                  _span_coordinate_map)
    result = [("original", [(label, format(i, f"0{width}b"))
                             for i, label in enumerate(labels)]),
              ("gray", [(label, format(i ^ (i >> 1), f"0{width}b"))
                         for i, label in enumerate(labels)])]
    bits = [_label_to_bits(label) for label in labels]
    weight = lambda value: (sum(c != "I" for c in _bits_to_label(value, system_size)), value)
    anchor = min(bits, key=weight)
    differences = sorted({value ^ anchor for value in bits}, key=weight)
    if _gf2_rank(differences) <= width:
        basis = []
        for delta in differences:
            if _gf2_rank(basis + [delta]) > len(basis):
                basis.append(delta)
        coordinates = _span_coordinate_map(basis)
        result.append(("affine", [(label, format(coordinates[value ^ anchor], f"0{width}b"))
                                   for label, value in zip(labels, bits)]))
    return result


def symbolic_select(residuals, address_width, system_size, *, phase_on_address=False):
    """Build ordered residuals; optionally lower scalar residuals on addresses.

    A scalar residual has no system target. Keep generic lowering available as
    a separate candidate because specialized lowering need not be cheaper for
    every angle and control count.
    """
    address = QuantumRegister(address_width, "ctrl")
    system = QuantumRegister(system_size, "sys")
    circuit = QuantumCircuit(address, system)
    for label, address_string, phase in residuals:
        controls = [i for i, bit in enumerate(address_string[::-1]) if bit == "1"]
        angle = float(np.pi * phase / 2)
        if phase_on_address and set(label) <= {"I"}:
            if not controls:
                circuit.global_phase += angle
                continue
            wrapped = angle % (2 * np.pi)
            if min(wrapped, 2 * np.pi - wrapped) < 1e-12:
                continue
            if abs(wrapped - np.pi) < 1e-12:
                gate = ZGate()
            elif abs(wrapped - np.pi / 2) < 1e-12:
                gate = SGate()
            elif abs(wrapped - 3 * np.pi / 2) < 1e-12:
                gate = SdgGate()
            else:
                gate = PhaseGate(angle)
            if len(controls) > 1:
                gate = gate.control(len(controls) - 1)
            circuit.append(gate, controls)
            continue
        pauli = QuantumCircuit(system_size)
        pauli.append(Pauli(label), range(system_size))
        pauli.global_phase = angle
        gate = pauli.decompose().to_gate()
        if controls:
            gate = gate.control(len(controls))
        circuit.append(gate, controls + list(range(address_width, address_width + system_size)))
    return circuit


def _reverse_address_controls(circuit, outer_controls):
    """Permute conjunction variables only, preserving residual and outer order.

    Move each ctrl_state bit with its wire, including open controls. No Pauli
    operations are commuted and outer control wires stay at the front.
    """
    output = circuit.copy_empty_like()
    output.global_phase = circuit.global_phase
    for instruction in circuit.data:
        op = instruction.operation
        qubits = list(instruction.qubits)
        if isinstance(op, ControlledGate):
            count = op.num_ctrl_qubits
            fixed = [i for i, q in enumerate(qubits[:count])
                     if circuit.find_bit(q).index < outer_controls]
            movable = [i for i in range(count) if i not in fixed]
            order = fixed + movable[::-1]
            if order != list(range(count)):
                state = sum(((int(op.ctrl_state) >> old) & 1) << new
                            for new, old in enumerate(order))
                op = op.base_gate.control(count, ctrl_state=state)
                qubits = [qubits[i] for i in order] + qubits[count:]
        output.append(op, qubits, instruction.clbits)
    return output


def lower_select(symbolic, options, *, outer_controls=0):
    from block_encoding import expand_large_controlled_gates, optimize_expanded_circuit
    variants = [("forward", symbolic)]
    if options.control_order_search:
        reversed_circuit = _reverse_address_controls(symbolic, outer_controls)
        if reversed_circuit != symbolic:
            variants.append(("reverse-address", reversed_circuit))
    maximum = options.anf_max_iterations if options.iterative_anf else 1
    measure = options.control_order_search or (options.iterative_anf and maximum > 1)
    trials = []
    best = None
    for name, candidate in variants:
        current = expand_large_controlled_gates(candidate, if_optimize=True)
        scratch = current.qubits[candidate.num_qubits:]
        for iteration in range(1, maximum + 1):
            groups = len(getattr(optimize_expanded_circuit, "last_boolean_expressions", []))
            cost = nam_cost(current) if measure else None
            trial = {"control_order": name, "anf_iterations": iteration,
                     "nam_gates": cost, "simplified_control_groups": groups}
            trials.append(trial)
            if best is None or (cost is not None and cost < best[0]):
                best = (cost, current, trial)
            if iteration == maximum or not scratch:
                break
            following = optimize_expanded_circuit(current, scratch)
            if following == current:
                break
            current = following
    return best[1], {**best[2], "trials": trials, "max_iterations": maximum}


def options_dict(options):
    return asdict(options)
