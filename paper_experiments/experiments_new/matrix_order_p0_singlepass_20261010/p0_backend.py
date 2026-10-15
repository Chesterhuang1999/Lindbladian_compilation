"""Isolated P0 experiment adapter; production sources are never modified."""

import inspect
from itertools import combinations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PhaseGate, ZGate, SGate, SdgGate
from qiskit.quantum_info import Pauli

import block_encoding as bm
import matrix_order_pipeline as mp
import placement_search as ps

PATCH_VERSION = "p0-isolated-v1"
ORIGINAL_APPLY = ps._apply_layer_assignment
ORIGINAL_SEARCH = ps.search_for_pauli_placement
ORIGINAL_CIRCUIT = bm._PlacementBlockEncoding.circuit
STATE_STATS = {"rebuilt": 0, "fast": 0}


def rebuild(placement, vectors, num_qubits, supersets):
    prefix = [0] * len(supersets)
    cost = 0
    for term, address in sorted(placement, key=lambda pair: pair[1]):
        residual = vectors[term] ^ prefix[address]
        cost += address.bit_count() * ps._pauli_weight(residual, num_qubits)
        for upper in supersets[address]:
            prefix[upper] ^= residual
    return tuple(prefix), cost


def corrected_apply(state, assignment, vectors, num_qubits, supersets):
    new_addresses = [address for _, address in assignment]
    all_addresses = [address for _, address in state.placement] + new_addresses
    comparable = any(a != b and a & b == a for a in new_addresses for b in all_addresses)
    if not comparable:
        STATE_STATS["fast"] += 1
        return ORIGINAL_APPLY(state, assignment, vectors, num_qubits, supersets)
    STATE_STATS["rebuilt"] += 1
    placement = state.placement + assignment
    prefix, cost = rebuild(placement, vectors, num_qubits, supersets)
    added = {term for term, _ in assignment}
    return ps._SearchState(tuple(t for t in state.remaining if t not in added),
                           prefix, cost, placement)


def phase_select(residuals, address_width, system_size):
    """Put pure scalar residuals on address wires, without a system target."""
    address = QuantumRegister(address_width, "ctrl")
    system = QuantumRegister(system_size, "sys")
    qc = QuantumCircuit(address, system)
    for label, bits, phase in residuals:
        controls = [i for i, bit in enumerate(bits[::-1]) if bit == "1"]
        angle = float(np.pi * phase / 2)
        if set(label) <= {"I"}:
            if not controls:
                qc.global_phase += angle
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
            qc.append(gate, controls)
        else:
            term = QuantumCircuit(system_size)
            term.append(Pauli(label), range(system_size))
            term.global_phase = angle
            gate = term.decompose().to_gate()
            if controls:
                gate = gate.control(len(controls))
            qc.append(gate, controls + list(range(address_width, address_width + system_size)))
    return qc


def extend_candidates(backend, candidates, width):
    """Bounded bit permutations; exact NAM selection remains in the backend.

    Retain existing candidates. Add reversal and a swap of the lowest two
    address bits for each placement, with phase-aware residual recomputation.
    Keep generic pure-phase lowering as a competing candidate.
    """
    full_lookup = {bm._pauli_label_to_symplectic_vector(a.expr).tobytes():
                   2 * np.angle(a.phase) / np.pi for a, _ in backend.J.instances}
    expanded = list(candidates)
    seen = {tuple(sorted(item["modes"])) for item in candidates}
    permutations = []
    if width >= 2:
        permutations.append(("address-reverse", list(reversed(range(width)))))
        low_swap = list(range(width))
        low_swap[0], low_swap[1] = low_swap[1], low_swap[0]
        if low_swap != permutations[0][1]:
            permutations.append(("address-swap01", low_swap))
    for base in candidates:
        for name, order in permutations:
            def permute(value):
                return sum(((value >> old) & 1) << new for new, old in enumerate(order))
            modes = [(label, format(permute(int(bits, 2)), f"0{width}b"))
                     for label, bits in base["modes"]]
            key = tuple(sorted(modes))
            if key in seen:
                continue
            seen.add(key)
            lookup = dict(full_lookup)
            for label, _ in modes:
                lookup.setdefault(bm._pauli_label_to_symplectic_vector(label).tobytes(), 0.0)
            mobius = bm.mobius_invert_modes_with_phases(width, modes, lookup, 0)
            coefficients = np.zeros_like(base["coefficients"])
            for index, coefficient in enumerate(base["coefficients"]):
                coefficients[permute(index)] = coefficient
            expanded.append({"name": base["name"] + "/" + name, "modes": modes,
                             "mobius": mobius, "coefficients": coefficients,
                             "phase_lookup": lookup,
                             "symbolic": mp.symbolic_select(mobius["g_modes_with_phase"],
                                                            width, backend.sys_size)})
    for base in list(expanded):
        residuals = base["mobius"]["g_modes_with_phase"]
        if any(set(label) <= {"I"} and abs(phase % 4) > 1e-12
               for label, _, phase in residuals):
            expanded.append({**base, "name": base["name"] + "/phase-on-address",
                             "symbolic": phase_select(residuals, width, backend.sys_size)})
    return expanded


def install():
    ps._apply_layer_assignment = corrected_apply
    source = inspect.getsource(ORIGINAL_SEARCH)
    needle = "    labels = _extract_pauli_labels(matrixsum)\n"
    assert source.count(needle) == 1
    source = source.replace(needle, "    padding_to_full_address = False\n" + needle, 1)
    start = source.index("    best_state = min(complete_states, key=lambda state: state.cost)\n")
    end = source.index("    if return_debug:\n", start)
    replacement = '''    best_state = min(complete_states, key=lambda state: state.cost)
    final_placement = [(labels[t], format(a, f"0{width}b"))
                       for t, a in sorted(best_state.placement, key=lambda item: item[1])]
    padded_search_cost = _placement_cost_from_modes(final_placement, num_qubits)
    assert best_state.cost == padded_search_cost, (best_state.cost, padded_search_cost)
    pre_refinement_cost = padded_search_cost
    refinement_applied = False
    refinement_skipped = False
    if swap_refinement and (swap_refinement_max_terms is None or len(final_placement) <= swap_refinement_max_terms):
        final_placement, refined_cost = _basis_nonbasis_swap_refine(
            final_placement, width=width, num_qubits=num_qubits)
        refinement_applied = refined_cost < pre_refinement_cost
    else:
        refined_cost = pre_refinement_cost
        refinement_skipped = swap_refinement and (swap_refinement_max_terms is not None and len(final_placement) > swap_refinement_max_terms)
    padded_placement = list(final_placement)
    real_labels = {labels[index] for index in real_output_indices}
    placement = [(label, address) for label, address in final_placement if label in real_labels]
    assert refined_cost == _placement_cost_from_modes(padded_placement, num_qubits)
'''
    source = source[:start] + replacement + source[end:]
    exec(compile(source, "<isolated-P0-3>", "exec"), ps.__dict__)
    bm.search_for_pauli_placement = ps.search_for_pauli_placement
    source = inspect.getsource(ORIGINAL_CIRCUIT)
    import textwrap
    source = textwrap.dedent(source)
    needle = '    for candidate in candidates:\n        candidate["proxy_cost"]'
    assert source.count(needle) == 1
    source = source.replace(needle, '    candidates = _p0_extend_candidates(self, candidates, ctrl_size)\n\n' + needle)
    bm._p0_extend_candidates = extend_candidates
    exec(compile(source, "<isolated-P0-1>", "exec"), bm.__dict__)
    bm._PlacementBlockEncoding.circuit = bm.circuit
    bm.BACKEND_ID = "matrix-order-anf-p0-isolated-v1"


def verify_selected_residuals(backend, selected_name):
    """Exact Pauli algebra on every occupied address of the selected candidate."""
    if len(backend.J.instances) == 1:
        return {"addresses_checked": 1, "max_phase_error": 0.0}
    candidate = next(c for c in backend._matrix_order_candidates if c["name"] == selected_name)
    phase_by_label = {a.expr: complex(a.phase) for a, _ in backend.J.instances}
    residuals = candidate["mobius"]["g_modes_with_phase"]
    max_error = 0.0
    checked = 0
    for label, bits in candidate["modes"]:
        if label not in phase_by_label:
            continue
        address = int(bits, 2)
        value = Pauli("I" * backend.sys_size)
        scalar = 1 + 0j
        for g_label, g_bits, phase in residuals:
            lower = int(g_bits, 2)
            if lower & address == lower:
                value = Pauli(g_label) @ value
                scalar *= np.exp(1j * np.pi * phase / 2)
        target = Pauli(label)
        assert np.array_equal(value.x, target.x) and np.array_equal(value.z, target.z)
        error = abs(scalar * (-1j) ** value.phase - phase_by_label[label])
        max_error = max(max_error, float(error))
        checked += 1
    assert max_error < 1e-10, max_error
    return {"addresses_checked": checked, "max_phase_error": max_error}
