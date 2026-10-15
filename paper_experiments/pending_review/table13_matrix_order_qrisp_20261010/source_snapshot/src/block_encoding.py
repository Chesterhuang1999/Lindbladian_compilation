"""Canonical block encoding with placement search and corrected zero-address residuals.

All active and compatibility imports resolve to this implementation.  Direct,
ctrl-line, and tree SELECTs share the same coefficient and phase convention;
Matrix-order uses placement_search and the fixed residual construction.
"""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, replace
from placement_search import analyze_placement_structure, search_for_pauli_placement
from matrix_order_pipeline import (resolve_options, options_dict, extra_placements,
                                   prepare_coefficients, symbolic_select, lower_select, nam_cost)

from channel_IR import *
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix, partial_trace, PauliList, Pauli
from qiskit.circuit.library import StatePreparation, UnitaryGate
from subroutine import lcu_prepare_tree, count_multiq_gates
import numpy as np
try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None
from qiskit.circuit.controlledgate import ControlledGate
from itertools import product


def _require_aer_simulator():
    if AerSimulator is None:
        raise ImportError("qiskit_aer.AerSimulator is required for simulation helpers.")
    return AerSimulator
def sort_symplectic_matrix(matrix, phases, crit = 'z'):

    n = matrix.shape[1] // 2    

    x_part = matrix[:, :n]
    z_part = matrix[:, n:]

    ## Primary key: Pauli-string Hamming weight (non-identity count)
    non_identity = x_part | z_part 
    hamming_weight = np.sum(non_identity, axis=1)

    ## Secondary key: full symplectic vector weight (sum over X/Z halves)
    vector_weight = np.sum(matrix, axis=1)

    ## Tertiary key: X- or Z-half weight
    if crit == 'x':
        x_weight = np.sum(x_part, axis = 1)
        sorted_indices = np.lexsort((x_weight, vector_weight, hamming_weight))
    else:
        z_weight = np.sum(z_part, axis = 1)
        sorted_indices = np.lexsort((z_weight, vector_weight, hamming_weight))

    
    sorted_matrix = matrix[sorted_indices]
    phases = phases[sorted_indices]
    return sorted_matrix, phases

def greedy_generator_selection(sorted_matrix, max_length = None, max_gens = None):
    M, N = sorted_matrix.shape
    n = N // 2
    max_elements = 2 ** max_length if max_length is not None else 1e9

    rows_as_bytes = [row.tobytes() for row in sorted_matrix]
    row_weights = np.sum(sorted_matrix[:, :n] | sorted_matrix[:, n:], axis=1)
    U_set = set(rows_as_bytes)

    candidates_indices = list(range(M))

    first_idx = 0
    first_vec = sorted_matrix[first_idx]
    selected_indices = [first_idx]

    identity_vec = np.zeros(N, dtype = int)
    current_group_map = {
        identity_vec.tobytes(): identity_vec,
        first_vec.tobytes(): first_vec,
    }

    candidates_indices.pop(0)

    current_covered_in_U = {b for b in current_group_map if b in U_set}

    while candidates_indices:
        if (max_gens and  len(selected_indices) >= max_gens):
            break

        best_score = -1.0
        best_idx = -1 
        best_new_covered_elements = None

        to_remove = []
        candidate_found = False

        for i, idx in enumerate(candidates_indices):
            vec_bytes = rows_as_bytes[idx]
            if vec_bytes in current_group_map:
                to_remove.append(i)
                continue

            vec_arr = sorted_matrix[idx]
            new_covered_chunk = []

            for g_arr in current_group_map.values():
                new_vec = g_arr ^ vec_arr
                new_vec_bytes = new_vec.tobytes()   
                if new_vec_bytes in U_set:
                    if new_vec_bytes not in current_covered_in_U:
                        new_covered_chunk.append(new_vec_bytes)
            
            count_delta = len(new_covered_chunk)
            wt = row_weights[idx]
            wt = wt if wt > 0 else 1e-9
            score = count_delta / wt

            if score > best_score:
                best_score = score
                best_idx = i
                best_new_covered_elements = new_covered_chunk
                candidate_found = True

        for i in reversed(to_remove):
            candidates_indices.pop(i)
            if i < best_idx:
                best_idx -= 1

        if not candidate_found or best_score <= 0:
            break

        target_idx = candidates_indices[best_idx]
        selected_indices.append(target_idx)

        target_vec = sorted_matrix[target_idx]
        new_group_map = {}
        current_group_values = list(current_group_map.values())
        for g_arr in current_group_values:
            new_vec = g_arr ^ target_vec
            new_vec_bytes = new_vec.tobytes()
            if new_vec_bytes not in current_group_map:
                new_group_map[new_vec_bytes] = new_vec
        
        current_group_map.update(new_group_map)
        
        if best_new_covered_elements:
            current_covered_in_U.update(best_new_covered_elements)

        ## Stop criterion is evaluated AFTER adding one new generator:
        ## given current subset S, decide whether there is still enough remaining
        ## address capacity to cover currently uncovered targets.
        current_group_size = len(current_group_map)
        remaining_uncovered = M - len(current_covered_in_U)
        lhs = max_elements - current_group_size
        rhs = remaining_uncovered
        if lhs < rhs:
            break
        candidates_indices.pop(best_idx)

    return selected_indices, len(current_covered_in_U), current_covered_in_U

def find_pivots(matrix, ids):
    ## Return the pivotal map for a given binary matrix.
    pivot_map = {}
    rows, cols = matrix.shape
    for i in range(rows):
        curr_vec = matrix[i].copy()
        curr_id = ids[i]
        while True:
            nonzero_indices = np.nonzero(curr_vec)[0]
            if len(nonzero_indices) == 0:
                break
            pivot = nonzero_indices[0]
            if pivot in pivot_map:
                existing_vec, existing_id = pivot_map[pivot]
                curr_vec = curr_vec ^ existing_vec 
                curr_id  = ''.join(str(int(a) ^ int(b)) for a, b in zip(curr_id, existing_id))  
            else:
                pivot_map[pivot] = (curr_vec, curr_id)
                break
    return pivot_map
def assign_subspace_modes(w, selected_matrix, U_covered):
    assigned_modes = [] 
    basis_ids = []
    current_vec_bytes = set()   
    def vec_to_pauli(vec):
        n = len(vec) // 2
        x = vec[:n]
        z = vec[n:]
        pauli = Pauli((z, x))
        return pauli.to_label()
    
    for i in range(selected_matrix.shape[0]):
        vec = selected_matrix[i]
        pauli_label = vec_to_pauli(vec)
        ctrl_value = bin(2**i)[2:].zfill(w)
        assigned_modes.append((pauli_label, ctrl_value))
        basis_ids.append(ctrl_value)
        current_vec_bytes.add(vec.tobytes())
    pivot_map = find_pivots(selected_matrix, basis_ids)
    for vec_bytes in U_covered:
        if vec_bytes not in current_vec_bytes:
            vec = np.frombuffer(vec_bytes, dtype = int)
            pauli_label = vec_to_pauli(vec)
            accu_id = '0' * w

            is_generated = True
            while True:
                nonz_indices = np.nonzero(vec)[0]
                if len(nonz_indices) == 0:
                    break
                pivot = nonz_indices[0]
                if pivot in pivot_map:
                    existing_vec, existing_id = pivot_map[pivot]
                    vec = vec ^ existing_vec
                    accu_id = ''.join(str(int(a) ^ int(b)) for a, b in zip(accu_id, existing_id)) # Default: Z placed before X

                else:
                    is_generated = False
                    break
            if is_generated:
                assigned_modes.append((pauli_label, accu_id))
            
    return assigned_modes

def assign_additional_modes(w, assigned_modes, remaining_tuple):
    ## Input/Output semantics:
    ## - assigned_modes: already allocated phase-free P entries, each as (pauli_label, ctrl_value)
    ## - remaining_tuple: unallocated phase-free vectors (or legacy tuples whose first entry is vector)
    ## - return: merged phase-free P table entries as (pauli_label, ctrl_value)
    assigned_modes = list(assigned_modes)
    def vec_to_pauli(vec):
        n = len(vec) // 2
        x = vec[:n]
        z = vec[n:]
        pauli = Pauli((z, x))
        return pauli.to_label()
    
    def pauli_to_vec(pauli_label):
        pauli = Pauli(pauli_label)
        return np.hstack((pauli.x.astype(int), pauli.z.astype(int)))

    def vec_weight(vec):
        n = len(vec) // 2
        x = vec[:n]
        z = vec[n:]
        return int(np.sum(x | z))

    def ctrl_hamming_weight(ctrl_value):
        return ctrl_value.count('1')

    if len(remaining_tuple) == 0:
        return assigned_modes

    basis_candidates = []
    ## Base subgroup generators are encoded as single-1 control strings.
    for mode in assigned_modes:
        pauli_label, ctrl_value = mode[0], mode[1]
        if ctrl_value.count('1') == 1:
            basis_candidates.append((pauli_label, ctrl_value))

    if len(basis_candidates) == 0:
        return assigned_modes

    basis_candidates.sort(key=lambda x: int(x[1], 2))
    d = len(basis_candidates)
    if d > w:
        d = w
        basis_candidates = basis_candidates[:d]

    mode_bits = w - d
    if mode_bits < 0:
        return assigned_modes

    basis_info = []
    for idx, (pauli_label, ctrl_value) in enumerate(basis_candidates):
        basis_info.append({
            "index": idx,
            "ctrl_d": bin(1 << idx)[2:].zfill(d),
            "vec": pauli_to_vec(pauli_label),
        })

    ## Enumerate the whole subspace generated by basis vectors:
    ## c in {0,1}^d  ->  v(c) = XOR_i c_i * B_i
    subspace_vectors = []
    vec_len = len(basis_info[0]["vec"])
    for c_int in range(2 ** d):
        coeff_id = bin(c_int)[2:].zfill(d)
        coeff_vec = np.zeros(vec_len, dtype=int)
        for bit_i in range(d):
            ## coeff_id uses standard binary string (MSB on the left), while basis index i
            ## is encoded as 1<<i. Therefore we must read from the right (LSB-first).
            if coeff_id[d - 1 - bit_i] == '1':
                coeff_vec = coeff_vec ^ basis_info[bit_i]["vec"]
        subspace_vectors.append((coeff_id, coeff_vec))

    remaining = []
    ## Backward compatible parsing: accept vec or (vec, phase).
    for item in remaining_tuple:
        if isinstance(item, tuple) or isinstance(item, list):
            vec = item[0]
        else:
            vec = item
        remaining.append(vec.copy().astype(int))

    assigned_mode_keys = {(mode[0], mode[1]) for mode in assigned_modes}

    def append_mode_if_new(label, ctrl_value):
        key = (label, ctrl_value)
        if key in assigned_mode_keys:
            return
        assigned_modes.append((label, ctrl_value))
        assigned_mode_keys.add(key)

    max_modes = 2 ** mode_bits
    zero_coeff_id = '0' * d
    base_subspace_map = {coeff_id: vec for coeff_id, vec in subspace_vectors}

    def greedy_assign_coeffs(remaining_vecs, allowed_coeff_ids, effective_subspace_map):
        if len(remaining_vecs) == 0 or len(allowed_coeff_ids) == 0:
            return {}

        coeff_ids = list(allowed_coeff_ids)
        coeff_vecs = np.asarray([effective_subspace_map[cid] for cid in coeff_ids], dtype=np.uint8)
        rem_vecs = np.asarray(remaining_vecs, dtype=np.uint8)

        n_half = rem_vecs.shape[1] // 2
        rem_x = rem_vecs[:, :n_half]
        rem_z = rem_vecs[:, n_half:]
        coeff_x = coeff_vecs[:, :n_half]
        coeff_z = coeff_vecs[:, n_half:]

        xor_x = rem_x[:, None, :] ^ coeff_x[None, :, :]
        xor_z = rem_z[:, None, :] ^ coeff_z[None, :, :]
        cost_matrix = np.sum(xor_x | xor_z, axis=2)

        used_coeff_mask = np.zeros(len(coeff_ids), dtype=bool)
        assigned_row_mask = np.zeros(len(remaining_vecs), dtype=bool)
        assignments = {}

        while True:
            best_choice = None
            for ridx in range(len(remaining_vecs)):
                if assigned_row_mask[ridx]:
                    continue

                row_best_coeff_idx = -1
                row_best_cost = None
                for coeff_idx in range(len(coeff_ids)):
                    if used_coeff_mask[coeff_idx]:
                        continue

                    cost = int(cost_matrix[ridx, coeff_idx])
                    if (row_best_cost is None) or (cost < row_best_cost):
                        row_best_cost = cost
                        row_best_coeff_idx = coeff_idx

                if row_best_coeff_idx < 0:
                    continue

                choice = (row_best_cost, row_best_coeff_idx, ridx)
                if (best_choice is None) or (choice < best_choice):
                    best_choice = choice

            if best_choice is None:
                break

            _, coeff_idx, ridx = best_choice
            coeff_id = coeff_ids[coeff_idx]
            assignments[ridx] = coeff_id
            used_coeff_mask[coeff_idx] = True
            assigned_row_mask[ridx] = True

        return assignments

    
    ## Enumerate non-zero modes m in the high bits.
    for mode_int in range(1, max_modes):
        if len(remaining) == 0:
            break

        mode_prefix = bin(mode_int)[2:].zfill(mode_bits)

        remaining_modes_count = max_modes - mode_int + 1
        nonzero_capacity_total = remaining_modes_count * ((2 ** d) - 1)
        need_zero_coeff = len(remaining) > nonzero_capacity_total

        nonzero_coeff_ids = [coeff_id for coeff_id, _ in subspace_vectors if coeff_id != zero_coeff_id]
        effective_map = {coeff_id: base_subspace_map[coeff_id] for coeff_id in nonzero_coeff_ids}

        assignments = {}
        zero_anchor_vec = None

        if need_zero_coeff:
            ## Only when nonzero addresses are globally insufficient,
            ## enable coeff=0 and make it a dynamic basis offset in this mode.
            best_zero_idx = min(range(len(remaining)), key=lambda ridx: vec_weight(remaining[ridx]))
            assignments[best_zero_idx] = zero_coeff_id
            zero_anchor_vec = remaining[best_zero_idx].copy()

            shifted_map = {}
            for coeff_id in nonzero_coeff_ids:
                shifted_map[coeff_id] = zero_anchor_vec ^ base_subspace_map[coeff_id]

            reduced_remaining = [vec for ridx, vec in enumerate(remaining) if ridx != best_zero_idx]
            reduced_assignments = greedy_assign_coeffs(reduced_remaining, nonzero_coeff_ids, shifted_map)
            reduced_to_original = [ridx for ridx in range(len(remaining)) if ridx != best_zero_idx]
            for local_idx, coeff_id in reduced_assignments.items():
                assignments[reduced_to_original[local_idx]] = coeff_id
        else:
            ## Default mode: do not use coeff=0 to avoid introducing unwanted shared offsets.
            assignments = greedy_assign_coeffs(remaining, nonzero_coeff_ids, effective_map)

        newly_assigned_indices = set()
        for ridx, coeff_id in assignments.items():
            newly_assigned_indices.add(ridx)
            full_ctrl = mode_prefix + coeff_id
            pauli_label = vec_to_pauli(remaining[ridx])
            append_mode_if_new(pauli_label, full_ctrl)

        remaining = [item for ridx, item in enumerate(remaining) if ridx not in newly_assigned_indices]

    if len(remaining) > 0:
        ctrl_to_pauli_weight = {}
        for pauli_label, ctrl_value in assigned_modes:
            ctrl_to_pauli_weight[ctrl_value] = vec_weight(pauli_to_vec(pauli_label))

        occupied_ctrls = {ctrl for _, ctrl in assigned_modes}
        all_ctrls = [bin(i)[2:].zfill(w) for i in range(2 ** w)]
        zero_ctrl = '0' * w
        free_ctrls = [ctrl for ctrl in all_ctrls if ctrl not in occupied_ctrls and ctrl != zero_ctrl]

        def prefix_assigned_weight_sum(ctrl_value):
            b_int = int(ctrl_value, 2)
            accum = 0
            for l in range(b_int):
                l_ctrl = bin(l)[2:].zfill(w)
                accum += ctrl_to_pauli_weight.get(l_ctrl, 0)
            return accum

        while len(remaining) > 0 and len(free_ctrls) > 0:
            free_ctrls.sort(
                key=lambda ctrl: (
                    ctrl_hamming_weight(ctrl),
                    prefix_assigned_weight_sum(ctrl),
                    int(ctrl, 2),
                )
            )
            target_ctrl = free_ctrls.pop(0)

            coeff_id = target_ctrl[-d:] if d > 0 else ''
            ref_vec = base_subspace_map.get(coeff_id)
            if ref_vec is None:
                continue

            best_ridx = min(
                range(len(remaining)),
                key=lambda ridx: vec_weight(remaining[ridx] ^ ref_vec)
            )

            chosen_vec = remaining.pop(best_ridx)
            chosen_label = vec_to_pauli(chosen_vec)
            append_mode_if_new(chosen_label, target_ctrl)
            ctrl_to_pauli_weight[target_ctrl] = vec_weight(pauli_to_vec(chosen_label))

    return assigned_modes

def build_vec_phase_lookup(matrix, phases):
    vec_phase_lookup = {}
    for idx in range(matrix.shape[0]):
        vec = np.asarray(matrix[idx], dtype=int)
        vec_phase_lookup[vec.tobytes()] = phases[idx] % 4
        
    return vec_phase_lookup

def extract_basis_modes_with_phases(additional_modes, vec_phase_lookup, zero_phase):
    basis_modes = []
    nonbasis_modes = []
    for mode in additional_modes:
        pauli_label, ctrl_value = mode[0], mode[1]
        pauli = Pauli(pauli_label)
        vec = np.hstack((pauli.x.astype(int), pauli.z.astype(int)))
        phase = vec_phase_lookup.get(vec.tobytes(), 0)
        if pauli_label == 'I' * len(pauli_label):
            phase = zero_phase
        mode_with_phase = (pauli_label, ctrl_value, phase)
        if ctrl_value.count('1') == 1:
            basis_modes.append(mode_with_phase)
        else:
            nonbasis_modes.append(mode_with_phase)
    basis_modes.sort(key=lambda x: int(x[1], 2))
    nonbasis_modes.sort(key=lambda x: int(x[1], 2))

    return basis_modes, nonbasis_modes

# def mobius_invert_modes_with_phases_legacy(w, additional_modes, vec_phase_lookup):
#     def vec_to_pauli(vec):
#         n = len(vec) // 2
#         x = vec[:n]
#         z = vec[n:]
#         pauli = Pauli((z, x))
#         return pauli
#     if len(additional_modes) == 0:
#         return {
#             "p_phase_table": np.zeros(2 ** w, dtype=float),
#             "g_phase_table": np.zeros(2 ** w, dtype=float),
#             "phi_ad_table": np.zeros(2 ** w, dtype=float),
#             "basis_modes_with_phase": [],
#             "nonbasis_modes_with_phase": [],
#             "g_modes_with_phase": [],
#         }

#     first_label = additional_modes[0][0]
#     n = len(first_label)
#     vec_len = 2 * n
#     table_size = 2 ** w

#     p_vec_table = np.zeros((table_size, vec_len), dtype=int)
#     p_phase_table = np.zeros(table_size, dtype=float)
#     relevant_addr_mask = np.zeros(table_size, dtype=bool)

#     for mode in additional_modes:
#         pauli_label, ctrl_value = mode[0], mode[1]
#         idx = int(ctrl_value, 2)
#         pauli = Pauli(pauli_label)
#         vec = np.hstack((pauli.x.astype(int), pauli.z.astype(int)))
#         vec_key = vec.tobytes()
#         if vec_key in vec_phase_lookup:
#             ## Relevant entry (belongs to the target U set): keep as hard P-table constraint.
#             p_vec_table[idx] = vec
#             p_phase_table[idx] = vec_phase_lookup[vec_key]
#             relevant_addr_mask[idx] = True
#         else:
#             ## Filler entry: unconstrained in P table; leave as identity/zero phase.
#             p_vec_table[idx] = np.zeros(vec_len, dtype=int)
#             p_phase_table[idx] = 0.0
#     g_vec_table = p_vec_table.copy()
#     g_phase_table = p_phase_table.copy()
#     for i in range(w):
#         bit = 1 << i
#         for b in range(table_size):
#             if b & bit: 
#                 ## First count overlap (phases)
#                 pauli_curr, pauli_basis = vec_to_pauli(g_vec_table[b]), vec_to_pauli(g_vec_table[b ^ bit])
#                 overlap_phase = int(4 - (pauli_curr @ pauli_basis).phase)
#                 g_vec_table[b] = g_vec_table[b] ^ g_vec_table[b ^ bit]
#                 g_phase_table[b] = (g_phase_table[b] - g_phase_table[b ^ bit] - int(overlap_phase)) % 4
                


#     ## For unconstrained (filler) addresses, force G_s = I with zero phase.
#     for idx in range(table_size):
#         if not relevant_addr_mask[idx]:
#             g_vec_table[idx] = np.zeros(vec_len, dtype=int)
#             g_phase_table[idx] = 0.0

#     ## Reconstruct phase(P_b) from G_s by explicit Pauli multiplication order,
#     ## so non-commuting overlaps contribute extra phase consistently.
#     phase_from_g = np.zeros(table_size, dtype=float)
#     for b in range(table_size):
#         accum_vec = np.zeros(vec_len, dtype=int)
#         accum_phase = 0.0
#         for s in range(table_size):
#             if (s & b) != s:
#                 continue
#             if not np.any(g_vec_table[s]) and int(g_phase_table[s]) % 4 == 0:
#                 continue
#             pauli_accum = vec_to_pauli(accum_vec)
#             pauli_s = vec_to_pauli(g_vec_table[s])
#             overlap_phase = int((pauli_s @ pauli_accum).phase) 
#             accum_phase = (accum_phase + g_phase_table[s] + int(overlap_phase)) % 4
#             accum_vec = accum_vec ^ g_vec_table[s]
#         phase_from_g[b] = accum_phase

#     phi_ad_table = (p_phase_table - phase_from_g) % 4

#     basis_modes_with_phase, nonbasis_modes_with_phase = extract_basis_modes_with_phases(additional_modes, vec_phase_lookup)

#     g_modes_with_phase = []
#     for idx in range(table_size):
#         vec = g_vec_table[idx]
#         has_nontrivial_vec = np.any(vec)
#         has_nontrivial_phase = (int(g_phase_table[idx]) % 4) != 0
#         if not has_nontrivial_vec and not has_nontrivial_phase:
#             continue
#         x = vec[:n]
#         z = vec[n:]
#         pauli_label = Pauli((z, x)).to_label()
#         ctrl_value = bin(idx)[2:].zfill(w)
#         g_modes_with_phase.append((pauli_label, ctrl_value, g_phase_table[idx] % 4))

#     return {
#         "p_phase_table": p_phase_table,
#         "g_phase_table": g_phase_table,
#         "phi_ad_table": phi_ad_table,
#         "relevant_addr_mask": relevant_addr_mask,
#         "basis_modes_with_phase": basis_modes_with_phase,
#         "nonbasis_modes_with_phase": nonbasis_modes_with_phase,
#         "g_modes_with_phase": g_modes_with_phase,
#     }


def mobius_invert_modes_with_phases_bottom_up(w, additional_modes, vec_phase_lookup, zero_phase):
    def vec_to_pauli(vec):
        n = len(vec) // 2
        x = vec[:n]
        z = vec[n:]
        return Pauli((z, x))

    if len(additional_modes) == 0:
        return {
            "p_phase_table": np.zeros(2 ** w, dtype=float),
            "g_phase_table": np.zeros(2 ** w, dtype=float),
            "phi_ad_table": np.zeros(2 ** w, dtype=float),
            "basis_modes_with_phase": [],
            "nonbasis_modes_with_phase": [],
            "g_modes_with_phase": [],
        }

    first_label = additional_modes[0][0]
    n = len(first_label)
    vec_len = 2 * n
    table_size = 2 ** w

    p_vec_table = np.zeros((table_size, vec_len), dtype=int)
    p_phase_table = np.zeros(table_size, dtype=float)
    relevant_addr_mask = np.zeros(table_size, dtype=bool)

    for mode in additional_modes:
        pauli_label, ctrl_value = mode[0], mode[1]
        idx = int(ctrl_value, 2)
        pauli = Pauli(pauli_label)
        vec = np.hstack((pauli.x.astype(int), pauli.z.astype(int)))
        vec_key = vec.tobytes()
        if vec_key in vec_phase_lookup:
            p_vec_table[idx] = vec
            p_phase_table[idx] = vec_phase_lookup[vec_key] % 4
            relevant_addr_mask[idx] = True
        elif vec_key == (np.zeros(vec_len, dtype=int)).tobytes() and zero_phase != 0.0:
            p_vec_table[idx] = np.zeros(vec_len, dtype=int)
            p_phase_table[idx] = zero_phase % 4
            relevant_addr_mask[idx] = True
        else:
            p_vec_table[idx] = np.zeros(vec_len, dtype=int)
            p_phase_table[idx] = 0
    if zero_phase != 0.0:
        p_phase_table[0] = zero_phase % 4
    g_vec_table = np.zeros_like(p_vec_table)
    g_phase_table = np.zeros_like(p_phase_table)

    for b in range(table_size):
        if not relevant_addr_mask[b]:
            continue

        # Include g_0 even for one-hot addresses: g_b = P_b F_b^dagger.
        curr_vec = p_vec_table[b].copy()
        curr_phase = p_phase_table[b] % 4

        # for c in range(1, b):
        for c in range(b):
            if (c & b) != c:
                continue
            if not relevant_addr_mask[c]:
                continue
            if not np.any(g_vec_table[c]) and g_phase_table[c] % 4 == 0.0:
                continue

            pauli_curr = vec_to_pauli(curr_vec)
            pauli_basis = vec_to_pauli(g_vec_table[c])
            overlap_phase = 4 - (pauli_curr @ pauli_basis).phase % 4 if (pauli_curr @ pauli_basis).phase != 0 else 0

            curr_vec = curr_vec ^ g_vec_table[c]
            curr_phase = (curr_phase - g_phase_table[c] + overlap_phase) % 4

        g_vec_table[b] = curr_vec
        g_phase_table[b] = curr_phase

    phase_from_g = np.zeros(table_size, dtype=float)
    phase_from_g[0] = g_phase_table[0]
    for b in range(1, table_size):
        accum_vec = np.zeros(vec_len, dtype=int)
        accum_phase = 0
        for s in range(table_size):
            if (s & b) != s:
                continue
            if not np.any(g_vec_table[s]) and g_phase_table[s] % 4 == 0.0:
                continue
            pauli_accum = vec_to_pauli(accum_vec)
            pauli_s = vec_to_pauli(g_vec_table[s])
            overlap_phase = 4 - (pauli_s @ pauli_accum).phase % 4 if (pauli_s @ pauli_accum).phase != 0 else 0
            accum_phase = (accum_phase + g_phase_table[s] + int(overlap_phase)) % 4
            accum_vec = accum_vec ^ g_vec_table[s]
        phase_from_g[b] = accum_phase

    phi_ad_table = (p_phase_table - phase_from_g) % 4
    additional_indices = {int(mode[1], 2) for mode in additional_modes}
    for idx in additional_indices:
        g_phase_table[idx] = g_phase_table[idx] + phi_ad_table[idx] % 4

    basis_modes_with_phase, nonbasis_modes_with_phase = extract_basis_modes_with_phases(additional_modes, vec_phase_lookup, zero_phase)

    g_modes_with_phase = []
    for idx in range(table_size):
        vec = g_vec_table[idx]
        has_nontrivial_vec = np.any(vec)
        has_nontrivial_phase = (g_phase_table[idx]) % 4 != 0
        if not has_nontrivial_vec and not has_nontrivial_phase:
            continue
        x = vec[:n]
        z = vec[n:]
        pauli_label = Pauli((z, x)).to_label()
        ctrl_value = bin(idx)[2:].zfill(w)
        g_phase_value = g_phase_table[idx] % 4 if g_phase_table[idx] < 2.0 else g_phase_table[idx] % 4 - 4
        g_modes_with_phase.append((pauli_label, ctrl_value, g_phase_value))

    return {
        "p_phase_table": p_phase_table,
        "g_phase_table": g_phase_table,
        "phi_ad_table": phi_ad_table,
        "relevant_addr_mask": relevant_addr_mask,
        "basis_modes_with_phase": basis_modes_with_phase,
        "nonbasis_modes_with_phase": nonbasis_modes_with_phase,
        "g_modes_with_phase": g_modes_with_phase,
    }


def mobius_invert_modes_with_phases(w, additional_modes, vec_phase_lookup, zero_phase, method="bottom-up"):
    return mobius_invert_modes_with_phases_bottom_up(w, additional_modes, vec_phase_lookup, zero_phase)

class _DirectBlockEncoding:
    """
    Block Encoding Class for a given Matrixsum Operator J.
    Constructs the block-encoding circuit for the operator and provides
    resource estimation such as ancilla qubit usage and multi-qubit gate counts.
    """
    def __init__(self, J: Matrixsum):
        self.J = J
        self.coeff_list = [coeff for _, coeff in J.instances]
        self.mat_list = []
        for matrix, _ in J.instances:
            if isinstance(matrix, PauliAtom):
                self.mat_list.append((matrix.expr, matrix.phase))
            else:
                self.mat_list.append(matrix.to_operator().data)
        self.ctrl_size = int(np.ceil(np.log2(len(self.coeff_list))))
        self.sys_size = J.size
        self.circuit_width = 0
    ## Basic version of multiplexed_u implementation: directly control each unitary by the control register, without optimization over the structure of the matrices.
    def mulplex_U(self, mat_list, ctrl_size, sys_size):
        tcount, mccount, cxcount = 0, 0, 0
        t_count_per_ctrl = 4 
        cx_count_per_ctrl = 4
        mccount = 0
        if ctrl_size == 0:
            qc = QuantumCircuit(sys_size)
            assert len(mat_list) == 1
            pauli_op, phase = Pauli(mat_list[0][0]), mat_list[0][1]
            qc_pauli = QuantumCircuit(pauli_op.num_qubits) #type: ignore
            qc_pauli.append(pauli_op, range(pauli_op.num_qubits)) #type: ignore
            qc_pauli.global_phase = np.angle(phase)
            qc_pauli = qc_pauli.decompose()
            return qc_pauli, 0, 0, 0
    
        qc = QuantumCircuit(ctrl_size + sys_size)
        ### For test: genearte the order for matrices
        # ctrlv_dict = {"ZII": "0100", "IZI": "0010", "IIZ": "0001", "ZZI": "0110", "ZIZ": "0101", "IZZ": "0011", "XII": "1100", "IXI": "1010", "IIX": "1001"}
        for i, ms in enumerate(mat_list):
            
            if isinstance(ms, tuple) or isinstance(ms, list):
                pauli_op, phase = Pauli(ms[0]), ms[1]
                qc_pauli = QuantumCircuit(pauli_op.num_qubits) #type:ignore
                qc_pauli.append(pauli_op, range(pauli_op.num_qubits)) #type: ignore
                qc_pauli.global_phase = np.angle(phase)
                qc_pauli = qc_pauli.decompose()
                U_elem = qc_pauli.to_gate()
                
            else:
                if ms.shape[0] < 2**sys_size:
                    pad_size = 2**sys_size // ms.shape[0]
                    ms = np.kron(ms, np.eye(pad_size))
            
            control_values =  bin(i)[2:].zfill(ctrl_size) 
            # control_values = ctrlv_dict.get(label)
            # if len(qc_pauli.data) > 0 : ### Identity is ignored
            ctrl_U_elem = U_elem.control(num_ctrl_qubits = ctrl_size, ctrl_state = control_values)
            qc.append(ctrl_U_elem, range(ctrl_size + sys_size))
            if len(qc_pauli.data) > 0 : ### Identity is ignored
                mccount += (ctrl_size - 1) * len(qc_pauli.data)
            else:
                mccount += ctrl_size - 2
                
        tcount = mccount * t_count_per_ctrl
        cxcount = mccount * cx_count_per_ctrl
        return qc, tcount, mccount, cxcount
    ## Optimized version of multiplexed_u implementation (Babbush 2018)
    def mulplex_U_opt(self, mat_list, ctrl_size, sys_size, gateable=True):

        from qiskit.circuit.library import XGate
        mccount = 0
        tcount = 0
        cxcount = 0
        t_counts_per_ccx = 4
        cx_counts_per_ccx = 4  
        
        opt_circuit = QuantumCircuit(2 * ctrl_size + sys_size)
        # sel_regs = [2 * j  for j in range(ctrl_size)]
        # anc_regs = [2 * j + 1 for j in range(ctrl_size)]
        anc_regs = [2 * (ctrl_size - 1 - j) for j in range(ctrl_size)]
        sel_regs = [2 * (ctrl_size - 1 - j) + 1 for j in range(ctrl_size)]
        def apply_left_enc(j):
            if not gateable:
                opt_circuit.reset(anc_regs[j])
            ctrl_bval = control_values[j]
            ccxgate_c = XGate().control(num_ctrl_qubits = 2, ctrl_state = ctrl_bval + '1')
            cxgate_c  = XGate().control(num_ctrl_qubits = 1, ctrl_state = ctrl_bval)
            if j == 0:
                opt_circuit.append(cxgate_c, [2 * ctrl_size - 1, anc_regs[0]])
            # top = 0 if j == 0 else anc_regs[j - 1]
            else:
                opt_circuit.append(ccxgate_c, [anc_regs[j - 1], sel_regs[j], anc_regs[j]])
        def apply_right_enc(j):
            ctrl_bval = control_values[j]
            ccxgate_c = XGate().control(num_ctrl_qubits = 2, ctrl_state = ctrl_bval + '1')
            cxgate_c  = XGate().control(num_ctrl_qubits = 1, ctrl_state = ctrl_bval)
            # top = 0 if j == 0 else anc_regs[j - 1]
            if j == 0:
                opt_circuit.append(cxgate_c, [2 * ctrl_size - 1, anc_regs[0]])
            else:
                opt_circuit.append(ccxgate_c, [anc_regs[j - 1], sel_regs[j], anc_regs[j]])
            if not gateable:
                opt_circuit.reset(anc_regs[j])

        maxctrl_value = bin(len(mat_list) - 1)[2:].zfill(ctrl_size)

        def find_bit_to_remove(max_val, cur_val):
            candidate_bits = []
            for j in range(ctrl_size):
                if max_val[j] != cur_val[j]:
                    return candidate_bits
                elif max_val[j] == '0' and cur_val[j] == '0':
                    candidate_bits.append(j)
                else:
                    continue
            return candidate_bits
        
        for i, mat in enumerate(mat_list):
            
            pauli_op, phase = Pauli(mat[0]), mat[1]
            
            qc_pauli = QuantumCircuit(pauli_op.num_qubits) #type: ignore 
            qc_pauli.append(pauli_op, range(pauli_op.num_qubits)) #type: ignore
            qc_pauli.global_phase = np.angle(phase)
            qc_pauli = qc_pauli.decompose()
            numq = qc_pauli.num_qubits
            control_values = bin(i)[2:].zfill(ctrl_size)
            # index_remove = find_bit_to_remove(maxctrl_value, control_values)
            pauli_length = len(qc_pauli.data)

            if i == 0:
                cval_next = bin(1)[2:].zfill(ctrl_size)   
                for j in range(ctrl_size):
                    apply_left_enc(j)
                    mccount += 1
                    tcount += t_counts_per_ccx
                    cxcount += cx_counts_per_ccx
                # opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size - 1 , 2 * ctrl_size + numq)))
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), [0] + list(range(2 * ctrl_size, 2 * ctrl_size + numq)))
                if ctrl_size >= 2:
                    opt_circuit.cx(anc_regs[ctrl_size - 2], anc_regs[ctrl_size - 1])
                    cxcount += 1 + pauli_length
                else:
                    # For a single ancilla line, CX(a, a) is invalid in Qiskit.
                    # Use X to advance the ancilla state for the next control value.
                    opt_circuit.x(anc_regs[0])
                    cxcount += pauli_length
            elif i == len(mat_list) - 1:
                cval_prev = bin(i - 1)[2:].zfill(ctrl_size)
                diff_prev = next(j for j in range(ctrl_size) if cval_prev[j] != control_values[j])
                if diff_prev != ctrl_size - 1:
                    for j in range(diff_prev + 1, ctrl_size):
                        apply_left_enc(j)
                        mccount += 1
                        tcount += t_counts_per_ccx
                        cxcount += cx_counts_per_ccx
                # opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size - 1,2 * ctrl_size + numq)))
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), [0] + list(range(2 * ctrl_size, 2 * ctrl_size + numq)))
                cxcount += pauli_length
                for j in range(ctrl_size - 1, -1, -1):
                    apply_right_enc(j)
                    mccount += 1
                    tcount += t_counts_per_ccx
                    cxcount += cx_counts_per_ccx
            else:
                cval_prev, cval_next = bin(i - 1)[2:].zfill(ctrl_size), bin(i + 1)[2:].zfill(ctrl_size)
                ## Find the first bit that differs
                diff_prev = next(j for j in range(ctrl_size) if cval_prev[j] != control_values[j])
                ## Apply left encodings from diff_prev to the end
                if diff_prev != ctrl_size - 1:
                    for j in range(diff_prev + 1, ctrl_size):
                        apply_left_enc(j)
                        mccount += 1
                        tcount += t_counts_per_ccx
                        cxcount += cx_counts_per_ccx
                ## Apply the controlled circuit
                # opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size - 1, 2 * ctrl_size  + numq)))
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), [0] + list(range(2 * ctrl_size, 2 * ctrl_size + numq)))
                cxcount += pauli_length
                diff_next = next(j for j in range(ctrl_size) if cval_next[j] != control_values[j])
                ## Apply right encodings from diff_next to the end
                if diff_next != ctrl_size - 1:
                    for j in range(ctrl_size - 1, diff_next, -1):
                        apply_right_enc(j)
                        mccount += 1
                        tcount += t_counts_per_ccx
                        cxcount += cx_counts_per_ccx
                ## Apply a CX to flip the differed bit
                if diff_next != 0:
                    opt_circuit.cx(anc_regs[diff_next - 1], anc_regs[diff_next])
                    cxcount += 1
                else:
                    # opt_circuit.cx(0, anc_regs[0])
                    opt_circuit.x(anc_regs[0])
        # opt_circuit.x(0)
        return opt_circuit, tcount, mccount, cxcount

    def mulplex_U_tree(self, mat_list, ctrl_size, sys_size):
        """
        Balanced-tree unary-iteration SELECT.

        This mirrors Qrisp's default q_switch(..., method="tree") traversal:
        a path predicate is maintained in ``ctrl_size`` clean ancillas, two
        neighboring leaves are handled per leaf visit, and the path is moved by
        up/bounce/down operations instead of recomputing a full equality test
        for every SELECT branch.
        """
        if ctrl_size == 0:
            return self.mulplex_U(mat_list, ctrl_size, sys_size)

        ctrl_reg = QuantumRegister(ctrl_size, 'ctrl')
        anc_reg = QuantumRegister(ctrl_size, 'tree_anc')
        sys_reg = QuantumRegister(sys_size, 'sys')
        qc = QuantumCircuit(ctrl_reg, anc_reg, sys_reg)

        mccount = 0
        cxcount = 0

        def ctrl_qubit(depth):
            return ctrl_reg[ctrl_size - 1 - depth]

        def anc_qubit(depth):
            return anc_reg[depth]

        def append_cx(control, target):
            nonlocal cxcount
            qc.cx(control, target)
            cxcount += 1

        def append_ccx(control_a, control_b, target):
            nonlocal mccount, cxcount
            qc.ccx(control_a, control_b, target)
            mccount += 1
            cxcount += 4

        def append_pauli_branch(branch_index, control):
            nonlocal cxcount
            if branch_index >= len(mat_list):
                return

            ms = mat_list[branch_index]
            if isinstance(ms, tuple) or isinstance(ms, list):
                pauli_op, phase = Pauli(ms[0]), ms[1]
                qc_pauli = QuantumCircuit(pauli_op.num_qubits)  # type: ignore
                qc_pauli.append(pauli_op, range(pauli_op.num_qubits))  # type: ignore
                qc_pauli.global_phase = np.angle(phase)
                qc_pauli = qc_pauli.decompose()
                controlled = qc_pauli.to_gate().control(num_ctrl_qubits=1, ctrl_state='1')
                qc.append(controlled, [control] + list(sys_reg))
                cxcount += sum(1 for ch in str(pauli_op.to_label()) if ch != 'I')
            else:
                if ms.shape[0] < 2**sys_size:
                    pad_size = 2**sys_size // ms.shape[0]
                    ms = np.kron(ms, np.eye(pad_size))
                controlled = UnitaryGate(ms).control(num_ctrl_qubits=1, ctrl_state='1')
                qc.append(controlled, [control] + list(sys_reg))

        def down(depth):
            ctrl = ctrl_qubit(depth)
            target = anc_qubit(depth)
            qc.x(ctrl)
            if depth == 0:
                append_cx(ctrl, target)
            else:
                append_ccx(ctrl, anc_qubit(depth - 1), target)
            qc.x(ctrl)

        def up(depth):
            ctrl = ctrl_qubit(depth)
            target = anc_qubit(depth)
            if depth == 0:
                append_cx(ctrl, target)
            else:
                append_ccx(ctrl, anc_qubit(depth - 1), target)

        def bounce(depth):
            if depth <= 0:
                raise ValueError("Tree bounce depth must be positive.")
            previous = anc_qubit(depth - 1)
            target = anc_qubit(depth)
            if depth == 1:
                qc.x(previous)
            else:
                append_cx(anc_qubit(depth - 2), previous)

            append_cx(previous, target)

            ctrl = ctrl_qubit(depth)
            if depth == 1:
                append_cx(ctrl, target)
            else:
                append_ccx(ctrl, anc_qubit(depth - 2), target)

        def leaf(first_branch):
            leaf_ctrl = anc_qubit(ctrl_size - 1)
            append_pauli_branch(first_branch, leaf_ctrl)

            if ctrl_size == 1:
                qc.x(leaf_ctrl)
            else:
                append_cx(anc_qubit(ctrl_size - 2), leaf_ctrl)

            append_pauli_branch(first_branch + 1, leaf_ctrl)

        def last_leaf(branch_index):
            append_pauli_branch(branch_index, anc_qubit(ctrl_size - 1))

        def bitwise_count_diff(left, right):
            return int((left ^ right).bit_count())

        branch_amount = len(mat_list)

        for depth in range(ctrl_size):
            down(depth)

        for pos in range(0, -(-branch_amount // 2) - 1):
            leaf(2 * pos)

            diff_depth = bitwise_count_diff(pos, pos + 1)
            for j in range(0, diff_depth - 1):
                up(ctrl_size - j - 1)
            bounce(ctrl_size - diff_depth)
            for j in range(0, diff_depth - 1):
                down(ctrl_size - (diff_depth - 1) + j)

        if branch_amount % 2 == 0:
            leaf(branch_amount - 2)
        else:
            last_leaf(branch_amount - 1)

        diff = 2**ctrl_size - branch_amount
        for j in range(ctrl_size):
            depth = ctrl_size - j - 1
            up(depth)
            if (diff >> j) & 1:
                if depth == 0:
                    qc.x(anc_qubit(depth))
                else:
                    append_cx(anc_qubit(depth - 1), anc_qubit(depth))

        tcount = 4 * mccount
        return qc, tcount, mccount, cxcount
    

            
    def mulplex_B_opt_order(self, ctrl_size):
        coeff_list_ordered = self.coeff_list_ordered
        sum_coeff = sum([abs(c) for c in coeff_list_ordered])
        norm_coeffs = [abs(c)/sum_coeff for c in coeff_list_ordered]
        amps = np.zeros(2**ctrl_size, dtype = float)
        for i, nc in enumerate(norm_coeffs):

            amps[i] = np.sqrt(nc)
        
        # qc = lcu_prepare_tree(probs) 
        qc = QuantumCircuit(ctrl_size)
        qc.append(StatePreparation(Statevector(amps)), range(ctrl_size))
        # qc = lcu_prepare_tree(norm_coeffs)

        return qc
    @staticmethod
    def _count_ctrl_qubits(qc: QuantumCircuit):
        cxcount = 0
        mccount = 0
        for inst, _, _  in qc.data:
            if isinstance(inst, ControlledGate):
                if inst.num_ctrl_qubits == 1:
                    cxcount += 1
                else:
                    mccount += inst.num_ctrl_qubits - 1
            else:
                cxcount += getattr(inst, 'num_ctrl_qubits', 0)
        return cxcount, mccount
    
    ## Coefficient encoding circuit
    def mulplex_B(self, coeff_list, ctrl_size):
        cx_per_toffoli = 4
        sum_coeff = sum([abs(c) for c in coeff_list])
        norm_coeffs = [abs(c)/sum_coeff for c in coeff_list]

        probs = np.zeros(2**ctrl_size, dtype = float)
        amps = np.zeros(2**ctrl_size, dtype = float)
        for i, nc in enumerate(norm_coeffs):
            probs[i] = nc
            amps[i] = np.sqrt(nc)

        # qc = lcu_prepare_tree(probs) 
        # print(probs)
        qc = QuantumCircuit(ctrl_size)
        qc.append(StatePreparation(Statevector(amps)), range(ctrl_size))
        
        return qc #type: ignore

    def prepare_tree_with_side_effect_count(self, weights):
        """
        Build the same recursive coefficient-preparation tree as `lcu_prepare_tree`,
        while counting effective k-controlled gate resources during recursion.

        This method is intentionally standalone and is NOT wired into `circuit()`.
        """
        weights = np.asarray(weights, dtype=float)
        if len(weights) == 0:
            return QuantumCircuit(0), {
                "k_ctrl_ry_count": {},
                "k_ctrl_x_count": {},
                "k_ctrl_x_after_ry_decomp": {},
                "mc_control_count": 0,
                "t_count": 0,
                "cx_count": 0,
                "non_clifford_count": 0,
            }

        n = int(np.log2(len(weights)))
        assert 2**n == len(weights)

        k_ctrl_ry_count = {}
        k_ctrl_x_count = {}

        def add_count(table, key, value=1):
            table[key] = table.get(key, 0) + value

        def recurse(level, probs, inherited_ctrls):
            if level == n:
                return None

            half = len(probs) // 2
            p0 = float(np.sum(probs[:half]))
            p1 = float(np.sum(probs[half:]))
            if p0 + p1 == 0:
                return None

            theta = 2 * np.arccos(np.sqrt(p0 / (p0 + p1)))
            t = n - level - 1
            qc_local = QuantumCircuit(t + 1)
            qc_local.ry(theta, t)
            add_count(k_ctrl_ry_count, inherited_ctrls, 1)

            qc_sub = recurse(level + 1, probs[:half], inherited_ctrls + 1)
            if qc_sub is not None:
                qc_local.x(t)
                add_count(k_ctrl_x_count, inherited_ctrls, 1)
                u_sub = qc_sub.to_gate().control(1, ctrl_state='1')
                qc_local.append(u_sub, [qc_local.qubits[-1]] + qc_local.qubits[:-1])
                qc_local.x(t)
                add_count(k_ctrl_x_count, inherited_ctrls, 1)

            qc_sub = recurse(level + 1, probs[half:], inherited_ctrls + 1)
            if qc_sub is not None:
                u_sub = qc_sub.to_gate().control(1, ctrl_state='1')
                qc_local.append(u_sub, [qc_local.qubits[-1]] + qc_local.qubits[:-1])

            return qc_local

        qc = recurse(0, weights, 0)
        if qc is None:
            qc = QuantumCircuit(n)

        ## User-specified decomposition rule:
        ## k-controlled RY -> 2 * (k-controlled X) + 2 * RY
        k_ctrl_x_after_ry_decomp = dict(k_ctrl_x_count)
        for k, cnt in k_ctrl_ry_count.items():
            k_ctrl_x_after_ry_decomp[k] = k_ctrl_x_after_ry_decomp.get(k, 0) + 2 * cnt

        ## Convert k-controlled X into MC-control counts with rule:
        ## t-controlled X contributes (t - 1) MC controls for t > 1.
        mc_control_count = 0
        for k, cnt in k_ctrl_x_after_ry_decomp.items():
            if k > 1:
                mc_control_count += (k - 1) * cnt

        t_count = 4 * mc_control_count
        cx_count = 4 * mc_control_count + k_ctrl_x_after_ry_decomp.get(1, 0)
        non_clifford_count = t_count + 2 * sum(k_ctrl_ry_count.values())

        stats = {
            "k_ctrl_ry_count": k_ctrl_ry_count,
            "k_ctrl_x_count": k_ctrl_x_count,
            "k_ctrl_x_after_ry_decomp": k_ctrl_x_after_ry_decomp,
            "mc_control_count": mc_control_count,
            "t_count": t_count,
            "cx_count": cx_count,
            "non_clifford_count": non_clifford_count,
        }
        return qc, stats
    def select_circuit(self, opt = None, gateable=True):
        if opt == 'No':
            qc_u, tcount, mccount, cxcount = self.mulplex_U(self.mat_list, self.ctrl_size, self.sys_size)
            return qc_u
        elif opt == 'Ctrl-line':
            qc_u, tcount, mccount, cxcount = self.mulplex_U_opt(self.mat_list, self.ctrl_size, self.sys_size, gateable=gateable)
            return qc_u
        elif opt == 'Tree':
            qc_u, tcount, mccount, cxcount = self.mulplex_U_tree(self.mat_list, self.ctrl_size, self.sys_size)
            return qc_u
        elif opt == 'Matrix-order':
            qc_u, tcount, mccount, cxcount, ctrl_size = self.mulplex_U_opt_order()
            return qc_u
    def circuit(self, opt = None, gateable=True):
        """
        Returns the block-encoding QuantumCircuit for the operator J.
        """
        self.mccount = 0
        self.cxcount = 0
        
        if opt == 'No':
            qc_u, tcount, mccount, cxcount = self.mulplex_U(self.mat_list, self.ctrl_size, self.sys_size)
            ctrl = QuantumRegister(self.ctrl_size, 'ctrl')
            sys = QuantumRegister(self.sys_size, 'sys')
            qc = QuantumCircuit(ctrl, sys)
            qc_select = self.mulplex_B(self.coeff_list, self.ctrl_size)
            qc.compose(qc_select, qubits=ctrl, inplace=True) #type: ignore
            qc.compose(qc_u, qubits=qc.qubits, inplace=True)
            qc.compose(qc_select.inverse(), qubits=ctrl, inplace=True) #type: ignore
        elif opt == 'Ctrl-line':
            qc_u, tcount, mccount, cxcount = self.mulplex_U_opt(self.mat_list, self.ctrl_size, self.sys_size, gateable=gateable)
            qc = QuantumCircuit(qc_u.num_qubits, name = "BlockEncoding")
            qc_select = self.mulplex_B(self.coeff_list, self.ctrl_size)
            ctrl_index = [2 * j + 1 for j in range(self.ctrl_size)]
            qc.compose(qc_select, qubits = ctrl_index, inplace = True) #type: ignore
            qc.compose(qc_u, qubits = qc.qubits, inplace = True)
            qc.compose(qc_select.inverse(), qubits = ctrl_index, inplace = True) #type: ignore
        elif opt == 'Tree':
            qc_u, tcount, mccount, cxcount = self.mulplex_U_tree(
                self.mat_list,
                self.ctrl_size,
                self.sys_size,
            )
            ctrl = QuantumRegister(self.ctrl_size, 'ctrl')
            anc = QuantumRegister(self.ctrl_size, 'tree_anc')
            sys = QuantumRegister(self.sys_size, 'sys')
            qc = QuantumCircuit(ctrl, anc, sys, name='BlockEncoding')
            qc_select = self.mulplex_B(self.coeff_list, self.ctrl_size)
            qc.compose(qc_select, qubits=ctrl, inplace=True)  # type: ignore
            qc.compose(qc_u, qubits=list(ctrl) + list(anc) + list(sys), inplace=True)
            qc.compose(qc_select.inverse(), qubits=ctrl, inplace=True)  # type: ignore
        self.tcount = tcount
        self.mccount += mccount
        self.cxcount = cxcount
        self.succ_prob = np.sum(self.coeff_list)
        self.circuit_width = qc.num_qubits
        self._lcu_parts = (
            qc_select, qc_u,
            tuple(2 * j + 1 for j in range(self.ctrl_size))
            if opt == 'Ctrl-line' else tuple(range(self.ctrl_size)),
        )
        return qc

    def pauli_norm(self):
        """
        Returns the sum of coefficients (success probability).
        """
        return self.J.pauli_norm()

    def resource_counts(self):
        """
        Returns a dictionary with resource estimates: ancilla qubits and multi-qubit gate counts.
        """
        qc = self.circuit()
        multiq, tcount = count_multiq_gates(qc)
        return {
            "ancilla_qubits": self.ctrl_size,
            "circuit_width": self.circuit_width,
            "multi_controlled_gates": self.mccount,
            "t_gates": self.tcount,
        }
    

class AlgCircuitSimulator:
    """
    Base class: 
    Simulator for the algorithmic circuits 
    designed by block-encoding and LCU methods. 
    This class can be used to simulate the final statevector or density matrix,
    and find the final density matrix on system qubits according to the register sizes. 
    """
    def __init__(self, circuit: QuantumCircuit, reg_sizes: list[int]):
        self.circuit = circuit
        self.reg_sizes = reg_sizes
        self.transpiled_circuit = None

    def simulate(self, *args, **kwargs):
        raise NotImplementedError("simulate must be implemented in subclasses.")
    def transpile_circuit(self, *args, **kwargs):
        raise NotImplementedError("transpile_circuit must be implemented in subclasses.")


class AlgCircuitSVSimulator(AlgCircuitSimulator):
    def transpile_circuit(self, gate_class: list[str] | None = None, optimization_level: int = 1):
        backend = _require_aer_simulator()(method="statevector")
        self.transpiled_circuit = transpile(
            self.circuit,
            backend=backend,
            basis_gates=gate_class,
            optimization_level=optimization_level,
        )
        multiq, tcount = count_multiq_gates(self.transpiled_circuit)
        return self.transpiled_circuit, {"multi_qubit_gates": multiq, "t_gates": tcount}


    def simulate(self, initial_state: Statevector | None = None):
        qc = self.transpiled_circuit or self.circuit
        if initial_state is None:
            return Statevector.from_instruction(qc)
        qc_sim = QuantumCircuit(qc.qubits, qc.clbits)
        qc_sim.initialize(initial_state, range(len(initial_state)))  # type: ignore
        qc_sim.compose(qc, qc.qubits, qc.clbits, inplace=True)
        simulator = _require_aer_simulator()(method="statevector")

        self.result_sv = simulator.run(qc_sim, shots = 1).result().data['final_state']


    def purification_sys(self):
        sv = self.result_sv
        total_dens = DensityMatrix(sv)
        sel_size, ctrl_size, sys_size = self.reg_sizes
        proj_0 = Operator.from_label('0' * ctrl_size) ## ctrl_register must be 0
        idenf = lambda x: Operator.from_label('I' * x)
        proj_full = idenf(sys_size).tensor(proj_0).tensor(idenf(sel_size))
        projected_dens = DensityMatrix(np.array(proj_full @ total_dens @ proj_full))
        system_dens = partial_trace(projected_dens, list(range(sel_size + ctrl_size)))
        self.dens_sys = system_dens 

        return system_dens


        
class AlgCircuitTNSimulator(AlgCircuitSimulator):
    def transpile_circuit(self, gate_class: list[str] | None = None, optimization_level: int = 1):
        backend = _require_aer_simulator()(method="matrix_product_state")
        self.transpiled_circuit = transpile(
            self.circuit,
            backend=backend,
            basis_gates=gate_class,
            optimization_level=optimization_level,
        )
        multiq, tcount = count_multiq_gates(self.transpiled_circuit)
        return self.transpiled_circuit, {"multi_qubit_gates": multiq, "t_gates": tcount}

    def simulate(self, initial_state: Statevector | None = None, bond_dim: int = 64):
        qc = self.transpiled_circuit or self.circuit
        simulator = _require_aer_simulator()(method="matrix_product_state")
        simulator.set_options(matrix_product_state_max_bond_dimension=bond_dim)
        qc_sim = QuantumCircuit(qc.qubits, qc.clbits)
        if initial_state is not None:
            qc_sim.initialize(initial_state)
        qc_sim.compose(qc, qc.qubits, qc.clbits, inplace=True)
        qc_sim.save_density_matrix(label="final_dm")  # type: ignore
        qc_sim = transpile(qc_sim, simulator, optimization_level=1)
        result = simulator.run(qc_sim, shots=1).result()
        return DensityMatrix(result.data()["final_dm"])
    
class Channels: 
    pass


def _pauli_label_to_symplectic_vector(label: str) -> np.ndarray:
    pauli = Pauli(label)
    return np.hstack((pauli.x.astype(int), pauli.z.astype(int))).astype(int)


def _ctrl_state_bits(gate: ControlledGate) -> str:
    """Return control-state bits in Qiskit's control-string convention."""
    ctrl_count = int(gate.num_ctrl_qubits)
    ctrl_state = gate.ctrl_state
    if ctrl_state is None:
        ctrl_state = (1 << ctrl_count) - 1
    return format(int(ctrl_state), f"0{ctrl_count}b")


def _instruction_controls_targets(instruction):
    op = instruction.operation
    qargs = list(instruction.qubits)
    if isinstance(op, ControlledGate):
        ctrl_count = int(op.num_ctrl_qubits)
        return qargs[:ctrl_count], qargs[ctrl_count:]
    return [], qargs


def _is_controlled_x_gate(op) -> bool:
    return isinstance(op, ControlledGate) and getattr(op.base_gate, "name", None) == "x"


def _clean_pauli_label(label: object, qubit_count: int) -> str | None:
    if label is None:
        return None
    cleaned = str(label).replace(" ", "")
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    if len(cleaned) != qubit_count:
        return None
    cleaned = cleaned.upper()
    if any(char not in {"I", "X", "Y", "Z"} for char in cleaned):
        return None
    return cleaned


def _pauli_label_from_operation(operation, qubit_count: int) -> str | None:
    label_method = getattr(operation, "to_label", None)
    if callable(label_method):
        label = _clean_pauli_label(label_method(), qubit_count)
        if label is not None:
            return label

    for attr_name in ("label", "name"):
        label = _clean_pauli_label(getattr(operation, attr_name, None), qubit_count)
        if label is not None:
            return label

    for param in getattr(operation, "params", ()):
        label = _clean_pauli_label(param, qubit_count)
        if label is not None:
            return label
        label_method = getattr(param, "to_label", None)
        if callable(label_method):
            label = _clean_pauli_label(label_method(), qubit_count)
            if label is not None:
                return label

    return None


def _pauli_ops_from_label(label: str) -> list[tuple[str, int]]:
    return [
        (char, len(label) - 1 - label_index)
        for label_index, char in enumerate(label)
        if char != "I"
    ]


def _single_qubit_pauli_data_from_definition(
    operation,
    seen: set[int] | None = None,
) -> dict[str, object] | None:
    if seen is None:
        seen = set()
    operation_id = id(operation)
    if operation_id in seen:
        return None
    seen.add(operation_id)

    definition = getattr(operation, "definition", None)
    if definition is None:
        return None

    pauli_by_index = ["I"] * int(operation.num_qubits)
    phase_angle = getattr(definition, "global_phase", 0)

    def apply_pauli(target_index: int, pauli: str) -> bool:
        nonlocal phase_angle
        if target_index < 0 or target_index >= len(pauli_by_index):
            return False
        if pauli == "I":
            return True
        current = pauli_by_index[target_index]
        if current == "I":
            pauli_by_index[target_index] = pauli
            return True
        if current == pauli:
            pauli_by_index[target_index] = "I"
            return True

        product_table = {
            ("X", "Y"): ("Z", np.pi / 2),
            ("Y", "X"): ("Z", -np.pi / 2),
            ("Y", "Z"): ("X", np.pi / 2),
            ("Z", "Y"): ("X", -np.pi / 2),
            ("Z", "X"): ("Y", np.pi / 2),
            ("X", "Z"): ("Y", -np.pi / 2),
        }
        result = product_table.get((pauli, current))
        if result is None:
            return False
        pauli_by_index[target_index], phase_delta = result
        phase_angle += phase_delta
        return True

    for instruction in definition.data:
        if instruction.clbits:
            return None
        op = instruction.operation
        qargs = list(instruction.qubits)
        label = _pauli_label_from_operation(op, len(qargs))
        if label is not None:
            for char, offset in _pauli_ops_from_label(label):
                target_index = definition.find_bit(qargs[offset]).index
                if not apply_pauli(target_index, char):
                    return None
            continue

        nested_pauli_data = _single_qubit_pauli_data(op, set(seen))
        if nested_pauli_data is not None:
            phase_angle += nested_pauli_data["phase"]
            for char, offset in nested_pauli_data["ops"]:
                if offset >= len(qargs):
                    return None
                target_index = definition.find_bit(qargs[offset]).index
                if not apply_pauli(target_index, char):
                    return None
            continue

        name = getattr(op, "name", None)
        if len(qargs) != 1 or name not in {"x", "y", "z"}:
            return None
        target_index = definition.find_bit(qargs[0]).index
        if not apply_pauli(target_index, str(name).upper()):
            return None

    return {
        "ops": [(char, index) for index, char in enumerate(pauli_by_index) if char != "I"],
        "phase": phase_angle,
    }


def _single_qubit_pauli_data(
    operation,
    seen: set[int] | None = None,
) -> dict[str, object] | None:
    label = _pauli_label_from_operation(operation, int(operation.num_qubits))
    if label is not None:
        return {
            "ops": _pauli_ops_from_label(label),
            "phase": 0,
        }
    return _single_qubit_pauli_data_from_definition(operation, seen)


def _controlled_gate_single_qubit_pauli_data(gate: ControlledGate) -> dict[str, object] | None:
    target_count = int(gate.num_qubits) - int(gate.num_ctrl_qubits)
    base_operation = getattr(gate, "base_gate", None)
    if base_operation is None:
        return None
    pauli_data = _single_qubit_pauli_data(base_operation)
    if pauli_data is None:
        return None
    pauli_ops = pauli_data["ops"]
    if any(target_index >= target_count for _pauli, target_index in pauli_ops):
        return None
    return pauli_data


def _is_ancilla_target_controlled_gate(instruction, ancilla_qubits: set) -> bool:
    op = instruction.operation
    if not isinstance(op, ControlledGate):
        return False
    _controls, targets = _instruction_controls_targets(instruction)
    return len(targets) == 1 and targets[0] in ancilla_qubits


def _is_optimizable_ancilla_target_controlled_gate(instruction, ancilla_qubits: set) -> bool:
    op = instruction.operation
    if instruction.clbits:
        return False
    if not _is_ancilla_target_controlled_gate(instruction, ancilla_qubits):
        return False
    return _is_controlled_x_gate(op) and int(op.num_ctrl_qubits) <= 2


def _can_move_controlled_gate_right(moving_instruction, other_instruction) -> bool:
    moving_controls, moving_targets = _instruction_controls_targets(moving_instruction)
    other_controls, other_targets = _instruction_controls_targets(other_instruction)

    moving_control_set = set(moving_controls)
    moving_target_set = set(moving_targets)
    other_control_set = set(other_controls)
    other_target_set = set(other_targets)

    if moving_target_set & other_control_set:
        return False
    if moving_control_set & other_target_set:
        return False
    if moving_target_set & other_target_set:
        return _is_controlled_x_gate(moving_instruction.operation) and _is_controlled_x_gate(
            other_instruction.operation
        )
    return True


def _copy_circuit_with_instructions(qc: QuantumCircuit, instructions: list) -> QuantumCircuit:
    copied = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    copied.global_phase = qc.global_phase
    copied.metadata = None if qc.metadata is None else dict(qc.metadata)
    for instruction in instructions:
        copied.append(
            instruction.operation,
            list(instruction.qubits),
            list(instruction.clbits),
        )
    return copied


def _append_instruction(qc: QuantumCircuit, instruction) -> None:
    qc.append(
        instruction.operation,
        list(instruction.qubits),
        list(instruction.clbits),
    )


def _qubit_label(qc: QuantumCircuit, qubit) -> str:
    location = qc.find_bit(qubit)
    registers = getattr(location, "registers", ())
    if registers:
        reg, index = registers[0]
        return f"{reg.name}[{index}]"
    return f"q[{location.index}]"


def _control_literals(qc: QuantumCircuit, instruction) -> list[dict[str, object]]:
    controls, _targets = _instruction_controls_targets(instruction)
    ctrl_bits = _ctrl_state_bits(instruction.operation)[::-1]
    return [
        {
            "qubit": _qubit_label(qc, qubit),
            "positive": bit == "1",
        }
        for qubit, bit in zip(controls, ctrl_bits)
    ]


def _format_product_expression(literals: list[dict[str, object]]) -> str:
    if not literals:
        return "1"
    terms = [
        str(literal["qubit"]) if literal["positive"] else f"~{literal['qubit']}"
        for literal in literals
    ]
    if len(terms) == 1:
        return terms[0]
    return "(" + " & ".join(terms) + ")"


def _anf_terms_from_product_literals(
    product_literals: list[list[dict[str, object]]],
) -> set[frozenset[str]]:
    def multiply_by_positive(poly: set[frozenset[str]], variable: str) -> set[frozenset[str]]:
        return {frozenset(set(monomial) | {variable}) for monomial in poly}

    def multiply_by_negative(poly: set[frozenset[str]], variable: str) -> set[frozenset[str]]:
        positive_part = multiply_by_positive(poly, variable)
        return poly ^ positive_part

    def product_to_anf(literals: list[dict[str, object]]) -> set[frozenset[str]]:
        poly = {frozenset()}
        for literal in literals:
            variable = str(literal["qubit"])
            if literal["positive"]:
                poly = multiply_by_positive(poly, variable)
            else:
                poly = multiply_by_negative(poly, variable)
        return poly

    anf_terms: set[frozenset[str]] = set()
    for literals in product_literals:
        anf_terms ^= product_to_anf(literals)
    return anf_terms


def _monomial_sort_key(
    monomial: frozenset[str],
    control_order: dict[str, int],
) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    ordered = sorted(monomial, key=lambda label: (control_order.get(label, len(control_order)), label))
    return (len(ordered), tuple(control_order.get(label, len(control_order)) for label in ordered), tuple(ordered))


def _monomial_expr(monomial: frozenset[str], control_order: dict[str, int]):
    if len(monomial) == 0:
        return {"kind": "const", "value": 1}
    ordered = sorted(monomial, key=lambda label: (control_order.get(label, len(control_order)), label))
    if len(ordered) == 1:
        return {"kind": "var", "name": ordered[0]}
    return {"kind": "and", "terms": [{"kind": "var", "name": label} for label in ordered]}


def _xor_expr(terms: list[dict]) -> dict:
    flat_terms = []
    for term in terms:
        if term["kind"] == "const" and term["value"] == 0:
            continue
        if term["kind"] == "xor":
            flat_terms.extend(term["terms"])
        else:
            flat_terms.append(term)
    if not flat_terms:
        return {"kind": "const", "value": 0}
    if len(flat_terms) == 1:
        return flat_terms[0]
    return {"kind": "xor", "terms": flat_terms}


def _and_expr(terms: list[dict]) -> dict:
    flat_terms = []
    for term in terms:
        if term["kind"] == "const":
            if term["value"] == 0:
                return {"kind": "const", "value": 0}
            continue
        if term["kind"] == "and":
            flat_terms.extend(term["terms"])
        else:
            flat_terms.append(term)
    if not flat_terms:
        return {"kind": "const", "value": 1}
    if len(flat_terms) == 1:
        return flat_terms[0]
    return {"kind": "and", "terms": flat_terms}


def _expr_to_string(expr: dict) -> str:
    kind = expr["kind"]
    if kind == "const":
        return str(expr["value"])
    if kind == "var":
        return str(expr["name"])
    if kind == "xor":
        return " ^ ".join(_expr_to_string(term) for term in expr["terms"])
    if kind == "and":
        pieces = []
        for term in expr["terms"]:
            piece = _expr_to_string(term)
            if term["kind"] == "xor":
                piece = f"({piece})"
            pieces.append(piece)
        return " & ".join(pieces)
    raise ValueError(f"Unsupported expression kind: {kind}")


def _expr_cost(expr: dict) -> tuple[int, int]:
    kind = expr["kind"]
    if kind == "const":
        return (0 if expr["value"] == 0 else 1, 1)
    if kind == "var":
        return (1, len(str(expr["name"])))
    if kind == "xor":
        child_costs = [_expr_cost(term) for term in expr["terms"]]
        return (sum(cost for cost, _ in child_costs), len(_expr_to_string(expr)))
    if kind == "and":
        child_costs = [_expr_cost(term) for term in expr["terms"]]
        non_var_count = sum(1 for term in expr["terms"] if term["kind"] != "var")
        return (sum(cost for cost, _ in child_costs) + max(len(expr["terms"]) - 1, 1) + 2 * non_var_count, len(_expr_to_string(expr)))
    raise ValueError(f"Unsupported expression kind: {kind}")


def _candidate_common_factors(terms: set[frozenset[str]]) -> list[frozenset[str]]:
    from itertools import combinations

    factors = set()
    term_list = list(terms)
    for left, right in combinations(term_list, 2):
        common = sorted(set(left) & set(right))
        for size in range(1, len(common) + 1):
            for subset in combinations(common, size):
                factors.add(frozenset(subset))
    return sorted(factors, key=lambda factor: (-len(factor), tuple(sorted(factor))))


def _factor_anf_terms_to_expr(
    anf_terms: set[frozenset[str]],
    control_order: dict[str, int],
    memo: dict[frozenset[frozenset[str]], dict] | None = None,
) -> dict:
    if memo is None:
        memo = {}
    key = frozenset(anf_terms)
    if key in memo:
        return memo[key]

    if not anf_terms:
        result = {"kind": "const", "value": 0}
        memo[key] = result
        return result

    base_expr = _xor_expr(
        [
            _monomial_expr(monomial, control_order)
            for monomial in sorted(
                anf_terms,
                key=lambda monomial: _monomial_sort_key(monomial, control_order),
            )
        ]
    )
    best_expr = base_expr
    best_cost = _expr_cost(base_expr)

    for common in _candidate_common_factors(anf_terms):
        group = {term for term in anf_terms if common <= term}
        if len(group) < 2:
            continue
        residual_terms = {frozenset(set(term) - set(common)) for term in group}
        remaining_terms = anf_terms - group
        factored_group = _and_expr(
            [
                _factor_anf_terms_to_expr(residual_terms, control_order, memo),
                _monomial_expr(common, control_order),
            ]
        )
        candidate_expr = _xor_expr(
            [
                _factor_anf_terms_to_expr(remaining_terms, control_order, memo),
                factored_group,
            ]
        )
        candidate_cost = _expr_cost(candidate_expr)
        if candidate_cost < best_cost:
            best_expr = candidate_expr
            best_cost = candidate_cost

    memo[key] = best_expr
    return best_expr


def _gate_sequence_for_boolean_expr(expr: dict, target: str) -> list[str]:
    steps = []
    temp_index = 0

    def new_temp() -> str:
        nonlocal temp_index
        temp = f"tmp{temp_index}"
        temp_index += 1
        return temp

    def inverse_step(step: str) -> str:
        if step.startswith("ALLOC(") and step.endswith("=0)"):
            return f"FREE({step[len('ALLOC('):-len('=0)')]})"
        if step.startswith("FREE(") and step.endswith(")"):
            return f"ALLOC({step[len('FREE('):-1]}=0)"
        return step

    def inverse_steps(step_list: list[str]) -> list[str]:
        return [inverse_step(step) for step in reversed(step_list)]

    def linear_xor_terms(subexpr: dict) -> tuple[list[str], int] | None:
        if subexpr["kind"] == "var":
            return [str(subexpr["name"])], 0
        if subexpr["kind"] == "const":
            return [], int(subexpr["value"]) & 1
        if subexpr["kind"] != "xor":
            return None

        variables = []
        const_value = 0
        for term in subexpr["terms"]:
            term_linear = linear_xor_terms(term)
            if term_linear is None:
                return None
            term_variables, term_const = term_linear
            const_value ^= term_const
            for variable in term_variables:
                if variable in variables:
                    variables.remove(variable)
                else:
                    variables.append(variable)
        return variables, const_value

    def synthesize_linear_xor_control(subexpr: dict) -> tuple[str | None, list[str], list[str]] | None:
        linear_terms = linear_xor_terms(subexpr)
        if linear_terms is None:
            return None
        variables, const_value = linear_terms
        if len(variables) == 0:
            return (None, [], []) if const_value == 0 else ("1", [], [])

        accumulator = variables[-1]
        compute_steps = [f"CNOT({variable}, {accumulator})" for variable in variables[:-1]]
        if const_value:
            compute_steps.append(f"X({accumulator})")
        cleanup_steps = list(reversed(compute_steps))
        return accumulator, compute_steps, cleanup_steps

    def synthesize_toggle(subexpr: dict, out: str) -> list[str]:
        kind = subexpr["kind"]
        if kind == "const":
            return [] if subexpr["value"] == 0 else [f"X({out})"]
        if kind == "var":
            return [f"CNOT({subexpr['name']}, {out})"]
        if kind == "xor":
            result = []
            for term in subexpr["terms"]:
                result.extend(synthesize_toggle(term, out))
            return result
        if kind == "and":
            return synthesize_and(subexpr["terms"], out)
        raise ValueError(f"Unsupported expression kind: {kind}")

    def synthesize_and(factors: list[dict], out: str) -> list[str]:
        local_steps = []
        cleanup_steps = []
        controls = []
        for factor in factors:
            if factor["kind"] == "var":
                controls.append(str(factor["name"]))
            elif factor["kind"] == "const" and factor["value"] == 1:
                continue
            elif factor["kind"] == "const" and factor["value"] == 0:
                return []
            else:
                linear_control = synthesize_linear_xor_control(factor)
                if linear_control is not None:
                    control, compute_steps, cleanup = linear_control
                    if control is None:
                        return []
                    if control != "1":
                        local_steps.extend(compute_steps)
                        controls.append(control)
                        cleanup_steps.extend(cleanup)
                    continue

                temp = new_temp()
                compute_steps = synthesize_toggle(factor, temp)
                local_steps.append(f"ALLOC({temp}=0)")
                local_steps.extend(compute_steps)
                controls.append(temp)
                cleanup_steps.extend(inverse_steps(compute_steps))
                cleanup_steps.append(f"FREE({temp})")

        if len(controls) == 0:
            local_steps.append(f"X({out})")
        elif len(controls) == 1:
            local_steps.append(f"CNOT({controls[0]}, {out})")
        elif len(controls) == 2:
            local_steps.append(f"CCU({controls[0]}, {controls[1]}, {out}; U=X)")
        else:
            ladder = []
            ladder_cleanup = []
            prev = controls[0]
            for control in controls[1:-1]:
                temp = new_temp()
                local_steps.append(f"ALLOC({temp}=0)")
                local_steps.append(f"CCU({prev}, {control}, {temp}; U=X)")
                ladder.append(temp)
                ladder_cleanup.append(f"CCU({prev}, {control}, {temp}; U=X)")
                prev = temp
            local_steps.append(f"CCU({prev}, {controls[-1]}, {out}; U=X)")
            for temp, undo_step in zip(reversed(ladder), reversed(ladder_cleanup)):
                local_steps.append(undo_step)
                local_steps.append(f"FREE({temp})")

        local_steps.extend(cleanup_steps)
        return local_steps

    steps.extend(synthesize_toggle(expr, target))
    return steps


def _append_boolean_expr_toggle(
    qc: QuantumCircuit,
    expr: dict,
    target,
    label_to_qubit: dict[str, object],
) -> None:
    def get_qubit(label: str):
        if label not in label_to_qubit:
            raise ValueError(f"Unknown qubit label in boolean expression: {label}")
        return label_to_qubit[label]

    def linear_xor_terms(subexpr: dict) -> tuple[list[str], int] | None:
        if subexpr["kind"] == "var":
            return [str(subexpr["name"])], 0
        if subexpr["kind"] == "const":
            return [], int(subexpr["value"]) & 1
        if subexpr["kind"] != "xor":
            return None

        variables = []
        const_value = 0
        for term in subexpr["terms"]:
            term_linear = linear_xor_terms(term)
            if term_linear is None:
                return None
            term_variables, term_const = term_linear
            const_value ^= term_const
            for variable in term_variables:
                if variable in variables:
                    variables.remove(variable)
                else:
                    variables.append(variable)
        return variables, const_value

    def append_inverse_operations(operations: list[tuple[str, str, str | None]]) -> None:
        for name, left, right in reversed(operations):
            if name == "cx":
                qc.cx(get_qubit(left), get_qubit(str(right)))
            elif name == "x":
                qc.x(get_qubit(left))
            else:
                raise ValueError(f"Unsupported inverse operation: {name}")

    def synthesize_linear_xor_control(
        subexpr: dict,
        avoid_variables: set[str],
    ) -> tuple[str | None, list[tuple[str, str, str | None]]] | None:
        linear_terms = linear_xor_terms(subexpr)
        if linear_terms is None:
            return None
        variables, const_value = linear_terms
        if len(variables) == 0:
            return (None, []) if const_value == 0 else ("1", [])

        accumulator = next(
            (variable for variable in reversed(variables) if variable not in avoid_variables),
            None,
        )
        if accumulator is None:
            raise ValueError("Linear-XOR control overlaps all other controls; a clean temp would be required.")
        operations = []
        for variable in variables:
            if variable == accumulator:
                continue
            qc.cx(get_qubit(variable), get_qubit(accumulator))
            operations.append(("cx", variable, accumulator))
        if const_value:
            qc.x(get_qubit(accumulator))
            operations.append(("x", accumulator, None))
        return accumulator, operations

    def synthesize_toggle(subexpr: dict, out) -> None:
        kind = subexpr["kind"]
        if kind == "const":
            if subexpr["value"] == 1:
                qc.x(out)
            return
        if kind == "var":
            qc.cx(get_qubit(str(subexpr["name"])), out)
            return
        if kind == "xor":
            for term in subexpr["terms"]:
                synthesize_toggle(term, out)
            return
        if kind == "and":
            synthesize_and(subexpr["terms"], out)
            return
        raise ValueError(f"Unsupported expression kind: {kind}")

    def synthesize_and(factors: list[dict], out) -> None:
        linear_variables_by_factor = []
        for factor in factors:
            linear_terms = linear_xor_terms(factor)
            if linear_terms is None:
                linear_variables_by_factor.append(set())
            else:
                variables, _const_value = linear_terms
                linear_variables_by_factor.append(set(variables))

        controls = []
        cleanup_operations = []
        for factor_index, factor in enumerate(factors):
            if factor["kind"] == "var":
                controls.append(get_qubit(str(factor["name"])))
            elif factor["kind"] == "const" and factor["value"] == 1:
                continue
            elif factor["kind"] == "const" and factor["value"] == 0:
                return
            else:
                other_variables = set()
                for other_index, variables in enumerate(linear_variables_by_factor):
                    if other_index != factor_index:
                        other_variables.update(variables)
                linear_control = synthesize_linear_xor_control(factor, other_variables)
                if linear_control is None:
                    raise ValueError("Only linear-XOR factors can be used as synthesized controls.")
                control, operations = linear_control
                if control is None:
                    return
                if control != "1":
                    controls.append(get_qubit(control))
                    cleanup_operations.extend(operations)

        if len(controls) == 0:
            qc.x(out)
        elif len(controls) == 1:
            qc.cx(controls[0], out)
        elif len(controls) == 2:
            qc.ccx(controls[0], controls[1], out)
        else:
            raise ValueError("Synthesizing an AND with more than two controls requires extra clean ancillas.")

        append_inverse_operations(cleanup_operations)

    synthesize_toggle(expr, target)


def _simplify_xor_product_data(
    control_qubits: list[str],
    product_literals: list[list[dict[str, object]]],
    target: str,
) -> dict[str, object]:
    control_order = {label: idx for idx, label in enumerate(control_qubits)}
    anf_terms = _anf_terms_from_product_literals(product_literals)
    expr = _factor_anf_terms_to_expr(anf_terms, control_order)
    return {
        "anf_terms": [
            sorted(monomial, key=lambda label: (control_order.get(label, len(control_order)), label))
            for monomial in sorted(
                anf_terms,
                key=lambda monomial: _monomial_sort_key(monomial, control_order),
            )
        ],
        "expression_tree": expr,
        "simplified_expression": _expr_to_string(expr),
        "gate_sequence": _gate_sequence_for_boolean_expr(expr, target),
    }


def encode_control_block_boolean_expressions(
    qc: QuantumCircuit,
    ancilla_qubits: list | tuple | set,
) -> list[dict[str, object]]:
    """
    Analyze ancilla-target controlled blocks and encode their boolean forms.

    The analysis first permutes each contiguous ancilla-target controlled block
    in-memory so gates with the same target are adjacent. For each target group
    with at least two gates, it encodes the target's toggling condition as the
    XOR of each gate's control-product expression and records a simplified
    boolean expression. The input circuit is not modified.
    """
    ancilla_set = set(ancilla_qubits)
    if not ancilla_set:
        return []

    instructions = list(qc.data)
    blocks = []
    current_block = []
    for index, instruction in enumerate(instructions):
        if _is_ancilla_target_controlled_gate(instruction, ancilla_set):
            current_block.append((index, instruction))
        elif current_block:
            blocks.append(current_block)
            current_block = []
    if current_block:
        blocks.append(current_block)

    encoded_blocks = []
    for block_id, block in enumerate(blocks):
        target_order = []
        target_groups = {}
        for instruction_index, instruction in block:
            _controls, targets = _instruction_controls_targets(instruction)
            target = targets[0]
            if target not in target_groups:
                target_order.append(target)
                target_groups[target] = []
            target_groups[target].append((instruction_index, instruction))

        permuted_indices = [
            instruction_index
            for target in target_order
            for instruction_index, _instruction in target_groups[target]
        ]

        for target in target_order:
            group = target_groups[target]
            if len(group) < 2:
                continue

            product_literals = []
            products = []
            control_qubits = []
            for instruction_index, instruction in group:
                literals = _control_literals(qc, instruction)
                product_literals.append(literals)
                products.append(
                    {
                        "gate_index": instruction_index,
                        "expression": _format_product_expression(literals),
                        "literals": literals,
                    }
                )
                for literal in literals:
                    qubit = str(literal["qubit"])
                    if qubit not in control_qubits:
                        control_qubits.append(qubit)

            xor_expression = " ^ ".join(str(product["expression"]) for product in products)
            target_label = _qubit_label(qc, target)
            simplified_data = _simplify_xor_product_data(
                control_qubits,
                product_literals,
                target_label,
            )
            encoded_blocks.append(
                {
                    "block_id": block_id,
                    "block_gate_indices": [instruction_index for instruction_index, _ in block],
                    "permuted_gate_indices": permuted_indices,
                    "target": target_label,
                    "control_qubits": control_qubits,
                    "products": products,
                    "anf_terms": simplified_data["anf_terms"],
                    "xor_expression": xor_expression,
                    "simplified_expression": simplified_data["simplified_expression"],
                    "gate_sequence": simplified_data["gate_sequence"],
                }
            )

    return encoded_blocks


def swap_phase_controlled_ancilla_targets(
    qc: QuantumCircuit,
    ancilla_qubits: list | tuple | set,
) -> QuantumCircuit:
    """
    Move eligible ancilla-target controlled gates next to the next such gate.

    For a controlled gate whose single target is an ancilla, find the next
    controlled gate with a single ancilla target. If the two target ancillas are
    not used as controls by intervening gates, and the first gate can commute
    through the intervening instructions, move the first gate immediately before
    the second. This is the first swap phase before later cancellation passes.
    """
    ancilla_set = set(ancilla_qubits)
    if not ancilla_set:
        return qc

    instructions = list(qc.data)
    i = 0
    while i < len(instructions):
        first = instructions[i]
        if not _is_ancilla_target_controlled_gate(first, ancilla_set):
            i += 1
            continue

        j = i + 1
        while j < len(instructions) and not _is_ancilla_target_controlled_gate(
            instructions[j],
            ancilla_set,
        ):
            j += 1
        if j >= len(instructions):
            break
        if j == i + 1:
            i += 1
            continue

        _first_controls, first_targets = _instruction_controls_targets(first)
        _second_controls, second_targets = _instruction_controls_targets(instructions[j])
        protected_ancillas = set(first_targets) | set(second_targets)

        can_move = True
        for between in instructions[i + 1:j]:
            between_controls, _between_targets = _instruction_controls_targets(between)
            if protected_ancillas & set(between_controls):
                can_move = False
                break
            if not _can_move_controlled_gate_right(first, between):
                can_move = False
                break

        if can_move:
            moved = instructions.pop(i)
            instructions.insert(j - 1, moved)
            i = max(i - 1, 0)
        else:
            i += 1

    swapped = _copy_circuit_with_instructions(qc, instructions)
    swap_phase_controlled_ancilla_targets.last_boolean_expressions = (
        encode_control_block_boolean_expressions(swapped, ancilla_set)
    )
    return swapped


def _permute_control_block_by_target_safely(block: list[tuple[int, object]]) -> list[tuple[int, object]]:
    target_order = []
    target_rank = {}
    for _instruction_index, instruction in block:
        _controls, targets = _instruction_controls_targets(instruction)
        target = targets[0]
        if target not in target_rank:
            target_rank[target] = len(target_order)
            target_order.append(target)

    permuted = list(block)
    changed = True
    while changed:
        changed = False
        for index in range(len(permuted) - 1):
            _left_index, left = permuted[index]
            _right_index, right = permuted[index + 1]
            _left_controls, left_targets = _instruction_controls_targets(left)
            _right_controls, right_targets = _instruction_controls_targets(right)
            if target_rank[left_targets[0]] <= target_rank[right_targets[0]]:
                continue
            if _can_move_controlled_gate_right(left, right):
                permuted[index], permuted[index + 1] = permuted[index + 1], permuted[index]
                changed = True
    return permuted


def _append_simplified_control_group(
    output: QuantumCircuit,
    source: QuantumCircuit,
    group: list[tuple[int, object]],
    label_to_qubit: dict[str, object],
) -> dict[str, object]:
    _controls, targets = _instruction_controls_targets(group[0][1])
    target = targets[0]
    target_label = _qubit_label(source, target)
    product_literals = []
    products = []
    control_qubits = []
    for instruction_index, instruction in group:
        literals = _control_literals(source, instruction)
        product_literals.append(literals)
        products.append(
            {
                "gate_index": instruction_index,
                "expression": _format_product_expression(literals),
                "literals": literals,
            }
        )
        for literal in literals:
            qubit = str(literal["qubit"])
            if qubit not in control_qubits:
                control_qubits.append(qubit)

    control_order = {label: idx for idx, label in enumerate(control_qubits)}
    anf_terms = _anf_terms_from_product_literals(product_literals)
    expr = _factor_anf_terms_to_expr(anf_terms, control_order)
    _append_boolean_expr_toggle(output, expr, target, label_to_qubit)

    return {
        "target": target_label,
        "control_qubits": control_qubits,
        "products": products,
        "anf_terms": [
            sorted(monomial, key=lambda label: (control_order.get(label, len(control_order)), label))
            for monomial in sorted(
                anf_terms,
                key=lambda monomial: _monomial_sort_key(monomial, control_order),
            )
        ],
        "xor_expression": " ^ ".join(str(product["expression"]) for product in products),
        "simplified_expression": _expr_to_string(expr),
    }


def optimize_expanded_circuit(
    qc: QuantumCircuit,
    ancilla_qubits: list | tuple | set,
) -> QuantumCircuit:
    """
    Optimize expanded multi-control ladders using boolean simplification.

    The pass first runs the swap phase, then scans maximal consecutive blocks of
    ancilla-target controlled-X gates with at most two controls. Within each
    block it safely permutes gates toward target-grouped order, derives the ANF
    for each repeated target group, simplifies it, and appends the resulting
    CNOT/Toffoli implementation directly to the returned circuit.
    """
    ancilla_set = set(ancilla_qubits)
    if not ancilla_set:
        optimize_expanded_circuit.last_boolean_expressions = []
        return qc

    swapped = swap_phase_controlled_ancilla_targets(qc, ancilla_set)
    optimized = QuantumCircuit(*swapped.qregs, *swapped.cregs, name=swapped.name)
    optimized.global_phase = swapped.global_phase
    optimized.metadata = None if swapped.metadata is None else dict(swapped.metadata)

    label_to_qubit = {_qubit_label(swapped, qubit): qubit for qubit in swapped.qubits}
    instructions = list(swapped.data)
    encoded_blocks = []
    block_id = 0
    index = 0
    while index < len(instructions):
        instruction = instructions[index]
        if not _is_optimizable_ancilla_target_controlled_gate(instruction, ancilla_set):
            _append_instruction(optimized, instruction)
            index += 1
            continue

        block = []
        while index < len(instructions) and _is_optimizable_ancilla_target_controlled_gate(
            instructions[index],
            ancilla_set,
        ):
            block.append((index, instructions[index]))
            index += 1

        permuted_block = _permute_control_block_by_target_safely(block)
        permuted_indices = [instruction_index for instruction_index, _instruction in permuted_block]
        group_start = 0
        while group_start < len(permuted_block):
            _group_index, first_instruction = permuted_block[group_start]
            _first_controls, first_targets = _instruction_controls_targets(first_instruction)
            target = first_targets[0]
            group_end = group_start + 1
            while group_end < len(permuted_block):
                _next_controls, next_targets = _instruction_controls_targets(permuted_block[group_end][1])
                if next_targets[0] != target:
                    break
                group_end += 1

            group = permuted_block[group_start:group_end]
            if len(group) == 1:
                _append_instruction(optimized, group[0][1])
            else:
                entry = _append_simplified_control_group(optimized, swapped, group, label_to_qubit)
                entry.update(
                    {
                        "block_id": block_id,
                        "block_gate_indices": [instruction_index for instruction_index, _ in block],
                        "permuted_gate_indices": permuted_indices,
                    }
                )
                encoded_blocks.append(entry)
            group_start = group_end

        block_id += 1

    optimize_expanded_circuit.last_boolean_expressions = encoded_blocks
    optimize_expanded_circuit.swapped_circuit = swapped
    return optimized


def _append_full_control_pauli_expansion(
    expanded: QuantumCircuit,
    controls: list,
    targets: list,
    work: list,
    pauli_ops: list[tuple[str, int]],
    phase_angle=0,
) -> None:
    expanded.ccx(controls[0], controls[1], work[0])
    for ctrl_index in range(2, len(controls)):
        expanded.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])

    full_control = work[len(controls) - 2]
    if phase_angle != 0:
        expanded.p(phase_angle, full_control)
    for pauli, target_index in pauli_ops:
        target = targets[target_index]
        if pauli == "X":
            expanded.cx(full_control, target)
        elif pauli == "Y":
            expanded.cy(full_control, target)
        elif pauli == "Z":
            expanded.cz(full_control, target)
        else:
            raise ValueError(f"Unsupported Pauli component: {pauli}")

    for ctrl_index in range(len(controls) - 1, 1, -1):
        expanded.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])
    expanded.ccx(controls[0], controls[1], work[0])


def expand_large_controlled_gates(
    qc: QuantumCircuit,
    ancilla_name: str = "mc_anc",
    if_optimize = False,
) -> QuantumCircuit:
    """
    Expand every controlled gate with N >= 3 controls, plus two-control
    high-weight controlled Pauli-string gates.

    A gate C^N(U) is usually rewritten using N - 2 clean ancilla qubits as a
    Toffoli compute ladder, one two-controlled U gate, and the reverse Toffoli
    ladder. If U is a tensor Pauli with weight at least 3, the pass instead
    computes the full conjunction of all N controls into N - 1 clean ancillas,
    applies one single-control Pauli gate per non-identity target, then
    uncomputes the conjunction. The same full-control Pauli path is also used
    for C^2(P) when P is a tensor Pauli with weight at least 3; that case uses
    one clean work qubit for the temporary conjunction of the two controls.
    Open controls are handled by temporary X conjugations, matching Qiskit's
    ctrl_state convention where the rightmost control-state bit corresponds to
    the first control qubit in qargs.

    The returned circuit reuses the original circuit registers and appends one
    ancilla register sized for the largest expanded gate. Non-Pauli gates with
    fewer than three controls are copied unchanged.
    """
    max_ancillas = 0
    for instruction in qc.data:
        op = instruction.operation
        if isinstance(op, ControlledGate):
            ctrl_count = int(op.num_ctrl_qubits)
            pauli_data = _controlled_gate_single_qubit_pauli_data(op)
            use_full_control_pauli_expansion = (
                pauli_data is not None
                and len(pauli_data["ops"]) >= 3
            )
            if ctrl_count >= 3:
                if use_full_control_pauli_expansion:
                    max_ancillas = max(max_ancillas, ctrl_count - 1)
                else:
                    max_ancillas = max(max_ancillas, ctrl_count - 2)
            elif ctrl_count == 2 and use_full_control_pauli_expansion:
                max_ancillas = max(max_ancillas, 1)

    expanded = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    expanded.global_phase = qc.global_phase
    expanded.metadata = None if qc.metadata is None else dict(qc.metadata)

    ancillas = []
    if max_ancillas > 0:
        existing_names = {reg.name for reg in qc.qregs + qc.cregs}
        reg_name = ancilla_name
        suffix = 0
        while reg_name in existing_names:
            suffix += 1
            reg_name = f"{ancilla_name}_{suffix}"
        ancilla_reg = QuantumRegister(max_ancillas, reg_name)
        expanded.add_register(ancilla_reg)
        ancillas = list(ancilla_reg)

    for instruction in qc.data:
        op = instruction.operation
        qargs = list(instruction.qubits)
        clbits = list(instruction.clbits)

        if not isinstance(op, ControlledGate):
            expanded.append(op, qargs, clbits)
            continue

        if clbits:
            raise ValueError("Cannot expand classically-bit-attached controlled gates.")

        ctrl_count = int(op.num_ctrl_qubits)
        controls = qargs[:ctrl_count]
        targets = qargs[ctrl_count:]
        pauli_data = _controlled_gate_single_qubit_pauli_data(op)
        pauli_ops = None if pauli_data is None else pauli_data["ops"]
        use_full_control_pauli_expansion = pauli_ops is not None and len(pauli_ops) >= 3
        should_expand = (
            ctrl_count >= 3
            or (ctrl_count == 2 and use_full_control_pauli_expansion)
        )
        if not should_expand:
            expanded.append(op, qargs, clbits)
            continue

        work_size = (
            ctrl_count - 1
            if use_full_control_pauli_expansion
            else ctrl_count - 2
        )
        work = ancillas[:work_size]
        ctrl_bits = _ctrl_state_bits(op)

        for ctrl, bit in zip(controls, ctrl_bits[::-1]):
            if bit == "0":
                expanded.x(ctrl)

        if use_full_control_pauli_expansion:
            _append_full_control_pauli_expansion(
                expanded,
                controls,
                targets,
                work,
                pauli_ops,
                pauli_data["phase"],
            )
        else:
            expanded.ccx(controls[0], controls[1], work[0])
            for ctrl_index in range(2, ctrl_count - 1):
                expanded.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])

            cc_u = op.base_gate.control(num_ctrl_qubits=2, ctrl_state="11")
            expanded.append(cc_u, [work[-1], controls[-1]] + targets)

            for ctrl_index in range(ctrl_count - 2, 1, -1):
                expanded.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])
            expanded.ccx(controls[0], controls[1], work[0])

        for ctrl, bit in zip(controls, ctrl_bits[::-1]):
            if bit == "0":
                expanded.x(ctrl)
    if if_optimize:
        expanded = optimize_expanded_circuit(expanded, ancillas)
        expand_large_controlled_gates.last_boolean_expressions = getattr(
            optimize_expanded_circuit,
            "last_boolean_expressions",
            [],
        )
    else:
        expand_large_controlled_gates.last_boolean_expressions = (
            encode_control_block_boolean_expressions(expanded, ancillas)
        )
    return expanded


expand_larger_controlled_gates = expand_large_controlled_gates


class _PlacementBlockEncoding(_DirectBlockEncoding):
    """Placement-search SELECT with phase-aware, zero-address residuals."""

    def mulplex_U_opt_order(self):
        sys_size = self.sys_size
        coeff_mode_dict, ctrl_size = self.find_optimal_order_matrices()
        modes_with_phase = self.mobius_phase_result['g_modes_with_phase']
        ctrl_reg, sys_reg = QuantumRegister(ctrl_size, 'ctrl'), QuantumRegister(sys_size, 'sys')
        qc_u = QuantumCircuit(ctrl_reg, sys_reg)
        mccount = 0
        cxcount = 0
        self.coeff_list_ordered = np.zeros((1 << ctrl_size), dtype=complex)
        for ctrl_value, coeff in coeff_mode_dict.items():
            self.coeff_list_ordered[int(ctrl_value, 2)] = coeff

        for pauli_label, ctrl_value, phase in modes_with_phase:
            qc_pauli = QuantumCircuit(len(pauli_label))
            pauli_op = Pauli(pauli_label)
            qc_pauli.append(pauli_op, range(len(pauli_label)))
            qc_pauli.global_phase = np.pi * phase / 2
            qc_pauli = qc_pauli.decompose()
            ## Match qiskit's ctrl_state ordering in mulplex_U:
            ## rightmost bit in ctrl_value corresponds to lower-index control qubit.
            active_ctrls = [i for i, bit in enumerate(ctrl_value[::-1]) if bit == '1']
            pauli_weight = sum(1 for ch in pauli_label if ch != 'I')
            if len(active_ctrls) == 0:
                qc_u.append(qc_pauli.to_gate(), list(range(ctrl_size, ctrl_size + sys_size)))
            else:
                ctrl_U_elem = qc_pauli.to_gate().control(
                    num_ctrl_qubits=len(active_ctrls),
                    ctrl_state='1' * len(active_ctrls),
                )
                qargs = active_ctrls + list(range(ctrl_size, ctrl_size + sys_size))
                qc_u.append(ctrl_U_elem, qargs)
                if len(active_ctrls) == 1:
                    cxcount += pauli_weight
                else:
                    mccount += (len(active_ctrls) - 1) * pauli_weight

        tcount = mccount * 4
        cxcount += mccount * 4
        return qc_u, tcount, mccount, cxcount, ctrl_size

    def circuit(self, opt=None):
        if opt != 'Matrix-order':
            return super().circuit(opt=opt)

        self.mccount = 0
        self.cxcount = 0
        qc_u, tcount, mccount, cxcount, ctrl_size = self.mulplex_U_opt_order()
        options = self.matrix_order_options
        candidates = [{"name": "placement-search", "modes": self.additional_modes,
                       "mobius": self.mobius_phase_result,
                       "coefficients": self.coeff_list_ordered.copy(), "symbolic": qc_u,
                       "phase_lookup": self.vec_phase_lookup}]
        if options.placement_candidates:
            labels = [atom.expr for atom, _ in self.J.instances]
            # Include the identity in the lookup: it need not remain at address
            # zero in these candidates. Phases travel with labels, as do weights.
            phase_lookup = {_pauli_label_to_symplectic_vector(atom.expr).tobytes():
                            2 * np.angle(atom.phase) / np.pi for atom, _ in self.J.instances}
            coeff_by_label = dict(zip(labels, self.coeff_list))
            seen = {tuple(sorted(self.additional_modes))}
            for name, modes in extra_placements(labels, ctrl_size, self.sys_size):
                key = tuple(sorted(modes))
                if key in seen:
                    continue
                seen.add(key)
                mobius = mobius_invert_modes_with_phases(ctrl_size, modes, phase_lookup, 0)
                coefficients = np.zeros(1 << ctrl_size, dtype=complex)
                for label, address in modes:
                    coefficients[int(address, 2)] = coeff_by_label[label]
                candidates.append({"name": name, "modes": modes, "mobius": mobius,
                                   "coefficients": coefficients, "phase_lookup": phase_lookup,
                                   "symbolic": symbolic_select(mobius['g_modes_with_phase'],
                                                               ctrl_size, self.sys_size)})

        for candidate in candidates:
            candidate["proxy_cost"] = sum(address.count('1') * sum(p != 'I' for p in label)
                                          for label, address, _ in candidate["mobius"]['g_modes_with_phase'])
        if not options.actual_nam_cost:
            # With the cost switch off, use the historical residual-weight
            # objective. Other switches remain independently usable.
            candidates = [min(candidates, key=lambda item: item["proxy_cost"])]
        for candidate in candidates:
            prepare, prepare_info = prepare_coefficients(candidate["coefficients"], options)
            select, lowering_info = lower_select(candidate["symbolic"], options)
            circuit = QuantumCircuit(*select.qregs, name='BlockEncoding')
            circuit.compose(prepare, qubits=range(ctrl_size), inplace=True)
            circuit.compose(select, inplace=True)
            circuit.compose(prepare.inverse(), qubits=range(ctrl_size), inplace=True)
            candidate.update(prepare=prepare, select=select, circuit=circuit)
            candidate["nam_cost"] = nam_cost(circuit) if options.actual_nam_cost else None
            circuit.metadata = {"matrix_order": {
                "options": options_dict(options), "selected_candidate": candidate["name"],
                "placement": list(candidate["modes"]), "proxy_cost": candidate["proxy_cost"],
                "prepare": prepare_info, "select_lowering": lowering_info,
                "cost_context": "standalone", "nam_gates": candidate["nam_cost"],
            }}
        best = min(candidates, key=lambda item: item["nam_cost"] if options.actual_nam_cost else item["proxy_cost"])
        comparison = [{"candidate": item["name"], "nam_gates": item["nam_cost"],
                       "proxy_cost": item["proxy_cost"]} for item in candidates]
        for candidate in candidates:
            candidate["circuit"].metadata["matrix_order"]["candidates"] = comparison
        self._matrix_order_candidates = candidates
        self._symbolic_select = best["symbolic"].copy()
        self.coeff_list_ordered = best["coefficients"].copy()
        self.additional_modes = best["modes"]
        self.coeff_mode_dict = {address: best["coefficients"][int(address, 2)]
                                for _, address in best["modes"]}
        self.mobius_phase_result = best["mobius"]
        self.vec_phase_lookup = best["phase_lookup"]
        self.basis_modes_with_phase, self.nonbasis_modes_with_phase = extract_basis_modes_with_phases(
            self.additional_modes, self.vec_phase_lookup,
            next((2 * np.angle(atom.phase) / np.pi for atom, _ in self.J.instances
                  if atom.expr == 'I' * self.sys_size), 0))
        self.selected_crit = best["name"]
        # These remain legacy resource proxies; final counts come from NAM
        # lowering. Keep the proxies aligned with the selected residuals.
        selected_residuals = best["mobius"]['g_modes_with_phase']
        self.mccount = sum(max(address.count('1') - 1, 0) * sum(p != 'I' for p in label)
                           for label, address, _ in selected_residuals)
        self.tcount = 4 * self.mccount
        self.cxcount = self.tcount + sum(sum(p != 'I' for p in label)
                                        for label, address, _ in selected_residuals
                                        if address.count('1') == 1)
        self.succ_prob = np.sum(self.coeff_list)
        self.circuit_width = best["circuit"].num_qubits
        self._lcu_parts = (best["prepare"], best["select"], tuple(range(ctrl_size)))
        return best["circuit"]

    def find_optimal_order_matrices(self):
        pauli_list = []
        phase_list = []
        zero_phase = 0.0
        for i, ms in enumerate(self.mat_list):
            if isinstance(ms, tuple) or isinstance(ms, list):
                pauli_op, coeff = ms
                if pauli_op != 'I' * len(pauli_op):  # type: ignore
                    pauli_list.append(pauli_op)  # type: ignore
                    phase_list.append(coeff)
                else:
                    zero_phase = 2 * np.angle(coeff) / np.pi
        pauli_list = PauliList(pauli_list)
        x_part, z_part = pauli_list.x, pauli_list.z
        symplectic_matrix = np.hstack((x_part, z_part)).astype(int)
        phase_array = np.array([2 * np.angle(phase) / np.pi for phase in phase_list])

        w = int(np.ceil(np.log2(len(self.coeff_list))))

        identity_label = None
        for ms in self.mat_list:
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            if pauli_op == 'I' * len(pauli_op):
                identity_label = pauli_op
                break

        def _pauli_weight(label):
            return sum(1 for c in label if c != 'I')


        real_pauli_count = len(pauli_list)
        if real_pauli_count <= 256:
            search_profile = {
                "beam_width": 16,
                "layer_candidate_limit": 16,
                "max_exact_layer_assignments": 16,
                "mixed_initial_beam_width": 128,
                "mixed_low_weight_count": 128,
                "mixed_random_count": 256,
                "future_exact_address_limit": 8,
            }
        else:
            search_profile = {
                "beam_width": 16,
                "layer_candidate_limit": 16,
                "max_exact_layer_assignments": 16,
                "mixed_initial_beam_width": 32,
                "mixed_low_weight_count": 32,
                "mixed_random_count": 64,
                "future_exact_address_limit": 8,
            }
        explicit_options = getattr(self, "placement_options", None)
        for key in ("beam_width", "layer_candidate_limit", "max_exact_layer_assignments",
                    "mixed_initial_beam_width"):
            env_name = f"CHANNEL_LCU_PLACEMENT_{key.upper()}"
            if explicit_options is None and env_name in os.environ:
                value = int(os.environ[env_name])
                if value < 1:
                    raise ValueError(f"{env_name} must be a positive integer")
                search_profile[key] = value
        # The public experiment backend resolves every search default explicitly.
        # The public constructor resolves legacy environment overrides before search.
        import inspect
        search_options = {
            key: parameter.default
            for key, parameter in inspect.signature(search_for_pauli_placement).parameters.items()
            if parameter.default is not inspect.Parameter.empty
            and key not in {"m", "return_debug", "padding_to_full_address", "allow_zero_address"}
        }
        search_options.update({key: value for key, value in search_profile.items()
                               if key != "mixed_initial_beam_width"})
        search_options.update(
            initial_strategy="mixed", mixed_initial_beam_width=None,
            mixed_full_initial_beam_width=search_profile["mixed_initial_beam_width"],
            mixed_nonfull_initial_beam_width=search_profile["mixed_initial_beam_width"],
            swap_refinement=True, swap_refinement_max_terms=16,
        )
        if explicit_options is not None:
            unknown = set(explicit_options) - set(search_options) - {
                "rank_width_heuristic", "force_original_ctrl_width"}
            if unknown:
                raise ValueError(f"Unsupported placement options: {sorted(unknown)}")
            search_options.update({key: value for key, value in explicit_options.items()
                                   if key in search_options})
        placement_structure = analyze_placement_structure(self.J)

        def _axis_analysis_to_debug(axis_info):
            return {
                "axis": axis_info.axis,
                "term_count": axis_info.term_count,
                "unique": axis_info.unique,
                "rank": axis_info.rank,
                "generator_labels": list(axis_info.generator_labels),
                "shortcut_available": axis_info.shortcut_available,
                "rank_width": axis_info.rank_width,
                "viable_width": axis_info.viable_width,
                "fixed_cost": axis_info.fixed_cost,
            }

        placement_structure_debug = {
            "num_qubits": placement_structure.num_qubits,
            "term_count": placement_structure.term_count,
            "identity_index": placement_structure.identity_index,
            "min_width": placement_structure.min_width,
            "subgroup_rank": placement_structure.subgroup_rank,
            "subgroup_viable_width": placement_structure.subgroup_viable_width,
            "axis": {
                name: _axis_analysis_to_debug(axis_info)
                for name, axis_info in placement_structure.axis.items()
            },
            "suggested_widths": list(placement_structure.suggested_widths),
        }
        rank_width_heuristic_enabled = (
            os.environ.get("CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC") == "1"
        ) if explicit_options is None else bool(explicit_options.get("rank_width_heuristic", False))
        if rank_width_heuristic_enabled:
            structural_widths = {
                axis_info.viable_width
                for axis_info in placement_structure.axis.values()
                if (
                    axis_info.shortcut_available
                    and axis_info.viable_width is not None
                    and axis_info.viable_width >= w
                )
            }
            if (
                placement_structure.subgroup_viable_width is not None
                and placement_structure.subgroup_viable_width >= w
            ):
                structural_widths.add(placement_structure.subgroup_viable_width)
            candidate_widths = tuple(
                sorted(width for width in structural_widths if width is not None)
            )
            if len(candidate_widths) == 0:
                candidate_widths = tuple(
                    width
                    for width in placement_structure.suggested_widths
                    if width >= w
                )
            if len(candidate_widths) == 0:
                candidate_widths = (w,)
        else:
            candidate_widths = (w,)
        force_original_width = (os.environ.get("CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH") == "1"
                                if explicit_options is None else
                                bool(explicit_options.get("force_original_ctrl_width", False)))
        if force_original_width:
            candidate_widths = (w,)
        placement_structure_debug["rank_width_heuristic_enabled"] = (
            rank_width_heuristic_enabled
        )
        self.resolved_placement_options = {
            **search_options, "candidate_widths": list(candidate_widths),
            "rank_width_heuristic": rank_width_heuristic_enabled,
            "force_original_ctrl_width": force_original_width,
        }

        def _run_width_candidate(candidate_w):
            allow_zero_address = (
                identity_label is None and len(pauli_list) > (2 ** candidate_w - 1)
            )
            padding_to_full_address = (
                not allow_zero_address
                and identity_label is None
                and len(pauli_list) < (2 ** candidate_w - 1)
            )
            candidate_modes, candidate_debug = search_for_pauli_placement(
                self.J,
                m=candidate_w,
                **search_options,
                padding_to_full_address=padding_to_full_address,
                allow_zero_address=allow_zero_address,
                return_debug=True,
            )
            candidate_debug["search_profile"] = dict(search_options)
            candidate_debug["candidate_width"] = candidate_w
            candidate_debug["candidate_widths"] = list(candidate_widths)
            candidate_debug["placement_structure"] = placement_structure_debug
            if candidate_debug.get("padding_count", 0) > 0:
                candidate_modes = candidate_debug["padded_placement"]
            candidate_modes = sorted(candidate_modes, key=lambda x: int(x[1], 2))

            candidate_vec_phase_lookup = build_vec_phase_lookup(
                symplectic_matrix, phase_array
            )
            for padding_label in candidate_debug.get("padding_labels", []):
                padding_vec = _pauli_label_to_symplectic_vector(padding_label)
                candidate_vec_phase_lookup.setdefault(padding_vec.tobytes(), 0.0)
            candidate_mobius = mobius_invert_modes_with_phases(
                candidate_w,
                candidate_modes,
                candidate_vec_phase_lookup,
                zero_phase,
            )
            g_modes = candidate_mobius['g_modes_with_phase']
            final_cost = sum(
                ctrl.count('1') * _pauli_weight(label)
                for label, ctrl, _ in g_modes
            )
            return {
                "w": candidate_w,
                "additional_modes": candidate_modes,
                "placement_debug": candidate_debug,
                "vec_phase_lookup": candidate_vec_phase_lookup,
                "mobius_result": candidate_mobius,
                "final_cost": final_cost,
            }

        width_candidate_results = [
            _run_width_candidate(candidate_w)
            for candidate_w in candidate_widths
        ]
        best_width_candidate = min(
            width_candidate_results,
            key=lambda result: (
                result["final_cost"],
                result["placement_debug"].get("best_cost", float("inf")),
                result["w"],
            ),
        )

        w = best_width_candidate["w"]
        additional_modes = best_width_candidate["additional_modes"]
        placement_debug = best_width_candidate["placement_debug"]
        vec_phase_lookup = best_width_candidate["vec_phase_lookup"]
        mobius_result = best_width_candidate["mobius_result"]
        final_cost = best_width_candidate["final_cost"]

        self.additional_modes = additional_modes
        self.candidate_costs = [
            (
                'placement-search',
                placement_debug.get('initial_strategy'),
                placement_debug.get('pre_refinement_cost'),
                placement_debug.get('best_cost'),
                placement_debug.get('padded_search_cost'),
                final_cost,
                placement_debug.get('padding_count'),
                search_profile,
            )
        ]
        self.width_candidate_costs = [
            {
                "w": result["w"],
                "pre_refinement_cost": result["placement_debug"].get("pre_refinement_cost"),
                "best_cost": result["placement_debug"].get("best_cost"),
                "padded_search_cost": result["placement_debug"].get("padded_search_cost"),
                "final_cost": result["final_cost"],
                "padding_count": result["placement_debug"].get("padding_count"),
                "shortcut": result["placement_debug"].get("shortcut"),
                "shortcut_axis": result["placement_debug"].get("shortcut_axis"),
                "shortcut_rank": result["placement_debug"].get("shortcut_rank"),
                "shortcut_terms": result["placement_debug"].get("shortcut_terms"),
            }
            for result in width_candidate_results
        ]
        self.placement_width_analysis = placement_structure_debug
        self.selected_k = w
        self.selected_crit = 'mixed-placement-search'
        self.placement_search_debug = placement_debug

        ## Build coefficient list aligned with additional_modes order.
        coeff_pool_by_label = {}
        for idx, ms in enumerate(self.mat_list):
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            coeff = self.coeff_list[idx]
            coeff_pool_by_label.setdefault(pauli_op, []).append(coeff)

        coeff_mode_dict = {}
        for pauli_label, ctrl_value in additional_modes:
            coeff_pool = coeff_pool_by_label.get(pauli_label, [])
            if len(coeff_pool) > 0:
                coeff_mode = coeff_pool.pop(0)
            else:
                coeff_mode = 0.0 + 0.0j
            coeff_mode_dict[ctrl_value] = coeff_mode

        self.coeff_mode_dict = coeff_mode_dict
        self.vec_phase_lookup = vec_phase_lookup
        self.basis_modes_with_phase, self.nonbasis_modes_with_phase = extract_basis_modes_with_phases(
            additional_modes, self.vec_phase_lookup, zero_phase
        )
        self.mobius_phase_result = mobius_result

        return coeff_mode_dict, w




BACKEND_ID = "matrix-order-anf-pipeline-v2"
BLOCK_OPTIONS = ("No", "Ctrl-line", "Tree", "Matrix-order")


@dataclass(frozen=True)
class BlockEncodingComponents:
    """B = P† S P, with P and S in the same workspace/system layout as B."""

    circuit: QuantumCircuit
    prepare: QuantumCircuit
    select: QuantumCircuit
    symbolic_select: QuantumCircuit | None = None
    expanded_controlled_select: QuantumCircuit | None = None
    outer_controls: int = 0
    matrix_order_options: object | None = None
    placement_candidates: tuple = ()

    def system_last(self, system_size):
        return replace(self, **{
            field: system_last(getattr(self, field), system_size)
            for field in ("circuit", "prepare", "select")
        })

    @property
    def symbolic_address_qubits(self):
        if self.symbolic_select is None:
            return None
        return self.symbolic_select.num_qubits - len(self.circuit.metadata["system_qubits"])

    def with_outer_control(self, num_ctrl_qubits=1, *, shared_address_qubits=None):
        """Control symbolic residuals, expand, and reserve disjoint clean scratch.

        Every channel branch must use the same address region. A preceding
        block can leave that region nonzero; expansion scratch cannot overlap
        it even when a later block needs fewer address wires. Scratch starts
        in zero and is uncomputed by the expansion pass.
        """
        if self.symbolic_select is None:
            return self
        if num_ctrl_qubits < 1:
            raise ValueError("A controlled block requires at least one control.")
        own_address = self.symbolic_address_qubits
        old_address = self.circuit.metadata.get("shared_address_qubits", own_address)
        shared_address = old_address if shared_address_qubits is None else shared_address_qubits
        if shared_address < old_address:
            raise ValueError("Shared address space is too short for this block.")
        if self.expanded_controlled_select is not None and self.outer_controls == num_ctrl_qubits and shared_address == old_address:
            return self
        if self.placement_candidates:
            # Select in the actual F* context. All candidates share the same
            # address width; scratch is allocated only after selecting a winner.
            variants = [parts.with_outer_control(num_ctrl_qubits,
                         shared_address_qubits=shared_address) for parts in self.placement_candidates]
            costs = [nam_cost(parts.controlled(num_ctrl_qubits)) for parts in variants]
            best_index = min(range(len(variants)), key=costs.__getitem__)
            winner = variants[best_index]
            comparison = [{"candidate": parts.circuit.metadata["matrix_order"]["selected_candidate"],
                           "nam_gates": cost} for parts, cost in zip(variants, costs)]
            updated = {}
            for field in ("circuit", "prepare", "select"):
                part = getattr(winner, field).copy()
                part.metadata = deepcopy(part.metadata)
                part.metadata["matrix_order"].update(
                    candidates=comparison, cost_context=f"Fstar-{num_ctrl_qubits}-controls",
                    nam_gates=costs[best_index])
                updated[field] = part
            return replace(winner, **updated, placement_candidates=self.placement_candidates)
        system_size = len(self.circuit.metadata["system_qubits"])
        control = QuantumRegister(num_ctrl_qubits, "outer")
        address = QuantumRegister(shared_address, "address")
        system = QuantumRegister(system_size, "sys")
        symbolic = QuantumCircuit(control, address, system)
        # symbolic_select has [own address, system], with no expansion scratch.
        targets = list(address[:own_address]) + list(system)
        phase = float(self.symbolic_select.global_phase)
        if phase:
            if num_ctrl_qubits == 1:
                symbolic.p(phase, control[0])
            else:
                symbolic.mcp(phase, list(control[:-1]), control[-1])
        max_controls = num_ctrl_qubits
        for instruction in self.symbolic_select.data:
            if instruction.clbits:
                raise ValueError("A symbolic SELECT must be unitary.")
            op = instruction.operation
            qargs = [targets[self.symbolic_select.find_bit(q).index] for q in instruction.qubits]
            if isinstance(op, ControlledGate):
                # Preserve the Pauli base gate instead of controlling its
                # decomposed definition. New controls are the low state bits.
                count = num_ctrl_qubits + op.num_ctrl_qubits
                state = (int(op.ctrl_state) << num_ctrl_qubits) | ((1 << num_ctrl_qubits) - 1)
                gate = op.base_gate.control(count, ctrl_state=state)
            else:
                count = num_ctrl_qubits
                gate = op.control(count)
            symbolic.append(gate, list(control) + qargs)
            max_controls = max(max_controls, count)
        if self.matrix_order_options is None:
            expanded = expand_large_controlled_gates(symbolic, if_optimize=True)
            lowering_info = {"simplified_control_groups": len(expand_large_controlled_gates.last_boolean_expressions)}
        else:
            expanded, lowering_info = lower_select(symbolic, self.matrix_order_options,
                                                   outer_controls=num_ctrl_qubits)
        old_system = list(self.circuit.metadata["system_qubits"])
        old_work = [q for q in range(self.circuit.num_qubits) if q not in old_system]
        old_scratch = len(old_work) - old_address
        added_scratch = expanded.num_qubits - symbolic.num_qubits
        scratch = max(old_scratch, added_scratch)
        workspace = shared_address + scratch
        width = workspace + system_size
        metadata = {
            **self.circuit.metadata,
            "system_qubits": list(range(workspace, width)),
            "workspace_qubits": list(range(workspace)),
            "address_qubits": own_address,
            "shared_address_qubits": shared_address,
            "scratch_qubits": scratch,
            "scratch_input": "zero; disjoint from every block's address region",
            "fstar_control_lowering": "symbolic-residual-control-before-expansion",
            "controlled_select_expansion": {
                "pass": "expand_large_controlled_gates", "if_optimize": True,
                "outer_controls": num_ctrl_qubits, "max_residual_controls": max_controls,
                "added_scratch_qubits": added_scratch,
                **lowering_info,
            },
        }
        if "matrix_order" in metadata:
            metadata["matrix_order"] = {**metadata["matrix_order"],
                                        "controlled_select_lowering": lowering_info,
                                        "cost_context": f"Fstar-{num_ctrl_qubits}-controls",
                                        "nam_gates": None}

        def pack(part):
            destination = (list(range(old_address))
                           + list(range(shared_address, shared_address + old_scratch))
                           + list(range(workspace, width)))
            mapping = dict(zip(old_work + old_system, destination))
            output = QuantumCircuit(width, name=part.name)
            output.compose(part, qubits=[mapping[q] for q in range(part.num_qubits)], inplace=True)
            output.metadata = deepcopy(metadata)
            return output

        controlled = QuantumCircuit(num_ctrl_qubits + width, name="ControlledResidualSelect")
        # expanded layout: [outer, shared address, system, new scratch].
        mapping = (list(range(num_ctrl_qubits + shared_address))
                   + list(range(num_ctrl_qubits + workspace, num_ctrl_qubits + width))
                   + list(range(num_ctrl_qubits + shared_address,
                                num_ctrl_qubits + shared_address + added_scratch)))
        controlled.compose(expanded, qubits=mapping, inplace=True)
        return replace(self, circuit=pack(self.circuit), prepare=pack(self.prepare), select=pack(self.select),
                       expanded_controlled_select=controlled, outer_controls=num_ctrl_qubits)

    def controlled(self, num_ctrl_qubits=1, ctrl_state=None):
        """Implement C(B) as P, C(S), P†; retain SELECT's global phase.

        The inactive branch is P†P = I on the declared scratch-zero domain.
        Address and system inputs may be arbitrary.
        PREPARE uses coefficient magnitudes; coefficient phases belong to S.
        Control wires precede the shared block workspace and system wires.
        """
        if num_ctrl_qubits < 1:
            raise ValueError("A controlled block requires at least one control.")
        if self.symbolic_select is not None and self.outer_controls != num_ctrl_qubits:
            return self.with_outer_control(num_ctrl_qubits).controlled(num_ctrl_qubits, ctrl_state)
        qc = QuantumCircuit(num_ctrl_qubits + self.circuit.num_qubits, name="FstarBlock")
        targets = list(range(num_ctrl_qubits, qc.num_qubits))
        qc.compose(self.prepare, qubits=targets, inplace=True)
        if self.expanded_controlled_select is None:
            qc.append(self.select.to_gate().control(num_ctrl_qubits, ctrl_state=ctrl_state),
                      list(range(qc.num_qubits)))
        else:
            state = ((1 << num_ctrl_qubits) - 1 if ctrl_state is None else
                     int(ctrl_state, 2) if isinstance(ctrl_state, str) else int(ctrl_state))
            if not 0 <= state < (1 << num_ctrl_qubits):
                raise ValueError("Control state does not fit the outer control width.")
            open_controls = [q for q in range(num_ctrl_qubits) if not (state >> q) & 1]
            for q in open_controls:
                qc.x(q)
            qc.compose(self.expanded_controlled_select, inplace=True)
            for q in open_controls:
                qc.x(q)
        qc.compose(self.prepare.inverse(), qubits=targets, inplace=True)
        return qc


class BlockEncoding(_PlacementBlockEncoding):
    """Corrected backend with explicit options and circuit provenance.

    matrix_order_options accepts MatrixOrderOptions or a dictionary of its
    fields. All five passes default on, with at most three total ANF rounds.
    Disable the five booleans to reproduce the historical single-pass route.
    placement_options independently configures placement search; its default
    swap_refinement_max_terms stays 16.
    """

    def __init__(self, matrix_sum, *, placement_options=None, matrix_order_options=None):
        if not matrix_sum.instances or matrix_sum.pauli_norm() <= 0:
            raise ValueError("A block encoding requires a nonzero operator.")
        super().__init__(deepcopy(matrix_sum))
        self.matrix_order_options = resolve_options(matrix_order_options)
        self.placement_options = dict(placement_options or {})
        if placement_options is None:
            # Preserve intentional legacy overrides, then record their resolved
            # values. New callers can pass {} to use environment-free defaults.
            for key in ("beam_width", "layer_candidate_limit", "max_exact_layer_assignments",
                        "mixed_initial_beam_width"):
                variable = f"CHANNEL_LCU_PLACEMENT_{key.upper()}"
                if variable in os.environ:
                    value = int(os.environ[variable])
                    if value < 1:
                        raise ValueError(f"{variable} must be positive")
                    self.placement_options[key] = value
            for key, variable in (("rank_width_heuristic", "CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC"),
                                  ("force_original_ctrl_width", "CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH")):
                if variable in os.environ:
                    self.placement_options[key] = os.environ[variable] == "1"

    @staticmethod
    def _option(opt):
        opt = "No" if opt is None else opt
        if opt not in BLOCK_OPTIONS:
            raise ValueError(f"Unsupported block option {opt!r}; use {BLOCK_OPTIONS}.")
        return opt

    def _single_term(self):
        system = QuantumRegister(self.sys_size, "sys")
        circuit = QuantumCircuit(system)
        atom, _ = self.J.instances[0]
        if isinstance(atom, PauliAtom):
            circuit.pauli(atom.expr, system)
            circuit.global_phase = float(np.angle(atom.phase))
        else:
            circuit.unitary(atom.to_operator(), system)
        self.tcount = self.mccount = self.cxcount = 0
        self.succ_prob = self.J.pauli_norm()
        self.circuit_width = circuit.num_qubits
        self._lcu_parts = (QuantumCircuit(0), circuit.copy(), ())
        self._symbolic_select = circuit.copy()
        return circuit

    def _record(self, circuit, opt):
        system = [circuit.find_bit(q).index for reg in circuit.qregs
                  if reg.name == "sys" for q in reg]
        if len(system) != self.sys_size:
            system = list(range(circuit.num_qubits - self.sys_size, circuit.num_qubits))
        circuit.metadata = {
            **(circuit.metadata or {}),
            "block_backend": BACKEND_ID,
            "block_opt": opt,
            "single_term": len(self.J.instances) == 1,
            "alpha": float(self.J.pauli_norm()),
            "system_qubits": system,
            "workspace_qubits": [i for i in range(circuit.num_qubits) if i not in system],
            "placement": deepcopy(getattr(self, "resolved_placement_options", None)),
            "selected_crit": getattr(self, "selected_crit", None),
            "selected_k": getattr(self, "selected_k", None),
            "matrix_order_options": options_dict(self.matrix_order_options) if opt == "Matrix-order" else None,
        }
        return circuit

    def circuit(self, opt="No", gateable=True):
        opt = self._option(opt)
        if not gateable:
            raise ValueError("Experiment block encodings must be unitary; gateable=False is unsupported.")
        circuit = self._single_term() if len(self.J.instances) == 1 else super().circuit(opt=opt)
        return self._record(circuit, opt)

    def circuit_components(self, opt="No"):
        """Build once and expose the exact PREPARE/SELECT used by circuit().

        Calling circuit() also preserves instrumentation in backend subclasses.
        All three circuits include idle wires and the same explicit system map.
        """
        circuit = self.circuit(opt=opt)
        if opt == "Matrix-order" and len(self.J.instances) > 1:
            candidates = []
            for item in self._matrix_order_candidates:
                block = item["circuit"].copy()
                block = self._record(block, opt)
                block.metadata["selected_crit"] = item["name"]
                prepare = QuantumCircuit(*block.qregs)
                prepare.compose(item["prepare"], qubits=range(item["prepare"].num_qubits), inplace=True)
                select = item["select"].copy()
                for part in (prepare, select):
                    part.metadata = deepcopy(block.metadata)
                candidates.append(BlockEncodingComponents(
                    block, prepare, select, symbolic_select=item["symbolic"].copy(),
                    matrix_order_options=self.matrix_order_options))
            best = next(parts for parts in candidates
                        if parts.circuit.metadata["selected_crit"] == self.selected_crit)
            # No need to retain alternatives when the placement cost switch is
            # disabled. Repeated controlled() calls reuse the lowered winner.
            return replace(best, placement_candidates=tuple(candidates)
                           if self.matrix_order_options.actual_nam_cost else ())
        prepare, select, prepare_qubits = self._lcu_parts
        padded_prepare = QuantumCircuit(*circuit.qregs)
        padded_prepare.compose(prepare, qubits=list(prepare_qubits), inplace=True)
        padded_select = QuantumCircuit(*circuit.qregs)
        padded_select.compose(select, qubits=list(range(circuit.num_qubits)), inplace=True)
        for part in (padded_prepare, padded_select):
            part.metadata = deepcopy(circuit.metadata)
        symbolic = self._symbolic_select.copy() if opt == "Matrix-order" else None
        return BlockEncodingComponents(circuit, padded_prepare, padded_select, symbolic_select=symbolic,
                                       matrix_order_options=self.matrix_order_options if opt == "Matrix-order" else None)

    def select_circuit(self, opt="No", gateable=True):
        opt = self._option(opt)
        if not gateable:
            raise ValueError("Experiment SELECT must be unitary.")
        if opt == "Matrix-order" and len(self.J.instances) > 1:
            # Preserve this API's symbolic SELECT contract, but use exactly the
            # placement selected by circuit(), with aligned coefficients.
            self.circuit(opt)
            circuit = self._symbolic_select.copy()
        elif len(self.J.instances) == 1:
            circuit = self._single_term()
        else:
            circuit = super().select_circuit(opt=opt, gateable=True)
        return self._record(circuit, opt)


def system_last(circuit, system_size):
    """Map explicit block registers to the outer selector's workspace/system ABI."""
    system = (circuit.metadata or {}).get("system_qubits")
    if system is None:
        system = [circuit.find_bit(q).index for reg in circuit.qregs
                  if reg.name == "sys" for q in reg]
        if len(system) != system_size:
            system = list(range(circuit.num_qubits - system_size, circuit.num_qubits))
    if len(system) != system_size or len(set(system)) != system_size:
        raise ValueError("Invalid block system-qubit mapping.")
    workspace = [q for q in range(circuit.num_qubits) if q not in system]
    order = workspace + list(system)
    if order == list(range(circuit.num_qubits)):
        return circuit
    output = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    inverse = {old: new for new, old in enumerate(order)}
    output.compose(circuit, qubits=[inverse[q] for q in range(circuit.num_qubits)], inplace=True)
    output.metadata = {
        **(circuit.metadata or {}),
        "system_qubits": list(range(len(workspace), output.num_qubits)),
        "workspace_qubits": list(range(len(workspace))),
    }
    return output
