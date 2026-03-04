from channel_IR import *
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix, partial_trace, PauliList, Pauli
from qiskit.circuit.library import StatePreparation
from subroutine import lcu_prepare_tree, count_multiq_gates
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.circuit.controlledgate import ControlledGate
from itertools import product
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
        vec_phase_lookup[vec.tobytes()] = int(phases[idx]) % 4
    return vec_phase_lookup

def extract_basis_modes_with_phases(additional_modes, vec_phase_lookup):
    basis_modes = []
    nonbasis_modes = []
    for mode in additional_modes:
        pauli_label, ctrl_value = mode[0], mode[1]
        pauli = Pauli(pauli_label)
        vec = np.hstack((pauli.x.astype(int), pauli.z.astype(int)))
        phase = int(vec_phase_lookup.get(vec.tobytes(), 0))
        mode_with_phase = (pauli_label, ctrl_value, phase)
        if ctrl_value.count('1') == 1:
            basis_modes.append(mode_with_phase)
        else:
            nonbasis_modes.append(mode_with_phase)
    basis_modes.sort(key=lambda x: int(x[1], 2))
    nonbasis_modes.sort(key=lambda x: int(x[1], 2))
    return basis_modes, nonbasis_modes

def mobius_invert_modes_with_phases_legacy(w, additional_modes, vec_phase_lookup):
    def vec_to_pauli(vec):
        n = len(vec) // 2
        x = vec[:n]
        z = vec[n:]
        pauli = Pauli((z, x))
        return pauli
    if len(additional_modes) == 0:
        return {
            "p_phase_table": np.zeros(2 ** w, dtype=int),
            "g_phase_table": np.zeros(2 ** w, dtype=int),
            "phi_ad_table": np.zeros(2 ** w, dtype=int),
            "basis_modes_with_phase": [],
            "nonbasis_modes_with_phase": [],
            "g_modes_with_phase": [],
        }

    first_label = additional_modes[0][0]
    n = len(first_label)
    vec_len = 2 * n
    table_size = 2 ** w

    p_vec_table = np.zeros((table_size, vec_len), dtype=int)
    p_phase_table = np.zeros(table_size, dtype=int)
    relevant_addr_mask = np.zeros(table_size, dtype=bool)

    for mode in additional_modes:
        pauli_label, ctrl_value = mode[0], mode[1]
        idx = int(ctrl_value, 2)
        pauli = Pauli(pauli_label)
        vec = np.hstack((pauli.x.astype(int), pauli.z.astype(int)))
        vec_key = vec.tobytes()
        if vec_key in vec_phase_lookup:
            ## Relevant entry (belongs to the target U set): keep as hard P-table constraint.
            p_vec_table[idx] = vec
            p_phase_table[idx] = int(vec_phase_lookup[vec_key])
            relevant_addr_mask[idx] = True
        else:
            ## Filler entry: unconstrained in P table; leave as identity/zero phase.
            p_vec_table[idx] = np.zeros(vec_len, dtype=int)
            p_phase_table[idx] = 0
    g_vec_table = p_vec_table.copy()
    g_phase_table = p_phase_table.copy()
    for i in range(w):
        bit = 1 << i
        for b in range(table_size):
            if b & bit: 
                ## First count overlap (phases)
                pauli_curr, pauli_basis = vec_to_pauli(g_vec_table[b]), vec_to_pauli(g_vec_table[b ^ bit])
                overlap_phase = int(4 - (pauli_curr @ pauli_basis).phase)
                g_vec_table[b] = g_vec_table[b] ^ g_vec_table[b ^ bit]
                g_phase_table[b] = (g_phase_table[b] - g_phase_table[b ^ bit] - int(overlap_phase)) % 4
                


    ## For unconstrained (filler) addresses, force G_s = I with zero phase.
    for idx in range(table_size):
        if not relevant_addr_mask[idx]:
            g_vec_table[idx] = np.zeros(vec_len, dtype=int)
            g_phase_table[idx] = 0

    ## Reconstruct phase(P_b) from G_s by explicit Pauli multiplication order,
    ## so non-commuting overlaps contribute extra phase consistently.
    phase_from_g = np.zeros(table_size, dtype=int)
    for b in range(table_size):
        accum_vec = np.zeros(vec_len, dtype=int)
        accum_phase = 0
        for s in range(table_size):
            if (s & b) != s:
                continue
            if not np.any(g_vec_table[s]) and int(g_phase_table[s]) % 4 == 0:
                continue
            pauli_accum = vec_to_pauli(accum_vec)
            pauli_s = vec_to_pauli(g_vec_table[s])
            overlap_phase = int((pauli_s @ pauli_accum).phase) 
            accum_phase = (accum_phase + int(g_phase_table[s]) + int(overlap_phase)) % 4
            accum_vec = accum_vec ^ g_vec_table[s]
        phase_from_g[b] = accum_phase

    phi_ad_table = (p_phase_table - phase_from_g) % 4

    basis_modes_with_phase, nonbasis_modes_with_phase = extract_basis_modes_with_phases(additional_modes, vec_phase_lookup)

    g_modes_with_phase = []
    for idx in range(table_size):
        vec = g_vec_table[idx]
        has_nontrivial_vec = np.any(vec)
        has_nontrivial_phase = (int(g_phase_table[idx]) % 4) != 0
        if not has_nontrivial_vec and not has_nontrivial_phase:
            continue
        x = vec[:n]
        z = vec[n:]
        pauli_label = Pauli((z, x)).to_label()
        ctrl_value = bin(idx)[2:].zfill(w)
        g_modes_with_phase.append((pauli_label, ctrl_value, int(g_phase_table[idx]) % 4))

    return {
        "p_phase_table": p_phase_table,
        "g_phase_table": g_phase_table,
        "phi_ad_table": phi_ad_table,
        "relevant_addr_mask": relevant_addr_mask,
        "basis_modes_with_phase": basis_modes_with_phase,
        "nonbasis_modes_with_phase": nonbasis_modes_with_phase,
        "g_modes_with_phase": g_modes_with_phase,
    }


def mobius_invert_modes_with_phases_bottom_up(w, additional_modes, vec_phase_lookup):
    def vec_to_pauli(vec):
        n = len(vec) // 2
        x = vec[:n]
        z = vec[n:]
        return Pauli((z, x))

    if len(additional_modes) == 0:
        return {
            "p_phase_table": np.zeros(2 ** w, dtype=int),
            "g_phase_table": np.zeros(2 ** w, dtype=int),
            "phi_ad_table": np.zeros(2 ** w, dtype=int),
            "basis_modes_with_phase": [],
            "nonbasis_modes_with_phase": [],
            "g_modes_with_phase": [],
        }

    first_label = additional_modes[0][0]
    n = len(first_label)
    vec_len = 2 * n
    table_size = 2 ** w

    p_vec_table = np.zeros((table_size, vec_len), dtype=int)
    p_phase_table = np.zeros(table_size, dtype=int)
    relevant_addr_mask = np.zeros(table_size, dtype=bool)

    for mode in additional_modes:
        pauli_label, ctrl_value = mode[0], mode[1]
        idx = int(ctrl_value, 2)
        pauli = Pauli(pauli_label)
        vec = np.hstack((pauli.x.astype(int), pauli.z.astype(int)))
        vec_key = vec.tobytes()
        if vec_key in vec_phase_lookup:
            p_vec_table[idx] = vec
            p_phase_table[idx] = int(vec_phase_lookup[vec_key]) % 4
            relevant_addr_mask[idx] = True
        else:
            p_vec_table[idx] = np.zeros(vec_len, dtype=int)
            p_phase_table[idx] = 0

    g_vec_table = np.zeros_like(p_vec_table)
    g_phase_table = np.zeros_like(p_phase_table)

    for b in range(table_size):
        if not relevant_addr_mask[b]:
            continue

        wt = int(bin(b).count("1"))
        if wt <= 1:
            g_vec_table[b] = p_vec_table[b].copy()
            g_phase_table[b] = int(p_phase_table[b]) % 4
            continue

        curr_vec = p_vec_table[b].copy()
        curr_phase = int(p_phase_table[b]) % 4

        for c in range(1, b):
            if (c & b) != c:
                continue
            if not relevant_addr_mask[c]:
                continue
            if not np.any(g_vec_table[c]) and int(g_phase_table[c]) % 4 == 0:
                continue

            pauli_curr = vec_to_pauli(curr_vec)
            pauli_basis = vec_to_pauli(g_vec_table[c])
            overlap_phase = int( 4 - (pauli_curr @ pauli_basis).phase) % 4

            curr_vec = curr_vec ^ g_vec_table[c]
            curr_phase = (curr_phase - int(g_phase_table[c]) + overlap_phase) % 4

        g_vec_table[b] = curr_vec
        g_phase_table[b] = curr_phase

    phase_from_g = np.zeros(table_size, dtype=int)
    for b in range(table_size):
        accum_vec = np.zeros(vec_len, dtype=int)
        accum_phase = 0
        for s in range(table_size):
            if (s & b) != s:
                continue
            if not np.any(g_vec_table[s]) and int(g_phase_table[s]) % 4 == 0:
                continue
            pauli_accum = vec_to_pauli(accum_vec)
            pauli_s = vec_to_pauli(g_vec_table[s])
            overlap_phase = int( 4 - (pauli_s @ pauli_accum).phase) % 4
            accum_phase = (accum_phase + int(g_phase_table[s]) + int(overlap_phase)) % 4
            accum_vec = accum_vec ^ g_vec_table[s]
        phase_from_g[b] = accum_phase

    phi_ad_table = (p_phase_table - phase_from_g) % 4
    additional_indices = {int(mode[1], 2) for mode in additional_modes}
    for idx in additional_indices:
        g_phase_table[idx] = (int(g_phase_table[idx]) + int(phi_ad_table[idx])) % 4

    basis_modes_with_phase, nonbasis_modes_with_phase = extract_basis_modes_with_phases(additional_modes, vec_phase_lookup)

    g_modes_with_phase = []
    for idx in range(table_size):
        vec = g_vec_table[idx]
        has_nontrivial_vec = np.any(vec)
        has_nontrivial_phase = (int(g_phase_table[idx]) % 4) != 0
        if not has_nontrivial_vec and not has_nontrivial_phase:
            continue
        x = vec[:n]
        z = vec[n:]
        pauli_label = Pauli((z, x)).to_label()
        ctrl_value = bin(idx)[2:].zfill(w)
        g_modes_with_phase.append((pauli_label, ctrl_value, int(g_phase_table[idx]) % 4))

    return {
        "p_phase_table": p_phase_table,
        "g_phase_table": g_phase_table,
        "phi_ad_table": phi_ad_table,
        "relevant_addr_mask": relevant_addr_mask,
        "basis_modes_with_phase": basis_modes_with_phase,
        "nonbasis_modes_with_phase": nonbasis_modes_with_phase,
        "g_modes_with_phase": g_modes_with_phase,
    }


def mobius_invert_modes_with_phases(w, additional_modes, vec_phase_lookup, method="bottom-up"):
    if method == "legacy":
        return mobius_invert_modes_with_phases_legacy(w, additional_modes, vec_phase_lookup)
    return mobius_invert_modes_with_phases_bottom_up(w, additional_modes, vec_phase_lookup)

class BlockEncoding:
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
            return qc_pauli
    
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
    def mulplex_U_opt(self, mat_list, ctrl_size, sys_size):
        from qiskit.circuit.library import XGate
        mccount = 0
        tcount = 0
        cxcount = 0
        t_counts_per_ccx = 4
        cx_counts_per_ccx = 4  
        opt_circuit = QuantumCircuit(1 + 2 * ctrl_size + sys_size)
        opt_circuit.x(0)
        sel_regs = [2 * j + 1 for j in range(ctrl_size)]
        anc_regs = [2 * j + 2 for j in range(ctrl_size)]
        def apply_left_enc(j):
            opt_circuit.reset(anc_regs[j])
            ctrl_bval = control_values[j]
            ccxgate_c = XGate().control(num_ctrl_qubits = 2, ctrl_state = ctrl_bval + '1')
            top = 0 if j == 0 else anc_regs[j - 1]
            opt_circuit.append(ccxgate_c, [top, sel_regs[j], anc_regs[j]])
        def apply_right_enc(j):
            ctrl_bval = control_values[j]
            ccxgate_c = XGate().control(num_ctrl_qubits = 2, ctrl_state = ctrl_bval + '1')
            top = 0 if j == 0 else anc_regs[j - 1]
            opt_circuit.append(ccxgate_c, [top, sel_regs[j], anc_regs[j]])
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
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size,2 * ctrl_size + 1 +  numq)))
                opt_circuit.cx(anc_regs[ctrl_size - 2], anc_regs[ctrl_size - 1])
                cxcount += 1 + pauli_length
            elif i == len(mat_list) - 1:
                cval_prev = bin(i - 1)[2:].zfill(ctrl_size)
                diff_prev = next(j for j in range(ctrl_size) if cval_prev[j] != control_values[j])
                if diff_prev != ctrl_size - 1:
                    for j in range(diff_prev + 1, ctrl_size):
                        apply_left_enc(j)
                        mccount += 1
                        tcount += t_counts_per_ccx
                        cxcount += cx_counts_per_ccx
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size,2 * ctrl_size + 1 + numq)))
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
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size, 2 * ctrl_size + 1 + numq)))
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
                    opt_circuit.cx(0, anc_regs[0])
        opt_circuit.x(0)
        return opt_circuit, tcount, mccount, cxcount
    

    def find_optimal_order_matrices(self):
        pauli_list = []
        phase_list = []
        for i, ms in enumerate(self.mat_list):    
            if isinstance(ms, tuple) or isinstance(ms, list):
                pauli_op, coeff = ms
               
                if pauli_op != 'I' * len(pauli_op): #type: ignore
                    pauli_list.append(pauli_op) #type: ignore
                    phase_list.append(coeff)
        pauli_list = PauliList(pauli_list)
        x_part, z_part = pauli_list.x, pauli_list.z
        symplectic_matrix = np.hstack((x_part, z_part)).astype(int)
        phase_array = np.array([int(2 * np.angle(phase) / np.pi) for phase in phase_list])
        sorted_matrix_z, phases_z = sort_symplectic_matrix(symplectic_matrix, phase_array, crit = 'z')
        sorted_matrix_x, phases_x = sort_symplectic_matrix(symplectic_matrix, phase_array, crit = 'x')

        M = sorted_matrix_z.shape[0]
        w = int(np.ceil(np.log2(len(self.coeff_list)))) ## length of control register for full combination
        # w = sorted_matrix_z.shape[1] // 2 + 1 ## length of control register

        ## Pick up independent rows that serve as the basis
        selected_indices_z, maxcoverage_z, U_covered_z = greedy_generator_selection(sorted_matrix_x, w, w)
        
        selected_indices_x, maxcoverage_x, U_covered_x = greedy_generator_selection(sorted_matrix_z, w, w)

        if maxcoverage_z > maxcoverage_x:
            chosen_matrix = sorted_matrix_x
            chosen_phases = phases_x
            selected_matrix = sorted_matrix_x[selected_indices_z]
            remained_indices = set(range(M)) - set(selected_indices_z)
            U_covered = U_covered_z
            U_covered_new = set(U_covered)
            remaining_tuple = [sorted_matrix_x[i] for i in list(remained_indices) if sorted_matrix_x[i].tobytes() not in U_covered]
        else:
            chosen_matrix = sorted_matrix_z
            chosen_phases = phases_z
            selected_matrix = sorted_matrix_z[selected_indices_x]
            remained_indices = set(range(M)) - set(selected_indices_x)
            U_covered = U_covered_x
            U_covered_new = set(U_covered)
            remaining_tuple = [sorted_matrix_z[i] for i in list(remained_indices) if sorted_matrix_z[i].tobytes() not in U_covered]

        subspace_modes = assign_subspace_modes(w, selected_matrix, U_covered_new)
        
        additional_modes = assign_additional_modes(w, subspace_modes, remaining_tuple)
        ## Keep an explicit identity slot aligned with self.mat_list terms.
        identity_label = None
        for ms in self.mat_list:
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            if pauli_op == 'I' * len(pauli_op):
                identity_label = pauli_op
                break
        if identity_label is not None:
            zero_ctrl = '0' * w
            used_ctrl_ids = {ctrl for _, ctrl in additional_modes}
            zero_idx = next((i for i, (_, ctrl) in enumerate(additional_modes) if ctrl == zero_ctrl), None)
            if zero_idx is None:
                additional_modes.append((identity_label, zero_ctrl))
            else:
                old_label, _ = additional_modes[zero_idx]
                if old_label != identity_label:
                    displaced_mode = additional_modes[zero_idx]
                    new_ctrl = None
                    for ctrl_int in range(2 ** w):
                        candidate = bin(ctrl_int)[2:].zfill(w)
                        if candidate not in used_ctrl_ids:
                            new_ctrl = candidate
                            break
                    additional_modes[zero_idx] = (identity_label, zero_ctrl)
                    if new_ctrl is not None:
                        additional_modes.append((displaced_mode[0], new_ctrl))
        additional_modes = sorted(additional_modes, key=lambda x: int(x[1], 2))
        self.additional_modes = additional_modes

        ## Build coefficient list aligned with additional_modes order.
        ## Mapping rule: if mode Pauli exists in original operator terms, consume one
        ## coefficient from that Pauli pool; otherwise treat as filler with coeff = 0.
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
        self.vec_phase_lookup = build_vec_phase_lookup(chosen_matrix, chosen_phases)
        self.basis_modes_with_phase, self.nonbasis_modes_with_phase = extract_basis_modes_with_phases(additional_modes, self.vec_phase_lookup)
        self.mobius_phase_result = mobius_invert_modes_with_phases(w, additional_modes, self.vec_phase_lookup)
        
        return coeff_mode_dict, w
    def mulplex_U_opt_order(self):
        sys_size = self.sys_size
        coeff_mode_dict, ctrl_size = self.find_optimal_order_matrices()
        modes_with_phase = self.mobius_phase_result['g_modes_with_phase']
        
        ctrl_reg, sys_reg = QuantumRegister(ctrl_size, 'ctrl'), QuantumRegister(sys_size, 'sys')
        qc_u = QuantumCircuit(ctrl_reg, sys_reg)
        mccount = 0
        cxcount = 0
        self.coeff_list_ordered = np.zeros((1 << ctrl_size), dtype = float)
        # if '0' * ctrl_size in coeff_mode_dict:
        #     self.coeff_list_ordered[0] = coeff_mode_dict['0' * ctrl_size]
        for ctrl_value, coeff in coeff_mode_dict.items():
            self.coeff_list_ordered[int(ctrl_value, 2)] = coeff
        
        for pauli_label, ctrl_value, phase in modes_with_phase:
            position = int(ctrl_value, 2)
            # self.coeff_list_ordered[position] = coeff_mode_dict[ctrl_value]
            qc_pauli = QuantumCircuit(len(pauli_label))
            pauli_op = Pauli(pauli_label)
            qc_pauli.append(pauli_op, range(len(pauli_label)))
            qc_pauli.global_phase = np.pi * phase / 2
            qc_pauli = qc_pauli.decompose()
            ## Match qiskit's ctrl_state ordering in mulplex_U:
            ## rightmost bit in ctrl_value corresponds to lower-index control qubit.
            active_ctrls = [i for i, bit in enumerate(ctrl_value[::-1]) if bit == '1']
            if len(active_ctrls) == 0:
                qc_u.append(qc_pauli.to_gate(), list(range(ctrl_size, ctrl_size + sys_size)))
            else:
                ctrl_U_elem = qc_pauli.to_gate().control(num_ctrl_qubits = len(active_ctrls), ctrl_state = '1' * len(active_ctrls))
                qargs = active_ctrls + list(range(ctrl_size, ctrl_size + sys_size))
                qc_u.append(ctrl_U_elem, qargs)
                if len(active_ctrls) == 1:
                    cxcount += 1
                else:
                    mccount += len(active_ctrls) - 1

        tcount = mccount * 4
        cxcount += mccount * 4
        return qc_u, tcount, mccount, cxcount, ctrl_size
            
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

    def circuit(self, opt = None):
        """
        Returns the block-encoding QuantumCircuit for the operator J.
        """
        self.mccount = 0
        self.cxcount = 0
        if self.ctrl_size == 0:
            sys = QuantumRegister(self.sys_size, 'sys')
            qc = QuantumCircuit(sys)
            if opt == False:
                qc_u, tcount, mccount, cxcount = self.mulplex_U(self.mat_list, 0, self.sys_size)
            else:
                qc_u, tcount, mccount, cxcount = self.mulplex_U_opt(self.mat_list, 0, self.sys_size)
            qc.compose(qc_u, qubits=sys[:], inplace=True)
            self.circuit_width = qc.num_qubits
            return qc
        
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
            qc_u, tcount, mccount, cxcount = self.mulplex_U_opt(self.mat_list, self.ctrl_size, self.sys_size)
            qc_select = self.mulplex_B(self.coeff_list, self.ctrl_size)
            ctrl_index = [2 * j + 1 for j in range(self.ctrl_size)]
            qc = QuantumCircuit(qc_u.num_qubits, name = "BlockEncoding")
            qc.compose(qc_select, qubits = ctrl_index, inplace = True) #type: ignore 
            qc.compose(qc_u, qubits = qc.qubits, inplace = True)
            qc.compose(qc_select.inverse(), qubits = ctrl_index, inplace = True) #type: ignore
        elif opt == 'Matrix-order':
            # coeff_mode_dict, ctrl_size = self.find_optimal_order_matrices()
            qc_u, tcount, mccount, cxcount, ctrl_size = self.mulplex_U_opt_order()
            # qc = qc_u.copy()
            qc_select = self.mulplex_B_opt_order(ctrl_size)
            ctrl, sys = QuantumRegister(ctrl_size, 'ctrl'), QuantumRegister(self.sys_size, 'sys')
            qc = QuantumCircuit(ctrl, sys, name = "BlockEncoding")
            qc.compose(qc_select, qubits = ctrl, inplace = True) #type: ignore 
            qc.compose(qc_u, qubits = qc.qubits, inplace = True)
            qc.compose(qc_select.inverse(), qubits = ctrl, inplace = True) #type: ignore
        self.tcount = tcount
        self.mccount += mccount
        self.cxcount = cxcount
        self.succ_prob = np.sum(self.coeff_list)
        self.circuit_width = qc.num_qubits
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
        backend = AerSimulator(method="statevector")
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
        simulator = AerSimulator(method="statevector")

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
        backend = AerSimulator(method="matrix_product_state")
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
        simulator = AerSimulator(method="matrix_product_state")
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

if __name__ == "__main__":
   
    from channel_LCU import Lindblad_to_channel 
    from qiskit.quantum_info import random_pauli_list
    N = 3
    H = []
    L_list = []
    gamma = np.sqrt(0.1)/2 
    for i in range(N):
        iden_str = 'I' * N
        Z_ind = [i, (i + 1) % N]
        Z_str = ''.join(['Z' if j in Z_ind else 'I' for j in range(N)])
        H.append((Z_str, 1j))
        X_str = ''.join([('X' if j == i else 'I') for j in range(N)]) 
        H.append((X_str, -1))
        Y_str = ''.join([('Y' if j == i else 'I') for j in range(N)])
        L_list.append([(X_str, gamma), (Y_str, -1j * gamma)])
    delta_t = 0.1
    gamma = np.sqrt(0.1)/2 
    # H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
    # L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
    TFIM_lind = Lindbladian(H, L_list)
    channel_Lind, success_prob_th, coeff_sum = Lindblad_to_channel(TFIM_lind, delta_t)
    channel_Lind = channel_Lind.channels[0][1]
    ms = channel_Lind[0]
    random_pauli = random_pauli_list(num_qubits = 3, size = 6, phase = False)
    H_eff = [(ms.to_label(), -1.0) for ms in random_pauli] #type: ignore
    Random_Lind = Lindbladian(H_eff, [])
    H_eff = Random_Lind.H
    J = BlockEncoding(H_eff)
    qc_opt_line = J.circuit(opt = 'Ctrl-line')
    print(qc_opt_line.draw())
    qc_no = J.circuit(opt = 'No')
    # J = BlockEncoding(ms)
    # J.find_optimal_order_matrices()
    # qc_nopt = J.circuit(opt = 'No')
    # qc_opt_line = J.circuit(opt = 'Matrix-order')
    # print(qc_nopt.draw())
    # print(qc_opt_line.draw())
    # print(J.tcount, J.mccount, J.cxcount)
    # ms = build_all_pauli_matrixsum(n = 4)
    # J = BlockEncoding(ms)
    # # J.find_optimal_order_matrices()
    # qc_opt_mo = J.circuit(opt = 'Matrix-order')
    # print(J.tcount, J.mccount, J.cxcount)
#     TFIM_lind = Lindbladian(H, L_list)

#     channel_Lind, success_prob_th, coeff_sum = Lindblad_to_channel(TFIM_lind, delta_t)

#     channel_Lind = channel_Lind.channels[0][1]
#     ms = channel_Lind[0]
#     ## Unoptimized version
#     print(ms)
#     ms_be = BlockEncoding(ms)

#     print("Unoptimized version:")
#     qc_be = ms_be.circuit(opt = False)   
#     tcount, mccount, cxcount = ms_be.tcount, ms_be.mccount, ms_be.cxcount
#     print(f"T-count: {tcount}, Multi-controlled gate count: {mccount}, CX count: {cxcount}")

#     ## Optimized version I: unary iteration
#     ms_be = BlockEncoding(ms)

#     qc_be_opt1 = ms_be.circuit(opt = True)
#     tcount_opt1, mccount_opt1, cxcount_opt1 = ms_be.tcount, ms_be.mccount, ms_be.cxcount
#     print("Optimized version I (unary iteration):")
#     print(f"T-count: {tcount_opt1}, Multi-controlled gate count: {mccount_opt1}, CX count: {cxcount_opt1}")

#     ## Optimized version II: optimization over gate structures

#     qc = QuantumCircuit(7)

#     mccount, tcount, cxcount = 0, 0, 0
#     L = ms.length
#     w = int(np.ceil(np.log2(L)))
#     for i in range(N):
#         ## Implement -Z on first three control lines
#         qc.p(np.pi, i)
#         qc.cz(i, i + w)
#         cxcount += 1
#         ## Implement -Y on (0, 3), (1, 3), (2, 3)
#     for i in range(N):
#         qc.cp(np.pi, i, w - 1)
#         ccy = YGate().control(num_ctrl_qubits = 2, ctrl_state = '11')
#         qc.append(ccy, [i, w - 1, i + w])
#         mccount += 1
#         tcount += 4
#         cxcount += 4
#     print("Optimized version II: exploring Pauli structures in LCU")
#     print(f"T-count: {tcount}, Multi-controlled gate count: {mccount}, CX count: {cxcount}")
        