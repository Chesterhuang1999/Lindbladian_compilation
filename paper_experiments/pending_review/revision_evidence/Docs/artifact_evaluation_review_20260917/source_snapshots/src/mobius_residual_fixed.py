"""Pauli residual inversion with the zero-address factor included at every layer."""

import numpy as np
from qiskit.quantum_info import Pauli
from block_encoding import extract_basis_modes_with_phases


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
