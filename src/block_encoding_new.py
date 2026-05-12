"""Multi-candidate variant of `BlockEncoding`.

Differs from the legacy `block_encoding.BlockEncoding` only in
`find_optimal_order_matrices`, which evaluates two candidate basis sizes
(`k = w` and `k = w - 1`) and selects the one with smaller weighted control
cost `C = Σ w(ctrl) · w_s(g_l)` (Möbius output). All other behaviour
(circuit synthesis, resource counting, etc.) is inherited unchanged.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import PauliList

from block_encoding import (
    BlockEncoding as _LegacyBlockEncoding,
    sort_symplectic_matrix,
    greedy_generator_selection,
    assign_subspace_modes,
    assign_additional_modes,
    build_vec_phase_lookup,
    extract_basis_modes_with_phases,
    mobius_invert_modes_with_phases,
)


class BlockEncoding(_LegacyBlockEncoding):
    """Block-encoding with multi-candidate (k = w, w-1) basis selection."""

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

        sorted_matrix_z, phases_z = sort_symplectic_matrix(symplectic_matrix, phase_array, crit='z')
        sorted_matrix_x, phases_x = sort_symplectic_matrix(symplectic_matrix, phase_array, crit='x')

        M = sorted_matrix_z.shape[0]
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

        def _build_candidate(k):
            ## Stage I: greedy basis on both x-prior and z-prior sortings.
            sel_z, cov_z, U_z = greedy_generator_selection(sorted_matrix_x, w, k)
            sel_x, cov_x, U_x = greedy_generator_selection(sorted_matrix_z, w, k)

            if cov_z > cov_x:
                cand_matrix = sorted_matrix_x
                cand_phases = phases_x
                cand_selected = sorted_matrix_x[sel_z]
                remained_indices = set(range(M)) - set(sel_z)
                cand_U = U_z
                cand_remaining = [sorted_matrix_x[i] for i in remained_indices
                                  if sorted_matrix_x[i].tobytes() not in cand_U]
            else:
                cand_matrix = sorted_matrix_z
                cand_phases = phases_z
                cand_selected = sorted_matrix_z[sel_x]
                remained_indices = set(range(M)) - set(sel_x)
                cand_U = U_x
                cand_remaining = [sorted_matrix_z[i] for i in remained_indices
                                  if sorted_matrix_z[i].tobytes() not in cand_U]

            ## Stages II/III: subspace + additional placement.
            subspace_modes = assign_subspace_modes(w, cand_selected, set(cand_U))
            cand_additional = assign_additional_modes(w, subspace_modes, cand_remaining)

            ## Identity handling: keep an explicit identity slot at ctrl = 0...0.
            if identity_label is not None:
                zero_ctrl = '0' * w
                used_ctrl_ids = {ctrl for _, ctrl in cand_additional}
                zero_idx = next((i for i, (_, c) in enumerate(cand_additional) if c == zero_ctrl), None)
                if zero_idx is None:
                    cand_additional.append((identity_label, zero_ctrl))
                else:
                    old_label, _ = cand_additional[zero_idx]
                    if old_label != identity_label:
                        displaced_mode = cand_additional[zero_idx]
                        new_ctrl = None
                        for ctrl_int in range(2 ** w):
                            candidate = bin(ctrl_int)[2:].zfill(w)
                            if candidate not in used_ctrl_ids:
                                new_ctrl = candidate
                                break
                        cand_additional[zero_idx] = (identity_label, zero_ctrl)
                        if new_ctrl is not None:
                            cand_additional.append((displaced_mode[0], new_ctrl))
            cand_additional = sorted(cand_additional, key=lambda x: int(x[1], 2))

            ## Möbius inversion gives g_l. Cost C = Σ w(ctrl) · w_s(g_l).
            cand_vec_phase_lookup = build_vec_phase_lookup(cand_matrix, cand_phases)
            cand_mobius = mobius_invert_modes_with_phases(
                w, cand_additional, cand_vec_phase_lookup, zero_phase
            )
            g_modes = cand_mobius['g_modes_with_phase']
            cost = sum(ctrl.count('1') * _pauli_weight(label) for label, ctrl, _ in g_modes)

            return {
                'k': k,
                'cost': cost,
                'additional_modes': cand_additional,
                'chosen_matrix': cand_matrix,
                'chosen_phases': cand_phases,
                'vec_phase_lookup': cand_vec_phase_lookup,
                'mobius_result': cand_mobius,
            }

        ## Two candidates: k = w and k = w - 1 (skip when w-1 < 1).
        candidate_ks = [w]
        if w - 1 >= 1:
            candidate_ks.append(w - 1)

        candidates = [_build_candidate(k) for k in candidate_ks]
        ## Tie-break: prefer larger k (closer to legacy behaviour) on equal cost.
        best = min(candidates, key=lambda c: (c['cost'], -c['k']))

        additional_modes = best['additional_modes']

        self.additional_modes = additional_modes
        self.candidate_costs = [(c['k'], c['cost']) for c in candidates]
        self.selected_k = best['k']

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
        self.vec_phase_lookup = best['vec_phase_lookup']
        self.basis_modes_with_phase, self.nonbasis_modes_with_phase = extract_basis_modes_with_phases(
            additional_modes, self.vec_phase_lookup, zero_phase
        )
        self.mobius_phase_result = best['mobius_result']

        return coeff_mode_dict, w
