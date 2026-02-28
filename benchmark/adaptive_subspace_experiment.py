import numpy as np
from qiskit.quantum_info import PauliList, Pauli
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from channel_IR import Matrixsum, PauliAtom
from block_encoding import (
    BlockEncoding,
    sort_symplectic_matrix,
    greedy_generator_selection,
    assign_subspace_modes,
    assign_additional_modes,
)


def pauli_to_vec(pauli_label: str) -> np.ndarray:
    pauli = Pauli(pauli_label)
    return np.hstack((pauli.x.astype(int), pauli.z.astype(int)))


def vec_to_pauli(vec: np.ndarray) -> str:
    n = len(vec) // 2
    x = vec[:n]
    z = vec[n:]
    return Pauli((z, x)).to_label()


def vec_weight(vec: np.ndarray) -> int:
    n = len(vec) // 2
    x = vec[:n]
    z = vec[n:]
    return int(np.sum(x | z))


def gf2_rank(mat: np.ndarray) -> int:
    if mat.size == 0:
        return 0
    a = mat.copy().astype(np.uint8)
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if a[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            a[[r, pivot]] = a[[pivot, r]]
        for i in range(rows):
            if i != r and a[i, c]:
                a[i, :] ^= a[r, :]
        r += 1
        if r == rows:
            break
    return r


def is_independent(candidate: np.ndarray, basis_rows: list[np.ndarray]) -> bool:
    if len(basis_rows) == 0:
        return bool(np.any(candidate))
    mat = np.vstack(basis_rows)
    r0 = gf2_rank(mat)
    r1 = gf2_rank(np.vstack([mat, candidate]))
    return r1 > r0


def build_subspace_map(basis_rows: list[np.ndarray]):
    d = len(basis_rows)
    if d == 0:
        return {}
    vec_len = len(basis_rows[0])
    subspace_map = {}
    for c_int in range(2 ** d):
        coeff_id = bin(c_int)[2:].zfill(d)
        coeff_vec = np.zeros(vec_len, dtype=int)
        for bit_i in range(d):
            if coeff_id[d - 1 - bit_i] == '1':
                coeff_vec = coeff_vec ^ basis_rows[bit_i]
        subspace_map[coeff_id] = coeff_vec
    return subspace_map


def build_candidate_pool_from_covered_subspace(target_matrix: np.ndarray, max_combo_order: int = 2) -> list[np.ndarray]:
    """
    Build a lightweight candidate pool that includes vectors from covered subspace.

    - Always includes all original target rows.
    - Adds low-order XOR combinations (currently order-2 by default), which is often
      enough to expose low-weight Z-like directions from X/Y pairs.
    """
    pool_map: dict[bytes, np.ndarray] = {}

    for row in target_matrix:
        b = row.tobytes()
        if b not in pool_map:
            pool_map[b] = row.copy()

    if max_combo_order >= 2:
        rows = [row.copy() for row in target_matrix]
        n_rows = len(rows)
        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                cand = rows[i] ^ rows[j]
                if not np.any(cand):
                    continue
                b = cand.tobytes()
                if b not in pool_map:
                    pool_map[b] = cand

    candidates = list(pool_map.values())
    candidates.sort(key=lambda v: (vec_weight(v), int.from_bytes(v.tobytes(), 'little')))
    return candidates


def vec_y_count(vec: np.ndarray) -> int:
    n = len(vec) // 2
    x = vec[:n]
    z = vec[n:]
    return int(np.sum(x & z))


def vec_z_weight(vec: np.ndarray) -> int:
    n = len(vec) // 2
    z = vec[n:]
    return int(np.sum(z))


def minimum_weight_basis(candidate_pool: list[np.ndarray], rank_target: int) -> list[np.ndarray]:
    ranked = sorted(
        [v for v in candidate_pool if np.any(v)],
        key=lambda v: (
            vec_weight(v),
            vec_y_count(v),
            -vec_z_weight(v),
            int.from_bytes(v.tobytes(), 'little'),
        )
    )

    basis: list[np.ndarray] = []
    for cand in ranked:
        if is_independent(cand, basis):
            basis.append(cand.copy())
            if len(basis) >= rank_target:
                break
    return basis


def build_all_pauli_matrixsum_from_labels(labels: list[str]) -> Matrixsum:
    instances = [(PauliAtom(label, phase=1.0), 1.0) for label in labels]
    return Matrixsum(instances)


class AdaptiveBlockEncoding(BlockEncoding):
    def _collect_symplectic_targets(self):
        pauli_list = []
        phase_list = []
        for ms in self.mat_list:
            if not isinstance(ms, (tuple, list)):
                continue
            pauli_op, coeff = ms
            if pauli_op != 'I' * len(pauli_op):
                pauli_list.append(pauli_op)
                phase_list.append(coeff)

        pauli_list = PauliList(pauli_list)
        x_part, z_part = pauli_list.x, pauli_list.z
        symplectic_matrix = np.hstack((x_part, z_part)).astype(int)
        phase_array = np.array([int(2 * np.angle(phase) / np.pi) for phase in phase_list])
        return symplectic_matrix, phase_array

    def _evaluate_basis_state(self, basis_rows: list[np.ndarray], target_matrix: np.ndarray, w: int):
        if len(basis_rows) == 0:
            return {
                'objective': float('inf'),
                'distance_sum': float('inf'),
                'unassigned': target_matrix.shape[0],
                'covered': 0,
                'assigned_subspace': [],
                'assigned_full': [],
                'remaining': [row.copy() for row in target_matrix],
            }

        basis_matrix = np.vstack(basis_rows).astype(int)
        target_bytes_set = {row.tobytes() for row in target_matrix}

        span_set = {np.zeros(target_matrix.shape[1], dtype=int).tobytes()}
        for b in basis_rows:
            old = list(span_set)
            for g_bytes in old:
                g = np.frombuffer(g_bytes, dtype=int)
                span_set.add((g ^ b).tobytes())

        covered_set = {x for x in span_set if x in target_bytes_set}

        assigned_subspace = assign_subspace_modes(w, basis_matrix, covered_set)

        remaining = [
            row.copy() for row in target_matrix
            if row.tobytes() not in covered_set
        ]

        assigned_full = assign_additional_modes(w, assigned_subspace, remaining)

        assigned_sub_keys = {(p, c) for p, c in assigned_subspace}
        d = len(basis_rows)
        subspace_map = build_subspace_map(basis_rows)

        distance_sum = 0
        for pauli_label, ctrl_value in assigned_full:
            if (pauli_label, ctrl_value) in assigned_sub_keys:
                continue
            coeff_id = ctrl_value[-d:] if d > 0 else ''
            if coeff_id not in subspace_map:
                continue
            vec = pauli_to_vec(pauli_label)
            ref_vec = subspace_map[coeff_id]
            distance_sum += vec_weight(vec ^ ref_vec)

        assigned_vecs = {pauli_to_vec(p).tobytes() for p, _ in assigned_full}
        unassigned = sum(1 for row in target_matrix if row.tobytes() not in assigned_vecs)
        basis_weight_sum = int(sum(vec_weight(row) for row in basis_rows))

        objective = distance_sum + 1000 * unassigned
        return {
            'objective': float(objective),
            'distance_sum': int(distance_sum),
            'unassigned': int(unassigned),
            'basis_weight_sum': basis_weight_sum,
            'covered': int(len(covered_set)),
            'assigned_subspace': assigned_subspace,
            'assigned_full': assigned_full,
            'remaining': remaining,
        }

    def baseline_pipeline(self, crit: str = 'z'):
        matrix, phases = self._collect_symplectic_targets()
        sorted_matrix, _ = sort_symplectic_matrix(matrix, phases, crit=crit)
        w = int(np.ceil(np.log2(len(self.coeff_list))))

        selected_idx, _, covered = greedy_generator_selection(sorted_matrix, w, w)
        basis_rows = [sorted_matrix[i].copy() for i in selected_idx]
        eval_info = self._evaluate_basis_state(basis_rows, sorted_matrix, w)

        return {
            'w': w,
            'd': len(basis_rows),
            'basis_indices': selected_idx,
            'basis_labels': [vec_to_pauli(v) for v in basis_rows],
            **eval_info,
        }

    def adaptive_pipeline(self, crit: str = 'z', verbose: bool = True, max_combo_order: int = 2):
        matrix, phases = self._collect_symplectic_targets()
        sorted_matrix, _ = sort_symplectic_matrix(matrix, phases, crit=crit)
        w = int(np.ceil(np.log2(len(self.coeff_list))))
        candidate_pool = build_candidate_pool_from_covered_subspace(sorted_matrix, max_combo_order=max_combo_order)

        def score_tuple(eval_info, d_value):
            return (
                int(eval_info['unassigned']),
                int(eval_info['distance_sum']),
                int(eval_info.get('basis_weight_sum', 10**9)),
                int(d_value),
            )

        def run_greedy_from_seed(seed_idx: int):
            seed_vec = candidate_pool[seed_idx]
            if not np.any(seed_vec):
                return None

            basis_rows: list[np.ndarray] = [seed_vec.copy()]
            chosen_indices: list[int] = [seed_idx]
            chosen_basis_keys: set[bytes] = {seed_vec.tobytes()}
            current_eval = self._evaluate_basis_state(basis_rows, sorted_matrix, w)

            if verbose:
                print(
                    f"[adaptive] seed idx={seed_idx}, basis={vec_to_pauli(seed_vec)}, "
                    f"objective={current_eval['objective']}, distance={current_eval['distance_sum']}"
                )

            while len(basis_rows) < w:
                best_eval = current_eval
                best_idx = None
                best_basis = None
                best_score = score_tuple(current_eval, len(basis_rows))

                for idx, cand in enumerate(candidate_pool):
                    if cand.tobytes() in chosen_basis_keys:
                        continue
                    if not is_independent(cand, basis_rows):
                        continue

                    trial_basis = basis_rows + [cand.copy()]
                    trial_eval = self._evaluate_basis_state(trial_basis, sorted_matrix, w)
                    trial_score = score_tuple(trial_eval, len(trial_basis))

                    if trial_score < best_score:
                        best_eval = trial_eval
                        best_idx = idx
                        best_basis = trial_basis
                        best_score = trial_score

                if best_idx is None:
                    break

                basis_rows = best_basis
                chosen_indices.append(best_idx)
                chosen_basis_keys.add(candidate_pool[best_idx].tobytes())
                current_eval = best_eval

                if verbose:
                    print(
                        f"[adaptive] accept basis idx={best_idx}, basis={vec_to_pauli(candidate_pool[best_idx])}, d={len(basis_rows)}, "
                        f"objective={current_eval['objective']}, "
                        f"distance={current_eval['distance_sum']}, "
                        f"unassigned={current_eval['unassigned']}"
                    )

            return {
                'w': w,
                'd': len(basis_rows),
                'basis_indices': chosen_indices,
                'basis_labels': [vec_to_pauli(v) for v in basis_rows],
                **current_eval,
            }

        all_results = []
        for seed_idx in range(len(candidate_pool)):
            result = run_greedy_from_seed(seed_idx)
            if result is not None:
                all_results.append(result)

        if len(all_results) == 0:
            return {
                'w': w,
                'd': 0,
                'basis_indices': [],
                'basis_labels': [],
                'objective': float('inf'),
                'distance_sum': float('inf'),
                'unassigned': sorted_matrix.shape[0],
                'basis_weight_sum': float('inf'),
                'covered': 0,
                'assigned_subspace': [],
                'assigned_full': [],
                'remaining': [row.copy() for row in sorted_matrix],
            }

        all_results.sort(
            key=lambda r: (
                int(r['unassigned']),
                int(r['distance_sum']),
                int(r.get('basis_weight_sum', 10**9)),
                int(r['d']),
            )
        )
        best_result = all_results[0]

        if int(best_result['unassigned']) == 0 and int(best_result['distance_sum']) == 0:
            rank_target = gf2_rank(sorted_matrix)
            mw_basis_rows = minimum_weight_basis(candidate_pool, rank_target)
            if len(mw_basis_rows) == rank_target:
                mw_eval = self._evaluate_basis_state(mw_basis_rows, sorted_matrix, w)
                mw_result = {
                    'w': w,
                    'd': len(mw_basis_rows),
                    'basis_indices': [],
                    'basis_labels': [vec_to_pauli(v) for v in mw_basis_rows],
                    **mw_eval,
                }
                mw_score = (
                    int(mw_result['unassigned']),
                    int(mw_result['distance_sum']),
                    int(mw_result.get('basis_weight_sum', 10**9)),
                    int(mw_result['d']),
                )
                best_score = (
                    int(best_result['unassigned']),
                    int(best_result['distance_sum']),
                    int(best_result.get('basis_weight_sum', 10**9)),
                    int(best_result['d']),
                )
                if mw_score <= best_score:
                    best_result = mw_result

        return best_result


def run_case(pauli_labels: list[str], title: str):
    print('=' * 90)
    print(title)
    ms = build_all_pauli_matrixsum_from_labels(pauli_labels)
    tester = AdaptiveBlockEncoding(ms)

    baseline = tester.baseline_pipeline(crit='z')
    adaptive = tester.adaptive_pipeline(crit='z', verbose=True)

    print('-' * 90)
    print('[baseline]')
    print(
        f"w={baseline['w']}, d={baseline['d']}, covered={baseline['covered']}, "
        f"unassigned={baseline['unassigned']}, distance={baseline['distance_sum']}, "
        f"objective={baseline['objective']}"
    )
    print(f"basis={baseline['basis_labels']}")

    print('-' * 90)
    print('[adaptive]')
    print(
        f"w={adaptive['w']}, d={adaptive['d']}, covered={adaptive['covered']}, "
        f"unassigned={adaptive['unassigned']}, distance={adaptive['distance_sum']}, "
        f"objective={adaptive['objective']}"
    )
    print(f"basis={adaptive['basis_labels']}")


def run_all_3bit_pauli_case():
    from itertools import product

    labels = [''.join(p) for p in product('IXYZ', repeat=3)]
    print('=' * 90)
    print('Case C: all 3-qubit Pauli strings (including III)')

    ms = build_all_pauli_matrixsum_from_labels(labels)
    tester = AdaptiveBlockEncoding(ms)

    matrix, phases = tester._collect_symplectic_targets()
    sorted_matrix, _ = sort_symplectic_matrix(matrix, phases, crit='z')
    w = int(np.ceil(np.log2(len(tester.coeff_list))))

    expected_basis_labels = ['ZII', 'IZI', 'IIZ', 'XII', 'IXI', 'IIX']
    expected_basis_rows = [pauli_to_vec(label) for label in expected_basis_labels]
    expected_eval = tester._evaluate_basis_state(expected_basis_rows, sorted_matrix, w)

    adaptive = tester.adaptive_pipeline(crit='z', verbose=False, max_combo_order=1)

    print('[expected basis check]')
    print(f"basis={expected_basis_labels}")
    print(
        f"w={w}, d={len(expected_basis_labels)}, covered={expected_eval['covered']}, "
        f"unassigned={expected_eval['unassigned']}, distance={expected_eval['distance_sum']}, "
        f"objective={expected_eval['objective']}"
    )

    print('[adaptive]')
    print(
        f"w={adaptive['w']}, d={adaptive['d']}, covered={adaptive['covered']}, "
        f"unassigned={adaptive['unassigned']}, distance={adaptive['distance_sum']}, "
        f"objective={adaptive['objective']}"
    )
    print(f"basis={adaptive['basis_labels']}")


def main():
    # run_case(
    #     ['ZZI', 'ZIZ', 'IZZ', 'XII', 'IXI', 'IIX'],
    #     title='Case A: user TFIM-like 6 Pauli terms',
    # )

    # run_case(
    #     ['ZZI', 'ZIZ', 'IZZ', 'XII', 'IXI', 'IIX', 'ZII', 'IZI', 'IIZ'],
    #     title='Case B: 9 Pauli terms (extended)',
    # )

    run_all_3bit_pauli_case()


if __name__ == '__main__':
    main()
