import time
from itertools import product
from pathlib import Path
import sys

import numpy as np
from qiskit.quantum_info import PauliList

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from block_encoding import greedy_generator_selection, sort_symplectic_matrix


def greedy_generator_selection_old(sorted_matrix, max_length=None, max_gens=None):
    M, N = sorted_matrix.shape
    n = N // 2
    max_elements = 2 ** max_length if max_length is not None else 1e9

    rows_as_bytes = [row.tobytes() for row in sorted_matrix]
    U_set = set(rows_as_bytes)

    candidates_indices = list(range(M))

    first_idx = 0
    first_vec = sorted_matrix[first_idx]
    selected_indices = [first_idx]

    identity_vec = np.zeros(N, dtype=int)
    current_group_elements = {identity_vec.tobytes(), first_vec.tobytes()}

    candidates_indices.pop(0)

    current_covered_in_U = {b for b in current_group_elements if b in U_set}

    def get_weight(vec_bytes):
        vec = np.frombuffer(vec_bytes, dtype=int)
        x = vec[:n]
        z = vec[n:]
        non_identity = x | z
        return np.sum(non_identity)

    while candidates_indices:
        if (max_gens and len(selected_indices) >= max_gens):
            break

        best_score = -1.0
        best_idx = -1
        best_new_covered_elements = None

        to_remove = []
        candidate_found = False

        for i, idx in enumerate(candidates_indices):
            vec_bytes = rows_as_bytes[idx]
            if vec_bytes in current_group_elements:
                to_remove.append(i)
                continue

            vec_arr = sorted_matrix[idx]
            new_covered_chunk = []

            for g_bytes in current_group_elements:
                g_arr = np.frombuffer(g_bytes, dtype=int)

                new_vec = g_arr ^ vec_arr
                new_vec_bytes = new_vec.tobytes()
                if new_vec_bytes in U_set:
                    if new_vec_bytes not in current_covered_in_U:
                        new_covered_chunk.append(new_vec_bytes)

            count_delta = len(new_covered_chunk)
            wt = get_weight(vec_bytes)
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
        new_group_elements = set()
        for g_bytes in current_group_elements:
            g_arr = np.frombuffer(g_bytes, dtype=int)
            new_vec = g_arr ^ target_vec
            new_group_elements.add(new_vec.tobytes())

        current_group_elements.update(new_group_elements)

        if best_new_covered_elements:
            current_covered_in_U.update(best_new_covered_elements)

        current_group_size = len(current_group_elements)
        remaining_uncovered = M - len(current_covered_in_U)
        lhs = max_elements - current_group_size
        rhs = remaining_uncovered
        if lhs < rhs:
            break

        candidates_indices.pop(best_idx)

    return selected_indices, len(current_covered_in_U), current_covered_in_U


def build_all_pauli_symplectic(n_qubits: int):
    labels = [''.join(p) for p in product('IXYZ', repeat=n_qubits)]
    labels = [p for p in labels if p != 'I' * n_qubits]
    pauli_list = PauliList(labels)
    x_part, z_part = pauli_list.x, pauli_list.z
    return np.hstack((x_part, z_part)).astype(int)


def run_once(matrix, w):
    phases = np.zeros(matrix.shape[0], dtype=int)
    sorted_matrix, _ = sort_symplectic_matrix(matrix, phases, crit='z')

    t0 = time.perf_counter()
    old_sel, old_cov, old_set = greedy_generator_selection_old(sorted_matrix, w, w)
    t_old = time.perf_counter() - t0

    t1 = time.perf_counter()
    new_sel, new_cov, new_set = greedy_generator_selection(sorted_matrix, w, w)
    t_new = time.perf_counter() - t1

    same = (old_sel == new_sel) and (old_cov == new_cov) and (old_set == new_set)
    return same, t_old, t_new, len(old_sel), old_cov


def main():
    np.random.seed(0)

    print('=== Consistency + Timing: old vs optimized greedy_generator_selection ===')

    for n in [3, 4]:
        matrix = build_all_pauli_symplectic(n)
        w = int(np.ceil(np.log2(matrix.shape[0] + 1)))
        same, t_old, t_new, nsel, cov = run_once(matrix, w)
        speedup = t_old / t_new if t_new > 0 else float('inf')
        print(f'[all-pauli n={n}] same={same} selected={nsel} covered={cov} old={t_old:.4f}s new={t_new:.4f}s speedup={speedup:.2f}x')

    for test_id in range(5):
        rows = 120
        n = 5
        cols = 2 * n
        matrix = np.random.randint(0, 2, size=(rows, cols), dtype=int)
        matrix = matrix[np.any(matrix, axis=1)]
        w = int(np.ceil(np.log2(len(matrix) + 1)))
        same, t_old, t_new, nsel, cov = run_once(matrix, w)
        speedup = t_old / t_new if t_new > 0 else float('inf')
        print(f'[random #{test_id}] same={same} selected={nsel} covered={cov} old={t_old:.4f}s new={t_new:.4f}s speedup={speedup:.2f}x')


if __name__ == '__main__':
    main()
