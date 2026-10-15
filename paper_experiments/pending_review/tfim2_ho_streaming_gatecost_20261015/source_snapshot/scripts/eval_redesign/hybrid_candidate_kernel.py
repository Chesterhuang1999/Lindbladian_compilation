"""Compiled per-pair arithmetic for the standalone streaming-beam experiment."""
import numpy as np
from numba import njit


def build_kernel(local_score):
    @njit(cache=False)
    def candidates(A, i, j, tol, neutral, original_rank, cutoff, parent_support, active_count):
        n = A.shape[1]
        scores = np.zeros((2 * n, 8), np.int64)
        new_is = np.zeros((2 * n, n), np.complex128)
        new_js = np.zeros((2 * n, n), np.complex128)
        unitaries = np.zeros((2 * n, 2, 2), np.complex128)
        neutral_flags = np.zeros(2 * n, np.bool_)
        seen = np.zeros((2 * n + 1, 3), np.float64)
        seen[0, 0] = 1.0  # Identity is already seen.
        seen_count = 1
        used = 0
        generated = 1
        old_pair = 0
        for k in range(n):
            old_pair += int(abs(A[i, k]) > tol) + int(abs(A[j, k]) > tol)
        before_ready = False
        before = (0, 0, 0, 0, 0)
        for k in range(n):
            for direction in range(2):
                ai, aj = A[i, k], A[j, k]
                if (direction == 0 and abs(aj) <= tol) or (direction == 1 and abs(ai) <= tol):
                    continue
                ratio = ai / aj if direction == 0 else aj / ai
                a = 1.0 / np.sqrt(1.0 + abs(ratio) ** 2)
                b = a * np.conj(ratio) if direction == 0 else -a * ratio
                ua, ur, ui = round(a, 10), round(b.real, 10), round(b.imag, 10)
                duplicate = False
                for q in range(seen_count):
                    if seen[q, 0] == ua and seen[q, 1] == ur and seen[q, 2] == ui:
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen[seen_count, 0], seen[seen_count, 1], seen[seen_count, 2] = ua, ur, ui
                seen_count += 1
                generated += 1
                ni, nj = np.empty(n, np.complex128), np.empty(n, np.complex128)
                si, sj = 0, 0
                unchanged = True
                for q in range(n):
                    x = a * A[i, q] - np.conj(b) * A[j, q]
                    y = b * A[i, q] + a * A[j, q]
                    ni[q] = x if abs(x) >= tol else 0j
                    nj[q] = y if abs(y) >= tol else 0j
                    si += int(abs(ni[q]) > tol)
                    sj += int(abs(nj[q]) > tol)
                    if abs(ni[q] - A[i, q]) > tol + tol * abs(A[i, q]) or abs(nj[q] - A[j, q]) > tol + tol * abs(A[j, q]):
                        unchanged = False
                if unchanged:
                    continue
                S = parent_support - old_pair + si + sj
                eq = S == parent_support
                if S > parent_support or (eq and not neutral) or S > cutoff:
                    continue
                after = local_score(A, i, j, ni, nj, tol) if original_rank or eq else (0, 0, 0, 0, 0)
                if eq:
                    if not before_ready:
                        before = local_score(A, i, j, A[i], A[j], tol)
                        before_ready = True
                    if after <= before:
                        continue
                scores[used, 0] = S
                offset = 2
                if original_rank:
                    scores[used, 1] = int(eq)
                else:
                    scores[used, 1] = active_count - 2 + int(si > 0) + int(sj > 0)
                    scores[used, 2] = int(eq)
                    offset = 3
                for q in range(5):
                    scores[used, offset + q] = -after[q]
                new_is[used], new_js[used] = ni, nj
                unitaries[used, 0, 0] = a
                unitaries[used, 0, 1] = -np.conj(b)
                unitaries[used, 1, 0] = b
                unitaries[used, 1, 1] = a
                neutral_flags[used] = eq
                used += 1
        return scores[:used], new_is[:used], new_js[:used], unitaries[:used], neutral_flags[:used], generated
    return candidates
