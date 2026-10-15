"""Context-aware support-frontier selection and gate-cost polishing.

Algorithm copied from the frozen 2026-10-12 streaming/hybrid experiment;
Basic and tree score their own outer controls and PREPARE control scope.
No imports or sys.path changes load a historical backend snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np
from numba import njit

from channel_rewrite_cost import (CostSettings, ceil_log2, _row_cost,
                                  components_from_rows, h_length_components,
                                  outer_flag_ccx_count, summarize_row)


def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


@njit(cache=False)
def pair_score(ri, rj, tol):
    n = len(ri)
    keys = np.empty(n, np.complex128)
    counts = np.zeros(n, np.int64)
    used = 0
    shared = 0
    singleton = 0
    for k in range(n):
        x, y = abs(ri[k]) > tol, abs(rj[k]) > tol
        if x and y:
            shared += 1
            z = ri[k] / rj[k]
            key = 0j if abs(z) <= tol else complex(round(z.real, 10), round(z.imag, 10))
            found = -1
            for q in range(used):
                if keys[q] == key:
                    found = q
                    break
            if found < 0:
                keys[used] = key
                counts[used] = 1
                used += 1
            else:
                counts[found] += 1
        elif x or y:
            singleton += 1
    aligned = 0
    multi = 0
    for q in range(used):
        aligned = max(aligned, counts[q])
        if counts[q] >= 2:
            multi += counts[q]
    return (aligned - singleton, aligned, -singleton, shared, multi)


@njit(cache=False)
def local_score(A, i, j, new_i, new_j, tol):
    best = (-10**9, 0, -10**9, 0, 0)
    for p in (i, j):
        for q in range(len(A)):
            if q == p or (p == j and q == i):
                continue
            first, second = min(p, q), max(p, q)
            r1 = new_i if first == i else (new_j if first == j else A[first])
            r2 = new_i if second == i else (new_j if second == j else A[second])
            score = pair_score(r1, r2, tol)
            if score > best:
                best = score
    return best


from hybrid_candidate_kernel import build_kernel
fast_candidates = build_kernel(local_score)


def row_key(row, tol):
    row = row.copy()
    row[np.abs(row) < tol] = 0
    nz = np.flatnonzero(np.abs(row) > tol)
    if len(nz):
        row *= np.exp(-1j * np.angle(row[nz[0]]))
        row[np.abs(row) < tol] = 0
    # As in the source: decimal quantization, row phase/order invariance.
    vals = np.round(row.view(np.float64), 10)
    vals[vals == 0] = 0.0
    return vals.tobytes()


def state_key(keys):
    return hashlib.sha256(b"".join(sorted(keys))).digest()


@dataclass
class State:
    A: np.ndarray
    support: int
    keys: tuple
    steps: tuple = ()


class TopK:
    """Exactly the stable top K unique keys of the stream seen so far."""
    def __init__(self, k):
        self.k = k
        self.heap = []
        self.by_key = {}
        self.offered = 0
        self.replacements = 0
        self.peak = 0

    def competitive(self, rank):
        return len(self.heap) < self.k or rank < self.heap[0][3]

    def offer(self, rank, key, payload):
        self.offered += 1
        if key in self.by_key:
            old = self.by_key[key]
            if rank >= old[3]:
                return False
            self.heap.remove(old)
            heapq.heapify(self.heap)
            del self.by_key[key]
        elif not self.competitive(rank):
            return False
        if len(self.heap) == self.k:
            old = heapq.heappop(self.heap)
            del self.by_key[old[1]]
            self.replacements += 1
        entry = (tuple(-x for x in rank), key, payload, rank)
        heapq.heappush(self.heap, entry)
        self.by_key[key] = entry
        self.peak = max(self.peak, len(self.heap))
        return True

    def sorted(self):
        return sorted(self.heap, key=lambda x: x[3])


def metrics(A, tol, settings):
    return {
        "M": int(np.count_nonzero(np.any(np.abs(A) > tol, axis=1))),
        "support": int(np.count_nonzero(np.abs(A) > tol)),
        "static_cost": h_length_components(A, tol, structure=settings.structure, fstar=settings.fstar),
    }


def verify(A0, final, steps, tol):
    replay = A0.copy()
    maximum_unitarity_error = 0.0
    for i, j, U in steps:
        replay[[i, j]] = U @ replay[[i, j]]
        maximum_unitarity_error = max(maximum_unitarity_error, float(np.linalg.norm(U.conj().T @ U - np.eye(2))))
    gram = A0.conj().T @ A0
    scale = max(float(np.linalg.norm(gram)), 1e-30)
    gram_error = float(np.linalg.norm(final.conj().T @ final - gram) / scale)
    replay_error = float(np.linalg.norm(replay.conj().T @ replay - gram) / scale)
    final_error = float(np.linalg.norm(replay - final) / max(float(np.linalg.norm(final)), 1e-30))
    return {"passed": max(gram_error, replay_error, final_error, maximum_unitarity_error) < 1e-10,
            "gram_relative_error": gram_error, "unthresholded_replay_gram_error": replay_error,
            "replay_final_relative_error": final_error, "maximum_step_unitarity_error": maximum_unitarity_error}, replay

def support_search(A0, out, width=8, max_steps=583, tol=1e-12, neutral=True,
           timeout=1800, collector="heap", rank_mode="support_rank", max_stagnant=32, on_layer=None):
    init = State(A0.copy(), int(np.count_nonzero(np.abs(A0) > tol)), tuple(row_key(r, tol) for r in A0))
    beam = [init]
    if on_layer:
        on_layer(0, beam)
    history = {state_key(init.keys)}
    best = init
    trajectory = []
    t0 = time.perf_counter()
    last_improvement = 0
    reason = "max_steps_reached"
    for layer in range(max_steps):
        top = TopK(width)
        full = {}  # Only used by bounded self-test reference, never benchmark.
        seq = 0
        generated = 0
        accepted = 0
        layer_start = time.perf_counter()
        for parent in beam:
            A = parent.A
            supp = np.count_nonzero(np.abs(A) > tol, axis=1)
            active = np.flatnonzero(supp)
            for pi, i in enumerate(active):
                if time.perf_counter() - t0 > timeout:
                    reason = "timeout"
                    break
                for j in active[pi + 1:]:
                    if not np.any((np.abs(A[i]) > tol) & (np.abs(A[j]) > tol)):
                        continue
                    cutoff = top.heap[0][3][0] if collector == "heap" and len(top.heap) == width else parent.support
                    scores, nis, njs, Us, flags, n_generated = fast_candidates(
                        A, int(i), int(j), tol, neutral, rank_mode == "original", cutoff, parent.support, len(active))
                    generated += int(n_generated)
                    score_len = 7 if rank_mode == "original" else 8
                    candidates = [(tuple(int(x) for x in scores[q, :score_len]), nis[q], njs[q], Us[q], bool(flags[q])) for q in range(len(scores))]
                    improving = sorted((c for c in candidates if not c[4]), key=lambda c: c[0])
                    neutrals = sorted((c for c in candidates if c[4]), key=lambda c: c[0])[:3]
                    for score, ni, nj, U, _ in improving + neutrals:
                        rank = (*score, seq)
                        seq += 1
                        accepted += 1
                        if collector == "heap" and not top.competitive(rank):
                            continue
                        keys = list(parent.keys)
                        keys[i], keys[j] = row_key(ni, tol), row_key(nj, tol)
                        keys = tuple(keys)
                        key = state_key(keys)
                        if key in history:
                            continue
                        # Only competitively ranked candidates allocate a full matrix/path.
                        B = A.copy()
                        B[i], B[j] = ni, nj
                        state = State(B, score[0], keys, parent.steps + ((int(i), int(j), U.copy()),))
                        if collector == "heap":
                            top.offer(rank, key, state)
                        elif key not in full or rank < full[key][3]:
                            full[key] = (None, key, state, rank)
                if reason == "timeout":
                    break
            if reason == "timeout":
                break
        if reason == "timeout":
            break  # Incomplete layer is not committed as an exact top-K layer.
        selected = top.sorted() if collector == "heap" else sorted(full.values(), key=lambda c: c[3])[:width]
        if not selected:
            reason = "frontier_exhausted"
            break
        beam = [item[2] for item in selected]
        if on_layer:
            on_layer(layer + 1, beam)
        history.update(item[1] for item in selected)
        if beam[0].support < best.support:
            best = beam[0]
            last_improvement = layer + 1
        record = {"layer": layer + 1, "best_support": best.support,
                  "best_M": int(np.count_nonzero(np.any(np.abs(best.A) > tol, axis=1))),
                  "frontier_supports": [s.support for s in beam],
                  "frontier_keys": [x[1].hex() for x in selected],
                  "generated_rotations": generated, "eligible_candidates": accepted,
                  "history_size": len(history), "heap_peak_entries": top.peak,
                  "heap_replacements": top.replacements,
                  "layer_time_s": time.perf_counter() - layer_start,
                  "elapsed_s": time.perf_counter() - t0}
        trajectory.append(record)
        if out:
            dump(out / "progress.json", record)
        if layer % 10 == 0 or layer + 1 == max_steps:
            print(json.dumps({k: record[k] for k in ("layer", "best_support", "best_M", "elapsed_s", "history_size")}), flush=True)
        if max_stagnant and layer + 1 - last_improvement >= max_stagnant:
            reason = "stagnation_budget"
            break
    return best, {"stop_reason": reason, "completed_layers": len(trajectory),
                  "accepted_steps": len(best.steps), "max_steps": max_steps,
                  "search_time_s": time.perf_counter() - t0, "trajectory": trajectory}


def cost_tables(m, n, settings):
    # Basic changes every row's control count when M crosses a power of two.
    levels = ceil_log2(m) + 1 if settings.structure == "basic" else 2
    row_table = np.zeros((levels, n + 1, 4), np.int64)
    for branch in range(levels):
        for s in range(1, n + 1):
            p, sel, _ = _row_cost(s, (0, 0, 0), branch, settings.fstar)
            row_table[branch, s, 0] = p + sel
            for q in range(3):
                phases = tuple(int(k == q) for k in range(3))
                row_table[branch, s, q + 1] = _row_cost(s, phases, branch, settings.fstar)[2]
    outer = np.array([7 * ((1 << max(0, (j - 1).bit_length())) - 1)
                      + 7 * outer_flag_ccx_count(j, settings.structure) if j else 0
                      for j in range(m + 1)], dtype=np.int64)
    controls = np.array([ceil_log2(j) if settings.structure == "basic" else int(j > 1)
                         for j in range(m + 1)], dtype=np.int64)
    return row_table, outer, controls


@njit(cache=False)
def row_summary(row, tol, row_table):
    values = row[np.abs(row) > tol]
    s = len(values)
    if not s:
        return 0, 0.0, np.zeros(row_table.shape[0], np.int64)
    alpha2 = np.sum(np.abs(values)) ** 2
    angles = np.angle(values)
    ordered = np.sort(angles)
    extended = np.concatenate((ordered - 2 * np.pi, ordered, ordered + 2 * np.pi))
    counts = (np.searchsorted(extended, angles + 1e-10, side="right")
              - np.searchsorted(extended, angles - 1e-10, side="left"))
    pivot = angles[np.argmax(counts)]
    phases = np.zeros(3, np.int64)
    for angle in angles:
        distance = abs(np.angle(np.exp(1j * (angle - pivot))))
        if abs(distance - np.pi) <= 1e-10:
            phases[0] += 1
        elif abs(distance - np.pi / 2) <= 1e-10:
            phases[1] += 1
        elif distance > 1e-10:
            phases[2] += 1
    costs = np.zeros(row_table.shape[0], np.int64)
    for b in range(row_table.shape[0]):
        costs[b] = row_table[b, s, 0]
        for k in range(3):
            costs[b] += phases[k] * row_table[b, s, k + 1]
    return s, alpha2, costs


@njit(cache=False)
def gate_candidates(A, i, j, tol, neutral, cutoff, parent_cost, parent_support,
               active_count, alpha, row_costs, norm, cost_sums, row_table, outer, controls,
               admit_structural=False):
    n = A.shape[1]
    scores = np.zeros((2 * n, 12), np.float64)
    new_is = np.zeros((2 * n, n), np.complex128)
    new_js = np.zeros((2 * n, n), np.complex128)
    unitaries = np.zeros((2 * n, 2, 2), np.complex128)
    seen = np.zeros((2 * n + 1, 3), np.float64)
    seen[0, 0] = 1.0
    seen_count, used, generated = 1, 0, 1
    old_pair = np.count_nonzero(np.abs(A[i]) > tol) + np.count_nonzero(np.abs(A[j]) > tol)
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
            unchanged = True
            for q in range(n):
                x = a * A[i, q] - np.conj(b) * A[j, q]
                y = b * A[i, q] + a * A[j, q]
                ni[q] = x if abs(x) >= tol else 0j
                nj[q] = y if abs(y) >= tol else 0j
                if abs(ni[q] - A[i, q]) > tol + tol * abs(A[i, q]) or abs(nj[q] - A[j, q]) > tol + tol * abs(A[j, q]):
                    unchanged = False
            if unchanged:
                continue
            si, li, ci = row_summary(ni, tol, row_table)
            sj, lj, cj = row_summary(nj, tol, row_table)
            m = active_count - 2 + int(si > 0) + int(sj > 0)
            branch = controls[m]
            cost = ((norm - alpha[i] - alpha[j] + li + lj)
                    * (outer[m] + cost_sums[branch] - row_costs[i, branch] - row_costs[j, branch]
                       + ci[branch] + cj[branch]))
            # Conservative rounding margin. Python recomputes exact competitive costs.
            if cost > parent_cost + 1e-12 * abs(parent_cost) or cost > cutoff + 1e-12 * abs(cutoff):
                continue
            eq = abs(cost - parent_cost) <= 1e-12 * max(abs(cost), abs(parent_cost))
            structural = admit_structural and (si + sj < old_pair or m < active_count)
            if eq and not neutral and not structural:
                continue
            after = local_score(A, i, j, ni, nj, tol) if neutral else (0, 0, 0, 0, 0)
            if eq and not structural:
                if not before_ready:
                    before = local_score(A, i, j, A[i], A[j], tol)
                    before_ready = True
                if after <= before:
                    continue
            scores[used, 0] = cost
            # Structural progress on a cost plateau is an improvement, so it
            # also bypasses the three-neutral-candidates-per-pair cap.
            scores[used, 1] = int(eq and not structural)
            scores[used, 2] = parent_support - old_pair + si + sj
            scores[used, 3] = max(si, sj)
            scores[used, 4] = si + sj
            for q in range(5):
                scores[used, 5 + q] = -after[q]
            scores[used, 10], scores[used, 11] = i, j
            new_is[used], new_js[used] = ni, nj
            unitaries[used, 0, 0], unitaries[used, 0, 1] = a, -np.conj(b)
            unitaries[used, 1, 0], unitaries[used, 1, 1] = b, a
            used += 1
    return scores[:used], new_is[:used], new_js[:used], unitaries[:used], generated


def row_data(A, tol, settings, levels):
    rows = tuple(summarize_row(r, tol) for r in A)
    alpha = np.array([r.alpha_squared for r in rows])
    costs = np.array([[sum(_row_cost(r.support, r.phases, b, settings.fstar)) if r.support else 0
                       for b in range(levels)] for r in rows], dtype=np.int64)
    return rows, alpha, costs


@dataclass
class GateState:
    A: np.ndarray
    rows: tuple
    cost: float
    support: int
    keys: tuple
    steps: tuple = ()


def gate_search(A0, out=None, width=8, max_steps=583, tol=1e-12, neutral=False,
           timeout=600, collector="heap", mode="legacy", settings=None):
    settings = CostSettings() if settings is None else settings
    row_table, outer, controls = cost_tables(*A0.shape, settings)
    levels = row_table.shape[0]
    rows, _, _ = row_data(A0, tol, settings, levels)
    init_cost = components_from_rows(rows, settings)
    admit = mode in ("admit_support", "tolerant", "support_order")
    # A fixed cost unit yields a transitive ordering. Pairwise isclose is not
    # a comparator. This bucket is for selection only; the actual H is saved.
    cost_unit = max(init_cost["total"] * 1e-12, 1e-30)
    init = GateState(A0.copy(), rows, init_cost["total"], init_cost["total_support"], tuple(row_key(r, tol) for r in A0))
    beam, history, best = [init], {state_key(init.keys)}, init
    trajectory = []
    t0 = time.perf_counter()
    reason = "max_steps_reached"
    for layer in range(max_steps):
        top, full = TopK(width), {}
        seq = generated = eligible = exact_evaluations = support_increasing = 0
        layer_start = time.perf_counter()
        for parent in beam:
            A, rows = parent.A, parent.rows
            alpha = np.array([r.alpha_squared for r in rows])
            row_costs = np.array([[sum(_row_cost(r.support, r.phases, b, settings.fstar)) if r.support else 0
                                   for b in range(levels)] for r in rows], dtype=np.int64)
            sums = row_costs.sum(axis=0)
            norm = math.fsum(alpha)
            active = np.array([k for k, row in enumerate(rows) if row.support])
            masks = np.abs(A) > tol
            for pi, i in enumerate(active):
                if time.perf_counter() - t0 > timeout:
                    reason = "timeout"
                    break
                for j in active[pi + 1:]:
                    if not np.any(masks[i] & masks[j]):
                        continue
                    cutoff = math.inf
                    if collector == "heap" and len(top.heap) == width and mode != "support_order":
                        cutoff = top.heap[0][3][0]
                        if mode == "tolerant":
                            cutoff = (cutoff + 0.500001) * cost_unit
                    scores, nis, njs, Us, ng = gate_candidates(A, i, j, tol, neutral, cutoff, parent.cost,
                        parent.support, len(active), alpha, row_costs, norm, sums, row_table, outer, controls, admit)
                    generated += int(ng)
                    ranked = []
                    for q, raw_score in enumerate(scores):
                        updated = list(rows)
                        updated[i], updated[j] = summarize_row(nis[q], tol), summarize_row(njs[q], tol)
                        exact = components_from_rows(updated, settings)["total"]
                        exact_evaluations += 1
                        assert math.isclose(exact, raw_score[0], rel_tol=2e-12, abs_tol=1e-12)
                        eq = math.isclose(exact, parent.cost, rel_tol=1e-12, abs_tol=0.0)
                        new_m = len(active) - 2 + int(updated[i].support > 0) + int(updated[j].support > 0)
                        structural = admit and (int(raw_score[2]) < parent.support or new_m < len(active))
                        if (exact > parent.cost and not eq) or (eq and not neutral and not structural):
                            continue
                        score = (exact, int(eq and not structural), *(int(x) for x in raw_score[2:]))
                        selection = score
                        if mode == "tolerant":
                            selection = (round(exact / cost_unit), score[2], new_m, *score[1:])
                        elif mode == "support_order":
                            support_equal = score[2] == parent.support
                            if score[2] > parent.support:
                                continue
                            # Match support's admission and ranking on its
                            # own plateaus, in addition to H nonincrease.
                            if support_equal:
                                after = local_score(A, i, j, nis[q], njs[q], tol)
                                before = local_score(A, i, j, A[i], A[j], tol)
                                if not neutral or after <= before:
                                    continue
                            else:
                                after = (0, 0, 0, 0, 0)
                            selection = (score[2], new_m, int(support_equal), *(-v for v in after))
                        ranked.append((score, q, tuple(updated), selection))
                    improved = sorted((c for c in ranked if not c[0][1]), key=lambda c: c[3])
                    neutrals = sorted((c for c in ranked if c[0][1]), key=lambda c: c[3])[:3]
                    for score, q, updated, selection in improved + neutrals:
                        rank = (*selection, seq)
                        seq += 1
                        eligible += 1
                        support_increasing += int(score[2] > parent.support)
                        if collector == "heap" and not top.competitive(rank):
                            continue
                        keys = list(parent.keys)
                        keys[i], keys[j] = row_key(nis[q], tol), row_key(njs[q], tol)
                        keys = tuple(keys)
                        key = state_key(keys)
                        if key in history:
                            continue
                        B = A.copy()
                        B[i], B[j] = nis[q], njs[q]
                        state = GateState(B, updated, score[0], score[2], keys,
                                      parent.steps + ((int(i), int(j), Us[q].copy()),))
                        if collector == "heap":
                            top.offer(rank, key, state)
                        elif key not in full or rank < full[key][3]:
                            full[key] = (None, key, state, rank)
                if reason == "timeout":
                    break
            if reason == "timeout":
                break
        if reason == "timeout":
            break
        selected = top.sorted() if collector == "heap" else sorted(full.values(), key=lambda c: c[3])[:width]
        if not selected:
            reason = "frontier_exhausted"
            break
        beam = [item[2] for item in selected]
        history.update(item[1] for item in selected)
        # Saved best agrees with the production search on equal-cost support
        # progress. This has no effect on which states are expanded.
        for state in beam:
            equal_best = math.isclose(state.cost, best.cost, rel_tol=1e-12, abs_tol=0)
            if ((state.cost < best.cost and not equal_best)
                    or (equal_best and state.support < best.support)):
                best = state
        record = {"layer": layer + 1, "best_cost": best.cost, "best_support": best.support,
                  "best_M": sum(bool(r.support) for r in best.rows),
                  "frontier_costs": [s.cost for s in beam], "frontier_supports": [s.support for s in beam],
                  "frontier_keys": [x[1].hex() for x in selected], "generated_rotations": generated,
                  "eligible_after_heap_prefilter": eligible, "support_increasing_eligible": support_increasing,
                  "exact_evaluations": exact_evaluations, "history_size": len(history),
                  "heap_peak_entries": top.peak, "heap_replacements": top.replacements,
                  "frontier_kraus_counts": [sum(r.support > 0 for r in s.rows) for s in beam],
                  "layer_time_s": time.perf_counter() - layer_start, "elapsed_s": time.perf_counter() - t0}
        trajectory.append(record)
        if out:
            dump(out / "progress.json", record)
        if layer % 10 == 0 or layer + 1 == max_steps:
            print({k: record[k] for k in ("layer", "best_cost", "best_support", "best_M", "elapsed_s")}, flush=True)
    return best, {"stop_reason": reason, "completed_layers": len(trajectory), "accepted_steps": len(best.steps),
                  "max_steps": max_steps, "search_time_s": time.perf_counter() - t0, "trajectory": trajectory}



def hybrid_search(A0, *, contexts=("basic", "tree"), out=None, width=8,
                  max_steps=None, tol=1e-12, timeout=300, max_stagnant=32):
    """Run the frozen two-seed policy with an explicit cost context per output.

    The support beam is independent of the lowering context. One shared search
    observes the same committed frontiers for every requested context; each
    context then polishes its own minimum-H archive and support-final seed.
    Original seeds remain candidates, even when polishing exhausts its budget.
    """
    if not contexts or len(set(contexts)) != len(contexts):
        raise ValueError("Choose distinct basic/tree contexts.")
    settings = {name: CostSettings(name, name != "basic") for name in contexts}
    if set(settings) - {"basic", "tree"}:
        raise ValueError("Hybrid ablation contexts are basic and tree.")
    if max_steps is None:
        max_steps = max(256, int(np.count_nonzero(np.abs(A0) > tol)) + 1)
    archives = {name: {"state": None, "H": math.inf, "layer": 0,
                       "evaluated": 0, "time_s": 0.0} for name in contexts}

    def observe(layer, beam):
        for name, context in settings.items():
            started = time.perf_counter()
            saved = archives[name]
            for state in beam:
                cost = metrics(state.A, tol, context)["static_cost"]
                saved["evaluated"] += 1
                rank = (cost["total"], state.support, cost["kraus_count"])
                previous = saved["state"]
                previous_rank = (saved["H"], previous.support,
                                 int(np.count_nonzero(np.any(np.abs(previous.A) > tol, axis=1)))) if previous else None
                if previous_rank is None or rank < previous_rank:
                    saved.update(state=state, H=cost["total"], layer=layer)
            saved["time_s"] += time.perf_counter() - started

    support_out = out / "support" if out else None
    if support_out:
        support_out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pure, support_term = support_search(
        A0, support_out, width=width, max_steps=max_steps, tol=tol,
        neutral=True, timeout=timeout, rank_mode="support_rank",
        max_stagnant=max_stagnant, on_layer=observe)
    support_s = time.perf_counter() - started
    result = {"support_state": pure, "support_termination": support_term,
              "support_wall_s": support_s, "contexts": {}}
    for name, context in settings.items():
        saved = archives[name]
        selected = saved["state"]
        seeds = [selected]
        if state_key(pure.keys) != state_key(selected.keys):
            seeds.append(pure)
        candidates = [selected, pure]
        polishes = []
        for index, seed in enumerate(seeds):
            polish_out = out / name / ("polish_" + str(index)) if out else None
            if polish_out:
                polish_out.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            state, termination = gate_search(
                seed.A, polish_out, width=width, max_steps=max_steps, tol=tol,
                neutral=True, timeout=timeout, mode="admit_support", settings=context)
            wall_s = time.perf_counter() - started
            combined = State(state.A, state.support, state.keys, seed.steps + state.steps)
            check, _ = verify(A0, combined.A, combined.steps, tol)
            if not check["passed"]:
                raise AssertionError(check)
            polishes.append({"seed_H": metrics(seed.A, tol, context)["static_cost"]["total"],
                             "termination": termination, "wall_s": wall_s, "verification": check})
            candidates.append(combined)
        best = min(candidates, key=lambda s: (metrics(s.A, tol, context)["static_cost"]["total"], s.support))
        check, replay = verify(A0, best.A, best.steps, tol)
        if not check["passed"]:
            raise AssertionError(check)
        complete = (support_term["stop_reason"] == "frontier_exhausted"
                    and all(p["termination"]["stop_reason"] == "frontier_exhausted" for p in polishes))
        final_metrics = metrics(best.A, tol, context)
        initial_H = metrics(A0, tol, context)["static_cost"]["total"]
        if final_metrics["static_cost"]["total"] > initial_H * (1 + 1e-12):
            raise AssertionError("Hybrid lost its initial-state fallback.")
        # Observer overhead for other contexts is excluded from this context's
        # effective compilation time; the shared wall time is reported too.
        observer_other_s = sum(a["time_s"] for n, a in archives.items() if n != name)
        record = {
            "settings": context.metadata(), "metrics": final_metrics,
            "archive": {key: value for key, value in saved.items() if key != "state"},
            "support_termination": support_term, "polishes": polishes,
            "termination": {"stop_reason": "frontier_exhausted" if complete else "incomplete_stage",
                            "support_stop": support_term["stop_reason"],
                            "polish_stops": [p["termination"]["stop_reason"] for p in polishes],
                            "accepted_steps": len(best.steps)},
            "status": "ok" if complete else "rewrite_incomplete",
            "support_shared_wall_s": support_s, "other_context_observer_s": observer_other_s,
            "rewrite_time_s": support_s - observer_other_s + sum(p["wall_s"] for p in polishes),
            "verification": check,
        }
        result["contexts"][name] = {"state": best, "replay": replay, "record": record}
    return result


def warmup():
    """Compile arithmetic kernels on independent data outside search timings."""
    matrix = np.array([[1, 1j], [1, -1j]], dtype=complex)
    started = time.perf_counter()
    support_search(matrix, None, max_steps=1)
    gate_search(matrix, None, max_steps=1, mode="admit_support", neutral=True,
                settings=CostSettings("basic", False))
    return time.perf_counter() - started
