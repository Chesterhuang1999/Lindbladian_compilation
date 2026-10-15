"""Isolated streaming beam using the frozen static gate-cost objective.

All rotations in a layer are scanned. A bounded heap retains the best eight;
only selected state digests enter history. Pauli-support increases are allowed.
Compiled pair arithmetic is a prefilter; competitive scores are recalculated
with the original Python cost functions before heap insertion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import resource
import sys
import time

# The five paper fixtures use the maintained entrypoint, not this legacy CLI.
if __name__ == "__main__":
    import runpy as _retirement_runpy
    _retirement_runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "_fixture_entrypoints.py"),
        init_globals={"default_case": "Thermal2_HO"}, run_name="__main__")

import numpy as np
from numba import njit

from fast_probe import (ChannelIR, SNAPSHOT, TopK, dump, local_score, metrics,
                        row_key, state_key, verify)
from channel_rewrite_cost import (CostSettings, RowSummary, _row_cost,
                                  components_from_rows, h_length_components,
                                  outer_flag_ccx_count, summarize_row)

SETTINGS = CostSettings("tree", True)


def tables(m, n):
    row_table = np.zeros((2, n + 1, 4), np.int64)
    for branch in (0, 1):
        for s in range(1, n + 1):
            p, sel, _ = _row_cost(s, (0, 0, 0), branch, True)
            row_table[branch, s, 0] = p + sel
            for q in range(3):
                phases = tuple(int(k == q) for k in range(3))
                row_table[branch, s, q + 1] = _row_cost(s, phases, branch, True)[2]
    outer = np.array([7 * ((1 << max(0, (j - 1).bit_length())) - 1)
                      + 7 * outer_flag_ccx_count(j, "tree") if j else 0
                      for j in range(m + 1)], dtype=np.int64)
    return row_table, outer


@njit(cache=False)
def row_summary(row, tol, row_table):
    values = row[np.abs(row) > tol]
    s = len(values)
    if not s:
        return 0, 0.0, 0, 0
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
    costs = np.zeros(2, np.int64)
    for b in range(2):
        costs[b] = row_table[b, s, 0]
        for k in range(3):
            costs[b] += phases[k] * row_table[b, s, k + 1]
    return s, alpha2, costs[0], costs[1]


@njit(cache=False)
def candidates(A, i, j, tol, neutral, cutoff, parent_cost, parent_support,
               active_count, alpha, row_costs, norm, cost_sums, row_table, outer):
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
            si, li, ci0, ci1 = row_summary(ni, tol, row_table)
            sj, lj, cj0, cj1 = row_summary(nj, tol, row_table)
            m = active_count - 2 + int(si > 0) + int(sj > 0)
            branch = int(m > 1)
            cost = ((norm - alpha[i] - alpha[j] + li + lj)
                    * (outer[m] + cost_sums[branch] - row_costs[i, branch] - row_costs[j, branch]
                       + (ci1 + cj1 if branch else ci0 + cj0)))
            # Conservative rounding margin. Python recomputes exact competitive costs.
            if cost > parent_cost + 1e-12 * abs(parent_cost) or cost > cutoff + 1e-12 * abs(cutoff):
                continue
            eq = abs(cost - parent_cost) <= 1e-12 * max(abs(cost), abs(parent_cost))
            if eq and not neutral:
                continue
            after = local_score(A, i, j, ni, nj, tol) if neutral else (0, 0, 0, 0, 0)
            if eq:
                if not before_ready:
                    before = local_score(A, i, j, A[i], A[j], tol)
                    before_ready = True
                if after <= before:
                    continue
            scores[used, 0] = cost
            scores[used, 1] = int(eq)
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


def row_data(A, tol):
    rows = tuple(summarize_row(r, tol) for r in A)
    alpha = np.array([r.alpha_squared for r in rows])
    costs = np.array([[sum(_row_cost(r.support, r.phases, b, True)) if r.support else 0
                       for b in (0, 1)] for r in rows], dtype=np.int64)
    return rows, alpha, costs


@dataclass
class State:
    A: np.ndarray
    rows: tuple
    cost: float
    support: int
    keys: tuple
    steps: tuple = ()


def search(A0, out=None, width=8, max_steps=583, tol=1e-12, neutral=False,
           timeout=600, collector="heap"):
    row_table, outer = tables(*A0.shape)
    rows, _, _ = row_data(A0, tol)
    init_cost = components_from_rows(rows, SETTINGS)
    init = State(A0.copy(), rows, init_cost["total"], init_cost["total_support"], tuple(row_key(r, tol) for r in A0))
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
            row_costs = np.array([[sum(_row_cost(r.support, r.phases, b, True)) if r.support else 0
                                   for b in (0, 1)] for r in rows], dtype=np.int64)
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
                    cutoff = top.heap[0][3][0] if collector == "heap" and len(top.heap) == width else math.inf
                    scores, nis, njs, Us, ng = candidates(A, i, j, tol, neutral, cutoff, parent.cost,
                        parent.support, len(active), alpha, row_costs, norm, sums, row_table, outer)
                    generated += int(ng)
                    ranked = []
                    for q, raw_score in enumerate(scores):
                        updated = list(rows)
                        updated[i], updated[j] = summarize_row(nis[q], tol), summarize_row(njs[q], tol)
                        exact = components_from_rows(updated, SETTINGS)["total"]
                        exact_evaluations += 1
                        assert math.isclose(exact, raw_score[0], rel_tol=2e-12, abs_tol=1e-12)
                        eq = math.isclose(exact, parent.cost, rel_tol=1e-12, abs_tol=0.0)
                        if (exact > parent.cost and not eq) or (eq and not neutral):
                            continue
                        score = (exact, int(eq), *(int(x) for x in raw_score[2:]))
                        ranked.append((score, q, tuple(updated)))
                    improved = sorted((c for c in ranked if not c[0][1]), key=lambda c: c[0])
                    neutrals = sorted((c for c in ranked if c[0][1]), key=lambda c: c[0])[:3]
                    for score, q, updated in improved + neutrals:
                        rank = (*score, seq)
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
                        state = State(B, updated, score[0], score[2], keys,
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
        if beam[0].cost < best.cost:
            best = beam[0]
        record = {"layer": layer + 1, "best_cost": best.cost, "best_support": best.support,
                  "best_M": sum(bool(r.support) for r in best.rows),
                  "frontier_costs": [s.cost for s in beam], "frontier_supports": [s.support for s in beam],
                  "frontier_keys": [x[1].hex() for x in selected], "generated_rotations": generated,
                  "eligible_after_heap_prefilter": eligible, "support_increasing_eligible": support_increasing,
                  "exact_evaluations": exact_evaluations, "history_size": len(history),
                  "heap_peak_entries": top.peak, "heap_replacements": top.replacements,
                  "layer_time_s": time.perf_counter() - layer_start, "elapsed_s": time.perf_counter() - t0}
        trajectory.append(record)
        if out:
            dump(out / "progress.json", record)
        if layer % 10 == 0 or layer + 1 == max_steps:
            print({k: record[k] for k in ("layer", "best_cost", "best_support", "best_M", "elapsed_s")}, flush=True)
    return best, {"stop_reason": reason, "completed_layers": len(trajectory), "accepted_steps": len(best.steps),
                  "max_steps": max_steps, "search_time_s": time.perf_counter() - t0, "trajectory": trajectory}


def self_test(out, thermal_input):
    from channelIR_gatecost_rewrite_search import _candidate_gatecost_rotations
    rng = np.random.default_rng(8715)
    rows_checked = candidates_checked = support_increases = 0
    max_error = 0.0
    matrices = [np.array([[1, 1j, -1, -1j], [1, -1j, -1, 1j]], complex),
                np.array([[1, 2], [2, 4]], complex)]
    thermal = np.load(thermal_input)["A_initial"]
    matrices += [thermal[:4], thermal[[0, 4, 12, 20, 71, 72]]]
    for _ in range(35):
        A = rng.normal(size=(5, 9)) + 1j * rng.normal(size=(5, 9))
        A[rng.random(A.shape) < .5] = 0
        matrices.append(A)
    for A in matrices:
        rt, outer = tables(*A.shape)
        rows, alpha, rc = row_data(A, 1e-12)
        for r, reference in zip(A, rows):
            s, a2, c0, c1 = row_summary(r, 1e-12, rt)
            assert s == reference.support and math.isclose(a2, reference.alpha_squared, rel_tol=1e-13, abs_tol=1e-14)
            assert [c0, c1] == [sum(_row_cost(s, reference.phases, b, True)) if s else 0 for b in (0, 1)]
            rows_checked += 1
        parent = components_from_rows(rows, SETTINGS)
        original = _candidate_gatecost_rotations(ChannelIR, A, np.array([r.support for r in rows]),
            parent["total"], 1e-12, False, 3, True, SETTINGS)
        reference = {(c["pair"], ChannelIR._unitary_key(c["U"], 1e-12)): c for c in original}
        got = {}
        active = [k for k, r in enumerate(rows) if r.support]
        for pi, i in enumerate(active):
            for j in active[pi + 1:]:
                if not np.any((np.abs(A[i]) > 1e-12) & (np.abs(A[j]) > 1e-12)):
                    continue
                scores, nis, njs, Us, _ = candidates(A, i, j, 1e-12, False, math.inf, parent["total"],
                    parent["total_support"], len(active), alpha, rc, math.fsum(alpha), rc.sum(axis=0), rt, outer)
                for q, score in enumerate(scores):
                    B = A.copy()
                    B[i], B[j] = nis[q], njs[q]
                    exact = h_length_components(B, structure="tree", fstar=True)["total"]
                    key = ((i, j), ChannelIR._unitary_key(Us[q], 1e-12))
                    assert key in reference
                    max_error = max(max_error, abs(exact - score[0]) / max(abs(exact), 1e-30))
                    assert math.isclose(exact, reference[key]["cost"], rel_tol=2e-12, abs_tol=1e-12)
                    assert np.allclose(B, reference[key]["A"], rtol=1e-12, atol=1e-12)
                    got[key] = True
                    support_increases += int(score[2] > parent["total_support"])
                    candidates_checked += 1
        assert set(reference) == set(got), (len(reference), len(got))
    assert support_increases > 0
    for A in (matrices[0], matrices[2], matrices[5]):
        h, ht = search(A, max_steps=4, collector="heap")
        f, ft = search(A, max_steps=4, collector="full")
        assert [r["frontier_keys"] for r in ht["trajectory"]] == [r["frontier_keys"] for r in ft["trajectory"]]
        assert np.allclose(h.A, f.A)
    dump(out / "self_test.json", {"passed": True, "rows_checked": rows_checked,
         "candidate_set_matches_frozen_source_matrices": len(matrices), "candidates_checked": candidates_checked,
         "support_increasing_candidates_checked": support_increases, "maximum_relative_score_error": max_error,
         "heap_equals_full_sort_matrices": 3})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--array", default="A_initial")
    parser.add_argument("--case", default="Thermal2_HO")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=583)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--rank", default="gatecost")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(out, args.input)
        return
    start = time.perf_counter()
    source = np.load(args.input)
    A = np.asarray(source[args.array], dtype=complex)
    rt, outer = tables(*A.shape)
    rows, alpha, rc = row_data(A, 1e-12)
    cost = components_from_rows(rows, SETTINGS)
    jit_start = time.perf_counter()
    candidates(A, 0, 1, 1e-12, not args.strict, math.inf, cost["total"], cost["total_support"],
               cost["kraus_count"], alpha, rc, math.fsum(alpha), rc.sum(axis=0), rt, outer)
    jit_time = time.perf_counter() - jit_start
    manifest = {"case": args.case, "pid": os.getpid(), "input": str(Path(args.input).resolve()),
                "input_file_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                "array": args.array, "shape": list(A.shape), "python": sys.executable,
                "beam_width": 8, "max_steps": args.max_steps, "neutral": not args.strict,
                "rank_mode": "static_gatecost", "cost_settings": SETTINGS.metadata(),
                "history": "selected states only; SHA256 of canonical quantized rows",
                "max_stagnant_layers": None, "timeout_s": args.timeout,
                "support_increases_allowed": True,
                "scope": "rewrite only; independent prototype; no source modifications",
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), Path(__file__).with_name("fast_probe.py"), SNAPSHOT / "channel_IR.py",
                     SNAPSHOT / "channel_rewrite_cost.py", SNAPSHOT / "channelIR_gatecost_rewrite_search.py"]}}
    dump(out / "manifest.json", manifest)
    best, termination = search(A, out, max_steps=args.max_steps, neutral=not args.strict, timeout=args.timeout)
    verification, replay = verify(A, best.A, best.steps, 1e-12)
    final = metrics(best.A, 1e-12)
    assert math.isclose(final["static_cost"]["total"], best.cost, rel_tol=1e-13, abs_tol=1e-12)
    result = {"case": args.case, "status": "ok" if verification["passed"] and termination["stop_reason"] == "frontier_exhausted" else "incomplete",
              "initial": metrics(A, 1e-12), "final": final, "termination": termination,
              "verification": verification, "jit_setup_s": jit_time, "total_time_s": time.perf_counter() - start,
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)}
    np.savez_compressed(out / "coefficients.npz", A_initial=A, A_final=best.A, A_replayed=replay,
                        labels=source["labels"] if "labels" in source.files else np.arange(A.shape[1]))
    dump(out / "rewrite_steps.json", [{"pair": [i, j], "U_real": U.real.tolist(), "U_imag": U.imag.tolist()} for i, j, U in best.steps])
    dump(out / "results.json", result)
    print({k: v for k, v in result.items() if k != "termination"}, flush=True)


if __name__ == "__main__":
    main()
