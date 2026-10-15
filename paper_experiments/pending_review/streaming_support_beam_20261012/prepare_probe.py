"""Standalone rewrite experiment. Does not patch/import-replace production search.

Scan every pair/rotation in a layer, retain eight states using a worst-first
heap. Deterministic sequence IDs break equal-rank ties. Only selected states
enter history. Per-pair candidate storage is bounded by twice the column count.
Numba accelerates the existing local cancellation-potential calculation only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import heapq
import json
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

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / "paper_experiments/pending_review/thermal2_ho_remote_20261011/source_snapshot/src"
sys.path.insert(0, str(SNAPSHOT))
from channel_IR import channel as ChannelIR
from channel_rewrite_cost import h_length_components


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


from candidate_kernel import build_kernel
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


def metrics(A, tol):
    return {
        "M": int(np.count_nonzero(np.any(np.abs(A) > tol, axis=1))),
        "support": int(np.count_nonzero(np.abs(A) > tol)),
        "static_cost": h_length_components(A, tol, structure="tree", fstar=True),
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


def search(A0, out, width=8, max_steps=583, tol=1e-12, neutral=True,
           timeout=1800, collector="heap", rank_mode="original", max_stagnant=32):
    init = State(A0.copy(), int(np.count_nonzero(np.abs(A0) > tol)), tuple(row_key(r, tol) for r in A0))
    beam = [init]
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
            padding = sum(1 << (int(v)-1).bit_length() for v in supp if v)
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
                    if rank_mode == "support_prepare":
                        revised = []
                        for score, ni, nj, U, flag in candidates:
                            si = int(np.count_nonzero(np.abs(ni) > tol))
                            sj = int(np.count_nonzero(np.abs(nj) > tol))
                            pad = padding - (1 << (int(supp[i])-1).bit_length()) - (1 << (int(supp[j])-1).bit_length())
                            pad += (1 << (si-1).bit_length()) if si else 0
                            pad += (1 << (sj-1).bit_length()) if sj else 0
                            revised.append(((*score[:2], pad, *score[2:]), ni, nj, U, flag))
                        candidates = revised
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


def self_test(out):
    rng = np.random.default_rng(481)
    tests = {}
    for count in (9, 100, 1000):
        stream = [(tuple(map(int, rng.integers(0, 8, 3))) + (i,), str(rng.integers(0, 30)), i) for i in range(count)]
        stream.append(((-1, 0, 0, count), "late_winner", count))
        top = TopK(8)
        full = {}
        for rank, key, payload in stream:
            top.offer(rank, key, payload)
            if key not in full or rank < full[key][0]:
                full[key] = (rank, key, payload)
        want = sorted(full.values())[:8]
        got = [(x[3], x[1], x[2]) for x in top.sorted()]
        assert want == got and got[0][1] == "late_winner"
    tests["heap_equals_full_sort_including_late_winner_and_duplicates"] = True
    for _ in range(100):
        A = rng.normal(size=(5, 6)) + 1j * rng.normal(size=(5, 6))
        A[rng.random(A.shape) < 0.35] = 0
        assert local_score(A, 1, 3, A[1], A[3], 1e-12) == ChannelIR._local_future_potential(A, (1, 3), 1e-12)["score"]
    tests["compiled_potential_matches_source_100_matrices"] = True
    A = np.array([[1, 1, 0, 1], [1, -1, 0, 1], [0, 1, 1, 0], [0, 1, -1, 0]], complex)
    heap, h = search(A, None, max_steps=4, timeout=60, collector="heap")
    full, f = search(A, None, max_steps=4, timeout=60, collector="full")
    assert [r["frontier_keys"] for r in h["trajectory"]] == [r["frontier_keys"] for r in f["trajectory"]]
    assert np.allclose(heap.A, full.A)
    tests["multi_layer_heap_frontiers_equal_full_sort"] = True
    original = ChannelIR([])._beam_search(A, [str(i) for i in range(4)], 4, int(np.count_nonzero(A)), 8, 4, 1e-12, False, False)
    tests["small_source_support"] = int(original["final_support"])
    tests["small_prototype_support"] = heap.support
    assert heap.support == original["final_support"]
    dump(out / "self_test.json", tests)
    print(json.dumps(tests), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--array", default="A_initial")
    parser.add_argument("--case", default="Thermal2_HO")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=583)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--rank", choices=("original", "support_rank", "support_prepare"), default="original")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(out)
        return
    start = time.perf_counter()
    source = np.load(args.input)
    A = np.asarray(source[args.array], dtype=complex)
    # Separate JIT setup from search time; report both.
    jit_start = time.perf_counter()
    local_score(A, 0, 1, A[0], A[1], 1e-12)
    fast_candidates(A, 0, 1, 1e-12, not args.strict, args.rank == "original", int(np.count_nonzero(A)), int(np.count_nonzero(A)), len(A))
    jit_time = time.perf_counter() - jit_start
    manifest = {"case": args.case, "pid": os.getpid(), "input": str(Path(args.input).resolve()),
                "input_file_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                "array": args.array, "shape": list(A.shape), "python": sys.executable,
                "beam_width": 8, "max_steps": args.max_steps, "neutral": not args.strict,
                "rank_mode": args.rank, "history": "selected states only; SHA256 of canonical quantized rows",
                "max_stagnant_layers": 32, "timeout_s": args.timeout,
                "scope": "rewrite only; independent prototype; no source modifications",
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), SNAPSHOT / "channel_IR.py", SNAPSHOT / "channel_rewrite_cost.py", Path(__file__).with_name("candidate_kernel.py")]}}
    dump(out / "manifest.json", manifest)
    best, termination = search(A, out, max_steps=args.max_steps, neutral=not args.strict,
                               timeout=args.timeout, rank_mode=args.rank)
    verification, replay = verify(A, best.A, best.steps, 1e-12)
    result = {"case": args.case, "status": "ok" if verification["passed"] and termination["stop_reason"] == "frontier_exhausted" else "incomplete",
              "initial": metrics(A, 1e-12), "final": metrics(best.A, 1e-12),
              "termination": termination, "verification": verification, "jit_setup_s": jit_time,
              "total_time_s": time.perf_counter() - start,
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
              "resource_note": "process high-water RSS includes imports/JIT; external monitor records process-tree peak"}
    np.savez_compressed(out / "coefficients.npz", A_initial=A, A_final=best.A, A_replayed=replay,
                        labels=source["labels"] if "labels" in source.files else np.arange(A.shape[1]))
    dump(out / "rewrite_steps.json", [{"pair": [i, j], "U_real": U.real.tolist(), "U_imag": U.imag.tolist()} for i, j, U in best.steps])
    dump(out / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "termination"}), flush=True)


if __name__ == "__main__":
    main()
