"""Frozen support search with a read-only callback on committed frontiers."""
import json
import time
import numpy as np
from fast_probe import State, TopK, dump, row_key, state_key, fast_candidates

def search(A0, out, width=8, max_steps=583, tol=1e-12, neutral=True,
           timeout=1800, collector="heap", rank_mode="original", max_stagnant=32, on_layer=None):
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
