"""Small CPTP channels with fully overlapping Pauli supports: objective audit."""
from pathlib import Path
import contextlib
import hashlib
import io
import itertools
import json
import resource
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "streaming_support_beam_20261012"
sys.path.insert(0, str(PREVIOUS))
import fast_probe as support
import diagnostic_gatecost as gate


def paulis(n):
    one = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
           "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    labels, matrices = [], []
    for word in itertools.product("IXYZ", repeat=n):
        matrix = np.array([[1]], complex)
        for char in word:
            matrix = np.kron(matrix, one[char])
        labels.append("".join(word))
        matrices.append(matrix)
    return np.array(labels), np.array(matrices)


def sample_channel(seed, m=4, n=2):
    rng = np.random.default_rng(seed)
    d = 1 << n
    z = rng.normal(size=(m * d, d)) + 1j * rng.normal(size=(m * d, d))
    q, r = np.linalg.qr(z, mode="reduced")
    q *= np.diag(r) / np.abs(np.diag(r))
    kraus = q.reshape(m, d, d)
    labels, basis = paulis(n)
    A = np.einsum("lab,jab->jl", basis.conj(), kraus) / d
    return A, labels, float(np.linalg.norm(sum(k.conj().T @ k for k in kraus) - np.eye(d)))


def overlap(A):
    masks = np.abs(A) > 1e-12
    scores = [np.count_nonzero(a & b) / np.count_nonzero(a | b)
              for a, b in itertools.combinations(masks, 2) if np.any(a | b)]
    return {"mean_pair_jaccard": float(np.mean(scores)), "min_pair_jaccard": float(min(scores)),
            "row_supports": masks.sum(axis=1).tolist()}


def run_case(name, A, labels, metadata):
    dest = HERE / "cases" / name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "results.json").exists():
        return json.loads((dest / "results.json").read_text())
    results = {}
    np.savez_compressed(dest / "input.npz", A=A, labels=labels)
    manifest = {"case": name, "metadata": metadata, "input_sha256": hashlib.sha256(A.tobytes()).hexdigest(),
                "overlap": overlap(A), "width": 8, "max_steps": 256, "per_search_timeout_s": 20,
                "gate_mode": "admit_support", "neutral": True, "cost_model": gate.SETTINGS.metadata(),
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [Path(__file__), PREVIOUS / "fast_probe.py", PREVIOUS / "diagnostic_gatecost.py"]}}
    support.dump(dest / "manifest.json", manifest)
    with (dest / "run.log").open("w") as log, contextlib.redirect_stdout(log):
        for method in ("support", "gatecost"):
            started = time.perf_counter()
            if method == "support":
                state, termination = support.search(A, None, width=8, max_steps=256,
                    rank_mode="support_rank", neutral=True, timeout=20)
            else:
                state, termination = gate.search(A, None, width=8, max_steps=256,
                    neutral=True, timeout=20, mode="admit_support")
            verify, replay = support.verify(A, state.A, state.steps, 1e-12)
            np.savez_compressed(dest / (method + ".npz"), A_initial=A, A_final=state.A, A_replayed=replay, labels=labels)
            support.dump(dest / (method + "_steps.json"), [{"pair": [i, j], "U_real": U.real.tolist(), "U_imag": U.imag.tolist()} for i, j, U in state.steps])
            results[method] = {"metrics": support.metrics(state.A, 1e-12), "termination": termination,
                               "verification": verify, "time_with_validation_s": time.perf_counter() - started,
                               "converged": termination["stop_reason"] == "frontier_exhausted" and verify["passed"]}
    hs = results["support"]["metrics"]["static_cost"]["total"]
    hg = results["gatecost"]["metrics"]["static_cost"]["total"]
    results.update(case=name, input=support.metrics(A, 1e-12), metadata=metadata, overlap=overlap(A),
                   support_H_over_gate_H=hs/hg, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    support.dump(dest / "results.json", results)
    return results


def main():
    raise SystemExit("The Petz/Collective legacy probe is retired; use paper_experiments/experiments_new/fixture_benchmarks/run.py")


if __name__ == "__main__":
    main()
