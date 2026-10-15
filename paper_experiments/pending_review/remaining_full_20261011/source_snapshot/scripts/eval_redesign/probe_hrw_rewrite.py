#!/usr/bin/env python3
"""Measure HRW rewrite alternatives without changing the default compiler.

The sparse shortcut discovers two-row buckets with identical Pauli supports,
then validates an equal-size split by an existing unitary-mixing candidate.
It is an opt-in experiment, not a replacement for generic rewrite search.
"""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("src", "scripts", "Baseline_scripts")]


def paired_support_shortcut(kraus, tol=1e-12):
    """Return valid local unitary rewrites, plus explicit detected/missed counts.

    Match actual labels/coefficients, never a benchmark name or row adjacency.
    Only accept nonzero equal-size output supports that halve each input row.
    The Kraus count therefore stays fixed and each accepted H_length cost drops.
    Unmatched rows are retained verbatim for a subsequent generic fallback.
    """
    import numpy as np
    from channel_IR import channel as ChannelIR, Matrixsum, PauliAtom

    output = deepcopy(kraus)
    groups = defaultdict(list)
    for i, ms in enumerate(output):
        groups[frozenset(a.expr for a, c in ms.instances if abs(c) > tol)].append(i)
    steps = []
    checked = 0
    for support, indices in groups.items():
        if len(indices) != 2 or len(support) < 2 or len(support) % 2:
            continue
        i, j = indices
        # Preserve input label order to compare actual generated circuits too.
        labels = [a.expr for a, _ in output[i].instances]
        if len(set(labels)) != len(labels) or set(labels) != set(support):
            continue
        mappings = [{a.expr: c * a.phase for a, c in output[k].instances} for k in (i, j)]
        A = np.array([[mapping[label] for label in labels] for mapping in mappings])
        checked += 1
        for U in ChannelIR._candidate_unitaries_for_pair(A, 0, 1, tol):
            if not np.allclose(U.conj().T @ U, np.eye(2), atol=tol, rtol=0):
                continue
            # Same arithmetic as the legacy search, avoiding matrix-product rounding.
            B = np.stack([U[0, 0] * A[0] + U[0, 1] * A[1],
                          U[1, 0] * A[0] + U[1, 1] * A[1]])
            residual = B.copy()
            B[np.abs(B) < tol] = 0
            sizes = np.count_nonzero(np.abs(B) > tol, axis=1)
            if not np.all(sizes == len(labels) // 2):
                continue
            for index, row in zip((i, j), B):
                output[index] = Matrixsum([(PauliAtom(label, phase=value / abs(value)), float(abs(value)))
                                          for label, value in zip(labels, row) if abs(value) > tol])
            steps.append({"pair": [i, j], "support_before": [len(labels)] * 2,
                          "support_after": sizes.tolist(), "unitary_real": U.real.tolist(),
                          "unitary_imag": U.imag.tolist(),
                          "unitarity_error": float(np.linalg.norm(U.conj().T @ U - np.eye(2))),
                          "thresholded_residual_norm": float(np.linalg.norm(residual - B))})
            break
    return output, {"method": "paired_support_shortcut", "steps": steps,
                    "support_buckets": len(groups), "pair_buckets_checked": checked,
                    "unmatched_rows": len(output) - 2 * len(steps)}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args):
    import numpy as np
    from qiskit import qasm2
    from block_encoding import BlockEncoding
    from channel_IR import channel as ChannelIR, channel_ensemble
    from channel_LCU import channel_to_LCU
    from channelIR_gatecost_rewrite_search import h_length_cost
    from experiment_protocol import source_identity, settings, transpile_channel_circuit
    from rq1_pipeline import canonicalize, deserialize_kraus, serialize_kraus, verify_channel

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "settings.json").exists():
        raise ValueError("Use a fresh output directory")
    sources = source_identity()
    reference = ROOT / "paper_experiments/pending_review/table13_matrix_order_qrisp_20261010"
    old_settings = json.loads((reference / "settings.json").read_text())
    if sources != old_settings["sources"]:
        raise ValueError("Compiler differs from the saved beam baseline")
    dump(out / "settings.json", {"sources": sources, "protocol": settings("RQ1"),
         "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
         "reference": str(reference), "warmups": 1, "repetitions": args.repeats,
         "scope": "opt-in alternatives; compiler default unchanged",
         "timing": "rewrite includes copying, detection/search, reconstruction and zero elimination; compile adds canonicalization and unchanged tree+F* Matrix-order NAM opt0 lowering"})
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    methods = ["support_greedy_incremental", "paired_support_shortcut", "gatecost_greedy"]
    summary = []
    for n in args.sizes:
        case = f"HRW-{n}"
        input_record = json.loads((reference / case / "input.json").read_text())
        raw = deserialize_kraus(input_record["input_kraus"])
        canonical, _ = canonicalize(raw)
        A, labels = ChannelIR(canonical)._coeff_matrix()
        # In the independent Pauli basis this coefficient Gram matrix specifies
        # the complete unnormalized channel, without a 2**n dense simulation.
        gram = A.conj().T @ A
        baseline = json.loads((reference / case / "ours/run-1.json").read_text())
        refrows = baseline["assembly"]["compiled_kraus"]
        refmap = [{a: c * complex(re, im) for a, c, re, im in row} for row in refrows]
        refA = np.array([[row.get(label, 0) for label in labels] for row in refmap])
        dump(out / case / "input.json", input_record)
        for method in methods:
            measurements = []
            for repetition in range(args.repeats + 1):
                started = time.perf_counter()
                can, _ = canonicalize(raw)
                rewrite_start = time.perf_counter()
                if method == "paired_support_shortcut":
                    rewritten, result = paired_support_shortcut(can)
                    search_info = result
                else:
                    ir = ChannelIR(deepcopy(can))
                    if method == "gatecost_greedy":
                        result = ir.gatecost_rewrite_search(strategy="greedy", max_steps=50)
                    else:
                        result = ir.rewrite_search(strategy="greedy_incremental", max_steps=50)
                    ir.apply_rewrite_result(result).zero_elim()
                    rewritten = ir.kraus_ops
                    search_info = {key: result[key] for key in ("initial_support", "final_support", "termination")}
                    search_info["accepted_steps"] = len(result["steps"])
                rewrite_end = time.perf_counter()
                with (out / case / f"{method}.log").open("a") as log, redirect_stdout(log), redirect_stderr(log):
                    qc, registers = channel_to_LCU(channel_ensemble([deepcopy(rewritten)]),
                        structure="tree", opt="Matrix-order", fstar=True,
                        block_encoding_cls=lambda ms: BlockEncoding(ms, placement_options={}, matrix_order_options={}))
                    lowered = transpile_channel_circuit(qc, basis_gates=["rz", "h", "x", "cx"], optimization_level=0,
                                                       seed_transpiler=20261009)
                ended = time.perf_counter()
                B, blabels = ChannelIR(rewritten)._coeff_matrix()
                by_label = {label: B[:, i] for i, label in enumerate(blabels)}
                B = np.stack([by_label.get(label, np.zeros(B.shape[0])) for label in labels], axis=1)
                semantic_error = float(np.linalg.norm(B.conj().T @ B - gram) / np.linalg.norm(gram))
                coefficient_error = float(np.max(np.abs(B - refA)))
                record = {"case": case, "method": method, "warmup": repetition == 0,
                          "repetition": repetition, "rewrite_time_s": rewrite_end - rewrite_start,
                          "compile_time_s": ended - started, "assembly_lowering_s": ended - rewrite_end,
                          "G": int(sum(lowered.count_ops().values())), "counts": dict(lowered.count_ops()),
                          "W": lowered.num_qubits, "Lambda": qc.metadata["subnormalization_factor"],
                          "initial_support": int(np.count_nonzero(A)), "final_support": int(np.count_nonzero(B)),
                          "h_length": h_length_cost(B), "coefficient_gram_relative_error": semantic_error,
                          "ordered_coefficient_max_difference_from_beam": coefficient_error,
                          "same_gate_count_as_beam": sum(lowered.count_ops().values()) == baseline["G"],
                          "search": search_info}
                assert semantic_error < 1e-10, record
                if repetition == 1:
                    qasm2.dump(lowered, out / case / f"{method}.qasm")
                    dump(out / case / f"{method}_kraus.json", serialize_kraus(rewritten))
                    if n == 4:
                        record["full_choi"] = verify_channel(lowered, registers, raw)
                        assert record["full_choi"]["passed"]
                dump(out / case / method / f"run-{repetition}.json", record)
                if repetition:
                    measurements.append(record)
                print(json.dumps({k: record[k] for k in ("case", "method", "repetition", "rewrite_time_s", "G", "ordered_coefficient_max_difference_from_beam")}), flush=True)
            assert len({r["G"] for r in measurements}) == 1
            summary.append({"case": case, "method": method,
                            "rewrite_median_s": median(r["rewrite_time_s"] for r in measurements),
                            "compile_median_s": median(r["compile_time_s"] for r in measurements),
                            "G": measurements[0]["G"], "support": measurements[0]["final_support"],
                            "semantic_max_error": max(r["coefficient_gram_relative_error"] for r in measurements),
                            "coefficient_max_difference_from_beam": max(r["ordered_coefficient_max_difference_from_beam"] for r in measurements)})
            dump(out / "summary.json", summary)
            with (out / "comparison.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
    if source_identity() != sources:
        raise RuntimeError("Compiler sources changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, choices=(4, 8, 12, 20, 28), default=[4, 8, 12, 20, 28])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    with tempfile.TemporaryDirectory(prefix="hrw-rewrite-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["NUMBA_CACHE_DIR"] = cache
        run(args)


if __name__ == "__main__":
    main()
