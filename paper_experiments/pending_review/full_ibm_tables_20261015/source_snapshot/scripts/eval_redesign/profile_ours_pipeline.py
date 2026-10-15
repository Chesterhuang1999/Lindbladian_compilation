#!/usr/bin/env python3
"""Reproduce saved paper inputs with exclusive timing of the current Ours pipeline.

Timers wrap stage boundaries without changing compiler arguments or results.
Nested durations are subtracted from parents, so exclusive times add to total
compilation time. Candidate NAM probes belong to O, not to rewrite scoring.
Input loading, imports, verification, recounting and output I/O are not timed.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import csv
from functools import wraps
import gc
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
from statistics import median
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("src", "scripts", "Baseline_scripts")]
STAGES = ("canonicalization", "rewrite", "placement", "inner_prepare",
          "select_anf_lowering", "internal_nam_cost_probes", "block_assembly",
          "fstar_control_assembly", "outer_tree_assembly", "final_nam_lowering", "pipeline_other")


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class StageTimers:
    def __init__(self):
        self.stack = []
        self.exclusive = defaultdict(float)
        self.inclusive = defaultdict(float)
        self.calls = defaultdict(int)
        self.rewrite_termination = None

    @contextmanager
    def stage(self, name):
        frame = [name, time.perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[1]
            self.stack.pop()
            if self.stack:
                self.stack[-1][2] += elapsed
            self.exclusive[name] += elapsed - frame[2]
            self.inclusive[name] += elapsed
            self.calls[name] += 1

    def wrap(self, function, stage):
        @wraps(function)
        def measured(*args, **kwargs):
            with self.stage(stage):
                result = function(*args, **kwargs)
                if stage == "rewrite" and isinstance(result, dict) and "termination" in result:
                    self.rewrite_termination = result["termination"]
                return result
        return measured

    @contextmanager
    def installed(self):
        import block_encoding as be
        import matrix_order_pipeline as mo
        import rq1_pipeline as rq
        targets = [
            (rq, "run_rewrite_search", "rewrite"),
            (rq.ChannelIR, "rewrite_search", "rewrite"),
            (rq.ChannelIR, "apply_rewrite_result", "rewrite"),
            (rq, "channel_to_LCU", "outer_tree_assembly"),
            (be.BlockEncoding, "circuit_components", "block_assembly"),
            (be._PlacementBlockEncoding, "find_optimal_order_matrices", "placement"),
            (be, "extra_placements", "placement"),
            (be._PlacementBlockEncoding, "_extend_phase_candidates", "placement"),
            (be, "prepare_coefficients", "inner_prepare"),
            (be, "lower_select", "select_anf_lowering"),
            (be, "nam_cost", "internal_nam_cost_probes"),
            (mo, "nam_cost", "internal_nam_cost_probes"),
            (be.BlockEncodingComponents, "with_outer_control", "fstar_control_assembly"),
            (be.BlockEncodingComponents, "controlled", "fstar_control_assembly"),
            (be.BlockEncodingComponents, "system_last", "fstar_control_assembly"),
        ]
        patches = [patch.object(obj, name, self.wrap(getattr(obj, name), stage))
                   for obj, name, stage in targets]
        try:
            for item in patches:
                item.start()
            yield
        finally:
            for item in reversed(patches):
                item.stop()


def compile_one(raw, *, rewrite_options=None):
    import rq1_pipeline as rq
    from experiment_protocol import DEFAULT_TRANSPILE_SEED, transpile_channel_circuit
    timer = StageTimers()
    with timer.installed():
        with timer.stage("pipeline_other"):
            with timer.stage("canonicalization"):
                canonical, canonical_info = rq.canonicalize(raw)
            circuit, registers, assembly = rq.build_pipeline(
                canonical, "Ours", matrix_order_options={}, **(rewrite_options or {}))
            with timer.stage("final_nam_lowering"):
                lowered = transpile_channel_circuit(
                    circuit, basis_gates=["rz", "h", "x", "cx"], optimization_level=0,
                    seed_transpiler=DEFAULT_TRANSPILE_SEED)
    stages = {name: timer.exclusive[name] for name in STAGES}
    elapsed = timer.inclusive["pipeline_other"]
    assert abs(sum(stages.values()) - elapsed) < 1e-8 * max(elapsed, 1)
    counts = {str(k): int(v) for k, v in lowered.count_ops().items()}
    assert not set(counts) - {"rz", "h", "x", "cx"}
    return lowered, {
        "compile_time_s": elapsed, "exclusive_stage_times_s": stages,
        "inclusive_stage_times_s": dict(timer.inclusive), "stage_calls": dict(timer.calls),
        "G": sum(counts.values()), "counts": counts, "W": lowered.num_qubits,
        "Lambda": float(circuit.metadata["subnormalization_factor"]),
        "canonicalization": canonical_info, "assembly": assembly,
        "rewrite_termination": timer.rewrite_termination,
        "registers": registers, "metadata": circuit.metadata,
    }


def coefficient_gram_error(raw, compiled):
    """Check the full unnormalized channel in the independent Pauli basis."""
    import numpy as np
    from rq1_pipeline import serialize_kraus
    before = serialize_kraus(raw)
    labels = sorted({a for rows in (before, compiled) for row in rows for a, *_ in row})
    index = {label: i for i, label in enumerate(labels)}
    def matrix(rows):
        result = np.zeros((len(rows), len(labels)), dtype=complex)
        for i, row in enumerate(rows):
            for label, magnitude, re, im in row:
                result[i, index[label]] += magnitude * complex(re, im)
        return result
    a, b = matrix(before), matrix(compiled)
    ga, gb = a.conj().T @ a, b.conj().T @ b
    error = float(np.linalg.norm(ga - gb) / max(np.linalg.norm(ga), 1e-30))
    assert error < 1e-10, error
    return error


def run(args):
    from qiskit import qasm2
    import rq1_pipeline as rq
    from experiment_protocol import settings, source_identity
    from baseline_common import count_qasm_file
    from channel_rewrite_cost import h_length_components
    output = args.out_dir.resolve()
    if (output / "settings.json").exists():
        raise ValueError("Use a fresh output directory")
    identities = source_identity()
    rewrite_options = {
        "rewrite_method": args.rewrite_method, "strategy": args.rewrite_strategy,
        "beam_width": args.beam_width, "max_steps": args.max_steps,
    }
    policy = {"protocol": settings("RQ1"), "sources": identities,
              "rewrite_overrides": rewrite_options,
              "rewrite_objective": ("total_pauli_support" if args.rewrite_method == "rewrite_search"
                                    else "H_length_static_nc_v1"),
              "cases": args.cases, "input_root": str(args.input_root.resolve()),
              "repetitions": args.repeats, "warmups": args.warmups,
              "timing": "serial warm process; exclusive nested stage wall times sum to total",
              "excluded": ["imports", "input loading", "verification", "QASM export/recount", "output I/O"],
              "summary": "stage breakdown from the repetition at median total compilation time",
              "stage_names": STAGES, "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "machine": {"platform": platform.platform(), "python": sys.version},
              "versions": {p: version(p) for p in ("qiskit", "numpy")},
              "thread_environment": {k: os.environ.get(k) for k in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "QISKIT_NUM_PROCS")}}
    policy["protocol"]["rewrite"].update(
        method=args.rewrite_method, strategy=args.rewrite_strategy,
        beam_width=args.beam_width, max_steps=args.max_steps,
        cost_model=policy["rewrite_objective"])
    dump(output / "settings.json", policy)
    for source in (ROOT / "src").glob("*.py"):
        dest = output / "source_snapshot/src" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    shutil.copy2(__file__, output / "runner.py")
    summaries, samples = [], []
    for case in args.cases:
        input_path = args.input_root / case / "input.json"
        payload = json.loads(input_path.read_text())
        raw = rq.deserialize_kraus(payload["input_kraus"])
        dump(output / case / "input.json", {**payload, "source": str(input_path.resolve()),
             "source_sha256": sha256(input_path.read_bytes()).hexdigest()})
        results = []
        for iteration in range(args.warmups + args.repeats):
            kind = "warmup" if iteration < args.warmups else "run"
            number = iteration + 1 if kind == "warmup" else iteration - args.warmups + 1
            print(json.dumps({"case": case, "stage": "start", "kind": kind, "number": number}), flush=True)
            gc.collect()
            with (output / case / f"{kind}-{number}.log").open("w") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    lowered, result = compile_one(raw, rewrite_options=rewrite_options)
            assert source_identity() == identities, "Compiler changed during measurement"
            result.update(case=case, kind=kind, repetition=number, sources=identities)
            result["rewrite_gram_relative_error"] = coefficient_gram_error(raw, result["assembly"]["compiled_kraus"])
            # Audit the current F* cost even when an explicitly selected
            # support search does not use it as its optimization objective.
            compiled = rq.deserialize_kraus(result["assembly"]["compiled_kraus"])
            matrix, _ = rq.ChannelIR(compiled)._coeff_matrix()
            audit = h_length_components(matrix, structure="tree", fstar=True)
            result["static_fstar_cost_audit"] = audit
            expected_lambda = audit["Lambda"]
            assert abs(expected_lambda - result["Lambda"]) < 1e-10 * max(1, expected_lambda)
            if kind == "run":
                qasm_path = output / case / f"run-{number}_nam.qasm"
                qasm2.dump(lowered, qasm_path)
                recount = count_qasm_file(qasm_path, gate_set="nam")
                assert recount["metric_total"] == result["G"]
                result["non_clifford"] = recount["clifford"]["non_clifford"]
                result["qasm_sha256"] = sha256(qasm_path.read_bytes()).hexdigest()
                results.append(result)
                samples.append({"case": case, "repetition": number, "G": result["G"],
                                "compile_time_s": result["compile_time_s"], **result["exclusive_stage_times_s"]})
                write_csv(output / "samples.csv", samples)
            dump(output / case / f"{kind}-{number}.json", result)
            print(json.dumps({"case": case, "kind": kind, "number": number,
                              "G": result["G"], "s": result["compile_time_s"],
                              "stages": result["exclusive_stage_times_s"]}), flush=True)
        assert len({r["G"] for r in results}) == 1, "Gate counts varied"
        representative = sorted(results, key=lambda r: r["compile_time_s"])[len(results)//2]
        row = {"case": case, "G": representative["G"], "W": representative["W"],
               "Lambda": representative["Lambda"], "repetitions": args.repeats,
               "representative_repetition": representative["repetition"],
               "compile_time_median_s": median(r["compile_time_s"] for r in results),
               "compile_time_min_s": min(r["compile_time_s"] for r in results),
               "compile_time_max_s": max(r["compile_time_s"] for r in results),
               **representative["exclusive_stage_times_s"]}
        summaries.append(row)
        write_csv(output / "comparison.csv", summaries)
        dump(output / "aggregates.json", summaries)


def main():
    from experiment_protocol import REWRITE_DEFAULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["HRW-28", "TFIM-12", "RepDQE-d5"])
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "paper_experiments/pending_review/table13_matrix_order_qrisp_20261010")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rewrite-method", choices=("gatecost_rewrite_search", "rewrite_search"),
                        default=REWRITE_DEFAULTS.method)
    parser.add_argument("--rewrite-strategy", choices=("beam", "greedy", "greedy_incremental"),
                        default=REWRITE_DEFAULTS.strategy)
    parser.add_argument("--beam-width", type=int, default=REWRITE_DEFAULTS.beam_width)
    parser.add_argument("--max-steps", type=int, default=REWRITE_DEFAULTS.max_steps)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1 or args.repeats % 2 != 1:
        parser.error("Require at least one warmup and a positive odd number of repetitions")
    if args.rewrite_strategy == "greedy_incremental" and args.rewrite_method != "rewrite_search":
        parser.error("greedy_incremental currently optimizes support; explicitly use --rewrite-method rewrite_search")
    if args.max_steps < 1 or args.beam_width < 1:
        parser.error("max-steps and beam-width must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-", dir=args.out_dir.resolve()) as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(cache) / "numba"))
        run(args)


if __name__ == "__main__":
    main()
