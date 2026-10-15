#!/usr/bin/env python3
"""Serial warm-process timing of Qrisp+F* and the default Matrix-order pipeline.

Constructed paper inputs are loaded before timing. Each measurement starts
before canonicalization and ends after NAM opt0 lowering. File I/O, imports,
gate recounts, and verification are outside the measured interval. Ours runs
the default rewrite; HRW additionally measures an explicit no-rewrite route.
"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import csv
import gc
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
from statistics import median, mean
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / folder) for folder in ("src", "scripts", "Baseline_scripts")]
CASES = ("TFIM-3", "TFIM-4", "TFIM-8", "HRW-4", "RepDQE-d3", "RepDQE-d5")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args):
    import numpy as np
    from qiskit import qasm2
    from block_encoding import BlockEncoding
    from channel_IR import channel_ensemble
    from channel_LCU import channel_to_LCU
    from experiment_protocol import (DEFAULT_TRANSPILE_SEED, source_identity,
                                     settings, transpile_channel_circuit)
    from matrix_order_pipeline import options_dict, resolve_options
    # Import both backends before timing. Full-case warmups also initialize
    # lazy imports, numerical runtimes, and Qrisp's first-use compilation.
    from qrisp_channel_adapter import qrisp_backend_for_channel  # noqa: F401
    from rq1_pipeline import (build_pipeline, canonicalize, deserialize_kraus,
                              input_identity, serialize_kraus)

    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "settings.json").exists():
        raise ValueError("Choose a fresh output directory; earlier results are never overwritten")
    identities = source_identity()
    script_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    policy = {
        "timing": "wall clock from canonicalization through final NAM opt0 lowering",
        "includes": ["canonicalization", "rewrite when enabled", "placement and its NAM cost probes",
                     "block and outer tree assembly", "NAM opt0 lowering"],
        "excludes": ["input construction/loading", "module imports", "warmups", "verification",
                     "gate counting", "result serialization", "QASM output"],
        "execution": "serial, same process, alternate route order between repetitions",
        "warmups_per_route": args.warmups, "repetitions": args.repeats,
        "suite": args.suite, "cases": args.cases,
        "summary_statistic": "median", "protocol": settings("RQ1"),
        "matrix_order_options": options_dict(resolve_options()),
        "sources": identities, "runner_sha256": script_hash,
        "machine": {"platform": platform.platform(), "architecture": platform.machine(),
                    "python": sys.version, "logical_cpus": os.cpu_count()},
        "versions": {package: version(package) for package in ("qiskit", "qrisp", "numpy")},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "QISKIT_NUM_PROCS")},
    }
    dump(output / "settings.json", policy)
    # Retain the actual implementation, including transitive local helpers.
    for folder in ("src", "scripts", "Baseline_scripts", "scripts/eval_redesign"):
        for source in (ROOT / folder).glob("*.py"):
            target = output / "source_snapshot" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    aggregates = []
    all_samples = []

    def compile_one(raw, route):
        started = time.perf_counter()
        canonical, canonical_info = canonicalize(raw)
        if route == "Ours-no-rewrite":
            assembly_start = time.perf_counter()
            circuit, registers = channel_to_LCU(
                channel_ensemble([deepcopy(canonical)]), structure="tree", opt="Matrix-order",
                block_encoding_cls=lambda ms: BlockEncoding(ms, placement_options={}), fstar=True)
            assembly = {"pipeline": route, "configuration": {"rewrite": False, "inner": "Matrix-order",
                        "structure": "tree", "fstar": True}, "rewrite": {"method": "none", "time_s": 0.0},
                        "assembly_time_s": time.perf_counter() - assembly_start,
                        "compiled_kraus": serialize_kraus(canonical),
                        "canonical_input_sha256": input_identity(canonical),
                        "matrix_order_options": options_dict(resolve_options())}
        else:
            circuit, registers, assembly = build_pipeline(canonical, route, matrix_order_options={})
        lowering_start = time.perf_counter()
        lowered = transpile_channel_circuit(circuit, basis_gates=["rz", "h", "x", "cx"],
                                            optimization_level=0, seed_transpiler=DEFAULT_TRANSPILE_SEED)
        ended = time.perf_counter()
        elapsed = ended - started
        # Counting and provenance collection below are not compilation time.
        counts = {str(name): int(count) for name, count in lowered.count_ops().items()}
        assert not (set(counts) - {"rz", "h", "x", "cx"}), counts
        return lowered, {
            "route": route, "compile_time_s": elapsed,
            "canonicalization_time_s": canonical_info["time_s"],
            "rewrite_time_s": assembly["rewrite"]["time_s"],
            "assembly_time_s": assembly["assembly_time_s"],
            "lowering_time_s": ended - lowering_start,
            "G": sum(counts.values()), "counts": counts, "W": lowered.num_qubits,
            "Lambda": float(circuit.metadata["subnormalization_factor"]),
            "canonicalization": canonical_info, "assembly": assembly,
            "registers": registers, "metadata": circuit.metadata,
        }

    for case in args.cases:
        if args.suite == "table13":
            from paper_rq1_inputs import build_table13_case
            raw, model = build_table13_case(case)
            input_record = {"input_kraus": serialize_kraus(raw), "model": model}
        else:
            source = (ROOT / "paper_experiments/pending_review/rq1_early_control_20261010/full/runs"
                      / case / "ours/results.json")
            payload = json.loads(source.read_text())
            raw = deserialize_kraus(payload["input_kraus"])
            input_record = {"input_kraus": payload["input_kraus"], "source": str(source),
                            "source_sha256": sha256(source.read_bytes()).hexdigest()}
        case_dir = output / case
        case_dir.mkdir()
        dump(case_dir / "input.json", input_record)
        print(json.dumps({"case": case, "stage": "input", "qubits": raw[0].size,
                          "kraus": len(raw), "term_counts": [len(ms.instances) for ms in raw]}), flush=True)
        routes = ["Qrisp", "Ours"] + (["Ours-no-rewrite"] if case == "HRW-4" and args.suite == "diagnostic" else [])
        measurements = {route: [] for route in routes}
        with (case_dir / "compilation.log").open("w") as log:
            for route in routes:
                for iteration in range(args.warmups):
                    gc.collect()
                    with redirect_stdout(log), redirect_stderr(log):
                        _, result = compile_one(raw, route)
                    dump(case_dir / route.lower() / f"warmup-{iteration + 1}.json", result)
                    print(json.dumps({"case": case, "route": route, "stage": "warmup",
                                      "G": result["G"], "s": round(result["compile_time_s"], 3)}), flush=True)
            for repetition in range(args.repeats):
                order = routes if repetition % 2 == 0 else routes[::-1]
                for route in order:
                    gc.collect()
                    with redirect_stdout(log), redirect_stderr(log):
                        lowered, result = compile_one(raw, route)
                    result.update(case=case, repetition=repetition + 1)
                    path = case_dir / route.lower() / f"run-{repetition + 1}.json"
                    dump(path, result)
                    if repetition == 0:
                        qasm2.dump(lowered, path.with_name("channel_nam_opt0.qasm"))
                    measurements[route].append(result)
                    sample = {key: result[key] for key in ("case", "route", "repetition", "G", "W", "Lambda",
                              "compile_time_s", "canonicalization_time_s", "rewrite_time_s", "assembly_time_s", "lowering_time_s")}
                    all_samples.append(sample)
                    dump(output / "samples.json", all_samples)
                    print(json.dumps({"case": case, "route": route, "repeat": repetition + 1,
                                      "G": result["G"], "s": round(result["compile_time_s"], 3)}), flush=True)
        for route, results in measurements.items():
            assert len({result["G"] for result in results}) == 1, (case, route, "gate counts varied")
            assert len({result["assembly"]["canonical_input_sha256"] for result in results}) == 1
            times = [result["compile_time_s"] for result in results]
            row = {"case": case, "route": route, "G": results[0]["G"], "W": results[0]["W"],
                   "Lambda": results[0]["Lambda"], "compile_time_median_s": median(times),
                   "compile_time_min_s": min(times), "compile_time_max_s": max(times),
                   "repetitions": args.repeats}
            aggregates.append(row)
        # Both paths consume the same canonical channel; Ours may then rewrite.
        assert len({results[0]["assembly"]["canonical_input_sha256"] for results in measurements.values()}) == 1
        dump(output / "aggregates.json", aggregates)
        with (output / "comparison.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
        grouped = {}
        for row in aggregates:
            label = row["case"].split("-seed")[0]
            grouped.setdefault(label, {}).setdefault(row["route"], []).append(row)
        table = []
        for label, route_rows in grouped.items():
            if "Qrisp" not in route_rows or "Ours" not in route_rows:
                continue
            row = {"benchmark": label, "inputs": len(route_rows["Ours"])}
            for route in ("Qrisp", "Ours"):
                values = route_rows[route]
                row[route + "_G"] = mean(v["G"] for v in values)
                row[route + "_compile_time_s"] = mean(v["compile_time_median_s"] for v in values)
            row["gate_reduction_fraction"] = 1 - row["Ours_G"] / row["Qrisp_G"]
            table.append(row)
        with (output / "comparison_table.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    if source_identity() != identities:
        raise RuntimeError("Compiler sources changed during measurement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("diagnostic", "table13"), default="diagnostic")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    from paper_rq1_inputs import table13_case_names
    available = table13_case_names() if args.suite == "table13" else list(CASES)
    args.cases = args.cases or available
    if len(set(args.cases)) != len(args.cases) or any(case not in available for case in args.cases):
        parser.error("Cases must be unique and belong to the selected suite")
    if args.repeats < 1 or args.warmups < 1:
        parser.error("At least one warmup and one measurement are required")
    # Runtime caches are temporary; retained files are experimental evidence.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-", dir=args.out_dir.resolve()) as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(cache) / "numba"))
        run(args)


if __name__ == "__main__":
    main()
