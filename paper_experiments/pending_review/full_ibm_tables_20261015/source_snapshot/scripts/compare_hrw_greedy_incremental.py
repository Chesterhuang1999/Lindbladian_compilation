#!/usr/bin/env python3
"""Compare unchanged greedy and incremental greedy on full-channel HRW cases."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("greedy", "greedy_incremental")
BASIS = {"ibm": ("u1", "u2", "u3", "cx", "reset"),
         "nam": ("h", "x", "rz", "cx", "reset")}
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "QISKIT_PARALLEL": "FALSE",
              "RAYON_NUM_THREADS": "1", "PYTHONHASHSEED": "7"}
SOURCE_PATHS = [Path(__file__).resolve(), ROOT / "scripts/run_hypercube_n4_ours_nam.py",
                *sorted((ROOT / "src").glob("*.py")),
                ROOT / "Baseline_scripts/baseline_common.py"]
DEFAULT_ROOT = ROOT / "Baseline_results/hrw_greedy_incremental_20260909"
DEFAULT_REPORT = ROOT / "Results/hrw_greedy_incremental_comparison_20260909.md"
LEGACY_ARCHIVE = ROOT / "Baseline_results/scalability_compile_20260909_larger/compiler_sources.executed.tar.gz"
TIMINGS = ("rewrite_time_s", "circuit_build_time_s", "ibm_lowering_time_s",
           "nam_lowering_time_s", "ibm_compile_time_s", "nam_compile_time_s")
PAPER_SIZES = (4, 8, 12, 20, 28)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS}


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def snapshot_sources(out_root):
    with tarfile.open(LEGACY_ARCHIVE, "r:gz") as archive:
        source = archive.extractfile("src/channel_IR.py").read()
    original = out_root / "original_channel_IR.py"
    original.write_bytes(source)
    current = (ROOT / "src/channel_IR.py").read_bytes()

    def methods(raw):
        tree = ast.parse(raw)
        channel = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "channel")
        return {node.name: ast.dump(node, include_attributes=False)
                for node in channel.body if isinstance(node, ast.FunctionDef)}

    before, after = methods(source), methods(current)
    preserved = ("_greedy_search", "_candidate_unitaries_for_pair", "_apply_unitary",
                 "_row_supports", "_support", "_coeff_matrix")
    checks = {name: before[name] == after[name] for name in preserved}
    if not all(checks.values()):
        raise ValueError(f"Original greedy implementation differs from archived source: {checks}")
    if "greedy_incremental" not in current.decode("utf-8"):
        raise ValueError("The greedy_incremental strategy is not installed yet")
    snapshot = out_root / "compiler_sources.executed.tar.gz"
    with tarfile.open(snapshot, "w:gz") as archive:
        for path in SOURCE_PATHS:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=True)
    (out_root / "environment.pip-freeze.txt").write_text(frozen.stdout, encoding="utf-8")
    return {"source_sha256": source_hashes(), "source_archive": str(snapshot),
            "source_archive_sha256": sha256(snapshot), "original_source": str(original),
            "original_source_sha256": sha256(original), "original_archive": str(LEGACY_ARCHIVE),
            "original_archive_sha256": sha256(LEGACY_ARCHIVE),
            "preserved_greedy_method_ast_equal": checks,
            "git_head": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--short")}


def circuit_stream(qc):
    return {"width": qc.num_qubits, "classical_bits": qc.num_clbits,
            "global_phase": float(qc.global_phase),
            "instructions": [{"name": item.operation.name,
                              "params": [float(value) for value in item.operation.params],
                              "qubits": [qc.find_bit(bit).index for bit in item.qubits],
                              "clbits": [qc.find_bit(bit).index for bit in item.clbits]}
                             for item in qc.data]}


def run_worker(args):
    import numpy as np
    import psutil
    import qiskit
    import qiskit_aer
    import scipy
    from qiskit import qasm2, qpy, transpile
    import run_hypercube_n4_ours_nam as hrw
    from baseline_common import count_qasm_file
    from qasm_export import export_openqasm2_quartz_input

    n, repetition, strategy = args.worker.split(":")
    n, repetition = int(n), int(repetition)
    run_dir = args.out_root / f"hrw_{n}" / f"rep_{repetition}_{strategy}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "summary.json"
    random.seed(args.seed)
    np.random.seed(args.seed)
    expected = json.loads((args.out_root / "manifest.json").read_text())["source_sha256"]
    hashes = source_hashes()
    summary = {"case": f"HRW-{n}", "n": n, "repetition": repetition,
               "strategy": strategy, "status": "running", "stage": "model",
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": hashes, "source_hashes_match_launch": hashes == expected,
               "environment": {"python": sys.version, "executable": sys.executable,
                               "platform": platform.platform(), "qiskit": qiskit.__version__,
                               "qiskit_aer": qiskit_aer.__version__, "numpy": np.__version__,
                               "scipy": scipy.__version__, "psutil": psutil.__version__,
                               "physical_memory_bytes": psutil.virtual_memory().total,
                               "threads": {key: os.environ.get(key) for key in THREAD_ENV},
                               "compiler_environment": {key: value for key, value in os.environ.items()
                                                        if key.startswith("CHANNEL_LCU_")}},
               "configuration": {"scope": "full_channel", "seed": args.seed,
                                 "tol": args.tol, "max_steps": args.max_steps, "beam_width": 3,
                                 "outer_select": "mulplex_U_opt", "block_opt": "Matrix-order",
                                 "optimization_level": 0, "seed_transpiler": args.seed,
                                 "basis": BASIS, "warmup": "none; initial coefficients captured before timing",
                                 "model_generator": "build_hypercube_section52_channel"},
               "simulation_status": "not_run_compile_only"}
    write_json(summary_path, summary)
    started = time.perf_counter()
    try:
        if not summary["source_hashes_match_launch"]:
            raise ValueError("Source changed after batch launch")
        model_started = time.perf_counter()
        raw = hrw.build_hypercube_channel(n)
        summary["model_time_s"] = time.perf_counter() - model_started
        summary["raw_metrics"] = hrw.channel_metrics(raw)
        summary["raw_subnormalization"] = float(sum(ms.pauli_norm()**2 for ms in raw))
        initial = hrw.ChannelIR(copy.deepcopy(raw))
        initial.zero_elim()
        initial_A, initial_labels = initial._coeff_matrix(args.tol)
        summary["stage"] = "rewrite"
        write_json(summary_path, summary)
        rewritten, result, elapsed = hrw.rewrite_channel(
            raw, strategy=strategy, beam_width=3, max_steps=args.max_steps, tol=args.tol)
        summary["rewrite_time_s"] = elapsed
        summary["rewrite"] = {key: result[key] for key in ("initial_support", "final_support", "termination")}
        summary["rewrite"]["steps"] = len(result["steps"])
        summary["rewritten_metrics"] = hrw.channel_metrics(rewritten)
        summary["subnormalization"] = float(sum(ms.pauli_norm()**2 for ms in rewritten))
        summary["inverse_subnormalization"] = 1 / summary["subnormalization"]

        # Save full numeric trajectories after the timed search, without stringifying arrays.
        state = initial_A.copy()
        states, pairs, unitaries, supports = [state.copy()], [], [], []
        for pair, unitary, support in result["steps"]:
            state = hrw.ChannelIR._apply_unitary(state, *pair, unitary)
            state[np.abs(state) < args.tol] = 0.0
            states.append(state.copy())
            pairs.append(pair)
            unitaries.append(unitary)
            supports.append(int(support))
        summary["replay_final_exact"] = bool(np.array_equal(state, result["A_final"]))
        summary["replay_final_allclose"] = bool(np.allclose(state, result["A_final"], rtol=0, atol=args.tol))
        summary["initial_labels_equal"] = initial_labels == result["labels"]
        if not summary["replay_final_allclose"] or not summary["initial_labels_equal"]:
            raise ValueError("Saved rewrite trajectory failed replay validation")
        np.savez_compressed(run_dir / "rewrite_arrays.npz", initial_A=initial_A,
                            final_A=result["A_final"], states=np.stack(states),
                            pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                            unitaries=np.asarray(unitaries, dtype=np.complex128).reshape(-1, 2, 2),
                            supports=np.asarray(supports, dtype=np.int64))
        write_json(run_dir / "rewrite_result.json", {"labels": result["labels"],
                   "initial_support": int(result["initial_support"]),
                   "final_support": int(result["final_support"]), "termination": result["termination"],
                   "steps": [{"pair": [int(i) for i in pair], "support_after": support,
                              "unitary_real": unitary.real.tolist(), "unitary_imag": unitary.imag.tolist()}
                             for pair, unitary, support in zip(pairs, unitaries, supports)]})

        summary["stage"] = "circuit_build"
        write_json(summary_path, summary)
        circuit, qubit_indexes, build_time = hrw.build_rewrite_opt_matrix_order_circuit(rewritten)
        summary.update(circuit_build_time_s=build_time, width=circuit.num_qubits,
                       qubit_indexes=qubit_indexes, basis_results={})
        for basis, gates in BASIS.items():
            summary["stage"] = f"lower_{basis}"
            write_json(summary_path, summary)
            lower_started = time.perf_counter()
            lowered = transpile(circuit, basis_gates=list(gates), optimization_level=0,
                                seed_transpiler=args.seed)
            lower_time = time.perf_counter() - lower_started
            summary[f"{basis}_lowering_time_s"] = lower_time
            summary[f"{basis}_compile_time_s"] = elapsed + build_time + lower_time
            qasm_path, qpy_path = run_dir / f"circuit_{basis}.qasm", run_dir / f"circuit_{basis}.qpy"
            export_openqasm2_quartz_input(lowered, qasm_path, basis_gates=gates)
            stats = count_qasm_file(qasm_path, gate_set=basis)
            with qpy_path.open("wb") as output:
                qpy.dump(lowered, output)
            with qpy_path.open("rb") as source:
                qpy_equal = qpy.load(source)[0] == lowered
            restored = qasm2.load(qasm_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            qasm_equal = restored.num_qubits == lowered.num_qubits and restored.count_ops() == lowered.count_ops()
            stream = run_dir / f"circuit_{basis}_instructions.json"
            write_json(stream, circuit_stream(lowered))
            summary["basis_results"][basis] = {
                "metric_gates": stats["metric_total"], "non_clifford": stats["clifford"]["non_clifford"],
                "width": stats["num_qubits"], "ops": stats["ops"], "stats": stats,
                "qpy_roundtrip_equal": qpy_equal, "qasm_roundtrip_width_ops_equal": qasm_equal,
                "instruction_stream_sha256": sha256(stream), "qasm_sha256": sha256(qasm_path)}
            if not qpy_equal or not qasm_equal:
                raise ValueError(f"{basis} circuit round-trip check failed")
        summary.update(status="ok", stage="complete")
    except Exception as error:
        summary.update(status="error", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary.update(worker_wall_time_s=time.perf_counter() - started,
                       peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(summary_path, summary)
    print(json.dumps({key: summary.get(key) for key in ("case", "repetition", "strategy", "status", *TIMINGS)}), flush=True)
    return 0 if summary["status"] == "ok" else 1


def compare_runs(reference_dir, candidate_dir, tol):
    import numpy as np
    from qiskit import qpy

    old = json.loads((reference_dir / "summary.json").read_text())
    new = json.loads((candidate_dir / "summary.json").read_text())
    comparisons = {"reference": str(reference_dir), "candidate": str(candidate_dir),
                   "fields_equal": {}, "arrays": {}, "basis": {}}
    fields = ("rewrite", "raw_metrics", "rewritten_metrics", "width", "qubit_indexes",
              "raw_subnormalization", "subnormalization", "inverse_subnormalization")
    comparisons["fields_equal"] = {key: old[key] == new[key] for key in fields}
    labels = [json.loads((directory / "rewrite_result.json").read_text())["labels"]
              for directory in (reference_dir, candidate_dir)]
    comparisons["fields_equal"]["labels"] = labels[0] == labels[1]
    with np.load(reference_dir / "rewrite_arrays.npz") as left, np.load(candidate_dir / "rewrite_arrays.npz") as right:
        for field in left.files:
            a, b = left[field], right[field]
            same_shape = a.shape == b.shape
            comparisons["arrays"][field] = {
                "exact_equal": bool(np.array_equal(a, b)), "shape_equal": same_shape,
                "allclose": bool(same_shape and np.allclose(a, b, rtol=0, atol=tol)),
                "max_abs_error": float(np.max(np.abs(a - b), initial=0)) if same_shape else None}
    for basis in BASIS:
        with (reference_dir / f"circuit_{basis}.qpy").open("rb") as source:
            a = qpy.load(source)[0]
        with (candidate_dir / f"circuit_{basis}.qpy").open("rb") as source:
            b = qpy.load(source)[0]
        old_stats, new_stats = old["basis_results"][basis], new["basis_results"][basis]
        comparisons["basis"][basis] = {
            "qpy_circuit_equal": a == b,
            "instruction_stream_exact_equal": old_stats["instruction_stream_sha256"] == new_stats["instruction_stream_sha256"],
            "metric_fields_equal": {key: old_stats[key] == new_stats[key]
                                    for key in ("metric_gates", "non_clifford", "width", "ops")}}
    comparisons["all_exact"] = (all(comparisons["fields_equal"].values())
        and all(value["exact_equal"] for value in comparisons["arrays"].values())
        and all(value["qpy_circuit_equal"] and value["instruction_stream_exact_equal"]
                and all(value["metric_fields_equal"].values()) for value in comparisons["basis"].values()))
    return comparisons


def aggregate(args, manifest, runs):
    comparisons, medians, historical = [], [], []
    for n in args.sizes:
        reference = args.out_root / f"hrw_{n}/rep_1_greedy"
        for run in runs:
            if run["n"] == n:
                directory = Path(run["summary_path"]).parent
                if directory != reference:
                    comparisons.append(compare_runs(reference, directory, args.tol))
        row = {"n": n, "paper_case": n in PAPER_SIZES}
        for strategy in STRATEGIES:
            samples = [run for run in runs if run["n"] == n and run["strategy"] == strategy]
            row[strategy] = {field: statistics.median(run[field] for run in samples) for field in TIMINGS}
            row[strategy]["ranges"] = {field: {"min": min(run[field] for run in samples),
                                              "max": max(run[field] for run in samples)} for field in TIMINGS}
            row[strategy]["samples"] = len(samples)
            row[strategy]["peak_rss_mib_median"] = statistics.median(run["peak_rss_bytes"] / 2**20 for run in samples)
        row["rewrite_speedup_old_over_new"] = row["greedy"]["rewrite_time_s"] / row["greedy_incremental"]["rewrite_time_s"]
        row["ibm_compile_speedup_old_over_new"] = row["greedy"]["ibm_compile_time_s"] / row["greedy_incremental"]["ibm_compile_time_s"]
        reference_summary = json.loads((reference / "summary.json").read_text())
        row["width"] = reference_summary["width"]
        row["subnormalization"] = reference_summary["subnormalization"]
        row["raw_subnormalization"] = reference_summary["raw_subnormalization"]
        row["metrics"] = {basis: {key: reference_summary["basis_results"][basis][key]
                                 for key in ("metric_gates", "non_clifford")} for basis in BASIS}
        medians.append(row)
        source = ROOT / f"Baseline_results/eval_redesign/matrixsum_phase_refresh_20260706/main/hrw/hypercube_random_walk_n{n}/hypercube_random_walk_n{n}_matrixsum_phase_ours_summary.json"
        if n in PAPER_SIZES and source.is_file():
            previous = json.loads(source.read_text())
            checks = {}
            for basis in BASIS:
                old_stats, new_stats = previous["basis_results"][basis], reference_summary["basis_results"][basis]
                checks[basis] = {"gate_count_equal": old_stats["gate_count"] == new_stats["metric_gates"],
                                 "non_clifford_equal": old_stats["non_clifford"] == new_stats["non_clifford"],
                                 "width_equal": old_stats["stats"]["num_qubits"] == new_stats["width"]}
            historical.append({"n": n, "source": str(source), "sha256": sha256(source),
                               "native_metric_comparison": checks, "summary": previous})
    result = {"all_exact": all(item["all_exact"] for item in comparisons), "comparisons": comparisons,
              "medians": medians, "historical_references": historical}
    write_json(args.out_root / "comparison.json", result)
    fields = ("n", "repetition", "strategy", "status", *TIMINGS, "width", "raw_subnormalization",
              "subnormalization", "peak_rss_bytes", "summary_path")
    with (args.out_root / "raw_timings.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: run[key] for key in fields} for run in runs)
    lines = ["# HRW Greedy / Incremental Greedy Comparison", "",
             f"All {len(runs)} fresh sequential worker runs completed. Each size has {args.repetitions} runs per strategy.",
             f"Full trajectory, coefficients, IBM/NAM circuit instructions, gate metrics, width, and subnormalization exactly equal: **{result['all_exact']}**.", "",
             "The published HRW sizes are 4, 8, 12, 20, 28; these are the default comparison sizes.",
             "Order alternates old/new then new/old by repetition. Seeds are fixed, worker threads are one, and each worker imports afresh.",
             "There is no circuit warmup or reuse. Initial coefficients are captured before timed rewrite for later validation.",
             "Imports, model construction, saving, metric counting, and validation are excluded from compile timing.",
             "Rewrite uses the existing helper, including input copying, search, applying its result, and zero elimination.",
             f"IBM/NAM total = rewrite + circuit build + lowering to the named basis. Lowering is opt0, seed {args.seed}; reset is excluded from metric gates.",
             f"Search uses max_steps={args.max_steps}, tol={args.tol:g}, with the original greedy strategy retained as the default.",
             "Values below are medians of separate runs; speedup is old median / new median. No fitted scaling model or statistical significance is claimed.", "",
             "| HRW | Rewrite old (s) | Rewrite new (s) | Rewrite speedup | IBM total old (s) | IBM total new (s) | IBM speedup | NAM total old (s) | NAM total new (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in medians:
        old, new = row["greedy"], row["greedy_incremental"]
        lines.append(f"| {row['n']} | {old['rewrite_time_s']:.6f} | {new['rewrite_time_s']:.6f} | {row['rewrite_speedup_old_over_new']:.3f}x | {old['ibm_compile_time_s']:.6f} | {new['ibm_compile_time_s']:.6f} | {row['ibm_compile_speedup_old_over_new']:.3f}x | {old['nam_compile_time_s']:.6f} | {new['nam_compile_time_s']:.6f} |")
    lines += ["", "| HRW | Strategy | Rewrite min-max (s) | IBM total min-max (s) | NAM total min-max (s) |",
              "|---|---|---:|---:|---:|"]
    for row in medians:
        for strategy in STRATEGIES:
            ranges = row[strategy]["ranges"]
            fields = ("rewrite_time_s", "ibm_compile_time_s", "nam_compile_time_s")
            lines.append(f"| {row['n']} | {strategy} | " + " | ".join(
                f"{ranges[key]['min']:.6f} - {ranges[key]['max']:.6f}" for key in fields) + " |")
    lines += ["", "| HRW | Build old/new (s) | IBM lowering old/new (s) | NAM lowering old/new (s) | RSS old/new (MiB) |",
              "|---|---:|---:|---:|---:|"]
    for row in medians:
        old, new = row["greedy"], row["greedy_incremental"]
        fields = ("circuit_build_time_s", "ibm_lowering_time_s", "nam_lowering_time_s", "peak_rss_mib_median")
        lines.append(f"| {row['n']} | " + " | ".join(f"{old[key]:.6f} / {new[key]:.6f}" for key in fields) + " |")
    lines += ["", "| HRW | Width | IBM gates / non-Clifford | NAM gates / non-Clifford | Raw / rewritten subnormalization |",
              "|---|---:|---:|---:|---:|"]
    for row in medians:
        ibm, nam = row["metrics"]["ibm"], row["metrics"]["nam"]
        lines.append(f"| {row['n']} | {row['width']} | {ibm['metric_gates']} / {ibm['non_clifford']} | {nam['metric_gates']} / {nam['non_clifford']} | {row['raw_subnormalization']:.12g} / {row['subnormalization']:.12g} |")
    lines += ["", "Subnormalization is freshly measured as sum_j ||K_j||_Pauli,1^2; its inverse is recorded separately. Historical summaries do not record this field.",
              "Coefficient arrays are saved losslessly in NPZ, including the full replayed state at every accepted step. Replay checks occur outside timed search.",
              f"Each of the other {2 * args.repetitions - 1} runs per size is compared against old repetition 1, including every chosen row pair, 2x2 unitary, support value, and coefficient state.",
              "Native QPY equality and canonical instruction streams include global phase; QASM loading checks additionally confirm width and operation counts.",
              "No statevector/density-matrix simulation or external optimizer is run. Circuit equality here compares the two strategies; it is not an independent channel-semantic proof.", "",
              "| HRW | Published width | Published IBM gates / non-Clifford | Published NAM gates / non-Clifford | Fresh native metrics match |",
              "|---|---:|---:|---:|---|"]
    for item in historical:
        ibm, nam = item["summary"]["basis_results"]["ibm"], item["summary"]["basis_results"]["nam"]
        equal = all(value for checks in item["native_metric_comparison"].values() for value in checks.values())
        lines.append(f"| {item['n']} | {ibm['stats']['num_qubits']} | {ibm['gate_count']} / {ibm['non_clifford']} | {nam['gate_count']} / {nam['non_clifford']} | {equal} |")
    lines += ["", "Historical native counts above come from the matrixsum_phase_refresh_20260706 full-channel RQ1 summaries; source paths and hashes are retained in comparison.json.",
              "Historical times are not included in the new/old speedup calculation. Any historical metric mismatch is retained explicitly rather than replacing published data.", "",
              f"Environment: {runs[0]['environment']['platform']}; Python {platform.python_version()}; Qiskit {runs[0]['environment']['qiskit']}; NumPy {runs[0]['environment']['numpy']}. Package freeze and per-worker environment metadata are saved in the result root.",
              f"Git HEAD: `{manifest['git_head']}` with a dirty worktree. The source archive records {len(manifest['source_sha256'])} files, not the full repository or installed environment.",
              "The original greedy search and its coefficient/candidate/application helpers are AST-equal to the archived pre-change compiler. Both strategies run from the same new source snapshot.",
              f"Source archive SHA256: `{manifest['source_archive_sha256']}`.",
              f"[Raw runs and artifacts]({args.out_root}) | [Numeric comparison]({args.out_root / 'comparison.json'}) | [Manifest]({args.out_root / 'manifest.json'})", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(PAPER_SIZES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out_root, args.report = args.out_root.resolve(), args.report.resolve()
    if args.worker:
        return run_worker(args)
    if args.repetitions < 3 or args.max_steps <= max(args.sizes) or len(set(args.sizes)) != len(args.sizes) or min(args.sizes) < 2:
        parser.error("Require >=3 repetitions, distinct sizes >=2, and max_steps greater than every size")
    if args.tol <= 0 or not args.tol < float("inf"):
        parser.error("Tolerance must be finite and positive")
    args.out_root.mkdir(parents=True, exist_ok=False)
    manifest = snapshot_sources(args.out_root)
    manifest.update(started_at_utc=datetime.now(timezone.utc).isoformat(), sizes=args.sizes,
                    repetitions=args.repetitions, strategies=STRATEGIES, max_steps=args.max_steps,
                    seed=args.seed, tol=args.tol, thread_environment=THREAD_ENV,
                    timing_definition="rewrite helper + circuit build + basis lowering; imports/model/export/count/validation excluded",
                    execution_order=[], status="running", report=str(args.report))
    write_json(args.out_root / "manifest.json", manifest)
    environment = dict(os.environ, **THREAD_ENV)
    for key in tuple(environment):
        if key.startswith("CHANNEL_LCU_"):
            del environment[key]
    runs = []
    for n in args.sizes:
        for repetition in range(1, args.repetitions + 1):
            order = STRATEGIES if repetition % 2 else tuple(reversed(STRATEGIES))
            for strategy in order:
                worker = f"{n}:{repetition}:{strategy}"
                log_path = args.out_root / f"hrw_{n}_rep_{repetition}_{strategy}.log"
                manifest["execution_order"].append(worker)
                write_json(args.out_root / "manifest.json", manifest)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", worker,
                           "--out-root", str(args.out_root), "--max-steps", str(args.max_steps),
                           "--seed", str(args.seed), "--tol", str(args.tol)]
                print(f"Running HRW-{n} repetition {repetition}: {strategy}", flush=True)
                started = time.perf_counter()
                with log_path.open("w", encoding="utf-8") as output:
                    process = subprocess.run(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
                summary_path = args.out_root / f"hrw_{n}/rep_{repetition}_{strategy}/summary.json"
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "initialization_error"}
                summary.update(process_returncode=process.returncode, process_wall_time_s=time.perf_counter() - started,
                               summary_path=str(summary_path), log_path=str(log_path))
                write_json(summary_path, summary)
                runs.append(summary)
                if process.returncode or summary["status"] != "ok":
                    manifest.update(status="error", failed_worker=worker)
                    write_json(args.out_root / "manifest.json", manifest)
                    raise RuntimeError(f"Worker failed; see {log_path}")
    if source_hashes() != manifest["source_sha256"]:
        raise RuntimeError("Source changed during the measured batch")
    result = aggregate(args, manifest, runs)
    manifest.update(status="ok" if result["all_exact"] else "comparison_mismatch",
                    finished_at_utc=datetime.now(timezone.utc).isoformat(), all_exact=result["all_exact"],
                    completed_runs=len(runs), environment=runs[0]["environment"],
                    source_hashes_match_at_finish=True)
    write_json(args.out_root / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "runs": len(runs), "report": str(args.report)}), flush=True)
    return 0 if result["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
