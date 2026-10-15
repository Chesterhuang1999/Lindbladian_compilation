"""Fresh VCBE fits of raw Kraus inputs, sharing only outer LCU assembly.

Historical fitted circuits and rewritten Kraus families are never loaded.
Outputs are isolated from the existing RQ1 tables and experiment batches.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import psutil
from qiskit import qpy
from qiskit.quantum_info import Operator
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_IMPLEMENTATION = ROOT / "Baseline_results/matrix_synthesis_hrw_tfim_20261012"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts"),
               str(HISTORICAL_IMPLEMENTATION), str(Path(__file__).parent)]
from channel_IR import channel as ChannelIR
from experiment_protocol import source_identity
from rq1_pipeline import deserialize_kraus, input_identity, serialize_kraus
from run_rq1_vcbe_scaling import (check_adjoint, compile_method, digest, dump,
                                ReversibleAdjointAnsatz, save_circuit)
import run_rq1_vcbe_scaling as assembly_implementation
from vcbe_probe import StopSearch
from least_squares_precision import residual_jacobian

CASES = ("HRW-4", "HRW-8", "HRW-12", "RepDQE-d3", "Petz_Ising3")
METHODS = ("vcbe_source_tree", "vcbe_preopt3_tree")
TOL = 1e-10


def forbid_rewrite(*args, **kwargs):
    raise AssertionError("Kraus rewrite is prohibited in this experiment")


def raw_ensemble(channels):
    # The public ensemble constructor drops branch global phases. Preserve the
    # original targets here; channel_to_LCU only needs its channels attribute.
    assert len(channels) == 1
    return SimpleNamespace(channels=[(1.0, channels[0])])


def load_raw(name):
    if name.startswith("HRW-"):
        if name == "HRW-8":
            path = ROOT / "Baseline_results/table1_scaling_ours_20261013/cases" / name / "input.json"
        else:
            path = ROOT / "paper_experiments/pending_review/tfim_hrw_full_20261011/cases" / name / "input.json"
    elif name == "RepDQE-d3":
        path = ROOT / "paper_experiments/pending_review/remaining_full_20261011/cases" / name / "input.json"
    else:
        path = ROOT / "paper_experiments/pending_review/raw_baseline_20261012/inputs/Petz_Ising3.json"
    document = json.loads(path.read_text())
    kraus = deserialize_kraus(document["input_kraus"])
    assert serialize_kraus(kraus) == document["input_kraus"]
    return kraus, dict(input_path=str(path), input_file_sha256=digest(path),
                       raw_kraus_sha256=input_identity(kraus), model=document["model"])


def active_target(ms):
    """Remove only exact spectator identities; retain every Kraus and coefficient."""
    active = [q for q in range(ms.size)
              if any(atom.expr[ms.size - 1 - q] != "I" for atom, _ in ms.instances)]
    assert active
    encoded = serialize_kraus([ms])[0]
    local = deserialize_kraus([[["".join(label[ms.size - 1 - q] for q in active[::-1]),
                                magnitude, real, imag]
                               for label, magnitude, real, imag in encoded]])[0]
    alpha = float(ms.pauli_norm())
    assert np.isclose(local.pauli_norm(), alpha, rtol=1e-14, atol=0)
    return (np.asarray(local.eff_op().to_matrix(), complex) / alpha,
            active, float(np.sqrt(2 ** (ms.size - len(active)))))


def fit_target(target, spectator_factor, directory, args):
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "target.npy", target)
    n = int(np.log2(len(target)))
    real = bool(np.max(np.abs(target.imag)) < 1e-14)
    desired = 1e-12 / spectator_factor
    started = time.perf_counter()
    best = {"error": None}
    trials = []
    evaluations = 0

    for depth in (1, 2, 4, 8, 16, 32):
        ansatz = ReversibleAdjointAnsatz(n, depth, real)
        for seed in (0, 1):
            if time.perf_counter() - started >= args.fit_budget:
                break
            trial_started = time.perf_counter()
            local = {"error": None, "evaluations": 0}
            initial = np.random.default_rng(20261012 + 1000 * depth + seed).uniform(
                -np.pi, np.pi, ansatz.num_params)

            def remember(theta, error):
                if local["error"] is None or error < local["error"]:
                    local.update(error=error, parameters=theta.copy())
                if best["error"] is None or error < best["error"]:
                    best.update(error=error, layers=depth, seed=seed, parameters=theta.tolist())

            def objective(theta):
                nonlocal evaluations
                if time.perf_counter() - started >= args.fit_budget:
                    raise StopSearch("fit_time_budget")
                if time.perf_counter() - trial_started >= args.trial_budget:
                    raise StopSearch("trial_time_budget")
                error, gradient = ansatz.evaluate(theta, target)
                evaluations += 1
                local["evaluations"] += 1
                remember(theta, error)
                if error <= desired:
                    raise StopSearch("error_threshold_met")
                return error, gradient

            try:
                result = minimize(objective, initial, method="BFGS", jac=True,
                                  options={"gtol": 1e-8, "maxiter": 1500})
                stop = str(result.message)
            except StopSearch as exc:
                stop = str(exc)
            bfgs_seconds = time.perf_counter() - trial_started

            # Refine this fresh fit in the same ansatz, with no reference circuit.
            polish_started = time.perf_counter()
            polish_stop = "not_needed"
            polish_evaluations = 0
            cache = {}
            if local["error"] is not None and desired < local["error"] < 0.03:
                def evaluate(theta):
                    nonlocal polish_evaluations
                    if time.perf_counter() - started >= args.fit_budget:
                        raise StopSearch("fit_time_budget")
                    if time.perf_counter() - polish_started >= args.polish_budget:
                        raise StopSearch("polish_time_budget")
                    if "theta" in cache and np.array_equal(theta, cache["theta"]):
                        return cache["residual"], cache["jacobian"]
                    residual, jacobian = residual_jacobian(ansatz, theta, target)
                    error = float(np.linalg.norm(residual))
                    polish_evaluations += 1
                    remember(theta, error)
                    cache.update(theta=theta.copy(), residual=residual, jacobian=jacobian)
                    if error <= desired:
                        raise StopSearch("error_threshold_met")
                    return residual, jacobian
                try:
                    result = least_squares(lambda x: evaluate(x)[0], local["parameters"],
                        jac=lambda x: evaluate(x)[1], ftol=None, gtol=None, xtol=1e-15,
                        max_nfev=100)
                    polish_stop = str(result.message)
                except StopSearch as exc:
                    polish_stop = str(exc)
            trial = dict(layers=depth, seed=seed, parameters=ansatz.num_params,
                         best_error=local["error"], bfgs_stop=stop,
                         bfgs_s=bfgs_seconds, bfgs_evaluations=local["evaluations"],
                         polish_stop=polish_stop, polish_evaluations=polish_evaluations,
                         polish_s=time.perf_counter() - polish_started)
            trials.append(trial)
            dump(directory / "progress.json", dict(best=best, trials=trials,
                 elapsed_s=time.perf_counter() - started))
            print("FIT_TRIAL", directory.name, json.dumps(trial), flush=True)
            if best["error"] is not None and best["error"] <= desired:
                break
        if best["error"] is not None and best["error"] <= desired:
            break
        if time.perf_counter() - started >= args.fit_budget:
            break

    assert best["error"] is not None
    circuit = ReversibleAdjointAnsatz(n, best["layers"], real).circuit(np.array(best["parameters"]))
    error = float(np.linalg.norm(Operator(circuit).data[:len(target), :len(target)] - target))
    assert abs(error - best["error"]) < 1e-11
    seconds = time.perf_counter() - started
    record = dict(**best, n=n, real_ansatz=real, fit_error=error,
        status="success" if error * spectator_factor < TOL else
               "fit_budget_exhausted" if seconds >= args.fit_budget else "fit_search_exhausted",
        total_search_s=seconds, newly_measured_fit_s=seconds, cached=False,
        historical_fit_reused=False, fit_budget_s=args.fit_budget,
        per_bfgs_trial_budget_s=args.trial_budget, per_polish_budget_s=args.polish_budget,
        acceptance_tolerance_full_embedded_frobenius=TOL, active_fitting_target=desired,
        target_file_sha256=digest(directory / "target.npy"), trials=trials,
        bfgs_evaluations=evaluations, template=directory.name,
        precision_method="analytic-Jacobian least squares after fresh BFGS, same ansatz")
    dump(directory / "results.json", record)
    save_circuit(directory / "source.qpy", circuit)
    return circuit, record


def verify_channel(directory, method, registers, kraus):
    # These three small instances permit all system columns and a complete Choi.
    import importlib.util
    path = ROOT / "Baseline_results/vcbe_channel_smoke_20261012/run_smoke.py"
    spec = importlib.util.spec_from_file_location("vcbe_small_channel_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (directory / f"{method}_nam_opt0.qpy").open("rb") as stream:
        circuit = qpy.load(stream)[0]
    return module.verify(circuit, registers, kraus)


def worker(args):
    directory = args.out_dir / "cases" / args.case
    directory.mkdir(parents=True, exist_ok=False)
    smoke = json.loads((args.out_dir / "smoke.json").read_text())
    assert smoke["passed"] and smoke["sources"] == source_identity()
    for name in ("rewrite_search", "gatecost_rewrite_search", "apply_rewrite_result"):
        setattr(ChannelIR, name, forbid_rewrite)
    assembly_implementation.channel_ensemble = raw_ensemble
    started = time.perf_counter()
    kraus, provenance = load_raw(args.case)
    initial_hash = input_identity(kraus)
    row = dict(case=args.case, status="started", n=kraus[0].size, M=len(kraus),
        kraus=serialize_kraus(kraus), raw_kraus_sha256=initial_hash, provenance=provenance,
        sources=source_identity(), rewrite=dict(enabled=False, steps=0, runtime_guard=True),
        canonicalization="disabled: no Kraus merging, rotations, phase changes or reordering",
        ensemble_adapter="preserve raw Kraus phases; bypass constructor phase canonicalization",
        alpha=[float(ms.pauli_norm()) for ms in kraus],
        Lambda=float(sum(ms.pauli_norm() ** 2 for ms in kraus)),
        shared=["outer PREPARE construction", "outer Kraus unary-iteration tree assembly"],
        normalization="Pauli 1-norm of each original Kraus, computed from raw input",
        cached_fit=False, methods={}, vcbe_blocks=[])
    dump(directory / "results.json", row)
    unique = []
    blocks, details = [], []
    for index, ms in enumerate(kraus):
        target, active, spectator = active_target(ms)
        matched = next((item for item in unique if np.array_equal(item[0], target)), None)
        if matched is None:
            block, info = fit_target(target, spectator, directory / f"fit_{index:02d}", args)
            unique.append((target.copy(), block.copy(), info))
            reused_in_batch = False
        else:
            _, block, info = matched
            block = block.copy()
            reused_in_batch = True
        error = float(np.linalg.norm(Operator(block).data[:len(target), :len(target)] - target))
        detail = dict(info, index=index, alpha=float(ms.pauli_norm()),
            active_wires=active, embedding_wires=active + [ms.size],
            preprocessing="exact spectator identity removal only; original Kraus preserved",
            reused_identical_raw_target_in_this_batch=reused_in_batch,
            active_frobenius_block_error=error, full_embedded_frobenius_block_error=error * spectator,
            spectator_frobenius_factor=spectator, block_qualified=error * spectator < TOL)
        blocks.append((block, active + [ms.size], target))
        details.append(detail)
        row["vcbe_blocks"] = details
        dump(directory / "results.json", row)
        print("BLOCK", args.case, index, detail["block_qualified"], error * spectator, flush=True)
    row["unique_raw_targets"] = len(unique)
    row["fresh_unique_fit_s"] = sum(item[2]["total_search_s"] for item in unique)
    for method in METHODS:
        measured = compile_method(kraus, blocks, details, method, directory, row["Lambda"])
        metric = measured["metrics"]["nam_opt0"]
        if measured["status"] == "ok" and kraus[0].size <= 4:
            verification = verify_channel(directory, method, measured["registers"], kraus)
            measured["verification"] = verification
            metric.update(full_choi_status="passed" if verification["passed"] else "failed",
                full_choi_relative_error=verification["relative_choi_frobenius_error"],
                validation_scope=verification["method"],
                status="ok" if verification["passed"] else "channel_validation_failed")
            measured["status"] = metric["status"]
        elif measured["status"] == "ok":
            metric["full_choi_status"] = "skipped_scaling_no_full_channel_matrix"
            metric["validation_scope"] = "all active blocks, exact spectator embedding and shared outer tree; no full compiled-channel Choi"
        row["methods"][method] = measured
        dump(directory / "results.json", row)
    assert input_identity(kraus) == initial_hash
    assert digest(provenance["input_path"]) == provenance["input_file_sha256"]
    row["raw_kraus_unchanged"] = True
    row["production_sources_unchanged"] = source_identity() == row["sources"]
    row["total_elapsed_s"] = time.perf_counter() - started
    row["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    row["status"] = "success" if all(m["status"] == "ok" for m in row["methods"].values()) else "unqualified_fit_or_validation"
    dump(directory / "results.json", row)
    print("DONE", args.case, row["status"], row["total_elapsed_s"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Baseline_results/vcbe_no_rewrite_20261014")
    parser.add_argument("--hrw", nargs="+", choices=("HRW-4", "HRW-8", "HRW-12"),
                        default=["HRW-4", "HRW-8", "HRW-12"])
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--fit-budget", type=float, default=600)
    parser.add_argument("--trial-budget", type=float, default=50)
    parser.add_argument("--polish-budget", type=float, default=20)
    parser.add_argument("--rss-budget-gib", type=float, default=4)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.case:
        worker(args)
        return
    assert not (args.out_dir / "results.json").exists(), "Use a new output directory for a new run"
    cases = tuple(dict.fromkeys(args.hrw)) + ("RepDQE-d3", "Petz_Ising3")
    record = dict(cases={}, selected_cases=cases, sources=source_identity(),
        runner_sha256=digest(__file__), gradient_checks=check_adjoint(),
        vcbe="independent generic Fig.1/Eq.(7) reproduction, arXiv:2507.17658v2",
        implementation_sources={str(p.relative_to(ROOT)): digest(p) for p in
            (Path(__file__), Path(__file__).with_name("run_rq1_vcbe_scaling.py"),
             HISTORICAL_IMPLEMENTATION / "vcbe_probe.py", HISTORICAL_IMPLEMENTATION / "least_squares_precision.py",
             ROOT / "Baseline_results/vcbe_channel_smoke_20261012/run_smoke.py")},
        rewrite_enabled=False, historical_fit_reused=False,
        fitting_recipe=dict(layers=[1, 2, 4, 8, 16, 32], seeds=[0, 1], method="BFGS",
            gtol=1e-8, maxiter=1500, polish="same-ansatz analytic-Jacobian least squares",
            per_unique_raw_target_budget_s=args.fit_budget, per_bfgs_trial_budget_s=args.trial_budget,
            per_polish_budget_s=args.polish_budget),
        environment=dict(python=sys.version, platform=platform.platform(),
            packages={name: version(name) for name in ("qiskit", "numpy", "scipy", "qiskit-aer")}),
        tolerance=TOL, basis="NAM", final_optimization_level=0, seed_transpiler=20261009,
        qubits_initially_zero=False, rss_budget_bytes=int(args.rss_budget_gib * 1024 ** 3))
    dump(args.out_dir / "results.json", record)
    for case in cases:
        command = [sys.executable, "-B", __file__, "--case", case, "--out-dir", str(args.out_dir),
            "--fit-budget", str(args.fit_budget), "--trial-budget", str(args.trial_budget),
            "--polish-budget", str(args.polish_budget)]
        log_path = args.out_dir / f"{case}.log"
        stop_reason = None
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            monitor = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    rss = monitor.memory_info().rss
                    if rss > record["rss_budget_bytes"]:
                        stop_reason = "rss_budget_exhausted"
                        process.terminate()
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.5)
            process.wait()
        path = args.out_dir / "cases" / case / "results.json"
        measured = json.loads(path.read_text()) if path.exists() else {"status": "worker_failed"}
        measured.update(worker_exit_code=process.returncode, runner_stop_reason=stop_reason, log_path=str(log_path))
        record["cases"][case] = measured
        dump(args.out_dir / "results.json", record)
        print("CASE_DONE", case, measured["status"], process.returncode, flush=True)
    fields = ["case", "method", "status", "n", "M", "G", "Lambda", "Lambda_G", "W", "CX", "NC",
              "fresh_unique_fit_s", "assembly_plus_lowering_s", "full_choi_status", "full_choi_relative_error",
              "rewrite_enabled", "historical_fit_reused", "raw_kraus_sha256", "qasm_path", "qasm_sha256"]
    with (args.out_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case, row in record["cases"].items():
            for method, measured in row.get("methods", {}).items():
                writer.writerow(dict(measured["metrics"]["nam_opt0"], case=case, method=method,
                    n=row["n"], M=row["M"], fresh_unique_fit_s=row["fresh_unique_fit_s"],
                    rewrite_enabled=False, historical_fit_reused=False, raw_kraus_sha256=row["raw_kraus_sha256"]))


if __name__ == "__main__":
    main()
