"""Add whole-circuit IBM opt3 to the seven frozen IBM opt0 measurements.

Run an archive smoke test into OUT first. The original opt0 measurements and
their runner stay unchanged. No TFIM VCBE or Stinespring jobs are started.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import gc
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Baseline_results/rq1_tfim_dqe_petz_ibm_opt3_20261014"
BASE = ROOT / "Baseline_results/rq1_tfim_dqe_petz_ibm_opt0_20261014"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["QISKIT_PARALLEL"] = "FALSE"
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / name) for name in ("src", "Baseline_scripts", "scripts/eval_redesign")]

import numpy as np
import psutil
import qiskit
from qiskit import QuantumCircuit, qasm2
from experiment_protocol import source_identity, transpile_channel_circuit
import run_rq1_tfim_dqe_petz_ibm_opt0 as base
from run_rq1_hrw_ibm_opt3 import digest, dump, counts, require_identity_layout
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")

CASES = base.CASES
WORKERS = [(c, m) for c, m in base.WORKERS]
IBM, SEED = base.IBM, base.SEED
BUDGET_S, RSS_BUDGET = 600, 4 * 1024 ** 3
HELPERS = [Path(base.__file__), Path(base.__file__).with_name("run_rq1_hrw_ibm_opt3.py")]


def row_path(case, method):
    return OUT / "cases" / case / f"{method}_opt3_results.json"


def progress(case, method, stage):
    dump(OUT / "progress.json", dict(case=case, method=method, stage=stage, pid=os.getpid()))
    print(case, method, stage, flush=True)


def worker(case, method):
    manifest = base.read(OUT / "manifest.json")
    assert source_identity() == manifest["sources"]
    assert digest(__file__) == manifest["runner_sha256"]
    for path, value in manifest["helper_sha256"].items():
        assert digest(path) == value
    previous_path = BASE / "cases" / case / f"{method}_results.json"
    previous = base.read(previous_path)
    for path in (previous_path, Path(previous["qasm_path"])):
        assert digest(path) == manifest["files"][str(path)]
    assert previous["status"] == "ok" and previous["basis"] == "IBM"
    assert previous["lowering_optimization_level"] == 0 and previous["optimizer_scope"] == "none"
    assert previous["independent_qasm_recount"].startswith("passed")
    progress(case, method, "load_frozen_ibm_opt0_qasm")
    started = time.perf_counter()
    source = qasm2.load(previous["qasm_path"])
    load_s = time.perf_counter() - started
    assert counts(source, IBM) == (previous["G"], previous["counts"])
    assert source.num_qubits == previous["W"]
    progress(case, method, "whole_circuit_ibm_opt3")
    started = time.perf_counter()
    circuit = transpile_channel_circuit(source, basis_gates=IBM, optimization_level=3,
        seed_transpiler=SEED, approximation_degree=1.0, qubits_initially_zero=False, num_processes=1)
    optimizer_s = time.perf_counter() - started
    del source
    gc.collect()
    require_identity_layout(circuit)
    assert circuit.num_qubits == previous["W"]
    G, operations = counts(circuit, IBM)
    progress(case, method, "save_ibm_opt3_qasm")
    directory = OUT / "cases" / case
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{method}_opt3_ibm.qasm"
    flat = QuantumCircuit(circuit.num_qubits)
    flat.compose(circuit, inplace=True)
    phase = float(flat.global_phase)
    qasm2.dump(flat, path)
    del circuit, flat
    gc.collect()
    progress(case, method, "independent_qasm_recount")
    total, artifact_operations, width, nc = base.recount_qasm(path)
    assert (total, artifact_operations, width) == (G, operations, previous["W"])
    row = dict(previous)
    row.update(method=method + "+Qiskit opt3", G=G, counts=operations,
        CX=operations.get("cx", 0), non_clifford=nc, Lambda_G=previous["Lambda"] * G,
        Lambda_G_display=format(Decimal(previous["Lambda_display"]) * G, "f"),
        lowering_optimization_level=3, optimizer_scope="whole assembled IBM opt0 circuit",
        optimizer_s=optimizer_s, lowering_s=previous["lowering_s"],
        lowering_s_scope="input IBM opt0 lowering, copied from baseline row; separate from optimizer_s",
        input_qasm_load_s=load_s, source_load_or_reassembly_s=load_s,
        measurement_time_scope="optimizer_s measures whole-circuit IBM opt3 only; imports, QASM loading, output I/O, classification and validation excluded",
        input_G=previous["G"], input_baseline_result_path=str(previous_path),
        input_baseline_result_sha256=digest(previous_path), input_qasm_path=previous["qasm_path"],
        input_qasm_sha256=previous["qasm_sha256"], qasm_path=str(path), qasm_sha256=digest(path),
        global_phase_rad=phase, nam_reproduction="inherited baseline source histogram; not NAM-native opt3",
        independent_qasm_recount="passed G, NC, counts and width",
        validation_scope=previous["validation_scope"] + "; exact opt3 configuration and identity layout; full optimized-channel Choi skipped",
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    assert source_identity() == manifest["sources"]
    dump(row_path(case, method), row)
    print("RESULT", case, method, "G", G, "optimizer_s", optimizer_s, flush=True)


def create_manifest():
    smoke = base.read(OUT / "smoke.json")
    baseline = base.read(BASE / "results.json")
    assert smoke["passed"] and smoke["sources"] == source_identity()
    assert baseline["status"] == "complete" and len(baseline["rows"]) == 7
    assert base.read(BASE / "manifest.json")["sources"] == source_identity()
    assert not (OUT / "manifest.json").exists(), "Choose a fresh measurement batch"
    files = {str(BASE / name): digest(BASE / name) for name in ("manifest.json", "results.json")}
    for row in baseline["rows"]:
        for path in (BASE / "cases" / row["case"] / f"{row['method']}_results.json", Path(row["qasm_path"])):
            files[str(path)] = digest(path)
    dump(OUT / "manifest.json", dict(cases=CASES, workers=WORKERS, sources=source_identity(),
        runner_sha256=digest(__file__), helper_sha256={str(p): digest(p) for p in HELPERS},
        files=files, base_batch=str(BASE), basis_gates=IBM, downstream_optimization_level=3,
        optimization_scope="whole assembled IBM opt0 channel circuit for both Raw and VCBE",
        seed_transpiler=SEED, approximation_degree=1.0, qubits_initially_zero=False,
        numerical_threads=1, num_processes=1, full_choi_status="skipped_user_request",
        worker_budget_s=BUDGET_S, worker_rss_budget_bytes=RSS_BUDGET,
        environment=dict(python=sys.version, qiskit=qiskit.__version__, numpy=np.__version__)))


def export(state):
    baseline = base.read(BASE / "results.json")["rows"]
    optimized = [base.read(row_path(c, m)) for c, m in WORKERS if row_path(c, m).exists()]
    rows = baseline + optimized
    order = {"Raw": 0, "Raw+Qiskit opt3": 1, "VCBE": 2, "VCBE+Qiskit opt3": 3}
    rows.sort(key=lambda r: (CASES.index(r["case"]), order[r["method"]]))
    dump(OUT / "results.json", dict(status=state["status"], rows=rows,
        optimized_rows=optimized, workers=state["workers"], manifest_path=str(OUT / "manifest.json")))
    fields = ["case", "method", "status", "basis", "G", "Lambda_display", "Lambda_G_display", "W",
        "CX", "non_clifford", "lowering_s", "optimizer_s", "measurement_time_scope", "optimizer_scope",
        "reference_NAM_G", "historical_compile_s", "historical_unique_fit_s", "reused_vcbe_fit",
        "target_kraus_sha256", "canonical_map_coefficient_gram_relative_error", "full_choi_status",
        "nam_reproduction", "independent_qasm_recount", "validation_scope", "input_qasm_path",
        "input_qasm_sha256", "qasm_path", "qasm_sha256", "original_result_path"]
    with (ROOT / "Results/rq1_tfim_dqe_petz_ibm_opt3_20261014.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format(Decimal(str(v)), "f") if isinstance(v, float) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--method", choices=("Raw", "VCBE"))
    args = parser.parse_args()
    if args.case:
        assert (args.case, args.method) in WORKERS
        worker(args.case, args.method)
        return
    create_manifest()
    state, process = dict(status="running", workers=[], current=None), None
    def terminate():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    def stop(signum, frame):
        terminate()
        state.update(status="paused_user_request", current=None)
        dump(OUT / "status.json", state)
        export(state)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for case, method in WORKERS:
        started, peak, reason = time.monotonic(), 0, None
        log_path = OUT / f"{case}_{method}_opt3.log"
        with log_path.open("w") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--case", case,
                "--method", method], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state["current"] = dict(case=case, method=method, pid=process.pid)
            dump(OUT / "status.json", state)
            try:
                while process.poll() is None:
                    try:
                        peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                    except (psutil.Error, PermissionError):
                        pass
                    if peak > RSS_BUDGET:
                        reason = "memory_limit"
                    elif time.monotonic() - started > BUDGET_S:
                        reason = "timeout"
                    if reason:
                        terminate()
                        break
                    time.sleep(0.5)
            finally:
                terminate()
                code = process.wait()
        measured = dict(case=case, method=method, exit_code=code, termination_reason=reason,
            wall_s=time.monotonic() - started, peak_rss_bytes=peak, log_path=str(log_path))
        state["workers"].append(measured)
        if code:
            old = base.read(BASE / "cases" / case / f"{method}_results.json")
            dump(row_path(case, method), dict(case=case, method=method + "+Qiskit opt3",
                status=reason or "worker_failed", basis="IBM", G=None, Lambda_display=old["Lambda_display"],
                Lambda_G_display=None, W=old["W"], optimizer_scope="whole assembled IBM opt0 circuit",
                input_qasm_path=old["qasm_path"], input_qasm_sha256=old["qasm_sha256"],
                full_choi_status="skipped_user_request", worker=measured))
        dump(OUT / "status.json", state)
        export(state)
        print(measured, flush=True)
    state.update(status="complete" if all(w["exit_code"] == 0 for w in state["workers"])
                 else "complete_with_failures", current=None)
    assert source_identity() == base.read(OUT / "manifest.json")["sources"]
    export(state)
    assert len(base.read(OUT / "results.json")["rows"]) == 14
    dump(OUT / "status.json", state)
    print("Completed IBM with/without opt3 comparison:", state["status"], flush=True)


if __name__ == "__main__":
    main()
