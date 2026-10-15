"""IBM opt0 only: TFIM-8/12/20 Raw and RepDQE/Petz Raw plus raw-Kraus VCBE.

Reuse frozen input Kraus representations and qualified VCBE fits. Never run
Kraus rewrite, Qiskit opt3, Stinespring, or a full-system channel matrix.
Run the archive smoke test into OUT first, then execute in qiskit_simulate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import gc
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Baseline_results/rq1_tfim_dqe_petz_ibm_opt0_20261014"
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
from qiskit import QuantumCircuit, qasm2, qpy
from baseline_common import (QasmOperation, is_ibm_operation_clifford,
    parse_qasm_operation, strip_qasm_comment, QREG2_RE)
from channel_IR import channel as ChannelIR
from experiment_protocol import source_identity, transpile_channel_circuit
from rq1_pipeline import build_pipeline, deserialize_kraus, input_identity
from run_rq1_hrw_ibm_opt3 import (digest, dump, counts, map_error, local_matrix,
                                 save_qpy, require_identity_layout)
# The imported helper establishes its historical cache path; use this batch's
# cache for the new workers. The completed historical runner is unchanged.
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")

CASES = ("TFIM-8", "TFIM-12", "TFIM-20", "RepDQE-d3", "Petz_Ising3")
SMALL = ("RepDQE-d3", "Petz_Ising3")
NAM, IBM = ["rz", "h", "x", "cx"], ["u1", "u2", "u3", "cx"]
SEED, TOL = 20261009, 1e-10
WORKERS = [(case, "Raw") for case in SMALL] + [(case, "VCBE") for case in SMALL]
WORKERS += [(case, "Raw") for case in CASES if case not in SMALL]
BUDGET_S, RSS_BUDGET = 600, 4 * 1024 ** 3
HELPER = Path(__file__).with_name("run_rq1_hrw_ibm_opt3.py")


def read(path):
    return json.loads(Path(path).read_text())


def paths(case):
    suite = ROOT / "paper_experiments/pending_review"
    raw = suite / "rq1_paper_suite_20261010/full_v2/runs" / case / "raw"
    if case in ("TFIM-20", "Petz_Ising3"):
        raw = suite / "raw_baseline_20261012/cases" / case / case / "raw"
    if case.startswith("TFIM-"):
        target = suite / "tfim_hrw_full_20261011/cases" / case / "input.json"
    elif case == "RepDQE-d3":
        target = suite / "remaining_full_20261011/cases" / case / "input.json"
    else:
        target = suite / "raw_baseline_20261012/inputs/Petz_Ising3.json"
    return raw, target, ROOT / "Baseline_results/vcbe_no_rewrite_20261014/cases" / case


def progress(case, method, stage):
    dump(OUT / "progress.json", dict(case=case, method=method, stage=stage, pid=os.getpid()))
    print(case, method, stage, flush=True)


def lower(circuit, basis=IBM):
    return transpile_channel_circuit(circuit, basis_gates=basis, optimization_level=0,
        seed_transpiler=SEED, approximation_degree=1.0, qubits_initially_zero=False, num_processes=1)


def local_checks(record, folder):
    checks = []
    for template in sorted({b["template"] for b in record["vcbe_blocks"]}):
        target = np.load(folder / template / "target.npy")
        with (folder / template / "source.qpy").open("rb") as stream:
            source = qpy.load(stream)[0]
        converted = lower(source)
        require_identity_layout(converted)
        matrix = local_matrix(converted)
        d = len(target)
        active_error = float(np.linalg.norm(matrix[:d, :d] - target))
        unitary_error = float(np.linalg.norm(matrix - local_matrix(source)))
        references = [b for b in record["vcbe_blocks"] if b["template"] == template]
        embedded_error = active_error * max(b["spectator_frobenius_factor"] for b in references)
        assert embedded_error < TOL and unitary_error < TOL
        for detail in references:
            terms, active = record["kraus"][detail["index"]], detail["active_wires"]
            n = len(terms[0][0])
            local_terms = [["".join(p[n-1-q] for q in active[::-1]), c, re, im] for p, c, re, im in terms]
            ms = deserialize_kraus([local_terms])[0]
            assert ms.size <= 3
            actual = np.asarray(ms.eff_op().to_matrix()) / detail["alpha"]
            assert np.linalg.norm(actual - target) < 1e-14
        checks.append(dict(template=template, qualified=True, active_block_frobenius_error=active_error,
            full_embedded_block_frobenius_error=embedded_error, unitary_frobenius_error=unitary_error))
    return checks


def recount_qasm(path):
    # Stream the existing operation parser instead of retaining millions of
    # Python operation objects for TFIM-20's independent artifact recount.
    operations, parameters, width = Counter(), Counter(), 0
    with path.open() as stream:
        for raw_line in stream:
            line = strip_qasm_comment(raw_line)
            width += sum(int(m.group(1)) for m in QREG2_RE.finditer(line))
            operation = parse_qasm_operation(line)
            if operation is None:
                continue
            assert operation.name in {*IBM, "reset", "barrier"}, operation
            operations[operation.name] += 1
            if operation.name in {"u1", "u2", "u3"}:
                parameters[operation] += 1
    nc = 0
    for operation, multiplicity in parameters.items():
        clifford = is_ibm_operation_clifford(operation)
        assert clifford is not None
        if not clifford:
            nc += multiplicity
    return sum(operations.get(g, 0) for g in IBM), dict(operations), width, nc


def worker(case, method):
    plan = read(OUT / "manifest.json")
    assert source_identity() == plan["sources"]
    assert digest(__file__) == plan["runner_sha256"] and digest(HELPER) == plan["helper_sha256"]
    for path, expected in plan["files"][case].items():
        assert digest(path) == expected, path
    raw_folder, target_path, vcbe_folder = paths(case)
    fixture, raw = read(target_path), read(raw_folder / "results.json")
    target = fixture["input_kraus"]
    target_hash = input_identity(deserialize_kraus(target))
    assert target_hash == raw["canonicalization"]["raw_input_sha256"]
    canonical_hash = input_identity(deserialize_kraus(raw["canonical_kraus"]))
    assert canonical_hash == raw["canonicalization"]["canonical_input_sha256"]
    error = map_error(target, raw["canonical_kraus"])
    assert error < TOL
    assert digest(raw_folder / "channel_nam_opt0.qasm") == raw["qasm_sha256"]
    def no_rewrite(*args, **kwargs):
        raise AssertionError("No Kraus rewrite in this batch")
    for name in ("rewrite_search", "gatecost_rewrite_search", "apply_rewrite_result"):
        setattr(ChannelIR, name, no_rewrite)
    directory = OUT / "cases" / case
    directory.mkdir(parents=True, exist_ok=True)
    progress(case, method, "load_frozen_source")
    started = time.perf_counter()
    extra = {}
    if method == "Raw":
        source, registers, _ = build_pipeline(deserialize_kraus(raw["canonical_kraus"]), "Raw")
        assert registers == raw["registers"]
        metric = raw["metrics"]
        lam, original_path = raw["Lambda"], raw_folder / "results.json"
        extra.update(reused_vcbe_fit=False, reference_NAM_NC=raw["non_clifford"],
                     historical_compile_s=raw["compile_time_s"])
    else:
        assert case in SMALL
        record = read(vcbe_folder / "results.json")
        assert record["status"] == "success" and record["raw_kraus_unchanged"]
        assert record["rewrite"]["enabled"] is False
        assert target == record["kraus"] and target_hash == record["raw_kraus_sha256"]
        assert digest(target_path) == record["provenance"]["input_file_sha256"]
        measured = record["methods"]["vcbe_source_tree"]
        metric = measured["metrics"]["nam_opt0"]
        assert metric["block_qualified"] and all(b["qualified"] for b in measured["block_validation"])
        assert digest(metric["qasm_path"]) == metric["qasm_sha256"]
        with (vcbe_folder / "vcbe_source_tree_source.qpy").open("rb") as stream:
            source = qpy.load(stream)[0]
        assert source.metadata["outer_control_scope"] == "whole-block"
        assert source.metadata["alpha_per_kraus"] == record["alpha"]
        registers = measured["registers"]
        lam, original_path = record["Lambda"], vcbe_folder / "results.json"
        extra.update(reused_vcbe_fit=True, reference_NAM_NC=metric["NC"],
            historical_unique_fit_s=record["fresh_unique_fit_s"],
            historical_compile_s=record["fresh_unique_fit_s"] + metric["assembly_plus_lowering_s"],
            historical_NAM_full_choi=measured.get("verification"), inner_preoptimization_level=0,
            outer_control_scope="whole-block")
    source_load_or_reassembly_s = time.perf_counter() - started
    assert np.isclose(source.metadata["subnormalization_factor"], lam, rtol=1e-12)
    assert np.isclose(lam, raw["Lambda"], rtol=1e-12)
    progress(case, method, "reproduce_archived_nam_histogram")
    nam = lower(source, NAM)
    assert counts(nam, NAM) == (metric["G"], metric["counts"])
    del nam
    gc.collect()
    save_qpy(source, directory / f"{method}_source.qpy")
    if method == "VCBE":
        progress(case, method, "local_block_validation")
        checks = local_checks(record, vcbe_folder)
        dump(directory / "vcbe_local_validation.json", checks)
        extra["max_embedded_block_error"] = max(b["full_embedded_block_frobenius_error"] for b in checks)
    progress(case, method, "ibm_opt0")
    started = time.perf_counter()
    circuit = lower(source)
    lowering_s = time.perf_counter() - started
    del source
    gc.collect()
    require_identity_layout(circuit)
    G, operations = counts(circuit, IBM)
    assert circuit.num_qubits == metric["W"]
    progress(case, method, "classify_and_save")
    classified, nc = {}, 0
    for instruction in circuit.data:
        op = instruction.operation
        if op.name not in {"u1", "u2", "u3"}:
            continue
        key = QasmOperation(op.name, tuple(str(float(p)) for p in op.params))
        if key not in classified:
            classified[key] = is_ibm_operation_clifford(key)
            assert classified[key] is not None
        nc += classified[key] is False
    qasm_path = directory / f"{method}_ibm.qasm"
    flat = QuantumCircuit(circuit.num_qubits)
    flat.compose(circuit, inplace=True)
    qasm2.dump(flat, qasm_path)
    phase = float(flat.global_phase)
    del flat, circuit
    gc.collect()
    progress(case, method, "independent_qasm_recount")
    assert recount_qasm(qasm_path) == (G, operations, metric["W"], nc)
    # Remove roundoff only for rational TFIM/DQE display. Preserve the actual
    # measured float normalization, and all Petz digits, in the data record.
    display_lam = str(round(lam, 12)) if case != "Petz_Ising3" else str(lam)
    assert abs(float(display_lam) - lam) < 1e-12
    row = dict(case=case, method=method, status="ok", basis="IBM", G=G, W=metric["W"],
        Lambda=lam, Lambda_G=lam * G, Lambda_display=display_lam,
        Lambda_G_display=format(Decimal(display_lam) * G, "f"), CX=operations.get("cx", 0),
        non_clifford=int(nc), counts=operations, registers=registers,
        lowering_s=lowering_s, source_load_or_reassembly_s=source_load_or_reassembly_s,
        measurement_time_scope="final IBM opt0 lowering only; imports, reassembly/load, NAM reproduction, validation, classification and artifact I/O excluded",
        optimizer_s=0.0, optimizer_scope="none", lowering_optimization_level=0,
        basis_gates=IBM, seed_transpiler=SEED, approximation_degree=1.0,
        qubits_initially_zero=False, numerical_threads=1, num_processes=1,
        reference_NAM_G=metric["G"], reference_NAM_counts=metric["counts"],
        nam_reproduction="passed exact archived histogram", independent_qasm_recount="passed G, NC, counts and width",
        target_kraus_sha256=target_hash, raw_canonical_kraus_sha256=canonical_hash,
        canonical_map_coefficient_gram_relative_error=error, target_path=str(target_path),
        target_file_sha256=digest(target_path), model=fixture["model"],
        original_result_path=str(original_path), original_result_sha256=digest(original_path),
        qasm_path=str(qasm_path), qasm_sha256=digest(qasm_path), global_phase_rad=phase,
        source_qpy_path=str(directory / f"{method}_source.qpy"), source_qpy_sha256=digest(directory / f"{method}_source.qpy"),
        full_choi_status="skipped_user_request", rewrite_enabled=False,
        validation_scope="frozen input hashes, sparse coefficient Gram equivalence, NAM histogram reproduction and independent IBM QASM recount; VCBE local blocks and exact spectator embedding; no new full-channel Choi",
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024), **extra)
    assert source_identity() == plan["sources"]
    dump(directory / f"{method}_results.json", row)
    print("RESULT", case, method, "G", G, "LambdaG", row["Lambda_G_display"],
          "W", row["W"], "lowering_s", lowering_s, flush=True)


def create_manifest():
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and smoke["sources"] == source_identity()
    assert not (OUT / "manifest.json").exists(), "Choose a fresh measurement batch"
    files = {}
    for case in CASES:
        raw_folder, fixture, vcbe_folder = paths(case)
        selected = [fixture, raw_folder / "results.json", raw_folder / "channel_nam_opt0.qasm"]
        if case in SMALL:
            record = read(vcbe_folder / "results.json")
            selected += [vcbe_folder / "results.json", vcbe_folder / "vcbe_source_tree_source.qpy",
                         vcbe_folder / "vcbe_source_tree_nam_opt0.qasm"]
            for template in {b["template"] for b in record["vcbe_blocks"]}:
                selected += [vcbe_folder / template / f for f in ("results.json", "source.qpy", "target.npy")]
        files[case] = {str(p): digest(p) for p in selected}
    dump(OUT / "manifest.json", dict(cases=CASES, workers=WORKERS, sources=source_identity(),
        files=files, runner_sha256=digest(__file__), helper_sha256=digest(HELPER),
        basis_gates=IBM, lowering_optimization_level=0, downstream_optimizer=None,
        vcbe_policy="original Kraus, whole-block outer controls, saved successful fits; no inner opt3",
        full_choi_status="skipped_user_request", seed_transpiler=SEED,
        approximation_degree=1.0, qubits_initially_zero=False, numerical_threads=1, num_processes=1,
        worker_budget_s=BUDGET_S, worker_rss_budget_bytes=RSS_BUDGET,
        environment=dict(python=sys.version, qiskit=qiskit.__version__, numpy=np.__version__)))


def export(state):
    rows = [read(OUT / "cases" / c / f"{m}_results.json") for c in CASES
            for m in ("Raw", "VCBE") if (OUT / "cases" / c / f"{m}_results.json").exists()]
    dump(OUT / "results.json", dict(status=state["status"], rows=rows, workers=state["workers"],
                                   manifest_path=str(OUT / "manifest.json")))
    fields = ["case", "method", "status", "basis", "G", "Lambda_display", "Lambda_G_display", "W",
        "CX", "non_clifford", "lowering_s", "source_load_or_reassembly_s", "measurement_time_scope",
        "optimizer_s", "optimizer_scope", "reference_NAM_G", "historical_compile_s",
        "historical_unique_fit_s", "reused_vcbe_fit", "nam_reproduction", "independent_qasm_recount",
        "target_kraus_sha256", "canonical_map_coefficient_gram_relative_error", "max_embedded_block_error",
        "full_choi_status", "validation_scope", "qasm_path", "qasm_sha256", "original_result_path"]
    with (ROOT / "Results/rq1_tfim_dqe_petz_ibm_opt0_20261014.csv").open("w", newline="") as stream:
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
        assert args.method and (args.method == "Raw" or args.case in SMALL)
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
        log_path = OUT / f"{case}_{method}.log"
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
        dump(OUT / "status.json", state)
        export(state)
        print(json.dumps(measured), flush=True)
        if code:
            state.update(status="failed", current=None)
            dump(OUT / "status.json", state)
            export(state)
            raise RuntimeError(f"{case} {method} failed; inspect {log_path}")
    state.update(status="complete", current=None)
    assert source_identity() == read(OUT / "manifest.json")["sources"]
    export(state)
    assert len(read(OUT / "results.json")["rows"]) == 7
    dump(OUT / "status.json", state)
    print("Completed seven IBM opt0 measurements", flush=True)


if __name__ == "__main__":
    main()
