"""Frozen HRW-12/28 inputs: IBM Raw and VCBE, with/without whole-circuit opt3.

Keep the existing Ours Kraus rewrite and NAM-based placement options. Reuse
qualified raw-Kraus VCBE fits. Also recount the previous inner-only opt3 route
as a separately named diagnostic. No Stinespring or full-system matrices.

First run smoke_block_encoding_archive.py into OUT, then run this script in
qiskit_simulate. Measurements are isolated, sequential, and checkpointed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Baseline_results/rq1_hrw12_hrw28_ibm_opt3_20261014"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["QISKIT_PARALLEL"] = "FALSE"
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / folder) for folder in ("src", "Baseline_scripts")]

import numpy as np
import psutil
import qiskit
from qiskit import QuantumCircuit, qasm2, qpy
from qiskit.quantum_info import Operator
from baseline_common import QasmOperation, is_ibm_operation_clifford
from block_encoding import BlockEncoding
from channel_IR import channel as ChannelIR, channel_ensemble
from channel_LCU import channel_to_LCU
from experiment_protocol import source_identity, transpile_channel_circuit
from rq1_pipeline import build_pipeline, deserialize_kraus, input_identity

CASES = ("HRW-12", "HRW-28")
METHODS = ("Raw", "Raw+Qiskit opt3", "VCBE", "VCBE+Qiskit opt3", "Ours")
DIAGNOSTIC = "VCBE+inner opt3"
WORKERS = ("ours", "vcbe", "vcbe_opt3", "vcbe_inner", "raw", "raw_opt3")
METHOD_BY_WORKER = dict(zip(WORKERS, ("Ours", "VCBE", "VCBE+Qiskit opt3",
                                    DIAGNOSTIC, "Raw", "Raw+Qiskit opt3")))
NAM = ["rz", "h", "x", "cx"]
IBM = ["u1", "u2", "u3", "cx"]
SEED = 20261009
TOL = 1e-10
BUDGET_S = 600
RSS_BUDGET = 4 * 1024 ** 3


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def folders(case):
    return {
        "raw": ROOT / "paper_experiments/pending_review/rq1_paper_suite_20261010/full_v2/runs" / case / "raw",
        "ours": ROOT / "paper_experiments/pending_review/tfim_hrw_full_20261011/cases" / case,
        "vcbe": ROOT / ("Baseline_results/vcbe_no_rewrite_hrw28_20261014" if case == "HRW-28"
                        else "Baseline_results/vcbe_no_rewrite_20261014") / "cases" / case,
    }


def slug(method):
    return method.replace("+", "_").replace(" ", "_")


def result_path(case, method):
    return OUT / "cases" / case / f"{slug(method)}_results.json"


def progress(case, method, stage):
    dump(OUT / "progress.json", dict(case=case, method=method, stage=stage,
         pid=os.getpid(), updated_at_utc=datetime.now(timezone.utc).isoformat()))
    print(case, method, stage, flush=True)


def map_error(left, right):
    # Sparse Pauli coefficient Gram; independent of Kraus phases/order, no
    # Hilbert-space matrix. Both HRW representations have two terms per Kraus.
    def gram(encoded):
        out = Counter()
        for terms in encoded:
            row = Counter()
            for label, magnitude, real, imag in terms:
                row[label] += magnitude * complex(real, imag)
            for p, a in row.items():
                for q, b in row.items():
                    out[p, q] += a.conjugate() * b
        return out
    a, b = gram(left), gram(right)
    return float(np.sqrt(sum(abs(a[k] - b[k]) ** 2 for k in a.keys() | b.keys())
                         / sum(abs(v) ** 2 for v in a.values())))


def counts(circuit, basis):
    operations = {str(k): int(v) for k, v in circuit.count_ops().items()}
    assert set(operations) <= {*basis, "barrier", "reset"}, operations
    return sum(operations.get(gate, 0) for gate in basis), operations


def local_matrix(circuit):
    assert circuit.num_qubits <= 4, "Only local validation may form a matrix"
    return Operator(circuit).data


def lower(source, level=0):
    return transpile_channel_circuit(source, basis_gates=IBM, optimization_level=level,
        seed_transpiler=SEED, approximation_degree=1.0, qubits_initially_zero=False,
        num_processes=1)


def require_identity_layout(circuit):
    if circuit.layout is not None:
        expected = list(range(circuit.num_qubits))
        assert circuit.layout.initial_index_layout() == expected
        assert circuit.layout.final_index_layout() == expected


def save_qpy(circuit, path):
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)


def source_audit(case):
    plan = read(OUT / "manifest.json")
    assert plan["sources"] == source_identity()
    assert plan["runner_sha256"] == digest(__file__)
    for path, value in plan["files"][case].items():
        assert digest(path) == value, path
    dirs = folders(case)
    raw, ours, vcbe = [read(dirs[name] / "results.json") for name in ("raw", "ours", "vcbe")]
    fixture = read(dirs["ours"] / "input.json")
    target = fixture["input_kraus"]
    key = input_identity(deserialize_kraus(target))
    assert key == raw["canonicalization"]["raw_input_sha256"] == vcbe["raw_kraus_sha256"]
    assert key == raw["canonicalization"]["canonical_input_sha256"]
    assert target == raw["canonical_kraus"] == vcbe["kraus"]
    assert digest(dirs["ours"] / "input.json") == ours["input_sha256"] == vcbe["provenance"]["input_file_sha256"]
    assert digest(dirs["raw"] / "channel_nam_opt0.qasm") == raw["qasm_sha256"]
    assert digest(dirs["ours"] / "channel_nam.qasm") == ours["qasm_sha256"]
    assert vcbe["status"] == "success" and vcbe["raw_kraus_unchanged"]
    assert vcbe["rewrite"]["enabled"] is False
    for value in vcbe["methods"].values():
        metric = value["metrics"]["nam_opt0"]
        assert metric["status"] == "ok" and metric["block_qualified"]
        assert digest(metric["qasm_path"]) == metric["qasm_sha256"]
        assert all(v["qualified"] for v in value["block_validation"])
    error = map_error(target, ours["assembly"]["compiled_kraus"])
    assert error < TOL
    # Runtime guard: this experiment is reconstruction, not another search.
    def no_rewrite(*args, **kwargs):
        raise AssertionError("Frozen experiment must not perform Kraus rewrite")
    for name in ("rewrite_search", "gatecost_rewrite_search", "apply_rewrite_result"):
        setattr(ChannelIR, name, no_rewrite)
    return dirs, raw, ours, vcbe, dict(target_kraus_sha256=key,
        target_map_coefficient_gram_relative_error=error,
        target_file_sha256=digest(dirs["ours"] / "input.json"),
        target_path=str(dirs["ours"] / "input.json"), model=fixture["model"])


def vcbe_local_checks(case, vcbe, folder):
    checks = []
    for template in sorted({b["template"] for b in vcbe["vcbe_blocks"]}):
        block_dir = folder / template
        target = np.load(block_dir / "target.npy")
        assert target.shape == (2, 2)
        with (block_dir / "source.qpy").open("rb") as stream:
            source = qpy.load(stream)[0]
        references = [b for b in vcbe["vcbe_blocks"] if b["template"] == template]
        base = local_matrix(source)
        errors = {}
        for level in (0, 3):
            circuit = lower(source, level)
            require_identity_layout(circuit)
            matrix = local_matrix(circuit)
            phase_error = float(np.linalg.norm(matrix - base))
            active_error = float(np.linalg.norm(matrix[:2, :2] - target))
            embedded_error = active_error * max(b["spectator_frobenius_factor"] for b in references)
            assert phase_error < TOL and embedded_error < TOL
            errors[f"ibm_opt{level}"] = dict(unitary_frobenius_error=phase_error,
                active_block_frobenius_error=active_error,
                full_embedded_block_frobenius_error=embedded_error)
        # Reconstruct every branch's target from its Pauli coefficients, after
        # removing only exact spectator identities. Never form a 2**n matrix.
        for detail in references:
            terms = vcbe["kraus"][detail["index"]]
            active = detail["active_wires"]
            n = len(terms[0][0])
            local_terms = [["".join(label[n-1-q] for q in active[::-1]), c, re, im]
                           for label, c, re, im in terms]
            local_kraus = deserialize_kraus([local_terms])[0]
            assert local_kraus.size == 1
            actual = np.asarray(local_kraus.eff_op().to_matrix()) / detail["alpha"]
            assert np.linalg.norm(actual - target) < 1e-14
        checks.append(dict(template=template, multiplicity=len(references), checks=errors,
                           qualified=True, only_local_matrix_qubits=2))
    dump(OUT / "cases" / case / "vcbe_local_validation.json", checks)
    return checks


def save_result(case, method, circuit, reference, provenance, lowering_s, **extra):
    G, operations = counts(circuit, IBM)
    assert circuit.num_qubits == reference["W"]
    require_identity_layout(circuit)
    progress(case, method, "classify_and_save")
    started = time.perf_counter()
    nc, classified = 0, {}
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name not in {"u1", "u2", "u3"}:
            continue
        key = (operation.name, tuple(str(float(p)) for p in operation.params))
        if key not in classified:
            classified[key] = is_ibm_operation_clifford(QasmOperation(*key))
            assert classified[key] is not None
        nc += classified[key] is False
    classification_s = time.perf_counter() - started
    directory = OUT / "cases" / case
    path = directory / f"{slug(method)}_ibm.qasm"
    flat = QuantumCircuit(circuit.num_qubits)
    flat.compose(circuit, inplace=True)
    qasm2.dump(flat, path)
    qpy_path = path.with_suffix(".qpy")
    save_qpy(flat, qpy_path)
    lam = int(round(reference["Lambda"]))
    assert np.isclose(reference["Lambda"], lam, atol=1e-12, rtol=1e-12)
    row = dict(case=case, n=int(case.split("-")[1]), method=method, status="ok", basis="IBM",
        G=G, Lambda=lam, Lambda_numeric=reference["Lambda"], Lambda_G=lam * G,
        W=circuit.num_qubits, CX=operations.get("cx", 0), non_clifford=int(nc), counts=operations,
        reference_NAM_G=reference["G"], reference_NAM_counts=reference["counts"],
        reference_NAM_NC=reference.get("non_clifford"),
        original_result_path=reference["source_result_path"],
        original_result_sha256=digest(reference["source_result_path"]),
        historical_compile_s=reference.get("historical_compile_s"),
        historical_compile_time_scope=reference.get("historical_compile_time_scope"),
        measurement_time_scope="final IBM lowering or whole-circuit opt3 only; reconstruction, imports, validation, classification and I/O excluded",
        lowering_s=lowering_s, optimizer_s=lowering_s if "Qiskit opt3" in method else 0.0,
        classification_s=classification_s, seed_transpiler=SEED,
        lowering_optimization_level=3 if "Qiskit opt3" in method else 0,
        approximation_degree=1.0, qubits_initially_zero=False,
        qasm_path=str(path), qasm_sha256=digest(path), qpy_path=str(qpy_path), qpy_sha256=digest(qpy_path),
        global_phase_rad=float(flat.global_phase), qasm_global_phase_policy="overall phase omitted by QASM2, channel unchanged; QPY retains phase",
        full_choi_status="skipped_user_request", validation_scope="frozen input hashes; sparse Pauli coefficient Gram; NAM histogram reproduction; local VCBE blocks and exact spectator embeddings; exact opt3 settings and identity layout; no full compiled-channel Choi",
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        **provenance, **extra)
    assert source_identity() == read(OUT / "manifest.json")["sources"]
    dump(result_path(case, method), row)
    print("RESULT", case, method, "G", G, "LambdaG", row["Lambda_G"], "W", row["W"],
          "lowering_s", lowering_s, flush=True)
    return row


def worker(case, kind):
    method = METHOD_BY_WORKER[kind]
    directory = OUT / "cases" / case
    directory.mkdir(parents=True, exist_ok=True)
    dirs, raw, ours, vcbe, provenance = source_audit(case)
    if kind in {"raw_opt3", "vcbe_opt3"}:
        baseline = "Raw" if kind == "raw_opt3" else "VCBE"
        previous = read(result_path(case, baseline))
        assert digest(previous["qasm_path"]) == previous["qasm_sha256"]
        progress(case, method, "load_frozen_ibm_opt0_qasm")
        source = qasm2.load(previous["qasm_path"])
        G, operations = counts(source, IBM)
        assert G == previous["G"] and operations == previous["counts"]
        reference = dict(G=previous["reference_NAM_G"], counts=previous["reference_NAM_counts"],
            Lambda=previous["Lambda_numeric"], W=previous["W"],
            source_result_path=previous["original_result_path"],
            historical_compile_s=previous["historical_compile_s"],
            historical_compile_time_scope=previous["historical_compile_time_scope"],
            non_clifford=previous["reference_NAM_NC"])
        progress(case, method, "whole_circuit_ibm_opt3")
        started = time.perf_counter()
        circuit = lower(source, 3)
        elapsed = time.perf_counter() - started
        save_result(case, method, circuit, reference, provenance, elapsed,
            optimizer_scope="whole assembled IBM opt0 channel circuit", input_G=G,
            input_qasm_path=previous["qasm_path"], input_qasm_sha256=previous["qasm_sha256"],
            reused_vcbe_fit=kind == "vcbe_opt3", rewrite_enabled=False,
            nam_reproduction="inherited from unoptimized source row; not a NAM-native opt3 measurement")
        return
    progress(case, method, "load_or_reassemble_frozen_source")
    started = time.perf_counter()
    if kind == "raw":
        source, registers, _ = build_pipeline(deserialize_kraus(raw["canonical_kraus"]), "Raw")
        reference = dict(G=raw["metrics"]["G"], counts=raw["metrics"]["counts"],
            Lambda=raw["Lambda"], W=raw["metrics"]["W"], non_clifford=raw["non_clifford"],
            source_result_path=str(dirs["raw"] / "results.json"),
            historical_compile_s=raw["compile_time_s"],
            historical_compile_time_scope="saved Raw assembly and NAM opt0 lowering")
        assert registers == raw["registers"]
        extra = dict(rewrite_enabled=False, optimizer_scope="none")
    elif kind == "ours":
        kraus = deserialize_kraus(ours["assembly"]["compiled_kraus"])
        options = ours["assembly"]["matrix_order_options"]
        def backend(ms):
            return BlockEncoding(ms, placement_options={}, matrix_order_options=options)
        source, registers = channel_to_LCU(channel_ensemble([deepcopy(kraus)]),
            structure="tree", opt="Matrix-order", fstar=True, block_encoding_cls=backend)
        reference = dict(G=ours["G"], counts=ours["counts"], Lambda=ours["Lambda"],
            W=ours["W"], non_clifford=ours["non_clifford"],
            source_result_path=str(dirs["ours"] / "results.json"),
            historical_compile_s=ours["compile_time_s"], historical_compile_time_scope=ours["timing_scope"])
        assert registers == ours["registers"]
        extra = dict(rewrite_enabled=True, reused_rewrite=True, optimizer_scope="none",
            backend_change_scope="final lowering only; saved rewrite and NAM-based placement options retained",
            matrix_order_options=options)
    else:
        name = "vcbe_source_tree" if kind == "vcbe" else "vcbe_preopt3_tree"
        record = vcbe["methods"][name]
        metric = record["metrics"]["nam_opt0"]
        with (dirs["vcbe"] / f"{name}_source.qpy").open("rb") as stream:
            source = qpy.load(stream)[0]
        registers = record["registers"]
        assert source.metadata["outer_control_scope"] == "whole-block"
        reference = dict(G=metric["G"], counts=metric["counts"], Lambda=metric["Lambda"],
            W=metric["W"], non_clifford=metric["NC"], source_result_path=str(dirs["vcbe"] / "results.json"),
            historical_compile_s=vcbe["fresh_unique_fit_s"] + metric["assembly_plus_lowering_s"],
            historical_compile_time_scope="saved fresh unique VCBE fit plus assembly and NAM opt0 lowering")
        extra = dict(rewrite_enabled=False, reused_vcbe_fit=True,
            optimizer_scope="each Kraus inner block before outer control" if kind == "vcbe_inner" else "none",
            inner_opt3_basis="IBM" if kind == "vcbe_inner" else "none",
            historical_inner_opt3_s=metric["inner_preoptimization_s"],
            vcbe_fit_source_path=str(dirs["vcbe"]), outer_control_scope="whole-block")
    reconstruction_s = time.perf_counter() - started
    assert np.isclose(source.metadata["subnormalization_factor"], reference["Lambda"], rtol=1e-12)
    progress(case, method, "reproduce_archived_nam_histogram")
    nam = transpile_channel_circuit(source, basis_gates=NAM, optimization_level=0,
        seed_transpiler=SEED, qubits_initially_zero=False, num_processes=1)
    assert counts(nam, NAM) == (reference["G"], reference["counts"])
    del nam
    save_qpy(source, directory / f"{slug(method)}_source.qpy")
    if kind == "vcbe":
        progress(case, method, "check_local_fits_and_ibm_lowering")
        checks = vcbe_local_checks(case, vcbe, dirs["vcbe"])
        extra["max_local_ibm_embedded_block_error"] = max(
            v["full_embedded_block_frobenius_error"] for c in checks for v in c["checks"].values())
    progress(case, method, "ibm_opt0")
    started = time.perf_counter()
    circuit = lower(source)
    elapsed = time.perf_counter() - started
    save_result(case, method, circuit, reference, provenance, elapsed, registers=registers,
        reconstruction_s=reconstruction_s, nam_reproduction="passed exact archived gate histogram", **extra)


def manifest():
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and smoke["sources"] == source_identity()
    assert not (OUT / "manifest.json").exists(), "Use a fresh batch for fresh measurements"
    files = {}
    for case in CASES:
        dirs = folders(case)
        paths = [dirs["raw"] / "results.json", dirs["raw"] / "channel_nam_opt0.qasm",
                 dirs["ours"] / "results.json", dirs["ours"] / "input.json", dirs["ours"] / "channel_nam.qasm",
                 dirs["vcbe"] / "results.json"]
        for name in ("vcbe_source_tree", "vcbe_preopt3_tree"):
            paths.extend([dirs["vcbe"] / f"{name}_source.qpy", dirs["vcbe"] / f"{name}_nam_opt0.qasm"])
        for template in ("fit_00", "fit_01"):
            paths.extend(dirs["vcbe"] / template / name for name in ("results.json", "target.npy", "source.qpy"))
        files[case] = {str(p): digest(p) for p in paths}
    dump(OUT / "manifest.json", dict(cases=CASES, methods=METHODS, supplementary_method=DIAGNOSTIC,
        sources=source_identity(), runner_sha256=digest(__file__), files=files,
        basis_gates=IBM, lowering_optimization_level=0, downstream_optimization_level=3,
        opt3_scope="whole assembled IBM opt0 circuit for both Raw and VCBE; inner-only VCBE opt3 recorded separately",
        vcbe_representation="original Kraus, no rewrite; reuse qualified local fits",
        ours_policy="saved Kraus rewrite and NAM-based placement options; final IBM lowering only",
        full_choi_status="skipped_user_request", full_system_matrix_construction=False,
        seed_transpiler=SEED, qubits_initially_zero=False, approximation_degree=1.0,
        num_processes=1, numerical_threads=1, worker_budget_s=BUDGET_S, worker_rss_budget_bytes=RSS_BUDGET,
        environment=dict(python=sys.version, platform=platform.platform(), qiskit=qiskit.__version__, numpy=np.__version__)))


def export(state):
    rows = [read(result_path(case, method)) for case in CASES for method in METHODS
            if result_path(case, method).exists()]
    diagnostics = [read(result_path(case, DIAGNOSTIC)) for case in CASES
                   if result_path(case, DIAGNOSTIC).exists()]
    for row in rows + diagnostics:
        path = result_path(row["case"], "Ours")
        if path.exists():
            ours = read(path)
            row.update(G_over_Ours=row["G"] / ours["G"], Lambda_G_over_Ours=row["Lambda_G"] / ours["Lambda_G"])
    dump(OUT / "results.json", dict(status=state["status"], rows=rows,
        diagnostics=diagnostics, manifest_path=str(OUT / "manifest.json"), workers=state["workers"]))
    columns = ["case", "method", "status", "basis", "G", "Lambda", "Lambda_G", "W", "CX", "non_clifford",
        "G_over_Ours", "Lambda_G_over_Ours", "reference_NAM_G", "lowering_s", "optimizer_s",
        "optimizer_scope", "historical_inner_opt3_s", "rewrite_enabled", "reused_rewrite", "reused_vcbe_fit",
        "historical_compile_s", "historical_compile_time_scope", "measurement_time_scope", "nam_reproduction",
        "target_kraus_sha256", "target_map_coefficient_gram_relative_error", "full_choi_status", "validation_scope",
        "qasm_path", "qasm_sha256", "original_result_path", "original_result_sha256"]
    for filename, selected in (("rq1_hrw12_hrw28_ibm_opt3_20261014.csv", rows),
                               ("rq1_hrw12_hrw28_ibm_inner_opt3_20261014.csv", diagnostics)):
        with (ROOT / "Results" / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=WORKERS)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.worker:
        assert args.case
        worker(args.case, args.worker)
        return
    manifest()
    state = dict(status="running", workers=[], current=None)
    process = None
    def terminate_worker():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    def stop(signum, frame):
        terminate_worker()
        state.update(status="paused_user_request", current=None)
        dump(OUT / "status.json", state)
        export(state)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for case in CASES:
        for kind in WORKERS:
            started, peak, reason = time.monotonic(), 0, None
            log_path = OUT / f"{case}_{kind}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", kind,
                    "--case", case], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                state["current"] = dict(case=case, worker=kind, pid=process.pid)
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
                            terminate_worker()
                            break
                        time.sleep(0.5)
                finally:
                    terminate_worker()
                    code = process.wait()
            record = dict(case=case, worker=kind, exit_code=code, termination_reason=reason,
                wall_s=time.monotonic() - started, peak_rss_bytes=peak, log_path=str(log_path))
            state["workers"].append(record)
            if code:
                state.update(status="failed", current=None)
                dump(OUT / "status.json", state)
                export(state)
                raise RuntimeError(f"{case} {kind} failed; inspect {log_path}")
            dump(OUT / "status.json", state)
            export(state)
            print(json.dumps(record), flush=True)
    state.update(status="complete", current=None)
    export(state)
    assert len(read(OUT / "results.json")["rows"]) == len(CASES) * len(METHODS)
    dump(OUT / "status.json", state)
    print("Completed HRW-12/28 IBM whole-circuit opt3 comparison", flush=True)


if __name__ == "__main__":
    main()
