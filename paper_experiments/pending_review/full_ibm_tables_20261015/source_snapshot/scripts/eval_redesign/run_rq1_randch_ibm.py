"""Frozen seed-0 RandCh-N9/N15: IBM resources for the split RQ1 tables.

Only N9 receives new VCBE fitting and Stinespring synthesis. N15's matrix
methods are explicitly unexecuted. Reuse archived Raw/Ours representations;
no rewrite search or paused external-optimizer queue is dispatched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Baseline_results/rq1_randch_ibm_20261015"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["QISKIT_PARALLEL"] = "FALSE"
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")
sys.path[:0] = [str(ROOT / f) for f in ("src", "Baseline_scripts", "scripts/eval_redesign")]

import numpy as np
import psutil
import qiskit
from qiskit import QuantumCircuit, qasm2, qpy
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import Operator, Pauli
from baseline_common import QasmOperation, is_ibm_operation_clifford
from block_encoding import BlockEncoding
from channel_IR import channel as ChannelIR
from channel_LCU import channel_to_LCU
from experiment_protocol import source_identity, transpile_channel_circuit
from rq1_pipeline import build_pipeline, deserialize_kraus, input_identity, serialize_kraus
import run_rq1_vcbe_no_rewrite as fitting
from run_rq1_hrw_ibm_opt3 import map_error
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")

CASES = ("RandCh-N9-8q-seed0", "RandCh-N15-12q-seed0")
IBM, NAM = ["u1", "u2", "u3", "cx"], ["rz", "h", "x", "cx"]
SEED, TOL = 20261009, 1e-10
METHODS = ("Stinespring", "VCBE", "VCBE+Qiskit opt3", "Raw", "Raw+Qiskit opt3", "Ours")
WORKERS = [(CASES[0], "VCBE"), (CASES[0], "Stinespring")]
WORKERS += [(c, m) for c in CASES for m in ("Ours", "Raw", "Raw+Qiskit opt3")]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            result.update(chunk)
    return result.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def folders(case):
    suite = ROOT / "paper_experiments/pending_review"
    return (suite / "remaining_full_20261011/cases" / case,
            suite / "raw_baseline_20261012/cases" / case / case / "raw")


def slug(method):
    return method.replace("+", "_").replace(" ", "_")


def result_path(case, method):
    return OUT / "cases" / case / (slug(method) + "_results.json")


def progress(case, method, stage):
    dump(OUT / "progress.json", dict(case=case, method=method, stage=stage,
        pid=os.getpid(), updated_at_utc=datetime.now(timezone.utc).isoformat()))
    print(case, method, stage, flush=True)


def lower(circuit, level=0, seed=SEED, basis=IBM):
    return transpile_channel_circuit(circuit, basis_gates=basis, optimization_level=level,
        seed_transpiler=seed, approximation_degree=1.0, qubits_initially_zero=False, num_processes=1)


def base_row(case, method, original=None):
    ours, _ = folders(case)
    target_path = ours / "input.json"
    document = read(target_path)
    return dict(case=case, method=method, status="running", basis="IBM",
        target_path=str(target_path), target_file_sha256=digest(target_path),
        target_kraus_sha256=input_identity(deserialize_kraus(document["input_kraus"])),
        model=document["model"], measurement_attempted=True,
        full_choi_status="skipped_user_request", original_result_path=str(original) if original else "",
        original_result_sha256=digest(original) if original else "",
        basis_gates=IBM, lowering_optimization_level=3 if "+Qiskit opt3" in method else 0,
        optimizer_scope="whole assembled IBM opt0 circuit" if "+Qiskit opt3" in method else "none",
        approximation_degree=1.0, qubits_initially_zero=False, seed_transpiler=SEED,
        numerical_threads=1, num_processes=1)


def save_counts(row, circuit, lam, elapsed, write_circuit=True):
    G = sum(int(circuit.count_ops().get(g, 0)) for g in IBM)
    ops = {str(g): int(c) for g, c in circuit.count_ops().items()}
    assert set(ops) <= set(IBM) and G > 0
    if circuit.layout is not None:
        assert circuit.layout.initial_index_layout() == list(range(circuit.num_qubits))
        assert circuit.layout.final_index_layout() == list(range(circuit.num_qubits))
    row.update(status="ok", G=G, W=circuit.num_qubits, Lambda=float(lam), Lambda_G=float(lam) * G,
        counts=ops, CX=ops.get("cx", 0), synthesis_or_optimizer_s=elapsed,
        optimizer_s=elapsed if row["lowering_optimization_level"] == 3 else 0,
        lowering_s=elapsed if row["lowering_optimization_level"] == 0 else 0,
        time_scope="final IBM lowering or whole-circuit opt3 only; excludes imports, reconstruction, fitting, validation and I/O")
    dump(result_path(row["case"], row["method"]), row)
    progress(row["case"], row["method"], "resource_counts_saved")
    if write_circuit:
        path = result_path(row["case"], row["method"]).with_name(slug(row["method"]) + "_ibm.qasm")
        flat = QuantumCircuit(circuit.num_qubits)
        flat.compose(circuit, inplace=True)
        qasm2.dump(flat, path)
        row.update(qasm_path=str(path), qasm_sha256=digest(path), global_phase_rad=float(flat.global_phase))
        del flat
    parameters = Counter()
    for instruction in circuit.data:
        op = instruction.operation
        if op.name in {"u1", "u2", "u3"}:
            parameters[QasmOperation(op.name, tuple(str(float(p)) for p in op.params))] += 1
    nc = 0
    for op, multiplicity in parameters.items():
        clifford = is_ibm_operation_clifford(op)
        assert clifford is not None
        if not clifford:
            nc += multiplicity
    row.update(non_clifford=int(nc), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(result_path(row["case"], row["method"]), row)
    print("RESULT", row["case"], row["method"], G, lam, row["W"], flush=True)
    return row


def forbid_rewrite(*args, **kwargs):
    raise AssertionError("Use archived Kraus representations; no new rewrite search")


def vcbe(case):
    assert case == CASES[0]
    directory = OUT / "cases" / case
    target_path = folders(case)[0] / "input.json"
    kraus = deserialize_kraus(read(target_path)["input_kraus"])
    row = base_row(case, "VCBE")
    row.update(kraus=serialize_kraus(kraus), rewrite_enabled=False, inner_preoptimization_level=0,
        historical_fit_reused=False, vcbe_blocks=[], per_unique_fit_budget_s=600,
        fit_recipe=dict(layers=[1, 2, 4, 8, 16, 32], seeds=[0, 1], trial_budget_s=50, polish_budget_s=20,
                        acceptance_frobenius=TOL, desired_active_error=1e-12),
        Lambda=float(sum(ms.pauli_norm() ** 2 for ms in kraus)))
    dump(result_path(case, "VCBE"), row)
    args = SimpleNamespace(fit_budget=600, trial_budget=50, polish_budget=20)
    blocks = []
    for index, ms in enumerate(kraus):
        progress(case, "VCBE", f"fit_original_kraus_{index}")
        target, active, spectator = fitting.active_target(ms)
        block, info = fitting.fit_target(target, spectator, directory / f"fit_{index:02d}", args)
        detail = dict(info, index=index, active_wires=active, alpha=float(ms.pauli_norm()),
            spectator_frobenius_factor=spectator, full_embedded_frobenius_block_error=info["fit_error"] * spectator)
        row["vcbe_blocks"].append(detail)
        row["fresh_unique_fit_s"] = sum(d["total_search_s"] for d in row["vcbe_blocks"])
        if detail["full_embedded_frobenius_block_error"] >= TOL:
            row.update(status=info["status"], absence_reason="An original Kraus block failed the fixed Frobenius tolerance; no qualified complete circuit",
                stopped_stage=f"fit_original_kraus_{index}")
            dump(result_path(case, "VCBE"), row)
            optimized = dict(row, method="VCBE+Qiskit opt3", measurement_attempted=False,
                optimizer_scope="whole assembled IBM opt0 circuit", lowering_optimization_level=3,
                absence_reason="No qualified VCBE opt0 circuit exists for downstream optimization")
            dump(result_path(case, optimized["method"]), optimized)
            return
        embedded = QuantumCircuit(ms.size + 1)
        embedded.compose(block, qubits=active + [ms.size], inplace=True)
        embedded.metadata = dict(system_qubits=list(range(ms.size)), workspace_qubits=[ms.size],
                                 alpha=float(ms.pauli_norm()))
        blocks.append(embedded)
        dump(result_path(case, "VCBE"), row)
    prepared = {input_identity([ms]): block for ms, block in zip(kraus, blocks)}
    class Backend:
        def __init__(self, ms):
            self.block = prepared[input_identity([ms])]
        def circuit(self, opt=None):
            return self.block.copy()
    source, registers = channel_to_LCU(SimpleNamespace(channels=[(1.0, kraus)]),
        structure="tree", opt="VCBE", fstar=False, block_encoding_cls=Backend)
    assert source.metadata["outer_control_scope"] == "whole-block"
    assert np.isclose(source.metadata["subnormalization_factor"], row["Lambda"], rtol=1e-12)
    row.update(registers=registers, outer_control_scope="whole-block")
    progress(case, "VCBE", "ibm_opt0")
    started = time.perf_counter()
    circuit = lower(source)
    save_counts(row, circuit, row["Lambda"], time.perf_counter() - started)
    progress(case, "VCBE+Qiskit opt3", "whole_circuit_ibm_opt3")
    optimized = base_row(case, "VCBE+Qiskit opt3")
    optimized.update(input_G=row["G"], input_qasm_sha256=row["qasm_sha256"],
        input_baseline_result_path=str(result_path(case, "VCBE")), inner_preoptimization_level=0,
        historical_fit_reused=False, vcbe_blocks=row["vcbe_blocks"])
    started = time.perf_counter()
    circuit = lower(circuit, 3)
    save_counts(optimized, circuit, row["Lambda"], time.perf_counter() - started)


def stinespring(case):
    assert case == CASES[0]
    directory = OUT / "cases" / case
    encoded = read(folders(case)[0] / "input.json")["input_kraus"]
    n, rank = len(encoded[0][0][0]), len(encoded)
    d = 1 << n
    row = base_row(case, "Stinespring")
    row["seed_transpiler"] = 20261013
    dump(result_path(case, "Stinespring"), row)
    progress(case, "Stinespring", "build_kraus_isometry")
    matrices = np.array([sum((c * complex(re, im) * Pauli(p).to_matrix() for p, c, re, im in terms),
                            np.zeros((d, d), complex)) for terms in encoded])
    vectors = np.column_stack([k.reshape(-1, order="F") for k in matrices])
    left, values, _ = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.sum(values > 1e-12 * values[0]))
    if rank < len(matrices):
        matrices = np.array([(left[:, j] * values[j]).reshape((d, d), order="F") for j in range(rank)])
    completeness = sum((k.conj().T @ k for k in matrices), np.zeros((d, d), complex))
    eig = np.linalg.eigvalsh(completeness)
    tp_error = float(np.linalg.norm(completeness - np.eye(d)))
    native = tp_error < TOL
    beta = 1.0 if native else float(max(1.0, np.nextafter(eig[-1], np.inf)))
    env_bits = (rank - 1).bit_length()
    env_dim = 1 << env_bits
    width = n + env_bits + int(not native)
    padded = np.zeros((env_dim, d * (1 if native else 2), d), complex)
    padded[:rank, :d] = matrices / np.sqrt(beta)
    if not native:
        remainder = np.eye(d) - completeness / beta
        rvalues, rvectors = np.linalg.eigh((remainder + remainder.conj().T) / 2)
        assert rvalues.min() > -1e-12
        padded[0, d:] = (rvectors * np.sqrt(np.maximum(rvalues, 0))) @ rvectors.conj().T
    iso = padded.transpose(1, 0, 2).reshape(-1, d)
    orthogonality = float(np.linalg.norm(iso.conj().T @ iso - np.eye(d)))
    errors = [float(np.linalg.norm(iso[j:d * env_dim:env_dim, :] * np.sqrt(beta) - k))
              for j, k in enumerate(matrices)]
    assert orthogonality < TOL and max(errors) < TOL
    path = directory / "isometry.npy"
    np.save(path, iso)
    row.update(kraus_rank=rank, native_cptp=native, tp_frobenius_error=tp_error,
        mode="native" if native else "flagged_minimum_beta", isometry_path=str(path), isometry_sha256=digest(path),
        isometry_orthogonality_error=orthogonality, isometry_success_kraus_max_error=max(errors),
        completeness_eigenvalue_bounds=[float(eig[0]), float(eig[-1])], Lambda=beta,
        input_system_wires=list(range(n)), output_environment_wires=list(range(env_bits)),
        output_system_wires=list(range(env_bits, env_bits + n)), success_flag_wire=None if native else width - 1,
        planned_W=width, isometry_shape=list(iso.shape), isometry_bytes=int(iso.nbytes))
    dump(result_path(case, "Stinespring"), row)
    source = QuantumCircuit(width)
    source.append(Isometry(iso, 0, 0), range(width))
    progress(case, "Stinespring", "ibm_opt0_synthesis")
    started = time.perf_counter()
    circuit = lower(source, seed=20261013)
    save_counts(row, circuit, beta, time.perf_counter() - started, write_circuit=False)


def pauli(case, method):
    ours_dir, raw_dir = folders(case)
    fixture = read(ours_dir / "input.json")
    original = (ours_dir if method == "Ours" else raw_dir) / "results.json"
    archived = read(original)
    row = base_row(case, method, original)
    progress(case, method, "reuse_archived_representation")
    if method == "Raw+Qiskit opt3":
        baseline = read(result_path(case, "Raw"))
        assert baseline["status"] == "ok"
        assert digest(baseline["qasm_path"]) == baseline["qasm_sha256"]
        source = qasm2.load(baseline["qasm_path"])
        assert dict(source.count_ops()) == baseline["counts"] and source.num_qubits == baseline["W"]
        row.update(input_G=baseline["G"], input_qasm_sha256=baseline["qasm_sha256"],
            input_baseline_result_path=str(result_path(case, "Raw")), reused_raw_representation=True)
        lam = baseline["Lambda"]
    elif method == "Raw":
        key = input_identity(deserialize_kraus(fixture["input_kraus"]))
        assert key == archived["canonicalization"]["raw_input_sha256"]
        assert digest(raw_dir / "channel_nam_opt0.qasm") == archived["qasm_sha256"]
        error = map_error(fixture["input_kraus"], archived["canonical_kraus"])
        assert error < TOL
        source, registers, _ = build_pipeline(deserialize_kraus(archived["canonical_kraus"]), "Raw")
        assert source.num_qubits == archived["metrics"]["W"]
        row.update(registers=registers, reused_raw_representation=True, canonical_map_gram_error=error,
            reference_NAM_G=archived["metrics"]["G"], reference_NAM_counts=archived["metrics"]["counts"],
            backend_change_scope="final lowering only; archived Raw Kraus and construction retained")
        lam = archived["Lambda"]
    else:
        assert digest(ours_dir / "input.json") == archived["input_sha256"]
        assert digest(ours_dir / "channel_nam.qasm") == archived["qasm_sha256"]
        kraus = deserialize_kraus(archived["assembly"]["compiled_kraus"])
        error = map_error(fixture["input_kraus"], archived["assembly"]["compiled_kraus"])
        assert error < TOL
        options = archived["assembly"]["matrix_order_options"]
        def backend(ms):
            return BlockEncoding(ms, placement_options={}, matrix_order_options=options)
        from channel_IR import channel_ensemble
        source, registers = channel_to_LCU(channel_ensemble([deepcopy(kraus)]), structure="tree",
            opt="Matrix-order", fstar=True, block_encoding_cls=backend)
        assert registers == archived["registers"] and source.num_qubits == archived["W"]
        reference = lower(source, basis=NAM)
        assert dict(reference.count_ops()) == archived["counts"]
        del reference
        row.update(reused_rewrite=True, rewrite_gram_error=error, matrix_order_options=options,
            nam_reproduction="passed exact histogram", registers=registers, reference_NAM_G=archived["G"],
            backend_change_scope="final lowering only; original NAM placement and Kraus rewrite retained")
        lam = archived["Lambda"]
    if method != "Raw+Qiskit opt3":
        assert np.isclose(source.metadata["subnormalization_factor"], lam, rtol=1e-12)
    dump(result_path(case, method), row)
    progress(case, method, "whole_circuit_ibm_opt3" if "+Qiskit opt3" in method else "ibm_opt0")
    started = time.perf_counter()
    circuit = lower(source, 3 if "+Qiskit opt3" in method else 0)
    elapsed = time.perf_counter() - started
    del source
    gc.collect()
    save_counts(row, circuit, lam, elapsed)


def worker(case, method):
    plan = read(OUT / "manifest.json")
    assert plan["sources"] == source_identity() and plan["runner_sha256"] == digest(__file__)
    for path, value in plan["files"][case].items():
        assert digest(path) == value
    for path, value in plan["helpers"].items():
        assert digest(path) == value
    for name in ("rewrite_search", "gatecost_rewrite_search", "apply_rewrite_result"):
        setattr(ChannelIR, name, forbid_rewrite)
    if method == "VCBE":
        vcbe(case)
    elif method == "Stinespring":
        stinespring(case)
    else:
        pauli(case, method)
    assert plan["sources"] == source_identity()
    progress(case, method, "finished")


def export(state):
    rows = [read(result_path(c, m)) for c in CASES for m in METHODS if result_path(c, m).exists()]
    dump(OUT / "results.json", dict(status=state["status"], rows=rows, workers=state["completed"],
        manifest_path=str(OUT / "manifest.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.case:
        assert (args.case, args.method) in WORKERS
        worker(args.case, args.method)
        return
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and smoke["sources"] == source_identity()
    assert not (OUT / "manifest.json").exists(), "Use a new batch; preserve earlier measurements"
    files = {}
    for case in CASES:
        ours, raw = folders(case)
        paths = (ours / "input.json", ours / "results.json", ours / "channel_nam.qasm",
                 raw / "results.json", raw / "channel_nam_opt0.qasm")
        files[case] = {str(p): digest(p) for p in paths}
        (OUT / "cases" / case).mkdir(parents=True, exist_ok=True)
    helpers = (Path(fitting.__file__), Path(fitting.__file__).with_name("run_rq1_vcbe_scaling.py"),
        Path(fitting.__file__).with_name("run_rq1_hrw_ibm_opt3.py"),
        ROOT / "Baseline_results/matrix_synthesis_hrw_tfim_20261012/vcbe_probe.py",
        ROOT / "Baseline_results/matrix_synthesis_hrw_tfim_20261012/least_squares_precision.py")
    dump(OUT / "manifest.json", dict(cases=CASES, workers=WORKERS, sources=source_identity(),
        runner_path=str(Path(__file__).resolve()), runner_sha256=digest(__file__), files=files,
        helpers={str(p): digest(p) for p in helpers}, gradient_checks=fitting.check_adjoint(),
        selected_seed=0, basis_gates=IBM, optimization_scope="whole assembled IBM opt0 circuit",
        qubits_initially_zero=False, approximation_degree=1.0, numerical_threads=1, num_processes=1,
        fit_budget_per_original_kraus_s=600, circuit_worker_budget_s=600,
        vcbe_stinespring_rss_limit_bytes=4 * 1024 ** 3, pauli_rss_limit_bytes=6 * 1024 ** 3,
        full_choi_status="skipped_user_request", qiskit_version=qiskit.__version__,
        exclusions="N15 Stinespring and VCBE not executed by explicit user decision; Petz-Ising3 excluded"))
    for method in ("Stinespring", "VCBE", "VCBE+Qiskit opt3"):
        row = base_row(CASES[1], method)
        row.update(status="not_run", measurement_attempted=False,
            absence_reason="User requested omitting N15 Stinespring and VCBE experiments")
        dump(result_path(CASES[1], method), row)
    state = dict(status="running", pid=os.getpid(), completed=[])
    process = None
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
        if "+Qiskit opt3" in method and read(result_path(case, "Raw"))["status"] != "ok":
            row = base_row(case, method)
            row.update(status="not_run", measurement_attempted=False, absence_reason="No IBM opt0 circuit available")
            dump(result_path(case, method), row)
            continue
        budget = 600 * len(read(folders(case)[0] / "input.json")["input_kraus"]) + 60 if method == "VCBE" else 600
        rss_limit = (4 if method in {"VCBE", "Stinespring"} else 6) * 1024 ** 3
        started, peak, reason = time.monotonic(), 0, None
        with (OUT / f"{case}_{slug(method)}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                "--case", case, "--method", method], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL)
            state["current"] = dict(case=case, method=method, pid=process.pid)
            dump(OUT / "status.json", state)
            while process.poll() is None:
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > rss_limit:
                    reason = "memory_limit"
                elif time.monotonic() - started > budget:
                    reason = "timeout"
                if reason:
                    terminate()
                    break
                time.sleep(0.5)
            code = process.wait()
        path = result_path(case, method)
        row = read(path) if path.exists() else base_row(case, method)
        if code:
            # Preserve a completed resource measurement if only optional
            # artifact output or gate classification exhausted its budget.
            row.update(status="synthesis_ok_postprocessing_timeout" if row.get("G") and reason else reason or "worker_error")
            row["stopped_stage"] = read(OUT / "progress.json")["stage"]
            dump(path, row)
            if method == "VCBE" and not result_path(case, "VCBE+Qiskit opt3").exists():
                other = dict(row, method="VCBE+Qiskit opt3", measurement_attempted=False,
                    absence_reason="VCBE fit has no qualified circuit; optimizer not started")
                dump(result_path(case, "VCBE+Qiskit opt3"), other)
        record = dict(case=case, method=method, status=row["status"], exit_code=code,
            termination_reason=reason, wall_s=time.monotonic() - started, peak_rss_bytes=peak,
            result_path=str(path))
        dump(path.with_name(slug(method) + "_resources.json"), record)
        state["completed"].append(record)
        dump(OUT / "status.json", state)
        export(state)
        print(json.dumps(record), flush=True)
    state.update(status="complete", current=None)
    dump(OUT / "status.json", state)
    export(state)


if __name__ == "__main__":
    main()
