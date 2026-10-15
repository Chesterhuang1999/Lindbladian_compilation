"""Rerun the frozen TFIM-8 and Petz-Ising3 Stinespring inputs in IBM opt0.

Run the archive smoke check into OUT before starting this isolated batch.
No complete Choi/superoperator is formed. Resource counts are checkpointed
before circuit output or optional small-circuit column validation.
"""
from __future__ import annotations

import argparse
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
OUT = ROOT / "Baseline_results/rq1_tfim8_petz_stinespring_ibm_20261014"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["QISKIT_PARALLEL"] = "FALSE"
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil
import qiskit
from qiskit import QuantumCircuit, qasm2, qpy
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import Pauli, Statevector
from experiment_protocol import source_identity, transpile_channel_circuit

INPUTS = {
    "Petz_Ising3": ROOT / "paper_experiments/pending_review/raw_baseline_20261012/inputs/Petz_Ising3.json",
    "TFIM-8": ROOT / "paper_experiments/pending_review/tfim_hrw_full_20261011/cases/TFIM-8/input.json",
}
CACHED = ROOT / "Baseline_results/table1_scaling_stinespring_20261013/cases/TFIM-8"
IBM = ["u1", "u2", "u3", "cx"]
SEED = 20261013
TIMEOUT = 600
RSS_LIMIT = 4 * 1024 ** 3
TOL = 1e-10


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


def identity(encoded):
    normalized = [[[str(p), float(c), float(re), float(im)] for p, c, re, im in row]
                  for row in encoded]
    return sha256(json.dumps(normalized, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def progress(case, stage):
    dump(OUT / "cases" / case / "progress.json",
         dict(case=case, stage=stage, pid=os.getpid(),
              updated_at_utc=datetime.now(timezone.utc).isoformat()))
    print(case, stage, flush=True)


def worker(case):
    plan = read(OUT / "manifest.json")
    assert plan["sources"] == source_identity()
    assert plan["runner_sha256"] == digest(__file__)
    for path, value in plan["files"][case].items():
        assert digest(path) == value, path
    folder = OUT / "cases" / case
    blob = INPUTS[case].read_bytes()
    (folder / "input.json").write_bytes(blob)
    encoded = json.loads(blob)["input_kraus"]
    n = len(encoded[0][0][0])
    d = 1 << n
    row = dict(case=case, method="Stinespring", status="running", basis="IBM",
        target_path=str(INPUTS[case]), target_file_sha256=digest(INPUTS[case]),
        target_kraus_sha256=identity(encoded), input_kraus_sha256=identity(encoded),
        input_file_sha256=digest(INPUTS[case]), system_qubits=n,
        input_kraus_count=len(encoded), lowering_optimization_level=0,
        seed_transpiler=SEED, approximation_degree=1.0, qubits_initially_zero=False,
        full_choi_status="skipped_user_request", non_clifford_status="not_required_for_resource_table",
        protocol=dict(qiskit=qiskit.__version__, numpy=np.__version__,
            python=sys.version, platform=platform.platform(), basis_gates=IBM,
            num_processes=1, numerical_threads=1, timeout_s=TIMEOUT, rss_limit_bytes=RSS_LIMIT))
    path = folder / "results.json"
    row.update(original_result_path=str(path))
    dump(path, row)
    progress(case, "materialize_and_check_kraus_rank")
    started = time.perf_counter()
    cache = {p: Pauli(p).to_matrix() for p in {t[0] for terms in encoded for t in terms}}
    matrices = np.array([sum((c * complex(re, im) * cache[p] for p, c, re, im in terms),
                            np.zeros((d, d), complex)) for terms in encoded])
    del cache
    vectors = np.column_stack([k.reshape(-1, order="F") for k in matrices])
    left, values, _ = np.linalg.svd(vectors, full_matrices=False)
    assert values[0] > 0
    rank = int(np.sum(values > 1e-12 * values[0]))
    compressed = rank < len(matrices)
    if compressed:
        matrices = np.array([(left[:, j] * values[j]).reshape((d, d), order="F")
                             for j in range(rank)])
    row.update(kraus_rank=rank, rank_relative_threshold=1e-12,
        rank_compression_applied=compressed, kraus_singular_values=values.tolist(),
        discarded_kraus_squared_weight=float(np.sum(values[rank:] ** 2)))
    del left, vectors
    completeness = sum((k.conj().T @ k for k in matrices), np.zeros((d, d), complex))
    eig = np.linalg.eigvalsh(completeness)
    tp_error = float(np.linalg.norm(completeness - np.eye(d)))
    native = tp_error < TOL
    beta = 1.0 if native else float(max(1.0, np.nextafter(eig[-1], np.inf)))
    env_bits = (rank - 1).bit_length()
    env_dim = 1 << env_bits
    width = n + env_bits + int(not native)
    output_dimension = d * (1 if native else 2)
    shape = (output_dimension * env_dim, d)
    row.update(native_cptp=native, tp_frobenius_error=tp_error,
        completeness_eigenvalue_bounds=[float(eig[0]), float(eig[-1])],
        Lambda=beta, planned_W=width, isometry_shape=list(shape),
        isometry_bytes=16 * shape[0] * shape[1],
        mode="native" if native else "flagged_minimum_beta",
        materialization_and_rank_s=time.perf_counter() - started,
        input_system_wires=list(range(n)), output_environment_wires=list(range(env_bits)),
        output_system_wires=list(range(env_bits, env_bits + n)),
        success_flag_wire=None if native else width - 1,
        output_wire_relabelling="explicit logical map; no routing swaps")
    if row["isometry_bytes"] > RSS_LIMIT:
        row.update(status="preflight_memory_limit", stopped_stage="isometry_preflight")
        dump(path, row)
        return
    progress(case, "prepare_isometry")
    started = time.perf_counter()
    padded = np.zeros((env_dim, output_dimension, d), complex)
    padded[:rank, :d] = matrices / np.sqrt(beta)
    if not native:
        residual = np.eye(d) - completeness / beta
        residual_values, residual_vectors = np.linalg.eigh((residual + residual.conj().T) / 2)
        assert residual_values.min() > -1e-12
        padded[0, d:] = (residual_vectors * np.sqrt(np.maximum(residual_values, 0))) @ residual_vectors.conj().T
    iso = padded.transpose(1, 0, 2).reshape(shape)
    row["reused_isometry"] = False
    if case == "TFIM-8":
        old = read(CACHED / "results.json")
        assert old["input_kraus_sha256"] == row["target_kraus_sha256"]
        assert old["input_file_sha256"] == row["target_file_sha256"]
        assert old["protocol"]["seed_transpiler"] == SEED
        assert not compressed and old["rank_compression_applied"] == compressed
        assert old["native_cptp"] == native and old["planned_W"] == width
        assert abs(old["Lambda"] - beta) < 1e-12
        frozen = np.load(CACHED / "isometry.npy")
        cached_error = float(np.linalg.norm(frozen - iso))
        assert cached_error < TOL
        row.update(reused_isometry=True, cached_isometry_path=str(CACHED / "isometry.npy"),
            cached_isometry_sha256=digest(CACHED / "isometry.npy"),
            cached_isometry_reconstruction_error=cached_error,
            cached_original_result_path=str(CACHED / "results.json"),
            cached_original_result_sha256=digest(CACHED / "results.json"))
        iso = frozen
        beta = old["Lambda"]
        row["Lambda"] = beta
    orthogonality = float(np.linalg.norm(iso.conj().T @ iso - np.eye(d)))
    assert orthogonality < TOL
    branch_errors = [float(np.linalg.norm(iso[j:d * env_dim:env_dim, :] * np.sqrt(beta) - k))
                     for j, k in enumerate(matrices)]
    assert max(branch_errors) < TOL
    iso_path = folder / "isometry.npy"
    np.save(iso_path, iso)
    row.update(isometry_path=str(iso_path), isometry_sha256=digest(iso_path),
        isometry_orthogonality_error=orthogonality,
        isometry_success_kraus_max_error=max(branch_errors),
        isometry_preparation_s=time.perf_counter() - started)
    dump(path, row)
    source = QuantumCircuit(width)
    source.append(Isometry(iso, 0, 0), range(width))
    progress(case, "ibm_opt0_synthesis")
    started = time.perf_counter()
    ibm = transpile_channel_circuit(source, basis_gates=IBM, optimization_level=0,
        seed_transpiler=SEED, approximation_degree=1.0, qubits_initially_zero=False,
        num_processes=1)
    lowering_s = time.perf_counter() - started
    if ibm.layout is not None:
        assert ibm.layout.initial_index_layout() == list(range(width))
        assert ibm.layout.final_index_layout() == list(range(width))
    operations = {str(g): int(count) for g, count in ibm.count_ops().items()}
    assert set(operations) <= set(IBM)
    G = sum(operations.get(g, 0) for g in IBM)
    assert ibm.num_qubits == width and G > 0
    row.update(status="ok", G=G, W=width, counts=operations, CX=operations.get("cx", 0),
        Lambda_G=beta * G, lowering_s=lowering_s, synthesis_s=lowering_s,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        time_scope="IBM opt0 synthesis only; excludes imports, matrix construction, validation and I/O",
        circuit_artifact="IBM circuit counted in memory; input, isometry and runner preserved",
        compiled_isometry_validation_status="not_run_large_circuit")
    assert plan["sources"] == source_identity()
    dump(path, row)
    progress(case, "resource_counts_saved")
    if width <= 8:
        qasm_path = folder / "stinespring_ibm.qasm"
        qasm2.dump(ibm, qasm_path)
        with (folder / "stinespring_ibm.qpy").open("wb") as stream:
            qpy.dump(ibm, stream)
        errors = [float(np.linalg.norm(Statevector.from_int(column, 1 << width).evolve(ibm).data - iso[:, column]))
                  for column in range(d)]
        assert max(errors) < TOL
        row.update(qasm_path=str(qasm_path), qasm_sha256=digest(qasm_path),
            qpy_path=str(folder / "stinespring_ibm.qpy"),
            qpy_sha256=digest(folder / "stinespring_ibm.qpy"),
            compiled_isometry_column_max_error=max(errors),
            compiled_isometry_validation_status="passed_all_input_columns")
        dump(path, row)
    progress(case, "finished")
    print(json.dumps({k: row[k] for k in ("case", "status", "G", "Lambda", "W", "synthesis_s")}), flush=True)


def coordinator():
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and smoke["sources"] == source_identity()
    assert not (OUT / "manifest.json").exists(), "Use a fresh batch; preserve historical measurements"
    files = {case: {str(path): digest(path)} for case, path in INPUTS.items()}
    for filename in ("input.json", "isometry.npy", "results.json"):
        files["TFIM-8"][str(CACHED / filename)] = digest(CACHED / filename)
    dump(OUT / "manifest.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        cases=list(INPUTS), methods=["Stinespring"], sources=source_identity(),
        runner_path=str(Path(__file__).resolve()), runner_sha256=digest(__file__), files=files,
        basis_gates=IBM, lowering_optimization_level=0, seed_transpiler=SEED,
        approximation_degree=1.0, qubits_initially_zero=False, num_processes=1,
        numerical_threads=1, per_case_timeout_s=TIMEOUT, per_case_rss_limit_bytes=RSS_LIMIT,
        full_choi_status="skipped_user_request", qiskit_version=qiskit.__version__,
        scope="Stinespring IBM reruns only; existing Raw optimizer queues stay paused"))
    state = dict(status="running", pid=os.getpid(), completed=[])
    process = None

    def stop(signum, frame):
        if process is not None and process.poll() is None:
            process.terminate()
        state["status"] = "paused_user_request"
        dump(OUT / "status.json", state)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    rows = []
    for case in INPUTS:
        folder = OUT / "cases" / case
        folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        peak, reason = 0, None
        with (folder / "run.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", case],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state["current"] = dict(case=case, pid=process.pid)
            dump(OUT / "status.json", state)
            while process.poll() is None:
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > RSS_LIMIT:
                    reason = "memory_limit"
                elif time.monotonic() - started > TIMEOUT:
                    reason = "timeout"
                if reason:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(0.5)
            code = process.wait()
        path = folder / "results.json"
        row = read(path) if path.exists() else dict(case=case, method="Stinespring", status="worker_error", basis="IBM")
        stage_path = folder / "progress.json"
        stage = read(stage_path)["stage"] if stage_path.exists() else "imports"
        record = dict(case=case, exit_code=code, termination_reason=reason,
            wall_s=time.monotonic() - started, peak_rss_bytes=peak, last_stage=stage,
            result_path=str(path))
        if code != 0:
            row.update(status=reason or "worker_error", stopped_stage=stage)
        row.update(resources=record)
        dump(path, row)
        dump(folder / "resources.json", record)
        state["completed"].append(dict(record, status=row["status"]))
        dump(OUT / "status.json", state)
        print(json.dumps(state["completed"][-1]), flush=True)
        rows.append(row)
    state.update(status="complete", current=None)
    dump(OUT / "results.json", dict(status="complete", rows=rows, manifest_path=str(OUT / "manifest.json")))
    dump(OUT / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=INPUTS)
    args = parser.parse_args()
    worker(args.worker) if args.worker else coordinator()


if __name__ == "__main__":
    main()
