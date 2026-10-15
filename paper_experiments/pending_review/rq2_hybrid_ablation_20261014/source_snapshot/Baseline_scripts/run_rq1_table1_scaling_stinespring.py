#!/usr/bin/env python3
"""Bounded NAM Stinespring comparison on the frozen RQ1 scaling inputs.

Keep dense Choi/superoperator verification out of the synthesis budget. Large
matrix storage figures are exact representation sizes, not allocations or RSS.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PENDING = ROOT / "paper_experiments/pending_review"
INPUTS = {
    "TFIM-8": PENDING / "tfim_hrw_full_20261011/cases/TFIM-8/input.json",
    "HRW-8": PENDING / "rq1_paper_suite_20261010/full_v2/inputs/HRW-8.json",
    "HRW-12": PENDING / "tfim_hrw_full_20261011/cases/HRW-12/input.json",
}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(path):
    return sha256(path.read_bytes()).hexdigest()


def worker(args):
    import numpy as np
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry
    from qiskit.quantum_info import Pauli
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
    from experiment_protocol import transpile_channel_circuit
    from baseline_common import QasmOperation, is_nam_operation_clifford

    case = args.worker
    source = INPUTS[case]
    folder = args.out_dir / "cases" / case
    blob = source.read_bytes()
    (folder / "input.json").write_bytes(blob)
    data = json.loads(blob)
    encoded = data["input_kraus"]
    n = len(encoded[0][0][0])
    dim = 1 << n

    def progress(stage):
        dump(folder / "progress.json", {"case": case, "stage": stage,
             "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
        print(json.dumps({"case": case, "stage": stage}), flush=True)

    progress("preflight")
    started = time.perf_counter()
    labels = sorted({term[0] for row in encoded for term in row})
    index = {label: j for j, label in enumerate(labels)}
    coefficients = np.zeros((len(encoded), len(labels)), complex)
    for i, row in enumerate(encoded):
        for label, mag, re, im in row:
            coefficients[i, index[label]] += mag * complex(re, im)
    singular = np.linalg.svd(coefficients, compute_uv=False)
    rank = int(np.sum(singular > 1e-12 * singular[0]))
    completeness_coefficients = defaultdict(complex)
    for row in encoded:
        for left, a, ar, ai in row:
            for right, b, br, bi in row:
                product = Pauli(left).dot(Pauli(right)).to_label()
                phase = 1
                for prefix, value in (("-i", -1j), ("i", 1j), ("-", -1)):
                    if product.startswith(prefix):
                        phase, product = value, product[len(prefix):]
                        break
                completeness_coefficients[product] += (a * complex(ar, ai)).conjugate() * b * complex(br, bi) * phase
    residual = dict(completeness_coefficients)
    residual["I" * n] = residual.get("I" * n, 0) - 1
    tp_error = float(np.sqrt(dim * sum(abs(z) ** 2 for z in residual.values())))
    native = tp_error < 1e-10
    env_bits = (rank - 1).bit_length()
    width = n + env_bits + int(not native)
    iso_shape = [dim * (1 << env_bits) * (2 if not native else 1), dim]
    iso_bytes = 16 * iso_shape[0] * iso_shape[1]
    preflight_s = time.perf_counter() - started
    result = {
        "case": case, "status": "running", "system_qubits": n,
        "input_source": str(source), "input_file_sha256": sha256(blob).hexdigest(),
        "input_kraus_sha256": sha256(json.dumps(encoded, separators=(",", ":")).encode()).hexdigest(),
        "input_kraus_count": len(encoded), "kraus_rank": rank,
        "rank_relative_threshold": 1e-12, "native_cptp": native,
        "tp_frobenius_error": tp_error, "rank_compression_applied": rank < len(encoded),
        "isometry_shape": iso_shape, "isometry_bytes": iso_bytes,
        "planned_W": width, "Lambda": 1.0 if native else None,
        "preflight_s": preflight_s, "variants": [],
        "classical_dense": {"rho_shape": [dim, dim], "rho_bytes": 16 * dim * dim,
            "raw_kraus_collection_bytes": len(encoded) * 16 * dim * dim,
            "compressed_kraus_collection_bytes": rank * 16 * dim * dim,
            "superoperator_shape": [dim * dim, dim * dim], "superoperator_bytes": 16 * dim ** 4,
            "dtype": "complex128", "storage_status": "exact_shape_calculation",
            "scope": "same finite CP map; no large superoperator or Choi allocated"},
        "protocol": {"compiler": "Qiskit Isometry", "qiskit": qiskit.__version__,
            "numpy": np.__version__, "python": sys.version, "platform": platform.platform(),
            "basis": "nam", "basis_gates": ["rz", "h", "x", "cx"],
            "opt_level": 0, "seed_transpiler": 20261013, "qubits_initially_zero": False,
            "numerical_threads": 1, "timeout_s": args.timeout,
            "rss_limit_bytes": int(args.rss_limit_gib * 1024 ** 3),
            "dense_matrix_preflight": "reject when the single isometry alone exceeds RSS limit",
            "input_system_wires": list(range(n)), "output_environment_wires": list(range(env_bits)),
            "output_system_wires": list(range(env_bits, env_bits + n)),
            "output_wire_relabelling": "explicit logical map; no routing swaps"},
        "full_choi": {"status": "not_run", "reason": "larger-than-small verification budget; synthesis does not require dense Choi"},
    }
    if iso_bytes > result["protocol"]["rss_limit_bytes"]:
        result.update(status="preflight_memory_limit", stopped_stage="preflight",
                      resource_reason="single dense isometry exceeds the fixed per-process memory budget")
        dump(folder / "results.json", result)
        progress("preflight_memory_limit")
        return
    dump(folder / "results.json", result)

    progress("matrix_materialization")
    started = time.perf_counter()
    cache = {label: Pauli(label).to_matrix() for label in labels}
    matrices = np.array([sum((a * complex(re, im) * cache[label] for label, a, re, im in row),
                            np.zeros((dim, dim), complex)) for row in encoded])
    del cache
    if rank < len(matrices):
        vectors = np.column_stack([k.reshape(-1, order="F") for k in matrices])
        left, values, _ = np.linalg.svd(vectors, full_matrices=False)
        matrices = np.array([(left[:, j] * values[j]).reshape((dim, dim), order="F") for j in range(rank)])
    completeness = sum((k.conj().T @ k for k in matrices), np.zeros((dim, dim), complex))
    eigenvalues = np.linalg.eigvalsh(completeness)
    beta = 1.0 if native else float(max(1.0, np.nextafter(eigenvalues[-1], np.inf)))
    materialization_s = time.perf_counter() - started
    result.update(materialization_s=materialization_s, Lambda=beta,
                  completeness_eigenvalue_bounds=[float(eigenvalues[0]), float(eigenvalues[-1])])
    progress("isometry_completion")
    started = time.perf_counter()
    output_dimension = dim * (1 if native else 2)
    padded = np.zeros((1 << env_bits, output_dimension, dim), complex)
    padded[:rank, :dim] = matrices / np.sqrt(beta)
    if not native:
        residual_matrix = np.eye(dim) - completeness / beta
        values, vectors = np.linalg.eigh((residual_matrix + residual_matrix.conj().T) / 2)
        assert values.min() > -1e-12
        padded[0, dim:] = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T
    iso = padded.transpose(1, 0, 2).reshape(iso_shape)
    orthogonality_error = float(np.linalg.norm(iso.conj().T @ iso - np.eye(dim)))
    assert orthogonality_error < 1e-10, orthogonality_error
    np.save(folder / "isometry.npy", iso)
    circuit = QuantumCircuit(width)
    circuit.append(Isometry(iso, 0, 0), range(width))
    completion_s = time.perf_counter() - started
    result.update(completion_and_wrapper_s=completion_s, isometry_orthogonality_error=orthogonality_error)
    dump(folder / "results.json", result)
    progress("nam_synthesis")
    started = time.perf_counter()
    compiled = transpile_channel_circuit(circuit, basis_gates=["rz", "h", "x", "cx"],
        optimization_level=0, seed_transpiler=20261013, approximation_degree=1.0)
    synthesis_s = time.perf_counter() - started
    counts = {str(gate): int(count) for gate, count in compiled.count_ops().items()}
    assert set(counts) <= {"rz", "h", "x", "cx"}
    G = sum(counts.values())
    variant = {"mode": "native" if native else "flagged_minimum_beta", "basis": "nam",
        "status": "synthesized", "G": G, "CX": counts.get("cx", 0), "W": compiled.num_qubits,
        "Lambda": beta, "Lambda_G": beta * G, "counts": counts, "synthesis_s": synthesis_s,
        "compile_time_s": preflight_s + materialization_s + completion_s + synthesis_s,
        "time_scope": "preflight + materialization/rank + completion + NAM synthesis; excludes imports, verification and output I/O",
        "success_contract": "E_original(rho)" if native else "E_original(rho)/Lambda",
        "full_choi": result["full_choi"], "circuit_artifact": "basis circuit measured in memory; input isometry and source preserved"}
    result["variants"] = [variant]
    dump(folder / "results.json", result)
    progress("gate_classification")
    nc = 0
    for instruction in compiled.data:
        operation = instruction.operation
        if operation.name == "rz":
            is_clifford = is_nam_operation_clifford(QasmOperation("rz", (str(float(operation.params[0])),)))
            assert is_clifford is not None
            nc += not is_clifford
    variant.update(status="ok", non_clifford=int(nc))
    result.update(status="ok")
    dump(folder / "results.json", result)
    progress("finished")
    print(json.dumps({"case": case, "status": "ok", "G": G, "W": compiled.num_qubits,
                      "Lambda": beta, "compile_time_s": variant["compile_time_s"]}), flush=True)


def coordinator(args):
    import psutil
    out = args.out_dir.resolve()
    args.out_dir = out
    if (out / "status.json").exists():
        raise ValueError("Choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": args.cases, "host": "local", "jobs": 1, "numerical_threads": 1,
        "per_case_timeout_s": args.timeout, "per_case_rss_limit_bytes": int(args.rss_limit_gib * 1024 ** 3),
        "input_files": {case: {"path": str(INPUTS[case]), "sha256": identity(INPUTS[case])} for case in args.cases},
        "sources": {str(path): identity(path) for path in [Path(__file__).resolve(), ROOT / "src/experiment_protocol.py", ROOT / "Baseline_scripts/baseline_common.py"]},
        "scope": "Stinespring additions only; no Raw optimizer dispatch; no large dense Choi/superoperator"})
    state = {"status": "running", "pid": os.getpid(), "completed": []}
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    for case in args.cases:
        folder = out / "cases" / case
        folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        peak, stop = 0, None
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--out-dir", str(out),
                   "--worker", case, "--timeout", str(args.timeout), "--rss-limit-gib", str(args.rss_limit_gib)]
        with (folder / "run.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            state["current"] = {"case": case, "pid": process.pid}
            dump(out / "status.json", state)
            while process.poll() is None:
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > args.rss_limit_gib * 1024 ** 3:
                    stop = "memory_limit"
                elif time.monotonic() - started > args.timeout:
                    stop = "timeout"
                if stop:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(0.2)
            code = process.wait()
        path = folder / "results.json"
        result = json.loads(path.read_text()) if path.exists() else {"case": case, "variants": []}
        stage = json.loads((folder / "progress.json").read_text()).get("stage") if (folder / "progress.json").exists() else "importing"
        if stop or code != 0:
            result.update(status=stop or "failed", stopped_stage=stage)
            dump(path, result)
        record = {"case": case, "status": result["status"], "exit_code": code,
                  "resource_stop": stop, "wall_s": time.monotonic() - started,
                  "peak_rss_bytes": peak, "last_stage": stage, "result": str(path)}
        dump(folder / "resources.json", record)
        state["completed"].append(record)
        print(json.dumps(record), flush=True)
    state.update(status="finished", current=None)
    dump(out / "status.json", state)
    dump(out / "results.json", [json.loads((out / "cases" / case / "results.json").read_text()) for case in args.cases])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=INPUTS, default=["HRW-12", "TFIM-8", "HRW-8"])
    parser.add_argument("--worker", choices=INPUTS)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--rss-limit-gib", type=float, default=3)
    arguments = parser.parse_args()
    if arguments.timeout <= 0 or arguments.rss_limit_gib <= 0:
        parser.error("Budgets must be positive")
    try:
        worker(arguments) if arguments.worker else coordinator(arguments)
    except Exception:
        traceback.print_exc()
        raise
