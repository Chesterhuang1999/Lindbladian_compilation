"""Compare the HRW-8 table methods after changing only final NAM to IBM lowering.

Run in qiskit_simulate after this batch's archive smoke check. Reuse the saved
Kraus representation, fitted VCBE circuits, and Stinespring isometry. The old
NAM optimizer batch and external queues stay paused.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Baseline_results/rq1_hrw8_ibm_backend_20261014"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["QISKIT_PARALLEL"] = "FALSE"
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / folder) for folder in ("src", "Baseline_scripts", "scripts/eval_redesign")]

import numpy as np
import psutil
import qiskit
from qiskit import QuantumCircuit, qasm2, qpy
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import Operator
from baseline_common import QasmOperation, is_ibm_operation_clifford
from block_encoding import BlockEncoding
from build_rq1_merged_table import map_error
from channel_IR import channel_ensemble
from channel_LCU import channel_to_LCU
from experiment_protocol import source_identity, transpile_channel_circuit
from rq1_pipeline import build_pipeline, deserialize_kraus, input_identity

NAM = ["rz", "h", "x", "cx"]
IBM = ["u1", "u2", "u3", "cx"]
METHODS = ["Stinespring", "VCBE", "VCBE+opt3", "Raw", "Raw+Qiskit opt3", "Ours"]
RAW = ROOT / "paper_experiments/pending_review/rq1_paper_suite_20261010/full_v2/runs/HRW-8/raw"
OURS = ROOT / "Baseline_results/table1_scaling_ours_20261013/cases/HRW-8"
VCBE = ROOT / "Baseline_results/vcbe_scaling_20261014/cases/HRW-8"
STINE = ROOT / "Baseline_results/table1_scaling_stinespring_20261013/cases/HRW-8"
OLD_OPT = ROOT / "Baseline_results/rq1_raw_qiskit_opt3_20261014/cases/HRW-8/results.json"
FILES = [RAW / "results.json", RAW / "channel_nam_opt0.qasm", OURS / "input.json",
         OURS / "results.json", OURS / "channel_nam.qasm", VCBE / "results.json",
         VCBE / "vcbe_source_tree_source.qpy", VCBE / "vcbe_preopt3_tree_source.qpy",
         STINE / "results.json", STINE / "isometry.npy", STINE / "input.json", OLD_OPT]
FILES += [VCBE / f"vcbe_active_block_{j}.qpy" for j in range(16)]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 ** 2), b""):
            hasher.update(part)
    return hasher.hexdigest()


def dump(path, data):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def progress(method, stage):
    dump(OUT / "progress.json", dict(method=method, stage=stage, pid=os.getpid(),
         updated_at_utc=datetime.now(timezone.utc).isoformat()))
    print(method, stage, flush=True)


def save_qasm(circuit, path):
    # A single q register avoids legacy qreg s collisions in QASM parsers.
    flat = QuantumCircuit(circuit.num_qubits)
    flat.compose(circuit, inplace=True)
    qasm2.dump(flat, path)


def counts(circuit, gates):
    operations = {str(k): int(v) for k, v in circuit.count_ops().items()}
    assert set(operations) <= {*gates, "reset", "barrier"}, operations
    return sum(operations.get(gate, 0) for gate in gates), operations


def save_result(method, original, circuit, lowering_s, **extra):
    G, operations = counts(circuit, IBM)
    assert circuit.num_qubits == original["W"]
    progress(method, "classification")
    started = time.perf_counter()
    nc, classifications = 0, {}
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name not in {"u1", "u2", "u3"}:
            continue
        key = (operation.name, tuple(str(float(value)) for value in operation.params))
        if key not in classifications:
            classifications[key] = is_ibm_operation_clifford(QasmOperation(*key))
            assert classifications[key] is not None
        nc += classifications[key] is False
    classification_s = time.perf_counter() - started
    row = dict(case="HRW-8", method=method, status="ok", basis="IBM",
        G_NAM=original["G"], G_IBM=G, Lambda=original["Lambda"], W=original["W"],
        C_NAM=original["Lambda"] * original["G"], C_IBM=original["Lambda"] * G,
        G_IBM_over_NAM=G / original["G"], CX_NAM=original["counts"].get("cx", 0),
        CX_IBM=operations.get("cx", 0), NC_NAM=original.get("non_clifford"), NC_IBM=int(nc),
        counts_NAM=original["counts"], counts_IBM=operations,
        original_result_path=original["source_result_path"],
        lowering_s=lowering_s, classification_s=classification_s,
        lowering_optimization_level=0 if method != "Raw+Qiskit opt3" else 3,
        approximation_degree=1.0, qubits_initially_zero=False,
        full_choi_status="skipped_user_request", peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **extra)
    slug = method.replace("+", "_").replace(" ", "_")
    if method != "Stinespring":
        path = OUT / f"{slug}_ibm.qasm"
        save_qasm(circuit, path)
        row.update(qasm_path=str(path), qasm_sha256=digest(path))
    else:
        row.update(circuit_artifact="native circuit counted in memory; cached isometry and reproducible source retained")
    dump(OUT / f"{slug}_results.json", row)
    print("RESULT", method, "NAM", original["G"], "IBM", G, "C", row["C_IBM"], flush=True)
    return row


def worker(kind):
    plan = read(OUT / "manifest.json")
    assert plan["sources"] == source_identity() and plan["runner_sha256"] == digest(__file__)
    for path, value in plan["files"].items():
        assert digest(path) == value, path
    target = read(OURS / "input.json")["input_kraus"]
    raw, ours, vcbe, stine = [read(folder / "results.json") for folder in (RAW, OURS, VCBE, STINE)]
    canonical_hash = input_identity(deserialize_kraus(target))
    assert canonical_hash == raw["canonicalization"]["canonical_input_sha256"] == stine["input_kraus_sha256"]
    assert map_error(target, ours["assembly"]["compiled_kraus"]) < 1e-10
    assert map_error(target, vcbe["kraus"]) < 1e-10
    if kind == "small":
        progress("Raw", "reassemble_frozen_canonical_input")
        source, registers, _ = build_pipeline(deserialize_kraus(raw["canonical_kraus"]), "Raw")
        start = time.perf_counter()
        nam = transpile_channel_circuit(source, basis_gates=NAM, optimization_level=0, seed_transpiler=20261009, num_processes=1)
        G, nam_counts = counts(nam, NAM)
        assert G == raw["metrics"]["G"] and nam_counts == raw["metrics"]["counts"]
        del nam
        reference = dict(G=G, counts=nam_counts, W=source.num_qubits, Lambda=raw["Lambda"],
                         non_clifford=raw["non_clifford"], source_result_path=str(RAW / "results.json"))
        progress("Raw", "ibm_opt0")
        start = time.perf_counter()
        ibm = transpile_channel_circuit(source, basis_gates=IBM, optimization_level=0, seed_transpiler=20261009, num_processes=1)
        save_result("Raw", reference, ibm, time.perf_counter() - start,
                    nam_reproduction="passed exact gate histogram", registers=registers)
        progress("Raw+Qiskit opt3", "ibm_opt3")
        imported = qasm2.load(OUT / "Raw_ibm.qasm")
        start = time.perf_counter()
        optimized = transpile_channel_circuit(imported, basis_gates=IBM, optimization_level=3,
                    seed_transpiler=20261009, approximation_degree=1.0, num_processes=1)
        optimizer_s = time.perf_counter() - start
        assert optimized.layout is None or optimized.layout.final_index_layout() == list(range(imported.num_qubits))
        old = read(OLD_OPT)
        assert old["input_qasm_sha256"] == raw["qasm_sha256"] and old["status"] == "ok"
        reference = dict(G=old["G"], counts=old["counts"], W=old["W"], Lambda=old["Lambda"],
                         non_clifford=old["non_clifford"], source_result_path=str(OLD_OPT))
        save_result("Raw+Qiskit opt3", reference, optimized, optimizer_s, optimizer_s=optimizer_s,
                    optimizer_s_NAM=old["optimizer_s"], input_qasm_sha256=digest(OUT / "Raw_ibm.qasm"))
        del imported, optimized, ibm, source
        for method, name in (("VCBE", "vcbe_source_tree"), ("VCBE+opt3", "vcbe_preopt3_tree")):
            progress(method, "load_saved_pre_lowering_circuit")
            with (VCBE / f"{name}_source.qpy").open("rb") as stream:
                source = qpy.load(stream)[0]
            metric = vcbe["methods"][name]["metrics"]["nam_opt0"]
            assert metric["block_qualified"]
            nam = transpile_channel_circuit(source, basis_gates=NAM, optimization_level=0, seed_transpiler=20261009, num_processes=1)
            G, nam_counts = counts(nam, NAM)
            assert G == metric["G"] and nam_counts == metric["counts"]
            del nam
            progress(method, "ibm_opt0")
            start = time.perf_counter()
            ibm = transpile_channel_circuit(source, basis_gates=IBM, optimization_level=0, seed_transpiler=20261009, num_processes=1)
            reference = dict(G=G, counts=nam_counts, W=metric["W"], Lambda=metric["Lambda"],
                             non_clifford=metric["NC"], source_result_path=str(VCBE / "results.json"))
            save_result(method, reference, ibm, time.perf_counter() - start,
                        nam_reproduction="passed exact gate histogram", reused_vcbe_fit=True,
                        inner_opt3_basis="IBM" if method == "VCBE+opt3" else "none")
        progress("VCBE", "active_block_basis_checks")
        checks=[]
        for j in range(16):
            with (VCBE / f"vcbe_active_block_{j}.qpy").open("rb") as stream:
                block = qpy.load(stream)[0]
            converted = transpile_channel_circuit(block, basis_gates=IBM, optimization_level=0, seed_transpiler=20261009)
            error = float(np.linalg.norm(Operator(block).data - Operator(converted).data))
            assert error < 1e-10
            checks.append(dict(block=j, ibm_lowering_unitary_error=error))
        dump(OUT / "vcbe_basis_checks.json", checks)
        progress("Ours", "reassemble_frozen_rewritten_kraus")
        kraus = deserialize_kraus(ours["assembly"]["compiled_kraus"])
        options = ours["assembly"]["matrix_order_options"]
        def backend(ms):
            return BlockEncoding(ms, placement_options={}, matrix_order_options=options)
        source, registers = channel_to_LCU(channel_ensemble([deepcopy(kraus)]), structure="tree", opt="Matrix-order", fstar=True, block_encoding_cls=backend)
        nam = transpile_channel_circuit(source, basis_gates=NAM, optimization_level=0, seed_transpiler=20261009, num_processes=1)
        G, nam_counts = counts(nam, NAM)
        assert G == ours["G"] and nam_counts == ours["counts"]
        assert registers == ours["registers"] and np.isclose(source.metadata["subnormalization_factor"],ours["Lambda"],rtol=1e-12)
        del nam
        progress("Ours", "ibm_opt0")
        start = time.perf_counter()
        ibm = transpile_channel_circuit(source, basis_gates=IBM, optimization_level=0, seed_transpiler=20261009, num_processes=1)
        reference = dict(G=G, counts=nam_counts, W=ours["W"], Lambda=ours["Lambda"],
                         non_clifford=ours["non_clifford"], source_result_path=str(OURS / "results.json"))
        save_result("Ours", reference, ibm, time.perf_counter() - start,
                    nam_reproduction="passed exact gate histogram", registers=registers,
                    backend_change_scope="final lowering only; original NAM-based placement cost retained", reused_rewrite=True)
    else:
        progress("Stinespring", "load_saved_isometry")
        iso = np.load(STINE / "isometry.npy")
        assert not stine["rank_compression_applied"] and stine["native_cptp"]
        env_size = 2 ** (stine["planned_W"] - 8)
        errors=[]
        for j, ms in enumerate(deserialize_kraus(target)):
            errors.append(float(np.linalg.norm(iso[j::env_size, :] - ms.eff_op().to_matrix())))
        assert max(errors) < 1e-10
        source = QuantumCircuit(stine["planned_W"])
        source.append(Isometry(iso,0,0), range(source.num_qubits))
        progress("Stinespring", "ibm_opt0_from_saved_isometry")
        start = time.perf_counter()
        ibm = transpile_channel_circuit(source, basis_gates=IBM, optimization_level=0,
                    seed_transpiler=stine["protocol"]["seed_transpiler"], approximation_degree=1.0, num_processes=1)
        lowering_s = time.perf_counter() - start
        original = stine["variants"][0]
        reference = dict(original, source_result_path=str(STINE / "results.json"))
        row=save_result("Stinespring", reference, ibm, lowering_s,
            isometry_sha256=digest(STINE / "isometry.npy"), reused_isometry=True,
            isometry_target_kraus_max_error=max(errors), seed_transpiler=stine["protocol"]["seed_transpiler"])
        assert row["CX_IBM"] == row["CX_NAM"]
    assert source_identity() == plan["sources"]
    progress(kind, "done")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("small","stinespring"))
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    smoke=read(OUT/"smoke.json")
    assert smoke["passed"] and smoke["sources"]==source_identity()
    assert not (OUT/"manifest.json").exists(), "Choose a fresh batch; do not overwrite measurements"
    dump(OUT/"manifest.json",dict(case="HRW-8", methods=METHODS, basis_gates=IBM,
        qubits_initially_zero=False, approximation_degree=1.0, numerical_threads=1,
        sources=source_identity(), runner_sha256=digest(__file__),
        files={str(path):digest(path) for path in FILES}, qiskit_version=qiskit.__version__,
        time_scope="final lowering or opt3 only; excludes reconstruction, imports, validation and I/O",
        budget_s=600, rss_budget_bytes=4*1024**3, full_choi_status="skipped_user_request"))
    state=dict(status="running", workers=[])
    process=None
    def stop(signum, frame):
        if process is not None and process.poll() is None: process.terminate()
        state["status"]="paused_user_request"
        dump(OUT/"status.json",state)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    for kind in ("small","stinespring"):
        start=time.monotonic(); peak=0;reason=None
        with (OUT/f"{kind}.log").open("w") as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker",kind],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state["current"]=dict(worker=kind,pid=process.pid)
            dump(OUT/"status.json",state)
            try:
                while process.poll() is None:
                    try: peak=max(peak,psutil.Process(process.pid).memory_info().rss)
                    except (psutil.Error,PermissionError): pass
                    if peak>4*1024**3: reason="memory_limit"
                    elif time.monotonic()-start>600: reason="timeout"
                    if reason: process.kill();break
                    time.sleep(.5)
            finally:
                if process.poll() is None: process.terminate()
                code=process.wait()
        record=dict(worker=kind,exit_code=code,termination_reason=reason,wall_s=time.monotonic()-start,peak_rss_bytes=peak)
        state["workers"].append(record);dump(OUT/"status.json",state)
        print(json.dumps(record),flush=True)
        if code: raise RuntimeError(f"{kind} failed; inspect its preserved log")
    rows=[read(OUT/(method.replace("+","_").replace(" ","_")+"_results.json")) for method in METHODS]
    dump(OUT/"results.json",dict(status="complete",case="HRW-8",methods=rows,manifest_path=str(OUT/"manifest.json")))
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT/"Results/rq1_hrw8_ibm_backend_20261014.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows([{key:json.dumps(value,separators=(",",":")) if isinstance(value,(dict,list)) else value for key,value in row.items()} for row in rows])
    state.update(status="complete",current=None);dump(OUT/"status.json",state)
    print("Completed HRW-8 IBM backend comparison",flush=True)


if __name__=="__main__":
    main()
