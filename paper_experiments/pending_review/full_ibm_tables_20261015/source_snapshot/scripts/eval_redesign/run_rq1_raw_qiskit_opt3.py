"""Optimize the current table's saved Raw NAM circuits with Qiskit level 3.

Run in qiskit_simulate. This is a separate local batch; the external optimizer
queue stays paused. No VCBE fitting or Stinespring synthesis is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["QISKIT_PARALLEL"] = "FALSE"
os.environ["QISKIT_NUM_PROCS"] = "1"

import psutil
import qiskit
from qiskit import qasm2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Baseline_results/rq1_raw_qiskit_opt3_20261014"
CASES = ("TFIM-4", "TFIM-8", "TFIM-12", "HRW-8", "HRW-12", "RepDQE-d3", "Petz_Ising3")
NAM = ("rz", "h", "x", "cx")
SEED = 20261009
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
from baseline_common import QREG2_RE, is_nam_operation_clifford, parse_qasm_operation
from experiment_protocol import source_identity, transpile_channel_circuit
from rq1_pipeline import deserialize_kraus, verify_channel


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


def recount(path):
    counts, width, non_clifford = Counter(), 0, 0
    classifications = {}
    with Path(path).open() as stream:
        for line in stream:
            match = QREG2_RE.search(line)
            if match:
                width += int(match.group(1))
            operation = parse_qasm_operation(line.strip())
            if operation:
                if operation.name not in {*NAM, "reset", "barrier"}:
                    raise ValueError(f"Unexpected output operation: {operation.name}")
                counts[operation.name] += 1
                if operation.name in NAM:
                    key = (operation.name, operation.params)
                    if key not in classifications:
                        classifications[key] = is_nam_operation_clifford(operation)
                    if classifications[key] is None:
                        raise ValueError(f"Unclassified operation: {operation}")
                    non_clifford += classifications[key] is False
    return sum(counts[gate] for gate in NAM), width, dict(counts), int(non_clifford)


def worker(batch, case):
    plan = read(batch / "manifest.json")
    job = plan["inputs"][case]
    directory = batch / "cases" / case
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress = lambda stage: dump(directory / "progress.json", {
        "stage": stage, "elapsed_s": time.perf_counter() - started})
    progress("input_load")
    source = read(job["source_result_path"])
    input_path = Path(job["qasm_path"])
    if digest(input_path) != job["qasm_sha256"]:
        raise ValueError("Raw QASM changed since collection")
    if source["canonicalization"]["canonical_input_sha256"] != job["canonical_input_sha256"]:
        raise ValueError("Raw target identity changed")
    # Older exports use qreg s, which collides with qelib1's S gate in Qiskit's
    # current parser. Alpha-renaming preserves register order and bit indices.
    text = input_path.read_text()
    names = re.findall(r"\b(?:qreg|creg)\s+([A-Za-z_]\w*)\s*\[", text)
    register_rename = {name: f"raw_register_{i}" for i, name in enumerate(names)}
    pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in names) + r")(?=\s*\[)")
    adapted = directory / "input_alpha_renamed.qasm"
    adapted.write_text(pattern.sub(lambda match: register_rename[match.group(1)], text))
    del text
    circuit = qasm2.load(adapted)
    if circuit.num_qubits != int(job["W"]) or sum(circuit.count_ops().get(g, 0) for g in NAM) != int(job["G"]):
        raise ValueError("Raw circuit metrics disagree with its collected row")
    input_load_s = time.perf_counter() - started
    progress("qiskit_opt3")
    stage_started = time.perf_counter()
    optimized = transpile_channel_circuit(
        circuit, basis_gates=list(NAM), optimization_level=3,
        approximation_degree=1.0, seed_transpiler=SEED, num_processes=1,
        qubits_initially_zero=False)
    optimizer_s = time.perf_counter() - stage_started
    if optimized.num_qubits != circuit.num_qubits:
        raise ValueError("Unexpected width change without placement or routing")
    if optimized.layout is not None and optimized.layout.final_index_layout() != list(range(circuit.num_qubits)):
        raise ValueError("Output interface was permuted")
    del circuit
    progress("save_and_recount")
    stage_started = time.perf_counter()
    output = directory / "channel_nam_qiskit_opt3.qasm"
    qasm2.dump(optimized, output)
    gates, width, counts, nc = recount(output)
    if gates != sum(optimized.count_ops().get(g, 0) for g in NAM) or width != optimized.num_qubits:
        raise ValueError("Saved NAM output differs from the optimized circuit")
    serialize_recount_s = time.perf_counter() - stage_started
    progress("validation")
    stage_started = time.perf_counter()
    if int(job["n"]) <= 4 and width <= 18:
        optimized.metadata = {"structure": "basic", "subnormalization_factor": float(job["Lambda"])}
        validation = verify_channel(optimized, source["registers"], deserialize_kraus(source["input_kraus"]))
        if not validation["passed"]:
            raise ValueError(f"Optimized channel failed the Choi check: {validation}")
        choi_status = "passed"
    else:
        validation = {"status": "not_run_budget", "scope": "full Choi budget: <=4 system / <=18 total qubits"}
        choi_status = "not_run_budget"
    verification_s = time.perf_counter() - stage_started
    result = dict(
        case=case, method="Raw+Qiskit opt3", status="ok", host="local", basis="NAM",
        n=int(job["n"]), G=gates, W=width, Lambda=float(job["Lambda"]),
        Lambda_G=float(job["Lambda"]) * gates, CX=counts.get("cx", 0),
        non_clifford=nc, counts=counts, input_G=int(job["G"]),
        input_qasm_path=str(input_path), input_qasm_sha256=job["qasm_sha256"],
        adapted_input_qasm_path=str(adapted), adapted_input_qasm_sha256=digest(adapted),
        input_adapter="register alpha-renaming only; declaration order and bit indices preserved",
        register_rename=register_rename,
        raw_source_result_path=job["source_result_path"],
        raw_source_result_sha256=digest(job["source_result_path"]),
        canonical_input_sha256=job["canonical_input_sha256"],
        input_pairing="passed exact current Raw NAM QASM hash",
        qasm_path=str(output), qasm_sha256=digest(output),
        source_result_path=str(directory / "results.json"),
        optimization_level=3, downstream_optimizer="qiskit", seed_transpiler=SEED,
        approximation_degree=1.0, qubits_initially_zero=False,
        optimizer_s=optimizer_s, reported_compile_s=optimizer_s,
        time_scope="Qiskit opt3 only; excludes Raw compilation, input loading, I/O/recount and validation",
        input_load_s=input_load_s, serialize_recount_s=serialize_recount_s,
        verification_s=verification_s, validation=validation,
        full_choi_status=choi_status,
        full_choi_relative_error=validation.get("relative_frobenius_error", ""),
        full_choi_scope=validation.get("method", validation.get("scope")),
        G_over_Raw=gates / int(job["G"]),
        Lambda_G_over_Raw=gates / int(job["G"]),
        metric_evidence="exact input hash; saved output NAM recount, width and count_ops agree",
        worker_elapsed_s=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        qiskit_version=qiskit.__version__, python_version=platform.python_version(),
        runner_sha256=plan["runner_sha256"])
    dump(directory / "results.json", result)
    progress("done")


def write_csv(batch, cases):
    rows = []
    for case in CASES:
        row = dict(cases[case])
        for key, value in list(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, separators=(",", ":"))
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = ROOT / "Results/rq1_raw_qiskit_opt3_20261014.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--rss-gib", type=float, default=6)
    args = parser.parse_args()
    batch = args.out_dir.resolve()
    if args.worker:
        try:
            worker(batch, args.worker)
        except Exception:
            traceback.print_exc()
            raise
        return
    smoke = read(batch / "smoke.json")
    if not smoke["passed"] or smoke["sources"] != source_identity():
        raise ValueError("A passing smoke check matching current sources is required")
    input_rows = []
    for name in ("rq1_raw_optimizer_20261014.csv", "rq1_scaling_20261014.csv"):
        input_rows.extend(csv.DictReader((ROOT / "Results" / name).open()))
    inputs = {case: next(r for r in input_rows if r["case"] == case and r["method"] == "Raw" and r["status"] == "ok")
              for case in CASES}
    plan = dict(inputs=inputs, runner_sha256=digest(__file__), sources=source_identity(),
                timeout_s=args.timeout_s, rss_budget_bytes=int(args.rss_gib * 1024 ** 3),
                basis="NAM", optimization_level=3, seed_transpiler=SEED,
                approximation_degree=1.0, qubits_initially_zero=False)
    manifest = batch / "manifest.json"
    if manifest.exists():
        if not args.resume or read(manifest) != plan:
            raise ValueError("Existing batch requires --resume and unchanged inputs/settings")
    else:
        dump(manifest, plan)
    document = read(batch / "results.json") if args.resume and (batch / "results.json").exists() else dict(config=plan, cases={})
    for case in ("Petz_Ising3", "TFIM-4", "RepDQE-d3", "HRW-8", "HRW-12", "TFIM-8", "TFIM-12"):
        if document["cases"].get(case, {}).get("status") == "ok":
            saved = document["cases"][case]
            if digest(saved["qasm_path"]) != saved["qasm_sha256"]:
                raise ValueError("Saved optimized output changed")
            continue
        directory = batch / "cases" / case
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"Starting {case}", flush=True)
        with (directory / "run.log").open("w") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--out-dir", str(batch), "--worker", case],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            observed_peak, reason = 0, None
            observer = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    observed_peak = max(observed_peak, observer.memory_info().rss)
                except psutil.Error:
                    pass
                if observed_peak > plan["rss_budget_bytes"]:
                    reason = "memory_limit"
                elif time.perf_counter() - started > args.timeout_s:
                    reason = "timeout"
                if reason:
                    process.kill()
                    break
                time.sleep(.5)
            exit_code = process.wait()
        path = directory / "results.json"
        if exit_code == 0 and path.exists():
            row = read(path)
        else:
            row = dict(case=case, method="Raw+Qiskit opt3", status=reason or "worker_error",
                       source_result_path=str(path), G="", W="", Lambda_G="")
        row["supervisor"] = dict(exit_code=exit_code, termination_reason=reason,
                                 elapsed_s=time.perf_counter() - started,
                                 sampled_peak_rss_bytes=observed_peak,
                                 timeout_s=args.timeout_s, rss_budget_bytes=plan["rss_budget_bytes"])
        document["cases"][case] = row
        dump(batch / "results.json", document)
        print(f"{case}: {row['status']}, G={row.get('G')}, opt3_s={row.get('optimizer_s')}", flush=True)
    write_csv(batch, document["cases"])


if __name__ == "__main__":
    main()
