#!/usr/bin/env python3
"""Run frozen selected TFIM/HRW cases serially, retaining per-case failures.

    python run_selected_tfim_hrw.py --out-dir ...

Each case receives a fresh process and one complete measured compilation.
There is no wall-time cutoff. This is a full-suite run, not a repeated timing
microbenchmark. Imports, input construction, verification and output I/O are
excluded from compilation timing. Run the coordinator from its frozen snapshot.
"""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("src", "scripts", "Baseline_scripts")]
CONFIG = Path("paper_experiments/configs/full_channel_suite_20261011.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selected(config):
    cases = [c for c in config["cases"] if c["family"] in ("tfim_first_order", "hrw")]
    cases.sort(key=lambda c: (c["system_qubits"], c["family"] != "hrw"))
    cases += [c for c in config["cases"] if c["id"] in ("TFIM2-HO", "TFIM3_HO")]
    return cases


def sparse_gram_check(before, after):
    """Exact Pauli-basis channel check without exponential Hilbert matrices."""
    import numpy as np
    from scipy.sparse import csr_matrix
    labels = {label for rows in (before, after) for row in rows for label, *_ in row}
    index = {label: i for i, label in enumerate(sorted(labels))}
    def matrix(rows):
        rr, cc, vv = [], [], []
        for i, row in enumerate(rows):
            for label, magnitude, real, imag in row:
                rr.append(i)
                cc.append(index[label])
                vv.append(magnitude * complex(real, imag))
        return csr_matrix((vv, (rr, cc)), shape=(len(rows), len(index)), dtype=complex)
    a, b = matrix(before), matrix(after)
    ga, gb = a.conj().T @ a, b.conj().T @ b
    difference = ga - gb
    error = float(np.linalg.norm(difference.data) / max(np.linalg.norm(ga.data), 1e-30))
    return {"method": "sparse Pauli coefficient Gram", "relative_error": error,
            "passed": error < 1e-10, "tolerance": 1e-10, "pauli_basis_size": len(index)}


def worker(spec, out):
    directory = out / "cases" / spec["id"]
    directory.mkdir(parents=True, exist_ok=True)
    def progress(stage, **extra):
        dump(directory / "progress.json", {"case": spec["id"], "stage": stage,
             "updated_at_utc": now(), "pid": os.getpid(), **extra})
        print(json.dumps({"case": spec["id"], "stage": stage, **extra}), flush=True)
    progress("importing")
    from qiskit import qasm2
    from channel_IR import channel as ChannelIR
    from channel_rewrite_cost import h_length_components
    from experiment_protocol import REWRITE_DEFAULTS, settings, source_identity
    from rq1_pipeline import serialize_kraus, deserialize_kraus, verify_channel
    from profile_ours_pipeline import compile_one
    from selected_channel_inputs import build_selected_case
    from baseline_common import count_qasm_file
    progress("input_construction")
    started = time.perf_counter()
    raw, model = build_selected_case(spec)
    input_time = time.perf_counter() - started
    input_rows = serialize_kraus(raw)
    input_path = directory / "input.json"
    dump(input_path, {"case": spec["id"], "model": model, "input_kraus": input_rows,
                      "input_construction_s": input_time})
    options = {"rewrite_method": REWRITE_DEFAULTS.method, "strategy": REWRITE_DEFAULTS.strategy,
               "beam_width": REWRITE_DEFAULTS.beam_width, "max_steps": REWRITE_DEFAULTS.max_steps,
               "tol": REWRITE_DEFAULTS.tol, **spec.get("rewrite_options", {})}
    identities = source_identity()
    progress("full_pipeline", rewrite_options=options)
    lowered, result = compile_one(raw, rewrite_options=options)
    assert source_identity() == identities
    dump(directory / "compiled.json", result)
    progress("validation", G=result["G"], compile_time_s=result["compile_time_s"])
    rewritten = result["assembly"]["compiled_kraus"]
    verification = sparse_gram_check(input_rows, rewritten)
    assert verification["passed"], verification
    final, _ = ChannelIR(deserialize_kraus(rewritten))._coeff_matrix()
    cost = h_length_components(final, structure="tree", fstar=True)
    del final
    assert abs(cost["Lambda"] - result["Lambda"]) < 1e-10 * max(1, result["Lambda"])
    choi = {"status": "skipped", "reason": "budget: <=4 system and <=18 total qubits"}
    if spec["system_qubits"] <= 4 and result["W"] <= 18:
        choi = verify_channel(lowered, result["registers"], raw)
        assert choi["passed"], choi
    progress("export", G=result["G"])
    qasm = directory / "channel_nam.qasm"
    qasm2.dump(lowered, qasm)
    recount = count_qasm_file(qasm, gate_set="nam")
    assert recount["metric_total"] == result["G"]
    termination = result["assembly"]["rewrite"]["termination"]
    truncated = termination.get("stop_reason") == "max_steps_reached"
    result.update(case=spec["id"], status="rewrite_truncated" if truncated else "ok",
                  model=model, input_sha256=sha256(input_path.read_bytes()).hexdigest(),
                  input_construction_s=input_time, sources=identities,
                  rewrite_verification=verification, full_choi=choi, static_fstar_cost_audit=cost,
                  qasm_sha256=sha256(qasm.read_bytes()).hexdigest(),
                  non_clifford=recount["clifford"]["non_clifford"],
                  effective_rewrite_options=options, protocol=settings("RQ1"),
                  timing_scope="one fresh-process compilation; imports, input construction, verification and I/O excluded")
    result["protocol"]["rewrite"] = result["assembly"]["rewrite"]
    dump(directory / "results.json", result)
    progress(result["status"], G=result["G"], compile_time_s=result["compile_time_s"],
             rewrite_time_s=result["exclusive_stage_times_s"]["rewrite"])


def prepare(out, config):
    if (out / "manifest.json").exists():
        raise ValueError("Choose a fresh output directory")
    snapshot = out / "source_snapshot"
    hashes = {}
    for folder in ("src", "scripts", "Baseline_scripts"):
        for source in sorted((ROOT / folder).rglob("*.py")):
            relative = source.relative_to(ROOT)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[str(relative)] = sha256(destination.read_bytes()).hexdigest()
    config_path = snapshot / CONFIG
    dump(config_path, config)
    hashes[str(CONFIG)] = sha256(config_path.read_bytes()).hexdigest()
    for spec in selected(config):
        if "fixture" in spec:
            path = Path(spec["fixture"])
            destination = snapshot / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, destination)
            hashes[str(path)] = sha256(destination.read_bytes()).hexdigest()
    manifest = {"created_at_utc": now(), "status": "prepared", "pipeline": "Ours",
                "cases": selected(config), "host": "local", "jobs": 1, "numerical_threads": 1,
                "repetitions": 1, "warmups": 0, "wall_timeout_s": None,
                "source_snapshot": str(snapshot), "snapshot_sha256": hashes,
                "scope": "8 first-order TFIM, 2 higher-order TFIM, 11 HRW",
                "concurrency_note": "Separate subagent rewrite probes may overlap; these are full compilation measurements, not isolated microbenchmark medians"}
    dump(out / "manifest.json", manifest)
    dump(out / "status.json", {"status": "queued", "updated_at_utc": now(),
         "jobs": [{"case": s["id"], "status": "queued"} for s in selected(config)]})


def coordinator(out):
    manifest = json.loads((out / "manifest.json").read_text())
    snapshot = Path(manifest["source_snapshot"])
    for name, expected in manifest["snapshot_sha256"].items():
        assert sha256((snapshot / name).read_bytes()).hexdigest() == expected, name
    records = [{"case": s["id"], "status": "queued"} for s in manifest["cases"]]
    started = now()
    def save(state="running"):
        dump(out / "status.json", {"status": state, "pid": os.getpid(),
             "started_at_utc": started, "updated_at_utc": now(), "jobs": records})
        fields = ("case", "status", "G", "W", "Lambda", "compile_time_s",
                  "rewrite_time_s", "termination", "results")
        with (out / "results.csv.writing").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        (out / "results.csv.writing").replace(out / "results.csv")
    save()
    for spec, record in zip(manifest["cases"], records):
        directory = out / "cases" / spec["id"]
        directory.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-B", str(snapshot / "scripts/eval_redesign/run_selected_tfim_hrw.py"),
               "--out-dir", str(out), "--worker", spec["id"]]
        with (directory / "run.log").open("w") as log:
            child = subprocess.Popen(cmd, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            record.update(status="running", pid=child.pid, started_at_utc=now(), command=cmd)
            save()
            code = child.wait()
        record.update(exit_code=code, finished_at_utc=now())
        path = directory / "results.json"
        if code == 0 and path.exists():
            data = json.loads(path.read_text())
            record.update(status=data["status"], G=data["G"], W=data["W"], Lambda=data["Lambda"],
                          compile_time_s=data["compile_time_s"],
                          rewrite_time_s=data["exclusive_stage_times_s"]["rewrite"],
                          termination=data["assembly"]["rewrite"]["termination"]["stop_reason"],
                          results=str(path))
        else:
            record.update(status="failed", log=str(directory / "run.log"))
        save()
    save("complete" if all(r["status"] == "ok" for r in records) else "complete_with_issues")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".runtime"
    cache.mkdir(exist_ok=True)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "QISKIT_NUM_PROCS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(cache / "matplotlib"),
                      NUMBA_CACHE_DIR=str(cache / "numba"))
    if args.prepare_only:
        prepare(out, json.loads((ROOT / CONFIG).read_text()))
    elif args.worker:
        config = json.loads((ROOT / CONFIG).read_text())
        spec = next(s for s in selected(config) if s["id"] == args.worker)
        try:
            worker(spec, out)
        except Exception:
            dump(out / "cases" / args.worker / "failure.json",
                 {"case": args.worker, "time_utc": now(), "traceback": traceback.format_exc()})
            raise
    else:
        coordinator(out)


if __name__ == "__main__":
    main()
