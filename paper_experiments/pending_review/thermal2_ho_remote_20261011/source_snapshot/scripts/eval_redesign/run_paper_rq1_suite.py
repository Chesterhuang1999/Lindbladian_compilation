#!/usr/bin/env python3
"""Run all paper RQ1 inputs through Raw / Qrisp+F* / Ours in isolated jobs.

Preserve inputs, commands, per-job logs/status, QASM and JSON metrics. Limits
are operational limits for this new run, not historical paper timeouts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / f) for f in ("src", "scripts", "Baseline_scripts")]


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    from paper_rq1_inputs import case_names
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=case_names())
    parser.add_argument("--pipelines", nargs="+", choices=("raw", "qrisp", "ours"), default=["raw", "qrisp", "ours"])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--rss-limit-gib", type=float, default=4)
    parser.add_argument("--matrix-order-options", type=json.loads, default={}, metavar="JSON")
    args = parser.parse_args()
    from matrix_order_pipeline import resolve_options, options_dict
    try:
        args.matrix_order_options = options_dict(resolve_options(args.matrix_order_options))
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.jobs < 1 or args.timeout <= 0 or args.rss_limit_gib <= 0:
        parser.error("jobs, timeout and memory budget must be positive")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        parser.error("Choose a fresh --out-dir; never overwrite a prior suite")
    with tempfile.TemporaryDirectory(prefix=".suite-runtime-", dir=out) as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(cache) / "numba"))
        execute(args, out)


def execute(args, out):
    import psutil
    from experiment_protocol import settings, source_identity
    from paper_rq1_inputs import case_names, build_case
    from rq1_pipeline import serialize_kraus
    names = args.only or case_names()
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    tasks = []
    for name in names:
        raw, model = build_case(name)
        dump(inputs / f"{name}.json", {"input_kraus": serialize_kraus(raw), "model": model})
        print(json.dumps({"input": name, "qubits": raw[0].size, "kraus": len(raw),
                          "terms": sum(len(k.instances) for k in raw)}), flush=True)
        tasks.extend((name, pipeline) for pipeline in args.pipelines)
    sources = dict(source_identity())
    snapshot = out / "source_snapshot"
    for name in sources:
        target = snapshot / "src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / name, target)
    for folder in (ROOT / "scripts", ROOT / "Baseline_scripts"):
        for source in folder.glob("*.py"):
            target = snapshot / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for source in Path(__file__).parent.glob("*.py"):
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Include transitive local backend helpers, not just the protocol's key files.
    for source in (ROOT / "src").glob("*.py"):
        shutil.copy2(source, snapshot / "src" / source.name)
    dump(out / "manifest.json", {"cases": names, "pipelines": args.pipelines,
         "jobs": args.jobs, "per_job_timeout_s": args.timeout, "per_job_rss_limit_gib": args.rss_limit_gib,
         "limits_scope": "new run operational budget; not historical paper settings",
         "settings": {**settings("RQ1"), "matrix_order_options": args.matrix_order_options}, "sources": sources,
         "input_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in inputs.glob("*.json")},
         "timing_note": "jobs run concurrently; compilation times are not isolated timing measurements",
         "random_scope": "preserve paper RQ1 actual2q/3q, not D07 corrected4q",
         "decay_scope": "same finite channel materialized as Pauli Kraus sums; legacy concatenated-gadget counts not directly comparable"})

    def worker(task):
        name, pipeline = task
        directory = out / "runs" / name / pipeline
        directory.mkdir(parents=True)
        cmd = [sys.executable, "-B", str(Path(__file__).with_name("run_rq1_pipeline.py")),
               "--input-json", str(inputs / f"{name}.json"), "--pipeline", pipeline,
               "--out-dir", str(out / "runs"), "--verify-auto",
               "--matrix-order-options", json.dumps(args.matrix_order_options)]
        started = time.monotonic()
        record = {"example": name, "pipeline": pipeline, "command": cmd, "status": "running"}
        dump(directory / "job.json", record)
        peak = 0
        reason = None
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", QISKIT_NUM_PROCS="1")
        with (directory / "run.log").open("w") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = process.pid
            dump(directory / "job.json", record)
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    # QISKIT_NUM_PROCS=1; inspect only this owned process. Listing
                    # every process/descendant is prohibited by the local sandbox.
                    rss = parent.memory_info().rss
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                except (psutil.Error, OSError) as exc:
                    reason = "monitor_unavailable"
                    record["monitor_error"] = str(exc)
                if peak > args.rss_limit_gib * 1024**3:
                    reason = "memory_limit"
                elif time.monotonic() - started > args.timeout:
                    reason = "timeout"
                if reason:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(1)
            process.wait()
        record.update(status=reason or ("ok" if process.returncode == 0 else "failed"),
                      exit_code=process.returncode, elapsed_s=time.monotonic()-started, peak_rss_bytes=peak)
        result = directory / "results.json"
        if result.exists():
            data = json.loads(result.read_text())
            record.update(G=data["metrics"]["G"], W=data["metrics"]["W"], Lambda=data["Lambda"],
                          Lambda_G=data["Lambda_G"], compile_time_s=data["compile_time_s"],
                          verification=data["verification"], result=str(result))
        elif record["status"] == "ok":
            record["status"] = "missing_result"
        dump(directory / "job.json", record)
        print(json.dumps({k: record.get(k) for k in ("example", "pipeline", "status", "G", "W", "elapsed_s")}), flush=True)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(worker, task) for task in tasks]):
            records.append(future.result())
            records.sort(key=lambda r: (names.index(r["example"]), args.pipelines.index(r["pipeline"])))
            dump(out / "results.json", records)
    fields = ("example", "pipeline", "status", "G", "W", "Lambda", "Lambda_G", "compile_time_s", "elapsed_s", "peak_rss_bytes", "result")
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
