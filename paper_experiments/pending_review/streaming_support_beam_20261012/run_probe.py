"""Resource-guarded launcher for the isolated experiment (one worker at a time)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# The five paper fixtures use the maintained entrypoint, not this legacy CLI.
if __name__ == "__main__":
    import runpy as _retirement_runpy
    _retirement_runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "_fixture_entrypoints.py"),
        init_globals={"default_case": "Thermal2_HO"}, run_name="__main__")

import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--array", default="A_initial")
parser.add_argument("--case", default="Thermal2_HO")
parser.add_argument("--out", required=True)
parser.add_argument("--max-steps", type=int, default=583)
parser.add_argument("--timeout", type=float, default=1800)
parser.add_argument("--strict", action="store_true")
parser.add_argument("--rank", default="original")
parser.add_argument("--script", default="fast_probe.py")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if (out / "results.json").exists() or (out / "resources.json").exists():
    raise SystemExit("Refusing to overwrite previous result")
cmd = [sys.executable, "-B", str(Path(__file__).with_name(args.script)),
       "--input", args.input, "--array", args.array, "--case", args.case, "--out", str(out),
       "--max-steps", str(args.max_steps), "--timeout", str(args.timeout), "--rank", args.rank]
if args.strict:
    cmd.append("--strict")
env = os.environ.copy()
env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
with (out / "run.log").open("w") as log:
    worker = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    start = time.monotonic()
    peak = 0
    reason = None
    print(json.dumps({"worker_pid": worker.pid, "out": str(out)}), flush=True)
    try:
        while worker.poll() is None:
            try:
                rss = psutil.Process(worker.pid).memory_info().rss
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                rss = 0
            elapsed = time.monotonic() - start
            if rss > 6 * 1024**3 or elapsed > args.timeout + 40:
                reason = "memory_limit" if rss > 6 * 1024**3 else "wall_timeout"
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
                break
            time.sleep(0.2)
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=10)
    result = {"returncode": worker.wait(), "peak_sampled_rss_bytes": peak,
              "wall_time_s": time.monotonic() - start, "termination_reason": reason,
              "memory_limit_bytes": 6 * 1024**3, "worker_command": cmd}
    (out / "resources.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
