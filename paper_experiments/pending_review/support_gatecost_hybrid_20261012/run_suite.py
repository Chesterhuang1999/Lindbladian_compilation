"""Serial resource-guarded coordinator for isolated hybrid rewrite probes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
CASES = ["Decay-HO", "HRW-12", "Color-n4q4-linear", "Petz_Ising3", "CPTP2_M4_seed4"]


def dump(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def main():
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLCONFIGDIR=str(HERE / ".mplcache"))
    status = {"coordinator_pid": os.getpid(), "cases": {}, "status": "running"}
    for name in CASES:
        dest = HERE / "cases" / name
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "resources.json").exists():
            status["cases"][name] = json.loads((dest / "resources.json").read_text())
            continue
        start = time.monotonic()
        with (dest / "run.log").open("w") as log:
            child = subprocess.Popen([sys.executable,"-B",str(HERE / "probe.py"),"--case",name],env=env,stdout=log,stderr=subprocess.STDOUT)
            status["current_case"] = name
            status["worker_pid"] = child.pid
            dump(HERE / "status.json", status)
            peak = 0
            reason = None
            while child.poll() is None:
                try:
                    proc = psutil.Process(child.pid)
                    rss = sum(p.memory_info().rss for p in [proc] + proc.children(recursive=True))
                    peak = max(peak,rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > 6*1024**3 or time.monotonic()-start > 1500:
                    reason = "memory_limit" if peak > 6*1024**3 else "wall_timeout"
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    break
                time.sleep(.3)
        record = {"exit_code": child.wait(), "termination_reason": reason,"peak_rss_bytes":peak,"wall_s":time.monotonic()-start}
        dump(dest / "resources.json",record)
        status["cases"][name]=record
        dump(HERE / "status.json",status)
        print(json.dumps({"case":name,**record}),flush=True)
    status["status"]="finished"
    status.pop("worker_pid",None)
    status.pop("current_case",None)
    dump(HERE / "status.json",status)


if __name__ == "__main__": main()
