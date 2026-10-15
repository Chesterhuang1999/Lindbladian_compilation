"""Take over scheduling without restarting workers; prioritize authorized HRW-512 opt3.

Retain a stopped original parent until its child exits so Linux exposes the
actual wait status in /proc. No result is accepted from disappearance alone.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import psutil
import remote_hrw_5000 as original
import run as batch

BASE = Path(__file__).resolve().parent
STATUS = original.STATUS
GIB = 2**30


def exited_code(pid, created):
    process = psutil.Process(pid)
    assert abs(process.create_time() - created) < .01, "PID identity mismatch"
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] != "Z":
        return None
    return os.waitstatus_to_exitcode(int(fields[49]))


def check():
    plan = batch.read(BASE / "hrw_5000_plan.json")
    manifest = batch.read(BASE / "manifest.json")
    assert batch.digest(original.__file__) == plan["runner_sha256"]
    assert batch.digest(BASE / "manifest.json") == plan["frozen_manifest_sha256"]
    assert batch.digest(BASE / "hrw_series_full_worker.py") == plan["ours_worker_sha256"]
    assert batch.digest(BASE / "run.py") == manifest["runner_sha256"]
    assert all(batch.digest(batch.SNAP / p) == h for p, h in manifest["sources"].items())
    state = batch.read(STATUS)
    assert state["status"] == "running" and state["pid"] == 1235651
    parent = psutil.Process(state["pid"])
    assert abs(parent.create_time() - 1792040044.76) < .1
    assert parent.cmdline()[-1].endswith("/remote_hrw_5000.py")
    selected = next(j for j in state["jobs"] if (j["case"], j["mode"]) == ("HRW-512", "opt3"))
    assert selected["status"] == "queued"
    directory = BASE / "cases" / selected["case"] / selected["kind"]
    assert not directory.exists() or not any(directory.iterdir())
    raw_path = BASE / "cases/HRW-512/raw/results.json"
    raw = batch.read(raw_path)
    assert raw["status"] == "ok" and raw["host"] == "remote"
    assert raw["basis"] == "IBM" and raw["optimization_level"] == 0
    assert raw["input_sha256"] == manifest["cases"]["HRW-512"]["input_sha256"]
    assert batch.digest(BASE / "inputs/HRW-512.json") == raw["input_sha256"]
    assert batch.digest(raw["qasm_path"]) == raw["qasm_sha256"] == plan["saved_raw_hashes"]["HRW-512"]
    assert psutil.virtual_memory().available >= 160 * GIB
    return state, parent


def main(check_only=False):
    before, parent = check()
    if check_only:
        print("Admission preflight passed", flush=True)
        return
    lock = (BASE / "remote_hrw_5000_admission.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (BASE / "hrw_5000_admission_handoff.json").exists()
    os.kill(parent.pid, signal.SIGSTOP)
    try:
        for _ in range(100):
            if parent.status() == psutil.STATUS_STOPPED:
                break
            time.sleep(.01)
        assert parent.status() == psutil.STATUS_STOPPED
        state = batch.read(STATUS)
        assert state["pid"] == parent.pid
        jobs = state["jobs"]
        adopted = {}
        for j in jobs:
            if j["status"] != "running":
                continue
            p = psutil.Process(j["worker_pid"])
            assert p.ppid() == parent.pid and abs(p.create_time() - j["start_wall"]) < 3
            assert "--worker" in p.cmdline() and str(BASE / "remote_hrw_5000.py") in p.cmdline()
            adopted[j["worker_pid"]] = p.create_time()
        assert {p.pid for p in parent.children()} == set(adopted), "Untracked child; abort handoff"
        assert next(j for j in jobs if (j["case"], j["mode"]) == ("HRW-512", "opt3"))["status"] == "queued"
        batch.dump(BASE / "remote_hrw_5000_status_before_admission.json", state)
        handoff = dict(old_pid=parent.pid, old_created=parent.create_time(),
            new_pid=os.getpid(), new_created=psutil.Process().create_time(), adopted=adopted,
            worker_limits_unchanged=True, worker_start_times_preserved=True,
            priority=["HRW-512 opt3", "HRW-1024 raw"],
            one_time_admission_exception="HRW-512 opt3 ignores unused reservation of running HRW-1024 Ours only; requires 160 GiB available; all RSS, aggregate and timeout guards retained",
            scheduler_sha256=batch.digest(__file__), updated=time.time())
        batch.dump(BASE / "hrw_5000_admission_handoff.json", handoff)
    except BaseException:
        os.kill(parent.pid, signal.SIGCONT)
        raise
    handles = {}
    policy = dict(state["policy"], admission_exception=handoff["one_time_admission_exception"])

    def checkpoint(status="running"):
        batch.dump(STATUS, dict(status=status, host="remote", pid=os.getpid(),
            created=handoff["new_created"], updated=time.time(), policy=policy, jobs=jobs,
            previous_coordinator_pid=handoff["old_pid"], handoff_record="hrw_5000_admission_handoff.json",
            hrw1440=state["hrw1440"]))

    def poll(j):
        pid = j["worker_pid"]
        return exited_code(pid, adopted[pid]) if pid in adopted else handles[pid].poll()

    def stop(j):
        os.killpg(j["worker_pid"], signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and poll(j) is None:
            time.sleep(.05)
        if poll(j) is None:
            os.killpg(j["worker_pid"], signal.SIGKILL)
            while poll(j) is None:
                time.sleep(.05)

    checkpoint()
    try:
        while any(j["status"] in ("queued", "running") for j in jobs):
            running = [j for j in jobs if j["status"] == "running"]
            usage = {j["worker_pid"]: original.rss(j["worker_pid"]) for j in running}
            pressure = sum(usage.values()) > 256 * GIB or psutil.virtual_memory().available < 32 * GIB
            victim = max(usage, key=usage.get) if pressure and usage else None
            for j in running:
                directory = BASE / "cases" / j["case"] / j["kind"]
                mark = directory / "progress.json"
                if mark.exists():
                    progress = batch.read(mark)
                    if progress["stage"] != j["stage"]:
                        j.update(stage=progress["stage"], stage_start_wall=progress["updated"])
                j.update(elapsed_s=time.time() - j["start_wall"],
                         peak_rss_bytes=max(j["peak_rss_bytes"], usage[j["worker_pid"]]))
                reason = None
                if poll(j) is None:
                    if usage[j["worker_pid"]] > j["worker_rss_gib"] * GIB:
                        reason = "memory_limit"
                    if victim == j["worker_pid"]:
                        reason = "aggregate_memory_guard"
                    if j["elapsed_s"] >= 5000:
                        reason = "timeout"
                    if reason:
                        stop(j)
                code = poll(j)
                if code is not None:
                    if mark.exists():
                        j["stage"] = batch.read(mark)["stage"]
                    path = directory / "results.json"
                    result = batch.read(path) if path.exists() else {}
                    success = code == 0 and reason is None and result.get("status") == "ok" and j["stage"] == "done"
                    j.update(status=reason or ("ok" if success else "failed"), exit_code=code, resource_stop=reason,
                             exit_evidence="Linux /proc zombie wait status" if j["worker_pid"] in adopted else "Popen.wait")
                    if success:
                        j.update(full_compile_s=result["compile_time_s"], optimizer_s=result.get("optimizer_s", 0),
                                 fresh_rewrite_s=result.get("rewrite_s", 0) if j["mode"] == "ours" else 0)
                    batch.dump(directory / "resources.json", j)
            if parent is not None and not any(j["status"] == "running" and j["worker_pid"] in adopted for j in jobs):
                # The scientific workers have exited and their exact exit statuses are saved.
                os.kill(parent.pid, signal.SIGKILL)
                handoff["old_coordinator_retired_wall"] = time.time()
                batch.dump(BASE / "hrw_5000_admission_handoff.json", handoff)
                parent = None
            running = [j for j in jobs if j["status"] == "running"]
            queued = sorted((j for j in jobs if j["status"] == "queued"),
                            key=lambda j: (j["case"], j["mode"]) != ("HRW-512", "opt3"))
            for j in queued:
                if j['mode'] == 'raw' and any(x['case'] == 'HRW-512' and x['mode'] == 'opt3'
                        and x['status'] in ('queued', 'running') for x in jobs):
                    continue
                if len(running) >= 3:
                    break
                if sum(x["worker_rss_gib"] for x in running) + j["worker_rss_gib"] > 256:
                    continue
                exception = (j["case"], j["mode"]) == ("HRW-512", "opt3")
                unused = sum(max(0, x["worker_rss_gib"] * GIB - original.rss(x["worker_pid"]))
                             for x in running if not (exception and (x["case"], x["mode"]) == ("HRW-1024", "ours")))
                if psutil.virtual_memory().available < unused + (j["worker_rss_gib"] + 32) * GIB:
                    continue
                directory = BASE / "cases" / j["case"] / j["kind"]
                assert not directory.exists() or not any(directory.iterdir())
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / "run.log").open("x") as log:
                    p = subprocess.Popen([sys.executable, "-B", str(BASE / "remote_hrw_5000.py"),
                        "--worker", j["case"], "--mode", j["mode"]], stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                handles[p.pid] = p
                now = time.time()
                j.update(status="running", worker_pid=p.pid, worker_created=psutil.Process(p.pid).create_time(),
                         start_wall=now, stage_start_wall=now, stage="starting", elapsed_s=0, peak_rss_bytes=0,
                         admission_exception_used=exception)
                running.append(j)
                checkpoint()
            checkpoint()
            time.sleep(.5)
        checkpoint("complete" if all(j["status"] == "ok" for j in jobs) else "complete_with_issues")
    except BaseException:
        batch.dump(BASE / "remote_hrw_5000_admission_failure.json", dict(traceback=traceback.format_exc(), updated=time.time()))
        for j in jobs:
            if j["status"] == "running" and poll(j) is None:
                stop(j)
                j.update(status="supervisor_failure", exit_code=poll(j), resource_stop="supervisor_failure")
                batch.dump(BASE / "cases" / j["case"] / j["kind"] / "resources.json", j)
        checkpoint("supervisor_failure")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    main(args.check)
