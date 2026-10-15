"""Run the three authorized fresh HRW Raw + IBM opt3 measurements serially."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import run as batch
from remote_ours_parallel import rss

BASE = Path(__file__).resolve().parent
CASES = ('HRW-8', 'HRW-12', 'HRW-28')
POLICY = dict(max_workers=1, worker_rss_gib=32, aggregate_rss_gib=32,
              minimum_available_gib=32, stage_timeout_s=43200, worker_timeout_s=86400)


def preflight():
    assert not (BASE / 'remote_hrw_raw_full_status.json').exists()
    manifest = batch.read(BASE / 'manifest.json')
    plan = batch.read(BASE / 'hrw_raw_full_plan.json')
    assert plan['cases'] == list(CASES) and plan['policy'] == POLICY
    assert batch.digest(BASE / 'manifest.json') == plan['frozen_manifest_sha256']
    assert all(batch.digest(BASE / name) == value for name, value in plan['workers'].items())
    assert batch.digest(BASE / 'run.py') == manifest['runner_sha256']
    assert all(batch.digest(batch.SNAP / p) == h for p, h in manifest['sources'].items())
    assert batch.read(BASE / 'preflight_remote.json')['passed']
    assert batch.read(BASE / 'smoke.json')['passed']
    assert batch.read(BASE / 'remote_hrw_series_full_status.json')['status'] == 'complete'
    for case in CASES:
        assert batch.digest(BASE / 'inputs' / f'{case}.json') == manifest['cases'][case]['input_sha256']
        assert batch.read(BASE / 'inputs' / f'{case}.json')['Raw']['status'] == 'ok'
        directory = BASE / 'cases' / case / 'raw_full_remote'
        assert not directory.exists() or not any(directory.iterdir()), case


def main():
    preflight()
    if '--check' in sys.argv:
        print('Preflight passed:', CASES, POLICY, flush=True)
        return
    lock = os.open(BASE / 'remote_hrw_raw_full.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock, str(os.getpid()).encode())
    os.close(lock)
    jobs = [dict(case=c, kind='raw_full_remote', host='remote', status='queued',
                 purpose='fresh Raw compilation followed by Qiskit opt3',
                 worker_rss_gib=POLICY['worker_rss_gib']) for c in CASES]

    def checkpoint(status='running'):
        batch.dump(BASE / 'remote_hrw_raw_full_status.json', dict(
            status=status, host='remote', pid=os.getpid(), updated=time.time(),
            policy=POLICY, jobs=jobs,
            time_scope='Fresh canonicalization, Raw assembly, IBM opt0 lowering and opt3; serial single observations'))

    checkpoint()
    for job in jobs:
        while psutil.virtual_memory().available < (POLICY['worker_rss_gib'] + POLICY['minimum_available_gib']) * 2**30:
            checkpoint('waiting_for_memory')
            time.sleep(10)
        directory = BASE / 'cases' / job['case'] / job['kind']
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'run.log').open('x') as log:
            process = subprocess.Popen([
                sys.executable, '-B', str(BASE / 'hrw_raw_full_worker.py'),
                '--worker', job['case'], '--host', 'remote', '--output-kind', job['kind']],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        now = time.time()
        job.update(status='running', worker_pid=process.pid, start_wall=now,
                   stage_start_wall=now, stage='starting', elapsed_s=0, peak_rss_bytes=0)
        checkpoint()
        stop = None
        while True:
            progress_path = directory / 'progress.json'
            if progress_path.exists():
                state = batch.read(progress_path)
                if state['stage'] != job['stage']:
                    job.update(stage=state['stage'], stage_start_wall=state['updated'])
            usage = rss(process.pid)
            job.update(elapsed_s=time.time()-job['start_wall'],
                       peak_rss_bytes=max(job['peak_rss_bytes'], usage))
            if process.poll() is None:
                if usage > POLICY['worker_rss_gib'] * 2**30:
                    stop = 'memory_limit'
                elif psutil.virtual_memory().available < POLICY['minimum_available_gib'] * 2**30:
                    stop = 'available_memory_guard'
                elif time.time()-job['stage_start_wall'] > POLICY['stage_timeout_s']:
                    stop = 'timeout'
                elif job['elapsed_s'] > POLICY['worker_timeout_s']:
                    stop = 'worker_timeout'
                if stop:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            if process.poll() is not None:
                code = process.wait()
                if progress_path.exists():
                    job['stage'] = batch.read(progress_path)['stage']
                result_path = directory / 'results.json'
                success = (not stop and code == 0 and result_path.exists()
                           and batch.read(result_path)['status'] == 'ok' and job['stage'] == 'done')
                job.update(status=stop or ('ok' if success else 'failed'), exit_code=code, resource_stop=stop)
                if success:
                    result = batch.read(result_path)
                    job.update(raw_compile_s=result['raw_compile_time_s'], optimizer_s=result['optimizer_s'],
                               full_compile_s=result['compile_time_s'])
                batch.dump(directory / 'resources.json', job)
                checkpoint()
                break
            checkpoint()
            time.sleep(1)
    checkpoint('complete' if all(j['status'] == 'ok' for j in jobs) else 'complete_with_issues')


if __name__ == '__main__':
    main()
