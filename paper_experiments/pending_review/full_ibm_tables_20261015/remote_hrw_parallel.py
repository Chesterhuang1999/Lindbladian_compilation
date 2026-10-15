"""Run the three user-requested HRW migrations independently of the TFIM queue."""
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
CASES = ('HRW-512', 'HRW-1024', 'HRW-1440')
POLICY = dict(max_workers=3, worker_rss_gib=32, aggregate_rss_gib=96,
              minimum_available_gib=32, launch_available_gib=128,
              stage_timeout_s=43200, worker_timeout_s=86400)


def preflight():
    assert not (BASE/'remote_hrw_status.json').exists(), 'Refusing duplicate HRW scheduler'
    manifest = batch.read(BASE/'manifest.json')
    assert batch.digest(BASE/'run.py') == manifest['runner_sha256']
    assert all(batch.digest(batch.SNAP/p) == h for p,h in manifest['sources'].items())
    assert batch.read(BASE/'preflight_remote.json')['passed']
    assert psutil.virtual_memory().available > POLICY['launch_available_gib']*2**30
    for case in CASES:
        directory = BASE/'cases'/case/'ours'
        assert not directory.exists() or not any(directory.iterdir()), str(directory)


def main():
    preflight()
    if '--check' in sys.argv:
        print('HRW migration preflight passed', POLICY)
        return
    path = BASE/'remote_hrw_status.json'
    jobs = [dict(case=c, kind='ours', host='remote', status='queued') for c in CASES]
    handles = {}

    def checkpoint(status='running'):
        batch.dump(path, dict(status=status, host='remote', pid=os.getpid(),
                   updated=time.time(), policy=POLICY, jobs=jobs,
                   scope='Three HRW Ours migrations; independent of TFIM and Raw queues'))

    checkpoint()
    for j in jobs:
        directory = BASE/'cases'/j['case']/'ours'
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'run.log').open('w') as log:
            p = subprocess.Popen([sys.executable, '-B', str(BASE/'run.py'),
                '--worker', j['case'], '--kind', 'ours', '--host', 'remote'],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        handles[j['case']] = p
        now = time.time()
        j.update(status='running', worker_pid=p.pid, start_wall=now,
                 stage_start_wall=now, stage='starting', peak_rss_bytes=0)
        checkpoint()
    while any(j['status']=='running' for j in jobs):
        running = [j for j in jobs if j['status']=='running']
        usage = {j['case']:rss(j['worker_pid']) for j in running}
        pressure = (sum(usage.values()) > POLICY['aggregate_rss_gib']*2**30 or
                    psutil.virtual_memory().available < POLICY['minimum_available_gib']*2**30)
        victim = max(usage, key=usage.get) if pressure and usage else None
        for j in running:
            p = handles[j['case']]
            directory = BASE/'cases'/j['case']/'ours'
            progress = directory/'progress.json'
            if progress.exists():
                state = batch.read(progress)
                if state['stage'] != j['stage']:
                    j.update(stage=state['stage'], stage_start_wall=state['updated'])
            j.update(elapsed_s=time.time()-j['start_wall'],
                     peak_rss_bytes=max(j['peak_rss_bytes'], usage[j['case']]))
            stop = None
            if p.poll() is None:
                if usage[j['case']] > POLICY['worker_rss_gib']*2**30: stop='memory_limit'
                if victim == j['case']: stop='aggregate_memory_guard'
                if time.time()-j['stage_start_wall'] > POLICY['stage_timeout_s']: stop='timeout'
                if j['elapsed_s'] > POLICY['worker_timeout_s']: stop='worker_timeout'
                if stop:
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
            if p.poll() is not None:
                code = p.wait()
                # Re-read the final marker after exit to avoid a completion race.
                if progress.exists(): j['stage'] = batch.read(progress)['stage']
                result = directory/'results.json'
                success = (code==0 and result.exists() and
                           batch.read(result)['status']=='ok' and j['stage']=='done')
                j.update(status=stop or ('ok' if success else 'failed'),
                         exit_code=code, resource_stop=stop)
                batch.dump(directory/'resources.json', j)
        checkpoint()
        time.sleep(1)
    checkpoint('complete' if all(j['status']=='ok' for j in jobs) else 'complete_with_issues')


if __name__=='__main__':
    main()
