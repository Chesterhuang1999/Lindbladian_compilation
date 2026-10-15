"""Resource-guarded replacement scheduler; workers and frozen sources stay intact."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import run as batch

BASE = Path(__file__).resolve().parent
POLICY = {
    'max_workers': 2, 'stage_timeout_s': 43200, 'worker_timeout_s': 86400,
    'worker_rss_bytes': 3 * 2**30, 'aggregate_rss_bytes': 4 * 2**30,
    'launch_available_bytes': 3 * 2**30, 'minimum_available_bytes': 1 * 2**30,
    'poll_seconds': 1,
}


def alive(pid, created):
    try:
        p = psutil.Process(pid)
        return p.create_time() == created and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def usage(pid):
    try:
        p = psutil.Process(pid)
        total = 0
        for child in [p] + p.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return 0


def stop_process(job):
    if alive(job['worker_pid'], job['worker_created']):
        os.killpg(job['worker_pid'], signal.SIGTERM)
        deadline = time.monotonic() + 5
        while alive(job['worker_pid'], job['worker_created']) and time.monotonic() < deadline:
            time.sleep(.2)
        if alive(job['worker_pid'], job['worker_created']):
            os.killpg(job['worker_pid'], signal.SIGKILL)


def main():
    manifest = batch.read(BASE / 'manifest.json')
    assert batch.digest(BASE / 'run.py') == manifest['runner_sha256']
    assert all(batch.digest(batch.SNAP / p) == h for p, h in manifest['sources'].items())
    old = batch.read(BASE / 'local_status.json')
    old_pid = old['pid']
    proc = psutil.Process(old_pid)
    command = proc.cmdline()
    migrate_hrw = '--migrate-hrw-remote' in sys.argv
    migrate = '--migrate-tfim-remote' in sys.argv or migrate_hrw
    targets = ('HRW-512', 'HRW-1024', 'HRW-1440') if migrate_hrw else ('TFIM-54', 'TFIM-60')
    backup = BASE / ('local_status_before_hrw_remote_migration.json' if migrate_hrw else
                     'local_status_before_remote_migration.json' if migrate else 'local_status_before_parallel.json')
    if migrate:
        assert str(BASE / 'local_parallel.py') in command
    else:
        assert str(BASE / 'run.py') in command and '--host' in command and 'local' in command
    assert not backup.exists(), 'Already handed over'
    if migrate:
        for case in targets:
            assert not (BASE / 'cases' / case / 'ours_local_interrupted_for_remote').exists()
            assert not (BASE / 'cases' / case / 'ours_local_failed_before_remote').exists()
    # Freeze dispatch while taking the state snapshot. The active child keeps running.
    os.kill(old_pid, signal.SIGSTOP)
    try:
        old = batch.read(BASE / 'local_status.json')
        jobs = old['jobs']
        batch.dump(backup, old)
        batch.dump(BASE / 'local_parallel_policy.json', POLICY)
    except BaseException:
        os.kill(old_pid, signal.SIGCONT)
        raise
    os.kill(old_pid, signal.SIGKILL)  # only the dispatcher, never its worker group
    handles = {}
    for j in jobs:
        if migrate and j['case'] in targets:
            was_running = j['status'] == 'running'
            if was_running:
                stop_process(j)
            directory = BASE / 'cases' / j['case']
            archive = directory / ('ours_local_interrupted_for_remote' if was_running else 'ours_local_failed_before_remote')
            assert not archive.exists()
            (directory / 'ours').rename(archive)
            if was_running:
                interrupted = dict(j, status='interrupted_for_remote_migration',
                                   stopped_at=time.time(), elapsed_s=time.time()-j['start_wall'],
                                   reason='user requested remote execution; no assembly checkpoint')
                batch.dump(archive / 'resources.json', interrupted)
            j.update(status='delegated', host='remote', previous_attempt=str(archive),
                     remote_status_file='remote_hrw_status.json' if migrate_hrw else 'remote_ours_status.json')
            continue
        if j['status'] == 'running':
            worker = psutil.Process(j['worker_pid'])
            progress = batch.read(BASE / 'cases' / j['case'] / 'ours/progress.json')
            j.update(worker_created=worker.create_time(), start_wall=time.time()-j['elapsed_s'],
                     stage_start_wall=progress['updated'], adopted=True)
        elif j['case'] == 'TFIM-54' and j['status'] == 'timeout':
            directory = BASE / 'cases' / j['case']
            archive = directory / 'ours_attempt_1_timeout'
            assert not archive.exists()
            (directory / 'ours').rename(archive)
            j.clear()
            j.update(case='TFIM-54', kind='ours', host='local', priority=18582,
                     status='queued', attempt=2, previous_attempt=str(archive))

    def checkpoint(status='running'):
        batch.dump(BASE / 'local_status.json', {
            'status': status, 'host': 'local', 'pid': os.getpid(), 'updated': time.time(),
            'scheduler': str(Path(__file__)), 'policy': POLICY,
            'previous_coordinator_pid': old_pid, 'jobs': jobs})

    checkpoint()
    while any(j['status'] in ('queued', 'running') for j in jobs):
        running = [j for j in jobs if j['status'] == 'running']
        rss = {j['case']: usage(j['worker_pid']) for j in running}
        available = psutil.virtual_memory().available
        aggregate = sum(rss.values())
        pressure_victim = None
        if running and (aggregate > POLICY['aggregate_rss_bytes'] or available < POLICY['minimum_available_bytes']):
            pressure_victim = max(running, key=lambda j: rss[j['case']])['case']
        for j in running:
            directory = BASE / 'cases' / j['case'] / 'ours'
            now = time.time()
            ppath = directory / 'progress.json'
            if ppath.exists():
                progress = batch.read(ppath)
                if progress['stage'] != j.get('stage'):
                    j.update(stage=progress['stage'], stage_start_wall=progress['updated'])
            j.update(elapsed_s=now-j['start_wall'],
                     peak_rss_bytes=max(j.get('peak_rss_bytes', 0), rss[j['case']]))
            stop = None
            if rss[j['case']] > POLICY['worker_rss_bytes']:
                stop = 'memory_limit'
            if j['case'] == pressure_victim:
                stop = 'aggregate_memory_guard'
            if now-j['stage_start_wall'] > POLICY['stage_timeout_s']:
                stop = 'timeout'
            if j['elapsed_s'] > POLICY['worker_timeout_s']:
                stop = 'worker_timeout'
            if stop:
                stop_process(j)
            handle = handles.get(j['case'])
            code = handle.poll() if handle else None
            if stop or not alive(j['worker_pid'], j['worker_created']):
                if handle:
                    code = handle.wait()
                result_path = directory / 'results.json'
                valid = result_path.exists() and batch.read(result_path).get('status') == 'ok'
                # For an adopted non-child, completion is established by the worker's
                # final progress marker and verified result; its OS exit code is unknown.
                done = valid and j.get('stage') == 'done' and (code == 0 or (j.get('adopted') and code is None))
                j.update(status=stop or ('ok' if done else 'failed'),
                         exit_code=code, resource_stop=stop,
                         completion_evidence='result + final progress' if done else 'resource/process termination')
                batch.dump(directory / 'resources.json', j)
        running = [j for j in jobs if j['status'] == 'running']
        while len(running) < POLICY['max_workers']:
            pending = sorted((j for j in jobs if j['status'] == 'queued'), key=lambda j:j['priority'])
            if not pending or psutil.virtual_memory().available < POLICY['launch_available_bytes']:
                break
            if sum(usage(j['worker_pid']) for j in running) > POLICY['aggregate_rss_bytes'] - 512*2**20:
                break
            j = pending[0]
            directory = BASE / 'cases' / j['case'] / 'ours'
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / 'run.log').open('w') as log:
                handle = subprocess.Popen([sys.executable, '-B', str(BASE/'run.py'),
                    '--worker', j['case'], '--kind', 'ours', '--host', 'local'],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            handles[j['case']] = handle
            now = time.time()
            j.update(status='running', worker_pid=handle.pid,
                     worker_created=psutil.Process(handle.pid).create_time(), start_wall=now,
                     stage_start_wall=now, stage='starting', elapsed_s=0, peak_rss_bytes=0)
            running.append(j)
            checkpoint()
        checkpoint()
        time.sleep(POLICY['poll_seconds'])
    checkpoint('complete' if all(j['status'] in ('ok','delegated') for j in jobs) else 'complete_with_issues')


if __name__ == '__main__':
    main()
