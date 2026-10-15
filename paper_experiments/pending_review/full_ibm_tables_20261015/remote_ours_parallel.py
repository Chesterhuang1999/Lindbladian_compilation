"""Run the two explicitly migrated Ours jobs with remote resource guards."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import run as batch

BASE = Path(__file__).resolve().parent
POLICY = {'max_workers':2, 'worker_rss_gib':8, 'aggregate_rss_gib':16,
          'minimum_available_gib':16, 'stage_timeout_s':43200, 'worker_timeout_s':86400}


def rss(pid):
    try:
        root = psutil.Process(pid)
        value = 0
        for proc in [root] + root.children(recursive=True):
            try:
                value += proc.memory_info().rss
            except psutil.Error:
                pass
        return value
    except psutil.Error:
        return 0


def main():
    path = BASE / 'remote_ours_status.json'
    assert not path.exists(), 'Refusing duplicate remote Ours scheduler'
    manifest = batch.read(BASE/'manifest.json')
    assert batch.digest(BASE/'run.py') == manifest['runner_sha256']
    assert all(batch.digest(batch.SNAP/p)==h for p,h in manifest['sources'].items())
    assert batch.read(BASE/'preflight_remote.json')['passed']
    assert psutil.virtual_memory().available > 32*2**30
    jobs = [{'case':case,'kind':'ours','host':'remote','status':'queued'} for case in ('TFIM-54','TFIM-60')]
    handles = {}
    def checkpoint(status='running'):
        batch.dump(path, {'status':status,'host':'remote','pid':os.getpid(),
            'updated':time.time(),'policy':POLICY,'jobs':jobs,
            'scope':'Ours migrated from local; separate from completed Raw queue'})
    checkpoint()
    for j in jobs:
        directory = BASE/'cases'/j['case']/'ours'
        assert not (directory/'results.json').exists()
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'run.log').open('w') as log:
            p = subprocess.Popen([sys.executable,'-B',str(BASE/'run.py'),
                '--worker',j['case'],'--kind','ours','--host','remote'],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        handles[j['case']] = p
        now = time.time()
        j.update(status='running',worker_pid=p.pid,start_wall=now,
                 stage_start_wall=now,stage='starting',peak_rss_bytes=0)
        checkpoint()
    while any(j['status']=='running' for j in jobs):
        running = [j for j in jobs if j['status']=='running']
        usage = {j['case']:rss(j['worker_pid']) for j in running}
        pressure = sum(usage.values())>16*2**30 or psutil.virtual_memory().available<16*2**30
        victim = max(usage,key=usage.get) if pressure and usage else None
        for j in running:
            p = handles[j['case']]
            directory = BASE/'cases'/j['case']/'ours'
            progress = directory/'progress.json'
            if progress.exists():
                state = batch.read(progress)
                if state['stage'] != j['stage']:
                    j.update(stage=state['stage'],stage_start_wall=state['updated'])
            j.update(elapsed_s=time.time()-j['start_wall'],
                     peak_rss_bytes=max(j['peak_rss_bytes'],usage[j['case']]))
            stop = None
            if usage[j['case']]>8*2**30:stop='memory_limit'
            if victim == j['case']:stop='aggregate_memory_guard'
            if time.time()-j['stage_start_wall']>43200:stop='timeout'
            if j['elapsed_s']>86400:stop='worker_timeout'
            if stop and p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
            if p.poll() is not None:
                code = p.wait()
                result = directory/'results.json'
                success = code==0 and result.exists() and batch.read(result)['status']=='ok' and j['stage']=='done'
                j.update(status=stop or ('ok' if success else 'failed'),exit_code=code,resource_stop=stop)
                batch.dump(directory/'resources.json',j)
        checkpoint()
        time.sleep(1)
    checkpoint('complete' if all(j['status']=='ok' for j in jobs) else 'complete_with_issues')


if __name__=='__main__':
    main()
