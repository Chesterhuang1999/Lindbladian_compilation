"""Serial fresh full-pipeline HRW measurements; preserve all previous outputs."""
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
CASES = ('HRW-4', 'HRW-8', 'HRW-12', 'HRW-20', 'HRW-28', 'HRW-48', 'HRW-60', 'HRW-128', 'HRW-256', 'HRW-512')
POLICY = dict(max_workers=1, large_worker_rss_gib=128, small_worker_rss_gib=8,
              aggregate_rss_gib=128, minimum_available_gib=32,
              stage_timeout_s=43200, worker_timeout_s=86400)


def dependency_ready(state):
    expected = {'TFIM-40', 'TFIM-4', 'TFIM-8', 'TFIM-12', 'TFIM-20'}
    terminal = {'ok', 'failed', 'memory_limit', 'aggregate_memory_guard', 'timeout', 'worker_timeout'}
    jobs = state.get('jobs', [])
    return (state.get('status') in ('complete', 'complete_with_issues')
            and len(jobs) == len(expected)
            and {j.get('case') for j in jobs} == expected
            and all(j.get('status') in terminal and j.get('exit_code') is not None for j in jobs))


def preflight():
    assert not (BASE/'remote_hrw_series_full_status.json').exists()
    m = batch.read(BASE/'manifest.json')
    plan = batch.read(BASE/'hrw_series_full_plan.json')
    assert batch.digest(BASE/'hrw_series_full_worker.py') == plan['worker_sha256']
    assert batch.digest(BASE/'manifest.json') == plan['frozen_manifest_sha256']
    assert list(CASES) == plan['cases']
    assert batch.digest(BASE/'run.py')==m['runner_sha256']
    assert all(batch.digest(batch.SNAP/p)==h for p,h in m['sources'].items())
    assert batch.read(BASE/'preflight_remote.json')['passed']
    for c in CASES:
        assert batch.digest(BASE/'inputs'/f'{c}.json')==m['cases'][c]['input_sha256']
        d=BASE/'cases'/c/'ours_full_remote_series'
        assert not d.exists() or not any(d.iterdir()), c



def main():
    preflight()
    if '--check' in sys.argv:
        print('Preflight passed:', CASES, POLICY)
        return
    lock = os.open(BASE/'remote_hrw_series_full.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock, str(os.getpid()).encode());os.close(lock)
    jobs=[dict(case=c,kind='ours_full_remote_series',host='remote',status='queued',
               purpose='fresh full remote compilation', worker_rss_gib=128 if int(c.split('-')[1])>=512 else 8) for c in CASES]
    handles={}
    def checkpoint(status='running'):
        batch.dump(BASE/'remote_hrw_series_full_status.json',dict(status=status,host='remote',
            pid=os.getpid(),updated=time.time(),policy=POLICY,jobs=jobs,
            dependency='remote_tfim_full_status.json',
            time_scope='Fresh canonicalization, rewrite, assembly and IBM lowering on the remote host; serial single observations'))
    checkpoint('waiting_for_tfim')
    while not dependency_ready(batch.read(BASE/'remote_tfim_full_status.json')):
        checkpoint('waiting_for_tfim')
        time.sleep(10)
    checkpoint()
    while any(j['status'] in ('queued','running') for j in jobs):
        running=[j for j in jobs if j['status']=='running']
        usage={j['case']:rss(j['worker_pid']) for j in running}
        pressure=(sum(usage.values())>POLICY['aggregate_rss_gib']*2**30 or
                  psutil.virtual_memory().available<POLICY['minimum_available_gib']*2**30)
        victim=max(usage,key=usage.get) if pressure and usage else None
        for j in running:
            p=handles[j['case']];d=BASE/'cases'/j['case']/'ours_full_remote_series';progress=d/'progress.json'
            if progress.exists():
                state=batch.read(progress)
                if state['stage']!=j['stage']:j.update(stage=state['stage'],stage_start_wall=state['updated'])
            j.update(elapsed_s=time.time()-j['start_wall'],peak_rss_bytes=max(j['peak_rss_bytes'],usage[j['case']]))
            stop=None
            if p.poll() is None:
                if usage[j['case']]>j['worker_rss_gib']*2**30:stop='memory_limit'
                if victim==j['case']:stop='aggregate_memory_guard'
                if time.time()-j['stage_start_wall']>POLICY['stage_timeout_s']:stop='timeout'
                if j['elapsed_s']>POLICY['worker_timeout_s']:stop='worker_timeout'
                if stop:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            if p.poll() is not None:
                code=p.wait()
                if progress.exists():j['stage']=batch.read(progress)['stage']
                result=d/'results.json'
                success=code==0 and result.exists() and batch.read(result)['status']=='ok' and j['stage']=='done'
                j.update(status=stop or ('ok' if success else 'failed'),exit_code=code,resource_stop=stop)
                if success:
                    r=batch.read(result)
                    j['remote_compile_stages_s']=sum(r[k] for k in ('canonicalization_s','assembly_s','lowering_s'))
                    j['fresh_rewrite_s']=r['rewrite_s']
                    j['full_compile_s']=r['compile_time_s']
                batch.dump(d/'resources.json',j)
        running=[j for j in jobs if j['status']=='running']
        for j in [j for j in jobs if j['status']=='queued']:
            if len(running)>=POLICY['max_workers']:break
            reserved=sum(x['worker_rss_gib'] for x in running)
            unused=sum(max(0,x['worker_rss_gib']*2**30-rss(x['worker_pid'])) for x in running)
            if reserved+j['worker_rss_gib']>POLICY['aggregate_rss_gib']:break
            if psutil.virtual_memory().available<unused+(j['worker_rss_gib']+POLICY['minimum_available_gib'])*2**30:break
            d=BASE/'cases'/j['case']/'ours_full_remote_series';d.mkdir(parents=True,exist_ok=True)
            with (d/'run.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-B',str(BASE/'hrw_series_full_worker.py'),'--worker',j['case'], '--host','remote', '--output-kind','ours_full_remote_series'],stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            handles[j['case']]=p;now=time.time()
            j.update(status='running',worker_pid=p.pid,start_wall=now,stage_start_wall=now,
                     stage='starting',elapsed_s=0,peak_rss_bytes=0)
            running.append(j);checkpoint()
        checkpoint();time.sleep(1)
    checkpoint('complete' if all(j['status']=='ok' for j in jobs) else 'complete_with_issues')


if __name__=='__main__':main()

