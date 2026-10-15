"""Parallel remote backend timing reruns and the authorized HRW-1440 retry."""
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
TIMING_CASES = ('TFIM-4','TFIM-8','TFIM-12','TFIM-20','TFIM-32',
                'HRW-4','HRW-8','HRW-12','HRW-20','HRW-28','HRW-48',
                'HRW-60','HRW-128','HRW-256')
CASES = ('HRW-1440',) + TIMING_CASES
POLICY = dict(max_workers=4, large_worker_rss_gib=128, small_worker_rss_gib=8,
              aggregate_rss_gib=160, minimum_available_gib=32,
              stage_timeout_s=43200, worker_timeout_s=86400)


def preflight():
    assert not (BASE/'remote_ours_timing_status.json').exists()
    m = batch.read(BASE/'manifest.json')
    assert batch.digest(BASE/'run.py')==m['runner_sha256']
    assert all(batch.digest(batch.SNAP/p)==h for p,h in m['sources'].items())
    assert batch.read(BASE/'preflight_remote.json')['passed']
    assert psutil.virtual_memory().available>160*2**30
    for c in CASES:
        assert batch.digest(BASE/'inputs'/f'{c}.json')==m['cases'][c]['input_sha256']
        d=BASE/'cases'/c/'ours'
        if c=='HRW-1440':
            old=batch.read(d/'resources.json')
            assert old['status']=='memory_limit' and old['exit_code']==-15
            assert not psutil.pid_exists(old['worker_pid']), 'Previous worker still exists'
            assert not (d.parent/'ours_remote_attempt_1_memory_limit').exists()
        else:
            assert not d.exists() or not any(d.iterdir()), c


def main():
    preflight()
    if '--check' in sys.argv:
        print('Preflight passed:', CASES, POLICY)
        return
    d=BASE/'cases/HRW-1440/ours'
    d.rename(d.parent/'ours_remote_attempt_1_memory_limit')
    jobs=[dict(case=c,kind='ours',host='remote',status='queued',
               purpose='128 GiB retry' if c=='HRW-1440' else 'remote backend timing',
               worker_rss_gib=128 if c=='HRW-1440' else 8) for c in CASES]
    handles={}
    def checkpoint(status='running'):
        batch.dump(BASE/'remote_ours_timing_status.json',dict(status=status,host='remote',
            pid=os.getpid(),updated=time.time(),policy=POLICY,jobs=jobs,
            time_scope='Fresh remote canonicalization, assembly and lowering; saved local rewrite time remains separately labeled'))
    checkpoint()
    while any(j['status'] in ('queued','running') for j in jobs):
        running=[j for j in jobs if j['status']=='running']
        usage={j['case']:rss(j['worker_pid']) for j in running}
        pressure=(sum(usage.values())>POLICY['aggregate_rss_gib']*2**30 or
                  psutil.virtual_memory().available<POLICY['minimum_available_gib']*2**30)
        victim=max(usage,key=usage.get) if pressure and usage else None
        for j in running:
            p=handles[j['case']];d=BASE/'cases'/j['case']/'ours';progress=d/'progress.json'
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
                    j['saved_rewrite_s']=r['rewrite_s']
                batch.dump(d/'resources.json',j)
        running=[j for j in jobs if j['status']=='running']
        for j in [j for j in jobs if j['status']=='queued']:
            if len(running)>=POLICY['max_workers']:break
            reserved=sum(x['worker_rss_gib'] for x in running)
            unused=sum(max(0,x['worker_rss_gib']*2**30-rss(x['worker_pid'])) for x in running)
            if reserved+j['worker_rss_gib']>POLICY['aggregate_rss_gib']:break
            if psutil.virtual_memory().available<unused+(j['worker_rss_gib']+POLICY['minimum_available_gib'])*2**30:break
            d=BASE/'cases'/j['case']/'ours';d.mkdir(parents=True,exist_ok=True)
            with (d/'run.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-B',str(BASE/'run.py'),'--worker',j['case'],
                    '--kind','ours','--host','remote'],stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            handles[j['case']]=p;now=time.time()
            j.update(status='running',worker_pid=p.pid,start_wall=now,stage_start_wall=now,
                     stage='starting',elapsed_s=0,peak_rss_bytes=0)
            running.append(j);checkpoint()
        checkpoint();time.sleep(1)
    checkpoint('complete' if all(j['status']=='ok' for j in jobs) else 'complete_with_issues')


if __name__=='__main__':main()
