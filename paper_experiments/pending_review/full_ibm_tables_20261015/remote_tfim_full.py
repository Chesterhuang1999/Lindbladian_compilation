"""Parallel fresh full-pipeline TFIM measurements; preserve all previous outputs."""
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
CASES = ('TFIM-40', 'TFIM-4', 'TFIM-8', 'TFIM-12', 'TFIM-20')
POLICY = dict(max_workers=4, raw_worker_rss_gib=32, ours_worker_rss_gib=8,
              aggregate_rss_gib=104, minimum_available_gib=32,
              stage_timeout_s=43200, worker_timeout_s=86400)


def preflight():
    assert not (BASE/'remote_tfim_full_status.json').exists()
    assert not (BASE/'remote_tfim40_timing_status.json').exists(), 'Inspect existing TFIM-40 backend-only run before proceeding'
    m = batch.read(BASE/'manifest.json')
    plan = batch.read(BASE/'tfim_full_plan.json')
    assert all(batch.digest(BASE/name) == value for name,value in plan['workers'].items())
    assert list(CASES) == plan['cases']
    assert batch.digest(BASE/'run.py')==m['runner_sha256']
    assert all(batch.digest(batch.SNAP/p)==h for p,h in m['sources'].items())
    assert batch.read(BASE/'preflight_remote.json')['passed']
    assert psutil.virtual_memory().available>40*2**30
    for c in CASES:
        assert batch.digest(BASE/'inputs'/f'{c}.json')==m['cases'][c]['input_sha256']
        d=BASE/'cases'/c/('ours_full_remote' if c=='TFIM-40' else 'raw_full_remote')
        assert not d.exists() or not any(d.iterdir()), c



def main():
    preflight()
    if '--check' in sys.argv:
        print('Preflight passed:', CASES, POLICY)
        return
    lock = os.open(BASE/'remote_tfim_full.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock, str(os.getpid()).encode());os.close(lock)
    jobs=[dict(case=c,kind='ours_full_remote' if c=='TFIM-40' else 'raw_full_remote',host='remote',status='queued',
               purpose='fresh Ours pipeline' if c=='TFIM-40' else 'fresh Raw pipeline followed by opt3',
               worker_rss_gib=8 if c=='TFIM-40' else 32) for c in CASES]
    handles={}
    def checkpoint(status='running'):
        batch.dump(BASE/'remote_tfim_full_status.json',dict(status=status,host='remote',
            pid=os.getpid(),updated=time.time(),policy=POLICY,jobs=jobs,
            time_scope='Fresh Ours including rewrite for TFIM-40; fresh Raw and opt3 for TFIM-4/8/12/20; parallel single observations on one remote host'))
    checkpoint()
    while any(j['status'] in ('queued','running') for j in jobs):
        running=[j for j in jobs if j['status']=='running']
        usage={j['case']:rss(j['worker_pid']) for j in running}
        pressure=(sum(usage.values())>POLICY['aggregate_rss_gib']*2**30 or
                  psutil.virtual_memory().available<POLICY['minimum_available_gib']*2**30)
        victim=max(usage,key=usage.get) if pressure and usage else None
        for j in running:
            p=handles[j['case']];d=BASE/'cases'/j['case']/j['kind'];progress=d/'progress.json'
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
                    if j['kind']=='raw_full_remote':
                        j['raw_compile_s']=r['raw_compile_time_s']
                        j['optimizer_s']=r['optimizer_s']
                batch.dump(d/'resources.json',j)
        running=[j for j in jobs if j['status']=='running']
        for j in [j for j in jobs if j['status']=='queued']:
            if len(running)>=POLICY['max_workers']:break
            reserved=sum(x['worker_rss_gib'] for x in running)
            unused=sum(max(0,x['worker_rss_gib']*2**30-rss(x['worker_pid'])) for x in running)
            if reserved+j['worker_rss_gib']>POLICY['aggregate_rss_gib']:break
            if psutil.virtual_memory().available<unused+(j['worker_rss_gib']+POLICY['minimum_available_gib'])*2**30:break
            d=BASE/'cases'/j['case']/j['kind'];d.mkdir(parents=True,exist_ok=True)
            worker='tfim_full_worker.py' if j['kind']=='ours_full_remote' else 'tfim_raw_full_worker.py'
            with (d/'run.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-B',str(BASE/worker),'--worker',j['case'], '--host','remote'],stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            handles[j['case']]=p;now=time.time()
            j.update(status='running',worker_pid=p.pid,start_wall=now,stage_start_wall=now,
                     stage='starting',elapsed_s=0,peak_rss_bytes=0)
            running.append(j);checkpoint()
        checkpoint();time.sleep(1)
    checkpoint('complete' if all(j['status']=='ok' for j in jobs) else 'complete_with_issues')


if __name__=='__main__':main()
