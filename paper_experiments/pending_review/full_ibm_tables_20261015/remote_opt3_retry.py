"""User-authorized opt3 retries using existing, hash-verified Raw IBM QASM."""
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import psutil
import run as batch
from remote_ours_parallel import rss

BASE = Path(__file__).resolve().parent
CASES = ('TFIM-40', 'TFIM-54')
POLICY = dict(max_workers=1, worker_rss_gib=128, minimum_available_gib=32,
              launch_available_gib=160, stage_timeout_s=43200, worker_timeout_s=86400)


def preflight():
    m = batch.read(BASE/'manifest.json')
    assert batch.digest(BASE/'run.py') == m['runner_sha256']
    assert all(batch.digest(batch.SNAP/p)==h for p,h in m['sources'].items())
    assert batch.read(BASE/'preflight_remote.json')['passed']
    for c in CASES:
        d = BASE/'cases'/c
        r = batch.read(d/'raw/results.json')
        assert r['status']=='ok' and r['basis']=='IBM' and r['optimization_level']==0
        assert batch.digest(r['qasm_path'])==r['qasm_sha256']
        assert batch.digest(BASE/'inputs'/f'{c}.json')==r['input_sha256']==m['cases'][c]['input_sha256']
        assert not (d/'raw_opt3').exists(), 'Refuse to overwrite an opt3 attempt'


def worker(case):
    sys.path.insert(0, str(batch.SNAP/'src'))
    import qiskit
    from qiskit import qasm2
    from experiment_protocol import transpile_channel_circuit
    directory = BASE/'cases'/case/'raw_opt3'
    raw = batch.read(BASE/'cases'/case/'raw/results.json')
    def progress(stage):
        batch.dump(directory/'progress.json', dict(stage=stage, updated=time.time(), pid=os.getpid()))
    progress('verify_saved_raw')
    assert batch.digest(raw['qasm_path'])==raw['qasm_sha256']
    progress('load_saved_raw')
    start = time.perf_counter()
    circuit = qasm2.load(raw['qasm_path'], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    load_s = time.perf_counter()-start
    assert circuit.num_qubits==raw['W']
    assert dict(circuit.count_ops())==raw['counts']
    progress('qiskit_opt3')
    start = time.perf_counter()
    optimized = transpile_channel_circuit(circuit, basis_gates=batch.IBM,
        optimization_level=3, seed_transpiler=20261009, approximation_degree=1.0,
        qubits_initially_zero=False, num_processes=1)
    optimizer_s = time.perf_counter()-start
    del circuit
    counts = {str(k):int(v) for k,v in optimized.count_ops().items()}
    assert not set(counts)-set(batch.IBM)-{'reset','barrier'}
    assert optimized.num_qubits==raw['W']
    if optimized.layout is not None:
        assert optimized.layout.final_index_layout()==list(range(optimized.num_qubits))
    progress('export_opt3')
    path = directory/'channel_ibm.qasm'
    qasm2.dump(optimized,path)
    recount = Counter()
    width = 0
    with path.open() as stream:
        for line in stream:
            if line.startswith('qreg '): width += int(re.search(r'\[(\d+)\]',line)[1])
            op = re.match(r'^(u1|u2|u3|cx|reset|barrier)(?:\(|\s)',line)
            if op: recount[op[1]] += 1
    assert dict(recount)==counts and width==raw['W']
    result = dict(raw, method='Raw+Qiskit opt3', status='ok', host='remote',
        G=sum(counts.get(g,0) for g in batch.IBM), counts=counts,
        optimization_level=3, optimizer_s=optimizer_s,
        compile_time_s=raw['compile_time_s']+optimizer_s,
        backend_time_s=raw['backend_time_s']+optimizer_s,
        time_scope='saved Raw compilation stages + newly measured IBM opt3; excludes QASM import/export, verification and I/O',
        timing_host_note='saved remote Raw compilation; opt3 retry on remote',
        saved_raw_qasm_sha256=raw['qasm_sha256'], saved_raw_result_sha256=batch.digest(BASE/'cases'/case/'raw/results.json'),
        qasm_import_s=load_s, qasm_path=str(path), qasm_sha256=batch.digest(path),
        qiskit_version=qiskit.__version__, retry_policy=POLICY,
        full_choi=dict(status='not_run', scope='saved input QASM hash/count/width; exact opt3 settings; output gate set, width and exported recount checked'))
    batch.dump(directory/'results.json',result)
    progress('done')


def coordinate(status_file='remote_opt3_retry_status.json'):
    path = BASE/status_file
    assert not path.exists(), 'Refusing duplicate retry coordinator'
    preflight()
    if '--check' in sys.argv:
        print('Saved Raw inputs verified; retry policy', POLICY)
        return
    jobs = [dict(case=c,kind='raw',host='remote',status='queued',attempt='opt3_saved_raw_retry') for c in CASES]
    def checkpoint(status='running'):
        batch.dump(path, dict(status=status, pid=os.getpid(), updated=time.time(),
                   policy=POLICY, jobs=jobs))
    checkpoint()
    for j in jobs:
        while psutil.virtual_memory().available < POLICY['launch_available_gib']*2**30:
            checkpoint();time.sleep(10)
        directory = BASE/'cases'/j['case']/'raw_opt3'
        directory.mkdir(parents=True,exist_ok=False)
        with (directory/'run.log').open('w') as log:
            p = subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',j['case']],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        start = time.time()
        j.update(status='running',worker_pid=p.pid,start_wall=start,stage_start_wall=start,
                 stage='starting',peak_rss_bytes=0)
        stop = None
        while True:
            progress = directory/'progress.json'
            if progress.exists():
                state=batch.read(progress)
                if state['stage']!=j['stage']:j.update(stage=state['stage'],stage_start_wall=state['updated'])
            used=rss(p.pid)
            j.update(elapsed_s=time.time()-start,peak_rss_bytes=max(j['peak_rss_bytes'],used))
            if p.poll() is not None:break
            if used>POLICY['worker_rss_gib']*2**30:stop='memory_limit'
            if psutil.virtual_memory().available<POLICY['minimum_available_gib']*2**30:stop='available_memory_guard'
            if time.time()-j['stage_start_wall']>POLICY['stage_timeout_s']:stop='timeout'
            if j['elapsed_s']>POLICY['worker_timeout_s']:stop='worker_timeout'
            if stop:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            checkpoint();time.sleep(1)
        code=p.wait()
        result=directory/'results.json'
        if progress.exists():j['stage']=batch.read(progress)['stage']
        success=code==0 and result.exists() and batch.read(result)['status']=='ok' and j['stage']=='done'
        j.update(status=stop or ('ok' if success else 'failed'),exit_code=code,resource_stop=stop)
        batch.dump(directory/'resources.json',j)
        checkpoint()
    checkpoint('complete' if all(j['status']=='ok' for j in jobs) else 'complete_with_issues')


if __name__=='__main__':
    os.environ['QISKIT_PARALLEL']='FALSE'
    os.environ['MPLCONFIGDIR']=str(BASE/'.runtime/mpl')
    if '--worker' in sys.argv:worker(sys.argv[sys.argv.index('--worker')+1])
    else:coordinate()
