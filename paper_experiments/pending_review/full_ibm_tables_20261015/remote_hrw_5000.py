"""User-authorized HRW extension, with a 5000 s hard limit per worker.

Keep the frozen compiler, earlier coordinators and all historical attempts intact.
"""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

import psutil
import run as batch
from remote_ours_parallel import rss

BASE = Path(__file__).resolve().parent
STATUS = BASE/'remote_hrw_5000_status.json'
POLICY = dict(max_workers=3, worker_timeout_s=5000, stage_timeout_s=5000,
              aggregate_rss_gib=256, minimum_available_gib=32,
              timeout_scope='worker wall clock including import, verification and export',
              excluded_cases=['HRW-1440'])
JOBS = [dict(case='HRW-1024', kind='ours_full_remote_5000s', mode='ours', worker_rss_gib=64),
        dict(case='HRW-256', kind='raw_opt3_remote_5000s', mode='opt3', worker_rss_gib=64),
        dict(case='HRW-1024', kind='raw_full_remote_5000s', mode='raw', worker_rss_gib=128),
        dict(case='HRW-512', kind='raw_opt3_remote_5000s', mode='opt3', worker_rss_gib=128)]


def progress(directory, stage):
    batch.dump(directory/'progress.json', dict(stage=stage, updated=time.time(), pid=os.getpid()))


def preflight():
    manifest = batch.read(BASE/'manifest.json')
    plan = batch.read(BASE/'hrw_5000_plan.json')
    assert batch.digest(__file__) == plan['runner_sha256']
    assert batch.digest(BASE/'manifest.json') == plan['frozen_manifest_sha256']
    assert batch.digest(BASE/'hrw_series_full_worker.py') == plan['ours_worker_sha256']
    assert batch.digest(BASE/'run.py') == manifest['runner_sha256']
    assert all(batch.digest(batch.SNAP/p) == h for p,h in manifest['sources'].items())
    assert batch.read(BASE/'preflight_remote.json')['passed']
    for j in JOBS:
        case = j['case']
        assert batch.digest(BASE/'inputs'/f'{case}.json') == manifest['cases'][case]['input_sha256']
        directory = BASE/'cases'/case/j['kind']
        assert not directory.exists() or not any(directory.iterdir()), str(directory)
        if j['mode'] == 'opt3':
            raw = batch.read(BASE/'cases'/case/'raw/results.json')
            assert raw['status'] == 'ok' and raw['host'] == 'remote'
            assert raw['basis'] == 'IBM' and raw['optimization_level'] == 0
            assert raw['input_sha256'] == manifest['cases'][case]['input_sha256']
            assert batch.digest(raw['qasm_path']) == raw['qasm_sha256'] == plan['saved_raw_hashes'][case]


def raw_worker(case, directory):
    sys.path[:0] = [str(batch.SNAP/p) for p in ('src','scripts/eval_redesign','Baseline_scripts')]
    import qiskit
    from channel_IR import channel_ensemble
    from channel_LCU import channel_to_LCU
    from experiment_protocol import source_identity, transpile_channel_circuit
    from rq1_pipeline import canonicalize, deserialize_kraus, serialize_kraus
    from run_selected_tfim_hrw import sparse_gram_check

    entry = batch.read(BASE/'inputs'/f'{case}.json')
    sources = source_identity()
    assert sources == batch.read(BASE/'smoke.json')['sources']
    target = deserialize_kraus(entry['target'])
    progress(directory, 'canonicalization'); start = time.perf_counter(); pipeline_start = start
    canonical, canonical_info = canonicalize(target)
    canonical_s = time.perf_counter()-start
    progress(directory, 'assembly'); start = time.perf_counter()
    circuit, registers = channel_to_LCU(channel_ensemble([canonical]), structure='basic', opt='No', fstar=False)
    assembly_s = time.perf_counter()-start
    lam = float(circuit.metadata['subnormalization_factor'])
    width = circuit.num_qubits
    assert math.isclose(lam, entry['Raw']['Lambda'], rel_tol=1e-10)
    progress(directory, 'ibm_lowering'); start = time.perf_counter()
    converted = transpile_channel_circuit(circuit, basis_gates=batch.IBM,
        optimization_level=0, seed_transpiler=20261009, approximation_degree=1.0,
        qubits_initially_zero=False, num_processes=1)
    lowering_s = time.perf_counter()-start
    pipeline_elapsed_s = time.perf_counter()-pipeline_start
    del circuit
    progress(directory, 'validation')
    gram = sparse_gram_check(entry['target'], serialize_kraus(canonical))
    assert gram['passed'], gram
    assert converted.num_qubits == width and width >= entry['n']
    system = registers.get('system') if isinstance(registers, dict) else None
    if system is not None:
        assert len(system) == entry['n']
    result = dict(case=case, method='Raw', status='ok', host='remote', basis='IBM',
        timing_kind='fresh_full_pipeline', saved_raw_reused=False, W=width, Lambda=lam,
        canonicalization_s=canonical_s, assembly_s=assembly_s, lowering_s=lowering_s,
        optimizer_s=0, rewrite_s=0, compile_time_s=canonical_s+assembly_s+lowering_s,
        backend_time_s=assembly_s+lowering_s, pipeline_elapsed_s=pipeline_elapsed_s,
        input_sha256=batch.digest(BASE/'inputs'/f'{case}.json'), sources=sources,
        worker_sha256=batch.digest(__file__), input_map_check=gram, canonicalization=canonical_info,
        NAM_reproduction='not_run: no successful historical Raw reference',
        full_choi=dict(status='not_run', scope='coefficient Gram; IBM gate set, width and exported recount'),
        time_scope='fresh canonicalization + Raw assembly + IBM opt0 lowering on remote; excludes imports, input deserialization, validation and export')
    save_circuit(converted, directory, result)
    assert source_identity() == sources


def opt3_worker(case, directory):
    sys.path.insert(0, str(batch.SNAP/'src'))
    from qiskit import qasm2
    from experiment_protocol import transpile_channel_circuit
    raw_path = BASE/'cases'/case/'raw/results.json'
    raw = batch.read(raw_path)
    progress(directory, 'verify_saved_raw')
    assert batch.digest(raw['qasm_path']) == raw['qasm_sha256']
    progress(directory, 'load_saved_raw'); start = time.perf_counter()
    circuit = qasm2.load(raw['qasm_path'], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    import_s = time.perf_counter()-start
    assert circuit.num_qubits == raw['W'] and dict(circuit.count_ops()) == raw['counts']
    progress(directory, 'qiskit_opt3'); start = time.perf_counter()
    optimized = transpile_channel_circuit(circuit, basis_gates=batch.IBM,
        optimization_level=3, seed_transpiler=20261009, approximation_degree=1.0,
        qubits_initially_zero=False, num_processes=1)
    optimizer_s = time.perf_counter()-start
    del circuit
    result = dict(raw, method='Raw+Qiskit opt3', optimization_level=3,
        timing_kind='saved_remote_raw_plus_new_opt3', saved_raw_reused=True,
        optimizer_s=optimizer_s, compile_time_s=raw['compile_time_s']+optimizer_s,
        backend_time_s=raw['backend_time_s']+optimizer_s,
        saved_raw_qasm_sha256=raw['qasm_sha256'], saved_raw_result_sha256=batch.digest(raw_path),
        qasm_import_s=import_s, worker_sha256=batch.digest(__file__),
        full_choi=dict(status='not_run', scope='saved Raw QASM hash/count/width; arbitrary-input-safe opt3; output gate set/width/exported recount'),
        time_scope='saved remote Raw compilation + newly measured remote IBM opt3; excludes QASM import/export and validation')
    save_circuit(optimized, directory, result)


def save_circuit(circuit, directory, result):
    import qiskit
    from qiskit import qasm2
    assert circuit.num_qubits == result['W']
    if circuit.layout is not None:
        assert circuit.layout.final_index_layout() == list(range(circuit.num_qubits))
    counts = {str(k):int(v) for k,v in circuit.count_ops().items()}
    assert set(counts) <= set(batch.IBM) | {'reset','barrier'}
    progress(directory, 'export')
    path = directory/'channel_ibm.qasm'
    qasm2.dump(circuit, path)
    recount = Counter(); width = 0
    with path.open() as stream:
        for line in stream:
            if line.startswith('qreg '): width += int(re.search(r'\[(\d+)\]',line)[1])
            op = re.match(r'^(u1|u2|u3|cx|reset|barrier)(?:\(|\s)',line)
            if op: recount[op[1]] += 1
    assert dict(recount) == counts and width == result['W']
    result.update(G=sum(counts.get(k,0) for k in batch.IBM), counts=counts,
        qasm_path=str(path), qasm_sha256=batch.digest(path), qiskit_version=qiskit.__version__,
        qubits_initially_zero=False, seed_transpiler=20261009, approximation_degree=1.0,
        optimization_level=result.get('optimization_level',0), timeout_policy=POLICY)
    batch.dump(directory/'results.json', result)
    progress(directory, 'done')


def worker(case, mode):
    job = next(j for j in JOBS if j['case']==case and j['mode']==mode)
    directory = BASE/'cases'/case/job['kind']
    try:
        if mode == 'ours':
            from hrw_series_full_worker import worker as ours_worker
            ours_worker(case, 'remote', job['kind'])
        elif mode == 'raw': raw_worker(case, directory)
        else: opt3_worker(case, directory)
    except Exception:
        batch.dump(directory/'failure.json',dict(status='failed', traceback=traceback.format_exc(), updated=time.time()))
        raise


def coordinate(check_only):
    assert not STATUS.exists(), 'Refusing duplicate coordinator'
    preflight()
    if check_only:
        print('Preflight passed', POLICY, JOBS, flush=True)
        return
    lock = os.open(BASE/'remote_hrw_5000.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    os.write(lock, str(os.getpid()).encode()); os.close(lock)
    jobs = [dict(j, host='remote', status='queued') for j in JOBS]
    handles = {}
    def checkpoint(status='running'):
        batch.dump(STATUS, dict(status=status, host='remote', pid=os.getpid(), updated=time.time(),
            policy=POLICY, jobs=jobs, hrw1440='no active process at preflight; excluded from new jobs'))
    checkpoint()
    while any(j['status'] in ('queued','running') for j in jobs):
        running = [j for j in jobs if j['status']=='running']
        usage = {(j['case'],j['mode']):rss(j['worker_pid']) for j in running}
        pressure = (sum(usage.values()) > POLICY['aggregate_rss_gib']*2**30 or
                    psutil.virtual_memory().available < POLICY['minimum_available_gib']*2**30)
        victim = max(usage, key=usage.get) if pressure and usage else None
        for j in running:
            key = j['case'],j['mode']; p = handles[key]; directory = BASE/'cases'/j['case']/j['kind']
            mark = directory/'progress.json'
            if mark.exists():
                state = batch.read(mark)
                if state['stage']!=j['stage']: j.update(stage=state['stage'],stage_start_wall=state['updated'])
            j.update(elapsed_s=time.time()-j['start_wall'], peak_rss_bytes=max(j['peak_rss_bytes'],usage[key]))
            stop = None
            if p.poll() is None:
                if usage[key]>j['worker_rss_gib']*2**30: stop='memory_limit'
                if key==victim: stop='aggregate_memory_guard'
                if j['elapsed_s']>=POLICY['worker_timeout_s']: stop='timeout'
                if stop:
                    os.killpg(p.pid,signal.SIGTERM)
                    try: p.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
            if p.poll() is not None:
                code=p.wait()
                if mark.exists(): j['stage']=batch.read(mark)['stage']
                result=directory/'results.json'
                success=code==0 and result.exists() and batch.read(result)['status']=='ok' and j['stage']=='done'
                j.update(status=stop or ('ok' if success else 'failed'), exit_code=code, resource_stop=stop)
                if success:
                    r=batch.read(result)
                    j.update(full_compile_s=r['compile_time_s'], optimizer_s=r.get('optimizer_s',0),
                             fresh_rewrite_s=r.get('rewrite_s',0) if j['mode']=='ours' else 0)
                batch.dump(directory/'resources.json',j)
        running=[j for j in jobs if j['status']=='running']
        for j in [j for j in jobs if j['status']=='queued']:
            if len(running)>=POLICY['max_workers']: break
            reserved=sum(x['worker_rss_gib'] for x in running)
            unused=sum(max(0,x['worker_rss_gib']*2**30-rss(x['worker_pid'])) for x in running)
            if reserved+j['worker_rss_gib']>POLICY['aggregate_rss_gib']: continue
            if psutil.virtual_memory().available<unused+(j['worker_rss_gib']+POLICY['minimum_available_gib'])*2**30: continue
            directory=BASE/'cases'/j['case']/j['kind']; directory.mkdir(parents=True,exist_ok=True)
            with (directory/'run.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)), '--worker',j['case'],'--mode',j['mode']],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            handles[j['case'],j['mode']]=p; now=time.time()
            j.update(status='running', worker_pid=p.pid,start_wall=now,stage_start_wall=now,
                     stage='starting',elapsed_s=0,peak_rss_bytes=0)
            running.append(j); checkpoint()
        checkpoint(); time.sleep(.5)
    checkpoint('complete' if all(j['status']=='ok' for j in jobs) else 'complete_with_issues')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker'); parser.add_argument('--mode',choices=['ours','raw','opt3'])
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS'):
        os.environ[name]='1'
    os.environ.update(QISKIT_PARALLEL='FALSE', PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(BASE/'.runtime/hrw5000/tmp'), MPLCONFIGDIR=str(BASE/'.runtime/hrw5000/mpl'),
        XDG_CACHE_HOME=str(BASE/'.runtime/hrw5000/cache'))
    for name in ('TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME'): Path(os.environ[name]).mkdir(parents=True,exist_ok=True)
    if args.worker: worker(args.worker,args.mode)
    else: coordinate(args.check)
