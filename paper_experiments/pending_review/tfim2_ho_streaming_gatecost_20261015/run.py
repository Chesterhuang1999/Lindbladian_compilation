"""One frozen, resource-monitored rerun; never invokes the old beam entry point."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SNAP = OUT / "source_snapshot"
CASE = OUT / "cases/TFIM2-HO"
PYTHON = Path('/Users/chester_hqf/miniconda3/envs/qiskit_simulate/bin/python')

def now():
    return datetime.now(timezone.utc).isoformat()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def progress(stage, **kwargs):
    value = dict(stage=stage, pid=os.getpid(), updated_at_utc=now(), **kwargs)
    dump(CASE / 'progress.json', value)
    print(json.dumps(value), flush=True)

def prepare():
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Output already prepared; do not overwrite or restart.')
    sources = {}
    for folder in ('src', 'scripts', 'Baseline_scripts'):
        for source in sorted((ROOT / folder).rglob('*.py')):
            relative = source.relative_to(ROOT)
            target = SNAP / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources[str(relative)] = digest(target)
    old = ROOT / 'paper_experiments/pending_review/tfim_hrw_full_20261011/cases/TFIM2-HO/input.json'
    CASE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old, CASE / 'input.json')
    dump(OUT / 'manifest.json', {
        'case': 'TFIM2-HO', 'prepared_at_utc': now(), 'input_source': str(old),
        'input_sha256': digest(old), 'sources': sources, 'runner_sha256': digest(Path(__file__)),
        'rewrite': {'implementation': 'scripts/eval_redesign/hybrid_rewrite.py::gate_search',
                    'objective': 'static_gate_cost', 'mode': 'admit_support', 'collector': 'heap',
                    'beam_width': 8, 'max_steps': 1167, 'tol': 1e-12, 'neutral': True,
                    'timeout_s': 1800, 'hybrid_support_primary_search': False},
        'lowering': {'structure': 'tree', 'fstar': True, 'opt': 'Matrix-order',
                     'basis': 'nam', 'optimization_level': 0, 'terminal_optimizer': None},
        'limits': {'worker_rss_gib': 6, 'worker_wall_s': 3600, 'verification_wall_s': 300},
        'comparison_note': 'Old run used width 8 and max_steps 50; this run permits 1167 layers to test convergence.'})
    for relative, expected in sources.items():
        assert digest(ROOT / relative) == expected
    env = os.environ.copy()
    done = subprocess.run([str(PYTHON), '-B', str(ROOT / 'Baseline_scripts/smoke_block_encoding_archive.py'),
                           '--out-dir', str(OUT)], env=env)
    assert done.returncode == 0
    assert json.loads((OUT / 'smoke.json').read_text())['passed']
    for relative, expected in sources.items():
        assert digest(ROOT / relative) == expected, relative
    print('Prepared frozen source and passed backend smoke.', flush=True)

def worker():
    sys.path[:0] = [str(SNAP / p) for p in ('src', 'scripts/eval_redesign', 'scripts', 'Baseline_scripts')]
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for relative, expected in manifest['sources'].items():
        assert digest(SNAP / relative) == expected, relative
    assert digest(CASE / 'input.json') == manifest['input_sha256']
    progress('importing')
    import numpy as np
    from qiskit import qasm2
    from channel_IR import channel as ChannelIR, channel_ensemble
    from rq1_pipeline import canonicalize, deserialize_kraus, serialize_kraus, verify_channel
    from hybrid_rewrite import gate_search, warmup, verify, metrics
    from channel_rewrite_cost import CostSettings
    from run_rq2_hybrid_ablation import matrix_rows
    from run_selected_tfim_hrw import sparse_gram_check
    from profile_ours_pipeline import StageTimers
    from block_encoding import BlockEncoding
    import rq1_pipeline as rq
    from experiment_protocol import transpile_channel_circuit, source_identity, DEFAULT_TRANSPILE_SEED
    from baseline_common import count_qasm_file
    assert json.loads((OUT / 'smoke.json').read_text())['sources'] == source_identity()
    raw_data = json.loads((CASE / 'input.json').read_text())
    raw = deserialize_kraus(raw_data['input_kraus'])
    started = time.perf_counter()
    canonical, canonical_info = canonicalize(raw)
    A0, labels = ChannelIR(canonical)._coeff_matrix(tol=0.0)
    canonical_s = time.perf_counter() - started
    assert sparse_gram_check(raw_data['input_kraus'], serialize_kraus(canonical))['passed']
    settings = CostSettings('tree', True)
    progress('jit_warmup', shape=list(A0.shape))
    jit_s = warmup()
    search_dir = CASE / 'search'
    search_dir.mkdir(exist_ok=True)
    progress('rewrite')
    started = time.perf_counter()
    state, term = gate_search(A0, search_dir, width=8, max_steps=1167, tol=1e-12,
                             neutral=True, timeout=1800, collector='heap',
                             mode='admit_support', settings=settings)
    rewrite_s = time.perf_counter() - started
    final = state.A.copy()
    final[np.abs(final) <= 1e-12] = 0
    check, replay = verify(A0, final, state.steps, 1e-12)
    assert check['passed'], check
    kraus = matrix_rows(final, labels, 1e-12)
    raw_check = sparse_gram_check(raw_data['input_kraus'], serialize_kraus(kraus))
    assert raw_check['passed'], raw_check
    rewrite = {'method': manifest['rewrite'], 'initial': metrics(A0, 1e-12, settings),
               'final': metrics(final, 1e-12, settings), 'verification': check,
               'raw_input_verification': raw_check, 'termination': term, 'rewrite_time_s': rewrite_s}
    dump(CASE / 'rewrite.json', rewrite)
    np.savez_compressed(CASE / 'rewrite.npz', A_initial=A0, A_final=final, A_replayed=replay, labels=labels)
    dump(CASE / 'rewrite_steps.json', [{'pair': [i,j], 'U_real': U.real.tolist(), 'U_imag': U.imag.tolist()}
                                      for i,j,U in state.steps])
    progress('assembly', rewrite_time_s=rewrite_s, termination=term['stop_reason'])
    timer = StageTimers()
    start = time.perf_counter()
    with timer.installed():
        backend = lambda ms: BlockEncoding(ms, placement_options={}, matrix_order_options={})
        circuit, registers = rq.channel_to_LCU(channel_ensemble([kraus]), structure='tree',
                       opt='Matrix-order', fstar=True, block_encoding_cls=backend)
        assembly_s = time.perf_counter() - start
        progress('lowering', assembly_s=assembly_s)
        with timer.stage('final_nam_lowering'):
            lowered = transpile_channel_circuit(circuit, basis_gates=['rz','h','x','cx'], optimization_level=0,
                                               seed_transpiler=DEFAULT_TRANSPILE_SEED)
    circuit_s = time.perf_counter() - start
    counts = {str(k): int(v) for k,v in lowered.count_ops().items()}
    assert not set(counts) - {'rz','h','x','cx'}
    lam = float(circuit.metadata['subnormalization_factor'])
    assert abs(lam-rewrite['final']['static_cost']['Lambda']) <= 1e-10*max(lam,1)
    for key, value in [('fstar',True),('block_opt','Matrix-order'),('controlled_prepare',False)]:
        assert circuit.metadata[key] == value
    result = {'case':'TFIM2-HO', 'status':'compiled_pending_validation', 'G':sum(counts.values()),
              'W':lowered.num_qubits, 'Lambda':lam, 'Lambda_G':lam*sum(counts.values()), 'counts':counts,
              'compile_time_s':canonical_s+rewrite_s+circuit_s,
              'canonicalization_s':canonical_s, 'rewrite_time_s':rewrite_s,
              'circuit_time_s':circuit_s, 'assembly_s':assembly_s,
              'exclusive_stage_times_s':dict(timer.exclusive), 'jit_warmup_s':jit_s,
              'timing_scope':'canonicalization + rewrite + assembly/lowering; excludes JIT, verification, export, input/imports',
              'input_sha256':manifest['input_sha256'], 'rewrite':rewrite,
              'metadata':circuit.metadata, 'compiled_kraus':serialize_kraus(kraus), 'sources':source_identity()}
    dump(CASE / 'compiled.json', result)
    progress('export', G=result['G'], W=result['W'])
    qasm_path = CASE / 'channel_nam.qasm'
    qasm2.dump(lowered, qasm_path)
    recount = count_qasm_file(qasm_path, gate_set='nam')
    assert recount['metric_total'] == result['G']
    result['qasm_sha256'] = digest(qasm_path)
    result['non_clifford'] = recount['clifford']['non_clifford']
    progress('verification')
    vstart = time.perf_counter()
    if lowered.num_qubits <= 18:
        result['full_choi'] = verify_channel(lowered, registers, raw)
        assert result['full_choi']['passed'], result['full_choi']
    else:
        result['full_choi'] = {'status':'skipped', 'reason':'total qubits >18; rewrite Gram and step replay passed'}
    result['verification_time_s'] = time.perf_counter()-vstart
    result['status'] = 'ok' if term['stop_reason']=='frontier_exhausted' else 'rewrite_incomplete'
    dump(CASE / 'results.json', result)
    progress(result['status'], G=result['G'], W=result['W'])

def coordinator():
    import psutil
    if (OUT / 'status.json').exists():
        raise RuntimeError('Refusing to restart an existing run.')
    started = time.monotonic()
    peak, reason, stage, stage_since = 0, None, 'starting', started
    with (CASE / 'run.log').open('w') as log:
        child = subprocess.Popen([str(PYTHON), '-B', str(Path(__file__)), '--worker'],
                                 stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        while child.poll() is None:
            p = CASE / 'progress.json'
            progress_data = json.loads(p.read_text()) if p.exists() else {}
            current = progress_data.get('stage',stage)
            if current != stage:
                stage, stage_since = current, time.monotonic()
            try:
                process = psutil.Process(child.pid)
                rss = sum(p.memory_info().rss for p in [process]+process.children(recursive=True))
                peak = max(peak,rss)
                if rss > 6*1024**3: reason='memory_limit'
            except psutil.Error:
                pass
            if time.monotonic()-started > 3600: reason='wall_timeout'
            if stage=='verification' and time.monotonic()-stage_since >300: reason='verification_timeout'
            status = dict(status='running', worker_pid=child.pid, coordinator_pid=os.getpid(),
                          updated_at_utc=now(), stage=stage, elapsed_s=time.monotonic()-started,
                          peak_sampled_rss_bytes=peak, resource_stop=reason)
            dump(OUT / 'status.json',status)
            if reason:
                import signal
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(1)
        code=child.wait()
    path=CASE/'results.json'
    result=json.loads(path.read_text()) if path.exists() else {}
    terminal=reason or (result.get('status','missing_results') if code==0 else 'failed')
    status.update(status=terminal, exit_code=code, updated_at_utc=now(),
                  elapsed_s=time.monotonic()-started, resource_stop=reason)
    dump(OUT/'status.json',status)
    dump(CASE/'resources.json',status)

if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['MPLCONFIGDIR']=str(OUT/'.runtime/matplotlib')
    if '--prepare' in sys.argv:
        prepare()
    elif '--worker' in sys.argv:
        try: worker()
        except Exception:
            dump(CASE/'failure.json',{'traceback':traceback.format_exc(),'time':now()})
            raise
    else:
        coordinator()
