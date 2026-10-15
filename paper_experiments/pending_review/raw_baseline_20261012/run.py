"""Isolated raw-NAM inventory and bounded generation; existing outputs stay intact."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = Path('/Users/chester_hqf/miniconda3/envs/qiskit_simulate/bin/python')
MEMORY_BYTES = 6 * 1024**3
TIMEOUT_S = 600


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    if (HERE / 'manifest.json').exists():
        raise RuntimeError('Inventory already exists; do not replace an experiment.')
    import numpy as np
    sys.path.insert(0, str(ROOT / 'src'))
    from rq1_pipeline import canonicalize, deserialize_kraus, input_identity
    inputs = HERE / 'inputs'
    inputs.mkdir(exist_ok=True)
    previous = HERE.parent
    entries = []
    historical = {'TFIM-4': 116187, 'TFIM-8': 777722, 'TFIM-12': 1900972,
                  'HRW-4': 33293, 'HRW-12': 254260, 'HRW-20': 612213,
                  'HRW-28': 857049, 'Toom-L3': 1339909, 'RepDQE-d3': 302786}
    for folder in ('tfim_hrw_full_20261011', 'remaining_full_20261011'):
        for source in sorted((previous / folder / 'cases').glob('*/input.json')):
            name = source.parent.name
            entries.append((name, json.loads(source.read_text()), source, 'selected'))
    source = previous / 'new_rewrite_incremental_20261011/Petz_Ising3/input.json'
    data = json.loads(source.read_text())
    entries.append(('Petz_Ising3', {'input_kraus': data['canonical_kraus'],
                    'model': data['parameters']}, source, 'rewrite_supplement'))
    for source in sorted((previous / 'support_gatecost_counterexamples_20261012/cases').glob('CPTP*/input.npz')):
        with np.load(source) as data:
            rows = [[[str(label), float(abs(v)), float((v/abs(v)).real),
                      float((v/abs(v)).imag)] for label, v in zip(data['labels'], row) if v != 0]
                    for row in data['A']]
        entries.append((source.parent.name, {'input_kraus': rows,
                       'model': {'family': 'random_Stinespring_isometry', 'system_qubits': 2}},
                       source, 'rewrite_supplement'))
    records = []
    for name, data, source, group in entries:
        can, info = canonicalize(deserialize_kraus(data['input_kraus']))
        lam = float(sum(sum(abs(c * atom.phase) for atom, c in row.instances)**2 for row in can))
        input_file = inputs / (name + '.json')
        dump(input_file, data)
        old_result = previous / 'rq1_paper_suite_20261010/full_v2/runs' / name / 'raw/results.json'
        reuse = None
        if old_result.exists():
            old = json.loads(old_result.read_text())
            qasm = Path(old['qasm'])
            settings = old['metrics']['settings']
            if (input_identity(can) == old['canonicalization']['canonical_input_sha256']
                    and qasm.exists() and digest(qasm) == old['qasm_sha256']
                    and settings['basis'] == 'nam' and settings['lowering_optimization_level'] == 0
                    and settings['qubits_initially_zero'] is False
                    and old['verification'].get('passed', True)
                    and abs(lam - old['Lambda']) < 1e-10 * max(1, lam)):
                reuse = {'result': str(old_result), 'qasm': str(qasm), 'G': old['metrics']['G'],
                         'Lambda': old['Lambda'], 'qasm_sha256': old['qasm_sha256'],
                         'canonical_input_sha256': info['canonical_input_sha256']}
        record = {'case': name, 'group': group, 'optional': name in ('Toom-L6', 'Toom-L8'),
                  'source': str(source), 'source_sha256': digest(source), 'input': str(input_file),
                  'input_sha256': digest(input_file), 'canonicalization': info,
                  'system_qubits': can[0].size, 'kraus_count': len(can),
                  'pauli_terms': sum(len(k.instances) for k in can), 'Lambda_raw': lam,
                  'historical_paper_G': historical.get(name), 'reuse': reuse}
        records.append(record)
        print(json.dumps({k: record[k] for k in ('case', 'Lambda_raw', 'historical_paper_G')}
                         | {'reuse': bool(reuse)}), flush=True)
    snapshot = HERE / 'source_snapshot'
    for folder in ('src', 'scripts', 'scripts/eval_redesign', 'Baseline_scripts'):
        for source in (ROOT / folder).glob('*.py'):
            dest = snapshot / source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    dump(HERE / 'manifest.json', {
        'scope': '47 selected instances plus 17 recent rewrite supplements; retired Table13 sizes excluded',
        'raw': {'structure': 'basic', 'inner': 'No', 'rewrite': False, 'fstar': False,
                'basis': 'nam', 'optimization_level': 0, 'qubits_initially_zero': False,
                'canonicalization': 'same A_can as Ours, exact zero removal, proportional merge tol 1e-12'},
        'historical_reuse_note': 'Paper optimizer counts only pair with their own historical input QASM, not current raw QASM.',
        'per_job_memory_bytes': MEMORY_BYTES, 'per_job_timeout_s': TIMEOUT_S,
        'optimizer_stage': {'tools': ['voqc', 'feynman'], 'timeout_s': 3600,
                            'order': 'after raw generation, ascending raw gate count',
                            'basis': 'NAM; each tool independently receives the raw QASM'},
        'sources': {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')},
        'cases': records})


def execute():
    import psutil
    manifest = json.loads((HERE / 'manifest.json').read_text())
    if (HERE / 'status.json').exists():
        raise RuntimeError('Queue already started; do not duplicate it.')
    records = []
    pending = []
    for entry in manifest['cases']:
        if entry['reuse']:
            records.append({'case': entry['case'], 'status': 'reused', **entry['reuse']})
        else:
            pending.append(entry)
    # Small physical fixtures first, then synthetic, random, and scaling inputs.
    priority = {'Petz_TFIM2': 0, 'Collective4': 1, 'Local_QSW3': 2, 'Thermal2_HO': 3,
                'Petz_Ising3': 4, 'TFIM2-HO': 6, 'TFIM3_HO': 7}
    pending.sort(key=lambda r: (priority.get(r['case'], 5 if r['group']=='rewrite_supplement' else 8),
                                r['system_qubits'], r['pauli_terms'], r['case']))
    state = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(),
             'queue': [r['case'] for r in pending], 'completed': records}
    dump(HERE / 'status.json', state)
    for entry in pending:
        name = entry['case']
        dest = HERE / 'cases' / name
        dest.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), '-B', str(HERE / 'source_snapshot/scripts/eval_redesign/run_rq1_pipeline.py'),
                   '--pipeline', 'raw', '--input-json', entry['input'], '--out-dir', str(dest), '--verify-auto']
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', QISKIT_NUM_PROCS='1',
                   MPLCONFIGDIR=str(HERE / '.runtime/matplotlib'), NUMBA_CACHE_DIR=str(HERE / '.runtime/numba'))
        start = time.monotonic()
        peak, reason = 0, None
        with (dest / 'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            state['current'] = {'case': name, 'pid': process.pid, 'command': command}
            dump(HERE / 'status.json', state)
            while process.poll() is None:
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                except (psutil.Error, OSError) as exc:
                    reason = 'resource_monitor_failed'
                    state['current']['monitor_error'] = str(exc)
                if peak > MEMORY_BYTES:
                    reason = 'memory_limit'
                elif time.monotonic() - start > TIMEOUT_S:
                    reason = 'timeout'
                if reason:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(1)
            code = process.wait()
        result_path = dest / name / 'raw/results.json'
        record = {'case': name, 'status': reason or ('ok' if code == 0 else 'failed'),
                  'exit_code': code, 'elapsed_s': time.monotonic()-start, 'peak_rss_bytes': peak,
                  'Lambda_raw': entry['Lambda_raw'], 'result': str(result_path) if result_path.exists() else None}
        if record['status'] == 'ok':
            if not result_path.exists():
                record['status'] = 'missing_result'
            else:
                data = json.loads(result_path.read_text())
                valid = (data['canonicalization']['canonical_input_sha256'] == entry['canonicalization']['canonical_input_sha256']
                         and data['verification'].get('passed', True)
                         and abs(data['Lambda'] - entry['Lambda_raw']) <= 1e-10 * max(1, entry['Lambda_raw']))
                if not valid:
                    record['status'] = 'validation_failed'
                record.update(G=data['metrics']['G'], Lambda=data['Lambda'], qasm=data['qasm'],
                              qasm_sha256=data['qasm_sha256'], compile_time_s=data['compile_time_s'],
                              verification=data['verification'])
        dump(dest / 'resources.json', record)
        records.append(record)
        dump(HERE / 'results.json', records)
        state['completed'] = records
        state['current'] = None
        dump(HERE / 'status.json', state)
        print(json.dumps(record), flush=True)
    state.update(status='raw_finished', raw_finished_at=time.time())
    dump(HERE / 'status.json', state)
    command = [str(PYTHON), '-B', str(HERE / 'optimize.py'), 'queue']
    code = subprocess.call(command, cwd=ROOT)
    state.update(status='finished' if code == 0 else 'optimizer_stage_failed',
                 finished_at=time.time(), optimizer_exit_code=code)
    dump(HERE / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    args = parser.parse_args()
    (HERE / '.runtime').mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.runtime/matplotlib'))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.runtime/numba'))
    prepare() if args.mode == 'prepare' else execute()
