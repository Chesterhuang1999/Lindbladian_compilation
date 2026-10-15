"""Bounded remote retries of saved raw inputs; never overwrite earlier runs."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    mismatches = [name for name, expected in manifest['files'].items()
                  if not (HERE / name).is_file() or digest(HERE / name) != expected]
    versions = {name: importlib.metadata.version(name) for name in manifest['versions']}
    result = {'passed': not mismatches and versions == manifest['versions'],
              'file_mismatches': mismatches, 'versions': versions,
              'verified_files': len(manifest['files']), 'python': sys.version,
              'timestamp': time.time()}
    dump(HERE / 'preflight.json', result)
    if not result['passed']:
        raise RuntimeError(result)
    return manifest


def worker(name):
    snapshot = HERE / 'source_snapshot'
    sys.path[:0] = [str(snapshot / f) for f in ('src', 'scripts', 'Baseline_scripts')]
    import rq1_pipeline
    import experiment_protocol
    import baseline_common
    from qiskit import qasm2
    destination = HERE / 'cases' / name
    stages = []

    def wrap(module, attribute, stage):
        original = getattr(module, attribute)

        def traced(*args, **kwargs):
            start = time.time()
            dump(destination / 'progress.json', {'stage': stage, 'stage_started_at': start,
                                                'completed_stages': stages})
            print(json.dumps({'stage': stage, 'event': 'start', 'time': start}), flush=True)
            value = original(*args, **kwargs)
            stages.append({'stage': stage, 'elapsed_s': time.time() - start})
            print(json.dumps({'stage': stage, 'event': 'done', **stages[-1]}), flush=True)
            return value

        setattr(module, attribute, traced)

    for module, attribute, stage in (
        (rq1_pipeline, 'canonicalize', 'canonicalization'),
        (rq1_pipeline, 'build_pipeline', 'raw_assembly'),
        (experiment_protocol, 'lower_and_count', 'nam_lowering'),
        (qasm2, 'dump', 'qasm_export'),
        (baseline_common, 'count_qasm_file', 'nam_recount'),
    ):
        wrap(module, attribute, stage)
    runner = snapshot / 'scripts/eval_redesign/run_rq1_pipeline.py'
    sys.argv = [str(runner), '--pipeline', 'raw', '--input-json',
                str(HERE / 'inputs' / (name + '.json')), '--out-dir', str(destination), '--verify-auto']
    runpy.run_path(str(runner), run_name='__main__')
    dump(destination / 'progress.json', {'stage': 'finished', 'completed_stages': stages})


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def queue():
    import psutil
    manifest = preflight()
    if (HERE / 'status.json').exists():
        raise RuntimeError('This queue has already started.')
    limits = manifest['limits']
    completed = []
    state = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(),
             'queue': [r['case'] for r in manifest['cases']], 'completed': completed, 'current': None}
    dump(HERE / 'status.json', state)
    for entry in manifest['cases']:
        name = entry['case']
        destination = HERE / 'cases' / name
        destination.mkdir(parents=True, exist_ok=True)
        record = {'case': name, 'Lambda_raw': entry['Lambda_raw']}
        parent = entry.get('requires')
        prior = next((r for r in completed if r['case'] == parent), None)
        if parent and (not prior or prior['status'] != 'ok'
                       or prior['peak_rss_bytes'] > limits['advance_peak_bytes']
                       or prior['wall_s'] > limits['advance_wall_s']):
            record.update(status='deferred', reason='predecessor exceeded advancement budget', predecessor=parent)
        elif psutil.virtual_memory().available < limits['minimum_available_bytes']:
            record.update(status='deferred', reason='insufficient available server memory')
        elif psutil.disk_usage(HERE).free < limits['minimum_free_disk_bytes']:
            record.update(status='deferred', reason='insufficient free disk')
        else:
            start = time.monotonic()
            peak = 0
            reason = None
            command = [sys.executable, '-B', str(Path(__file__)), 'worker', '--case', name]
            with (destination / 'run.log').open('w') as log:
                process = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                state['current'] = {'case': name, 'pid': process.pid, 'started_at': time.time()}
                dump(HERE / 'status.json', state)
                while process.poll() is None:
                    try:
                        root = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [root] + root.children(recursive=True)
                                  if p.is_running())
                        peak = max(peak, rss)
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.Error as error:
                        reason = 'resource_monitor_failed'
                        record['monitor_error'] = str(error)
                    elapsed = time.monotonic() - start
                    if peak > limits['memory_bytes']:
                        reason = 'memory_limit'
                    elif elapsed > limits['timeout_s']:
                        reason = 'timeout'
                    elif psutil.virtual_memory().available < limits['runtime_available_floor_bytes']:
                        reason = 'server_memory_pressure'
                    dump(destination / 'resources.json', {'status': 'running', 'pid': process.pid,
                                                         'wall_s': elapsed, 'peak_rss_bytes': peak})
                    if reason:
                        stop(process)
                        break
                    time.sleep(1)
                code = process.wait()
            record.update(status=reason or ('ok' if code == 0 else 'failed'), exit_code=code,
                          wall_s=time.monotonic() - start, peak_rss_bytes=peak)
            result = destination / name / 'raw/results.json'
            if record['status'] == 'ok':
                if not result.exists():
                    record['status'] = 'missing_result'
                else:
                    data = json.loads(result.read_text())
                    cfg = data['assembly']['configuration']
                    settings = data['metrics']['settings']
                    valid = (data['canonicalization']['canonical_input_sha256'] == entry['canonical_input_sha256']
                             and data['model']['input_json_sha256'] == entry['input_sha256']
                             and cfg == {'rewrite': False, 'inner': 'No', 'structure': 'basic', 'fstar': False}
                             and settings['qubits_initially_zero'] is False
                             and settings['lowering_optimization_level'] == 0
                             and data['assembly']['rewrite']['method'] == 'none'
                             and data['verification'].get('passed', True)
                             and set(data['metrics']['counts']) <= {'rz', 'h', 'x', 'cx', 'reset'}
                             and abs(data['Lambda'] - entry['Lambda_raw']) < 1e-10 * max(1, entry['Lambda_raw']))
                    record.update(status='ok' if valid else 'validation_failed', result=str(result),
                                  G=data['metrics']['G'], Lambda=data['Lambda'],
                                  compile_time_s=data['compile_time_s'], qasm=data['qasm'],
                                  qasm_sha256=data['qasm_sha256'], verification=data['verification'])
        dump(destination / 'resources.json', record)
        completed.append(record)
        state['current'] = None
        dump(HERE / 'results.json', completed)
        dump(HERE / 'status.json', state)
        print(json.dumps(record), flush=True)
    state.update(status='finished', finished_at=time.time())
    dump(HERE / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'queue', 'worker'))
    parser.add_argument('--case')
    args = parser.parse_args()
    runtime = HERE / '.runtime'
    runtime.mkdir(exist_ok=True)
    for key, value in {'TMPDIR': str(runtime), 'MPLCONFIGDIR': str(runtime / 'matplotlib'),
                       'NUMBA_CACHE_DIR': str(runtime / 'numba'), 'PYTHONDONTWRITEBYTECODE': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                       'QISKIT_NUM_PROCS': '1', 'PYTHONUNBUFFERED': '1'}.items():
        os.environ[key] = value
    if args.mode == 'worker':
        worker(args.case)
    elif args.mode == 'preflight':
        print(json.dumps(preflight()), flush=True)
    else:
        queue()
