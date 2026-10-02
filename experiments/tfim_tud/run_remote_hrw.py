"""Native-beta full HRW channel compilation through the existing opt tree."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
               'QISKIT_NUM_PROCS', 'RAYON_NUM_THREADS', 'OMP_THREAD_LIMIT')
os.environ.update({key: '1' for key in THREAD_KEYS})
os.environ.update(QISKIT_PARALLEL='FALSE', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True

import numpy as np
import psutil
import qiskit
from qiskit import qpy
import scipy
import threadpoolctl
import yaml

from remote_channel_adapter import compile_branch, assemble_opt, zero_input_channel_trace
from tfim_tud_reference import lower_to_u_cx, native_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ('pauli_placement', 'tud_factored_diagonal', 'tud_separate')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                   default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()) + '\n')
    temporary.replace(path)


class Stages:
    def __init__(self, directory):
        self.directory, self.times = directory, {}
        self.totals = {'compile': 0., 'serialization': 0.}

    def run(self, name, function, category='compile'):
        started = time.perf_counter()
        state = {'stage': name, 'category': category, 'started': started,
                 'completed_seconds': self.totals.copy()}
        save(self.directory / 'progress.json', state)
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(state, event='start')) + '\n')
        result = function()
        elapsed = time.perf_counter() - started
        self.times[name] = elapsed
        self.totals[category] += elapsed
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'event': 'end', 'stage': name, 'category': category,
                                     'seconds': elapsed}) + '\n')
        save(self.directory / 'progress.json', {'stage': 'between_stages', 'category': None,
                                               'started': time.perf_counter(),
                                               'completed_seconds': self.totals.copy()})
        return result


def write_qpy(path, circuit):
    with path.open('wb') as stream:
        qpy.dump(circuit, stream)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def worker(path):
    job = json.loads(path.read_text())
    directory = path.parent
    os.environ.update(job['placement_environment'])
    stages = Stages(directory)
    result = dict(job, status='COMPILING', branches=[], native0=None,
                  opt3_executed=False, probes_executed=False)
    save(directory / 'result.json', result)
    try:
        source = ROOT / job['input_relative_path']
        assert sha(source) == job['input_sha256']
        data = json.loads(source.read_text())
        n = data['n']
        assert n == job['n'] and len(data['branches']) == 2 * n
        blocks = []
        for j, branch in enumerate(data['branches']):
            prefix = f'kraus_{j:02d}_'
            block = compile_branch(branch, n, job['method'],
                lambda name, function: stages.run(prefix + name, function))
            blocks.append(block)
            row = {'index': j, 'beta': block.beta, 'metadata': block.metadata,
                   'native0': native_metrics(block.circuit),
                   'stage_times_seconds': {k[len(prefix):]: v for k, v in stages.times.items() if k.startswith(prefix)}}
            row['compile_to_native0_seconds'] = sum(row['stage_times_seconds'].values())
            result['branches'].append(row)
            if j == 0:
                selected_time = sum(v for k, v in row['stage_times_seconds'].items()
                                    if k != 'tree_register_adaptation')
                result['selected_kraus'] = {
                    'canonical_name': branch['canonical_name'], 'beta': block.beta,
                    'native0': row['native0'], 'compile_to_native0_seconds': selected_time,
                    'beta_squared_weighted_gates': block.beta**2 * row['native0']['gates_total'],
                    'trace_on_zero_input': 1.0 / n,
                    'expected_zero_input_gates': n * block.beta**2 * row['native0']['gates_total'],
                }
                result['selected_kraus']['circuit_record'] = stages.run(
                    'save_selected_kraus_native0',
                    lambda: write_qpy(directory / 'selected_kraus_native0.qpy', block.circuit),
                    'serialization')
            result['stage_times_seconds'] = stages.times.copy()
            save(directory / 'result.json', result)
        result['lambda'] = sum(block.beta**2 for block in blocks)
        expected_lambda = 4.0 if job['method'] == 'pauli_placement' else 2.0
        assert np.isclose(result['lambda'], expected_lambda, rtol=1e-12, atol=1e-12)
        result['all_kraus_compile_to_native0_seconds'] = stages.totals['compile']
        save(directory / 'result.json', result)
        raw, layout = stages.run('outer_prepare_tree_control_assembly', lambda: assemble_opt(blocks, n))
        result['channel_layout'] = layout
        result['lambda'] = layout['lambda']
        save(directory / 'result.json', result)
        circuit = stages.run('full_channel_native0_lowering', lambda: lower_to_u_cx(raw, optimization_level=0))
        count = native_metrics(circuit)
        denominator = zero_input_channel_trace(data['branches'], n)
        assert np.isclose(denominator, 1.0, rtol=1e-12, atol=1e-12)
        result.update(native0=count, compile_to_native0_seconds=stages.totals['compile'],
                      status='COMPILED_VALIDATION_SKIPPED_BY_USER',
                      costs={'lambda_weighted_gates': layout['lambda'] * count['gates_total'],
                             'zero_input_channel_trace': denominator,
                             'zero_input_success_probability': denominator / layout['lambda'],
                             'expected_zero_input_gates': layout['lambda'] * count['gates_total'] / denominator,
                             'strict_expectation_formula': 'Lambda*G/Tr(Phi(rho)); independent repeated preparation; preparation/measurement costs excluded'})
        save(directory / 'result.json', result)
        result['circuit_record'] = stages.run('save_full_channel_native0',
            lambda: write_qpy(directory / 'channel_native0.qpy', circuit), 'serialization')
    except Exception:
        result.update(status='EXCEPTION', traceback=traceback.format_exc())
    result.update(stage_times_seconds=stages.times, category_times_seconds=stages.totals)
    save(directory / 'result.json', result)
    return result['status'] == 'COMPILED_VALIDATION_SKIPPED_BY_USER'


def monitor(path, compile_budget, memory_gib, worker_script=None):
    directory = path.parent
    reason, peak = None, 0
    start = time.perf_counter()
    with (directory / 'worker.log').open('w') as stream:
        process = subprocess.Popen([sys.executable, str(worker_script or Path(__file__).resolve()), '_worker', str(path)],
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    peak = max(peak, sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True)))
                except psutil.NoSuchProcess:
                    pass
                progress_file = directory / 'progress.json'
                if progress_file.exists():
                    progress = json.loads(progress_file.read_text())
                    category = progress['category']
                    if category:
                        elapsed = progress['completed_seconds'][category] + time.perf_counter() - progress['started']
                        if elapsed > {'compile': compile_budget, 'serialization': 600}[category]:
                            reason = 'TIMEOUT_DURING_' + progress['stage']
                if peak > memory_gib * 1024**3:
                    reason = 'MEMORY_BUDGET_EXCEEDED'
                if time.perf_counter() - start > compile_budget + 900:
                    reason = 'TIMEOUT_DURING_WORKER_LIFETIME'
                if reason:
                    break
                time.sleep(.2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    result_path = directory / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'EXCEPTION'}
    if reason:
        result['status'] = reason
    elif process.returncode:
        result['status'] = 'EXCEPTION'
    result['monitor'] = {'returncode': process.returncode, 'observed_worker_tree_peak_rss_bytes': peak,
                         'worker_wall_seconds': time.perf_counter() - start, 'poll_interval_seconds': .2}
    if (directory / 'progress.json').exists():
        progress = json.loads((directory / 'progress.json').read_text())
        result['monitor']['final_progress'] = progress
        if progress['category']:
            result['monitor']['interrupted_stage_seconds'] = time.perf_counter() - progress['started']
    save(result_path, result)
    return result


def run(out, sizes, compile_budget, memory_gib):
    assert platform.python_version() == '3.12.9'
    assert (qiskit.__version__, np.__version__, scipy.__version__) == ('2.1.1', '2.4.2', '1.17.1')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'jobs').mkdir()
    plan = yaml.safe_load((HERE / 'experiment_plan.lock.yaml').read_text())
    deployed_manifest = json.loads((ROOT / 'remote_source_manifest.json').read_text())
    deployed_hashes = {r['relative_path']: r['sha256'] for r in deployed_manifest['files']}
    expected = {n: deployed_hashes[f'remote_inputs/HRW{n}.json'] for n in sizes}
    for n in sizes:
        assert sha(ROOT / f'remote_inputs/HRW{n}.json') == expected[n]
    source_paths = list((ROOT / 'src').glob('*.py')) + [HERE / name for name in
        ('run_remote_hrw.py', 'remote_channel_adapter.py',
         'adapter_channelir.py', 'tfim_tud_reference.py', 'experiment_plan.lock.yaml')]
    manifest = [{'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for path in sorted(source_paths)]
    for row in manifest:
        assert deployed_hashes[row['path']] == row['sha256'], row['path']
    save(out / 'source_manifest.json', manifest)
    save(out / 'input_manifest.json', [{'n': n, 'relative_path': f'remote_inputs/HRW{n}.json',
                                      'sha256': expected[n]} for n in sizes])
    save(out / 'environment.json', {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'host': platform.node(), 'platform': platform.platform(), 'python': sys.version,
        'executable': sys.executable, 'versions': {name: importlib.metadata.version(name) for name in
            ('qiskit', 'numpy', 'scipy', 'psutil', 'qutip', 'sympy', 'PyYAML', 'qsppack', 'pyqsp')},
        'threads': {key: os.environ[key] for key in THREAD_KEYS},
        'threadpools': threadpoolctl.threadpool_info(), 'affinity': psutil.Process().cpu_affinity()})
    save(out / 'trial.lock.json', {'sizes': sizes, 'methods': METHODS, 'beta_mode': 'native_only',
        'method_display_names': {'pauli_placement': 'Pauli Matrix-order (fixed residual backend)',
                                 'tud_factored_diagonal': 'TUD factored', 'tud_separate': 'TUD separate'},
        'structure': 'opt', 'branch_lowering_before_outer_assembly': True, 'optimization_levels': [0],
        'selected_kraus': 'K_0_0 = T_0_0^dagger / sqrt(2n)',
        'rewrite_enabled': False, 'kraus_family': 'fixed original Sec.5.2 family, 2n branches',
        'repetitions': 1, 'large_probes': 'skipped_by_user_request',
        'compile_budget_seconds': compile_budget, 'worker_tree_rss_gib': memory_gib,
        'outer_assembly': 'prep_sup_state and mulplex_U_tree from existing src/subroutine.py',
        'local_kraus_policy': 'independent one-qubit support compilation for every Kraus and every route',
        'pauli_inner_opt_policy': 'All Kraus Matrix-order using src/block_encoding_new_fixed.py and src/mobius_residual_fixed.py; legacy src/block_encoding_new.py remains unchanged',
        'lambda_definition': 'sum_j(actual_branch_beta_j**2)',
        'timing_scope': 'all individual block builds/lowering/register adaptation plus outer prepare/tree/controlled assembly including its legacy opt1 count work and final native0 lowering; excludes imports, counting, serialization, verification',
        'native_basis': ['u', 'cx'], 'seed': 646, 'approximation_degree': 1.,
        'qubits_initially_zero': False, 'placement_environment': plan['freeze']['placement_environment']})
    results = []
    for n in sizes:
        for method in METHODS:
            directory = out / 'jobs' / f'HRW{n}_{method}'
            directory.mkdir()
            job = {'n': n, 'case_id': f'HRW{n}_full_channel', 'method': method,
                   'input_relative_path': f'remote_inputs/HRW{n}.json',
                   'input_sha256': expected[n], 'placement_environment': plan['freeze']['placement_environment']}
            save(directory / 'job.json', job)
            print('START ' + directory.name, flush=True)
            result = monitor(directory / 'job.json', compile_budget, memory_gib)
            results.append(result)
            save(out / 'results.json', results)
            print(json.dumps({k: result.get(k) for k in ('n', 'method', 'status', 'native0', 'lambda',
                                                      'compile_to_native0_seconds', 'costs')}), flush=True)
    for row in manifest:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    save(out / 'summary.json', {'jobs': len(results), 'compiled_jobs': sum(bool(r.get('native0')) for r in results),
                              'opt3_executed': False, 'probes_executed': False,
                              'statuses': [r['status'] for r in results]})
    (out / 'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n'
        for p in sorted(out.rglob('*')) if p.is_file()))
    return all(r['status'] == 'COMPILED_VALIDATION_SKIPPED_BY_USER' for r in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', '_worker'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, choices=[4, 8, 12, 20, 28], default=[4, 8, 12, 20, 28])
    parser.add_argument('--compile-budget-seconds', type=float, default=3600)
    parser.add_argument('--worker-rss-gib', type=float, default=64)
    args = parser.parse_args()
    ok = worker(args.path) if args.command == '_worker' else run(
        args.path.resolve(), args.sizes, args.compile_budget_seconds, args.worker_rss_gib)
    sys.exit(0 if ok else 1)
