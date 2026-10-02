"""Run the frozen cycle-4, q=4 coloring comparison, stopping at native0."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from run_remote_hrw import (METHODS, THREAD_KEYS, Stages, monitor, save, sha,
                            write_qpy, np, qiskit, scipy, psutil, threadpoolctl)
import argparse
from datetime import datetime, timezone
import json
import os
import platform
import traceback
import yaml
from remote_channel_adapter import compile_branch, assemble_opt, zero_input_channel_trace
from tfim_tud_reference import lower_to_u_cx, native_metrics

CASE = 'Coloring_cycle4_q4'


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
        assert n == 8 and len(data['branches']) == 16
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
                trace = zero_input_channel_trace([branch], n)
                weighted = block.beta**2 * row['native0']['gates_total']
                result['selected_kraus'] = {
                    'canonical_name': branch['canonical_name'], 'beta': block.beta,
                    'native0': row['native0'],
                    'compile_to_native0_seconds': sum(v for k, v in row['stage_times_seconds'].items()
                                                      if k != 'tree_register_adaptation'),
                    'beta_squared_weighted_gates': weighted, 'trace_on_zero_input': trace,
                    'expected_zero_input_gates': weighted / trace if trace > 0 else None}
                result['selected_kraus']['circuit_record'] = stages.run(
                    'save_selected_kraus_native0',
                    lambda: write_qpy(directory / 'selected_kraus_native0.qpy', block.circuit), 'serialization')
            result['stage_times_seconds'] = stages.times.copy()
            save(directory / 'result.json', result)
        result['lambda'] = sum(block.beta**2 for block in blocks)
        result['all_kraus_compile_to_native0_seconds'] = stages.totals['compile']
        save(directory / 'result.json', result)
        raw, layout = stages.run('outer_prepare_tree_control_assembly', lambda: assemble_opt(blocks, n))
        result['channel_layout'] = layout
        result['lambda'] = layout['lambda']
        save(directory / 'result.json', result)
        circuit = stages.run('full_channel_native0_lowering', lambda: lower_to_u_cx(raw, optimization_level=0))
        count = native_metrics(circuit)
        trace = zero_input_channel_trace(data['branches'], n)
        result.update(native0=count, compile_to_native0_seconds=stages.totals['compile'],
            status='COMPILED_VALIDATION_SKIPPED_BY_USER',
            input_validation=data['input_validation'],
            costs={'lambda_weighted_gates': layout['lambda'] * count['gates_total'],
                   'zero_input_channel_trace': trace,
                   'zero_input_success_probability': trace / layout['lambda'],
                   'expected_zero_input_gates': layout['lambda'] * count['gates_total'] / trace,
                   'strict_expectation_formula': 'Lambda*G/Tr(Phi(rho)); independent repeated preparation; preparation/measurement costs excluded'})
        save(directory / 'result.json', result)
        result['circuit_record'] = stages.run('save_full_channel_native0',
            lambda: write_qpy(directory / 'channel_native0.qpy', circuit), 'serialization')
    except Exception:
        result.update(status='EXCEPTION', traceback=traceback.format_exc())
    result.update(stage_times_seconds=stages.times, category_times_seconds=stages.totals)
    save(directory / 'result.json', result)
    return result['status'] == 'COMPILED_VALIDATION_SKIPPED_BY_USER'


def run(out, compile_budget, memory_gib):
    assert platform.python_version() == '3.12.9'
    assert (qiskit.__version__, np.__version__, scipy.__version__) == ('2.1.1', '2.4.2', '1.17.1')
    manifest = json.loads((ROOT / 'remote_source_manifest.json').read_text())
    for row in manifest['files']:
        assert sha(ROOT / row['relative_path']) == row['sha256'], row['relative_path']
    input_relative = f'remote_inputs/{CASE}.json'
    source_hash = sha(ROOT / input_relative)
    plan = yaml.safe_load((HERE / 'experiment_plan.lock.yaml').read_text())
    out.mkdir(parents=True, exist_ok=False)
    (out / 'jobs').mkdir()
    save(out / 'source_manifest.json', manifest)
    save(out / 'input.json', json.loads((ROOT / input_relative).read_text()))
    save(out / 'environment.json', {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'host': platform.node(), 'python': sys.version, 'executable': sys.executable,
        'versions': {'qiskit': qiskit.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__},
        'threads': {k: os.environ[k] for k in THREAD_KEYS},
        'threadpools': threadpoolctl.threadpool_info(), 'affinity': psutil.Process().cpu_affinity()})
    save(out / 'trial.lock.json', {'case': CASE, 'system_qubits': 8, 'vertices': 4, 'colors': 4,
        'support_per_kraus': 6, 'kraus_count': 16, 'methods': METHODS, 'beta_mode': 'native_only',
        'structure': 'opt', 'branch_lowering_before_outer_assembly': True,
        'selected_kraus': 'K_0_0', 'rewrite_enabled': False, 'repetitions': 1,
        'large_probes': 'skipped_by_user_request', 'compile_budget_seconds': compile_budget,
        'worker_tree_rss_gib': memory_gib, 'local_kraus_policy': 'independent local support compilation per branch and route',
        'lambda_definition': 'sum_j(actual_branch_beta_j**2)', 'native_basis': ['u', 'cx'],
        'optimization_level': 0, 'seed': 646, 'approximation_degree': 1., 'qubits_initially_zero': False,
        'timing_scope': 'block construction, local lowering, embedding, outer opt tree (including legacy opt1 counting), final native0 lowering; excludes imports, metric counting, serialization, validation',
        'placement_environment': plan['freeze']['placement_environment']})
    results = []
    for method in METHODS:
        directory = out / 'jobs' / f'{CASE}_{method}'
        directory.mkdir()
        save(directory / 'job.json', {'n': 8, 'vertices': 4, 'colors': 4, 'case_id': CASE,
            'method': method, 'input_relative_path': input_relative, 'input_sha256': source_hash,
            'placement_environment': plan['freeze']['placement_environment']})
        print('START ' + directory.name, flush=True)
        result = monitor(directory / 'job.json', compile_budget, memory_gib, Path(__file__).resolve())
        results.append(result)
        save(out / 'results.json', results)
        print(json.dumps({k: result.get(k) for k in ('method', 'status', 'native0', 'lambda',
                                                   'compile_to_native0_seconds', 'costs')}), flush=True)
    for row in manifest['files']:
        assert sha(ROOT / row['relative_path']) == row['sha256']
    save(out / 'summary.json', {'jobs': len(results), 'compiled_jobs': sum(bool(r.get('native0')) for r in results),
                              'statuses': [r['status'] for r in results]})
    (out / 'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n'
        for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'checksums.sha256'))
    return all(r['status'] == 'COMPILED_VALIDATION_SKIPPED_BY_USER' for r in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', '_worker'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--compile-budget-seconds', type=float, default=7200)
    parser.add_argument('--worker-rss-gib', type=float, default=64)
    args = parser.parse_args()
    ok = worker(args.path) if args.command == '_worker' else run(
        args.path.resolve(), args.compile_budget_seconds, args.worker_rss_gib)
    sys.exit(0 if ok else 1)
