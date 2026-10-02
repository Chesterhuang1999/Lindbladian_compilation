"""Pauli rewrite-search HRW compilation, compared with existing raw-Kraus TUD."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_remote_hrw import (THREAD_KEYS, Stages, monitor, save, sha, write_qpy,
                            np, qiskit, scipy, psutil, threadpoolctl)
import argparse
from datetime import datetime, timezone
import json
import os
import platform
import traceback
import yaml
from hrw_rewrite_adapter import (REWRITE_PARAMETERS, input_operators, rewrite_operators,
                                 freeze_rewritten, audit_rewrite)
from remote_channel_adapter import compile_branch, assemble_opt, zero_input_channel_trace
from tfim_tud_reference import lower_to_u_cx, native_metrics

SIZES = (4, 8, 12, 20, 28)


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
        raw_operators = input_operators(data)
        rewritten, search = stages.run('rewrite_search_and_apply', lambda: rewrite_operators(raw_operators))
        branches = stages.run('rewritten_support_extraction', lambda: freeze_rewritten(rewritten, n))
        audit, before, after = audit_rewrite(data, branches, search)
        result['rewrite'] = audit
        stages.run('save_rewrite_audit', lambda: save(directory / 'rewrite.json', audit), 'serialization')
        stages.run('save_rewritten_input', lambda: save(directory / 'rewritten_input.json',
            {'n': n, 'branches': branches, 'origin': 'rewrite_search of the frozen original HRW family'}), 'serialization')
        stages.run('save_rewrite_coefficients', lambda: np.savez(directory / 'rewrite_coefficients.npz',
            before=before, after=after, labels=np.asarray(search['labels'])), 'serialization')
        save(directory / 'result.json', result)
        blocks = []
        for j, branch in enumerate(branches):
            prefix = f'kraus_{j:02d}_'
            block = compile_branch(branch, n, 'pauli_placement',
                lambda name, function: stages.run(prefix + name, function))
            blocks.append(block)
            row = {'index': j, 'canonical_name': branch['canonical_name'], 'beta': block.beta,
                'metadata': block.metadata, 'native0': native_metrics(block.circuit),
                'stage_times_seconds': {k[len(prefix):]: v for k, v in stages.times.items() if k.startswith(prefix)}}
            row['compile_to_native0_seconds'] = sum(row['stage_times_seconds'].values())
            result['branches'].append(row)
            save(directory / 'result.json', result)
        result['lambda'] = sum(block.beta**2 for block in blocks)
        raw, layout = stages.run('outer_prepare_tree_control_assembly', lambda: assemble_opt(blocks, n))
        result['channel_layout'] = layout
        assert np.isclose(result['lambda'], layout['lambda'], atol=1e-12)
        save(directory / 'result.json', result)
        circuit = stages.run('full_channel_native0_lowering', lambda: lower_to_u_cx(raw, optimization_level=0))
        count = native_metrics(circuit)
        trace = zero_input_channel_trace(branches, n)
        assert np.isclose(trace, 1., atol=1e-12)
        result.update(native0=count, compile_to_native0_seconds=stages.totals['compile'],
            rewrite_seconds=stages.times['rewrite_search_and_apply'],
            post_rewrite_compile_seconds=stages.totals['compile'] - stages.times['rewrite_search_and_apply'],
            status='COMPILED_VALIDATION_SKIPPED_BY_USER',
            costs={'lambda_weighted_gates': layout['lambda'] * count['gates_total'],
                   'zero_input_channel_trace': trace,
                   'zero_input_success_probability': trace / layout['lambda'],
                   'expected_zero_input_gates': layout['lambda'] * count['gates_total'] / trace,
                   'strict_expectation_formula': 'HRW is trace preserving: Lambda*G; excludes input preparation and measurement'})
        save(directory / 'result.json', result)
        result['circuit_record'] = stages.run('save_full_channel_native0',
            lambda: write_qpy(directory / 'channel_native0.qpy', circuit), 'serialization')
    except Exception:
        result.update(status='EXCEPTION', traceback=traceback.format_exc())
    result.update(stage_times_seconds=stages.times, category_times_seconds=stages.totals)
    save(directory / 'result.json', result)
    return result['status'] == 'COMPILED_VALIDATION_SKIPPED_BY_USER'


def run(out, budget, memory_gib):
    assert platform.python_version() == '3.12.9'
    assert (qiskit.__version__, np.__version__, scipy.__version__) == ('2.1.1', '2.4.2', '1.17.1')
    manifest = json.loads((ROOT / 'remote_source_manifest.json').read_text())
    for row in manifest['files']:
        assert sha(ROOT / row['relative_path']) == row['sha256'], row['relative_path']
    plan = yaml.safe_load((HERE / 'experiment_plan.lock.yaml').read_text())
    out.mkdir(parents=True, exist_ok=False)
    (out / 'jobs').mkdir()
    save(out / 'source_manifest.json', manifest)
    save(out / 'environment.json', {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'host': platform.node(), 'python': sys.version, 'executable': sys.executable,
        'versions': {'qiskit': qiskit.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__},
        'threads': {k: os.environ[k] for k in THREAD_KEYS},
        'threadpools': threadpoolctl.threadpool_info(), 'affinity': psutil.Process().cpu_affinity()})
    baseline = json.loads((ROOT / 'baseline/hrw_raw_results.json').read_text())
    save(out / 'raw_baseline_results.json', baseline)
    save(out / 'trial.lock.json', {'sizes': SIZES, 'method': 'pauli_rewrite',
        'comparison': 'Pauli basis + rewrite_search versus raw Kraus + TUD synthesis',
        'rewrite_enabled': True, 'rewrite_parameters': REWRITE_PARAMETERS,
        'structure': 'opt', 'branch_lowering_before_outer_assembly': True,
        'normalization': 'each route native Lambda=sum_j beta_j^2, using its actual compiled Kraus family',
        'local_support_policy': 'independent compilation on each rewritten Kraus support, then embed',
        'raw_tud_reused': True, 'selected_kraus_comparison': 'not defined across a channel-level Kraus rewrite',
        'native_basis': ['u', 'cx'], 'optimization_level': 0, 'seed': 646,
        'approximation_degree': 1., 'qubits_initially_zero': False, 'repetitions': 1,
        'compile_budget_seconds': budget, 'worker_tree_rss_gib': memory_gib,
        'timing_scope': 'rewrite search including copy/apply, support extraction, all block compilation/lowering/embedding, outer opt tree (including legacy opt1 counting), final native0 lowering; excludes input generation, imports, metric counting, serialization, validation',
        'large_probes': 'skipped_by_user_request; coefficient Gram equality checked for every size',
        'placement_environment': plan['freeze']['placement_environment']})
    results = []
    for n in SIZES:
        source = f'remote_inputs/HRW{n}.json'
        source_hash = sha(ROOT / source)
        assert {r['input_sha256'] for r in baseline if r['n'] == n} == {source_hash}
        directory = out / 'jobs' / f'HRW{n}_pauli_rewrite'
        directory.mkdir()
        save(directory / 'job.json', {'n': n, 'case_id': f'HRW{n}', 'method': 'pauli_rewrite',
            'input_relative_path': source, 'input_sha256': source_hash,
            'placement_environment': plan['freeze']['placement_environment']})
        print('START ' + directory.name, flush=True)
        result = monitor(directory / 'job.json', budget, memory_gib, Path(__file__).resolve())
        results.append(result)
        save(out / 'results.json', results)
        print(json.dumps({k: result.get(k) for k in ('n', 'status', 'native0', 'lambda',
            'rewrite_seconds', 'compile_to_native0_seconds', 'costs')}), flush=True)
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
    parser.add_argument('--compile-budget-seconds', type=float, default=3600)
    parser.add_argument('--worker-rss-gib', type=float, default=64)
    args = parser.parse_args()
    ok = worker(args.path) if args.command == '_worker' else run(
        args.path.resolve(), args.compile_budget_seconds, args.worker_rss_gib)
    sys.exit(0 if ok else 1)
