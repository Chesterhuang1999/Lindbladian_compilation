"""Render the two maintained TFIM tables from an archived remote snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
METHODS = {'pauli_placement': 'Ours (Pauli Matrix-order)',
           'tud_separate': 'TUD separate'}
SUCCESS = {'COMPILED_NATIVE0_PROBES_SKIPPED', 'COMPILED_VALIDATION_SKIPPED_BY_USER'}


def status(job):
    value = job.get('status')
    if value in SUCCESS:
        return 'Compiled'
    return {'COMPILING': 'Running', 'MEMORY_BUDGET_EXCEEDED': 'Memory limit',
            'TIMEOUT': 'Time limit', None: 'Not started'}.get(value, value)


def html_row(n, method, values, memory_footnote=False):
    cells = []
    if method == next(iter(METHODS)):
        cells.append(f'<td rowspan="{len(METHODS)}" align="center" valign="middle">{n}</td>')
    label = escape(METHODS[method])
    if memory_footnote:
        label += '<sup><a href="#tfim12-memory-note">1</a></sup>'
    cells.append(f'<td>{label}</td>')
    cells.extend(f'<td align="right">{escape(value)}</td>' for value in values)
    return '<tr>' + ''.join(cells) + '</tr>'


def table_start():
    headings = ('n', 'Route', 'per-attempt gate count', 'expected gate count', 'Compile time (s)')
    return ['<table>', '<thead><tr>' + ''.join(f'<th>{h}</th>' for h in headings)
            + '</tr></thead>', '<tbody>']


def row(n, method, job, channel=False):
    if job.get('status') not in SUCCESS:
        memory_footnote = (not channel and n == 12 and method != 'pauli_placement'
                           and job.get('status') == 'MEMORY_BUDGET_EXCEEDED')
        return html_row(n, method, ('', '', ''), memory_footnote=memory_footnote)
    block = job if channel or n == 12 else job['k0']
    metrics = block['native0']
    gates = metrics['gates_total']
    assert gates == metrics['gates_cx'] + metrics['gates_u']
    factor = job['lambda'] if channel else block['beta'] ** 2
    weighted = gates * factor
    seconds = block['compile_to_native0_seconds']
    if not channel and n != 12:
        stages = block['stage_times_seconds']
        assert math.isclose(sum(stages.values()), seconds, rel_tol=1e-12)
        seconds = sum(v for k, v in stages.items() if k != 'tree_register_adaptation')
    if channel:
        assert math.isclose(weighted, job['costs']['lambda_weighted_gates'], rel_tol=1e-12)
    elif n == 12:
        assert math.isclose(weighted, job['costs']['beta_squared_weighted_gates'], rel_tol=1e-12)
    assert seconds >= 0
    return html_row(n, method, (f'{gates:,}', f'{weighted:,.2f}', f'{seconds:.3f}'))


def render(snapshot, source, supplemental=None, pauli_results=None):
    experiments = {e['name']: e for e in snapshot['experiments']}
    channel = experiments['full_channel_opt_fixed_20260910']
    large = experiments['tfim12_k0_fixed_20260910']
    jobs = {(j['n'], j['method']): j for j in channel['jobs']}
    if supplemental is not None:
        additions = json.loads(supplemental.read_text())
        for job in additions:
            key = (job['n'], job['method'])
            if key in jobs:
                raise ValueError(f'Supplemental result would replace an existing result: {key}')
            jobs[key] = job
    k12 = {j['method']: j for j in large['jobs']}
    if pauli_results is not None:
        replacement = json.loads(pauli_results.read_text())
        assert sorted(j['n'] for j in replacement) == [3, 4, 6, 8, 12]
        for measured in replacement:
            assert measured['method'] == 'pauli_placement' and measured['status'] in SUCCESS
            job = dict(measured, k0=measured['branches'][0])
            jobs[(job['n'], job['method'])] = job
            if job['n'] == 12:
                block = dict(job['k0'], status=job['status'])
                block['compile_to_native0_seconds'] = sum(v for k, v in block['stage_times_seconds'].items()
                                                          if k != 'tree_register_adaptation')
                block['costs'] = {'beta_squared_weighted_gates': block['beta']**2 * block['native0']['gates_total']}
                k12['pauli_placement'] = block
    timestamp = datetime.fromisoformat(snapshot['timestamp_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
    if pauli_results is not None:
        environment = json.loads((pauli_results.parent / 'environment.json').read_text())
        timestamp = datetime.fromisoformat(environment['timestamp_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
    lines = ['# TFIM Native0 Results', '',
        f'Updated: {timestamp:%Y-%m-%d %H:%M:%S} Asia/Shanghai.', '',
        'All populated cells use the current remote run on 192.168.20.20. '
        'Pauli uses the corrected Matrix-order backend. '
        'Only each route\'s native beta is used; matched-beta results are excluded.', '',
        'Displayed routes: Ours (Pauli Matrix-order) and raw-Kraus TUD separate. '
        'These archived TFIM measurements use a fixed Kraus family without channel '
        'rewrite_search; Ours here denotes the measured Pauli compilation backend, '
        'not the Pauli + rewrite pipeline in the revised HRW comparison. '
        + ('Pauli rows were rerun with initial beam width 32, beam width 8, layer candidate '
           'limit 8, and exact layer assignment threshold 8; these replace the old Pauli '
           'measurements in both tables. Unitary results retain their original measurements.'
           if pauli_results is not None else
           'This table update filters existing results and does not rerun compilation.'), '',
        '- Per-attempt gate count: `G = N_u + N_cx`, native0 `{u,cx}`, optimization level 0.',
        '- Expected gate count: `beta^2 G` for K0; `Lambda G` for the channel, with `Lambda = sum_j beta_j^2`.',
        '- These weighted counts are expected-cost numerators. Actual repeated-attempt costs divide by '
        '`Tr(K0 rho K0^dagger)` or `Tr(Phi(rho))`, respectively.',
        '- Time is classical compilation through native0, in seconds. No opt3, serialization, '
        'imports, metric counting, or numerical validation time is included.',
        '- Empty numeric cells mean no completed measurement, not zero. '
        'Large probes were skipped by request.', '',
        '## Table 1: TFIM K0', '',
        'Complete single-Kraus blocks, including PREPARE, SELECT, and uncompute. '
        + ('For every Pauli size, including n=12, K0 is extracted from the new full-channel '
           'run on CPU 72; its time excludes tree-register adaptation. Unitary n<=8 rows '
           'come from the previous full-channel runs; unitary n=12 retains the dedicated '
           'K0 failure on CPU 8.' if pauli_results is not None else
        'For n=3/4/6/8, data come from the independently compiled K0 branch before outer '
        'channel assembly; time excludes the later tree-register adaptation. '
        'TFIM12 comes from the dedicated K0 experiment. '
        'Thus all rows use the same remote software environment; n<=8 ran on CPU 72 and n=12 on CPU 8.'), '']
    lines += table_start()
    for n in (3, 4, 6, 8, 12):
        for method in METHODS:
            job = k12.get(method, {}) if n == 12 else jobs.get((n, method), {})
            lines.append(row(n, method, job))
    lines += ['</tbody>', '</table>', '']
    if any(k12.get(method, {}).get('status') == 'MEMORY_BUDGET_EXCEEDED'
           for method in METHODS if method != 'pauli_placement'):
        lines += ['<p id="tfim12-memory-note"><sup>1</sup> For TFIM-12, the marked '
                  'unitary-synthesis runs exceeded the 128 GiB worker-process-tree RSS '
                  'budget during QSD/controlled-circuit assembly and were terminated '
                  'before native0 compilation completed. Final gate counts and complete '
                  'compilation times are therefore unavailable. This records a resource '
                  'limit under the stated experimental budget, not algorithmic '
                  'infeasibility.</p>', '']
    lines += ['## Table 2: Full TFIM Channel', '',
        'Every Kraus is synthesized independently and assembled with `structure="opt"`. '
        'G and time include outer PREPARE, tree SELECT, controlled blocks, cleanup, '
        'and final native0 lowering. '
        + ('The new Pauli runs cover all five sizes on CPU 72. TFIM12 unitary full-channel '
           'compilation was not launched after its dedicated K0 run exceeded the memory budget.'
           if pauli_results is not None else
           'The TFIM12 Pauli full-channel result is a supplemental run in the same remote '
           'environment on CPU 72. The TFIM12 unitary full-channel run was not launched '
           'after its dedicated K0 compilation exceeded the memory budget.'
           if (12, 'pauli_placement') in jobs else
           'Full-channel TFIM12 has not been launched; its blank rows are placeholders, '
           'independent of the dedicated K0 experiment.'), '']
    lines += table_start()
    for n in (3, 4, 6, 8, 12):
        for method in METHODS:
            lines.append(row(n, method, jobs.get((n, method), {}), channel=True))
    lines += ['</tbody>', '</table>', '', '## TFIM12 Status', '']
    for method in METHODS:
        job = k12.get(method, {})
        detail = status(job)
        if job.get('status') == 'MEMORY_BUDGET_EXCEEDED':
            peak = job['monitor']['observed_worker_tree_peak_rss_bytes'] / 1024**3
            elapsed = job['compile_elapsed_at_stop_seconds']
            detail += (f': stopped during {job["failed_stage"]} at {elapsed:.1f} s '
                       f'of compilation; sampled peak RSS {peak:.2f} GiB exceeded 128 GiB. '
                       'No final native0 circuit/count was produced; failed-attempt time is not '
                       'entered as completed compilation time in Table 1.')
        elif job.get('status') == 'COMPILING':
            progress = job['progress']
            elapsed = progress['completed_seconds']['compile']
            if progress.get('category') == 'compile':
                elapsed += job['stage_elapsed_now_seconds']
            detail += f': {progress["stage"]}; approximately {elapsed / 60:.1f} min compilation elapsed.'
        lines.append(f'- {METHODS[method]}: {detail}')
    lines += ['', '## Provenance', '',
        f'- Snapshot: [{source.name}]({source.resolve()})',
        '- Full-channel source: `results/full_channel_opt_fixed_20260910/jobs/*/result.json`.',
        '- Dedicated K0 source: `results/tfim12_k0_fixed_20260910/jobs/*/result.json`.',
        '- Remote result paths above are under `/home/huangqf/tfim_tud_remote_20260910_1925`.',
        '- Snapshot retains per-route beta/Lambda, unrounded times, phase-stage records, '
        'source paths, and resource/failure status. Times are single runs on a shared server.', '',
        'Regenerate after archiving a fresh snapshot with:', '', '```sh',
        f'python experiments/tfim_tud/update_markdown_tables.py {source.relative_to(ROOT)}', '```', '']
    if supplemental is not None:
        lines.insert(lines.index('## Provenance') + 2,
                     f'- Supplemental full-channel data: [{supplemental.name}]({supplemental.resolve()}).')
        command = f'python experiments/tfim_tud/update_markdown_tables.py {source.relative_to(ROOT)}'
        lines[lines.index(command)] = command + f' --supplemental {supplemental.relative_to(ROOT)}'
    if pauli_results is not None:
        lines.insert(lines.index('## Provenance') + 2,
                     f'- Replacement Pauli results (32/8/8/8): [{pauli_results.name}]({pauli_results.resolve()}).')
        command_index = next(i for i, line in enumerate(lines) if line.startswith('python experiments/'))
        lines[command_index] += f' --pauli-results {pauli_results.relative_to(ROOT)}'
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--supplemental', type=Path)
    parser.add_argument('--pauli-results', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'Results/tfim_tud/TFIM-NATIVE0-TABLES.md')
    args = parser.parse_args()
    result = render(json.loads(args.snapshot.read_text()), args.snapshot.resolve(),
                    args.supplemental.resolve() if args.supplemental else None,
                    args.pauli_results.resolve() if args.pauli_results else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result)
    print(args.out)
