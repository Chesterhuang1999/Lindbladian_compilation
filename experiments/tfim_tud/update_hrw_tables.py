"""Render measured selected-Kraus and full-channel HRW tables."""
import argparse
from html import escape
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = {'pauli_placement': 'Pauli', 'tud_factored_diagonal': 'TUD factored',
           'tud_separate': 'TUD separate'}
SIZES = (4, 8, 12, 20, 28)


def table(results, selected):
    lines = ['<table>', '<thead><tr><th>n</th><th>Route</th>'
             '<th>per-attempt gate count</th><th>expected gate count</th>'
             '<th>Compile time (s)</th></tr></thead>', '<tbody>']
    for n in SIZES:
        for method, label in METHODS.items():
            job = results.get((n, method), {})
            record = job.get('selected_kraus', {}) if selected else job
            cells = [f'<td rowspan="3" align="center" valign="middle">{n}</td>'] if method == 'pauli_placement' else []
            cells.append(f'<td>{escape(label)}</td>')
            values = ('', '', '')
            if record.get('native0') and record.get('compile_to_native0_seconds') is not None:
                g = record['native0']['gates_total']
                assert g == record['native0']['gates_u'] + record['native0']['gates_cx']
                weighted = record['beta_squared_weighted_gates'] if selected else job['costs']['lambda_weighted_gates']
                factor = record['beta']**2 if selected else job['lambda']
                assert math.isclose(weighted, factor * g, rel_tol=1e-12)
                values = (f'{g:,}', f'{weighted:,.4f}', f'{record["compile_to_native0_seconds"]:.3f}')
            cells.extend(f'<td align="right">{v}</td>' for v in values)
            lines.append('<tr>' + ''.join(cells) + '</tr>')
    return lines + ['</tbody>', '</table>', '']


def render(source):
    rows = json.loads(source.read_text())
    results = {(r['n'], r['method']): r for r in rows}
    lines = ['# HRW Native0 Results', '',
        '> Fixed-Kraus backend ablation only. This table does not test channel rewrite. '
        'For the corrected reviewer-B comparison, see '
        '[Pauli + rewrite vs raw-Kraus TUD](HRW-REWRITE-VS-RAW-TUD.md).', '',
        'Paper cases HRW-4, 8, 12, 20, and 28; measured on 192.168.20.20 in the same '
        'isolated environment as TFIM. Pauli uses the corrected Matrix-order backend. '
        'Original Kraus operators are fixed; no cross-Kraus rewrite, matched beta, opt3, or large probes.', '',
        'The paper input is `K_(i,b) = T_(i,b)^dagger / sqrt(2n)`, with local matrix '
        '`(I + X + (-1)^b Z - i(-1)^b Y) / (2 sqrt(2n))`. There are 2n Kraus operators, '
        'each supported on one system qubit. Each branch is independently synthesized on that '
        'support and embedded; no full 2^n matrix, analytic gate-count substitution, or '
        'cross-branch circuit cache is used.', '',
        '- Per-attempt gate count: G = N_u + N_cx, native0 {u,cx}, optimization level 0.',
        '- Expected gate count uses each route\'s native beta: beta^2 G for one Kraus; '
        'Lambda G for the full channel, where Lambda=sum_j beta_j^2.',
        '- For one Kraus, beta^2 G remains the expected-cost numerator, consistent with the TFIM table. '
        'For a fixed input rho, divide by Tr(K rho K^dagger). For K_(0,0) on |0^n>, '
        'that trace is 1/n; the actual expected cost is n times the displayed weighted value.',
        '- HRW is trace preserving, so the full-channel Lambda G is the actual independent-repeat '
        'expected gate count for any normalized input, excluding re-preparation and measurement.',
        '- Compilation includes block construction through native0. The single-Kraus time excludes '
        'later register adaptation; full-channel time includes all branches, register adaptation, '
        'outer assembly and final lowering. Imports, counting, serialization, and validation are excluded.',
        '- Every branch has beta_Pauli=sqrt(2/n), beta_TUD=1/sqrt(n); independently checked against '
        'the backend values. Thus Lambda_Pauli=4 and Lambda_TUD=2.', '',
        '## Table 1: Selected Kraus K_(0,0)', '',
        'The same source branch (i=0,b=0) is selected at every n. Counts include PREPARE, '
        'SELECT, and uncompute. Each local circuit acts on a full n-qubit system after embedding.', '']
    lines += table(results, True)
    lines += ['## Table 2: Full HRW Channel', '',
        'All 2n independently compiled Kraus blocks are assembled with structure="opt", '
        'including outer PREPARE, tree SELECT, controlled blocks, cleanup, and native0 lowering.', '']
    lines += table(results, False)
    failures = [r for r in rows if r.get('status') != 'COMPILED_VALIDATION_SKIPPED_BY_USER']
    lines += ['## Run Notes', '',
        f'{len(rows) - len(failures)}/{len(SIZES) * len(METHODS)} cases completed. '
        'Small semantic contracts are checked separately; completed large runs have no statevector probes.',
        'CPU 72, one numerical thread, one worker at a time, 64 GiB worker-tree RSS limit and '
        '3600 s compilation budget per size/route. All times are single measurements on a shared server.', '']
    lines += [f'- HRW-{r["n"]} {METHODS[r["method"]]}: {r["status"]}.' for r in failures]
    lines += [f'- Raw per-Kraus and channel results: [{source.name}]({source.resolve()}).',
        '- Input/source hashes and frozen protocol are archived with the deployment manifest.',
        '- These results use the same paper channel inputs but exclude the paper\'s optional '
        'cross-Kraus rewrite, matching the fixed-Kraus TFIM experiment.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'Results/hrw_tud/HRW-NATIVE0-TABLES.md')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.results))
    print(args.out)
