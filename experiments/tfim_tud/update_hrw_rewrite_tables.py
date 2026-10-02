"""Compare Pauli+rewrite with direct, separate synthesis of two unitaries."""
import argparse
from html import escape
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = [('pauli_rewrite', 'Pauli + rewrite'),
           ('tud_separate', 'Raw Kraus + TUD separate')]


def render(source):
    rewritten = json.loads(source.read_text())
    raw_path = source.parent / 'raw_baseline_results.json'
    raw = json.loads(raw_path.read_text())
    results = {(r['n'], r['method']): r for r in raw + rewritten}
    lines = ['# HRW: Pauli Rewrite vs Raw-Kraus Unitary Synthesis', '',
        'This is the corrected full-channel comparison for reviewer B: Pauli basis + '
        'rewrite_search versus raw original Kraus + TUD separate. The baseline forms '
        'two unitary matrices for each Kraus and synthesizes them separately. Only the Pauli '
        'route is rerun. The raw-Kraus TUD separate route reuses the completed remote measurements '
        'from 2026-09-10, with identical frozen input hashes.', '',
        'This comparison evaluates that direct two-unitary synthesis implementation; '
        'it does not establish a bound over all possible two-unitary synthesis algorithms.', '',
        'Pauli applies the existing rewrite_search(strategy="greedy", beam_width=3, '
        'max_steps=50, tol=1e-12), then compiles every retained rewritten Kraus with the '
        'corrected Matrix-order backend. TUD uses the original 2n Kraus without rewrite. '
        'All routes independently compile on each operator\'s effective local support, '
        'embed, assemble using structure="opt", and lower to native0 {u,cx}, opt0, '
        'seed 646, approximation_degree=1, qubits_initially_zero=False.', '',
        '- Per-attempt gate count: G=N_u+N_cx, including all inner blocks and the outer LCU.',
        '- Expected gate count: Lambda*G, where Lambda=sum_j beta_j^2 is recomputed from '
        'the actual Kraus family compiled by each route. HRW is trace preserving, so this '
        'is the independent-repeat gate cost, excluding input preparation and measurement.',
        '- Pauli compile time includes rewrite copying/search/application, support extraction, '
        'all block construction/lowering/register adaptation, outer assembly (including its '
        'legacy opt1 counting work), and final native0 lowering. Raw TUD time includes the '
        'corresponding synthesis stages. Input generation, imports, metric counting, '
        'serialization, and validation are excluded.',
        '- A rewritten Kraus is not the original selected Kraus. Accordingly, the claim is '
        'evaluated on the full channel; the old selected-Kraus table remains an unrevised '
        'backend ablation, not evidence for the rewrite-system comparison.', '',
        '## Full-Channel Comparison', '', '<table>',
        '<thead><tr><th>n</th><th>Route</th><th>per-attempt gate count</th>'
        '<th>expected gate count</th><th>Compile time (s)</th></tr></thead>', '<tbody>']
    for n in (4, 8, 12, 20, 28):
        for index, (method, label) in enumerate(METHODS):
            row = results.get((n, method), {})
            cells = [f'<td rowspan="{len(METHODS)}" align="center" valign="middle">{n}</td>'] if index == 0 else []
            cells += [f'<td>{escape(label)}</td>']
            values = ('', '', '')
            if row.get('native0'):
                g = row['native0']['gates_total']
                weighted = row['costs']['lambda_weighted_gates']
                assert g == row['native0']['gates_u'] + row['native0']['gates_cx']
                assert math.isclose(weighted, row['lambda'] * g, rel_tol=1e-12)
                values = (f'{g:,}', f'{weighted:,.4f}', f'{row["compile_to_native0_seconds"]:.3f}')
            cells.extend(f'<td align="right">{v}</td>' for v in values)
            lines.append('<tr>' + ''.join(cells) + '</tr>')
    lines += ['</tbody>', '</table>', '', '## Rewrite Audit', '',
        '| n | Kraus before / after | Pauli terms before / after | Search steps | Lambda after | Rewrite time (s) | Compilation after rewrite (s) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rewritten:
        audit = row.get('rewrite')
        if not audit or not row.get('native0'):
            continue
        lines.append(f'| {row["n"]} | {audit["initial_kraus_count"]} / {audit["final_kraus_count"]} | '
            f'{audit["initial_support"]} / {audit["final_support"]} | {audit["step_count"]} | '
            f'{row["lambda"]:.8f} | {row["rewrite_seconds"]:.6f} | {row["post_rewrite_compile_seconds"]:.6f} |')
    lines += ['', '## Effect Relative to the Previous Pauli Ablation', '',
        '| n | Previous Pauli G | Pauli + rewrite G | Previous expected gates | Pauli + rewrite expected gates | Raw TUD separate / Pauli + rewrite expected ratio |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in rewritten:
        if not row.get('native0'):
            continue
        n = row['n']
        previous = results[(n, 'pauli_placement')]
        tud = results[(n, 'tud_separate')]
        e = row['costs']['lambda_weighted_gates']
        lines.append(f'| {n} | {previous["native0"]["gates_total"]:,} | {row["native0"]["gates_total"]:,} | '
            f'{previous["costs"]["lambda_weighted_gates"]:,.4f} | {e:,.4f} | '
            f'{tud["costs"]["lambda_weighted_gates"] / e:.4f} |')
    errors = [r['rewrite']['coefficient_gram_inf_error'] for r in rewritten if r.get('rewrite')]
    lines += ['', '## Validation and Provenance', '',
        '- Every size checks channel equivalence through the coefficient Gram matrix in '
        'the common Pauli basis and independently replays all recorded two-row unitary '
        'rotations. This validates the rewrite without constructing a dense n-qubit channel.',
        f'- Maximum coefficient Gram infinity-norm residual: {max(errors):.3e}.' if errors else '- Rewrite audit pending.',
        '- A small HRW-2 test checks the emitted full native0 success blocks and their Choi '
        'matrix against the raw channel; large statevector probes remain skipped as requested.',
        '- Same remote host 192.168.20.20 and isolated environment: Python 3.12.9, Qiskit '
        '2.1.1, NumPy 2.4.2, SciPy 1.17.1; CPU 72, one numerical thread, sequential workers. '
        'Limits: 64 GiB worker-tree RSS and 3600 s compilation per size. Times are single '
        'measurements on a shared server; raw TUD and rewritten Pauli were measured in separate batches.',
        f'- New Pauli results: [{source.name}]({source.resolve()}).',
        f'- Reused raw-Kraus baseline: [{raw_path.name}]({raw_path.resolve()}).',
        '- Previous fixed-Kraus tables: [HRW-NATIVE0-TABLES.md](HRW-NATIVE0-TABLES.md).', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'Results/hrw_tud/HRW-REWRITE-VS-RAW-TUD.md')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.results))
    print(args.out)
