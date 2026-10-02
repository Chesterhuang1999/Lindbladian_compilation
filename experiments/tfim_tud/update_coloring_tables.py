"""Render cycle-4 q=4 coloring results using the TFIM/HRW table contract."""
import argparse
import json
from pathlib import Path
import update_hrw_tables as tables

ROOT = Path(__file__).resolve().parents[2]


def render(source):
    rows = json.loads(source.read_text())
    results = {(r['n'], r['method']): r for r in rows}
    tables.SIZES = (8,)
    lines = ['# Random Coloring Native0 Results', '',
        'Cycle graph with 4 vertices and 4 colors: 8 system qubits, 16 fixed original '
        'Kraus operators, 6-qubit effective support and 256 Pauli terms per Kraus. '
        'Selected branch: K_(0,0).', '',
        'Same remote environment and compilation protocol as TFIM/HRW: corrected Pauli '
        'Matrix-order, TUD factored, and TUD separate; independently compile every local '
        'Kraus, embed, and assemble with structure="opt". No cross-Kraus rewrite or circuit reuse.', '',
        'The 256 terms are the distinct nonzero Pauli coefficients after input expansion '
        'and duplicate merging. Matrix-order optimizes their circuit placement; neither '
        'rewrite_search nor Walsh-Hadamard color-branch mixing is applied. Thus these '
        'measurements compare synthesis backends for a fixed Kraus family, not the full '
        'channel-rewrite workflow.', '',
        '- Per-attempt gate count: G=N_u+N_cx, native0 {u,cx}, optimization level 0.',
        '- Expected gate count follows the agreed weighted metric: beta^2 G for the '
        'selected Kraus and Lambda G for the full construction; Lambda=sum_j beta_j^2 '
        'uses each route\'s own normalization.',
        '- The existing local full-space coloring extension is not trace preserving. '
        'These weighted values are expected-cost numerators. For a specified input rho, '
        'the actual independent-repeat cost is beta^2 G/Tr(K rho K^dagger) or '
        'Lambda G/Tr(Phi(rho)), excluding re-preparation and measurement.',
        '- For the unchanged input family, the eigenvalues of sum_j K_j^dagger K_j '
        'range from 0.25 to 1.5. On |0^8>, the selected-Kraus trace is 1/16 and '
        'the full-map trace is 1/4, so actual repeat costs are respectively 16 and 4 '
        'times the displayed weighted values.',
        '- Compile time stops at native0. Selected-Kraus time excludes subsequent register '
        'adaptation; full time includes every branch, adaptation, outer assembly, and final '
        'lowering (including the existing tree\'s internal legacy opt1 counting). Imports, '
        'metric counting, serialization, and validation are excluded.',
        '- n in the tables denotes system qubits, not graph vertices.', '',
        '## Table 1: Selected Kraus K_(0,0)', '']
    lines += tables.table(results, True)
    lines += ['## Table 2: Full Coloring Construction', '']
    lines += tables.table(results, False)
    lines += ['## Normalization and Run Notes', '']
    lines += ['For an individual Kraus, block encoding requires beta >= ||K||_2, '
              'not beta >= 1. Here K=A^dagger/sqrt(4*4)=A^dagger/4. On a no-conflict '
              'neighbor sector, A resets four colors to one color and has norm sqrt(4)=2; '
              'on a conflict sector A is identity. Hence ||K||_2=1/2 and TUD uses beta=0.5 '
              'in K=(beta/2)(U_+ + U_-). The full construction has Lambda=16*(0.5)^2=4 '
              'and amplitude scaling sqrt(Lambda)=2.', '']
    lines += ['Pauli uses the coefficient L1 sum: beta_Pauli=2.5703125=10.28125/4. '
              'The input factor 1/4 is already included in both routes. The ratio '
              'beta_Pauli^2/beta_TUD^2 is 26.426025390625.', '']
    for row in rows:
        selected = row.get('selected_kraus', {})
        if selected:
            lines.append(f'- {tables.METHODS[row["method"]]}: selected beta={selected["beta"]:.12g}; '
                         f'Lambda={row.get("lambda", float("nan")):.12g}.')
        if row.get('status') != 'COMPILED_VALIDATION_SKIPPED_BY_USER':
            lines.append(f'- {tables.METHODS[row["method"]]}: {row["status"]}; incomplete counts are blank.')
    if all(results.get((8, method), {}).get('native0') for method in tables.METHODS):
        pauli = results[(8, 'pauli_placement')]
        for method in ('tud_factored_diagonal', 'tud_separate'):
            tud = results[(8, method)]
            ratio = pauli['costs']['lambda_weighted_gates'] / tud['costs']['lambda_weighted_gates']
            lines.append(f'- Full construction: Pauli / {tables.METHODS[method]} expected gate count '
                         f'ratio = {ratio:.4f}.')
        lines.append('- This case favors TUD factored in both gate metrics, while Pauli compiles '
                     'faster. Effective support alone does not predict the comparison: Pauli '
                     'term count and the native normalization penalty also matter.')
    lines += ['- CPU 72, one numerical thread, sequential route workers; 64 GiB worker-tree '
              'RSS budget and 7200 s compilation budget per route. One timing measurement on a shared server.',
              '- Large circuit probes skipped as requested; input matrices are checked against '
              'the classical recoloring definition, and small support/embedding contracts are tested.',
              f'- Raw results: [{source.name}]({source.resolve()}).', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'Results/coloring_tud/COLORING-NATIVE0-TABLES.md')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.results))
    print(args.out)
