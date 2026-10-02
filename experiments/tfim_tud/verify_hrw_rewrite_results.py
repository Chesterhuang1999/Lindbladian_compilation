"""Independently recount the five saved HRW rewrite circuits and audit totals."""
import argparse
import gc
import json
import math
from pathlib import Path
from run_remote_hrw import sha, np
from qiskit import qpy


def verify(out):
    files = 0
    for line in (out / 'checksums.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert sha(out / relative) == expected, relative
        files += 1
    rows = json.loads((out / 'results.json').read_text())
    baseline = json.loads((out / 'raw_baseline_results.json').read_text())
    assert [r['n'] for r in rows] == [4, 8, 12, 20, 28]
    records = []
    for row in rows:
        n = row['n']
        assert row['status'] == 'COMPILED_VALIDATION_SKIPPED_BY_USER'
        assert {b['input_sha256'] for b in baseline if b['n'] == n} == {row['input_sha256']}
        d = out / 'jobs' / f'HRW{n}_pauli_rewrite'
        with np.load(d / 'rewrite_coefficients.npz') as arrays:
            before, after = arrays['before'], arrays['after']
        np.testing.assert_allclose(before.conj().T @ before, after.conj().T @ after, atol=1e-12)
        audit = row['rewrite']
        assert audit['initial_support'] == np.count_nonzero(np.abs(before) > 1e-12)
        assert audit['final_support'] == np.count_nonzero(np.abs(after) > 1e-12)
        branches = json.loads((d / 'rewritten_input.json').read_text())['branches']
        assert len(branches) == len(row['branches']) == audit['final_kraus_count']
        betas = []
        for b, compiled in zip(branches, row['branches']):
            beta = sum(abs(complex(t['real'], t['imag'])) for t in b['canonical_terms'])
            assert math.isclose(beta, compiled['beta'], rel_tol=1e-12)
            betas.append(beta)
        assert math.isclose(sum(b*b for b in betas), row['lambda'], rel_tol=1e-12)
        assert math.isclose(row['costs']['lambda_weighted_gates'], row['lambda'] * row['native0']['gates_total'], rel_tol=1e-12)
        assert math.isclose(row['compile_to_native0_seconds'],
            sum(v for k, v in row['stage_times_seconds'].items() if not k.startswith('save_')), rel_tol=1e-12)
        assert math.isclose(row['rewrite_seconds'] + row['post_rewrite_compile_seconds'],
                            row['compile_to_native0_seconds'], rel_tol=1e-12)
        circuit_path = d / 'channel_native0.qpy'
        assert sha(circuit_path) == row['circuit_record']['sha256']
        with circuit_path.open('rb') as stream:
            circuits = qpy.load(stream)
        assert len(circuits) == 1
        circuit = circuits[0]
        counts = dict(circuit.count_ops())
        assert set(counts) <= {'u', 'cx'}
        assert counts == row['native0']['counts']
        assert sum(counts.values()) == row['native0']['gates_total']
        assert circuit.num_qubits == row['native0']['width']
        records.append({'n': n, 'counts': counts, 'width': circuit.num_qubits,
                        'sha256': row['circuit_record']['sha256']})
        del circuit, circuits
        gc.collect()
        print(f'VERIFIED HRW-{n}', flush=True)
    return {'checksum_files_verified': files, 'circuits_recounted': records,
            'coefficient_gram_checks_passed': True, 'normalization_and_timing_checks_passed': True,
            'raw_baseline_input_hashes_match': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.results_directory)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'circuits_recounted'}))
