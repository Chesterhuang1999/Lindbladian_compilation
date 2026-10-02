"""Verify new TFIM native0 circuits and the search parameters actually used."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tfim_tud_mplconfig')
from qiskit import qpy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(out):
    checked = 0
    for line in (out / 'checksums.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert sha(out / relative) == expected, relative
        checked += 1
    results = json.loads((out / 'results.json').read_text())
    assert [r['n'] for r in results] == [3, 4, 6, 8, 12]
    expected_profile = {'beam_width': 8, 'layer_candidate_limit': 8,
                        'max_exact_layer_assignments': 8, 'mixed_initial_beam_width': 32}
    audit = []
    for r in results:
        assert r['status'] == 'COMPILED_VALIDATION_SKIPPED_BY_USER'
        assert r['method'] == 'pauli_placement'
        assert len(r['branches']) == r['n'] + 1
        for b in r['branches']:
            debug = b['metadata']['pauli_adapter']['placement_search_debug']
            assert {key: debug['search_profile'][key] for key in expected_profile} == expected_profile
            assert debug['beam_width'] == 8
            for layer in debug['layers']:
                assert layer['states_kept'] <= (32 if layer['hamming_weight'] == 1 else 8)
        assert math.isclose(sum(b['beta']**2 for b in r['branches']), r['lambda'], rel_tol=1e-12)
        assert math.isclose(r['lambda'] * r['native0']['gates_total'], r['costs']['lambda_weighted_gates'], rel_tol=1e-12)
        assert math.isclose(sum(v for k, v in r['stage_times_seconds'].items() if not k.startswith('save_')),
                            r['compile_to_native0_seconds'], rel_tol=1e-12)
        path = out / 'jobs' / f'TFIM{r["n"]}_pauli_placement' / 'channel_native0.qpy'
        assert sha(path) == r['circuit_record']['sha256']
        with path.open('rb') as stream:
            circuits = qpy.load(stream)
        assert len(circuits) == 1
        circuit = circuits[0]
        counts = dict(circuit.count_ops())
        assert counts == r['native0']['counts'] and set(counts) <= {'u', 'cx'}
        assert sum(counts.values()) == r['native0']['gates_total']
        assert circuit.num_qubits == r['native0']['width']
        audit.append({'n': r['n'], 'profile': expected_profile, 'native0': r['native0'],
                      'lambda': r['lambda'], 'compile_seconds': r['compile_to_native0_seconds'],
                      'k0': r['branches'][0]['native0']})
        print(f'VERIFIED TFIM-{r["n"]}', flush=True)
    return {'checksum_files_verified': checked, 'all_branch_profiles_verified': True,
            'timing_and_normalization_verified': True, 'circuits_recounted': audit}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(args.results_directory), indent=2) + '\n')
