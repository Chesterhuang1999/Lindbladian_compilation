"""Verify archived results and recount saved native0 circuits after compilation."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from qiskit import qpy


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(out, recount):
    verified = 0
    for line in (out / 'checksums.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert sha(out / relative) == expected, relative
        verified += 1
    rows = json.loads((out / 'results.json').read_text())
    inputs = json.loads((out / 'input.json').read_text())
    assert len(rows) == 3
    assert {r['method'] for r in rows} == {'pauli_placement', 'tud_factored_diagonal', 'tud_separate'}
    circuits = []
    for row in rows:
        assert row['status'] == 'COMPILED_VALIDATION_SKIPPED_BY_USER'
        assert len(row['branches']) == len(inputs['branches']) == 16
        assert math.isclose(row['lambda'], sum(b['beta']**2 for b in row['branches']), rel_tol=1e-12)
        for b, source in zip(row['branches'], inputs['branches']):
            assert b['metadata']['support_system_qubits'] == 6
            assert tuple(b['metadata']['local_target_system']) == tuple(source['local_support_qubits'])
            # ||A^dagger||=sqrt(q), with the input scale 1/sqrt(vertices*q).
            beta = sum(abs(complex(t['real'], t['imag'])) for t in source['canonical_terms']) \
                if row['method'] == 'pauli_placement' else 1 / np.sqrt(inputs['vertices'])
            assert math.isclose(beta, b['beta'], rel_tol=1e-12)
            assert set(b['native0']['counts']) <= {'u', 'cx'}
            assert b['native0']['gates_total'] == sum(b['native0']['counts'].values())
        selected = row['selected_kraus']
        assert selected['canonical_name'] == 'K_0_0'
        assert math.isclose(selected['beta_squared_weighted_gates'], selected['beta']**2 * selected['native0']['gates_total'], rel_tol=1e-12)
        assert math.isclose(row['costs']['lambda_weighted_gates'], row['lambda'] * row['native0']['gates_total'], rel_tol=1e-12)
        assert math.isclose(selected['trace_on_zero_input'], 1 / 16, rel_tol=1e-12)
        assert math.isclose(row['costs']['zero_input_channel_trace'], .25, rel_tol=1e-12)
        selected_stages = row['branches'][0]['stage_times_seconds']
        assert math.isclose(selected['compile_to_native0_seconds'], sum(v for k, v in selected_stages.items() if k != 'tree_register_adaptation'), rel_tol=1e-12)
        assert math.isclose(row['compile_to_native0_seconds'], sum(v for k, v in row['stage_times_seconds'].items() if not k.startswith('save_')), rel_tol=1e-12)
        for record, name in ((selected, 'selected_kraus_native0.qpy'), (row, 'channel_native0.qpy')):
            path = out / 'jobs' / f'{row["case_id"]}_{row["method"]}' / name
            assert sha(path) == record['circuit_record']['sha256']
            if recount:
                with path.open('rb') as stream:
                    loaded = qpy.load(stream)
                assert len(loaded) == 1
                circuit = loaded[0]
                counts = dict(circuit.count_ops())
                assert counts == record['native0']['counts']
                assert circuit.num_qubits == record['native0']['width']
                assert sum(counts.values()) == record['native0']['gates_total']
                del circuit, loaded
                gc.collect()
            circuits.append({'path': str(path.relative_to(out)), 'sha256': sha(path),
                             'recounted': recount, 'native0': record['native0']})
            print('VERIFIED ' + str(path.relative_to(out)), flush=True)
    return {'checksum_files_verified': verified, 'complete_routes': len(rows),
            'normalization_checked_against_input': True, 'timing_scope_checked': True,
            'circuits': circuits}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_directory', type=Path)
    parser.add_argument('--recount', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.results_directory, args.recount)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'circuits'}), flush=True)
