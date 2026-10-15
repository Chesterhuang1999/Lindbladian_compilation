#!/usr/bin/env python3
"""One compilation per input with the production P0-2 default configuration."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
import tempfile
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BEFORE = ROOT / 'paper_experiments/pending_review/table13_matrix_order_qrisp_20261010'
ALL_P0 = OUT.parent / 'matrix_order_p0_singlepass_20261010'
CASES = ['TFIM-8', 'RepDQE-d5'] + [f'RandCh-N9-8q-seed{s}' for s in range(5)]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def snapshot():
    files = sorted((ROOT / 'src').glob('*.py'))
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = OUT / 'source_snapshot/manifest.json'
    if manifest.exists():
        assert hashes == json.loads(manifest.read_text())['sha256'], 'Production source changed after snapshot'
    else:
        for source in files:
            dest = OUT / 'source_snapshot/src' / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
        dump(manifest, {'original_directory': str(ROOT / 'src'), 'sha256': hashes})
    for name, digest in hashes.items():
        assert sha256((OUT / 'source_snapshot/src' / name).read_bytes()).hexdigest() == digest
    return hashes


def coefficient_check(left, right):
    import numpy as np
    labels = sorted({a.expr for family in (left, right) for ms in family for a, _ in ms.instances})
    position = {label: i for i, label in enumerate(labels)}
    def matrix(family):
        out = np.zeros((len(family), len(labels)), complex)
        for i, ms in enumerate(family):
            for atom, coefficient in ms.instances:
                out[i, position[atom.expr]] += coefficient * atom.phase
        return out
    a, b = matrix(left), matrix(right)
    gram = a.conj().T @ a
    error = float(np.linalg.norm(gram - b.conj().T @ b) / max(np.linalg.norm(gram), 1e-30))
    ordered = float(np.max(np.abs(a - b))) if a.shape == b.shape else None
    assert error < 1e-10
    return {'gram_relative_error': error, 'ordered_coefficient_max_error': ordered}


def check_residuals(block, selected):
    import numpy as np
    from qiskit.quantum_info import Pauli
    if len(block.J.instances) == 1:
        return {'occupied_addresses_checked': 1, 'maximum_phase_error': 0.0}
    candidate = next(c for c in block._matrix_order_candidates if c['name'] == selected)
    phases = {a.expr: complex(a.phase) for a, _ in block.J.instances}
    maximum, checked = 0.0, 0
    for label, bits in candidate['modes']:
        if label not in phases:
            continue
        address = int(bits, 2)
        value, scalar = Pauli('I' * block.sys_size), 1 + 0j
        for residual, lower_bits, phase in candidate['mobius']['g_modes_with_phase']:
            lower = int(lower_bits, 2)
            if lower & address == lower:
                value = Pauli(residual) @ value
                scalar *= np.exp(1j * np.pi * phase / 2)
        target = Pauli(label)
        assert np.array_equal(value.x, target.x) and np.array_equal(value.z, target.z)
        maximum = max(maximum, float(abs(scalar * (-1j) ** value.phase - phases[label])))
        checked += 1
    assert maximum < 1e-10
    return {'occupied_addresses_checked': checked, 'maximum_phase_error': maximum}


def instruction_fingerprint(circuit):
    # A circuit-wide scalar phase cancels in the implemented channel.
    items = [(inst.operation.name,
              [circuit.find_bit(q).index for q in inst.qubits],
              [float(value).hex() for value in inst.operation.params]) for inst in circuit.data]
    return sha256(json.dumps([circuit.num_qubits, items], separators=(',', ':')).encode()).hexdigest()


def preflight(hashes):
    import block_encoding as bm
    import placement_search as ps
    from channel_IR import Matrixsum, PauliAtom
    from matrix_order_pipeline import resolve_options, options_dict
    options = resolve_options()
    assert not options.phase_aware_candidates and options.random_padding
    vectors = [ps._label_to_bits(label) for label in ('ZI', 'IZ', 'ZZ')]
    supersets = [ps._strict_supersets(a, 3) for a in range(8)]
    empty = ps._SearchState((0, 1, 2), (0,) * 8, 0, ())
    simultaneous = ps._apply_layer_assignment(empty, ((0, 1), (1, 2), (2, 3)), vectors, 2, supersets)
    upper = ps._apply_layer_assignment(empty, ((2, 3),), vectors, 2, supersets)
    lower = ps._apply_layer_assignment(upper, ((0, 1), (1, 2)), vectors, 2, supersets)
    assert simultaneous.cost == lower.cost == 2 and simultaneous.prefix == lower.prefix
    ms = Matrixsum([(PauliAtom(label), 1.) for label in ('XIIII', 'IXIII', 'IIXII', 'IIIXI', 'IIIIX')])
    padding = {}
    for enabled in (True, False):
        block = bm.BlockEncoding(ms, placement_options={}, matrix_order_options={'random_padding': enabled})
        block.find_optimal_order_matrices()
        padding[str(enabled)] = block.placement_search_debug['padding_count']
    assert padding == {'True': 2, 'False': 0}
    hrw = Matrixsum([(PauliAtom(label, phase=phase), .25)
                    for label, phase in [('II', 1), ('IX', 1), ('IZ', 1), ('IY', -1j)]])
    candidate_names = {}
    for enabled in (False, True):
        block = bm.BlockEncoding(hrw, placement_options={}, matrix_order_options={'phase_aware_candidates': enabled})
        block.circuit('Matrix-order')
        names = [item['name'] for item in block._matrix_order_candidates]
        assert any('/' in name for name in names) == enabled
        candidate_names[str(enabled)] = names
    dump(OUT / 'preflight.json', {'passed': True, 'source_hashes': hashes,
         'options': options_dict(options), 'prefix_cost': simultaneous.cost,
         'late_lower_address_cost': lower.cost, 'random_padding_counts': padding,
         'phase_candidate_switch': candidate_names})
    print(json.dumps({'preflight_passed': True, 'options': options_dict(options)}), flush=True)


def summaries(rows):
    columns = ['case', 'G_before_P0', 'G_all_P0', 'G_P02', 'W_before_P0', 'W_all_P0', 'W_P02',
               'before_P0_compile_median_s', 'all_P0_compile_s', 'P02_compile_s', 'rewrite_s',
               'assembly_s', 'lowering_s', 'verification']
    with (OUT / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in rows)
    dump(OUT / 'comparison.json', [{key: row[key] for key in columns} for row in rows])
    grouped = {}
    for row in rows:
        grouped.setdefault(row['case'].split('-seed')[0], []).append(row)
    table = []
    for benchmark, values in grouped.items():
        table.append({'benchmark': benchmark, 'inputs': len(values),
                      **{key: mean(r[key] for r in values) for key in columns[1:-1]},
                      'W_P02_values': sorted({r['W_P02'] for r in values})})
    dump(OUT / 'comparison_table.json', table)


def run(hashes):
    import block_encoding as bm
    import rq1_pipeline as rp
    from qiskit import qasm2
    from importlib.metadata import version
    from experiment_protocol import REWRITE_DEFAULTS, transpile_channel_circuit
    from matrix_order_pipeline import resolve_options, options_dict
    pre = json.loads((OUT / 'preflight.json').read_text())
    assert pre['passed'] and pre['source_hashes'] == hashes
    assert not (OUT / 'settings.json').exists(), 'Measured inputs must not be repeated'
    dump(OUT / 'settings.json', {'cases': CASES, 'repetitions': 1, 'warmups': 0,
         'pipeline': 'Ours: canonicalization + rewrite + tree/Fstar + Matrix-order',
         'matrix_order_options': options_dict(resolve_options()), 'rewrite': asdict(REWRITE_DEFAULTS),
         'P0_policy': 'P0-2 active; P0-1 off; legacy random padding enabled; final refinement table consistency retained',
         'basis': ['rz', 'h', 'x', 'cx'], 'optimization_level': 0, 'seed_transpiler': 20261009,
         'qubits_initially_zero': False, 'source_hashes': hashes,
         'timing': 'canonicalization through final NAM lowering; exclude imports, input loading, verification and output',
         'comparison': 'before-P0 times are warmed medians of three runs; all-P0 and this run are single observations',
         'reference_before_P0': str(BEFORE), 'reference_all_P0': str(ALL_P0),
         'versions': {name: version(name) for name in ('qiskit', 'numpy')},
         'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'QISKIT_NUM_PROCS')},
         'runner_sha256': sha256(Path(__file__).read_bytes()).hexdigest()})
    blocks = []
    class RecordedBlock(bm.BlockEncoding):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            blocks.append(self)
    rp.BlockEncoding = RecordedBlock
    rows, failures = [], []
    for case in CASES:
        directory = OUT / case
        input_record = json.loads((ALL_P0 / case / 'input.json').read_text())
        raw = rp.deserialize_kraus(input_record['input_kraus'])
        dump(directory / 'input.json', input_record)
        old = [json.loads((BEFORE / case / 'ours' / f'run-{i}.json').read_text()) for i in (1, 2, 3)]
        full = json.loads((ALL_P0 / case / 'run-1.json').read_text())
        blocks.clear()
        print(json.dumps({'case': case, 'stage': 'started'}), flush=True)
        try:
            with (directory / 'compilation.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                started = time.perf_counter()
                canonical, canon_info = rp.canonicalize(raw)
                circuit, registers, assembly = rp.build_pipeline(canonical, 'Ours', matrix_order_options={})
                lower_started = time.perf_counter()
                lowered = transpile_channel_circuit(circuit, basis_gates=['rz','h','x','cx'],
                    optimization_level=0, seed_transpiler=20261009)
                ended = time.perf_counter()
            counts = {str(k): int(v) for k, v in lowered.count_ops().items()}
            assert not (set(counts) - {'rz', 'h', 'x', 'cx'})
            row = {'case': case, 'status': 'compiled', 'repetition': 1,
                   'G_before_P0': old[0]['G'], 'G_all_P0': full['G_after'], 'G_P02': sum(counts.values()),
                   'W_before_P0': old[0]['W'], 'W_all_P0': full['W_after'], 'W_P02': lowered.num_qubits,
                   'before_P0_compile_median_s': median(r['compile_time_s'] for r in old),
                   'all_P0_compile_s': full['compile_time_s'], 'P02_compile_s': ended - started,
                   'rewrite_s': assembly['rewrite']['time_s'], 'assembly_s': assembly['assembly_time_s'],
                   'lowering_s': ended - lower_started, 'Lambda': circuit.metadata['subnormalization_factor'],
                   'counts': counts, 'registers': registers, 'canonicalization': canon_info,
                   'assembly': assembly, 'metadata': circuit.metadata,
                   'padding': [getattr(b, 'placement_search_debug', {}) for b in blocks]}
            dump(directory / 'run-1.json', row)
            qasm2.dump(lowered, directory / 'channel_nam_opt0.qasm')
            print(json.dumps({'case': case, 'stage': 'compiled', **{k: row[k] for k in
                  ('G_P02','W_P02','P02_compile_s','assembly_s')}}), flush=True)
            rewritten = rp.deserialize_kraus(assembly['compiled_kraus'])
            validation = {'rewrite': coefficient_check(raw, rewritten)}
            previous_rewrite = coefficient_check(rp.deserialize_kraus(full['assembly']['compiled_kraus']), rewritten)
            assert previous_rewrite['ordered_coefficient_max_error'] < 1e-10
            validation['matches_previous_rewrite'] = previous_rewrite
            assert abs(row['Lambda'] - full['Lambda']) < 1e-10
            assert len(blocks) == len(circuit.metadata['block_metadata'])
            symbolic = []
            for block, meta in zip(blocks, circuit.metadata['block_metadata']):
                info = meta.get('matrix_order', {})
                assert all('/' not in c['candidate'] for c in info.get('candidates', []))
                symbolic.append(check_residuals(block, info.get('selected_candidate')))
            validation['selected_residuals'] = symbolic
            if case == 'RepDQE-d5':
                previous = qasm2.load(ALL_P0 / case / 'channel_nam_opt0.qasm')
                current_fingerprint = instruction_fingerprint(lowered)
                previous_fingerprint = instruction_fingerprint(previous)
                same = current_fingerprint == previous_fingerprint and registers == full['registers'] and row['Lambda'] == full['Lambda']
                old_check = full['validation']['full_channel']
                if same and old_check.get('passed'):
                    validation['full_channel'] = {'method': 'reuse verified Choi after exact lowered instruction/ABI/Lambda equality',
                        'passed': True, 'source': str(ALL_P0 / case / 'run-1.json'),
                        'source_sha256': sha256((ALL_P0 / case / 'run-1.json').read_bytes()).hexdigest(),
                        'instruction_fingerprint': current_fingerprint, 'reference_check': old_check,
                        'global_phase': 'ignored because a circuit-wide phase cancels in a channel'}
                    row['verification'] = 'exact circuit/ABI/Lambda match to previous full Choi + symbolic residuals + Gram'
                else:
                    print(json.dumps({'case':case,'stage':'full Choi verification; circuit differs from verified reference'}),flush=True)
                    validation['full_channel'] = rp.verify_channel(lowered, registers, raw)
                    assert validation['full_channel']['passed']
                    row['verification'] = 'full Choi + symbolic residuals + Gram'
            else:
                validation['full_channel'] = {'method': 'not simulated at this width'}
                row['verification'] = 'symbolic residuals + Gram; no full lowered-channel simulation'
            row['validation'] = validation
            row['status'] = 'verified'
            dump(directory / 'run-1.json', row)
            rows.append(row)
            summaries(rows)
            print(json.dumps({'case':case,'stage':'verified','method':row['verification']}),flush=True)
        except Exception as error:
            import traceback
            failure = {'case':case,'error':str(error),'traceback':traceback.format_exc()}
            failures.append(failure)
            dump(directory / 'failure.json', failure)
            print(json.dumps({'case':case,'stage':'failed','error':str(error)}),flush=True)
    dump(OUT / 'completion.json', {'passed':not failures and len(rows)==len(CASES),
         'requested':len(CASES),'completed':len(rows),'failures':failures})
    assert not failures, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    hashes = snapshot()
    sys.path.insert(0, str(OUT / 'source_snapshot/src'))
    with tempfile.TemporaryDirectory(prefix='matrix-p02-cache-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        os.environ['NUMBA_CACHE_DIR'] = cache
        preflight(hashes) if args.preflight else run(hashes)


if __name__ == '__main__':
    main()
