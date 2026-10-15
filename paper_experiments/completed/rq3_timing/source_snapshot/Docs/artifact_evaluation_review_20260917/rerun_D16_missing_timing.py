"""Rerun the missing n=12 notebook simulation timing in an isolated directory."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lindbladian_D16_matplotlib')
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
import channel_LCU as lcu
from channel_IR import Lindbladian, channel_ensemble

OUT = ROOT / 'Baseline_results/eval_redesign/artifact_review_20260917/D16_missing_n12_rerun'


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Refusing to overwrite an initialized run'
    n, delta = 12, 0.1
    gamma = np.sqrt(0.1) / 2
    sources = ['src/channel_LCU.py', 'src/channel_IR.py', 'src/block_encoding.py',
               'src/subroutine.py', 'src/baseline.py',
               str(Path(__file__).relative_to(ROOT))]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    machine_info = subprocess.run(['sysctl', '-n', 'hw.model'], capture_output=True, text=True)
    metadata = {
        'n': n, 'delta_t': delta, 'gamma': float(gamma), 'input_state': '|0>^12',
        'structure': 'basic', 'opt': 'No', 'block_encoding_class': 'block_encoding.BlockEncoding',
        'simulate_entry': 'channel_LCU.simulate_circuit',
        'timing_scope': 'simulate_circuit only: deepcopy, Aer transpile opt1, simulation, approx_state(tol=1e-6), postselection and density extraction',
        'excluded_from_timing': ['model construction', 'channel lowering', 'circuit construction',
                                 'initial state allocation', 'export', 'validation'],
        'repetitions': 1, 'reason': 'original n=12 188.03s timing record not found',
        'reference': {'status': 'historical_timeout_user_confirmed',
                      'timeout_seconds': None, 'rerun': False},
        'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
        'logical_cpus': os.cpu_count(),
        'machine_model': machine_info.stdout.strip() if machine_info.returncode == 0 else None,
        'machine_model_read_error': machine_info.stderr.strip() if machine_info.returncode else None,
        'versions': {p: importlib.metadata.version(p) for p in ['qiskit', 'qiskit-aer', 'numpy', 'scipy', 'qutip']},
        'thread_environment': {p: os.environ.get(p) for p in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'aer_default_options': dict(lcu._require_aer_simulator()().options),
        'source_hashes_before': hashes, 'argv': sys.argv,
    }
    write('protocol.json', metadata)
    write('progress.json', {'stage': 'constructing', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    begin = time.perf_counter()
    h_terms, jump_terms = [], []
    for i in range(n):
        z = ''.join('Z' if j in [i, (i + 1) % n] else 'I' for j in range(n))
        x = ''.join('X' if j == i else 'I' for j in range(n))
        y = ''.join('Y' if j == i else 'I' for j in range(n))
        h_terms.extend([(z, -1.0), (x, -1.0)])
        jump_terms.append([(x, gamma), (y, -1j * gamma)])
    model = Lindbladian(h_terms, jump_terms)
    ensemble, _, _ = lcu.Lindblad_to_channel(model, delta)
    kraus = ensemble.channels[0][1]
    qc, regs = lcu.channel_to_LCU(channel_ensemble([kraus]), structure='basic', opt='No')
    build_s = time.perf_counter() - begin
    with (OUT / 'circuit.qpy').open('wb') as stream:
        qpy.dump(qc, stream)
    initial = Statevector.from_label('0' * sum(map(len, regs)))
    write('progress.json', {'stage': 'simulating', 'build_time_s': build_s,
                            'circuit_qubits': qc.num_qubits, 'register_sizes': list(map(len, regs)),
                            'simulation_started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('Simulation started:', n, qc.num_qubits, 'qubits', flush=True)
    start = time.perf_counter()
    density = lcu.simulate_circuit(qc, ini_state=initial, qubit_regs=regs)
    elapsed = time.perf_counter() - start
    # Persist the completed timing before untimed numerical checks.
    result = dict(metadata, status='simulation_complete', simulation_time_s=elapsed,
                  build_time_s=build_s, circuit_qubits=qc.num_qubits,
                  register_sizes=list(map(len, regs)), qubit_indexes=regs)
    write('result.json', result)
    rho = np.asarray(density.data)
    # Exact zero-input Kraus action, calculated from Pauli strings without dense operators.
    vectors = []
    for ms in kraus:
        vector = np.zeros(2 ** n, dtype=complex)
        for atom, coeff in ms.instances:
            index, phase = 0, complex(atom.phase)
            for bit, pauli in enumerate(reversed(atom.expr)):
                if pauli in 'XY':
                    index |= 1 << bit
                if pauli == 'Y':
                    phase *= 1j
            vector[index] += coeff * phase
        vectors.append(vector)
    support = np.flatnonzero(np.any(np.abs(vectors) > 1e-14, axis=0))
    reference = sum(np.outer(v[support], v[support].conj()) for v in vectors)
    reference /= np.trace(reference)
    block = rho[np.ix_(support, support)]
    diagonal = np.real(np.diag(rho))
    outside_mass = float(np.sum(diagonal) - np.sum(diagonal[support]))
    block_max_error = float(np.max(np.abs(block - reference)))
    check = {'reference': 'direct Kraus map on zero input, normalized',
             'support_indices': support.tolist(), 'trace': repr(complex(np.trace(rho))),
             'outside_reference_support_diagonal_mass': outside_mass,
             'max_error_on_reference_support': block_max_error,
             'tolerance': 1e-6,
             'passed': block_max_error < 1e-6 and abs(outside_mass) < 1e-6}
    rr, cc = np.nonzero(rho)
    np.savez_compressed(OUT / 'output_density_sparse.npz', row=rr, col=cc,
                        value=rho[rr, cc], shape=np.asarray(rho.shape))
    result.update(status='ok' if check['passed'] else 'validation_failed', validation=check,
                  source_hashes_unchanged={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == v
                                            for p, v in hashes.items()})
    write('result.json', result)
    write('progress.json', {'stage': 'complete', 'status': result['status'], 'simulation_time_s': elapsed})
    print(json.dumps({'simulation_time_s': elapsed, 'validation': check}), flush=True)
    assert check['passed'] and all(result['source_hashes_unchanged'].values())


if __name__ == '__main__':
    main()
