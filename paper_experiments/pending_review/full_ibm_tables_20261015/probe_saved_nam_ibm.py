"""Measure saved-NAM conversion separately from the formal direct IBM route."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS'):
    os.environ[key] = '1'
os.environ['QISKIT_PARALLEL'] = 'FALSE'
import sys
import time
from pathlib import Path
import run as batch
sys.path.insert(0, str(batch.SNAP / 'src'))
from experiment_protocol import transpile_channel_circuit, source_identity
from qiskit import qasm2

assert batch.read(batch.OUT / 'smoke.json')['sources'] == source_identity()
output = batch.OUT / 'diagnostics/saved_nam_to_ibm'
for case in ('TFIM-4', 'TFIM-54'):
    info = batch.read(batch.OUT / 'inputs' / f'{case}.json')['Ours']
    original = Path(info['source'])
    assert batch.digest(original) == info['source_sha256']
    record = batch.read(original)
    qasm = original.parent / 'channel_nam.qasm'
    assert batch.digest(qasm) == record['qasm_sha256']
    circuit = qasm2.load(qasm)
    assert dict(circuit.count_ops()) == record['counts']
    assert circuit.num_qubits == info['W']
    start = time.perf_counter()
    converted = transpile_channel_circuit(circuit, basis_gates=batch.IBM,
        optimization_level=0, seed_transpiler=20261009, approximation_degree=1.0,
        qubits_initially_zero=False, num_processes=1)
    duration = time.perf_counter()-start
    counts = dict(converted.count_ops())
    assert not set(counts)-set(batch.IBM)-{'reset','barrier'}
    dest = output / case
    dest.mkdir(parents=True, exist_ok=True)
    qasm2.dump(converted, dest / 'channel_ibm_from_nam.qasm')
    direct_path = batch.saved(case, 'Ours')
    direct = batch.read(direct_path) if direct_path.exists() else None
    result = {'case':case, 'status':'ok', 'route':'saved NAM QASM -> IBM opt0',
        'source':str(qasm), 'source_sha256':batch.digest(qasm), 'NAM_G':info['NAM_G'],
        'G':sum(counts.get(g,0) for g in batch.IBM), 'counts':counts,
        'W':converted.num_qubits, 'Lambda':info['Lambda'], 'conversion_s':duration,
        'direct_IBM_G':direct['G'] if direct else None,
        'formal_table_eligible':False,
        'reason':'additional NAM decomposition before IBM; not the formal direct high-level-to-IBM route',
        'output_sha256':batch.digest(dest/'channel_ibm_from_nam.qasm'),
        'verification':'archived source hash, NAM histogram, width, output basis; full Choi not run'}
    batch.dump(dest/'results.json',result)
    print(case, result['G'], result['direct_IBM_G'], duration, flush=True)
