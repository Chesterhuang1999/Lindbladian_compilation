"""Fresh HRW Ours compilation using the existing immutable compiler snapshot.

Results are separate from saved-rewrite backend reproductions. No recorded
rewrite time or rewritten representation is used as a compilation input.
"""
import argparse
from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

import run as batch


def worker(case, host, output_kind):
    assert re.fullmatch(r'HRW-\d+', case)
    sys.path[:0] = [str(batch.SNAP / p) for p in ('src', 'scripts/eval_redesign', 'Baseline_scripts')]
    import qiskit
    from qiskit import QuantumCircuit, qasm2
    from block_encoding import BlockEncoding
    from channel_IR import channel as ChannelIR, channel_ensemble
    from channel_LCU import channel_to_LCU
    from experiment_protocol import source_identity, transpile_channel_circuit
    from rq1_pipeline import canonicalize, deserialize_kraus, serialize_kraus, verify_channel
    from run_selected_tfim_hrw import sparse_gram_check

    manifest = batch.read(batch.OUT / 'manifest.json')
    assert batch.digest(batch.OUT / 'run.py') == manifest['runner_sha256']
    assert all(batch.digest(batch.SNAP / p) == h for p, h in manifest['sources'].items())
    sources = source_identity()
    assert sources == batch.read(batch.OUT / 'smoke.json')['sources']
    input_path = batch.OUT / 'inputs' / f'{case}.json'
    assert batch.digest(input_path) == manifest['cases'][case]['input_sha256']
    entry = batch.read(input_path)
    expected = entry['Ours']
    policy = expected['rewrite_policy']
    assert policy['rewrite_method'] == 'rewrite_search'
    directory = batch.OUT / 'cases' / case / output_kind
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / 'results.json').exists()

    def progress(stage):
        batch.dump(directory / 'progress.json', dict(stage=stage, updated=time.time(), pid=os.getpid()))

    def lower(circuit, basis):
        return transpile_channel_circuit(circuit, basis_gates=basis, optimization_level=0,
            seed_transpiler=20261009, approximation_degree=1.0,
            qubits_initially_zero=False, num_processes=1)

    # Loading/deserialization is outside the four-stage compilation timer.
    raw = deserialize_kraus(entry['target'])
    progress('canonicalization')
    pipeline_start = time.perf_counter()
    start = time.perf_counter()
    canonical, canonical_info = canonicalize(raw)
    canonical_s = time.perf_counter() - start
    progress('rewrite')
    start = time.perf_counter()
    source = ChannelIR(deepcopy(canonical))
    parameters = {k: policy[k] for k in ('strategy', 'beam_width', 'max_steps', 'tol')}
    rewrite = source.rewrite_search(**parameters)
    source.apply_rewrite_result(rewrite, tol=policy['tol'])
    source.zero_elim()
    kraus = source.kraus_ops
    rewrite_s = time.perf_counter() - start
    progress('assembly')
    start = time.perf_counter()
    backend = lambda ms: BlockEncoding(ms, placement_options={}, matrix_order_options=expected['matrix_order_options'])
    circuit, registers = channel_to_LCU(channel_ensemble([deepcopy(kraus)]),
        structure='tree', opt='Matrix-order', fstar=True, block_encoding_cls=backend)
    assembly_s = time.perf_counter() - start
    progress('ibm_lowering')
    start = time.perf_counter()
    ibm = lower(circuit, batch.IBM)
    lowering_s = time.perf_counter() - start
    pipeline_elapsed_s = time.perf_counter() - pipeline_start
    stages = dict(canonicalization_s=canonical_s, rewrite_s=rewrite_s,
                  assembly_s=assembly_s, lowering_s=lowering_s)
    compile_s = sum(stages.values())

    # Independent semantic and reproduction checks follow the timed pipeline.
    progress('validation')
    rewritten = serialize_kraus(kraus)
    gram = sparse_gram_check(entry['target'], rewritten)
    assert gram['passed'], gram
    termination = rewrite.get('termination') or {}
    assert termination.get('stop_reason') != 'max_steps_reached', termination
    lam = float(circuit.metadata['subnormalization_factor'])
    assert math.isclose(lam, expected['Lambda'], rel_tol=1e-10)
    assert ibm.num_qubits == expected['W']
    counts = {str(k): int(v) for k, v in ibm.count_ops().items()}
    assert set(counts) <= set(batch.IBM) | {'reset', 'barrier'}
    if ibm.layout is not None:
        assert ibm.layout.final_index_layout() == list(range(ibm.num_qubits))
    progress('nam_reproduction')
    nam = lower(circuit, batch.NAM)
    nam_g = sum(nam.count_ops().get(k, 0) for k in batch.NAM)
    assert nam_g == expected['NAM_G'] and nam.num_qubits == expected['W']
    choi = dict(status='not_run', scope='fresh rewrite Pauli coefficient Gram; NAM G/W reproduction; IBM gate set/count/width')
    if entry['n'] <= 4 and ibm.num_qubits <= 18:
        choi = verify_channel(ibm, registers, raw)
        assert choi['passed'], choi
    assert source_identity() == sources
    progress('export')
    flat = QuantumCircuit(ibm.num_qubits)
    flat.compose(ibm, inplace=True)
    qasm_path = directory / 'channel_ibm.qasm'
    qasm2.dump(flat, qasm_path)
    recount, width = Counter(), 0
    for line in qasm_path.open():
        if line.startswith('qreg '):
            width += int(re.search(r'\[(\d+)\]', line)[1])
        op = re.match(r'^(u1|u2|u3|cx|reset|barrier)(?:\(|\s)', line)
        if op:
            recount[op[1]] += 1
    assert dict(recount) == counts and width == expected['W']
    batch.dump(directory / 'compiled_kraus.json', dict(case=case, compiled_kraus=rewritten,
        rewrite_policy=policy, termination=termination, map_check=gram))
    result = dict(case=case, method='Ours', status='ok', host=host, basis='IBM',
        timing_kind='fresh_full_pipeline', rewrite_reused=False,
        G=sum(counts.get(k, 0) for k in batch.IBM), NAM_G=nam_g, W=width, Lambda=lam,
        counts=counts, compile_time_s=compile_s, **stages,
        pipeline_elapsed_s=pipeline_elapsed_s,
        pipeline_bookkeeping_s=pipeline_elapsed_s-compile_s,
        time_scope='fresh canonicalization + fresh rewrite + assembly + IBM opt0 lowering on one host; excludes imports, input deserialization, validation, NAM reproduction and export',
        rewrite_policy=policy, rewrite_termination=termination, canonicalization=canonical_info,
        input_sha256=manifest['cases'][case]['input_sha256'], sources=sources,
        worker_sha256=batch.digest(__file__), original_source=expected['source'],
        original_source_sha256=expected['source_sha256'], original_source_role='reproduction reference only; not a timing or rewrite input',
        rewrite_map_check=gram, NAM_reproduction='passed G and W', full_choi=choi,
        qubits_initially_zero=False, optimization_level=0, seed_transpiler=20261009,
        qasm_path=str(qasm_path), qasm_sha256=batch.digest(qasm_path),
        compiled_kraus_sha256=batch.digest(directory/'compiled_kraus.json'),
        qiskit_version=qiskit.__version__, machine=dict(hostname=platform.node(), platform=platform.platform()),
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS')})
    batch.dump(directory / 'results.json', result)
    progress('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    parser.add_argument('--host', choices=['local', 'remote'], default='remote')
    parser.add_argument('--output-kind', default='ours_full_remote')
    args = parser.parse_args()
    assert args.output_kind in ('ours_full_remote', 'ours_full_smoke_local')
    try:
        worker(args.worker, args.host, args.output_kind)
    except Exception:
        batch.dump(batch.OUT/'cases'/args.worker/args.output_kind/'failure.json',
                   dict(status='failed', traceback=traceback.format_exc(), updated=time.time()))
        raise
