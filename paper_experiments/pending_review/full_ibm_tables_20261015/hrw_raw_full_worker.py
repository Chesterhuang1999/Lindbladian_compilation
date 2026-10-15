"""Fresh HRW-8/12/28 Raw compilation followed by IBM Qiskit opt3."""
import argparse
from collections import Counter
import math
import os
import platform
import re
import sys
import time
import traceback

import run as batch


def worker(case, host, output_kind):
    assert case in ('HRW-8', 'HRW-12', 'HRW-28')
    sys.path[:0] = [str(batch.SNAP / p) for p in ('src', 'scripts/eval_redesign', 'Baseline_scripts')]
    import qiskit
    from qiskit import QuantumCircuit, qasm2
    from channel_IR import channel_ensemble
    from channel_LCU import channel_to_LCU
    from experiment_protocol import source_identity, transpile_channel_circuit
    from rq1_pipeline import canonicalize, deserialize_kraus, serialize_kraus, verify_channel
    from run_selected_tfim_hrw import sparse_gram_check

    manifest = batch.read(batch.OUT/'manifest.json')
    assert batch.digest(batch.OUT/'run.py') == manifest['runner_sha256']
    assert all(batch.digest(batch.SNAP/p) == h for p,h in manifest['sources'].items())
    sources = source_identity()
    assert sources == batch.read(batch.OUT/'smoke.json')['sources']
    input_path = batch.OUT/'inputs'/f'{case}.json'
    assert batch.digest(input_path) == manifest['cases'][case]['input_sha256']
    entry = batch.read(input_path); expected = entry['Raw']
    assert expected['status'] == 'ok'
    directory = batch.OUT/'cases'/case/output_kind
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory/'results.json').exists()

    def progress(stage):
        batch.dump(directory/'progress.json', dict(stage=stage, updated=time.time(), pid=os.getpid()))

    def lower(circuit, level, basis):
        return transpile_channel_circuit(circuit, basis_gates=basis, optimization_level=level,
            seed_transpiler=20261009, approximation_degree=1.0, qubits_initially_zero=False, num_processes=1)

    target = deserialize_kraus(entry['target'])
    progress('canonicalization'); pipeline_start = time.perf_counter(); start = time.perf_counter()
    canonical, canonical_info = canonicalize(target)
    canonical_s = time.perf_counter()-start
    progress('assembly'); start = time.perf_counter()
    circuit, registers = channel_to_LCU(channel_ensemble([canonical]), structure='basic', opt='No', fstar=False)
    assembly_s = time.perf_counter()-start
    progress('ibm_lowering'); start = time.perf_counter()
    raw = lower(circuit, 0, batch.IBM)
    lowering_s = time.perf_counter()-start
    raw_pipeline_elapsed_s = time.perf_counter()-pipeline_start
    progress('qiskit_opt3'); start = time.perf_counter()
    optimized = lower(raw, 3, batch.IBM)
    optimizer_s = time.perf_counter()-start
    pipeline_elapsed_s = time.perf_counter()-pipeline_start
    raw_s = canonical_s+assembly_s+lowering_s
    progress('validation')
    gram = sparse_gram_check(entry['target'], serialize_kraus(canonical))
    assert gram['passed'], gram
    lam = float(circuit.metadata['subnormalization_factor'])
    assert math.isclose(lam, expected['Lambda'], rel_tol=1e-10)
    progress('nam_reproduction')
    nam = lower(circuit, 0, batch.NAM)
    nam_g = sum(nam.count_ops().get(k,0) for k in batch.NAM)
    assert nam_g == expected['NAM_G'] and nam.num_qubits == expected['W']
    del nam, circuit
    records = {}
    for name, obj, level in [('Raw',raw,0), ('Raw+Qiskit opt3',optimized,3)]:
        assert obj.num_qubits == expected['W']
        if obj.layout is not None:
            assert obj.layout.final_index_layout() == list(range(obj.num_qubits))
        counts = {str(k):int(v) for k,v in obj.count_ops().items()}
        assert set(counts) <= set(batch.IBM) | {'reset','barrier'}
        choi = dict(status='not_run', scope='input coefficient Gram, NAM G/W reproduction, IBM gate set/count/width; opt3 arbitrary-input-safe settings')
        if entry['n'] <= 4 and obj.num_qubits <= 18:
            progress('choi_raw' if level == 0 else 'choi_opt3')
            choi = verify_channel(obj, registers, target)
            assert choi['passed'], choi
        progress('export_raw' if level == 0 else 'export_opt3')
        flat = QuantumCircuit(obj.num_qubits);flat.compose(obj, inplace=True)
        qasm_path = directory/('raw_ibm.qasm' if level == 0 else 'opt3_ibm.qasm')
        qasm2.dump(flat, qasm_path)
        recount, width = Counter(), 0
        for line in qasm_path.open():
            if line.startswith('qreg '): width += int(re.search(r'\[(\d+)\]',line)[1])
            m = re.match(r'^(u1|u2|u3|cx|reset|barrier)(?:\(|\s)',line)
            if m: recount[m[1]] += 1
        assert dict(recount) == counts and width == expected['W']
        record = dict(case=case, method=name, status='ok', host=host, basis='IBM',
            timing_kind='fresh_full_pipeline', saved_raw_reused=False,
            G=sum(counts.get(k,0) for k in batch.IBM), W=width, Lambda=lam, counts=counts,
            canonicalization_s=canonical_s, assembly_s=assembly_s, lowering_s=lowering_s,
            optimizer_s=optimizer_s if level else 0, raw_compile_time_s=raw_s,
            compile_time_s=raw_s+(optimizer_s if level else 0), rewrite_s=0,
            backend_time_s=assembly_s+lowering_s+(optimizer_s if level else 0),
            pipeline_elapsed_s=pipeline_elapsed_s if level else raw_pipeline_elapsed_s,
            time_scope='fresh canonicalization + Raw assembly + IBM opt0 lowering + opt3 for optimized output, all on one host; excludes imports, input deserialization, post-compilation validation, NAM reproduction and export',
            input_sha256=manifest['cases'][case]['input_sha256'], sources=sources,
            worker_sha256=batch.digest(__file__), canonicalization=canonical_info,
            input_map_check=gram, NAM_G=nam_g, NAM_reproduction='passed G and W', full_choi=choi,
            qubits_initially_zero=False, optimization_level=level, seed_transpiler=20261009,
            approximation_degree=1.0,
            qasm_path=str(qasm_path), qasm_sha256=batch.digest(qasm_path), qiskit_version=qiskit.__version__,
            machine=dict(hostname=platform.node(), platform=platform.platform()),
            thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS')})
        records[name] = record
        if level == 0: batch.dump(directory/'raw_results.json', record)
    assert source_identity() == sources
    final = records['Raw+Qiskit opt3']
    final.update(raw_G=records['Raw']['G'], raw_qasm_sha256=records['Raw']['qasm_sha256'])
    batch.dump(directory/'results.json', final)
    progress('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    parser.add_argument('--host', choices=['local','remote'], default='remote')
    parser.add_argument('--output-kind', choices=['raw_full_remote','raw_full_smoke_local'], default='raw_full_remote')
    args = parser.parse_args()
    try:
        worker(args.worker, args.host, args.output_kind)
    except Exception:
        batch.dump(batch.OUT/'cases'/args.worker/args.output_kind/'failure.json',
                   dict(status='failed', traceback=traceback.format_exc(), updated=time.time()))
        raise
