"""Resolve a small historical NAM-count mismatch with full channel validation."""
import json
import os
from pathlib import Path
import sys
import time
BASE=Path(__file__).resolve().parent
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(BASE/'.runtime/mpl')
sys.path[:0]=[str(BASE/'source_snapshot'/p) for p in ('src','Baseline_scripts')]
from qiskit import qasm2,QuantumCircuit
from rq1_pipeline import build_pipeline,deserialize_kraus,verify_channel,canonicalize
from experiment_protocol import transpile_channel_circuit
from baseline_common import count_qasm_file
from run import dump,digest,saved,IBM,NAM
case='Petz_TFIM2';entry=json.loads((BASE/'inputs'/f'{case}.json').read_text());info=entry['Raw']
assert not saved(case,'Raw').exists() and not saved(case,'Raw+Qiskit opt3').exists()
diag=BASE/'diagnostics/Petz_TFIM2_local.json';assert json.loads(diag.read_text())['verification']['passed']
start=time.perf_counter();canonicalize(deserialize_kraus(entry['target']));canonical_s=time.perf_counter()-start
start=time.perf_counter();high,regs,_=build_pipeline(deserialize_kraus(info['encoded']),'Raw');assembly_s=time.perf_counter()-start
def lower(circuit,level,basis):return transpile_channel_circuit(circuit,basis_gates=basis,optimization_level=level,
    qubits_initially_zero=False,approximation_degree=1.0,seed_transpiler=20261009,num_processes=1)
nam=lower(high,0,NAM);nam_counts=dict(nam.count_ops());del nam
start=time.perf_counter();raw=lower(high,0,IBM);lowering_s=time.perf_counter()-start;del high
for method,circuit,opt_s in [('Raw',raw,0)]:
    pass
def save(method,circuit,opt_s):
    check=verify_channel(circuit,regs,deserialize_kraus(entry['target']));assert check['passed'],check
    assert circuit.num_qubits==info['W']
    if circuit.layout is not None:assert circuit.layout.final_index_layout()==list(range(info['W']))
    counts=dict(circuit.count_ops());assert not set(counts)-set(IBM)
    out=saved(case,method);out.parent.mkdir(parents=True,exist_ok=True);qasm=out.parent/'channel_ibm_validated_recovery.qasm'
    flat=QuantumCircuit(circuit.num_qubits);flat.compose(circuit,inplace=True);qasm2.dump(flat,qasm)
    rec=count_qasm_file(qasm,gate_set='ibm');assert rec['metric_total']==sum(counts.values()) and rec['num_qubits']==info['W']
    result={'case':case,'method':method,'status':'ok','basis':'IBM','host':'local','G':rec['metric_total'],
        'W':info['W'],'Lambda':info['Lambda'],'counts':counts,'compile_time_s':canonical_s+assembly_s+lowering_s+opt_s,
        'backend_time_s':assembly_s+lowering_s+opt_s,'canonicalization_s':canonical_s,'assembly_s':assembly_s,
        'lowering_s':lowering_s,'optimizer_s':opt_s,'time_scope':'fresh canonicalization + assembly + IBM opt0 + optional opt3; excludes NAM audit and full Choi validation',
        'input_sha256':digest(BASE/'inputs'/f'{case}.json'),'original_source':info['source'],'original_source_sha256':info['source_sha256'],
        'NAM_reproduction':'counts differ; accepted only after full unnormalized Choi verification of each IBM output',
        'reference_NAM_G':info['NAM_G'],'observed_NAM_counts':nam_counts,'full_choi':check,
        'recovery_reason':'small NAM count difference on serialized-input reassembly, also reproduced locally; not a resource-limit retry',
        'prior_failure':str(BASE/'cases/Petz_TFIM2/raw/failure.json'),'recovery_runner_sha256':digest(__file__),
        'qasm_path':str(qasm),'qasm_sha256':digest(qasm),'qubits_initially_zero':False,'optimization_level':3 if opt_s else 0,'seed_transpiler':20261009}
    dump(out,result);print(method,result['G'],check,flush=True)
save('Raw',raw,0)
start=time.perf_counter();optimized=lower(raw,3,IBM);opt_s=time.perf_counter()-start
save('Raw+Qiskit opt3',optimized,opt_s)
