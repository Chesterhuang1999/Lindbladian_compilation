"""Compare fresh raw/ours IBM circuits and explicit Aer simulation methods."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Baseline_results/simulation_tfim8_hrw20_20260911'

def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', type=Path)
    parser.add_argument('--method', default='matrix_product_state')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from qiskit import qpy, qasm2, transpile
    import qiskit, qiskit_aer, numpy
    if args.simulate:
        with (args.simulate / 'circuit.qpy').open('rb') as f:
            circuit = qpy.load(f)[0]
        circuit.measure_all()
        options = dict(method=args.method, max_parallel_threads=1, enable_truncation=False,
                       precision='double', max_memory_mb=4096)
        if args.method == 'matrix_product_state':
            options.update(matrix_product_state_truncation_threshold=1e-16, mps_log_data=True)
        sim = qiskit_aer.AerSimulator(**options)
        records = []
        for repeat in range(3):
            start = time.perf_counter()
            result = sim.run(circuit, shots=1, seed_simulator=7).result()
            elapsed = time.perf_counter() - start
            records.append(dict(time_s=elapsed, success=result.success, status=result.status,
                                metadata=result.results[0].metadata,
                                counts=result.get_counts() if result.success else None))
            save(args.simulate / (args.method + '.json'), dict(options=options, input='all_zero',
                 shots=1, seed=7, repeats=records))
            if not result.success:
                break
        return
    import run_tfim4_channel_lcu_ours_nam as tfim
    import run_hypercube_n4_ours_nam as hrw
    sys.path.insert(0, str(ROOT / 'Baseline_scripts'))
    from baseline_common import count_qasm_file
    if args.resume:
        return simulate_all()
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / 'provenance.json', dict(python=sys.version, qiskit=qiskit.__version__,
         aer=qiskit_aer.__version__, numpy=numpy.__version__, environment={k:v for k,v in os.environ.items() if k.startswith('CHANNEL_LCU_') or k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
         source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__), Path(tfim.__file__), Path(hrw.__file__), *sorted((ROOT/'src').glob('*.py'))]}))
    for family,n,mod in [('tfim',8,tfim), ('hrw',20,hrw)]:
        for variant in ('raw','ours'):
            folder = OUT / f'{family}_{n}_{variant}'
            folder.mkdir()
            t = time.perf_counter()
            kraus = mod.build_periodic_tfim_channel(n,0.1)[0] if family=='tfim' else mod.build_hypercube_channel(n)
            model_s = time.perf_counter()-t
            rewrite_s=0
            if variant=='ours':
                kraus,_,rewrite_s=mod.rewrite_channel(kraus,strategy='greedy_incremental',beam_width=3,max_steps=400,tol=1e-12)
                qc,indexes,build_s=mod.build_rewrite_opt_matrix_order_circuit(kraus)
            else:
                qc,indexes,build_s=mod.build_channel_lcu_circuit(kraus,structure='basic',opt='No')
            t=time.perf_counter()
            qc=transpile(qc,basis_gates=['u1','u2','u3','cx','reset'],optimization_level=0,seed_transpiler=7)
            lowering_s=time.perf_counter()-t
            with (folder/'circuit.qpy').open('wb') as f: qpy.dump(qc,f)
            qasm2.dump(qc,str(folder/'circuit.qasm'))
            stats=count_qasm_file(folder/'circuit.qasm',gate_set='ibm')
            record=dict(family=family,n=n,variant=variant,width=qc.num_qubits,indexes=indexes,
                        model_time_s=model_s,rewrite_time_s=rewrite_s,build_time_s=build_s,
                        lowering_time_s=lowering_s,compile_time_s=rewrite_s+build_s+lowering_s,
                        ibm_gates=stats['metric_total'],stats=stats,statevector_bytes=16*2**qc.num_qubits)
            save(folder/'compile.json',record)
            print(json.dumps({k:v for k,v in record.items() if k not in ('stats','indexes')}),flush=True)
    simulate_all()

def simulate_all():
    for method in ('matrix_product_state','statevector'):
        for folder in sorted(OUT.glob('*_raw'))+sorted(OUT.glob('*_ours')):
            with (folder/(method+'.log')).open('w') as log:
                start=time.perf_counter()
                try:
                    p=subprocess.run([sys.executable,__file__,'--simulate',str(folder),'--method',method],stdout=log,stderr=subprocess.STDOUT,timeout=180)
                    status=dict(returncode=p.returncode,wall_s=time.perf_counter()-start)
                except subprocess.TimeoutExpired:
                    status=dict(status='timeout',limit_s=180)
                save(folder/(method+'_process.json'),status)
            print(folder.name,method,status,flush=True)

if __name__=='__main__': main()
