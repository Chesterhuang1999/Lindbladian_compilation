"""D11-D13: isolated rerun of paper parameters, corrected O/F labels, state probes only."""
from pathlib import Path
import sys, json, time, traceback, hashlib, platform, argparse
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Baseline_results/eval_redesign/artifact_review_20260917/semantic'
sys.path[:0] = [str(ROOT/'Baseline_results/eval_redesign/cp4_rq3_semantic_backend_validation'), str(ROOT/'scripts')]
import numpy as np
import qiskit
import run_cp4b_semantic_backend_validation as cp
import random_coloring_no_approx_zero_check as probe
from run_random_coloring_channel_lcu_ours_nam import build_random_coloring_channel

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixed-residual',action='store_true')
    args=parser.parse_args()
    if args.fixed_residual:
        import block_encoding_new
        from mobius_residual_fixed import mobius_invert_modes_with_phases
        block_encoding_new.mobius_invert_modes_with_phases=mobius_invert_modes_with_phases
        OUT=OUT.parent/'semantic_fixed'
    OUT.mkdir(parents=True, exist_ok=True)
    benches = cp.build_benchmarks()
    ops, model = build_random_coloring_channel(2, 'linear', q=2, tol=1e-12,
        max_total_pauli_terms=None, max_terms_per_kraus=None, color_mixing_preprocess=False)
    benches.append(cp.Benchmark('random_coloring_n2_q2_linear', 'random_coloring', ops, model))
    configs = [cp.BackendConfig('Basic',False,'basic','No','basic,No'),
        cp.BackendConfig('R',True,'basic','No','rewrite+basic,No'),
        cp.BackendConfig('O',False,'basic','Matrix-order','ordering'),
        cp.BackendConfig('F',False,'opt','No','flattening'),
        cp.BackendConfig('All',True,'opt','Matrix-order','rewrite+flattening+ordering')]
    result = {'python':sys.version,'qiskit':qiskit.__version__,'platform':platform.platform(),'fixed_residual':args.fixed_residual,
        'input_state':'zero_density','approx_state':False,'threshold':1e-10,
        'note':'state sanity checks, not a full-channel equivalence assertion', 'rows':[], 'rewrite_rows':[]}
    for b in benches:
        rw, info, elapsed = cp.rewrite_channel(b.kraus_ops,max_steps=50,tol=1e-12)
        diff=cp.diff_metrics(cp.channel_choi(b.kraus_ops),cp.channel_choi(rw))
        result['rewrite_rows'].append(dict(benchmark=b.name,**diff,elapsed_s=elapsed))
        for c in configs:
            started=time.perf_counter()
            row={'benchmark':b.name,'configuration':c.name,'structure':c.structure,'opt':c.opt,'rewrite':c.use_rewrite}
            try:
                target=rw if c.use_rewrite else b.kraus_ops
                n=cp.channel_metrics(target)['system_qubits']
                circuit,regs,build_s=cp.build_backend_circuit(b,target,c)
                reference=probe.direct_density(target,n)
                candidate=probe.sim_no_approx(circuit,regs)
                td,fro,maxe=probe.metrics(reference,candidate)
                row.update(status='ok' if td<=1e-10 else 'discrepancy',trace_distance=td,
                    frobenius=fro,max_entry=maxe,circuit_qubits=circuit.num_qubits,build_s=build_s)
            except Exception:
                row.update(status='failed',error=traceback.format_exc())
            row['elapsed_s']=time.perf_counter()-started
            result['rows'].append(row)
            (OUT/'summary.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
            print(json.dumps(row),flush=True)
if __name__=='__main__': main()
