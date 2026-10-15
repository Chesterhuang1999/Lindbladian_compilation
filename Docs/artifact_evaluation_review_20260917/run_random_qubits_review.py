"""Review-only RQ2 runner: public tuple is (Kraus count, qubits, first terms, other terms)."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/eval_redesign'))
import run_cp2_random_ablation as ab
p=argparse.ArgumentParser()
p.add_argument('--kraus-count',type=int,default=5)
p.add_argument('--num-qubits',type=int,default=4)
p.add_argument('--first-kraus-terms',type=int,default=10)
p.add_argument('--other-kraus-terms',type=int,default=2)
p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
p.add_argument('--residual-implementation',choices=['current','fixed'],default='fixed')
p.add_argument('--pack-block-registers',action='store_true')
p.add_argument('--out-dir',type=Path,default=ROOT/'Baseline_results/eval_redesign/artifact_review_20260917/random_qubits4_fixed')
x=p.parse_args()
assert x.num_qubits>0
if x.residual_implementation=='fixed':
    import block_encoding_new
    from mobius_residual_fixed import mobius_invert_modes_with_phases
    block_encoding_new.mobius_invert_modes_with_phases=mobius_invert_modes_with_phases
if x.pack_block_registers:
    assert x.residual_implementation=='fixed'
    from review_backend_adapter import install
    install()
args=argparse.Namespace(example='random-pauli',out_dir=x.out_dir,num_vertices=4,q=4,graph='linear',
    kraus_count=x.kraus_count,hilbert_dim=2**x.num_qubits,first_kraus_terms=x.first_kraus_terms,
    other_kraus_terms=x.other_kraus_terms,seed=None,seeds=x.seeds,beam_width=8,max_steps=50,tol=1e-12,transpile_optimization_level=0)
x.out_dir.mkdir(parents=True,exist_ok=True)
(x.out_dir/'public_input.json').write_text(json.dumps(dict(kraus_count=x.kraus_count,num_qubits=x.num_qubits,
    hilbert_dim=args.hilbert_dim,first_kraus_terms=x.first_kraus_terms,other_kraus_terms=x.other_kraus_terms,
    seeds=x.seeds,argv=sys.argv,residual_implementation=x.residual_implementation,pack_block_registers=x.pack_block_registers,
    interface='qubit count; dimension is computed internally'),indent=2)+'\n')
norms=[]
for seed in x.seeds:
    args.seed=seed
    raw,metrics=ab.ensure_raw_cache(args,seed)
    assert metrics['system_qubits']==x.num_qubits,metrics
    rewritten,_=ab.ensure_rewrite_cache(args,seed)
    norms.append(dict(seed=seed,raw=sum(ms.pauli_norm()**2 for ms in raw),rewritten=sum(ms.pauli_norm()**2 for ms in rewritten),system_qubits=metrics['system_qubits']))
    (x.out_dir/'subnormalization.json').write_text(json.dumps(norms,indent=2)+'\n')
    for cfg in ab.CONFIGS:
        ab.run_config(args,cfg,seed)
ab.aggregate(args)
