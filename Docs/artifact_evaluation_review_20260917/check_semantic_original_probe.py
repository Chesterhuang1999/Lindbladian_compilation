"""Recheck D11 failing configurations using the original probe simulation path."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Baseline_results/eval_redesign/cp4_rq3_semantic_backend_validation'))
import state_probe_hrw_tfim_backend as p
import numpy as np
OUT=ROOT/'Baseline_results/eval_redesign/artifact_review_20260917/semantic'
p.OUT_JSONL=OUT/'original_probe_events.jsonl'
rows=[]
for b in p.cp4b.build_benchmarks():
    rw,_,_=p.cp4b.rewrite_channel(b.kraus_ops,max_steps=50,tol=1e-12)
    for name,structure,opt in [('O','basic','Matrix-order'),('All','opt','Matrix-order')]:
        cfg=p.cp4b.BackendConfig(name,name=='All',structure,opt,name)
        ops=rw if cfg.use_rewrite else b.kraus_ops
        n=p.cp4b.channel_metrics(ops)['system_qubits']
        rho=np.zeros((2**n,2**n),complex);rho[0,0]=1
        ref,_=p.apply_direct_channel(ops,rho,n)
        qc,regs,_=p.cp4b.build_backend_circuit(b,ops,cfg)
        cand,method=p.simulate_postselected_density(qc,regs)
        cand,_=p.normalize_density(cand)
        row=dict(benchmark=b.name,configuration=name,method=method,**p.diff_metrics(ref,cand))
        rows.append(row);print(json.dumps(row),flush=True)
(OUT/'original_probe_recheck.json').write_text(json.dumps(rows,indent=2)+'\n')
