"""D11 layer isolation using identical inputs/placement, with an in-memory residual swap."""
from pathlib import Path
import sys,json,copy,os,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Baseline_results/eval_redesign/cp4_rq3_semantic_backend_validation'),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from qiskit.quantum_info import Operator,Pauli
import run_cp4b_semantic_backend_validation as cp
import random_coloring_no_approx_zero_check as probe
import block_encoding_new as be
from mobius_residual_fixed import mobius_invert_modes_with_phases as fixed
OUT=ROOT/'Baseline_results/eval_redesign/artifact_review_20260917/semantic'
old=be.mobius_invert_modes_with_phases
result={'env':{k:v for k,v in os.environ.items() if k.startswith('CHANNEL_LCU_')},'rewrite':{'strategy':'greedy','beam_width':3,'max_steps':50,'tol':1e-12},'blocks':[],'states':[]}
def save():(OUT/'pipeline_diagnosis.json').write_text(json.dumps(result,indent=2,default=cp.json_default)+'\n')
for b in cp.build_benchmarks():
    rw,_,_=cp.rewrite_channel(b.kraus_ops,max_steps=50,tol=1e-12)
    for label,ops in [('raw',b.kraus_ops),('rewritten',rw)]:
        for idx,ms in enumerate(ops):
            for variant,fn in [('current',old),('residual_only_fixed',fixed)]:
                be.mobius_invert_modes_with_phases=fn
                obj=be.BlockEncoding(copy.deepcopy(ms));qc=obj.circuit(opt='Matrix-order')
                sysids=[qc.find_bit(q).index for reg in qc.qregs if reg.name=='sys' for q in reg]
                n=ms.size;inds=[sum(((v>>j)&1)<<q for j,q in enumerate(sysids)) for v in range(2**n)]
                U=Operator(qc).data;block=U[np.ix_(inds,inds)]
                target=ms.eff_op().to_matrix()/ms.pauli_norm()
                overlap=np.vdot(target,block);phase=overlap/abs(overlap) if abs(overlap)>1e-14 else 1
                diff=float(np.max(np.abs(block-phase*target)))
                row=dict(benchmark=b.name,representation=label,kraus=idx,variant=variant,system_qubits=n,
                    control_width=obj.selected_k,block_width=qc.num_qubits,registers=[(r.name,len(r)) for r in qc.qregs],
                    max_block_diff_up_to_global_phase=diff,passed=diff<1e-10,
                    placement=obj.placement_search_debug,modes=obj.mobius_phase_result['g_modes_with_phase'],
                    coeff_modes=obj.coeff_mode_dict)
                result['blocks'].append(row);save()
    for variant,fn in [('current',old),('residual_only_fixed',fixed)]:
        be.mobius_invert_modes_with_phases=fn
        for name,structure,opt in [('O','basic','Matrix-order'),('All','opt','Matrix-order')]:
            cfg=cp.BackendConfig(name,name=='All',structure,opt,name)
            ops=rw if cfg.use_rewrite else b.kraus_ops
            qc,regs,_=cp.build_backend_circuit(b,ops,cfg)
            ref=probe.direct_density(ops,ops[0].size);cand=probe.sim_no_approx(qc,regs)
            td,fro,maxe=probe.metrics(ref,cand)
            row=dict(benchmark=b.name,configuration=name,variant=variant,trace_distance=td,passed=td<1e-10,
                channel_builder='build_rewrite_opt_matrix_order_circuit + mulplex_U_opt' if name=='All' else 'channel_to_LCU(structure=basic,opt=Matrix-order)',registers=regs)
            result['states'].append(row);save();print(json.dumps(row),flush=True)
be.mobius_invert_modes_with_phases=old
