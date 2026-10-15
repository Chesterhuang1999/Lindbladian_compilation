"""D14: SELECT-only Clifford+T rerun with exactly h,cx,t,s; no source mutations."""
from pathlib import Path
import sys,json,time,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Baseline_scripts'),str(ROOT/'scripts'),str(ROOT/'src')]
from qiskit import transpile,qasm2
from qiskit.quantum_info import Operator
import numpy as np
import qiskit
from probe_full_pauli_n6_select_clifford_t import BlockEncoding,expand_large_controlled_gates,full_pauli_matrixsum,CLIFFORD_T_BASIS
OUT=ROOT/'Baseline_results/eval_redesign/artifact_review_20260917/clifford'
def stats(qc): return dict(width=qc.num_qubits,depth=qc.depth(),gates=dict(qc.count_ops()),total=sum(qc.count_ops().values()))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result={'qiskit':qiskit.__version__,'basis':['h','cx','t','s'],'optimization_level':0,'rows':[]}
    for n in [3,4,6,8]:
        started=time.perf_counter()
        row={'n':n,'term_count':4**n}
        try:
            be=BlockEncoding(full_pauli_matrixsum(n))
            raw,*_=be.mulplex_U_opt_order()
            expanded=expand_large_controlled_gates(raw,if_optimize=True)
            old=transpile(expanded,basis_gates=CLIFFORD_T_BASIS,optimization_level=0)
            new=transpile(expanded,basis_gates=['h','cx','t','s'],optimization_level=0)
            row.update(status='ok',previous_basis=stats(old),requested_basis=stats(new),selected_k=getattr(be,'selected_k',None))
            # Validate every distinct local gate against its actual basis translation.
            local=[]
            seen=set()
            for ci in old.data:
                gate=ci.operation
                key=(gate.name,tuple(map(str,gate.params)),gate.num_qubits)
                if key in seen: continue
                seen.add(key)
                from qiskit import QuantumCircuit
                qc=QuantumCircuit(gate.num_qubits); qc.append(gate,range(gate.num_qubits))
                lowered=transpile(qc,basis_gates=['h','cx','t','s'],optimization_level=0)
                err=float(np.max(np.abs(Operator(qc).data-Operator(lowered).data)))
                local.append({'gate':gate.name,'max_operator_diff':err})
            row['local_translation_checks']=local
            row['translation_pass']=all(v['max_operator_diff']<1e-10 for v in local)
            qasm2.dump(new,str(OUT/f'full_pauli_n{n}_select_h_cx_t_s.qasm'))
            qasm2.dump(old,str(OUT/f'full_pauli_n{n}_select_previous_basis.qasm'))
        except Exception: row.update(status='failed',error=traceback.format_exc())
        row['elapsed_s']=time.perf_counter()-started
        result['rows'].append(row)
        (OUT/'summary.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps(row),flush=True)
if __name__=='__main__': main()
