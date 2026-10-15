"""Exact consecutive phase folding of saved D14 circuits; does not alter compiler."""
import json,sys
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit,qasm2
from qiskit.quantum_info import Operator
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Baseline_results/eval_redesign/artifact_review_20260917/clifford'
def fold(qc,allow_dagger):
    new=qc.copy_empty_like();pending={}
    def flush(q):
        k=pending.pop(q,0)%8
        if allow_dagger and k>4:
            k-=8
            for _ in range((-k)//2):new.sdg(q)
            if (-k)%2:new.tdg(q)
        else:
            for _ in range(k//2):new.s(q)
            if k%2:new.t(q)
    for item in qc.data:
        name=item.operation.name
        if name in {'s','sdg','t','tdg','z'}:
            q=item.qubits[0];pending[q]=pending.get(q,0)+dict(s=2,sdg=-2,t=1,tdg=-1,z=4)[name]
        else:
            for q in item.qubits:flush(q)
            new.append(item.operation,item.qubits,item.clbits)
    for q in list(pending):flush(q)
    return new
rows=[]
for n in [3,4,6,8]:
    path=OUT/f'full_pauli_n{n}_select_previous_basis.qasm'
    if not path.exists():continue
    qc=qasm2.load(str(path));row={'n':n}
    for dagger in [False,True]:
        new=fold(qc,dagger);key='exact_h_cx_t_s' if not dagger else 'dagger_counted_with_parent'
        row[key]={'depth':new.depth(),'counts':dict(new.count_ops()),'total':sum(new.count_ops().values())}
        qasm2.dump(new,str(OUT/f'full_pauli_n{n}_{key}.qasm'))
        # The family factors into disjoint 3-qubit blocks. Validate every block,
        # including its complete 8x8 operator, rather than a full 24-qubit matrix.
        errs=[]
        for i in range(n):
            ids=[2*i,2*i+1,2*n+i]
            def block(c):
                b=QuantumCircuit(3)
                for it in c.data:
                    inds=[c.find_bit(q).index for q in it.qubits]
                    if set(inds)&set(ids):
                        assert set(inds)<=set(ids)
                        b.append(it.operation,[ids.index(q) for q in inds])
                return b
            errs.append(float(np.max(np.abs(Operator(block(qc)).data-Operator(block(new)).data))))
        row[key]['max_block_operator_diff']=max(errs)
        row[key]['passed']=max(errs)<1e-10
    rows.append(row)
(OUT/'phase_fold_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
