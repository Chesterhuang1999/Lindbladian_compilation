"""Cross-host backend check on the frozen code, without benchmark reruns."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_NUM_PROCS'):
    os.environ[k]='1'
BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'source_snapshot'/p) for p in ('src','scripts/eval_redesign')]
from experiment_protocol import source_identity,transpile_channel_circuit
from rq1_pipeline import deserialize_kraus,verify_channel
from channel_IR import channel_ensemble
from channel_LCU import channel_to_LCU
encoded=[[['II',.6,1,0],['XZ',.2,0,1],['YX',.1,-1,0]],
         [['ZI',.3,0,1],['IX',.1,-1,0]],[['YY',.15,0,1]]]
assert source_identity()==json.loads((BASE/'smoke.json').read_text())['sources']
rows=[]
for structure,opt in [('basic','No'),('tree','Matrix-order')]:
    raw=deserialize_kraus(encoded)
    circuit,registers=channel_to_LCU(channel_ensemble([raw]),structure=structure,opt=opt,fstar=structure=='tree')
    for basis in (['rz','h','x','cx'],['u1','u2','u3','cx']):
        lowered=transpile_channel_circuit(circuit,basis_gates=basis,optimization_level=0,seed_transpiler=20261009)
        check=verify_channel(lowered,registers,deserialize_kraus(encoded))
        assert check['passed'],check
        rows.append({'structure':structure,'basis':basis,'counts':dict(lowered.count_ops()),'W':lowered.num_qubits,'verification':check})
versions={k:md.version(k) for k in json.loads((BASE/'environment_local.json').read_text())}
expected=json.loads((BASE/'environment_local.json').read_text())
assert all(versions[k]==v for k,v in expected.items() if k!='dill'),(versions,expected)
host=sys.argv[1]
if host=='remote':
    local=json.loads((BASE/'preflight_local.json').read_text())['rows']
    assert [(r['counts'],r['W']) for r in rows]==[(r['counts'],r['W']) for r in local]
result={'passed':True,'rows':rows,'versions':versions,'python':sys.version,'sources':source_identity(),
        'dill_version_evidence':'local metadata absent; pin from prior environment archive'}
(BASE/f'preflight_{host}.json').write_text(json.dumps(result,indent=2)+'\n')
print(host,'preflight passed',flush=True)
