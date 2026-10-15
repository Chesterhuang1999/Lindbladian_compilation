"""Historical multi-seed probe, retained only for CPTP2_M4_seed4."""
import argparse
import json
import time
import numpy as np
from probe import HERE, input_data, archived_search, support, gate, H, TOL, save_state, compile_output

parser=argparse.ArgumentParser()
parser.add_argument('--case',choices=['CPTP2_M4_seed4'],required=True)
name=parser.parse_args().case
A0, labels, source = input_data(name)
seen = {}
def observe(layer, beam):
    for state in beam:
        key = support.state_key(state.keys)
        seen.setdefault(key, (H(state.A), state, layer))
pure, term = archived_search(A0,None,width=8,max_steps=256,timeout=30,rank_mode="support_rank",neutral=True,on_layer=observe)
best_eight = sorted(seen.values(), key=lambda x:(x[0],x[1].support))[:8]
out = {"input":source,"support_termination":term,"unique_committed_states":len(seen),"seeds":[]}
gate.search(np.array([[1,1j],[1,-1j]],complex),None,max_steps=1,mode="admit_support",neutral=True)
best = None
total_polish = 0
for h, seed, layer in best_eight:
    start=time.perf_counter()
    final, stop=gate.search(seed.A,None,width=8,max_steps=256,timeout=30,mode="admit_support",neutral=True)
    validation,_=support.verify(A0,final.A,seed.steps+final.steps,TOL)
    assert validation['passed']
    elapsed=time.perf_counter()-start
    total_polish += elapsed
    out['seeds'].append({'layer':layer,'seed_H':h,'final_H':H(final.A),'S':final.support,
                         'termination':stop,'verification':validation,'wall_s':elapsed})
    if best is None or (H(final.A),final.support) < (H(best.A),best.support):
        best=support.State(final.A,final.support,final.keys,seed.steps+final.steps)
assert all(x['termination']['stop_reason']=='frontier_exhausted' for x in out['seeds'])
dest=HERE/'cases'/name
out['best']=save_state(dest,'hybrid8',A0,labels,best,{'stop_reason':'frontier_exhausted'},total_polish)
out['best']['circuit']=compile_output(dest,'hybrid8',A0,best.A,labels,out['best'])
support.dump(HERE/(name+'_multiseed.json'),out)
print(json.dumps({'unique_states':len(seen),'seeds':[{k:v for k,v in x.items() if k not in ['termination','verification']} for x in out['seeds']]}))
