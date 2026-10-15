"""Aggregate verified outputs without treating partial searches as successes."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
rows=[]
for path in sorted((HERE/'cases').glob('*/results.json')):
    r=json.loads(path.read_text())
    row={'case':r['case'],'status':r.get('status','working'),'historical':r['historical'],
         'source':str(path),'observer_reference_output_error':r.get('observer_reference_output_error'),
         'archive':r.get('archive'),'methods':{}}
    for method,v in r['methods'].items():
        c=v.get('circuit',{})
        row['methods'][method]={'status':v['status'],'S':v['metrics']['support'],'M':v['metrics']['M'],
             'H':v['metrics']['static_cost']['total'],'Lambda':v['metrics']['static_cost']['Lambda'],
             'rewrite_s':v['wall_s'],'stop_reason':v['termination']['stop_reason'],
             'rewrite_verification':v['verification'],**{k:c[k] for k in ['G','W','Lambda_G','NC','Lambda_NC','compile_s','verification'] if k in c}}
    if 'hybrid' in row['methods']:
        hybrid=row['methods']['hybrid']
        row['hybrid_change_pct']={}
        for baseline in ['support','gate']:
            if baseline not in row['methods']: continue
            b=row['methods'][baseline]
            if b['status']=='ok' and hybrid['status']=='ok':
                row['hybrid_change_pct'][baseline]={k:100*(hybrid[k]/b[k]-1) for k in ['H','G','Lambda_G','Lambda_NC'] if k in b and k in hybrid}
            else:
                row['hybrid_change_pct'][baseline]={'not_comparable_as_converged_outputs':True}
        assert hybrid['H'] <= row['methods']['support']['H']*(1+1e-12)
    rows.append(row)
out={'definition':'Two-seed hybrid: original support final plus minimum-H committed support-frontier state; refine distinct seeds, retain fallbacks.',
     'cost_note':'H is a static Lambda-weighted non-Clifford heuristic; G and NC are measured after tree/F*/Matrix-order NAM opt0 lowering.',
     'rows':rows}
extra=HERE/'collective_multiseed.json'
if extra.exists():
    x=json.loads(extra.read_text())
    out['collective_eight_seed_ablation']={'source':str(extra),'H':x['best']['metrics']['static_cost']['total'],
        'polish_s':sum(v['wall_s'] for v in x['seeds']),
        **{k:x['best']['circuit'][k] for k in ['G','Lambda_G','verification']}}
extra=HERE/'CPTP2_M4_seed4_multiseed.json'
if extra.exists():
    x=json.loads(extra.read_text())
    out['random_eight_seed_ablation']={'source':str(extra),'H':x['best']['metrics']['static_cost']['total'],
        'polish_s':sum(v['wall_s'] for v in x['seeds']),
        **{k:x['best']['circuit'][k] for k in ['G','Lambda_G','verification']}}
(HERE/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
    print(json.dumps({'case':r['case'],'status':r['status'],
        'methods':{m:{k:v[k] for k in ['S','M','H','G','Lambda_G','rewrite_s','stop_reason'] if k in v} for m,v in r['methods'].items()},
        'change':r.get('hybrid_change_pct')}))
