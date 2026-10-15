"""Read-only evidence collector for the review bundle; no benchmark or paper writes."""
from pathlib import Path
import csv,json,ast,statistics,hashlib,re,collections,subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text())
def rows(p):return list(csv.DictReader((ROOT/p).open()))
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
random=rows('Results/random_channel_benchmarks_summary.csv')
raw=[]
for label in ['N5_dim4_N1_10_N2_2','N9_dim8_N1_20_N2_4']:
    for basis in ['nam','ibm']:
        selected=[r for r in random if r['parameter_set']==label and r['method']=='raw' and r['basis']==basis and r['seed'] in list('01234')]
        vals=[int(r['metric_total']) for r in selected]
        raw.append(dict(parameter_set=label,basis=basis,seeds=[int(r['seed']) for r in selected],counts=vals,mean=statistics.mean(vals),sources=[r['summary_path'] for r in selected]))
q=[r for r in random if r['parameter_set']=='N5_dim4_N1_10_N2_2' and r['method']=='qiskit_opt3' and r['seed'] in list('01234')]
write('D03_D04_seed_means.json',dict(raw=raw,n5_qiskit=dict(counts=[int(r['metric_total']) for r in q],mean=statistics.mean(int(r['metric_total']) for r in q),ratio=statistics.mean(int(r['metric_total']) for r in q)/raw[1]['mean']),quartz=dict(seeds=[0,1],output_mean=115069,raw_5seed_mean=raw[0]['mean'],requested_5seed_denominator_ratio=115069/raw[0]['mean'],complete_five_seed_estimate=False)))

inventory=[]
for p in sorted((ROOT/'Baseline_results/eval_redesign/random_channel_benchmarks/cases').glob('*seed[0-4]/generic_tools/*/*native_summary.json')):
    j=json.loads(p.read_text());proc=j.get('process') or {};stat=j.get('output_stats') or {}
    inventory.append(dict(benchmark=p.parent.parent.parent.name,tool=p.parent.name,source=str(p.relative_to(ROOT)),
        output=stat.get('metric_total'),process_returncode=proc.get('returncode'),timed_out=proc.get('timed_out'),
        fallback=j.get('fallback_output_used'),checkpoint=j.get('checkpoint_output_used'),input=j.get('input'),
        input_count=(j.get('input_stats') or {}).get('metric_total'),output_qasm=j.get('output_qasm'),
        elapsed=proc.get('elapsed_seconds',(j.get('pytket_stats') or {}).get('elapsed_seconds'))))
write('D05_raw_record_inventory.json',inventory)
second=rows('Results/ours_second_stage_external_optimizers_merged.csv')
wisq=[r for r in second if 'N5_' in r['benchmark'] and r['tool']=='wisq']
partials=[]
for p in (ROOT/'Baseline_results/eval_redesign').glob('ours_second_stage*/**/*seed3*/nam/wisq/**/*.qasm'):
    partials.append(dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,classification='intermediate checkpoint; not a completed summary'))
write('D06_wisq_records.json',dict(merged_rows=wisq,mean_ratio=statistics.mean(float(r['ratio_vs_ours_input']) for r in wisq),seed3_intermediate=partials))

nb=read('src/evaluate.ipynb');text=''.join(''.join(o.get('text',[])) for o in nb['cells'][10]['outputs'])
lines=text.splitlines();lcu=ast.literal_eval(lines[0]);ref=ast.literal_eval(lines[1]);parts=lines[2].split('} {');err=ast.literal_eval(parts[0]+'}');bound=ast.literal_eval('{'+parts[1])
points=[]
for n in [4,5,6]:
    for k,(e,b) in enumerate(zip(err[n],bound[n])):points.append(dict(n=n,inverse_delta=10*(k+1),error=e,bound=b,passed=e<=b))
write('D09_D16_notebook_values.json',dict(error_points=points,all_bounds_pass=all(r['passed'] for r in points),single_segment=True,timing_rows=[dict(n=n,lcu_samples=v,lcu_mean=statistics.mean(v),reference_samples=ref.get(n,[]),reference_mean=statistics.mean(ref[n]) if ref.get(n) else None) for n,v in lcu.items()],missing_paper_n12=True,source='src/evaluate.ipynb:cells[9].source + cells[10].outputs',historical_runtime_code_snapshot_available=False))
points=[]
for r in rows('Baseline_results/eval_redesign/cp4_rq3_frontend_accuracy/higher_order_accuracy.csv'):
    if r['order'] not in ['1','2']:continue
    e=float(r['avg_trace_distance']);m=float(r['max_trace_distance']);b=float(r['theory_total_bound_eq75'])
    points.append(dict(case=int(r['order']),segments=int(r['segments']),mean_error=e,max_error=m,bound=b,mean_pass=e<=b,max_pass=m<=b,lambda0=r['lambda0']))
for r in rows('Baseline_results/eval_redesign/cp4_rq3_coherent_kqk1_accuracy/coherent_kqk1_accuracy.csv'):
    if tuple(r[k] for k in ['duhamel_order_K','sampling_points_q','coherent_taylor_order_K1'])!=('2','3','3'):continue
    e=float(r['avg_trace_distance']);m=float(r['max_trace_distance']);b=float(r['theory_total_bound_eq75'])
    points.append(dict(case=3,segments=int(r['segments']),mean_error=e,max_error=m,bound=b,mean_pass=e<=b,max_pass=m<=b))
write('D10_bound_check.json',dict(points=points,all_pass=all(p['mean_pass'] and p['max_pass'] for p in points)))

for line in (ROOT/'Baseline_results/eval_redesign/cp1_stinespring_sanity/raw_runs.jsonl').read_text().splitlines():
    j=json.loads(line)
    if j.get('benchmark_name')=='hypercube_random_walk_n8':
        j['proposed_ratios_percent']={k:100*j[k]/v for k,v in [('gates',84220),('depth',60268)]}
        j['proposed_ratios_percent'].update(cx=100*j['ops']['cx']/40816,u3=100*j['ops']['u3']/43404)
        write('D15_stinespring_record.json',j)

source_files=['src/channel_IR.py','src/block_encoding_new.py','src/channel_LCU.py','src/subroutine.py','scripts/run_tfim4_channel_lcu_ours_nam.py','src/hypercube_channel_pauli.py','scripts/build_random_channel_lcu_benchmark.py','scripts/eval_redesign/run_cp2_random_ablation.py','scripts/eval_redesign/run_cp2_hrw20_ablation.py','Baseline_scripts/run_qiskit_opt3_baseline.py','scripts/prepare_tfim_external_baseline_inputs.py']
source_files += [str(p.relative_to(ROOT)) for p in (ROOT.parent/'nonudymics').glob('Sources/Tables/*.tex')] if False else []
manifest=[]
for s in source_files:
    p=ROOT/s;manifest.append(dict(path=s,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
write('source_fingerprints.json',manifest)
write('repository_state.json',{key:subprocess.check_output(['git','-C',str(path),'status','--short'],text=True) for key,path in [('compiler',ROOT),('paper',ROOT.parent/'nonudymics')]})
print('Evidence collected:',OUT)
