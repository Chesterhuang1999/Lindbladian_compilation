"""Build public qubit-based candidate and verify the review package."""
from pathlib import Path
import json,csv,hashlib,statistics,re,ast,subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'Baseline_results/eval_redesign/artifact_review_20260917'
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
random=DATA/'random_qubits4_checked'
internal=random/'random_pauli_kraus_N5_dim16_N1_10_N2_2'
summary=list(csv.DictReader((internal/'summary.csv').open()))
norms=json.loads((random/'subnormalization.json').read_text())
public={'benchmark':'RandomPauliKraus_N5_q4_N1_10_N2_2','tuple_fields':['kraus_count','system_qubits','first_kraus_terms','other_kraus_terms'],
    'tuple':[5,4,10,2],'hilbert_dim':16,'seeds':[0,1,2,3,4],
    'ratio_policy':'mean of per-seed output/basic ratios; not ratio of rounded means',
    'subnormalization':{'raw_mean':statistics.mean(x['raw'] for x in norms),'rewritten_mean':statistics.mean(x['rewritten'] for x in norms)},
    'backend_candidate':'fixed zero-address residual + explicit [work,system] block register interface',
    'aggregates':[{'config':r['config'],'gate_mean':float(r['total_gates']),'mean_ratio_to_seed_basic':float(r['ratio_vs_basic']),'completed_seeds':int(float(r['completed_seed_count'])),'source':str((internal/'summary.csv').relative_to(ROOT))} for r in summary],
    'state_probes':json.loads((random/'all_state_checks_packed.json').read_text())}
save(OUT/'D07_public_qubit_summary.json',public)

issues=[]
statuses=['paper_change_ready','paper_change_ready','paper_change_ready','denominator_confirmed_seed_backfill_pending','backfill_plan_pending_review','backfill_plan_pending_review','rerun_candidate_passed_pending_review','new_five_seed_mean_ready','paper_change_ready','closed_per_user','root_cause_and_fix_validated_pending_review','closed_per_user','closed_per_user','rerun_candidate_passed_pending_review','paper_change_ready_with_basis_note','paper_change_ready_n12_missing','closed_per_user','builders_disentangled_pending_policy','preview_ready_pending_review']
for i,status in enumerate(statuses,1):issues.append({'issue':f'D{i:02d}','status':status,'applied_to_paper':False,'applied_to_formal_scripts_or_results':False})
save(OUT/'issue_status.json',issues)

# Capture exact source versions used; preserve references to original paths.
original=json.loads((OUT/'source_fingerprints.json').read_text())
for extra in ['src/mobius_residual_fixed.py','src/placement_search.py','src/block_encoding.py','scripts/eval_redesign/run_cp2_hrw20_ablation.py','Baseline_results/eval_redesign/cp4_rq3_semantic_backend_validation/run_cp4b_semantic_backend_validation.py']:
    if extra not in [x['path'] for x in original]:original.append({'path':extra,'sha256':hashlib.sha256((ROOT/extra).read_bytes()).hexdigest()})
checks=[]
for rec in original:
    p=ROOT/rec['path']; digest=hashlib.sha256(p.read_bytes()).hexdigest()
    checks.append(dict(path=rec['path'],unchanged=digest==rec['sha256']))
    dest=OUT/'source_snapshots'/rec['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
save(OUT/'original_sources_unchanged.json',checks)

semantic=json.loads((DATA/'semantic_fixed/summary.json').read_text())
blocks=json.loads((DATA/'semantic/pipeline_diagnosis.json').read_text())
fold=json.loads((DATA/'clifford/phase_fold_summary.json').read_text())
backfill=json.loads((OUT/'D05_D06_backfill_plan.json').read_text())
rows=list(internal.glob('seed_*/rows/*.json'))
validation={'issues':len(issues),'random_config_seed_rows':len(rows),
    'random_rows_success':all(json.loads(p.read_text())['status']=='ok' for p in rows),
    'random_state_probes_pass':all(r['passed'] for r in public['state_probes']),
    'semantic_state_probes':len(semantic['rows']),'semantic_all_pass':all(r['status']=='ok' for r in semantic['rows']),
    'local_kraus_blocks_fixed':sum(r['variant']=='residual_only_fixed' for r in blocks['blocks']),
    'local_kraus_blocks_all_pass':all(r['passed'] for r in blocks['blocks'] if r['variant']=='residual_only_fixed'),
    'clifford_n':[r['n'] for r in fold], 'clifford_exact_checks_pass':all(r['exact_h_cx_t_s']['passed'] for r in fold),
    'figure9_rows':len(list(csv.DictReader((OUT/'figure9_appendix_sync/channel_lcu_ratio_bars_draft.csv').open()))),
    'backfill_jobs_prepared_not_run':len(backfill['jobs']),
    'backfill_inputs_exist':all(Path(r['input']).exists() for r in backfill['jobs']),
    'original_sources_unchanged':all(r['unchanged'] for r in checks),
    'paper_git_status':subprocess.check_output(['git','-C',str(ROOT.parent/'nonudymics'),'status','--short'],text=True)}
assert validation['issues']==19 and validation['random_config_seed_rows']==40 and validation['random_rows_success']
assert validation['random_state_probes_pass'] and validation['semantic_all_pass'] and validation['local_kraus_blocks_all_pass']
assert validation['clifford_n']==[3,4,6,8] and validation['clifford_exact_checks_pass'] and validation['original_sources_unchanged']
save(OUT/'validation.json',validation)
files=[]
for base in [OUT,DATA]:
    for p in sorted(base.rglob('*')):
        if not p.is_file() or p.name in ['artifact_manifest.json'] or '__pycache__' in p.parts:continue
        files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
save(OUT/'artifact_manifest.json',files)
print(json.dumps(validation,ensure_ascii=False,indent=2))
