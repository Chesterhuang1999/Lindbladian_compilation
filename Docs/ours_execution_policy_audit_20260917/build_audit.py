"""Read existing evidence and write an execution-policy audit; never run benchmarks."""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REVIEW = ROOT / 'Docs/artifact_evaluation_review_20260917'
BASE = ROOT / 'Baseline_results/eval_redesign/artifact_review_20260917'
INDEX = ROOT / 'Docs/artifact_evaluation_audit_20260917/evaluation_data_index.json'
entries = json.loads(INDEX.read_text())['entries']
read_paths = {INDEX}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    read_paths.add(path)
    return json.loads(path.read_text())


def route(outer, placement, rewrite, residual='unpatched', packing='not_installed'):
    return dict(outer_multiplexer=outer, placement_search=placement,
                rewrite=rewrite, residual_implementation=residual if placement == 'yes' else 'not_applicable',
                work_system_adapter=packing if placement == 'yes' else 'not_applicable')


def ablation_route(benchmark, config, fixed=False):
    r = config in ['R', 'R+O', 'R+F*', 'All']
    o = config in ['O', 'R+O', 'O+F*', 'All']
    f = config in ['F*', 'R+F*', 'O+F*', 'All']
    common = benchmark in ['TFIM-8', 'HRW-20']
    outer = 'basic' if not f else ('tree' if common or not o else 'ctrl-line')
    return route(outer, 'yes' if o else 'disabled_by_ablation',
                 'gatecost_rewrite_search / beam / width=8' if r else 'disabled_by_ablation',
                 'fixed' if fixed else 'unpatched', 'installed' if fixed else 'not_installed')


rows = []
for e in entries:
    rq, method, bench = e['rq'], e['method'], e['benchmark']
    included = ('ours' in method.lower() or method == 'Optimized'
                or rq == 'RQ2.1' or rq.startswith('RQ3'))
    if not included:
        continue
    item = {k:e[k] for k in ['id', 'rq', 'benchmark', 'method', 'paper_locations', 'records', 'scripts']}
    item['version'] = 'paper_or_selected_existing_result'
    item['evidence_limit'] = 'Recorded results plus current linked source; historical source hashes are not generally available.'
    if rq == 'RQ1.1':
        rewrite = ('gatecost_rewrite_search / beam / width=8' if bench.startswith('Toom') or bench.startswith('Color')
                   else 'none_special_higher_order_constructor' if bench == 'Decay_HO'
                   else 'rewrite_search / greedy / support objective / beam_width=3 inactive')
        item.update(route('ctrl-line', 'yes', rewrite))
        if 'historical' in method:
            item['version'] = 'superseded_plot_result_D19'
        if bench.startswith('RandCh'):
            item['note'] = 'RQ1 N5/N9 remain the original 2q/3q inputs; D07 correction did not change them.'
    elif rq == 'RQ1.2':
        item.update(route('ctrl-line_in_linked_builders', 'yes', 'rewrite_search / greedy / support objective'))
        item['note'] = 'Audit the ours input only. Table 3 uses July 4/5 batches, not the July 6 RQ1 refresh. Historical per-run source snapshot missing.'
    elif rq == 'RQ1.4':
        item.update(route('unknown', 'unknown', 'unknown', 'unknown', 'unknown'))
        item['note'] = 'Exact Optimized records 836/1313 gates are paper-only; related raw opt3 summary cannot certify this route.'
    elif rq == 'RQ2.1':
        item.update(ablation_route(bench, method))
        if bench.startswith('Rand-Paulis'):
            item['version'] = 'paper_old_2q_superseded_by_D07_current_4q'
    elif rq == 'RQ2.2' and 'SELECT-only' in method:
        multi = bench in ['HRW_8', 'HRW_28', 'random_coloring_linear_n4_q4', 'random_coloring_cycle_n4_q4', 'repetition_dqe_d3']
        item.update(route('not_applicable_SELECT_only', 'yes',
                          'rewrite_search / greedy / support objective' if multi else 'not_applicable_single_Pauli_sum',
                          packing='not_applicable_SELECT_only'))
    elif rq == 'RQ2.2':
        item.update(route('not_applicable_SELECT_only', 'unknown_historical_record_missing', 'not_applicable_single_Pauli_sum'))
        item['version'] = 'paper_old_Cline_ours_superseded_by_D14'
    elif rq == 'RQ3.1':
        item.update(route('not_applicable_dense_proxy' if method in ['case1', 'case2'] else 'basic',
                          'not_enabled_accuracy_experiment', 'not_enabled_accuracy_experiment'))
        item['note'] = 'Accuracy experiment does not measure the fully optimized compiler route.'
    elif rq == 'RQ3.3':
        item.update(route('basic', 'not_enabled_simulation_timing', 'not_enabled_simulation_timing'))
    elif rq == 'RQ3.2':
        config = method.strip().split()[-1]
        if method.startswith('rewrite_exactness'):
            item.update(route('not_applicable_rewrite_only', 'not_applicable', 'rewrite_search / greedy / support objective'))
        elif method.startswith('no-approx'):
            item.update(route('mixed_legacy_Basic_O_F_All', 'mixed', 'rewrite_search / greedy where enabled'))
        else:
            item.update(route('ctrl-line' if config == 'All' else 'tree' if config == 'O' else 'basic',
                              'yes' if config in ['F', 'All'] else 'disabled_by_ablation',
                              'rewrite_search / greedy / support objective' if config in ['R', 'All'] else 'disabled_by_ablation'))
        item['version'] = 'paper_legacy_semantic_labels_replaced_by_review_probes'
        item['note'] = 'Legacy CP4 labels O=flattening and F=ordering were inverted; current review labels are corrected.'
    else:
        raise ValueError((rq, method))
    rows.append(item)

comparison = read(REVIEW / 'D07_paper_vs_4q_comparison.json')
config_names = dict(zip(['Basic', 'Rewrite only', 'Order only', 'Flat only', 'Rewrite + Order', 'Rewrite + Flat', 'Flat + Order', 'Rewrite + Flat + Order'],
                       ['Basic', 'R', 'O', 'F*', 'R+O', 'R+F*', 'O+F*', 'All']))
for raw in comparison['batches']['4q_current']['rows']:
    cfg = config_names[raw['config']]
    assert raw['status'] == 'ok'
    rows.append(dict(id=f'D07-current-{raw["seed"]}-{cfg}', rq='RQ2.1', benchmark='Random Kraus 4q current', method=cfg,
                     version='pending_review_current_4q', paper_locations=['Table 5'],
                     records=[dict(path=raw['_source'], selector='', role='raw configuration record')],
                     scripts=['Docs/artifact_evaluation_review_20260917/run_random_qubits_review.py',
                              'Docs/artifact_evaluation_review_20260917/run_D07_manual_fifth_ablation.py',
                              'Docs/artifact_evaluation_review_20260917/review_backend_adapter.py',
                              'scripts/eval_redesign/run_cp2_random_ablation.py'],
                     seed=raw['seed'], total_gates=raw['total_gates'],
                     note='seed 20260921 includes the user-requested K4 term0 edit; first four records are frozen.',
                     **ablation_route('Random Kraus 4q current', cfg, fixed=True)))

semantic_path = BASE / 'semantic_fixed/summary.json'
semantic = read(semantic_path)
assert semantic['fixed_residual'] and all(r['status'] == 'ok' for r in semantic['rows'])
for raw in semantic['rows']:
    cfg = raw['configuration']
    rows.append(dict(id=f'D11-current-{raw["benchmark"]}-{cfg}', rq='RQ3.2', benchmark=raw['benchmark'], method=cfg,
                     version='pending_review_semantic_fixed', paper_locations=['Table 8'],
                     records=[dict(path=str(semantic_path.relative_to(ROOT)), selector=f'benchmark={raw["benchmark"]}; configuration={cfg}')],
                     scripts=['Docs/artifact_evaluation_review_20260917/run_semantic_review.py',
                              'Baseline_results/eval_redesign/cp4_rq3_semantic_backend_validation/run_cp4b_semantic_backend_validation.py'],
                     trace_distance=raw['trace_distance'],
                     **route('ctrl-line' if cfg == 'All' else 'tree' if cfg == 'F' else 'basic',
                             'yes' if cfg in ['O', 'All'] else 'disabled_by_ablation',
                             'rewrite_search / greedy / support objective' if raw['rewrite'] else 'disabled_by_ablation',
                             'fixed', 'not_installed_state_probes_passed_for_these_inputs')))

clifford_path = BASE / 'clifford/phase_fold_summary.json'
for raw in read(clifford_path):
    rows.append(dict(id=f'D14-current-H{raw["n"]}', rq='RQ2.2', benchmark=f'Full Pauli H{raw["n"]}', method='Ours H/CX/T/S',
                     version='pending_review_D14', paper_locations=['Figure 10'],
                     records=[dict(path=str(clifford_path.relative_to(ROOT)), selector=f'n={raw["n"]}')],
                     scripts=['Docs/artifact_evaluation_review_20260917/run_clifford_review.py', 'Docs/artifact_evaluation_review_20260917/fold_clifford_review.py'],
                     note='Basis-conversion local operator checks passed; no full-channel claim is made.',
                     **route('not_applicable_SELECT_only', 'yes', 'not_applicable_single_Pauli_sum', packing='not_applicable_SELECT_only')))
rows.append(dict(id='D16-current-n12', rq='RQ3.3', benchmark='TFIM n12', method='LCU simulation', version='pending_review_D16',
                 paper_locations=['Figure 15'], records=[dict(path=str((BASE/'D16_missing_n12_rerun/result.json').relative_to(ROOT)), selector='')],
                 scripts=['Docs/artifact_evaluation_review_20260917/rerun_D16_missing_timing.py'],
                 **route('basic', 'not_enabled_simulation_timing', 'not_enabled_simulation_timing')))

# D18 is a diagnostic cross-product, not an automatically selected replacement table row.
hrw_path = BASE / 'hrw20/summary.json'
hrw_diagnostics = read(hrw_path)
for i, raw in enumerate(hrw_diagnostics):
    rows.append(dict(id=f'D18-diagnostic-{i}', rq='RQ1.1/RQ2.1', benchmark='HRW-20', method=raw['builder'],
                     version='diagnostic_candidate_not_selected', paper_locations=['Table 5', 'Table 14'],
                     records=[dict(path=str(hrw_path.relative_to(ROOT)), selector=f'array[{i}]')], scripts=[], total_gates=raw['total'],
                     **route('tree' if raw['builder'] == 'cp2_channel_to_lcu' else 'ctrl-line', 'yes',
                             'gatecost_rewrite_search / beam / width=8' if raw['rewrite_target']=='gatecost_beam_cached' else 'rewrite_search / greedy / support objective',
                             'fixed' if raw['residual']=='residual_fixed' else 'unpatched')))

# Extract saved placement evidence instead of inferring it just from an option name.
placement_evidence = []
def walk_placements(x, path, selector=''):
    if isinstance(x, dict):
        if 'selected_crit' in x:
            d = x.get('placement_search_debug', x.get('placement', {}))
            placement_evidence.append(dict(path=str(path.relative_to(ROOT)), selector=selector, selected_crit=x['selected_crit'],
                                           initial_strategy=d.get('initial_strategy'), search_profile=d.get('search_profile'),
                                           shortcut=d.get('shortcut'), candidate_widths=d.get('candidate_widths')))
        for k, v in x.items():
            walk_placements(v, path, selector + '/' + k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            walk_placements(v, path, selector + '/' + str(i))

select_paths = [ROOT/'Baseline_results'/n/'summary.json' for n in
                ['table_main_select_only_ours_qrisp_v1','table_main_select_only_random4q100_v1','table_main_select_only_random4q164_v1']]
for path in select_paths:
    walk_placements(read(path), path)
refresh = ROOT/'Baseline_results/eval_redesign/matrixsum_phase_refresh_20260706'
for path in refresh.rglob('*.json'):
    if path.name.endswith(('build_summary.json', 'ours_summary.json', 'dual_basis_summary.json')):
        walk_placements(read(path), path)

source_files = set(s for row in rows for s in row['scripts']) | {
    'src/channel_IR.py','src/channel_LCU.py','src/subroutine.py','src/block_encoding.py','src/block_encoding_new.py',
    'src/block_encoding_new_fixed.py','src/mobius_residual_fixed.py','src/placement_search.py','src/series_expansion.py',
    'scripts/build_periodic_tfim_higher_order_expansion.py','scripts/build_decay_higher_order_expansion.py',
    'scripts/run_tfim4_channel_lcu_ours_nam.py','scripts/run_hypercube_n4_ours_nam.py',
    'scripts/run_random_coloring_channel_lcu_ours_nam.py','scripts/build_random_channel_lcu_benchmark.py',
    'Baseline_scripts/sum_rewritten_kraus_block_counts_qrisp_vs_ours.py','Baseline_scripts/probe_full_pauli_n6_select_clifford_t.py',
    'Baseline_scripts/rerun_block_encoding_new_main_families.py',
    'Baseline_results/eval_redesign/channel_lcu_structure_opt_refresh_20260705/structured/rerun_structured_families_refresh.py',
    'Baseline_results/eval_redesign/block_encoding_new_ours_rerun_20260704/subagent_structured/rerun_structured_families.py',
}
source_symbols = []
for rel in sorted(source_files):
    p = ROOT/rel
    if p.exists():
        read_paths.add(p)
        if p.suffix == '.py':
            tree = ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and any(k in node.name for k in ['rewrite','build','circuit','mulplex','placement','install','mobius']):
                    source_symbols.append(dict(path=rel, symbol=node.name, line=node.lineno))

missing_records = []
for row in rows:
    for record in row['records']:
        p = Path(record['path'])
        if not p.is_absolute():
            p = ROOT/p
        if p.is_file():
            read_paths.add(p)
        else:
            missing_records.append(dict(id=row['id'], path=str(p)))
assert len([r for r in rows if r['version']=='pending_review_current_4q']) == 40
assert len([r for r in rows if r['version']=='pending_review_semantic_fixed']) == 15
assert len([p for p in placement_evidence if p['path'] in {str(x.relative_to(ROOT)) for x in select_paths}]) == 161

def signature(path):
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

data = dict(scope='All ours-related paper entries and current review candidates; static call-chain audit with saved execution metadata.',
            limitations=['No benchmark was rerun.', 'Current source trace is not a historical code snapshot.',
                         'Selecting placement-search does not certify residual reconstruction or full-channel semantics.',
                         'Basic/ablation-off/SELECT-only/frontend experiments have separate applicability.'],
            rows=rows, source_symbols=source_symbols, saved_placement_evidence=placement_evidence,
            missing_records=missing_records, counts_by_version=dict(Counter(r['version'] for r in rows)),
            modified_paper_or_formal_scripts=False)
(OUT/'ours_execution_policy_audit.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
detail=['# Ours执行口径：逐条索引','',
        '本表按论文索引保留旧记录，另列当前候选。ctrl-line/tree依据实际函数调用；Basic或关闭pass是实验配置，不自动判为错误。',
        'unpatched表示当前入口未安装已发现的残差修正，不等于每个实例都已证实语义错误。历史源码快照缺失的限制见主报告。','',
        '| ID / 版本 | 实例 / 配置 | 外层构造 | Placement | 残差 | Rewrite | 首个记录 |',
        '|---|---|---|---|---|---|---|']
for row in rows:
    record=row['records'][0]['path']
    p=Path(record); p=p if p.is_absolute() else ROOT/p
    detail.append(f'| {row["id"]}<br>{row["version"]} | {row["benchmark"]}<br>{row["method"]} | {row["outer_multiplexer"]} | {row["placement_search"]} | {row["residual_implementation"]} | {row["rewrite"]} | [记录](<{p}>) |')
(OUT/'ours_execution_policy_index.md').write_text('\n'.join(detail)+'\n')
paper=ROOT.parent/'nonudymics/main_v2.pdf'
assert hashlib.sha256(paper.read_bytes()).hexdigest()==json.loads(INDEX.read_text())['paper_sha256']
read_paths.add(paper)
(OUT/'source_manifest.json').write_text(json.dumps([signature(p) for p in sorted(read_paths)],indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(entries=len(rows),versions=data['counts_by_version'],placement_records=len(placement_evidence),missing_records=missing_records,source_files=len(read_paths)),ensure_ascii=False))
