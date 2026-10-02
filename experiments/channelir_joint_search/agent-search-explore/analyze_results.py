#!/usr/bin/env python3
"""Export auditable evaluation and best-so-far curves from completed sessions."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(folder):
    folder = Path(folder)
    manifest = {}
    for p in folder.rglob('summary.json'):
        for row in json.loads(p.read_text()):
            manifest[str(Path(row['run']).resolve())] = row
    evaluations, curves, runs = [], [], []
    for p in sorted(folder.rglob('state.json')):
        state = json.loads(p.read_text())
        meta = manifest.get(str(p.parent.resolve()), {})
        common = {'run': str(p.parent.resolve()), 'method': meta.get('method', 'diagnostic_or_manual'),
                  'repeat': meta.get('repeat', ''), 'family': state['instance']['family'],
                  'n': state['instance']['n'], 'dt': state['instance']['dt'],
                  'source_identity': state['source_identity']}
        for eid, e in state['evaluations'].items():
            bundle = state['bundles'][e['placement_bundle_id']]
            rep = state['representations'][e['repr_id']]
            evaluations.append(dict(common, evaluation_id=eid, repr_id=e['repr_id'],
                upstream_score=rep['statistics']['upstream_score'],
                support=rep['statistics']['total_support'], gates=e['gates'],
                success_probability=e['success_probability'], expected_gate_count=e['expected_gate_count'],
                normalization=e['lambda'], tau=e['tau'], width=e['width'],
                seed_index=bundle['query']['seed_index'],
                config=json.dumps(bundle['effective_config'], sort_keys=True),
                semantic_check=e['semantic_validation']['full_channel_check']))
        best, cpu, lowerings = None, 0., 0
        for event in state['events']:
            cpu += event['spent']['cpu_seconds']
            lowerings += event['spent']['lowering_calls']
            response = event['response']
            if event['operation'] == 'evaluate' and response.get('status') == 'completed':
                cost = response['expected_gate_count']
                best = cost if best is None else min(best, cost)
            curves.append(dict(common, event_id=event['event_id'],
                 elapsed_wall_seconds=event.get('completed_at', event['timestamp']+event['spent']['wall_seconds'])-state['created_at'],
                 used_tool_cpu_seconds=cpu, used_lowering_attempts=lowerings,
                 best_expected_gate_count=best, operation=event['operation'],
                 status=response.get('status','completed'), cache_hit=response.get('cache_hit',False)))
        runs.append(dict(common, best_expected_gate_count=best, finished=state['finished'],
                         pending=bool(state['pending']), stop_reason=state.get('stop_reason','not_finished'),
                         wall_budget=state['budget']['wall_seconds'], cpu_budget=state['budget']['cpu_seconds'],
                         lowering_budget=state['budget']['lowering_calls'], **state['used']))
    paired = []
    for baseline in ('current_layered', 'proxy_layered'):
        references = [r for r in runs if r['method'] == baseline]
        for r in runs:
            if r['method'] in (baseline, 'diagnostic_or_manual'):
                continue
            matches = [b for b in references if Path(b['run']).parent == Path(r['run']).parent and b['repeat'] == r['repeat']]
            if not matches:
                continue
            b = matches[0]
            numerator, denominator = r['best_expected_gate_count'], b['best_expected_gate_count']
            valid = numerator is not None and denominator is not None and r['finished'] and b['finished']
            ratio = (numerator/denominator if denominator else (1. if numerator == 0 else 'unbounded')) if valid else None
            paired.append({'method':r['method'], 'baseline':baseline, 'repeat':r['repeat'],
                           'family':r['family'], 'n':r['n'], 'dt':r['dt'], 'cpu_budget':r['cpu_budget'],
                           'method_run':r['run'], 'baseline_run':b['run'], 'valid_pair':valid,
                           'cost_ratio':ratio,
                           'outcome':('win' if numerator < denominator-1e-9 else 'loss' if numerator > denominator+1e-9 else 'tie') if valid else 'missing'})
    output = folder / 'analysis'
    output.mkdir(exist_ok=True)
    for name, rows in [('evaluations',evaluations), ('anytime',curves), ('runs',runs), ('paired',paired)]:
        write_csv(output / f'{name}.csv', rows)
    print(json.dumps({'output':str(output.resolve()), 'runs':len(runs), 'evaluations':len(evaluations), 'paired_rows':len(paired)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    analyze(parser.parse_args().folder)
