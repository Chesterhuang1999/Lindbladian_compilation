#!/usr/bin/env python3
"""Executable task-1 controllers. No LLM/API calls; all search uses agent_search."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time
from agent_search import Session, write_json

FAST = {'beam_width': 2, 'layer_candidate_limit': 2, 'max_exact_layer_assignments': 2,
        'mixed_full_initial_beam_width': 4, 'mixed_nonfull_initial_beam_width': 4,
        'mixed_low_weight_count': 4, 'mixed_random_count': 4}
PROFILES = {'fast': FAST, 'default': {}}
METHODS = ('current_layered', 'proxy_layered', 'alternating', 'nested', 'frontier')


def remaining(session):
    return session.call('inspect_ledger')['budget']['remaining']


def available(session):
    b = remaining(session)
    return b['wall_seconds'] > 2 and b['cpu_seconds'] > 1 and b['lowering_calls'] > 0


def evaluate_one(session, rid, profile, seed, per_call):
    limits = {'wall_seconds': per_call, 'cpu_seconds': per_call}
    placement = session.call('search_placements', dict(repr_id=rid, config=profile, seed_index=seed, limits=limits))
    if placement['status'] != 'completed':
        return None
    result = session.call('evaluate', dict(repr_id=rid,
                           placement_bundle_id=placement['placement_bundle_id'], limits=limits))
    return result if result['status'] == 'completed' else None


def expand(session, rid, cursor, page_size, per_call):
    result = session.call('propose_rewrites', {'parent_repr_id': rid, 'cursor': cursor,
                     'limits': {'max_pairs': 4, 'max_unitaries': 128, 'page_size': page_size,
                                'wall_seconds': per_call, 'cpu_seconds': per_call}})
    return result


def controller(session, method, *, profile, seed, candidates, rounds, per_call, feedback=True, adaptive=True,
               rewrite_beam=8):
    root = session.call('observe')['root_repr_id']
    # All methods receive the same measured input incumbent, and pay for it.
    initial = evaluate_one(session, root, profile, seed, per_call)
    costs = {root: initial['expected_gate_count']} if initial else {}
    scores = {root: session.call('observe')['candidates'][0]['upstream_score']}
    seen, cursors, exhausted = {root}, {}, set()
    order, evaluated = [root], {root}
    if method in ('current_layered', 'proxy_layered'):
        operation = 'run_legacy_rewrite' if method == 'current_layered' else 'run_cost_guided_rewrite'
        baseline = session.call(operation, {'repr_id': root, 'beam_width': rewrite_beam, 'max_steps': rounds,
                         'limits': {'wall_seconds': per_call, 'cpu_seconds': per_call}})
        target = baseline.get('repr_id', root)
        for j in range(1, candidates + 1):
            if not available(session):
                break
            evaluate_one(session, target, profile, seed + j, per_call)
        return
    parent = root
    for turn in range(candidates):
        if not available(session):
            break
        # Nested: fixed root-centered stream and fixed lower budget. Alternating:
        # round-robin fresh starts and greedy feedback chains. Frontier retains
        # candidates, revisits promising states and periodically explores the root.
        if method == 'nested':
            parent = root
        elif method == 'alternating' and turn % max(1, rounds) == 0:
            parent = root
        elif method == 'frontier':
            pool = [r for r in order if r not in exhausted]
            if not pool:
                break
            ranking = costs if feedback else scores
            parent = root if turn % 3 == 0 and root in pool else min(pool, key=lambda r: (ranking.get(r, float('inf')), scores[r]))
        if parent not in exhausted:
            new = expand(session, parent, cursors.get(parent), 4, per_call)
            if new['status'] != 'completed':
                break
            cursors[parent] = new['next_cursor']
            if not new['next_cursor']:
                exhausted.add(parent)
            children = []
            for item in new['candidates']:
                rid = item['repr_id']
                if rid not in seen:
                    seen.add(rid)
                    order.append(rid)
                    scores[rid] = item['upstream_score']
                    children.append(rid)
            session.call('update_frontier', {'activate': children})
        else:
            children = []
        if method == 'frontier':
            waiting = [r for r in order if r not in evaluated]
            target = min(waiting, key=lambda r: (scores[r], r)) if waiting else parent
            # Every third decision revisits the best measured representation.
            if adaptive and costs and turn % 3 == 2:
                target = min(costs, key=costs.get) if feedback else min(scores, key=scores.get)
        else:
            waiting = [r for r in order if r not in evaluated] if method == 'nested' else children
            if not waiting:
                if parent in exhausted and method == 'nested':
                    break
                parent = root
                continue
            target = waiting[0] if method == 'nested' else min(waiting, key=lambda r: (scores[r], r))
        use_profile = dict(profile)
        if method == 'frontier' and adaptive and target in evaluated:
            # Doubling is bounded, explicit in the ledger, and costs the same total budget.
            use_profile['beam_width'] = min(64, 2 * use_profile.get('beam_width', 16))
            use_profile['layer_candidate_limit'] = min(64, 2 * use_profile.get('layer_candidate_limit', 16))
        result = evaluate_one(session, target, use_profile, seed + turn + 1, per_call)
        evaluated.add(target)
        if result:
            costs[target] = min(costs.get(target, float('inf')), result['expected_gate_count'])
        if method == 'alternating':
            ranking = costs if feedback else scores
            if ranking.get(target, float('inf')) < ranking.get(parent, float('inf')):
                parent = target
            session.call('update_frontier', {'activate': [parent], 'archive': [r for r in children if r != parent]})


def compare(options):
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', vars(options))
    summaries = []
    for repeat in range(options.repeats):
        # Rotate order to reduce systematic machine-load/order effects.
        methods = options.methods[repeat % len(options.methods):] + options.methods[:repeat % len(options.methods)]
        for method in methods:
            run = output / f'{method}_seed{repeat}'
            session = Session.create(run, family=options.family, n=options.n, dt=options.dt,
                                     seed=options.input_seed, wall_seconds=options.wall,
                                     cpu_seconds=options.cpu, lowering_calls=options.lowerings)
            profile = dict(PROFILES[options.profile])
            if options.placement_beam is not None:
                profile['beam_width'] = options.placement_beam
            controller(session, method, profile=profile, seed=repeat*100,
                       candidates=options.candidates, rounds=options.rounds, per_call=options.per_call,
                       feedback=not options.no_feedback, adaptive=not options.no_adaptive,
                       rewrite_beam=options.rewrite_beam)
            summary = session.call('finish', {'reason': 'budget_or_controller_limit'})
            summary.update(method=method, repeat=repeat, run=str(run.resolve()))
            summaries.append(summary)
            write_json(output / 'summary.json', summaries)
            print(json.dumps(summary), flush=True)


def diagnose(options):
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', vars(options))
    session = Session.create(output / 'diagnostic', family=options.family, n=options.n, dt=options.dt,
                             seed=options.input_seed, wall_seconds=options.wall,
                             cpu_seconds=options.cpu, lowering_calls=options.lowerings)
    root = session.call('observe')['root_repr_id']
    pool, cursor = [root], None
    # Keep the actual legacy pruning trace for attribution; do not infer pruning
    # merely from a candidate's score. Join trace snapshots by representation ID.
    strong = session.call('run_cost_guided_rewrite', {'repr_id': root, 'beam_width': options.rewrite_beam,
                            'max_steps': options.rounds,
                            'limits': {'wall_seconds': options.per_call}})
    if strong.get('repr_id') and strong['repr_id'] not in pool:
        pool.append(strong['repr_id'])
    legacy = session.call('run_legacy_rewrite', {'repr_id': root, 'beam_width': options.rewrite_beam,
                            'max_steps': options.rounds, 'trace': True,
                            'limits': {'wall_seconds': options.per_call}})
    if legacy.get('repr_id') and legacy['repr_id'] not in pool:
        pool.append(legacy['repr_id'])
    reasons = {}
    if legacy.get('trace_ref'):
        from search_contract import stable_hash
        for line in Path(legacy['trace_ref']).read_text().splitlines():
            event = json.loads(line)
            if 'candidate' in event:
                key = 'r_' + stable_hash(event['candidate'])[:24]
                reasons.setdefault(key, set()).add(event['reason'])
    while len(pool) < options.candidates and available(session):
        new = expand(session, root, cursor, options.candidates, options.per_call)
        if new['status'] != 'completed':
            break
        for c in new['candidates']:
            if c['repr_id'] not in pool:
                pool.append(c['repr_id'])
        cursor = new['next_cursor']
        if not cursor:
            break
    pool = pool[:options.candidates]
    session.call('update_frontier', {'activate': pool})
    records = []
    for profile in options.profiles:
        for repeat in range(options.repeats):
            ordered = pool[repeat % len(pool):] + pool[:repeat % len(pool)]
            for rid in ordered:
                if not available(session):
                    break
                config = dict(PROFILES[profile])
                if options.placement_beam is not None:
                    config['beam_width'] = options.placement_beam
                result = evaluate_one(session, rid, config, repeat, options.per_call)
                obs = session.call('observe', {'repr_id': rid})['candidates'][0]
                records.append({'repr_id': rid, 'upstream_score': obs['upstream_score'],
                                'legacy_trace_reasons': sorted(reasons.get(rid, [])),
                                'profile': profile, 'repeat': repeat, 'result': result})
                write_json(output / 'diagnostic_records.json', records)
    finish = session.call('finish', {'reason': 'diagnostic_grid_or_budget_completed'})
    write_json(output / 'finish.json', finish)
    print(json.dumps(finish), flush=True)
    report_diagnostic(output)


def report_diagnostic(output):
    records = json.loads((Path(output) / 'diagnostic_records.json').read_text())
    pairs = []
    for profile in sorted({r['profile'] for r in records}):
        rows = [r for r in records if r['profile'] == profile and r['result']]
        ids = sorted({r['repr_id'] for r in rows})
        for a, b in itertools.combinations(ids, 2):
            da = {r['repeat']: r for r in rows if r['repr_id'] == a}
            db = {r['repeat']: r for r in rows if r['repr_id'] == b}
            common = sorted(set(da) & set(db))
            if not common:
                continue
            ha, hb = da[common[0]]['upstream_score'], db[common[0]]['upstream_score']
            ratios = [da[k]['result']['expected_gate_count'] / db[k]['result']['expected_gate_count'] for k in common]
            tied = abs(ha-hb) <= 1e-10 * max(1, abs(ha), abs(hb))
            reversal = sum((ratio-1)*(ha-hb) < 0 for ratio in ratios) / len(ratios)
            pairs.append({'profile': profile, 'a': a, 'b': b, 'paired_repeats': len(common),
                          'heuristic_tie': tied, 'median_cost_a_over_b': statistics.median(ratios),
                          'reversal_fraction': None if tied else reversal,
                          'stable_reversal_screen': not tied and len(common) >= 5 and reversal >= .8
                          and abs(statistics.median(ratios)-1) >= .05})
    write_json(Path(output) / 'pair_diagnostics.json', pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['compare', 'diagnose'])
    p.add_argument('--output', required=True)
    p.add_argument('--family', choices=['tfim', 'pauli_mixture', 'mixed_pauli', 'hrw',
                                      'hrw_scrambled', 'random_pauli'], default='tfim')
    p.add_argument('--n', type=int, default=2)
    p.add_argument('--dt', type=float, default=.01)
    p.add_argument('--input-seed', type=int, default=0)
    p.add_argument('--wall', type=float, default=600)
    p.add_argument('--cpu', type=float, default=600)
    p.add_argument('--lowerings', type=int, default=12)
    p.add_argument('--per-call', type=float, default=120)
    p.add_argument('--candidates', type=int, default=12)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--rewrite-beam', type=int, default=8, help='Provisional, freeze after calibration')
    p.add_argument('--placement-beam', type=int, default=None, help='Override profile beam width')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--profile', choices=PROFILES, default='default')
    p.add_argument('--profiles', nargs='+', choices=PROFILES, default=['fast', 'default'])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--no-feedback', action='store_true')
    p.add_argument('--no-adaptive', action='store_true')
    options = p.parse_args()
    if options.rewrite_beam < 1 or (options.placement_beam is not None and options.placement_beam < 1):
        p.error('Beam widths must be positive')
    if min(options.repeats, options.candidates, options.rounds, options.per_call) <= 0:
        p.error('Counts and limits must be positive')
    (compare if options.mode == 'compare' else diagnose)(options)


if __name__ == '__main__':
    main()
