#!/usr/bin/env python3
"""Persistent, single-writer JSON API for the ChannelIR joint-search pilot.

Python: Session.create(...); session.call('observe', {...}).
CLI: python agent_search.py call RUN observe --args '{}'.
Heavy operations always run in a killable subprocess, with one thread per library.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from search_contract import (build_input, equivalent, pack, stats, stable_hash, SCORE_ID,
                             placement_config)

HEAVY = {'propose_rewrites': 'propose', 'search_placements': 'search',
         'evaluate': 'evaluate', 'run_legacy_rewrite': 'legacy',
         'run_cost_guided_rewrite': 'costguided'}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def source_identity():
    files = [ROOT / 'agent_search.py', ROOT / 'pilot.py'] + sorted((ROOT / 'src').glob('*.py'))
    return stable_hash({'python': list(sys.version_info[:3]),
                        'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                        'runtime': {name: importlib.metadata.version(name) for name in ('numpy', 'qiskit', 'psutil')}})


class Session:
    def __init__(self, directory):
        self.path = Path(directory).resolve()

    @classmethod
    def create(cls, directory, *, family='tfim', n=2, dt=0.01, gamma=0.1, seed=0,
               wall_seconds=600, cpu_seconds=600, lowering_calls=20, representation=None):
        for v in (wall_seconds, cpu_seconds, lowering_calls):
            if not math.isfinite(v) or v <= 0:
                raise ValueError('Session budgets must be positive finite numbers')
        obj = cls(directory)
        obj.path.mkdir(parents=True, exist_ok=False)
        (obj.path / 'artifacts').mkdir()
        rep = representation or build_input(family, n, dt, gamma, seed)
        # Normalize and validate supplied input; preserve exact coefficients/row order.
        from search_contract import unpack
        A, labels = unpack(rep)
        rep = pack(A, labels)
        raw_rep = rep
        if representation is None:
            # Match the existing ChannelIR constructor's common input phase pass.
            from search_contract import channel_from_rep
            ir = channel_from_rep(rep).glob_phase_elim()
            rep = pack(*ir._coeff_matrix())
            equivalent(rep, raw_rep)
        rid = 'r_' + stable_hash(rep)[:24]
        now = time.time()
        state = {'schema': 1, 'created_at': now, 'deadline': now + wall_seconds,
                 'source_identity': source_identity(), 'runtime': {'python': sys.version,
                    **{name: importlib.metadata.version(name) for name in ('numpy','qiskit','psutil')}},
                 'input_preprocessing': 'legacy_global_phase_elimination' if representation is None else 'canonical_columns_only',
                 'raw_input_representation': raw_rep,
                 'instance': {'family': family if representation is None else 'imported',
                              'n': len(labels[0]), 'dt': dt, 'gamma': gamma, 'seed': seed},
                 'contract': {'metric': 'full_channel_NAM_opt0_G_over_p',
                              'basis': ['rz', 'h', 'x', 'cx'], 'reset_counted': False,
                              'rho_ref': 'maximally_mixed', 'score_model_id': SCORE_ID,
                              'width_policy': 'record_only', 'lambda_constraint': None,
                              'semantic_tol': 1e-10, 'coefficient_tol': 1e-12,
                              'cache_scope': 'this_session_only'},
                 'budget': {'wall_seconds': float(wall_seconds), 'cpu_seconds': float(cpu_seconds),
                            'lowering_calls': int(lowering_calls)},
                 'used': {'cpu_seconds': 0., 'tool_wall_seconds': 0., 'lowering_calls': 0,
                          'cache_hits': 0},
                 'revision': 0, 'root_repr_id': rid, 'active': [rid],
                 'representations': {rid: {'representation': rep, 'statistics': stats(rep),
                                          'parent_repr_id': None}},
                 'bundles': {}, 'evaluations': {}, 'events': [], 'cache': {},
                 'best_evaluation_id': None, 'finished': False, 'pending': None}
        write_json(obj.path / 'state.json', state)
        return obj

    @contextlib.contextmanager
    def locked(self):
        with (self.path / 'session.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.state = json.loads((self.path / 'state.json').read_text())
            if self.state['source_identity'] != source_identity():
                raise RuntimeError('Compiler/API changed since session creation; create a fresh run')
            yield

    def save(self):
        write_json(self.path / 'state.json', self.state)

    def budget_view(self):
        s = self.state
        elapsed = max(0., (s.get('finished_at') or time.time()) - s['created_at'])
        return {'used': dict(s['used'], elapsed_wall_seconds=elapsed),
                'remaining': {'wall_seconds': max(0., s['budget']['wall_seconds'] - elapsed),
                              'cpu_seconds': max(0., s['budget']['cpu_seconds'] - s['used']['cpu_seconds']),
                              'lowering_calls': max(0, s['budget']['lowering_calls'] - s['used']['lowering_calls'])}}

    def candidate(self, rid):
        if rid not in self.state['representations']:
            raise ValueError('Unknown representation')
        return self.state['representations'][rid]['representation']

    def add_rep(self, rep, parent, metadata=None):
        equivalent(rep, self.candidate(self.state['root_repr_id']))
        rid = 'r_' + stable_hash(rep)[:24]
        if rid not in self.state['representations']:
            self.state['representations'][rid] = {'representation': rep, 'statistics': stats(rep),
                                                  'parent_repr_id': parent, 'origin': metadata}
        return rid

    def normalize(self, operation, args):
        a = dict(args)
        rid = a.pop('repr_id', a.pop('parent_repr_id', None))
        self.candidate(rid)
        limits = dict(a.pop('limits', {}))
        if set(limits) - {'wall_seconds', 'cpu_seconds', 'max_pairs', 'max_unitaries', 'page_size'}:
            raise ValueError('Unknown limits')
        for v in limits.values():
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError('Limits must be positive finite numbers')
        normalized = {'repr_id': rid}
        if operation == 'propose_rewrites':
            selector = a.pop('pair_selector', 'all')
            cursor = a.pop('cursor', None)
            if cursor:
                if cursor.get('repr_id') != rid or cursor.get('pair_selector') != selector:
                    raise ValueError('Cursor belongs to a different query')
                position = cursor['position']
            else:
                position = None
            normalized.update(pair_selector=selector, cursor=position,
                              limits={k: int(limits.get(k, v)) for k, v in
                                      [('max_pairs', 2), ('max_unitaries', 64), ('page_size', 16)]})
        elif operation == 'search_placements':
            base = a.pop('base_bundle_id', None)
            if base and self.state['bundles'][base]['repr_id'] != rid:
                raise ValueError('Base bundle representation mismatch')
            config = a.pop('config', {})
            seed = a.pop('seed_index', 0)
            if not isinstance(seed, int) or seed < 0 or seed > 1000000:
                raise ValueError('Invalid seed_index')
            placement_config(config, seed)
            normalized.update(kraus_ids=a.pop('kraus_ids', 'all'), base_bundle_id=base,
                              config=config, seed_index=seed)
        elif operation == 'evaluate':
            bid = a.pop('placement_bundle_id')
            bundle = self.state['bundles'][bid]
            if bundle['repr_id'] != rid or bundle['missing_kraus_ids']:
                raise ValueError('Bundle is incomplete or belongs to another representation')
            normalized['placement_bundle_id'] = bid
        elif operation in ('run_legacy_rewrite', 'run_cost_guided_rewrite'):
            normalized.update(beam_width=int(a.pop('beam_width', 8)), max_steps=int(a.pop('max_steps', 50)),
                              trace=bool(a.pop('trace', False)))
            if normalized['beam_width'] < 1 or normalized['max_steps'] < 0:
                raise ValueError('Invalid legacy search bounds')
        if a:
            raise ValueError(f'Unknown arguments: {list(a)}')
        # Wall/CPU limits do not change a completed deterministic result.
        # Work bounds, seeds and solver options do change its identity.
        key = stable_hash({'source': self.state['source_identity'], 'operation': operation,
                           'args': normalized})
        return normalized, limits, key

    def run_worker(self, operation, normalized, limits, event_id):
        import psutil
        s = self.state
        remaining = self.budget_view()['remaining']
        wall = min(remaining['wall_seconds'], float(limits.get('wall_seconds', 300)))
        cpu = min(remaining['cpu_seconds'], float(limits.get('cpu_seconds', 300)))
        if wall <= 0 or cpu <= 0 or (operation == 'evaluate' and remaining['lowering_calls'] < 1):
            return {'status': 'budget_exhausted'}, None, {'cpu_seconds': 0., 'wall_seconds': 0., 'lowering_calls': 0}
        folder = self.path / 'artifacts' / event_id
        folder.mkdir()
        args = dict(normalized)
        rid = args.pop('repr_id')
        args.update(rep=self.candidate(rid), root=self.candidate(s['root_repr_id']), artifact_dir=str(folder))
        if operation == 'search_placements':
            base = args.pop('base_bundle_id')
            args['base_plans'] = s['bundles'][base]['plans'] if base else None
        if operation == 'evaluate':
            args['plans'] = s['bundles'][args.pop('placement_bundle_id')]['plans']
        write_json(folder / 'request.json', {'operation': HEAVY[operation], 'args': args})
        calls = int(operation == 'evaluate')
        s['used']['lowering_calls'] += calls  # Reserve before starting, including failed attempts.
        s['pending'] = {'event_id': event_id, 'started_at': time.time(), 'wall_limit': wall, 'cpu_limit': cpu}
        self.save()
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', RAYON_NUM_THREADS='1', QISKIT_PARALLEL='FALSE',
                   MPLCONFIGDIR=str(self.path / 'mpl-cache'))
        for k in list(env):
            if k.startswith('CHANNEL_LCU_'):
                del env[k]
        start = time.perf_counter()
        cpu_seen, status = {}, 'completed'
        with (folder / 'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(ROOT / 'src/search_worker.py'),
                                     str(folder / 'request.json'), str(folder / 'result.json')],
                                    stdout=log, stderr=log, env=env, start_new_session=True)
            s['pending']['pid'] = proc.pid
            self.save()
            try:
                while proc.poll() is None:
                    try:
                        parent = psutil.Process(proc.pid)
                        for process in [parent] + parent.children(recursive=True):
                            try:
                                ct = process.cpu_times()
                                cpu_seen[process.pid] = max(cpu_seen.get(process.pid, 0.), ct.user + ct.system)
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                    elapsed = time.perf_counter() - start
                    if elapsed >= wall or sum(cpu_seen.values()) >= cpu:
                        status = 'wall_timeout' if elapsed >= wall else 'cpu_timeout'
                        os.killpg(proc.pid, signal.SIGKILL)
                        break
                    time.sleep(0.02)
            finally:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                proc.wait()
        elapsed = time.perf_counter() - start
        result_file = folder / 'result.json'
        data = json.loads(result_file.read_text()) if result_file.exists() else None
        measured_cpu = max(sum(cpu_seen.values()), (data or {}).get('worker_cpu_seconds', 0.))
        # Sampling tolerance is charged, never silently accepted past the deadline.
        if status == 'completed' and (elapsed > wall or measured_cpu > cpu):
            status = 'wall_timeout' if elapsed > wall else 'cpu_timeout'
        spent = {'cpu_seconds': measured_cpu, 'wall_seconds': elapsed, 'lowering_calls': calls}
        s['used']['cpu_seconds'] += measured_cpu
        s['used']['tool_wall_seconds'] += elapsed
        s['pending'] = None
        if status != 'completed':
            return {'status': status, 'artifact_ref': str(folder)}, None, spent
        if not data or not data.get('ok'):
            return {'status': 'error', 'error': (data or {}).get('error', f'worker_exit_{proc.returncode}'),
                    'artifact_ref': str(folder)}, None, spent
        return {'status': 'completed'}, data['result'], spent

    def call(self, operation, args=None):
        args = dict(args or {})
        with self.locked():
            start_cpu = time.process_time()
            s = self.state
            if s['pending']:
                raise RuntimeError('Interrupted controller: pending worker/accounting; quarantine run and start fresh')
            if operation == 'observe':
                ids = [args['repr_id']] if 'repr_id' in args else list(s['active'])
                rows = []
                for rid in ids:
                    self.candidate(rid)
                    row = {'repr_id': rid, 'active': rid in s['active'],
                           'upstream_score': s['representations'][rid]['statistics']['upstream_score'],
                           'evaluations': [dict({k: e[k] for k in ('evaluation_id', 'placement_bundle_id', 'expected_gate_count')},
                                               seed_index=s['bundles'][e['placement_bundle_id']]['query']['seed_index'],
                                               config_ref=f"state.json#/bundles/{e['placement_bundle_id']}/effective_config")
                                           for e in s['evaluations'].values() if e['repr_id'] == rid]}
                    if args.get('include_structure'):
                        row['statistics'] = s['representations'][rid]['statistics']
                        row['representation'] = self.candidate(rid)
                    rows.append(row)
                return {'candidates': rows, 'root_repr_id': s['root_repr_id'],
                        'best_evaluation_id': s['best_evaluation_id'], 'revision': s['revision'],
                        'budget': self.budget_view(), 'finished': s['finished']}
            if operation == 'inspect_ledger':
                out = {'budget': self.budget_view(), 'revision': s['revision'],
                       'event_count': len(s['events']), 'ledger_ref': str(self.path / 'ledger.jsonl')}
                if args.get('repr_id') or args.get('detail'):
                    out['events'] = [e for e in s['events'] if not args.get('repr_id') or e.get('repr_id') == args['repr_id']]
                return out
            if operation == 'lookup_cache':
                target = args['operation']
                if target not in HEAVY:
                    raise ValueError('Only computational operations are cached')
                _, _, key = self.normalize(target, args['args'])
                cached = s['cache'].get(key)
                return {'hit': bool(cached), 'result_ref': cached['result_ref'] if cached else None,
                        'source_event': cached['source_event'] if cached else None}
            if operation == 'finish':
                s['finished'] = True
                s.setdefault('finished_at', time.time())
                s.setdefault('stop_reason', args.get('reason', 'controller_finished'))
                self.save()
                best = s['evaluations'].get(s['best_evaluation_id'])
                return {'best_evaluation_id': s['best_evaluation_id'],
                        'expected_gate_count': best['expected_gate_count'] if best else None,
                        'stop_reason': s['stop_reason'], 'budget': self.budget_view()}
            if s['finished']:
                raise RuntimeError('Run is finished; read-only queries remain available')
            s['revision'] += 1
            event_id = f"event_{s['revision']:06d}"
            event = {'event_id': event_id, 'operation': operation, 'timestamp': time.time(), 'args': args}
            if operation == 'update_frontier':
                activate, archive = list(args.get('activate', [])), list(args.get('archive', []))
                changed, rejected = {'activated': [], 'archived': []}, []
                for action, ids in [('activate', activate), ('archive', archive)]:
                    for rid in ids:
                        if rid not in s['representations'] or (rid in activate and rid in archive):
                            rejected.append(rid)
                        elif action == 'activate' and rid not in s['active']:
                            s['active'].append(rid)
                            changed['activated'].append(rid)
                        elif action == 'archive' and rid in s['active']:
                            s['active'].remove(rid)
                            changed['archived'].append(rid)
                response = dict(changed, rejected=sorted(set(rejected)), revision=s['revision'])
                spent = {'cpu_seconds': 0., 'wall_seconds': 0., 'lowering_calls': 0}
            elif operation in HEAVY:
                normalized, limits, key = self.normalize(operation, args)
                event['repr_id'] = normalized['repr_id']
                event['cache_key'] = key
                cached = s['cache'].get(key)
                if cached:
                    s['used']['cache_hits'] += 1
                    response = dict(cached['response'], cache_hit=True, source_event=cached['source_event'])
                    spent = {'cpu_seconds': 0., 'wall_seconds': 0., 'lowering_calls': 0}
                else:
                    response, result, spent = self.run_worker(operation, normalized, limits, event_id)
                    if result is not None:
                        rid = normalized['repr_id']
                        if operation == 'propose_rewrites':
                            candidates = []
                            for c in result['candidates']:
                                cid = self.add_rep(c['representation'], rid, {k:v for k,v in c.items() if k != 'representation'})
                                candidates.append({'repr_id': cid, 'upstream_score': c['statistics']['upstream_score']})
                            cursor = result['next_cursor']
                            response.update(candidates=candidates, next_cursor=(dict(repr_id=rid,
                                            pair_selector=normalized['pair_selector'], position=cursor) if cursor else None))
                        elif operation in ('run_legacy_rewrite', 'run_cost_guided_rewrite'):
                            cid = self.add_rep(result['representation'], rid, {'solver': operation, 'event_id': event_id})
                            response.update(repr_id=cid, upstream_score=result['statistics']['upstream_score'],
                                            termination=result['termination'], trace_ref=result['trace_ref'])
                        elif operation == 'search_placements':
                            # Provenance participates in identity even if two searches choose the same map.
                            bid = 'b_' + stable_hash({'repr_id': rid, 'plans': result['plans'], 'query': normalized})[:24]
                            s['bundles'][bid] = dict(result, repr_id=rid, query=normalized, source_event=event_id)
                            response.update(placement_bundle_id=bid, missing_kraus_ids=result['missing_kraus_ids'],
                                            effective_config_ref=f"state.json#/bundles/{bid}/effective_config",
                                            termination=result['termination'])
                        elif operation == 'evaluate':
                            eid = 'e_' + stable_hash({'repr_id': rid, 'bundle': normalized['placement_bundle_id']})[:24]
                            s['evaluations'][eid] = dict(result, evaluation_id=eid, repr_id=rid,
                                    placement_bundle_id=normalized['placement_bundle_id'], source_event=event_id)
                            best = s['evaluations'].get(s['best_evaluation_id'])
                            if result['verified'] and (best is None or result['expected_gate_count'] < best['expected_gate_count']):
                                s['best_evaluation_id'] = eid
                            response.update(evaluation_id=eid, expected_gate_count=result['expected_gate_count'])
                        ref = str(self.path / 'artifacts' / event_id / 'response.json')
                        write_json(ref, response)
                        s['cache'][key] = {'response': dict(response), 'source_event': event_id, 'result_ref': ref}
            else:
                raise ValueError(f'Unknown operation: {operation}')
            controller_cpu = time.process_time() - start_cpu
            s['used']['cpu_seconds'] += controller_cpu
            spent['cpu_seconds'] += controller_cpu
            response.update(event_id=event_id, spent=spent)
            event.update(response=response, spent=spent, completed_at=time.time())
            s['events'].append(event)
            self.save()
            with (self.path / 'ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
            return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('run')
    init.add_argument('--family', default='tfim')
    init.add_argument('--n', type=int, default=2)
    init.add_argument('--dt', type=float, default=0.01)
    init.add_argument('--gamma', type=float, default=0.1)
    init.add_argument('--seed', type=int, default=0)
    init.add_argument('--wall-seconds', type=float, default=600)
    init.add_argument('--cpu-seconds', type=float, default=600)
    init.add_argument('--lowering-calls', type=int, default=20)
    call = commands.add_parser('call')
    call.add_argument('run')
    call.add_argument('operation', choices=list(HEAVY)+['observe', 'update_frontier', 'inspect_ledger', 'lookup_cache', 'finish'])
    call.add_argument('--args', default='{}', help='JSON object')
    options = vars(parser.parse_args())
    command, run = options.pop('command'), options.pop('run')
    if command == 'init':
        session = Session.create(run, **options)
        response = session.call('observe')
    else:
        response = Session(run).call(options['operation'], json.loads(options['args']))
    print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
