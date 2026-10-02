"""Isolated CPU worker. Only the controller writes authoritative session state."""
from __future__ import annotations
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from search_contract import (TOL, SEMANTIC_TOL, pack, unpack, stats, equivalent,
                             matrixsum, channel_from_rep, stable_hash, placement_config)


def propose(args):
    from channel_IR import channel
    A, labels = unpack(args['rep'])
    pairs = (list(itertools.combinations(range(len(A)), 2)) if args['pair_selector'] == 'all'
             else [tuple(p) for p in args['pair_selector']])
    if any(len(p) != 2 or p[0] == p[1] or min(p) < 0 or max(p) >= len(A) for p in pairs):
        raise ValueError('Invalid row pair')
    pair_idx, unit_idx = args.get('cursor') or (0, 0)
    if pair_idx < 0 or pair_idx > len(pairs) or unit_idx < 0:
        raise ValueError('Invalid rewrite cursor')
    out, seen = [], set()
    touched, work = 0, 0
    limits = args['limits']
    while pair_idx < len(pairs) and touched < limits['max_pairs']:
        i, j = pairs[pair_idx]
        us = channel._candidate_unitaries_for_pair(A, i, j, TOL)
        touched += 1
        while unit_idx < len(us):
            if work >= limits['max_unitaries'] or len(out) >= limits['page_size']:
                return {'candidates': out, 'next_cursor': [pair_idx, unit_idx], 'unitaries_examined': work}
            U = us[unit_idx]
            unit_idx += 1
            work += 1
            B = channel._apply_unitary(A, i, j, U)
            B[np.abs(B) < TOL] = 0
            rep = pack(B, labels)
            key = stable_hash(rep)
            if key in seen or np.allclose(A, B, atol=TOL, rtol=TOL):
                continue
            seen.add(key)
            residual = equivalent(rep, args['root'])
            out.append({'representation': rep, 'statistics': stats(rep), 'gram_residual': residual,
                        'pair': [i, j], 'unitary': {'real': U.real.tolist(), 'imag': U.imag.tolist()},
                        'raw_generator_index': unit_idx - 1})
        pair_idx += 1
        unit_idx = 0
    return {'candidates': out, 'next_cursor': [pair_idx, unit_idx] if pair_idx < len(pairs) else None,
            'unitaries_examined': work}


def legacy(args):
    ir = channel_from_rep(args['rep'])
    labels = ir._coeff_matrix(TOL)[1]
    trace_path = Path(args['artifact_dir']) / 'legacy_trace.jsonl'
    with trace_path.open('w') as stream:
        def observer(reason, **kw):
            row = {'reason': reason}
            for k, v in kw.items():
                if k in ('parent', 'candidate'):
                    row[k] = pack(v, labels)
                elif k == 'unitary':
                    row[k] = {'real': v.real.tolist(), 'imag': v.imag.tolist()}
                else:
                    row[k] = v
            stream.write(json.dumps(row) + '\n')
        result = ir.rewrite_search(strategy='beam', beam_width=args['beam_width'],
                                   max_steps=args['max_steps'], tol=TOL,
                                   observer=observer if args.get('trace') else None)
    # _coeff_matrix keeps first-seen columns; retain result's actual labels.
    rep = pack(result['A_final'], result['labels'])
    equivalent(rep, args['root'])
    return {'representation': rep, 'statistics': stats(rep), 'termination': result['termination'],
            'trace_ref': str(trace_path) if args.get('trace') else None}


def costguided(args):
    from search_contract import SCORE_ID
    ir = channel_from_rep(args['rep'])
    labels = ir._coeff_matrix(TOL)[1]
    result = ir.gatecost_rewrite_search(
        strategy='beam', beam_width=args['beam_width'], max_steps=args['max_steps'],
        tol=TOL, score_fn=lambda A: stats(pack(A, labels))['upstream_score'],
        score_model_id=SCORE_ID)
    rep = pack(result['A_final'], result['labels'])
    equivalent(rep, args['root'])
    return {'representation': rep, 'statistics': stats(rep),
            'termination': dict(result['termination'], score_model_id=result['cost_model']),
            'trace_ref': None}


def search(args):
    from block_encoding_new_fixed import BlockEncoding
    A, labels = unpack(args['rep'])
    plans = dict(args.get('base_plans') or {})
    selected = ([j for j, row in enumerate(A) if np.any(abs(row) > TOL)]
                if args['kraus_ids'] == 'all' else args['kraus_ids'])
    config = placement_config(args['config'], args['seed_index'])
    debug = {}
    for j in selected:
        if not isinstance(j, int) or j < 0 or j >= len(A) or not np.any(abs(A[j]) > TOL):
            raise ValueError('Invalid live Kraus id')
        op = matrixsum(A[j], labels)
        if len(op.instances) == 1:
            atom, _ = op.instances[0]
            plans[str(j)] = {'width': 0, 'placement': [[atom.expr, '']],
                             'debug': {'padding_count': 0, 'padding_labels': []}, 'single_term': True}
            debug[str(j)] = {'shortcut': 'single_term'}
        else:
            block = BlockEncoding(op, placement_config=config)
            block.find_optimal_order_matrices()
            plans[str(j)] = block.exported_placement_plan
            debug[str(j)] = block.placement_search_debug
    return {'plans': plans, 'effective_config': config, 'debug': debug,
            'missing_kraus_ids': [j for j, row in enumerate(A)
                                  if np.any(abs(row) > TOL) and str(j) not in plans],
            'termination': 'completed'}


def verify_modes(op, block):
    """Polynomial-size check of every real SELECT address, including complex phase."""
    from qiskit.quantum_info import Pauli
    target = {atom.expr: complex(atom.phase) for atom, _ in op.instances}
    worst = 0.
    for label, address in block.exported_placement_plan['placement']:
        if label not in target:
            continue
        b = int(address, 2)
        acc, phase = Pauli('I' * op.size), 1 + 0j
        for p, a, phi in block.mobius_phase_result['g_modes_with_phase']:
            s = int(a, 2)
            if (s & b) == s:
                acc = Pauli(p) @ acc
                phase *= np.exp(1j * np.pi * phi / 2)
        phase *= (-1j) ** int(acc.phase)
        acc.phase = 0
        if acc.to_label() != label:
            raise ValueError('SELECT reconstruction Pauli mismatch')
        worst = max(worst, abs(phase - target[label]))
    if worst > SEMANTIC_TOL:
        raise ValueError(f'SELECT reconstruction phase mismatch: {worst}')
    return worst


def evaluate(args):
    started = time.perf_counter()
    from qiskit import QuantumCircuit, transpile, qpy
    from qiskit.quantum_info import Pauli, Statevector, SparsePauliOp
    from block_encoding_new_fixed import BlockEncoding
    from subroutine import prep_sup_state, mulplex_U_tree
    A, labels = unpack(args['rep'])
    equivalent(args['rep'], args['root'])
    n = len(labels[0])
    branches, alphas, residuals, live_rows = [], [], [], []
    for j, row in enumerate(A):
        if not np.any(abs(row) > TOL):
            continue
        plan = args['plans'].get(str(j))
        if plan is None:
            raise ValueError('Cannot evaluate an incomplete placement bundle')
        op = matrixsum(row, labels)
        if len(op.instances) == 1:
            atom, _ = op.instances[0]
            if plan.get('placement') != [[atom.expr, '']] or plan['width'] != 0:
                raise ValueError('Wrong single-term plan')
            raw = QuantumCircuit(n)
            raw.append(Pauli(atom.expr), range(n))
            raw.global_phase = float(np.angle(atom.phase))
            ctrl_size, residual = 0, 0.
        else:
            block = BlockEncoding(op, placement_plan=plan)
            raw = block.circuit(opt='Matrix-order')
            ctrl_size = int(block.selected_k)
            residual = verify_modes(op, block)
        anc_size = raw.num_qubits - n
        # Legacy fixed block order: [ctrl, sys, work]; tree expects [ctrl, work, sys].
        mapping = list(range(ctrl_size)) + list(range(anc_size, anc_size+n)) + list(range(ctrl_size, anc_size))
        reordered = QuantumCircuit(raw.num_qubits)
        reordered.compose(raw, qubits=mapping, inplace=True)
        branches.append(reordered)
        alphas.append(float(sum(abs(row))))
        residuals.append(residual)
        live_rows.append(row)
    branch_done = time.perf_counter()
    select_n = (len(branches)-1).bit_length()
    inner_n = max(c.num_qubits - n for c in branches)
    circuit = QuantumCircuit(2*select_n + inner_n + n)
    select = [2*j for j in range(select_n)]
    if select_n:
        circuit.compose(prep_sup_state(alphas), qubits=select, inplace=True)
    circuit, _, _, _ = mulplex_U_tree(circuit, branches, select_n, inner_n, n)
    assembly_done = time.perf_counter()
    details = stats(args['rep'])
    semantic = {'gram_relative_residual': equivalent(args['rep'], args['root']),
                'select_max_phase_residual': max(residuals),
                'full_channel_check': 'algebraic_plus_tested_assembly'}
    # Exhaustive success-branch check on small enough full circuits; no dense unitary.
    if n <= 3 and circuit.num_qubits <= 18:
        worst = 0.
        anc_total = 2*select_n + inner_n
        for x in range(1 << n):
            state = Statevector.from_int(x << anc_total, 1 << circuit.num_qubits).evolve(circuit).data
            for j, row in enumerate(live_rows):
                select_bits = [(j >> (select_n - 1 - k)) & 1 for k in range(select_n)]
                offset = sum(bit << q for bit, q in zip(select_bits, select))
                got = state[offset + (np.arange(1 << n) << anc_total)]
                target = SparsePauliOp.from_list([(p, c) for p, c in zip(labels, row)
                                                  if abs(c) > TOL]).to_matrix()[:, x] / np.sqrt(details['lambda'])
                worst = max(worst, float(np.max(abs(got-target))))
        if worst > SEMANTIC_TOL:
            raise ValueError(f'Full-channel success branch mismatch: {worst}')
        semantic.update(full_channel_check='exhaustive_input_basis', full_channel_max_residual=worst)
    verification_done = time.perf_counter()
    native = transpile(circuit, basis_gates=['rz', 'h', 'x', 'cx'], optimization_level=0,
                       seed_transpiler=0)
    lowering_done = time.perf_counter()
    counts = {str(k): int(v) for k, v in native.count_ops().items()}
    if set(counts) - {'rz', 'h', 'x', 'cx', 'reset', 'barrier'}:
        raise ValueError('Non-NAM output')
    gates = sum(counts.get(k, 0) for k in ('rz', 'h', 'x', 'cx'))
    path = Path(args['artifact_dir']) / 'full_channel_nam.qpy'
    with path.open('wb') as stream:
        qpy.dump(native, stream)
    return {'expected_gate_count': gates / details['success_probability'], 'gates': gates,
            'success_probability': details['success_probability'], 'lambda': details['lambda'],
            'tau': details['tau'], 'width': native.num_qubits, 'gate_histogram': counts,
            'timings': {'blocks_seconds': branch_done-started,
                        'channel_assembly_seconds': assembly_done-branch_done,
                        'semantic_check_seconds': verification_done-assembly_done,
                        'NAM_lowering_seconds': lowering_done-verification_done},
            'semantic_validation': semantic, 'circuit_ref': str(path), 'verified': True}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    request, output = map(Path, sys.argv[1:3])
    data = json.loads(request.read_text())
    start = time.perf_counter()
    try:
        result = {'ok': True, 'result': globals()[data['operation']](data['args'])}
    except Exception as error:
        result = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result['worker_cpu_seconds'] = usage.ru_utime + usage.ru_stime
    result['worker_body_wall_seconds'] = time.perf_counter() - start
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, default=json_default, allow_nan=False))
    os.replace(tmp, output)


if __name__ == '__main__':
    main()
