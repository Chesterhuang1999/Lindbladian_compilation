"""Apply the existing channel rewrite to the frozen HRW input family."""
import copy
import numpy as np

from adapter_channelir import canonical_terms
from channel_IR import Matrixsum, PauliAtom, channel as ChannelIR

REWRITE_PARAMETERS = {'strategy': 'greedy', 'beam_width': 3, 'max_steps': 50,
                      'tol': 1e-12, 'verbose': False}


def input_operators(data):
    operators = []
    for branch in data['branches']:
        terms = canonical_terms([(r['label'], complex(r['real'], r['imag']))
                                 for r in branch['canonical_terms']], data['n'])
        operators.append(Matrixsum([(PauliAtom(label, phase=value / abs(value)), abs(value))
                                    for label, value in terms]))
    return operators


def rewrite_operators(operators):
    # Follow the paper runner: search a copy, apply the result to a fresh copy.
    source = ChannelIR(copy.deepcopy(operators))
    source.zero_elim()
    result = source.rewrite_search(**REWRITE_PARAMETERS)
    rewritten = ChannelIR(copy.deepcopy(operators))
    rewritten.apply_rewrite_result(result, tol=REWRITE_PARAMETERS['tol'])
    rewritten.zero_elim()
    return rewritten.kraus_ops, result


def freeze_rewritten(operators, n):
    branches = []
    for index, operator in enumerate(operators):
        terms = canonical_terms([(atom.expr, complex(atom.phase * coefficient))
                                 for atom, coefficient in operator.instances], n)
        if not terms:
            continue
        support = sorted({q for label, c in terms for q, p in enumerate(reversed(label)) if p != 'I'})
        branches.append({'canonical_name': f'R_{index}', 'source_kraus_index': index,
            'source_kraus_index_semantics': 'rewritten row, not original Kraus index',
            'local_support_qubits': support or [0], 'effective_support_size': len(support),
            'canonical_terms': [{'label': label, 'real': c.real, 'imag': c.imag} for label, c in terms]})
    return branches


def coefficient_matrix(branches, labels):
    columns = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(branches), len(labels)), complex)
    for j, branch in enumerate(branches):
        for term in branch['canonical_terms']:
            matrix[j, columns[term['label']]] += complex(term['real'], term['imag'])
    return matrix


def audit_rewrite(data, branches, result):
    labels = result['labels']
    before = coefficient_matrix(data['branches'], labels)
    after = coefficient_matrix(branches, labels)
    replay = before.copy()
    max_unitarity_error = 0.
    steps = []
    for pair, unitary, support in result['steps']:
        max_unitarity_error = max(max_unitarity_error, float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(2))))
        replay[list(pair)] = unitary @ replay[list(pair)]
        steps.append({'pair': list(pair), 'unitary_real': unitary.real.tolist(),
                      'unitary_imag': unitary.imag.tolist(), 'support_after': int(support)})
    replay_error = float(np.max(np.abs(replay - result['A_final'])))
    gram_error = float(np.linalg.norm(before.conj().T @ before - after.conj().T @ after, ord=np.inf))
    assert max_unitarity_error < 1e-10 and replay_error < 1e-10 and gram_error < 1e-10
    retained = result['A_final'][np.any(np.abs(result['A_final']) > REWRITE_PARAMETERS['tol'], axis=1)]
    np.testing.assert_allclose(after, retained, atol=1e-12, rtol=1e-12)
    return {'parameters': REWRITE_PARAMETERS, 'steps': steps, 'step_count': len(steps),
        'termination': result['termination'], 'initial_support': int(result['initial_support']),
        'final_support': int(result['final_support']),
        'initial_kraus_count': len(data['branches']), 'final_kraus_count': len(branches),
        'terms_per_kraus_before': [len(b['canonical_terms']) for b in data['branches']],
        'terms_per_kraus_after': [len(b['canonical_terms']) for b in branches],
        'effective_support_per_kraus_after': [b['effective_support_size'] for b in branches],
        'coefficient_gram_inf_error': gram_error, 'step_replay_max_error': replay_error,
        'max_rotation_unitarity_error': max_unitarity_error,
        'channel_equivalence_check': 'C_before^dagger C_before = C_after^dagger C_after in the common Pauli basis; no dense n-qubit simulation'}, before, after
