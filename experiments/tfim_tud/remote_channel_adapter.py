"""Adapt independently compiled Kraus blocks to the existing opt channel tree."""
from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit import QuantumCircuit

from adapter_channelir import compile_existing_pauli_block
from block_encoding_new_fixed import BlockEncoding as FixedBlockEncoding
from subroutine import mulplex_U_tree, prep_sup_state
from tfim_tud_reference import (build_factored_tud, build_separate_tud,
                                dense_from_terms, factor_tud, lower_to_u_cx)


@dataclass
class ChannelBranch:
    circuit: QuantumCircuit
    beta: float
    metadata: dict


def branch_terms(branch, n):
    terms = [(r['label'], complex(r['real'], r['imag']))
             for r in branch['canonical_terms']]
    supports = branch.get('local_support_qubits')
    if supports is not None:
        targets = tuple(supports)
        if not targets or targets != tuple(sorted(set(targets))) or any(q < 0 or q >= n for q in targets):
            raise ValueError('Invalid local support qubits')
        for label, _ in terms:
            if len(label) != n or any(p != 'I' for q, p in enumerate(reversed(label)) if q not in targets):
                raise ValueError('Declared local support omits nonidentity terms')
        return [(''.join(label[n - 1 - q] for q in reversed(targets)), c)
                for label, c in terms], len(targets), targets
    support = branch.get('local_support_qubit')
    if support is None:
        return terms, n, tuple(range(n))
    if not 0 <= support < n:
        raise ValueError('Invalid local support')
    for label, _ in terms:
        if any(p != 'I' for q, p in enumerate(reversed(label)) if q != support):
            raise ValueError('Declared local jump has nonlocal terms')
    return [(label[n - 1 - support], c) for label, c in terms], 1, (support,)


def reorder_and_embed(circuit, system, success, work, n, target_system):
    """Tree branches require all inner ancillas first, then the entire system."""
    system, success, work = tuple(system), tuple(success), tuple(work)
    roles = success + work + system
    if sorted(roles) != list(range(circuit.num_qubits)):
        raise ValueError('Incomplete or overlapping branch qubit roles')
    if len(system) != len(target_system) or len(set(target_system)) != len(target_system):
        raise ValueError('Invalid system embedding')
    if any(q < 0 or q >= n for q in target_system):
        raise ValueError('System embedding out of range')
    anc = len(success) + len(work)
    mapping = [None] * circuit.num_qubits
    for new, old in enumerate(success + work):
        mapping[old] = new
    for old, target in zip(system, target_system):
        mapping[old] = anc + target
    out = QuantumCircuit(anc + n, name=circuit.name + '_tree_layout')
    out.compose(circuit, qubits=mapping, inplace=True)
    return out, {'source_system': system, 'source_success': success, 'source_work': work,
                 'old_to_tree_qubit': mapping, 'tree_inner_ancillas': tuple(range(anc)),
                 'tree_system': tuple(range(anc, anc + n)), 'local_target_system': target_system}


def compile_branch(branch, n, method, timed):
    terms, support_n, targets = branch_terms(branch, n)
    metadata = {'support_system_qubits': support_n, 'local_target_system': targets,
                'canonical_name': branch['canonical_name'],
                'source_kraus_index': branch['source_kraus_index']}
    if method == 'pauli_placement':
        block = timed('pauli_full_block_build', lambda: compile_existing_pauli_block(
            terms, support_n, placement_enabled=True, backend_class=FixedBlockEncoding))
        raw, beta = block.circuit, block.normalization
        system, success, work = block.system_qubits, block.success_qubits, block.clean_work_qubits
        metadata['pauli_adapter'] = block.metadata
        metadata['inner_opt'] = 'Matrix-order'
        metadata['inner_opt_reason'] = ('Use the isolated corrected placement backend for every Kraus; '
            'mobius_residual_fixed includes the nonidentity zero-address factor at every layer')
    else:
        dense = timed('dense_materialization', lambda: dense_from_terms(terms))
        factors = timed('svd_and_diagonal_vectors', lambda: factor_tud(dense))
        beta = factors.beta
        metadata['factorization'] = {'beta': beta, 'clipped_entries': factors.clipped_entries,
                                     'max_overshoot': factors.max_overshoot}
        builder = {'tud_factored_diagonal': build_factored_tud, 'tud_separate': build_separate_tud}[method]
        raw = timed('qsd_control_and_full_assembly', lambda: builder(factors))
        system, success, work = tuple(range(support_n)), (support_n,), ()
    native = timed('native0_lowering', lambda: lower_to_u_cx(raw, optimization_level=0))
    embedded, layout = timed('tree_register_adaptation',
                             lambda: reorder_and_embed(native, system, success, work, n, targets))
    metadata['layout'] = layout
    return ChannelBranch(embedded, beta, metadata)


def assemble_opt(branches, n):
    if len(branches) < 2:
        raise ValueError('This experiment requires at least two Kraus operators')
    betas = [branch.beta for branch in branches]
    if any(not math.isfinite(beta) or beta <= 0 for beta in betas):
        raise ValueError('Every retained branch must have a positive finite beta')
    select_n = (len(branches) - 1).bit_length()
    inner_n = max(branch.circuit.num_qubits - n for branch in branches)
    select = tuple(2 * j for j in range(select_n))
    tree = tuple(2 * j + 1 for j in range(select_n))
    inner = tuple(range(2 * select_n, 2 * select_n + inner_n))
    system = tuple(range(2 * select_n + inner_n, 2 * select_n + inner_n + n))
    circuit = QuantumCircuit(2 * select_n + inner_n + n, name='channel_lcu_opt')
    circuit.compose(prep_sup_state(betas), qubits=select, inplace=True)
    circuit, legacy_cx, legacy_t, legacy_mc = mulplex_U_tree(
        circuit, [branch.circuit for branch in branches], select_n, inner_n, n)
    return circuit, {'structure': 'opt', 'lambda': sum(beta**2 for beta in betas),
                     'betas': betas, 'select_qubits': select, 'tree_ancillas': tree,
                     'inner_ancillas': inner, 'system_qubits': system,
                     'success_qubits': tree + inner,
                     'select_contract': 'branch index bits in MSB-first order on select_qubits; retain label',
                     'legacy_tree_estimates_not_native_counts':
                         {'cx': legacy_cx, 't': legacy_t, 'mc': legacy_mc},
                     'tree_internal_transpile': 'opt1 u3/cx counting only; controlled input circuit is unchanged'}


def zero_input_channel_trace(branches, n):
    """Apply Pauli terms to |0^n> without building a dense operator."""
    total = 0.0
    for branch in branches:
        amplitudes = {}
        for row in branch['canonical_terms']:
            target, phase = 0, 1 + 0j
            for q, pauli in enumerate(reversed(row['label'])):
                if pauli in 'XY':
                    target |= 1 << q
                if pauli == 'Y':
                    phase *= 1j
            amplitudes[target] = amplitudes.get(target, 0j) + complex(row['real'], row['imag']) * phase
        total += sum(abs(a)**2 for a in amplitudes.values())
    return float(total)
