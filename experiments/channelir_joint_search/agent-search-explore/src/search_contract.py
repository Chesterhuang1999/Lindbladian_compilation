"""Frozen numerical contract for the isolated joint-search pilot (version 1)."""
from __future__ import annotations

import hashlib
import json
import math
import itertools
import numpy as np

TOL = 1e-12
SEMANTIC_TOL = 1e-10
SCORE_ID = 'paper_H_length_with_lambda_v1'


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def pack(A, labels):
    A = np.asarray(A, dtype=complex).copy()
    A[np.abs(A) < TOL] = 0
    if not np.isfinite(A).all() or A.ndim != 2 or A.shape[1] != len(labels):
        raise ValueError('Invalid coefficient matrix')
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Expected distinct canonical Pauli columns')
    if any(len(p) != len(labels[0]) or set(p) - set('IXYZ') for p in labels):
        raise ValueError('Invalid Pauli label')
    # Exact row order and complex coefficients are part of the identity.
    order = np.argsort(labels)
    A = A[:, order]
    return {'labels': [labels[int(i)] for i in order],
            'real': A.real.tolist(), 'imag': A.imag.tolist()}


def unpack(rep):
    return np.array(rep['real']) + 1j * np.array(rep['imag']), rep['labels']


def stats(rep):
    A, _ = unpack(rep)
    supports = np.count_nonzero(np.abs(A) > TOL, axis=1)
    live = supports[supports > 0]
    if not len(live):
        raise ValueError('Zero channel is outside this pilot')
    lam = float(np.sum(np.sum(np.abs(A), axis=1) ** 2))
    tau = float(np.sum(np.abs(A) ** 2))
    outer = (len(live) - 1).bit_length()
    tmc = lambda c: 7 * (2 * max(0, c - 2) + 1)
    core = 7 * ((1 << outer) - 1)
    for k in live:
        inner = (int(k) - 1).bit_length()
        core += int(k) * tmc(outer + inner)
        core += 2 * (16 * ((1 << inner) - 1) - 7 * inner) * tmc(outer)
    return {'upstream_score': lam * core, 'score_model_id': SCORE_ID,
            'score_includes_lambda': True, 'lambda': lam, 'tau': tau,
            'success_probability': tau / lam, 'total_support': int(sum(live)),
            'row_supports': supports.tolist(), 'live_kraus_count': len(live)}


def equivalent(rep, root):
    A, labels = unpack(rep)
    B, other = unpack(root)
    if labels != other:
        raise ValueError('Canonical Pauli columns differ')
    residual = float(np.linalg.norm(A.conj().T @ A - B.conj().T @ B))
    scale = max(1., float(np.linalg.norm(B.conj().T @ B)))
    if residual > SEMANTIC_TOL * scale:
        raise ValueError(f'Kraus Gram mismatch: {residual}')
    return residual / scale


def matrixsum(row, labels):
    from channel_IR import Matrixsum, PauliAtom
    return Matrixsum([(PauliAtom(label, complex(c / abs(c))), float(abs(c)))
                      for label, c in zip(labels, row) if abs(c) > TOL])


def channel_from_rep(rep):
    from channel_IR import channel
    A, labels = unpack(rep)
    # Avoid the legacy constructor's implicit global-phase rewrite.
    ir = channel.__new__(channel)
    ir.kraus_ops = [matrixsum(row, labels) for row in A]
    ir.size = len(labels[0])
    return ir


def build_input(family='tfim', n=2, dt=0.01, gamma=0.1, seed=0):
    """Euler Kraus input; finite-dt map is CP, not assumed exactly TP."""
    if n < 1 or n > 20 or dt <= 0 or gamma < 0:
        raise ValueError('Invalid instance parameters')
    def label(chars):
        p = ['I'] * n
        for q, c in chars.items():
            p[n - 1 - q] = c
        return ''.join(p)
    rows = []
    if family == 'tfim':
        # H=-sum Xi-sum Zi Zi+1; Li=sqrt(gamma)/2 (Xi-iYi).
        first = {'I' * n: 1 - dt * n * gamma / 4}
        for q in range(n):
            first[label({q: 'X'})] = 1j * dt
            first[label({q: 'Z'})] = -dt * gamma / 4
        for q in range(n - 1):
            first[label({q: 'Z', q + 1: 'Z'})] = 1j * dt
        rows = [first] + [{label({q: 'X'}): math.sqrt(dt * gamma) / 2,
                          label({q: 'Y'}): -1j * math.sqrt(dt * gamma) / 2}
                         for q in range(n)]
    elif family in ('hrw', 'hrw_scrambled'):
        # Section 5.2 hypercube channel: T_i,b^dagger / sqrt(2n).
        c = 1 / (2 * math.sqrt(2*n))
        rows = [{'I'*n: c, label({q: 'X'}): c, label({q: 'Z'}): s*c,
                 label({q: 'Y'}): -1j*s*c} for q in range(n) for s in (1, -1)]
        if family == 'hrw_scrambled':
            # Bijective column relabeling preserves supports, Lambda and the
            # row-unitary search landscape, but changes downstream Pauli algebra.
            # This is a new CP input, not an equivalent representation of HRW.
            rng = np.random.default_rng(seed)
            old = sorted(set().union(*(set(row) for row in rows)))
            indices = rng.choice(4**n, size=len(old), replace=False)
            def decode(index):
                out = ''
                for _ in range(n):
                    out = 'IXYZ'[int(index) % 4] + out
                    index //= 4
                return out
            mapping = dict(zip(old, map(decode, indices)))
            rows = [{mapping[p]: v for p, v in row.items()} for row in rows]
    elif family == 'random_pauli':
        # Input-only reproduction of build_random_channel_lcu_benchmark.py.
        # Tuples are (Kraus count, Hilbert dimension, first terms, other terms).
        presets = {2: (5, 10, 2), 3: (9, 20, 4)}
        if n not in presets:
            raise ValueError('random_pauli supports n=2 or n=3')
        m, first, other = presets[n]
        pool = [''.join(p) for p in itertools.product('IXYZ', repeat=n) if set(p) != {'I'}]
        rng = np.random.default_rng(seed)
        phases = np.array([1, -1, 1j, -1j], dtype=complex)
        for j in range(m):
            chosen = rng.choice(pool, size=first if j == 0 else other, replace=False)
            rows.append({str(p): complex(rng.choice(phases))*float(rng.uniform(.5, 1.5)) for p in chosen})
    elif family == 'pauli_mixture':
        rng = np.random.default_rng(seed)
        weights = rng.dirichlet(np.ones(n + 1))
        rows = [{'I' * n: math.sqrt(weights[0])}] + [
            {label({q: 'X'}): math.sqrt(weights[q + 1])} for q in range(n)]
    elif family == 'mixed_pauli':
        base = build_input('pauli_mixture', n, dt, gamma, seed)
        A, labels = unpack(base)
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.normal(size=(n+1, n+1)) + 1j*rng.normal(size=(n+1, n+1)))
        return pack(Q @ A, labels)
    else:
        raise ValueError('Unsupported input family')
    labels = sorted(set().union(*(set(row) for row in rows)))
    return pack([[row.get(p, 0) for p in labels] for row in rows], labels)


def placement_config(config, seed):
    """Complete, environment-independent small fixed-backend profile."""
    defaults = dict(beam_width=16, layer_candidate_limit=16, max_exact_layer_assignments=16,
                    future_exact_address_limit=8, future_weight=1.0,
                    lookahead_depth=0, lookahead_width=16, lookahead_start_weight=1,
                    initial_strategy='mixed', mixed_initial_beam_width=None,
                    mixed_full_initial_beam_width=128, mixed_nonfull_initial_beam_width=128,
                    mixed_low_weight_count=128, mixed_random_count=256,
                    subgroup_initial_beam_width=None, subgroup_candidate_count=256,
                    subgroup_random_count=1024, subgroup_max_distance=1,
                    swap_refinement=True, swap_refinement_max_terms=16)
    if set(config) - set(defaults):
        raise ValueError(f'Unsupported placement options: {set(config) - set(defaults)}')
    defaults.update(config)
    if defaults['lookahead_depth'] not in (0, 1):
        raise ValueError('lookahead_depth must be 0 or 1')
    for k, v in defaults.items():
        if v is not None and isinstance(v, (float, int)) and not isinstance(v, bool):
            if not math.isfinite(v) or v < 0 or v > 100000:
                raise ValueError(f'Out-of-range search control: {k}')
    for k in ('beam_width', 'layer_candidate_limit', 'max_exact_layer_assignments'):
        if defaults[k] < 1:
            raise ValueError(f'{k} must be positive')
    defaults.update(mixed_random_seed=20260608 + seed,
                    subgroup_random_seed=20260610 + seed, padding_random_seed=20260609 + seed)
    return defaults
