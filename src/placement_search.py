"""Layered beam search for Pauli placement in matrix-order block encoding.

This module keeps the search independent from ``BlockEncoding`` so that the
placement heuristic can be benchmarked before it is wired into circuit
synthesis.  It optimizes the phase-free proxy

    sum_l hw(l) * pauli_weight(g_l),

where ``g_l`` is the Boolean-lattice Möbius residual of the placed Pauli table.
The implementation is heuristic: it searches control addresses layer by layer
according to Hamming weight and keeps a bounded beam of candidate states.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations, permutations
from math import ceil, comb, factorial, inf, log2
from typing import Iterable

from channel_IR import Matrixsum, PauliAtom


@dataclass(frozen=True)
class _SearchState:
    remaining: tuple[int, ...]
    prefix: tuple[int, ...]
    cost: int
    placement: tuple[tuple[int, int], ...]


def _default_control_size(term_count: int) -> int:
    if term_count <= 1:
        return 0
    return int(ceil(log2(term_count)))


def _label_to_bits(label: str) -> int:
    """Encode a Pauli label as x|z bits packed into a Python int."""
    n = len(label)
    bits = 0
    for q, char in enumerate(label):
        if char in ("X", "Y"):
            bits |= 1 << q
        if char in ("Z", "Y"):
            bits |= 1 << (n + q)
        if char not in ("I", "X", "Y", "Z"):
            raise ValueError(f"Unsupported Pauli character {char!r} in {label!r}.")
    return bits


def _bits_to_label(bits: int, num_qubits: int) -> str:
    label = []
    for q in range(num_qubits):
        x_bit = (bits >> q) & 1
        z_bit = (bits >> (num_qubits + q)) & 1
        if x_bit == 0 and z_bit == 0:
            label.append("I")
        elif x_bit == 1 and z_bit == 0:
            label.append("X")
        elif x_bit == 0 and z_bit == 1:
            label.append("Z")
        else:
            label.append("Y")
    return "".join(label)


def _pauli_weight(bits: int, num_qubits: int) -> int:
    weight = 0
    for q in range(num_qubits):
        if ((bits >> q) & 1) or ((bits >> (num_qubits + q)) & 1):
            weight += 1
    return weight


def _pauli_y_count(label: str) -> int:
    return sum(char == "Y" for char in label)


def _gf2_rank(vectors: Iterable[int]) -> int:
    basis: dict[int, int] = {}
    for vector in vectors:
        value = vector
        while value:
            pivot = value.bit_length() - 1
            existing = basis.get(pivot)
            if existing is None:
                basis[pivot] = value
                break
            value ^= existing
    return len(basis)


def _is_independent_of_basis(vector: int, basis: dict[int, int]) -> bool:
    value = vector
    while value:
        pivot = value.bit_length() - 1
        existing = basis.get(pivot)
        if existing is None:
            return True
        value ^= existing
    return False


def _add_to_gf2_basis(vector: int, basis: dict[int, int]) -> bool:
    value = vector
    while value:
        pivot = value.bit_length() - 1
        existing = basis.get(pivot)
        if existing is None:
            basis[pivot] = value
            return True
        value ^= existing
    return False


def _low_weight_independent_terms(
    term_indices: Iterable[int],
    vectors: list[int],
    labels: list[str],
    num_qubits: int,
) -> list[int]:
    basis: dict[int, int] = {}
    selected = []
    for term_idx in sorted(
        term_indices,
        key=lambda idx: (
            _pauli_weight(vectors[idx], num_qubits),
            _pauli_y_count(labels[idx]),
            labels[idx],
            idx,
        ),
    ):
        if _add_to_gf2_basis(vectors[term_idx], basis):
            selected.append(term_idx)
    return selected


def _span_coordinate_map(basis_vectors: list[int]) -> dict[int, int]:
    span = {0: 0}
    for basis_pos, basis_vector in enumerate(basis_vectors):
        updates = {
            vector ^ basis_vector: coord | (1 << basis_pos)
            for vector, coord in list(span.items())
        }
        span.update(updates)
    return span


def _is_axis_vector(bits: int, num_qubits: int, axis: str) -> bool:
    x_mask = (1 << num_qubits) - 1
    x_part = bits & x_mask
    z_part = bits >> num_qubits
    if bits == 0:
        return False
    if axis == "z":
        return x_part == 0 and z_part != 0
    if axis == "x":
        return z_part == 0 and x_part != 0
    raise ValueError(f"Unsupported axis {axis!r}.")


def _placement_cost_from_index_modes(
    placement: list[tuple[int, int]],
    vectors: list[int],
    num_qubits: int,
) -> int:
    p_table = {address: vectors[term_idx] for term_idx, address in placement}
    g_table: dict[int, int] = {}
    total = 0
    for address in sorted(p_table):
        g_bits = p_table[address]
        for lower, lower_g in g_table.items():
            if (lower & address) == lower:
                g_bits ^= lower_g
        g_table[address] = g_bits
        total += address.bit_count() * _pauli_weight(g_bits, num_qubits)
    return total


def _placement_cost_from_index_and_label_modes(
    index_placement: list[tuple[int, int]],
    label_placement: list[tuple[str, int]],
    vectors: list[int],
    num_qubits: int,
) -> int:
    p_table = {address: vectors[term_idx] for term_idx, address in index_placement}
    for label, address in label_placement:
        p_table[address] = _label_to_bits(label)
    g_table: dict[int, int] = {}
    total = 0
    for address in sorted(p_table):
        g_bits = p_table[address]
        for lower, lower_g in g_table.items():
            if (lower & address) == lower:
                g_bits ^= lower_g
        g_table[address] = g_bits
        total += address.bit_count() * _pauli_weight(g_bits, num_qubits)
    return total


def _padded_placement_from_index_and_label_modes(
    index_placement: list[tuple[int, int]],
    label_placement: list[tuple[str, int]],
    labels: list[str],
    width: int,
) -> list[tuple[str, str]]:
    rows = [
        (labels[term_idx], format(address, f"0{width}b"))
        for term_idx, address in index_placement
    ]
    rows.extend(
        (label, format(address, f"0{width}b"))
        for label, address in label_placement
    )
    return sorted(rows, key=lambda item: int(item[1], 2))


def _placement_from_index_modes(
    index_placement: list[tuple[int, int]],
    labels: list[str],
    width: int,
    real_output_indices: set[int],
) -> list[tuple[str, str]]:
    return [
        (labels[term_idx], format(address, f"0{width}b"))
        for term_idx, address in sorted(index_placement, key=lambda item: item[1])
        if term_idx in real_output_indices
    ]


def _try_subgroup_span_shortcut(
    searchable_indices: list[int],
    vectors: list[int],
    labels: list[str],
    *,
    width: int,
    num_qubits: int,
) -> tuple[list[tuple[int, int]], list[tuple[str, int]], dict[str, object]] | None:
    if len(searchable_indices) == 0:
        return None

    vector_to_term: dict[int, int] = {}
    for term_idx in searchable_indices:
        vector = vectors[term_idx]
        if vector == 0 or vector in vector_to_term:
            return None
        vector_to_term[vector] = term_idx

    rank = _gf2_rank(vector_to_term)
    if rank > width:
        return None

    generator_terms = _low_weight_independent_terms(
        searchable_indices,
        vectors,
        labels,
        num_qubits,
    )
    if len(generator_terms) != rank:
        return None

    basis_vectors = [vectors[term_idx] for term_idx in generator_terms]
    span = _span_coordinate_map(basis_vectors)

    placement = []
    padding_placement = []
    for vector, address in span.items():
        if vector == 0:
            continue
        term_idx = vector_to_term.get(vector)
        if term_idx is None:
            padding_placement.append((_bits_to_label(vector, num_qubits), address))
        else:
            placement.append((term_idx, address))

    if len(placement) != len(searchable_indices):
        return None

    return placement, padding_placement, {
        "shortcut": "subgroup-span",
        "shortcut_rank": rank,
        "shortcut_generators": [labels[term_idx] for term_idx in generator_terms],
        "shortcut_cost": sum(
            _pauli_weight(vectors[term_idx], num_qubits)
            for term_idx in generator_terms
        ),
        "shortcut_nonidentity_g_count": rank,
        "shortcut_nonbasis_nonidentity_g_count": 0,
        "shortcut_padding_labels": [label for label, _address in padding_placement],
        "shortcut_padding_count": len(padding_placement),
        "shortcut_span_size": (1 << rank) - 1,
    }


def _try_axis_span_shortcut(
    searchable_indices: list[int],
    vectors: list[int],
    labels: list[str],
    *,
    width: int,
    num_qubits: int,
) -> tuple[list[tuple[int, int]], dict[str, object]] | None:
    best: tuple[tuple[int, int, int], list[tuple[int, int]], dict[str, object]] | None = None
    for axis in ("z", "x"):
        axis_indices = [
            term_idx
            for term_idx in searchable_indices
            if _is_axis_vector(vectors[term_idx], num_qubits, axis)
        ]
        if len(axis_indices) == 0:
            continue

        unique_vectors = {vectors[term_idx] for term_idx in axis_indices}
        if len(unique_vectors) != len(axis_indices):
            continue

        rank = _gf2_rank(unique_vectors)
        if rank == 0 or rank > width or len(axis_indices) <= rank:
            continue

        generator_terms = _low_weight_independent_terms(
            axis_indices,
            vectors,
            labels,
            num_qubits,
        )
        if len(generator_terms) != rank:
            continue

        basis_vectors = [vectors[term_idx] for term_idx in generator_terms]
        span = _span_coordinate_map(basis_vectors)
        placement = []
        used_addresses = set()
        for term_idx in axis_indices:
            address = span.get(vectors[term_idx])
            if address is None or address == 0 or address in used_addresses:
                placement = []
                break
            used_addresses.add(address)
            placement.append((term_idx, address))
        if len(placement) != len(axis_indices):
            continue

        basis_cost = _placement_cost_from_index_modes(placement, vectors, num_qubits)
        score = (len(axis_indices), -basis_cost, -rank)
        info = {
            "shortcut": f"{axis}-axis-span",
            "shortcut_axis": axis,
            "shortcut_rank": rank,
            "shortcut_terms": len(axis_indices),
            "shortcut_generators": [labels[term_idx] for term_idx in generator_terms],
            "shortcut_fixed_cost": basis_cost,
        }
        if best is None or score > best[0]:
            best = (score, placement, info)

    if best is None:
        return None
    return best[1], best[2]


def _pauli_distance(lhs: int, rhs: int, num_qubits: int) -> int:
    distance = 0
    for q in range(num_qubits):
        lhs_pair = ((lhs >> q) & 1, (lhs >> (num_qubits + q)) & 1)
        rhs_pair = ((rhs >> q) & 1, (rhs >> (num_qubits + q)) & 1)
        if lhs_pair != rhs_pair:
            distance += 1
    return distance


def _extract_pauli_labels(matrixsum: Matrixsum) -> list[str]:
    labels = []
    for atom, _coeff in matrixsum.instances:
        if not isinstance(atom, PauliAtom):
            raise TypeError(
                "search_for_pauli_placement currently supports Matrixsum objects "
                "whose instances are all PauliAtom."
            )
        labels.append(atom.expr)
    return labels


def _generate_padding_labels(
    existing_labels: set[str],
    *,
    num_qubits: int,
    count: int,
    seed: int,
) -> list[str]:
    if count <= 0:
        return []

    rng = random.Random(seed)
    labels: list[str] = []
    seen = set(existing_labels)
    identity = "I" * num_qubits
    attempts = 0
    max_attempts = max(1000, 100 * count)
    while len(labels) < count and attempts < max_attempts:
        attempts += 1
        label = "".join(rng.choice("IXYZ") for _ in range(num_qubits))
        if label == identity or label in seen:
            continue
        seen.add(label)
        labels.append(label)

    if len(labels) < count:
        raise RuntimeError(
            f"Could not generate {count} distinct padding Pauli labels."
        )
    return labels


def _strict_supersets(address: int, width: int) -> list[int]:
    return [
        target
        for target in range(1 << width)
        if target != address and (target & address) == address
    ]


def _future_assignment_proxy(
    remaining: tuple[int, ...],
    future_addresses: list[int],
    prefix: tuple[int, ...],
    vectors: list[int],
    num_qubits: int,
    exact_address_limit: int,
) -> float:
    """Estimate future cost by assigning terms to addresses under current prefix.

    The exact branch computes a minimum injective assignment for small future
    address sets.  The greedy branch is only a ranking proxy; it is not used as
    an admissible lower bound.
    """
    if len(remaining) == 0:
        return 0.0
    if len(future_addresses) < len(remaining):
        return inf

    def pair_cost(term_idx: int, address: int) -> int:
        residual = vectors[term_idx] ^ prefix[address]
        return address.bit_count() * _pauli_weight(residual, num_qubits)

    if len(future_addresses) <= exact_address_limit:
        dp = {0: 0}
        for term_idx in remaining:
            next_dp = {}
            for mask, curr_cost in dp.items():
                for addr_pos, address in enumerate(future_addresses):
                    if (mask >> addr_pos) & 1:
                        continue
                    next_mask = mask | (1 << addr_pos)
                    next_cost = curr_cost + pair_cost(term_idx, address)
                    old_cost = next_dp.get(next_mask)
                    if old_cost is None or next_cost < old_cost:
                        next_dp[next_mask] = next_cost
            dp = next_dp
        return float(min(dp.values()))

    available = set(future_addresses)
    total = 0
    for term_idx in sorted(
        remaining,
        key=lambda idx: min(pair_cost(idx, address) for address in available),
    ):
        best_address = min(available, key=lambda address: pair_cost(term_idx, address))
        total += pair_cost(term_idx, best_address)
        available.remove(best_address)
    return float(total)


def _exact_layer_assignments(
    remaining: tuple[int, ...],
    layer_addresses: list[int],
    q: int,
) -> Iterable[tuple[tuple[int, int], ...]]:
    if q == 0:
        yield ()
        return

    for term_subset in combinations(remaining, q):
        for address_subset in combinations(layer_addresses, q):
            for address_perm in permutations(address_subset):
                yield tuple(zip(term_subset, address_perm))


def _greedy_layer_assignments(
    remaining: tuple[int, ...],
    layer_addresses: list[int],
    q: int,
    prefix: tuple[int, ...],
    vectors: list[int],
    num_qubits: int,
    limit: int,
) -> list[tuple[tuple[int, int], ...]]:
    if q == 0:
        return [()]

    def pair_cost(term_idx: int, address: int) -> int:
        residual = vectors[term_idx] ^ prefix[address]
        return address.bit_count() * _pauli_weight(residual, num_qubits)

    all_pairs = sorted(
        (pair_cost(term_idx, address), term_idx, address)
        for term_idx in remaining
        for address in layer_addresses
    )

    candidates: dict[tuple[tuple[int, int], ...], None] = {}

    def add_greedy(seed_pairs: list[tuple[int, int]]) -> None:
        used_terms = {term_idx for term_idx, _ in seed_pairs}
        used_addresses = {address for _, address in seed_pairs}
        assignment = list(seed_pairs)

        for _cost, term_idx, address in all_pairs:
            if len(assignment) >= q:
                break
            if term_idx in used_terms or address in used_addresses:
                continue
            assignment.append((term_idx, address))
            used_terms.add(term_idx)
            used_addresses.add(address)

        if len(assignment) == q:
            key = tuple(sorted(assignment))
            candidates.setdefault(key, None)

    add_greedy([])
    for _cost, term_idx, address in all_pairs[: max(limit, q)]:
        add_greedy([(term_idx, address)])
        if len(candidates) >= limit:
            break

    return list(candidates.keys())[:limit]


def _layer_assignment_count(
    remaining_count: int,
    address_count: int,
    q: int,
) -> int:
    if q == 0:
        return 1
    return comb(remaining_count, q) * comb(address_count, q) * factorial(q)


def _apply_layer_assignment(
    state: _SearchState,
    assignment: tuple[tuple[int, int], ...],
    vectors: list[int],
    num_qubits: int,
    supersets: list[list[int]],
) -> _SearchState:
    prefix = list(state.prefix)
    cost = state.cost
    placed_terms = {term_idx for term_idx, _address in assignment}
    new_placement = list(state.placement)
    g_updates: list[tuple[int, int]] = []

    for term_idx, address in assignment:
        g_bits = vectors[term_idx] ^ state.prefix[address]
        cost += address.bit_count() * _pauli_weight(g_bits, num_qubits)
        new_placement.append((term_idx, address))
        if g_bits != 0:
            g_updates.append((address, g_bits))

    for address, g_bits in g_updates:
        for target in supersets[address]:
            prefix[target] ^= g_bits

    new_remaining = tuple(idx for idx in state.remaining if idx not in placed_terms)
    return _SearchState(
        remaining=new_remaining,
        prefix=tuple(prefix),
        cost=cost,
        placement=tuple(new_placement),
    )


def _static_state_score(
    state: _SearchState,
    future_addresses: list[int],
    vectors: list[int],
    num_qubits: int,
    future_weight: float,
    future_exact_address_limit: int,
) -> float:
    proxy = _future_assignment_proxy(
        state.remaining,
        future_addresses,
        state.prefix,
        vectors,
        num_qubits,
        future_exact_address_limit,
    )
    return float(state.cost) + future_weight * proxy


def _mixed_initial_layer_states(
    initial_state: _SearchState,
    first_layer_addresses: list[int],
    future_addresses: list[int],
    vectors: list[int],
    labels: list[str],
    num_qubits: int,
    supersets: list[list[int]],
    *,
    initial_beam_width: int,
    low_weight_count: int,
    random_count: int,
    seed: int,
    future_weight: float,
    future_exact_address_limit: int,
) -> tuple[list[_SearchState], int]:
    """Build a bounded mixed candidate pool for the first basis layer."""
    if len(first_layer_addresses) == 0:
        return [initial_state], 1

    q_min = max(0, len(initial_state.remaining) - len(future_addresses))
    q_max = min(len(initial_state.remaining), len(first_layer_addresses))
    if q_min > q_max:
        return [], 0

    def score_assignment(assignment: tuple[tuple[int, int], ...]) -> tuple[float, _SearchState]:
        new_state = _apply_layer_assignment(
            initial_state, assignment, vectors, num_qubits, supersets
        )
        score = _static_state_score(
            new_state,
            future_addresses,
            vectors,
            num_qubits,
            future_weight,
            future_exact_address_limit,
        )
        return score, new_state

    def vector_weight(term_idx: int) -> int:
        return _pauli_weight(vectors[term_idx], num_qubits)

    def random_assignment_from_pool(
        rng: random.Random,
        term_pool: list[int],
        q: int,
    ) -> tuple[tuple[int, int], ...]:
        if q == 0:
            return ()
        term_subset = tuple(sorted(rng.sample(term_pool, q)))
        address_subset = rng.sample(first_layer_addresses, q)
        return tuple(sorted(zip(term_subset, address_subset)))

    ranked_low_weight: list[tuple[float, _SearchState]] = []
    seen_low_weight: set[tuple[tuple[int, int], ...]] = set()

    def add_low_weight_assignment(assignment: tuple[tuple[int, int], ...]) -> None:
        key = tuple(sorted(assignment))
        if key in seen_low_weight:
            return
        seen_low_weight.add(key)
        ranked_low_weight.append(score_assignment(key))

    for q in range(q_min, q_max + 1):
        assignments = _greedy_layer_assignments(
            initial_state.remaining,
            first_layer_addresses,
            q,
            initial_state.prefix,
            vectors,
            num_qubits,
            low_weight_count,
        )
        for assignment in assignments:
            add_low_weight_assignment(assignment)

    low_rng = random.Random(seed + 7919)
    sorted_terms = sorted(
        initial_state.remaining,
        key=lambda idx: (vector_weight(idx), labels[idx], idx),
    )
    attempts = 0
    max_attempts = max(low_weight_count * 40, low_weight_count)
    while len(ranked_low_weight) < low_weight_count and attempts < max_attempts:
        attempts += 1
        q = low_rng.randint(q_min, q_max)
        if q == 0:
            add_low_weight_assignment(())
            continue
        pool_size = min(
            len(sorted_terms),
            max(q, min(len(sorted_terms), q + 2 + attempts // max(low_weight_count // 8, 1))),
        )
        term_pool = sorted_terms[:pool_size]
        if len(term_pool) >= q:
            add_low_weight_assignment(random_assignment_from_pool(low_rng, term_pool, q))

    ranked_low_weight.sort(key=lambda item: (item[0], item[1].cost))
    ranked = ranked_low_weight[:low_weight_count]
    seen = {tuple(sorted(state.placement)) for _score, state in ranked}

    rng = random.Random(seed)
    attempts = 0
    max_attempts = max(random_count * 30, random_count)
    while attempts < max_attempts and len(ranked) < low_weight_count + random_count:
        attempts += 1
        q = rng.randint(q_min, q_max)
        if q == 0:
            assignment = ()
        else:
            term_subset = tuple(sorted(rng.sample(initial_state.remaining, q)))
            address_subset = rng.sample(first_layer_addresses, q)
            assignment = tuple(sorted(zip(term_subset, address_subset)))
        key = tuple(sorted(assignment))
        if key in seen:
            continue
        score, state = score_assignment(key)
        seen.add(tuple(sorted(state.placement)))
        ranked.append((score, state))

    ranked.sort(key=lambda item: (item[0], item[1].cost))
    return [state for _score, state in ranked[:initial_beam_width]], len(ranked)


def _subgroup_initial_layer_states(
    initial_state: _SearchState,
    first_layer_addresses: list[int],
    future_addresses: list[int],
    vectors: list[int],
    labels: list[str],
    num_qubits: int,
    supersets: list[list[int]],
    *,
    initial_beam_width: int,
    candidate_count: int,
    random_count: int,
    seed: int,
    max_distance: int,
    future_weight: float,
    future_exact_address_limit: int,
) -> tuple[list[_SearchState], int]:
    """Build first-layer states from subgroup-like generator candidates.

    The score favors basis generators whose weight-1 and weight-2 products
    exactly or approximately cover many input Paulis.  This is deliberately a
    first-layer heuristic; upper layers still use the regular beam search.
    """
    if len(first_layer_addresses) == 0:
        return [initial_state], 1

    q_min = max(0, len(initial_state.remaining) - len(future_addresses))
    q_max = min(len(initial_state.remaining), len(first_layer_addresses))
    if q_min > q_max:
        return [], 0

    vector_to_indices: dict[int, list[int]] = {}
    for term_idx in initial_state.remaining:
        vector_to_indices.setdefault(vectors[term_idx], []).append(term_idx)
    input_vectors = [vectors[term_idx] for term_idx in initial_state.remaining]
    input_vector_set = set(input_vectors)

    def score_assignment(assignment: tuple[tuple[int, int], ...]) -> tuple[float, _SearchState]:
        new_state = _apply_layer_assignment(
            initial_state, assignment, vectors, num_qubits, supersets
        )
        score = _static_state_score(
            new_state,
            future_addresses,
            vectors,
            num_qubits,
            future_weight,
            future_exact_address_limit,
        )
        return score, new_state

    def subgroup_proxy(term_subset: tuple[int, ...]) -> tuple[int, int, int, int]:
        generated = {0}
        subset_vectors = [vectors[term_idx] for term_idx in term_subset]
        generated.update(subset_vectors)
        for left_idx in range(len(subset_vectors)):
            for right_idx in range(left_idx + 1, len(subset_vectors)):
                generated.add(subset_vectors[left_idx] ^ subset_vectors[right_idx])

        exact_cover = len(generated & input_vector_set)
        approx_cover = 0
        for target_vec in input_vectors:
            if target_vec in generated:
                continue
            if min(
                _pauli_distance(target_vec, generated_vec, num_qubits)
                for generated_vec in generated
            ) <= max_distance:
                approx_cover += 1
        generator_weight = sum(_pauli_weight(vectors[term_idx], num_qubits) for term_idx in term_subset)
        pair_cover = 0
        for left_idx in range(len(subset_vectors)):
            for right_idx in range(left_idx + 1, len(subset_vectors)):
                if subset_vectors[left_idx] ^ subset_vectors[right_idx] in input_vector_set:
                    pair_cover += 1
        total_near_cover = exact_cover + approx_cover
        return (-total_near_cover, -exact_cover, -pair_cover, generator_weight)

    def add_assignment_from_subset(
        term_subset: tuple[int, ...],
        ranked: list[tuple[tuple[int, int, int, int], float, _SearchState]],
        seen: set[tuple[tuple[int, int], ...]],
    ) -> None:
        if len(term_subset) == 0:
            assignments = [()]
        else:
            assignments = []
            for address_perm in permutations(first_layer_addresses, len(term_subset)):
                assignments.append(tuple(sorted(zip(term_subset, address_perm))))

        subgroup_score = subgroup_proxy(term_subset)
        for assignment in assignments:
            key = tuple(sorted(assignment))
            if key in seen:
                continue
            seen.add(key)
            static_score, state = score_assignment(key)
            ranked.append((subgroup_score, static_score, state))

    ranked: list[tuple[tuple[int, int, int, int], float, _SearchState]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    exact_limit = 50000
    for q in range(q_min, q_max + 1):
        count = _layer_assignment_count(len(initial_state.remaining), len(first_layer_addresses), q)
        if count <= exact_limit:
            for term_subset in combinations(initial_state.remaining, q):
                add_assignment_from_subset(tuple(term_subset), ranked, seen)

    rng = random.Random(seed + 104729)
    sorted_terms = sorted(
        initial_state.remaining,
        key=lambda idx: (_pauli_weight(vectors[idx], num_qubits), labels[idx], idx),
    )
    attempts = 0
    max_attempts = max(random_count * 40, random_count)
    while len(ranked) < candidate_count + random_count and attempts < max_attempts:
        attempts += 1
        q = rng.randint(q_min, q_max)
        if q == 0:
            term_subset = ()
        else:
            if rng.random() < 0.6:
                pool_size = min(len(sorted_terms), max(q, q + attempts // max(random_count // 8, 1) + 4))
                pool = sorted_terms[:pool_size]
            else:
                pool = list(initial_state.remaining)
            term_subset = tuple(sorted(rng.sample(pool, q)))
        add_assignment_from_subset(term_subset, ranked, seen)

    ranked.sort(key=lambda item: (item[0], item[1], item[2].cost))
    states = [state for _subgroup_score, _static_score, state in ranked[:initial_beam_width]]
    return states, len(ranked)


def _placement_cost_from_modes(
    placement: list[tuple[str, str]],
    num_qubits: int,
) -> int:
    p_table = {int(ctrl, 2): _label_to_bits(label) for label, ctrl in placement}
    g_table: dict[int, int] = {}
    total = 0
    for address in sorted(p_table):
        g_bits = p_table[address]
        for lower, lower_g in g_table.items():
            if (lower & address) == lower:
                g_bits ^= lower_g
        g_table[address] = g_bits
        total += address.bit_count() * _pauli_weight(g_bits, num_qubits)
    return total


def _best_improvement_swap_refine(
    placement: dict[int, str],
    num_qubits: int,
    allowed_pairs: list[tuple[int, int]],
) -> tuple[dict[int, str], int]:
    current = dict(placement)
    current_cost = _placement_cost_from_modes(
        [(label, format(address, "b")) for address, label in current.items()],
        num_qubits,
    )
    while True:
        best_pair = None
        best_cost = current_cost
        for left, right in allowed_pairs:
            current[left], current[right] = current[right], current[left]
            candidate_cost = _placement_cost_from_modes(
                [(label, format(address, "b")) for address, label in current.items()],
                num_qubits,
            )
            current[left], current[right] = current[right], current[left]
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_pair = (left, right)
        if best_pair is None:
            return current, current_cost
        left, right = best_pair
        current[left], current[right] = current[right], current[left]
        current_cost = best_cost


def _basis_nonbasis_swap_refine(
    placement: list[tuple[str, str]],
    *,
    width: int,
    num_qubits: int,
) -> tuple[list[tuple[str, str]], int]:
    """Local refinement: swap basis/nonbasis entries and reoptimize nonbasis swaps."""
    placement_by_address = {int(ctrl, 2): label for label, ctrl in placement}
    basis_addresses = [
        address for address in sorted(placement_by_address) if address.bit_count() == 1
    ]
    nonbasis_addresses = [
        address for address in sorted(placement_by_address) if address.bit_count() > 1
    ]
    nonbasis_pairs = list(combinations(nonbasis_addresses, 2))

    current, current_cost = _best_improvement_swap_refine(
        placement_by_address, num_qubits, nonbasis_pairs
    )
    while True:
        best_candidate = current
        best_cost = current_cost
        for basis_address in basis_addresses:
            for nonbasis_address in nonbasis_addresses:
                candidate = dict(current)
                candidate[basis_address], candidate[nonbasis_address] = (
                    candidate[nonbasis_address],
                    candidate[basis_address],
                )
                candidate, candidate_cost = _best_improvement_swap_refine(
                    candidate, num_qubits, nonbasis_pairs
                )
                if candidate_cost < best_cost:
                    best_candidate = candidate
                    best_cost = candidate_cost
        if best_cost >= current_cost:
            refined = [
                (label, format(address, f"0{width}b"))
                for address, label in sorted(current.items())
            ]
            return refined, current_cost
        current = best_candidate
        current_cost = best_cost


def _rank_layer_successors_static(
    state: _SearchState,
    layer_addresses: list[int],
    future_addresses: list[int],
    vectors: list[int],
    num_qubits: int,
    supersets: list[list[int]],
    layer_candidate_limit: int,
    max_exact_layer_assignments: int,
    future_weight: float,
    future_exact_address_limit: int,
) -> list[tuple[float, _SearchState]]:
    remaining_count = len(state.remaining)
    future_capacity = len(future_addresses)
    q_min = max(0, remaining_count - future_capacity)
    q_max = min(remaining_count, len(layer_addresses))
    if q_min > q_max:
        return []

    ranked: list[tuple[float, _SearchState]] = []
    for q in range(q_min, q_max + 1):
        exact_count = _layer_assignment_count(remaining_count, len(layer_addresses), q)
        if exact_count <= max_exact_layer_assignments:
            assignments = _exact_layer_assignments(state.remaining, layer_addresses, q)
        else:
            assignments = _greedy_layer_assignments(
                state.remaining,
                layer_addresses,
                q,
                state.prefix,
                vectors,
                num_qubits,
                layer_candidate_limit,
            )

        for assignment in assignments:
            new_state = _apply_layer_assignment(
                state, assignment, vectors, num_qubits, supersets
            )
            score = _static_state_score(
                new_state,
                future_addresses,
                vectors,
                num_qubits,
                future_weight,
                future_exact_address_limit,
            )
            ranked.append((score, new_state))

    ranked.sort(key=lambda item: (item[0], item[1].cost))
    return ranked[:layer_candidate_limit]


def _lookahead_state_score(
    state: _SearchState,
    next_layer_addresses: list[int],
    after_next_future_addresses: list[int],
    vectors: list[int],
    num_qubits: int,
    supersets: list[list[int]],
    lookahead_width: int,
    max_exact_layer_assignments: int,
    future_weight: float,
    future_exact_address_limit: int,
) -> float:
    if len(state.remaining) == 0:
        return float(state.cost)
    if len(next_layer_addresses) == 0:
        return _static_state_score(
            state,
            after_next_future_addresses,
            vectors,
            num_qubits,
            future_weight,
            future_exact_address_limit,
        )

    successors = _rank_layer_successors_static(
        state,
        next_layer_addresses,
        after_next_future_addresses,
        vectors,
        num_qubits,
        supersets,
        lookahead_width,
        max_exact_layer_assignments,
        future_weight,
        future_exact_address_limit,
    )
    if len(successors) == 0:
        return inf
    return min(score for score, _state in successors[:lookahead_width])


def search_for_pauli_placement(
    matrixsum: Matrixsum,
    m: int | None = None,
    *,
    beam_width: int = 64,
    layer_candidate_limit: int = 256,
    max_exact_layer_assignments: int = 20000,
    future_weight: float = 1.0,
    future_exact_address_limit: int = 8,
    lookahead_depth: int = 0,
    lookahead_width: int = 16,
    lookahead_start_weight: int = 1,
    initial_strategy: str = "standard",
    mixed_initial_beam_width: int | None = None,
    mixed_full_initial_beam_width: int = 512,
    mixed_nonfull_initial_beam_width: int = 128,
    mixed_low_weight_count: int = 256,
    mixed_random_count: int = 1024,
    mixed_random_seed: int = 20260608,
    subgroup_initial_beam_width: int | None = None,
    subgroup_candidate_count: int = 256,
    subgroup_random_count: int = 1024,
    subgroup_random_seed: int = 20260610,
    subgroup_max_distance: int = 1,
    swap_refinement: bool = False,
    swap_refinement_max_terms: int | None = 64,
    padding_to_full_address: bool = False,
    padding_random_seed: int = 20260609,
    allow_zero_address: bool = False,
    return_debug: bool = False,
):
    """Search for a Pauli placement table.

    Args:
        matrixsum: Pauli ``Matrixsum`` to place.
        m: Number of control/ancilla bits.  Defaults to
            ``ceil(log2(matrixsum.length))``.
        beam_width: Number of states kept after each Hamming-weight layer.
        layer_candidate_limit: Maximum candidate layer assignments generated
            per state when exact enumeration would be too large.
        max_exact_layer_assignments: Exact enumeration threshold per layer and
            per state.
        future_weight: Weight of the future assignment proxy in beam ranking.
        future_exact_address_limit: Address-count cutoff for exact future
            assignment proxy.
        lookahead_depth: Optional lookahead depth for beam scoring.  Only
            ``0`` and ``1`` are currently supported.  With depth 1, each
            current-layer candidate is scored by simulating one real next layer
            before falling back to the static future proxy.
        lookahead_width: Maximum number of next-layer candidates used when
            ``lookahead_depth=1``.
        lookahead_start_weight: Lowest Hamming-weight layer at which lookahead
            scoring is enabled.  Set this to ``2`` to keep the first basis
            layer on the static proxy and only look ahead from the second layer.
        initial_strategy: ``"standard"`` keeps the regular layer-by-layer beam
            search. ``"mixed"`` uses a larger bounded mixed pool for the first
            Hamming-weight-one basis layer, combining low-weight/proxy and
            random candidates. ``"subgroup"`` ranks first-layer generators by
            exact and approximate coverage under weight-1/2 generated products.
        mixed_initial_beam_width: First-layer beam width for ``"mixed"``.  If
            omitted, full-address instances use ``mixed_full_initial_beam_width``
            and non-full-address instances use
            ``mixed_nonfull_initial_beam_width``.
        mixed_full_initial_beam_width: Default first-layer beam for full-address
            instances.
        mixed_nonfull_initial_beam_width: Default first-layer beam for non-full
            address instances.
        mixed_low_weight_count: Number of low-weight/proxy first-layer
            candidates kept in the mixed candidate pool.
        mixed_random_count: Number of random first-layer candidates added to the
            mixed candidate pool.
        mixed_random_seed: Seed for random first-layer candidates.
        subgroup_initial_beam_width: First-layer beam width for
            ``"subgroup"``.  Defaults to the same full/non-full values as the
            mixed strategy.
        subgroup_candidate_count: Number of structure-ranked subgroup
            candidates to build before truncation.
        subgroup_random_count: Number of random generator subsets used to find
            additional subgroup candidates.
        subgroup_random_seed: Seed for random subgroup generator subsets.
        subgroup_max_distance: Approximate coverage threshold in Pauli-distance
            units.  Only weight-1 and weight-2 products of the proposed basis
            generators are used in this coverage proxy.
        swap_refinement: If true, run a local basis/nonbasis swap refinement on
            the final placement.
        swap_refinement_max_terms: Skip local swap refinement when the returned
            real placement is larger than this value. ``None`` means no limit.
        padding_to_full_address: If true, append random padding Paulis during
            search so the nonzero control-address table is full.  The regular
            returned placement contains only the original terms, while debug
            info stores ``padded_placement`` and ``padded_search_cost`` for the
            full select/g implementation.  Downstream block encoding should
            keep padding entries in the select circuit and set their prepared
            coefficients to zero.
        padding_random_seed: Seed for random padding Pauli generation.
        allow_zero_address: If true, allow non-identity terms at all-zero
            control address.  The default mirrors the current matrix-order
            convention and reserves zero only for an identity term.
        return_debug: If true, return ``(placement, debug_info)``.

    Returns:
        A list of ``(pauli_label, ctrl_value)`` tuples sorted by control value,
        matching the shape returned by ``assign_additional_modes``.
    """
    labels = _extract_pauli_labels(matrixsum)
    if len(labels) == 0:
        return ([], {"best_cost": 0, "layers": []}) if return_debug else []

    if m is None:
        m = _default_control_size(len(labels))
    if m < 0:
        raise ValueError("m must be non-negative.")
    if lookahead_depth not in (0, 1):
        raise ValueError("Only lookahead_depth values 0 and 1 are supported.")
    if lookahead_width < 1:
        raise ValueError("lookahead_width must be positive.")
    if lookahead_start_weight < 0:
        raise ValueError("lookahead_start_weight must be non-negative.")
    if initial_strategy not in {"standard", "mixed", "subgroup"}:
        raise ValueError("initial_strategy must be 'standard', 'mixed', or 'subgroup'.")
    if mixed_full_initial_beam_width < 1 or mixed_nonfull_initial_beam_width < 1:
        raise ValueError("mixed initial beam widths must be positive.")
    if mixed_low_weight_count < 0 or mixed_random_count < 0:
        raise ValueError("mixed candidate counts must be non-negative.")
    if subgroup_candidate_count < 0 or subgroup_random_count < 0:
        raise ValueError("subgroup candidate counts must be non-negative.")
    if subgroup_max_distance < 0:
        raise ValueError("subgroup_max_distance must be non-negative.")
    if swap_refinement_max_terms is not None and swap_refinement_max_terms < 0:
        raise ValueError("swap_refinement_max_terms must be non-negative or None.")
    if padding_to_full_address and allow_zero_address:
        raise ValueError("padding_to_full_address currently requires allow_zero_address=False.")

    num_qubits = matrixsum.size
    width = int(m)
    table_size = 1 << width
    zero_label = "I" * num_qubits
    original_label_count = len(labels)
    real_output_indices = set(range(original_label_count))

    fixed_placement: list[tuple[int, int]] = []
    searchable_indices = list(range(len(labels)))
    if not allow_zero_address:
        identity_idx = next((idx for idx, label in enumerate(labels) if label == zero_label), None)
        if identity_idx is not None:
            fixed_placement.append((identity_idx, 0))
            searchable_indices.remove(identity_idx)

    available_addresses = list(range(table_size)) if allow_zero_address else list(range(1, table_size))
    original_searchable_count = len(searchable_indices)
    padding_labels: list[str] = []
    if padding_to_full_address and len(searchable_indices) < len(available_addresses):
        padding_count = len(available_addresses) - len(searchable_indices)
        padding_labels = _generate_padding_labels(
            set(labels),
            num_qubits=num_qubits,
            count=padding_count,
            seed=padding_random_seed,
        )
        labels.extend(padding_labels)
        padding_indices = range(original_label_count, original_label_count + padding_count)
        searchable_indices.extend(padding_indices)
    else:
        padding_count = 0

    if not allow_zero_address and len(searchable_indices) > len(available_addresses):
        raise ValueError(
            f"m={m} gives only {len(available_addresses)} nonzero control addresses, "
            f"but {len(searchable_indices)} non-identity/searchable terms need placement."
        )
    if allow_zero_address and len(searchable_indices) > len(available_addresses):
        raise ValueError(
            f"m={m} gives only {len(available_addresses)} control addresses, "
            f"but {len(searchable_indices)} terms need placement."
        )

    vectors = [_label_to_bits(label) for label in labels]
    supersets = [_strict_supersets(address, width) for address in range(table_size)]
    address_layers = [
        [address for address in available_addresses if address.bit_count() == h]
        for h in range(width + 1)
    ]

    shortcut_info: dict[str, object] | None = None
    if not allow_zero_address:
        shortcut_source_indices = [
            idx for idx in searchable_indices if idx < original_label_count
        ]
        subgroup_shortcut = _try_subgroup_span_shortcut(
            shortcut_source_indices,
            vectors,
            labels,
            width=width,
            num_qubits=num_qubits,
        )
        if subgroup_shortcut is not None:
            shortcut_placement, shortcut_padding_placement, shortcut_info = subgroup_shortcut
            padded_placement = _padded_placement_from_index_and_label_modes(
                fixed_placement + shortcut_placement,
                shortcut_padding_placement,
                labels,
                width,
            )
            placement = _placement_from_index_modes(
                fixed_placement + shortcut_placement,
                labels,
                width,
                real_output_indices,
            )
            shortcut_padding_labels = [
                label for label, _address in shortcut_padding_placement
            ]
            cost = int(shortcut_info["shortcut_cost"])
            effective_padding_labels = shortcut_padding_labels
            effective_padding_count = len(shortcut_padding_labels)
            if return_debug:
                return placement, {
                    "best_cost": cost,
                    "pre_refinement_cost": cost,
                    "padded_search_cost": cost,
                    "swap_refinement": False,
                    "refinement_applied": False,
                    "refinement_skipped": False,
                    "swap_refinement_max_terms": swap_refinement_max_terms,
                    "m": width,
                    "beam_width": beam_width,
                    "layers": [],
                    "initial_strategy": initial_strategy,
                    "is_full_address": len(searchable_indices) == len(available_addresses),
                    "original_is_full_address": original_searchable_count == len(available_addresses),
                    "padding_to_full_address": padding_to_full_address,
                    "padding_count": effective_padding_count,
                    "padding_labels": effective_padding_labels,
                    "padded_placement": padded_placement,
                    "initial_basis_placements": [],
                    "allow_zero_address": allow_zero_address,
                    "lookahead_depth": lookahead_depth,
                    "lookahead_width": lookahead_width,
                    "lookahead_start_weight": lookahead_start_weight,
                    **shortcut_info,
                }
            return placement

    initial_prefix = [0] * table_size
    initial_placement = list(fixed_placement)
    shortcut_fixed: list[tuple[int, int]] = []
    if not allow_zero_address:
        shortcut_source_indices = [
            idx for idx in searchable_indices if idx < original_label_count
        ]
        axis_shortcut = _try_axis_span_shortcut(
            shortcut_source_indices,
            vectors,
            labels,
            width=width,
            num_qubits=num_qubits,
        )
        if axis_shortcut is not None:
            shortcut_fixed, shortcut_info = axis_shortcut
            used_addresses = {address for _term_idx, address in shortcut_fixed}
            if all(address in available_addresses for address in used_addresses):
                state_for_shortcut = _SearchState(
                    remaining=tuple(searchable_indices),
                    prefix=tuple(initial_prefix),
                    cost=0,
                    placement=tuple(initial_placement),
                )
                shortcut_state = _apply_layer_assignment(
                    state_for_shortcut,
                    tuple(shortcut_fixed),
                    vectors,
                    num_qubits,
                    supersets,
                )
                initial_prefix = list(shortcut_state.prefix)
                initial_placement = list(shortcut_state.placement)
                searchable_indices = list(shortcut_state.remaining)
                available_addresses = [
                    address for address in available_addresses if address not in used_addresses
                ]
                address_layers = [
                    [address for address in available_addresses if address.bit_count() == h]
                    for h in range(width + 1)
                ]
            else:
                shortcut_fixed = []
                shortcut_info = None

    initial_remaining = tuple(searchable_indices)

    states = [
        _SearchState(
            remaining=initial_remaining,
            prefix=tuple(initial_prefix),
            cost=_placement_cost_from_index_modes(initial_placement, vectors, num_qubits),
            placement=tuple(initial_placement),
        )
    ]
    debug_layers = []
    initial_basis_placements: list[tuple[tuple[str, str], ...]] = []
    start_h = 0

    is_full_address = (
        not allow_zero_address
        and len(searchable_indices) == len(available_addresses)
    )
    original_is_full_address = (
        not allow_zero_address
        and original_searchable_count == len(available_addresses)
    )
    if (
        initial_strategy in {"mixed", "subgroup"}
        and not allow_zero_address
        and width > 0
    ):
        first_layer_addresses = address_layers[1]
        future_addresses = [
            address
            for future_h in range(2, width + 1)
            for address in address_layers[future_h]
        ]
        if initial_strategy == "mixed":
            if mixed_initial_beam_width is None:
                mixed_initial_beam_width = (
                    mixed_full_initial_beam_width
                    if is_full_address
                    else mixed_nonfull_initial_beam_width
                )
            states, initial_candidate_count = _mixed_initial_layer_states(
                states[0],
                first_layer_addresses,
                future_addresses,
                vectors,
                labels,
                num_qubits,
                supersets,
                initial_beam_width=mixed_initial_beam_width,
                low_weight_count=mixed_low_weight_count,
                random_count=mixed_random_count,
                seed=mixed_random_seed,
                future_weight=future_weight,
                future_exact_address_limit=future_exact_address_limit,
            )
            initial_beam_width_used = mixed_initial_beam_width
        else:
            if subgroup_initial_beam_width is None:
                subgroup_initial_beam_width = (
                    mixed_full_initial_beam_width
                    if is_full_address
                    else mixed_nonfull_initial_beam_width
                )
            states, initial_candidate_count = _subgroup_initial_layer_states(
                states[0],
                first_layer_addresses,
                future_addresses,
                vectors,
                labels,
                num_qubits,
                supersets,
                initial_beam_width=subgroup_initial_beam_width,
                candidate_count=subgroup_candidate_count,
                random_count=subgroup_random_count,
                seed=subgroup_random_seed,
                max_distance=subgroup_max_distance,
                future_weight=future_weight,
                future_exact_address_limit=future_exact_address_limit,
            )
            initial_beam_width_used = subgroup_initial_beam_width
        initial_basis_placements = [
            tuple(
                (labels[term_idx], format(address, f"0{width}b"))
                for term_idx, address in sorted(state.placement, key=lambda item: item[1])
                if address.bit_count() == 1
            )
            for state in states
        ]
        debug_layers.append(
            {
                "hamming_weight": 1,
                "layer_addresses": len(first_layer_addresses),
                "states_kept": len(states),
                "best_cost_so_far": min((state.cost for state in states), default=inf),
                "initial_strategy": initial_strategy,
                "mixed_initial_beam_width": initial_beam_width_used,
                "mixed_initial_candidates": initial_candidate_count,
                "subgroup_max_distance": subgroup_max_distance if initial_strategy == "subgroup" else None,
                "is_full_address": is_full_address,
                "fixed_layer_occupancy": is_full_address,
            }
        )
        start_h = 2

    for h, layer_addresses in enumerate(address_layers):
        if h < start_h:
            continue
        if h == 0 and not allow_zero_address:
            continue

        future_addresses = [
            address
            for future_h in range(h + 1, width + 1)
            for address in address_layers[future_h]
        ]
        next_states: list[tuple[float, _SearchState]] = []

        for state in states:
            remaining_count = len(state.remaining)
            future_capacity = len(future_addresses)
            if is_full_address:
                q_min = q_max = min(remaining_count, len(layer_addresses))
            else:
                q_min = max(0, remaining_count - future_capacity)
                q_max = min(remaining_count, len(layer_addresses))
            if q_min > q_max:
                continue

            for q in range(q_min, q_max + 1):
                exact_count = _layer_assignment_count(remaining_count, len(layer_addresses), q)
                if exact_count <= max_exact_layer_assignments:
                    assignments = _exact_layer_assignments(state.remaining, layer_addresses, q)
                else:
                    assignments = _greedy_layer_assignments(
                        state.remaining,
                        layer_addresses,
                        q,
                        state.prefix,
                        vectors,
                        num_qubits,
                        layer_candidate_limit,
                    )

                ranked_for_state = []
                for assignment in assignments:
                    new_state = _apply_layer_assignment(
                        state, assignment, vectors, num_qubits, supersets
                    )
                    if (
                        lookahead_depth == 1
                        and h >= lookahead_start_weight
                        and h + 1 < len(address_layers)
                    ):
                        next_layer_addresses = address_layers[h + 1]
                        after_next_future_addresses = [
                            address
                            for future_h in range(h + 2, width + 1)
                            for address in address_layers[future_h]
                        ]
                        score = _lookahead_state_score(
                            new_state,
                            next_layer_addresses,
                            after_next_future_addresses,
                            vectors,
                            num_qubits,
                            supersets,
                            lookahead_width,
                            max_exact_layer_assignments,
                            future_weight,
                            future_exact_address_limit,
                        )
                    else:
                        score = _static_state_score(
                            new_state,
                            future_addresses,
                            vectors,
                            num_qubits,
                            future_weight,
                            future_exact_address_limit,
                        )
                    ranked_for_state.append((score, new_state))

                ranked_for_state.sort(key=lambda item: (item[0], item[1].cost))
                next_states.extend(ranked_for_state[:layer_candidate_limit])

        best_by_key: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, _SearchState]] = {}
        for score, state in next_states:
            key = (state.remaining, state.prefix)
            old = best_by_key.get(key)
            if old is None or (score, state.cost) < (old[0], old[1].cost):
                best_by_key[key] = (score, state)

        states = [
            state
            for _score, state in sorted(
                best_by_key.values(), key=lambda item: (item[0], item[1].cost)
            )[:beam_width]
        ]
        debug_layers.append(
            {
                "hamming_weight": h,
                "layer_addresses": len(layer_addresses),
                "states_kept": len(states),
                "best_cost_so_far": min((state.cost for state in states), default=inf),
                "fixed_layer_occupancy": is_full_address,
            }
        )

        if len(states) == 0:
            break

    complete_states = [state for state in states if len(state.remaining) == 0]
    if len(complete_states) == 0:
        raise RuntimeError(
            "Layered beam search did not find a complete placement. "
            "Try increasing m, beam_width, or layer_candidate_limit."
        )

    best_state = min(complete_states, key=lambda state: state.cost)
    padded_placement = [
        (labels[term_idx], format(address, f"0{width}b"))
        for term_idx, address in sorted(best_state.placement, key=lambda item: item[1])
    ]
    placement = [
        (labels[term_idx], format(address, f"0{width}b"))
        for term_idx, address in sorted(best_state.placement, key=lambda item: item[1])
        if term_idx in real_output_indices
    ]
    padded_search_cost = best_state.cost
    pre_refinement_cost = _placement_cost_from_modes(placement, num_qubits)
    refinement_applied = False
    refinement_skipped = False
    if (
        swap_refinement
        and (
            swap_refinement_max_terms is None
            or len(placement) <= swap_refinement_max_terms
        )
    ):
        placement, refined_cost = _basis_nonbasis_swap_refine(
            placement,
            width=width,
            num_qubits=num_qubits,
        )
        refinement_applied = refined_cost < pre_refinement_cost
    else:
        refined_cost = pre_refinement_cost
        refinement_skipped = swap_refinement and not (
            swap_refinement_max_terms is None
            or len(placement) <= swap_refinement_max_terms
        )

    if return_debug:
        debug_info = {
            "best_cost": refined_cost,
            "pre_refinement_cost": pre_refinement_cost,
            "padded_search_cost": padded_search_cost,
            "swap_refinement": swap_refinement,
            "refinement_applied": refinement_applied,
            "refinement_skipped": refinement_skipped,
            "swap_refinement_max_terms": swap_refinement_max_terms,
            "m": width,
            "beam_width": beam_width,
            "layers": debug_layers,
            "initial_strategy": initial_strategy,
            "is_full_address": is_full_address,
            "original_is_full_address": original_is_full_address,
            "padding_to_full_address": padding_to_full_address,
            "padding_count": padding_count,
            "padding_labels": padding_labels,
            "padded_placement": padded_placement,
            "initial_basis_placements": initial_basis_placements,
            "allow_zero_address": allow_zero_address,
            "lookahead_depth": lookahead_depth,
            "lookahead_width": lookahead_width,
            "lookahead_start_weight": lookahead_start_weight,
        }
        if shortcut_info is not None:
            debug_info.update(shortcut_info)
        return placement, debug_info
    return placement
