"""Gate-cost-guided rewrite search for ChannelIR Kraus representations.

This module implements an H_length-style search objective.  The current
candidate generator is intentionally conservative: it reuses ChannelIR's
existing two-row unitary-mixing candidates, but ranks and accepts candidates by
an estimated generic StatePreparation/control T-count cost instead of total
Pauli support alone.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np


def ceil_log2(value: int) -> int:
    """Return ceil(log2(value)), with 0 for value <= 1."""
    if value <= 1:
        return 0
    return int(math.ceil(math.log2(value)))


def next_pow2_len(value: int) -> int:
    """Return 2**ceil(log2(value)), with 1 for value <= 1."""
    if value <= 1:
        return 1
    return 1 << ceil_log2(value)


def generic_stateprep_gate_count(length: int) -> int:
    """
    Generic non-trivial Qiskit StatePreparation gate-count proxy.

    For length N=2**n, the empirical NAM opt0 proxy is
        G_sp(N) ~= 10N - n - 10.
    The length argument is rounded by callers to a power of two.
    """
    if length <= 1:
        return 0
    return int(10 * length - ceil_log2(length) - 10)


def multi_control_t_count(num_controls: int) -> int:
    """
    Toffoli-style T-count proxy for a gate with num_controls controls.
    """
    if num_controls <= 1:
        return 0
    if num_controls == 2:
        return 7
    return 7 * (2 * (num_controls - 2) + 1)


def row_supports(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return the number of nonzero Pauli coefficients in each Kraus row."""
    return np.count_nonzero(np.abs(A) > tol, axis=1).astype(int)


def h_length_components_from_supports(supports: np.ndarray | list[int]) -> dict[str, int]:
    """
    Compute the H_length heuristic components from row support sizes.

    The cost model is:
        outer prep
      + outer select
      + inner block select
      + controlled block coefficient prep/inverse.

    Coefficient values are intentionally ignored; only Kraus count and per-row
    Pauli support lengths enter the estimate.
    """
    supports_arr = np.asarray(supports, dtype=int)
    supports_arr = supports_arr[supports_arr > 0]
    kraus_count = int(len(supports_arr))
    if kraus_count == 0:
        return {
            "total": 0,
            "kraus_count": 0,
            "total_support": 0,
            "outer_prep": 0,
            "outer_select": 0,
            "block_select": 0,
            "block_prep": 0,
            "outer_controls": 0,
        }

    outer_controls = ceil_log2(kraus_count)
    outer_length = next_pow2_len(kraus_count)

    outer_prep = generic_stateprep_gate_count(outer_length)
    outer_select = kraus_count * multi_control_t_count(outer_controls)

    block_select = 0
    block_prep = 0
    block_prep_control_factor = multi_control_t_count(outer_controls + 1)

    for support in supports_arr:
        support = int(support)
        inner_controls = ceil_log2(support)
        inner_length = next_pow2_len(support)
        block_select += support * multi_control_t_count(inner_controls)
        block_prep += (
            2
            * generic_stateprep_gate_count(inner_length)
            * block_prep_control_factor
        )

    total = outer_prep + outer_select + block_select + block_prep
    return {
        "total": int(total),
        "kraus_count": kraus_count,
        "total_support": int(np.sum(supports_arr)),
        "outer_prep": int(outer_prep),
        "outer_select": int(outer_select),
        "block_select": int(block_select),
        "block_prep": int(block_prep),
        "outer_controls": int(outer_controls),
    }


def h_length_cost_from_supports(supports: np.ndarray | list[int]) -> int:
    """Return the scalar H_length cost from row support sizes."""
    return h_length_components_from_supports(supports)["total"]


def h_length_cost(A: np.ndarray, tol: float = 1e-12) -> int:
    """Return the scalar H_length cost for a coefficient matrix."""
    return h_length_cost_from_supports(row_supports(A, tol))


def _future_potential(ir_cls: type, A: np.ndarray, pair: tuple[int, int], tol: float):
    if hasattr(ir_cls, "_local_future_potential"):
        return ir_cls._local_future_potential(A, pair, tol)
    return None


def _future_score(potential: Any) -> tuple[int, int, int, int, int]:
    if isinstance(potential, dict) and "score" in potential:
        return tuple(potential["score"])
    return (0, 0, 0, 0, 0)


def _candidate_gatecost_rotations(
    ir_cls: type,
    A: np.ndarray,
    supports: np.ndarray,
    current_cost: int,
    tol: float,
    keep_neutral: bool,
    max_neutral_candidates: int | None,
    skip_disjoint_pairs: bool,
) -> list[dict[str, Any]]:
    """Generate non-increasing H_length two-row rotation candidates."""
    candidates: list[dict[str, Any]] = []
    m = A.shape[0]

    for i, j in combinations(range(m), 2):
        shared = ir_cls._pair_shared_support(A, i, j, tol)
        if skip_disjoint_pairs and shared == 0:
            continue

        before_potential = _future_potential(ir_cls, A, (i, j), tol)
        before_score = _future_score(before_potential)
        seen_unitaries = set()
        pair_candidates: list[dict[str, Any]] = []

        for U in ir_cls._candidate_unitaries_for_pair(A, i, j, tol):
            unitary_key = ir_cls._unitary_key(U, tol)
            if unitary_key in seen_unitaries:
                continue
            seen_unitaries.add(unitary_key)

            new_i = U[0, 0] * A[i] + U[0, 1] * A[j]
            new_j = U[1, 0] * A[i] + U[1, 1] * A[j]
            new_i[np.abs(new_i) < tol] = 0.0
            new_j[np.abs(new_j) < tol] = 0.0

            if np.allclose(new_i, A[i], atol=tol, rtol=tol) and np.allclose(
                new_j,
                A[j],
                atol=tol,
                rtol=tol,
            ):
                continue

            new_support_i = int(np.count_nonzero(np.abs(new_i) > tol))
            new_support_j = int(np.count_nonzero(np.abs(new_j) > tol))
            new_supports = supports.copy()
            old_pair_support = int(supports[i] + supports[j])
            new_supports[i] = new_support_i
            new_supports[j] = new_support_j

            new_cost = h_length_cost_from_supports(new_supports)
            if new_cost > current_cost:
                continue

            reason = "improve" if new_cost < current_cost else "neutral"
            if reason == "neutral" and not keep_neutral:
                continue

            B = A.copy()
            B[i] = new_i
            B[j] = new_j

            after_potential = _future_potential(ir_cls, B, (i, j), tol)
            after_score = _future_score(after_potential)
            if reason == "neutral" and after_score <= before_score:
                continue

            new_total_support = int(np.sum(new_supports))
            pair_delta = int(new_support_i + new_support_j - old_pair_support)
            sort_key = (
                new_cost,
                0 if reason == "improve" else 1,
                new_total_support,
                max(new_support_i, new_support_j),
                new_support_i + new_support_j,
                -after_score[0],
                -after_score[1],
                -after_score[2],
                -after_score[3],
                -after_score[4],
                i,
                j,
            )
            pair_candidates.append(
                {
                    "pair": (i, j),
                    "U": U,
                    "A": B,
                    "cost": int(new_cost),
                    "support": new_total_support,
                    "row_supports": new_supports,
                    "pair_delta": pair_delta,
                    "reason": reason,
                    "shared_support": int(shared),
                    "before_potential": before_potential,
                    "after_potential": after_potential,
                    "sort_key": sort_key,
                }
            )

        improving = [c for c in pair_candidates if c["reason"] == "improve"]
        neutral = [c for c in pair_candidates if c["reason"] == "neutral"]
        improving.sort(key=lambda c: c["sort_key"])
        neutral.sort(key=lambda c: c["sort_key"])
        if max_neutral_candidates is not None:
            neutral = neutral[:max_neutral_candidates]
        candidates.extend(improving)
        candidates.extend(neutral)

    return candidates


def gatecost_rewrite_search(
    channel_ir,
    strategy: str = "beam",
    beam_width: int = 8,
    max_steps: int = 50,
    tol: float = 1e-12,
    verbose: bool = False,
    keep_neutral: bool = False,
    max_neutral_candidates: int | None = 3,
    skip_disjoint_pairs: bool = True,
) -> dict[str, Any]:
    """
    Search for two-row unitary rewrites using H_length as the objective.

    Args:
        channel_ir: A channel_IR.channel instance.
        strategy: "beam" or "greedy".  Greedy is implemented as beam width 1.
        beam_width: Number of frontier states for beam search.
        max_steps: Maximum number of accepted rewrite layers.
        tol: Numerical tolerance for Pauli coefficient support.
        verbose: Print per-step search diagnostics.
        keep_neutral: If True, allow equal-cost plateau moves only when the
            local future-potential score improves.
        max_neutral_candidates: Per-pair cap for neutral candidates.
        skip_disjoint_pairs: Skip pairs with no shared Pauli support.

    Returns:
        A result dictionary compatible with channel.apply_rewrite_result().
        The primary objective fields are initial_cost/final_cost.
    """
    if strategy not in {"beam", "greedy"}:
        raise ValueError(f"Unknown gatecost rewrite strategy: {strategy}")
    if strategy == "greedy":
        beam_width = 1

    A, labels = channel_ir._coeff_matrix(tol)
    ir_cls = channel_ir.__class__

    initial_supports = row_supports(A, tol)
    initial_components = h_length_components_from_supports(initial_supports)
    initial_cost = initial_components["total"]
    initial_support = int(np.sum(initial_supports))

    initial_key = ir_cls._canonical_state_key(A, tol)
    beam = [
        (
            initial_cost,
            (initial_cost, initial_support),
            A.copy(),
            initial_supports.copy(),
            [],
            [],
        )
    ]
    visited = {initial_key}
    global_best = (
        initial_cost,
        initial_support,
        A.copy(),
        initial_supports.copy(),
        [],
        [],
    )

    cost_trajectory = [initial_cost]
    support_trajectory = [initial_support]
    frontier_sizes: list[int] = []
    generated_counts: list[int] = []
    accepted_counts: list[int] = []
    stop_reason = "max_steps_reached"
    iterations = 0

    for step in range(max_steps):
        iterations += 1
        layer = []
        generated = 0
        accepted = 0

        for current_cost, _, state_A, supports, steps, metadata in beam:
            candidates = _candidate_gatecost_rotations(
                ir_cls,
                state_A,
                supports,
                current_cost,
                tol,
                keep_neutral,
                max_neutral_candidates,
                skip_disjoint_pairs,
            )
            generated += len(candidates)

            for cand in candidates:
                new_A = cand["A"]
                state_key = ir_cls._canonical_state_key(new_A, tol)
                if state_key in visited:
                    continue
                visited.add(state_key)
                accepted += 1

                new_cost = cand["cost"]
                new_support = cand["support"]
                new_steps = steps + [(cand["pair"], cand["U"], new_cost)]
                new_metadata = metadata + [
                    {
                        "pair": cand["pair"],
                        "reason": cand["reason"],
                        "cost": new_cost,
                        "support": new_support,
                        "cost_delta": int(new_cost - current_cost),
                        "pair_delta": cand["pair_delta"],
                        "shared_support": cand["shared_support"],
                    }
                ]
                layer.append(
                    (
                        new_cost,
                        cand["sort_key"],
                        new_A,
                        cand["row_supports"],
                        new_steps,
                        new_metadata,
                    )
                )

                best_key = (global_best[0], global_best[1])
                cand_key = (new_cost, new_support)
                if cand_key < best_key:
                    global_best = (
                        new_cost,
                        new_support,
                        new_A.copy(),
                        cand["row_supports"].copy(),
                        list(new_steps),
                        list(new_metadata),
                    )

        generated_counts.append(generated)
        accepted_counts.append(accepted)

        if not layer:
            stop_reason = "frontier_exhausted"
            if verbose:
                print(f"  Step {step}: no unseen non-increasing gate-cost candidates.")
            break

        seen_layer = {}
        for item in layer:
            key = ir_cls._canonical_state_key(item[2], tol)
            if key not in seen_layer or item[1] < seen_layer[key][1]:
                seen_layer[key] = item

        unique = sorted(seen_layer.values(), key=lambda item: item[1])
        beam = unique[:beam_width]
        frontier_sizes.append(len(beam))
        cost_trajectory.append(global_best[0])
        support_trajectory.append(global_best[1])

        if verbose:
            beam_costs = [item[0] for item in beam]
            print(
                f"  Step {step}: generated={generated}, accepted={accepted}, "
                f"beam costs={beam_costs}, global best={global_best[0]}"
            )

    final_cost, final_support, final_A, final_supports, final_steps, final_meta = global_best
    return {
        "cost_model": "H_length_generic_stateprep",
        "strategy": strategy,
        "initial_cost": int(initial_cost),
        "final_cost": int(final_cost),
        "initial_cost_components": initial_components,
        "final_cost_components": h_length_components_from_supports(final_supports),
        "initial_support": int(initial_support),
        "final_support": int(final_support),
        "initial_row_supports": initial_supports.tolist(),
        "final_row_supports": final_supports.tolist(),
        "steps": final_steps,
        "step_metadata": final_meta,
        "A_final": final_A,
        "labels": labels,
        "termination": {
            "stop_reason": stop_reason,
            "iterations": iterations,
            "max_steps": max_steps,
            "cost_trajectory": cost_trajectory,
            "support_trajectory": support_trajectory,
            "beam_width": beam_width,
            "keep_neutral": keep_neutral,
            "max_neutral_candidates": max_neutral_candidates,
            "skip_disjoint_pairs": skip_disjoint_pairs,
            "frontier_sizes": frontier_sizes,
            "generated_counts": generated_counts,
            "accepted_counts": accepted_counts,
            "visited_states": len(visited),
        },
    }
