#!/usr/bin/env python3
"""Reproduce sparse coherent alias sampling registers for TFIM_8 A0 PREPARE.

This script mirrors the register layout and classical alias-table preprocessing
used by Qualtran's ``SparseStatePreparationAliasSampling``.  It does not depend
on Qualtran being installed; the small preprocessing helpers are reproduced
from the published Qualtran/OpenFermion logic so that we can apply the scheme to
the matrix-order coefficient addresses produced by this project.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding_new import BlockEncoding  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    json_default,
)


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_sparse_alias_sampling_qubit_count_v1"
QUALTRAN_SOURCE_URL = (
    "https://github.com/quantumlib/Qualtran/blob/main/"
    "qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py"
)
LCU_UTIL_SOURCE_URL = (
    "https://github.com/quantumlib/Qualtran/blob/main/qualtran/linalg/lcu_util.py"
)
TOL = 1e-15


def partial_sums(values: Sequence[float]):
    total = 0.0
    for value in values:
        yield total
        total += float(value)
    yield total


def differences(values: Sequence[int]) -> list[int]:
    return [int(right - left) for left, right in zip(values[:-1], values[1:])]


def discretize_probability_distribution(
    unnormalized_probabilities: Sequence[float],
    sub_bit_precision: int,
) -> tuple[list[int], int]:
    """Qualtran/OpenFermion discretization over denominator n * 2**mu."""

    n = len(unnormalized_probabilities)
    bin_count = (1 << sub_bit_precision) * n
    cumulative = list(partial_sums(unnormalized_probabilities))
    total = cumulative[-1]
    discretized_cumulative = [
        int(math.floor(c / total * bin_count + 0.5)) for c in cumulative
    ]
    return differences(discretized_cumulative), bin_count


def preprocess_for_efficient_roulette_selection(
    discretized_probabilities: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Alias-table preprocessing used by Qualtran's lcu_util."""

    weights = [int(value) for value in discretized_probabilities]
    if not weights:
        raise ValueError("empty probability list")
    n = len(weights)
    target_weight = sum(weights) // n
    if sum(weights) != n * target_weight:
        raise ValueError(f"{sum(weights)=} must be a multiple of {n=}")

    alternates = list(range(n))
    keep_weights = [0] * n
    donor_position = 0
    for _ in range(2):
        for i in range(n):
            if weights[i] >= target_weight:
                continue
            while weights[donor_position] <= target_weight:
                donor_position += 1
            donated = target_weight - weights[i]
            weights[donor_position] -= donated
            alternates[i] = donor_position
            keep_weights[i] = weights[i]
            weights[i] = target_weight

    return alternates, keep_weights


def sub_bit_prec_from_epsilon(number_of_coefficients: int, precision: float) -> int:
    """mu = ceil(log2(1 / (L * epsilon))), with at least one bit."""

    mu = math.ceil(math.log2(1.0 / (precision * number_of_coefficients)))
    return max(1, int(mu))


def preprocess_probabilities_for_reversible_sampling(
    unnormalized_probabilities: Sequence[float],
    *,
    epsilon: float | None = None,
    sub_bit_precision: int | None = None,
) -> tuple[list[int], list[int], int, list[int], int]:
    if (epsilon is None) == (sub_bit_precision is None):
        raise ValueError("provide exactly one of epsilon or sub_bit_precision")
    if sub_bit_precision is None:
        assert epsilon is not None
        sub_bit_precision = sub_bit_prec_from_epsilon(len(unnormalized_probabilities), epsilon)
    if sub_bit_precision <= 0:
        raise ValueError("sub_bit_precision must be at least 1")

    numerators, denominator = discretize_probability_distribution(
        unnormalized_probabilities, sub_bit_precision
    )
    alternates, keep_numerators = preprocess_for_efficient_roulette_selection(numerators)
    return alternates, keep_numerators, sub_bit_precision, numerators, denominator


def extract_tfim8_a0_auto_coefficients() -> tuple[np.ndarray, dict[str, Any], float]:
    started = time.perf_counter()
    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    ms = kraus_ops[0]
    be = BlockEncoding(copy.deepcopy(ms))
    coeff_mode_dict, ctrl_size = be.find_optimal_order_matrices()
    coeff_list_ordered = np.zeros(1 << ctrl_size, dtype=complex)
    for ctrl_value, coeff in coeff_mode_dict.items():
        coeff_list_ordered[int(ctrl_value, 2)] = coeff

    elapsed = time.perf_counter() - started
    nonzero = np.nonzero(np.abs(coeff_list_ordered) > TOL)[0]
    metadata = {
        "benchmark": "TFIM_8",
        "kraus_index": 0,
        "model_info": model_info,
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "auto_ctrl_size": int(ctrl_size),
        "raw_ctrl_size": int(be.ctrl_size),
        "selected_k": int(getattr(be, "selected_k", ctrl_size)),
        "selected_crit": str(getattr(be, "selected_crit", "")),
        "coeff_slots": int(len(coeff_list_ordered)),
        "nonzero_coeff_slots": int(len(nonzero)),
        "nonzero_coefficients": [
            {
                "sparse_j": int(j),
                "index": int(index),
                "ctrl": format(int(index), f"0{ctrl_size}b"),
                "abs_coeff": float(abs(coeff_list_ordered[index])),
                "coeff": repr(coeff_list_ordered[index]),
            }
            for j, index in enumerate(nonzero)
        ],
    }
    return coeff_list_ordered, metadata, elapsed


def alias_distribution(
    indices: Sequence[int],
    alternates_actual: Sequence[int],
    keep: Sequence[int],
    mu: int,
    N: int,
) -> np.ndarray:
    d = len(indices)
    denom = d * (1 << mu)
    distribution = np.zeros(N, dtype=float)
    for index, alt, keep_value in zip(indices, alternates_actual, keep):
        distribution[int(index)] += float(keep_value) / denom
        distribution[int(alt)] += float((1 << mu) - int(keep_value)) / denom
    return distribution


def qubit_counts(*, N: int, d: int, mu: int) -> dict[str, int]:
    selection_bitsize = (N - 1).bit_length()
    sparse_index_bitsize = (d - 1).bit_length()
    signature_qubits = (
        selection_bitsize
        + sparse_index_bitsize
        + mu
        + selection_bitsize
        + mu
        + 1
    )
    qrom_signature_qubits = sparse_index_bitsize + selection_bitsize + selection_bitsize + mu
    qrom_unary_iteration_clean_ancilla = max(sparse_index_bitsize - 1, 0)
    qrom_static_qubits = qrom_signature_qubits + qrom_unary_iteration_clean_ancilla
    peak_with_qrom_static_extra = signature_qubits + qrom_unary_iteration_clean_ancilla

    # The SparseStatePreparationAliasSampling docstring currently states
    # "+ log(L) ancillas for QROM".  QROM.my_static_costs instead adds the
    # selection-register size of the QROM, i.e. log(d)-1 extra clean ancillas
    # for the sparse-index-addressed QROM.  We record both interpretations.
    docstring_qrom_clean_ancilla = selection_bitsize
    peak_with_docstring_qrom_extra = signature_qubits + docstring_qrom_clean_ancilla

    return {
        "selection_bitsize_log_N": int(selection_bitsize),
        "sparse_index_bitsize_log_d": int(sparse_index_bitsize),
        "sigma_mu_bitsize": int(mu),
        "alt_bitsize": int(selection_bitsize),
        "keep_bitsize": int(mu),
        "less_than_equal_bitsize": 1,
        "signature_qubits": int(signature_qubits),
        "qrom_signature_qubits": int(qrom_signature_qubits),
        "qrom_unary_iteration_clean_ancilla": int(qrom_unary_iteration_clean_ancilla),
        "qrom_static_qubits": int(qrom_static_qubits),
        "peak_qubits_with_qrom_static_extra": int(peak_with_qrom_static_extra),
        "docstring_qrom_clean_ancilla_log_N": int(docstring_qrom_clean_ancilla),
        "peak_qubits_with_docstring_qrom_extra": int(peak_with_docstring_qrom_extra),
    }


def dense_alias_qubit_counts(*, N: int, mu: int) -> dict[str, int]:
    selection_bitsize = (N - 1).bit_length()
    signature_qubits = selection_bitsize + mu + selection_bitsize + mu + 1
    qrom_unary_iteration_clean_ancilla = max(selection_bitsize - 1, 0)
    return {
        "selection_bitsize_log_N": int(selection_bitsize),
        "sigma_mu_bitsize": int(mu),
        "alt_bitsize": int(selection_bitsize),
        "keep_bitsize": int(mu),
        "less_than_equal_bitsize": 1,
        "signature_qubits": int(signature_qubits),
        "qrom_unary_iteration_clean_ancilla": int(qrom_unary_iteration_clean_ancilla),
        "peak_qubits_with_qrom_static_extra": int(
            signature_qubits + qrom_unary_iteration_clean_ancilla
        ),
        "qrom_iteration_length": int(N),
    }


def unary_iteration_counts(iteration_length: int, selection_bitsize: int) -> dict[str, int]:
    """Reproduce Qualtran's one-dimensional zero-control unary-iteration counts.

    This covers the QROM used by sparse alias sampling, where the QROM is
    addressed by ``sparse_index`` and has no external controls.
    """

    if iteration_length <= 0:
        raise ValueError("iteration_length must be positive")
    if selection_bitsize <= 0:
        return {
            "x": 0,
            "and": 0,
            "and_adjoint": 0,
            "tree_cnot": 0,
            "leaf_count": 1,
        }

    def segtree_counts(l_iter: int, r_iter: int, l_range: int, r_range: int) -> tuple[int, int]:
        n = r_range - l_range
        n_levels = n.bit_length()
        marked = [False] * (2 * n)
        num_ands = 0
        leaves = 0
        step_size = n
        for lvl in range(1, n_levels + 1):
            r = l_range
            for i in range(1 << (lvl - 1), 1 << lvl):
                l = r
                r = l + step_size
                marked[i] = marked[i >> 1]
                if marked[i]:
                    continue
                if l >= r_iter or l_iter >= r:
                    marked[i] = True
                    continue
                if l_iter <= l < r <= r_iter and i >= n:
                    marked[i] = True
                    leaves += 1
                    continue
                m = (l + r) >> 1
                if r_iter <= m or l_iter >= m:
                    continue
                num_ands += 1
            step_size //= 2
        return num_ands, leaves

    while iteration_length <= 1 << (selection_bitsize - 1):
        selection_bitsize -= 1
    l_range = 0
    r_range = 1 << selection_bitsize
    midpoint = (l_range + r_range) >> 1
    left_ands, left_leaves = segtree_counts(0, iteration_length, l_range, midpoint)
    right_ands, right_leaves = segtree_counts(0, iteration_length, midpoint, r_range)
    and_count = left_ands + right_ands
    return {
        "x": 2,
        "and": int(and_count),
        "and_adjoint": int(and_count),
        "tree_cnot": int(and_count),
        "leaf_count": int(left_leaves + right_leaves),
    }


def qrom_gate_counts(
    *,
    indices: Sequence[int],
    alternates_actual: Sequence[int],
    keep: Sequence[int],
    N: int,
    d: int,
    mu: int,
) -> dict[str, Any]:
    selection_bitsize = (N - 1).bit_length()
    sparse_index_bitsize = (d - 1).bit_length()
    rows = []
    load_cnot_total = 0
    index_load_cnot_total = 0
    alt_load_cnot_total = 0
    keep_load_cnot_total = 0
    for sparse_j, (index, alt, keep_value) in enumerate(zip(indices, alternates_actual, keep)):
        index_cnot = int(index).bit_count()
        alt_cnot = int(alt).bit_count()
        keep_cnot = int(keep_value).bit_count()
        row_total = index_cnot + alt_cnot + keep_cnot
        index_load_cnot_total += index_cnot
        alt_load_cnot_total += alt_cnot
        keep_load_cnot_total += keep_cnot
        load_cnot_total += row_total
        rows.append(
            {
                "sparse_j": int(sparse_j),
                "index": int(index),
                "index_bitcount": int(index_cnot),
                "alt": int(alt),
                "alt_bitcount": int(alt_cnot),
                "keep": int(keep_value),
                "keep_bitcount": int(keep_cnot),
                "load_cnot": int(row_total),
            }
        )

    unary = unary_iteration_counts(d, sparse_index_bitsize)
    tree_cnot = int(unary["tree_cnot"])
    return {
        "qrom_selection_register": "sparse_index",
        "qrom_selection_bitsize_log_d": int(sparse_index_bitsize),
        "qrom_iteration_length_d": int(d),
        "target_bitsizes": {
            "index": int(selection_bitsize),
            "alt": int(selection_bitsize),
            "keep": int(mu),
            "total_per_row": int(2 * selection_bitsize + mu),
        },
        "unary_iteration": unary,
        "load_cnot_by_target": {
            "index": int(index_load_cnot_total),
            "alt": int(alt_load_cnot_total),
            "keep": int(keep_load_cnot_total),
        },
        "load_cnot_total": int(load_cnot_total),
        "tree_cnot_total": tree_cnot,
        "total_cnot_in_qrom_callgraph": int(load_cnot_total + tree_cnot),
        "and": int(unary["and"]),
        "and_adjoint": int(unary["and_adjoint"]),
        "x": int(unary["x"]),
        "row_load_counts": rows,
    }


def build_case(
    weights_dense: np.ndarray,
    *,
    case_name: str,
    precision: float | None,
    mu: int | None,
) -> dict[str, Any]:
    N = int(len(weights_dense))
    indices = [int(index) for index in np.nonzero(weights_dense > TOL)[0]]
    values = [float(weights_dense[index]) for index in indices]
    d = len(values)

    if mu is None:
        if precision is None:
            raise ValueError("precision or mu is required")
        alt_compressed, keep, mu, numerators, denominator = (
            preprocess_probabilities_for_reversible_sampling(values, epsilon=precision)
        )
        precision_used = float(precision)
    else:
        alt_compressed, keep, mu, numerators, denominator = (
            preprocess_probabilities_for_reversible_sampling(values, sub_bit_precision=mu)
        )
        precision_used = float(1.0 / (d * (1 << mu)))

    alternates_actual = [indices[int(alt_j)] for alt_j in alt_compressed]
    approx_distribution = alias_distribution(indices, alternates_actual, keep, mu, N)
    target_distribution = weights_dense / float(np.sum(weights_dense))
    max_abs_probability_error = float(np.max(np.abs(approx_distribution - target_distribution)))
    l1_probability_error = float(np.sum(np.abs(approx_distribution - target_distribution)))

    table = []
    for sparse_j, (index, value, alt_j, alt_actual, keep_value, numerator) in enumerate(
        zip(indices, values, alt_compressed, alternates_actual, keep, numerators)
    ):
        table.append(
            {
                "sparse_j": int(sparse_j),
                "index": int(index),
                "ctrl": format(int(index), f"0{(N - 1).bit_length()}b"),
                "abs_coeff": float(value),
                "target_probability": float(target_distribution[index]),
                "discretized_numerator": int(numerator),
                "keep": int(keep_value),
                "keep_probability": float(keep_value / (1 << mu)),
                "alt_compressed_j": int(alt_j),
                "alt_index": int(alt_actual),
                "alt_ctrl": format(int(alt_actual), f"0{(N - 1).bit_length()}b"),
            }
        )

    return {
        "case": case_name,
        "precision_input": None if precision is None else float(precision),
        "effective_precision_bound": precision_used,
        "mu": int(mu),
        "N": int(N),
        "d_nonzero": int(d),
        "sum_abs_coeff": float(np.sum(weights_dense)),
        "alias_denominator": int(denominator),
        "max_abs_probability_error": max_abs_probability_error,
        "l1_probability_error": l1_probability_error,
        "qubit_counts_sparse_alias": qubit_counts(N=N, d=d, mu=mu),
        "qubit_counts_dense_alias_same_mu": dense_alias_qubit_counts(N=N, mu=mu),
        "qrom_iteration_lengths": {
            "sparse_alias_qrom_rows": int(d),
            "dense_alias_qrom_rows": int(N),
        },
        "qrom_gate_counts_sparse_alias": qrom_gate_counts(
            indices=indices,
            alternates_actual=alternates_actual,
            keep=keep,
            N=N,
            d=d,
            mu=mu,
        ),
        "alias_table": table,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    metadata = summary["metadata"]
    lines = [
        "# TFIM_8 A0 Sparse Coherent Alias Sampling PREPARE",
        "",
        "This reproduces the register layout and alias table for Qualtran's `SparseStatePreparationAliasSampling` on the TFIM_8 A0 matrix-order coefficient vector.",
        "",
        f"- Qualtran source: {QUALTRAN_SOURCE_URL}",
        f"- Qualtran lcu_util source: {LCU_UTIL_SOURCE_URL}",
        f"- auto matrix-order width `w`: `{metadata['auto_ctrl_size']}`",
        f"- nonzero coefficients `d`: `{metadata['nonzero_coeff_slots']}`",
        f"- total coefficient slots `N`: `{metadata['coeff_slots']}`",
        "",
        "## Algorithm Registers",
        "",
        "Sparse coherent alias sampling uses `sparse_index` to sample one of the nonzero entries uniformly, QROM-loads `(index, alt, keep)`, compares `keep` against a uniform `sigma_mu`, then conditionally swaps `selection` with `alt`.",
        "",
        "## Qubit Counts",
        "",
        "| case | mu | max prob err | signature qubits | QROM extra | peak qubits | docstring peak | dense-alias peak same mu | QROM rows sparse/dense |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        qc = case["qubit_counts_sparse_alias"]
        dense = case["qubit_counts_dense_alias_same_mu"]
        qrom_rows = case["qrom_iteration_lengths"]
        lines.append(
            f"| {case['case']} | {case['mu']} | {case['max_abs_probability_error']:.3e} | "
            f"{qc['signature_qubits']} | {qc['qrom_unary_iteration_clean_ancilla']} | "
            f"{qc['peak_qubits_with_qrom_static_extra']} | "
            f"{qc['peak_qubits_with_docstring_qrom_extra']} | "
            f"{dense['peak_qubits_with_qrom_static_extra']} | "
            f"{qrom_rows['sparse_alias_qrom_rows']}/{qrom_rows['dense_alias_qrom_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Sparse Alias QROM Main Gate Counts",
            "",
            "Counts follow Qualtran's unary-iteration QROM callgraph style: `And/And†` for address iteration and CNOTs for loading classical data bits. The QROM is addressed by `sparse_index`, not by the final matrix-order `selection` register.",
            "",
            "| case | mu | target bits/row | load CNOT index | load CNOT alt | load CNOT keep | load CNOT total | tree CNOT | total CNOT | And | And† | X |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in summary["cases"]:
        qrom = case["qrom_gate_counts_sparse_alias"]
        by_target = qrom["load_cnot_by_target"]
        lines.append(
            f"| {case['case']} | {case['mu']} | {qrom['target_bitsizes']['total_per_row']} | "
            f"{by_target['index']} | {by_target['alt']} | {by_target['keep']} | "
            f"{qrom['load_cnot_total']} | {qrom['tree_cnot_total']} | "
            f"{qrom['total_cnot_in_qrom_callgraph']} | {qrom['and']} | "
            f"{qrom['and_adjoint']} | {qrom['x']} |"
        )

    default_case = summary["cases"][0]
    lines.extend(
        [
            "",
            "## Default Alias Table",
            "",
            f"Case `{default_case['case']}` uses `mu={default_case['mu']}`.",
            "",
            "| j | index | coeff | target p | keep | keep/2^mu | alt index |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in default_case["alias_table"]:
        lines.append(
            f"| {row['sparse_j']} | `{row['ctrl']}` | {row['abs_coeff']:.12g} | "
            f"{row['target_probability']:.12g} | {row['keep']} | "
            f"{row['keep_probability']:.12g} | `{row['alt_ctrl']}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--precision",
        type=float,
        default=1.0e-5,
        help="Qualtran-style sparse alias precision for the default case.",
    )
    parser.add_argument(
        "--mu-sweep",
        type=int,
        nargs="*",
        default=[8, 10, 12, 16],
        help="Additional fixed-mu cases to report.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coeffs, metadata, extraction_seconds = extract_tfim8_a0_auto_coefficients()
    weights_dense = np.asarray([abs(value) for value in coeffs], dtype=float)

    cases = [
        build_case(
            weights_dense,
            case_name=f"qualtran_default_precision_{args.precision:g}",
            precision=args.precision,
            mu=None,
        )
    ]
    default_mu = cases[0]["mu"]
    for mu in args.mu_sweep:
        if mu == default_mu:
            case_name = f"fixed_mu_{mu}_same_as_default"
        else:
            case_name = f"fixed_mu_{mu}"
        cases.append(build_case(weights_dense, case_name=case_name, precision=None, mu=mu))

    summary = {
        "script": str(Path(__file__).resolve()),
        "result_dir": str(out_dir),
        "qualtran_sources": {
            "sparse_state_preparation_alias_sampling": QUALTRAN_SOURCE_URL,
            "lcu_util": LCU_UTIL_SOURCE_URL,
        },
        "metadata": metadata,
        "coefficient_extraction_seconds": float(extraction_seconds),
        "cases": cases,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(out_dir / "report.md", summary)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
