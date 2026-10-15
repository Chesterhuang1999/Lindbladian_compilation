"""Materialize the selected benchmark inputs for the full channel pipeline."""

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def legacy_tfim_higher_order(spec):
    """Materialize the finite trajectory operators of the legacy gadget input.

    The coherent Taylor arithmetic and quadrature indexing follow the existing
    builder. Multiplying its finite factors gives Pauli Kraus inputs for the
    unified channel compiler; gadget-level gate counts are not comparable.
    """
    from qiskit.quantum_info import Pauli
    from channel_IR import Matrixsum, PauliAtom, matsum_mul
    from build_periodic_tfim_higher_order_expansion import build_periodic_tfim_lindbladian
    from series_expansion import create_index_set, quadrature_points_N_weights

    n, params = spec["system_qubits"], spec["parameters"]
    gamma = np.sqrt(0.1) / 2
    lind = build_periodic_tfim_lindbladian(n, gamma)
    generator = lind.effective_H()
    jumps = [op.eff_op().to_matrix() for op in lind.L_list]
    dimension = 1 << n
    coherent_cache = {}

    def coherent(interval):
        key = float(interval)
        if key not in coherent_cache:
            total, term = generator.identity(n), None
            for degree in range(1, params["K1"] + 1):
                term = deepcopy(generator) if degree == 1 else matsum_mul(term, generator)
                term.mul_coeffs(interval / degree)
                total = total.add(term)
            coherent_cache[key] = total.eff_op().to_matrix()
        return coherent_cache[key]

    matrices = [coherent(params["delta_t"])]
    indices = [[0]]
    for order in range(1, params["K"] + 1):
        for index in create_index_set(order, params["q"], len(jumps)):
            points, weights = quadrature_points_N_weights(
                order, params["q"], params["delta_t"], index[order + 1:2 * order + 1])
            points.append(0)
            value = np.eye(dimension, dtype=complex)
            for segment in range(order + 1):
                interval = (params["delta_t"] - points[0] if segment == order
                            else points[order - segment - 1] - points[order - segment])
                value = coherent(interval) @ value
                if segment < order:
                    value = jumps[index[1:order + 1][order - segment - 1]] @ value
            matrices.append(np.sqrt(np.prod(weights)) * value)
            indices.append(index)
    basis = [(label, Pauli(label).to_matrix())
             for label in map("".join, product("IXYZ", repeat=n))]
    rows, errors = [], []
    for matrix in matrices:
        terms = []
        for label, pauli in basis:
            value = np.trace(pauli @ matrix) / dimension
            if value != 0:
                terms.append((PauliAtom(label, phase=value / abs(value)), float(abs(value))))
        ms = Matrixsum(terms)
        actual = ms.eff_op().to_matrix() if terms else np.zeros_like(matrix)
        errors.append(float(np.linalg.norm(actual - matrix)))
        if terms:
            rows.append(ms)
    assert max(errors, default=0) < 1e-12
    return rows, {**params, "gamma": float(gamma), "logical_kraus_count": len(matrices),
                  "exact_zero_kraus_removed": len(matrices) - len(rows), "indices": indices,
                  "dense_to_pauli_max_error": max(errors, default=0),
                  "representation": "materialized finite Duhamel/Taylor Kraus operators",
                  "legacy_builder": "scripts/build_periodic_tfim_higher_order_expansion.py"}


def build_selected_case(spec):
    family, n = spec["family"], spec["system_qubits"]
    if family == "tfim_first_order":
        from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel
        rows, model = build_periodic_tfim_channel(n, spec["parameters"]["delta_t"])
    elif family == "hrw":
        from run_hypercube_n4_ours_nam import build_hypercube_channel
        rows, model = build_hypercube_channel(n), {"definition": "HRW section 5.2"}
    elif family == "tfim_higher_order_legacy":
        rows, model = legacy_tfim_higher_order(spec)
    elif family == "new_benchmark_fixture":
        from channel_IR import Matrixsum, PauliAtom
        path = ROOT / spec["fixture"]
        payload = json.loads(path.read_text())
        rows = []
        for row in payload["stages"]["raw"]["kraus"]:
            terms = []
            for label, pair in row.items():
                value = complex(*pair)
                if value != 0:
                    terms.append((PauliAtom(label, phase=value / abs(value)), float(abs(value))))
            rows.append(Matrixsum(terms))
        model = {"parameters": payload["parameters"], "fixture": spec["fixture"],
                 "fixture_sha256": sha256(path.read_bytes()).hexdigest(),
                 "input_stage": "raw"}
    elif family == "toom":
        from build_toom_2d_channel_lcu import build_toom_channel
        p = spec["parameters"]
        rows, model = build_toom_channel(p["L"], p["gamma"], p["delta_t"])
    elif family in ("graph_cluster_grid", "graph_cluster_random"):
        from build_graph_cluster_state_channel_lcu import (
            grid_2d_cluster_graph, random_bounded_degree_graph, build_channel_for_spec)
        p = spec["parameters"]
        graph = (grid_2d_cluster_graph(p["rows"], p["columns"])
                 if family == "graph_cluster_grid" else
                 random_bounded_degree_graph(n, p["max_degree"], p["seed"]))
        rows, model = build_channel_for_spec(graph, p["gamma"], p["delta_t"])
    elif family == "random_coloring":
        from run_random_coloring_channel_lcu_ours_nam import build_random_coloring_channel
        p = spec["parameters"]
        rows, model = build_random_coloring_channel(
            p["num_vertices"], p["graph"], q=p["colors"], tol=1e-12,
            max_total_pauli_terms=250000, max_terms_per_kraus=50000,
            color_mixing_preprocess=p["color_mixing_preprocess"])
    elif family == "repetition_dqe":
        from build_repetition_code_dqe_channel import (
            DqeSpec, build_repetition_dqe_channel, pauli_dict_to_matrixsum)
        dictionaries, model = build_repetition_dqe_channel(DqeSpec(**spec["parameters"]))
        rows = [pauli_dict_to_matrixsum(row) for row in dictionaries]
    elif family == "decay_higher_order":
        from paper_rq1_inputs import decay_kraus
        rows, model = decay_kraus()
        assert all(model[k] == v for k, v in spec["parameters"].items())
    elif family == "random_pauli_kraus_channel":
        from build_random_channel_lcu_benchmark import (
            RandomChannelSpec, RandomChannelCase, build_random_kraus_ops)
        p = spec["parameters"]
        random_spec = RandomChannelSpec.from_qubits(
            kraus_count=p["kraus_count"], num_qubits=p["num_qubits"],
            first_kraus_terms=p["first_kraus_terms"], other_kraus_terms=p["other_kraus_terms"])
        rows, model = build_random_kraus_ops(RandomChannelCase(random_spec, p["seed"]))
        expected = [p["first_kraus_terms"]] + [p["other_kraus_terms"]] * (p["kraus_count"] - 1)
        actual = [len(row.instances) for row in rows]
        assert actual == expected and n == p["num_qubits"], (actual, expected)
        model["input_validation"] = {"passed": True, "term_counts": actual,
                                     "system_qubits": n, "hilbert_dim": 1 << n}
    else:
        raise ValueError(f"Unsupported selected input: {spec['id']}")
    assert rows and all(op.size == n for op in rows)
    return rows, {**model, "system_qubits": n, "suite_case": spec["id"], "family": family}
