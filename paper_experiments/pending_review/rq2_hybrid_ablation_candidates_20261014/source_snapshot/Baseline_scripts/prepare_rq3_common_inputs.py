"""Freeze the seven common-outer QSVT inputs, including MaxCut cycle-4."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pennylane as qml
from qiskit.quantum_info import SparsePauliOp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "Baseline_results/rq3_common_outer_20261014"


def prepare(out_dir):
    previous_path = ROOT / "Baseline_results/rq3_refresh_20261013/qsvt_inputs.json"
    previous = json.loads(previous_path.read_text())
    cases = [deepcopy(case) for case in previous["cases"]
             if case["id"] != "fullpauli2_nonseparable"]
    assert len(cases) == 6
    graph = nx.cycle_graph(4)
    cost, _ = qml.qaoa.maxcut(graph)
    # Retain API term order after moving the combined identity term first,
    # matching the existing triangle's ordered-input convention.
    terms = []
    for word, coefficient in cost.pauli_rep.items():
        label = "".join(word.get(bit, "I") for bit in reversed(range(4)))
        assert abs(complex(coefficient).imag) < 1e-14
        terms.append([label, float(complex(coefficient).real)])
    terms = [term for term in terms if term[0] == "IIII"] + [
        term for term in terms if term[0] != "IIII"]
    assert len(terms) == 5 and terms[0] == ["IIII", -2.0]
    edges = [list(edge) for edge in graph.edges]
    matrix = SparsePauliOp.from_list(terms).to_matrix()
    cut_values = [sum(((basis >> u) & 1) != ((basis >> v) & 1)
                      for u, v in graph.edges) for basis in range(16)]
    expected = -np.diag(cut_values)
    error = float(np.max(np.abs(matrix - expected)))
    assert error < 1e-14
    definition = {
        "id": "maxcut_cycle4", "name": "MaxCut 四节点环（官方 API）",
        "origin": "official_api_graph_variant",
        "source": "https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.maxcut.html",
        "formula": "sum_edges (Z_u Z_v - I)/2", "edges": edges,
        "terms": terms, "api": "qml.qaoa.maxcut(nx.cycle_graph(4))[0]",
        "basis_state_cut_validation": {
            "qiskit_bit_order": "node i corresponds to system qubit i",
            "diagonal_equals_negative_cut_value": True,
            "checked_states": 16, "cut_values": cut_values,
            "max_abs_error": error,
        },
    }
    cases.append({
        "id": definition["id"], "name": definition["name"],
        "terms": terms, "alpha": sum(abs(c) for _, c in terms),
        "poly": deepcopy(cases[0]["poly"]), "angles": deepcopy(cases[0]["angles"]),
        "source_member": None, "origin": definition["origin"], "case_definition": definition,
    })
    assert all(case["poly"] == cases[0]["poly"] and case["angles"] == cases[0]["angles"]
               for case in cases)
    frozen = {
        "archive": previous["archive"], "archive_sha256": previous["archive_sha256"],
        "inherited_inputs": str(previous_path.relative_to(ROOT)),
        "inherited_inputs_sha256": sha256(previous_path.read_bytes()).hexdigest(),
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"pennylane": qml.__version__, "networkx": nx.__version__},
        "cases": cases,
        "policy": {
            "outer": "H-C1(U_phi)-C0(U_phi.adjoint)-H",
            "projector": "logical block address zero; clean expansion scratch excluded",
            "degree": 5, "query_count": 10, "projector_count": 12,
            "alpha": "sum(abs(coefficients))", "target": "P(H/alpha)",
            "polynomial_parity": "odd", "frozen_time_ordered_phases": True,
            "routes": ["pl_native", "qrisp_native", "ours"],
            "retired_active_case": "fullpauli2_nonseparable",
            "historical_artifacts_preserved": True,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "qsvt_inputs.json"
    text = json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output.exists() and output.read_text() != text:
        raise ValueError("Existing frozen inputs differ; use a new batch directory.")
    output.write_text(text)
    print(json.dumps({"path": str(output.relative_to(ROOT)), "cases": [c["id"] for c in cases],
                      "cycle4_terms": terms, "cycle4_alpha": cases[-1]["alpha"],
                      "cut_validation_error": error}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    prepare(parser.parse_args().out_dir)
