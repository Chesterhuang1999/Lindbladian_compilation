#!/usr/bin/env python3
"""Validate context-aware costs, frozen tree outputs, and channel semantics."""
import argparse
import contextlib
import io
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("src", "scripts", "Baseline_scripts")]


def check_costs():
    import numpy as np
    from channel_IR import channel as ChannelIR
    from channel_rewrite_cost import CostSettings, _row_cost, components_from_rows, h_length_components
    from channelIR_gatecost_rewrite_search import _candidate_gatecost_rotations
    from hybrid_rewrite import cost_tables, gate_candidates, row_data, row_summary

    rng = np.random.default_rng(8715)
    # Proportional row removal crosses selector widths at M=5->4 and M=3->2.
    matrices = [np.array([[1, 2], [2, 4]], complex),
                np.array([[1, 1j, -1, -1j], [1, -1j, -1, 1j]], complex),
                np.array([[1, 2, 0], [2, 4, 0], [0, 1j, 1]], complex),
                np.array([[1, 2, 0, 0], [2, 4, 0, 0], [0, 1j, 1, 0],
                          [1, 0, 0, 2j], [0, 0, 1, -1]], complex)]
    for _ in range(16):
        matrix = rng.normal(size=(5, 9)) + 1j * rng.normal(size=(5, 9))
        matrix[rng.random(matrix.shape) < .5] = 0
        matrices.append(matrix)
    row_count = candidate_count = width_crossings = 0
    maximum_error = 0.0
    for settings in (CostSettings("basic", False), CostSettings("tree", True),
                     CostSettings("basic", True), CostSettings("tree", False)):
        for matrix in matrices:
            table, outer, controls = cost_tables(*matrix.shape, settings)
            rows, alpha, row_costs = row_data(matrix, 1e-12, settings, table.shape[0])
            for row, reference in zip(matrix, rows):
                support, norm, costs = row_summary(row, 1e-12, table)
                assert support == reference.support
                assert math.isclose(norm, reference.alpha_squared, rel_tol=1e-13, abs_tol=1e-14)
                assert costs.tolist() == [sum(_row_cost(support, reference.phases, b, settings.fstar))
                                          if support else 0 for b in range(table.shape[0])]
                row_count += 1
            parent = components_from_rows(rows, settings)
            original = _candidate_gatecost_rotations(
                ChannelIR, matrix, np.array([row.support for row in rows]), parent["total"],
                1e-12, False, 3, True, settings)
            reference = {(c["pair"], ChannelIR._unitary_key(c["U"], 1e-12)): c for c in original}
            found = {}
            active = [i for i, row in enumerate(rows) if row.support]
            for pi, i in enumerate(active):
                for j in active[pi + 1:]:
                    if not np.any((np.abs(matrix[i]) > 1e-12) & (np.abs(matrix[j]) > 1e-12)):
                        continue
                    scores, new_i, new_j, unitaries, _ = gate_candidates(
                        matrix, i, j, 1e-12, False, math.inf, parent["total"], parent["total_support"],
                        len(active), alpha, row_costs, math.fsum(alpha), row_costs.sum(axis=0), table, outer, controls)
                    for q, score in enumerate(scores):
                        changed = matrix.copy()
                        changed[i], changed[j] = new_i[q], new_j[q]
                        exact = h_length_components(changed, structure=settings.structure, fstar=settings.fstar)
                        key = ((i, j), ChannelIR._unitary_key(unitaries[q], 1e-12))
                        assert key in reference, (settings, key)
                        error = abs(exact["total"] - score[0]) / max(abs(exact["total"]), 1e-30)
                        assert error < 2e-12, (settings, error)
                        maximum_error = max(maximum_error, error)
                        width_crossings += int(settings.structure == "basic" and exact["branch_controls"] != parent["branch_controls"])
                        found[key] = score[0]
                        candidate_count += 1
            assert set(found) == set(reference), (settings, len(found), len(reference))
    assert width_crossings > 0
    return {"rows_checked": row_count, "candidates_checked": candidate_count,
            "control_width_crossings_checked": width_crossings, "maximum_cost_relative_error": maximum_error}


def check_frozen():
    import numpy as np
    from hybrid_rewrite import hybrid_search
    results = {}
    root = ROOT / "paper_experiments/pending_review/support_gatecost_hybrid_20261012/cases"
    for name in ("Collective4", "Petz_TFIM2"):
        initial = np.load(root / name / "input.npz")["A"]
        expected = np.load(root / name / "hybrid.npz")["A_final"]
        result = hybrid_search(initial, contexts=("tree",))
        final = result["contexts"]["tree"]["state"].A
        error = float(np.linalg.norm(final - expected))
        assert error < 1e-12, (name, error)
        assert result["contexts"]["tree"]["record"]["status"] == "ok"
        results[name] = {"final_array_error": error,
                         "rewrite_verification": result["contexts"]["tree"]["record"]["verification"]}
    return results


def check_channels():
    import numpy as np
    from block_encoding import BlockEncoding
    from channel_IR import channel_ensemble
    from channel_LCU import channel_to_LCU
    from experiment_protocol import lower_and_count
    from hybrid_rewrite import hybrid_search
    from rq1_pipeline import verify_channel
    from run_rq2_hybrid_ablation import configurations, matrix_rows

    labels = np.array(["II", "IX", "IY", "IZ", "XX", "XY"])
    initial = np.array([[1, 1j, .5, -.2, .3, 0], [1, -1j, .5, -.2, 0, .4],
                        [.2, 0, 0, 0, 0, .7]], complex)
    hybrid = hybrid_search(initial)
    target = matrix_rows(initial, labels, 0)
    results = {}
    for slug, config in configurations().items():
        matrix = hybrid["contexts"][config["rewrite_context"]]["state"].A if config["rewrite"] else initial
        rows = matrix_rows(matrix, labels, 1e-12 if config["rewrite"] else 0)
        circuit, registers = channel_to_LCU(
            channel_ensemble([rows]), structure=config["structure"], opt=config["opt"], fstar=config["fstar"],
            block_encoding_cls=lambda ms: BlockEncoding(ms, placement_options={}, matrix_order_options={}))
        assert circuit.metadata["structure"] == config["structure"]
        assert circuit.metadata["block_opt"] == config["opt"]
        assert circuit.metadata["fstar"] == config["fstar"]
        assert circuit.metadata["controlled_prepare"] == (len(rows) > 1 and not config["fstar"])
        lowered, metrics = lower_and_count(circuit, rq="RQ2")
        assert metrics["settings"]["qubits_initially_zero"] is False
        check = verify_channel(lowered, registers, target)
        assert check["passed"], (slug, check)
        results[slug] = {"G": metrics["G"], "W": metrics["W"], "verification": check}
    return results


def check_random():
    from selected_channel_inputs import build_selected_case
    rows, model = build_selected_case({
        "id": "Random-5K-4q-10-2-seed0", "family": "random_pauli_kraus_channel", "system_qubits": 4,
        "parameters": {"kraus_count": 5, "num_qubits": 4, "first_kraus_terms": 10,
                       "other_kraus_terms": 2, "seed": 0}})
    assert len(rows) == 5 and all(row.size == 4 for row in rows)
    assert [len(row.instances) for row in rows] == [10, 2, 2, 2, 2]
    return model["input_validation"]


def main():
    import json
    import platform
    import time
    from run_rq2_hybrid_ablation import configure_environment, dump
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-archived", action="store_true")
    args = parser.parse_args()
    configure_environment(args.output.resolve().parent)
    from hybrid_rewrite import warmup
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        jit_s = warmup()
        result = {"passed": True, "python": sys.version, "host": platform.node(),
                  "jit_warmup_s": jit_s, "costs": check_costs(), "channels": check_channels(),
                  "random_input": check_random()}
        if not args.skip_archived:
            result["frozen_tree_parity"] = check_frozen()
    result["wall_s"] = time.perf_counter() - started
    dump(args.output.resolve(), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
