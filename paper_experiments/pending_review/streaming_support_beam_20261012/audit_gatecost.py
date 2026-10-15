"""Replay saved support path and audit its rejection by gate-cost rules."""
from pathlib import Path
import json
import math

import numpy as np

from fast_probe import ChannelIR, dump, h_length_components, row_key, state_key

HERE = Path(__file__).resolve().parent
OUT = HERE / "diagnosis_validation"


def main():
    OUT.mkdir(exist_ok=True)
    saved = np.load(HERE / "Thermal2_HO_support_rank/coefficients.npz")
    A = saved["A_initial"].copy()
    steps = json.loads((HERE / "Thermal2_HO_support_rank/rewrite_steps.json").read_text())
    gate = json.loads((HERE / "Thermal2_HO_gatecost_neutral/results.json").read_text())
    frontier = gate["termination"]["trajectory"]
    rows = []
    first_absent = None
    component_keys = ("total", "Lambda", "unweighted_nc", "outer_prep", "outer_flags",
                      "block_prep", "block_select", "phase", "kraus_count", "total_support")
    for index, step in enumerate(steps, 1):
        i, j = step["pair"]
        U = np.array(step["U_real"]) + 1j * np.array(step["U_imag"])
        before = h_length_components(A)
        before_supports = np.count_nonzero(np.abs(A[[i, j]]) > 1e-12, axis=1).tolist()
        old_pair = A[[i, j]].copy()
        ni = U[0, 0] * A[i] + U[0, 1] * A[j]
        nj = U[1, 0] * A[i] + U[1, 1] * A[j]
        ni[np.abs(ni) < 1e-12] = 0
        nj[np.abs(nj) < 1e-12] = 0
        A[i], A[j] = ni, nj
        after = h_length_components(A)
        eq = math.isclose(after["total"], before["total"], rel_tol=1e-12, abs_tol=0)
        row = {"step": index, "pair": [i, j], "H_equal": eq,
               "before": {k: before[k] for k in component_keys},
               "after": {k: after[k] for k in component_keys}}
        if eq:
            after_score = ChannelIR._local_future_potential(A, (i, j), 1e-12)["score"]
            A[[i, j]] = old_pair
            before_score = ChannelIR._local_future_potential(A, (i, j), 1e-12)["score"]
            A[i], A[j] = ni, nj
            row.update(before_pair_supports=before_supports,
                       after_pair_supports=np.count_nonzero(np.abs(A[[i, j]]) > 1e-12, axis=1).tolist(),
                       before_potential=before_score, after_potential=after_score,
                       rejected_by_neutral_screen=after_score <= before_score)
        key = state_key(tuple(row_key(r, 1e-12) for r in A)).hex()
        if index <= len(frontier):
            row["present_in_gate_frontier"] = key in frontier[index - 1]["frontier_keys"]
            if not row["present_in_gate_frontier"] and first_absent is None:
                first_absent = {"step": index, "pair": [i, j], "support_path_H": after["total"],
                                "gate_frontier_costs": frontier[index - 1]["frontier_costs"],
                                "support": after["total_support"],
                                "gate_frontier_supports": frontier[index - 1]["frontier_supports"]}
        rows.append(row)
    assert np.max(np.abs(A - saved["A_final"])) < 1e-14
    rejected = [r for r in rows if r.get("rejected_by_neutral_screen")]
    result = {"replay_matches_saved": True, "first_support_path_prefix_absent_from_gate_beam": first_absent,
              "rejected_by_neutral_screen": len(rejected),
              "rejected_singleton_merges": sum(r["before_pair_supports"] == [1, 1]
                                               and sum(r["after_pair_supports"]) == 1 for r in rejected),
              "first_rejected_step": rejected[0] if rejected else None, "path": rows}
    dump(OUT / "path_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "path"}, indent=2))


if __name__ == "__main__":
    main()
