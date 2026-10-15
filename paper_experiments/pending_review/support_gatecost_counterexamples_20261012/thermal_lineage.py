"""Trace the frozen Thermal2_HO rewrites back to labelled input trajectories."""
from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "streaming_support_beam_20261012"
sys.path.insert(0, str(PREVIOUS))
from fast_probe import dump, verify


def main():
    fixture = HERE.parent / "thermal2_ho_remote_20261011/cases/Thermal2_HO/input.json"
    source = json.loads(fixture.read_text())
    trajectories = source["model"]["parameters"]["trajectory_labels"]
    reference = np.load(PREVIOUS / "Thermal2_HO_support_rank/coefficients.npz")
    A0, labels = reference["A_initial"], reference["labels"]
    raw = np.zeros_like(A0)
    columns = {p: i for i, p in enumerate(labels)}
    for j, row in enumerate(source["input_kraus"]):
        for p, mag, re, im in row:
            raw[j, columns[p]] += mag * complex(re, im)
    phases = np.array([np.vdot(x, y) / np.vdot(x, x) for x, y in zip(raw, A0)])
    phase_error = float(np.max(np.abs(A0 - phases[:, None] * raw)))
    assert phase_error < 1e-14
    groups = {k: [i for i, label in enumerate(trajectories) if label["k"] == k] for k in (0, 1, 2)}
    result = {"indexing": "zero-based original input row numbers; intermediate rows are rotated mixtures",
              "initial_shape": list(A0.shape), "input_rows_match_up_to_global_phase_error": phase_error,
              "initial_ranks_by_jump_order": {str(k): int(np.linalg.matrix_rank(A0[idx], tol=1e-12)) for k, idx in groups.items()},
              "initial_full_rank": int(np.linalg.matrix_rank(A0, tol=1e-12)), "methods": {}}
    for method in ("Thermal2_HO_support_rank", "Thermal2_HO_gatecost_neutral"):
        steps = json.loads((PREVIOUS / method / "rewrite_steps.json").read_text())
        A = A0.copy()
        transform = np.eye(len(A), dtype=complex)
        ancestry = [{i} for i in range(len(A))]
        events, rotations = [], []
        cross_orders = 0
        for index, step in enumerate(steps, 1):
            i, j = step["pair"]
            U = np.array(step["U_real"]) + 1j * np.array(step["U_imag"])
            before = np.count_nonzero(np.abs(A[[i, j]]) > 1e-12, axis=1)
            origins = sorted(ancestry[i] | ancestry[j])
            order_set = sorted({trajectories[q]["k"] for q in origins})
            cross_orders += int(len(order_set) > 1)
            before_pair = A[[i, j]].copy()
            A[[i, j]] = np.stack([U[0, 0] * A[i] + U[0, 1] * A[j], U[1, 0] * A[i] + U[1, 1] * A[j]])
            A[np.abs(A) < 1e-12] = 0
            transform[[i, j]] = U @ transform[[i, j]]
            ancestry[i] = set(origins)
            ancestry[j] = set(origins)
            after = np.count_nonzero(np.abs(A[[i, j]]) > 1e-12, axis=1)
            entry = {"step": index, "pair_slots": [i, j], "before_supports": before.tolist(), "after_supports": after.tolist(),
                     "support_drop": int(before.sum() - after.sum()), "kraus_drop": int(np.count_nonzero(before) - np.count_nonzero(after)),
                     "original_jump_orders": order_set, "number_of_original_rows_involved": len(origins)}
            rotations.append(entry)
            if entry["kraus_drop"]:
                eliminated = [q for q, s in zip((i, j), after) if not s]
                survivors = [q for q, s in zip((i, j), after) if s]
                entry = dict(entry, eliminated_slots=eliminated, surviving_slots=survivors,
                    original_trajectory_indices=origins,
                    original_jump_words=sorted({tuple(trajectories[q]["jump_indices"]) for q in origins}),
                    pair_relative_proportionality_residual=float(np.linalg.svd(before_pair, compute_uv=False)[-1]
                            / max(np.linalg.norm(before_pair), 1e-30)))
                events.append(entry)
        saved = np.load(PREVIOUS / method / "coefficients.npz")["A_final"]
        assert np.max(np.abs(A - saved)) < 1e-14
        summaries = {}
        for k, idx in groups.items():
            final_rows = [i for i in idx if np.any(np.abs(A[i]) > 1e-12)]
            summaries[str(k)] = {"initial_rows": len(idx), "initial_support": int(np.count_nonzero(np.abs(A0[idx]) > 1e-12)),
                "final_rows": len(final_rows), "final_support": int(np.count_nonzero(np.abs(A[idx]) > 1e-12)),
                "final_row_slots": final_rows, "final_row_supports": [int(np.count_nonzero(np.abs(A[i]) > 1e-12)) for i in final_rows]}
        singular = np.linalg.svd(A, compute_uv=False)
        nojump_projection_residual = float(np.linalg.norm(A0[0] - A0[0] @ np.linalg.pinv(A0[groups[2]]) @ A0[groups[2]]) / np.linalg.norm(A0[0]))
        result["methods"][method] = {"replay_matches_saved": True, "cross_jump_order_rotations": cross_orders,
            "total_rotations": len(rotations), "zero_row_eliminations": len(events),
            "eliminations_by_original_order": dict(Counter(str(e["original_jump_orders"]) for e in events)),
            "eliminations_by_pair_support_before": dict(Counter(str(e["before_supports"]) for e in events)),
            "by_jump_order": summaries, "nojump_ever_touched": any(0 in s["pair_slots"] for s in rotations),
            "nojump_in_initial_double_jump_span_relative_residual": nojump_projection_residual,
            "full_rank_1e12": int(np.count_nonzero(singular > 1e-12)), "singular_values": singular.tolist(),
            "largest_support_drop_rotations": sorted(rotations, key=lambda e: -e["support_drop"])[:12],
            "elimination_events": events}
        np.savez_compressed(HERE / (method + "_lineage.npz"), transform=transform, A_initial=A0, A_final=A, labels=labels)
    support = np.load(HERE / "Thermal2_HO_support_rank_lineage.npz")
    gate = np.load(HERE / "Thermal2_HO_gatecost_neutral_lineage.npz")
    conversion = support["transform"] @ gate["transform"].conj().T
    result["gate_to_support_equivalence"] = {
        "convention": "pad both coefficient matrices to 73 rows; U = U_support @ U_gate.conj().T",
        "unitarity_frobenius_error": float(np.linalg.norm(conversion @ conversion.conj().T - np.eye(73))),
        "coefficient_relative_error": float(np.linalg.norm(conversion @ gate["A_final"] - support["A_final"])
                                            / np.linalg.norm(support["A_final"])),
        "zero_threshold": 1e-12,
    }
    result["trajectories"] = trajectories
    dump(HERE / "thermal_lineage.json", result)
    print(json.dumps({"input_rank": result["initial_full_rank"], "order_ranks": result["initial_ranks_by_jump_order"],
        "methods": {k: {x: v[x] for x in ["cross_jump_order_rotations", "zero_row_eliminations", "by_jump_order",
            "eliminations_by_pair_support_before", "nojump_ever_touched", "nojump_in_initial_double_jump_span_relative_residual",
            "largest_support_drop_rotations"]} for k, v in result["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
