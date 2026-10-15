#!/usr/bin/env python3
"""Single-pass complete-Ours experiment using an immutable source snapshot."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
import tempfile
import time

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
REFERENCE = REPO / "paper_experiments/pending_review/table13_matrix_order_qrisp_20261010"
sys.path[:0] = [str(OUT / "source_snapshot/src"), str(OUT / "source_snapshot/scripts/eval_redesign")]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def preflight():
    import numpy as np
    from qiskit.quantum_info import Operator, Pauli
    from channel_IR import Matrixsum, PauliAtom
    import p0_backend as patch
    import placement_search as ps
    import matrix_order_pipeline as mp
    import block_encoding as bm
    patch.install()
    rows = []
    for width in range(4):
        for phase in (1.0, 2.0, -1.0, 0.37):
            residuals = [("II", "1" * width, phase)]
            generic = mp.symbolic_select(residuals, width, 2)
            special = patch.phase_select(residuals, width, 2)
            for outer in (0, 1):
                expected = generic.control(outer) if outer else generic
                actual = special.control(outer) if outer else special
                error = float(np.max(np.abs(Operator(expected).data - Operator(actual).data)))
                assert error < 1e-10
                lowered, _ = mp.lower_select(actual, mp.resolve_options(), outer_controls=outer)
                original_dimension = 1 << actual.num_qubits
                U = Operator(lowered).data
                target = Operator(expected).data
                lowered_error = float(np.max(np.abs(U[:original_dimension, :original_dimension] - target)))
                leakage = float(np.max(np.abs(U[original_dimension:, :original_dimension]))) if len(U) > original_dimension else 0.0
                assert max(lowered_error, leakage) < 1e-10
                rows.append({"width": width, "phase_pi_over_2": phase, "outer": outer,
                             "symbolic_error": error, "lowered_error": lowered_error,
                             "scratch_leakage": leakage})
    vectors = [ps._label_to_bits(x) for x in ("ZI", "IZ", "ZZ")]
    supersets = [ps._strict_supersets(i, 3) for i in range(8)]
    empty = ps._SearchState((0, 1, 2), (0,) * 8, 0, ())
    simultaneous = patch.corrected_apply(empty, ((0, 1), (1, 2), (2, 3)), vectors, 2, supersets)
    high_first = patch.corrected_apply(empty, ((2, 3),), vectors, 2, supersets)
    filled = patch.corrected_apply(high_first, ((0, 1), (1, 2)), vectors, 2, supersets)
    assert simultaneous.cost == filled.cost == 2
    assert simultaneous.prefix == filled.prefix
    singletons = Matrixsum([(PauliAtom(x), 1.0) for x in ("XIIII", "IXIII", "IIXII", "IIIXI", "IIIIX")])
    with redirect_stdout(sys.stderr):
        block = bm.BlockEncoding(singletons, placement_options={})
        block.find_optimal_order_matrices()
    assert block.placement_search_debug["padding_count"] == 0
    assert block.placement_search_debug["padded_placement"] == block.additional_modes
    subgroup = Matrixsum([(PauliAtom(x), 1.0) for x in ("XI", "IX")])
    modes, debug = ps.search_for_pauli_placement(subgroup, m=2, return_debug=True)
    assert debug["shortcut"] == "subgroup-span" and debug["padding_count"] == 1
    from rq1_pipeline import deserialize_kraus
    raw = deserialize_kraus(json.loads((REFERENCE / "HRW-4/input.json").read_text())["input_kraus"])
    block = bm.BlockEncoding(raw[0], placement_options={})
    parts = block.circuit_components("Matrix-order").with_outer_control(1)
    qc = parts.controlled(1)
    U = Operator(qc).data
    system = [q + 1 for q in parts.circuit.metadata["system_qubits"]]
    indices = [sum(((s >> i) & 1) << q for i, q in enumerate(system)) for s in range(16)]
    active = [i + 1 for i in indices]
    target = sum(c * a.phase * Pauli(a.expr).to_matrix() for a, c in raw[0].instances) / raw[0].pauli_norm()
    active_error = float(np.max(abs(U[np.ix_(active, active)] - target)))
    expected_inactive = np.zeros((len(U), 16), complex)
    expected_inactive[indices, np.arange(16)] = 1
    inactive_error = float(np.max(abs(U[:, indices] - expected_inactive)))
    assert max(active_error, inactive_error) < 1e-10
    symbolic = patch.verify_selected_residuals(block, parts.circuit.metadata["matrix_order"]["selected_candidate"])
    result = {"passed": True, "pure_phase_cases": rows,
              "axis_prefix": {"simultaneous_cost": simultaneous.cost, "late_lower_address_cost": filled.cost},
              "random_padding_removed": True, "structural_padding_retained": True,
              "HRW4_unrewritten_block": {"G_Fstar": mp.nam_cost(qc), "active_error": active_error,
                    "inactive_error": inactive_error, "symbolic": symbolic,
                    "selected": parts.circuit.metadata["matrix_order"]["selected_candidate"]},
              "patch_sha256": sha256((OUT / "p0_backend.py").read_bytes()).hexdigest()}
    dump(OUT / "preflight.json", result)
    print(json.dumps({"preflight_passed": True, "phase_tests": len(rows),
                      "HRW4_Fstar_G": result["HRW4_unrewritten_block"]["G_Fstar"]}), flush=True)


def coefficient_comparison(left, right):
    import numpy as np
    labels = sorted({a.expr for rows in (left, right) for ms in rows for a, _ in ms.instances})
    index = {label: i for i, label in enumerate(labels)}
    def matrix(rows):
        A = np.zeros((len(rows), len(labels)), complex)
        for i, ms in enumerate(rows):
            for atom, c in ms.instances:
                A[i, index[atom.expr]] += c * atom.phase
        return A
    A, B = matrix(left), matrix(right)
    gram = A.conj().T @ A
    error = float(np.linalg.norm(gram - B.conj().T @ B) / max(np.linalg.norm(gram), 1e-30))
    same_rows = A.shape == B.shape
    ordered_error = float(np.max(np.abs(A - B))) if same_rows else None
    return {"coefficient_gram_relative_error": error, "same_ordered_shape": same_rows,
            "ordered_coefficient_max_error": ordered_error}


def summarize(rows):
    fields = ("case", "G_before", "G_after", "gates_saved", "W_before", "W_after",
              "compile_time_s", "rewrite_time_s", "previous_compile_median_s",
              "previous_rewrite_median_s", "matches_previous_rewritten_kraus", "verification")
    compact = [{key: row[key] for key in fields} for row in rows]
    dump(OUT / "comparison.json", compact)
    with (OUT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(compact)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"].split("-seed")[0], []).append(row)
    table = []
    for case, values in grouped.items():
        table.append({"benchmark": case, "inputs": len(values),
                      "G_before": mean(r["G_before"] for r in values),
                      "G_after": mean(r["G_after"] for r in values),
                      "W_before": sorted(set(r["W_before"] for r in values)),
                      "W_after": sorted(set(r["W_after"] for r in values)),
                      "compile_time_s": mean(r["compile_time_s"] for r in values),
                      "rewrite_time_s": mean(r["rewrite_time_s"] for r in values)})
    dump(OUT / "comparison_table.json", table)


def run():
    import numpy as np
    from qiskit import qasm2
    from dataclasses import asdict
    from importlib.metadata import version
    import p0_backend as patch
    import block_encoding as bm
    import rq1_pipeline as rp
    from channel_IR import channel_ensemble
    from channel_LCU import channel_to_LCU
    from experiment_protocol import transpile_channel_circuit, REWRITE_DEFAULTS
    from matrix_order_pipeline import options_dict, resolve_options
    from probe_hrw_rewrite import paired_support_shortcut
    pre = json.loads((OUT / "preflight.json").read_text())
    assert pre["passed"] and pre["patch_sha256"] == sha256((OUT / "p0_backend.py").read_bytes()).hexdigest()
    assert not (OUT / "settings.json").exists(), "A measured run already exists; do not repeat it"
    patch.install()
    cases = json.loads((REFERENCE / "settings.json").read_text())["cases"]
    settings = {"patch_version": patch.PATCH_VERSION, "cases": cases, "repetitions": 1,
                "warmups": 0, "source_snapshot": json.loads((OUT / "source_snapshot/manifest.json").read_text()),
                "matrix_order_options": options_dict(resolve_options()), "rewrite": asdict(REWRITE_DEFAULTS),
                "HRW_rewrite": "paired_support_shortcut; validate coefficient Gram and old beam output",
                "address_permutations": ["reverse", "swap lowest two bits"],
                "padding": "no forced random padding; retain subgroup structural completion",
                "basis": ["rz", "h", "x", "cx"], "opt": 0, "qubits_initially_zero": False,
                "seed_transpiler": 20261009, "reference": str(REFERENCE),
                "timing": "single sample; canonicalization, rewrite, assembly, NAM lowering; exclude imports/I/O/verification",
                "comparison_caveat": "previous times are warmed medians; rewrite cost model has since changed",
                "versions": {name: version(name) for name in ("qiskit", "qiskit-aer", "numpy")},
                "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "patch_sha256": pre["patch_sha256"]}
    dump(OUT / "settings.json", settings)
    objects = []
    class TrackedBlock(bm.BlockEncoding):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); objects.append(self)
    rp.BlockEncoding = TrackedBlock
    rows = []
    failures = []
    for case in cases:
        case_dir = OUT / case
        input_record = json.loads((REFERENCE / case / "input.json").read_text())
        dump(case_dir / "input.json", input_record)
        previous = [json.loads((REFERENCE / case / "ours" / f"run-{i}.json").read_text()) for i in (1, 2, 3)]
        baseline = previous[0]
        raw = rp.deserialize_kraus(input_record["input_kraus"])
        objects.clear()
        patch.STATE_STATS.update(rebuilt=0, fast=0)
        print(json.dumps({"case": case, "stage": "started"}), flush=True)
        try:
            with (case_dir / "compilation.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                started = time.perf_counter()
                canonical, canonical_info = rp.canonicalize(raw)
                if case.startswith("HRW-"):
                    rewrite_started = time.perf_counter()
                    rewritten, rewrite = paired_support_shortcut(canonical)
                    rewrite_s = time.perf_counter() - rewrite_started
                    assert rewrite["unmatched_rows"] == 0
                    assembly_started = time.perf_counter()
                    circuit, registers = channel_to_LCU(channel_ensemble([deepcopy(rewritten)]),
                        structure="tree", opt="Matrix-order", fstar=True,
                        block_encoding_cls=lambda ms: TrackedBlock(ms, placement_options={}, matrix_order_options={}))
                    assembly = {"rewrite": {**rewrite, "time_s": rewrite_s},
                                "compiled_kraus": rp.serialize_kraus(rewritten),
                                "assembly_time_s": time.perf_counter() - assembly_started}
                else:
                    circuit, registers, assembly = rp.build_pipeline(canonical, "Ours", matrix_order_options={})
                    rewritten = rp.deserialize_kraus(assembly["compiled_kraus"])
                    rewrite_s = assembly["rewrite"]["time_s"]
                lowering_started = time.perf_counter()
                lowered = transpile_channel_circuit(circuit, basis_gates=["rz", "h", "x", "cx"],
                    optimization_level=0, seed_transpiler=20261009)
                ended = time.perf_counter()
            counts = {str(k): int(v) for k, v in lowered.count_ops().items()}
            assert not (set(counts) - {"rz", "h", "x", "cx"})
            original_kraus = rp.deserialize_kraus(baseline["assembly"]["compiled_kraus"])
            old_match = coefficient_comparison(original_kraus, rewritten)
            same_rewrite = old_match["ordered_coefficient_max_error"] is not None and old_match["ordered_coefficient_max_error"] < 1e-10
            row = {"case": case, "status": "compiled", "repetition": 1,
                   "G_before": baseline["G"], "G_after": sum(counts.values()),
                   "gates_saved": baseline["G"] - sum(counts.values()),
                   "W_before": baseline["W"], "W_after": lowered.num_qubits, "counts": counts,
                   "Lambda": float(circuit.metadata["subnormalization_factor"]),
                   "compile_time_s": ended - started, "rewrite_time_s": rewrite_s,
                   "assembly_time_s": assembly["assembly_time_s"], "lowering_time_s": ended - lowering_started,
                   "previous_compile_median_s": median(r["compile_time_s"] for r in previous),
                   "previous_rewrite_median_s": median(r["rewrite_time_s"] for r in previous),
                   "matches_previous_rewritten_kraus": same_rewrite,
                   "old_rewrite_comparison": old_match, "assembly": assembly,
                   "canonicalization": canonical_info, "registers": registers,
                   "metadata": circuit.metadata, "state_updates": dict(patch.STATE_STATS),
                   "padding_counts": [getattr(o, "placement_search_debug", {}).get("padding_count", 0) for o in objects]}
            dump(case_dir / "run-1.json", row)
            qasm2.dump(lowered, case_dir / "channel_nam_opt0.qasm")
            print(json.dumps({"case": case, "stage": "compiled", **{k: row[k] for k in
                ("G_before", "G_after", "W_before", "W_after", "compile_time_s", "rewrite_time_s")}}), flush=True)
            rewrite_check = coefficient_comparison(raw, rewritten)
            assert rewrite_check["coefficient_gram_relative_error"] < 1e-10
            symbolic_checks = []
            for obj, meta in zip(objects, circuit.metadata["block_metadata"]):
                selected = meta.get("matrix_order", {}).get("selected_candidate")
                symbolic_checks.append(patch.verify_selected_residuals(obj, selected))
            validation = {"rewrite": rewrite_check, "selected_residuals": symbolic_checks,
                          "prepare": "Qiskit StatePreparation or existing amplitude-checked product preparation",
                          "pure_phase_lowering": "preflight verifies symbolic and lowered controlled operators"}
            if raw[0].size <= 6 and lowered.num_qubits <= 22:
                validation["full_channel"] = rp.verify_channel(lowered, registers, raw)
                assert validation["full_channel"]["passed"]
                row["verification"] = "full Choi + symbolic residuals + coefficient Gram"
            else:
                validation["full_channel"] = {"method": "not simulated at this width",
                    "scope": "symbolic placement on every occupied address and coefficient Gram; not a full lowered-channel simulation"}
                row["verification"] = "symbolic residuals + coefficient Gram; no full statevector"
            row["validation"] = validation
            row["status"] = "verified"
            dump(case_dir / "run-1.json", row)
            rows.append(row)
            summarize(rows)
            print(json.dumps({"case": case, "stage": "verified", "method": row["verification"]}), flush=True)
        except Exception as error:
            import traceback
            failure = {"case": case, "error": str(error), "traceback": traceback.format_exc()}
            failures.append(failure); dump(case_dir / "failure.json", failure)
            print(json.dumps({"case": case, "stage": "failed", "error": str(error)}), flush=True)
    dump(OUT / "completion.json", {"completed": len(rows), "requested": len(cases), "failures": failures,
                                   "passed": len(rows) == len(cases) and not failures})
    assert not failures, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="matrix-p0-cache-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["NUMBA_CACHE_DIR"] = cache
        preflight() if args.preflight else run()


if __name__ == "__main__":
    main()
