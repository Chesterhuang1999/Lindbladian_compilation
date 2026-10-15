#!/usr/bin/env python3
"""Compare one versus three padding samples inside complete Ours compilation."""

import argparse
import ast
import inspect
from contextlib import redirect_stderr, redirect_stdout
import csv
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean
import sys
import tempfile
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REFERENCE = OUT.parent / "matrix_order_p02_only_20261010"
CASES = ["TFIM-8", "RepDQE-d5"] + [f"RandCh-N9-8q-seed{s}" for s in range(5)]
HISTORY = OUT.parent / "matrix_order_p0_singlepass_20261010/source_snapshot/src/placement_search.py"
VARIANTS = {
    "single_seed": {"phase_aware_candidates": False, "random_padding": True},
    "three_seeds": {"phase_aware_candidates": False, "random_padding": True},
}
PADDING_SEEDS = {
    "single_seed": [20260609],
    "three_seeds": [20260609, 20260610, 20260611],
}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def snapshot():
    sources = {p.name: p.read_bytes() for p in sorted((ROOT / "src").glob("*.py"))}
    hashes = {name: sha256(content).hexdigest() for name, content in sources.items()}
    baseline_hashes = json.loads((REFERENCE / "source_snapshot/manifest.json").read_text())["sha256"]
    assert hashes == baseline_hashes, "Source changed since the P0-2 comparison baseline"
    manifest = OUT / "source_snapshot/manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["sha256"] == hashes
    else:
        for name, content in sources.items():
            destination = OUT / "source_snapshot/src" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        validator = (REFERENCE / "run_probe.py").read_bytes()
        (OUT / "source_snapshot/reference_probe.py").write_bytes(validator)
        dump(manifest, {"original_directory": str(ROOT / "src"), "sha256": hashes,
                       "validation_reference_sha256": sha256(validator).hexdigest()})
    historical_copy = OUT / "source_snapshot/historical_placement_search.py"
    if not historical_copy.exists():
        historical_copy.write_bytes(HISTORY.read_bytes())
    assert historical_copy.read_bytes() == HISTORY.read_bytes()
    record = json.loads(manifest.read_text())
    record["historical_prefix_source"] = str(HISTORY)
    record["historical_prefix_source_sha256"] = sha256(historical_copy.read_bytes()).hexdigest()
    dump(manifest, record)
    for name, digest in hashes.items():
        assert sha256((OUT / "source_snapshot/src" / name).read_bytes()).hexdigest() == digest
    assert sha256((OUT / "source_snapshot/reference_probe.py").read_bytes()).hexdigest() == record["validation_reference_sha256"]
    return record


def load_validation():
    spec = importlib.util.spec_from_file_location("reference_probe", OUT / "source_snapshot/reference_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def circuit_comparison(current, previous):
    """Compare native instructions, allowing only serialization roundoff."""
    if current.num_qubits != previous.num_qubits or len(current.data) != len(previous.data):
        return {"equivalent": False, "reason": "width or instruction count differs"}
    maximum = 0.0
    for left, right in zip(current.data, previous.data):
        left_wires = [current.find_bit(q).index for q in left.qubits]
        right_wires = [previous.find_bit(q).index for q in right.qubits]
        if left.operation.name != right.operation.name or left_wires != right_wires:
            return {"equivalent": False, "reason": "gate or wire sequence differs"}
        if len(left.operation.params) != len(right.operation.params):
            return {"equivalent": False, "reason": "parameter arity differs"}
        for a, b in zip(left.operation.params, right.operation.params):
            maximum = max(maximum, abs(float(a) - float(b)))
    return {"equivalent": maximum <= 1e-14, "maximum_parameter_error": maximum,
            "parameter_tolerance": 1e-14, "instruction_count": len(current.data),
            "global_phase": "ignored because a circuit-wide scalar cancels in the channel"}


def summarize():
    columns = ["variant", "case", "G_reference", "G", "W_reference", "W",
               "reference_compile_s", "compile_time_s", "gate_savings_percent", "time_ratio",
               "rewrite_s", "assembly_s", "lowering_s", "verification"]
    rows = []
    for variant in VARIANTS:
        for case in CASES:
            path = OUT / variant / case / "run-1.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["status"] == "verified":
                    rows.append({key: row[key] for key in columns})
    with (OUT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    dump(OUT / "comparison.json", rows)
    grouped = {}
    for row in rows:
        key = (row["variant"], row["case"].split("-seed")[0])
        grouped.setdefault(key, []).append(row)
    table = []
    for (variant, benchmark), values in grouped.items():
        table.append({"variant": variant, "benchmark": benchmark, "inputs": len(values),
                      **{key: mean(v[key] for v in values) for key in columns[2:-1]},
                      "W_values": sorted({v["W"] for v in values})})
    dump(OUT / "comparison_table.json", table)



def install_historical_prefix():
    """Restore historical ranking only within this process, retaining diagnostics."""
    import block_encoding as bm
    import placement_search as ps
    historical = (OUT / "source_snapshot/historical_placement_search.py").read_text()
    node = next(n for n in ast.parse(historical).body
                if isinstance(n, ast.FunctionDef) and n.name == "_apply_layer_assignment")
    function_source = ast.get_source_segment(historical, node)
    exec(compile(function_source, str(HISTORY), "exec"), ps.__dict__)
    # Historical ranking can disagree with reconstructed cost. Record that
    # evidence, as the old implementation did not reject the candidate.
    search_source = inspect.getsource(ps.search_for_pauli_placement)
    needle = '        raise RuntimeError("Placement state cost disagrees with its reconstructed residuals.")'
    assert search_source.count(needle) == 1
    replacement = '''        _P02_DISABLED_COST_EVENTS.append({
            "case": _P02_DISABLED_CASE, "state_cost": best_state.cost,
            "reconstructed_cost": padded_search_cost, "address_width": width,
            "term_count": len(final_placement),
        })'''
    search_source = search_source.replace(needle, replacement)
    ps._P02_DISABLED_COST_EVENTS = []
    ps._P02_DISABLED_CASE = None
    exec(compile(search_source, "<padding-only-legacy-prefix-diagnostics>", "exec"), ps.__dict__)
    bm.search_for_pauli_placement = ps.search_for_pauli_placement
    vectors = [ps._label_to_bits(label) for label in ("ZI", "IZ", "ZZ")]
    supersets = [ps._strict_supersets(address, 3) for address in range(8)]
    empty = ps._SearchState((0, 1, 2), (0,) * 8, 0, ())
    legacy = ps._apply_layer_assignment(empty, ((0, 1), (1, 2), (2, 3)), vectors, 2, supersets)
    assert legacy.cost == 6
    dump(OUT / "adapter.json", {
        "P02_enabled": False, "phase_candidates_enabled": False, "random_padding_enabled": True,
        "refinement_writeback_consistency": "retained independently",
        "historical_function_sha256": sha256(function_source.encode()).hexdigest(),
        "instrumented_search_sha256": sha256(search_source.encode()).hexdigest(),
        "preflight_legacy_prefix_cost": legacy.cost, "correct_reconstructed_cost": 2,
        "cost_mismatch_policy": "record historical ranking discrepancy; validate emitted channel separately",
    })


def run(variant, manifest):
    import block_encoding as bm
    import rq1_pipeline as rp
    import placement_search as ps
    install_historical_prefix()
    from experiment_protocol import REWRITE_DEFAULTS, transpile_channel_circuit
    from importlib.metadata import version
    from matrix_order_pipeline import options_dict, resolve_options
    from qiskit import qasm2

    checks = load_validation()
    parameters = VARIANTS[variant]
    options = options_dict(resolve_options(parameters))
    padding_seeds = PADDING_SEEDS[variant]
    directory = OUT / variant
    assert not directory.exists(), "Each variant must be measured only once"
    dump(directory / "settings.json", {
        "cases": CASES, "repetitions": 1, "warmups": 0,
        "variant": variant, "matrix_order_options": options,
        "padding_seeds": padding_seeds,
        "padding_selection": "compare all sampled placements in the actual Fstar NAM cost context; retain original and Gray candidates",
        "pair_cost_cache": False, "bitcount_weight": False,
        "pipeline": "Ours: canonicalization + rewrite + tree/Fstar + Matrix-order",
        "rewrite": asdict(REWRITE_DEFAULTS), "basis": ["rz", "h", "x", "cx"],
        "optimization_level": 0, "seed_transpiler": 20261009, "qubits_initially_zero": False,
        "P02_enabled": False,
        "refinement_writeback_consistency": "retained independently",
        "historical_prefix_adapter": str(OUT / "adapter.json"),
        "structural_subgroup_completion": "unchanged; only randomly padded blocks receive extra samples",
        "reference": str(REFERENCE), "source_manifest": manifest,
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": "canonicalization through final NAM lowering; exclude imports, input loading, verification and output",
        "versions": {name: version(name) for name in ("qiskit", "numpy")},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "QISKIT_NUM_PROCS")},
    })
    blocks = []

    class RecordedBlock(bm.BlockEncoding):
        def __init__(self, *args, **kwargs):
            kwargs["placement_options"] = {
                **(kwargs.get("placement_options") or {}),
                "padding_random_seed": padding_seeds[0],
            }
            super().__init__(*args, **kwargs)
            self.padding_trials = []
            blocks.append(self)

        def circuit_components(self, opt="No"):
            started = time.perf_counter()
            base_parts = super().circuit_components(opt=opt)
            base_time = time.perf_counter() - started
            debug = getattr(self, "placement_search_debug", {})
            if opt != "Matrix-order" or debug.get("padding_count", 0) == 0:
                return base_parts

            def rename_part(parts, name):
                changed = {}
                for field in ("circuit", "prepare", "select"):
                    circuit = getattr(parts, field).copy()
                    circuit.metadata = deepcopy(circuit.metadata)
                    circuit.metadata["selected_crit"] = name
                    circuit.metadata["matrix_order"]["selected_candidate"] = name
                    changed[field] = circuit
                return replace(parts, **changed)

            def collect_trial(backend, parts, seed, elapsed):
                name = f"placement-search-padding-{seed}"
                candidate = next(item for item in backend._matrix_order_candidates
                                 if item["name"] == "placement-search")
                candidate["name"] = name
                candidate["circuit"].metadata = deepcopy(candidate["circuit"].metadata)
                candidate["circuit"].metadata["matrix_order"]["selected_candidate"] = name
                self.padding_trials.append({
                    "seed": seed, "candidate": name, "standalone_build_s": elapsed,
                    "debug": deepcopy(backend.placement_search_debug),
                })
                candidates = list(parts.placement_candidates)
                changed = []
                for part in candidates:
                    old_name = part.circuit.metadata["matrix_order"]["selected_candidate"]
                    changed.append(rename_part(part, name) if old_name == "placement-search" else part)
                return candidate, changed

            original_selected = self.selected_crit
            _, combined = collect_trial(self, base_parts, padding_seeds[0], base_time)
            if original_selected == "placement-search":
                self.selected_crit = f"placement-search-padding-{padding_seeds[0]}"
            initial_best = next(part for part in combined
                                if part.circuit.metadata["selected_crit"] == self.selected_crit)
            for seed in padding_seeds[1:]:
                trial_started = time.perf_counter()
                trial = bm.BlockEncoding(self.J,
                    placement_options={**self.placement_options, "padding_random_seed": seed},
                    matrix_order_options=self.matrix_order_options)
                parts = trial.circuit_components(opt=opt)
                candidate, trial_parts = collect_trial(
                    trial, parts, seed, time.perf_counter() - trial_started)
                self._matrix_order_candidates.append(candidate)
                selected = next(part for part in trial_parts
                                if part.circuit.metadata["selected_crit"] == candidate["name"])
                # Original and Gray are deterministic and already present.
                combined.append(selected)
            return replace(initial_best, placement_candidates=tuple(combined))

    rp.BlockEncoding = RecordedBlock
    completed, failures = [], []
    for case in CASES:
        case_directory = directory / case
        input_record = json.loads((REFERENCE / case / "input.json").read_text())
        base = json.loads((REFERENCE / case / "run-1.json").read_text())
        raw = rp.deserialize_kraus(input_record["input_kraus"])
        dump(case_directory / "input.json", input_record)
        blocks.clear()
        ps._P02_DISABLED_CASE = case
        print(json.dumps({"variant": variant, "case": case, "stage": "started"}), flush=True)
        try:
            with (case_directory / "compilation.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                started = time.perf_counter()
                canonical, canonicalization = rp.canonicalize(raw)
                circuit, registers, assembly = rp.build_pipeline(canonical, "Ours", matrix_order_options=parameters)
                lower_started = time.perf_counter()
                lowered = transpile_channel_circuit(circuit, basis_gates=["rz", "h", "x", "cx"],
                                                    optimization_level=0, seed_transpiler=20261009)
                ended = time.perf_counter()
            counts = {str(k): int(v) for k, v in lowered.count_ops().items()}
            assert not (set(counts) - {"rz", "h", "x", "cx"})
            assert assembly["matrix_order_options"] == options
            gates = sum(counts.values())
            row = {"variant": variant, "case": case, "status": "compiled", "repetition": 1,
                   "G_reference": base["G_P02"], "G": gates,
                   "W_reference": base["W_P02"], "W": lowered.num_qubits,
                   "reference_compile_s": base["P02_compile_s"], "compile_time_s": ended - started,
                   "gate_savings_percent": 100 * (1 - gates / base["G_P02"]),
                   "time_ratio": (ended - started) / base["P02_compile_s"],
                   "rewrite_s": assembly["rewrite"]["time_s"], "assembly_s": assembly["assembly_time_s"],
                   "lowering_s": ended - lower_started, "counts": counts,
                   "Lambda": circuit.metadata["subnormalization_factor"], "registers": registers,
                   "canonicalization": canonicalization, "assembly": assembly, "metadata": circuit.metadata,
                   "padding": [getattr(block, "placement_search_debug", {}) for block in blocks],
                   "padding_trials": [block.padding_trials for block in blocks],
                   "prefix_cost_diagnostics": [entry for entry in ps._P02_DISABLED_COST_EVENTS if entry["case"] == case]}
            dump(case_directory / "run-1.json", row)
            qasm2.dump(lowered, case_directory / "channel_nam_opt0.qasm")
            print(json.dumps({"case": case, "stage": "compiled", **{key: row[key] for key in
                              ("G", "W", "compile_time_s", "gate_savings_percent", "time_ratio")}}), flush=True)

            rewritten = rp.deserialize_kraus(assembly["compiled_kraus"])
            validation = {"rewrite": checks.coefficient_check(raw, rewritten)}
            rewrite_match = checks.coefficient_check(rp.deserialize_kraus(base["assembly"]["compiled_kraus"]), rewritten)
            assert rewrite_match["ordered_coefficient_max_error"] < 1e-10
            validation["matches_reference_rewrite"] = rewrite_match
            assert abs(row["Lambda"] - base["Lambda"]) < 1e-10
            assert len(blocks) == len(circuit.metadata["block_metadata"])
            residuals, selections, candidate_counts = [], [], []
            for block, metadata in zip(blocks, circuit.metadata["block_metadata"]):
                info = metadata.get("matrix_order", {})
                names = [candidate["candidate"] for candidate in info.get("candidates", [])]
                if not parameters["phase_aware_candidates"]:
                    assert all("/" not in name for name in names)
                residuals.append(checks.check_residuals(block, info.get("selected_candidate")))
                selections.append(info.get("selected_candidate"))
                candidate_counts.append(len(names))
            if not parameters["random_padding"]:
                assert all(p.get("padding_count", 0) == 0 for p in row["padding"])
            validation["selected_residuals"] = residuals
            validation["padding_trial_residuals"] = [
                [{"seed": trial["seed"], **checks.check_residuals(block, trial["candidate"])}
                 for trial in block.padding_trials] for block in blocks]
            for block in blocks:
                if block.padding_trials:
                    assert [trial["seed"] for trial in block.padding_trials] == padding_seeds
            row["selected_candidates"] = selections
            row["candidate_counts"] = candidate_counts

            if case == "RepDQE-d5":
                previous = qasm2.load(REFERENCE / case / "channel_nam_opt0.qasm")
                equality = circuit_comparison(lowered, previous)
                equality["same_registers"] = registers == base["registers"]
                equality["same_Lambda"] = row["Lambda"] == base["Lambda"]
                current_roundtrip = qasm2.load(case_directory / "channel_nam_opt0.qasm")
                equality["serialized_instruction_fingerprint_equal"] = (
                    checks.instruction_fingerprint(current_roundtrip) == checks.instruction_fingerprint(previous))
                validation["reference_circuit_comparison"] = equality
                old_check = base["validation"]["full_channel"]
                if (equality["equivalent"] and equality["same_registers"] and equality["same_Lambda"]
                        and equality["serialized_instruction_fingerprint_equal"] and old_check.get("passed")):
                    validation["full_channel"] = {
                        "method": "reuse prior full Choi after native circuit/ABI/Lambda comparison",
                        "passed": True, "source": str(REFERENCE / case / "run-1.json"),
                        "source_sha256": sha256((REFERENCE / case / "run-1.json").read_bytes()).hexdigest(),
                        "reference_check": old_check, "comparison": equality}
                    row["verification"] = "native circuit matches verified reference; residuals + Gram"
                else:
                    print(json.dumps({"case": case, "stage": "full Choi verification"}), flush=True)
                    validation["full_channel"] = rp.verify_channel(lowered, registers, raw)
                    assert validation["full_channel"]["passed"]
                    row["verification"] = "full Choi + residuals + Gram"
            else:
                validation["full_channel"] = {"method": "not simulated at this width"}
                row["verification"] = "symbolic residuals + Gram; no full lowered-channel simulation"
            row["validation"] = validation
            row["status"] = "verified"
            dump(case_directory / "run-1.json", row)
            completed.append(case)
            summarize()
            print(json.dumps({"case": case, "stage": "verified", "method": row["verification"]}), flush=True)
        except Exception as error:
            import traceback
            failure = {"case": case, "error": str(error), "traceback": traceback.format_exc()}
            failures.append(failure)
            dump(case_directory / "failure.json", failure)
            print(json.dumps({"case": case, "stage": "failed", "error": str(error)}), flush=True)
    dump(directory / "completion.json", {"passed": not failures and completed == CASES,
                                         "requested": len(CASES), "completed": completed, "failures": failures})
    assert not failures, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS)
    args = parser.parse_args()
    manifest = snapshot()
    sys.path.insert(0, str(OUT / "source_snapshot/src"))
    with tempfile.TemporaryDirectory(prefix="matrix-phase-padding-cache-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["NUMBA_CACHE_DIR"] = cache
        run(args.variant, manifest)


if __name__ == "__main__":
    main()


