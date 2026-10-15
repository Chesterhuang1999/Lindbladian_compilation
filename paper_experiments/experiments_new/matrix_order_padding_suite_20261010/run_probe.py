#!/usr/bin/env python3
"""Audit random-padding activation and complete the five-seed N5 comparison.

Uses the immutable source and in-process adapters from the preceding probe.
Production source and earlier experiment records are never modified.
"""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from functools import wraps
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = OUT.parent / "matrix_order_padding_three_seeds_20261010"
INPUTS = OUT.parent / "matrix_order_p0_singlepass_20261010"
SNAPSHOT = PREVIOUS / "source_snapshot/src"
CASES = [f"RandCh-N5-4q-seed{seed}" for seed in range(5)]
VARIANTS = {
    "single_seed": (True, [20260609]),
    "three_seeds": (True, [20260609, 20260610, 20260611]),
    "no_random_padding": (False, [20260609]),
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_activation(bm, rp):
    """Capture the real backend's search arguments without running its search."""
    original = bm.search_for_pauli_placement
    rows = []

    class Captured(Exception):
        pass

    @wraps(original)
    def capture(matrixsum, **kwargs):
        raise Captured(kwargs)

    bm.search_for_pauli_placement = capture
    try:
        for case in json.loads((INPUTS / "settings.json").read_text())["cases"]:
            source = INPUTS / case / "run-1.json"
            record = json.loads(source.read_text())
            assert record["status"] == "verified"
            blocks = []
            for index, ms in enumerate(rp.deserialize_kraus(record["assembly"]["compiled_kraus"])):
                if len(ms.instances) == 1:
                    blocks.append({"index": index, "terms": 1, "random_padding_requested": False,
                                   "path": "single-term"})
                    continue
                block = bm.BlockEncoding(ms, placement_options={},
                    matrix_order_options={"random_padding": True, "phase_aware_candidates": False})
                try:
                    block.find_optimal_order_matrices()
                except Captured as captured:
                    parameters = captured.args[0]
                else:
                    raise AssertionError("Search arguments were not captured")
                blocks.append({"index": index, "terms": len(block.J.instances),
                               "random_padding_requested": parameters["padding_to_full_address"],
                               "allow_zero_address": parameters["allow_zero_address"],
                               "address_width": parameters["m"], "search_arguments": parameters})
            rows.append({"case": case, "compiled_input_source": str(source),
                         "source_sha256": sha256(source.read_bytes()).hexdigest(),
                         "random_padding_blocks": [b["index"] for b in blocks if b["random_padding_requested"]],
                         "blocks": blocks})
    finally:
        bm.search_for_pauli_placement = original
    dump(OUT / "activation_audit.json", rows)
    print(json.dumps({"stage": "activation-audit", "cases": len(rows),
                      "random_padding_cases": [r["case"] for r in rows if r["random_padding_blocks"]]}), flush=True)
    return rows


def recorded_block_class(bm, padding_seeds, blocks):
    """Reuse exactly the previous probe's multi-sample candidate assembly."""
    source = (PREVIOUS / "run_probe.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "run")
    node = next(node for node in function.body
                if isinstance(node, ast.ClassDef) and node.name == "RecordedBlock")
    namespace = {"bm": bm, "padding_seeds": padding_seeds, "blocks": blocks,
                 "deepcopy": deepcopy, "replace": replace, "time": time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PREVIOUS / "run_probe.py"), "exec"), namespace)
    return namespace["RecordedBlock"]


def run():
    import block_encoding as bm
    import placement_search as ps
    import rq1_pipeline as rp
    from experiment_protocol import REWRITE_DEFAULTS, transpile_channel_circuit
    from matrix_order_pipeline import options_dict, resolve_options
    from qiskit import qasm2

    manifest = json.loads((PREVIOUS / "source_snapshot/manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert sha256((SNAPSHOT / name).read_bytes()).hexdigest() == digest
        assert sha256((ROOT / "src" / name).read_bytes()).hexdigest() == digest
    assert not (OUT / "settings.json").exists(), "Do not overwrite or repeat an existing measurement"
    previous = load_module("padding_previous_probe", PREVIOUS / "run_probe.py")
    checks = load_module("padding_validation", PREVIOUS / "source_snapshot/reference_probe.py")
    adapter_records = []
    previous.dump = lambda path, data: adapter_records.append(data)
    previous.install_historical_prefix()
    dump(OUT / "adapter.json", adapter_records)
    dump(OUT / "settings.json", {
        "cases": CASES, "variants": VARIANTS, "repetitions_per_input_and_variant": 1,
        "warmups": 0, "P02_enabled": False, "phase_aware_candidates": False,
        "refinement_writeback_consistency": "retained",
        "established_matrix_order_options": options_dict(resolve_options()),
        "pipeline": "canonicalization + gate-cost beam rewrite + tree/Fstar + Matrix-order",
        "rewrite": asdict(REWRITE_DEFAULTS), "basis": ["rz", "h", "x", "cx"],
        "optimization_level": 0, "seed_transpiler": 20261009, "qubits_initially_zero": False,
        "source_manifest": str(PREVIOUS / "source_snapshot/manifest.json"),
        "source_manifest_sha256": sha256((PREVIOUS / "source_snapshot/manifest.json").read_bytes()).hexdigest(),
        "shared_padding_runner": str(PREVIOUS / "run_probe.py"),
        "shared_padding_runner_sha256": sha256((PREVIOUS / "run_probe.py").read_bytes()).hexdigest(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": "canonicalization through final NAM opt0 lowering; exclude input I/O, imports and verification",
        "N9_evidence": [str(PREVIOUS), str(OUT.parent / "matrix_order_padding_only_20261010")],
    })
    with (OUT / "activation_audit.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        audit = audit_activation(bm, rp)
    print(json.dumps({"stage": "audit-complete", "cases": len(audit),
                      "cases_requesting_random_padding": sum(bool(r["random_padding_blocks"]) for r in audit)}), flush=True)
    all_results = []
    fingerprints = {}
    for variant, (enabled, seeds) in VARIANTS.items():
        blocks = []
        rp.BlockEncoding = recorded_block_class(bm, seeds, blocks)
        options = {"random_padding": enabled, "phase_aware_candidates": False}
        for case in CASES:
            directory = OUT / variant / case
            directory.mkdir(parents=True, exist_ok=False)
            input_file = INPUTS / case / "input.json"
            input_record = json.loads(input_file.read_text())
            raw = rp.deserialize_kraus(input_record["input_kraus"])
            reference = json.loads((INPUTS / case / "run-1.json").read_text())
            blocks.clear()
            identifier = f"{variant}/{case}"
            ps._P02_DISABLED_CASE = identifier
            print(json.dumps({"case": case, "variant": variant, "stage": "started"}), flush=True)
            with (directory / "compilation.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                started = time.perf_counter()
                canonical, canonicalization = rp.canonicalize(raw)
                circuit, registers, assembly = rp.build_pipeline(canonical, "Ours", matrix_order_options=options)
                lowering_started = time.perf_counter()
                lowered = transpile_channel_circuit(circuit, basis_gates=["rz", "h", "x", "cx"],
                    optimization_level=0, seed_transpiler=20261009)
                ended = time.perf_counter()
            counts = dict(lowered.count_ops())
            assert set(counts) <= {"rz", "h", "x", "cx"}
            rewritten = rp.deserialize_kraus(assembly["compiled_kraus"])
            validation = {"rewrite": checks.coefficient_check(raw, rewritten),
                          "matches_reference_rewrite": checks.coefficient_check(
                              rp.deserialize_kraus(reference["assembly"]["compiled_kraus"]), rewritten)}
            assert validation["matches_reference_rewrite"]["ordered_coefficient_max_error"] < 1e-10
            assert abs(circuit.metadata["subnormalization_factor"] - reference["Lambda"]) < 1e-10
            assert len(blocks) == len(circuit.metadata["block_metadata"])
            validation["selected_residuals"] = [
                checks.check_residuals(block, meta.get("matrix_order", {}).get("selected_candidate"))
                for block, meta in zip(blocks, circuit.metadata["block_metadata"])]
            validation["padding_trial_residuals"] = [
                [{"seed": trial["seed"], **checks.check_residuals(block, trial["candidate"])}
                 for trial in block.padding_trials] for block in blocks]
            if not enabled:
                assert all(not getattr(b, "placement_search_debug", {}).get("padding_count", 0) for b in blocks)
            else:
                assert [t["seed"] for t in blocks[0].padding_trials] == seeds
                assert all(t["debug"]["padding_count"] == 5 for t in blocks[0].padding_trials)
            qasm_file = directory / "channel_nam_opt0.qasm"
            qasm2.dump(lowered, qasm_file)
            fingerprints[identifier] = sha256(qasm_file.read_bytes()).hexdigest()
            if variant == "three_seeds":
                same = fingerprints[identifier] == fingerprints[f"single_seed/{case}"]
                validation["identical_to_single_seed_qasm"] = same
            row = {"case": case, "variant": variant, "status": "verified",
                   "G": sum(counts.values()), "W": lowered.num_qubits, "counts": counts,
                   "Lambda": circuit.metadata["subnormalization_factor"],
                   "compile_time_s": ended - started, "rewrite_s": assembly["rewrite"]["time_s"],
                   "assembly_s": assembly["assembly_time_s"], "lowering_s": ended - lowering_started,
                   "input_source": str(input_file), "input_sha256": sha256(input_file.read_bytes()).hexdigest(),
                   "qasm_sha256": fingerprints[identifier], "registers": registers,
                   "canonicalization": canonicalization, "assembly": assembly, "metadata": circuit.metadata,
                   "padding": [getattr(block, "placement_search_debug", {}) for block in blocks],
                   "padding_trials": [block.padding_trials for block in blocks],
                   "prefix_cost_diagnostics": [event for event in ps._P02_DISABLED_COST_EVENTS if event["case"] == identifier],
                   "validation": validation,
                   "verification_scope": "coefficient Gram and phase-aware residuals on real addresses; no new full statevector simulation"}
            dump(directory / "run-1.json", row)
            summary = {key: row[key] for key in ("case", "variant", "G", "W", "compile_time_s", "rewrite_s", "assembly_s")}
            summary["first_block_candidates"] = row["metadata"]["block_metadata"][0]["matrix_order"]["candidates"]
            all_results.append(summary)
            dump(OUT / "measurements.json", all_results)
            print(json.dumps({"stage": "verified", **summary}), flush=True)
    dump(OUT / "completion.json", {"passed": True, "audit_cases": len(audit), "new_compilations": len(all_results),
                                   "qasm_equal_for_single_and_three_seeds": all(
                                       fingerprints[f"single_seed/{case}"] == fingerprints[f"three_seeds/{case}"] for case in CASES)})


if __name__ == "__main__":
    sys.path.insert(0, str(SNAPSHOT))
    with tempfile.TemporaryDirectory(prefix="padding-suite-cache-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["NUMBA_CACHE_DIR"] = cache
        run()
