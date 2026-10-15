#!/usr/bin/env python3
"""Single-run comparisons of phase candidates and nonrandom partial placement."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict
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
VARIANTS = {
    "phase_expansion": {"phase_aware_candidates": True, "random_padding": True},
    "no_random_padding": {"phase_aware_candidates": False, "random_padding": False},
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
    record = json.loads(manifest.read_text())
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


def run(variant, manifest):
    import block_encoding as bm
    import rq1_pipeline as rp
    from experiment_protocol import REWRITE_DEFAULTS, transpile_channel_circuit
    from importlib.metadata import version
    from matrix_order_pipeline import options_dict, resolve_options
    from qiskit import qasm2

    checks = load_validation()
    parameters = VARIANTS[variant]
    options = options_dict(resolve_options(parameters))
    directory = OUT / variant
    assert not directory.exists(), "Each variant must be measured only once"
    dump(directory / "settings.json", {
        "cases": CASES, "repetitions": 1, "warmups": 0,
        "variant": variant, "matrix_order_options": options,
        "pipeline": "Ours: canonicalization + rewrite + tree/Fstar + Matrix-order",
        "rewrite": asdict(REWRITE_DEFAULTS), "basis": ["rz", "h", "x", "cx"],
        "optimization_level": 0, "seed_transpiler": 20261009, "qubits_initially_zero": False,
        "P02_and_refinement_consistency": "retained in both variants",
        "structural_subgroup_completion": "unchanged in both variants",
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
            super().__init__(*args, **kwargs)
            blocks.append(self)

    rp.BlockEncoding = RecordedBlock
    completed, failures = [], []
    for case in CASES:
        case_directory = directory / case
        input_record = json.loads((REFERENCE / case / "input.json").read_text())
        base = json.loads((REFERENCE / case / "run-1.json").read_text())
        raw = rp.deserialize_kraus(input_record["input_kraus"])
        dump(case_directory / "input.json", input_record)
        blocks.clear()
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
                   "padding": [getattr(block, "placement_search_debug", {}) for block in blocks]}
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
