#!/usr/bin/env python3
"""Supplement the saved random4q SELECT sweep with the native PennyLane baseline.

The sixty saved inputs and both existing methods are retained byte-for-byte.
Only PennyLane graph-work Select circuits are compiled and measured here.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from qiskit import qasm2, qpy, transpile
from qiskit.quantum_info import Operator, Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]

from baseline_common import count_qasm_file
from experiment_protocol import source_identity
from run_rq3_random4q_select import (
    BASES, SETTINGS, TOL, dump, expected_terms, generate, lowering_probes,
    qasm_roundtrip_error, rel,
)
from search_rq3_placement_counterexamples import pl_select
from pennylane_lcu_adapter import build_lcu_tape, compile_lcu_tape, tape_to_qiskit

DEFAULT = ROOT / "Baseline_results/rq3_random4q_20261014"


def sparse_columns(circuit, initial):
    """Simulate all clean-work input columns without a full dense operator.

    Native graph SELECT is a sparse computation even between its relative-phase
    Toffoli gates. Columns remain independent, and complex amplitudes are kept.
    Only numerical cancellation below 1e-13 is dropped; its total L1 magnitude
    is retained as a conservative error bound for each input column.
    """
    column = np.arange(len(initial), dtype=np.int64)
    states = initial.copy()
    amplitudes = np.full(len(initial), np.exp(1j * float(circuit.global_phase)), dtype=complex)
    discarded = np.zeros(len(initial))
    cache = {}
    largest_support = len(initial)
    max_local_error = 0.0
    for instruction in circuit.data:
        op = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        key = (op.name, op.num_qubits, tuple(map(str, op.params)))
        if key not in cache:
            matrix = np.asarray(Operator(op).data)
            output = np.argmax(np.abs(matrix), axis=0)
            values = matrix[output, np.arange(len(output))]
            reference = np.zeros_like(matrix)
            reference[output, np.arange(len(output))] = values
            error = max(float(np.max(np.abs(matrix - reference))),
                        float(np.max(np.abs(np.abs(values) - 1))))
            monomial = len(set(output)) == len(output) and error < 1e-12
            cache[key] = matrix, output, values, monomial
            max_local_error = max(max_local_error, float(np.max(np.abs(matrix.conj().T @ matrix - np.eye(len(matrix))))))
        matrix, output, values, monomial = cache[key]
        local = sum(((states >> q) & 1) << bit for bit, q in enumerate(qubits))
        if monomial:
            amplitudes *= values[local]
            target = output[local]
            for bit, q in enumerate(qubits):
                states &= ~(1 << q)
                states |= ((target >> bit) & 1) << q
            continue
        if op.num_qubits != 1:
            raise ValueError(f"Unexpected non-monomial graph operation: {op.name}")
        q = qubits[0]
        # Column-major copies represent the two possible output values of this
        # one-qubit gate; coalescing sums all paths to the same output state.
        new_states = np.concatenate((states & ~(1 << q), states | (1 << q)))
        new_column = np.tile(column, 2)
        new_amplitudes = np.concatenate((amplitudes * matrix[0, local], amplitudes * matrix[1, local]))
        keys = (new_column << circuit.num_qubits) | new_states
        unique, inverse = np.unique(keys, return_inverse=True)
        merged = np.bincount(inverse, weights=new_amplitudes.real) + 1j * np.bincount(inverse, weights=new_amplitudes.imag)
        keep = np.abs(merged) > 1e-13
        dropped_columns = unique[~keep] >> circuit.num_qubits
        discarded += np.bincount(dropped_columns, weights=np.abs(merged[~keep]), minlength=len(initial))
        unique = unique[keep]
        amplitudes = merged[keep]
        states = unique & ((1 << circuit.num_qubits) - 1)
        column = unique >> circuit.num_qubits
        largest_support = max(largest_support, len(states))
    return column, states, amplitudes, discarded, largest_support, max_local_error


def validate(circuit, inputs, placement, address, system, scratch, require_pass=True):
    _, initial, expected, phases, occupied = expected_terms(inputs, placement, address, system)
    column, states, amplitudes, discarded, max_support, local_error = sparse_columns(circuit, initial)
    correct = states == expected[column]
    actual = np.bincount(column[correct], weights=amplitudes[correct].real, minlength=len(initial))
    actual = actual + 1j * np.bincount(column[correct], weights=amplitudes[correct].imag, minlength=len(initial))
    phase_error = float(np.max(np.abs(actual - phases)))
    leakage = float(np.max(np.bincount(column[~correct], weights=np.abs(amplitudes[~correct]) ** 2, minlength=len(initial))))
    dirty = np.zeros(len(states), dtype=bool)
    for q in scratch:
        dirty |= ((states >> q) & 1).astype(bool)
    dirty_probability = float(np.max(np.bincount(column[dirty], weights=np.abs(amplitudes[dirty]) ** 2, minlength=len(initial))))
    passed = phase_error < TOL and leakage < TOL**2 and dirty_probability < TOL**2 and float(np.max(discarded)) < TOL
    checks = {
        "verification": "sparse complex-amplitude simulation of compiled PennyLane graph SELECT; all logical addresses and all 16 system inputs",
        "all_address_system_basis_columns": len(initial),
        "occupied_address_system_basis_columns": int(np.count_nonzero(occupied)),
        "includes_unused_identity_addresses": True,
        "phase_sensitive": True,
        "phase_max_error": phase_error,
        "max_wrong_output_probability": leakage,
        "max_dirty_scratch_probability": dirty_probability,
        "numerical_cancellation_cutoff": 1e-13,
        "max_discarded_l1_bound_per_column": float(np.max(discarded)),
        "max_intermediate_sparse_entries": max_support,
        "local_unitarity_max_error": local_error,
        "passed": passed,
        "bad_addresses": sorted(set(int(i // 16) for i in np.flatnonzero(np.abs(actual - phases) > TOL))),
    }
    if require_pass:
        assert passed, checks
    return checks, (initial, expected, phases, np.ones(len(initial), dtype=bool))


def valid_pl_select(inputs):
    m = (len(inputs) - 1).bit_length()
    padding = (1 << m) - len(inputs)
    circuit, metadata = pl_select(inputs + [("IIII", 0.0)] * padding)
    metadata["identity_extension"] = {
        "original_terms": len(inputs), "select_branches": 1 << m,
        "padded_identity_branches": padding,
        "unused_branch_coefficient": 0.0,
        "preserves_full_select_unitary": True,
        "reason": "PennyLane 0.44 nonpartial unary closing ladder reverses work triples but not closing control bits; complete identity extension avoids this bug",
    }
    return circuit, metadata


def audit_unary_cleanup(directory, original, fingerprints):
    """Preserve a minimal failing input and an independent dense reproduction."""
    path = directory / "pennylane_unary_cleanup_audit.json"
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached["source_fingerprints"] == fingerprints
        return
    first = original["rows"][0]
    input_terms = json.loads((ROOT / first["input"]).read_text())["inputs"]
    records = []
    for terms in (10, 20):
        inputs = input_terms[:terms]
        m = (terms - 1).bit_length()
        placement = [(label, format(i, f"0{m}b")) for i, (label, _) in enumerate(inputs)]
        for optimize in (False, True):
            tape, wires, metadata = build_lcu_tape(inputs, 4, scope="select", work_wires=m-1)
            expanded, compile_metadata = compile_lcu_tape(tape, wires, optimize=optimize, decomposition="graph")
            circuit = tape_to_qiskit(expanded, wires)
            address = list(range(4 + m - 1, 4 + m - 1 + m))
            scratch = list(range(4, 4 + m - 1))
            checks, columns = validate(circuit, inputs, placement, address, list(range(4)), scratch, require_pass=False)
            assert not checks["passed"]
            artifact = directory / "pennylane_failed_graph" / f"terms{terms}_compile{int(optimize)}"
            artifact.mkdir(parents=True, exist_ok=True)
            with (artifact / "source.qpy").open("wb") as handle:
                qpy.dump(circuit, handle)
            for basis, gates in BASES.items():
                lowered = transpile(circuit, basis_gates=gates, **SETTINGS)
                (artifact / f"{basis}_opt0.qasm").write_text(qasm2.dumps(lowered) + "\n")
                dump(artifact / f"{basis}_opt0.global_phase.json", {"global_phase_radians": float(lowered.global_phase)})
            initial, expected, phases, _ = columns
            failed_address = checks["bad_addresses"][0]
            column = failed_address * 16
            dense = Statevector.from_int(int(initial[column]), 1 << circuit.num_qubits).evolve(circuit).data
            nonzero = np.flatnonzero(np.abs(dense) > 1e-10)
            records.append({"terms": terms, "compile": optimize, "partial": False,
                "work_wires": m-1, "input": inputs, "metadata": metadata, "compile_metadata": compile_metadata,
                "source_validation": checks, "failed_artifacts": rel(artifact),
                "independent_statevector_probe": {"address": failed_address, "system_basis": 0,
                    "input_basis": int(initial[column]), "expected_output_basis": int(expected[column]),
                    "expected_phase_real_imag": [float(phases[column].real), float(phases[column].imag)],
                    "actual_outputs": [{"basis": int(i), "amplitude_real_imag": [float(dense[i].real), float(dense[i].imag)]} for i in nonzero],
                    "expected_amplitude_error": float(abs(dense[expected[column]] - phases[column]))}})
        fixed, meta = valid_pl_select(inputs)
        checks, _ = validate(fixed, inputs, placement, meta["address_qubits"], meta["system_qubits"], meta["clean_scratch_qubits"])
        records.append({"terms": terms, "identity_extended_graph": True,
            "identity_extension": meta["identity_extension"], "source_validation": checks})
    select_path = Path(importlib.import_module("pennylane.templates.subroutines.select").__file__)
    dump(path, {"pennylane_version": version("pennylane"), "source_fingerprints": fingerprints,
        "bug_reproduced": True, "valid_workaround_passed": True,
        "trigger": "closing bits after the first two address bits are not a palindrome; K=10 and K=20 are reproduced",
        "source": str(select_path), "source_sha256": sha256(select_path.read_bytes()).hexdigest(),
        "source_function": "_select_decomp_unary_not_partial",
        "source_expression": "zip(closing_ctrl_bits[2:], reversed(unary_triples[: c - 1]))",
        "compiled_false_and_true_both_fail": True,
        "wire_export_verified_by_independent_statevector": True,
        "records": records})


def run(inputs, path, seed, input_hash, fingerprints):
    started = time.perf_counter()
    circuit, metadata = valid_pl_select(inputs)
    build_s = time.perf_counter() - started
    m = (len(inputs) - 1).bit_length()
    placement = [(label, format(i, f"0{m}b")) for i, (label, _) in enumerate(inputs)]
    checks, columns = validate(circuit, inputs, placement, metadata["address_qubits"],
                               metadata["system_qubits"], metadata["clean_scratch_qubits"])
    path.mkdir(parents=True, exist_ok=True)
    with (path / "source.qpy").open("wb") as handle:
        qpy.dump(circuit, handle)
    dump(path / "circuit_metadata.json", metadata)
    row = {"method": "pennylane", "input_sha256": input_hash,
           "source_fingerprints": fingerprints, "build_s": build_s,
           "source_validation": checks, "metadata": metadata, "basis": {}}
    for basis, gates in BASES.items():
        started = time.perf_counter()
        lowered = transpile(circuit, basis_gates=gates, **SETTINGS)
        seconds = time.perf_counter() - started
        qasm = path / f"{basis}_opt0.qasm"
        qasm.write_text(qasm2.dumps(lowered) + "\n")
        phase_path = qasm.with_suffix(".global_phase.json")
        dump(phase_path, {"global_phase_radians": float(lowered.global_phase)})
        saved = qasm2.load(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        saved.global_phase = float(lowered.global_phase)
        roundtrip_error = qasm_roundtrip_error(lowered, saved)
        validation = lowering_probes(saved, columns, seed)
        counts = count_qasm_file(qasm, gate_set=basis)
        assert set(lowered.count_ops()) <= set(gates)
        assert counts["clifford"]["unknown"] == 0 and roundtrip_error < TOL
        row["basis"][basis] = {"G": int(counts["metric_total"]),
            "NC": int(counts["clifford"]["non_clifford"]), "W": lowered.num_qubits,
            "CX": int(counts["ops"].get("cx", 0)), "ops": counts["ops"],
            "lowering_s": seconds, "validation": validation,
            "qasm_roundtrip_local_operator_error": roundtrip_error,
            "qasm": rel(qasm), "qasm_sha256": sha256(qasm.read_bytes()).hexdigest(),
            "global_phase_radians": float(lowered.global_phase), "phase_sidecar": rel(phase_path)}
    row["passed"] = True
    dump(path / "result.json", row)
    return row


def aggregate(rows):
    output = []
    for terms in (20, 64, 100, 129, 200, 256):
        group = [row for row in rows if row["terms"] == terms]
        if not group:
            continue
        entry = {"terms": terms, "samples": len(group), "basis": {}}
        for basis in BASES:
            entry["basis"][basis] = {}
            for method in ("qrisp", "ours", "pennylane"):
                entry["basis"][basis][method] = {
                    metric: {"mean": statistics.mean(r[method]["basis"][basis][metric] for r in group),
                        "population_std": statistics.pstdev(r[method]["basis"][basis][metric] for r in group),
                        "min": min(r[method]["basis"][basis][metric] for r in group),
                        "max": max(r[method]["basis"][basis][metric] for r in group)}
                    for metric in ("G", "NC", "W")}
            for method in ("qrisp", "pennylane"):
                entry["basis"][basis][f"{method}_over_ours_ratio_of_means"] = {
                    metric: entry["basis"][basis][method][metric]["mean"] / entry["basis"][basis]["ours"][metric]["mean"]
                    for metric in ("G", "NC", "W")}
        output.append(entry)
    return output


def export_csv(directory, data):
    with (directory / "samples_with_pl.csv").open("w", newline="") as handle:
        fields = ["terms", "sample", "seed", "basis", "method", "G", "NC", "W", "CX"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in data["rows"]:
            for basis in BASES:
                for method in ("qrisp", "ours", "pennylane"):
                    writer.writerow({**{key: row[key] for key in ("terms", "sample", "seed")},
                        "basis": basis, "method": method,
                        **{key: row[method]["basis"][basis][key] for key in ("G", "NC", "W", "CX")}})
    with (directory / "aggregate_with_pl.csv").open("w", newline="") as handle:
        fields = ["terms", "samples", "basis", "method", "G_mean", "G_std", "NC_mean", "NC_std", "W_mean", "W_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in data["aggregate"]:
            for basis in BASES:
                for method in ("qrisp", "ours", "pennylane"):
                    result = {"terms": group["terms"], "samples": group["samples"], "basis": basis, "method": method}
                    for metric in ("G", "NC", "W"):
                        result[metric + "_mean"] = group["basis"][basis][method][metric]["mean"]
                        result[metric + "_std"] = group["basis"][basis][method][metric]["population_std"]
                    writer.writerow(result)


def main(directory, selected_terms):
    summary_path = directory / "summary.json"
    original_bytes = summary_path.read_bytes()
    original = json.loads(original_bytes)
    smoke = json.loads((ROOT / original["smoke"]).read_text())
    assert original["passed"] and len(original["rows"]) == 60
    assert smoke["passed"] and source_identity() == smoke["sources"] == original["sources"]
    fingerprints = {rel(ROOT / name): sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("Baseline_scripts/supplement_rq3_random4q_pennylane.py",
                     "Baseline_scripts/search_rq3_placement_counterexamples.py",
                     "Baseline_scripts/pennylane_lcu_adapter.py",
                     "Baseline_scripts/run_rq3_random4q_select.py")}
    select_path = Path(importlib.import_module("pennylane.templates.subroutines.select").__file__)
    fingerprints[str(select_path)] = sha256(select_path.read_bytes()).hexdigest()
    audit_unary_cleanup(directory, original, fingerprints)
    rows = []
    for old in original["rows"]:
        if old["terms"] not in selected_terms:
            continue
        input_path = ROOT / old["input"]
        assert sha256(input_path.read_bytes()).hexdigest() == old["input_sha256"]
        inputs = json.loads(input_path.read_text())["inputs"]
        assert inputs == generate(old["terms"], old["seed"])
        path = input_path.parent / "pennylane"
        cache = path / "result.json"
        if cache.exists():
            result = json.loads(cache.read_text())
            assert result["input_sha256"] == old["input_sha256"] and result["source_fingerprints"] == fingerprints
        else:
            result = run(inputs, path, old["seed"], old["input_sha256"], fingerprints)
        rows.append({**old, "pennylane": result})
        assert source_identity() == original["sources"]
        assert summary_path.read_bytes() == original_bytes
        data = {**original, "experiment": "random4q SELECT sweep with native PennyLane supplementary baseline",
            "scope": "SELECT only; PREPARE and its inverse excluded from all methods",
            "original_summary": rel(summary_path), "original_summary_sha256": sha256(original_bytes).hexdigest(),
            "pl_source_fingerprints": fingerprints,
            "versions": {name: version(name) for name in ("pennylane", "qiskit", "qrisp")},
            "policy": {**original["policy"], "pennylane": "native Select partial=False; explicitly identity-extend unused addresses to 2^m; graph decomposition; work=m-1; qml.compile; identical NAM/IBM lowering",
                "pennylane_cleanup_audit": rel(directory / "pennylane_unary_cleanup_audit.json")},
            "rows": rows, "aggregate": aggregate(rows),
            "passed": all(r[m]["passed"] for r in rows for m in ("qrisp", "ours", "pennylane"))}
        dump(directory / "summary_with_pl.partial.json", data)
        print(json.dumps({"terms": old["terms"], "sample": old["sample"], "seed": old["seed"],
            "pennylane": {b: {k: result["basis"][b][k] for k in ("G", "NC", "W")} for b in BASES},
            "source_phase_error": result["source_validation"]["phase_max_error"]}), flush=True)
    assert len(rows) == 10 * len(selected_terms)
    if len(rows) == 60:
        dump(directory / "summary_with_pl.json", data)
        export_csv(directory, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--terms", nargs="+", type=int, default=[20, 64, 100, 129, 200, 256])
    args = parser.parse_args()
    main(args.directory.resolve(), args.terms)
