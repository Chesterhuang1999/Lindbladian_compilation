"""PennyLane queries under the shared logical-projector QSVT outer circuit.

Recompile Ours for the seven active inputs and the new PL-native cycle4 input.
The six existing PL-native circuits are reused with their original provenance.
Core backends, earlier runners, and frozen inputs are never modified.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]

import numpy as np
import pennylane as qml
import qiskit
from qiskit import QuantumCircuit, qasm2, qpy, transpile
from qiskit.quantum_info import SparsePauliOp

from baseline_common import count_qasm_file
from experiment_protocol import source_identity
from pennylane_lcu_adapter import tape_to_qiskit
from run_rq3_pennylane_qsvt import (
    ATOL, BASIS, SIMULATOR, ADAPTERS, CanonicalBlockQuery, build_adapter,
    columns, compile_official, load_official_wrapper, max_error, polynomial,
    relative, source_record, write_json,
)

ACTIVE = ("resource_three_term", "h2_molecular_api", "hubbard_chain2_api",
          "heisenberg_square4_api", "maxcut_triangle3", "cluster3_weighted", "maxcut_cycle4")
DEFAULT_ROOT = ROOT / "Baseline_results/rq3_common_outer_20261014"
OLD_INPUTS = ROOT / "Baseline_results/rq3_refresh_20261013/qsvt_inputs.json"
OLD_NATIVE_CHECKS = ROOT / "Baseline_results/rq3_refresh_20261013/archived_pl_qsvt_correctness.json"


def official_logical_tape(wrapper, encoding, angles, system_size, address_size, scratch_size):
    """Use the archived H-C1(U)-C0(U.adjoint)-H wrapper with address-only Pi.

    Qiskit order is [system, address, scratch, signal]. PennyLane order is its
    reverse: signal wire 0, scratch 1..r, address r+1..r+a, then system.
    The projector has rank one only on the logical address register, tensor
    identity on the system and on previously verified clean expansion scratch.
    """
    assert len(encoding.wires) == system_size + address_size + scratch_size
    if address_size < 1:
        raise ValueError("The active comparison requires a nonempty logical address register")
    address_wires = list(encoding.wires)[scratch_size:scratch_size + address_size]
    with qml.QueuingManager.stop_recording():
        projectors = [qml.PCPhase(float(phi), dim=1, wires=address_wires) for phi in angles]
    scope = {"qp": qml, "dev": qml.device("default.qubit"),
             "block_encode": encoding, "projectors": projectors}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "official_logical_address_wrapper", "exec"), scope)
    tape = qml.workflow.construct_tape(scope["circuit"])().copy(measurements=[])
    assert [op.name for op in tape.operations] == ["Hadamard", "C(QSVT)", "C(Adjoint(QSVT))", "Hadamard"]
    assert list(tape.operations[1].control_values) == [True]
    assert list(tape.operations[2].control_values) == [False]
    return tape


def logical_source_reference(source, angles, system_size, address_size, scratch_size):
    """Independent full-output reference using address-only projector phases."""
    from qiskit.quantum_info import Statevector
    from run_rq3_pennylane_qsvt import BRIDGE
    assert source.num_qubits == system_size + address_size + scratch_size
    forward = transpile(source, basis_gates=list(BRIDGE), optimization_level=0,
                        seed_transpiler=7, qubits_initially_zero=False)
    inverse = forward.inverse()
    dimension = 1 << system_size
    address_mask = ((1 << address_size) - 1) << system_size
    projected = (np.arange(1 << forward.num_qubits) & address_mask) == 0

    def evolve(vector, circuit):
        return Statevector(np.array(vector, dtype=complex, copy=True)).evolve(circuit).data.copy()

    def project(vector, phi):
        result = vector * np.exp(-1j * phi)
        result[projected] = vector[projected] * np.exp(1j * phi)
        return result

    def branch(vector, adjoint=False):
        if not adjoint:
            for index, phi in enumerate(angles):
                vector = project(vector, phi)
                if index + 1 < len(angles):
                    vector = evolve(vector, forward if index % 2 == 0 else inverse)
        else:
            for index in reversed(range(len(angles))):
                vector = project(vector, -angles[index])
                if index:
                    vector = evolve(vector, inverse if (index - 1) % 2 == 0 else forward)
        return vector

    block, output = [], []
    for column in range(dimension):
        initial = np.zeros(1 << forward.num_qubits, dtype=complex)
        initial[column] = 1
        block.append(evolve(initial, forward))
        positive, negative = branch(initial), branch(initial, True)
        output.append(np.concatenate(((positive + negative) / 2, (negative - positive) / 2)))
    return np.column_stack(block), np.column_stack(output)


def _arbitrary_columns(circuit, input_indices, *, batch_size=64):
    from qiskit.quantum_info import Statevector
    from run_rq3_pennylane_qsvt import BRIDGE
    compiled = transpile(circuit, basis_gates=list(BRIDGE), optimization_level=0, seed_transpiler=7,
                         qubits_initially_zero=False)
    input_indices = list(input_indices)
    output = np.empty((1 << circuit.num_qubits, len(input_indices)), dtype=complex)
    for column, index in enumerate(input_indices):
        initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
        initial[index] = 1
        # Each input owns its storage: phase gates must not mutate a shared column.
        output[:, column] = Statevector(initial.copy()).evolve(compiled).data.copy()
    return output


def validate_query_domain(source, controlled_queries, system_size, address_size, scratch_size):
    """Prove scratch-zero invariance over every address/system basis input.

    controlled_queries maps ctrl_state 0/1 to actual synthesized forward-query
    circuits, already in the shared canonical order with signal highest.
    A square unitary action on the complete scratch-zero subspace establishes
    the same invariant subspace for the adjoint queries.
    """
    block_width = system_size + address_size + scratch_size
    assert source.num_qubits == block_width and set(controlled_queries) == {0, 1}
    domain = 1 << (system_size + address_size)
    original = _arbitrary_columns(source, range(domain))
    block_rows = np.arange(1 << block_width)
    scratch_mask = ((1 << scratch_size) - 1) << (system_size + address_size)
    scratch_bad = (block_rows & scratch_mask) != 0
    scratch_error = float(np.max(np.abs(original[scratch_bad]), initial=0.0))
    unitary_error = max_error(original.conj().T @ original, np.eye(domain))
    results = {}
    signal_offset = 1 << block_width
    full_bad = (np.arange(1 << (block_width + 1)) & scratch_mask) != 0
    for state, controlled in controlled_queries.items():
        assert controlled.num_qubits == block_width + 1
        inactive_signal = 1 - state
        indices = [value + signal * signal_offset for signal in (0, 1) for value in range(domain)]
        actual = _arbitrary_columns(controlled, indices)
        active_columns = actual[:, state * domain:(state + 1) * domain]
        expected_active = np.zeros_like(active_columns)
        expected_active[state * signal_offset:(state + 1) * signal_offset] = original
        inactive_columns = actual[:, inactive_signal * domain:(inactive_signal + 1) * domain]
        expected_inactive = np.zeros_like(inactive_columns)
        expected_inactive[np.arange(domain) + inactive_signal * signal_offset, np.arange(domain)] = 1
        results[str(state)] = {
            "basis_inputs_checked": 2 * domain,
            "active_source_max_error": max_error(active_columns, expected_active),
            "inactive_identity_max_error": max_error(inactive_columns, expected_inactive),
            "clean_scratch_max_error": float(np.max(np.abs(actual[full_bad]), initial=0.0)),
        }
    errors = [scratch_error, unitary_error] + [row[key] for row in results.values()
               for key in ("active_source_max_error", "inactive_identity_max_error", "clean_scratch_max_error")]
    return {"passed": max(errors) < ATOL, "system_basis_inputs": 1 << system_size,
            "all_logical_addresses": 1 << address_size, "source_basis_inputs_checked": domain,
            "controlled_basis_inputs_checked": 4 * domain,
            "source_clean_scratch_max_error": scratch_error,
            "source_domain_unitarity_max_error": unitary_error,
            "controlled": results,
            "adjoint_scratch_invariance": "complete scratch-zero subspace is invariant and source action on it is square unitary",
            "projector_excludes_scratch_authorized_by_validation": max(errors) < ATOL}


def projector_contract(system_size, address_size, scratch_size):
    width = system_size + address_size + scratch_size
    return {"condition": "logical address == 0; identity on system and clean expansion scratch",
            "system_qiskit_bits": list(range(system_size)),
            "address_qiskit_bits": list(range(system_size, system_size + address_size)),
            "scratch_qiskit_bits": list(range(system_size + address_size, width)),
            "signal_qiskit_bit": width,
            "projector_pennylane_wires": list(range(scratch_size + 1, scratch_size + address_size + 1)),
            "qml_PCPhase_dim": 1, "logical_projector_rank": 1 << system_size}


def native_encoding(case):
    n = len(case["terms"][0][0])
    address = (len(case["terms"]) - 1).bit_length()
    system_wires = list(range(address + 1, address + n + 1))
    with qml.QueuingManager.stop_recording():
        ops = []
        for label, _ in case["terms"]:
            factors = [{"X": qml.X, "Y": qml.Y, "Z": qml.Z}[letter](wire)
                       for letter, wire in zip(label, system_wires) if letter != "I"]
            ops.append(qml.Identity(system_wires[0]) if not factors else
                       factors[0] if len(factors) == 1 else qml.prod(*factors))
        hamiltonian = qml.ops.LinearCombination([coefficient for _, coefficient in case["terms"]], ops)
        encoding = qml.PrepSelPrep(hamiltonian, control=list(range(1, address + 1)))
    assert sorted(encoding.wires) == list(range(1, address + n + 1))
    return encoding, n, address


def build_native(case):
    encoding, n, address = native_encoding(case)
    block_wires = list(range(1, n + address + 1))
    source_tape = qml.tape.QuantumScript([encoding])
    source_tape, source_compile = compile_official(source_tape, block_wires)
    source = tape_to_qiskit(source_tape, block_wires)
    controlled = {}
    for state in (0, 1):
        with qml.QueuingManager.stop_recording():
            query = qml.ctrl(encoding, control=0, control_values=[state])
        tape, _ = compile_official(qml.tape.QuantumScript([query]), [0] + block_wires)
        controlled[state] = tape_to_qiskit(tape, [0] + block_wires)
    return source, encoding, controlled, {"backend": "qml.PrepSelPrep", "matrix_synthesis_used": False,
           "address_size": address, "scratch_size": 0, "source_compile": source_compile}


def save_qpy(circuit, path):
    with Path(path).open("wb") as stream:
        qpy.dump(circuit, stream)
    return {"path": relative(path), "sha256": sha256(Path(path).read_bytes()).hexdigest()}


def run_new_route(case, route, wrapper, directory, sources):
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "input.json", case)
    n = len(case["terms"][0][0])
    dimension = 1 << n
    print(json.dumps({"case": case["id"], "route": route, "stage": "build"}), flush=True)
    if route == "ours":
        source, encoding, build = build_adapter(case, directory / "placement.log")
        control_metadata = build["controlled_context_metadata"]
        scratch = int(control_metadata.get("scratch_qubits", 0))
        address = int(control_metadata.get("shared_address_qubits", source.num_qubits - n - scratch))
        controlled = {state: ADAPTERS[case["id"]]["lowered_queries"][(False, state)] for state in (0, 1)}
    else:
        source, encoding, controlled, build = build_native(case)
        address, scratch = build["address_size"], build["scratch_size"]
    print(json.dumps({"case": case["id"], "route": route, "stage": "all_query_domain_inputs",
                      "system_inputs": dimension, "logical_addresses": 1 << address, "scratch": scratch}), flush=True)
    domain_checks = validate_query_domain(source, controlled, n, address, scratch)
    write_json(directory / "query_domain_validation.json", domain_checks)
    if not domain_checks["passed"]:
        raise ValueError("Cannot exclude expansion scratch from the projector: query-domain validation failed")
    angles = np.asarray(case["angles"], dtype=float)
    tape = official_logical_tape(wrapper, encoding, angles, n, address, scratch)
    full_wires = [0] + list(range(1, source.num_qubits + 1))
    print(json.dumps({"case": case["id"], "route": route, "stage": "compile_qsvt", "W": len(full_wires)}), flush=True)
    lowered_tape, compile_metadata = compile_official(tape, full_wires)
    circuit = tape_to_qiskit(lowered_tape, full_wires)
    source_qpy = save_qpy(circuit, directory / "source.qpy")
    query_qpy = save_qpy(source, directory / "query.qpy")
    h = SparsePauliOp.from_list([(label, complex(coefficient)) for label, coefficient in case["terms"]]).to_matrix() / case["alpha"]
    target = polynomial(h, case["poly"])
    block, expected = logical_source_reference(source, angles, n, address, scratch)
    native = columns(circuit, dimension)
    scratch_mask = ((1 << scratch) - 1) << (n + address)
    scratch_bad = (np.arange(1 << circuit.num_qubits) & scratch_mask) != 0
    checks = {"source_block_encoding_error": max_error(block[:dimension], h),
              "complete_complex_output_reference_error": max_error(native, expected),
              "physical_polynomial_block_error": max_error(native[:dimension], target),
              "reference_physical_polynomial_block_error": max_error(expected[:dimension], target),
              "columns_orthogonality_error": max_error(native.conj().T @ native, np.eye(dimension)),
              "clean_scratch_output_error": float(np.max(np.abs(native[scratch_bad]), initial=0.0))}
    row = {"id": case["id"], "name": case["name"], "route": route, "reused": False,
           "terms": case["terms"], "alpha": case["alpha"], "poly": case["poly"], "angles": case["angles"],
           "input_sha256": sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
           "queries": 2 * (len(angles) - 1), "projectors": 2 * len(angles),
           "outer": "H-C1(U_phi)-C0(U_phi.adjoint)-H", "projector": projector_contract(n, address, scratch),
           "build": build, "compile": compile_metadata, "query_domain_validation": domain_checks,
           "source_qpy": source_qpy, "query_qpy": query_qpy, "checks": checks,
           "system_basis_inputs_checked": dimension, "all_complex_output_amplitudes_checked": True,
           "source_fingerprints": sources, "basis": {}}
    compiled = directory / "compiled"
    compiled.mkdir(parents=True, exist_ok=True)
    for basis, gates in BASIS.items():
        begun = time.perf_counter()
        lowered = transpile(circuit, basis_gates=gates, optimization_level=0, seed_transpiler=7,
                            qubits_initially_zero=False)
        lowering_seconds = time.perf_counter() - begun
        qasm_path = compiled / f"{route}_{basis}_opt0.qasm"
        qasm_path.write_text(qasm2.dumps(lowered) + "\n")
        phase_path = qasm_path.with_suffix(".global_phase.json")
        phase = float(lowered.global_phase)
        write_json(phase_path, {"global_phase_radians": phase, "restore_policy": "restore after QASM2 import"})
        actual = columns(lowered, dimension)
        saved = qasm2.load(qasm_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        saved.global_phase = json.loads(phase_path.read_text())["global_phase_radians"]
        roundtrip = columns(saved, dimension)
        counted = count_qasm_file(qasm_path, gate_set=basis)
        assert set(lowered.count_ops()) <= set(gates)
        assert counted["clifford"]["unknown"] == 0
        row["basis"][basis] = {"G": int(counted["metric_total"]), "NC": int(counted["clifford"]["non_clifford"]),
            "CX": int(counted["ops"].get("cx", 0)), "W": lowered.num_qubits, "depth": lowered.depth(),
            "ops": counted["ops"], "lowering_s": lowering_seconds,
            "lowering_max_error": max_error(actual, native),
            "complete_reference_max_error": max_error(actual, expected),
            "target_max_error": max_error(actual[:dimension], target),
            "clean_scratch_max_error": float(np.max(np.abs(actual[scratch_bad]), initial=0.0)),
            "saved_qasm_roundtrip_max_error": max_error(roundtrip, actual),
            "saved_qasm_target_max_error": max_error(roundtrip[:dimension], target),
            "global_phase_radians": phase, "qasm": relative(qasm_path),
            "qasm_sha256": sha256(qasm_path.read_bytes()).hexdigest(), "global_phase_sidecar": relative(phase_path)}
        write_json(directory / "summary.partial.json", row)
        print(json.dumps({"case": case["id"], "route": route, "basis": basis,
                          **{key: row["basis"][basis][key] for key in ("G", "NC", "CX", "W", "target_max_error")}}), flush=True)
    errors = list(checks.values()) + [metric[key] for metric in row["basis"].values()
        for key in ("lowering_max_error", "complete_reference_max_error", "target_max_error", "clean_scratch_max_error",
                    "saved_qasm_roundtrip_max_error", "saved_qasm_target_max_error")]
    row["passed"] = max(errors) < ATOL and domain_checks["passed"]
    row["status"] = "passed" if row["passed"] else "semantic_failure"
    row["seconds"] = time.perf_counter() - started
    write_json(directory / "summary.json", row)
    return row


def reuse_old_native(case, manifest, directory, sources):
    old_inputs = json.loads(OLD_INPUTS.read_text())
    previous = next(item for item in old_inputs["cases"] if item["id"] == case["id"])
    for key in ("terms", "alpha", "poly", "angles", "pl_metrics"):
        assert previous[key] == case[key], (case["id"], key)
    assert old_inputs["archive_sha256"] == manifest["archive_sha256"]
    old_validation = json.loads(OLD_NATIVE_CHECKS.read_text())
    assert old_validation["completed"] and old_validation["passed"]
    validation = next(item for item in old_validation["cases"] if item["id"] == case["id"])
    assert validation["passed"]
    n = len(case["terms"][0][0])
    address = (len(case["terms"]) - 1).bit_length()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "input.json", case)
    basis = {}
    source = None
    with tarfile.open(ROOT / manifest["archive"]) as archive:
        for name in BASIS:
            metric = case["pl_metrics"][name]
            assert metric["width"] == 1 + n + address, "Archived PL native has no extra scratch to exclude"
            text = archive.extractfile(metric["qasm"]).read().decode()
            assert sha256(text.encode()).hexdigest() == validation["basis"][name]["qasm_sha256"]
            compiled = directory / "compiled"
            compiled.mkdir(parents=True, exist_ok=True)
            path = compiled / f"pl_native_{name}_opt0.qasm"
            path.write_text(text)
            phase = float(metric["global_phase_radians"])
            phase_path = path.with_suffix(".global_phase.json")
            write_json(phase_path, {"global_phase_radians": phase, "restore_policy": "restore original archived phase after QASM2 import",
                                   "archive_member": metric["qasm"]})
            counted = count_qasm_file(path, gate_set=name)
            assert counted["metric_total"] == metric["gate_count"]
            assert counted["clifford"]["non_clifford"] == metric["non_clifford"]
            source = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS) if source is None else source
            if name == "nam":
                source.global_phase = phase
            basis[name] = {"G": metric["gate_count"], "NC": metric["non_clifford"],
                "CX": metric["cx_count"], "W": metric["width"], "depth": metric["depth"], "ops": metric["ops"],
                "global_phase_radians": phase, "qasm": relative(path), "qasm_sha256": sha256(path.read_bytes()).hexdigest(),
                "global_phase_sidecar": relative(phase_path), "archive_member": metric["qasm"],
                "target_max_error": validation["basis"][name]["target_max_abs_error"]}
    row = {"id": case["id"], "name": case["name"], "route": "pl_native", "reused": True,
           "passed": True, "status": "passed", "terms": case["terms"], "alpha": case["alpha"],
           "poly": case["poly"], "angles": case["angles"],
           "input_sha256": sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
           "queries": 2 * (len(case["angles"]) - 1), "projectors": 2 * len(case["angles"]),
           "outer": "H-C1(U_phi)-C0(U_phi.adjoint)-H", "projector": projector_contract(n, address, 0),
           "reuse_checks": {"frozen_input_contract_equal": True, "original_projector_equivalent": True,
                           "argument": "without scratch, the original rank-2**system projector on all block wires is exactly address==0 tensor I_system",
                           "compilation_rerun": False, "archive_sha256": manifest["archive_sha256"],
                           "physical_validation": validation, "physical_validation_path": relative(OLD_NATIVE_CHECKS)},
           "source_qpy": save_qpy(source, directory / "source.qpy"),
           "source_qpy_origin": "original archived NAM circuit with restored global phase",
           "source_fingerprints": sources, "basis": basis}
    write_json(directory / "summary.json", row)
    print(json.dumps({"case": case["id"], "route": "pl_native", "stage": "reused_archived_circuit"}), flush=True)
    return row


def collect_summary(out_dir, manifest, sources, official):
    cases = []
    for case in manifest["cases"]:
        routes = {}
        for route in ("pl_native", "ours"):
            path = out_dir / "cases" / case["id"] / route / "summary.json"
            if path.exists():
                row = json.loads(path.read_text())
                assert row["source_fingerprints"] == sources
                routes[route] = row
        if routes:
            cases.append({"id": case["id"], "name": case["name"], "routes": routes})
    return {"measured_at": datetime.now(timezone.utc).isoformat(), "cases": cases,
            "completed": len(cases) == len(ACTIVE) and all(set(case["routes"]) == {"pl_native", "ours"} for case in cases),
            "passed": bool(cases) and all(row.get("passed") for case in cases for row in case["routes"].values()),
            "sources": source_identity(), "source_fingerprints": sources, "official": official,
            "policy": {"outer": "H-C1(U_phi)-C0(U_phi.adjoint)-H", "projector": "logical address == 0; clean expansion scratch excluded",
                       "system_low_qiskit_bits": True, "queries": 10, "projectors": 12,
                       "basis_gates": BASIS, "optimization_level": 0, "seed_transpiler": 7,
                       "qubits_initially_zero": False, "NC_counter": "baseline_common"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_ROOT / "qsvt_inputs.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "pennylane_qsvt")
    parser.add_argument("--smoke", type=Path, default=DEFAULT_ROOT / "smoke.json")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--route", choices=("all", "pl_native", "ours"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    smoke = json.loads(args.smoke.read_text())
    assert smoke.get("passed") is True and source_identity() == smoke["sources"], "Fresh smoke and matching source hashes are required"
    manifest = json.loads(args.inputs.read_text())
    assert tuple(case["id"] for case in manifest["cases"]) == ACTIVE
    assert sha256((ROOT / manifest["archive"]).read_bytes()).hexdigest() == manifest["archive_sha256"]
    assert sha256(OLD_INPUTS.read_bytes()).hexdigest() == manifest["inherited_inputs_sha256"]
    wrapper, official = load_official_wrapper(ROOT / manifest["archive"])
    paths = [Path(__file__).resolve(), args.inputs.resolve(), args.smoke.resolve(), OLD_INPUTS, OLD_NATIVE_CHECKS,
             ROOT / "Baseline_scripts/run_rq3_pennylane_qsvt.py", ROOT / "Baseline_scripts/pennylane_lcu_adapter.py",
             ROOT / "Baseline_scripts/baseline_common.py", Path(inspect.getfile(qml.QSVT)), Path(inspect.getfile(qml.PCPhase))]
    paths += [ROOT / "src" / name for name in smoke["sources"]]
    sources = [source_record(path) for path in paths]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out_dir / "source_snapshot.tar.gz", "w:gz") as snapshot:
        for path in paths:
            snapshot.add(path, arcname=relative(path), recursive=False)
    write_json(args.out_dir / "source_manifest.json", sources)
    write_json(args.out_dir / "versions.json", {"pennylane": qml.__version__, "qiskit": qiskit.__version__, "python": sys.version})
    selected = [case for case in manifest["cases"] if not args.cases or case["id"] in args.cases]
    if args.cases:
        assert {case["id"] for case in selected} == set(args.cases), "Unknown active input"
    routes = ("pl_native", "ours") if args.route == "all" else (args.route,)
    for case in selected:
        for route in routes:
            directory = args.out_dir / "cases" / case["id"] / route
            path = directory / "summary.json"
            if args.resume and path.exists():
                cached = json.loads(path.read_text())
                if cached.get("passed") and cached.get("source_fingerprints") == sources:
                    print(json.dumps({"case": case["id"], "route": route, "stage": "already_completed"}), flush=True)
                    continue
            try:
                row = (reuse_old_native(case, manifest, directory, sources)
                       if route == "pl_native" and case["id"] != "maxcut_cycle4" else
                       run_new_route(case, route, wrapper, directory, sources))
            except Exception as exception:
                row = {"id": case["id"], "name": case["name"], "route": route, "passed": False,
                       "status": "exception", "error": repr(exception), "traceback": traceback.format_exc(),
                       "source_fingerprints": sources}
                write_json(path, row)
                print(row["traceback"], flush=True)
            summary = collect_summary(args.out_dir, manifest, sources, official)
            write_json(args.out_dir / "summary.json", summary)
            if not row["passed"]:
                raise SystemExit(1)
    assert source_identity() == smoke["sources"], "Backend drift during the run"
    summary = collect_summary(args.out_dir, manifest, sources, official)
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "completed": summary["completed"],
                      "summary": relative(args.out_dir / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
