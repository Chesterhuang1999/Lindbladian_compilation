#!/usr/bin/env python3
"""Qrisp from_lcu queries in the shared two-branch QSVT comparison.

Qrisp constructs the actual plain and controlled block queries. The shared
PennyLane QSVT core, logical-address projectors, official two-branch outer
circuit and compile pipeline are then applied to these query leaves. This is
an adapted native Qrisp query comparison, not Qrisp's public QSVT output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
import importlib
from importlib.metadata import version
import inspect
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "QISKIT_NUM_PROCS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-rq3-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/codex-rq3-numba")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]

import numpy as np
import pennylane as qml
import qiskit
from qiskit import QuantumCircuit, qasm2, qpy, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qrisp import QuantumBool, QuantumVariable, control
from qrisp.jasp import q_switch

from baseline_common import count_qasm_file
import run_rq3_pennylane_qsvt as legacy
import run_rq3_qrisp_qsvt as native

BATCH = ROOT / "Baseline_results/rq3_common_outer_20261014"
ATOL = 1e-8


def dump(path, data):
    legacy.write_json(path, data)


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_qpy(path, circuit):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)
    return {"path": legacy.relative(path), "sha256": file_hash(path)}


def common_module():
    return importlib.import_module("run_rq3_common_pl_qsvt")


def source_records(inputs, smoke):
    paths = [Path(__file__).resolve(), inputs.resolve(), smoke.resolve(),
             ROOT / "Baseline_scripts/run_rq3_common_pl_qsvt.py",
             Path(legacy.__file__), Path(native.__file__),
             ROOT / "Baseline_scripts/pennylane_lcu_adapter.py",
             ROOT / "Baseline_scripts/baseline_common.py",
             Path(inspect.getfile(qml.QSVT)), Path(inspect.getfile(qml.PCPhase))]
    for name in ("block_encoding.py", "placement_search.py", "matrix_order_pipeline.py",
                 "channel_IR.py", "subroutine.py", "mobius_residual_fixed.py"):
        paths.append(ROOT / "src" / name)
    for module in ("qrisp.block_encodings.block_encoding_base",
                   "qrisp.block_encodings.constructors.from_lcu",
                   "qrisp.environments.control_environment",
                   "qrisp.interface.converter.qiskit_converter"):
        imported = importlib.import_module(module)
        paths.append(Path(imported.__file__))
    paths.append(Path(inspect.getfile(q_switch)))
    return [legacy.source_record(path) for path in dict.fromkeys(paths)]


def check_smoke(path):
    smoke = json.loads(path.read_text())
    if smoke.get("passed") is not True:
        raise ValueError("The fresh archived-interface smoke gate must pass before execution")
    for name, expected in smoke["sources"].items():
        source = ROOT / "src" / name
        if not source.exists() or file_hash(source) != expected:
            raise ValueError(f"Backend source changed after the fresh smoke gate: {name}")
    return {"path": legacy.relative(path), "sha256": file_hash(path), "passed": True,
            "backend_fingerprints_match": True}


def construct_native_query(case, state=None):
    """Use Qrisp's control environment before any gate-definition expansion."""
    started = time.perf_counter()
    block, block_metadata = native.native_block(case)
    if not np.isclose(float(block.alpha), float(case["alpha"]), atol=1e-12, rtol=0):
        raise ValueError("Qrisp from_lcu changed the frozen alpha")
    n = len(case["terms"][0][0])
    signal = QuantumBool(name="signal") if state is not None else None
    system = QuantumVariable(n, name="system")
    indices = block.create_ancillas()
    address = sum(len(variable.reg) for variable in indices)
    with redirect_stdout(io.StringIO()):
        if state is None:
            block.unitary(*indices, system)
        else:
            with control(signal[0], ctrl_state=state):
                block.unitary(*indices, system)
    permanent = indices if signal is None else [signal] + indices
    packed, ports = native.pack_fixed_ports(system.qs.copy(), permanent, system)
    output = legacy.canonicalize(packed, ports["system_qubits"], controlled=signal is not None)
    projection_width = address + int(signal is not None)
    scratch = len(ports["clean_scratch_qubits"])
    # pack_fixed_ports places projection, scratch, system in that order.
    # canonicalize moves system low and, for a controlled query, signal high.
    remap = {old: n + old - int(signal is not None)
             for old in range(int(signal is not None), projection_width)}
    remap.update({old: n + address + old - projection_width
                  for old in ports["clean_scratch_qubits"]})
    remap.update({old: bit for bit, old in enumerate(ports["system_qubits"])})
    if signal is not None:
        remap[0] = output.num_qubits - 1
    releases = [{"position": checkpoint["position"],
                 "scratch_slots": [remap[slot] for slot in checkpoint["scratch_slots"]]}
                for checkpoint in ports["release_checks"]]
    metadata = {"backend": block_metadata, "system_size": n, "address_size": address,
        "scratch_size": scratch, "ctrl_state": state,
        "control_synthesis": "Qrisp control(signal, ctrl_state) applied to native from_lcu.unitary"
            if state is not None else "native from_lcu.unitary",
        "prepare_control_policy": "native from_lcu conjugate(prepare) and Qrisp control synthesis",
        "packed_ports": ports, "canonical_release_checks": releases,
        "canonical_system_qubits": list(range(n)),
        "canonical_address_qubits": list(range(n, n + address)),
        "canonical_scratch_qubits": list(range(n + address, n + address + scratch)),
        "canonical_signal_qubit": output.num_qubits - 1 if signal is not None else None,
        "construct_seconds": time.perf_counter() - started,
        "op_counts_before_shared_leaf_import": dict(output.count_ops()),
        "quantum_session_compile_used": False, "matrix_synthesis_used": False}
    output.metadata = metadata
    return output, metadata


def pad_query(circuit, n, address, scratch, controlled=False):
    width = n + address + scratch + int(controlled)
    old_signal = circuit.num_qubits - 1
    mapping = list(range(circuit.num_qubits))
    if controlled:
        mapping[old_signal] = width - 1
    output = QuantumCircuit(width, name=circuit.name)
    output.compose(circuit, qubits=mapping, inplace=True)
    output.metadata = dict(circuit.metadata or {})
    return output


def build_adapter(case, case_dir):
    started = time.perf_counter()
    plain, plain_meta = construct_native_query(case)
    controlled, control_meta = {}, {}
    for state in (0, 1):
        controlled[state], control_meta[state] = construct_native_query(case, state)
    n, address = plain_meta["system_size"], plain_meta["address_size"]
    if any(meta["address_size"] != address for meta in control_meta.values()):
        raise ValueError("Controlled native synthesis changed the logical index interface")
    scratch = max(plain_meta["scratch_size"], *(meta["scratch_size"] for meta in control_meta.values()))
    plain = pad_query(plain, n, address, scratch)
    controlled = {state: pad_query(query, n, address, scratch, controlled=True)
                  for state, query in controlled.items()}
    token = "common_qrisp_" + case["id"]
    leaves, lowered_queries = {}, {}
    for inverse in (False, True):
        source = plain.inverse() if inverse else plain
        leaves[(inverse, None)], lowered_queries[(inverse, None)] = legacy.leaf_operations(source)
        for state in (0, 1):
            source = controlled[state].inverse() if inverse else controlled[state]
            leaves[(inverse, state)], lowered_queries[(inverse, state)] = legacy.leaf_operations(
                source, controlled=True)
    legacy.ADAPTERS[token] = {"leaves": leaves, "lowered_queries": lowered_queries}
    with qml.QueuingManager.stop_recording():
        encoding = legacy.CanonicalBlockQuery(token, list(range(1, plain.num_qubits + 1)))
    artifacts = {"plain": save_qpy(case_dir / "queries/plain.qpy", plain)}
    for state in (0, 1):
        artifacts[f"controlled_{state}"] = save_qpy(
            case_dir / f"queries/controlled_{state}.qpy", controlled[state])
    metadata = {"backend": "qrisp.block_encodings.BlockEncoding.from_lcu",
        "query_control_backend": "native Qrisp control environment before leaf decomposition",
        "system_size": n, "logical_address_size": address, "clean_scratch_size": scratch,
        "block_width": plain.num_qubits, "normalization_alpha": float(case["alpha"]),
        "plain": plain_meta, "controlled": {str(state): meta for state, meta in control_meta.items()},
        "pad_policy": "Identity pad to the maximum actual query scratch width; signal kept highest",
        "projector": "logical index all zero; identity on system and verified clean SELECT scratch",
        "controlled_context_used_for_early_synthesis": True,
        "matrix_synthesis_used": False, "public_qrisp_qsvt_called": False,
        "canonical_inverse_policy": "exact adjoint of the actual plain/controlled query circuit",
        "leaf_counts": {str(key): dict(Counter(op.name for op in ops)) for key, ops in leaves.items()},
        "query_artifacts": artifacts, "construction_seconds": time.perf_counter() - started}
    return plain, controlled, encoding, metadata


def release_domain_columns(circuit, indices, checkpoints):
    """Check each actual release boundary while evolving the full input domain."""
    indices = list(indices)
    if any(not 0 <= item["position"] <= len(circuit.data) for item in checkpoints):
        raise ValueError("Native scratch release boundary is outside the exported query")
    boundaries = sorted(set([item["position"] for item in checkpoints] + [len(circuit.data)]))
    segments = []
    previous = 0
    for segment_number, position in enumerate(boundaries):
        segment = QuantumCircuit(circuit.num_qubits)
        for instruction in circuit.data[previous:position]:
            segment.append(instruction.operation,
                           [circuit.find_bit(bit).index for bit in instruction.qubits])
        if segment_number == 0:
            segment.global_phase = circuit.global_phase
        compiled = transpile(segment, basis_gates=list(legacy.BRIDGE), optimization_level=0,
                             seed_transpiler=7, qubits_initially_zero=False)
        segments.append((position, compiled))
        previous = position
    rows = np.arange(1 << circuit.num_qubits)
    records = []
    for item in checkpoints:
        mask = sum(1 << bit for bit in item["scratch_slots"])
        records.append({"position": item["position"], "scratch_slots": item["scratch_slots"],
                        "max_error": 0.0, "bad_rows": (rows & mask) != 0})
    output = np.empty((1 << circuit.num_qubits, len(indices)), dtype=complex)
    for column, index in enumerate(indices):
        initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
        initial[index] = 1
        vector = initial.copy()
        for position, segment in segments:
            vector = Statevector(np.array(vector, dtype=complex, copy=True)).evolve(segment).data.copy()
            for record in records:
                if record["position"] == position:
                    record["max_error"] = max(record["max_error"],
                        float(np.max(np.abs(vector[record["bad_rows"]]), initial=0.0)))
        output[:, column] = vector
    for record in records:
        del record["bad_rows"]
    max_error = max((record["max_error"] for record in records), default=0.0)
    return output, {"passed": max_error < ATOL, "basis_inputs_checked": len(indices),
        "checkpoint_count": len(records), "basis_checkpoint_checks": len(indices) * len(records),
        "max_error": max_error, "checkpoints": records,
        "method": "independent Statevector evolution at every original Qrisp temporary release boundary"}


def validate_native_query_domain(source, controlled, n, address, scratch, build):
    """Shared domain contract plus evidence that every reused slot was clean."""
    assert source.num_qubits == n + address + scratch and set(controlled) == {0, 1}
    assert all(query.num_qubits == source.num_qubits + 1 for query in controlled.values())
    domain, width = 1 << (n + address), source.num_qubits
    original, plain_releases = release_domain_columns(
        source, range(domain), build["plain"]["canonical_release_checks"])
    scratch_mask = ((1 << scratch) - 1) << (n + address)
    scratch_error = float(np.max(np.abs(original[(np.arange(1 << width) & scratch_mask) != 0]), initial=0.0))
    unitary_error = legacy.max_error(original.conj().T @ original, np.eye(domain))
    signal_offset = 1 << width
    full_bad = (np.arange(1 << (width + 1)) & scratch_mask) != 0
    results, releases = {}, {"plain": plain_releases}
    for state, query in controlled.items():
        indices = [value + signal * signal_offset for signal in (0, 1) for value in range(domain)]
        actual, releases[str(state)] = release_domain_columns(
            query, indices, build["controlled"][str(state)]["canonical_release_checks"])
        active = actual[:, state * domain:(state + 1) * domain]
        expected_active = np.zeros_like(active)
        expected_active[state * signal_offset:(state + 1) * signal_offset] = original
        inactive_signal = 1 - state
        inactive = actual[:, inactive_signal * domain:(inactive_signal + 1) * domain]
        expected_inactive = np.zeros_like(inactive)
        expected_inactive[np.arange(domain) + inactive_signal * signal_offset, np.arange(domain)] = 1
        results[str(state)] = {"basis_inputs_checked": 2 * domain,
            "active_source_max_error": legacy.max_error(active, expected_active),
            "inactive_identity_max_error": legacy.max_error(inactive, expected_inactive),
            "clean_scratch_max_error": float(np.max(np.abs(actual[full_bad]), initial=0.0))}
    release_validation = {"passed": all(item["passed"] for item in releases.values()),
        "checkpoint_count": sum(item["checkpoint_count"] for item in releases.values()),
        "basis_checkpoint_checks": sum(item["basis_checkpoint_checks"] for item in releases.values()),
        "max_error": max(item["max_error"] for item in releases.values()),
        "plain": releases["plain"], "controlled": {state: releases[state] for state in ("0", "1")}}
    errors = [scratch_error, unitary_error] + [row[key] for row in results.values()
        for key in ("active_source_max_error", "inactive_identity_max_error", "clean_scratch_max_error")]
    passed = max(errors) < ATOL and release_validation["passed"]
    domain_validation = {"passed": passed, "system_basis_inputs": 1 << n,
        "all_logical_addresses": 1 << address, "source_basis_inputs_checked": domain,
        "controlled_basis_inputs_checked": 4 * domain,
        "source_clean_scratch_max_error": scratch_error, "source_domain_unitarity_max_error": unitary_error,
        "controlled": results,
        "adjoint_scratch_invariance": "complete scratch-zero subspace is invariant and source action on it is square unitary",
        "projector_excludes_scratch_authorized_by_validation": passed}
    return domain_validation, release_validation


def run_case(case, wrapper, out_dir):
    started = time.perf_counter()
    folder = out_dir / "cases" / case["id"] / "qrisp_native"
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / "input.json", case)
    print(f"Building native Qrisp query {case['id']}", flush=True)
    source, controlled, encoding, build = build_adapter(case, folder)
    n, address, scratch = build["system_size"], build["logical_address_size"], build["clean_scratch_size"]
    common = common_module()
    validation_started = time.perf_counter()
    print(f"Checking full query domain {case['id']}: {1 << (n + address)} address/system inputs", flush=True)
    query_forward, release_validation = validate_native_query_domain(source, controlled, n, address, scratch, build)
    query_inverse = {"passed": query_forward["passed"],
        "method": "Exact saved-circuit adjoints preserve the proven complete scratch-zero invariant subspace",
        "source_and_controlled_queries_are_exact_adjoints": True,
        "forward_validation_dependency": True,
        "controlled_active_and_inactive": "Adjoints of the validated active source and inactive identity actions",
        "new_basis_inputs_simulated": 0,
        "forward_basis_inputs_proving_inverse": query_forward["source_basis_inputs_checked"] +
            query_forward["controlled_basis_inputs_checked"]}
    dump(folder / "query_domain_validation.json", {"forward": query_forward, "inverse": query_inverse,
        "release_validation": release_validation})
    if not query_forward.get("passed") or not query_inverse.get("passed"):
        dump(folder / "query_domain_validation.json", {"forward": query_forward, "inverse": query_inverse})
        raise ValueError("Native Qrisp query failed the complete scratch-zero / controlled identity domain check")
    query_validation_seconds = time.perf_counter() - validation_started
    angles = np.asarray(case["angles"], dtype=float)
    tape = common.official_logical_tape(wrapper, encoding, angles, n, address, scratch)
    full_wires = [0] + list(encoding.wires)
    print(f"Compiling common outer {case['id']}, width={len(full_wires)}", flush=True)
    expanded, compile_meta = legacy.compile_official(tape, full_wires)
    exporting = time.perf_counter()
    circuit = legacy.tape_to_qiskit(expanded, full_wires)
    export_seconds = time.perf_counter() - exporting
    source_artifact = save_qpy(folder / "source.qpy", circuit)
    dimension = 1 << n
    normalized_h = SparsePauliOp.from_list(case["terms"]).to_matrix() / case["alpha"]
    target = legacy.polynomial(normalized_h, case["poly"])
    validation_started = time.perf_counter()
    block_columns, expected = common.logical_source_reference(source, angles, n, address, scratch)
    actual = legacy.columns(circuit, dimension)
    np.savez_compressed(folder / "validated_columns.npz", source_query=block_columns,
                        common_reference=expected, qsvt_source=actual,
                        normalized_h=normalized_h, polynomial_target=target)
    scratch_mask = sum(1 << bit for bit in range(n + address, n + address + scratch))
    bad_scratch = (np.arange(1 << circuit.num_qubits) & scratch_mask) != 0
    checks = {"source_block_encoding_error": legacy.max_error(block_columns[:dimension], normalized_h),
        "phase_sensitive_valid_input_isometry_error": legacy.max_error(actual, expected),
        "physical_polynomial_block_error": legacy.max_error(actual[:dimension], target),
        "reference_polynomial_block_error": legacy.max_error(expected[:dimension], target),
        "checked_columns_orthogonality_error": legacy.max_error(actual.conj().T @ actual, np.eye(dimension)),
        "clean_scratch_output_error": float(np.max(np.abs(actual[bad_scratch]))) if np.any(bad_scratch) else 0.0}
    route = {"id": case["id"], "name": case["name"], "route": "qrisp_native",
        "status": "verifying", "passed": False, "terms": case["terms"], "alpha": case["alpha"],
        "reused": False, "input_sha256": sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
        "poly": case["poly"], "angles": case["angles"], "build": build, "compile": compile_meta,
        "source_qpy": source_artifact, "export_seconds": export_seconds,
        "queries": 2 * (len(angles) - 1), "projectors": 2 * len(angles),
        "queries_per_branch": len(angles) - 1, "projectors_per_branch": len(angles),
        "outer": "H-C1(U_phi)-C0(U_phi adjoint)-H; shared official wrapper",
        "single_branch_engine": "PennyLane qml.QSVT applied to adapted native Qrisp queries",
        "untouched_public_qrisp_qsvt": False,
        "projector_wires": list(range(scratch + 1, scratch + address + 1)),
        "projector_dim": 1, "projector": common.projector_contract(n, address, scratch), "checks": checks,
        "system_basis_inputs_checked": dimension, "all_complex_output_amplitudes_checked": True,
        "query_validation": {"forward": query_forward, "inverse": query_inverse,
                             "seconds": query_validation_seconds},
        "query_domain_validation": query_forward,
        "release_validation": release_validation,
        "basis": {}, "verification": {
            "method": "all clean-auxiliary system basis inputs; every complex output amplitude",
            "system_basis_inputs": dimension, "all_output_amplitudes_checked": True,
            "physical_block_compared_without_real_projection": True,
            "independent_reference": "actual source query and reverse-adjoint logical-projector sequence",
            "reference_and_native_seconds": time.perf_counter() - validation_started}}
    for basis, gates in legacy.BASIS.items():
        lowering_start = time.perf_counter()
        lowered = transpile(circuit, basis_gates=gates, optimization_level=0,
                            seed_transpiler=7, qubits_initially_zero=False)
        lowering_seconds = time.perf_counter() - lowering_start
        qasm_path = folder / f"qrisp_native_{basis}_opt0.qasm"
        qasm_path.write_text(qasm2.dumps(lowered) + "\n")
        phase_path = qasm_path.with_suffix(".global_phase.json")
        phase = float(lowered.global_phase)
        dump(phase_path, {"global_phase_radians": phase,
                         "restore_policy": "Restore circuit.global_phase after importing OpenQASM 2"})
        validation_start = time.perf_counter()
        lowered_columns = legacy.columns(lowered, dimension)
        roundtrip = qasm2.load(qasm_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        roundtrip.global_phase = json.loads(phase_path.read_text())["global_phase_radians"]
        roundtrip_columns = legacy.columns(roundtrip, dimension)
        np.savez_compressed(folder / f"validated_{basis}_columns.npz",
                            lowered=lowered_columns, saved_qasm=roundtrip_columns)
        stats = count_qasm_file(qasm_path, gate_set=basis)
        metric = {"G": int(stats["metric_total"]), "NC": int(stats["clifford"]["non_clifford"]),
            "CX": int(stats["ops"].get("cx", 0)), "W": lowered.num_qubits,
            "gate_count": int(stats["metric_total"]), "non_clifford": int(stats["clifford"]["non_clifford"]),
            "cx_count": int(stats["ops"].get("cx", 0)), "width": lowered.num_qubits,
            "depth": lowered.depth(), "ops": stats["ops"], "basis_lowering_seconds": lowering_seconds,
            "total_compile_seconds": build["construction_seconds"] + compile_meta["compile_seconds"] +
                export_seconds + lowering_seconds,
            "phase_sensitive_lowering_error": legacy.max_error(lowered_columns, actual),
            "phase_sensitive_reference_error": legacy.max_error(lowered_columns, expected),
            "physical_polynomial_block_error": legacy.max_error(lowered_columns[:dimension], target),
            "saved_qasm_target_max_error": legacy.max_error(roundtrip_columns[:dimension], target),
            "clean_scratch_max_error": float(np.max(np.abs(lowered_columns[bad_scratch]))) if np.any(bad_scratch) else 0.0,
            "saved_qasm_roundtrip_error": legacy.max_error(roundtrip_columns, lowered_columns),
            "global_phase_radians": phase, "qasm": legacy.relative(qasm_path),
            "qasm_sha256": file_hash(qasm_path), "global_phase_sidecar": legacy.relative(phase_path),
            "global_phase_sidecar_sha256": file_hash(phase_path),
            "verification_seconds": time.perf_counter() - validation_start}
        metric.update(lowering_max_error=metric["phase_sensitive_lowering_error"],
                      complete_reference_max_error=metric["phase_sensitive_reference_error"],
                      target_max_error=metric["physical_polynomial_block_error"],
                      saved_qasm_roundtrip_max_error=metric["saved_qasm_roundtrip_error"])
        route["basis"][basis] = metric
        dump(folder / "summary.json", route)
        print(json.dumps({"case": case["id"], "basis": basis, "G": metric["G"],
                          "NC": metric["NC"], "W": metric["W"]}), flush=True)
    errors = list(checks.values()) + [metric[key] for metric in route["basis"].values()
        for key in ("phase_sensitive_lowering_error", "phase_sensitive_reference_error",
                    "physical_polynomial_block_error", "saved_qasm_roundtrip_error",
                    "saved_qasm_target_max_error", "clean_scratch_max_error")]
    route["passed"] = bool(query_forward["passed"] and query_inverse["passed"] and max(errors) < ATOL)
    route["status"] = "ok" if route["passed"] else "semantic_failure"
    route["wall_seconds"] = time.perf_counter() - started
    dump(folder / "summary.json", route)
    return route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=BATCH / "qsvt_inputs.json")
    parser.add_argument("--out-dir", type=Path, default=BATCH / "qrisp_qsvt")
    parser.add_argument("--smoke", type=Path, default=BATCH / "smoke.json")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    smoke = check_smoke(args.smoke)
    manifest = json.loads(args.inputs.read_text())
    archive = ROOT / manifest["archive"]
    if file_hash(archive) != manifest["archive_sha256"]:
        raise ValueError("Frozen official source archive fingerprint changed")
    wrapper, official = legacy.load_official_wrapper(archive)
    common_module()
    sources = source_records(args.inputs, args.smoke)
    summary = {"measured_at": datetime.now(timezone.utc).isoformat(), "passed": False,
        "versions": {"qrisp": version("qrisp"), "pennylane": qml.__version__,
                     "qiskit": qiskit.__version__, "python": sys.version},
        "source_fingerprints": sources, "fresh_smoke": smoke, "official": official,
        "scope": "native Qrisp from_lcu controlled queries in shared QSVT outer/PCPhase/compile pipeline",
        "public_qrisp_qsvt_executed": False,
        "basis_policy": {"bases": legacy.BASIS, "optimization_level": 0, "seed_transpiler": 7,
                         "qubits_initially_zero": False, "metric": "baseline_common.count_qasm_file"},
        "cases": []}
    cases = [case for case in manifest["cases"] if not args.cases or case["id"] in args.cases]
    if args.cases and {case["id"] for case in cases} != set(args.cases):
        raise ValueError("Unknown active QSVT case")
    for case in cases:
        summary_path = args.out_dir / "cases" / case["id"] / "qrisp_native/summary.json"
        row = None
        if args.resume and summary_path.exists():
            cached = json.loads(summary_path.read_text())
            if cached.get("passed") and cached.get("source_fingerprints") == sources:
                row = cached
        if row is None:
            try:
                row = run_case(case, wrapper, args.out_dir)
            except Exception as exc:
                row = {"id": case["id"], "name": case["name"], "status": "exception", "passed": False,
                       "error": repr(exc), "traceback": traceback.format_exc()}
                print(row["traceback"], flush=True)
            row["source_fingerprints"] = sources
            row["fresh_smoke"] = smoke
            dump(summary_path, row)
        summary["cases"].append({"id": case["id"], "name": case["name"], "routes": {"qrisp_native": row}})
        summary["passed"] = all(case["routes"]["qrisp_native"].get("passed") for case in summary["cases"])
        dump(args.out_dir / "summary.json", summary)
    summary["passed"] = bool(summary["cases"]) and all(
        case["routes"]["qrisp_native"].get("passed") for case in summary["cases"])
    dump(args.out_dir / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "summary": legacy.relative(args.out_dir / "summary.json")}), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
