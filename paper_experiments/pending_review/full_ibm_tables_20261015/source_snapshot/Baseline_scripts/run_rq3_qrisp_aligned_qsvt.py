#!/usr/bin/env python3
"""Aligned local-control QSVT for PL, Qrisp, and frozen-placement Ours.

All routes use H-C1(U_phi)-C0(U_phi.adjoint)-H, uncontrolled coefficient
preparation, and the same address-only compute/controlled-RZ/uncompute
projector.  Query builders are public helpers without exponential validation,
so the large diagonal cases can use their independent symbolic/MPS checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
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
import tarfile
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
from qiskit.quantum_info import SparsePauliOp
from qrisp import QuantumBool, QuantumFloat, QuantumVariable, control, gphase, x, y, z
from qrisp.jasp import q_switch

from baseline_common import count_qasm_file
from block_encoding import BlockEncoding
from channel_IR import Matrixsum, PauliAtom
from experiment_protocol import source_identity
from pennylane_lcu_adapter import (
    LEAF_GATE_SET, _audit_tape, _decomposition_mode, tape_to_qiskit,
)
import run_rq3_pennylane_qsvt as legacy
import run_rq3_common_pl_qsvt as common
import run_rq3_common_qrisp_qsvt as native

OUT = ROOT / "Baseline_results/rq3_qrisp_aligned_qsvt_20261014"
INPUTS = ROOT / "Baseline_results/rq3_common_outer_20261014/qsvt_inputs.json"
SMOKE = OUT / "interface_smoke/smoke.json"
BASES = legacy.BASIS
SETTINGS = dict(optimization_level=0, seed_transpiler=7, qubits_initially_zero=False)
TOL = 1e-8
ROUTES = ("pl_native", "qrisp_native", "ours")


def dump(path, value):
    legacy.write_json(path, value)


def file_record(path):
    return legacy.source_record(Path(path))


def save_qpy(path, circuit):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)
    return file_record(path)


def _terms(case):
    return [(label, complex(coefficient)) for label, coefficient in case["terms"]]


def compose_controlled_query(prepare, controlled_select, *, system_size,
                             address_size, scratch_size):
    """P, C1(SELECT), P.adjoint in [system,address,scratch,signal] order."""
    width = system_size + address_size + scratch_size
    if prepare.num_qubits != width or controlled_select.num_qubits != width + 1:
        raise ValueError("PREPARE/CSELECT do not match the canonical query port contract")
    output = QuantumCircuit(width + 1, name="LocallyControlledBlockQuery")
    output.compose(prepare, qubits=range(width), inplace=True)
    output.compose(controlled_select, inplace=True)
    output.compose(prepare.inverse(), qubits=range(width), inplace=True)
    return output


def controlled_logical_projector(phi, *, system_size, address_size, scratch_size):
    """Address predicate compute is uncontrolled; only RZ has signal control.

    Ports: [system,address,scratch,independent clean predicate flag,signal].
    The phase is +phi for address zero and -phi otherwise on signal one;
    signal zero is identity.  Compute.inverse(), not a second assumed-
    Hermitian synthesis, clears any phases introduced by predicate lowering.
    """
    if address_size < 1:
        raise ValueError("The retained QSVT cases require a logical address register")
    block_width = system_size + address_size + scratch_size
    flag, signal = block_width, block_width + 1
    width = block_width + 2
    addresses = list(range(system_size, system_size + address_size))
    compute = QuantumCircuit(width, name="AddressZeroPredicate")
    for bit in addresses:
        compute.x(bit)
    compute.mcx(addresses, flag)
    for bit in addresses:
        compute.x(bit)
    output = QuantumCircuit(width, name="ControlledLogicalPCPhase")
    output.compose(compute, inplace=True)
    output.crz(2 * float(phi), signal, flag)
    output.compose(compute.inverse(), inplace=True)
    output.metadata = {
        "condition": "logical address == 0; system and SELECT scratch excluded",
        "compute_outer_control": False, "uncompute_outer_control": False,
        "rotation": "Csignal RZ(2*phi) on the independent clean predicate flag",
        "uncompute": "exact adjoint of the actual predicate compute circuit",
        "flag": flag, "signal": signal, "address": addresses,
    }
    return output


def assemble_aligned_outer(controlled_query, angles, *, system_size,
                           address_size, scratch_size, optimize=True):
    """Compile the same two-branch outer after each route's local synthesis.

    Input query ports are [system,address,scratch,signal].  The returned
    circuit inserts one independent flag before signal and uses a common
    Qiskit bridge plus PennyLane's default basic compile passes.  No matrix
    synthesis and no NAM prelowering is used before the IBM export.
    """
    block_width = system_size + address_size + scratch_size
    if controlled_query.num_qubits != block_width + 1:
        raise ValueError("Controlled query does not match canonical ports")
    flag, signal, width = block_width, block_width + 1, block_width + 2
    query = QuantumCircuit(width)
    query.compose(controlled_query, qubits=list(range(block_width)) + [signal], inplace=True)
    positive = QuantumCircuit(width, name="C1_U_phi")
    for index, phi in enumerate(angles):
        phase = controlled_logical_projector(phi, system_size=system_size,
            address_size=address_size, scratch_size=scratch_size)
        positive.compose(phase, inplace=True)
        if index + 1 < len(angles):
            positive.compose(query if index % 2 == 0 else query.inverse(), inplace=True)
    raw = QuantumCircuit(width, name="AlignedTwoBranchQSVT")
    raw.h(signal)
    raw.compose(positive, inplace=True)
    raw.x(signal)
    raw.compose(positive.inverse(), inplace=True)
    raw.x(signal)
    raw.h(signal)
    raw_counts = dict(raw.count_ops())
    leaves, bridge = legacy.leaf_operations(raw)
    wire_order = list(range(1, width + 1))
    tape = qml.tape.QuantumScript(list(leaves))
    if optimize:
        expanded, compile_metadata = legacy.compile_official(tape, wire_order)
    else:
        expanded, compile_metadata = tape, {"pipeline": [], "compile_seconds": 0.0}
    compiled = tape_to_qiskit(expanded, wire_order)
    metadata = {
        "outer": "H-C1(U_phi)-C0(U_phi.adjoint)-H",
        "ctrl0_implementation": "Xsignal-C1(U_phi.adjoint)-Xsignal",
        "queries": 2 * (len(angles) - 1), "projectors": 2 * len(angles),
        "queries_per_branch": len(angles) - 1,
        "projectors_per_branch": len(angles),
        "PREPARE_has_outer_control": False,
        "projector_compute_has_outer_control": False,
        "projector_uncompute_has_outer_control": False,
        "projector_rotation": "Csignal RZ(2phi) on an independent clean flag",
        "projector_excludes_query_scratch": True,
        "projector_flag_qubit": flag, "signal_qubit": signal,
        "system_qubits": list(range(system_size)),
        "address_qubits": list(range(system_size, system_size + address_size)),
        "query_scratch_qubits": list(range(system_size + address_size, block_width)),
        "raw_operation_counts": raw_counts,
        "bridge_operation_counts": dict(bridge.count_ops()),
        "shared_basic_compile": compile_metadata,
        "matrix_synthesis_used": False,
    }
    compiled.metadata = metadata
    return compiled, metadata


def _native_pl_circuit(operations, width):
    """Native graph decomposition, before any primitive-level control."""
    wire_order = list(reversed(range(width)))
    tape = qml.tape.QuantumScript(operations)
    with _decomposition_mode("graph"):
        [expanded], _ = qml.transforms.decompose(
            tape, gate_set=LEAF_GATE_SET, num_work_wires=0)
    _audit_tape(expanded, wire_order)
    return tape_to_qiskit(expanded, wire_order), {
        "mode": "native PennyLane graph decomposition",
        "dynamic_work_wire_budget": 0,
        "input_operations": dict(Counter(op.name for op in operations)),
        "leaf_operations": dict(Counter(op.name for op in expanded.operations)),
        "primitive_control_added_after_lowering": False,
    }


def build_pl_components(case, *, work_wires=None):
    """Native coefficient PREPARE and graph C(SELECT), without validation.

    The explicit graph-work budget defaults to address_size-1, matching the
    established standalone graph_work SELECT strategy.  The native partial
    Select contract of PrepSelPrep is retained; full power-of-two inputs have
    no unused addresses.  Returns PREPARE, plain SELECT, {0: C0_SELECT,
    1: C1_SELECT}, and JSON metadata in canonical ports.
    """
    started = time.perf_counter()
    terms = _terms(case)
    n, address = len(terms[0][0]), (len(terms) - 1).bit_length()
    scratch = max(0, address - 1) if work_wires is None else int(work_wires)
    width = n + address + scratch
    controls = list(reversed(range(n, n + address)))
    work = list(range(n + address, width))
    with qml.QueuingManager.stop_recording():
        operators = []
        for label, coefficient in terms:
            factors = [{"X": qml.X, "Y": qml.Y, "Z": qml.Z}[letter](bit)
                for bit, letter in enumerate(reversed(label)) if letter != "I"]
            pauli = qml.Identity(0) if not factors else factors[0] if len(factors) == 1 else qml.prod(*factors)
            operators.append(qml.prod(pauli, qml.GlobalPhase(-float(np.angle(coefficient)), wires=pauli.wires)))
        prep = qml.AmplitudeEmbedding(np.sqrt([abs(c) for _, c in terms]),
            wires=controls, normalize=True, pad_with=0)
        select = qml.Select(operators, control=controls, work_wires=work, partial=True)
    prepare, prepare_meta = _native_pl_circuit([prep], width)
    plain_select, select_meta = _native_pl_circuit([select], width)
    controlled, controls_meta = {}, {}
    for state in (0, 1):
        with qml.QueuingManager.stop_recording():
            native_control = qml.ctrl(select, control=width, control_values=[state])
        controlled[state], controls_meta[str(state)] = _native_pl_circuit([native_control], width + 1)
    metadata = {
        "backend": "native PennyLane AmplitudeEmbedding + Select(graph)",
        "system_size": n, "logical_address_size": address, "clean_scratch_size": scratch,
        "graph_work_wire_budget": scratch,
        "normalization_alpha": float(sum(abs(c) for _, c in terms)),
        "prepare": prepare_meta, "select": select_meta, "controlled_select": controls_meta,
        "prepare_has_outer_control": False,
        "controlled_select_native_graph_before_leaf_expansion": True,
        "partial_select": True, "placement_reselection": False,
        "matrix_synthesis_used": False, "construction_seconds": time.perf_counter() - started,
    }
    return prepare, plain_select, controlled, metadata


def build_pl_select(case, *, work_wires=None):
    """Return native plain/CSELECT and metadata; root may supply shared PREPARE."""
    _, plain, controlled, metadata = build_pl_components(case, work_wires=work_wires)
    return plain, controlled, metadata


def build_pl_query(case, *, work_wires=None):
    prepare, plain_select, controlled_selects, metadata = build_pl_components(case, work_wires=work_wires)
    plain = QuantumCircuit(prepare.num_qubits)
    plain.compose(prepare, inplace=True)
    plain.compose(plain_select, inplace=True)
    plain.compose(prepare.inverse(), inplace=True)
    controlled = {state: compose_controlled_query(prepare, select,
        system_size=metadata["system_size"], address_size=metadata["logical_address_size"],
        scratch_size=metadata["clean_scratch_size"]) for state, select in controlled_selects.items()}
    return plain, controlled, metadata


def _construct_qrisp_select(case, state=None):
    """The exact native from_lcu conjugand, with a fresh branch list."""
    terms = _terms(case)
    n, address = len(terms[0][0]), (len(terms) - 1).bit_length()

    def branch(label, coefficient):
        def unitary(system):
            if coefficient.real < 0:
                gphase(np.pi, system[0])
            if abs(coefficient.imag) > 1e-12:
                raise ValueError("Retained native Qrisp cases have real coefficients")
            for bit, letter in enumerate(reversed(label)):
                if letter != "I":
                    {"X": x, "Y": y, "Z": z}[letter](system[bit])
        return unitary

    branches = [branch(label, coefficient) for label, coefficient in terms]
    signal = QuantumBool(name="signal") if state is not None else None
    system = QuantumVariable(n, name="system")
    index = QuantumFloat(address, name="index")
    with redirect_stdout(io.StringIO()):
        if state is None:
            q_switch(index, branches, system)
        else:
            with control(signal[0], ctrl_state=state):
                q_switch(index, branches, system)
    permanent = [index] if signal is None else [signal, index]
    packed, ports = native.native.pack_fixed_ports(system.qs.copy(), permanent, system)
    output = legacy.canonicalize(packed, ports["system_qubits"], controlled=signal is not None)
    shift = int(signal is not None)
    projection_width = address + shift
    remap = {old: n + old - shift for old in range(shift, projection_width)}
    remap.update({old: n + address + old - projection_width for old in ports["clean_scratch_qubits"]})
    remap.update({old: bit for bit, old in enumerate(ports["system_qubits"])})
    if signal is not None:
        remap[0] = output.num_qubits - 1
    return output, {
        "system_size": n, "address_size": address, "scratch_size": len(ports["clean_scratch_qubits"]),
        "ctrl_state": state, "packed_ports": ports,
        "canonical_release_checks": [{"position": record["position"],
            "scratch_slots": [remap[slot] for slot in record["scratch_slots"]]}
            for record in ports["release_checks"]],
        "backend": "native qrisp q_switch; exact positive-magnitude from_lcu SELECT conjugand",
        "fresh_branch_list": True, "native_custom_control": state is not None,
        "outer_control_enters_tree_root": state is not None,
        "primitive_control_added_after_lowering": False,
    }


def build_qrisp_select(case):
    """Expose native SELECT separately for the common-H^m large-case PREPARE."""
    started = time.perf_counter()
    plain, plain_meta = _construct_qrisp_select(case)
    controlled, control_meta = {}, {}
    for state in (0, 1):
        controlled[state], control_meta[state] = _construct_qrisp_select(case, state)
    n, address = plain_meta["system_size"], plain_meta["address_size"]
    scratch = max(plain_meta["scratch_size"], *(meta["scratch_size"] for meta in control_meta.values()))
    plain = native.pad_query(plain, n, address, scratch)
    controlled = {state: native.pad_query(query, n, address, scratch, controlled=True)
        for state, query in controlled.items()}
    metadata = {"backend": "native q_switch from_lcu conjugand",
        "system_size": n, "logical_address_size": address, "clean_scratch_size": scratch,
        "plain": plain_meta, "controlled": {str(state): meta for state, meta in control_meta.items()},
        "native_q_switch_custom_control": True, "fresh_branch_list_per_control_context": True,
        "control_enters_tree_root": True, "prepare_not_included": True,
        "matrix_synthesis_used": False, "construction_seconds": time.perf_counter() - started}
    return plain, controlled, metadata


def build_qrisp_query(case):
    """Fresh from_lcu for each plain/control context; native custom_control."""
    started = time.perf_counter()
    plain, plain_meta = native.construct_native_query(case)
    controlled, controls_meta = {}, {}
    for state in (0, 1):
        controlled[state], controls_meta[state] = native.construct_native_query(case, state)
    n, address = plain_meta["system_size"], plain_meta["address_size"]
    assert all(item["address_size"] == address for item in controls_meta.values())
    scratch = max(plain_meta["scratch_size"], *(meta["scratch_size"] for meta in controls_meta.values()))
    plain = native.pad_query(plain, n, address, scratch)
    controlled = {state: native.pad_query(query, n, address, scratch, controlled=True)
                  for state, query in controlled.items()}
    metadata = {
        "backend": "qrisp.block_encodings.BlockEncoding.from_lcu",
        "system_size": n, "logical_address_size": address, "clean_scratch_size": scratch,
        "normalization_alpha": float(case["alpha"]),
        "plain": plain_meta, "controlled": {str(state): meta for state, meta in controls_meta.items()},
        "prepare_has_outer_control": False,
        "prepare_control_policy": "native conjugate(prepare); outer control only on q_switch conjugand",
        "native_q_switch_custom_control": True,
        "q_switch_method": "native auto selects balanced tree",
        "control_enters_tree_root": True,
        "fresh_from_lcu_per_control_context": True,
        "primitive_control_added_after_lowering": False,
        "public_qrisp_QSVT_called": False, "placement_reselection": False,
        "matrix_synthesis_used": False, "construction_seconds": time.perf_counter() - started,
    }
    return plain, controlled, metadata


def build_ours_query(case):
    """Freeze the standalone winner before adding symbolic residual control."""
    started = time.perf_counter()
    terms = _terms(case)
    operator = Matrixsum([(PauliAtom(label, phase=coefficient / abs(coefficient)), abs(coefficient))
                          for label, coefficient in terms])
    with redirect_stdout(io.StringIO()):
        standalone = BlockEncoding(operator, placement_options={}).circuit_components("Matrix-order")
        frozen = replace(standalone, placement_candidates=())
        parts = frozen.with_outer_control(1)
    system = list(parts.circuit.metadata["system_qubits"])
    plain = legacy.canonicalize(parts.circuit, system)
    controlled = {state: legacy.canonicalize(parts.controlled(1, ctrl_state=state),
        [q + 1 for q in system], controlled=True) for state in (0, 1)}
    standalone_matrix_order = standalone.circuit.metadata["matrix_order"]
    controlled_matrix_order = parts.circuit.metadata["matrix_order"]
    assert standalone_matrix_order["placement"] == controlled_matrix_order["placement"]
    assert standalone_matrix_order["selected_candidate"] == controlled_matrix_order["selected_candidate"]
    assert not parts.placement_candidates
    n = len(terms[0][0])
    address = int(parts.circuit.metadata["shared_address_qubits"])
    scratch = int(parts.circuit.metadata["scratch_qubits"])
    metadata = {
        "backend": "block_encoding.BlockEncoding.circuit_components('Matrix-order')",
        "system_size": n, "logical_address_size": address, "clean_scratch_size": scratch,
        "normalization_alpha": float(case["alpha"]),
        "standalone_metadata": standalone.circuit.metadata,
        "controlled_context_metadata": parts.circuit.metadata,
        "fixed_placement": standalone_matrix_order["placement"],
        "standalone_selected_candidate": standalone_matrix_order["selected_candidate"],
        "candidate_count_before_freezing": len(standalone.placement_candidates),
        "candidate_count_after_freezing": len(parts.placement_candidates),
        "placement_reselection": False, "standalone_placement_preserved": True,
        "prepare_has_outer_control": False,
        "symbolic_residual_control_before_standard_lowering": True,
        "matrix_synthesis_used": False, "construction_seconds": time.perf_counter() - started,
    }
    return plain, controlled, metadata


def validate_projectors(angles, address_size):
    """Exhaust all logical addresses and both signal states, flag starts zero."""
    width, domain = address_size + 2, 1 << address_size
    inputs = [value + state * (1 << (width - 1)) for state in (0, 1) for value in range(domain)]
    records = []
    for phi in angles:
        circuit = controlled_logical_projector(phi, system_size=0,
            address_size=address_size, scratch_size=0)
        actual = common._arbitrary_columns(circuit, inputs)
        expected = np.zeros_like(actual)
        for column, index in enumerate(inputs):
            active, address = index >> (width - 1), index & (domain - 1)
            expected[index, column] = np.exp(1j * float(phi) * (1 if address == 0 else -1)) if active else 1
        error = legacy.max_error(actual, expected)
        records.append({"phi": float(phi), "complex_max_error": error,
            "address_signal_basis_inputs": len(inputs)})
    return {"passed": all(row["complex_max_error"] < TOL for row in records),
        "all_logical_addresses_and_signal_states": True,
        "independent_flag_clean_verified": True, "max_error": max(row["complex_max_error"] for row in records),
        "records": records}


def source_records(inputs, smoke):
    paths = [Path(__file__), Path(inputs), Path(smoke), Path(legacy.__file__),
        Path(common.__file__), Path(native.__file__),
        ROOT / "Baseline_scripts/run_rq3_qrisp_qsvt.py",
        ROOT / "Baseline_scripts/pennylane_lcu_adapter.py", ROOT / "Baseline_scripts/baseline_common.py",
        ROOT / "src/experiment_protocol.py", Path(inspect.getfile(qml.Select)),
        Path(inspect.getfile(qml.AmplitudeEmbedding))]
    paths.extend(ROOT / "src" / name for name in source_identity())
    for module in ("qrisp.block_encodings.block_encoding_base",
        "qrisp.block_encodings.constructors.from_lcu",
        "qrisp.environments.control_environment", "qrisp.environments.custom_control_environment",
        "qrisp.environments.conjugation_environment", "qrisp.alg_primitives.program_control.quantum_switch",
        "qrisp.interface.converter.qiskit_converter"):
        paths.append(Path(importlib.import_module(module).__file__))
    return [file_record(path) for path in dict.fromkeys(path.resolve() for path in paths)]


def run_case(case, route, out_dir, sources, projector_validation):
    started = time.perf_counter()
    directory = out_dir / "cases" / case["id"] / route
    directory.mkdir(parents=True, exist_ok=True)
    dump(directory / "input.json", case)
    print(json.dumps({"case": case["id"], "route": route, "stage": "query_build"}), flush=True)
    builders = {"pl_native": build_pl_query, "qrisp_native": build_qrisp_query, "ours": build_ours_query}
    source, controlled, build = builders[route](case)
    n, address, scratch = build["system_size"], build["logical_address_size"], build["clean_scratch_size"]
    assert np.isclose(build["normalization_alpha"], case["alpha"], atol=1e-12, rtol=0)
    artifacts = {"plain": save_qpy(directory / "queries/plain.qpy", source)}
    for state, query in controlled.items():
        artifacts[f"controlled_{state}"] = save_qpy(directory / f"queries/controlled_{state}.qpy", query)
    print(json.dumps({"case": case["id"], "route": route, "stage": "all_query_inputs",
        "address_system_inputs": 1 << (n + address), "query_width": source.num_qubits}), flush=True)
    if route == "qrisp_native":
        domain, releases = native.validate_native_query_domain(source, controlled, n, address, scratch, build)
    else:
        domain = common.validate_query_domain(source, controlled, n, address, scratch)
        releases = None
    dump(directory / "query_domain_validation.json", {"domain": domain, "releases": releases})
    if not domain["passed"]:
        raise ValueError("Local controlled query failed active/inactive or scratch-domain validation")
    circuit, upper = assemble_aligned_outer(controlled[1], case["angles"],
        system_size=n, address_size=address, scratch_size=scratch)
    source_qpy = save_qpy(directory / "source.qpy", circuit)
    dimension = 1 << n
    block, original_reference = common.logical_source_reference(source, case["angles"], n, address, scratch)
    # The independent reference reserves the same additional clean flag as
    # the experimental circuit, between query workspace and highest signal.
    old_width = source.num_qubits
    expected = np.zeros((1 << (old_width + 2), dimension), dtype=complex)
    expected[:1 << old_width] = original_reference[:1 << old_width]
    expected[1 << (old_width + 1):(1 << (old_width + 1)) + (1 << old_width)] = original_reference[1 << old_width:]
    actual = legacy.columns(circuit, dimension)
    normalized_h = SparsePauliOp.from_list(_terms(case)).to_matrix() / case["alpha"]
    target = legacy.polynomial(normalized_h, case["poly"])
    scratch_mask = sum(1 << bit for bit in range(n + address, old_width + 1))
    bad_scratch = (np.arange(1 << circuit.num_qubits) & scratch_mask) != 0
    checks = {
        "source_block_encoding_error": legacy.max_error(block[:dimension], normalized_h),
        "complete_complex_reference_error": legacy.max_error(actual, expected),
        "physical_polynomial_block_error": legacy.max_error(actual[:dimension], target),
        "reference_physical_polynomial_block_error": legacy.max_error(expected[:dimension], target),
        "all_input_columns_orthogonality_error": legacy.max_error(actual.conj().T @ actual, np.eye(dimension)),
        "query_scratch_and_projector_flag_clean_error": float(np.max(np.abs(actual[bad_scratch]), initial=0.0)),
    }
    np.savez_compressed(directory / "validated_columns.npz", source_query=block,
        independent_reference=expected, circuit=actual, normalized_h=normalized_h, polynomial_target=target)
    row = {"id": case["id"], "name": case["name"], "route": route,
        "terms": case["terms"], "alpha": case["alpha"], "poly": case["poly"], "angles": case["angles"],
        "queries": upper["queries"], "projectors": upper["projectors"],
        "outer": upper["outer"], "build": build, "upper": upper,
        "query_artifacts": artifacts, "source_qpy": source_qpy,
        "query_domain_validation": domain, "release_validation": releases,
        "projector_validation": projector_validation, "checks": checks,
        "source_fingerprints": sources, "system_basis_inputs_checked": dimension,
        "all_complex_output_amplitudes_checked": True, "basis": {}, "passed": False}
    for basis, gates in BASES.items():
        lowered = transpile(circuit, basis_gates=gates, **SETTINGS)
        path = directory / f"{route}_{basis}_opt0.qasm"
        path.write_text(qasm2.dumps(lowered) + "\n")
        phase_path = path.with_suffix(".global_phase.json")
        phase = float(lowered.global_phase)
        dump(phase_path, {"global_phase_radians": phase, "restore_policy": "restore after QASM2 import"})
        lowered_columns = legacy.columns(lowered, dimension)
        restored = qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        restored.global_phase = json.loads(phase_path.read_text())["global_phase_radians"]
        saved_columns = legacy.columns(restored, dimension)
        stats = count_qasm_file(path, gate_set=basis)
        assert set(lowered.count_ops()) <= set(gates) and stats["clifford"]["unknown"] == 0
        metric = {"G": int(stats["metric_total"]), "NC": int(stats["clifford"]["non_clifford"]),
            "CX": int(stats["ops"].get("cx", 0)), "W": lowered.num_qubits, "depth": lowered.depth(),
            "ops": stats["ops"], "phase_sensitive_lowering_error": legacy.max_error(lowered_columns, actual),
            "complete_reference_error": legacy.max_error(lowered_columns, expected),
            "physical_polynomial_block_error": legacy.max_error(lowered_columns[:dimension], target),
            "query_scratch_and_projector_flag_clean_error": float(np.max(np.abs(lowered_columns[bad_scratch]), initial=0.0)),
            "saved_qasm_roundtrip_error": legacy.max_error(saved_columns, lowered_columns),
            "saved_qasm_physical_polynomial_block_error": legacy.max_error(saved_columns[:dimension], target),
            "qasm": legacy.relative(path), "qasm_sha256": file_record(path)["sha256"],
            "global_phase_radians": phase, "global_phase_sidecar": legacy.relative(phase_path),
            "global_phase_sidecar_sha256": file_record(phase_path)["sha256"]}
        row["basis"][basis] = metric
        dump(directory / "summary.partial.json", row)
        print(json.dumps({"case": case["id"], "route": route, "basis": basis,
            **{key: metric[key] for key in ("G", "NC", "W")}}), flush=True)
    errors = list(checks.values()) + [value for metric in row["basis"].values()
        for key, value in metric.items() if key.endswith("error")]
    row["passed"] = domain["passed"] and projector_validation["passed"] and max(errors) < TOL
    row["status"] = "passed" if row["passed"] else "semantic_failure"
    row["seconds"] = time.perf_counter() - started
    dump(directory / "summary.json", row)
    return row


def collect_summary(out_dir, manifest, sources, smoke):
    cases = []
    for case in manifest["cases"]:
        routes = {}
        for route in ROUTES:
            path = out_dir / "cases" / case["id"] / route / "summary.json"
            if path.exists():
                value = json.loads(path.read_text())
                assert value["source_fingerprints"] == sources
                routes[route] = value
        if routes:
            cases.append({"id": case["id"], "name": case["name"], "routes": routes})
    completed = len(cases) == len(common.ACTIVE) and all(set(case["routes"]) == set(ROUTES) for case in cases)
    return {"measured_at": datetime.now(timezone.utc).isoformat(), "completed": completed,
        "passed": bool(cases) and all(row["passed"] for case in cases for row in case["routes"].values()),
        "cases": cases, "source_fingerprints": sources, "sources": source_identity(), "smoke": file_record(smoke),
        "policy": {"outer": "H-C1(U_phi)-C0(U_phi.adjoint)-H", "queries": 10, "projectors": 12,
            "prepare_outer_control": False, "projector_predicate_outer_control": False,
            "projector_rotation": "Csignal RZ(2phi)", "extra_independent_clean_flags": 1,
            "projector_excludes_SELECT_scratch": True, "ours_placement_reselection": False,
            "Qrisp_SELECT_native_custom_control": True, "PL_SELECT_native_graph_control": True,
            "public_qrisp_QSVT_called": False, "base_compile": "common PennyLane default compile passes",
            "basis_gates": BASES, **SETTINGS},
        "versions": {"python": sys.version, "pennylane": qml.__version__,
            "qiskit": qiskit.__version__, "qrisp": version("qrisp")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--smoke", type=Path, default=SMOKE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--route", choices=("all", *ROUTES), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    smoke = json.loads(args.smoke.read_text())
    assert smoke["passed"] and smoke["sources"] == source_identity(), "Fresh smoke and exact backend hashes required"
    manifest = json.loads(args.inputs.read_text())
    assert tuple(case["id"] for case in manifest["cases"]) == common.ACTIVE
    sources = source_records(args.inputs, args.smoke)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump(args.out_dir / "source_manifest.json", sources)
    with tarfile.open(args.out_dir / "source_snapshot.tar.gz", "w:gz") as archive:
        for record in sources:
            path = ROOT / record["path"]
            archive.add(path, arcname=record["path"].lstrip("/"), recursive=False)
    selected = [case for case in manifest["cases"] if not args.cases or case["id"] in args.cases]
    assert not args.cases or {case["id"] for case in selected} == set(args.cases)
    routes = ROUTES if args.route == "all" else (args.route,)
    for case in selected:
        address = (len(case["terms"]) - 1).bit_length()
        projector_validation = validate_projectors(case["angles"], address)
        assert projector_validation["passed"]
        for route in routes:
            path = args.out_dir / "cases" / case["id"] / route / "summary.json"
            if args.resume and path.exists():
                previous = json.loads(path.read_text())
                if previous.get("passed") and previous.get("source_fingerprints") == sources:
                    print(json.dumps({"case": case["id"], "route": route, "stage": "resume_verified"}), flush=True)
                    continue
            try:
                row = run_case(case, route, args.out_dir, sources, projector_validation)
            except Exception as exc:
                row = {"id": case["id"], "name": case["name"], "route": route,
                    "passed": False, "status": "exception", "error": repr(exc),
                    "traceback": traceback.format_exc(), "source_fingerprints": sources}
                dump(path, row)
                print(row["traceback"], flush=True)
            assert source_identity() == smoke["sources"], "Backend changed during aligned QSVT execution"
            dump(args.out_dir / "summary.json", collect_summary(args.out_dir, manifest, sources, args.smoke))
            if not row["passed"]:
                raise SystemExit(1)
    summary = collect_summary(args.out_dir, manifest, sources, args.smoke)
    dump(args.out_dir / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "completed": summary["completed"],
        "summary": legacy.relative(args.out_dir / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
