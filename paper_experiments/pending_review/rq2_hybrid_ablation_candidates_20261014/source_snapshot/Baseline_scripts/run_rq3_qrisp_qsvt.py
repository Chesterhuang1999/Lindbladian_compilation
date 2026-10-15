#!/usr/bin/env python3
"""Refresh the seven frozen QSVT cases through Qrisp's native QSVT API.

Only the angle provider is replaced: Qrisp's native reflection, alternating
block/inverse queries, and physical real-polynomial extraction stay active.
System and projection ports are permanently reserved.  Temporaries created
and deleted by Qrisp reuse disjoint clean scratch by their explicit lifetimes;
no QuantumSession zero-input allocator is invoked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from hashlib import sha256
import importlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, qasm2, qpy, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Baseline_scripts"))

from baseline_common import count_qasm_text
from block_encoding import BlockEncoding as OursBlockEncoding
from channel_IR import Matrixsum, PauliAtom
from qrisp import QuantumCircuit as QrispCircuit
from qrisp import QuantumVariable, gphase, x, y, z
from qrisp.block_encodings import BlockEncoding as NativeBlockEncoding


QSVT_MODULE = importlib.import_module("qrisp.algorithms.gqsp.qsvt")
OUT = ROOT / "Baseline_results/rq3_refresh_20261013/qrisp_qsvt"
INPUT = ROOT / "Baseline_results/rq3_refresh_20261013/qsvt_inputs.json"
BASES = {"nam": ["rz", "h", "x", "cx"], "ibm": ["u1", "u2", "u3", "cx"]}
SETTINGS = dict(optimization_level=0, seed_transpiler=7, qubits_initially_zero=False)
TOL = 2e-8


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def relpath(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def source_hashes():
    resources = [ROOT / "src" / name for name in (
        "block_encoding.py", "placement_search.py", "matrix_order_pipeline.py", "channel_IR.py",
        "mobius_residual_fixed.py", "subroutine.py")]
    resources += [Path(__file__), ROOT / "Baseline_scripts/baseline_common.py", INPUT]
    resources += [Path(QSVT_MODULE.__file__)]
    for module in (
        "qrisp.algorithms.gqsp.gqsp_angles",
        "qrisp.block_encodings.block_encoding_base",
        "qrisp.block_encodings.constructors.from_lcu",
        "qrisp.interface.converter.qiskit_converter",
    ):
        resources.append(Path(importlib.import_module(module).__file__))
    return {relpath(path): sha256(path.read_bytes()).hexdigest() for path in resources}


def freeze_sources(hashes):
    smoke_path = ROOT / "Baseline_results/rq3_refresh_20261013/smoke.json"
    smoke = json.loads(smoke_path.read_text())
    if not smoke.get("passed"):
        raise ValueError("The mandatory archived-interface smoke gate has not passed.")
    checked = {}
    for source, digest in hashes.items():
        if source.startswith("src/"):
            expected = smoke["sources"][Path(source).name]
            if digest != expected:
                raise ValueError(f"{source} changed after the mandatory smoke gate.")
            checked[source] = digest
    snapshots = {}
    for source, digest in hashes.items():
        path = Path(source) if Path(source).is_absolute() else ROOT / source
        if Path(source).is_absolute():
            suffix = Path(*path.parts[path.parts.index("qrisp"):])
            snapshot = OUT / "source_snapshot" / "package" / suffix
        else:
            snapshot = OUT / "source_snapshot" / "project" / source
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        if sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError("Source changed during snapshot creation.")
        snapshots[source] = {"sha256": digest, "snapshot": relpath(snapshot)}
    smoke_snapshot = OUT / "source_snapshot" / "smoke.json"
    smoke_snapshot.write_bytes(smoke_path.read_bytes())
    manifest = {
        "snapshots": snapshots, "smoke_gate": relpath(smoke_path),
        "smoke_sha256": sha256(smoke_path.read_bytes()).hexdigest(),
        "smoke_snapshot": relpath(smoke_snapshot),
        "smoke_passed": True, "backend_fingerprint_matches_smoke": True,
        "matched_backend_sources": checked,
    }
    dump(OUT / "source_manifest.json", manifest)
    return manifest


@contextmanager
def frozen_angles(case):
    """Native Qrisp applies phi[d] first; PL's saved list is time ordered."""
    provider = QSVT_MODULE.qsvt_angles
    phi = np.asarray(case["angles"], dtype=float)[::-1].copy()

    def fixed_provider(coefficients, parity="odd"):
        expected = np.polynomial.chebyshev.poly2cheb(case["poly"])
        if parity != "odd" or not np.allclose(np.asarray(coefficients), expected, atol=1e-12, rtol=0):
            raise ValueError("Native QSVT requested a different polynomial or parity.")
        return phi.copy(), 1.0

    QSVT_MODULE.qsvt_angles = fixed_provider
    try:
        yield phi
    finally:
        QSVT_MODULE.qsvt_angles = provider


def native_block(case):
    def pauli(label, coefficient):
        def unitary(system):
            if coefficient < 0:
                gphase(np.pi, system[0])
            for bit, letter in enumerate(reversed(label)):
                if letter != "I":
                    {"X": x, "Y": y, "Z": z}[letter](system[bit])
        return unitary

    coefficients = np.asarray([abs(c) for _, c in case["terms"]], dtype=float)
    be = NativeBlockEncoding.from_lcu(
        coefficients, [pauli(label, c) for label, c in case["terms"]], is_hermitian=True)
    return be, {"backend": "qrisp.block_encodings.BlockEncoding.from_lcu",
                "coefficient_convention": "positive magnitudes; signs in Pauli unitaries",
                "alpha": float(be.alpha)}


def ours_block(case):
    n = len(case["terms"][0][0])
    matrix_sum = Matrixsum([(PauliAtom(label), c) for label, c in case["terms"]])
    parts = OursBlockEncoding(matrix_sum, placement_options={}).circuit_components("Matrix-order").system_last(n)
    source = parts.circuit
    # Preserve the canonical high-level gate definitions so the final circuit
    # can be lowered independently to each hardware basis.  In particular,
    # do not preselect the NAM decomposition before the IBM experiment.
    imported = QrispCircuit.from_qiskit(source)
    system_qubits = list(range(source.num_qubits - n, source.num_qubits))
    bridge_validation_started = time.perf_counter()
    canonical_columns, _ = simulation_columns(source, system_qubits)
    imported_columns, _ = simulation_columns(imported.to_qiskit(), system_qubits)
    bridge_validation_seconds = time.perf_counter() - bridge_validation_started
    bridge_error = float(np.max(np.abs(canonical_columns - imported_columns)))
    if bridge_error >= TOL:
        raise ValueError("Qiskit-to-Qrisp bridge did not preserve canonical block amplitudes and phase.")
    gate = imported.to_gate(name="OursCanonicalBE")
    workspace = source.num_qubits - n

    def unitary(ancilla, system):
        system.qs.append(gate, ancilla.reg + system.reg)

    be = NativeBlockEncoding(float(case["alpha"]),
                             [QuantumVariable(workspace).template()], unitary)
    return be, {
        "backend": "block_encoding.BlockEncoding.circuit_components('Matrix-order')",
        "placement_options": {}, "alpha": float(be.alpha),
        "width": source.num_qubits,
        "canonical_metadata": source.metadata,
        "qiskit_to_qrisp_phase_radians": float(source.global_phase),
        "qiskit_to_qrisp_bridge": "direct canonical high-level circuit; no basis prelowering",
        "qiskit_to_qrisp_bridge_max_abs_error": bridge_error,
        "qiskit_to_qrisp_bridge_inputs": 1 << n,
        "bridge_validation_seconds": bridge_validation_seconds,
        "projection": "all canonical workspace wires, including clean expansion scratch",
    }


def pack_fixed_ports(logical, ancillas, system):
    """Recycle only allocated/deallocated temporaries; reserve all input ports."""
    projection = [q for variable in ancillas for q in variable.reg]
    system_qubits = list(system.reg)
    permanent = projection + system_qubits
    positions = {q: i for i, q in enumerate(permanent)}
    if len(positions) != len(permanent):
        raise ValueError("Projection and system ports overlap.")
    scratch_map, free, next_slot = {}, [], len(permanent)
    mapped, checkpoints = [], []
    for instruction in logical.data:
        name = instruction.op.name
        if instruction.clbits:
            raise ValueError("QSVT unitary must not contain classical operations.")
        if name == "qb_alloc":
            for q in instruction.qubits:
                if q in positions:
                    continue
                if q in scratch_map:
                    raise ValueError("A temporary was allocated twice without release.")
                if free:
                    slot = min(free)
                    free.remove(slot)
                else:
                    slot = next_slot
                    next_slot += 1
                scratch_map[q] = slot
            continue
        if name == "qb_dealloc":
            slots = []
            for q in instruction.qubits:
                if q in positions:
                    raise ValueError("Native QSVT released a permanent input port.")
                slot = scratch_map.pop(q)
                free.append(slot)
                slots.append(slot)
            checkpoints.append({"position": len(mapped), "scratch_slots": slots})
            continue
        qargs = [positions[q] if q in positions else scratch_map[q] for q in instruction.qubits]
        mapped.append((instruction.op, qargs))
    if scratch_map:
        raise ValueError("Qrisp left live temporaries at the end of QSVT.")

    # Put system last after lifetime assignment, keeping projection before scratch.
    projection_width = len(projection)
    system_width = len(system_qubits)
    scratch_width = next_slot - len(permanent)
    remap = {i: i for i in range(projection_width)}
    remap.update({projection_width + i: projection_width + scratch_width + i
                  for i in range(system_width)})
    remap.update({len(permanent) + i: projection_width + i for i in range(scratch_width)})
    packed = QrispCircuit(next_slot)
    for op, qargs in mapped:
        packed.append(op, [packed.qubits[remap[i]] for i in qargs])
    for checkpoint in checkpoints:
        checkpoint["scratch_slots"] = [remap[i] for i in checkpoint["scratch_slots"]]
    qc = packed.to_qiskit()
    if len(qc.data) != len(mapped):
        raise ValueError("Qrisp export changed operation boundaries used for clean-scratch checks.")
    metadata = {
        "export": "fixed logical input ports; temporary-only lifetime reuse",
        "quantum_session_compile_used": False,
        "raw_logical_width": len(logical.qubits), "packed_width": next_slot,
        "projection_qubits": list(range(projection_width)),
        "clean_scratch_qubits": list(range(projection_width, projection_width + scratch_width)),
        "system_qubits": list(range(projection_width + scratch_width, next_slot)),
        "released_temporary_count": len(checkpoints), "release_checks": checkpoints,
    }
    return qc, metadata


def build_qsvt(case, route):
    started = time.perf_counter()
    be, block_metadata = {"qrisp_native": native_block, "ours": ours_block}[route](case)
    if not np.isclose(float(be.alpha), float(case["alpha"]), atol=1e-12, rtol=0):
        raise ValueError("Block normalization does not match frozen alpha.")
    with frozen_angles(case) as phi:
        native = QSVT_MODULE.QSVT(be, np.asarray(case["poly"], dtype=float),
                                  kind="Polynomial", parity="odd", rescale=False)
    # Angle lookup runs inside QSVT construction, while its native circuit is a closure.
    n = len(case["terms"][0][0])
    system = QuantumVariable(n, name="system")
    ancillas = native.create_ancillas()
    # Native prepare's own semantic checks print progress; save clean JSON output.
    with redirect_stdout(io.StringIO()):
        native.unitary(*ancillas, system)
    qc, port_metadata = pack_fixed_ports(system.qs.copy(), ancillas, system)
    elapsed = time.perf_counter() - started - block_metadata.get("bridge_validation_seconds", 0.0)
    metadata = {
        "native_api": "qrisp.gqsp.QSVT", "kind": "Polynomial", "parity": "odd",
        "rescale": False, "input_alpha": float(be.alpha), "output_alpha": float(native.alpha),
        "polynomial": case["poly"], "pennylane_time_ordered_angles": case["angles"],
        "qrisp_angles": phi.tolist(), "conversion": "phi_Qrisp[k] = phi_PL[d-k]",
        "angle_provider": "temporary replacement of module qsvt_angles; native QSVT unchanged",
        "query_count": len(phi) - 1,
        "real_polynomial_extraction": "native signal QuantumBool; H, paired Rz reflections, H",
        "build_seconds": elapsed, "block": block_metadata, "ports": port_metadata,
    }
    qc.metadata = metadata
    return qc, metadata


def simulation_columns(qc, system_qubits, checkpoints=()):
    """Simulate every system basis input with all auxiliary inputs zero."""
    backend = AerSimulator(method="statevector", max_parallel_threads=1,
                           max_parallel_experiments=1)
    probe = QuantumCircuit(qc.num_qubits)
    by_position = {}
    for checkpoint in checkpoints:
        position = checkpoint["position"]
        by_position.setdefault(position, set()).update(checkpoint["scratch_slots"])
    labels = {}
    for i, instruction in enumerate(qc.data, 1):
        probe.append(instruction.operation,
                     [probe.qubits[qc.find_bit(q).index] for q in instruction.qubits])
        if i in by_position:
            label = f"release_{i}"
            labels[label] = sorted(by_position[i])
            probe.save_statevector(label=label)
    probe.global_phase = qc.global_phase
    probe.save_statevector(label="final")
    lowered = transpile(probe, backend=backend, basis_gates=["u", "cx", "save_statevector"], **SETTINGS)
    circuits = []
    for column in range(1 << len(system_qubits)):
        initial = QuantumCircuit(qc.num_qubits)
        for bit, qubit in enumerate(system_qubits):
            if (column >> bit) & 1:
                initial.x(qubit)
        initial.compose(lowered, inplace=True)
        circuits.append(initial)
    result = backend.run(circuits).result()
    if not result.success:
        raise RuntimeError("Aer failed to simulate native QSVT outputs.")
    columns, release_error = [], 0.0
    indices = np.arange(1 << qc.num_qubits)
    for column in range(len(circuits)):
        data = result.data(column)
        columns.append(np.asarray(data["final"], dtype=complex).copy())
        for label, slots in labels.items():
            mask = np.zeros(len(indices), dtype=bool)
            for slot in slots:
                mask |= ((indices >> slot) & 1).astype(bool)
            release_error = max(release_error, float(np.max(np.abs(np.asarray(data[label])[mask]), initial=0)))
    return np.column_stack(columns), release_error


def verify_target(case, qc, metadata, columns):
    system = metadata["ports"]["system_qubits"]
    address_factor = 1 << min(system)
    selected = np.arange(1 << len(system)) * address_factor
    block = columns[selected, :]
    h_normalized = SparsePauliOp.from_list(case["terms"]).to_matrix() / float(case["alpha"])
    target = sum(float(coefficient) * np.linalg.matrix_power(h_normalized, exponent)
                 for exponent, coefficient in enumerate(case["poly"]))
    target_error = float(np.max(np.abs(block - target)))
    scratch_mask = np.zeros(1 << qc.num_qubits, dtype=bool)
    indices = np.arange(len(scratch_mask))
    for qubit in metadata["ports"]["clean_scratch_qubits"]:
        scratch_mask |= ((indices >> qubit) & 1).astype(bool)
    scratch_error = float(np.max(np.abs(columns[scratch_mask, :]), initial=0))
    return {"target_max_abs_error": target_error, "final_clean_scratch_max_amplitude": scratch_error,
            "system_basis_inputs": 1 << len(system), "output_amplitudes_per_input": 1 << qc.num_qubits,
            "target": "P(H/alpha), all projection and scratch outputs postselected zero",
            "phase_sensitive": True}


def run_case(case):
    case_dir = OUT / "cases" / case["id"]
    rows = {}
    for route in ("qrisp_native", "ours"):
        path = case_dir / route
        path.mkdir(parents=True, exist_ok=True)
        qc, metadata = build_qsvt(case, route)
        with (path / "source.qpy").open("wb") as handle:
            qpy.dump(qc, handle)
        dump(path / "circuit_metadata.json", metadata)
        started = time.perf_counter()
        source_columns, release_error = simulation_columns(
            qc, metadata["ports"]["system_qubits"], metadata["ports"]["release_checks"])
        verification = verify_target(case, qc, metadata, source_columns)
        verification["temporary_release_max_amplitude"] = release_error
        verification["source_validation_seconds"] = time.perf_counter() - started
        if max(verification["target_max_abs_error"], verification["final_clean_scratch_max_amplitude"],
               release_error) >= TOL:
            raise ValueError(f"{case['id']}/{route}: source semantic validation failed: {verification}")
        np.save(path / "validated_source_columns.npy", source_columns)
        metrics = {}
        for basis, gates in BASES.items():
            started = time.perf_counter()
            lowered = transpile(qc, basis_gates=gates, **SETTINGS)
            lowering_seconds = time.perf_counter() - started
            text = qasm2.dumps(lowered)
            qasm_path = path / f"{basis}_opt0.qasm"
            qasm_path.write_text(text)
            sidecar = path / f"{basis}_opt0.global_phase.json"
            dump(sidecar, {"global_phase_radians": float(lowered.global_phase),
                           "restore": "add this phase to loaded OpenQASM2 circuit before simulation/control"})
            stats = count_qasm_text(text, path=relpath(qasm_path), gate_set=basis)
            started = time.perf_counter()
            # Verify the saved QASM plus its phase, not just the in-memory circuit.
            restored = qasm2.loads(text)
            restored.global_phase = float(lowered.global_phase)
            columns, _ = simulation_columns(restored, metadata["ports"]["system_qubits"])
            lowering_error = float(np.max(np.abs(columns - source_columns)))
            hardware_validation = verify_target(case, restored, metadata, columns)
            validation_seconds = time.perf_counter() - started
            if lowering_error >= TOL or hardware_validation["target_max_abs_error"] >= TOL:
                raise ValueError(f"{case['id']}/{route}/{basis}: phase-sensitive export validation failed.")
            metrics[basis] = {
                "gate_count": stats["metric_total"], "non_clifford": stats["clifford"]["non_clifford"],
                "width": lowered.num_qubits, "cx_count": int(lowered.count_ops().get("cx", 0)),
                "depth": lowered.depth(), "ops": dict(lowered.count_ops()),
                "basis_lowering_seconds": lowering_seconds,
                "total_compile_seconds": metadata["build_seconds"] + lowering_seconds,
                "verification_seconds": validation_seconds,
                "phase_sensitive_lowering_error": lowering_error,
                "target_max_abs_error": hardware_validation["target_max_abs_error"],
                "global_phase_radians": float(lowered.global_phase),
                "qasm": relpath(qasm_path), "global_phase_sidecar": relpath(sidecar),
            }
        row = {"status": "passed", "metadata": metadata, "verification": verification, "metrics": metrics}
        dump(path / "result.json", row)
        rows[route] = row
        print(json.dumps({"case": case["id"], "route": route, "status": "passed", "metrics": metrics,
                          "verification": verification}, ensure_ascii=False), flush=True)
    ratios = {basis: rows["qrisp_native"]["metrics"][basis]["gate_count"] /
                      rows["ours"]["metrics"][basis]["gate_count"] for basis in BASES}
    result = {"id": case["id"], "name": case["name"], "terms": case["terms"], "alpha": case["alpha"],
              "poly": case["poly"], "routes": rows, "qrisp_over_ours": ratios, "status": "passed"}
    dump(case_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Restrict to a frozen case id.")
    parser.add_argument("--reuse", action="store_true", help="Reuse already passed cases with the current source hashes.")
    args = parser.parse_args()
    inputs = json.loads(INPUT.read_text())
    cases = [case for case in inputs["cases"] if not args.case or case["id"] in args.case]
    hashes = source_hashes()
    manifest = freeze_sources(hashes)
    info = {
        "available": True, "qrisp_version": version("qrisp"),
        "api": "qrisp.gqsp.QSVT", "implementation": str(Path(QSVT_MODULE.__file__)),
        "signature": "QSVT(A, p, kind='Polynomial', parity='odd', rescale=True)",
        "native_block_api": "qrisp.block_encodings.BlockEncoding.from_lcu",
        "source_hashes": hashes,
        "source_manifest": relpath(OUT / "source_manifest.json"),
        "backend_fingerprint_matches_smoke": manifest["backend_fingerprint_matches_smoke"],
        "angle_parameterization": "freeze phi_Qrisp = reversed(saved PL angles); return output alpha 1",
        "native_algorithm": "unchanged Qrisp QSVT reflection/query/Hadamard closure",
        "input_normalization": "alpha = sum(abs(coefficients)), rescale=False",
        "hardware_target": "P(H/alpha) including native signal qubit and complete block queries",
        "downstream": {"qiskit_version": version("qiskit"), "basis_gates": BASES, **SETTINGS},
        "validation": "every system basis input; phase-sensitive full output amplitudes; clean scratch at releases; saved QASM+phase",
        "archive_smoke_gate": "root agent passed smoke_block_encoding_archive.py before dispatch; not repeated here",
        "framework_comparison": "Qrisp-native / Qrisp-with-Ours, separately from the PL framework",
    }
    dump(OUT / "native_api.json", info)
    results = []
    for case in cases:
        existing = OUT / "cases" / case["id"] / "result.json"
        existing_hashes = OUT / "cases" / case["id"] / "source_hashes.json"
        if args.reuse and existing.exists() and existing_hashes.exists() and json.loads(existing_hashes.read_text()) == hashes:
            result = json.loads(existing.read_text())
            if result.get("status") != "passed":
                result = run_case(case)
        else:
            print(json.dumps({"case": case["id"], "status": "running"}), flush=True)
            result = run_case(case)
        dump(existing_hashes, hashes)
        results.append(result)
        dump(OUT / "summary.json", {**info, "cases": results})
    print(json.dumps({"summary": relpath(OUT / "summary.json"), "passed": len(results)}), flush=True)


if __name__ == "__main__":
    main()
