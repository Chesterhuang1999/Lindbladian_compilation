"""Phase-sensitive checks of large diagonal SELECT counterexamples.

The logical proof simulates only address/scratch. At each system Z operation,
its control is proved definite for every address. Tracking the resulting Z
word proves SELECT for every system state, without enumerating 2**n inputs.
Saved NAM/IBM QASM is additionally exercised with calibrated MPS contraction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, qasm2, qpy
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import PauliGate
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
from experiment_protocol import source_identity
from baseline_common import count_qasm_file
from search_rq3_placement_counterexamples import OUT, dump

TOL = 1e-8
MPS = AerSimulator(method="matrix_product_state", max_parallel_threads=1,
                   max_parallel_experiments=1, matrix_product_state_truncation_threshold=1e-16)


def diagonal_pauli(op, depth=0):
    """Recognize a tensor Z word without allocating its exponential matrix."""
    if depth > 10:
        return None
    if isinstance(op, PauliGate):
        label = str(op.params[0])
        if set(label) <= set("IZ"):
            return {i for i, letter in enumerate(reversed(label)) if letter == "Z"}, 1 + 0j
        return None
    if op.name in ("z", "id"):
        return ({0} if op.name == "z" else set()), 1 + 0j
    if op.definition is None or isinstance(op, ControlledGate):
        return None
    word = set()
    phase = np.exp(1j * float(op.definition.global_phase))
    for item in op.definition.data:
        result = diagonal_pauli(item.operation, depth + 1)
        if result is None:
            return None
        positions, scalar = result
        phase *= scalar
        for i in positions:
            position = op.definition.find_bit(item.qubits[i]).index
            word.symmetric_difference_update({position})
    return word, phase


def small_operations(circuit):
    current = circuit
    for _ in range(20):
        wide = {i.operation.name for i in current.data if i.operation.num_qubits > 9
                and diagonal_pauli(i.operation.base_gate if isinstance(i.operation, ControlledGate)
                                   else i.operation) is None}
        if not wide:
            return current
        current = current.decompose(gates_to_decompose=sorted(wide))
    raise ValueError("Cannot locally expand wide gate definitions")


def apply_local(vectors, matrix, positions, width):
    axes = [width - 1 - q for q in positions[::-1]]
    order = axes + [q for q in range(width) if q not in axes] + [width]
    tensor = vectors.reshape((2,) * width + (vectors.shape[1],))
    moved = tensor.transpose(order)
    transformed = (matrix @ moved.reshape(1 << len(positions), -1)).reshape(moved.shape)
    return transformed.transpose(np.argsort(order)).reshape(vectors.shape)


def verify_logical(circuit, system, addresses, scratch, terms, placement):
    circuit = small_operations(circuit)
    workspace = [q for q in range(circuit.num_qubits) if q not in system]
    assert set(workspace) == set(addresses + scratch)
    work_map = {q: i for i, q in enumerate(workspace)}
    n, m, w = len(system), len(addresses), len(workspace)
    columns = np.arange(1 << m)
    initial = sum(((columns >> j) & 1) << work_map[q] for j, q in enumerate(addresses))
    vectors = np.zeros((1 << w, 1 << m), dtype=complex)
    vectors[initial, columns] = np.exp(1j * float(circuit.global_phase))
    z_words = np.zeros(1 << m, dtype=np.uint64)
    rows = np.arange(1 << w)
    max_local_error = 0.0
    max_condition_error = 0.0
    cache = {}
    for instruction in circuit.data:
        op = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        sys_local = [i for i, q in enumerate(qubits) if q in system]
        base_gate = op.base_gate if isinstance(op, ControlledGate) else op
        recognized = diagonal_pauli(base_gate)
        if recognized is not None and sys_local:
            positions, phase = recognized
            ctrl_count = op.num_ctrl_qubits if isinstance(op, ControlledGate) else 0
            assert all(q in workspace for q in qubits[:ctrl_count])
            if ctrl_count:
                control_values = sum(((rows >> work_map[qubits[i]]) & 1) << i
                                     for i in range(ctrl_count))
                active = control_values == int(op.ctrl_state)
            else:
                active = np.ones(len(rows), dtype=bool)
            expectations = (abs(vectors) ** 2).T @ (1 - 2 * active.astype(int))
            condition_error = float(np.max(np.minimum(abs(expectations - 1), abs(expectations + 1))))
            max_condition_error = max(max_condition_error, condition_error)
            assert condition_error < TOL
            work_phase = np.full(len(rows), phase, dtype=complex)
            for local in positions:
                q = qubits[ctrl_count + local]
                if q in system:
                    z_words[expectations.real < 0] ^= np.uint64(1 << system.index(q))
                else:
                    work_phase *= 1 - 2 * ((rows >> work_map[q]) & 1)
            vectors *= np.where(active, work_phase, 1)[:, None]
            continue
        key = (op.name, op.num_qubits, tuple(map(str, op.params)))
        if key not in cache:
            cache[key] = np.asarray(Operator(op).data)
        matrix = cache[key]
        if not sys_local:
            vectors = apply_local(vectors, matrix, [work_map[q] for q in qubits], w)
            continue
        diagonal = np.diag(matrix)
        max_local_error = max(max_local_error, float(np.max(np.abs(matrix - np.diag(diagonal)))))
        assert max_local_error < TOL, (op.name, "non-diagonal system operation")
        work_local = [i for i in range(len(qubits)) if i not in sys_local]
        local_zero_system = np.zeros(len(rows), dtype=np.int64)
        for i in work_local:
            local_zero_system |= ((rows >> work_map[qubits[i]]) & 1) << i
        base = diagonal[local_zero_system]
        assert np.max(np.abs(np.abs(base) - 1)) < TOL
        local_columns = np.arange(len(diagonal))
        local_base_indices = local_columns.copy()
        for i in sys_local:
            local_base_indices &= ~(1 << i)
        local_reconstruction = diagonal[local_base_indices].copy()
        for i in sys_local:
            ratio = diagonal[local_zero_system | (1 << i)] / base
            assert np.max(np.minimum(abs(ratio - 1), abs(ratio + 1))) < TOL
            # A definite control makes the system action an explicit Pauli Z.
            expectations = (abs(vectors) ** 2).T @ ratio
            condition_error = float(np.max(np.minimum(abs(expectations - 1), abs(expectations + 1))))
            max_condition_error = max(max_condition_error, condition_error)
            assert condition_error < TOL, (op.name, "system gate control is not definite")
            z_words[expectations.real < 0] ^= np.uint64(1 << system.index(qubits[i]))
            local_ratio = diagonal[local_base_indices | (1 << i)] / diagonal[local_base_indices]
            local_reconstruction *= np.where((local_columns >> i) & 1, local_ratio, 1)
        max_local_error = max(max_local_error, float(np.max(abs(diagonal - local_reconstruction))))
        assert max_local_error < TOL, "System phase contains an unaccounted nonlinear term"
        vectors *= base[:, None]
    amplitudes = vectors[initial, columns]
    leakage_vectors = vectors.copy()
    leakage_vectors[initial, columns] = 0
    leakage = float(np.max(abs(leakage_vectors)))
    phase_error = 0.0
    for label, real, imag in terms:
        assert set(label) <= set("IZ")
        address = int(placement[label], 2)
        mask = sum(1 << j for j, letter in enumerate(reversed(label)) if letter == "Z")
        assert int(z_words[address]) == mask, (label, address, int(z_words[address]))
        coefficient = complex(real, imag)
        phase_error = max(phase_error, float(abs(amplitudes[address] - coefficient / abs(coefficient))))
    assert max(leakage, phase_error) < TOL
    return {"passed": True, "logical_addresses_checked": 1 << m,
            "system_dimension_covered_symbolically": 1 << n,
            "workspace_return_max_error": leakage, "coefficient_phase_max_error": phase_error,
            "local_diagonal_factorization_max_error": max_local_error,
            "system_control_definiteness_max_error": max_condition_error,
            "method": "exhaustive address/workspace simulation; symbolic system Pauli Z words",
            "scratch_zero_invariance_on_entire_input_domain": True}, (z_words, amplitudes)


def tensor_amplitude(mps, index):
    gammas, lambdas = mps
    amplitude = np.asarray([[1.0 + 0j]])
    for bit, gamma in enumerate(gammas):
        amplitude = amplitude @ gamma[(index >> bit) & 1]
        if bit < len(lambdas):
            amplitude *= np.asarray(lambdas[bit])[None, :]
    return complex(amplitude.item())


def calibrate():
    from qiskit import transpile
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.rz(.47, 3)
    circuit.ry(.31, 1)
    circuit.cp(.29, 1, 3)
    circuit.global_phase = .19
    expected = Statevector.from_instruction(circuit).data
    compiled = transpile(circuit, MPS, optimization_level=0, qubits_initially_zero=False)
    compiled.save_matrix_product_state(label="mps")
    result = MPS.run(compiled).result()
    actual = [tensor_amplitude(result.data(0)["mps"], i) for i in range(16)]
    error = float(np.max(abs(np.asarray(actual) - expected)))
    assert error < TOL
    return {"passed": True, "amplitudes_checked": 16, "statevector_max_error": error}


def verify_lowered(qc, system, addresses, z_words, phases):
    n = len(system)
    rng = random.Random(20261014)
    system_values = [0, (1 << n) - 1, *[rng.randrange(1 << n) for _ in range(2)]]
    jobs, expected = [], []
    for address in range(len(z_words)):
        for system_value in system_values:
            index = sum(((address >> j) & 1) << q for j, q in enumerate(addresses))
            index |= sum(((system_value >> j) & 1) << q for j, q in enumerate(system))
            phase = phases[address] * (-1) ** ((system_value & int(z_words[address])).bit_count() % 2)
            probe = QuantumCircuit(qc.num_qubits)
            for q in range(qc.num_qubits):
                if (index >> q) & 1:
                    probe.x(q)
            probe.compose(qc, inplace=True)
            probe.save_matrix_product_state(label="mps")
            jobs.append(probe)
            expected.append((index, phase))
    errors = []
    for start in range(0, len(jobs), 32):
        result = MPS.run(jobs[start:start + 32]).result()
        assert result.success
        for j, (index, phase) in enumerate(expected[start:start + 32]):
            amplitude = tensor_amplitude(result.data(j)["mps"], index)
            errors.append(float(abs(amplitude - phase)))
    error = max(errors)
    assert error < TOL, error
    return {"passed": True, "all_addresses_checked": len(z_words),
            "system_basis_probes_per_address": len(system_values), "basis_inputs_checked": len(jobs),
            "system_values": system_values, "expected_unit_amplitude_max_error": error,
            "simulator": "Aer MPS; directly contract saved tensors, including global phase",
            "unit_amplitude_checks_include_clean_scratch_and_all_output_wires": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    smoke = json.loads((OUT / "smoke.json").read_text())
    assert smoke["passed"] and source_identity() == smoke["sources"]
    calibration = calibrate()
    dump(OUT / "mps_calibration.json", calibration)
    for name in args.cases:
        directory = OUT / "cases" / name
        record = json.loads((directory / "result.json").read_text())
        checks = {}
        for method, entry in record["methods"].items():
            begun = time.perf_counter()
            with (directory / method / "select.qpy").open("rb") as handle:
                qc = qpy.load(handle)[0]
            meta = entry["metadata"]
            system = meta["system_qubits"]
            addresses = meta["projection_qubits"] if method == "qrisp" else meta["address_qubits"]
            scratch = meta["clean_scratch_qubits"]
            width = len(addresses)
            placement = (meta["placement"] if method == "ours" else
                         {label: format(i, f"0{width}b") for i, (label, _, _) in enumerate(record["terms"])})
            logical, (words, phases) = verify_logical(qc, system, addresses, scratch, record["terms"], placement)
            lowered = {}
            for basis, metrics in entry["basis"].items():
                qasm_path = ROOT / metrics["qasm"]
                recount = count_qasm_file(qasm_path, gate_set=basis)
                assert (recount["metric_total"], recount["clifford"]["non_clifford"], recount["num_qubits"]) == (
                    metrics["G"], metrics["NC"], metrics["W"])
                saved = qasm2.load(qasm_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
                saved.global_phase = json.loads(qasm_path.with_suffix(".global_phase.json").read_text())["radians"]
                lowered[basis] = verify_lowered(saved, system, addresses, words, phases)
            checks[method] = {"logical": logical, "lowerings": lowered,
                              "seconds": time.perf_counter() - begun}
            print(json.dumps({"case": name, "method": method, "passed": True,
                              "max_error": max(v["expected_unit_amplitude_max_error"] for v in lowered.values()),
                              "seconds": round(checks[method]["seconds"], 3)}), flush=True)
            dump(directory / "verification.json", {"case": name, "methods": checks, "calibration": calibration})


if __name__ == "__main__":
    main()
