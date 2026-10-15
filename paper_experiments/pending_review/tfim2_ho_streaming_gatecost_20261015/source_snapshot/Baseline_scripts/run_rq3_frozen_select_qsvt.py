"""Lift verified SELECTs into QSVT with one shared, literal control rule.

No placement is rebuilt or rescored. All three routes use the saved NAM
SELECT, the same uniform PREPARE, the same projector phases, and a control
on every primitive gate, including PREPARE. QSVT components are lowered
separately at opt0 and concatenated without cross-component optimization.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]

import numpy as np
from qiskit import QuantumCircuit, qasm2, qpy, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

from baseline_common import count_qasm_file
from experiment_protocol import source_identity
from verify_rq3_diagonal_counterexamples import tensor_amplitude

OUT = ROOT / "Baseline_results/rq3_frozen_select_qsvt_20261014"
SELECTS = ROOT / "Baseline_results/rq3_placement_counterexamples_20261014"
PHASES = ROOT / "Baseline_results/rq3_common_outer_20261014/qsvt_inputs.json"
CASES = ("qrisp_magnetization_32q", "qrisp_ising_zero_field_chain33")
BASES = {"nam": ["rz", "h", "x", "cx"], "ibm": ["u1", "u2", "u3", "cx"]}
SETTINGS = dict(optimization_level=0, seed_transpiler=7, qubits_initially_zero=False)
TOL = 1e-8
SIMULATOR = AerSimulator(method="matrix_product_state", max_parallel_threads=1,
    max_parallel_experiments=1, matrix_product_state_truncation_threshold=1e-16,
    mps_lapack=True)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def rel(path):
    return str(path.relative_to(ROOT))


def file_record(path):
    return {"path": rel(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def controlled_primitives(circuit, signal, width):
    """C(U1 ... Uk), with global phase promoted to a signal phase gate."""
    output = QuantumCircuit(width)
    phase = float(circuit.global_phase)
    if phase:
        output.p(phase, signal)
    for instruction in circuit.data:
        if instruction.clbits:
            raise ValueError("Frozen input must be unitary")
        qargs = [circuit.find_bit(q).index for q in instruction.qubits]
        if signal in qargs:
            raise ValueError("Signal overlaps a query operand")
        output.append(instruction.operation.control(1, annotated=False), [signal] + qargs)
    return output


def projector(phi, address):
    """PCPhase(phi, dim=1): exp(+i phi) on zero, exp(-i phi) elsewhere."""
    qc = QuantumCircuit(address)
    qc.global_phase = -float(phi)
    for q in range(address):
        qc.x(q)
    qc.mcp(2 * float(phi), list(range(address - 1)), address - 1)
    for q in range(address):
        qc.x(q)
    return transpile(qc, basis_gates=BASES["nam"], **SETTINGS)


def embed(circuit, qubits, width):
    qc = QuantumCircuit(width)
    qc.compose(circuit, qubits=qubits, inplace=True)
    return qc


def saved_select(case, method):
    entry = case["methods"][method]
    meta = entry["metadata"]
    original_path = ROOT / entry["basis"]["nam"]["qasm"]
    phase_path = original_path.with_suffix(".global_phase.json")
    circuit = qasm2.load(original_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    circuit.global_phase = json.loads(phase_path.read_text())["radians"]
    counts = count_qasm_file(original_path, gate_set="nam")
    assert (counts["metric_total"], counts["clifford"]["non_clifford"], counts["num_qubits"]) == (
        entry["basis"]["nam"]["G"], entry["basis"]["nam"]["NC"], entry["basis"]["nam"]["W"])
    assert set(circuit.count_ops()) <= set(BASES["nam"])
    system = meta["system_qubits"]
    address = meta["projection_qubits"] if method == "qrisp" else meta["address_qubits"]
    scratch = meta["clean_scratch_qubits"]
    order = system + address + scratch
    assert sorted(order) == list(range(circuit.num_qubits))
    destination = {old: new for new, old in enumerate(order)}
    frozen = embed(circuit, [destination[q] for q in range(circuit.num_qubits)], circuit.num_qubits)
    m = len(address)
    placement = meta["placement"] if method == "ours" else {
        label: format(i, f"0{m}b") for i, (label, _, _) in enumerate(case["terms"])}
    return frozen, placement, {"select_qasm": file_record(original_path),
        "phase_sidecar": file_record(phase_path), "canonical_to_original_qubits": order,
        "original_metrics": entry["basis"]["nam"], "placement": placement,
        "standalone_selected_candidate": meta.get("canonical", {}).get("matrix_order", {}).get("selected_candidate")}


def lower(circuit, basis):
    qc = transpile(circuit, basis_gates=BASES[basis], **SETTINGS)
    assert set(qc.count_ops()) <= set(BASES[basis])
    return qc


def write_circuit(path, circuit, basis):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(qasm2.dumps(circuit) + "\n")
    phase_path = path.with_suffix(".global_phase.json")
    dump(phase_path, {"global_phase_radians": float(circuit.global_phase)})
    restored = qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    restored.global_phase = float(circuit.global_phase)
    assert restored.num_qubits == circuit.num_qubits and len(restored.data) == len(circuit.data)
    error = abs(np.exp(1j * float(restored.global_phase)) - np.exp(1j * float(circuit.global_phase)))
    cache = {}
    for left, right in zip(circuit.data, restored.data, strict=True):
        assert [circuit.find_bit(q).index for q in left.qubits] == [restored.find_bit(q).index for q in right.qubits]
        key = (left.operation.name, tuple(map(str, left.operation.params)), tuple(map(str, right.operation.params)))
        if key not in cache:
            cache[key] = float(np.max(abs(Operator(left.operation).data - Operator(right.operation).data)))
        error = max(error, cache[key])
    assert error < TOL
    count = count_qasm_file(path, gate_set=basis)
    assert count["clifford"]["unknown"] == 0
    return restored, {"G": count["metric_total"], "NC": count["clifford"]["non_clifford"],
        "W": circuit.num_qubits, "CX": count["ops"].get("cx", 0), "ops": count["ops"],
        "qasm": file_record(path), "phase_sidecar": file_record(phase_path),
        "qasm_roundtrip_local_operator_error": float(error)}


def reference(case, placement, system_value, angles):
    m = (case["K"] - 1).bit_length()
    size = 1 << m
    hadamard = np.array([[(-1) ** ((row & col).bit_count() % 2)
        for col in range(size)] for row in range(size)], dtype=complex) / np.sqrt(size)
    diagonal = np.empty(size, dtype=complex)
    for label, real, imag in case["terms"]:
        coefficient = complex(real, imag)
        mask = sum(1 << i for i, letter in enumerate(reversed(label)) if letter == "Z")
        diagonal[int(placement[label], 2)] = (coefficient / abs(coefficient)
            * (-1) ** ((mask & system_value).bit_count() % 2))
    block = (hadamard * diagonal[None, :]) @ hadamard
    value = float(np.mean(diagonal).real)
    assert np.max(abs(block.conj().T @ block - np.eye(size))) < TOL
    assert abs(block[0, 0] - value) < TOL

    def phase(vector, phi):
        result = vector * np.exp(-1j * phi)
        result[0] = vector[0] * np.exp(1j * phi)
        return result

    initial = np.zeros(size, dtype=complex)
    initial[0] = 1
    forward, inverse = initial.copy(), initial.copy()
    for i, phi in enumerate(angles):
        forward = phase(forward, phi)
        if i + 1 < len(angles):
            forward = (block if i % 2 == 0 else block.conj().T) @ forward
    for i in reversed(range(len(angles))):
        inverse = phase(inverse, -angles[i])
        if i:
            inverse = (block.conj().T if (i - 1) % 2 == 0 else block) @ inverse
    expected = np.concatenate(((forward + inverse) / 2, (inverse - forward) / 2))
    target = sum(coefficient * value ** degree for degree, coefficient in enumerate(case["poly"]))
    assert abs(expected[0] - target) < TOL
    assert abs(np.vdot(expected, expected) - 1) < TOL
    return expected, value, target


def spectral_representatives(case):
    if "magnetization" in case["id"]:
        return [(1 << k) - 1 for k in range(case["n"] + 1)]
    result = []
    for walls in range(case["n"]):
        value, bit = 0, 0
        for position in range(1, case["n"]):
            if position <= walls:
                bit ^= 1
            value |= bit << position
        result.append(value)
    return result


def subspace_probability(mps, fixed):
    gammas, lambdas = mps
    env = np.ones((1, 1), dtype=complex)
    for q, gamma in enumerate(gammas):
        matrices = [gamma[fixed[q]]] if q in fixed else gamma
        updated = sum(np.einsum("ij,ia,jb->ab", env, matrix, matrix.conj()) for matrix in matrices)
        if q < len(lambdas):
            lam = np.asarray(lambdas[q])
            updated *= lam[:, None] * lam[None, :]
        env = updated
    assert abs(env.item().imag) < TOL
    return float(env.item().real)


def calibrate():
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.ry(.31, 1)
    circuit.cp(.29, 1, 3)
    circuit.global_phase = .19
    expected = Statevector.from_instruction(circuit).data
    compiled = transpile(circuit, SIMULATOR, **SETTINGS)
    compiled.save_matrix_product_state(label="mps")
    result = SIMULATOR.run(compiled).result()
    assert result.success
    mps = result.data(0)["mps"]
    actual = np.array([tensor_amplitude(mps, i) for i in range(16)])
    error = float(np.max(abs(actual - expected)))
    probability_error = abs(subspace_probability(mps, {0: 1, 1: 0}) -
        sum(abs(expected[i]) ** 2 for i in range(16) if (i & 3) == 1))
    assert max(error, probability_error) < TOL
    return {"passed": True, "complex_amplitude_max_error": error,
        "projected_probability_error": float(probability_error)}


def verify_saved(circuit, case, placement, angles, directory):
    n, m = case["n"], (case["K"] - 1).bit_length()
    scratch = list(range(n + m, circuit.num_qubits - 1))
    signal = circuit.num_qubits - 1
    systems = spectral_representatives(case)
    references = [reference(case, placement, system, angles) for system in systems]
    assert len(set(round(value, 12) for _, value, _ in references)) == len(systems)
    records = []
    for system, (expected, value, target) in zip(systems, references, strict=True):
        probe = QuantumCircuit(circuit.num_qubits)
        for q in range(n):
            if (system >> q) & 1:
                probe.x(q)
        probe.compose(circuit, inplace=True)
        probe.save_matrix_product_state(label="mps")
        result = SIMULATOR.run(probe).result()
        if not result.success:
            raise ValueError("MPS failed on the saved full circuit")
        mps = result.data(0)["mps"]
        indices = [system | (address << n) | (state << signal)
            for state in (0, 1) for address in range(1 << m)]
        actual = np.array([tensor_amplitude(mps, index) for index in indices])
        error = float(np.max(abs(actual - expected)))
        fixed = {q: (system >> q) & 1 for q in range(n)}
        fixed.update({q: 0 for q in scratch})
        probability = subspace_probability(mps, fixed)
        record = {"system_basis": system, "normalized_eigenvalue": value,
            "target_polynomial": target, "full_workspace_output_max_error": error,
            "physical_polynomial_block_error": float(abs(actual[0] - target)),
            "system_and_scratch_invariance_probability_error": float(abs(1 - probability))}
        assert max(error, record["physical_polynomial_block_error"], abs(1 - probability)) < TOL, record
        records.append(record)
        dump(directory / "verification.partial.json", {"records": records})
        print(json.dumps({"case": case["id"], "stage": "spectral_probe", "probe": len(records),
            "of": len(systems), "eigenvalue": value, "error": error}), flush=True)
    return {"passed": True, "basis_representatives": len(systems),
        "all_distinct_spectral_values_covered": True,
        "all_address_signal_amplitudes_per_probe": 2 << m,
        "full_output_max_error": max(r["full_workspace_output_max_error"] for r in records),
        "physical_polynomial_block_max_error": max(r["physical_polynomial_block_error"] for r in records),
        "system_scratch_invariance_probability_max_error": max(r["system_and_scratch_invariance_probability_error"] for r in records),
        "full_system_basis_exhaustively_simulated": False,
        "entire_system_domain_SELECT_proof": "previous symbolic all-address proof; generic gate control preserves both signal sectors exactly",
        "simulator": "Aer MPS, LAPACK SVD, truncation threshold 1e-16; complex amplitudes via direct tensor contraction",
        "records": records}


def run_case(case, method, angles, directory):
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    frozen, placement, provenance = saved_select(case, method)
    n, m = case["n"], (case["K"] - 1).bit_length()
    assert m == 5 and case["K"] == 32
    magnitudes = [abs(complex(real, imag)) for _, real, imag in case["terms"]]
    assert max(magnitudes) - min(magnitudes) < 1e-14
    width, signal = frozen.num_qubits + 1, frozen.num_qubits
    prepare = QuantumCircuit(frozen.num_qubits)
    for q in range(n, n + m):
        prepare.h(q)
    raw_c_prepare = controlled_primitives(prepare, signal, width)
    raw_c_select = controlled_primitives(frozen, signal, width)
    raw_c_phases = [controlled_primitives(embed(projector(phi, m), list(range(n, n + m)),
        frozen.num_qubits), signal, width) for phi in angles]
    result = {"case": case["id"], "method": method, "input": case,
        "frozen_select": provenance, "outer": "H-C1(U_phi)-C0(U_phi.adjoint)-H",
        "query_count": 2 * (len(angles) - 1), "projector_count": 2 * len(angles),
        "prepare": "uniform Hadamards on the five logical addresses; shared by all routes",
        "control_policy": "one signal control on every saved NAM primitive, including every PREPARE gate; explicit controlled global phase",
        "upper_layer_optimization": False, "placement_reselection": False,
        "controlled_residual_resynthesis": False, "basis": {}}
    with (directory / "frozen_select.qpy").open("wb") as handle:
        qpy.dump(frozen, handle)
    for basis in BASES:
        positive = QuantumCircuit(width)
        c_prepare = lower(raw_c_prepare, basis)
        c_select = lower(raw_c_select, basis)
        c_phases = [lower(raw, basis) for raw in raw_c_phases]
        c_query = QuantumCircuit(width)
        c_query.compose(c_prepare, inplace=True)
        c_query.compose(c_select, inplace=True)
        c_query.compose(c_prepare.inverse(), inplace=True)
        for i, phase in enumerate(c_phases):
            positive.compose(phase, inplace=True)
            if i + 1 < len(angles):
                positive.compose(c_query if i % 2 == 0 else c_query.inverse(), inplace=True)
        full = QuantumCircuit(width)
        shell = QuantumCircuit(width)
        shell.h(signal)
        h = lower(shell, basis)
        shell = QuantumCircuit(width)
        shell.x(signal)
        x = lower(shell, basis)
        full.compose(h, inplace=True)
        full.compose(positive, inplace=True)
        full.compose(x, inplace=True)
        full.compose(positive.inverse(), inplace=True)
        full.compose(x, inplace=True)
        full.compose(h, inplace=True)
        expected_gates = 2 * len(positive.data) + 2 * len(h.data) + 2 * len(x.data)
        assert len(full.data) == expected_gates
        with (directory / f"qsvt_{basis}.qpy").open("wb") as handle:
            qpy.dump(full, handle)
        saved, metrics = write_circuit(directory / f"qsvt_{basis}_opt0.qasm", full, basis)
        metrics["components"] = {"controlled_SELECT_G": len(c_select.data),
            "controlled_PREPARE_G": len(c_prepare.data), "controlled_query_G": len(c_query.data),
            "controlled_projector_G": [len(qc.data) for qc in c_phases],
            "positive_branch_G": len(positive.data), "outer_H_and_X_G": 2 * len(h.data) + 2 * len(x.data),
            "literal_additivity_verified": metrics["G"] == expected_gates}
        assert metrics["components"]["literal_additivity_verified"]
        print(json.dumps({"case": case["id"], "method": method, "basis": basis,
            "stage": "counts", "G": metrics["G"], "NC": metrics["NC"], "W": metrics["W"]}), flush=True)
        metrics["validation"] = verify_saved(saved, case, placement, angles, directory / basis)
        result["basis"][basis] = metrics
        dump(directory / "result.partial.json", result)
    result["passed"] = all(v["validation"]["passed"] for v in result["basis"].values())
    result["seconds"] = time.perf_counter() - started
    dump(directory / "result.json", result)
    return result


def main(args):
    smoke = json.loads((args.out / "smoke.json").read_text())
    assert smoke["passed"] and smoke["sources"] == source_identity()
    phases = json.loads(PHASES.read_text())["cases"][0]
    calibration = calibrate()
    rows = []
    for name in args.cases:
        case_path = SELECTS / "cases" / name / "result.json"
        verification_path = case_path.with_name("verification.json")
        original = json.loads(case_path.read_text())
        proof = json.loads(verification_path.read_text())
        assert original["sources"] == source_identity()
        assert all(proof["methods"][method]["logical"]["passed"] for method in args.methods)
        case = {k: v for k, v in original.items() if k != "methods"}
        case.update(poly=phases["poly"], angles=phases["angles"])
        case["source_result"] = file_record(case_path)
        case["source_validation"] = file_record(verification_path)
        case["methods"] = original["methods"]
        for method in args.methods:
            directory = args.out / "cases" / name / method
            path = directory / "result.json"
            if path.exists() and not args.force:
                row = json.loads(path.read_text())
            else:
                row = run_case(case, method, phases["angles"], directory)
            rows.append(row)
            assert smoke["sources"] == source_identity()
            summary = {"scope": "full QSVT on frozen verified NAM SELECTs, with shared literal gate control",
                "runner": file_record(Path(__file__)), "phase_source": file_record(PHASES),
                "sources": smoke["sources"], "smoke": file_record(args.out / "smoke.json"),
                "policy": {"basis": BASES, **SETTINGS, "NC_counter": "baseline_common",
                    "PREPARE": "identical H^5; all gates controlled uniformly",
                    "upper_layer_optimization": False, "qml_compile_after_SELECT": False,
                    "new_placement_search": False, "with_outer_control_called": False,
                    "component_lowering": "opt0; independent components; exact concatenation"},
                "calibration": calibration, "rows": rows, "passed": all(r["passed"] for r in rows)}
            dump(args.out / "summary.partial.json", summary)
    dump(args.out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--methods", nargs="+", choices=("ours", "pl", "qrisp"), default=["ours", "pl", "qrisp"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
