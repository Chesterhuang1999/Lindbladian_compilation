#!/usr/bin/env python3
"""Rerun the historical four-qubit SELECT density sweep with fixed input ports.

Historical seeds and random-positive input generation are retained, extending
each density to ten samples. PREPARE is excluded from both gate counts.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import importlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]

import numpy as np
from qiskit import QuantumCircuit, qasm2, qpy, transpile
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qrisp import QuantumFloat, QuantumVariable, x, y, z
from qrisp.jasp import q_switch

from baseline_common import count_qasm_file
from block_encoding import BlockEncoding
from channel_IR import Matrixsum, PauliAtom
from experiment_protocol import source_identity
from run_rq3_qrisp_qsvt import pack_fixed_ports


OUT = ROOT / "Baseline_results/rq3_random4q_20261014"
SMOKE = ROOT / "Baseline_results/rq3_placement_counterexamples_20261014/smoke.json"
SEED_BASE = {20: 20260703, 64: 20260706, 100: 20290706,
             129: 20360706, 200: 20460706, 256: 20560706}
BASES = {"nam": ["rz", "h", "x", "cx"], "ibm": ["u1", "u2", "u3", "cx"]}
SETTINGS = dict(optimization_level=0, seed_transpiler=7, qubits_initially_zero=False)
TOL = 1e-9
BACKEND = AerSimulator(method="matrix_product_state", max_parallel_threads=1,
                       max_parallel_experiments=1, matrix_product_state_truncation_threshold=0)
HISTORICAL = [ROOT / "Baseline_results" / name / "summary.json" for name in (
    "table_main_select_only_ours_qrisp_v1", "table_main_select_only_random4q100_v1",
    "dense_random_pauli4_block_encoding_random_coeff_v1")]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def rel(path):
    return str(path.relative_to(ROOT))


def generate(terms, seed):
    rng = random.Random(seed)
    labels = ["".join(chars) for chars in itertools.product("IXYZ", repeat=4)]
    if terms != 256:
        labels = rng.sample(labels[1:], terms)
    weights = [rng.random() for _ in labels]
    total = sum(weights)
    return [[label, weight / total] for label, weight in zip(labels, weights, strict=True)]


def old_records():
    records = {}
    for path in HISTORICAL:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        cases = data.get("cases", [])
        cases += data.get("random", {}).get("cases", [])
        for case in cases:
            for sample in case.get("samples", []):
                metrics = sample.get("metrics", sample.get("input_validation", sample.get("validation", {})))
                count = metrics.get("term_count", case.get("num_terms", case.get("case", {}).get("num_terms")))
                seed = sample["seed"]
                if count in SEED_BASE:
                    records.setdefault((count, seed), []).append({"summary": rel(path),
                        "summary_sha256": sha256(path.read_bytes()).hexdigest(),
                        "preview": metrics.get("terms_preview", [])})
    return records


def verify_historical_input(terms, seed, generated, records):
    matches = records.get((terms, seed), [])
    result = []
    for record in matches:
        preview = record["preview"]
        errors = []
        for actual, expected in zip(generated, preview, strict=False):
            if actual[0] != expected["pauli"]:
                raise ValueError("Regenerated label order differs from historical input")
            errors.append(abs(actual[1] - complex(expected["coeff"])))
        if max(errors, default=0) > 1e-15:
            raise ValueError("Regenerated coefficient differs from historical input")
        result.append({**record, "preview_terms_checked": len(preview),
                       "max_coefficient_error": max(errors, default=0)})
    return result


def grid(address_qubits, system_qubits):
    addresses = np.repeat(np.arange(1 << len(address_qubits), dtype=np.int64), 16)
    system = np.tile(np.arange(16, dtype=np.int64), 1 << len(address_qubits))
    states = np.zeros(len(system), dtype=np.int64)
    for bit, physical in enumerate(address_qubits):
        states |= ((addresses >> bit) & 1) << physical
    for bit, physical in enumerate(system_qubits):
        states |= ((system >> bit) & 1) << physical
    return addresses, system, states


def pauli_action(label, system):
    output = system.copy()
    phases = np.ones(len(system), dtype=complex)
    for bit, letter in enumerate(label[::-1]):
        value = (system >> bit) & 1
        if letter in "XY":
            output ^= 1 << bit
        if letter == "Y":
            phases *= 1j * (1 - 2 * value)
        elif letter == "Z":
            phases *= 1 - 2 * value
    return output, phases


def expected_terms(inputs, placement, address_qubits, system_qubits):
    addresses, system, states = grid(address_qubits, system_qubits)
    expected = states.copy()
    phases = np.ones(len(states), dtype=complex)
    occupied = np.zeros(len(states), dtype=bool)
    coefficients = dict(inputs)
    for label, bit_string in placement:
        if label not in coefficients:
            continue
        selected = addresses == int(bit_string, 2)
        occupied |= selected
        output, phase = pauli_action(label, system[selected])
        for bit, physical in enumerate(system_qubits):
            expected[selected] &= ~(1 << physical)
            expected[selected] |= ((output >> bit) & 1) << physical
        phases[selected] = phase
    return addresses, states, expected, phases, occupied


def simulate_monomial(circuit, initial, checkpoints=()):
    """Track every computational-basis column of a monomial SELECT exactly.

    Each operation is independently checked to have one unit-modulus matrix
    entry per column. Complex phases remain explicit; no state preparation or
    clean-system assumption is used. Released scratch is checked before reuse.
    """
    states = initial.copy()
    phases = np.full(len(states), np.exp(1j * float(circuit.global_phase)), dtype=complex)
    cache = {}
    release = {}
    for checkpoint in checkpoints:
        release.setdefault(checkpoint["position"], set()).update(checkpoint["scratch_slots"])
    matrix_error = 0.0
    for position, instruction in enumerate(circuit.data, 1):
        op = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        active = np.ones(len(states), dtype=bool)
        if isinstance(op, ControlledGate):
            controls = qubits[:op.num_ctrl_qubits]
            value = sum(((states >> q) & 1) << bit for bit, q in enumerate(controls))
            active = value == int(op.ctrl_state)
            qubits = qubits[op.num_ctrl_qubits:]
            op = op.base_gate
        if op.num_qubits > 9:
            raise ValueError("Unexpected large top-level operation in SELECT validation")
        key = (op.name, op.num_qubits, tuple(map(str, op.params)))
        if key not in cache:
            matrix = np.asarray(Operator(op).data)
            rows = np.argmax(np.abs(matrix), axis=0)
            nonzero = matrix[rows, np.arange(len(rows))]
            reference = np.zeros_like(matrix)
            reference[rows, np.arange(len(rows))] = nonzero
            error = max(float(np.max(np.abs(matrix - reference))),
                        float(np.max(np.abs(np.abs(nonzero) - 1))))
            if len(set(rows)) != len(rows) or error > 1e-11:
                raise ValueError(f"Non-monomial source SELECT operation: {op.name}, {error}")
            matrix_error = max(matrix_error, error)
            cache[key] = rows, nonzero
        rows, nonzero = cache[key]
        local = sum(((states >> q) & 1) << bit for bit, q in enumerate(qubits))
        output = np.where(active, rows[local], local)
        phases *= np.where(active, nonzero[local], 1)
        for bit, q in enumerate(qubits):
            states &= ~(1 << q)
            states |= ((output >> bit) & 1) << q
        for q in release.get(position, ()):
            if np.any((states >> q) & 1):
                raise ValueError(f"Temporary qubit {q} released dirty at operation {position}")
    return states, phases, matrix_error


def validate_source(circuit, inputs, placement, address_qubits, system_qubits, scratch, releases=()):
    addresses, initial, expected, target_phases, occupied = expected_terms(
        inputs, placement, address_qubits, system_qubits)
    output, phases, matrix_error = simulate_monomial(circuit, initial, releases)
    incorrect = int(np.count_nonzero(output[occupied] != expected[occupied]))
    phase_error = float(np.max(np.abs(phases[occupied] - target_phases[occupied])))
    scratch_dirty = sum(int(np.count_nonzero((output >> q) & 1)) for q in scratch)
    if incorrect or scratch_dirty or phase_error > TOL:
        raise ValueError(f"SELECT semantics failed: {incorrect}, {scratch_dirty}, {phase_error}")
    return {"all_address_system_basis_columns": len(initial),
            "occupied_address_system_basis_columns": int(np.count_nonzero(occupied)),
            "incorrect_occupied_outputs": incorrect, "phase_max_error": phase_error,
            "dirty_scratch_outputs": scratch_dirty,
            "local_monomial_unitary_max_error": matrix_error,
            "released_temporary_checks": len(releases), "phase_sensitive": True,
            "verification": "all logical address and all 16 system inputs; monomial unitary columns"}, (initial, output, phases, occupied)


def native_qrisp(inputs):
    width = (len(inputs) - 1).bit_length()
    index = QuantumFloat(width, name="index")
    system = QuantumVariable(4, name="system")

    def unitary(label):
        def invoke(qv):
            for bit, letter in enumerate(label[::-1]):
                if letter != "I":
                    {"X": x, "Y": y, "Z": z}[letter](qv[bit])
        return invoke

    q_switch(index, [unitary(label) for label, _ in inputs], system)
    circuit, ports = pack_fixed_ports(system.qs.copy(), [index], system)
    placement = [(label, format(i, f"0{width}b")) for i, (label, _) in enumerate(inputs)]
    circuit.metadata = {"native_api": "qrisp.jasp.q_switch", "select_of": "qrisp.BlockEncoding.from_lcu",
                        "public_qrisp_from_lcu_executed": False, "ports": ports, "placement": placement}
    return circuit, ports, placement


def qasm_roundtrip_error(source, saved):
    if source.num_qubits != saved.num_qubits or len(source.data) != len(saved.data):
        raise ValueError("QASM roundtrip changed circuit shape")
    error = abs(np.exp(1j * float(source.global_phase)) - np.exp(1j * float(saved.global_phase)))
    for left, right in zip(source.data, saved.data, strict=True):
        if ([source.find_bit(q).index for q in left.qubits] !=
                [saved.find_bit(q).index for q in right.qubits]):
            raise ValueError("QASM roundtrip changed gate operands")
        error = max(error, float(np.max(np.abs(Operator(left.operation).data - Operator(right.operation).data))))
    return float(error)


def lowering_probes(circuit, columns, seed, exhaustive=False):
    initial, expected, phases, occupied = columns
    choices = np.flatnonzero(occupied).tolist()
    if not exhaustive:
        rng = random.Random(seed)
        choices = sorted(set([choices[0], choices[-1], *rng.sample(choices, min(14, len(choices)))]))
    jobs = []
    for column in choices:
        probe = QuantumCircuit(circuit.num_qubits)
        for q in range(circuit.num_qubits):
            if (initial[column] >> q) & 1:
                probe.x(q)
        probe.compose(circuit, inplace=True)
        # Aer 0.17.2 on this host returns zero from MPS save_amplitudes even
        # when save_probabilities_dict and Statevector give unit probability.
        # Contract the exported MPS tensors directly, preserving global phase.
        probe.save_matrix_product_state(label="mps")
        jobs.append(probe)
    results = BACKEND.run(jobs).result()
    if not results.success:
        raise ValueError("Aer failed to validate lowering")
    error = 0.0
    for i, column in enumerate(choices):
        gammas, lambdas = results.data(i)["mps"]
        amplitude = np.asarray([[1.0 + 0.0j]])
        for bit, gamma in enumerate(gammas):
            amplitude = amplitude @ gamma[(int(expected[column]) >> bit) & 1]
            if bit < len(lambdas):
                amplitude *= np.asarray(lambdas[bit])[None, :]
        error = max(error, float(abs(complex(amplitude.item()) - phases[column])))
    if error > TOL:
        raise ValueError(f"Phase-sensitive lowering failed: {error}")
    return {"phase_sensitive_expected_amplitude_max_error": error, "basis_columns_checked": len(choices),
            "columns": choices, "simulator": "Aer matrix_product_state; direct tensor amplitude contraction",
            "effective_truncation_threshold": results.results[0].metadata["matrix_product_state_truncation_threshold"],
            "max_bond_dimension": results.results[0].metadata["matrix_product_state_max_bond_dimension"],
            "unit_amplitude_certifies_all_other_outputs_zero": True}


def run_method(method, inputs, path, seed, sample):
    begun = time.perf_counter()
    if method == "ours":
        matrix_sum = Matrixsum([(PauliAtom(label, phase=1.0), coefficient) for label, coefficient in inputs])
        be = BlockEncoding(matrix_sum, placement_options={})
        parts = be.circuit_components("Matrix-order")
        circuit = parts.select
        metadata = circuit.metadata
        placement = metadata["matrix_order"]["placement"]
        system = metadata["system_qubits"]
        width = len(placement[0][1])
        address = list(range(width))
        scratch = [q for q in range(circuit.num_qubits) if q not in system + address]
        releases = []
        symbolic_checks, _ = validate_source(parts.symbolic_select, inputs, placement,
            list(range(width)), list(range(width, width + 4)), [])
        selected = next(candidate for candidate in be._matrix_order_candidates
                        if candidate["name"] == metadata["selected_crit"])
        dump(path / "symbolic_residuals.json", {"placement": placement,
            "residuals": selected["mobius"]["g_modes_with_phase"], "validation": symbolic_checks})
    else:
        circuit, ports, placement = native_qrisp(inputs)
        metadata = circuit.metadata
        address, system, scratch = (ports[key] for key in
            ("projection_qubits", "system_qubits", "clean_scratch_qubits"))
        releases = ports["release_checks"]
        symbolic_checks = None
    build_s = time.perf_counter() - begun
    validation, columns = validate_source(circuit, inputs, placement, address, system, scratch, releases)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "source.qpy").open("wb") as handle:
        qpy.dump(circuit, handle)
    dump(path / "circuit_metadata.json", metadata)
    row = {"method": method, "build_s": build_s, "source_validation": validation,
           "symbolic_validation": symbolic_checks, "metadata": metadata, "basis": {}}
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
        checks = lowering_probes(saved, columns, seed, exhaustive=False)
        counted = count_qasm_file(qasm, gate_set=basis)
        assert set(lowered.count_ops()) <= set(gates)
        assert counted["clifford"]["unknown"] == 0 and roundtrip_error < TOL
        row["basis"][basis] = {"G": int(counted["metric_total"]),
            "NC": int(counted["clifford"]["non_clifford"]), "W": lowered.num_qubits,
            "CX": int(counted["ops"].get("cx", 0)), "ops": counted["ops"],
            "lowering_s": seconds, "validation": checks, "qasm_roundtrip_local_operator_error": roundtrip_error,
            "qasm": rel(qasm), "qasm_sha256": sha256(qasm.read_bytes()).hexdigest(),
            "global_phase_radians": float(lowered.global_phase), "phase_sidecar": rel(phase_path)}
    row["passed"] = True
    dump(path / "result.json", row)
    return row


def aggregate(rows):
    result = []
    for count in SEED_BASE:
        group = [row for row in rows if row["terms"] == count]
        if not group:
            continue
        item = {"terms": count, "samples": len(group), "seed_base": SEED_BASE[count], "basis": {}}
        for basis in BASES:
            item["basis"][basis] = {}
            for method in ("qrisp", "ours"):
                item["basis"][basis][method] = {
                    metric: {"mean": statistics.mean(row[method]["basis"][basis][metric] for row in group),
                             "population_std": statistics.pstdev(row[method]["basis"][basis][metric] for row in group),
                             "min": min(row[method]["basis"][basis][metric] for row in group),
                             "max": max(row[method]["basis"][basis][metric] for row in group)}
                    for metric in ("G", "NC", "W")}
            item["basis"][basis]["qrisp_over_ours_ratio_of_means"] = {
                metric: item["basis"][basis]["qrisp"][metric]["mean"] / item["basis"][basis]["ours"][metric]["mean"]
                for metric in ("G", "NC", "W")}
        result.append(item)
    return result


def main(args):
    smoke = json.loads(SMOKE.read_text())
    assert smoke["passed"] and source_identity() == smoke["sources"], "Fresh smoke/source identity gate failed"
    records = old_records()
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprints = {rel(Path(__file__)): sha256(Path(__file__).read_bytes()).hexdigest()}
    for module in ("qrisp.jasp.program_control.q_switch", "qrisp.interface.converter.qiskit_converter"):
        try:
            path = Path(importlib.import_module(module).__file__)
        except ModuleNotFoundError:
            continue
        fingerprints[str(path)] = sha256(path.read_bytes()).hexdigest()
    rows = []
    for count in args.terms:
        for sample in range(args.samples):
            seed = SEED_BASE[count] + sample
            path = args.out / f"terms{count}" / f"sample{sample:02d}_seed{seed}"
            inputs = generate(count, seed)
            historical = verify_historical_input(count, seed, inputs, records)
            dump(path / "input.json", {"num_qubits": 4, "terms": count, "seed": seed,
                "sample": sample, "generator": "random.Random(seed); sample(nonidentity IXYZ labels,K); rng.random positive coefficients / L1",
                "full_support_order": "lexicographic IXYZ including identity" if count == 256 else None,
                "has_identity": count == 256, "inputs": inputs, "historical_matches": historical})
            row = {"terms": count, "sample": sample, "seed": seed,
                   "input": rel(path / "input.json"), "input_sha256": sha256((path / "input.json").read_bytes()).hexdigest()}
            for method in ("qrisp", "ours"):
                cached = path / method / "result.json"
                if cached.exists() and not args.force:
                    row[method] = json.loads(cached.read_text())
                else:
                    log = path / method / "build.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("w") as handle, redirect_stdout(handle):
                        row[method] = run_method(method, inputs, path / method, seed, sample)
                print(json.dumps({"terms": count, "sample": sample, "seed": seed, "method": method,
                    "basis": {basis: {metric: row[method]["basis"][basis][metric] for metric in ("G", "NC", "W")}
                              for basis in BASES}}), flush=True)
            assert source_identity() == smoke["sources"], "Backend changed during run"
            rows.append(row)
            summary = {"experiment": "current-default fixed-input-port random4q SELECT density sweep",
                "scope": "SELECT only; PREPARE and its inverse excluded from both methods",
                "smoke": rel(SMOKE), "sources": smoke["sources"], "source_fingerprints": fingerprints,
                "policy": {"basis": BASES, **SETTINGS, "NC_counter": "baseline_common",
                    "coefficients": "positive uniform draws normalized to L1=1",
                    "sampling": "without replacement excluding identity; 256 complete support includes identity",
                    "selected_candidate": "current full-block NAM-cost default winner; SELECT extracted from it",
                    "std": "population", "samples_per_terms_requested": args.samples,
                    "source_validation": "every logical address x every system basis; clean scratch and release checks",
                    "lowering_validation": "16 deterministic occupied-basis probes per sample and basis; complex expected amplitude"},
                "rows": rows, "aggregate": aggregate(rows), "passed": all(row[m]["passed"] for row in rows for m in ("qrisp", "ours"))}
            dump(args.out / "summary.partial.json", summary)
    dump(args.out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=int, nargs="+", choices=list(SEED_BASE), default=list(SEED_BASE))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
