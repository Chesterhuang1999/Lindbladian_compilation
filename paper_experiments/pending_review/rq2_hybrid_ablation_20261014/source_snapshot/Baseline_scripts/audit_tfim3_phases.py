#!/usr/bin/env python3
"""Audit TFIM-3 model phases, frozen SELECT semantics, and current RQ1 channels.

This targeted audit is independent of the three RQ3 benchmark runs. Start it
only after those runs finish; --after-main-experiments makes that explicit.
The physical reference uses direct dense Pauli matrices, not ChannelIR algebra.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / folder) for folder in ("src", "scripts", "Baseline_scripts")]
OUT = ROOT / "Baseline_results/rq3_refresh_20261013/tfim3_phase_audit"
FROZEN_SELECT = ROOT / "Baseline_results/pennylane_lcu_v1/inputs.json"
FROZEN_RQ1 = ROOT / "paper_experiments/pending_review/rq1_paper_suite_20261010/full_v2/inputs/TFIM-3.json"
HIST_SELECT = ROOT / "Baseline_results/table_main_select_only_ours_qrisp_v1"
OLD_FULL = ROOT / "Baseline_results/block_encoding_new_qrisp_comparison_v1/summary.json"
FIXED_FULL = ROOT / "Baseline_results/block_encoding_new_qrisp_comparison_phasefix_20260706/summary.json"
HIST_COMMIT = "fa39aea"
TOL = 1e-10


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    def native(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=native) + "\n")
    temporary.replace(path)


def complex_pair(value):
    return [float(np.real(value)), float(np.imag(value))]


def max_error(actual, expected):
    return float(np.max(np.abs(actual - expected)))


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def pauli_dense(label):
    matrices = {"I": np.eye(2, dtype=complex),
                "X": np.array([[0, 1], [1, 0]], dtype=complex),
                "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
                "Z": np.diag([1.0, -1.0]).astype(complex)}
    result = np.ones((1, 1), dtype=complex)
    for letter in label:
        result = np.kron(result, matrices[letter])
    return result


def dense_terms(terms):
    return sum(coefficient * pauli_dense(label) for label, coefficient in terms)


def choi_dense(kraus):
    vectors = [matrix.reshape(-1, order="F") for matrix in kraus]
    return sum(np.outer(vector, vector.conj()) for vector in vectors)


def effective_terms(matrix_sum):
    return [(atom.expr, complex(coefficient * atom.phase))
            for atom, coefficient in matrix_sum.instances]


def term_records(terms):
    return [{"pauli": label, "coefficient": complex_pair(value)} for label, value in terms]


def fit_shared_phase(actual, expected):
    overlap = np.vdot(expected, actual)
    phase = complex(overlap / abs(overlap)) if abs(overlap) > 1e-30 else 1.0 + 0j
    return phase, max_error(actual, phase * expected)


def physical_reference():
    n, dt = 3, 0.1
    gamma = float(np.sqrt(0.1) / 2)
    local = lambda letter, site: "".join(letter if i == site else "I" for i in range(n))
    h_terms, l_terms = [], []
    for site in range(n):
        zz = "".join("Z" if i in (site, (site + 1) % n) else "I" for i in range(n))
        h_terms.extend([(zz, -1.0), (local("X", site), -1.0)])
        l_terms.append([(local("X", site), gamma), (local("Y", site), -1j * gamma)])
    hamiltonian = dense_terms(h_terms)
    jumps = [dense_terms(terms) for terms in l_terms]
    jump_products = [jump.conj().T @ jump for jump in jumps]
    product_errors = [max_error(product, 2 * gamma**2 * (np.eye(8) + pauli_dense(local("Z", i))))
                      for i, product in enumerate(jump_products)]
    a0 = np.eye(8) - 1j * dt * hamiltonian - (dt / 2) * sum(jump_products)
    kraus = [a0] + [np.sqrt(dt) * jump for jump in jumps]
    expected_fixed_terms = [("III", -1j * 0.9925)]
    expected_fixed_terms += [(label, 0.1) for label, _ in h_terms]
    expected_fixed_terms += [(local("Z", site), 0.0025j) for site in range(n)]
    return {"n": n, "dt": dt, "gamma": gamma, "h_terms": h_terms, "l_terms": l_terms,
            "kraus": kraus, "choi": choi_dense(kraus), "fixed_a0": -1j * a0,
            "fixed_a0_terms": expected_fixed_terms, "jump_product_errors": product_errors}


def frozen_select_terms():
    case = next(case for case in read(FROZEN_SELECT)["cases"] if case["name"] == "TFIM_3_A0")
    item = case["inputs"][0]
    return [(term["pauli"], complex(term["real"], term["imag"])) for term in item["terms"]], item


def model_audit(reference):
    from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel
    generated, model = build_periodic_tfim_channel(3, reference["dt"])
    generated_dense = [dense_terms(effective_terms(operator)) for operator in generated]
    frozen, item = frozen_select_terms()
    frozen_channel = read(FROZEN_RQ1)
    frozen_dense = [dense_terms([(label, magnitude * complex(real, imag))
                                for label, magnitude, real, imag in row])
                    for row in frozen_channel["input_kraus"]]
    target_fixed = [reference["fixed_a0"]] + reference["kraus"][1:]
    checks = {"jump_product_max_errors": reference["jump_product_errors"],
              "analytical_fixed_a0_error": max_error(dense_terms(reference["fixed_a0_terms"]), reference["fixed_a0"]),
              "current_builder_kraus_errors": [max_error(a, b) for a, b in zip(generated_dense, target_fixed)],
              "frozen_select_a0_error": max_error(dense_terms(frozen), reference["fixed_a0"]),
              "frozen_rq1_kraus_errors": [max_error(a, b) for a, b in zip(frozen_dense, target_fixed)],
              "current_builder_physical_choi_error": relative_error(choi_dense(generated_dense), reference["choi"]),
              "frozen_rq1_physical_choi_error": relative_error(choi_dense(frozen_dense), reference["choi"])}
    errors = [value for values in checks.values() for value in (values if isinstance(values, list) else [values])]
    assert len(generated) == len(frozen_dense) == len(target_fixed) == 4
    return {"passed": max(errors) < TOL, "model": model,
            "definition": {"Hamiltonian": "-sum_periodic(ZZ_neighbours + X_i)",
                           "jump": "gamma*(X_i - i*Y_i)", "gamma": reference["gamma"], "delta_t": reference["dt"],
                           "kraus": "A0=I-i*dt*H-(dt/2)*sum(L_i^dagger L_i), Ai=sqrt(dt)*L_i",
                           "frozen_gauge": "A0_fixed=-i*A0_physical; jump Kraus unchanged"},
            "current_a0_terms": term_records(effective_terms(generated[0])),
            "frozen_select_sha256": item["input_sha256"], "checks": checks}


def preview_error(terms, metrics):
    values = dict(terms)
    return max(float(abs(values[term["pauli"]] - complex(term["coeff"])))
               for term in metrics["terms_preview"])


def historical_mechanism(reference):
    source_ir = subprocess.check_output(["git", "show", f"{HIST_COMMIT}:src/channel_IR.py"], cwd=ROOT).decode()
    source_lcu = subprocess.check_output(["git", "show", f"{HIST_COMMIT}:src/channel_LCU.py"], cwd=ROOT).decode()
    namespace = {"__name__": "isolated_tfim3_historical_ir"}
    exec(compile(source_ir, f"git:{HIST_COMMIT}/src/channel_IR.py", "exec"), namespace)
    function = next(node for node in ast.parse(source_lcu).body
                    if isinstance(node, ast.FunctionDef) and node.name == "Lindblad_to_channel")
    isolated_function = ast.Module(body=[function], type_ignores=[])
    exec(compile(isolated_function, f"git:{HIST_COMMIT}/src/channel_LCU.py", "exec"), namespace)
    old_lind = namespace["Lindbladian"](reference["h_terms"], reference["l_terms"])
    ensemble, success_probability, norms = namespace["Lindblad_to_channel"](old_lind, reference["dt"])
    old_kraus = ensemble.channels[0][1]
    old_terms = effective_terms(old_kraus[0])
    old_dense = [dense_terms(effective_terms(operator)) for operator in old_kraus]
    old_record = next(case for case in read(OLD_FULL)["tfim"]["cases"] if case["case"] == "TFIM_3_A0")
    fixed_record = next(case for case in read(FIXED_FULL)["tfim"]["cases"] if case["case"] == "TFIM_3_A0")
    match_error = preview_error(old_terms, old_record["metrics"])
    norm_error = abs(norms[0] - old_record["metrics"]["l1_norm"])
    phase, phase_error = fit_shared_phase(old_dense[0], reference["fixed_a0"])
    return {"passed": bool(match_error < TOL and norm_error < TOL),
            "mechanism": "Matrixsum.add invokes simplify, whose historical return also eliminates global phase before the next dissipative term is added",
            "historical_source": {"git_ref": HIST_COMMIT,
                                  "channel_IR_sha256": sha256(source_ir.encode()).hexdigest(),
                                  "channel_LCU_sha256": sha256(source_lcu.encode()).hexdigest(),
                                  "exact_old_experiment_snapshot_available": False,
                                  "provenance_scope": "reproduces the mechanism, recorded eight-term preview and l1 norm; does not establish the exact executable source of the historical run"},
            "reproduced_a0_terms": term_records(old_terms), "reproduced_norms": list(map(float, norms)),
            "reproduced_success_probability": float(success_probability),
            "old_record_preview_max_error": match_error, "old_record_norm_error": float(norm_error),
            "phasefixed_record_preview_max_error": preview_error(reference["fixed_a0_terms"], fixed_record["metrics"]),
            "old_a0_best_shared_phase": complex_pair(phase), "old_a0_after_shared_phase_max_error": phase_error,
            "old_channel_physical_choi_relative_error": relative_error(choi_dense(old_dense), reference["choi"])}


def encode_bits(value, bits):
    return sum(((value >> i) & 1) << bit for i, bit in enumerate(bits))


def all_select_inputs(circuit, terms, addresses, system, index_bits, scratch, *, reverse_labels=False):
    from qiskit.quantum_info import Statevector
    dimension = 1 << len(system)
    system_rows = np.asarray([encode_bits(value, system) for value in range(dimension)])
    scratch_mask = sum(1 << bit for bit in scratch)
    index_mask = sum(1 << bit for bit in index_bits)
    rows = np.arange(1 << circuit.num_qubits)
    trials, exact_errors, overlaps = [], [], []
    for label, coefficient in terms:
        phase = coefficient / abs(coefficient)
        matrix = phase * pauli_dense(label[::-1] if reverse_labels else label)
        address = encode_bits(addresses[label], index_bits)
        valid = ((rows & scratch_mask) == 0) & ((rows & index_mask) == address)
        for value in range(dimension):
            initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
            initial[address + system_rows[value]] = 1
            # Statevector phase gates can mutate an input buffer. Expected is independent.
            expected = np.zeros_like(initial)
            expected[address + system_rows] = matrix[:, value]
            actual = Statevector(initial.copy()).evolve(circuit).data.copy()
            overlap = complex(np.vdot(expected, actual))
            overlaps.append(overlap)
            exact_errors.append(max_error(actual, expected))
            trials.append({"pauli": label, "address": addresses[label], "system_input": value,
                           "expected": expected, "actual": actual,
                           "invalid_register_max_amplitude": float(np.max(np.abs(actual[~valid]), initial=0.0))})
    phase_sum = sum(overlaps)
    common = complex(phase_sum / abs(phase_sum)) if abs(phase_sum) > 1e-30 else 1 + 0j
    residuals = [max_error(trial["actual"], common * trial["expected"]) for trial in trials]
    phases = [complex(overlap / abs(overlap)) if abs(overlap) > 1e-30 else 0j for overlap in overlaps]
    errors_by_label = {}
    for trial, exact, residual, phase in zip(trials, exact_errors, residuals, phases):
        record = errors_by_label.setdefault(trial["pauli"], {"address": trial["address"], "basis_inputs": dimension,
                     "exact_max_error": 0.0, "shared_phase_max_error": 0.0, "register_max_amplitude": 0.0,
                     "phase_at_system_zero": None})
        record["exact_max_error"] = max(record["exact_max_error"], exact)
        record["shared_phase_max_error"] = max(record["shared_phase_max_error"], residual)
        record["register_max_amplitude"] = max(record["register_max_amplitude"], trial["invalid_register_max_amplitude"])
        if trial["system_input"] == 0:
            record["phase_at_system_zero"] = complex_pair(phase)
    return {"basis_inputs_checked": len(trials), "index_addresses_checked": len(terms),
            "system": system, "index": index_bits, "scratch": scratch,
            "exact_max_error": max(exact_errors), "shared_global_factor": complex_pair(common),
            "shared_global_angle_radians": float(np.angle(common)), "shared_phase_max_error": max(residuals),
            "branch_phase_spread": max(float(abs(phase - common)) for phase in phases),
            "register_max_amplitude": max(trial["invalid_register_max_amplitude"] for trial in trials),
            "per_term": errors_by_label,
            "exact_passed": max(exact_errors) < TOL, "shared_phase_passed": max(residuals) < TOL}


def load_qasm(path):
    from qiskit import qasm2
    return qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def historical_select_audit():
    from baseline_common import count_qasm_file
    terms, _ = frozen_select_terms()
    archived = next(case for case in read(HIST_SELECT / "summary.json")["single"]["cases"]
                    if case["case"]["name"] == "TFIM_3_A0")
    addresses = {label: i for i, (label, _) in enumerate(terms)}
    directory = HIST_SELECT / "single/TFIM_3_A0"
    results = {}
    for basis, suffix in (("logical", "qiskit"), ("nam", "nam_opt0"), ("ibm", "ibm_opt0")):
        path = directory / f"TFIM_3_A0_qrisp_select_{suffix}.qasm"
        circuit = load_qasm(path)
        system, indices, scratch = [], [], []
        for register in circuit.qregs:
            bits = [circuit.find_bit(bit).index for bit in register]
            target = system if register.name.startswith("sys_") else indices if register.name.startswith("idx_") else scratch
            target.extend(bits)
        assert system == [0, 1, 2] and indices == [3, 4, 5, 6] and scratch == [7, 8, 9, 10]
        checks = all_select_inputs(circuit, terms, addresses, system, indices, scratch, reverse_labels=True)
        results[basis] = {"qasm": str(path.relative_to(ROOT)), "qasm_sha256": sha256(path.read_bytes()).hexdigest(),
                          "width": circuit.num_qubits, "checks": checks}
        if basis != "logical":
            recounted = count_qasm_file(path, gate_set=basis)
            results[basis]["G"] = recounted["metric_total"]
            results[basis]["NC"] = recounted["clifford"]["non_clifford"]
    passed = results["logical"]["checks"]["exact_passed"] and all(result["checks"]["shared_phase_passed"]
                                                                                for result in results.values())
    return {"passed": passed, "method": "all ten occupied addresses times all eight system basis columns, with scratch initially zero",
            "label_mapping": "historical Qrisp maps label character i to sys_i; labels are reversed in the little-endian dense reference",
            "phase_policy": "logical QASM must match complex branch phases exactly; QASM2 lowerings may omit one common global phase, which is fitted across all 80 inputs and explicitly reported",
            "historical_preview_error": preview_error(terms, archived["metrics"]),
            "historical_norm_error": float(abs(sum(abs(c) for _, c in terms) - archived["metrics"]["l1_norm"])),
            "magnitude_evidence_limit": "SELECT determines all ten phase-bearing Pauli branches, but does not independently recover coefficient magnitudes beyond the saved preview and norm",
            "results": results}


def encoded_block(circuit, system):
    from qiskit.quantum_info import Statevector
    rows = np.asarray([encode_bits(value, system) for value in range(1 << len(system))])
    columns = []
    for address in rows:
        initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
        initial[address] = 1
        columns.append(Statevector(initial.copy()).evolve(circuit).data.copy()[rows])
    return np.column_stack(columns)


def current_block_audit(reference):
    from qiskit import qasm2
    from baseline_common import count_qasm_text
    from block_encoding import BlockEncoding
    from channel_IR import Matrixsum, PauliAtom
    from experiment_protocol import transpile_channel_circuit
    terms, _ = frozen_select_terms()
    parts = BlockEncoding(Matrixsum([(PauliAtom(label), coefficient) for label, coefficient in terms]),
                          placement_options={}).circuit_components("Matrix-order")
    metadata = parts.circuit.metadata
    system = metadata["system_qubits"]
    placement = {label: int(address, 2) for label, address in metadata["matrix_order"]["placement"]}
    index_bits = list(range(metadata["selected_k"]))
    scratch = [bit for bit in metadata["workspace_qubits"] if bit not in index_bits]
    alpha = float(sum(abs(value) for _, value in terms))
    target = reference["fixed_a0"] / alpha
    checks = {"logical_select": all_select_inputs(parts.select, terms, placement, system, index_bits, scratch),
              "logical_encoded_block_max_error": max_error(encoded_block(parts.circuit, system), target)}
    archived = read(OUT.parent / "select/TFIM_3_A0/result.json")["inputs"][0]
    metrics = {}
    for basis, gates in (("nam", ["rz", "h", "x", "cx"]), ("ibm", ["u1", "u2", "u3", "cx"])):
        saved = archived["basis"][basis]
        saved_select = load_qasm(ROOT / saved["qasm"])
        saved_select.global_phase = saved["global_phase"]
        select_checks = all_select_inputs(saved_select, terms, placement, system, index_bits, scratch)
        checks[f"{basis}_saved_select"] = select_checks
        lowered_block = transpile_channel_circuit(parts.circuit, basis_gates=gates, optimization_level=0, seed_transpiler=7)
        checks[f"{basis}_encoded_block_max_error"] = max_error(encoded_block(lowered_block, system), target)
        path = OUT / f"current_block_{basis}_opt0.qasm"
        qasm = qasm2.dumps(lowered_block)
        path.write_text(qasm)
        count = count_qasm_text(qasm, gate_set=basis)
        metrics[basis] = {"G": count["metric_total"], "NC": count["clifford"]["non_clifford"],
                          "W": lowered_block.num_qubits, "global_phase": float(lowered_block.global_phase),
                          "qasm": str(path.relative_to(ROOT)), "SELECT_archived_metrics": {key: saved[key] for key in ("G", "NC", "W")}}
    passed = all(value["exact_passed"] if isinstance(value, dict) else value < TOL for value in checks.values())
    return {"passed": passed, "target": "physical first-order TFIM A0 multiplied by the explicit -i Kraus gauge, divided by alpha",
            "alpha": alpha, "metadata": metadata, "checks": checks, "full_block_metrics": metrics}


def recover_channel_choi(circuit, registers, scale):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    system = registers[-1]
    selector = registers[1] if circuit.metadata["structure"] == "basic" else registers[0]
    rows = np.asarray([encode_bits(value, system) for value in range(1 << len(system))])
    successful = np.asarray([rows + encode_bits(address, selector) for address in range(1 << len(selector))])
    recovered = np.empty((len(successful), len(rows), len(rows)), dtype=complex)
    simulator = AerSimulator(method="statevector", max_parallel_threads=1, max_parallel_experiments=1)
    for column, address in enumerate(rows):
        initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
        initial[address] = 1
        probe = QuantumCircuit(circuit.num_qubits)
        probe.set_statevector(initial)
        probe.compose(circuit, inplace=True)
        probe.save_statevector()
        state = np.asarray(simulator.run(probe).result().get_statevector()).copy()
        recovered[:, :, column] = state[successful]
    return scale * choi_dense(list(recovered)), {"system_basis_inputs": len(rows), "selector_outcomes": len(successful)}


def current_rq1_audit(reference):
    from qiskit import qasm2
    from baseline_common import count_qasm_text
    from experiment_protocol import lower_and_count, DEFAULT_TRANSPILE_SEED
    from rq1_pipeline import canonicalize, deserialize_kraus, serialize_kraus, build_pipeline
    raw = deserialize_kraus(read(FROZEN_RQ1)["input_kraus"])
    canonical, canonicalization = canonicalize(raw)
    results = {}
    for pipeline in ("Raw", "Qrisp", "Ours"):
        begun = time.perf_counter()
        circuit, registers, assembly = build_pipeline(deepcopy(canonical), pipeline)
        lowered, metrics = lower_and_count(circuit, rq="RQ1", basis="nam", seed_transpiler=DEFAULT_TRANSPILE_SEED)
        scale = float(circuit.metadata["subnormalization_factor"])
        actual, probes = recover_channel_choi(lowered, registers, scale)
        error = relative_error(actual, reference["choi"])
        path = OUT / f"rq1_{pipeline.lower()}_nam_opt0.qasm"
        qasm = qasm2.dumps(lowered)
        path.write_text(qasm)
        recount = count_qasm_text(qasm, gate_set="nam")
        assert recount["metric_total"] == metrics["G"]
        results[pipeline] = {"passed": error < TOL, "metrics": metrics, "NC": recount["clifford"]["non_clifford"],
                             "Lambda": scale, "registers": registers, "assembly": assembly,
                             "canonicalization": canonicalization, "compiled_kraus": serialize_kraus(deserialize_kraus(assembly["compiled_kraus"])),
                             "verification": {"method": "full unnormalized Choi against independent dense H/L first-order channel", **probes,
                                              "relative_frobenius_error": error, "tolerance": TOL, "passed": error < TOL},
                             "global_phase": float(lowered.global_phase), "qasm": str(path.relative_to(ROOT)),
                             "qasm_sha256": sha256(path.read_bytes()).hexdigest(), "seconds": time.perf_counter() - begun}
        dump(OUT / f"rq1_{pipeline.lower()}.json", results[pipeline])
        print(json.dumps({"stage": "RQ1", "pipeline": pipeline, "G": metrics["G"], "W": metrics["W"],
                          "NC": results[pipeline]["NC"], "Lambda": scale, "physical_choi_error": error}), flush=True)
    return {"passed": all(result["passed"] for result in results.values()), "results": results,
            "interpretation": "Independent per-Kraus global phases are channel gauge freedom; all relative Pauli phases within a Kraus are retained"}


def source_snapshot(sources):
    paths = [ROOT / "src" / name for name in sources]
    paths += [Path(__file__).resolve(), ROOT / "scripts/run_tfim4_channel_lcu_ours_nam.py", FROZEN_SELECT, FROZEN_RQ1]
    with tarfile.open(OUT / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        for name in ("channel_IR.py", "channel_LCU.py"):
            data = subprocess.check_output(["git", "show", f"{HIST_COMMIT}:src/{name}"], cwd=ROOT)
            info = tarfile.TarInfo(f"historical_git_{HIST_COMMIT}/src/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-main-experiments", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages when backend and frozen inputs match")
    args = parser.parse_args()
    if not args.after_main_experiments:
        parser.error("Wait for explicit confirmation that the main experiments finished, then pass --after-main-experiments.")
    from experiment_protocol import source_identity
    sources = source_identity()
    smoke = read(OUT.parent / "smoke.json")
    assert smoke["passed"] and sources == smoke["sources"], "Backend must match the required archive smoke gate."
    OUT.mkdir(parents=True, exist_ok=True)
    previous = read(OUT / "summary.partial.json") if args.resume and (OUT / "summary.partial.json").exists() else {}
    if previous:
        assert previous["sources"] == sources, "Cannot resume stages from a different backend."
        with tarfile.open(OUT / "source_snapshot.tar.gz") as archive:
            for path in (FROZEN_SELECT, FROZEN_RQ1):
                archived = archive.extractfile(str(path.relative_to(ROOT)))
                assert archived is not None and archived.read() == path.read_bytes(), "Frozen input changed."
    reference = physical_reference()
    summary = {"audit": "TFIM-3 model / phase / historical SELECT / current RQ1",
               "sources": sources, "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
               "versions": {name: importlib.metadata.version(name) for name in ("numpy", "qiskit", "qrisp")},
               "tolerance": TOL, "stages": {}}
    source_snapshot(sources)
    for name, operation in (("model", lambda: model_audit(reference)),
                            ("historical_mechanism", lambda: historical_mechanism(reference)),
                            ("historical_qrisp_select", historical_select_audit),
                            ("current_ours_block", lambda: current_block_audit(reference)),
                            ("current_rq1", lambda: current_rq1_audit(reference))):
        if previous.get("stages", {}).get(name, {}).get("passed"):
            summary["stages"][name] = previous["stages"][name]
            dump(OUT / "summary.partial.json", summary)
            print(json.dumps({"stage": name, "status": "already_completed"}), flush=True)
            continue
        print(json.dumps({"stage": name, "status": "start"}), flush=True)
        started = time.perf_counter()
        try:
            # Keep backend diagnostic prints out of the concise status stream.
            with redirect_stdout(io.StringIO()) if name != "current_rq1" else io.StringIO() as output:
                result = operation()
        except Exception:
            result = {"passed": False, "status": "error", "traceback": traceback.format_exc()}
        result["seconds"] = time.perf_counter() - started
        summary["stages"][name] = result
        dump(OUT / f"{name}.json", result)
        dump(OUT / "summary.partial.json", summary)
        print(json.dumps({"stage": name, "passed": result["passed"], "seconds": result["seconds"],
                          "error": result.get("traceback")}), flush=True)
    assert source_identity() == sources, "Source drift during the audit."
    summary["passed"] = all(stage["passed"] for stage in summary["stages"].values())
    summary["source_snapshot_sha256"] = sha256((OUT / "source_snapshot.tar.gz").read_bytes()).hexdigest()
    dump(OUT / "summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "result": str(OUT / "summary.json")}), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
