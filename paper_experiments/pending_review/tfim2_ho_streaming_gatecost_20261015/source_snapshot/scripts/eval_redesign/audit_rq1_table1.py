#!/usr/bin/env python3
"""Audit saved RQ1 artifacts without compilation or resuming optimizer jobs.

The collector is called read-only. Small saved circuits receive complete basis
checks; larger Raw/Ours circuits receive bounded coherent pure-state probes.
Saved large Stinespring isometries are checked without forming dense Choi
matrices. This cannot certify a final basis circuit that was never exported.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "QISKIT_NUM_PROCS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-rq3-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/codex-rq3-numba")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"),
               str(ROOT / "scripts/eval_redesign"), str(ROOT / "Baseline_scripts")]
CASES = ["TFIM-4", "TFIM-8", "HRW-4", "HRW-8", "HRW-12", "RepDQE-d3"]
SMALL = {"TFIM-4", "HRW-4", "RepDQE-d3"}
TOLERANCE = 1e-10


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def paths(case):
    pending = ROOT / "paper_experiments/pending_review"
    if case == "RepDQE-d3":
        ours = pending / "remaining_full_20261011/cases" / case
    elif case == "HRW-8":
        ours = ROOT / "Baseline_results/table1_scaling_ours_20261013/cases" / case
    else:
        ours = pending / "tfim_hrw_full_20261011/cases" / case
    raw = pending / "rq1_paper_suite_20261010/full_v2/runs" / case / "raw"
    stine = ROOT / "Baseline_results" / (
        "table_a_stinespring_20261013" if case in SMALL else
        "table1_scaling_stinespring_20261013") / "cases" / case
    return ours, raw, stine


def coefficient_rows(encoded):
    rows = []
    for terms in encoded:
        row = Counter()
        for label, magnitude, real, imag in terms:
            row[label] += magnitude * complex(real, imag)
        rows.append(dict(row))
    return rows


def model_input_audit(case, fixture):
    """Check declared TFIM/HRW formulas independently of circuit construction."""
    import numpy as np
    actual = coefficient_rows(fixture["input_kraus"])
    n = len(next(iter(actual[0])))
    model = fixture["model"]
    if case.startswith("TFIM-"):
        dt, gamma = model["delta_t"], model["gamma"]
        expected = [{"I" * n: -1j * (1 - dt * n * gamma ** 2)}]
        for i in range(n):
            site = lambda char: "".join(char if j == i else "I" for j in range(n))
            pair = "".join("Z" if j in {i, (i + 1) % n} else "I" for j in range(n))
            expected[0][pair] = expected[0].get(pair, 0) + dt
            expected[0][site("X")] = dt
            expected[0][site("Z")] = 1j * dt * gamma ** 2
            expected.append({site("X"): np.sqrt(dt) * gamma,
                             site("Y"): -1j * np.sqrt(dt) * gamma})
        description = ("K0=-i[I-i*dt*H-(dt/2)*sum(Ldagger*L)], "
                       "H=-sum(ZZ+X), L_i=gamma*(X_i-iY_i); "
                       "K_i=sqrt(dt)*L_i; K0's whole-Kraus phase cancels")
    elif case.startswith("HRW-"):
        expected = []
        scale = 1 / (2 * np.sqrt(2 * n))
        for i in range(n):
            site = lambda char: "I" * (n - 1 - i) + char + "I" * i
            for b in (0, 1):
                sign = (-1) ** b
                expected.append({"I" * n: scale, site("X"): scale,
                                 site("Z"): sign * scale, site("Y"): -1j * sign * scale})
        description = "K_(i,b)=[I+X_i+(-1)^b Z_i-i(-1)^b Y_i]/(2 sqrt(2n))"
    else:
        from build_repetition_code_dqe_channel import DqeSpec, build_repetition_dqe_channel
        expected, _ = build_repetition_dqe_channel(DqeSpec(
            d=model["d"], epsilon=model["epsilon"], symmetrized=model["symmetrized"]))
        description = "Rebuild M_enc o C_rep with the declared d/epsilon/symmetrized parameters"
    assert len(actual) == len(expected)
    error = max(abs(a.get(label, 0) - b.get(label, 0))
                for a, b in zip(actual, expected) for label in set(a) | set(b))
    assert error < 1e-12, (case, error)
    return {"passed": True, "method": description, "coefficient_max_abs_error": float(error)}


def compare_csv(rows, table_path):
    with table_path.open(newline="") as stream:
        frozen = list(csv.DictReader(stream))
    key = lambda row: (row["case"], row["method"], row["variant"])
    expected = {key(row): row for row in rows}
    actual = {key(row): row for row in frozen}
    assert len(frozen) == len(rows) == len(actual) == len(expected)
    assert set(expected) == set(actual)
    checked = 0
    for identity, row in expected.items():
        for field, value in row.items():
            observed = actual[identity][field]
            if isinstance(value, float):
                assert math.isclose(value, float(observed), rel_tol=1e-12, abs_tol=1e-14), (
                    identity, field, value, observed)
            else:
                assert ("" if value is None else str(value)) == observed, (
                    identity, field, value, observed)
            checked += 1
    return {"passed": True, "rows": len(rows), "candidate_rows": sum(
        row["table1_candidate"] for row in rows), "fields_checked": checked}


def low_rank_map_error(actual_vectors, expected_vectors):
    """Stable Frobenius error of A A† versus B B†, without dense output maps."""
    import numpy as np
    assert actual_vectors.shape[0] == expected_vectors.shape[0]
    _, r = np.linalg.qr(np.column_stack([actual_vectors, expected_vectors]), mode="reduced")
    split = actual_vectors.shape[1]
    a, b = r[:, :split], r[:, split:]
    reference = b @ b.conj().T
    difference = a @ a.conj().T - reference
    return float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30))


def kraus_vectors(matrices):
    import numpy as np
    return np.column_stack([matrix.reshape(-1, order="F") for matrix in matrices])


def dense_kraus(encoded):
    import numpy as np
    from qiskit.quantum_info import Pauli
    n = len(encoded[0][0][0])
    assert n <= 8, "Do not materialize larger dense Kraus collections"
    dimension = 1 << n
    return np.asarray([sum((magnitude * complex(real, imag) * Pauli(label).to_matrix()
                           for label, magnitude, real, imag in row),
                          np.zeros((dimension, dimension), complex)) for row in encoded])


def matrix_audit(case, stine_folder, result, encoded):
    import numpy as np
    from collect_rq1_table1 import file_hash
    n, dimension = result["system_qubits"], 1 << result["system_qubits"]
    if case == "HRW-12":
        assert result["status"] == "preflight_memory_limit"
        assert result["isometry_bytes"] == 16 * math.prod(result["isometry_shape"])
        assert result["isometry_bytes"] > result["protocol"]["rss_limit_bytes"]
        assert not result["variants"] and not (stine_folder / "isometry.npy").exists()
        return {"status": "not_synthesized", "preflight_arithmetic_passed": True,
                "isometry_bytes": result["isometry_bytes"],
                "rss_limit_bytes": result["protocol"]["rss_limit_bytes"],
                "circuit_semantics": "unavailable: preflight stopped before matrix/circuit construction"}
    reference = dense_kraus(encoded)
    audit = {"status": "passed", "isometries": []}
    matrix_path = stine_folder / "matrices.npz"
    if matrix_path.exists():
        with np.load(matrix_path) as archive:
            raw, compressed, superoperator = (archive[key] for key in (
                "raw_kraus", "compressed_kraus", "superoperator"))
        raw_error = float(np.linalg.norm(raw - reference) / np.linalg.norm(reference))
        compression_error = low_rank_map_error(kraus_vectors(compressed), kraus_vectors(reference))
        independent_superoperator = sum((np.kron(k.conj(), k) for k in reference),
                                       np.zeros_like(superoperator))
        superoperator_error = float(np.linalg.norm(superoperator - independent_superoperator) /
                                   np.linalg.norm(independent_superoperator))
        assert max(raw_error, compression_error, superoperator_error) < TOLERANCE
        rng = np.random.default_rng(20261013)
        action_errors = []
        for _ in range(3):
            x = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
            rho = x @ x.conj().T
            rho /= np.trace(rho)
            expected = sum(k @ rho @ k.conj().T for k in reference)
            actual = (superoperator @ rho.reshape(-1, order="F")).reshape(
                (dimension, dimension), order="F")
            action_errors.append(float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)))
        assert max(action_errors) < TOLERANCE
        audit["classical_dense"] = {"artifact": str(matrix_path), "sha256": file_hash(matrix_path),
            "raw_kraus_relative_error": raw_error, "rank_compression_choi_relative_error": compression_error,
            "superoperator_relative_error": superoperator_error, "action_relative_errors": action_errors}
    variants = {variant["mode"]: variant for variant in result["variants"]}
    for mode, variant in variants.items():
        path = stine_folder / (mode + "_isometry.npy" if case in SMALL else "isometry.npy")
        isometry = np.load(path, mmap_mode="r")
        assert list(isometry.shape) == variant.get("isometry_shape", result.get("isometry_shape"))
        orthogonality = float(np.linalg.norm(isometry.conj().T @ isometry - np.eye(dimension)))
        environment_bits = variant.get("environment_qubits", variant["W"] - n - int(not result["native_cptp"]))
        environment_dim = 1 << environment_bits
        output_dim = dimension * (1 if result["native_cptp"] else 2)
        successful = isometry.reshape(output_dim, environment_dim, dimension)[:dimension].transpose(1, 0, 2)
        choi_error = low_rank_map_error(np.sqrt(variant["Lambda"]) * kraus_vectors(successful),
                                        kraus_vectors(reference))
        assert orthogonality < TOLERANCE and choi_error < TOLERANCE
        audit["isometries"].append({"mode": mode, "path": str(path), "sha256": file_hash(path),
            "orthogonality_frobenius_error": orthogonality,
            "full_channel_choi_relative_error_via_low_rank_QR": choi_error,
            "Lambda_restored": variant["Lambda"], "passed": True})
    if case not in SMALL:
        audit["circuit_semantics"] = "not independently certified: final basis circuit/QASM was not preserved"
        audit["count_evidence"] = "recorded in-memory count_ops and basis check; no saved artifact for independent recount"
    return audit


def load_qasm(path):
    from qiskit import qasm2
    # Historical Raw QASM uses qreg s, colliding with qelib1's s gate under
    # Qiskit's strict parser. Rename register variables in memory only. Gate
    # names, parameters, declaration order, and indexed wires are unchanged.
    original_text = text = Path(path).read_text()
    registers = re.findall(r"\bqreg\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\[", text)
    renaming = {name: f"audit_register_{index}" for index, name in enumerate(registers)}
    for name, replacement in renaming.items():
        assert not re.search(r"\b" + re.escape(replacement) + r"\b", text)
        text = re.sub(r"\b" + re.escape(name) + r"\b(?=\s*\[)", replacement, text)
    restored_text = text
    for name, replacement in renaming.items():
        restored_text = re.sub(r"\b" + re.escape(replacement) + r"\b(?=\s*\[)", name, restored_text)
    assert restored_text == original_text
    circuit = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    circuit.metadata = {"audit_import_register_renaming": renaming}
    return circuit


def simulate_columns(circuit, input_system, initial_layout=None, final_layout=None):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    width = circuit.num_qubits
    initial_layout = initial_layout or list(range(width))
    final_layout = final_layout or list(range(width))
    def address(value, system, layout):
        return sum(((value >> bit) & 1) << layout[wire] for bit, wire in enumerate(system))
    simulator = AerSimulator(method="statevector", max_parallel_threads=1, max_parallel_experiments=1)
    columns = []
    for value in range(1 << len(input_system)):
        initial = np.zeros(1 << width, dtype=complex)
        initial[address(value, input_system, initial_layout)] = 1
        probe = QuantumCircuit(width)
        probe.set_statevector(initial)
        del initial
        probe.compose(circuit, inplace=True, copy=False)
        probe.save_statevector()
        outcome = simulator.run(probe).result()
        assert outcome.success
        state = np.array(outcome.get_statevector(), copy=True)
        columns.append(state)
    recovered = np.column_stack(columns)
    logical_rows = [address(value, list(range(width)), final_layout) for value in range(1 << width)]
    return recovered[logical_rows]


def small_circuit_audit(case, out_dir):
    import numpy as np
    from qiskit import qpy
    from collect_rq1_table1 import file_hash
    from rq1_pipeline import deserialize_kraus, verify_channel
    ours_folder, raw_folder, stine_folder = paths(case)
    fixture = read(ours_folder / "input.json")
    target = deserialize_kraus(fixture["input_kraus"])
    result = {"case": case, "kind": "complete_small_circuits", "status": "running", "channels": [], "stinespring": []}
    path = out_dir / case / "small_circuits.json"
    for method, folder, qasm in (("Raw", raw_folder, "channel_nam_opt0.qasm"),
                                 ("Ours", ours_folder, "channel_nam.qasm")):
        saved = read(folder / "results.json")
        circuit = load_qasm(folder / qasm)
        assert file_hash(folder / qasm) == saved["qasm_sha256"]
        renaming = circuit.metadata["audit_import_register_renaming"]
        circuit.metadata = saved.get("metadata", {
            "structure": "basic", "subnormalization_factor": saved["Lambda"]})
        check = verify_channel(circuit, saved["registers"], target)
        assert check["passed"], (case, method, check)
        result["channels"].append({"route": method, "artifact": str(folder / qasm),
                                    "sha256": file_hash(folder / qasm),
                                    "import_register_renaming": renaming, **check})
        dump(path, result)
        print(json.dumps({"case": case, "method": method, "verification": check}), flush=True)
        del circuit
    saved = read(stine_folder / "results.json")
    reference = dense_kraus(fixture["input_kraus"])
    dimension = 1 << saved["system_qubits"]
    for variant in saved["variants"]:
        qasm_path = Path(variant["qasm"])
        qpy_path = qasm_path.with_suffix(".qpy")
        with qpy_path.open("rb") as stream:
            stored = qpy.load(stream)
        assert len(stored) == 1
        qpy_circuit = stored[0]
        initial_map = qpy_circuit.layout.initial_index_layout() if qpy_circuit.layout else None
        final_map = qpy_circuit.layout.final_index_layout() if qpy_circuit.layout else None
        isometry = np.load(stine_folder / (variant["mode"] + "_isometry.npy"))
        for artifact, circuit in ((qpy_path, qpy_circuit), (qasm_path, load_qasm(qasm_path))):
            allowed = {"rz", "h", "x", "cx"} if variant["basis"] == "nam" else {"u1", "u2", "u3", "cx"}
            counts = dict(circuit.count_ops())
            assert set(counts) <= allowed
            assert sum(counts.values()) == variant["G"] and circuit.num_qubits == variant["W"]
            assert counts == variant["counts"]["ops"]
            recovered = simulate_columns(circuit, saved["protocol"]["input_system_wires"], initial_map, final_map)
            overlap = np.vdot(isometry, recovered)
            phase = overlap / abs(overlap)
            isometry_error = float(np.linalg.norm(recovered - phase * isometry) / np.linalg.norm(isometry))
            environment_dim = 1 << variant["environment_qubits"]
            output_dim = dimension * (2 if variant["success_flag_qubits"] else 1)
            successful = recovered.reshape(output_dim, environment_dim, dimension)[:dimension].transpose(1, 0, 2)
            channel_error = low_rank_map_error(np.sqrt(variant["Lambda"]) * kraus_vectors(successful),
                                               kraus_vectors(reference))
            assert max(isometry_error, channel_error) < TOLERANCE, (case, artifact, isometry_error, channel_error)
            result["stinespring"].append({"mode": variant["mode"], "basis": variant["basis"],
                "artifact": str(artifact), "sha256": file_hash(artifact), "G_recounted": sum(counts.values()),
                "import_register_renaming": (circuit.metadata or {}).get("audit_import_register_renaming"),
                "W_recounted": circuit.num_qubits, "basis_columns": dimension,
                "isometry_relative_error_up_to_global_phase": isometry_error,
                "full_unnormalized_choi_relative_error": channel_error, "passed": True,
                "global_phase_scope": "whole-circuit phase cancels in the channel; relative phases retained"})
            dump(path, result)
            print(json.dumps({"case": case, "Stinespring": variant["mode"], "basis": variant["basis"],
                              "artifact": artifact.suffix, "choi_relative_error": channel_error}), flush=True)
        del qpy_circuit, stored
    result["status"] = "passed"
    dump(path, result)
    return result


def pauli_action(label, vector):
    import numpy as np
    indices = np.arange(len(vector), dtype=np.int64)
    destination = indices.copy()
    phase = np.ones(len(vector), dtype=complex)
    for bit, char in enumerate(reversed(label)):
        if char in "XY":
            destination ^= 1 << bit
        if char in "YZ":
            phase *= np.where((indices >> bit) & 1, -1, 1)
        if char == "Y":
            phase *= 1j
    output = np.empty_like(vector)
    output[destination] = phase * vector
    return output


def large_probe_audit(case, method, out_dir, probe_count, rss_limit_bytes):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from collect_rq1_table1 import file_hash, encoded_hash
    ours_folder, raw_folder, _ = paths(case)
    folder = ours_folder if method == "Ours" else raw_folder
    saved = read(folder / "results.json")
    fixture = read(ours_folder / "input.json")
    qasm_path = folder / ("channel_nam.qasm" if method == "Ours" else "channel_nam_opt0.qasm")
    assert file_hash(qasm_path) == saved["qasm_sha256"]
    circuit = load_qasm(qasm_path)
    width = circuit.num_qubits
    result = {"case": case, "method": method, "kind": "bounded_pure_state_probes",
              "status": "running", "artifact": str(qasm_path), "sha256": file_hash(qasm_path),
              "W": width, "probe_results": [],
              "import_register_renaming": circuit.metadata["audit_import_register_renaming"],
              "scope": "actual saved circuit checks for selected pure states; not a complete channel proof"}
    path = out_dir / case / (method.lower() + "_probes.json")
    target_input_hash = encoded_hash(fixture["input_kraus"])
    previous = read(path) if path.exists() else {}
    if (previous.get("sha256") == result["sha256"] and
            previous.get("target_input_sha256", target_input_hash) == target_input_hash and
            previous.get("case") == case and previous.get("method") == method):
        result["probe_results"] = previous.get("probe_results", [])
        assert all(probe.get("passed") for probe in result["probe_results"])
    result["target_input_sha256"] = target_input_hash
    if 5 * 16 * (1 << width) > rss_limit_bytes:
        result.update(status="not_run_memory_preflight", reason="Five statevector copies exceed the audit budget")
        dump(path, result)
        return result
    registers = saved["registers"]
    system = registers[-1]
    selector = registers[0] if method == "Ours" else registers[1]
    assert len(set(sum(registers, []))) == width
    dimension = 1 << len(system)
    address = lambda value, wires: sum(((value >> bit) & 1) << wire for bit, wire in enumerate(wires))
    system_addresses = np.asarray([address(value, system) for value in range(dimension)])
    success_addresses = np.asarray([[address(value, selector) + row for row in system_addresses]
                                    for value in range(1 << len(selector))])
    target_rows = coefficient_rows(fixture["input_kraus"])
    rng = np.random.default_rng(20261013)
    simulator = AerSimulator(method="statevector", max_parallel_threads=1, max_parallel_experiments=1)
    for index in range(probe_count):
        if index == 0:
            vector = np.zeros(dimension, complex)
            vector[0] = 1
            label = "zero_basis"
        elif index == 1:
            vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
            vector /= np.linalg.norm(vector)
            label = "seeded_complex_superposition"
        elif index == 2:
            vector = np.full(dimension, 1 / np.sqrt(dimension), complex)
            label = "plus_product"
        else:
            vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
            vector /= np.linalg.norm(vector)
            label = f"seeded_complex_superposition_{index}"
        if index < len(result["probe_results"]):
            assert result["probe_results"][index]["input"] == label
            continue
        initial = np.zeros(1 << width, dtype=complex)
        initial[system_addresses] = vector
        probe = QuantumCircuit(width)
        probe.set_statevector(initial)
        del initial
        probe.compose(circuit, inplace=True, copy=False)
        probe.save_statevector()
        started = time.perf_counter()
        outcome = simulator.run(probe).result()
        assert outcome.success
        state = np.asarray(outcome.get_statevector())
        actual = np.sqrt(saved["Lambda"]) * state[success_addresses].T
        expected = np.column_stack([sum((coefficient * pauli_action(pauli, vector)
                                        for pauli, coefficient in row.items()),
                                       np.zeros(dimension, complex)) for row in target_rows])
        error = low_rank_map_error(actual, expected)
        assert error < TOLERANCE, (case, method, label, error)
        result["probe_results"].append({"input": label, "output_map_relative_frobenius_error": error,
            "success_probability": float(np.vdot(actual, actual).real / saved["Lambda"]),
            "expected_success_probability": float(np.vdot(expected, expected).real / saved["Lambda"]),
            "simulation_s": time.perf_counter() - started, "passed": True})
        dump(path, result)
        print(json.dumps({"case": case, "method": method, "probe": label, "relative_error": error}), flush=True)
        del probe, outcome, state, actual
    result["status"] = "passed_probes"
    dump(path, result)
    return result


def data_audit(cases):
    from collect_rq1_table1 import collect, file_hash
    rows = collect()
    comparison = compare_csv(rows, ROOT / "Results/rq1_table1_resources_20261013.csv")
    smoke = read(ROOT / "Baseline_results/rq3_refresh_20261013/smoke.json")
    assert smoke["passed"]
    sources = {}
    case_data = {}
    for case in cases:
        ours_folder, raw_folder, stine_folder = paths(case)
        ours, raw, stine = (read(folder / "results.json") for folder in (ours_folder, raw_folder, stine_folder))
        for name, expected in ours["sources"].items():
            actual = file_hash(ROOT / "src" / name)
            assert actual == expected, (case, name, expected, actual)
            if name in smoke["sources"]:
                assert actual == smoke["sources"][name]
            sources[name] = actual
        fixture = read(ours_folder / "input.json")
        case_data[case] = {"input_model": model_input_audit(case, fixture),
            "classical_and_isometry": matrix_audit(case, stine_folder, stine, fixture["input_kraus"]),
            "resource_rows": [{key: row.get(key) for key in (
                "method", "variant", "status", "G", "non_clifford", "W", "Lambda", "qasm_sha256",
                "qasm_recount", "full_choi_status")} for row in rows if row["case"] == case],
            "encoded_input_pairing": "passed by collector for original/canonical/rewritten Kraus families"}
        print(json.dumps({"case": case, "data_and_matrix_audit": "passed"}), flush=True)
    manifest = read(ROOT / "Baseline_results/table1_scaling_stinespring_20261013/manifest.json")
    for path, expected in manifest["sources"].items():
        assert file_hash(Path(path)) == expected, path
    return {"collector": "passed", "current_csv": comparison,
            "backend_matches_mandatory_archive_smoke": True, "source_sha256": sources,
            "scaling_stinespring_manifest_sources": "passed", "cases": case_data}


def bounded_worker(args, case, kind):
    import psutil
    folder = args.out_dir / case
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / ("small_circuits.json" if kind == "small" else kind.lower() + "_probes.json")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--out-dir", str(args.out_dir),
               "--worker-case", case, "--worker-kind", kind, "--probes", str(args.probes),
               "--rss-limit-gib", str(args.rss_limit_gib)]
    started = time.monotonic()
    peak = 0
    reason = None
    with (folder / (kind.lower() + ".log")).open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                peak = max(peak, psutil.Process(process.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak > args.rss_limit_gib * 1024 ** 3:
                reason = "memory_limit"
            elif time.monotonic() - started > args.timeout:
                reason = "timeout"
            if reason:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(0.2)
        code = process.wait()
    record = {"case": case, "kind": kind, "exit_code": code,
              "wall_s": time.monotonic() - started, "peak_rss_bytes": peak,
              "resource_stop": reason, "artifact": str(artifact)}
    if artifact.exists():
        record["result"] = read(artifact)
    record["status"] = reason or ("passed" if code == 0 else "failed")
    print(json.dumps({key: value for key, value in record.items() if key != "result"}), flush=True)
    return record


def main(args):
    from collect_rq1_table1 import file_hash
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    output = args.out_dir / "summary.json"
    previous = read(output) if args.resume and output.exists() else {}
    completed = {(record["case"], record["kind"]): record for record in previous.get("workers", [])
                 if record["status"] == "passed" and record.get("result", {}).get("status")
                 in {"passed", "passed_probes"}}
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
               "cases": args.cases, "protocol": {"timeout_per_worker_s": args.timeout,
                   "rss_limit_per_worker_bytes": int(args.rss_limit_gib * 1024 ** 3),
                   "large_pure_state_probes": args.probes, "tolerance": TOLERANCE,
                   "numerical_threads": 1, "semantic_mode": args.semantic_mode,
                   "compilation": "none; actual saved circuit artifacts only",
                   "paused_raw_optimizer_queue": "untouched"}, "workers": []}
    summary["audit_runner_sha256"] = file_hash(Path(__file__).resolve())
    summary["import_compatibility"] = ("Register variables renamed in memory for the strict QASM parser; "
        "declaration order, wire indices, gate operations, parameters and saved artifact hashes preserved")
    if previous:
        summary["prior_incomplete_attempts"] = previous.get("prior_incomplete_attempts", []) + [
            {key: record.get(key) for key in ("case", "kind", "status", "resource_stop", "exit_code")}
            for record in previous.get("workers", []) if record["status"] != "passed"]
    dump(output, summary)
    summary["data"] = data_audit(args.cases)
    dump(output, summary)
    if args.semantic_mode != "none":
        for case in args.cases:
            if case in SMALL:
                summary["workers"].append(completed.get((case, "small")) or bounded_worker(args, case, "small"))
            elif args.semantic_mode == "probes":
                for method in ("Raw", "Ours"):
                    summary["workers"].append(completed.get((case, method)) or bounded_worker(args, case, method))
                    dump(output, summary)
            dump(output, summary)
    gaps = ["TFIM-8/HRW-8 Stinespring: saved isometry fully audited; final basis circuit/QASM missing",
            "HRW-12 Stinespring: 8 GiB single-isometry preflight rejection, no synthesized circuit"]
    if args.semantic_mode == "probes":
        gaps.append("Larger Raw/Ours circuit checks cover the recorded pure-state probes, not all basis columns")
    else:
        gaps.append("Larger Raw/Ours final circuit semantics not re-simulated by this audit mode")
    summary["evidence_gaps"] = gaps
    summary["status"] = ("failed" if any(record["status"] == "failed" for record in summary["workers"])
                         else "completed_with_bounded_evidence")
    summary["wall_s"] = time.perf_counter() - started
    dump(output, summary)
    print(json.dumps({"output": str(output), "status": summary["status"], "wall_s": summary["wall_s"]}), flush=True)
    return 1 if summary["status"] == "failed" else 0


def finish_after_large_cancellation(args):
    """Preserve existing large evidence and complete only outstanding small checks."""
    from collect_rq1_table1 import file_hash
    args.out_dir = args.out_dir.resolve()
    output = args.out_dir / "summary.json"
    summary = read(output)
    assert "data" in summary, "The completed data/matrix checks must already be preserved"
    records = {(record["case"], record["kind"]): record for record in summary.get("workers", [])}
    workers = []
    large_stages = []
    summary["status"] = "finishing_small_checks_after_user_cancellation"
    summary["large_cancellation"] = {"requested_by": "user", "requested_at_utc":
        datetime.now(timezone.utc).isoformat(), "coordinator_pid": 82011, "worker_pid": 83255,
        "session_id": 9599, "case": "TFIM-8", "method": "Raw",
        "termination": "SIGTERM; both processes confirmed no longer running",
        "scope": "No further large circuit simulations, retries or probes"}
    dump(output, summary)
    for case in summary["cases"]:
        if case in SMALL:
            path = args.out_dir / case / "small_circuits.json"
            saved = read(path) if path.exists() else {}
            record = records.get((case, "small"))
            if saved.get("status") != "passed":
                record = bounded_worker(args, case, "small")
                saved = record.get("result", {})
            elif record is None:
                record = {"case": case, "kind": "small", "status": "passed",
                          "artifact": str(path), "evidence": "previously completed saved artifact"}
            if saved.get("status") == "passed":
                ours_folder, raw_folder, _ = paths(case)
                for channel in saved["channels"]:
                    artifact = Path(channel["artifact"])
                    if artifact == raw_folder / "channel_nam_opt0.qasm":
                        channel["route"] = "Raw"
                    else:
                        assert artifact == ours_folder / "channel_nam.qasm"
                        channel["route"] = "Ours"
                dump(path, saved)
            record["result"] = saved
            workers.append(record)
        else:
            for method in ("Raw", "Ours"):
                path = args.out_dir / case / (method.lower() + "_probes.json")
                saved = read(path) if path.exists() else {}
                passed = saved.get("probe_results", [])
                assert all(probe["passed"] for probe in passed)
                if len(passed) == args.probes and saved.get("status") == "passed_probes":
                    stage = "passed_completed_probes"
                elif case == "TFIM-8" and method == "Raw":
                    stage = "cancelled_by_user"
                elif case == "HRW-12" and method == "Raw":
                    stage = "passed_partial_probes_remaining_cancelled_by_user"
                else:
                    stage = "not_executed_after_user_cancellation"
                remaining = max(0, args.probes - len(passed))
                if remaining:
                    saved["status_before_cancellation"] = saved.get("status")
                    saved["status"] = stage
                    saved["case"], saved["method"] = case, method
                    saved["remaining_probes_not_executed"] = remaining
                    if "error" in saved:
                        saved["prior_import_error"] = saved.pop("error").strip().splitlines()[-1]
                    dump(path, saved)
                large_stages.append({"case": case, "method": method, "status": stage,
                                     "passed_probes": len(passed), "remaining_not_executed": remaining})
                workers.append({"case": case, "kind": method, "status": stage,
                                "artifact": str(path), "result": saved,
                                "evidence": "preserved at user cancellation; no additional large check executed"})
        summary["workers"] = workers
        summary["large_stages"] = large_stages
        dump(output, summary)
    summary["evidence_gaps"] = [
        "TFIM-8/HRW-8 Stinespring: saved isometry fully audited; final basis circuit/QASM missing",
        "HRW-12 Stinespring: 8 GiB single-isometry preflight rejection, no synthesized circuit",
        "Large Raw/Ours final circuit evidence is limited to the recorded completed pure-state probes",
        "User cancelled the remaining large circuit checks; no further large simulations performed"]
    summary["previous_audit_runner_sha256"] = summary.get("audit_runner_sha256")
    summary["audit_runner_sha256"] = file_hash(Path(__file__).resolve())
    summary["status"] = ("failed_small_check" if any(
        record["kind"] == "small" and record["status"] != "passed" for record in workers)
        else "completed_with_user_cancelled_large_checks")
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump(output, summary)
    print(json.dumps({"output": str(output), "status": summary["status"],
                      "large_stages": large_stages}), flush=True)
    return 1 if summary["status"] == "failed_small_check" else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Baseline_results/rq1_correctness_audit_20261013")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--semantic-mode", choices=["none", "small", "probes"], default="probes")
    parser.add_argument("--probes", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--rss-limit-gib", type=float, default=3)
    parser.add_argument("--worker-case", choices=CASES)
    parser.add_argument("--worker-kind", choices=["small", "Raw", "Ours"])
    parser.add_argument("--resume", action="store_true", help="Reuse completed audit workers; rerun incomplete checks")
    parser.add_argument("--finish-small-after-cancellation", action="store_true",
                        help="Preserve large evidence and complete only outstanding small circuit checks")
    arguments = parser.parse_args()
    try:
        if arguments.worker_case:
            assert arguments.worker_kind
            if arguments.worker_kind == "small":
                small_circuit_audit(arguments.worker_case, arguments.out_dir)
            else:
                large_probe_audit(arguments.worker_case, arguments.worker_kind, arguments.out_dir,
                                  arguments.probes, int(arguments.rss_limit_gib * 1024 ** 3))
        elif arguments.finish_small_after_cancellation:
            raise SystemExit(finish_after_large_cancellation(arguments))
        else:
            raise SystemExit(main(arguments))
    except Exception:
        traceback.print_exc()
        if arguments.worker_case:
            folder = arguments.out_dir / arguments.worker_case
            path = folder / ("small_circuits.json" if arguments.worker_kind == "small" else
                             arguments.worker_kind.lower() + "_probes.json")
            partial = read(path) if path.exists() else {"case": arguments.worker_case}
            partial.update(status="failed", error=traceback.format_exc())
            dump(path, partial)
        raise SystemExit(1)
