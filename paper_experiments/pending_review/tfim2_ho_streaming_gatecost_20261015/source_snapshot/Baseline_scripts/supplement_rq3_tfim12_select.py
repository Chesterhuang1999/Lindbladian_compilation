"""Add the 12-qubit TFIM A0 SELECT comparison using the existing table policy."""

from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("src", "scripts", "Baseline_scripts")]

import numpy as np
from qiskit import qasm2, qpy
from qiskit.quantum_info import Pauli
from qrisp import QuantumFloat, QuantumVariable
from qrisp.jasp import q_switch

from baseline_common import count_qasm_text
from block_encoding import BlockEncoding
from channel_IR import Matrixsum, PauliAtom
from experiment_protocol import source_identity, transpile_channel_circuit
from pennylane_lcu_adapter import build_lcu_tape, compile_lcu_tape, tape_to_qiskit
from run_rq3_random4q_select import lowering_probes, pauli_action, qasm_roundtrip_error
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel

OUT = ROOT / "Baseline_results/rq3_select_tfim12_20261014"
BASES = {"nam": ["rz", "h", "x", "cx"], "ibm": ["u1", "u2", "u3", "cx"]}
TOL = 1e-9


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rel(path):
    return str(path.relative_to(ROOT))


def qrisp_pauli_unitary_with_phase(label, phase):
    # Preserve the table3 phase-unitary convention without importing its
    # obsolete benchmark dependencies.
    from qrisp import gphase, x, y, z

    angle = math.atan2(phase.imag, phase.real)

    def unitary(qv):
        if abs(angle) > 1e-14:
            gphase(angle, qv[0])
        for index, letter in enumerate(label):
            if letter != "I":
                {"X": x, "Y": y, "Z": z}[letter](qv[index])

    unitary.__name__ = f"phase_pauli_{label}"
    return unitary


def input_terms(n):
    kraus, model = build_periodic_tfim_channel(n, 0.1)
    matrix_sum = next(operator for operator in kraus if operator.instances)
    terms = [(atom.expr, complex(coefficient * atom.phase)) for atom, coefficient in matrix_sum.instances]
    assert len(terms) == 3 * n + 1
    expected = [("I" * n, -1j * (1 - 0.0025 * n))]
    for site in range(n):
        expected += [("".join("Z" if q in (site, (site + 1) % n) else "I" for q in range(n)), 0.1),
                     ("".join("X" if q == site else "I" for q in range(n)), 0.1)]
    expected += [("".join("Z" if q == site else "I" for q in range(n)), 0.0025j) for site in range(n)]
    assert [label for label, _ in terms] == [label for label, _ in expected]
    assert max(abs(a - b) for (_, a), (_, b) in zip(terms, expected)) < 1e-12
    return terms, model


def ours_route(terms):
    matrix_sum = Matrixsum([(PauliAtom(label), coefficient) for label, coefficient in terms])
    backend = BlockEncoding(matrix_sum, placement_options={})
    parts = backend.circuit_components("Matrix-order")
    metadata = parts.circuit.metadata
    placement = dict(metadata["matrix_order"]["placement"])
    worst = 0.0
    for label, coefficient in terms:
        address = int(placement[label], 2)
        accumulated, scalar = Pauli("I" * len(label)), 1.0 + 0j
        for factor, mask, phase in backend.mobius_phase_result["g_modes_with_phase"]:
            mask = int(mask, 2)
            if address & mask == mask:
                accumulated = Pauli(factor).dot(accumulated)
                scalar *= np.exp(0.5j * np.pi * phase)
        assert Pauli((accumulated.z, accumulated.x)).to_label() == label
        worst = max(worst, float(abs(scalar * (-1j) ** int(accumulated.phase) - coefficient / abs(coefficient))))
    assert worst < TOL
    return parts.select, list(range(backend.ctrl_size)), metadata["system_qubits"], placement, {
        "metadata": metadata, "all_occupied_address_pauli_product_error": worst,
        "occupied_addresses_checked_symbolically": len(terms), "symbolic_system_input": "arbitrary",
    }


def pl_route(terms):
    n, k = len(terms[0][0]), (len(terms) - 1).bit_length()
    work = max(0, k - 1)
    # PennyLane 0.44.0's full-SELECT unary iterator closes its work ladder
    # incorrectly for 37 entries. Two zero-weight identity entries preserve
    # the entire SELECT operator and avoid that native decomposition error.
    padding = 2 if n == 12 and len(terms) == 37 and version("pennylane") == "0.44.0" else 0
    select_terms = terms + [("I" * n, 0j)] * padding
    tape, wires, build = build_lcu_tape(select_terms, n, scope="select", work_wires=work)
    compiled, compile_meta = compile_lcu_tape(tape, wires, optimize=True, decomposition="graph")
    circuit = tape_to_qiskit(compiled, wires)
    placement = {label: format(i, f"0{k}b") for i, (label, _) in enumerate(terms)}
    return circuit, list(range(n + work, n + work + k)), list(range(n)), placement, {
        "profile": "graph_work", "build": build, "compile": compile_meta,
        "zero_weight_identity_padding": padding,
        "padding_reason": "native full-SELECT unary closing-ladder cleanup" if padding else None,
        "unpadded_validation_failure": rel(OUT / "TFIM_12_A0/pl37_unpadded/validation_failure.json") if padding else None,
    }


def qrisp_route(terms):
    n, k = len(terms[0][0]), (len(terms) - 1).bit_length()
    system = QuantumVariable(n, name="sys")
    selector = QuantumFloat(k, name="idx")
    unitaries = [qrisp_pauli_unitary_with_phase(label, coefficient / abs(coefficient))
                 for label, coefficient in terms]
    q_switch(selector, unitaries, system)
    compiled = system.qs.compile(workspace=0)
    circuit = compiled.to_qiskit()
    ports = {register.name: circuit.find_bit(register[0]).index
             for register in circuit.qregs if len(register) == 1}
    system_bits = [ports[qubit.identifier] for qubit in system.reg]
    address_bits = [ports[qubit.identifier] for qubit in selector.reg]
    assert len(system_bits) == n and len(address_bits) == k
    placement = {label: format(i, f"0{k}b") for i, (label, _) in enumerate(terms)}
    # The existing Qrisp baseline maps label character i to sys[i]. Express
    # its port map in Qiskit's little-endian label convention for validation.
    return circuit, address_bits, system_bits[::-1], placement, {
        "native_api": "qrisp.jasp.q_switch", "workspace": 0,
        "system_order": "logical least-significant system bit is sys[-1]",
    }


def probe_columns(terms, addresses, system, placement):
    n = len(system)
    patterns = np.asarray([0, (1 << n) - 1, sum(1 << bit for bit in range(0, n, 2))], dtype=np.int64)
    initial, expected, phases = [], [], []
    for label, coefficient in terms:
        address = int(placement[label], 2)
        encoded_address = sum(((address >> bit) & 1) << physical for bit, physical in enumerate(addresses))
        outputs, signs = pauli_action(label, patterns)
        for value, output, sign in zip(patterns, outputs, signs):
            initial.append(encoded_address | sum(((int(value) >> bit) & 1) << physical
                                                for bit, physical in enumerate(system)))
            expected.append(encoded_address | sum(((int(output) >> bit) & 1) << physical
                                                 for bit, physical in enumerate(system)))
            phases.append(sign * coefficient / abs(coefficient))
    return (np.asarray(initial, dtype=np.int64), np.asarray(expected, dtype=np.int64),
            np.asarray(phases, dtype=complex), np.ones(len(initial), dtype=bool))


def measure(terms, directory, *, verify):
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for method, builder in (("ours", ours_route), ("pl", pl_route), ("qrisp", qrisp_route)):
        started = time.perf_counter()
        with (directory / f"{method}.log").open("w") as log, redirect_stdout(log):
            circuit, addresses, system, placement, metadata = builder(terms)
        columns = probe_columns(terms, addresses, system, placement)
        if metadata.get("zero_weight_identity_padding"):
            # Full SELECT promises identity on every unused address too.
            unused = []
            patterns = [0, (1 << len(system)) - 1, sum(1 << bit for bit in range(0, len(system), 2))]
            for address in range(len(terms), 1 << len(addresses)):
                for value in patterns:
                    unused.append(sum(((address >> bit) & 1) << physical for bit, physical in enumerate(addresses))
                                  | sum(((value >> bit) & 1) << physical for bit, physical in enumerate(system)))
            unused = np.asarray(unused, dtype=np.int64)
            columns = (np.concatenate((columns[0], unused)), np.concatenate((columns[1], unused)),
                       np.concatenate((columns[2], np.ones(len(unused), dtype=complex))),
                       np.concatenate((columns[3], np.ones(len(unused), dtype=bool))))
        with (directory / f"{method}.qpy").open("wb") as handle:
            qpy.dump(circuit, handle)
        metrics = {}
        for basis, gates in BASES.items():
            lowered = transpile_channel_circuit(circuit, basis_gates=gates,
                                                optimization_level=0, seed_transpiler=7)
            assert set(lowered.count_ops()) <= set(gates)
            path = directory / f"{method}_{basis}_opt0.qasm"
            path.write_text(qasm2.dumps(lowered))
            phase_path = path.with_suffix(".phase.json")
            dump(phase_path, {"global_phase_radians": float(lowered.global_phase)})
            restored = qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            restored.global_phase = float(lowered.global_phase)
            roundtrip_error = qasm_roundtrip_error(lowered, restored)
            assert roundtrip_error < TOL
            stats = count_qasm_text(path.read_text(), gate_set=basis)
            assert stats["clifford"]["unknown"] == 0
            metric = {"G": stats["metric_total"], "NC": stats["clifford"]["non_clifford"],
                      "W": stats["num_qubits"], "CX": stats["ops"].get("cx", 0),
                      "qasm": rel(path), "qasm_sha256": sha256(path.read_bytes()).hexdigest(),
                      "global_phase_sidecar": rel(phase_path), "saved_gate_roundtrip_error": roundtrip_error}
            if method in verify:
                metric["verification"] = lowering_probes(restored, columns, seed=7, exhaustive=True)
            metrics[basis] = metric
            print(json.dumps({"method": method, "basis": basis, "G": metric["G"], "NC": metric["NC"],
                              "W": metric["W"], "phase_sensitive_verified": method in verify}), flush=True)
        results[method] = {"basis": metrics, "metadata": metadata, "placement": placement,
                           "address_qubits": addresses, "system_qubits": system,
                           "seconds": time.perf_counter() - started}
        dump(directory / f"{method}.json", results[method])
    return results


def main():
    smoke_path = OUT / "smoke.json"
    smoke = json.loads(smoke_path.read_text())
    sources = source_identity()
    assert smoke["passed"] and smoke["channel_check"]["passed"] and smoke["sources"] == sources
    with (OUT / "input_generation.log").open("w") as log, redirect_stdout(log):
        calibration_terms, _ = input_terms(8)
        terms, model = input_terms(12)
    frozen = json.loads((ROOT / "Baseline_results/pennylane_lcu_v1/inputs.json").read_text())
    reference = next(case for case in frozen["cases"] if case["name"] == "TFIM_8_A0")["inputs"][0]["terms"]
    assert [label for label, _ in calibration_terms] == [row["pauli"] for row in reference]
    assert max(abs(value - complex(row["real"], row["imag"])) for (_, value), row in zip(calibration_terms, reference)) < 1e-12
    calibration = measure(calibration_terms, OUT / "calibration_TFIM_8_A0", verify={"qrisp"})
    old = json.loads((ROOT / "Baseline_results/rq3_refresh_20261013/comparison.json").read_text())
    calibration_comparison = {}
    for basis in BASES:
        row = next(row for row in old["select_rows"] if row["case"] == "TFIM_8_A0" and row["basis"] == basis)
        for method in ("ours", "pl", "qrisp"):
            keys = ("G", "NC", "W") if method == "ours" else ("gate_count", "non_clifford", "width")
            measured = [calibration[method]["basis"][basis][k] for k in ("G", "NC", "W")]
            reference_metrics = [row[method][k] for k in keys]
            calibration_comparison[f"{method}_{basis}"] = {
                "reference_G_NC_W": reference_metrics, "measured_G_NC_W": measured,
                "counts_match": measured == reference_metrics,
            }
            if method != "qrisp":
                assert measured == reference_metrics
            else:
                # The native compiler may choose a different equivalent gate
                # sequence. Verify the saved baseline as well as the new run.
                archive_dir = ROOT / "Baseline_results/table_main_select_only_ours_qrisp_v1/single/TFIM_8_A0"
                logical = qasm2.load(archive_dir / "TFIM_8_A0_qrisp_select_qiskit.qasm",
                                     custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
                lowered = transpile_channel_circuit(logical, basis_gates=BASES[basis],
                                                    optimization_level=0, seed_transpiler=7)
                archive_path = archive_dir / f"TFIM_8_A0_qrisp_select_{basis}_opt0.qasm"
                saved = qasm2.load(archive_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
                saved.global_phase = lowered.global_phase
                assert qasm_roundtrip_error(lowered, saved) < TOL
                stats = count_qasm_text(archive_path.read_text(), gate_set=basis)
                assert [stats["metric_total"], stats["clifford"]["non_clifford"], stats["num_qubits"]] == reference_metrics
                ports = calibration[method]
                columns = probe_columns(calibration_terms, ports["address_qubits"],
                                        ports["system_qubits"], ports["placement"])
                calibration_comparison[f"{method}_{basis}"]["archive_verification"] = lowering_probes(
                    saved, columns, seed=7, exhaustive=True)
                calibration_comparison[f"{method}_{basis}"].update({
                    "archive_qasm": rel(archive_path),
                    "archive_qasm_sha256": sha256(archive_path.read_bytes()).hexdigest(),
                    "restored_global_phase": float(saved.global_phase),
                })
    print("TFIM_8_A0 calibration: PL/Ours counts match; historical and current Qrisp circuits verified phase-sensitively.", flush=True)
    records = [{"pauli": label, "real": coefficient.real, "imag": coefficient.imag} for label, coefficient in terms]
    manifest = {"case": "TFIM_12_A0", "system_qubits": 12, "num_terms": len(terms), "terms": records,
                "alpha": sum(abs(coefficient) for _, coefficient in terms), "model": model,
                "input_sha256": sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                "physical_reference": "-i*[I-i*dt*H-(dt/2)*sum(L_i^dagger L_i)], dt=0.1, H=-sum(ZZ+X), L_i=sqrt(0.1)/2*(X_i-iY_i)"}
    dump(OUT / "inputs.json", manifest)
    routes = measure(terms, OUT / "TFIM_12_A0", verify={"ours", "pl", "qrisp"})
    rows = []
    for basis in BASES:
        ours = {k: routes["ours"]["basis"][basis][k] for k in ("G", "NC", "W", "CX")}
        row = {"case": "TFIM_12_A0", "basis": basis, "aggregation": "single", "inputs": 1,
               "ours": ours, "status": "ok"}
        for method in ("pl", "qrisp"):
            metric = routes[method]["basis"][basis]
            row[method] = {"gate_count": metric["G"], "non_clifford": metric["NC"], "width": metric["W"]}
            row[f"{method}_over_ours"] = metric["G"] / ours["G"]
        rows.append(row)
    assert source_identity() == sources
    result = {"case": "TFIM_12_A0", "rows": rows, "routes": routes, "calibration": calibration,
              "calibration_count_comparison": calibration_comparison,
              "sources": sources, "smoke": rel(smoke_path), "smoke_sha256": sha256(smoke_path.read_bytes()).hexdigest(),
              "input_manifest": rel(OUT / "inputs.json"), "input_sha256": manifest["input_sha256"],
              "runner": rel(Path(__file__)), "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {name: version(name) for name in ("qiskit", "pennylane", "qrisp", "qiskit-aer")},
              "policy": {"scope": "SELECT only", "optimization_level": 0, "seed_transpiler": 7,
                         "qubits_initially_zero": False, "block_interface": "block_encoding.BlockEncoding",
                         "block_opt": "Matrix-order", "qrisp_workspace": 0, "pl_profile": "graph_work",
                         "channel_circuits_constructed": False,
                         "lowered_validation": "all 37 occupied addresses x system states 0, all-ones, alternating; PL also all 27 unused identity addresses"},
              "passed": True}
    dump(OUT / "summary.json", result)
    print(json.dumps({"passed": True, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
