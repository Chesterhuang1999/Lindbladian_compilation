"""Measure documented QSVT inputs and explicit structural SELECT stress cases.

All cases use the current-default canonical placement interface, native Qrisp
q_switch with fixed input ports, and PennyLane graph-work Select. No backend
settings are changed to manufacture a counterexample.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from qiskit import qasm2, qpy, transpile
from qiskit.quantum_info import Pauli, SparsePauliOp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
from baseline_common import count_qasm_text
from block_encoding import BlockEncoding
from channel_IR import Matrixsum, PauliAtom
from experiment_protocol import source_identity
from pennylane_lcu_adapter import build_lcu_tape, compile_lcu_tape, tape_to_qiskit
from run_rq3_qrisp_qsvt import pack_fixed_ports

OUT = ROOT / "Baseline_results/rq3_placement_counterexamples_20261014"
BASES = {"nam": ["rz", "h", "x", "cx"], "ibm": ["u1", "u2", "u3", "cx"]}
SETTINGS = dict(optimization_level=0, seed_transpiler=7, qubits_initially_zero=False)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def terms_from_matrix(matrix):
    # This is input conversion, never circuit synthesis.
    return [(label, complex(c)) for label, c in SparsePauliOp.from_operator(matrix).to_list()
            if abs(c) > 1e-12]


def cases():
    yield {"id": "pl_hardware_three_term", "origin": "official_exact_input",
           "source": "https://pennylane.ai/demos/tutorial_qsvt_hardware",
           "terms": [("XI", .2), ("XZ", -.7), ("ZY", -.6)]}
    matrix = np.array([[.65713691, -.05349524, .08024556, -.07242864],
                       [-.05349524, .65713691, -.07242864, .08024556],
                       [.08024556, -.07242864, .65713691, -.05349524],
                       [-.07242864, .08024556, -.05349524, .65713691]])
    yield {"id": "pl_linear_system_matrix", "origin": "official_exact_matrix_pauli_input",
           "source": "https://pennylane.ai/demos/tutorial_apply_qsvt",
           "terms": terms_from_matrix(matrix)}
    matrix = np.array([[.73255474, .14516978, -.14510851, -.0391581],
                       [.14516978, .68701415, -.04929867, -.00999921],
                       [-.14510851, -.04929867, .76587818, -.03420339],
                       [-.0391581, -.00999921, -.03420339, .58862043]])
    yield {"id": "qrisp_qet_matrix", "origin": "official_exact_matrix_pauli_input",
           "source": "https://qrisp.eu/reference/Algorithms/generated/qrisp.gqsp.QET.html",
           "terms": terms_from_matrix(matrix)}
    source = "https://qrisp.eu/reference/Algorithms/generated/qrisp.gqsp.hamiltonian_simulation.html"
    def label(n, operations):
        letters = ["I"] * n
        for bit, letter in operations:
            letters[n - bit - 1] = letter
        return "".join(letters)
    for n in (6, 16, 32):
        yield {"id": f"qrisp_magnetization_{n}q",
               "origin": "official_exact_qsp_observable" if n == 6 else "official_qsp_observable_size_variant",
               "source": source, "formula": "M_n=(1/n)*sum_j Z_j",
               "document_role": "observable in QSP Hamiltonian-simulation example; used here as QSVT input",
               "terms": [(label(n, [(j, "Z")]), 1.0 / n) for j in range(n)]}
    yield {"id": "qrisp_ising_chain6", "origin": "official_exact_qsp_hamiltonian",
           "source": source,
           "terms": [(label(6, [(j, "Z"), (j+1, "Z")]), -.25) for j in range(5)]
                    + [(label(6, [(j, "X")]), .5) for j in range(6)]}
    for n in (9, 17, 33):
        yield {"id": f"qrisp_ising_zero_field_chain{n}",
               "origin": "official_qsp_hamiltonian_parameter_and_size_variant",
               "source": source, "formula": "H=-0.25*sum_j Z_j Z_(j+1); B=0",
               "terms": [(label(n, [(j, "Z"), (j+1, "Z")]), -.25) for j in range(n-1)]}
    import networkx as nx
    edges = list(nx.cycle_graph(6).edges())
    yield {"id": "qrisp_heisenberg_ring6", "origin": "official_exact_qubitization_hamiltonian",
           "source": "https://qrisp.eu/general/tutorial/BE_tutorial/BE_vol2.html",
           "terms": [(label(6, [(j, axis), (k, axis)]), .125 if axis == "Z" else .25)
                     for j, k in edges for axis in "XYZ"]}
    for n in (2, 3, 4):
        matrix = np.eye(1 << n, k=1) + 3 * np.eye(1 << n)
        matrix[-1, 0] = 1
        yield {"id": f"qrisp_cyclic_shift_{n}q",
               "origin": "official_exact_matrix_pauli_input" if n == 2 else "official_matrix_size_variant",
               "source": "https://qrisp.eu/reference/Algorithms/generated/qrisp.gqsp.QSVT.html",
               "native_document_lcu": "3I + decrement; all three measured routes instead share its exact Pauli expansion",
               "terms": terms_from_matrix(matrix)}
    for n in (4, 8, 16, 32):
        for axis in ("Z", "X"):
            yield {"id": f"independent_{axis.lower()}_fields_{n}q",
                   "origin": "constructed_structural_stress_case",
                   "source": None,
                   "terms": [("I" * (n - j - 1) + axis + "I" * j, 1.0) for j in range(n)]}


def native_qrisp_select(terms):
    from qrisp import QuantumVariable, QuantumFloat, gphase, x, y, z
    from qrisp.jasp import q_switch

    def pauli(label, coefficient):
        def unitary(system):
            angle = float(np.angle(coefficient))
            if abs(angle) > 1e-14:
                gphase(angle, system[0])
            for bit, letter in enumerate(reversed(label)):
                if letter != "I":
                    {"X": x, "Y": y, "Z": z}[letter](system[bit])
        return unitary

    index = QuantumFloat((len(terms) - 1).bit_length(), name="select_address")
    system = QuantumVariable(len(terms[0][0]), name="select_system")
    q_switch(index, [pauli(label, c) for label, c in terms], system)
    return pack_fixed_ports(system.qs.copy(), [index], system)


def pl_select(terms):
    n = len(terms[0][0])
    m = (len(terms) - 1).bit_length()
    tape, wires, source_meta = build_lcu_tape(terms, n, scope="select", work_wires=max(0, m - 1))
    expanded, compile_meta = compile_lcu_tape(tape, wires, optimize=True, decomposition="graph")
    return tape_to_qiskit(expanded, wires), {
        "source": source_meta, "compile": compile_meta,
        "system_qubits": list(range(n)),
        "address_qubits": list(range(n + max(0, m - 1), n + max(0, m - 1) + m)),
        "clean_scratch_qubits": list(range(n, n + max(0, m - 1))),
    }


def ours_select(terms):
    backend = BlockEncoding(Matrixsum([(PauliAtom(label), c) for label, c in terms]), placement_options={})
    parts = backend.circuit_components("Matrix-order")
    metadata = parts.circuit.metadata
    placement = dict(metadata["matrix_order"]["placement"])
    residuals = backend.mobius_phase_result["g_modes_with_phase"]
    error = 0.0
    for label, coefficient in terms:
        address = int(placement[label], 2)
        accumulated = Pauli("I" * len(label))
        scalar = 1 + 0j
        for factor, mask, phase in residuals:
            mask_value = int(mask, 2)
            if address & mask_value == mask_value:
                accumulated = Pauli(factor).dot(accumulated)
                scalar *= np.exp(0.5j * np.pi * phase)
        assert Pauli((accumulated.z, accumulated.x)).to_label() == label
        error = max(error, float(abs(scalar * (-1j) ** int(accumulated.phase) - coefficient / abs(coefficient))))
    assert error < 1e-10, error
    return parts.select, {"canonical": metadata, "system_qubits": metadata["system_qubits"],
        "address_qubits": list(range(backend.ctrl_size)),
        "clean_scratch_qubits": list(range(backend.ctrl_size + len(terms[0][0]), parts.select.num_qubits)),
        "placement": placement, "symbolic_occupied_address_max_error": error,
        "residuals": residuals}


def measure(qc, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "select.qpy").open("wb") as handle:
        qpy.dump(qc, handle)
    result = {}
    for name, gates in BASES.items():
        started = time.perf_counter()
        lowered = transpile(qc, basis_gates=gates, **SETTINGS)
        assert set(lowered.count_ops()) <= set(gates)
        qasm = qasm2.dumps(lowered)
        (directory / f"{name}_opt0.qasm").write_text(qasm)
        counted = count_qasm_text(qasm, gate_set=name)
        assert counted["clifford"]["unknown"] == 0
        dump(directory / f"{name}_opt0.global_phase.json", {"radians": float(lowered.global_phase)})
        result[name] = {"G": counted["metric_total"], "NC": counted["clifford"]["non_clifford"],
                        "W": lowered.num_qubits, "CX": int(lowered.count_ops().get("cx", 0)),
                        "counts": dict(lowered.count_ops()), "depth": lowered.depth(),
                        "qasm": str((directory / f"{name}_opt0.qasm").relative_to(ROOT)),
                        "lowering_seconds": time.perf_counter() - started,
                        "global_phase": float(lowered.global_phase)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--methods", nargs="+", default=["ours", "qrisp", "pl"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    smoke = json.loads((OUT / "smoke.json").read_text())
    sources = source_identity()
    assert smoke["passed"] and sources == smoke["sources"]
    policy = {"smoke": str((OUT / "smoke.json").relative_to(ROOT)), "sources": sources,
              "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {p: version(p) for p in ("qiskit", "pennylane", "qrisp")},
              "scope": "SELECT only", "placement_options": {}, "basis": BASES, **SETTINGS,
              "pl_policy": "native Select partial=False, graph decomposition, work=m-1, qml.compile",
              "qrisp_policy": "native q_switch; fixed logical input ports; explicit temporary lifetime reuse"}
    dump(OUT / "settings.json", policy)
    summary = []
    for case in cases():
        if args.cases and case["id"] not in args.cases:
            continue
        directory = OUT / "cases" / case["id"]
        terms = case["terms"]
        record = {**case, "terms": [[label, c.real, c.imag] for label, c in terms],
                  "n": len(terms[0][0]), "K": len(terms), "alpha": float(sum(abs(c) for _, c in terms)),
                  "scope": "SELECT only", "methods": {}, "sources": sources}
        old_path = directory / "result.json"
        if args.resume and old_path.exists():
            old = json.loads(old_path.read_text())
            assert old["terms"] == record["terms"] and old["sources"] == sources
            record["methods"] = old["methods"]
        for name in args.methods:
            if name in record["methods"]:
                continue
            print(json.dumps({"case": case["id"], "method": name, "stage": "start"}), flush=True)
            method_dir = directory / name
            method_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with (method_dir / "build.log").open("w") as log, redirect_stdout(log):
                qc, meta = {"ours": ours_select, "qrisp": native_qrisp_select, "pl": pl_select}[name](terms)
            built = time.perf_counter() - started
            basis = measure(qc, method_dir)
            entry = {"basis": basis, "metadata": meta, "build_seconds": built}
            record["methods"][name] = entry
            dump(old_path, record)
            print(json.dumps({"case": case["id"], "method": name,
                              "basis": {b: {k: v[k] for k in ("G", "NC", "W")} for b, v in basis.items()},
                              "build_seconds": round(built, 3)}), flush=True)
        summary.append(record)
        dump(OUT / "screening.json", {"policy": policy, "cases": summary})


if __name__ == "__main__":
    main()
