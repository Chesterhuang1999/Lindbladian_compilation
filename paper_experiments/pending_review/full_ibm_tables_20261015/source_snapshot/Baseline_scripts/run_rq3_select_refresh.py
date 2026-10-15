"""Replay frozen PennyLane comparison inputs through the canonical SELECT export."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
OUTPUT = ROOT / "Baseline_results/rq3_refresh_20261013/select"
INPUTS = ROOT / "Baseline_results/pennylane_lcu_v1/inputs.json"


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    import numpy as np
    from qiskit import qasm2
    from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
    from block_encoding import BlockEncoding
    from channel_IR import Matrixsum, PauliAtom
    from baseline_common import count_qasm_text
    from experiment_protocol import source_identity, transpile_channel_circuit

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    smoke = json.loads((OUTPUT.parent / "smoke.json").read_text())
    assert smoke["passed"], "Run the archive smoke check before benchmarks."
    sources = source_identity()
    assert sources == smoke["sources"], "Backend changed after the required smoke check."
    manifest = json.loads(INPUTS.read_text())
    historical = json.loads((ROOT / "Baseline_results/pennylane_lcu_comparison/comparison.json").read_text())
    policy = {"input_manifest": str(INPUTS.relative_to(ROOT)),
              "input_manifest_sha256": sha256(INPUTS.read_bytes()).hexdigest(),
              "sources": sources, "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "SELECT from circuit_components, excluding common preparation",
              "placement_options": {}, "qiskit_optimization_level": 0,
              "seed_transpiler": 7, "qubits_initially_zero": False,
              "pl_qrisp_policy": "reuse saved same-input SELECT comparisons; historical audit retained",
              "width_aggregation": "maximum across Kraus blocks; mean across random samples"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dump(OUTPUT / "settings.json", policy)
    all_rows = []
    for case in manifest["cases"]:
        if args.cases and case["name"] not in args.cases:
            continue
        directory = OUTPUT / case["name"]
        directory.mkdir(parents=True, exist_ok=True)
        completed_path = directory / "result.json"
        if args.resume and completed_path.exists():
            previous = json.loads(completed_path.read_text())
            if (previous.get("passed") and previous.get("sources") == sources
                    and [x["input_sha256"] for x in previous["inputs"]]
                    == [x["input_sha256"] for x in case["inputs"]]):
                all_rows.extend(previous["rows"])
                print(json.dumps({"case": case["name"], "stage": "already_completed"}), flush=True)
                continue
        entries = []
        begun = time.perf_counter()
        print(json.dumps({"case": case["name"], "stage": "start", "inputs": len(case["inputs"])}), flush=True)
        with (directory / "run.log").open("w") as log, redirect_stdout(log):
            for input_index, item in enumerate(case["inputs"]):
                terms = [(t["pauli"], complex(t["real"], t["imag"])) for t in item["terms"]]
                target = dict(terms)
                assert len(target) == len(terms), "Repeated labels require a separately specified address contract."
                matrix_sum = Matrixsum([(PauliAtom(label), coefficient) for label, coefficient in terms])
                started = time.perf_counter()
                backend = BlockEncoding(matrix_sum, placement_options={})
                parts = backend.circuit_components("Matrix-order")
                construction = time.perf_counter() - started
                residuals = backend.mobius_phase_result["g_modes_with_phase"]
                placement = dict(parts.circuit.metadata["matrix_order"]["placement"])
                phase_error = 0.0
                for label, coefficient in terms:
                    address = int(placement[label], 2)
                    accumulated = Pauli("I" * len(label))
                    scalar = 1.0 + 0j
                    for factor_label, mask, phase in residuals:
                        mask_value = int(mask, 2)
                        if address & mask_value == mask_value:
                            accumulated = Pauli(factor_label).dot(accumulated)
                            scalar *= np.exp(0.5j * np.pi * phase)
                    phase_free = Pauli((accumulated.z, accumulated.x)).to_label()
                    assert phase_free == label, (label, address, phase_free)
                    actual_phase = scalar * (-1j) ** int(accumulated.phase)
                    expected_phase = coefficient / abs(coefficient)
                    phase_error = max(phase_error, float(abs(actual_phase - expected_phase)))
                assert phase_error < 1e-10, phase_error
                basis = {}
                lowered_circuits = {}
                for name, gates in (("nam", ["rz", "h", "x", "cx"]), ("ibm", ["u1", "u2", "u3", "cx"])):
                    started = time.perf_counter()
                    lowered = transpile_channel_circuit(parts.select, basis_gates=gates,
                                                         optimization_level=0, seed_transpiler=7)
                    lowering_s = time.perf_counter() - started
                    qasm = qasm2.dumps(lowered)
                    path = directory / f"{item['id']}_{name}_opt0.qasm"
                    path.write_text(qasm)
                    counted = count_qasm_text(qasm, gate_set=name)
                    assert set(lowered.count_ops()) <= set(gates), dict(lowered.count_ops())
                    assert counted["clifford"]["unknown"] == 0
                    basis[name] = {"G": counted["metric_total"], "NC": counted["clifford"]["non_clifford"],
                                   "W": lowered.num_qubits, "CX": int(lowered.count_ops().get("cx", 0)),
                                   "depth": lowered.depth(), "counts": dict(lowered.count_ops()),
                                   "global_phase": float(lowered.global_phase), "lowering_s": lowering_s,
                                   "qasm": str(path.relative_to(ROOT))}
                    lowered_circuits[name] = lowered
                verification = {"method": "phase-sensitive Pauli products at every occupied address",
                                "occupied_addresses_checked": len(terms), "symbolic_max_error": phase_error}
                # Every Pauli product is checked above, including large-width inputs.
                # Small representative inputs additionally exercise both actual lowerings.
                if input_index == 0 and matrix_sum.size <= 4 and parts.circuit.num_qubits <= 12:
                    system = parts.circuit.metadata["system_qubits"]
                    index_values = np.zeros(1 << matrix_sum.size, dtype=np.int64)
                    for logical, physical in enumerate(system):
                        index_values |= ((np.arange(len(index_values)) >> logical) & 1) << physical
                    target_matrix = SparsePauliOp.from_list(terms).to_matrix() / matrix_sum.pauli_norm()
                    encoded = np.column_stack([
                        Statevector.from_int(int(i), 1 << parts.circuit.num_qubits).evolve(parts.circuit).data[index_values]
                        for i in index_values])
                    block_error = float(np.max(np.abs(encoded - target_matrix)))
                    assert block_error < 1e-10, block_error
                    verification["encoded_block_max_error"] = block_error
                    address_width = backend.ctrl_size
                    chosen = [int(placement[terms[0][0]], 2), int(placement[terms[-1][0]], 2)]
                    lowering_errors = {}
                    for name, lowered in lowered_circuits.items():
                        error = 0.0
                        for address in chosen:
                            for system_value in (0, (1 << matrix_sum.size) - 1):
                                index = int(index_values[system_value]) | address
                                expected = Statevector.from_int(index, 1 << lowered.num_qubits).evolve(parts.select).data
                                actual = Statevector.from_int(index, 1 << lowered.num_qubits).evolve(lowered).data
                                error = max(error, float(np.max(np.abs(actual - expected))))
                        assert error < 1e-10, error
                        lowering_errors[name] = error
                    verification["lowering_max_errors"] = lowering_errors
                    verification["select_input_states_checked_per_basis"] = 4
                record = {"id": item["id"], "seed": item["seed"], "input_sha256": item["input_sha256"],
                          "construction_s": construction, "basis": basis, "verification": verification,
                          "metadata": parts.circuit.metadata, "historical_input_check": item["historical_check"],
                          "passed": True}
                dump(directory / f"{item['id']}.json", record)
                entries.append(record)
                dump(directory / "result.partial.json", {"case": case["name"], "completed": len(entries),
                     "expected": len(case["inputs"]), "inputs": entries, "sources": sources})
        rows = []
        mode = case["aggregation"]
        for name in ("nam", "ibm"):
            values = [entry["basis"][name] for entry in entries]
            aggregate = lambda metric: (statistics.fmean(v[metric] for v in values) if mode == "sample_mean"
                                       else max(v[metric] for v in values) if metric == "W"
                                       else sum(v[metric] for v in values))
            ours = {metric: aggregate(metric) for metric in ("G", "NC", "W", "CX")}
            previous = next(row for row in historical["main_rows"] if row["case"] == case["name"]
                            and row["basis"] == name and row["scope"] == "select" and row["profile"] == "graph_work")
            qrisp = previous["references"]["qrisp"]
            pl = previous["pennylane"]
            rows.append({"case": case["name"], "basis": name, "aggregation": mode, "inputs": len(entries),
                         "ours": ours, "pl": pl, "qrisp": qrisp,
                         "qrisp_over_ours": qrisp["gate_count"] / ours["G"],
                         "pl_over_ours": pl["gate_count"] / ours["G"], "status": "ok"})
        result = {"case": case["name"], "aggregation": mode, "inputs": entries, "rows": rows,
                  "seconds": time.perf_counter() - begun, "sources": sources, "passed": True,
                  "runner_sha256": policy["runner_sha256"]}
        dump(directory / "result.json", result)
        all_rows.extend(rows)
        print(json.dumps({"case": case["name"], "stage": "complete", "rows": rows,
                          "seconds": round(result["seconds"], 3)}), flush=True)
        dump(OUTPUT / "summary.partial.json", {"policy": policy, "rows": all_rows, "completed": False})
    dump(OUTPUT / "summary.json", {"policy": policy, "rows": all_rows, "completed": True})


if __name__ == "__main__":
    main()
