"""Check canonical block exports against post-2026-10-07 archived fixtures."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]


def main(output=None):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from block_encoding import BlockEncoding
    from block_encoding_new import BlockEncoding as CompatibilityBlockEncoding
    from channel_IR import Matrixsum, PauliAtom
    from experiment_protocol import source_identity, transpile_channel_circuit

    assert CompatibilityBlockEncoding is BlockEncoding
    output = Path(output) if output is not None else ROOT / "Baseline_results/rq3_refresh_20261013"
    output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for case in ("TFIM-3", "TFIM-4"):
        path = ROOT / "paper_experiments/pending_review/matrix_order_pipeline_20261010" / case / "default/results.json"
        data = json.loads(path.read_text())
        terms = [(label, magnitude * complex(real, imag))
                 for label, magnitude, real, imag in data["compiled_kraus"][0]]
        fixtures.append((case, terms, data["metadata"]["block_metadata"][0], path))
    for case in ("Local_QSW3", "Decay-HO"):
        directory = ROOT / "paper_experiments/pending_review/support_gatecost_hybrid_20261012/cases" / case
        path = directory / "hybrid_circuit.json"
        data = json.loads(path.read_text())
        with np.load(directory / "hybrid.npz") as saved:
            row = next(row for row in saved["A_final"] if np.any(row != 0))
            terms = [(str(label), complex(coefficient))
                     for label, coefficient in zip(saved["labels"], row)
                     if coefficient != 0]
        fixtures.append((case, terms, data["metadata"]["block_metadata"][0], path))

    def indices(circuit, system, control=None):
        values = np.arange(1 << len(system), dtype=np.int64)
        result = np.zeros_like(values)
        for logical, physical in enumerate(system):
            result |= ((values >> logical) & 1) << physical
        if control is not None:
            result |= control
        return result

    def encoded(circuit, system):
        projected = indices(circuit, system)
        result = np.empty((len(projected), len(projected)), dtype=complex)
        states = []
        for column, input_index in enumerate(projected):
            initial = Statevector.from_int(int(input_index), 1 << circuit.num_qubits)
            evolved = initial.evolve(circuit).data
            result[:, column] = evolved[projected]
            states.append(evolved)
        return result, np.asarray(states).T

    checks = []
    with (output / "smoke.log").open("w") as log, redirect_stdout(log):
        for case, terms, reference, path in fixtures:
            started = time.perf_counter()
            matrix_sum = Matrixsum([(PauliAtom(label, phase=coefficient / abs(coefficient)), abs(coefficient))
                                    for label, coefficient in terms])
            backend = BlockEncoding(matrix_sum, placement_options={})
            parts = backend.circuit_components("Matrix-order")
            system = parts.circuit.metadata["system_qubits"]
            target = SparsePauliOp.from_list(terms).to_matrix() / sum(abs(c) for _, c in terms)
            block, outputs = encoded(parts.circuit, system)
            error = float(np.max(np.abs(block - target)))
            assembled = QuantumCircuit(parts.circuit.num_qubits)
            assembled.compose(parts.prepare, inplace=True)
            assembled.compose(parts.select, inplace=True)
            assembled.compose(parts.prepare.inverse(), inplace=True)
            _, composed_outputs = encoded(assembled, system)
            component_error = float(np.max(np.abs(outputs - composed_outputs)))
            controlled_parts = parts.with_outer_control(1)
            controlled = controlled_parts.controlled(1)
            controlled_system = [q + 1 for q in controlled_parts.circuit.metadata["system_qubits"]]
            active_indices = indices(controlled, controlled_system, 1)
            active = np.empty_like(target)
            inactive_error = 0.0
            for column, input_index in enumerate(active_indices):
                evolved = Statevector.from_int(int(input_index), 1 << controlled.num_qubits).evolve(controlled).data
                active[:, column] = evolved[active_indices]
                zero_index = int(input_index) ^ 1
                initial = Statevector.from_int(zero_index, 1 << controlled.num_qubits)
                expected_inactive = initial.data.copy()
                inactive_error = max(inactive_error, float(np.max(np.abs(initial.evolve(controlled).data - expected_inactive))))
            controlled_error = float(np.max(np.abs(active - target)))
            basis_metrics = {}
            for name, gates in (("nam", ["rz", "h", "x", "cx"]), ("ibm", ["u1", "u2", "u3", "cx"])):
                lowered = transpile_channel_circuit(controlled, basis_gates=gates, optimization_level=0,
                                                     seed_transpiler=20261009)
                active_lowered = np.column_stack([
                    Statevector.from_int(int(index), 1 << lowered.num_qubits).evolve(lowered).data[active_indices]
                    for index in active_indices])
                lowering_error = float(np.max(np.abs(active_lowered - active)))
                basis_metrics[name] = {"G": int(sum(lowered.count_ops().values())), "W": lowered.num_qubits,
                                       "counts": dict(lowered.count_ops()), "lowering_error": lowering_error}
            archived_cost = reference.get("matrix_order", {}).get("nam_gates")
            archive_matches = archived_cost is None or basis_metrics["nam"]["G"] == archived_cost
            entry = {"case": case, "reference": str(path.relative_to(ROOT)),
                     "reference_sha256": sha256(path.read_bytes()).hexdigest(),
                     "terms": [[label, coefficient.real, coefficient.imag] for label, coefficient in terms],
                     "block_error": error, "component_error": component_error,
                     "controlled_block_error": controlled_error, "inactive_error": inactive_error,
                     "metrics": basis_metrics, "archived_controlled_nam_G": archived_cost,
                     "archive_count_matches": archive_matches,
                     "metadata": parts.circuit.metadata,
                     "passed": archive_matches and max(error, component_error, controlled_error, inactive_error,
                         *(m["lowering_error"] for m in basis_metrics.values())) < 1e-10,
                     "seconds": time.perf_counter() - started}
            checks.append(entry)
    result = {"sources": source_identity(), "checks": checks, "passed": all(x["passed"] for x in checks)}
    (output / "smoke.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for check in checks:
        print(json.dumps({k: check[k] for k in ("case", "passed", "block_error", "controlled_block_error",
                                               "archived_controlled_nam_G", "archive_count_matches")}, allow_nan=False), flush=True)
    if not result["passed"]:
        raise SystemExit("Smoke validation failed; benchmark execution must remain gated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    main(parser.parse_args().out_dir)
