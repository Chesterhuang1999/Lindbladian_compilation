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
    from channel_IR import Matrixsum, PauliAtom, channel_ensemble
    from channel_LCU import channel_to_LCU
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
        # Exercise the optimized channel workflow on mixed block sizes. The
        # omitted fstar argument deliberately checks the tree route's default.
        kraus = [Matrixsum([(PauliAtom(label), coefficient) for label, coefficient in row])
                 for row in [[("II", 0.6), ("XZ", 0.2j), ("YX", -0.1)],
                             [("ZI", 0.3), ("IX", -0.1j)], [("YY", 0.15j)]]]
        # Capture the input before channel_ensemble canonicalizes Kraus phases.
        matrices = [operator.eff_op().to_matrix() for operator in kraus]
        expected_subnorm = sum(operator.pauli_norm() ** 2 for operator in kraus)
        channel, registers = channel_to_LCU(
            channel_ensemble([kraus]), structure="tree", opt="Matrix-order")
        metadata = channel.metadata
        assert metadata["block_backend"] == "block_encoding.BlockEncoding"
        assert metadata["outer_select"] == "tree" and metadata["block_opt"] == "Matrix-order"
        assert metadata["fstar"] is True and metadata["controlled_prepare"] is False
        assert metadata["outer_control_scope"] == "select-only"
        select, tree_work, address, system = registers
        subnorm = metadata["subnormalization_factor"]
        assert np.isclose(subnorm, expected_subnorm, rtol=1e-12, atol=1e-12)
        dimension = 1 << len(system)
        projected_kraus = np.zeros((1 << len(select), dimension, dimension), dtype=complex)
        reference_outputs = []
        tree_work_error = 0.0
        for value in range(dimension):
            initial_index = sum(((value >> bit) & 1) << physical for bit, physical in enumerate(system))
            evolved = Statevector.from_int(initial_index, 1 << channel.num_qubits).evolve(channel).data
            reference_outputs.append(evolved.copy())
            for branch in range(1 << len(select)):
                for row in range(dimension):
                    index = sum(((branch >> bit) & 1) << physical for bit, physical in enumerate(select))
                    index |= sum(((row >> bit) & 1) << physical for bit, physical in enumerate(system))
                    projected_kraus[branch, row, value] = evolved[index]
            dirty = np.zeros(len(evolved), dtype=bool)
            for physical in tree_work:
                dirty |= ((np.arange(len(evolved)) >> physical) & 1).astype(bool)
            tree_work_error = max(tree_work_error, float(np.max(np.abs(evolved[dirty]), initial=0)))
        # Choi equality checks the complete unnormalized CP map, including its
        # success scale, while allowing branch permutation and Kraus phases.
        actual_vectors = np.column_stack([matrix.reshape(-1, order="F") for matrix in projected_kraus])
        expected_vectors = np.column_stack([matrix.reshape(-1, order="F") for matrix in matrices])
        actual_choi = actual_vectors @ actual_vectors.conj().T
        expected_choi = expected_vectors @ expected_vectors.conj().T / expected_subnorm
        channel_error = float(np.max(np.abs(actual_choi - expected_choi)))
        channel_metrics = {}
        for name, gates in (("nam", ["rz", "h", "x", "cx"]), ("ibm", ["u1", "u2", "u3", "cx"])):
            lowered = transpile_channel_circuit(channel, basis_gates=gates, optimization_level=0,
                                                 seed_transpiler=20261009)
            error = 0.0
            for value, expected in enumerate(reference_outputs):
                index = sum(((value >> bit) & 1) << physical for bit, physical in enumerate(system))
                actual = Statevector.from_int(index, 1 << lowered.num_qubits).evolve(lowered).data
                error = max(error, float(np.max(np.abs(actual - expected))))
            channel_metrics[name] = {"G": int(sum(lowered.count_ops().values())), "W": lowered.num_qubits,
                                     "lowering_error": error}
        channel_check = {"structure": "tree", "opt": "Matrix-order", "fstar_argument": None,
                         "metadata": metadata, "system_basis_inputs": len(reference_outputs),
                         "channel_choi_error": channel_error, "tree_work_error": tree_work_error,
                         "channel_validation": "unnormalized projected Choi matrix on all system inputs",
                         "metrics": channel_metrics,
                         "passed": max(channel_error, tree_work_error,
                             *(value["lowering_error"] for value in channel_metrics.values())) < 1e-10}
    result = {"sources": source_identity(), "checks": checks, "channel_check": channel_check,
              "passed": all(x["passed"] for x in checks) and channel_check["passed"]}
    (output / "smoke.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for check in checks:
        print(json.dumps({k: check[k] for k in ("case", "passed", "block_error", "controlled_block_error",
                                               "archived_controlled_nam_G", "archive_count_matches")}, allow_nan=False), flush=True)
    print(json.dumps({"case": "channel-tree-matrix-order-default-fstar", "passed": channel_check["passed"],
                      "channel_choi_error": channel_error, "tree_work_error": tree_work_error,
                      "lowering_errors": {name: value["lowering_error"] for name, value in channel_metrics.items()}},
                     allow_nan=False), flush=True)
    if not result["passed"]:
        raise SystemExit("Smoke validation failed; benchmark execution must remain gated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    main(parser.parse_args().out_dir)
