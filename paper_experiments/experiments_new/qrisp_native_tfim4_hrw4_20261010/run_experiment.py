#!/usr/bin/env python3
"""Build full channels with Qrisp; lower to NAM with Qiskit opt0.

Inputs are serialized Kraus Pauli sums, without importing project compilers.
Run with the qiskit_simulate environment. Results and input snapshots are saved
next to this runner. Verification covers all 16 system basis columns.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

NAM = ["rz", "h", "x", "cx"]
SEED = 20261009


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["TFIM-4", "HRW-4"],
                        choices=["TFIM-4", "HRW-4"])
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    root = output.parents[2]
    inputs = args.input_dir or root / "paper_experiments/pending_review/rq1_paper_suite_20261010/full_v2/inputs"
    with tempfile.TemporaryDirectory(prefix="qrisp-native-nam-") as cache:
        os.environ["MPLCONFIGDIR"] = cache + "/mpl"
        os.environ["NUMBA_CACHE_DIR"] = cache + "/numba"
        os.environ["JAX_ENABLE_X64"] = "True"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.chdir(cache)
        import numpy as np
        from qiskit import QuantumCircuit, qasm2, qpy, transpile
        from qiskit.quantum_info import Pauli
        from qiskit_aer import AerSimulator
        from qrisp import QuantumFloat, QuantumVariable, barrier, gphase, prepare, x, y, z
        from qrisp.block_encodings import BlockEncoding
        from qrisp.jasp import q_switch

        versions = {p: importlib.metadata.version(p) for p in ("qrisp", "qiskit", "qiskit-aer", "numpy")}

        def phased_pauli(label, phase):
            def unitary(system):
                angle = float(np.angle(phase))
                if angle != 0:
                    gphase(angle, system[0])
                for bit, letter in enumerate(reversed(label)):
                    if letter != "I":
                        {"X": x, "Y": y, "Z": z}[letter](system[bit])
            return unitary

        def identity(*args):
            pass

        def branch(block):
            def apply(operand):
                address, system = operand
                block.unitary(address, system)
            return apply

        def encode(value, bits):
            return sum(((value >> bit) & 1) << position for bit, position in enumerate(bits))

        def choi(kraus):
            vectors = [k.reshape(-1, order="F") for k in kraus]
            return sum(np.outer(v, v.conj()) for v in vectors)

        def verify(circuit, interface, rows, scale):
            start = time.perf_counter()
            sb, ib = interface["system"], interface["kraus_index"]
            d = 1 << len(sb)
            system_rows = np.array([encode(s, sb) for s in range(d)])
            success_rows = np.array([[encode(j, ib) | s for s in system_rows]
                                     for j in range(1 << len(ib))])
            actual = np.empty((len(success_rows), d, d), dtype=complex)
            simulator = AerSimulator(method="statevector", max_parallel_threads=1,
                                     max_parallel_experiments=1)
            scratch_mask = sum(1 << b for b in interface["scratch"])
            leaked_rows = (np.arange(1 << circuit.num_qubits) & scratch_mask) != 0
            max_leakage = 0.0
            for column in range(d):
                test = QuantumCircuit(circuit.num_qubits)
                for bit, position in enumerate(sb):
                    if (column >> bit) & 1:
                        test.x(position)
                test.compose(circuit, inplace=True)
                test.save_statevector()
                state = np.asarray(simulator.run(test).result().get_statevector())
                actual[:, :, column] = state[success_rows]
                max_leakage = max(max_leakage, float(np.sum(np.abs(state[leaked_rows]) ** 2)))
            expected = [sum(float(c) * complex(re, im) * Pauli(label).to_matrix()
                            for label, c, re, im in row) for row in rows]
            target = choi(expected)
            residual = scale * choi(actual) - target
            max_error = float(np.max(np.abs(residual)))
            relative = float(np.linalg.norm(residual) / np.linalg.norm(target))
            trace_effect = sum(k.conj().T @ k for k in expected)
            expected_probability = float(np.trace(trace_effect).real / d / scale)
            measured_probability = float(sum(np.linalg.norm(k) ** 2 for k in actual) / d)
            result = {
                "method": "full unnormalized Choi from every system basis column of the saved NAM circuit",
                "system_basis_columns": d, "choi_max_abs_error": max_error,
                "choi_relative_frobenius_error": relative,
                "max_scratch_leakage_probability": max_leakage,
                "maximally_mixed_input_success_probability": measured_probability,
                "expected_maximally_mixed_input_success_probability": expected_probability,
                "input_trace_preservation_max_abs_error": float(np.max(np.abs(trace_effect - np.eye(d)))),
                "input_trace_effect_eigenvalue_range": np.linalg.eigvalsh(trace_effect)[[0, -1]].tolist(),
                "tolerance": 1e-10, "verification_time_s": time.perf_counter() - start,
                "passed": max(max_error, relative, max_leakage,
                              abs(measured_probability - expected_probability)) < 1e-10,
            }
            return result

        summaries = []
        for case in args.cases:
            case_dir = output / case
            case_dir.mkdir(exist_ok=True)
            local_input = case_dir / "input.json"
            source = inputs / (case + ".json")
            if not local_input.exists():
                local_input.write_bytes(source.read_bytes())
            payload = json.loads(local_input.read_text())
            rows = payload["input_kraus"]
            print(json.dumps({"case": case, "stage": "build", "term_counts": list(map(len, rows))}), flush=True)
            started = time.perf_counter()
            width = (max(map(len, rows)) - 1).bit_length()
            blocks = []
            for row in rows:
                coeffs = np.array([float(c) for _, c, _, _ in row])
                if not np.all(coeffs > 0):
                    raise ValueError("LCU magnitudes must be strictly positive before padding")
                unitaries = [phased_pauli(label, complex(re, im)) for label, _, re, im in row]
                # A common address register remains disjoint from all clean scratch.
                # Zero padding is representation padding, not extra Kraus terms.
                missing = (1 << width) - len(row)
                coeffs = np.pad(coeffs, (0, missing))
                unitaries += [identity] * missing
                blocks.append(BlockEncoding.from_lcu(coeffs, unitaries))
            alpha = np.array([float(b.alpha) for b in blocks])
            scale = float(alpha @ alpha)
            system = QuantumVariable(4, name="system")
            index = QuantumFloat((len(rows) - 1).bit_length(), name="kraus", qs=system.qs)
            address = QuantumFloat(width, name="lcu", qs=system.qs)
            # Make system ports live before scratch allocation. The full-channel
            # check below verifies the actual exported arbitrary-input interface.
            barrier(system.reg + index.reg + address.reg)
            weights = np.pad(alpha / np.sqrt(scale), (0, (1 << index.size) - len(rows)))
            prepare(index, weights)
            branches = [branch(block) for block in blocks]
            branches += [identity] * ((1 << index.size) - len(branches))
            q_switch(index, branches, (address, system), method="tree")
            build_seconds = time.perf_counter() - started
            started = time.perf_counter()
            compiled = system.qs.compile(workspace=0, cancel_qfts=False, compile_mcm=False)
            compile_seconds = time.perf_counter() - started
            interface = {name: [compiled.qubits.index(q) for q in reg.reg]
                         for name, reg in (("system", system), ("kraus_index", index), ("lcu_address", address))}
            live = set(sum(interface.values(), []))
            interface["scratch"] = [i for i in range(len(compiled.qubits)) if i not in live]
            started = time.perf_counter()
            exported = compiled.to_qiskit()
            export_seconds = time.perf_counter() - started
            started = time.perf_counter()
            lowered = transpile(exported, basis_gates=NAM, optimization_level=0,
                                seed_transpiler=SEED, qubits_initially_zero=False)
            lower_seconds = time.perf_counter() - started
            counts = {name: int(count) for name, count in lowered.count_ops().items()}
            if set(counts) - set(NAM):
                raise ValueError(f"Non-NAM operations: {counts}")
            non_clifford = sum(1 for inst in lowered.data if inst.operation.name == "rz"
                              and abs(float(inst.operation.params[0]) / (np.pi / 2)
                                      - round(float(inst.operation.params[0]) / (np.pi / 2))) > 1e-8)
            with (case_dir / "qrisp_export.qpy").open("wb") as f:
                qpy.dump(exported, f)
            with (case_dir / "nam_opt0.qpy").open("wb") as f:
                qpy.dump(lowered, f)
            qasm2.dump(lowered, case_dir / "nam_opt0.qasm")
            result = {
                "case": case, "versions": versions, "model": payload["model"],
                "input_source": str(source), "input_sha256": hashlib.sha256(local_input.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "configuration": {"native_outer": "prepare + q_switch(tree)", "native_inner": "BlockEncoding.from_lcu",
                                  "rewrite": False, "coefficients": "positive magnitudes; phases on Pauli unitaries",
                                  "address_policy": "all Kraus LCU lists zero-padded to a common address width",
                                  "address_qubits": width, "controlled_prepare": False,
                                  "qrisp_compile_workspace": 0, "qrisp_compile_mcm": False,
                                  "system_input": "arbitrary; initial barrier before scratch allocation",
                                  "basis_gates": NAM, "optimization_level": 0,
                                  "seed_transpiler": SEED, "qubits_initially_zero": False},
                "system_qubits": 4, "kraus_count": len(rows), "pauli_term_counts": list(map(len, rows)),
                "alpha_per_kraus": alpha.tolist(), "Lambda": scale,
                "G": sum(counts.values()), "Lambda_G": scale * sum(counts.values()),
                "W": lowered.num_qubits, "counts": counts, "non_clifford": non_clifford,
                "depth": lowered.depth(), "interface": interface,
                "timing_s": {"native_build": build_seconds, "qrisp_compile": compile_seconds,
                             "qiskit_export": export_seconds, "nam_opt0_lowering": lower_seconds,
                             "total_compile": build_seconds + compile_seconds + export_seconds + lower_seconds},
                "project_compiler_modules_imported": [name for name in ("channel_IR", "channel_LCU", "block_encoding", "qrisp_channel_adapter", "rq1_pipeline") if name in sys.modules],
                "status": "awaiting_semantic_verification",
            }
            save_json(case_dir / "results.json", result)
            print(json.dumps({"case": case, "stage": "verify", "G": result["G"], "W": result["W"], "counts": counts}), flush=True)
            result["verification"] = verify(lowered, interface, rows, scale)
            result["status"] = "passed" if result["verification"]["passed"] else "failed_semantics"
            save_json(case_dir / "results.json", result)
            summaries.append(result)
            print(json.dumps({"case": case, "status": result["status"], "G": result["G"], "W": result["W"], "verification": result["verification"]}), flush=True)
        save_json(output / "results.json", summaries)
        if any(r["status"] != "passed" for r in summaries):
            raise SystemExit("Full channel verification failed; do not use these metrics as valid results")


if __name__ == "__main__":
    main()
