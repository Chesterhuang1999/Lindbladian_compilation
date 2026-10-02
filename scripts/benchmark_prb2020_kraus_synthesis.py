#!/usr/bin/env python3
"""Reproducible synthesis experiment for PRB 102, 134310 (2020), KC2.

Generate new seeded realizations of the paper's ensemble, not the authors'
unpublished matrices. Keep Kraus-block and full-channel costs separate.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer
import scipy
from qiskit import QuantumCircuit, qasm2, qpy, transpile
from qiskit.circuit.library import Isometry, StatePreparation, UCRYGate, UCRZGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.stats import unitary_group

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
from baseline_common import count_qasm_text  # noqa: E402
from block_encoding_new import BlockEncoding  # noqa: E402
from channel_IR import Matrixsum, PauliAtom  # noqa: E402

BASIS = {"ibm": ["u1", "u2", "u3", "cx"], "nam": ["rz", "h", "x", "cx"]}
TOL = 1e-8


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def complex_json(a: np.ndarray) -> dict:
    return {"real": a.real.tolist(), "imag": a.imag.tolist()}


def choi(matrices: list[np.ndarray]) -> np.ndarray:
    vectors = [a.reshape(-1, order="F") for a in matrices]
    return sum(np.outer(v, v.conj()) for v in vectors)


def phase_error(actual: np.ndarray, target: np.ndarray) -> float:
    overlap = np.vdot(target, actual)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    return float(np.linalg.norm(actual - phase * target, ord="fro"))


def unitary_circuit(matrix: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(round(math.log2(len(matrix))))
    qc.unitary(matrix, range(qc.num_qubits))
    return transpile(qc, basis_gates=["u", "cx"], optimization_level=0, seed_transpiler=0)


def decomposition(a: np.ndarray) -> dict:
    left, singular, right = np.linalg.svd(a, full_matrices=True)
    alpha = float(singular[0])
    normalized = np.clip(singular / alpha, 0, 1)
    normalized[np.abs(normalized - 1) < 1e-12] = 1
    theta = np.arccos(normalized)
    plus = (left * np.exp(1j * theta)) @ right
    minus = (left * np.exp(-1j * theta)) @ right
    error = float(np.linalg.norm(a - alpha * (plus + minus) / 2))
    unitary_error = max(float(np.linalg.norm(u.conj().T @ u - np.eye(len(a))))
                        for u in (plus, minus))
    assert error < TOL and unitary_error < TOL
    return {"left": left, "right": right, "singular": singular, "alpha": alpha,
            "theta": theta, "plus": plus, "minus": minus,
            "decomposition_error": error, "unitarity_error": unitary_error,
            "scaled_unitary": bool(np.max(theta) < 1e-10)}


def blocks(a: np.ndarray, dec: dict) -> dict[str, tuple[QuantumCircuit, float, float]]:
    n = round(math.log2(len(a)))
    alpha = dec["alpha"]
    output = {}
    if dec["scaled_unitary"]:
        start = time.perf_counter()
        output["native_scaled_unitary"] = (unitary_circuit(a / alpha), alpha,
                                            time.perf_counter() - start)
        return output

    start = time.perf_counter()
    plus = unitary_circuit(dec["plus"])
    plus_seconds = time.perf_counter() - start
    start = time.perf_counter()
    minus = unitary_circuit(dec["minus"])
    minus_seconds = time.perf_counter() - start
    synthesis_seconds = plus_seconds + minus_seconds
    output["unitary_plus_only"] = (plus, 1.0, plus_seconds)
    output["unitary_minus_only"] = (minus, 1.0, minus_seconds)

    start = time.perf_counter()
    qc = QuantumCircuit(n + 1)
    qc.h(n)
    qc.append(plus.to_gate().control(1, ctrl_state=0), [n] + list(range(n)))
    qc.append(minus.to_gate().control(1, ctrl_state=1), [n] + list(range(n)))
    qc.h(n)
    output["two_unitary_independent_lcu"] = (qc, alpha,
                                             synthesis_seconds + time.perf_counter() - start)

    for method, gate in (("two_unitary_shared_svd", UCRZGate), ("svd_ucry_dilation", UCRYGate)):
        start = time.perf_counter()
        qc = QuantumCircuit(n + 1)
        if method == "two_unitary_shared_svd":
            qc.h(n)
        qc.compose(unitary_circuit(dec["right"]), qubits=range(n), inplace=True)
        sign = -1 if method == "two_unitary_shared_svd" else 1
        qc.append(gate((sign * 2 * dec["theta"]).tolist()), [n] + list(range(n)))
        qc.compose(unitary_circuit(dec["left"]), qubits=range(n), inplace=True)
        if method == "two_unitary_shared_svd":
            qc.h(n)
        output[method] = (qc, alpha, time.perf_counter() - start)

    start = time.perf_counter()
    right_vectors = dec["right"].conj().T
    defect = (right_vectors * np.sin(dec["theta"])) @ dec["right"]
    iso = np.vstack([a / alpha, defect])
    assert np.linalg.norm(iso.conj().T @ iso - np.eye(len(a))) < TOL
    qc = QuantumCircuit(n + 1)
    qc.append(Isometry(iso, 0, 0), range(n + 1))
    output["direct_block_isometry"] = (qc, alpha, time.perf_counter() - start)

    start = time.perf_counter()
    paulis = SparsePauliOp.from_operator(Operator(a), atol=1e-12, rtol=0)
    ms = Matrixsum([(PauliAtom(label, phase=complex(c / abs(c))), float(abs(c)))
                    for label, c in paulis.to_list()])
    with contextlib.redirect_stdout(io.StringIO()):
        original = BlockEncoding(ms).circuit(opt="Matrix-order")
    sys_old = [original.find_bit(q).index for reg in original.qregs if reg.name == "sys" for q in reg]
    assert len(sys_old) == n
    other_old = [q for q in range(original.num_qubits) if q not in sys_old]
    mapping = [0] * original.num_qubits
    for new, old in enumerate(sys_old + other_old):
        mapping[old] = new
    qc = QuantumCircuit(original.num_qubits)
    qc.compose(original, qubits=mapping, inplace=True)
    output["pauli_current_matrix_order"] = (qc, float(sum(abs(paulis.coeffs))), time.perf_counter() - start)
    return output


def compile_row(qc: QuantumCircuit, context: dict, alpha: float, build_time: float,
                target: np.ndarray | list[np.ndarray], output_dir: Path,
                basis: str, opt: int) -> dict:
    start = time.perf_counter()
    native = transpile(qc, basis_gates=BASIS[basis], optimization_level=opt,
                       seed_transpiler=0, approximation_degree=1.0)
    elapsed = time.perf_counter() - start
    assert set(native.count_ops()) <= set(BASIS[basis])
    text = qasm2.dumps(native)
    counts = count_qasm_text(text, gate_set=basis)
    suffix = f"{context['method']}_{basis}_opt{opt}"
    (output_dir / f"{suffix}.qasm").write_text(text, encoding="utf-8")
    with (output_dir / f"{suffix}.qpy").open("wb") as handle:
        qpy.dump(native, handle)
    dim = 2 ** context["system_qubits"]
    if context["scope"] == "full_channel":
        # All input columns with initialized environment suffice to verify the map.
        simulator = AerSimulator(method="statevector", max_parallel_threads=1)
        columns = []
        for column in range(dim):
            probe = QuantumCircuit(native.num_qubits)
            for bit in range(context["system_qubits"]):
                if column & (1 << bit):
                    probe.x(bit)
            probe.compose(native, inplace=True)
            probe.save_statevector()
            result = simulator.run(probe).result()
            assert result.success
            columns.append(np.asarray(result.get_statevector()))
        matrix = np.column_stack(columns)
    else:
        matrix = Operator(native).data
    if context["scope"] == "full_channel":
        r = len(target)
        if context["method"] == "direct_channel_isometry":
            realized = [matrix[j * dim:(j + 1) * dim, :dim] for j in range(r)]
        else:
            # System is low, block ancilla next, and Kraus index highest.
            realized = [matrix[j * 2 * dim:j * 2 * dim + dim, :dim] * math.sqrt(alpha)
                        for j in range(r)]
        reference = choi(target)
        error = float(np.linalg.norm(choi(realized) - reference) / np.linalg.norm(reference))
        success = 1.0 / alpha
    else:
        error = phase_error(matrix[:dim, :dim], target / alpha)
        success = None
    assert error < TOL, (context, basis, opt, error)
    return {**context, "basis": basis, "opt_level": opt, "status": "ok",
            "total_gates": counts["metric_total"], "cx": counts["ops"].get("cx", 0),
            "non_clifford": counts["clifford"]["non_clifford"],
            "width": native.num_qubits, "depth": native.depth(),
            "alpha_or_lambda": alpha, "channel_success": success,
            "build_seconds": build_time, "transpile_seconds": elapsed,
            "semantic_error": error, "global_phase": float(native.global_phase),
            "qasm": str(output_dir / f"{suffix}.qasm")}


def full_channel_circuits(kraus: list[np.ndarray], cached: list[dict]) -> dict:
    n = round(math.log2(len(kraus[0])))
    r = len(kraus)
    k = math.ceil(math.log2(r))
    dim = 2 ** n
    iso = np.zeros((dim * 2 ** k, dim), dtype=complex)
    iso[:r * dim] = np.vstack(kraus)
    start = time.perf_counter()
    qc = QuantumCircuit(n + k)
    qc.append(Isometry(iso, 0, 0), range(n + k))
    result = {"direct_channel_isometry": (qc, 1.0, time.perf_counter() - start)}
    for method in ("two_unitary_independent_lcu", "two_unitary_shared_svd", "svd_ucry_dilation"):
        start = time.perf_counter()
        alpha = np.array([d["alpha"] for d in cached])
        lam = float(np.sum(alpha ** 2))
        amplitudes = np.zeros(2 ** k)
        amplitudes[:r] = alpha / math.sqrt(lam)
        qc = QuantumCircuit(n + 1 + k)
        idx = list(range(n + 1, n + 1 + k))
        qc.append(StatePreparation(amplitudes), idx)
        local_build_seconds = 0.0
        for j, d in enumerate(cached):
            local, _, local_seconds = d["blocks"].get(method, d["blocks"].get("native_scaled_unitary"))
            local_build_seconds += local_seconds
            qc.append(local.to_gate().control(k, ctrl_state=j), idx + list(range(local.num_qubits)))
        result[method] = (qc, lam, local_build_seconds + time.perf_counter() - start)
    return result


def run(args) -> None:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((out / "rows.json").read_text()) if args.resume and (out / "rows.json").exists() else []
    metadata = []
    keys = ("seed", "kraus_index", "method", "basis", "opt_level")
    completed = {tuple(row[key] for key in keys) for row in rows}
    for seed in args.seeds:
        sample_dir = out / f"seed{seed}"
        sample_dir.mkdir(exist_ok=True)
        rng = np.random.default_rng(seed)
        n, d, p = args.system_qubits, args.dissipative_count, args.p
        dim = 2 ** n
        start = time.perf_counter()
        u = unitary_group.rvs(dim, random_state=rng)
        v = unitary_group.rvs(dim * d, random_state=rng)
        kraus = [math.sqrt(1 - p) * u] + [math.sqrt(p) * v[j * dim:(j + 1) * dim, :dim]
                                         for j in range(d)]
        generation_seconds = time.perf_counter() - start
        tp_error = float(np.linalg.norm(sum(a.conj().T @ a for a in kraus) - np.eye(dim)))
        assert tp_error < TOL
        np.savez_compressed(sample_dir / "matrices.npz", U=u, V=v, kraus=np.array(kraus))
        write_json(sample_dir / "kraus_matrices.json", {f"K{j}": complex_json(a) for j, a in enumerate(kraus)})
        cached, description = [], []
        for j, a in enumerate(kraus):
            dec = decomposition(a)
            paulis = SparsePauliOp.from_operator(Operator(a), atol=1e-12, rtol=0)
            description.append({"index": j, "spectral_norm": dec["alpha"],
                                "singular_values": dec["singular"].tolist(),
                                "scaled_unitary": dec["scaled_unitary"],
                                "pauli_terms": len(paulis), "pauli_l1": float(sum(abs(paulis.coeffs))),
                                "plus_pauli_terms": len(SparsePauliOp.from_operator(Operator(dec["plus"]), atol=1e-12, rtol=0)),
                                "minus_pauli_terms": len(SparsePauliOp.from_operator(Operator(dec["minus"]), atol=1e-12, rtol=0)),
                                "decomposition_error": dec["decomposition_error"],
                                "unitarity_error": dec["unitarity_error"]})
            selected = args.micro_indices is None or j in args.micro_indices
            need_full = not args.skip_full and seed == args.seeds[0]
            if selected or need_full:
                dec["blocks"] = blocks(a, dec)
            cached.append(dec)
            if not selected:
                continue
            block_dir = sample_dir / f"kraus{j:02d}"
            block_dir.mkdir(exist_ok=True)
            for method, (qc, alpha, build_time) in dec["blocks"].items():
                scope = "unitary_only" if method.startswith("unitary_") else "kraus_block"
                target = dec["plus"] if method == "unitary_plus_only" else dec["minus"] if method == "unitary_minus_only" else a
                context = {"seed": seed, "p": p, "system_qubits": n, "d": d,
                           "kraus_index": j, "scope": scope, "method": method}
                for basis in args.bases:
                    for opt in args.opt_levels:
                        if (seed, j, method, basis, opt) in completed:
                            continue
                        rows.append(compile_row(qc, context, alpha, build_time, target, block_dir, basis, opt))
            write_json(out / "rows.json", rows)
            print(f"seed={seed} kraus={j} complete rows={len(rows)}", flush=True)
        metadata.append({"seed": seed, "system_qubits": n, "d": d, "p": p,
                         "tp_error": tp_error, "generation_seconds": generation_seconds,
                         "lambda_two_unitary": float(sum(x["spectral_norm"] ** 2 for x in description)),
                         "lambda_pauli": float(sum(x["pauli_l1"] ** 2 for x in description)),
                         "kraus": description})
        write_json(out / "samples.json", metadata)
        if not args.skip_full and seed == args.seeds[0]:
            full_dir = sample_dir / "full_channel"
            full_dir.mkdir(exist_ok=True)
            for method, (qc, lam, build_time) in full_channel_circuits(kraus, cached).items():
                print(f"full channel {method}: compile", flush=True)
                context = {"seed": seed, "p": p, "system_qubits": n, "d": d,
                           "kraus_index": -1, "scope": "full_channel", "method": method}
                for basis in args.bases:
                    for opt in args.opt_levels:
                        if (seed, -1, method, basis, opt) in completed:
                            continue
                        rows.append(compile_row(qc, context, lam, build_time, kraus, full_dir, basis, opt))
                        write_json(out / "rows.json", rows)
                        print(f"full channel {method} {basis} opt{opt}: {rows[-1]['total_gates']} gates", flush=True)
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "manifest.json", {
        "paper": "https://doi.org/10.1103/PhysRevB.102.134310",
        "construction": "Sec II Eqs 2-3; Sec IV KC2; Fig 7 includes p=0.3, q=2, L=6, d=15; actual sample parameters are in args",
        "provenance": "New seeded samples from the specified ensemble; not authors' original matrices",
        "args": {**vars(args), "output": str(out)},
        "python": platform.python_version(), "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sampler": "scipy.stats.unitary_group.rvs, shared numpy.default_rng(seed), U then V",
        "basis": BASIS, "seed_transpiler": 0, "approximation_degree": 1.0,
        "topology": "all-to-all; no hardware routing",
        "metric_note": "Continuous-rotation gate counts, not fault-tolerant T counts; measurement/reset not counted",
        "timing_note": "Build and transpile timing excludes sampling and SVD. Resumed early rows use the original timing convention: U+/- build time is half their combined time. Timing is diagnostic, not used for route comparison.",
        "semantic_tolerance": TOL, "block_error": "Frobenius norm up to one overall phase",
        "channel_error": "relative Frobenius Choi error, successful branch rescaled by Lambda",
        "serialization": "QPY preserves global phase; QASM2 top-level phase is in each result row",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "block_backend_sha256": hashlib.sha256((ROOT / "src/block_encoding_new.py").read_bytes()).hexdigest(),
        "row_count": len(rows)})
    print(f"Finished {len(rows)} rows; output={out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Baseline_results/prb2020_kc2_synthesis_20260909")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--system-qubits", type=int, default=2)
    parser.add_argument("--dissipative-count", type=int, default=15)
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--bases", nargs="+", choices=BASIS, default=["ibm", "nam"])
    parser.add_argument("--opt-levels", type=int, nargs="+", choices=[0, 3], default=[0, 3])
    parser.add_argument("--micro-indices", type=int, nargs="+")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume identical sample/basis settings from rows.json")
    args = parser.parse_args()
    if not 0 < args.p < 1:
        parser.error("This experiment requires 0 < p < 1 (retains all Kraus operators).")
    run(args)


if __name__ == "__main__":
    main()
