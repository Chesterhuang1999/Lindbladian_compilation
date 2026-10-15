#!/usr/bin/env python3
"""Supplement the PRB KC2 experiment with joint QSD and a classical mixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import qs_decomposition

from benchmark_prb2020_kraus_synthesis import (
    BASIS,
    ROOT,
    TOL,
    choi,
    compile_row,
    count_qasm_text,
    decomposition,
    phase_error,
    unitary_circuit,
    write_json,
)


def context(method: str, scope: str, p: float, n: int, d: int, index: int) -> dict:
    return {"seed": 0, "p": p, "system_qubits": n, "d": d,
            "kraus_index": index, "scope": scope, "method": method}


def source_data(path: Path) -> tuple[list[np.ndarray], np.ndarray, float]:
    with np.load(path, allow_pickle=False) as source:
        kraus = list(source["kraus"])
        unitary = source["U"]
    dim = len(unitary)
    p = 1.0 - float(np.trace(kraus[0].conj().T @ kraus[0]).real / dim)
    assert 0 < p < 1
    assert np.linalg.norm(kraus[0] - math.sqrt(1 - p) * unitary) < TOL
    assert np.linalg.norm(sum(a.conj().T @ a for a in kraus) - np.eye(dim)) < TOL
    return kraus, unitary, p


def joint_qsd(kraus: list[np.ndarray], p: float, out: Path) -> tuple[list[dict], dict]:
    a = kraus[1]
    n = round(math.log2(len(a)))
    start = time.perf_counter()
    dec = decomposition(a)
    plus, minus = dec["plus"], dec["minus"]
    average, difference = (plus + minus) / 2, (plus - minus) / 2
    # The ancilla is highest: H diag(U+, U-) H has these four blocks.
    matrix = np.block([[average, difference], [difference, average]])
    assert np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(matrix))) < TOL
    assert np.linalg.norm(matrix[:len(a), :len(a)] - a / dec["alpha"]) < TOL
    qc = qs_decomposition(matrix)
    build_seconds = time.perf_counter() - start
    exact_error = float(np.linalg.norm(Operator(qc).data - matrix))
    assert exact_error < TOL
    rows = []
    for basis in BASIS:
        for opt in (0, 3):
            rows.append(compile_row(
                qc, context("two_unitary_joint_qsd", "kraus_block", p, n, len(kraus) - 1, 1),
                dec["alpha"], build_seconds, a, out, basis, opt,
            ))
    return rows, {"matrix_shape": list(matrix.shape), "qsd_exact_unitary_error": exact_error,
                  "alpha": dec["alpha"], "synthesis": "qiskit.synthesis.qs_decomposition"}


def native_unitary_metrics(unitary: np.ndarray, basis: str, opt: int) -> dict:
    start = time.perf_counter()
    qc = transpile(unitary_circuit(unitary), basis_gates=BASIS[basis],
                   optimization_level=opt, seed_transpiler=0, approximation_degree=1.0)
    elapsed = time.perf_counter() - start
    assert phase_error(Operator(qc).data, unitary) < TOL
    stats = count_qasm_text(qasm2.dumps(qc), gate_set=basis)
    return {"total_gates": stats["metric_total"], "cx": stats["ops"].get("cx", 0),
            "non_clifford": stats["clifford"]["non_clifford"], "width": qc.num_qubits,
            "depth": qc.depth(), "compile_seconds": elapsed}


def dissipative_isometry(kraus: list[np.ndarray], unitary: np.ndarray, p: float,
                        out: Path) -> tuple[list[dict], dict]:
    n = round(math.log2(len(unitary)))
    dim = len(unitary)
    start = time.perf_counter()
    dissipative = [a / math.sqrt(p) for a in kraus[1:]]
    tp_error = float(np.linalg.norm(sum(a.conj().T @ a for a in dissipative) - np.eye(dim)))
    assert tp_error < TOL
    env_qubits = math.ceil(math.log2(len(dissipative)))
    matrix = np.zeros((dim * 2 ** env_qubits, dim), dtype=complex)
    matrix[:dim * len(dissipative)] = np.vstack(dissipative)
    qc = QuantumCircuit(n + env_qubits)
    qc.append(Isometry(matrix, 0, 0), range(qc.num_qubits))
    build_seconds = time.perf_counter() - start
    rows, averages = [], []
    for basis in BASIS:
        for opt in (0, 3):
            # Reuse the direct-isometry extraction path, then rename its artifacts.
            row = compile_row(qc, context("direct_channel_isometry", "full_channel", p, n,
                                         len(dissipative), -1),
                              1.0, build_seconds, dissipative, out, basis, opt)
            old_qasm = Path(row["qasm"])
            method = "direct_dissipative_isometry"
            new_stem = f"{method}_{basis}_opt{opt}"
            for extension in (".qasm", ".qpy"):
                old_qasm.with_suffix(extension).rename(out / (new_stem + extension))
            row["method"] = method
            row["qasm"] = str(out / (new_stem + ".qasm"))
            rows.append(row)
            unitary_metrics = native_unitary_metrics(unitary, basis, opt)
            averages.append({"basis": basis, "opt_level": opt, "p": p,
                             "native_unitary": unitary_metrics,
                             "dissipative_isometry": {key: row[key] for key in
                                                       ("total_gates", "cx", "non_clifford", "width", "depth")},
                             "mixture_expected": {key: (1 - p) * unitary_metrics[key] + p * row[key]
                                                  for key in ("total_gates", "cx", "non_clifford")},
                             "mixture_max_width": max(row["width"], unitary_metrics["width"]),
                             "channel_success": 1.0})
    return rows, {"matrix_shape": list(matrix.shape), "tp_error": tp_error,
                  "dissipative_kraus_count": len(dissipative), "classical_mixture_costs": averages}


def classical_kraus_sampling(kraus: list[np.ndarray], p: float, source: Path,
                             main_rows_path: Path, out: Path) -> None:
    main_rows_bytes = main_rows_path.read_bytes()
    main_rows = json.loads(main_rows_bytes)
    assert len(kraus) == 16, "This derived experiment expects the seed0 KC2 rank-16 representation."
    reference = choi(kraus)
    spectral_norms = np.array([np.linalg.svd(a, compute_uv=False)[0] for a in kraus])
    hybrid_pauli_norms = np.array([
        spectral_norms[0],
        *[float(sum(abs(SparsePauliOp.from_operator(Operator(a), atol=1e-12, rtol=0).coeffs)))
          for a in kraus[1:]],
    ])
    methods = ("two_unitary_independent_lcu", "two_unitary_shared_svd", "svd_ucry_dilation",
               "direct_block_isometry", "pauli_current_matrix_order")
    results = []
    for method in methods:
        expected_norms = hybrid_pauli_norms if method == "pauli_current_matrix_order" else spectral_norms
        for basis in BASIS:
            selected = []
            for index in range(len(kraus)):
                actual_method = "native_scaled_unitary" if index == 0 else method
                matches = [row for row in main_rows
                           if row["seed"] == 0 and row["basis"] == basis and row["opt_level"] == 3
                           and row["scope"] == "kraus_block" and row["kraus_index"] == index
                           and row["method"] == actual_method]
                assert len(matches) == 1, (method, basis, index, len(matches))
                assert matches[0]["status"] == "ok"
                assert matches[0]["semantic_error"] < TOL
                selected.append(matches[0])
            assert {row["kraus_index"] for row in selected} == set(range(16))
            norms = np.array([row["alpha_or_lambda"] for row in selected])
            assert np.all(norms > 0)
            norm_error = float(np.max(abs(norms - expected_norms)))
            assert norm_error < TOL
            lam = float(np.sum(norms ** 2))
            expected_lam = float(np.sum(expected_norms ** 2))
            assert abs(lam - expected_lam) < TOL
            probabilities = norms ** 2 / lam
            assert abs(float(np.sum(probabilities)) - 1.0) < TOL
            realized = [math.sqrt(probability) * a / alpha
                        for probability, a, alpha in zip(probabilities, kraus, norms)]
            choi_error = float(np.linalg.norm(choi(realized) - reference / lam)
                               / np.linalg.norm(reference / lam))
            assert choi_error < TOL
            success_operator = sum(a.conj().T @ a for a in realized)
            success_error = float(np.linalg.norm(success_operator - np.eye(len(kraus[0])) / lam))
            assert success_error < TOL
            results.append({
                "method": method, "basis": basis, "opt_level": 3, "seed": 0,
                "kraus0_method": "native_scaled_unitary", "kraus_count": len(kraus),
                "lambda": lam, "channel_success": 1.0 / lam,
                "six_consecutive_success": (1.0 / lam) ** 6,
                "max_width": max(row["width"] for row in selected),
                "weighted_cost_per_attempt": {
                    key: float(sum(probability * row[key] for probability, row in zip(probabilities, selected)))
                    for key in ("total_gates", "cx", "non_clifford")},
                "norms": norms.tolist(), "sampling_probabilities": probabilities.tolist(),
                "norm_validation_error": norm_error, "choi_relative_error": choi_error,
                "state_independent_success_operator_error": success_error,
                "selected_rows": [{key: row[key] for key in
                                   ("kraus_index", "method", "alpha_or_lambda", "total_gates", "cx",
                                    "non_clifford", "width", "semantic_error", "qasm")}
                                  for row in selected],
            })
    destination = out / "classical_kraus_sampling.json"
    write_json(destination, {
        "experiment": "classical_kraus_sampling", "seed": 0, "p": p,
        "source_matrices": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "main_rows": str(main_rows_path), "main_rows_sha256": hashlib.sha256(main_rows_bytes).hexdigest(),
        "main_row_count": len(main_rows), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "derivation": "Select j classically with r_j=alpha_j^2/Lambda, Lambda=sum_j alpha_j^2; "
                      "postselect the block implementing K_j/alpha_j. The successful unnormalized map is Phi/Lambda.",
        "rows": results,
        "limitations": ["Data derivation from completed seed0 opt3 circuits; no additional synthesis.",
                        "Every route uses the native scaled-unitary K0, including the hybrid Pauli route.",
                        "Weighted gate counts are per attempt, not a claimed cost per successful arbitrary-state output.",
                        "Six-consecutive-success reports only Lambda^-6; no restart or amplitude-amplification cost is inferred.",
                        "Sampling, measurement, reset, routing, and fault-tolerant rotation-synthesis costs are excluded."]})
    for row in results:
        cost = row["weighted_cost_per_attempt"]
        print(f"{row['method']} {row['basis']}: Lambda={row['lambda']:.12g} "
              f"success={row['channel_success']:.8g} six={row['six_consecutive_success']:.8g} "
              f"G={cost['total_gates']:.8g} CX={cost['cx']:.8g} NC={cost['non_clifford']:.8g} "
              f"width={row['max_width']} error={row['choi_relative_error']:.3g}", flush=True)
    print(f"Output: {destination}", flush=True)


def run(args) -> None:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    kraus, unitary, p = source_data(source)
    if args.experiment == "kraus-sampling":
        classical_kraus_sampling(kraus, p, source, args.main_rows.resolve(), out)
        return
    rows, details = [], {}
    if args.experiment in ("all", "joint-qsd"):
        result, detail = joint_qsd(kraus, p, out)
        rows.extend(result)
        details["joint_qsd"] = detail
    if args.experiment in ("all", "dissipative"):
        result, detail = dissipative_isometry(kraus, unitary, p, out)
        rows.extend(result)
        details["dissipative_isometry"] = detail
    stem = "extensions" if args.experiment == "all" else args.experiment.replace("-", "_")
    write_json(out / f"{stem}_rows.json", rows)
    with (out / f"{stem}_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / f"{stem}_summary.json", {
        "source_matrices": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiment": args.experiment, "seed": 0, "p": p,
        "qiskit": qiskit.__version__, "details": details, "rows": rows,
        "limitations": ["One seeded local KC2 channel, not a six-site circuit layer.",
                        "Exact continuous-rotation synthesis with all-to-all connectivity; no fault-tolerant T count.",
                        "The classical-mixture mean weights mutually exclusive U and D branches, not coherent SELECT.",
                        "Classical random sampling, measurement, reset, and hardware routing costs are excluded.",
                        "Joint QSD synthesizes a full three-qubit unitary, not U+ and U- independently."]})
    for row in rows:
        print(f"{row['method']} {row['basis']} opt{row['opt_level']}: "
              f"G={row['total_gates']} CX={row['cx']} NC={row['non_clifford']} "
              f"width={row['width']} error={row['semantic_error']:.3g}", flush=True)
    print(f"Output: {out / (stem + '_summary.json')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "Baseline_results/prb2020_kc2_synthesis_20260909/seed0/matrices.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Baseline_results/prb2020_kc2_extensions_20260909")
    parser.add_argument("--main-rows", type=Path,
                        default=ROOT / "Baseline_results/prb2020_kc2_synthesis_20260909/rows.json")
    parser.add_argument("--experiment", choices=("all", "joint-qsd", "dissipative", "kraus-sampling"),
                        default="all")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
