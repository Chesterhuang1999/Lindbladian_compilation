#!/usr/bin/env python3
"""Prepare native Kraus inputs for UniversalQCompiler's small-case probe.

HRW is passed through unchanged. Non-TP inputs are rejected unless an explicit
LCU-normalized, success-flag completion is requested. No matrix whitening is
performed: projecting the flag to zero recovers the original CP map / Lambda.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from qiskit.quantum_info import Pauli

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("src", "scripts", "scripts/eval_redesign")]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pack(matrices):
    data = np.asarray(matrices, dtype=complex)
    return np.stack([data.real, data.imag], axis=-1).tolist()


def choi(kraus):
    vectors = np.stack([k.reshape(-1, order="F") for k in kraus], axis=1)
    return vectors @ vectors.conj().T


def prepare_case(name, heralded=False):
    from paper_rq1_inputs import build_case
    from rq1_pipeline import input_identity, serialize_kraus

    raw, model = build_case(name)
    rows = serialize_kraus(raw)
    d = 1 << model["system_qubits"]
    matrices = [sum((c * complex(re, im) * Pauli(label).to_matrix()
                     for label, c, re, im in row), np.zeros((d, d), complex))
                for row in rows]
    direct_matrix_error = max(np.linalg.norm(a - b.eff_op().to_matrix())
                              for a, b in zip(matrices, raw))
    assert direct_matrix_error < 1e-7
    completeness = sum(k.conj().T @ k for k in matrices)
    eigenvalues = np.linalg.eigvalsh(completeness)
    tp_error = float(np.linalg.norm(completeness - np.eye(d)))
    lcu_lambda = float(sum(sum(abs(c * complex(re, im)) for _, c, re, im in row)**2
                           for row in rows))
    native_eligible = tp_error < 1e-10
    result = {
        "case": name, "model": model, "input_identity": input_identity(raw),
        "input_pauli_kraus": rows, "original_kraus": pack(matrices),
        "system_qubits": model["system_qubits"], "input_dimension": d,
        "original_kraus_count": len(matrices), "raw_lcu_Lambda": lcu_lambda,
        "native_cptp_eligible": native_eligible,
        "original_completeness_eigenvalues": eigenvalues.tolist(),
        "original_tp_frobenius_error": tp_error,
        "matrix_materialization_error": float(direct_matrix_error),
    }
    if heralded:
        remainder = np.eye(d) - completeness / lcu_lambda
        values, vectors = np.linalg.eigh(remainder)
        if values.min() < -1e-10:
            raise ValueError("LCU normalization did not give a contraction")
        failure = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T
        zeros = np.zeros((d, d), complex)
        compiled = [np.vstack([k / np.sqrt(lcu_lambda), zeros]) for k in matrices]
        compiled.append(np.vstack([zeros, failure]))
        success = [k[:d] for k in compiled]
        good = choi(matrices) / lcu_lambda
        success_error = float(np.linalg.norm(choi(success) - good) / np.linalg.norm(good))
        assert success_error < 1e-12
        result.update(
            mode="heralded_lcu_completion", output_dimension=2*d,
            success_flag="most-significant output qubit; success = 0",
            success_contract="E_original(rho) / raw_lcu_Lambda",
            success_choi_relative_error=success_error,
            success_probability_eigenvalue_bounds=(eigenvalues / lcu_lambda).tolist(),
            failure_completeness_min_eigenvalue=float(values.min()),
        )
    else:
        result.update(mode="native_channel", output_dimension=d,
                      success_contract="E_original(rho); no postselection")
        if not native_eligible:
            result.update(status="rejected_non_tp", compiler_kraus=None)
            return result
        compiled = matrices

    residual = float(np.linalg.norm(sum(k.conj().T @ k for k in compiled) - np.eye(d)))
    assert residual < 1e-10
    # Padding the environment label to a power of two gives a true isometry.
    env_dim = 1 << (len(compiled)-1).bit_length()
    isometry = np.vstack(compiled + [np.zeros_like(compiled[0])] * (env_dim-len(compiled)))
    assert np.linalg.norm(isometry.conj().T @ isometry - np.eye(d)) < 1e-10
    result.update(
        status="prepared", compiler_kraus=pack(compiled),
        compiler_kraus_count=len(compiled), compiler_tp_frobenius_error=residual,
        uncompressed_isometry_shape=list(isometry.shape),
        uncompressed_circuit_qubits=(isometry.shape[0]-1).bit_length(),
        validation_scope="input and completion only; no synthesized circuit yet",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--uqc-source-dir", type=Path, required=True,
                        help="Directory containing upstream UniversalQCompiler.m and QI.m")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    if output.exists():
        raise ValueError("Use a new output directory to preserve prior runs")
    required = ("UniversalQCompiler.m", "QI.m")
    for name in required:
        if not (args.uqc_source_dir / name).is_file():
            raise FileNotFoundError(name)
    output.mkdir(parents=True)
    vendor = output / "vendor"
    vendor.mkdir()
    sources = {}
    for name in required:
        source = args.uqc_source_dir / name
        shutil.copy2(source, vendor / name)
        sources[name] = sha256(source.read_bytes()).hexdigest()
    for name in ("LICENSE", "QI.LICENSE"):
        source = args.uqc_source_dir / name
        if source.exists():
            shutil.copy2(source, vendor / name)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("run_universalqcompiler.wls"),
                 output / "run_universalqcompiler.wls")
    results = []
    for case, heralded in (("HRW-4", False), ("TFIM-3", False), ("TFIM-3", True)):
        payload = prepare_case(case, heralded)
        path = output / (case + ("_heralded" if heralded else "_native") + ".json")
        dump(path, payload)
        summary = {key: payload.get(key) for key in (
            "case", "mode", "status", "original_kraus_count", "compiler_kraus_count",
            "original_tp_frobenius_error", "compiler_tp_frobenius_error", "raw_lcu_Lambda",
            "uncompressed_isometry_shape", "uncompressed_circuit_qubits",
            "success_choi_relative_error")}
        if payload["status"] == "prepared":
            summary["command_arguments"] = ["wolframscript", "-file",
                str(output / "run_universalqcompiler.wls"), str(vendor),
                str(path), str(path.with_suffix(".result.json"))]
        results.append(summary)
    dump(output / "input_checks.json", results)
    dump(output / "provenance.json", {
        "upstream": {
            "UniversalQCompiler": "https://github.com/Q-Compiler/UniversalQCompiler/tree/c72cd84a537786756ae8490c07f3f599aa411323",
            "QI": "https://github.com/rogercolbeck/QI/tree/30103eb1ae501eeda6e5ed3c6ffa9fbec164c1f0"},
        "source_sha256": sources,
        "driver_status": "prepared; requires execution with an activated Wolfram kernel",
        "policy": "No benchmark or original Kraus data are overwritten. TFIM native rejection and heralded completion are separate cases.",
    })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
