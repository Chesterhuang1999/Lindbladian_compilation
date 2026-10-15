#!/usr/bin/env python3
"""Probe PennyLane's native Kraus simulation and circuit-compilation boundary."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pennylane-research-mpl")

import numpy as np
import pennylane as qml
from pennylane.devices.qubit_mixed.apply_operation import (
    apply_operation_einsum,
    apply_operation_tensordot,
)

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-10


def max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def encode_complex(value):
    array = np.asarray(value)
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def attempt(function):
    try:
        value = function()
        return {"status": "accepted", "result": value}
    except Exception as exc:  # Probe the public API's validation boundary.
        return {"status": "rejected", "exception": type(exc).__name__, "message": str(exc)}


def measured_channel(kraus, rho, *, builtin=False, device_name="default.mixed"):
    device = qml.device(device_name, wires=[0, 1], shots=None)

    @qml.qnode(device, interface=None)
    def circuit():
        qml.QubitDensityMatrix(rho, wires=[0, 1])
        if builtin:
            qml.AmplitudeDamping(0.3, wires=1)
        else:
            qml.QubitChannel(kraus, wires=1)
        return (
            qml.density_matrix(wires=[0, 1]),
            qml.density_matrix(wires=[1]),
            qml.probs(wires=[0, 1]),
            qml.expval(qml.Z(1)),
            qml.expval(qml.X(0) @ qml.X(1)),
        )

    return circuit()


def simulate_one_wire(kraus):
    device = qml.device("default.mixed", wires=1)

    @qml.qnode(device, interface=None)
    def circuit():
        qml.QubitChannel(kraus, wires=0)
        return qml.state()

    return encode_complex(circuit())


def pure_state_channel(kraus):
    device = qml.device("default.qubit", wires=1)

    @qml.qnode(device, interface=None)
    def circuit():
        qml.X(0)
        qml.QubitChannel(kraus, wires=0)
        return qml.probs(wires=0)

    return np.asarray(circuit()).tolist()


def compile_probe(operation):
    tape = qml.tape.QuantumScript([operation])
    basis = {qml.RZ, qml.RY, qml.CNOT, qml.GlobalPhase}

    def lower():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            [lowered], _ = qml.transforms.decompose(tape, gate_set=basis)
        return {
            "operation_names": [op.name for op in lowered.operations],
            "all_operations_in_requested_basis": all(type(op) in basis for op in lowered.operations),
            "warnings": [str(item.message) for item in caught],
        }

    def compile_tape():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            [compiled], _ = qml.compile(tape, basis_set=["RZ", "RY", "CNOT", "GlobalPhase"])
        return {
            "operation_names": [op.name for op in compiled.operations],
            "all_operations_in_requested_basis": all(type(op) in basis for op in compiled.operations),
            "warnings": [str(item.message) for item in caught],
        }

    return {
        "name": operation.name,
        "has_matrix": operation.has_matrix,
        "has_decomposition": operation.has_decomposition,
        "operation_decomposition": attempt(lambda: [op.name for op in operation.decomposition()]),
        "matrix": attempt(lambda: encode_complex(qml.matrix(operation))),
        "transforms_decompose": attempt(lower),
        "compile": attempt(compile_tape),
        "openqasm2": attempt(lambda: qml.to_openqasm(tape, measure_all=False)),
    }


def source_record(obj):
    filename = Path(inspect.getfile(obj))
    return {
        "path": str(filename),
        "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
        "first_line": inspect.getsourcelines(obj)[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Baseline_results/pennylane_channel_probe")
    args = parser.parse_args()

    gamma = 0.3
    kraus = [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex),
    ]
    # An entangled mixed input exercises coherence and the untouched subsystem.
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = 0.8 * np.outer(bell, bell.conj()) + 0.2 * np.eye(4) / 4
    extended_kraus = [np.kron(np.eye(2), item) for item in kraus]
    expected = sum(item @ rho @ item.conj().T for item in extended_kraus)
    reduced = np.trace(expected.reshape(2, 2, 2, 2), axis1=0, axis2=2)
    expected_probs = expected.diagonal().real
    z = np.diag([1, -1])
    x = np.array([[0, 1], [1, 0]])
    expected_z = np.trace(expected @ np.kron(np.eye(2), z)).real
    expected_xx = np.trace(expected @ np.kron(x, x)).real
    generic = measured_channel(kraus, rho)
    builtin = measured_channel(kraus, rho, builtin=True)
    channel = qml.QubitChannel(kraus, wires=1)
    tensor = rho.reshape(2, 2, 2, 2)
    errors = {
        "generic_density_vs_numpy": max_error(generic[0], expected),
        "builtin_density_vs_numpy": max_error(builtin[0], expected),
        "builtin_vs_generic_density": max_error(builtin[0], generic[0]),
        "reduced_density_vs_numpy": max_error(generic[1], reduced),
        "probabilities_vs_numpy": max_error(generic[2], expected_probs),
        "z_expectation_vs_numpy": max_error(generic[3], expected_z),
        "xx_expectation_vs_numpy": max_error(generic[4], expected_xx),
        "einsum_kernel_vs_numpy": max_error(apply_operation_einsum(channel, tensor).reshape(4, 4), expected),
        "tensordot_kernel_vs_numpy": max_error(apply_operation_tensordot(channel, tensor).reshape(4, 4), expected),
        "kraus_tp_identity": max_error(sum(item.conj().T @ item for item in kraus), np.eye(2)),
        "output_trace": float(abs(np.trace(generic[0]) - 1)),
    }

    invalid_inputs = {}
    for name, matrices in {
        "trace_decreasing_single_kraus": [0.5 * np.eye(2)],
        "trace_increasing_single_kraus": [2 * np.eye(2)],
        "rectangular_matrix": [np.zeros((2, 3))],
        "inconsistent_shapes": [np.eye(2), np.eye(4)],
        "rank_three_matrices": [np.ones((2, 2, 2))],
        "empty_list": [],
        "four_by_four_on_one_wire": [np.eye(4)],
        "three_by_three_on_one_wire": [np.eye(3)],
    }.items():
        construction = attempt(lambda matrices=matrices: qml.QubitChannel(matrices, wires=0).name)
        invalid_inputs[name] = {"construction": construction}
        if construction["status"] == "accepted":
            invalid_inputs[name]["default_mixed_execution"] = attempt(lambda matrices=matrices: simulate_one_wire(matrices))

    compilation = [compile_probe(channel), compile_probe(qml.AmplitudeDamping(gamma, wires=1))]
    default_qubit = attempt(lambda: pure_state_channel(kraus))
    mixed_state_result = simulate_one_wire([np.eye(2)])

    checks = {
        "numeric_channel_semantics": max(errors.values()) < TOLERANCE,
        "non_tp_inputs_rejected": all(
            invalid_inputs[name]["construction"]["status"] == "rejected"
            for name in ("trace_decreasing_single_kraus", "trace_increasing_single_kraus")
        ),
        "native_unitary_representation_unavailable": all(
            not row["has_matrix"] and not row["has_decomposition"]
            and row["openqasm2"]["status"] == "rejected"
            for row in compilation
        ),
        "default_qubit_rejects_generic_channel": default_qubit["status"] == "rejected",
    }
    report = {
        "pennylane_version": qml.__version__,
        "python_version": platform.python_version(),
        "scope": "Native Kraus density-matrix simulation and channel compilation boundary; not a gate-count benchmark.",
        "device": "default.mixed",
        "shots": None,
        "wire_order_msb_first": [0, 1],
        "channel_target_wires": [1],
        "gamma": gamma,
        "input_state": "0.8 |Phi+><Phi+| + 0.2 I_4/4",
        "input_density": encode_complex(rho),
        "kraus_matrices": [encode_complex(item) for item in kraus],
        "output_density": encode_complex(generic[0]),
        "output_reduced_density_wire1": encode_complex(generic[1]),
        "output_probabilities": np.asarray(generic[2]).tolist(),
        "output_expval_z1": float(generic[3]),
        "output_expval_x0x1": float(generic[4]),
        "output_shapes": [list(np.shape(value)) for value in generic],
        "state_measurement_on_default_mixed": mixed_state_result,
        "errors": errors,
        "invalid_inputs": invalid_inputs,
        "compilation": compilation,
        "registered_graph_decomposition_counts": {
            name: len(qml.list_decomps(getattr(qml, name)))
            for name in sorted(qml.ops.channel.__qubit_channels__)
        },
        "default_qubit_generic_channel": default_qubit,
        "sources": {
            "QubitChannel": source_record(qml.QubitChannel),
            "Channel": source_record(qml.operation.Channel),
            "DefaultMixed": source_record(qml.devices.DefaultMixed),
            "apply_operation_einsum": source_record(apply_operation_einsum),
            "apply_operation_tensordot": source_record(apply_operation_tensordot),
        },
        "tolerance": TOLERANCE,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": str(output), "passed": report["passed"], "max_error": max(errors.values()), "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
