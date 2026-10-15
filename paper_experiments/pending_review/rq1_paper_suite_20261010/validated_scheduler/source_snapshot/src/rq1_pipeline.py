"""Shared inputs, assembly, and semantic verification for the RQ1 pipelines."""

from copy import deepcopy
from hashlib import sha256
import json
import time

import numpy as np
from block_encoding import BlockEncoding
from channel_IR import Matrixsum, PauliAtom, channel as ChannelIR, channel_ensemble
from channel_LCU import channel_to_LCU
from experiment_protocol import RQ1_PIPELINES, REWRITE_DEFAULTS, run_rewrite_search


def serialize_kraus(kraus):
    return [[[atom.expr, float(c), float(atom.phase.real), float(atom.phase.imag)]
             for atom, c in ms.instances] for ms in kraus]


def deserialize_kraus(rows):
    return [Matrixsum([(PauliAtom(label, phase=complex(re, im)), c)
                       for label, c, re, im in row]) for row in rows]


def input_identity(kraus):
    return sha256(json.dumps(serialize_kraus(kraus), separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def canonicalize(kraus, *, merge_tol=1e-12):
    """Exact per-row term merging/zero elimination, then proportional Kraus merging."""
    if not kraus:
        raise ValueError("A channel needs at least one Kraus operator.")
    for ms in kraus:
        if any(not isinstance(a, PauliAtom) for a, _ in ms.instances):
            raise TypeError("RQ1 Pauli-LCU inputs require PauliAtom operators.")
    started = time.perf_counter()
    channel = ChannelIR([ms.simplify() for ms in deepcopy(kraus)])
    channel.zero_elim().merging(tol=merge_tol)
    if not channel.kraus_ops:
        raise ValueError("The zero channel has no nonzero LCU block.")
    return channel.kraus_ops, {
        "time_s": time.perf_counter() - started,
        "zero_policy": "exact", "proportional_merge_tolerance": merge_tol,
        "raw_kraus_count": len(kraus), "canonical_kraus_count": len(channel.kraus_ops),
        "raw_input_sha256": input_identity(kraus),
        "canonical_input_sha256": input_identity(channel.kraus_ops),
    }


def build_pipeline(canonical_kraus, pipeline, *, rewrite_method=REWRITE_DEFAULTS.method,
                   strategy=REWRITE_DEFAULTS.strategy, beam_width=REWRITE_DEFAULTS.beam_width,
                   max_steps=REWRITE_DEFAULTS.max_steps, tol=REWRITE_DEFAULTS.tol):
    """Consume A_can without mutating it; only Ours runs a unitary Kraus rewrite."""
    if pipeline not in RQ1_PIPELINES:
        raise ValueError(f"Unknown RQ1 pipeline: {pipeline}")
    cfg = RQ1_PIPELINES[pipeline]
    kraus = deepcopy(canonical_kraus)
    rewrite = {"method": "none", "time_s": 0.0}
    if cfg["rewrite"]:
        started = time.perf_counter()
        source = ChannelIR(kraus)
        parameters = dict(strategy=strategy, beam_width=beam_width, max_steps=max_steps, tol=tol)
        if rewrite_method == "rewrite_search":
            result = source.rewrite_search(**parameters)
        elif rewrite_method == "gatecost_rewrite_search":
            result = run_rewrite_search(source, **parameters)
        else:
            raise ValueError(f"Unsupported rewrite method: {rewrite_method}")
        source.apply_rewrite_result(result, tol=tol)
        source.zero_elim()
        kraus = source.kraus_ops
        rewrite = {"method": rewrite_method, **parameters,
                   "initial_support": result["initial_support"],
                   "final_support": result["final_support"], "steps": len(result["steps"]),
                   "cost_model": result.get("cost_model"),
                   "initial_cost": result.get("initial_cost"), "final_cost": result.get("final_cost"),
                   "time_s": time.perf_counter() - started}
    if pipeline == "Qrisp":
        from qrisp_channel_adapter import qrisp_backend_for_channel
        backend = qrisp_backend_for_channel(kraus)
    else:
        def backend(ms):
            return BlockEncoding(ms, placement_options={})
    started = time.perf_counter()
    circuit, registers = channel_to_LCU(
        channel_ensemble([deepcopy(kraus)]), structure=cfg["structure"], opt=cfg["inner"],
        block_encoding_cls=backend, fstar=cfg["fstar"],
    )
    return circuit, registers, {
        "pipeline": pipeline, "configuration": dict(cfg),
        "canonical_input_sha256": input_identity(canonical_kraus),
        "compiled_kraus": serialize_kraus(kraus), "rewrite": rewrite,
        "assembly_time_s": time.perf_counter() - started,
    }


def channel_choi(kraus):
    vectors = [ms.eff_op().to_matrix().reshape(-1, order="F") for ms in kraus]
    return sum(np.outer(v, v.conj()) for v in vectors)


def verify_channel(circuit, registers, target_kraus, *, tolerance=1e-10):
    """Recover all successful Kraus matrices; check the unnormalized channel map.

    The supplied circuit must already be lowered to native simulation gates.
    Basis columns determine the full Choi matrix, including coherent inputs.
    Qubit positions are an explicit input/output interface, never inferred from
    a state-preparation allocator's eventual output wires.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    system = registers[-1]
    if len(system) > 6 or circuit.num_qubits > 22:
        raise ValueError("Full Choi verification is limited to 6 system / 22 total qubits.")
    selector = registers[1] if circuit.metadata["structure"] == "basic" else registers[0]
    dimension = 1 << len(system)
    address = lambda value, bits: sum(((value >> b) & 1) << q for b, q in enumerate(bits))
    system_rows = np.asarray([address(s, system) for s in range(dimension)])
    successful_rows = np.asarray([[address(j, selector) + s for s in system_rows]
                                 for j in range(1 << len(selector))])
    recovered = np.empty((len(successful_rows), dimension, dimension), dtype=complex)
    simulator = AerSimulator(method="statevector", max_parallel_threads=1,
                             max_parallel_experiments=1)
    for column in range(dimension):
        initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
        initial[system_rows[column]] = 1
        probe = QuantumCircuit(circuit.num_qubits)
        probe.set_statevector(initial)
        probe.compose(circuit, inplace=True)
        probe.save_statevector()
        state = np.asarray(simulator.run(probe).result().get_statevector())
        recovered[:, :, column] = state[successful_rows]
    actual = circuit.metadata["subnormalization_factor"] * sum(
        np.outer(k.reshape(-1, order="F"), k.reshape(-1, order="F").conj()) for k in recovered
    )
    expected = channel_choi(target_kraus)
    error = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))
    return {"method": "full unnormalized Choi", "system_basis_inputs": dimension,
            "relative_frobenius_error": error, "tolerance": tolerance, "passed": error < tolerance}
