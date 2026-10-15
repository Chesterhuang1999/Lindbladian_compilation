#!/usr/bin/env python3
"""Check HRW channel-LCU with a reference entangled before Qrisp compilation.

Example:
    python Baseline_scripts/probe_qrisp_hrw_channel_lcu_correctness.py \
        --reproduce-output-injection

The default check constructs the entire Choi matrix from a Bell input. Injecting
arbitrary inputs into final output wires after compilation is not a valid test:
the allocator may have used those wires as clean workspace before system allocation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--raw", action="store_true", help="Validate before allocation/compilation.")
    parser.add_argument("--qrisp-source", type=Path, help="Optional directory containing a separate qrisp package.")
    parser.add_argument("--reproduce-output-injection", action="store_true", help="Also show why the former output-wire injection check is invalid.")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if args.raw and args.reproduce_output_injection:
        parser.error("--reproduce-output-injection demonstrates compiled allocation; use it without --raw")

    cache_root = Path(tempfile.gettempdir()) / "qrisp-hrw-correctness-probe"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    if args.qrisp_source:
        sys.path.insert(0, str(args.qrisp_source.resolve()))
    sys.path.insert(0, str(ROOT / "src"))

    import numpy as np
    from qiskit.quantum_info import Statevector
    from qrisp import QuantumFloat, QuantumVariable, cx, gphase, h, prepare, x, y, z
    from qrisp.block_encodings import BlockEncoding
    from qrisp.jasp import q_switch
    from hypercube_channel_pauli import build_hypercube_section52_channel

    channel = build_hypercube_section52_channel(args.n)
    terms = [[(p.expr, complex(c * p.phase)) for p, c in ms.instances]
             for ms in channel.channels[0][1]]
    target_kraus = [ms.eff_op().to_matrix() for ms in channel.channels[0][1]]
    dimension = 1 << args.n
    canonical = []
    for site in range(args.n):
        for bit in (0, 1):
            matrix = np.zeros((dimension, dimension), dtype=complex)
            for column in range(dimension):
                if ((column >> site) & 1) == bit:
                    matrix[column & ~(1 << site), column] = 1 / np.sqrt(2 * args.n)
                    matrix[column | (1 << site), column] = 1 / np.sqrt(2 * args.n)
            canonical.append(matrix)

    def choi(kraus):
        vectors = [k.reshape(-1, order="F") for k in kraus]
        return sum(np.outer(v, v.conj()) for v in vectors)

    construction_error = float(np.max(np.abs(choi(target_kraus) - choi(canonical))))
    tp_error = float(np.max(np.abs(sum(k.conj().T @ k for k in target_kraus) - np.eye(dimension))))

    def phased_pauli(label, coefficient):
        def unitary(system):
            angle = float(np.angle(coefficient))
            if abs(angle) > 1e-14:
                gphase(angle, system[0])
            for bit, pauli in enumerate(reversed(label)):
                if pauli != "I":
                    {"X": x, "Y": y, "Z": z}[pauli](system[bit])
        return unitary

    def branch(block):
        def apply(operand):
            ancilla, system = operand
            block.unitary(ancilla, system)
        return apply

    def identity(operand):
        pass

    def build(with_reference):
        system = QuantumVariable(args.n, name="sys")
        reference = QuantumVariable(args.n, name="ref") if with_reference else None
        if reference is not None:
            for bit in range(args.n):
                h(reference[bit])
                cx(reference[bit], system[bit])
        index_size = (len(terms) - 1).bit_length()
        term_size = (max(map(len, terms)) - 1).bit_length()
        index = QuantumFloat(index_size, name="kraus")
        ancilla = QuantumFloat(term_size, name="lcu")
        blocks = [BlockEncoding.from_lcu(np.array([abs(c) for _, c in row]),
                                        [phased_pauli(p, c) for p, c in row])
                  for row in terms]
        alpha = np.array([float(block.alpha) for block in blocks])
        scale = float(alpha @ alpha)
        prepare(index, np.pad(alpha / np.sqrt(scale), (0, (1 << index_size) - len(blocks))))
        branches = [branch(block) for block in blocks]
        branches += [identity] * ((1 << index_size) - len(branches))
        q_switch(index, branches, (ancilla, system), method="tree")
        circuit = system.qs.copy() if args.raw else system.qs.compile(workspace=0)
        return system, reference, index, ancilla, circuit, scale

    def positions(circuit, register):
        return [circuit.qubits.index(q) for q in register.reg]

    def encode(value, bits):
        return sum(((value >> bit) & 1) << position for bit, position in enumerate(bits))

    system, reference, index, ancilla, circuit, scale = build(True)
    qiskit = circuit.to_qiskit()
    state = Statevector.from_int(0, 1 << qiskit.num_qubits).evolve(qiskit).data
    sb, rb, ib, ab = [positions(circuit, qv) for qv in (system, reference, index, ancilla)]
    recovered = [np.array([[state[encode(j, ib) | encode(s, sb) | encode(r, rb)] * np.sqrt(dimension)
                           for r in range(dimension)] for s in range(dimension)])
                 for j in range(1 << len(ib))]
    choi_error = float(np.max(np.abs(scale * choi(recovered) - choi(target_kraus))))
    success = float(sum(np.linalg.norm(k) ** 2 for k in recovered) / dimension)
    live = set(sb + rb + ib + ab)
    workspace_mask = sum(1 << bit for bit in range(qiskit.num_qubits) if bit not in live)
    leakage = float(sum(abs(amplitude) ** 2 for row, amplitude in enumerate(state) if row & workspace_mask))
    print(json.dumps({"check": "bell_input_choi", "qrisp_version": importlib.metadata.version("qrisp"),
                      "n": args.n, "mode": "raw" if args.raw else "compiled", "width": qiskit.num_qubits,
                      "canonical_channel_error": construction_error, "trace_preservation_error": tp_error,
                      "choi_max_abs_error": choi_error, "success_probability": success,
                      "expected_success_probability": 1 / scale, "workspace_leakage": leakage}), flush=True)
    if max(construction_error, tp_error, choi_error, abs(success - 1 / scale), leakage) > 1e-9:
        raise SystemExit("Correctly prepared channel validation failed")

    if args.reproduce_output_injection:
        system, _, index, _, circuit, scale = build(False)
        qiskit = circuit.to_qiskit()
        sb, ib = positions(circuit, system), positions(circuit, index)
        columns = [encode(s, sb) for s in range(dimension)]
        images = np.column_stack([Statevector.from_int(column, 1 << qiskit.num_qubits).evolve(qiskit).data
                                  for column in columns])
        apparent_kraus = [images[[encode(j, ib) | column for column in columns], :]
                         for j in range(1 << len(ib))]
        apparent_error = float(np.max(np.abs(scale * choi(apparent_kraus) - choi(target_kraus))))
        print(json.dumps({"check": "invalid_output_wire_injection", "width": qiskit.num_qubits,
                          "apparent_choi_error": apparent_error,
                          "interpretation": "Final system output wires are not arbitrary-input ports; earlier clean-workspace use invalidates this check."}), flush=True)


if __name__ == "__main__":
    main()
