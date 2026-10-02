#!/usr/bin/env python3
"""Probe MCX/controlled-Pauli lowerings for matrix-order block encodings.

This is an experiment harness only.  The project implementation in
``src/block_encoding_new.py`` expands controlled gates with at least three
controls; this script adds a candidate lowering that also expands two-control
high-weight Pauli gates before running the existing boolean optimization pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import RCCXGate
from qiskit.quantum_info import Operator, Statevector


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding_new import (  # type: ignore  # noqa: E402
    BlockEncoding,
    _append_full_control_pauli_expansion,
    _controlled_gate_single_qubit_pauli_data,
    _ctrl_state_bits,
    expand_larger_controlled_gates,
    optimize_expanded_circuit,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
)


NAM_BASIS = ("h", "x", "rz", "cx", "reset")
RESULT_ROOT = ROOT / "Baseline_results" / "two_control_high_weight_pauli_lowering_probe"


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    family: str
    size: int
    kraus_index: int | None = None


EXAMPLES: tuple[ExampleSpec, ...] = (
    ExampleSpec("TFIM_3_A0", "tfim", 3, 0),
    ExampleSpec("TFIM_4_A0", "tfim", 4, 0),
    ExampleSpec("TFIM_5_A0", "tfim", 5, 0),
    ExampleSpec("HRW_4_first", "hrw", 4, None),
    ExampleSpec("HRW_8_first", "hrw", 8, None),
    ExampleSpec("repetition_dqe_d3_first", "repetition_dqe", 3, None),
)


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    started = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - started, None
    except Exception:
        return "error", None, time.perf_counter() - started, traceback.format_exc()


def pauli_weight(label: str) -> int:
    return sum(ch != "I" for ch in label)


def select_kraus(kraus_ops: list[Any], preferred_index: int | None) -> tuple[int, Any]:
    if preferred_index is not None:
        op = kraus_ops[preferred_index]
        if len(op.instances) == 0:
            raise ValueError(f"preferred Kraus index {preferred_index} is empty")
        return preferred_index, op

    for index, op in enumerate(kraus_ops):
        if len(op.instances) > 0:
            return index, op
    raise ValueError("example has no nonempty Kraus operator")


def append_exact_control_ladder(
    output: QuantumCircuit,
    op: ControlledGate,
    qargs: list[Any],
    work: list[Any],
) -> None:
    ctrl_count = int(op.num_ctrl_qubits)
    controls = qargs[:ctrl_count]
    targets = qargs[ctrl_count:]
    output.ccx(controls[0], controls[1], work[0])
    for ctrl_index in range(2, ctrl_count - 1):
        output.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])

    cc_u = op.base_gate.control(num_ctrl_qubits=2, ctrl_state="11")
    output.append(cc_u, [work[-1], controls[-1]] + targets)

    for ctrl_index in range(ctrl_count - 2, 1, -1):
        output.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])
    output.ccx(controls[0], controls[1], work[0])


def append_rccx_control_ladder(
    output: QuantumCircuit,
    op: ControlledGate,
    qargs: list[Any],
    work: list[Any],
) -> None:
    """Use relative-phase Toffolis for the compute/uncompute ladder.

    This candidate is deliberately treated as experimental and is always
    checked by the equivalence verifier below.
    """

    ctrl_count = int(op.num_ctrl_qubits)
    controls = qargs[:ctrl_count]
    targets = qargs[ctrl_count:]
    output.append(RCCXGate(), [controls[0], controls[1], work[0]])
    for ctrl_index in range(2, ctrl_count - 1):
        output.append(RCCXGate(), [work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1]])

    cc_u = op.base_gate.control(num_ctrl_qubits=2, ctrl_state="11")
    output.append(cc_u, [work[-1], controls[-1]] + targets)

    for ctrl_index in range(ctrl_count - 2, 1, -1):
        output.append(RCCXGate().inverse(), [work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1]])
    output.append(RCCXGate().inverse(), [controls[0], controls[1], work[0]])


def expand_controlled_gates_candidate(
    qc: QuantumCircuit,
    *,
    expand_two_control_high_weight: bool,
    boolean_optimize: bool,
    use_rccx_ladders: bool = False,
    ancilla_name: str = "probe_anc",
) -> QuantumCircuit:
    max_ancillas = 0
    for instruction in qc.data:
        op = instruction.operation
        if not isinstance(op, ControlledGate):
            continue

        pauli_data = _controlled_gate_single_qubit_pauli_data(op)
        pauli_ops = None if pauli_data is None else pauli_data["ops"]
        is_high_weight_pauli = pauli_ops is not None and len(pauli_ops) >= 3
        ctrl_count = int(op.num_ctrl_qubits)

        if ctrl_count >= 3:
            max_ancillas = max(max_ancillas, ctrl_count - 1 if is_high_weight_pauli else ctrl_count - 2)
        elif expand_two_control_high_weight and ctrl_count == 2 and is_high_weight_pauli:
            max_ancillas = max(max_ancillas, 1)

    expanded = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    expanded.global_phase = qc.global_phase
    expanded.metadata = None if qc.metadata is None else dict(qc.metadata)

    ancillas: list[Any] = []
    if max_ancillas > 0:
        existing_names = {reg.name for reg in qc.qregs + qc.cregs}
        reg_name = ancilla_name
        suffix = 0
        while reg_name in existing_names:
            suffix += 1
            reg_name = f"{ancilla_name}_{suffix}"
        anc_reg = QuantumRegister(max_ancillas, reg_name)
        expanded.add_register(anc_reg)
        ancillas = list(anc_reg)

    for instruction in qc.data:
        op = instruction.operation
        qargs = list(instruction.qubits)
        clbits = list(instruction.clbits)

        if not isinstance(op, ControlledGate):
            expanded.append(op, qargs, clbits)
            continue
        if clbits:
            raise ValueError("Cannot expand classically-bit-attached controlled gates.")

        ctrl_count = int(op.num_ctrl_qubits)
        pauli_data = _controlled_gate_single_qubit_pauli_data(op)
        pauli_ops = None if pauli_data is None else pauli_data["ops"]
        is_high_weight_pauli = pauli_ops is not None and len(pauli_ops) >= 3
        should_expand = ctrl_count >= 3 or (
            expand_two_control_high_weight and ctrl_count == 2 and is_high_weight_pauli
        )
        if not should_expand:
            expanded.append(op, qargs, clbits)
            continue

        controls = qargs[:ctrl_count]
        targets = qargs[ctrl_count:]
        ctrl_bits = _ctrl_state_bits(op)
        for ctrl, bit in zip(controls, ctrl_bits[::-1]):
            if bit == "0":
                expanded.x(ctrl)

        if is_high_weight_pauli:
            work = ancillas[: ctrl_count - 1]
            _append_full_control_pauli_expansion(
                expanded,
                controls,
                targets,
                work,
                pauli_ops,
                pauli_data["phase"],
            )
        else:
            work = ancillas[: ctrl_count - 2]
            if use_rccx_ladders:
                append_rccx_control_ladder(expanded, op, qargs, work)
            else:
                append_exact_control_ladder(expanded, op, qargs, work)

        for ctrl, bit in zip(controls, ctrl_bits[::-1]):
            if bit == "0":
                expanded.x(ctrl)

    if boolean_optimize:
        return optimize_expanded_circuit(expanded, ancillas)
    return expanded


def build_variants(raw: QuantumCircuit) -> dict[str, QuantumCircuit]:
    return {
        "current_expand": expand_larger_controlled_gates(raw),
        "current_expand_boolean": expand_larger_controlled_gates(raw, if_optimize=True),
        "rccx_ladder": expand_controlled_gates_candidate(
            raw,
            expand_two_control_high_weight=False,
            boolean_optimize=False,
            use_rccx_ladders=True,
            ancilla_name="rccx_anc",
        ),
        "ctrl2_high_weight_boolean": expand_controlled_gates_candidate(
            raw,
            expand_two_control_high_weight=True,
            boolean_optimize=True,
            use_rccx_ladders=False,
            ancilla_name="heavy_anc",
        ),
    }


def ancilla_zero_indices(num_main_qubits: int, num_total_qubits: int) -> tuple[list[int], list[int]]:
    zero: list[int] = []
    nonzero: list[int] = []
    for index in range(1 << num_total_qubits):
        ancilla_bits = index >> num_main_qubits
        if ancilla_bits == 0:
            zero.append(index)
        else:
            nonzero.append(index)
    return zero, nonzero


def exact_equivalence(raw: QuantumCircuit, candidate: QuantumCircuit) -> dict[str, Any]:
    raw_matrix = Operator(raw).data
    cand_matrix = Operator(candidate).data
    zero_indices, nonzero_indices = ancilla_zero_indices(raw.num_qubits, candidate.num_qubits)
    block = cand_matrix[np.ix_(zero_indices, zero_indices)]
    leak = cand_matrix[np.ix_(nonzero_indices, zero_indices)]
    diff_norm = float(np.linalg.norm(raw_matrix - block))
    inner = np.vdot(raw_matrix.reshape(-1), block.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
    phase_corrected_diff_norm = float(np.linalg.norm(raw_matrix * phase - block))
    leak_norm = float(np.linalg.norm(leak))
    return {
        "method": "exact_operator_ancilla_zero_block",
        "passed": bool(phase_corrected_diff_norm < 1e-8 and leak_norm < 1e-8),
        "strict_passed": bool(diff_norm < 1e-8 and leak_norm < 1e-8),
        "diff_norm": diff_norm,
        "phase_corrected_diff_norm": phase_corrected_diff_norm,
        "global_phase": repr(phase),
        "leakage_norm": leak_norm,
    }


def spot_equivalence(
    raw: QuantumCircuit,
    candidate: QuantumCircuit,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    main_dim = 1 << raw.num_qubits
    total_dim = 1 << candidate.num_qubits
    zero_indices, nonzero_indices = ancilla_zero_indices(raw.num_qubits, candidate.num_qubits)
    max_diff = 0.0
    max_phase_corrected_diff = 0.0
    max_leak = 0.0
    phases: list[str] = []
    for _sample in range(samples):
        state = rng.normal(size=main_dim) + 1j * rng.normal(size=main_dim)
        state = state / np.linalg.norm(state)
        raw_out = Statevector(state).evolve(raw).data

        embedded = np.zeros(total_dim, dtype=complex)
        embedded[zero_indices] = state
        cand_out = Statevector(embedded).evolve(candidate).data
        block_out = cand_out[zero_indices]
        leak = cand_out[nonzero_indices]
        inner = np.vdot(raw_out, block_out)
        phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
        phases.append(repr(phase))
        max_diff = max(max_diff, float(np.linalg.norm(raw_out - block_out)))
        max_phase_corrected_diff = max(
            max_phase_corrected_diff,
            float(np.linalg.norm(raw_out * phase - block_out)),
        )
        max_leak = max(max_leak, float(np.linalg.norm(leak)))

    return {
        "method": "statevector_random_spotcheck",
        "samples": int(samples),
        "seed": int(seed),
        "passed": bool(max_phase_corrected_diff < 1e-8 and max_leak < 1e-8),
        "strict_passed": bool(max_diff < 1e-8 and max_leak < 1e-8),
        "max_diff_norm": max_diff,
        "max_phase_corrected_diff_norm": max_phase_corrected_diff,
        "max_leakage_norm": max_leak,
        "global_phase_samples": phases,
    }


def verify_equivalence(raw: QuantumCircuit, candidate: QuantumCircuit, *, exact_qubit_limit: int) -> dict[str, Any]:
    if candidate.num_qubits <= exact_qubit_limit:
        return exact_equivalence(raw, candidate)
    return spot_equivalence(raw, candidate, samples=8, seed=12345)


def pauli_label(instance: Any) -> str | None:
    label = getattr(instance, "expr", None)
    if label is not None:
        return str(label)
    pauli_op = getattr(instance, "pauli_op", None)
    if pauli_op is not None:
        return str(pauli_op)
    return None


def nam_metrics(qc: QuantumCircuit) -> dict[str, Any]:
    started = time.perf_counter()
    tqc = transpile(qc, basis_gates=list(NAM_BASIS), optimization_level=0)
    elapsed = time.perf_counter() - started
    ops = {str(name): int(count) for name, count in tqc.count_ops().items()}
    total = sum(int(ops.get(gate, 0)) for gate in ("h", "x", "rz", "cx"))
    return {
        "num_qubits": int(tqc.num_qubits),
        "size": int(tqc.size()),
        "depth": int(tqc.depth()),
        "ops": ops,
        "nam_total_h_x_rz_cx": int(total),
        "transpile_time_s": float(elapsed),
    }


def raw_instruction_summary(qc: QuantumCircuit) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ops": {str(name): int(count) for name, count in qc.count_ops().items()},
        "controlled_by_width": {},
        "controlled_by_width_weight": {},
    }
    by_width: dict[int, int] = {}
    by_width_weight: dict[str, int] = {}
    for instruction in qc.data:
        op = instruction.operation
        if not isinstance(op, ControlledGate):
            continue
        ctrl_count = int(op.num_ctrl_qubits)
        by_width[ctrl_count] = by_width.get(ctrl_count, 0) + 1
        pauli_data = _controlled_gate_single_qubit_pauli_data(op)
        if pauli_data is not None:
            weight = len(pauli_data["ops"])
            key = f"{ctrl_count}:{weight}"
            by_width_weight[key] = by_width_weight.get(key, 0) + 1
    summary["controlled_by_width"] = {str(k): int(v) for k, v in sorted(by_width.items())}
    summary["controlled_by_width_weight"] = dict(sorted(by_width_weight.items()))
    return summary


def matrixsum_metrics(ms: Any) -> dict[str, Any]:
    labels = [pauli_label(instance) for instance, _coeff in ms.instances]
    weights = [pauli_weight(label) for label in labels if label is not None]
    return {
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "max_pauli_weight": int(max(weights, default=0)),
        "weight_histogram": {str(w): int(weights.count(w)) for w in sorted(set(weights))},
        "instances_sample": [
            {"pauli": repr(instance), "coeff": repr(coeff)}
            for instance, coeff in list(ms.instances)[:10]
        ],
    }


def run_example(spec: ExampleSpec, *, exact_qubit_limit: int, nam: bool) -> dict[str, Any]:
    bench_spec = BenchmarkSpec(
        spec.name.replace("_A0", ""),
        spec.family,
        spec.size,
    )
    kraus_ops, model_info = build_kraus_ops(bench_spec)
    kraus_index, ms = select_kraus(kraus_ops, spec.kraus_index)
    be = BlockEncoding(copy.deepcopy(ms))
    raw, tcount, mccount, cxcount, ctrl_size = be.mulplex_U_opt_order()

    result: dict[str, Any] = {
        "example": spec.name,
        "family": spec.family,
        "size": int(spec.size),
        "model_info": model_info,
        "kraus_index": int(kraus_index),
        "matrixsum": matrixsum_metrics(ms),
        "raw_mulplex": {
            "num_qubits": int(raw.num_qubits),
            "ctrl_size": int(ctrl_size),
            "sys_size": int(be.sys_size),
            "size": int(raw.size()),
            "depth": int(raw.depth()),
            "estimated_tcount": int(tcount),
            "estimated_mccount": int(mccount),
            "estimated_cxcount": int(cxcount),
            "instruction_summary": raw_instruction_summary(raw),
            "g_modes_count": int(len(be.mobius_phase_result["g_modes_with_phase"])),
        },
        "variants": {},
    }

    for variant_name, circuit in build_variants(raw).items():
        status, verification, verify_elapsed, verify_error = timed(
            lambda circuit=circuit: verify_equivalence(raw, circuit, exact_qubit_limit=exact_qubit_limit)
        )
        row = {
            "num_qubits": int(circuit.num_qubits),
            "size": int(circuit.size()),
            "depth": int(circuit.depth()),
            "ops": {str(name): int(count) for name, count in circuit.count_ops().items()},
            "verification": {
                "status": status,
                "elapsed_seconds": float(verify_elapsed),
                "error": verify_error,
                "result": verification,
            },
        }
        if nam:
            nam_status, nam_value, nam_elapsed, nam_error = timed(lambda circuit=circuit: nam_metrics(circuit))
            row["nam_opt0"] = {
                "status": nam_status,
                "elapsed_seconds": float(nam_elapsed),
                "error": nam_error,
                "result": nam_value,
            }
        result["variants"][variant_name] = row
    return result


def markdown_summary(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Example | Variant | Verify | Method | Qubits | NAM total | CX | RZ | H | Depth |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for variant, row in result["variants"].items():
            verification = row["verification"]
            verify_result = verification.get("result") or {}
            nam_result = (row.get("nam_opt0", {}).get("result") or {})
            ops = nam_result.get("ops") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(result["example"]),
                        str(variant),
                        "yes" if verify_result.get("passed") else "no",
                        str(verify_result.get("method", "-")),
                        str(nam_result.get("num_qubits", row["num_qubits"])),
                        str(nam_result.get("nam_total_h_x_rz_cx", "-")),
                        str(ops.get("cx", "-")),
                        str(ops.get("rz", "-")),
                        str(ops.get("h", "-")),
                        str(nam_result.get("depth", "-")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--exact-qubit-limit", type=int, default=11)
    parser.add_argument("--tfim8-only", action="store_true")
    parser.add_argument("--no-nam", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples = (ExampleSpec("TFIM_8_A0", "tfim", 8, 0),) if args.tfim8_only else EXAMPLES + (
        ExampleSpec("TFIM_8_A0", "tfim", 8, 0),
    )

    results = [
        run_example(spec, exact_qubit_limit=args.exact_qubit_limit, nam=not args.no_nam)
        for spec in examples
    ]
    summary = {
        "description": "Controlled-Pauli lowering probe for BlockEncoding.mulplex_U_opt_order()",
        "note": (
            "ctrl2_high_weight_boolean is implemented only in this experiment script; "
            "block_encoding_new.py currently has boolean optimization support but does not "
            "expand two-control high-weight Pauli gates in expand_larger_controlled_gates()."
        ),
        "nam_basis": list(NAM_BASIS),
        "exact_qubit_limit": int(args.exact_qubit_limit),
        "results": results,
    }
    json_path = args.out_dir / "summary.json"
    md_path = args.out_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, default=repr), encoding="utf-8")
    md_path.write_text(markdown_summary(results) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    print(markdown_summary(results))


if __name__ == "__main__":
    main()
