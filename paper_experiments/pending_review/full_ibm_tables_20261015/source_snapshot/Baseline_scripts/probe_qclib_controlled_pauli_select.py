#!/usr/bin/env python3
"""Probe qclib compatibility with matrix-order controlled-Pauli SELECT.

The input is the high-level ``BlockEncoding.mulplex_U_opt_order()`` circuit,
whose main operations are multi-controlled Pauli strings.  The central adapter
in this script converts each controlled Pauli string into a serial list of
same-control single-target controlled X/Y/Z gates, plus the conditional global
phase carried by Qiskit's controlled composite Pauli gate.  Several qclib
multi-controlled-gate decomposers are then tried on the same adapted structure.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import PhaseGate, XGate, YGate, ZGate
from qiskit.quantum_info import Operator, Statevector


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

extra_qclib_site = os.environ.get("QCLIB_EXTRA_SITE")
if extra_qclib_site and Path(extra_qclib_site).exists() and extra_qclib_site not in sys.path:
    # Append, do not prepend: the active environment's Qiskit/Numpy stack should
    # win, while qclib may be discovered from a separate local installation.
    sys.path.append(extra_qclib_site)

from block_encoding_new import (  # type: ignore  # noqa: E402
    BlockEncoding,
    _controlled_gate_single_qubit_pauli_data,
    _ctrl_state_bits,
    _single_qubit_pauli_data,
    expand_large_controlled_gates,
)
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore  # noqa: E402


NAM_BASIS = ("h", "x", "rz", "cx", "reset")
RESULT_ROOT = ROOT / "Baseline_results" / "qclib_controlled_pauli_select_probe"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    num_qubits: int
    kraus_index: int = 0


CASES: tuple[CaseSpec, ...] = (
    CaseSpec("TFIM_4_A0", 4, 0),
    CaseSpec("TFIM_8_A0", 8, 0),
)


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    started = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - started, None
    except Exception:
        return "error", None, time.perf_counter() - started, traceback.format_exc()


def qiskit_instruction_parts(instruction: Any) -> tuple[Any, list[Any], list[Any]]:
    return instruction.operation, list(instruction.qubits), list(instruction.clbits)


def qiskit_metrics(qc: QuantumCircuit) -> dict[str, Any]:
    return {
        "num_qubits": int(qc.num_qubits),
        "num_clbits": int(qc.num_clbits),
        "size": int(qc.size()),
        "depth": int(qc.depth()),
        "ops": {str(name): int(count) for name, count in qc.count_ops().items()},
    }


def write_qasm(qc: QuantumCircuit, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)
    return {"path": str(path), "bytes": int(path.stat().st_size)}


def nam_metrics(qc: QuantumCircuit, path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    tqc = transpile(qc, basis_gates=list(NAM_BASIS), optimization_level=0)
    elapsed = time.perf_counter() - started
    write_qasm(tqc, path)
    ops = {str(name): int(count) for name, count in tqc.count_ops().items()}
    total = sum(int(ops.get(gate, 0)) for gate in ("h", "x", "rz", "cx"))
    return {
        "basis": "NAM",
        "basis_gates": list(NAM_BASIS),
        "optimization_level": 0,
        "num_qubits": int(tqc.num_qubits),
        "size": int(tqc.size()),
        "depth": int(tqc.depth()),
        "ops": ops,
        "nam_total_h_x_rz_cx": int(total),
        "transpile_time_s": float(elapsed),
        "qasm": str(path),
    }


def append_controlled_global_phase_qiskit(
    out_qc: QuantumCircuit,
    phase: float,
    controls: list[Any],
    ctrl_state: str,
    stats: dict[str, Any],
) -> None:
    if abs(float(phase)) < 1e-15:
        return
    if not controls:
        out_qc.global_phase += phase
        stats["uncontrolled_global_phases"] += 1
        return

    # Qiskit control-state strings are big-endian: the leftmost bit belongs to
    # the last control qubit in qargs, while the rightmost bit belongs to qargs[0].
    target = controls[-1]
    target_state = ctrl_state[0]
    remaining_controls = controls[:-1]
    remaining_state = ctrl_state[1:]

    if target_state == "0":
        out_qc.x(target)
        stats["open_control_x_conjugations"] += 2
    if remaining_controls:
        phase_gate = PhaseGate(phase).control(
            num_ctrl_qubits=len(remaining_controls),
            ctrl_state=remaining_state,
        )
        out_qc.append(phase_gate, remaining_controls + [target])
        stats["controlled_global_phases"] += 1
    else:
        out_qc.p(phase, target)
        stats["single_control_global_phases"] += 1
    if target_state == "0":
        out_qc.x(target)


def append_pauli_ops_qiskit(
    out_qc: QuantumCircuit,
    pauli_ops: list[tuple[str, int]],
    controls: list[Any],
    ctrl_state: str,
    targets: list[Any],
    stats: dict[str, Any],
) -> None:
    gate_by_pauli = {"X": XGate(), "Y": YGate(), "Z": ZGate()}
    for pauli, target_index in pauli_ops:
        target = targets[target_index]
        base_gate = gate_by_pauli[str(pauli)]
        if controls:
            controlled = base_gate.control(
                num_ctrl_qubits=len(controls),
                ctrl_state=ctrl_state,
            )
            out_qc.append(controlled, controls + [target])
            stats["controlled_single_pauli_gates"] += 1
            stats["controlled_single_pauli_by_width"][str(len(controls))] = (
                int(stats["controlled_single_pauli_by_width"].get(str(len(controls)), 0)) + 1
            )
        else:
            out_qc.append(base_gate, [target])
            stats["uncontrolled_single_pauli_gates"] += 1


def controlled_pauli_to_serial_single_target(qc: QuantumCircuit) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Split C^k(Pauli string) into serial C^k(X/Y/Z) gates.

    The conversion preserves instruction order and control structure.  It does
    not factor across neighboring Pauli strings or change address placement.
    """

    converted = QuantumCircuit(*qc.qregs, *qc.cregs, name=f"{qc.name}_serial_pauli")
    converted.global_phase = qc.global_phase
    converted.metadata = None if qc.metadata is None else dict(qc.metadata)
    stats: dict[str, Any] = {
        "controlled_pauli_strings_split": 0,
        "uncontrolled_pauli_strings_split": 0,
        "copied_instructions": 0,
        "controlled_single_pauli_gates": 0,
        "controlled_single_pauli_by_width": {},
        "uncontrolled_single_pauli_gates": 0,
        "controlled_global_phases": 0,
        "single_control_global_phases": 0,
        "uncontrolled_global_phases": 0,
        "open_control_x_conjugations": 0,
    }

    for instruction in qc.data:
        op, qargs, cargs = qiskit_instruction_parts(instruction)
        if cargs:
            converted.append(op, qargs, cargs)
            stats["copied_instructions"] += 1
            continue

        if isinstance(op, ControlledGate):
            pauli_data = _controlled_gate_single_qubit_pauli_data(op)
            if pauli_data is None:
                converted.append(op, qargs, cargs)
                stats["copied_instructions"] += 1
                continue
            ctrl_count = int(op.num_ctrl_qubits)
            controls = qargs[:ctrl_count]
            targets = qargs[ctrl_count:]
            ctrl_state = _ctrl_state_bits(op)
            append_controlled_global_phase_qiskit(
                converted,
                float(pauli_data["phase"]),
                controls,
                ctrl_state,
                stats,
            )
            append_pauli_ops_qiskit(
                converted,
                list(pauli_data["ops"]),
                controls,
                ctrl_state,
                targets,
                stats,
            )
            stats["controlled_pauli_strings_split"] += 1
            continue

        pauli_data = _single_qubit_pauli_data(op)
        if pauli_data is not None:
            append_controlled_global_phase_qiskit(
                converted,
                float(pauli_data["phase"]),
                [],
                "",
                stats,
            )
            append_pauli_ops_qiskit(
                converted,
                list(pauli_data["ops"]),
                [],
                "",
                qargs,
                stats,
            )
            stats["uncontrolled_pauli_strings_split"] += 1
            continue

        converted.append(op, qargs, cargs)
        stats["copied_instructions"] += 1
    return converted, stats


def append_controlled_global_phase_qclib(
    out_qc: QuantumCircuit,
    phase: float,
    controls: list[int],
    ctrl_state: str,
    stats: dict[str, Any],
) -> None:
    from qclib.gates.mcg import Mcg

    if abs(float(phase)) < 1e-15:
        return
    if not controls:
        out_qc.global_phase += phase
        stats["uncontrolled_global_phases"] += 1
        return

    target = controls[-1]
    target_state = ctrl_state[0]
    remaining_controls = controls[:-1]
    remaining_state = ctrl_state[1:]

    if target_state == "0":
        out_qc.x(target)
        stats["open_control_x_conjugations"] += 2
    if remaining_controls:
        out_qc.append(
            Mcg(PhaseGate(phase).to_matrix(), len(remaining_controls), ctrl_state=remaining_state),
            remaining_controls + [target],
        )
        stats["qclib_mcg_inserted"] += 1
        stats["qclib_mcg_by_base"]["p"] = int(stats["qclib_mcg_by_base"].get("p", 0)) + 1
        stats["controlled_global_phases"] += 1
    else:
        out_qc.p(phase, target)
        stats["single_control_global_phases"] += 1
    if target_state == "0":
        out_qc.x(target)


def qclib_u2_to_su2_with_phase(unitary: np.ndarray) -> tuple[np.ndarray, float]:
    from qclib.gates.util import u2_to_su2

    su2, phase = u2_to_su2(np.asarray(unitary, dtype=complex))
    return su2, float(phase)


def append_pauli_ops_qclib_mcg(
    out_qc: QuantumCircuit,
    pauli_ops: list[tuple[str, int]],
    controls: list[int],
    ctrl_state: str,
    targets: list[int],
    stats: dict[str, Any],
    *,
    up_to_diagonal: bool = False,
    correct_su2_phase: bool = False,
    use_su2_unitary: bool = False,
) -> None:
    from qclib.gates.mcg import Mcg

    gate_by_pauli = {"X": XGate(), "Y": YGate(), "Z": ZGate()}
    for pauli, target_index in pauli_ops:
        target = targets[target_index]
        base_gate = gate_by_pauli[str(pauli)]
        if controls:
            unitary = base_gate.to_matrix()
            if correct_su2_phase or use_su2_unitary:
                su2, phase = qclib_u2_to_su2_with_phase(unitary)
            if correct_su2_phase:
                append_controlled_global_phase_qclib(out_qc, phase, controls, ctrl_state, stats)
                stats["qclib_su2_phase_corrections"] += 1
            if use_su2_unitary:
                unitary = su2
            out_qc.append(
                Mcg(
                    unitary,
                    len(controls),
                    ctrl_state=ctrl_state,
                    up_to_diagonal=up_to_diagonal,
                ),
                controls + [target],
            )
            stats["qclib_mcg_inserted"] += 1
            stats["qclib_mcg_by_base"][str(pauli).lower()] = (
                int(stats["qclib_mcg_by_base"].get(str(pauli).lower(), 0)) + 1
            )
            stats["max_controls"] = max(int(stats["max_controls"]), len(controls))
        else:
            out_qc.append(base_gate, [target])
            stats["uncontrolled_single_pauli_gates"] += 1


def controlled_pauli_to_qclib_mcg(
    qc: QuantumCircuit,
    *,
    up_to_diagonal: bool = False,
    correct_su2_phase: bool = False,
    use_su2_unitary: bool = False,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Convert controlled-Pauli strings directly to qclib.gates.Mcg gates."""

    # Import early so the caller gets a clear tool-availability error.
    import qclib  # noqa: F401
    from qiskit import QuantumCircuit as QiskitQuantumCircuit

    converted = QiskitQuantumCircuit(qc.num_qubits, qc.num_clbits, global_phase=qc.global_phase)
    qubit_index = {qubit: index for index, qubit in enumerate(qc.qubits)}
    clbit_index = {clbit: index for index, clbit in enumerate(qc.clbits)}
    stats: dict[str, Any] = {
        "adapter": "controlled Pauli string -> serial qclib.gates.Mcg X/Y/Z/P",
        "qclib_mcg_up_to_diagonal": bool(up_to_diagonal),
        "qclib_su2_phase_corrected": bool(correct_su2_phase),
        "qclib_su2_unitary_used": bool(use_su2_unitary),
        "qclib_su2_phase_corrections": 0,
        "controlled_pauli_strings_split": 0,
        "uncontrolled_pauli_strings_split": 0,
        "copied_instructions": 0,
        "qclib_mcg_inserted": 0,
        "qclib_mcg_by_base": {},
        "max_controls": 0,
        "uncontrolled_single_pauli_gates": 0,
        "controlled_global_phases": 0,
        "single_control_global_phases": 0,
        "uncontrolled_global_phases": 0,
        "open_control_x_conjugations": 0,
    }

    for instruction in qc.data:
        op, qargs, cargs = qiskit_instruction_parts(instruction)
        qids = [qubit_index[qubit] for qubit in qargs]
        cids = [clbit_index[clbit] for clbit in cargs]
        if cids:
            converted.append(op, qids, cids)
            stats["copied_instructions"] += 1
            continue

        if isinstance(op, ControlledGate):
            pauli_data = _controlled_gate_single_qubit_pauli_data(op)
            if pauli_data is None:
                converted.append(op, qids, cids)
                stats["copied_instructions"] += 1
                continue
            ctrl_count = int(op.num_ctrl_qubits)
            controls = qids[:ctrl_count]
            targets = qids[ctrl_count:]
            ctrl_state = _ctrl_state_bits(op)
            append_controlled_global_phase_qclib(
                converted,
                float(pauli_data["phase"]),
                controls,
                ctrl_state,
                stats,
            )
            append_pauli_ops_qclib_mcg(
                converted,
                list(pauli_data["ops"]),
                controls,
                ctrl_state,
                targets,
                stats,
                up_to_diagonal=up_to_diagonal,
                correct_su2_phase=correct_su2_phase,
                use_su2_unitary=use_su2_unitary,
            )
            stats["controlled_pauli_strings_split"] += 1
            continue

        pauli_data = _single_qubit_pauli_data(op)
        if pauli_data is not None:
            append_controlled_global_phase_qclib(
                converted,
                float(pauli_data["phase"]),
                [],
                "",
                stats,
            )
            append_pauli_ops_qclib_mcg(
                converted,
                list(pauli_data["ops"]),
                [],
                "",
                qids,
                stats,
                up_to_diagonal=up_to_diagonal,
                correct_su2_phase=correct_su2_phase,
                use_su2_unitary=use_su2_unitary,
            )
            stats["uncontrolled_pauli_strings_split"] += 1
            continue

        converted.append(op, qids, cids)
        stats["copied_instructions"] += 1
    return converted, stats


def qclib_mcx_ancillas_needed(num_controls: int, backend: str) -> int:
    if num_controls < 3:
        return 0
    if backend == "vchain":
        return max(0, num_controls - 2)
    if backend == "gidney_linear":
        return 1
    if backend == "gidney_log":
        return 2
    raise ValueError(f"Unknown qclib MCX backend: {backend}")


def append_qclib_mcx(
    out_qc: QuantumCircuit,
    controls: list[int],
    ctrl_state: str,
    target: int,
    work: list[int],
    backend: str,
    stats: dict[str, Any],
) -> None:
    ctrl_count = len(controls)
    if ctrl_count == 0:
        out_qc.x(target)
        return

    if ctrl_count < 3:
        for ctrl, bit in zip(controls, ctrl_state[::-1]):
            if bit == "0":
                out_qc.x(ctrl)
                stats["open_control_x_conjugations"] += 2
        if ctrl_count == 1:
            out_qc.cx(controls[0], target)
        else:
            out_qc.ccx(controls[0], controls[1], target)
        for ctrl, bit in zip(controls, ctrl_state[::-1]):
            if bit == "0":
                out_qc.x(ctrl)
        stats["small_control_mcx_inserted"] += 1
        return

    if backend == "vchain":
        from qclib.gates.mcx import McxVchainDirty

        anc_count = qclib_mcx_ancillas_needed(ctrl_count, backend)
        out_qc.append(
            McxVchainDirty(ctrl_count, ctrl_state=ctrl_state),
            controls + work[:anc_count] + [target],
        )
    elif backend == "gidney_linear":
        from qclib.gates.mcx import MCXGidneyLinearDepth

        for ctrl, bit in zip(controls, ctrl_state[::-1]):
            if bit == "0":
                out_qc.x(ctrl)
                stats["open_control_x_conjugations"] += 2
        out_qc.append(MCXGidneyLinearDepth(ctrl_count, clean=True), controls + [target] + work[:1])
        for ctrl, bit in zip(controls, ctrl_state[::-1]):
            if bit == "0":
                out_qc.x(ctrl)
    elif backend == "gidney_log":
        from qclib.gates.mcx import MCXGidneyLogDepth

        for ctrl, bit in zip(controls, ctrl_state[::-1]):
            if bit == "0":
                out_qc.x(ctrl)
                stats["open_control_x_conjugations"] += 2
        out_qc.append(MCXGidneyLogDepth(ctrl_count, clean=True), controls + [target] + work[:2])
        for ctrl, bit in zip(controls, ctrl_state[::-1]):
            if bit == "0":
                out_qc.x(ctrl)
    else:
        raise ValueError(f"Unknown qclib MCX backend: {backend}")

    stats["qclib_mcx_inserted"] += 1
    stats["qclib_mcx_by_backend"][backend] = int(stats["qclib_mcx_by_backend"].get(backend, 0)) + 1
    stats["max_controls"] = max(int(stats["max_controls"]), ctrl_count)
    stats["max_work_qubits_per_mcx"] = max(
        int(stats["max_work_qubits_per_mcx"]),
        qclib_mcx_ancillas_needed(ctrl_count, backend),
    )


def append_controlled_pauli_via_qclib_mcx(
    out_qc: QuantumCircuit,
    pauli: str,
    controls: list[int],
    ctrl_state: str,
    target: int,
    work: list[int],
    backend: str,
    stats: dict[str, Any],
) -> None:
    if pauli == "X":
        append_qclib_mcx(out_qc, controls, ctrl_state, target, work, backend, stats)
    elif pauli == "Y":
        out_qc.sdg(target)
        append_qclib_mcx(out_qc, controls, ctrl_state, target, work, backend, stats)
        out_qc.s(target)
    elif pauli == "Z":
        out_qc.h(target)
        append_qclib_mcx(out_qc, controls, ctrl_state, target, work, backend, stats)
        out_qc.h(target)
    else:
        raise ValueError(f"Unsupported Pauli component: {pauli}")


def append_controlled_global_phase_qclib_mcx(
    out_qc: QuantumCircuit,
    phase: float,
    controls: list[int],
    ctrl_state: str,
    work: list[int],
    backend: str,
    stats: dict[str, Any],
) -> None:
    if abs(float(phase)) < 1e-15:
        return
    if not controls:
        out_qc.global_phase += phase
        stats["uncontrolled_global_phases"] += 1
        return
    if len(controls) == 1:
        target = controls[0]
        if ctrl_state == "0":
            out_qc.x(target)
            stats["open_control_x_conjugations"] += 2
        out_qc.p(phase, target)
        if ctrl_state == "0":
            out_qc.x(target)
        stats["single_control_global_phases"] += 1
        return
    if len(controls) == 2:
        # Qiskit's single-control phase decomposition is cheaper here than
        # burning a clean work qubit and two Toffoli-style MCX calls.
        append_controlled_global_phase_qclib(out_qc, phase, controls, ctrl_state, stats)
        return

    if not work:
        raise ValueError("MCX phase mode requires at least one clean work qubit")
    phase_target = work[0]
    mcx_work = work[1:]
    append_qclib_mcx(out_qc, controls, ctrl_state, phase_target, mcx_work, backend, stats)
    out_qc.p(phase, phase_target)
    append_qclib_mcx(out_qc, controls, ctrl_state, phase_target, mcx_work, backend, stats)
    stats["controlled_global_phases"] += 1
    stats["controlled_global_phases_via_mcx"] += 1


def append_pauli_ops_qclib_mcx(
    out_qc: QuantumCircuit,
    pauli_ops: list[tuple[str, int]],
    controls: list[int],
    ctrl_state: str,
    targets: list[int],
    work: list[int],
    backend: str,
    stats: dict[str, Any],
) -> None:
    gate_by_pauli = {"X": XGate(), "Y": YGate(), "Z": ZGate()}
    for pauli, target_index in pauli_ops:
        target = targets[target_index]
        if controls:
            append_controlled_pauli_via_qclib_mcx(
                out_qc,
                str(pauli),
                controls,
                ctrl_state,
                target,
                work,
                backend,
                stats,
            )
            stats["controlled_single_pauli_gates"] += 1
            stats["controlled_single_pauli_by_width"][str(len(controls))] = (
                int(stats["controlled_single_pauli_by_width"].get(str(len(controls)), 0)) + 1
            )
            stats["controlled_single_pauli_by_base"][str(pauli).lower()] = (
                int(stats["controlled_single_pauli_by_base"].get(str(pauli).lower(), 0)) + 1
            )
        else:
            out_qc.append(gate_by_pauli[str(pauli)], [target])
            stats["uncontrolled_single_pauli_gates"] += 1


def max_mcx_work_qubits_for_circuit(qc: QuantumCircuit, backend: str, *, phase_mode: str) -> int:
    max_work = 0
    for instruction in qc.data:
        op, _qargs, cargs = qiskit_instruction_parts(instruction)
        if cargs or not isinstance(op, ControlledGate):
            continue
        pauli_data = _controlled_gate_single_qubit_pauli_data(op)
        if pauli_data is None:
            continue
        ctrl_count = int(op.num_ctrl_qubits)
        max_work = max(max_work, qclib_mcx_ancillas_needed(ctrl_count, backend))
        if phase_mode == "mcx_compute" and abs(float(pauli_data["phase"])) >= 1e-15 and ctrl_count >= 3:
            max_work = max(max_work, 1 + qclib_mcx_ancillas_needed(ctrl_count, backend))
    return max_work


def controlled_pauli_to_qclib_mcx(
    qc: QuantumCircuit,
    *,
    backend: str,
    phase_mode: str = "mcg",
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Convert controlled-Pauli strings to qclib MCX-based X/Y/Z decompositions."""

    import qclib  # noqa: F401

    if phase_mode not in {"mcg", "mcx_compute"}:
        raise ValueError(f"Unknown phase mode: {phase_mode}")

    max_work = max_mcx_work_qubits_for_circuit(qc, backend, phase_mode=phase_mode)
    converted = QuantumCircuit(qc.num_qubits + max_work, qc.num_clbits, global_phase=qc.global_phase)
    qubit_index = {qubit: index for index, qubit in enumerate(qc.qubits)}
    clbit_index = {clbit: index for index, clbit in enumerate(qc.clbits)}
    work = list(range(qc.num_qubits, qc.num_qubits + max_work))
    stats: dict[str, Any] = {
        "adapter": f"controlled Pauli string -> serial qclib {backend} MCX-conjugated X/Y/Z",
        "qclib_mcx_backend": backend,
        "qclib_phase_mode": phase_mode,
        "allocated_work_qubits": int(max_work),
        "controlled_pauli_strings_split": 0,
        "uncontrolled_pauli_strings_split": 0,
        "copied_instructions": 0,
        "controlled_single_pauli_gates": 0,
        "controlled_single_pauli_by_width": {},
        "controlled_single_pauli_by_base": {},
        "uncontrolled_single_pauli_gates": 0,
        "qclib_mcx_inserted": 0,
        "qclib_mcx_by_backend": {},
        "small_control_mcx_inserted": 0,
        "max_controls": 0,
        "max_work_qubits_per_mcx": 0,
        "qclib_mcg_inserted": 0,
        "qclib_mcg_by_base": {},
        "controlled_global_phases": 0,
        "controlled_global_phases_via_mcx": 0,
        "single_control_global_phases": 0,
        "uncontrolled_global_phases": 0,
        "open_control_x_conjugations": 0,
    }

    for instruction in qc.data:
        op, qargs, cargs = qiskit_instruction_parts(instruction)
        qids = [qubit_index[qubit] for qubit in qargs]
        cids = [clbit_index[clbit] for clbit in cargs]
        if cids:
            converted.append(op, qids, cids)
            stats["copied_instructions"] += 1
            continue

        if isinstance(op, ControlledGate):
            pauli_data = _controlled_gate_single_qubit_pauli_data(op)
            if pauli_data is None:
                converted.append(op, qids, cids)
                stats["copied_instructions"] += 1
                continue
            ctrl_count = int(op.num_ctrl_qubits)
            controls = qids[:ctrl_count]
            targets = qids[ctrl_count:]
            ctrl_state = _ctrl_state_bits(op)
            if phase_mode == "mcx_compute":
                append_controlled_global_phase_qclib_mcx(
                    converted,
                    float(pauli_data["phase"]),
                    controls,
                    ctrl_state,
                    work,
                    backend,
                    stats,
                )
            else:
                append_controlled_global_phase_qclib(
                    converted,
                    float(pauli_data["phase"]),
                    controls,
                    ctrl_state,
                    stats,
                )
            append_pauli_ops_qclib_mcx(
                converted,
                list(pauli_data["ops"]),
                controls,
                ctrl_state,
                targets,
                work,
                backend,
                stats,
            )
            stats["controlled_pauli_strings_split"] += 1
            continue

        pauli_data = _single_qubit_pauli_data(op)
        if pauli_data is not None:
            if phase_mode == "mcx_compute":
                append_controlled_global_phase_qclib_mcx(
                    converted,
                    float(pauli_data["phase"]),
                    [],
                    "",
                    work,
                    backend,
                    stats,
                )
            else:
                append_controlled_global_phase_qclib(converted, float(pauli_data["phase"]), [], "", stats)
            append_pauli_ops_qclib_mcx(
                converted,
                list(pauli_data["ops"]),
                [],
                "",
                qids,
                work,
                backend,
                stats,
            )
            stats["uncontrolled_pauli_strings_split"] += 1
            continue

        converted.append(op, qids, cids)
        stats["copied_instructions"] += 1
    return converted, stats


def exact_equivalence(a: QuantumCircuit, b: QuantumCircuit) -> dict[str, Any]:
    a_matrix = Operator(a).data
    b_matrix = Operator(b).data
    diff_norm = float(np.linalg.norm(a_matrix - b_matrix))
    inner = np.vdot(a_matrix.reshape(-1), b_matrix.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
    phase_diff_norm = float(np.linalg.norm(a_matrix * phase - b_matrix))
    return {
        "method": "exact_operator",
        "passed": bool(phase_diff_norm < 1e-8),
        "strict_passed": bool(diff_norm < 1e-8),
        "diff_norm": diff_norm,
        "phase_corrected_diff_norm": phase_diff_norm,
        "global_phase": repr(phase),
    }


def ancilla_zero_indices(num_main_qubits: int, num_total_qubits: int) -> tuple[list[int], list[int]]:
    zero: list[int] = []
    nonzero: list[int] = []
    for index in range(1 << num_total_qubits):
        if (index >> num_main_qubits) == 0:
            zero.append(index)
        else:
            nonzero.append(index)
    return zero, nonzero


def exact_ancilla_zero_equivalence(a: QuantumCircuit, b: QuantumCircuit) -> dict[str, Any]:
    a_matrix = Operator(a).data
    b_matrix = Operator(b).data
    zero_indices, nonzero_indices = ancilla_zero_indices(a.num_qubits, b.num_qubits)
    block = b_matrix[np.ix_(zero_indices, zero_indices)]
    leakage = b_matrix[np.ix_(nonzero_indices, zero_indices)]
    diff_norm = float(np.linalg.norm(a_matrix - block))
    inner = np.vdot(a_matrix.reshape(-1), block.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
    phase_diff_norm = float(np.linalg.norm(a_matrix * phase - block))
    leakage_norm = float(np.linalg.norm(leakage))
    return {
        "method": "exact_operator_ancilla_zero_block",
        "passed": bool(phase_diff_norm < 1e-8 and leakage_norm < 1e-8),
        "strict_passed": bool(diff_norm < 1e-8 and leakage_norm < 1e-8),
        "diff_norm": diff_norm,
        "phase_corrected_diff_norm": phase_diff_norm,
        "leakage_norm": leakage_norm,
        "global_phase": repr(phase),
    }


def spot_equivalence(a: QuantumCircuit, b: QuantumCircuit, *, samples: int = 8, seed: int = 12345) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dim = 1 << a.num_qubits
    max_diff = 0.0
    max_phase_diff = 0.0
    phases: list[str] = []
    for _sample in range(samples):
        state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        state = state / np.linalg.norm(state)
        a_out = Statevector(state).evolve(a).data
        b_out = Statevector(state).evolve(b).data
        inner = np.vdot(a_out, b_out)
        phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
        phases.append(repr(phase))
        max_diff = max(max_diff, float(np.linalg.norm(a_out - b_out)))
        max_phase_diff = max(max_phase_diff, float(np.linalg.norm(a_out * phase - b_out)))
    return {
        "method": "statevector_random_spotcheck",
        "samples": int(samples),
        "seed": int(seed),
        "passed": bool(max_phase_diff < 1e-8),
        "strict_passed": bool(max_diff < 1e-8),
        "max_diff_norm": max_diff,
        "max_phase_corrected_diff_norm": max_phase_diff,
        "global_phase_samples": phases,
    }


def spot_ancilla_zero_equivalence(
    a: QuantumCircuit,
    b: QuantumCircuit,
    *,
    samples: int = 8,
    seed: int = 12345,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    main_dim = 1 << a.num_qubits
    total_dim = 1 << b.num_qubits
    zero_indices, nonzero_indices = ancilla_zero_indices(a.num_qubits, b.num_qubits)
    max_diff = 0.0
    max_phase_diff = 0.0
    max_leakage = 0.0
    phases: list[str] = []
    for _sample in range(samples):
        state = rng.normal(size=main_dim) + 1j * rng.normal(size=main_dim)
        state = state / np.linalg.norm(state)
        a_out = Statevector(state).evolve(a).data
        embedded = np.zeros(total_dim, dtype=complex)
        embedded[zero_indices] = state
        b_out = Statevector(embedded).evolve(b).data
        block_out = b_out[zero_indices]
        leakage = b_out[nonzero_indices]
        inner = np.vdot(a_out, block_out)
        phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
        phases.append(repr(phase))
        max_diff = max(max_diff, float(np.linalg.norm(a_out - block_out)))
        max_phase_diff = max(max_phase_diff, float(np.linalg.norm(a_out * phase - block_out)))
        max_leakage = max(max_leakage, float(np.linalg.norm(leakage)))
    return {
        "method": "statevector_random_spotcheck_ancilla_zero_block",
        "samples": int(samples),
        "seed": int(seed),
        "passed": bool(max_phase_diff < 1e-8 and max_leakage < 1e-8),
        "strict_passed": bool(max_diff < 1e-8 and max_leakage < 1e-8),
        "max_diff_norm": max_diff,
        "max_phase_corrected_diff_norm": max_phase_diff,
        "max_leakage_norm": max_leakage,
        "global_phase_samples": phases,
    }


def verify_equivalence(a: QuantumCircuit, b: QuantumCircuit, *, exact_qubit_limit: int) -> dict[str, Any]:
    if a.num_qubits > b.num_qubits:
        return {
            "method": "qubit_count_check",
            "passed": False,
            "error": f"candidate has fewer qubits than reference: {a.num_qubits} vs {b.num_qubits}",
        }
    if a.num_qubits < b.num_qubits:
        if b.num_qubits <= exact_qubit_limit:
            return exact_ancilla_zero_equivalence(a, b)
        return spot_ancilla_zero_equivalence(a, b)
    if a.num_qubits <= exact_qubit_limit:
        return exact_equivalence(a, b)
    return spot_equivalence(a, b)


def module_candidate_attrs(module_name: str, keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return [{"module": module_name, "error": f"{type(exc).__name__}: {exc}"}]

    for attr in dir(module):
        lower = attr.lower()
        if not any(keyword in lower for keyword in keywords):
            continue
        obj = getattr(module, attr)
        try:
            signature = str(inspect.signature(obj)) if callable(obj) else None
        except Exception:
            signature = None
        rows.append({"module": module_name, "attribute": attr, "signature": signature, "repr": repr(obj)[:240]})
    return rows


def inspect_tool_modules() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for tool, modules in {
        "qclib": (
            "qclib",
            "qclib.gates",
            "qclib.gates.mcg",
            "qclib.gates.mcu",
            "qclib.gates.mcx",
            "qclib.gates.qdmcu",
        ),
    }.items():
        tool_evidence: dict[str, Any] = {}
        try:
            module = importlib.import_module(modules[0])
            tool_evidence["installed"] = True
            tool_evidence["version"] = getattr(module, "__version__", "unknown")
            tool_evidence["module_file"] = getattr(module, "__file__", None)
            tool_evidence["candidate_attrs"] = []
            for module_name in modules:
                tool_evidence["candidate_attrs"].extend(
                    module_candidate_attrs(
                        module_name,
                        ("qiskit", "qasm", "control", "ctrl", "mc", "multi", "compile", "transpile", "from_", "import"),
                    )
                )
        except Exception:
            tool_evidence["installed"] = False
            tool_evidence["error"] = traceback.format_exc()
        evidence[tool] = tool_evidence
    return evidence


def build_tfim_a0_select(num_qubits: int, kraus_index: int) -> tuple[QuantumCircuit, dict[str, Any]]:
    kraus_ops, model_info = build_periodic_tfim_channel(num_qubits, 0.1)
    ms = kraus_ops[kraus_index]
    if len(ms.instances) == 0:
        raise ValueError(f"TFIM_{num_qubits} Kraus index {kraus_index} is empty")

    be = BlockEncoding(copy.deepcopy(ms))
    qc, tcount, mccount, cxcount, ctrl_size = be.mulplex_U_opt_order()
    metadata = {
        "model_info": model_info,
        "kraus_index": int(kraus_index),
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "ctrl_size": int(ctrl_size),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
        "g_modes_count": int(len(be.mobius_phase_result["g_modes_with_phase"])),
    }
    return qc, metadata


def run_qclib_adapted(
    raw_qc: QuantumCircuit,
    case_dir: Path,
    exact_qubit_limit: int,
    *,
    key: str,
    variant: str,
    qasm_stem: str,
    adapter: Callable[[], tuple[QuantumCircuit, dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool": "qclib",
        "variant": variant,
        "status": "running",
    }
    try:
        import qclib
        from qclib.gates.mcg import Mcg

        result["api_evidence"] = {
            "qclib_version": getattr(qclib, "__version__", "unknown"),
            "mcg_signature": str(inspect.signature(Mcg)),
        }
    except Exception:
        result["status"] = "not_available"
        result["error"] = traceback.format_exc()
        return result

    status, adapted_value, elapsed, error = timed(adapter)
    result["adapter_run"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "failed_adapter"
        return result

    qclib_qc, adapter_stats = adapted_value
    result["adapter_run"].update(adapter_stats)
    result["qiskit_metrics"] = qiskit_metrics(qclib_qc)
    result["qasm"] = write_qasm(qclib_qc, case_dir / f"{qasm_stem}.qasm")
    status, verification, elapsed, error = timed(
        lambda: verify_equivalence(raw_qc, qclib_qc, exact_qubit_limit=exact_qubit_limit)
    )
    result["verification"] = {"status": status, "elapsed_seconds": elapsed, "error": error, "result": verification}
    status, nam_value, elapsed, error = timed(
        lambda: nam_metrics(qclib_qc, case_dir / f"{qasm_stem}_nam_opt0.qasm")
    )
    result["nam_opt0"] = {"status": status, "elapsed_seconds": elapsed, "error": error, "result": nam_value}
    result["result_key"] = key
    result["status"] = "executed_full" if status == "ok" else "failed_nam_recount"
    return result


def run_qclib_variants(raw_qc: QuantumCircuit, case_dir: Path, exact_qubit_limit: int) -> dict[str, Any]:
    variants: list[tuple[str, str, str, Callable[[], tuple[QuantumCircuit, dict[str, Any]]]]] = [
        (
            "qclib_serial_mcg",
            "qclib serial Mcg exact U(2)",
            "qclib_serial_mcg",
            lambda: controlled_pauli_to_qclib_mcg(raw_qc),
        ),
        (
            "qclib_serial_mcg_su2_phase",
            "qclib serial Mcg up-to-diagonal plus explicit phase",
            "qclib_serial_mcg_su2_phase",
            lambda: controlled_pauli_to_qclib_mcg(
                raw_qc,
                up_to_diagonal=True,
                correct_su2_phase=True,
            ),
        ),
        (
            "qclib_serial_mcg_su2_exact",
            "qclib serial Mcg SU(2) plus explicit phase",
            "qclib_serial_mcg_su2_exact",
            lambda: controlled_pauli_to_qclib_mcg(
                raw_qc,
                correct_su2_phase=True,
                use_su2_unitary=True,
            ),
        ),
        (
            "qclib_mcx_vchain",
            "qclib MCX v-chain X/Y/Z plus Mcg phase",
            "qclib_mcx_vchain",
            lambda: controlled_pauli_to_qclib_mcx(raw_qc, backend="vchain"),
        ),
        (
            "qclib_mcx_gidney_linear",
            "qclib MCX Gidney linear X/Y/Z plus Mcg phase",
            "qclib_mcx_gidney_linear",
            lambda: controlled_pauli_to_qclib_mcx(raw_qc, backend="gidney_linear"),
        ),
        (
            "qclib_mcx_gidney_log",
            "qclib MCX Gidney log X/Y/Z plus Mcg phase",
            "qclib_mcx_gidney_log",
            lambda: controlled_pauli_to_qclib_mcx(raw_qc, backend="gidney_log"),
        ),
        (
            "qclib_mcx_vchain_phase_mcx",
            "qclib MCX v-chain X/Y/Z plus MCX phase",
            "qclib_mcx_vchain_phase_mcx",
            lambda: controlled_pauli_to_qclib_mcx(raw_qc, backend="vchain", phase_mode="mcx_compute"),
        ),
        (
            "qclib_mcx_gidney_linear_phase_mcx",
            "qclib MCX Gidney linear X/Y/Z plus MCX phase",
            "qclib_mcx_gidney_linear_phase_mcx",
            lambda: controlled_pauli_to_qclib_mcx(raw_qc, backend="gidney_linear", phase_mode="mcx_compute"),
        ),
        (
            "qclib_mcx_gidney_log_phase_mcx",
            "qclib MCX Gidney log X/Y/Z plus MCX phase",
            "qclib_mcx_gidney_log_phase_mcx",
            lambda: controlled_pauli_to_qclib_mcx(raw_qc, backend="gidney_log", phase_mode="mcx_compute"),
        ),
    ]
    results: dict[str, Any] = {}
    for key, variant, qasm_stem, adapter in variants:
        results[key] = run_qclib_adapted(
            raw_qc,
            case_dir,
            exact_qubit_limit,
            key=key,
            variant=variant,
            qasm_stem=qasm_stem,
            adapter=adapter,
        )
    return results


def run_case(spec: CaseSpec, out_dir: Path, exact_qubit_limit: int) -> dict[str, Any]:
    case_dir = out_dir / spec.name
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_qc, metadata = build_tfim_a0_select(spec.num_qubits, spec.kraus_index)
    serial_qc, serial_stats = controlled_pauli_to_serial_single_target(raw_qc)
    expanded_boolean = expand_large_controlled_gates(raw_qc, if_optimize=True)

    result: dict[str, Any] = {
        "case": spec.name,
        "num_system_qubits": int(spec.num_qubits),
        "raw_select_metadata": metadata,
        "raw_select": {
            "qiskit_metrics": qiskit_metrics(raw_qc),
            "qasm": write_qasm(raw_qc, case_dir / "raw_mulplex_U_opt_order.qasm"),
            "nam_opt0": nam_metrics(raw_qc, case_dir / "raw_mulplex_U_opt_order_nam_opt0.qasm"),
        },
        "serial_same_control_pauli": {
            "conversion_stats": serial_stats,
            "qiskit_metrics": qiskit_metrics(serial_qc),
            "qasm": write_qasm(serial_qc, case_dir / "serial_same_control_pauli.qasm"),
            "verification": verify_equivalence(raw_qc, serial_qc, exact_qubit_limit=exact_qubit_limit),
            "nam_opt0": nam_metrics(serial_qc, case_dir / "serial_same_control_pauli_nam_opt0.qasm"),
        },
        "current_expand_boolean": {
            "qiskit_metrics": qiskit_metrics(expanded_boolean),
            "qasm": write_qasm(expanded_boolean, case_dir / "current_expand_boolean.qasm"),
            "verification": verify_equivalence(raw_qc, expanded_boolean, exact_qubit_limit=exact_qubit_limit),
            "nam_opt0": nam_metrics(expanded_boolean, case_dir / "current_expand_boolean_nam_opt0.qasm"),
        },
        "tool_runs": {},
    }
    result["tool_runs"].update(run_qclib_variants(raw_qc, case_dir, exact_qubit_limit))
    return result


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# qclib Controlled-Pauli SELECT Probe",
        "",
        "## Tool Availability",
        "",
        "| Tool | Installed | Version / note |",
        "|---|---:|---|",
    ]
    for tool in ("qclib",):
        evidence = summary["tool_evidence"].get(tool, {})
        version = evidence.get("version", evidence.get("note", evidence.get("error", "")))
        lines.append(f"| {tool} | {str(evidence.get('installed', False)).lower()} | {str(version).splitlines()[0][:120]} |")

    lines.extend(
        [
            "",
            "## NAM opt0 Results",
            "",
            "| Case | Method | Status | Verify | Qubits | NAM total | CX | RZ | H | X | Depth |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def add_row(case: str, method: str, status: str, verify: Any, nam: Any) -> None:
        nam_result = nam.get("result", nam) if isinstance(nam, dict) else {}
        ops = nam_result.get("ops", {}) if isinstance(nam_result, dict) else {}
        verify_result = verify.get("result", verify) if isinstance(verify, dict) else verify
        verify_cell = "-"
        if isinstance(verify_result, dict):
            verify_cell = "yes" if verify_result.get("passed") else "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    method,
                    status,
                    verify_cell,
                    str(nam_result.get("num_qubits", "-")),
                    str(nam_result.get("nam_total_h_x_rz_cx", "-")),
                    str(ops.get("cx", 0)),
                    str(ops.get("rz", 0)),
                    str(ops.get("h", 0)),
                    str(ops.get("x", 0)),
                    str(nam_result.get("depth", "-")),
                ]
            )
            + " |"
        )

    for result in summary["results"]:
        case = result["case"]
        add_row(case, "raw mulplex_U_opt_order", "ok", {"passed": True}, result["raw_select"]["nam_opt0"])
        add_row(
            case,
            "serial same-control C^k X/Y/Z",
            "ok",
            result["serial_same_control_pauli"]["verification"],
            result["serial_same_control_pauli"]["nam_opt0"],
        )
        add_row(
            case,
            "current expand boolean",
            "ok",
            result["current_expand_boolean"]["verification"],
            result["current_expand_boolean"]["nam_opt0"],
        )
        for key in (
            "qclib_serial_mcg",
            "qclib_serial_mcg_su2_phase",
            "qclib_serial_mcg_su2_exact",
            "qclib_mcx_vchain",
            "qclib_mcx_gidney_linear",
            "qclib_mcx_gidney_log",
            "qclib_mcx_vchain_phase_mcx",
            "qclib_mcx_gidney_linear_phase_mcx",
            "qclib_mcx_gidney_log_phase_mcx",
        ):
            qclib = result["tool_runs"].get(key, {})
            add_row(
                case,
                qclib.get("variant", key),
                qclib.get("status", "missing"),
                qclib.get("verification", {}),
                qclib.get("nam_opt0", {}),
            )

    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- JSON summary: `{summary['summary_path']}`")
    for result in summary["results"]:
        lines.append(f"- `{result['case']}` artifacts: `{Path(summary['out_dir']) / result['case']}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--cases", nargs="+", default=[case.name for case in CASES])
    parser.add_argument("--exact-qubit-limit", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.cases)
    cases = [case for case in CASES if case.name in selected]
    summary: dict[str, Any] = {
        "experiment": "qclib_controlled_pauli_select_probe",
        "out_dir": str(out_dir),
        "tool_evidence": inspect_tool_modules(),
        "exact_qubit_limit": int(args.exact_qubit_limit),
        "results": [],
    }
    for case in cases:
        summary["results"].append(run_case(case, out_dir, args.exact_qubit_limit))

    summary_path = out_dir / "summary.json"
    report_path = out_dir / "summary.md"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=repr) + "\n", encoding="utf-8")
    report_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}, indent=2))
    print(markdown_summary(summary))


if __name__ == "__main__":
    main()
