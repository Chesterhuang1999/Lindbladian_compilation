#!/usr/bin/env python3
"""qclib MCG adapter probe for the raw TFIM-8 A0 SELECT module.

This script deliberately uses the unoptimized ``BlockEncoding.mulplex_U`` path:
no channel rewrite, no matrix-order placement, and no coefficient PREPARE.
The qclib variants rewrite each controlled Pauli string into serial
multi-controlled single-qubit gates, which matches qclib's native MCG APIs.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import PhaseGate, XGate, YGate, ZGate
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
BASELINE_DIR = ROOT / "Baseline_scripts"
QCLIB_SITE = Path(
    "/Users/chester_hqf/miniconda3/envs/ChannelIR_test/lib/python3.12/site-packages"
)

for candidate in (SRC_DIR, SCRIPTS_DIR, BASELINE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

if str(QCLIB_SITE) not in sys.path:
    # Use qiskit_simulate as the active stack, but import qclib from ChannelIR_test.
    sys.path.append(str(QCLIB_SITE))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import (  # type: ignore  # noqa: E402
    BlockEncoding,
    _controlled_gate_single_qubit_pauli_data,
    _ctrl_state_bits,
    _single_qubit_pauli_data,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    qiskit_metrics,
    recount_qiskit_circuit,
    write_qasm2,
)
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore  # noqa: E402


RESULT_ROOT = ROOT / "Baseline_results" / "qclib_tfim8_a0_raw_select_mcg_v1"


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    started = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - started, None
    except Exception:
        return "error", None, time.perf_counter() - started, traceback.format_exc()


def write_qasm(qc: QuantumCircuit, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)
    return {"path": str(path), "bytes": int(path.stat().st_size)}


def recount_bases(qc: QuantumCircuit, stem: str, out_dir: Path) -> dict[str, Any]:
    return {
        "nam": recount_qiskit_circuit(
            qc,
            basis_name="nam",
            out_path=out_dir / f"{stem}_nam_opt0.qasm",
            optimization_level=0,
        ),
        "ibm": recount_qiskit_circuit(
            qc,
            basis_name="ibm",
            out_path=out_dir / f"{stem}_ibm_opt0.qasm",
            optimization_level=0,
        ),
    }


def qiskit_instruction_parts(instruction: Any) -> tuple[Any, list[Any], list[Any]]:
    return instruction.operation, list(instruction.qubits), list(instruction.clbits)


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
            stats["controlled_single_pauli_by_base"][str(pauli).lower()] = (
                int(stats["controlled_single_pauli_by_base"].get(str(pauli).lower(), 0)) + 1
            )
        else:
            out_qc.append(base_gate, [target])
            stats["uncontrolled_single_pauli_gates"] += 1


def controlled_pauli_to_serial_single_target(qc: QuantumCircuit) -> tuple[QuantumCircuit, dict[str, Any]]:
    converted = QuantumCircuit(*qc.qregs, *qc.cregs, name=f"{qc.name}_serial_pauli")
    converted.global_phase = qc.global_phase
    converted.metadata = None if qc.metadata is None else dict(qc.metadata)
    stats: dict[str, Any] = new_adapter_stats("qiskit serial C^k(X/Y/Z)")

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
            append_controlled_global_phase_qiskit(converted, float(pauli_data["phase"]), [], "", stats)
            append_pauli_ops_qiskit(converted, list(pauli_data["ops"]), [], "", qargs, stats)
            stats["uncontrolled_pauli_strings_split"] += 1
            continue

        converted.append(op, qargs, cargs)
        stats["copied_instructions"] += 1
    return converted, stats


def new_adapter_stats(adapter: str) -> dict[str, Any]:
    return {
        "adapter": adapter,
        "controlled_pauli_strings_split": 0,
        "uncontrolled_pauli_strings_split": 0,
        "copied_instructions": 0,
        "controlled_single_pauli_gates": 0,
        "controlled_single_pauli_by_width": {},
        "controlled_single_pauli_by_base": {},
        "uncontrolled_single_pauli_gates": 0,
        "controlled_global_phases": 0,
        "single_control_global_phases": 0,
        "uncontrolled_global_phases": 0,
        "open_control_x_conjugations": 0,
        "qclib_mcg_inserted": 0,
        "qclib_mcg_by_base": {},
        "qclib_su2_phase_corrections": 0,
        "max_controls": 0,
    }


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
        stats["controlled_global_phases"] += 1
        stats["qclib_mcg_inserted"] += 1
        stats["qclib_mcg_by_base"]["p"] = int(stats["qclib_mcg_by_base"].get("p", 0)) + 1
        stats["max_controls"] = max(int(stats["max_controls"]), len(remaining_controls))
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
    use_su2_unitary: bool = False,
) -> None:
    from qclib.gates.mcg import Mcg

    gate_by_pauli = {"X": XGate(), "Y": YGate(), "Z": ZGate()}
    for pauli, target_index in pauli_ops:
        target = targets[target_index]
        base_gate = gate_by_pauli[str(pauli)]
        if controls:
            unitary = np.asarray(base_gate.to_matrix(), dtype=complex)
            if use_su2_unitary:
                su2, phase = qclib_u2_to_su2_with_phase(unitary)
                append_controlled_global_phase_qclib(out_qc, phase, controls, ctrl_state, stats)
                stats["qclib_su2_phase_corrections"] += 1
                unitary = su2
            out_qc.append(
                Mcg(unitary, len(controls), ctrl_state=ctrl_state),
                controls + [target],
            )
            stats["qclib_mcg_inserted"] += 1
            stats["qclib_mcg_by_base"][str(pauli).lower()] = (
                int(stats["qclib_mcg_by_base"].get(str(pauli).lower(), 0)) + 1
            )
            stats["controlled_single_pauli_gates"] += 1
            stats["controlled_single_pauli_by_width"][str(len(controls))] = (
                int(stats["controlled_single_pauli_by_width"].get(str(len(controls)), 0)) + 1
            )
            stats["controlled_single_pauli_by_base"][str(pauli).lower()] = (
                int(stats["controlled_single_pauli_by_base"].get(str(pauli).lower(), 0)) + 1
            )
            stats["max_controls"] = max(int(stats["max_controls"]), len(controls))
        else:
            out_qc.append(base_gate, [target])
            stats["uncontrolled_single_pauli_gates"] += 1


def controlled_pauli_to_qclib_mcg(
    qc: QuantumCircuit,
    *,
    use_su2_unitary: bool,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    import qclib  # noqa: F401

    converted = QuantumCircuit(qc.num_qubits, qc.num_clbits, global_phase=qc.global_phase)
    qubit_index = {qubit: index for index, qubit in enumerate(qc.qubits)}
    clbit_index = {clbit: index for index, clbit in enumerate(qc.clbits)}
    adapter = "controlled Pauli string -> serial qclib Mcg SU(2)+phase"
    if not use_su2_unitary:
        adapter = "controlled Pauli string -> serial qclib Mcg U(2)"
    stats = new_adapter_stats(adapter)
    stats["qclib_mcg_su2_unitary_used"] = bool(use_su2_unitary)

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
                use_su2_unitary=use_su2_unitary,
            )
            stats["controlled_pauli_strings_split"] += 1
            continue

        pauli_data = _single_qubit_pauli_data(op)
        if pauli_data is not None:
            append_controlled_global_phase_qclib(converted, float(pauli_data["phase"]), [], "", stats)
            append_pauli_ops_qclib_mcg(
                converted,
                list(pauli_data["ops"]),
                [],
                "",
                qids,
                stats,
                use_su2_unitary=use_su2_unitary,
            )
            stats["uncontrolled_pauli_strings_split"] += 1
            continue

        converted.append(op, qids, cids)
        stats["copied_instructions"] += 1
    return converted, stats


def spot_equivalence(
    reference: QuantumCircuit,
    candidate: QuantumCircuit,
    *,
    samples: int = 8,
    seed: int = 20260705,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dim = 1 << reference.num_qubits
    max_diff = 0.0
    max_phase_diff = 0.0
    phases: list[str] = []
    for _ in range(samples):
        state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        state = state / np.linalg.norm(state)
        ref_out = Statevector(state).evolve(reference).data
        cand_out = Statevector(state).evolve(candidate).data
        inner = np.vdot(ref_out, cand_out)
        phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
        phases.append(repr(phase))
        max_diff = max(max_diff, float(np.linalg.norm(ref_out - cand_out)))
        max_phase_diff = max(max_phase_diff, float(np.linalg.norm(ref_out * phase - cand_out)))
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


def inspect_qclib_mcg_support() -> dict[str, Any]:
    modules = (
        "qclib.gates.mcg",
        "qclib.gates.ldmcsu",
        "qclib.gates.ldmcu",
        "qclib.gates.mcu",
        "qclib.gates.qdmcu",
        "qclib.gates.mcx",
    )
    evidence: dict[str, Any] = {
        "qclib_site_appended": str(QCLIB_SITE),
        "modules": {},
        "classification": {
            "mcg": "multi-controlled single-qubit unitary, i.e. U(2)/SU(2) with one target",
            "mcx": "specialized multi-controlled X families",
            "not_supported_directly": "arbitrary multi-controlled multi-qubit gates such as a controlled Pauli string without splitting",
        },
    }
    try:
        qclib = importlib.import_module("qclib")
        evidence["installed"] = True
        evidence["version"] = getattr(qclib, "__version__", "unknown")
        evidence["module_file"] = getattr(qclib, "__file__", None)
    except Exception:
        evidence["installed"] = False
        evidence["error"] = traceback.format_exc()
        return evidence

    for module_name in modules:
        row: dict[str, Any] = {}
        try:
            module = importlib.import_module(module_name)
            row["file"] = getattr(module, "__file__", None)
            row["callables"] = {}
            for attr in ("Mcg", "Ldmcsu", "Ldmcu", "MCU", "Qdmcu", "McxVchainDirty", "MCXGidneyLinearDepth", "MCXGidneyLogDepth"):
                if hasattr(module, attr):
                    obj = getattr(module, attr)
                    row["callables"][attr] = str(inspect.signature(obj))
        except Exception:
            row["error"] = traceback.format_exc()
        evidence["modules"][module_name] = row
    return evidence


def build_tfim8_a0_raw_select() -> tuple[QuantumCircuit, dict[str, Any]]:
    kraus_ops, model_info = build_periodic_tfim_channel(8, 0.1)
    ms = kraus_ops[0]
    if len(ms.instances) == 0:
        raise ValueError("TFIM_8 A0 is empty")

    be = BlockEncoding(copy.deepcopy(ms))
    qc, tcount, mccount, cxcount = be.mulplex_U(be.mat_list, be.ctrl_size, be.sys_size)
    metadata = {
        "benchmark": "TFIM_8_A0",
        "scope": "SELECT only",
        "input_boundary": "BlockEncoding.mulplex_U(mat_list, ctrl_size, sys_size), equivalent to structure='no'/opt='No' SELECT",
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "ctrl_size": int(be.ctrl_size),
        "pauli_norm": float(ms.pauli_norm()),
        "model_info": model_info,
        "analytical_counts_from_mulplex_U": {
            "tcount": int(tcount),
            "mccount": int(mccount),
            "cxcount": int(cxcount),
        },
    }
    return qc, metadata


def run_variant(
    raw_qc: QuantumCircuit,
    out_dir: Path,
    *,
    key: str,
    label: str,
    adapter: Callable[[], tuple[QuantumCircuit, dict[str, Any]]],
    verify_samples: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"key": key, "label": label, "status": "running"}
    status, adapted, elapsed, error = timed(adapter)
    result["adapter"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "failed_adapter"
        return result

    qc, stats = adapted
    result["adapter"].update(stats)
    result["qiskit_metrics"] = qiskit_metrics(qc)
    result["qasm"] = write_qasm(qc, out_dir / f"{key}.qasm")
    status, verification, elapsed, error = timed(lambda: spot_equivalence(raw_qc, qc, samples=verify_samples))
    result["verification"] = {"status": status, "elapsed_seconds": elapsed, "error": error, "result": verification}
    status, recounts, elapsed, error = timed(lambda: recount_bases(qc, key, out_dir))
    result["basis_recounts"] = {"status": status, "elapsed_seconds": elapsed, "error": error, "result": recounts}
    result["status"] = "ok" if status == "ok" else "failed_recount"
    return result


def row_from_recount(key: str, label: str, recount: dict[str, Any]) -> dict[str, Any]:
    stats = recount.get("stats", {})
    return {
        "key": key,
        "label": label,
        "basis": recount.get("basis"),
        "width": stats.get("num_qubits"),
        "metric_total": stats.get("metric_total"),
        "non_clifford": stats.get("clifford", {}).get("non_clifford"),
        "qasm": recount.get("qasm"),
    }


def build_comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_recounts = summary["raw_select"]["basis_recounts"]
    for basis in ("nam", "ibm"):
        rows.append(row_from_recount("raw_select", "raw mulplex_U SELECT", raw_recounts[basis]))
    for key, variant in summary["variants"].items():
        recounts = variant.get("basis_recounts", {}).get("result")
        if not recounts:
            continue
        for basis in ("nam", "ibm"):
            rows.append(row_from_recount(key, variant["label"], recounts[basis]))
    return rows


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# qclib TFIM-8 A0 Raw SELECT MCG Probe",
        "",
        f"- Input: `{summary['metadata']['input_boundary']}`",
        f"- Terms/control width/system width: `{summary['metadata']['pauli_terms']}` / `{summary['metadata']['ctrl_size']}` / `{summary['metadata']['system_qubits']}`",
        "",
        "## qclib MCG Support",
        "",
        "- `Mcg` accepts a 2x2 unitary and one target, dispatching SU(2) to `Ldmcsu` and general U(2) to `Ldmcu`.",
        "- Specialized MCX classes exist separately (`McxVchainDirty`, `MCXGidneyLinearDepth`, `MCXGidneyLogDepth`).",
        "- qclib does not directly optimize arbitrary multi-controlled multi-qubit gates; controlled Pauli strings must be split into same-control single-target gates.",
        "",
        "## Counts",
        "",
        "| Variant | Basis | Width | Gates | Non-Clifford | Verification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    verification_by_key = {
        "raw_select": {"passed": True},
        **{
            key: variant.get("verification", {}).get("result", {})
            for key, variant in summary.get("variants", {}).items()
        },
    }
    for row in summary["comparison_rows"]:
        verification = verification_by_key.get(row["key"], {})
        passed = verification.get("passed")
        if passed is True:
            vcell = "pass"
        elif passed is False:
            vcell = "fail"
        else:
            vcell = "-"
        lines.append(
            f"| {row['label']} | {str(row['basis']).upper()} | {row['width']} | "
            f"{row['metric_total']} | {row['non_clifford']} | {vcell} |"
        )
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- JSON summary: `{summary['summary_path']}`")
    lines.append(f"- Output directory: `{summary['out_dir']}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--verify-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_qc, metadata = build_tfim8_a0_raw_select()
    raw_qasm = write_qasm2(raw_qc, out_dir / "raw_mulplex_U_select.qasm")
    raw_recounts = recount_bases(raw_qc, "raw_mulplex_U_select", out_dir)

    summary: dict[str, Any] = {
        "experiment": "qclib_tfim8_a0_raw_select_mcg",
        "out_dir": str(out_dir),
        "metadata": metadata,
        "qclib_support": inspect_qclib_mcg_support(),
        "raw_select": {
            "qiskit_metrics": qiskit_metrics(raw_qc),
            "qasm": raw_qasm,
            "basis_recounts": raw_recounts,
        },
        "variants": {},
    }

    variants: list[tuple[str, str, Callable[[], tuple[QuantumCircuit, dict[str, Any]]]]] = [
        (
            "qiskit_serial_single_target",
            "Qiskit serial C^k(X/Y/Z)",
            lambda: controlled_pauli_to_serial_single_target(raw_qc),
        ),
        (
            "qclib_mcg_u2",
            "qclib Mcg U(2)",
            lambda: controlled_pauli_to_qclib_mcg(raw_qc, use_su2_unitary=False),
        ),
        (
            "qclib_mcg_su2_phase",
            "qclib Mcg SU(2)+phase",
            lambda: controlled_pauli_to_qclib_mcg(raw_qc, use_su2_unitary=True),
        ),
    ]

    for key, label, adapter in variants:
        summary["variants"][key] = run_variant(
            raw_qc,
            out_dir,
            key=key,
            label=label,
            adapter=adapter,
            verify_samples=args.verify_samples,
        )

    summary["comparison_rows"] = build_comparison_rows(summary)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "summary.md"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=repr) + "\n", encoding="utf-8")
    report_path.write_text(markdown_summary(summary), encoding="utf-8")

    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}, indent=2))
    print(markdown_summary(summary))


if __name__ == "__main__":
    main()
