#!/usr/bin/env python3
"""Test pruned Gray-code UCR PREPARE for TFIM_8 A0 matrix-order auto width.

This experiment only targets the coefficient PREPARE state

    sum_i sqrt(|c_i| / sum_j |c_j|) |i>

after ``block_encoding_new.BlockEncoding`` has chosen its automatic
matrix-order placement.  It does not synthesize or count SELECT.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding_new import BlockEncoding  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_auto9_pruned_ucr_prepare_v1"
TOL = 1e-15


def dense_prepare_from_weights(weights: np.ndarray) -> QuantumCircuit:
    n = int(round(math.log2(len(weights))))
    probs = weights / float(np.sum(weights))
    amps = np.sqrt(probs)
    qc = QuantumCircuit(n, name="dense_qiskit_prepare")
    qc.append(StatePreparation(Statevector(amps)), range(n))
    return qc


def target_state_from_weights(weights: np.ndarray) -> Statevector:
    probs = weights / float(np.sum(weights))
    return Statevector(np.sqrt(probs).astype(float))


def infer_ucry_angle_position(control_count: int) -> dict[tuple[int, ...], int]:
    """Map control-bit tuples in qarg order to UCRYGate angle-list positions."""

    if control_count == 0:
        return {(): 0}

    angles = [0.13 + 0.031 * idx for idx in range(1 << control_count)]
    mapping: dict[tuple[int, ...], int] = {}
    for bits in itertools.product((0, 1), repeat=control_count):
        qc = QuantumCircuit(control_count + 1)
        for offset, bit in enumerate(bits, start=1):
            if bit:
                qc.x(offset)
        qc.append(UCRYGate(angles), range(control_count + 1))
        sv = Statevector.from_instruction(qc).data
        base_index = sum(bit << offset for offset, bit in enumerate(bits, start=1))
        amp0 = float(np.real_if_close(sv[base_index]))
        amp1 = float(np.real_if_close(sv[base_index | 1]))
        theta = 2.0 * math.atan2(amp1, amp0)
        position = min(range(len(angles)), key=lambda idx: abs(theta - angles[idx]))
        mapping[bits] = int(position)
    return mapping


_UCRY_POSITION_CACHE: dict[int, dict[tuple[int, ...], int]] = {}


def ucry_position_map(control_count: int) -> dict[tuple[int, ...], int]:
    if control_count not in _UCRY_POSITION_CACHE:
        _UCRY_POSITION_CACHE[control_count] = infer_ucry_angle_position(control_count)
    return _UCRY_POSITION_CACHE[control_count]


def prefix_bits(prefix_int: int, level: int) -> tuple[int, ...]:
    return tuple((prefix_int >> (level - 1 - bit)) & 1 for bit in range(level))


def choose_minimal_discriminating_controls(
    prefix_angles: list[tuple[tuple[int, ...], float]],
    *,
    tol: float,
) -> tuple[int, ...]:
    """Choose a smallest previous-bit subset that preserves active angles.

    The multiplexor only has to be correct on active prefixes.  If two active
    prefixes have the same projection onto the chosen subset, they must require
    the same rotation angle.
    """

    if not prefix_angles:
        return ()

    level = len(prefix_angles[0][0])
    for subset_size in range(level + 1):
        for subset in itertools.combinations(range(level), subset_size):
            projected: dict[tuple[int, ...], float] = {}
            ok = True
            for bits, angle in prefix_angles:
                key = tuple(bits[index] for index in subset)
                previous = projected.get(key)
                if previous is None:
                    projected[key] = angle
                elif abs(previous - angle) > tol:
                    ok = False
                    break
            if ok:
                return tuple(int(index) for index in subset)
    return tuple(range(level))


def pruned_gray_ucr_prepare_from_weights(
    weights: np.ndarray,
    *,
    tol: float = TOL,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Prepare a sparse real nonnegative state with pruned UCRY layers.

    At level ``l`` the exact binary-tree state-preparation angle is known for
    each active prefix.  Instead of using all ``l`` previous bits as controls,
    choose the smallest subset that separates active prefixes with different
    angles; inactive prefixes have zero amplitude, so their angle is irrelevant.
    Qiskit's ``UCRYGate`` supplies the Gray-code ``Ry``/``CX`` multiplexor.
    """

    weights = np.asarray(weights, dtype=float)
    n = int(round(math.log2(len(weights))))
    if (1 << n) != len(weights):
        raise ValueError("weights length must be a power of two")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("weights must contain positive mass")

    probs = weights / total
    qc = QuantumCircuit(n, name="pruned_gray_ucr_prepare")

    layer_stats: list[dict[str, Any]] = []
    for level in range(n):
        target = n - level - 1
        suffix_size = 1 << (n - level)
        child_size = suffix_size >> 1
        active_angles: list[tuple[tuple[int, ...], float]] = []
        active_prefixes = 0
        nonzero_angle_prefixes = 0

        for prefix in range(1 << level):
            start = prefix * suffix_size
            middle = start + child_size
            end = start + suffix_size
            p0 = float(np.sum(probs[start:middle]))
            p1 = float(np.sum(probs[middle:end]))
            mass = p0 + p1
            if mass <= tol:
                continue
            active_prefixes += 1
            ratio = min(max(p0 / mass, 0.0), 1.0)
            angle = 2.0 * math.acos(math.sqrt(ratio))
            if abs(angle) > tol:
                nonzero_angle_prefixes += 1
            active_angles.append((prefix_bits(prefix, level), angle))

        if not active_angles:
            layer_stats.append(
                {
                    "level": int(level),
                    "target_qubit": int(target),
                    "active_prefixes": 0,
                    "status": "empty",
                }
            )
            continue

        subset = choose_minimal_discriminating_controls(active_angles, tol=1e-12)
        control_qubits = [n - index - 1 for index in subset]
        control_count = len(control_qubits)
        angle_list = [0.0 for _ in range(1 << control_count)]

        if control_count == 0:
            angle = active_angles[0][1]
            if abs(angle) > tol:
                qc.ry(angle, target)
            assigned_nonzero = int(abs(angle) > tol)
            layer_stats.append(
                {
                    "level": int(level),
                    "target_qubit": int(target),
                    "active_prefixes": int(active_prefixes),
                    "nonzero_angle_prefixes": int(nonzero_angle_prefixes),
                    "selected_prefix_bit_indices": [],
                    "selected_control_qubits": [],
                    "selected_control_count": 0,
                    "ucr_angles": 1,
                    "assigned_nonzero_angles": assigned_nonzero,
                    "status": "ry" if assigned_nonzero else "skip_zero",
                }
            )
            continue

        position_map = ucry_position_map(control_count)
        assigned: dict[tuple[int, ...], float] = {}
        for bits, angle in active_angles:
            key = tuple(bits[index] for index in subset)
            assigned[key] = angle
        for key, angle in assigned.items():
            position = position_map[key]
            angle_list[position] = float(angle)

        assigned_nonzero = sum(1 for angle in angle_list if abs(angle) > tol)
        if assigned_nonzero:
            qc.append(UCRYGate(angle_list), [target] + control_qubits)
            status = "ucry"
        else:
            status = "skip_zero"

        layer_stats.append(
            {
                "level": int(level),
                "target_qubit": int(target),
                "active_prefixes": int(active_prefixes),
                "nonzero_angle_prefixes": int(nonzero_angle_prefixes),
                "selected_prefix_bit_indices": [int(index) for index in subset],
                "selected_control_qubits": [int(qubit) for qubit in control_qubits],
                "selected_control_count": int(control_count),
                "ucr_angles": int(1 << control_count),
                "assigned_nonzero_angles": int(assigned_nonzero),
                "status": status,
            }
        )

    stats = {
        "layers": layer_stats,
        "active_layers": int(sum(1 for layer in layer_stats if layer.get("status") in {"ry", "ucry"})),
        "ucr_layers": int(sum(1 for layer in layer_stats if layer.get("status") == "ucry")),
        "single_ry_layers": int(sum(1 for layer in layer_stats if layer.get("status") == "ry")),
        "skipped_layers": int(sum(1 for layer in layer_stats if str(layer.get("status")).startswith("skip"))),
        "sum_ucr_angle_slots": int(sum(int(layer.get("ucr_angles", 0)) for layer in layer_stats)),
        "sum_assigned_nonzero_angles": int(
            sum(int(layer.get("assigned_nonzero_angles", 0)) for layer in layer_stats)
        ),
        "max_selected_control_count": int(
            max((int(layer.get("selected_control_count", 0)) for layer in layer_stats), default=0)
        ),
    }
    return qc, stats


def prepare_pair(qc: QuantumCircuit, name: str) -> QuantumCircuit:
    pair = QuantumCircuit(qc.num_qubits, name=name)
    pair.compose(qc, qubits=range(qc.num_qubits), inplace=True)
    pair.compose(qc.inverse(), qubits=range(qc.num_qubits), inplace=True)
    return pair


def safe_write_qasm2(qc: QuantumCircuit, path: Path) -> dict[str, Any]:
    status, result, elapsed, error = timed(lambda: write_qasm2(qc, path))
    return {
        "status": status,
        "elapsed_seconds": elapsed,
        "result": result,
        "error": error,
    }


def recount_bases(qc: QuantumCircuit, out_dir: Path, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for basis in ("nam", "ibm"):
        status, result, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=out_dir / f"{prefix}_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        rows[basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": result,
            "error": error,
        }
    return rows


def summarize_basis(entry: dict[str, Any]) -> dict[str, Any]:
    if entry["status"] != "ok" or entry["result"] is None:
        return {
            "status": entry["status"],
            "gate_count": None,
            "non_clifford": None,
            "elapsed_seconds": entry["elapsed_seconds"],
            "error": entry["error"],
        }
    stats = entry["result"]["stats"]
    return {
        "status": "ok",
        "gate_count": int(stats["metric_total"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "elapsed_seconds": float(entry["elapsed_seconds"]),
        "transpile_time_s": float(entry["result"]["transpile_time_s"]),
        "ops": entry["result"]["qiskit_metrics"]["ops"],
        "qasm": entry["result"]["qasm"],
    }


def circuit_row(
    qc: QuantumCircuit,
    *,
    name: str,
    variant: str,
    kind: str,
    target: Statevector,
    out_dir: Path,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prefix = f"tfim8_a0_auto9_{name}_{kind}"
    decomposed = qc.decompose(reps=10)
    fidelity = float(abs(np.vdot(target.data, Statevector.from_instruction(qc).data)) ** 2)
    qasm_info = safe_write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    decomp_qasm_info = safe_write_qasm2(decomposed, out_dir / f"{prefix}_decomposed_logical.qasm")
    basis_recounts = recount_bases(qc, out_dir, prefix)
    return {
        "name": name,
        "variant": variant,
        "kind": kind,
        "width": int(qc.num_qubits),
        "state_fidelity_to_dense_target": fidelity,
        "logical_metrics": qiskit_metrics(qc),
        "decomposed_logical_metrics": qiskit_metrics(decomposed),
        "stats": stats,
        "qasm2": qasm_info,
        "decomposed_qasm2": decomp_qasm_info,
        "basis_recounts": basis_recounts,
        "basis_summary": {basis: summarize_basis(entry) for basis, entry in basis_recounts.items()},
    }


def extract_auto_coefficients() -> tuple[np.ndarray, Any, dict[str, Any], float]:
    started = time.perf_counter()
    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    ms = kraus_ops[0]
    be = BlockEncoding(copy.deepcopy(ms))
    coeff_mode_dict, ctrl_size = be.find_optimal_order_matrices()
    coeff_list_ordered = np.zeros(1 << ctrl_size, dtype=complex)
    for ctrl_value, coeff in coeff_mode_dict.items():
        coeff_list_ordered[int(ctrl_value, 2)] = coeff
    elapsed = time.perf_counter() - started
    metadata = {
        "benchmark": "TFIM_8",
        "kraus_index": 0,
        "model_info": model_info,
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "auto_ctrl_size": int(ctrl_size),
        "raw_ctrl_size": int(be.ctrl_size),
        "selected_k": int(getattr(be, "selected_k", ctrl_size)),
        "selected_crit": str(getattr(be, "selected_crit", "")),
        "placement_search_debug": getattr(be, "placement_search_debug", {}),
        "nonzero_coeff_slots": int(np.count_nonzero(np.abs(coeff_list_ordered) > TOL)),
        "coeff_slots": int(len(coeff_list_ordered)),
        "nonzero_coefficients": [
            {
                "index": int(index),
                "ctrl": format(int(index), f"0{ctrl_size}b"),
                "abs_coeff": float(abs(coeff_list_ordered[index])),
                "coeff": repr(coeff_list_ordered[index]),
            }
            for index in np.nonzero(np.abs(coeff_list_ordered) > TOL)[0]
        ],
    }
    return coeff_list_ordered, ms, metadata, elapsed


def write_report(path: Path, summary: dict[str, Any]) -> None:
    meta = summary["metadata"]
    lines = [
        "# TFIM_8 A0 Auto-9 Pruned UCR PREPARE",
        "",
        "This experiment counts only the coefficient PREPARE state for the automatic matrix-order placement. SELECT is not synthesized or counted.",
        "",
        f"- auto ctrl width: `{meta['auto_ctrl_size']}`",
        f"- coefficient support: `{meta['nonzero_coeff_slots']}/{meta['coeff_slots']}`",
        f"- selected criterion: `{meta['selected_crit']}`",
        "",
        "## Counts",
        "",
        "| variant | circuit | width | fidelity | NAM gates | NAM non-Clifford | IBM gates | IBM non-Clifford | NAM time (s) | IBM time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        nam = row["basis_summary"]["nam"]
        ibm = row["basis_summary"]["ibm"]
        lines.append(
            f"| {row['variant']} | {row['kind']} | {row['width']} | "
            f"{row['state_fidelity_to_dense_target']:.12g} | "
            f"{nam['gate_count']} | {nam['non_clifford']} | "
            f"{ibm['gate_count']} | {ibm['non_clifford']} | "
            f"{nam['elapsed_seconds']:.6f} | {ibm['elapsed_seconds']:.6f} |"
        )

    pruned = summary["pruned_stats"]
    lines.extend(
        [
            "",
            "## Pruned UCR Layers",
            "",
            f"- active layers: `{pruned['active_layers']}`",
            f"- UCR layers: `{pruned['ucr_layers']}`",
            f"- single-Ry layers: `{pruned['single_ry_layers']}`",
            f"- skipped layers: `{pruned['skipped_layers']}`",
            f"- max selected controls: `{pruned['max_selected_control_count']}`",
            f"- total UCR angle slots: `{pruned['sum_ucr_angle_slots']}`",
            f"- nonzero assigned angles: `{pruned['sum_assigned_nonzero_angles']}`",
            "",
            "| level | target | active prefixes | nonzero prefixes | selected controls | angle slots | nonzero slots | status |",
            "| ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for layer in pruned["layers"]:
        lines.append(
            f"| {layer.get('level')} | {layer.get('target_qubit')} | "
            f"{layer.get('active_prefixes')} | {layer.get('nonzero_angle_prefixes', 0)} | "
            f"`{layer.get('selected_control_qubits', [])}` | "
            f"{layer.get('ucr_angles', 0)} | {layer.get('assigned_nonzero_angles', 0)} | "
            f"{layer.get('status')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = RESULT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    coeffs, _ms, metadata, coeff_elapsed = extract_auto_coefficients()
    weights = np.asarray([abs(value) for value in coeffs], dtype=float)
    target = target_state_from_weights(weights)

    dense_started = time.perf_counter()
    dense = dense_prepare_from_weights(weights)
    dense_build = time.perf_counter() - dense_started

    pruned_started = time.perf_counter()
    pruned, pruned_stats = pruned_gray_ucr_prepare_from_weights(weights)
    pruned_build = time.perf_counter() - pruned_started

    rows = []
    rows.append(
        circuit_row(
            dense,
            name="dense_qiskit",
            variant="dense_qiskit",
            kind="single_prep",
            target=target,
            out_dir=out_dir,
            stats={"build_seconds": dense_build},
        )
    )
    rows.append(
        circuit_row(
            prepare_pair(dense, "dense_qiskit_prepare_pair"),
            name="dense_qiskit",
            variant="dense_qiskit",
            kind="prep_pair",
            target=Statevector.from_label("0" * dense.num_qubits),
            out_dir=out_dir,
            stats={"build_seconds": dense_build},
        )
    )
    rows.append(
        circuit_row(
            pruned,
            name="pruned_gray_ucr",
            variant="pruned_gray_ucr",
            kind="single_prep",
            target=target,
            out_dir=out_dir,
            stats={"build_seconds": pruned_build, **pruned_stats},
        )
    )
    rows.append(
        circuit_row(
            prepare_pair(pruned, "pruned_gray_ucr_prepare_pair"),
            name="pruned_gray_ucr",
            variant="pruned_gray_ucr",
            kind="prep_pair",
            target=Statevector.from_label("0" * pruned.num_qubits),
            out_dir=out_dir,
            stats={"build_seconds": pruned_build, **pruned_stats},
        )
    )

    summary = {
        "script": str(Path(__file__).resolve()),
        "result_dir": str(out_dir),
        "metadata": metadata,
        "coefficient_extraction_seconds": coeff_elapsed,
        "pruned_stats": pruned_stats,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(out_dir / "report.md", summary)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
