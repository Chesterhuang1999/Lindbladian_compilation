#!/usr/bin/env python3
"""TFIM-4 A0 SELECT-only Mode A qclib McxVchainDirty multiplexor.

Mode A follows ``mode_a_mcxvchaindirty_multiplexor.md``:

* one row per Pauli string address;
* basis sidecars turn X/Y/Z targets into X-like targets;
* qclib ``McxVchainDirty(..., num_target_qubit=w_j).definition`` toggles all
  non-identity targets for that row;
* row phases are implemented as qclib ``Mcg(PhaseGate(theta))`` sidecars.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import PhaseGate
from qiskit.quantum_info import Operator, SparsePauliOp


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
    sys.path.append(str(QCLIB_SITE))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding, expand_large_controlled_gates  # type: ignore  # noqa: E402
from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    qiskit_metrics,
    recount_qiskit_circuit,
    write_qasm2,
)
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore  # noqa: E402


OUT_DIR = ROOT / "Baseline_results" / "tfim4_a0_select_qclib_mcxvchain_mode_a_v1"


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    started = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - started, None
    except Exception:
        return "error", None, time.perf_counter() - started, traceback.format_exc()


def complex_cell(value: complex, digits: int = 10) -> str:
    z = complex(value)
    real = 0.0 if abs(z.real) < 1e-12 else z.real
    imag = 0.0 if abs(z.imag) < 1e-12 else z.imag
    if abs(imag) < 1e-12:
        return f"{real:.{digits}g}"
    if abs(real) < 1e-12:
        return f"{imag:.{digits}g}j"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.{digits}g}{sign}{abs(imag):.{digits}g}j"


def complex_record(value: complex, tol: float = 1e-12) -> dict[str, Any]:
    z = complex(value)
    real = 0.0 if abs(z.real) < tol else float(z.real)
    imag = 0.0 if abs(z.imag) < tol else float(z.imag)
    return {
        "real": real,
        "imag": imag,
        "repr": repr(complex(real, imag)),
        "angle_pi": float(np.angle(z) / np.pi),
    }


def coeff_dict_from_matrixsum(ms: Matrixsum) -> dict[str, complex]:
    out: dict[str, complex] = defaultdict(complex)
    for atom, coeff in ms.instances:
        if not isinstance(atom, PauliAtom):
            raise TypeError("Mode A TFIM test expects PauliAtom-only Matrixsum")
        out[atom.expr] += complex(coeff) * complex(atom.phase)
    return dict(out)


def pauli_sum_rows(ms: Matrixsum) -> list[dict[str, Any]]:
    ctrl_size = 0 if len(ms.instances) <= 1 else int(math.ceil(math.log2(len(ms.instances))))
    rows = []
    for address, (atom, coeff) in enumerate(ms.instances):
        if not isinstance(atom, PauliAtom):
            raise TypeError("Mode A TFIM test expects PauliAtom-only Matrixsum")
        complex_coeff = complex(coeff) * complex(atom.phase)
        rows.append(
            {
                "address": int(address),
                "ctrl_state": format(address, f"0{ctrl_size}b"),
                "pauli": atom.expr,
                "coefficient_abs": float(coeff),
                "phase": complex_record(complex(atom.phase)),
                "complex_coefficient": complex_record(complex_coeff),
                "weight": int(sum(ch != "I" for ch in atom.expr)),
            }
        )
    return rows


def pauli_expression_from_rows(rows: list[dict[str, Any]]) -> str:
    pieces = []
    for row in rows:
        coeff = complex(row["complex_coefficient"]["real"], row["complex_coefficient"]["imag"])
        pieces.append(f"({complex_cell(coeff)}) {row['pauli']}")
    return " + ".join(pieces) if pieces else "0"


def dense_from_coeffs(coeffs: dict[str, complex]) -> np.ndarray:
    total = None
    for label, coeff in coeffs.items():
        mat = SparsePauliOp.from_list([(label, coeff)]).to_matrix()
        total = mat if total is None else total + mat
    if total is None:
        return np.zeros((1, 1), dtype=complex)
    return np.asarray(total, dtype=complex)


def canonical_tfim4_a0_coeffs(delta_t: float = 0.1) -> dict[str, complex]:
    n = 4
    gamma_sq = 0.1 / 4
    coeffs: dict[str, complex] = defaultdict(complex)
    coeffs["I" * n] += 1.0 - n * delta_t * gamma_sq
    for i in range(n):
        z_pair = "".join("Z" if j in {i, (i + 1) % n} else "I" for j in range(n))
        x_single = "".join("X" if j == i else "I" for j in range(n))
        z_single = "".join("Z" if j == i else "I" for j in range(n))
        coeffs[z_pair] += 1j * delta_t
        coeffs[x_single] += 1j * delta_t
        coeffs[z_single] += -delta_t * gamma_sq
    return dict(coeffs)


def verify_a0_global_phase(ms: Matrixsum) -> dict[str, Any]:
    actual = coeff_dict_from_matrixsum(ms)
    canonical = canonical_tfim4_a0_coeffs(0.1)
    actual_dense = dense_from_coeffs(actual)
    canonical_dense = dense_from_coeffs(canonical)
    inner = np.vdot(canonical_dense.reshape(-1), actual_dense.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
    phase_corrected_diff = float(np.linalg.norm(actual_dense - phase * canonical_dense))
    direct_diff = float(np.linalg.norm(actual_dense - canonical_dense))
    return {
        "canonical_expression": pauli_expression_from_coeffs(canonical),
        "actual_expression": pauli_expression_from_coeffs(actual),
        "phase_actual_over_canonical": repr(phase),
        "direct_dense_diff_norm": direct_diff,
        "phase_corrected_dense_diff_norm": phase_corrected_diff,
        "passed_up_to_global_phase": bool(phase_corrected_diff < 1e-10),
    }


def pauli_expression_from_coeffs(coeffs: dict[str, complex]) -> str:
    pieces = []
    for label in sorted(coeffs):
        coeff = coeffs[label]
        if abs(coeff) > 1e-12:
            pieces.append(f"({complex_cell(coeff)}) {label}")
    return " + ".join(pieces) if pieces else "0"


def address_size(num_rows: int) -> int:
    if num_rows < 1:
        raise ValueError("num_rows must be positive")
    return 0 if num_rows == 1 else int(math.ceil(math.log2(num_rows)))


def required_work_qubits(num_controls: int) -> int:
    return max(num_controls - 2, 0)


def qclib_ctrl_state_from_bits(bits_in_control_order: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in reversed(bits_in_control_order))


def ctrl_state_for_index(j: int, k: int, address_lsb_first: bool = True) -> str:
    if k == 0:
        return ""
    if not (0 <= j < (1 << k)):
        raise ValueError(f"row index {j} does not fit in {k} bits")
    if address_lsb_first:
        bits = [(j >> i) & 1 for i in range(k)]
    else:
        bits = [(j >> (k - 1 - i)) & 1 for i in range(k)]
    return qclib_ctrl_state_from_bits(bits)


def qiskit_pauli_targets(pauli: str, ctrl_size: int) -> list[dict[str, Any]]:
    sys_size = len(pauli)
    targets = []
    for label_index, letter in enumerate(pauli):
        if letter == "I":
            continue
        sys_qubit = sys_size - 1 - label_index
        targets.append(
            {
                "pauli": letter,
                "pauli_label_index": int(label_index),
                "sys_qubit": int(sys_qubit),
                "absolute_qubit": int(ctrl_size + sys_qubit),
            }
        )
    return targets


def apply_pauli_basis_sidecar(
    qc: QuantumCircuit,
    system_qubits: Sequence,
    targets: Sequence[dict[str, Any]],
    *,
    inverse: bool,
) -> None:
    for target in targets:
        letter = str(target["pauli"])
        q = system_qubits[int(target["sys_qubit"])]
        if letter == "Z":
            qc.h(q)
        elif letter == "Y":
            if inverse:
                qc.s(q)
            else:
                qc.sdg(q)
        elif letter == "X":
            pass
        else:
            raise ValueError(f"unsupported Pauli letter {letter!r}")


def ctrl_bit_for_qiskit_control_index(ctrl_state: str, control_index: int) -> str:
    return ctrl_state[len(ctrl_state) - 1 - control_index]


def append_row_phase_sidecar_qclib(
    qc: QuantumCircuit,
    controls: list,
    ctrl_state: str,
    phase: complex,
    stats: dict[str, int],
) -> None:
    from qclib.gates.mcg import Mcg

    angle = float(np.angle(phase))
    if abs(angle) < 1e-12:
        return
    if not controls:
        qc.global_phase += angle
        stats["global_phase_only"] += 1
        return

    phase_target_index = len(controls) - 1
    phase_target = controls[phase_target_index]
    target_state = ctrl_bit_for_qiskit_control_index(ctrl_state, phase_target_index)
    remaining_controls = controls[:-1]
    remaining_state = ctrl_state[1:]

    if target_state == "0":
        qc.x(phase_target)
        stats["phase_open_control_x_conjugations"] += 2

    if remaining_controls:
        qc.append(
            Mcg(PhaseGate(angle).to_matrix(), len(remaining_controls), ctrl_state=remaining_state),
            remaining_controls + [phase_target],
        )
        stats["qclib_mcg_phase_sidecars"] += 1
    else:
        qc.p(angle, phase_target)
        stats["single_control_phase_sidecars"] += 1

    if target_state == "0":
        qc.x(phase_target)


def append_mode_a_row(
    qc: QuantumCircuit,
    address_qubits: list,
    system_qubits: list,
    work_qubits: list,
    *,
    pauli: str,
    phase: complex,
    row_index: int,
    stats: dict[str, int],
) -> dict[str, Any]:
    from qclib.gates.mcx import McxVchainDirty

    k = len(address_qubits)
    a = required_work_qubits(k)
    ctrl_state = ctrl_state_for_index(row_index, k, address_lsb_first=True)
    targets = qiskit_pauli_targets(pauli, k)
    weight = len(targets)
    append_row_phase_sidecar_qclib(qc, address_qubits, ctrl_state, phase, stats)

    row_info = {
        "address": int(row_index),
        "ctrl_state": ctrl_state,
        "pauli": pauli,
        "phase": complex_record(phase),
        "weight": int(weight),
        "targets": targets,
        "basis_sidecars": [],
        "mcxvchain_definition_width": None,
    }

    if weight == 0:
        stats["identity_rows"] += 1
        return row_info

    if len(work_qubits) < a:
        raise ValueError(f"need {a} work qubits, got {len(work_qubits)}")

    for target in targets:
        letter = str(target["pauli"])
        if letter == "Z":
            row_info["basis_sidecars"].append({"sys_qubit": target["sys_qubit"], "pre": "h", "post": "h"})
            stats["basis_h_pairs"] += 1
        elif letter == "Y":
            row_info["basis_sidecars"].append({"sys_qubit": target["sys_qubit"], "pre": "sdg", "post": "s"})
            stats["basis_sdg_s_pairs"] += 1

    apply_pauli_basis_sidecar(qc, system_qubits, targets, inverse=False)
    sub = McxVchainDirty(
        num_controls=k,
        num_target_qubit=weight,
        ctrl_state=ctrl_state,
        relative_phase=False,
        action_only=False,
    ).definition
    row_info["mcxvchain_definition_width"] = int(sub.num_qubits)
    qc.compose(
        sub,
        list(address_qubits) + list(work_qubits[:a]) + [system_qubits[int(t["sys_qubit"])] for t in targets],
        inplace=True,
    )
    apply_pauli_basis_sidecar(qc, system_qubits, targets, inverse=True)

    stats["mcxvchain_rows"] += 1
    stats["total_x_like_targets"] += weight
    stats["max_row_weight"] = max(stats["max_row_weight"], weight)
    return row_info


def build_mode_a_select(ms: Matrixsum) -> tuple[QuantumCircuit, dict[str, Any], list[dict[str, Any]]]:
    rows = pauli_sum_rows(ms)
    paulis = [str(row["pauli"]) for row in rows]
    n = len(paulis)
    k = address_size(n)
    m = len(paulis[0]) if paulis else 0
    a = required_work_qubits(k)

    qc = QuantumCircuit(k + m + a, name="tfim4_a0_mcxvchain_mode_a_select")
    address_qubits = list(range(k))
    system_qubits = list(range(k, k + m))
    work_qubits = list(range(k + m, k + m + a))
    stats = {
        "num_rows": n,
        "address_qubits": k,
        "system_qubits": m,
        "work_qubits_mode_a_clean": a,
        "full_width_mode_a_clean": k + m + a,
        "identity_rows": 0,
        "mcxvchain_rows": 0,
        "qclib_mcg_phase_sidecars": 0,
        "single_control_phase_sidecars": 0,
        "global_phase_only": 0,
        "phase_open_control_x_conjugations": 0,
        "basis_h_pairs": 0,
        "basis_sdg_s_pairs": 0,
        "total_x_like_targets": 0,
        "max_row_weight": 0,
        "num_X": sum(p.count("X") for p in paulis),
        "num_Y": sum(p.count("Y") for p in paulis),
        "num_Z": sum(p.count("Z") for p in paulis),
        "num_I": sum(p.count("I") for p in paulis),
    }
    mode_a_rows = []
    for row in rows:
        phase = complex(row["phase"]["real"], row["phase"]["imag"])
        mode_a_rows.append(
            append_mode_a_row(
                qc,
                address_qubits,
                system_qubits,
                work_qubits,
                pauli=str(row["pauli"]),
                phase=phase,
                row_index=int(row["address"]),
                stats=stats,
            )
        )
    return qc, {"layout": {"address": address_qubits, "system": system_qubits, "work": work_qubits}, "stats": stats}, mode_a_rows


def build_tfim4_a0() -> tuple[Matrixsum, dict[str, Any]]:
    kraus_ops, model_info = build_periodic_tfim_channel(4, 0.1)
    ms = kraus_ops[0]
    return ms, {
        "benchmark": "TFIM_4_A0",
        "scope": "SELECT only",
        "system_qubits": int(ms.size),
        "pauli_terms": int(len(ms.instances)),
        "pauli_norm": float(ms.pauli_norm()),
        "model_info": model_info,
    }


def build_raw_select(ms: Matrixsum) -> tuple[QuantumCircuit, dict[str, Any]]:
    be = BlockEncoding(copy.deepcopy(ms))
    qc, tcount, mccount, cxcount = be.mulplex_U(be.mat_list, be.ctrl_size, be.sys_size)
    return qc, {
        "builder": "BlockEncoding.mulplex_U(mat_list, ctrl_size, sys_size)",
        "ctrl_size": int(be.ctrl_size),
        "sys_size": int(be.sys_size),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
    }


def build_matrix_order_select(ms: Matrixsum) -> tuple[QuantumCircuit, dict[str, Any]]:
    be = BlockEncoding(copy.deepcopy(ms))
    logical, tcount, mccount, cxcount, ctrl_size = be.mulplex_U_opt_order()
    expanded = expand_large_controlled_gates(logical, if_optimize=True)
    return expanded, {
        "builder": "BlockEncoding.mulplex_U_opt_order() + expand_large_controlled_gates(if_optimize=True)",
        "ctrl_size": int(ctrl_size),
        "sys_size": int(be.sys_size),
        "logical_width": int(logical.num_qubits),
        "expanded_width": int(expanded.num_qubits),
        "estimated_tcount": int(tcount),
        "estimated_mccount": int(mccount),
        "estimated_cxcount": int(cxcount),
        "selected_k": int(getattr(be, "selected_k", ctrl_size)),
        "selected_crit": str(getattr(be, "selected_crit", "")),
    }


def exact_equivalence_with_work_identity(
    raw_select: QuantumCircuit,
    mode_a_select: QuantumCircuit,
    *,
    k: int,
    m: int,
    a: int,
) -> dict[str, Any]:
    reference = QuantumCircuit(k + m + a)
    reference.compose(raw_select, qubits=list(range(k + m)), inplace=True)
    ref = Operator(reference).data
    cand = Operator(mode_a_select).data
    diff_norm = float(np.linalg.norm(ref - cand))
    inner = np.vdot(ref.reshape(-1), cand.reshape(-1))
    phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0
    phase_diff_norm = float(np.linalg.norm(ref * phase - cand))
    return {
        "method": "operator_norm_against_raw_select_tensor_identity_work",
        "strict_passed": bool(diff_norm < 1e-8),
        "global_phase_passed": bool(phase_diff_norm < 1e-8),
        "diff_norm": diff_norm,
        "phase_corrected_diff_norm": phase_diff_norm,
        "global_phase": repr(phase),
    }


def write_qasm(qc: QuantumCircuit, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)
    return {"path": str(path), "bytes": int(path.stat().st_size)}


def recount(qc: QuantumCircuit, key: str, basis_name: str, opt: int) -> dict[str, Any]:
    return recount_qiskit_circuit(
        qc,
        basis_name=basis_name,
        out_path=OUT_DIR / f"{key}_{basis_name}_opt{opt}.qasm",
        optimization_level=opt,
    )


def summarize_circuit(
    *,
    key: str,
    label: str,
    qc: QuantumCircuit,
    metadata: dict[str, Any],
    recount_specs: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    write_qasm(qc, OUT_DIR / f"{key}_source.qasm")
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "metadata": metadata,
        "source_metrics": qiskit_metrics(qc),
        "counts": {},
    }
    for basis_name, opt in recount_specs:
        status, result, elapsed, error = timed(lambda basis_name=basis_name, opt=opt: recount(qc, key, basis_name, opt))
        count_key = f"{basis_name}_opt{opt}"
        row["counts"][count_key] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
            "result": result,
        }
        if status == "ok":
            stats = result["stats"]
            row["counts"][count_key]["summary"] = {
                "basis": basis_name,
                "optimization_level": opt,
                "width": int(stats["num_qubits"]),
                "gate_count": int(stats["metric_total"]),
                "non_clifford": int(stats["clifford"]["non_clifford"]),
                "ops": stats["ops"],
                "qasm": result["qasm"],
                "transpile_time_s": float(result["transpile_time_s"]),
            }
    return row


def comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    raw_refs: dict[str, int] = {}
    for circuit in summary["circuits"]:
        if circuit["key"] == "raw_select":
            for count_key, data in circuit["counts"].items():
                item = data.get("summary")
                if item:
                    raw_refs[count_key] = int(item["gate_count"])
    for circuit in summary["circuits"]:
        for count_key, data in circuit["counts"].items():
            item = data.get("summary")
            if not item:
                continue
            gate_count = int(item["gate_count"])
            ref = raw_refs.get(count_key)
            rows.append(
                {
                    "variant": circuit["label"],
                    "key": circuit["key"],
                    "count_key": count_key,
                    "basis": item["basis"],
                    "qiskit_opt": item["optimization_level"],
                    "width": item["width"],
                    "gate_count": gate_count,
                    "non_clifford": item["non_clifford"],
                    "ratio_vs_raw_same_count": None if ref in (None, 0) else float(gate_count / ref),
                    "ops": item["ops"],
                    "qasm": item["qasm"],
                }
            )
    return rows


def write_mode_a_markdown(
    path: Path,
    *,
    pauli_rows: list[dict[str, Any]],
    mode_a_rows: list[dict[str, Any]],
    expression: str,
) -> None:
    lines = [
        "# TFIM_4 A0 Mode A McxVchainDirty Intermediates",
        "",
        "## Pauli Sum",
        "",
        "```text",
        expression,
        "```",
        "",
        "| Address | Ctrl state | Coefficient | Phase | Pauli | Weight |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for row in pauli_rows:
        coeff = complex(row["complex_coefficient"]["real"], row["complex_coefficient"]["imag"])
        phase = complex(row["phase"]["real"], row["phase"]["imag"])
        lines.append(
            f"| {row['address']} | `{row['ctrl_state']}` | `{complex_cell(coeff)}` | "
            f"`{complex_cell(phase)}` | `{row['pauli']}` | {row['weight']} |"
        )
    lines.extend(["", "## Mode A Rows", ""])
    for row in mode_a_rows:
        phase = complex(row["phase"]["real"], row["phase"]["imag"])
        lines.extend(
            [
                f"### Address {row['address']} / ctrl_state `{row['ctrl_state']}`",
                "",
                f"- Pauli: `{row['pauli']}`",
                f"- Row phase: `{complex_cell(phase)}`",
                f"- Weight: `{row['weight']}`",
                f"- Targets: `{row['targets']}`",
                f"- Basis sidecars: `{row['basis_sidecars']}`",
                f"- McxVchainDirty definition width: `{row['mcxvchain_definition_width']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TFIM_4 A0 SELECT: qclib McxVchainDirty Mode A",
        "",
        f"- A0 global-phase verification passed: `{summary['a0_global_phase_check']['passed_up_to_global_phase']}`",
        f"- A0 phase vs canonical: `{summary['a0_global_phase_check']['phase_actual_over_canonical']}`",
        f"- Mode A semantic check vs raw SELECT: `{summary['mode_a_equivalence_to_raw']['strict_passed']}`",
        f"- Mode A full width formula: `{summary['mode_a_metadata']['stats']['full_width_mode_a_clean']}`",
        "",
        "| Variant | Basis/opt | Width | Gates | Non-Clifford | Ratio vs raw |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison_rows"]:
        ratio = row["ratio_vs_raw_same_count"]
        ratio_cell = "-" if ratio is None else f"{ratio:.3f}"
        lines.append(
            f"| {row['variant']} | {row['count_key']} | {row['width']} | "
            f"{row['gate_count']} | {row['non_clifford']} | {ratio_cell} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Mode A rows: `{summary['mode_a_markdown']}`",
            f"- JSON summary: `{summary['summary_path']}`",
            f"- Output directory: `{summary['out_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ms, metadata = build_tfim4_a0()
    pauli_rows = pauli_sum_rows(ms)
    expression = pauli_expression_from_rows(pauli_rows)
    a0_check = verify_a0_global_phase(ms)

    raw_select, raw_meta = build_raw_select(ms)
    mode_a_select, mode_a_meta, mode_a_rows = build_mode_a_select(ms)
    matrix_select, matrix_meta = build_matrix_order_select(ms)

    k = mode_a_meta["stats"]["address_qubits"]
    m = mode_a_meta["stats"]["system_qubits"]
    a = mode_a_meta["stats"]["work_qubits_mode_a_clean"]
    equivalence = exact_equivalence_with_work_identity(raw_select, mode_a_select, k=k, m=m, a=a)

    mode_a_markdown_path = OUT_DIR / "pauli_sum_and_mode_a_rows.md"
    write_mode_a_markdown(
        mode_a_markdown_path,
        pauli_rows=pauli_rows,
        mode_a_rows=mode_a_rows,
        expression=expression,
    )

    circuits = [
        summarize_circuit(
            key="raw_select",
            label="Qiskit raw mulplex_U SELECT",
            qc=raw_select,
            metadata=raw_meta,
            recount_specs=(("ibm", 0), ("ibm", 3), ("nam", 0)),
        ),
        summarize_circuit(
            key="qclib_mcxvchain_mode_a",
            label="qclib McxVchainDirty Mode A SELECT",
            qc=mode_a_select,
            metadata=mode_a_meta,
            recount_specs=(("ibm", 0), ("ibm", 3), ("nam", 0)),
        ),
        summarize_circuit(
            key="ours_matrix_order_expanded",
            label="ours matrix-order SELECT expanded",
            qc=matrix_select,
            metadata=matrix_meta,
            recount_specs=(("ibm", 0), ("nam", 0)),
        ),
    ]

    summary: dict[str, Any] = {
        "experiment": "tfim4_a0_select_qclib_mcxvchain_mode_a",
        "out_dir": str(OUT_DIR.resolve()),
        "metadata": metadata,
        "pauli_sum_expression": expression,
        "pauli_sum_rows": pauli_rows,
        "a0_global_phase_check": a0_check,
        "mode_a_metadata": mode_a_meta,
        "mode_a_rows": mode_a_rows,
        "mode_a_equivalence_to_raw": equivalence,
        "mode_a_markdown": str(mode_a_markdown_path.resolve()),
        "circuits": circuits,
    }
    summary["comparison_rows"] = comparison_rows(summary)

    summary_path = OUT_DIR / "summary.json"
    report_path = OUT_DIR / "summary.md"
    summary["summary_path"] = str(summary_path.resolve())
    summary["report_path"] = str(report_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=repr) + "\n", encoding="utf-8")
    report_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": summary["summary_path"], "report": summary["report_path"]}, indent=2))
    print(markdown(summary))


if __name__ == "__main__":
    main()
