#!/usr/bin/env python3
"""Diagnose TFIM_8 first-Kraus SELECT circuits.

Outputs the selected Kraus Pauli-sum, the raw ``mulplex_U`` SELECT circuit, and
the matrix-order ``mulplex_U_opt_order`` placement metadata from
``src/block_encoding.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding import BlockEncoding  # type: ignore  # noqa: E402
from block_encoding import (  # type: ignore  # noqa: E402
    assign_additional_modes,
    assign_subspace_modes,
    build_vec_phase_lookup,
    extract_basis_modes_with_phases,
    greedy_generator_selection,
    mobius_invert_modes_with_phases,
    sort_symplectic_matrix,
)
import numpy as np  # noqa: E402
from qiskit.quantum_info import Pauli, PauliList  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from probe_single_kraus_select_mcg_tools import select_kraus_operator  # type: ignore  # noqa: E402


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_single_kraus_select_diagnostic_v1"


class ForcedWBlockEncoding(BlockEncoding):
    """BlockEncoding variant that forces matrix-order control width."""

    def __init__(self, *args: Any, forced_w: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.forced_w = int(forced_w)

    def find_optimal_order_matrices(self):  # type: ignore[override]
        pauli_list = []
        phase_list = []
        zero_phase = 0.0
        for ms in self.mat_list:
            if isinstance(ms, tuple) or isinstance(ms, list):
                pauli_op, coeff = ms
                if pauli_op != "I" * len(pauli_op):
                    pauli_list.append(pauli_op)
                    phase_list.append(coeff)
                else:
                    zero_phase = 2 * np.angle(coeff) / np.pi

        pauli_list_obj = PauliList(pauli_list)
        x_part, z_part = pauli_list_obj.x, pauli_list_obj.z
        symplectic_matrix = np.hstack((x_part, z_part)).astype(int)
        phase_array = np.array([2 * np.angle(phase) / np.pi for phase in phase_list])

        sorted_matrix_z, phases_z = sort_symplectic_matrix(symplectic_matrix, phase_array, crit="z")
        sorted_matrix_x, phases_x = sort_symplectic_matrix(symplectic_matrix, phase_array, crit="x")

        w = int(self.forced_w)
        if w < int(np.ceil(np.log2(len(self.coeff_list)))):
            raise ValueError("forced w is too small to address all coefficients")

        selected_indices_z, maxcoverage_z, u_covered_z = greedy_generator_selection(sorted_matrix_x, w, w)
        selected_indices_x, maxcoverage_x, u_covered_x = greedy_generator_selection(sorted_matrix_z, w, w)

        if maxcoverage_z > maxcoverage_x:
            chosen_matrix = sorted_matrix_x
            chosen_phases = phases_x
            selected_matrix = sorted_matrix_x[selected_indices_z]
            remained_indices = set(range(sorted_matrix_x.shape[0])) - set(selected_indices_z)
            u_covered = u_covered_z
            remaining_tuple = [
                sorted_matrix_x[i]
                for i in list(remained_indices)
                if sorted_matrix_x[i].tobytes() not in u_covered
            ]
            chosen_criterion = "x_sorted_z_selection"
            selected_indices = selected_indices_z
            maxcoverage = maxcoverage_z
        else:
            chosen_matrix = sorted_matrix_z
            chosen_phases = phases_z
            selected_matrix = sorted_matrix_z[selected_indices_x]
            remained_indices = set(range(sorted_matrix_z.shape[0])) - set(selected_indices_x)
            u_covered = u_covered_x
            remaining_tuple = [
                sorted_matrix_z[i]
                for i in list(remained_indices)
                if sorted_matrix_z[i].tobytes() not in u_covered
            ]
            chosen_criterion = "z_sorted_x_selection"
            selected_indices = selected_indices_x
            maxcoverage = maxcoverage_x

        subspace_modes = assign_subspace_modes(w, selected_matrix, set(u_covered))
        additional_modes = assign_additional_modes(w, subspace_modes, remaining_tuple)

        identity_label = None
        for ms in self.mat_list:
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            if pauli_op == "I" * len(pauli_op):
                identity_label = pauli_op
                break

        if identity_label is not None:
            zero_ctrl = "0" * w
            used_ctrl_ids = {ctrl for _, ctrl in additional_modes}
            zero_idx = next((i for i, (_, ctrl) in enumerate(additional_modes) if ctrl == zero_ctrl), None)
            if zero_idx is None:
                additional_modes.append((identity_label, zero_ctrl))
            else:
                old_label, _ = additional_modes[zero_idx]
                if old_label != identity_label:
                    displaced_mode = additional_modes[zero_idx]
                    new_ctrl = None
                    for ctrl_int in range(2**w):
                        candidate = bin(ctrl_int)[2:].zfill(w)
                        if candidate not in used_ctrl_ids:
                            new_ctrl = candidate
                            break
                    additional_modes[zero_idx] = (identity_label, zero_ctrl)
                    if new_ctrl is not None:
                        additional_modes.append((displaced_mode[0], new_ctrl))

        additional_modes = sorted(additional_modes, key=lambda x: int(x[1], 2))
        self.additional_modes = additional_modes

        coeff_pool_by_label = {}
        for idx, ms in enumerate(self.mat_list):
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            coeff = self.coeff_list[idx]
            coeff_pool_by_label.setdefault(pauli_op, []).append(coeff)

        coeff_mode_dict = {}
        for pauli_label, ctrl_value in additional_modes:
            coeff_pool = coeff_pool_by_label.get(pauli_label, [])
            if len(coeff_pool) > 0:
                coeff_mode = coeff_pool.pop(0)
            else:
                coeff_mode = 0.0 + 0.0j
            coeff_mode_dict[ctrl_value] = coeff_mode

        self.coeff_mode_dict = coeff_mode_dict
        self.vec_phase_lookup = build_vec_phase_lookup(chosen_matrix, chosen_phases)
        self.basis_modes_with_phase, self.nonbasis_modes_with_phase = extract_basis_modes_with_phases(
            additional_modes,
            self.vec_phase_lookup,
            zero_phase,
        )
        self.mobius_phase_result = mobius_invert_modes_with_phases(
            w,
            additional_modes,
            self.vec_phase_lookup,
            zero_phase,
        )
        self.forced_w_selection_metadata = {
            "forced_w": w,
            "chosen_criterion": chosen_criterion,
            "selected_indices": [int(i) for i in selected_indices],
            "selected_indices_count": int(len(selected_indices)),
            "maxcoverage": int(maxcoverage),
            "covered_count": int(len(u_covered)),
            "remaining_tuple_count": int(len(remaining_tuple)),
            "additional_modes_count": int(len(additional_modes)),
        }

        return coeff_mode_dict, w


def scalar_to_text(value: Any) -> str:
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    return str(value)


def pauli_sum_terms(be: BlockEncoding) -> list[dict[str, Any]]:
    rows = []
    for index, ((instance, coeff), mat_item) in enumerate(zip(be.J.instances, be.mat_list)):
        if isinstance(mat_item, tuple) or isinstance(mat_item, list):
            pauli_label, phase = mat_item
            effective_coeff = coeff * phase
        else:
            pauli_label = repr(instance)
            phase = None
            effective_coeff = coeff
        rows.append(
            {
                "term_index": int(index),
                "ctrl_state": bin(index)[2:].zfill(be.ctrl_size),
                "pauli": str(pauli_label),
                "matrixsum_coeff": scalar_to_text(coeff),
                "pauli_phase": None if phase is None else scalar_to_text(phase),
                "effective_coeff": scalar_to_text(effective_coeff),
                "instance_repr": repr(instance),
            }
        )
    return rows


def pauli_sum_expression(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        coeff = row["matrixsum_coeff"]
        phase = row["pauli_phase"]
        pauli = row["pauli"]
        if phase in (None, "1", "1.0", "(1+0j)"):
            parts.append(f"({coeff}) {pauli}")
        else:
            parts.append(f"(({coeff})*({phase})) {pauli}")
    return " + ".join(parts)


def raw_address_table(be: BlockEncoding) -> list[dict[str, Any]]:
    return [
        {
            "term_index": row["term_index"],
            "ctrl_state": row["ctrl_state"],
            "pauli": row["pauli"],
            "effective_coeff": row["effective_coeff"],
        }
        for row in pauli_sum_terms(be)
    ]


def matrix_order_metadata(be: BlockEncoding) -> dict[str, Any]:
    mobius = getattr(be, "mobius_phase_result", {})
    metadata = {
        "w": int(len(next(iter(getattr(be, "coeff_mode_dict", {"": 0}).keys()), ""))),
        "ctrl_size_returned": int(len(next(iter(getattr(be, "coeff_mode_dict", {"": 0}).keys()), ""))),
        "sys_size": int(be.sys_size),
        "additional_modes": [
            {"pauli": str(pauli), "ctrl_value": str(ctrl)}
            for pauli, ctrl in getattr(be, "additional_modes", [])
        ],
        "coeff_mode_dict": {
            str(ctrl): scalar_to_text(coeff)
            for ctrl, coeff in getattr(be, "coeff_mode_dict", {}).items()
        },
        "coeff_list_ordered": [scalar_to_text(value) for value in getattr(be, "coeff_list_ordered", [])],
        "basis_modes_with_phase": [
            {"pauli": str(pauli), "ctrl_value": str(ctrl), "phase_pi_over_2": scalar_to_text(phase)}
            for pauli, ctrl, phase in getattr(be, "basis_modes_with_phase", [])
        ],
        "nonbasis_modes_with_phase": [
            {"pauli": str(pauli), "ctrl_value": str(ctrl), "phase_pi_over_2": scalar_to_text(phase)}
            for pauli, ctrl, phase in getattr(be, "nonbasis_modes_with_phase", [])
        ],
        "g_modes_with_phase": [
            {"pauli": str(pauli), "ctrl_value": str(ctrl), "phase_pi_over_2": scalar_to_text(phase)}
            for pauli, ctrl, phase in mobius.get("g_modes_with_phase", [])
        ],
        "selection_metadata": getattr(be, "forced_w_selection_metadata", None),
        "p_phase_table": [scalar_to_text(value) for value in mobius.get("p_phase_table", [])],
        "g_phase_table": [scalar_to_text(value) for value in mobius.get("g_phase_table", [])],
        "phi_ad_table": [scalar_to_text(value) for value in mobius.get("phi_ad_table", [])],
    }
    return metadata


def markdown_table(headers: list[str], rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines)


def write_text_circuit(qc: Any, path: Path) -> None:
    path.write_text(str(qc.draw(output="text", fold=-1)), encoding="utf-8")


def recount(qc: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
    out = {}
    for basis in ("nam", "ibm"):
        status, result, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=out_dir / f"{prefix}_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        out[basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": result,
            "error": error,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--forced-w",
        type=int,
        default=None,
        help="Force matrix-order mulplex_U_opt_order control width. Default uses block_encoding.py behavior.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    selected = select_kraus_operator(kraus_ops)

    raw_be = BlockEncoding(copy.deepcopy(selected.operator))
    terms = pauli_sum_terms(raw_be)
    raw_select, raw_t, raw_mc, raw_cx = raw_be.mulplex_U(raw_be.mat_list, raw_be.ctrl_size, raw_be.sys_size)

    matrix_be_cls = BlockEncoding if args.forced_w is None else ForcedWBlockEncoding
    if args.forced_w is None:
        matrix_be = matrix_be_cls(copy.deepcopy(selected.operator))
    else:
        matrix_be = matrix_be_cls(copy.deepcopy(selected.operator), forced_w=args.forced_w)
    matrix_select, mo_t, mo_mc, mo_cx, mo_w = matrix_be.mulplex_U_opt_order()

    pauli_expr = pauli_sum_expression(terms)
    (out_dir / "tfim8_kraus0_pauli_sum.txt").write_text(pauli_expr + "\n", encoding="utf-8")
    (out_dir / "tfim8_mulplex_U_raw_select.txt").write_text(
        markdown_table(
            ["term_index", "ctrl_state", "pauli", "effective_coeff"],
            raw_address_table(raw_be),
            ["term_index", "ctrl_state", "pauli", "effective_coeff"],
        )
        + "\n",
        encoding="utf-8",
    )
    write_text_circuit(raw_select, out_dir / "tfim8_mulplex_U_raw_select_diagram.txt")
    write_text_circuit(matrix_select, out_dir / "tfim8_mulplex_U_opt_order_diagram.txt")

    raw_qasm = write_qasm2(raw_select, out_dir / "tfim8_mulplex_U_raw_select.qasm")
    matrix_qasm = write_qasm2(matrix_select, out_dir / "tfim8_mulplex_U_opt_order.qasm")
    raw_recounts = recount(raw_select, out_dir, "tfim8_mulplex_U_raw_select")
    matrix_recounts = recount(matrix_select, out_dir, "tfim8_mulplex_U_opt_order")

    placement = matrix_order_metadata(matrix_be)
    placement.update(
        {
            "w_returned_by_mulplex_U_opt_order": int(mo_w),
            "raw_ctrl_size": int(raw_be.ctrl_size),
            "raw_width": int(raw_select.num_qubits),
            "matrix_order_width": int(matrix_select.num_qubits),
            "extra_ancilla_vs_raw_select": int(matrix_select.num_qubits - raw_select.num_qubits),
            "matrix_order_estimated_tcount": int(mo_t),
            "matrix_order_estimated_mccount": int(mo_mc),
            "matrix_order_estimated_cxcount": int(mo_cx),
        }
    )
    placement_prefix = "tfim8_mulplex_U_opt_order"
    if args.forced_w is not None:
        placement_prefix = f"tfim8_mulplex_U_opt_order_forced_w{args.forced_w}"

    (out_dir / f"{placement_prefix}_placement.json").write_text(
        json.dumps(placement, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{placement_prefix}_placement.md").write_text(
        "\n\n".join(
            [
                "# TFIM_8 mulplex_U_opt_order Placement",
                f"- `w`: `{mo_w}`",
                f"- forced w requested: `{args.forced_w}`",
                f"- raw SELECT width: `{raw_select.num_qubits}`",
                f"- matrix-order SELECT width: `{matrix_select.num_qubits}`",
                f"- extra ancilla vs raw SELECT: `{matrix_select.num_qubits - raw_select.num_qubits}`",
                f"- selection metadata: `{placement.get('selection_metadata')}`",
                "## additional_modes",
                markdown_table(
                    ["ctrl_value", "pauli"],
                    placement["additional_modes"],
                    ["ctrl_value", "pauli"],
                ),
                "## g_modes_with_phase",
                markdown_table(
                    ["ctrl_value", "pauli", "phase_pi_over_2"],
                    placement["g_modes_with_phase"],
                    ["ctrl_value", "pauli", "phase_pi_over_2"],
                ),
                "## coeff_mode_dict",
                markdown_table(
                    ["ctrl_value", "coeff"],
                    [
                        {"ctrl_value": ctrl, "coeff": coeff}
                        for ctrl, coeff in placement["coeff_mode_dict"].items()
                    ],
                    ["ctrl_value", "coeff"],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "benchmark": "TFIM_8",
        "model_info": model_info,
        "selected_kraus_index": int(selected.index),
        "selected_kraus_pauli_terms": int(len(selected.operator.instances)),
        "selected_kraus_pauli_norm": float(selected.operator.pauli_norm()),
        "raw_mulplex_U": {
            "ctrl_size": int(raw_be.ctrl_size),
            "sys_size": int(raw_be.sys_size),
            "estimated_tcount": int(raw_t),
            "estimated_mccount": int(raw_mc),
            "estimated_cxcount": int(raw_cx),
            "qiskit_metrics": qiskit_metrics(raw_select),
            "qasm": raw_qasm,
            "basis_recounts": raw_recounts,
        },
        "matrix_order_mulplex_U_opt_order": {
            "w": int(mo_w),
            "estimated_tcount": int(mo_t),
            "estimated_mccount": int(mo_mc),
            "estimated_cxcount": int(mo_cx),
            "qiskit_metrics": qiskit_metrics(matrix_select),
            "qasm": matrix_qasm,
            "basis_recounts": matrix_recounts,
            "forced_w_requested": args.forced_w,
            "selection_metadata": placement.get("selection_metadata"),
            "placement_json": str(out_dir / f"{placement_prefix}_placement.json"),
            "placement_md": str(out_dir / f"{placement_prefix}_placement.md"),
        },
        "artifacts": {
            "pauli_sum": str(out_dir / "tfim8_kraus0_pauli_sum.txt"),
            "raw_select_table": str(out_dir / "tfim8_mulplex_U_raw_select.txt"),
            "raw_select_diagram": str(out_dir / "tfim8_mulplex_U_raw_select_diagram.txt"),
            "matrix_order_diagram": str(out_dir / "tfim8_mulplex_U_opt_order_diagram.txt"),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    report = [
        "# TFIM_8 Single-Kraus SELECT Diagnostic",
        "",
        f"- Kraus index: `{selected.index}`",
        f"- Pauli terms: `{len(selected.operator.instances)}`",
        f"- Pauli norm: `{float(selected.operator.pauli_norm())}`",
        f"- raw `mulplex_U` width: `{raw_select.num_qubits}`; ctrl_size: `{raw_be.ctrl_size}`; sys_size: `{raw_be.sys_size}`",
        f"- matrix-order `mulplex_U_opt_order` width: `{matrix_select.num_qubits}`; w: `{mo_w}`",
        f"- forced w requested: `{args.forced_w}`",
        f"- extra ancilla vs raw SELECT: `{matrix_select.num_qubits - raw_select.num_qubits}`",
        "",
        "## Pauli-Sum",
        "",
        pauli_expr,
        "",
        "## Raw Address Placement",
        "",
        markdown_table(
            ["term_index", "ctrl_state", "pauli", "effective_coeff"],
            raw_address_table(raw_be),
            ["term_index", "ctrl_state", "pauli", "effective_coeff"],
        ),
        "",
        "## Matrix-Order Placement",
        "",
        f"See `{out_dir / f'{placement_prefix}_placement.md'}` and `{out_dir / f'{placement_prefix}_placement.json'}`.",
        "",
        "## Circuit Artifacts",
        "",
        f"- raw QASM: `{out_dir / 'tfim8_mulplex_U_raw_select.qasm'}`",
        f"- raw NAM opt0: `{out_dir / 'tfim8_mulplex_U_raw_select_nam_opt0.qasm'}`",
        f"- raw text diagram: `{out_dir / 'tfim8_mulplex_U_raw_select_diagram.txt'}`",
        f"- matrix-order QASM: `{out_dir / 'tfim8_mulplex_U_opt_order.qasm'}`",
        f"- matrix-order NAM opt0: `{out_dir / 'tfim8_mulplex_U_opt_order_nam_opt0.qasm'}`",
        f"- summary: `{out_dir / 'summary.json'}`",
        "",
    ]
    (out_dir / "tfim8_single_kraus_select_diagnostic_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"report: {out_dir / 'tfim8_single_kraus_select_diagnostic_report.md'}")


if __name__ == "__main__":
    main()
