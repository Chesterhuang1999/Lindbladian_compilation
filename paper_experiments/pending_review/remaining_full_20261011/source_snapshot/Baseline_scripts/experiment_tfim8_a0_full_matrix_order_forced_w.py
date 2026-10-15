#!/usr/bin/env python3
"""Full matrix-order block-encoding experiment for TFIM_8 A0.

This script only adds an experiment harness.  It does not modify project
source.  For each requested matrix-order width ``w``, it subclasses
``BlockEncoding`` to force ``find_optimal_order_matrices()`` to use that width,
then builds the complete block-encoding circuit:

    PREPARE(coefficients) -> matrix-order SELECT -> PREPARE(coefficients)^dag

The output is lowered to NAM/IBM opt0 bases and counted with the shared
baseline helpers.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


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
from qiskit.quantum_info import PauliList  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    BenchmarkSpec,
    build_kraus_ops,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_full_matrix_order_forced_w_v1"


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

        w_min = int(np.ceil(np.log2(len(self.coeff_list))))
        w = int(self.forced_w)
        if w < w_min:
            raise ValueError(f"forced w={w} is smaller than required address width {w_min}")

        pauli_list_obj = PauliList(pauli_list)
        x_part, z_part = pauli_list_obj.x, pauli_list_obj.z
        symplectic_matrix = np.hstack((x_part, z_part)).astype(int)
        phase_array = np.array([2 * np.angle(phase) / np.pi for phase in phase_list])

        sorted_matrix_z, phases_z = sort_symplectic_matrix(symplectic_matrix, phase_array, crit="z")
        sorted_matrix_x, phases_x = sort_symplectic_matrix(symplectic_matrix, phase_array, crit="x")

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

        coeff_pool_by_label: dict[str, list[Any]] = {}
        for idx, ms in enumerate(self.mat_list):
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            coeff_pool_by_label.setdefault(pauli_op, []).append(self.coeff_list[idx])

        coeff_mode_dict = {}
        for pauli_label, ctrl_value in additional_modes:
            coeff_pool = coeff_pool_by_label.get(pauli_label, [])
            coeff_mode = coeff_pool.pop(0) if coeff_pool else 0.0 + 0.0j
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
            "forced_w": int(w),
            "required_min_w": int(w_min),
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


def select_kraus_zero(kraus_ops: list[Any]) -> Any:
    if not kraus_ops:
        raise ValueError("TFIM_8 builder returned no Kraus operators")
    if len(kraus_ops[0].instances) == 0:
        raise ValueError("TFIM_8 Kraus index 0 is empty")
    return kraus_ops[0]


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
                "raw_ctrl_state": bin(index)[2:].zfill(be.ctrl_size),
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


def placement_metadata(be: ForcedWBlockEncoding) -> dict[str, Any]:
    mobius = getattr(be, "mobius_phase_result", {})
    return {
        "forced_w": int(be.forced_w),
        "sys_size": int(be.sys_size),
        "ctrl_size_raw": int(be.ctrl_size),
        "ctrl_size_matrix_order": int(len(next(iter(getattr(be, "coeff_mode_dict", {"": 0}).keys()), ""))),
        "circuit_width": int(be.circuit_width),
        "additional_modes": [
            {"pauli": str(pauli), "ctrl_value": str(ctrl)}
            for pauli, ctrl in getattr(be, "additional_modes", [])
        ],
        "coeff_mode_dict": {
            str(ctrl): scalar_to_text(coeff)
            for ctrl, coeff in getattr(be, "coeff_mode_dict", {}).items()
        },
        "coeff_list_ordered_nonzero": [
            {"index": int(i), "ctrl_value": bin(int(i))[2:].zfill(be.forced_w), "coeff": scalar_to_text(value)}
            for i, value in enumerate(getattr(be, "coeff_list_ordered", []))
            if abs(complex(value)) > 0
        ],
        "g_modes_with_phase": [
            {"pauli": str(pauli), "ctrl_value": str(ctrl), "phase_pi_over_2": scalar_to_text(phase)}
            for pauli, ctrl, phase in mobius.get("g_modes_with_phase", [])
        ],
        "basis_modes_with_phase": [
            {"pauli": str(pauli), "ctrl_value": str(ctrl), "phase_pi_over_2": scalar_to_text(phase)}
            for pauli, ctrl, phase in getattr(be, "basis_modes_with_phase", [])
        ],
        "nonbasis_modes_with_phase": [
            {"pauli": str(pauli), "ctrl_value": str(ctrl), "phase_pi_over_2": scalar_to_text(phase)}
            for pauli, ctrl, phase in getattr(be, "nonbasis_modes_with_phase", [])
        ],
        "selection_metadata": getattr(be, "forced_w_selection_metadata", {}),
    }


def write_placement_md(path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        f"# TFIM_8 A0 Full Matrix-Order Placement, forced w={metadata['forced_w']}",
        "",
        f"- raw ctrl_size: `{metadata['ctrl_size_raw']}`",
        f"- matrix-order ctrl_size: `{metadata['ctrl_size_matrix_order']}`",
        f"- sys_size: `{metadata['sys_size']}`",
        f"- circuit_width: `{metadata['circuit_width']}`",
        "",
        "## Additional Modes",
        "",
        "| ctrl | pauli |",
        "| --- | --- |",
    ]
    for row in metadata["additional_modes"]:
        lines.append(f"| `{row['ctrl_value']}` | `{row['pauli']}` |")

    lines.extend(["", "## Nonzero Coefficient Slots", "", "| ctrl | coeff |", "| --- | --- |"])
    for row in metadata["coeff_list_ordered_nonzero"]:
        lines.append(f"| `{row['ctrl_value']}` | `{row['coeff']}` |")

    lines.extend(["", "## g_modes_with_phase", "", "| ctrl | pauli | phase*pi/2 |", "| --- | --- | --- |"])
    for row in metadata["g_modes_with_phase"]:
        lines.append(f"| `{row['ctrl_value']}` | `{row['pauli']}` | `{row['phase_pi_over_2']}` |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recount_bases(qc: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
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


def summarize_basis(recount_entry: dict[str, Any]) -> dict[str, Any] | None:
    if recount_entry.get("status") != "ok":
        return None
    result = recount_entry["result"]
    stats = result["stats"]
    return {
        "gate_count": int(stats["metric_total"]),
        "width": int(stats["num_qubits"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "unknown_clifford": int(stats["clifford"]["unknown"]),
        "transpile_time_s": float(result["transpile_time_s"]),
        "elapsed_seconds": float(recount_entry["elapsed_seconds"]),
        "ops": stats["ops"],
        "qasm": result["qasm"],
    }


def run_for_w(ms: Any, forced_w: int, out_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    be = ForcedWBlockEncoding(copy.deepcopy(ms), forced_w=forced_w)
    qc = be.circuit(opt="Matrix-order")
    build_seconds = time.perf_counter() - started

    prefix = f"tfim8_a0_full_matrix_order_w{forced_w}"
    write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")

    placement = placement_metadata(be)
    (out_dir / f"{prefix}_placement.json").write_text(
        json.dumps(placement, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_placement_md(out_dir / f"{prefix}_placement.md", placement)

    basis_recounts = recount_bases(qc, out_dir, prefix)
    basis_summary = {
        basis: summarize_basis(entry)
        for basis, entry in basis_recounts.items()
    }
    return {
        "status": "ok",
        "forced_w": int(forced_w),
        "build_seconds": float(build_seconds),
        "logical_qasm": str(out_dir / f"{prefix}_logical.qasm"),
        "qiskit_metrics_logical": qiskit_metrics(qc),
        "block_encoding_reported_width": int(be.circuit_width),
        "raw_ctrl_size": int(be.ctrl_size),
        "matrix_order_ctrl_size": int(forced_w),
        "sys_size": int(be.sys_size),
        "selection_metadata": getattr(be, "forced_w_selection_metadata", {}),
        "placement_files": {
            "json": str(out_dir / f"{prefix}_placement.json"),
            "markdown": str(out_dir / f"{prefix}_placement.md"),
        },
        "basis_recounts": basis_recounts,
        "basis_summary": basis_summary,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TFIM_8 A0 Full Matrix-Order Forced-w Experiment",
        "",
        "Circuit scope: complete `BlockEncoding.circuit(opt=\"Matrix-order\")`, i.e. coefficient PREPARE, matrix-order SELECT, and PREPARE inverse.",
        "",
        f"- benchmark: `{summary['benchmark']}`",
        f"- kraus_index: `{summary['kraus_index']}`",
        f"- system_qubits: `{summary['kraus_metrics']['system_qubits']}`",
        f"- pauli_terms: `{summary['kraus_metrics']['pauli_terms']}`",
        f"- pauli_norm: `{summary['kraus_metrics']['pauli_norm']}`",
        "",
        "## Counts",
        "",
        "| w | basis | status | width | gate count | non-Clifford | unknown Clifford | build time (s) | basis elapsed (s) | transpile time (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary["results"]:
        for basis in ("nam", "ibm"):
            entry = row["basis_summary"].get(basis)
            status = row["basis_recounts"][basis]["status"]
            if entry is None:
                lines.append(
                    f"| {row['forced_w']} | {basis.upper()} | {status} | - | - | - | - | "
                    f"{row['build_seconds']:.6f} | {row['basis_recounts'][basis]['elapsed_seconds']:.6f} | - |"
                )
                continue
            lines.append(
                f"| {row['forced_w']} | {basis.upper()} | {status} | {entry['width']} | "
                f"{entry['gate_count']} | {entry['non_clifford']} | {entry['unknown_clifford']} | "
                f"{row['build_seconds']:.6f} | {entry['elapsed_seconds']:.6f} | {entry['transpile_time_s']:.6f} |"
            )

    lines.extend(["", "## Placement Files", ""])
    for row in summary["results"]:
        lines.append(
            f"- w={row['forced_w']}: `{row['placement_files']['markdown']}` / `{row['placement_files']['json']}`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--w", type=int, nargs="+", default=[5, 8])
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    ms = select_kraus_zero(kraus_ops)

    baseline_be = BlockEncoding(copy.deepcopy(ms))
    terms = pauli_sum_terms(baseline_be)
    (out_dir / "tfim8_a0_pauli_sum.txt").write_text(pauli_sum_expression(terms) + "\n", encoding="utf-8")
    (out_dir / "tfim8_a0_pauli_sum.json").write_text(
        json.dumps(terms, indent=2, default=json_default),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "benchmark": "TFIM_8",
        "kraus_index": 0,
        "model_info": model_info,
        "result_dir": str(out_dir),
        "script": str(Path(__file__).resolve()),
        "scope": "complete BlockEncoding.circuit(opt='Matrix-order') for TFIM_8 A0",
        "kraus_metrics": {
            "system_qubits": int(ms.size),
            "pauli_terms": int(len(ms.instances)),
            "pauli_norm": float(ms.pauli_norm()),
            "raw_ctrl_size": int(baseline_be.ctrl_size),
        },
        "results": [],
    }

    for forced_w in args.w:
        try:
            row = run_for_w(ms, forced_w, out_dir)
        except Exception as exc:  # pragma: no cover - experiment reporting path
            row = {
                "status": "error",
                "forced_w": int(forced_w),
                "error": repr(exc),
            }
        summary["results"].append(row)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(out_dir / "report.md", summary)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
