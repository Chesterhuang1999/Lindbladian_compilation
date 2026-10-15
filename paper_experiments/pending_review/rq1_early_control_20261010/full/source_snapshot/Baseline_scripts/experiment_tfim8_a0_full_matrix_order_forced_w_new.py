#!/usr/bin/env python3
"""Full matrix-order block-encoding experiment for TFIM_8 A0.

This harness uses ``block_encoding_new.BlockEncoding`` as the base class and
forces the matrix-order address width in the new placement-search path.  For
each requested width it builds the complete block-encoding circuit:

    PREPARE(coefficients) -> matrix-order SELECT -> PREPARE(coefficients)^dag
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

from block_encoding import (  # type: ignore  # noqa: E402
    build_vec_phase_lookup,
    extract_basis_modes_with_phases,
    mobius_invert_modes_with_phases,
)
from block_encoding_new import (  # type: ignore  # noqa: E402
    BlockEncoding as NewBlockEncoding,
    _pauli_label_to_symplectic_vector,
)
from placement_search import search_for_pauli_placement  # type: ignore  # noqa: E402
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


RESULT_ROOT = ROOT / "Baseline_results" / "tfim8_a0_full_matrix_order_forced_w_new_v1"


class ForcedWNewBlockEncoding(NewBlockEncoding):
    """New BlockEncoding variant that forces matrix-order control width."""

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
        pauli_list = PauliList(pauli_list)
        x_part, z_part = pauli_list.x, pauli_list.z
        symplectic_matrix = np.hstack((x_part, z_part)).astype(int)
        phase_array = np.array([2 * np.angle(phase) / np.pi for phase in phase_list])

        natural_w = int(np.ceil(np.log2(len(self.coeff_list))))
        w = int(self.forced_w)
        if w < natural_w:
            raise ValueError(f"forced w={w} is smaller than natural address width {natural_w}")

        identity_label = None
        for ms in self.mat_list:
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            if pauli_op == "I" * len(pauli_op):
                identity_label = pauli_op
                break

        def _pauli_weight(label: str) -> int:
            return sum(1 for char in label if char != "I")

        allow_zero_address = identity_label is None and len(pauli_list) > (2**w - 1)
        padding_to_full_address = (
            not allow_zero_address
            and identity_label is None
            and len(pauli_list) < (2**w - 1)
        )
        is_full_address = (
            not allow_zero_address
            and (len(pauli_list) == (2**w - 1) or padding_to_full_address)
        )
        real_pauli_count = len(pauli_list)
        if real_pauli_count <= 256:
            search_profile = {
                "beam_width": 16,
                "layer_candidate_limit": 16,
                "max_exact_layer_assignments": 16,
                "mixed_initial_beam_width": 128,
                "mixed_low_weight_count": 128,
                "mixed_random_count": 256,
                "future_exact_address_limit": 8,
            }
        else:
            search_profile = {
                "beam_width": 16,
                "layer_candidate_limit": 16,
                "max_exact_layer_assignments": 16,
                "mixed_initial_beam_width": 32,
                "mixed_low_weight_count": 32,
                "mixed_random_count": 64,
                "future_exact_address_limit": 8,
            }

        additional_modes, placement_debug = search_for_pauli_placement(
            self.J,
            m=w,
            beam_width=search_profile["beam_width"],
            layer_candidate_limit=search_profile["layer_candidate_limit"],
            max_exact_layer_assignments=search_profile["max_exact_layer_assignments"],
            future_exact_address_limit=search_profile["future_exact_address_limit"],
            initial_strategy="mixed",
            mixed_full_initial_beam_width=search_profile["mixed_initial_beam_width"],
            mixed_nonfull_initial_beam_width=search_profile["mixed_initial_beam_width"],
            mixed_low_weight_count=search_profile["mixed_low_weight_count"],
            mixed_random_count=search_profile["mixed_random_count"],
            swap_refinement=True,
            swap_refinement_max_terms=16,
            padding_to_full_address=padding_to_full_address,
            allow_zero_address=allow_zero_address,
            return_debug=True,
        )
        placement_debug["search_profile"] = search_profile
        placement_debug["forced_w"] = w
        placement_debug["natural_w"] = natural_w
        placement_debug["is_full_address"] = is_full_address
        if placement_debug.get("padding_count", 0) > 0:
            additional_modes = placement_debug["padded_placement"]
        additional_modes = sorted(additional_modes, key=lambda item: int(item[1], 2))

        vec_phase_lookup = build_vec_phase_lookup(symplectic_matrix, phase_array)
        for padding_label in placement_debug.get("padding_labels", []):
            padding_vec = _pauli_label_to_symplectic_vector(padding_label)
            vec_phase_lookup.setdefault(padding_vec.tobytes(), 0.0)
        mobius_result = mobius_invert_modes_with_phases(
            w, additional_modes, vec_phase_lookup, zero_phase
        )
        g_modes = mobius_result["g_modes_with_phase"]
        final_cost = sum(ctrl.count("1") * _pauli_weight(label) for label, ctrl, _ in g_modes)

        self.additional_modes = additional_modes
        self.candidate_costs = [
            (
                "placement-search",
                placement_debug.get("initial_strategy"),
                placement_debug.get("pre_refinement_cost"),
                placement_debug.get("best_cost"),
                placement_debug.get("padded_search_cost"),
                final_cost,
                placement_debug.get("padding_count"),
                search_profile,
            )
        ]
        self.selected_k = w
        self.selected_crit = "mixed-placement-search-forced-w"
        self.placement_search_debug = placement_debug

        coeff_pool_by_label: dict[str, list[Any]] = {}
        for idx, ms in enumerate(self.mat_list):
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            coeff_pool_by_label.setdefault(pauli_op, []).append(self.coeff_list[idx])

        coeff_mode_dict = {}
        for pauli_label, ctrl_value in additional_modes:
            coeff_pool = coeff_pool_by_label.get(pauli_label, [])
            if len(coeff_pool) > 0:
                coeff_mode = coeff_pool.pop(0)
            else:
                coeff_mode = 0.0 + 0.0j
            coeff_mode_dict[ctrl_value] = coeff_mode

        self.coeff_mode_dict = coeff_mode_dict
        self.vec_phase_lookup = vec_phase_lookup
        self.basis_modes_with_phase, self.nonbasis_modes_with_phase = extract_basis_modes_with_phases(
            additional_modes, self.vec_phase_lookup, zero_phase
        )
        self.mobius_phase_result = mobius_result

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


def pauli_sum_terms(be: NewBlockEncoding) -> list[dict[str, Any]]:
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


def placement_metadata(be: ForcedWNewBlockEncoding) -> dict[str, Any]:
    mobius = getattr(be, "mobius_phase_result", {})
    return {
        "base_class": "block_encoding_new.BlockEncoding",
        "subclass": "ForcedWNewBlockEncoding",
        "forced_w": int(be.forced_w),
        "sys_size": int(be.sys_size),
        "ctrl_size_raw": int(be.ctrl_size),
        "ctrl_size_matrix_order": int(len(next(iter(getattr(be, "coeff_mode_dict", {"": 0}).keys()), ""))),
        "circuit_width": int(be.circuit_width),
        "selected_k": int(getattr(be, "selected_k", be.forced_w)),
        "selected_crit": str(getattr(be, "selected_crit", "")),
        "candidate_costs": getattr(be, "candidate_costs", []),
        "placement_search_debug": getattr(be, "placement_search_debug", {}),
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
    }


def write_placement_md(path: Path, metadata: dict[str, Any]) -> None:
    debug = metadata["placement_search_debug"]
    costs = metadata["candidate_costs"][0] if metadata["candidate_costs"] else ()
    lines = [
        f"# TFIM_8 A0 Full Matrix-Order Placement, forced w={metadata['forced_w']}",
        "",
        f"- base class: `{metadata['base_class']}`",
        f"- raw ctrl_size: `{metadata['ctrl_size_raw']}`",
        f"- matrix-order ctrl_size: `{metadata['ctrl_size_matrix_order']}`",
        f"- sys_size: `{metadata['sys_size']}`",
        f"- circuit_width: `{metadata['circuit_width']}`",
        f"- selected_crit: `{metadata['selected_crit']}`",
        f"- placement initial_strategy: `{debug.get('initial_strategy')}`",
        f"- placement shortcut: `{debug.get('shortcut')}`",
        f"- pre_refinement_cost: `{debug.get('pre_refinement_cost')}`",
        f"- best_cost: `{debug.get('best_cost')}`",
        f"- padded_search_cost: `{debug.get('padded_search_cost')}`",
        f"- final_cost: `{costs[5] if len(costs) > 5 else None}`",
        f"- padding_count: `{debug.get('padding_count')}`",
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
    be = ForcedWNewBlockEncoding(copy.deepcopy(ms), forced_w=forced_w)
    qc = be.circuit(opt="Matrix-order")
    build_seconds = time.perf_counter() - started

    prefix = f"tfim8_a0_full_matrix_order_new_w{forced_w}"
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
        "base_class": "block_encoding_new.BlockEncoding",
        "forced_w": int(forced_w),
        "build_seconds": float(build_seconds),
        "logical_qasm": str(out_dir / f"{prefix}_logical.qasm"),
        "qiskit_metrics_logical": qiskit_metrics(qc),
        "block_encoding_reported_width": int(be.circuit_width),
        "raw_ctrl_size": int(be.ctrl_size),
        "matrix_order_ctrl_size": int(forced_w),
        "sys_size": int(be.sys_size),
        "candidate_costs": getattr(be, "candidate_costs", []),
        "placement_search_debug": getattr(be, "placement_search_debug", {}),
        "placement_files": {
            "json": str(out_dir / f"{prefix}_placement.json"),
            "markdown": str(out_dir / f"{prefix}_placement.md"),
        },
        "basis_recounts": basis_recounts,
        "basis_summary": basis_summary,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TFIM_8 A0 Full Matrix-Order Forced-w New BlockEncoding Experiment",
        "",
        "Circuit scope: complete `BlockEncoding.circuit(opt=\"Matrix-order\")`, i.e. coefficient PREPARE, matrix-order SELECT, and PREPARE inverse.",
        "",
        f"- base class: `{summary['base_class']}`",
        f"- benchmark: `{summary['benchmark']}`",
        f"- kraus_index: `{summary['kraus_index']}`",
        f"- system_qubits: `{summary['kraus_metrics']['system_qubits']}`",
        f"- pauli_terms: `{summary['kraus_metrics']['pauli_terms']}`",
        f"- pauli_norm: `{summary['kraus_metrics']['pauli_norm']}`",
        "",
        "## Counts",
        "",
        "| w | basis | status | width | gate count | non-Clifford | unknown Clifford | build time (s) | placement shortcut | final cost | basis elapsed (s) | transpile time (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    for row in summary["results"]:
        debug = row.get("placement_search_debug", {})
        costs = row.get("candidate_costs", [[]])
        final_cost = costs[0][5] if costs and len(costs[0]) > 5 else None
        shortcut = debug.get("shortcut")
        for basis in ("nam", "ibm"):
            entry = row["basis_summary"].get(basis)
            status = row["basis_recounts"][basis]["status"]
            if entry is None:
                lines.append(
                    f"| {row['forced_w']} | {basis.upper()} | {status} | - | - | - | - | "
                    f"{row['build_seconds']:.6f} | `{shortcut}` | {final_cost} | "
                    f"{row['basis_recounts'][basis]['elapsed_seconds']:.6f} | - |"
                )
                continue
            lines.append(
                f"| {row['forced_w']} | {basis.upper()} | {status} | {entry['width']} | "
                f"{entry['gate_count']} | {entry['non_clifford']} | {entry['unknown_clifford']} | "
                f"{row['build_seconds']:.6f} | `{shortcut}` | {final_cost} | "
                f"{entry['elapsed_seconds']:.6f} | {entry['transpile_time_s']:.6f} |"
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

    baseline_be = NewBlockEncoding(copy.deepcopy(ms))
    terms = pauli_sum_terms(baseline_be)
    (out_dir / "tfim8_a0_pauli_sum.txt").write_text(pauli_sum_expression(terms) + "\n", encoding="utf-8")
    (out_dir / "tfim8_a0_pauli_sum.json").write_text(
        json.dumps(terms, indent=2, default=json_default),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "benchmark": "TFIM_8",
        "kraus_index": 0,
        "base_class": "block_encoding_new.BlockEncoding",
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
        except Exception as exc:
            row = {
                "status": "error",
                "base_class": "block_encoding_new.BlockEncoding",
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
