#!/usr/bin/env python3
"""Probe Qualtran sparse alias PREPARE resources for TFIM_8 A0.

The script intentionally has two phases because the project Qiskit environment
and the Qualtran environment can have incompatible dependencies.

* ``extract``: run in the existing project/Qiskit environment.  It extracts the
  real matrix-order coefficient/address table for the first TFIM_8 Kraus
  operator at the requested widths.
* ``qualtran``: run in the Qualtran venv.  It reads the extracted table and
  builds ``SparseStatePreparationAliasSampling`` to report Qualtran resource
  counts and width estimates.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "qualtran_sparse_alias_tfim8_a0_v1"


def add_project_paths() -> None:
    for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def scalar_to_json(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return str(value)


def json_default(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def complex_from_json(value: Any) -> complex:
    if isinstance(value, dict) and {"real", "imag"} <= set(value):
        return complex(float(value["real"]), float(value["imag"]))
    return complex(value)


def group_weights(nonzero: list[dict[str, Any]], *, rel_tol: float = 1e-12) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for row in nonzero:
        weight = float(row["weight"])
        placed = False
        for bucket in buckets:
            ref = float(bucket["representative_weight"])
            if abs(weight - ref) <= rel_tol * max(1.0, abs(ref)):
                bucket["count"] += 1
                bucket["indices"].append(int(row["index"]))
                placed = True
                break
        if not placed:
            buckets.append(
                {
                    "representative_weight": weight,
                    "count": 1,
                    "indices": [int(row["index"])],
                }
            )
    buckets.sort(key=lambda item: (-int(item["count"]), float(item["representative_weight"])))
    return buckets


def extract_coefficients(widths: list[int], out_dir: Path) -> dict[str, Any]:
    import numpy as np

    add_project_paths()
    from experiment_tfim8_a0_full_matrix_order_forced_w_new import (  # type: ignore
        ForcedWNewBlockEncoding,
    )
    from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore
        BenchmarkSpec,
        build_kraus_ops,
    )

    started_all = time.perf_counter()
    kraus_ops, model_info = build_kraus_ops(BenchmarkSpec("TFIM_8", "tfim", 8))
    if not kraus_ops or len(kraus_ops[0].instances) == 0:
        raise ValueError("TFIM_8 first Kraus operator is empty")
    ms = kraus_ops[0]

    rows: list[dict[str, Any]] = []
    for width in widths:
        started = time.perf_counter()
        be = ForcedWNewBlockEncoding(copy.deepcopy(ms), forced_w=int(width))
        coeff_mode_dict, ctrl_size = be.find_optimal_order_matrices()
        coeffs = np.zeros(1 << int(ctrl_size), dtype=complex)
        for ctrl_value, coeff in coeff_mode_dict.items():
            coeffs[int(str(ctrl_value), 2)] = complex(coeff)

        nonzero: list[dict[str, Any]] = []
        for index, coeff in enumerate(coeffs):
            if abs(coeff) <= 0.0:
                continue
            nonzero.append(
                {
                    "index": int(index),
                    "ctrl": format(index, f"0{int(ctrl_size)}b"),
                    "coeff": scalar_to_json(complex(coeff)),
                    "coeff_repr": repr(complex(coeff)),
                    "weight": float(abs(coeff)),
                    "phase": float(np.angle(coeff)),
                    "popcount": int(index.bit_count()),
                }
            )

        placement_debug = getattr(be, "placement_search_debug", {})
        mobius = getattr(be, "mobius_phase_result", {})
        row = {
            "benchmark": "TFIM_8",
            "kraus_index": 0,
            "forced_w": int(width),
            "ctrl_size": int(ctrl_size),
            "slot_count": int(len(coeffs)),
            "support": int(len(nonzero)),
            "sum_weight": float(sum(item["weight"] for item in nonzero)),
            "extract_seconds": float(time.perf_counter() - started),
            "system_qubits": int(ms.size),
            "pauli_terms": int(len(ms.instances)),
            "pauli_norm": float(ms.pauli_norm()),
            "selected_k": int(getattr(be, "selected_k", width)),
            "selected_crit": str(getattr(be, "selected_crit", "")),
            "placement_shortcut": placement_debug.get("shortcut"),
            "placement_initial_strategy": placement_debug.get("initial_strategy"),
            "placement_best_cost": placement_debug.get("best_cost"),
            "placement_pre_refinement_cost": placement_debug.get("pre_refinement_cost"),
            "placement_padding_count": placement_debug.get("padding_count"),
            "placement_search_debug": placement_debug,
            "additional_modes": [
                {"pauli": str(pauli), "ctrl": str(ctrl)}
                for pauli, ctrl in getattr(be, "additional_modes", [])
            ],
            "g_modes_with_phase": [
                {"pauli": str(pauli), "ctrl": str(ctrl), "phase_pi_over_2": scalar_to_json(phase)}
                for pauli, ctrl, phase in mobius.get("g_modes_with_phase", [])
            ],
            "nonzero_coefficients": nonzero,
            "weight_groups": group_weights(nonzero),
            "popcount_histogram": dict(sorted(Counter(item["popcount"] for item in nonzero).items())),
        }
        rows.append(row)

        (out_dir / f"tfim8_a0_matrix_order_w{width}_coefficients.json").write_text(
            json.dumps(row, indent=2, default=json_default),
            encoding="utf-8",
        )

    summary = {
        "phase": "extract",
        "script": str(Path(__file__).resolve()),
        "result_dir": str(out_dir),
        "model_info": model_info,
        "elapsed_seconds": float(time.perf_counter() - started_all),
        "rows": rows,
    }
    (out_dir / "coefficients_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    return summary


def sympy_to_json(value: Any) -> Any:
    try:
        if hasattr(value, "is_integer") and value.is_integer:
            return int(value)
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return str(value)


def gate_counts_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "asdict"):
        raw = value.asdict()
    else:
        raw = getattr(value, "__dict__", {})
    return {str(key): sympy_to_json(val) for key, val in raw.items()}


def t_complexity_to_dict(value: Any) -> dict[str, Any]:
    return {
        "t": sympy_to_json(getattr(value, "t", 0)),
        "clifford": sympy_to_json(getattr(value, "clifford", 0)),
        "rotations": sympy_to_json(getattr(value, "rotations", 0)),
        "repr": str(value),
    }


def bloq_count_rows(counts: dict[Any, Any]) -> list[dict[str, Any]]:
    rows = []
    for bloq, count in counts.items():
        rows.append(
            {
                "bloq_type": type(bloq).__name__,
                "bloq": str(bloq),
                "count": sympy_to_json(count),
            }
        )
    rows.sort(key=lambda item: (str(item["bloq_type"]), str(item["bloq"])))
    return rows


def load_extracted_rows(out_dir: Path) -> list[dict[str, Any]]:
    summary_path = out_dir / "coefficients_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))["rows"]
    rows = []
    for path in sorted(out_dir.glob("tfim8_a0_matrix_order_w*_coefficients.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"No extracted coefficient JSON found under {out_dir}")
    return rows


def qualtran_resources(out_dir: Path, epsilons: list[float]) -> dict[str, Any]:
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
    )
    from qualtran.resource_counting import (
        BloqCount,
        QECGatesCost,
        QubitCount,
        get_cost_value,
    )

    rows_in = load_extracted_rows(out_dir)
    results = []
    for row in rows_in:
        width = int(row["ctrl_size"])
        support = int(row["support"])
        slot_count = int(row["slot_count"])
        weights = {
            int(item["index"]): float(item["weight"])
            for item in row["nonzero_coefficients"]
            if float(item["weight"]) > 0
        }
        sum_weight = float(sum(weights.values()))
        log_d = int(math.ceil(math.log2(max(support, 1))))
        for eps in epsilons:
            precision = float(eps) * sum_weight
            started = time.perf_counter()
            bloq = SparseStatePreparationAliasSampling.from_sparse_dict(
                weights,
                slot_count,
                precision=precision,
            )
            construct_seconds = time.perf_counter() - started

            started = time.perf_counter()
            t_complexity = bloq.t_complexity()
            t_seconds = time.perf_counter() - started

            qec_error = None
            qec_counts = None
            try:
                qec_counts = get_cost_value(bloq, QECGatesCost())
            except Exception as exc:  # pragma: no cover - diagnostic path
                qec_error = repr(exc)

            width_error = None
            qualtran_width = None
            try:
                qualtran_width = get_cost_value(bloq, QubitCount())
            except Exception as exc:  # pragma: no cover - diagnostic path
                width_error = repr(exc)

            call_graph_error = None
            call_graph_summary = None
            leaf_rows: list[dict[str, Any]] = []
            try:
                call_graph, sigma = bloq.call_graph()
                call_graph_summary = {
                    "nodes": int(len(call_graph.nodes)),
                    "edges": int(len(call_graph.edges)),
                    "sigma": bloq_count_rows(sigma),
                }
                leaf_counts = get_cost_value(bloq, BloqCount.for_call_graph_leaf_bloqs(call_graph))
                leaf_rows = bloq_count_rows(leaf_counts)
            except Exception as exc:  # pragma: no cover - diagnostic path
                call_graph_error = repr(exc)

            qrom = bloq.qrom_bloq
            mu = int(bloq.mu)
            selection_bitsize = int(bloq.selection_bitsize)
            sparse_index_bitsize = int(bloq.sparse_index_bitsize)
            formula_t = 4 * support + 8 * mu + 7 * selection_bitsize + 12 * sparse_index_bitsize - 8
            formula_clifford = (
                support * mu * selection_bitsize**2
                + 13 * support
                + 47 * mu
                + 10 * selection_bitsize
                + 52 * sparse_index_bitsize
                - 12
            )
            signature_qubits = int(bloq.signature.n_qubits())
            doc_qrom_extra_ancilla = selection_bitsize
            doc_total_with_qrom_extra = signature_qubits + doc_qrom_extra_ancilla
            leaf_total = sum(int(item["count"]) for item in leaf_rows if isinstance(item["count"], int))

            alias_table = []
            index_array = list(getattr(bloq.index, "tolist", lambda: bloq.index)())
            alt_array = list(getattr(bloq.alt, "tolist", lambda: bloq.alt)())
            keep_array = list(getattr(bloq.keep, "tolist", lambda: bloq.keep)())
            for sparse_j, (index, alt, keep) in enumerate(zip(index_array, alt_array, keep_array)):
                alias_table.append(
                    {
                        "sparse_j": int(sparse_j),
                        "index": int(index),
                        "index_bits": format(int(index), f"0{selection_bitsize}b"),
                        "alt": int(alt),
                        "alt_bits": format(int(alt), f"0{selection_bitsize}b"),
                        "keep": int(keep),
                    }
                )

            results.append(
                {
                    "benchmark": row["benchmark"],
                    "kraus_index": int(row["kraus_index"]),
                    "w": width,
                    "slot_count": slot_count,
                    "support": support,
                    "epsilon_normalized": float(eps),
                    "precision_unnormalized": precision,
                    "construct_seconds": float(construct_seconds),
                    "t_complexity_seconds": float(t_seconds),
                    "mu": mu,
                    "selection_bitsize": selection_bitsize,
                    "sparse_index_bitsize": sparse_index_bitsize,
                    "sum_weight": sum_weight,
                    "qualtran_signature_qubits": signature_qubits,
                    "qualtran_qubit_count": sympy_to_json(qualtran_width),
                    "qualtran_qubit_count_error": width_error,
                    "doc_qrom_extra_ancilla": doc_qrom_extra_ancilla,
                    "doc_total_with_qrom_extra": doc_total_with_qrom_extra,
                    "qrom_signature_qubits": int(qrom.signature.n_qubits()),
                    "qrom_bloq": str(qrom),
                    "t_complexity": t_complexity_to_dict(t_complexity),
                    "sparse_symbolic_formula": {
                        "t": int(formula_t),
                        "clifford": int(formula_clifford),
                        "rotations": 0,
                        "formula": "T=4d+8mu+7logN+12logd-8; C=d*mu*logN^2+13d+47mu+10logN+52logd-12",
                    },
                    "qec_gate_counts": None if qec_counts is None else gate_counts_to_dict(qec_counts),
                    "qec_gate_counts_error": qec_error,
                    "leaf_gate_total_including_bookkeeping": int(leaf_total),
                    "leaf_gate_counts": leaf_rows,
                    "call_graph": call_graph_summary,
                    "call_graph_error": call_graph_error,
                    "alias_table": alias_table,
                    "weight_groups": row.get("weight_groups", []),
                    "placement_shortcut": row.get("placement_shortcut"),
                    "placement_best_cost": row.get("placement_best_cost"),
                    "placement_padding_count": row.get("placement_padding_count"),
                }
            )

    summary = {
        "phase": "qualtran",
        "script": str(Path(__file__).resolve()),
        "result_dir": str(out_dir),
        "qualtran_module": importlib.import_module("qualtran").__file__,
        "epsilons": epsilons,
        "rows": results,
    }
    (out_dir / "qualtran_sparse_alias_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(out_dir / "qualtran_sparse_alias_report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# Qualtran Sparse Alias PREPARE Probe",
        "",
        "This reports Qualtran resources for `SparseStatePreparationAliasSampling` on the TFIM_8 A0 matrix-order coefficient table.  Counts are Qualtran bloq/QEC resources, not NAM/IBM Qiskit opt0 gate counts.",
        "",
        "## Main Rows",
        "",
        "| w | support/slots | eps | mu | signature qubits | QubitCount | doc sig+QROM-extra | T | Clifford | rotations | QEC counts | leaf total |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        qec = row.get("qec_gate_counts") or {}
        qec_text = ", ".join(f"{key}={value}" for key, value in qec.items()) if qec else "n/a"
        tc = row["t_complexity"]
        lines.append(
            f"| {row['w']} | {row['support']}/{row['slot_count']} | "
            f"{row['epsilon_normalized']:.1e} | {row['mu']} | "
            f"{row['qualtran_signature_qubits']} | {row['qualtran_qubit_count']} | "
            f"{row['doc_total_with_qrom_extra']} | {tc['t']} | {tc['clifford']} | "
            f"{tc['rotations']} | `{qec_text}` | {row['leaf_gate_total_including_bookkeeping']} |"
        )

    lines.extend(
        [
            "",
            "## Gate-Set Notes",
            "",
            "- The bloq structure is `UNIFORM_d`, `H^mu`, QROM loading `(index, alt, keep)`, `LessThanEqual`, and controlled swaps between `selection` and `alt`.",
            "- `QubitCount` is Qualtran's recursive min-max width estimate. `signature qubits` is just visible registers. `doc sig+QROM-extra` adds the `log(L)` QROM extra-ancilla term mentioned in the Qualtran docstring.",
            "- `leaf total` includes Qualtran bookkeeping leaves such as split/join when present, so use it only as a diagnostic, not as a NAM/IBM metric.",
            "",
            "## Alias Tables",
            "",
        ]
    )
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["w"]), float(row["epsilon_normalized"]))].append(row)
    for (width, eps), group_rows in sorted(grouped.items()):
        row = group_rows[0]
        lines.extend(
            [
                f"### w={width}, eps={eps:.1e}",
                "",
                "| j | index | alt | keep |",
                "| ---: | --- | --- | ---: |",
            ]
        )
        for item in row["alias_table"]:
            lines.append(
                f"| {item['sparse_j']} | `{item['index_bits']}` ({item['index']}) | "
                f"`{item['alt_bits']}` ({item['alt']}) | {item['keep']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["extract", "qualtran", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--w", type=int, nargs="+", default=[5, 8, 9])
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3, 1e-4],
        help="Normalized probability precision. The Qualtran factory receives eps * sum(weights).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in {"extract", "both"}:
        extract_coefficients(args.w, out_dir)
    if args.phase in {"qualtran", "both"}:
        qualtran_resources(out_dir, args.eps)


if __name__ == "__main__":
    main()
