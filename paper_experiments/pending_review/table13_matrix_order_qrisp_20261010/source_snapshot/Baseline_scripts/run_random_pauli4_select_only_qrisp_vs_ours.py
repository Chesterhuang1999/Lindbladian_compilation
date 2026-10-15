#!/usr/bin/env python3
"""SELECT-only comparison for 4-qubit random/full Pauli LCU examples.

This intentionally removes PREP/PREP-dagger from both methods:

* Ours: ``BlockEncoding.mulplex_U_opt_order()`` followed by the current large
  controlled-gate lowering pass.
* Qrisp: direct ``q_switch(index, unitaries, sys)`` using the same Pauli
  unitary list as ``BlockEncoding.from_lcu`` would use.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding import (  # type: ignore  # noqa: E402
    BlockEncoding as CurrentBlockEncoding,
    expand_large_controlled_gates,
)
from channel_IR import Matrixsum  # type: ignore  # noqa: E402
from run_dense_random_pauli4_block_encoding import (  # type: ignore  # noqa: E402
    full_matrixsum,
    random_matrixsum,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from run_table3_extra_block_encoding_cases import (  # type: ignore  # noqa: E402
    qrisp_pauli_unitary_with_phase,
    term_to_label_phase,
)
from sum_rewritten_kraus_block_counts_qrisp_vs_ours import (  # type: ignore  # noqa: E402
    kraus_effective_terms,
    natural_width,
)


RESULT_ROOT = ROOT / "Baseline_results" / "random_pauli4_select_only_ours_qrisp_v1"
RESULTS_MD = ROOT / "Results" / "Table_random_pauli4_select_only_ours_qrisp.md"
RESULTS_TEX = ROOT / "Results" / "Table_random_pauli4_select_only_ours_qrisp.tex"


CASES = (
    {
        "name": "random_pauli_4q_20t",
        "label": "Random Pauli 4q/20t",
        "num_qubits": 4,
        "num_terms": 20,
        "samples": 10,
        "seed_base": 20260703,
        "full": False,
    },
    {
        "name": "random_pauli_4q_64t",
        "label": "Random Pauli 4q/64t",
        "num_qubits": 4,
        "num_terms": 64,
        "samples": 5,
        "seed_base": 20260706,
        "full": False,
    },
    {
        "name": "random_pauli_4q_129t",
        "label": "Random Pauli 4q/129t",
        "num_qubits": 4,
        "num_terms": 129,
        "samples": 5,
        "seed_base": 20360706,
        "full": False,
    },
    {
        "name": "random_pauli_4q_200t",
        "label": "Random Pauli 4q/200t",
        "num_qubits": 4,
        "num_terms": 200,
        "samples": 5,
        "seed_base": 20460706,
        "full": False,
    },
    {
        "name": "full_pauli_4q_256t",
        "label": "Full Pauli n=4",
        "num_qubits": 4,
        "num_terms": 256,
        "samples": 1,
        "seed_base": 20560706,
        "full": True,
    },
)


def summarize_basis(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry or entry.get("status") != "ok" or entry.get("result") is None:
        return None
    stats = entry["result"]["stats"]
    return {
        "gate_count": int(stats["metric_total"]),
        "width": int(stats["num_qubits"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "ops": stats["ops"],
        "qasm": entry["result"]["qasm"],
        "elapsed_seconds": float(entry["elapsed_seconds"]),
        "transpile_time_s": float(entry["result"]["transpile_time_s"]),
    }


def recount_bases(qc: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
    rows = {}
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


def validate_matrixsum(ms: Matrixsum) -> dict[str, Any]:
    terms = kraus_effective_terms(ms)
    labels = [label for label, _coeff in terms]
    identity = "I" * ms.size
    return {
        "num_qubits": int(ms.size),
        "term_count": int(len(terms)),
        "unique_terms": int(len(set(labels))),
        "has_identity": identity in labels,
        "l1_norm": float(sum(abs(coeff) for _label, coeff in terms)),
        "natural_width": int(natural_width(ms)),
    }


def build_matrixsum(case: dict[str, Any], seed: int) -> Matrixsum:
    if case["full"]:
        return full_matrixsum(case["num_qubits"], seed, coeff_mode="random-positive")
    return random_matrixsum(
        case["num_qubits"],
        case["num_terms"],
        seed,
        coeff_mode="random-positive",
    )


def count_ours_select_only(ms: Matrixsum, out_dir: Path, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "ours_select_only_matrix_order",
        "term_count": int(len(ms.instances)),
        "natural_width": int(natural_width(ms)),
    }
    started = time.perf_counter()
    try:
        be = CurrentBlockEncoding(copy.deepcopy(ms))
        raw_select, analytical_t, analytical_mcx, analytical_cx, ctrl_size = be.mulplex_U_opt_order()
        qc = expand_large_controlled_gates(raw_select, if_optimize=True)
    except Exception:
        result["status"] = "build_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        result["error"] = traceback.format_exc()
        return result

    result.update(
        {
            "status": "ok",
            "ctrl_size": int(ctrl_size),
            "selected_k": int(getattr(be, "selected_k", ctrl_size)),
            "selected_crit": str(getattr(be, "selected_crit", "")),
            "analytical": {
                "tcount": int(analytical_t),
                "mcx_like": int(analytical_mcx),
                "cx": int(analytical_cx),
            },
            "raw_select_metrics": qiskit_metrics(raw_select),
            "qiskit_metrics": qiskit_metrics(qc),
            "qasm": write_qasm2(qc, out_dir / f"{prefix}_logical.qasm"),
        }
    )
    result["basis_recounts"] = recount_bases(qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def count_qrisp_select_only(ms: Matrixsum, out_dir: Path, prefix: str, workspace: int) -> dict[str, Any]:
    import qrisp
    from qrisp import QuantumFloat, QuantumVariable
    from qrisp.jasp import q_switch

    terms = []
    for instance, coeff in copy.deepcopy(ms).instances:
        label, atom_phase = term_to_label_phase(instance)
        effective = complex(coeff) * atom_phase
        magnitude = abs(effective)
        phase = 1.0 + 0.0j if magnitude == 0 else effective / magnitude
        terms.append((label, phase))

    result: dict[str, Any] = {
        "method": "qrisp_select_only_q_switch",
        "term_count": int(len(terms)),
        "index_width": int(math.ceil(math.log2(max(1, len(terms))))),
        "workspace": int(workspace),
        "qrisp_version": getattr(qrisp, "__version__", None),
        "qrisp_file": getattr(qrisp, "__file__", None),
    }
    started = time.perf_counter()
    try:
        idx = QuantumFloat(result["index_width"], name="idx")
        sys_qv = QuantumVariable(ms.size, name="sys")
        unitaries = [
            qrisp_pauli_unitary_with_phase(label, phase)
            for label, phase in terms
        ]
        q_switch(idx, unitaries, sys_qv)
        compiled = sys_qv.qs.compile(workspace=workspace)
        qc = compiled.to_qiskit()
    except Exception:
        result["status"] = "build_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        result["error"] = traceback.format_exc()
        return result

    result["status"] = "ok"
    result["compiled_qrisp_metrics"] = {
        "num_qubits": int(compiled.num_qubits()),
        "depth": int(compiled.depth()),
        "ops": {str(k): int(v) for k, v in compiled.count_ops().items()},
    }
    result["qiskit_metrics"] = qiskit_metrics(qc)
    result["qasm"] = write_qasm2(qc, out_dir / f"{prefix}_qiskit.qasm")
    result["basis_recounts"] = recount_bases(qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def aggregate_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(statistics.fmean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_case(case_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for basis in ("nam", "ibm"):
        values = {
            "ours_select_only_matrix_order": {"gates": [], "non_clifford": [], "width": [], "time": []},
            "qrisp_select_only_q_switch": {"gates": [], "non_clifford": [], "width": [], "time": []},
        }
        for sample in case_summary["samples"]:
            for key in values:
                result = sample[key]
                summary = result.get("basis_summary", {}).get(basis) if result.get("status") == "ok" else None
                if summary is None:
                    continue
                values[key]["gates"].append(float(summary["gate_count"]))
                values[key]["non_clifford"].append(float(summary["non_clifford"]))
                values[key]["width"].append(float(summary["width"]))
                values[key]["time"].append(float(result.get("elapsed_seconds", 0.0)))
        for method, buckets in values.items():
            rows.append(
                {
                    "case": case_summary["case"]["name"],
                    "label": case_summary["case"]["label"],
                    "basis": basis,
                    "method": method,
                    "gate_count": aggregate_values(buckets["gates"]),
                    "non_clifford": aggregate_values(buckets["non_clifford"]),
                    "width": aggregate_values(buckets["width"]),
                    "elapsed_seconds": aggregate_values(buckets["time"]),
                    "completed_samples": int(len(buckets["gates"])),
                    "missing_samples": int(case_summary["num_samples"] - len(buckets["gates"])),
                }
            )
    for row in rows:
        if row["method"] != "qrisp_select_only_q_switch":
            continue
        ours_row = next(
            candidate
            for candidate in rows
            if candidate["case"] == row["case"]
            and candidate["basis"] == row["basis"]
            and candidate["method"] == "ours_select_only_matrix_order"
        )
        ours_mean = ours_row["gate_count"]["mean"]
        qrisp_mean = row["gate_count"]["mean"]
        ours_nc = ours_row["non_clifford"]["mean"]
        qrisp_nc = row["non_clifford"]["mean"]
        row["qrisp_over_ours_mean"] = None if not ours_mean or qrisp_mean is None else float(qrisp_mean / ours_mean)
        row["qrisp_over_ours_non_clifford_mean"] = (
            None if not ours_nc or qrisp_nc is None else float(qrisp_nc / ours_nc)
        )
    return rows


def fmt_stat(stat: dict[str, Any], digits: int = 1) -> str:
    if stat["mean"] is None:
        return "-"
    if stat["count"] == 1:
        return f"{stat['mean']:.{digits}f}"
    return f"{stat['mean']:.{digits}f} +/- {stat['stdev']:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for row in summary["aggregate_rows"]:
        ratio = row.get("qrisp_over_ours_mean")
        ratio_nc = row.get("qrisp_over_ours_non_clifford_mean")
        rows.append(
            [
                row["label"],
                row["basis"].upper(),
                row["method"],
                fmt_stat(row["gate_count"], 1),
                fmt_stat(row["non_clifford"], 1),
                fmt_stat(row["width"], 1),
                fmt_stat(row["elapsed_seconds"], 3),
                f"{row['completed_samples']}/{row['missing_samples']}",
                "" if ratio is None else f"{ratio:.4g}",
                "" if ratio_nc is None else f"{ratio_nc:.4g}",
            ]
        )

    lines = [
        "# Random Pauli 4q SELECT-Only Comparison",
        "",
        "- This table removes coefficient PREP/PREP-dagger from both methods.",
        "- Coefficients are still generated with the random-positive L1-normalized protocol for support consistency, but SELECT costs are coefficient-independent except for possible phase.",
        "- Ours uses current `BlockEncoding.mulplex_U_opt_order()`; Qrisp uses direct `q_switch(index, unitaries, sys)`.",
        "",
        markdown_table(
            [
                "case",
                "basis",
                "method",
                "gates mean +/- sd",
                "NC mean +/- sd",
                "width mean +/- sd",
                "time mean +/- sd (s)",
                "completed/missing",
                "qrisp/ours gates",
                "qrisp/ours NC",
            ],
            rows,
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def tex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_tex(summary: dict[str, Any], path: Path) -> None:
    rows_by_key = {
        (row["case"], row["basis"], row["method"]): row
        for row in summary["aggregate_rows"]
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{SELECT-only NAM-basis comparison for 4-qubit random/full Pauli LCU examples. PREP and PREP$^{\\dagger}$ are removed from both methods. Ratio is Qrisp/Ours.}",
        "\\label{tab:random-pauli4-select-only}",
        "\\begin{tabular}{lrrlrrrrrrrr}",
        "\\toprule",
        "& & & & \\multicolumn{3}{c}{Ours} & \\multicolumn{3}{c}{Qrisp} & \\multicolumn{2}{c}{Ratio} \\\\",
        "\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-12}",
        "Example & $n$ & Terms & Scope & Total & NC & Width & Total & NC & Width & Total & NC \\\\",
        "\\midrule",
    ]
    for case in CASES:
        ours = rows_by_key[(case["name"], "nam", "ours_select_only_matrix_order")]
        qrisp = rows_by_key[(case["name"], "nam", "qrisp_select_only_q_switch")]
        ratio = qrisp.get("qrisp_over_ours_mean")
        ratio_nc = qrisp.get("qrisp_over_ours_non_clifford_mean")
        sample_label = "single" if case["samples"] == 1 else f"avg. {case['samples']}"
        lines.append(
            f"{tex_escape(case['label'])} & {case['num_qubits']} & {case['num_terms']:,} & "
            f"{sample_label} & "
            f"{fmt_stat(ours['gate_count'], 1).replace('+/-', '$\\pm$')} & "
            f"{fmt_stat(ours['non_clifford'], 1).replace('+/-', '$\\pm$')} & "
            f"{fmt_stat(ours['width'], 1).replace('+/-', '$\\pm$')} & "
            f"{fmt_stat(qrisp['gate_count'], 1).replace('+/-', '$\\pm$')} & "
            f"{fmt_stat(qrisp['non_clifford'], 1).replace('+/-', '$\\pm$')} & "
            f"{fmt_stat(qrisp['width'], 1).replace('+/-', '$\\pm$')} & "
            f"{'' if ratio is None else f'{ratio:.3g}'} & "
            f"{'' if ratio_nc is None else f'{ratio_nc:.3g}'} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--workspace", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": "random_pauli4_select_only_ours_qrisp",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "cases": [],
        "aggregate_rows": [],
    }

    for case in CASES:
        case_summary = {
            "case": case,
            "num_samples": int(case["samples"]),
            "samples": [],
        }
        for sample_index in range(case["samples"]):
            seed = int(case["seed_base"] + sample_index)
            ms = build_matrixsum(case, seed)
            validation = validate_matrixsum(ms)
            if validation["unique_terms"] != validation["term_count"]:
                raise RuntimeError(f"duplicate terms detected: {validation}")
            if not case["full"] and validation["has_identity"]:
                raise RuntimeError(f"non-full random case unexpectedly contains identity: {validation}")

            sample_prefix = f"{case['name']}_sample{sample_index:02d}_seed{seed}"
            sample_dir = out_dir / case["name"] / f"sample{sample_index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            ours = count_ours_select_only(copy.deepcopy(ms), sample_dir, sample_prefix + "_ours_select")
            qrisp = count_qrisp_select_only(copy.deepcopy(ms), sample_dir, sample_prefix + "_qrisp_select", args.workspace)
            case_summary["samples"].append(
                {
                    "case": case["name"],
                    "sample_index": int(sample_index),
                    "seed": int(seed),
                    "validation": validation,
                    "ours_select_only_matrix_order": ours,
                    "qrisp_select_only_q_switch": qrisp,
                }
            )
            (out_dir / "summary.partial.json").write_text(
                json.dumps(summary, indent=2, default=json_default),
                encoding="utf-8",
            )

        summary["cases"].append(case_summary)
        summary["aggregate_rows"] = [
            row for case_item in summary["cases"] for row in aggregate_case(case_item)
        ]
        (out_dir / "summary.partial.json").write_text(
            json.dumps(summary, indent=2, default=json_default),
            encoding="utf-8",
        )
        write_report(summary, out_dir / "report.partial.md")

    summary["aggregate_rows"] = [
        row for case_item in summary["cases"] for row in aggregate_case(case_item)
    ]
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    write_report(summary, RESULTS_MD)
    write_tex(summary, RESULTS_TEX)
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"report: {out_dir / 'report.md'}")
    print(f"results md: {RESULTS_MD}")
    print(f"results tex: {RESULTS_TEX}")
    print(json.dumps(summary["aggregate_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
