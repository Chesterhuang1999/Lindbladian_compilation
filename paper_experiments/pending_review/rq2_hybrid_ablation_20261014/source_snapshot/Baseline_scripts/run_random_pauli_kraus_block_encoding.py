#!/usr/bin/env python3
"""Random Pauli-sum single-Kraus block-encoding benchmark.

Cases:
* 4 system qubits, 20 unique non-identity Pauli terms.
* 8 system qubits, 80 unique non-identity Pauli terms.

For each case, generate 10 random Pauli sums with uniform positive coefficients
``c_i = 1 / term_count``.  Count the complete block-encoding circuit
for the in-house matrix-order route and Qrisp ``BlockEncoding.from_lcu`` after
Qiskit opt0 lowering to NAM and IBM gate sets.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import json_default  # type: ignore  # noqa: E402
from sum_rewritten_kraus_block_counts_qrisp_vs_ours import (  # type: ignore  # noqa: E402
    count_ours_block,
    count_qrisp_from_lcu,
    kraus_effective_terms,
    natural_width,
)


RESULT_ROOT = ROOT / "Baseline_results" / "random_pauli_kraus_block_encoding_v1"


CASES = (
    {"name": "random_pauli_4q_20t", "num_qubits": 4, "num_terms": 20},
    {"name": "random_pauli_8q_80t", "num_qubits": 8, "num_terms": 80},
)


def all_nonidentity_pauli_labels(num_qubits: int) -> list[str]:
    labels = [
        "".join(chars)
        for chars in itertools.product("IXYZ", repeat=num_qubits)
        if any(char != "I" for char in chars)
    ]
    return labels


def random_pauli_matrixsum(num_qubits: int, num_terms: int, seed: int) -> Matrixsum:
    labels = all_nonidentity_pauli_labels(num_qubits)
    if num_terms > len(labels):
        raise ValueError(f"requested {num_terms} terms, but only {len(labels)} non-identity labels exist")

    py_rng = random.Random(seed)
    selected = py_rng.sample(labels, num_terms)
    coeff = 1.0 / float(num_terms)

    instances = []
    for label in selected:
        instances.append((PauliAtom(label, phase=1.0), coeff))
    return Matrixsum(instances)


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
        "terms_preview": [
            {"pauli": label, "coeff": repr(coeff)}
            for label, coeff in terms[: min(8, len(terms))]
        ],
    }


def run_sample(
    case: dict[str, Any],
    sample_index: int,
    seed: int,
    out_dir: Path,
    *,
    workspace: int,
) -> dict[str, Any]:
    ms = random_pauli_matrixsum(case["num_qubits"], case["num_terms"], seed)
    validation = validate_matrixsum(ms)
    if validation["unique_terms"] != validation["term_count"] or validation["has_identity"]:
        raise RuntimeError(f"invalid random Pauli sample: {validation}")

    sample_prefix = f"{case['name']}_sample{sample_index:02d}_seed{seed}"
    sample_dir = out_dir / case["name"] / f"sample{sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ours = count_ours_block(copy.deepcopy(ms), sample_dir, sample_prefix + "_ours")
    qrisp = count_qrisp_from_lcu(copy.deepcopy(ms), sample_dir, sample_prefix + "_qrisp", workspace=workspace)
    return {
        "case": case["name"],
        "sample_index": int(sample_index),
        "seed": int(seed),
        "validation": validation,
        "ours_matrix_order_forced_natural_w": ours,
        "qrisp_from_lcu": qrisp,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def basis_summary(result: dict[str, Any], basis: str) -> dict[str, Any] | None:
    if result.get("status") != "ok":
        return None
    summary = result.get("basis_summary", {}).get(basis)
    if summary is None:
        return None
    return summary


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
        ours_counts = []
        qrisp_counts = []
        ours_noncliff = []
        qrisp_noncliff = []
        ours_widths = []
        qrisp_widths = []
        ours_times = []
        qrisp_times = []
        for sample in case_summary["samples"]:
            ours = sample["ours_matrix_order_forced_natural_w"]
            qrisp = sample["qrisp_from_lcu"]
            ours_summary = basis_summary(ours, basis)
            qrisp_summary = basis_summary(qrisp, basis)
            if ours_summary is not None:
                ours_counts.append(float(ours_summary["gate_count"]))
                ours_noncliff.append(float(ours_summary["non_clifford"]))
                ours_widths.append(float(ours_summary["width"]))
                ours_times.append(float(ours.get("elapsed_seconds", 0.0)))
            if qrisp_summary is not None:
                qrisp_counts.append(float(qrisp_summary["gate_count"]))
                qrisp_noncliff.append(float(qrisp_summary["non_clifford"]))
                qrisp_widths.append(float(qrisp_summary["width"]))
                qrisp_times.append(float(qrisp.get("elapsed_seconds", 0.0)))

        for method, counts, noncliff, widths, times in (
            ("ours_matrix_order_forced_natural_w", ours_counts, ours_noncliff, ours_widths, ours_times),
            ("qrisp_from_lcu", qrisp_counts, qrisp_noncliff, qrisp_widths, qrisp_times),
        ):
            rows.append(
                {
                    "case": case_summary["case"]["name"],
                    "basis": basis,
                    "method": method,
                    "gate_count": aggregate_values(counts),
                    "non_clifford": aggregate_values(noncliff),
                    "width": aggregate_values(widths),
                    "elapsed_seconds": aggregate_values(times),
                    "completed_samples": int(len(counts)),
                    "missing_samples": int(case_summary["num_samples"] - len(counts)),
                }
            )
    for row in rows:
        if row["method"] != "qrisp_from_lcu":
            continue
        ours_row = next(
            candidate
            for candidate in rows
            if candidate["basis"] == row["basis"]
            and candidate["method"] == "ours_matrix_order_forced_natural_w"
        )
        ours_mean = ours_row["gate_count"]["mean"]
        qrisp_mean = row["gate_count"]["mean"]
        row["qrisp_over_ours_mean"] = None if not ours_mean or qrisp_mean is None else float(qrisp_mean / ours_mean)
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def fmt_stat(stat: dict[str, Any], digits: int = 2) -> str:
    if stat["mean"] is None:
        return "-"
    return f"{stat['mean']:.{digits}f} +/- {stat['stdev']:.{digits}f}"


def write_report(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for row in summary["aggregate_rows"]:
        ratio = row.get("qrisp_over_ours_mean")
        rows.append(
            [
                row["case"],
                row["basis"],
                row["method"],
                fmt_stat(row["gate_count"], 2),
                fmt_stat(row["non_clifford"], 2),
                fmt_stat(row["width"], 2),
                fmt_stat(row["elapsed_seconds"], 3),
                f"{row['completed_samples']}/{row['missing_samples']}",
                "" if ratio is None else f"{ratio:.6g}",
            ]
        )

    validation_rows = []
    for case_summary in summary["cases"]:
        validations = [sample["validation"] for sample in case_summary["samples"]]
        validation_rows.append(
            [
                case_summary["case"]["name"],
                case_summary["case"]["num_qubits"],
                case_summary["case"]["num_terms"],
                min(v["unique_terms"] for v in validations),
                max(v["unique_terms"] for v in validations),
                any(v["has_identity"] for v in validations),
                fmt_stat(aggregate_values([v["l1_norm"] for v in validations]), 6),
                validations[0]["natural_width"],
            ]
        )

    lines = [
        "# Random Pauli-Sum Kraus Block-Encoding",
        "",
        f"- samples per case: `{summary['arguments']['samples']}`",
        f"- base seed: `{summary['arguments']['seed']}`",
        "- Pauli labels are sampled without replacement from non-identity labels.",
        "- Coefficients are uniform positive values `c_i = 1 / term_count`, so Qrisp `from_lcu` receives native nonnegative LCU coefficients.",
        "- Ours uses forced natural matrix-order width `ceil(log2(term_count))`; Qrisp uses `BlockEncoding.from_lcu`.",
        "",
        markdown_table(
            [
                "case",
                "basis",
                "method",
                "gates mean +/- sd",
                "non-Cliff mean +/- sd",
                "width mean +/- sd",
                "time mean +/- sd (s)",
                "completed/missing",
                "qrisp/ours",
            ],
            rows,
        ),
        "",
        "## Validation",
        "",
        markdown_table(
            [
                "case",
                "qubits",
                "terms requested",
                "min unique",
                "max unique",
                "any identity",
                "L1 norm mean +/- sd",
                "natural width",
            ],
            validation_rows,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--workspace", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "experiment": "random_pauli_kraus_block_encoding",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "cases": [],
        "aggregate_rows": [],
    }

    for case in CASES:
        case_summary = {
            "case": case,
            "num_samples": int(args.samples),
            "samples": [],
        }
        for sample_index in range(args.samples):
            seed = int(args.seed + 100000 * CASES.index(case) + sample_index)
            sample = run_sample(case, sample_index, seed, out_dir, workspace=args.workspace)
            case_summary["samples"].append(sample)
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
        row for case_summary in summary["cases"] for row in aggregate_case(case_summary)
    ]
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary["aggregate_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
