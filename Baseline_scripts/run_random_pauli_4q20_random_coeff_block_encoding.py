#!/usr/bin/env python3
"""4q/20-term random Pauli block-encoding with random positive coefficients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_dense_random_pauli4_block_encoding import (  # type: ignore  # noqa: E402
    aggregate_case,
    fmt_stat,
    markdown_table,
    run_sample,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import json_default  # type: ignore  # noqa: E402


RESULT_ROOT = ROOT / "Baseline_results" / "random_pauli_4q20_block_encoding_random_coeff_v1"
RESULTS_MD = ROOT / "Results" / "random_pauli_4q20_random_coeff_ours_qrisp.md"

CASE = {
    "name": "random_pauli_4q_20t",
    "num_qubits": 4,
    "num_terms": 20,
    "samples": 10,
    "full": False,
}


def write_report(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for row in summary["aggregate_rows"]:
        ratio = row.get("qrisp_over_ours_mean")
        ratio_nc = row.get("qrisp_over_ours_non_clifford_mean")
        rows.append(
            [
                row["case"],
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

    validations = [sample["validation"] for sample in summary["cases"][0]["samples"]]
    validation_rows = [
        [
            CASE["name"],
            CASE["num_terms"],
            summary["arguments"]["samples"],
            min(v["unique_terms"] for v in validations),
            max(v["unique_terms"] for v in validations),
            any(v["has_identity"] for v in validations),
            fmt_stat(
                {
                    "mean": sum(v["l1_norm"] for v in validations) / len(validations),
                    "count": len(validations),
                    "stdev": 0.0,
                },
                6,
            ),
            validations[0]["natural_width"],
        ]
    ]

    lines = [
        "# Random Pauli 4q/20t Block-Encoding With Random Coefficients",
        "",
        f"- samples: `{summary['arguments']['samples']}`",
        f"- base seed: `{summary['arguments']['seed']}`",
        "- Pauli labels are sampled without replacement from non-identity 4-qubit Paulis.",
        "- Coefficients are sampled as `c_i ~ Uniform(0, 1)` and L1-normalized to `sum_i c_i = 1`.",
        "- Ours uses current `block_encoding_new.BlockEncoding.circuit(opt=\"Matrix-order\")`; Qrisp uses legal nonnegative `from_lcu` input.",
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
        "## Validation",
        "",
        markdown_table(
            ["case", "terms", "samples", "min unique", "max unique", "any identity", "L1 norm", "natural width"],
            validation_rows,
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "experiment": "random_pauli_4q20_block_encoding_random_coeff",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "cases": [],
        "aggregate_rows": [],
    }

    case_summary = {
        "case": {**CASE, "samples": int(args.samples)},
        "num_samples": int(args.samples),
        "samples": [],
    }
    for sample_index in range(args.samples):
        seed = int(args.seed + sample_index)
        sample = run_sample(
            CASE,
            sample_index,
            seed,
            out_dir,
            workspace=args.workspace,
            coeff_mode="random-positive",
        )
        case_summary["samples"].append(sample)
        (out_dir / "summary.partial.json").write_text(
            json.dumps(summary, indent=2, default=json_default),
            encoding="utf-8",
        )

    summary["cases"].append(case_summary)
    summary["aggregate_rows"] = aggregate_case(case_summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    write_report(summary, RESULTS_MD)
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"report: {out_dir / 'report.md'}")
    print(f"results md: {RESULTS_MD}")
    print(json.dumps(summary["aggregate_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
