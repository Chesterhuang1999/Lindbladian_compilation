#!/usr/bin/env python3
"""Dense 4-qubit random Pauli block-encoding benchmark.

Cases:
* 64, 129, 200 unique non-identity 4-qubit Pauli terms, 5 random seeds each;
* 256 terms, the complete 4-qubit Pauli group including identity, 1 sample.

By default, coefficients are positive random values normalized to L1 norm 1.
Use ``--coeff-mode uniform`` to reproduce the earlier equal-coefficient run.

Both tools receive the same Matrixsum.  Ours uses the current
``block_encoding_new.BlockEncoding.circuit(opt="Matrix-order")`` path, while
Qrisp uses legal ``BlockEncoding.from_lcu`` input: nonnegative coefficients
and phase-bearing Pauli unitaries.
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
from run_table3_extra_block_encoding_cases import (  # type: ignore  # noqa: E402
    count_ours_block_encoding_new,
    count_qrisp_from_lcu_phase_unitaries,
)
from sum_rewritten_kraus_block_counts_qrisp_vs_ours import (  # type: ignore  # noqa: E402
    kraus_effective_terms,
    natural_width,
)


RESULT_ROOT = ROOT / "Baseline_results" / "dense_random_pauli4_block_encoding_random_coeff_v1"
RESULTS_MD = ROOT / "Results" / "dense_random_pauli4_random_coeff_ours_qrisp.md"
RESULTS_TEX = ROOT / "Results" / "dense_random_pauli4_random_coeff_ours_qrisp.tex"


CASES = (
    {"name": "random_pauli_4q_64t", "num_qubits": 4, "num_terms": 64, "samples": 5, "full": False},
    {"name": "random_pauli_4q_129t", "num_qubits": 4, "num_terms": 129, "samples": 5, "full": False},
    {"name": "random_pauli_4q_200t", "num_qubits": 4, "num_terms": 200, "samples": 5, "full": False},
    {"name": "full_pauli_4q_256t", "num_qubits": 4, "num_terms": 256, "samples": 1, "full": True},
)


def pauli_labels(num_qubits: int, *, include_identity: bool) -> list[str]:
    labels = ["".join(chars) for chars in itertools.product("IXYZ", repeat=num_qubits)]
    if include_identity:
        return labels
    return [label for label in labels if any(char != "I" for char in label)]


def coefficients_for_labels(labels: list[str], *, coeff_mode: str, rng: random.Random) -> list[float]:
    if coeff_mode == "uniform":
        return [1.0 / float(len(labels)) for _label in labels]
    if coeff_mode == "random-positive":
        weights = [rng.random() for _label in labels]
        total = sum(weights)
        if total == 0:
            raise ValueError("random coefficient total is zero")
        return [weight / total for weight in weights]
    raise ValueError(f"unsupported coeff mode: {coeff_mode!r}")


def matrixsum_from_labels(labels: list[str], *, coeff_mode: str, rng: random.Random) -> Matrixsum:
    coeffs = coefficients_for_labels(labels, coeff_mode=coeff_mode, rng=rng)
    return Matrixsum(
        [
            (PauliAtom(label, phase=1.0), coeff)
            for label, coeff in zip(labels, coeffs, strict=True)
        ]
    )


def random_matrixsum(num_qubits: int, num_terms: int, seed: int, *, coeff_mode: str) -> Matrixsum:
    labels = pauli_labels(num_qubits, include_identity=False)
    if num_terms > len(labels):
        raise ValueError(f"requested {num_terms} non-identity terms, but only {len(labels)} exist")
    rng = random.Random(seed)
    return matrixsum_from_labels(rng.sample(labels, num_terms), coeff_mode=coeff_mode, rng=rng)


def full_matrixsum(num_qubits: int, seed: int, *, coeff_mode: str) -> Matrixsum:
    rng = random.Random(seed)
    return matrixsum_from_labels(
        pauli_labels(num_qubits, include_identity=True),
        coeff_mode=coeff_mode,
        rng=rng,
    )


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
    workspace: int,
    *,
    coeff_mode: str,
) -> dict[str, Any]:
    if case["full"]:
        ms = full_matrixsum(case["num_qubits"], seed, coeff_mode=coeff_mode)
    else:
        ms = random_matrixsum(case["num_qubits"], case["num_terms"], seed, coeff_mode=coeff_mode)

    validation = validate_matrixsum(ms)
    if validation["unique_terms"] != validation["term_count"]:
        raise RuntimeError(f"duplicate Pauli label detected: {validation}")
    if not case["full"] and validation["has_identity"]:
        raise RuntimeError(f"non-full random case unexpectedly contains identity: {validation}")

    sample_prefix = f"{case['name']}_sample{sample_index:02d}_seed{seed}"
    sample_dir = out_dir / case["name"] / f"sample{sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ours = count_ours_block_encoding_new(copy.deepcopy(ms), sample_dir, sample_prefix + "_ours")
    qrisp = count_qrisp_from_lcu_phase_unitaries(copy.deepcopy(ms), sample_dir, sample_prefix + "_qrisp", workspace)
    return {
        "case": case["name"],
        "sample_index": int(sample_index),
        "seed": int(seed),
        "coeff_mode": coeff_mode,
        "validation": validation,
        "ours_block_encoding_new_current": ours,
        "qrisp_from_lcu": qrisp,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def basis_summary(result: dict[str, Any], basis: str) -> dict[str, Any] | None:
    if result.get("status") != "ok":
        return None
    return result.get("basis_summary", {}).get(basis)


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
        values: dict[str, dict[str, list[float]]] = {
            "ours_block_encoding_new_current": {"gates": [], "non_clifford": [], "width": [], "time": []},
            "qrisp_from_lcu": {"gates": [], "non_clifford": [], "width": [], "time": []},
        }
        for sample in case_summary["samples"]:
            for method, result_key in (
                ("ours_block_encoding_new_current", "ours_block_encoding_new_current"),
                ("qrisp_from_lcu", "qrisp_from_lcu"),
            ):
                result = sample[result_key]
                summary = basis_summary(result, basis)
                if summary is None:
                    continue
                values[method]["gates"].append(float(summary["gate_count"]))
                values[method]["non_clifford"].append(float(summary["non_clifford"]))
                values[method]["width"].append(float(summary["width"]))
                values[method]["time"].append(float(result.get("elapsed_seconds", 0.0)))

        for method, buckets in values.items():
            rows.append(
                {
                    "case": case_summary["case"]["name"],
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
        if row["method"] != "qrisp_from_lcu":
            continue
        ours_row = next(
            candidate
            for candidate in rows
            if candidate["case"] == row["case"]
            and candidate["basis"] == row["basis"]
            and candidate["method"] == "ours_block_encoding_new_current"
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

    validation_rows = []
    for case_summary in summary["cases"]:
        validations = [sample["validation"] for sample in case_summary["samples"]]
        validation_rows.append(
            [
                case_summary["case"]["name"],
                case_summary["case"]["num_terms"],
                case_summary["num_samples"],
                min(v["unique_terms"] for v in validations),
                max(v["unique_terms"] for v in validations),
                any(v["has_identity"] for v in validations),
                validations[0]["natural_width"],
            ]
        )

    lines = [
        "# Dense 4q Random Pauli Block-Encoding",
        "",
        f"- base seed: `{summary['arguments']['seed']}`",
        "- Random cases sample unique non-identity Pauli labels without replacement.",
        "- The 256-term full case is the complete 4-qubit Pauli group including identity, with seeded coefficients.",
        f"- coefficient mode: `{summary['arguments']['coeff_mode']}`.",
        "- For `random-positive`, coefficients are sampled as `c_i ~ Uniform(0, 1)` and L1-normalized to `sum_i c_i = 1`.",
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
            ["case", "terms", "samples", "min unique", "max unique", "any identity", "natural width"],
            validation_rows,
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
        "\\caption{Dense 4-qubit random Pauli block-encoding comparison. Cells report mean gates / non-Clifford / width after Qiskit opt0 lowering; the full 256-term Pauli row has one deterministic sample.}",
        "\\label{tab:dense-random-pauli4-qrisp}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Case & Basis & Ours & Qrisp & Qrisp/Ours & Ours NC & Qrisp NC & Widths \\\\",
        "\\midrule",
    ]
    for case in CASES:
        for basis in ("ibm", "nam"):
            ours = rows_by_key[(case["name"], basis, "ours_block_encoding_new_current")]
            qrisp = rows_by_key[(case["name"], basis, "qrisp_from_lcu")]
            ratio = qrisp.get("qrisp_over_ours_mean")
            ours_cell = fmt_stat(ours["gate_count"], 1).replace("+/-", "$\\pm$")
            qrisp_cell = fmt_stat(qrisp["gate_count"], 1).replace("+/-", "$\\pm$")
            ours_nc = fmt_stat(ours["non_clifford"], 1).replace("+/-", "$\\pm$")
            qrisp_nc = fmt_stat(qrisp["non_clifford"], 1).replace("+/-", "$\\pm$")
            ours_width = fmt_stat(ours["width"], 1).replace("+/-", "$\\pm$")
            qrisp_width = fmt_stat(qrisp["width"], 1).replace("+/-", "$\\pm$")
            widths = f"{ours_width}/{qrisp_width}"
            lines.append(
                f"{tex_escape(case['name'])} & {basis.upper()} & {ours_cell} & {qrisp_cell} & "
                f"{'' if ratio is None else f'{ratio:.4g}'} & {ours_nc} & {qrisp_nc} & {widths} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--workspace", type=int, default=0)
    parser.add_argument(
        "--coeff-mode",
        choices=("random-positive", "uniform"),
        default="random-positive",
        help="Coefficient model. random-positive samples c_i ~ U(0,1) and L1-normalizes.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": "dense_random_pauli4_block_encoding",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "cases": [],
        "aggregate_rows": [],
    }

    for case_index, case in enumerate(CASES):
        case_summary = {
            "case": case,
            "num_samples": int(case["samples"]),
            "samples": [],
        }
        for sample_index in range(case["samples"]):
            seed = int(args.seed + 100000 * case_index + sample_index)
            sample = run_sample(
                case,
                sample_index,
                seed,
                out_dir,
                workspace=args.workspace,
                coeff_mode=args.coeff_mode,
            )
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
        row for case_item in summary["cases"] for row in aggregate_case(case_item)
    ]
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    write_report(summary, RESULTS_MD)
    write_tex(summary, RESULTS_TEX)
    print(f"summary: {summary_path}")
    print(f"report: {out_dir / 'report.md'}")
    print(f"results md: {RESULTS_MD}")
    print(f"results tex: {RESULTS_TEX}")
    print(json.dumps(summary["aggregate_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
