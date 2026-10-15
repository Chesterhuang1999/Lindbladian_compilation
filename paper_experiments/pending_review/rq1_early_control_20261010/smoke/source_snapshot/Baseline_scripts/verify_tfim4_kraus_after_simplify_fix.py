#!/usr/bin/env python3
"""Verify all TFIM-4 Kraus Matrixsum terms after the simplify() phase fix."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for candidate in (SRC_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: E402
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore  # noqa: E402


OUT_DIR = ROOT / "Baseline_results" / "tfim4_kraus_verify_after_simplify_fix_v1"


def complex_record(value: complex, tol: float = 1e-12) -> dict[str, Any]:
    z = complex(value)
    real = 0.0 if abs(z.real) < tol else float(z.real)
    imag = 0.0 if abs(z.imag) < tol else float(z.imag)
    return {"real": real, "imag": imag, "repr": repr(complex(real, imag))}


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


def coeff_dict_from_matrixsum(ms: Matrixsum) -> dict[str, complex]:
    out: dict[str, complex] = defaultdict(complex)
    for atom, coeff in ms.instances:
        if not isinstance(atom, PauliAtom):
            raise TypeError("TFIM verification expects PauliAtom-only Matrixsum")
        out[atom.expr] += complex(coeff) * complex(atom.phase)
    return dict(out)


def dense_from_coeffs(coeffs: dict[str, complex]) -> np.ndarray:
    total = None
    for label, coeff in coeffs.items():
        mat = SparsePauliOp.from_list([(label, coeff)]).to_matrix()
        total = mat if total is None else total + mat
    if total is None:
        return np.zeros((1, 1), dtype=complex)
    return np.asarray(total, dtype=complex)


def expected_tfim4_coeffs(delta_t: float = 0.1) -> list[dict[str, complex]]:
    n = 4
    gamma = np.sqrt(0.1) / 2
    sqdelta = np.sqrt(delta_t)

    a0: dict[str, complex] = defaultdict(complex)
    a0["I" * n] += 1.0

    for i in range(n):
        z_ind = [i, (i + 1) % n]
        z_str = "".join("Z" if j in z_ind else "I" for j in range(n))
        x_str = "".join("X" if j == i else "I" for j in range(n))
        y_str = "".join("Y" if j == i else "I" for j in range(n))

        # H contains -Z_i Z_{i+1} and -X_i, so -i dt H contributes +i dt.
        a0[z_str] += 1j * delta_t
        a0[x_str] += 1j * delta_t

        # L_i = gamma (X_i - i Y_i), so -dt/2 L_i^\dag L_i
        # contributes -dt gamma^2 I and -dt gamma^2 Z_i.
        a0["I" * n] += -delta_t * gamma * gamma
        a0[z_str_single := "".join("Z" if j == i else "I" for j in range(n))] += (
            -delta_t * gamma * gamma
        )
        _ = y_str  # keep formula correspondence explicit above

    expected = [dict(a0)]
    for i in range(n):
        x_str = "".join("X" if j == i else "I" for j in range(n))
        y_str = "".join("Y" if j == i else "I" for j in range(n))
        expected.append({x_str: sqdelta * gamma, y_str: -1j * sqdelta * gamma})
    return expected


def coeff_diff_report(
    actual: dict[str, complex],
    expected: dict[str, complex],
    tol: float = 1e-10,
) -> dict[str, Any]:
    labels = sorted(set(actual) | set(expected))
    rows = []
    max_abs_diff = 0.0
    for label in labels:
        a = actual.get(label, 0.0)
        e = expected.get(label, 0.0)
        diff = a - e
        max_abs_diff = max(max_abs_diff, abs(diff))
        if abs(a) > tol or abs(e) > tol or abs(diff) > tol:
            rows.append(
                {
                    "pauli": label,
                    "actual": complex_record(a),
                    "expected": complex_record(e),
                    "diff": complex_record(diff),
                    "abs_diff": float(abs(diff)),
                }
            )
    return {
        "passed": bool(max_abs_diff <= tol),
        "max_abs_coeff_diff": float(max_abs_diff),
        "rows": rows,
    }


def pauli_expression(coeffs: dict[str, complex]) -> str:
    pieces = []
    for label in sorted(coeffs):
        coeff = coeffs[label]
        if abs(coeff) > 1e-12:
            pieces.append(f"({complex_cell(coeff)}) {label}")
    return " + ".join(pieces) if pieces else "0"


def verify() -> dict[str, Any]:
    kraus_ops, model_info = build_periodic_tfim_channel(4, 0.1)
    expected = expected_tfim4_coeffs(0.1)
    if len(kraus_ops) != len(expected):
        raise AssertionError(f"expected {len(expected)} Kraus ops, got {len(kraus_ops)}")

    rows = []
    all_passed = True
    for index, (ms, expected_coeffs) in enumerate(zip(kraus_ops, expected)):
        actual_coeffs = coeff_dict_from_matrixsum(ms)
        coeff_report = coeff_diff_report(actual_coeffs, expected_coeffs)
        actual_dense = dense_from_coeffs(actual_coeffs)
        expected_dense = dense_from_coeffs(expected_coeffs)
        dense_diff = float(np.linalg.norm(actual_dense - expected_dense))
        dense_passed = dense_diff <= 1e-10
        all_passed = all_passed and coeff_report["passed"] and dense_passed
        rows.append(
            {
                "kraus_index": int(index),
                "term_count": int(len(actual_coeffs)),
                "actual_expression": pauli_expression(actual_coeffs),
                "expected_expression": pauli_expression(expected_coeffs),
                "coeff_check": coeff_report,
                "dense_diff_norm": dense_diff,
                "dense_check_passed": bool(dense_passed),
            }
        )

    return {
        "experiment": "verify_tfim4_kraus_after_simplify_fix",
        "benchmark": "TFIM_4",
        "delta_t": 0.1,
        "model_info": model_info,
        "kraus_count": int(len(kraus_ops)),
        "all_passed": bool(all_passed),
        "kraus": rows,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TFIM_4 Kraus Verification After simplify() Phase Fix",
        "",
        f"- All checks passed: `{summary['all_passed']}`",
        f"- Kraus count: `{summary['kraus_count']}`",
        "",
        "| Kraus | Terms | Max coeff diff | Dense diff norm | Passed |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["kraus"]:
        passed = row["coeff_check"]["passed"] and row["dense_check_passed"]
        lines.append(
            f"| {row['kraus_index']} | {row['term_count']} | "
            f"{row['coeff_check']['max_abs_coeff_diff']:.3e} | "
            f"{row['dense_diff_norm']:.3e} | {str(passed).lower()} |"
        )
    lines.append("")
    for row in summary["kraus"]:
        lines.extend(
            [
                f"## Kraus {row['kraus_index']}",
                "",
                "Actual:",
                "",
                "```text",
                row["actual_expression"],
                "```",
                "",
                "Expected:",
                "",
                "```text",
                row["expected_expression"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- JSON summary: `{summary['summary_path']}`",
            f"- Output directory: `{summary['out_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = verify()
    summary["out_dir"] = str(OUT_DIR.resolve())
    summary_path = OUT_DIR / "summary.json"
    report_path = OUT_DIR / "summary.md"
    summary["summary_path"] = str(summary_path.resolve())
    summary["report_path"] = str(report_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": summary["summary_path"], "report": summary["report_path"]}, indent=2))
    print(markdown(summary))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
