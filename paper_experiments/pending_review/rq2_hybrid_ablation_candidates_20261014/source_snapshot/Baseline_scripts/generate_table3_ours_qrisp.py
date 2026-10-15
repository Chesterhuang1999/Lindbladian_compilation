#!/usr/bin/env python3
"""Generate Results/Table3_ours_Qrisp.{md,tex} from saved block-encoding runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"


def load_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{int(round(float(value))):,}"


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_stat(mean: float | None, stdev: float | None, digits: int = 1) -> str:
    if mean is None:
        return "-"
    if stdev is None:
        stdev = 0.0
    return f"{mean:.{digits}f} +/- {stdev:.{digits}f}"


def cell(gates: Any, non_clifford: Any, width: Any, *, stat: bool = False) -> str:
    if gates is None:
        return "-"
    if stat:
        return f"{gates} / {non_clifford} / {width}"
    return f"{fmt_int(gates)} / {fmt_int(non_clifford)} / {fmt_float(width, 0)}"


def method_name(method: str) -> str:
    if method.startswith("ours"):
        return "ours"
    if method.startswith("qrisp"):
        return "qrisp"
    raise ValueError(method)


def rows_from_rewritten() -> list[dict[str, Any]]:
    summary = load_json("Baseline_results/rewritten_kraus_block_sum_qrisp_vs_ours_v1/summary.json")
    selected = {
        "HRW_8": "sum rewritten Kraus",
        "HRW_28": "sum rewritten Kraus",
        "random_coloring_linear_n4_q4": "sum rewritten Kraus",
        "random_coloring_cycle_n4_q4": "sum rewritten Kraus",
        "repetition_dqe_d3": "sum rewritten Kraus",
    }
    labels = {
        "HRW_8": "HRW_8",
        "HRW_28": "HRW_28",
        "random_coloring_linear_n4_q4": "random coloring linear",
        "random_coloring_cycle_n4_q4": "random coloring cycle",
        "repetition_dqe_d3": "repetition DQE d=3",
    }
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary["comparison_rows"]:
        bench = row["benchmark"]
        if bench not in selected:
            continue
        key = (bench, row["basis"].upper())
        grouped.setdefault(
            key,
            {
                "example": labels[bench],
                "scope": selected[bench],
                "basis": row["basis"].upper(),
            },
        )
        grouped[key][method_name(row["method"])] = {
            "gates": row["gate_count_sum"],
            "non_clifford": row["non_clifford_sum"],
            "width": row["max_width"],
        }
        if row.get("qrisp_over_ours") is not None:
            grouped[key]["ratio"] = row["qrisp_over_ours"]

    order = [
        ("HRW_8", "NAM"),
        ("HRW_8", "IBM"),
        ("HRW_28", "NAM"),
        ("HRW_28", "IBM"),
        ("random_coloring_linear_n4_q4", "NAM"),
        ("random_coloring_linear_n4_q4", "IBM"),
        ("random_coloring_cycle_n4_q4", "NAM"),
        ("random_coloring_cycle_n4_q4", "IBM"),
        ("repetition_dqe_d3", "NAM"),
        ("repetition_dqe_d3", "IBM"),
    ]
    return [grouped[key] for key in order]


def rows_from_random_pauli() -> list[dict[str, Any]]:
    summary = load_json("Baseline_results/random_pauli_kraus_block_encoding_v1/summary.json")
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary["aggregate_rows"]:
        if row["case"] != "random_pauli_4q_20t":
            continue
        key = (row["case"], row["basis"].upper())
        grouped.setdefault(
            key,
            {
                "example": "4q random Pauli, 20 terms",
                "scope": "10-sample mean",
                "basis": row["basis"].upper(),
                "stat": True,
            },
        )
        grouped[key][method_name(row["method"])] = {
            "gates": fmt_stat(row["gate_count"]["mean"], row["gate_count"]["stdev"], 1),
            "non_clifford": fmt_stat(row["non_clifford"]["mean"], row["non_clifford"]["stdev"], 1),
            "width": fmt_stat(row["width"]["mean"], row["width"]["stdev"], 1),
        }
        if row.get("qrisp_over_ours_mean") is not None:
            grouped[key]["ratio"] = row["qrisp_over_ours_mean"]
    return [grouped[("random_pauli_4q_20t", basis)] for basis in ("NAM", "IBM")]


def rows_from_tfim_extra() -> list[dict[str, Any]]:
    summary = load_json("Baseline_results/table3_extra_block_encoding_cases_v1/summary.partial.json")
    selected = {
        "TFIM_3_A0": "A0 block encoding",
        "TFIM_8_A0": "A0 block encoding",
    }
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary["comparison_rows"]:
        if row["case"] not in selected:
            continue
        key = (row["case"], row["basis"].upper())
        grouped.setdefault(
            key,
            {
                "example": row["case"].replace("_A0", " A0"),
                "scope": selected[row["case"]],
                "basis": row["basis"].upper(),
            },
        )
        grouped[key][method_name(row["method"])] = {
            "gates": row["gate_count"],
            "non_clifford": row["non_clifford"],
            "width": row["width"],
        }
        if row.get("qrisp_over_ours") is not None:
            grouped[key]["ratio"] = row["qrisp_over_ours"]
    order = [
        ("TFIM_3_A0", "NAM"),
        ("TFIM_3_A0", "IBM"),
        ("TFIM_8_A0", "NAM"),
        ("TFIM_8_A0", "IBM"),
    ]
    return [grouped[key] for key in order]


def rows_from_full_pauli() -> list[dict[str, Any]]:
    summary = load_json("Baseline_results/full_pauli_new_block_encoding_n4_n6_v1/summary.json")
    rows = []
    selected = ("full_pauli_n4", "full_pauli_n6")
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary["comparison_rows"]:
        if row["case"] not in selected:
            continue
        key = (row["case"], row["basis"].upper())
        grouped.setdefault(
            key,
            {
                "example": row["case"].replace("_", " "),
                "scope": "full Pauli, uniform",
                "basis": row["basis"].upper(),
            },
        )
        if row["method"].startswith("ours"):
            grouped[key]["ours"] = {
                "gates": row["gate_count"],
                "non_clifford": row["non_clifford"],
                "width": row["width"],
            }
        elif row["status"] == "ok":
            grouped[key]["qrisp"] = {
                "gates": row["gate_count"],
                "non_clifford": row["non_clifford"],
                "width": row["width"],
            }
            grouped[key]["ratio"] = row["qrisp_over_ours"]
        else:
            grouped[key]["qrisp"] = None
            grouped[key]["qrisp_status"] = "DNF"
    for case in selected:
        for basis in ("NAM", "IBM"):
            rows.append(grouped[(case, basis)])
    return rows


def build_rows() -> list[dict[str, Any]]:
    return (
        rows_from_rewritten()
        + rows_from_tfim_extra()
        + rows_from_random_pauli()
        + rows_from_full_pauli()
    )


def md_cell_for_method(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        return data.get(f"{key}_status", "-")
    return cell(
        value["gates"],
        value["non_clifford"],
        value["width"],
        stat=bool(data.get("stat")),
    )


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Table 3: Ours vs Qrisp Block-Encoding Counts",
        "",
        "Cells report `gates / non-Clifford / width` after Qiskit opt0 lowering to the named basis.",
        "",
        "| example | scope | basis | ours | Qrisp | Qrisp/ours |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        ratio = row.get("ratio")
        lines.append(
            f"| {row['example']} | {row['scope']} | {row['basis']} | "
            f"{md_cell_for_method(row, 'ours')} | {md_cell_for_method(row, 'qrisp')} | "
            f"{'' if ratio is None else f'{ratio:.3g}'} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- HRW, random coloring, and repetition DQE rows sum block-encoding counts over all nonempty rewritten Kraus operators.",
            "- TFIM rows are the first nonempty Kraus operator A0.",
            "- The 4q random row is the mean over 10 unique, non-identity 20-term Pauli samples with uniform positive coefficients.",
            "- Full-Pauli rows use the generic `block_encoding_new` matrix-order interface for ours. Qrisp receives the literal `BlockEncoding.from_lcu` list.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def tex_num_cell(text: str) -> str:
    if text in {"-", "DNF"}:
        return text
    return tex_escape(text).replace("+/-", "$\\pm$")


def write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Ours vs. Qrisp block-encoding counts. Cells are gates / non-Clifford / width after Qiskit opt0 lowering.}",
        "\\label{tab:ours-qrisp-block-encoding}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Example & Basis & Ours & Qrisp & Qrisp/Ours \\\\",
        "\\midrule",
    ]
    for row in rows:
        ratio = row.get("ratio")
        lines.append(
            f"{tex_escape(row['example'])} & {row['basis']} & "
            f"{tex_num_cell(md_cell_for_method(row, 'ours'))} & "
            f"{tex_num_cell(md_cell_for_method(row, 'qrisp'))} & "
            f"{'' if ratio is None else f'{ratio:.3g}'} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.5ex}",
            "\\begin{minipage}{0.98\\linewidth}",
            "\\footnotesize\\emph{Notes.} HRW, random coloring, and repetition DQE sum over nonempty rewritten Kraus operators. TFIM rows use A0. The random-Pauli row is a 10-sample mean. Full-Pauli rows use the generic block\\_encoding\\_new matrix-order interface for ours and Qrisp receives the literal from\\_lcu list.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_markdown(rows, RESULTS / "Table3_ours_Qrisp.md")
    write_latex(rows, RESULTS / "Table3_ours_Qrisp.tex")
    print(f"wrote {RESULTS / 'Table3_ours_Qrisp.md'}")
    print(f"wrote {RESULTS / 'Table3_ours_Qrisp.tex'}")


if __name__ == "__main__":
    main()
