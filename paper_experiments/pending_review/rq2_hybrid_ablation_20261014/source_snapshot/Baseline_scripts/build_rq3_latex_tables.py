"""Export the validated SELECT and aligned QSVT comparisons as LaTeX tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECT_SOURCE = ROOT / "Baseline_results/rq3_refresh_20261013/comparison.json"
QSVT_SOURCE = ROOT / "Baseline_results/rq3_qrisp_aligned_qsvt_20261014/comparison.json"
TABLE_DIRECTORY = ROOT.parent / "nonudymics/Sources/Tables"

QSVT_NAMES = {
    "resource_three_term": "Three-term Pauli sum",
    "h2_molecular_api": r"H$_2$ (STO-3G)",
    "hubbard_chain2_api": "Hubbard (2 sites)",
    "heisenberg_square4_api": r"Heisenberg ($2\times2$)",
    "maxcut_triangle3": "MaxCut (triangle)",
    "cluster3_weighted": "Weighted cluster (3 qubits)",
    "maxcut_cycle4": "MaxCut (4-cycle)",
    "qrisp_magnetization_32q": "Magnetization (32 qubits)",
    "qrisp_ising_zero_field_chain33": "Zero-field Ising (33 qubits)",
}


def number(value):
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def triple(metric, *, baseline=False):
    keys = ("gate_count", "non_clifford", "width") if baseline else ("G", "NC", "W")
    return " / ".join(number(metric[key]) for key in keys)


def cells(name, methods, ratios):
    return " & ".join([name, *methods, *(number(value) for value in ratios)]) + r" \\"


def table(caption, label, rows, source):
    return "\n".join([
        "% Requires \\usepackage{booktabs,graphicx}.",
        f"% Data: {source.relative_to(ROOT.parent)} (relative to the shared project directory).",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.18}",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"Benchmark & \shortstack{PennyLane (PL)\\$G/\mathrm{NC}/W$}"
        r" & \shortstack{Qrisp\\$G/\mathrm{NC}/W$}"
        r" & \shortstack{Ours\\$G/\mathrm{NC}/W$} & PL/Ours & Qrisp/Ours \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}%", "}",
        r"\end{table*}", "",
    ])


def main(output_directory):
    select = json.loads(SELECT_SOURCE.read_text())
    qsvt = json.loads(QSVT_SOURCE.read_text())
    assert select["passed"] and qsvt["passed"]
    select_rows = [row for row in select["select_rows"] if row["basis"] == "nam"]
    assert len(select_rows) == 10
    rows = []
    for index, row in enumerate(select_rows):
        if index == 5:
            rows.append(r"\midrule")
        name = row["case"].replace("random_coloring_", "coloring_", 1)
        name = r"\texttt{" + name.replace("_", r"\_") + "}"
        ratios = [row["pl"]["gate_count"] / row["ours"]["G"],
                  row["qrisp"]["gate_count"] / row["ours"]["G"]]
        assert math.isclose(ratios[0], row["pl_over_ours"], rel_tol=1e-12)
        assert math.isclose(ratios[1], row["qrisp_over_ours"], rel_tol=1e-12)
        rows.append(cells(name, [triple(row["pl"], baseline=True),
            triple(row["qrisp"], baseline=True), triple(row["ours"])], ratios))
    select_caption = (
        r"Controlled Pauli selection (SELECT) in the NAM basis $\{R_z,H,X,\mathrm{CNOT}\}$, excluding state preparation. "
        r"After lowering, entries give total gates $G$, non-Clifford gates $\mathrm{NC}$, and total qubits $W$ including work qubits. "
        r"For the first five benchmarks, gate counts sum over SELECT circuits for nonzero Kraus operators, and $W$ is their maximum. "
        r"Random-Pauli entries are ten-sample means. Ratios use the reported $G$ (baseline/Ours); values above 1 favor Ours."
    )
    select_text = table(select_caption, "tab:rq3-select-comparison", rows, SELECT_SOURCE)
    assert len(qsvt["qsvt_rows"]) == 9
    assert [row["id"] for row in qsvt["qsvt_rows"]] == list(QSVT_NAMES)
    rows = []
    for index, basis in enumerate(("nam", "ibm")):
        if index:
            rows.append(r"\midrule")
        gates = r"\{R_z,H,X,\mathrm{CNOT}\}" if basis == "nam" else r"\{u_1,u_2,u_3,\mathrm{CNOT}\}"
        rows.append(r"\multicolumn{6}{@{}l}{\textit{" + basis.upper() + r" basis} $" + gates + r"$} \\[2pt]")
        for row in qsvt["qsvt_rows"]:
            metrics = row["metrics"]
            ratios = [metrics["pl_native"][basis]["G"] / metrics["ours"][basis]["G"],
                      metrics["qrisp_native"][basis]["G"] / metrics["ours"][basis]["G"]]
            assert math.isclose(ratios[0], row["ratios"][basis]["pl_over_ours"], rel_tol=1e-12)
            assert math.isclose(ratios[1], row["ratios"][basis]["qrisp_over_ours"], rel_tol=1e-12)
            rows.append(cells(QSVT_NAMES[row["id"]],
                [triple(metrics[method][basis]) for method in ("pl_native", "qrisp_native", "ours")], ratios))
    qsvt_caption = (
        r"Quantum singular value transformation (QSVT) resources for $P(x)=-x+x^3/2+x^5/2$. "
        r"Each circuit uses QSVT controlled on $|1\rangle$ and its adjoint controlled on $|0\rangle$, enclosed by Hadamards on their shared control qubit (ten block-encoding calls). "
        r"After lowering, entries give total gates $G$, non-Clifford gates $\mathrm{NC}$, and total qubits $W$ including work qubits. "
        r"Ratios are $G_{\mathrm{baseline}}/G_{\mathrm{Ours}}$; values above 1 favor Ours."
    )
    qsvt_text = table(qsvt_caption, "tab:rq3-qsvt-comparison", rows, QSVT_SOURCE)
    output_directory.mkdir(parents=True, exist_ok=True)
    destinations = [(output_directory / "rq3_select_comparison.tex", select_text),
                    (output_directory / "rq3_qsvt_comparison.tex", qsvt_text)]
    for destination, text in destinations:
        assert "\\caption{" in text and text.index("\\caption{") < text.index("\\begin{tabular}")
        assert text.endswith("\\end{table*}\n")
        assert "\\footnote" not in text and "\\begin{minipage}" not in text
        destination.write_text(text)
        print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=TABLE_DIRECTORY)
    main(parser.parse_args().output_directory)
