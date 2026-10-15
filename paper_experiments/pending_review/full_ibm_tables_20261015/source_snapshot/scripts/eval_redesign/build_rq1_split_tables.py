"""Generate standalone main/appendix resource figures from the audited IBM CSV.

The main table uses one C(W) cell per implementation and the Raw-to-Ours
subnormalization change. Both figures omit
captions and notes. Historical combined previews and the manuscript are not
modified. PDF/PNG rendering is a separate, reproducible LaTeX build step.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from build_rq1_resource_layout import decimal_text

ROOT = Path(__file__).resolve().parents[2]
CASES = ["TFIM-4", "TFIM-8", "TFIM-12", "TFIM-20", "HRW-8", "HRW-12", "HRW-28",
         "RepDQE-d3", "RandCh-N9-8q-seed0", "RandCh-N15-12q-seed0"]
METHODS = ["Stinespring", "VCBE", "VCBE+Qiskit opt3", "Raw", "Raw+Qiskit opt3", "Ours"]
SUCCESS = {"ok", "synthesis_ok_postprocessing_timeout"}
STATUS = {
    "not_run": "/", "not_run_user_assigned_memory_limit": "OOM",
    "not_run_user_assigned_fit_failed": "Fail", "preflight_memory_limit": "OOM",
    "memory_limit": "OOM", "fit_budget_exhausted": "Fail", "fit_search_exhausted": "Fail",
    "timeout": "Timeout", "worker_error": "Error", "running": "Pending",
}
PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage{booktabs,amsmath,multirow,siunitx,graphicx}
\sisetup{group-separator={,},group-minimum-digits=4,group-digits=integer,detect-weight=true}
\newcommand{\status}[1]{\textnormal{\footnotesize #1}}
\begin{document}
"""


def case_name(case):
    return {"RandCh-N9-8q-seed0": "RandCh-N9", "RandCh-N15-12q-seed0": "RandCh-N15"}.get(case, case)


def measured(row):
    return row["status"] in SUCCESS and bool(row["G"])


def main_table(rows):
    lines = [r"\resizebox{6.5in}{!}{%", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
        r"$C(W)$ & Dilation & \multicolumn{2}{c}{Matrix channel-LCU} & \multicolumn{3}{c}{Pauli channel-LCU} & Subnormalization\\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(l){8-8}",
        r"Benchmark & Stinespring & VCBE & +opt3 & Raw & +opt3 & Ours & $\Lambda_{\mathrm{Raw}}\to\Lambda_{\mathrm{Ours}}$\\", r"\midrule"]
    for i, case in enumerate(CASES):
        if i:
            lines.append(r"\midrule" if case in {"HRW-8", "RepDQE-d3", "RandCh-N9-8q-seed0"}
                         else r"\addlinespace[3pt]")
        available = [Decimal(rows[case, m]["Lambda_G"]) for m in METHODS if measured(rows[case, m])]
        best = min(available) if available else None
        cells = []
        for method in METHODS:
            row = rows[case, method]
            if measured(row):
                cost = r"\num{" + decimal_text(row["Lambda_G"], 2) + "}"
                if Decimal(row["Lambda_G"]) == best:
                    cost = r"{\bfseries " + cost + "}"
                cells.append(cost + r"\,({\footnotesize " + row["W"] + "})")
            else:
                cells.append(r"\status{" + STATUS[row["status"]] + "}")
        # Round displayed scales to two decimal places; keep integers plain.
        # Resource costs above still use the full-precision source values.
        scales = []
        for method in ("Raw", "Ours"):
            value = Decimal(rows[case, method]["Lambda"])
            scales.append(str(int(value)) if value == value.to_integral_value()
                          else format(Decimal(decimal_text(str(value), 2)), ".2f"))
        before, after = [r"\num{" + value + "}" for value in scales]
        cells.append(before + r"\,$\to$\," + after)
        lines.append(case_name(case) + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}"])
    return "\n".join(lines)


def appendix_table(rows):
    lines = [r"\small", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{0.97}",
        r"\begin{tabular*}{6.5in}{@{\extracolsep{\fill}}llrrr@{}}", r"\toprule",
        r"Benchmark & Implementation & Gates $G$ & Scale $\Lambda$ & Qubits $W$\\", r"\midrule"]
    for i, case in enumerate(CASES):
        if i:
            lines.append(r"\addlinespace[4pt]")
        for j, method in enumerate(METHODS):
            prefix = r"\multirow{6}{*}{" + case_name(case) + "}" if j == 0 else ""
            row = rows[case, method]
            if measured(row):
                cells = [r"\num{" + row["G"] + "}", r"\num{" + decimal_text(row["Lambda"], 8) + "}", row["W"]]
                lines.append(prefix + " & " + method + " & " + " & ".join(cells) + r" \\")
            else:
                lines.append(prefix + " & " + method + r" & \multicolumn{3}{c}{\status{"
                             + STATUS[row["status"]] + r"}} \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "Results/rq1_resource_tables_ibm_20261015.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Results")
    args = parser.parse_args()
    all_rows = list(csv.DictReader(args.data.open()))
    rows = {(r["case"], r["method"]): r for r in all_rows}
    assert set(rows) == {(c, m) for c in CASES for m in METHODS}
    assert len(rows) == len(all_rows), "Duplicate benchmark/method record"
    for row in rows.values():
        if measured(row):
            assert row["basis"] == "IBM" and int(row["W"]) > 0 and int(row["G"]) > 0
            expected = Decimal(row["Lambda"]) * int(row["G"])
            assert abs(expected - Decimal(row["Lambda_G"])) < max(expected, 1) * Decimal("1e-12")
        else:
            assert row["status"] in STATUS, row["status"]
    header = f"% Source CSV: {args.data.name}; SHA256: {sha256(args.data.read_bytes()).hexdigest()}\n"
    for name, table in (("rq1_main_table", main_table(rows)), ("rq1_appendix_table", appendix_table(rows))):
        path = args.out_dir / (name + ".tex")
        content = header + PREAMBLE + table + "\n\\end{document}\n"
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
        print(path)


if __name__ == "__main__":
    main()
