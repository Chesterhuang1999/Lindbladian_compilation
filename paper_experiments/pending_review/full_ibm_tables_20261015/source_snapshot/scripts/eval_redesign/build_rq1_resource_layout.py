"""Refresh the supplied two-table layout from the collected RQ1 measurements.

Run: python scripts/eval_redesign/build_rq1_resource_layout.py
The Results layout is updated in place; no experiment or manuscript is changed.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CASES = ["TFIM-4", "TFIM-8", "TFIM-12", "TFIM-20", "HRW-8", "HRW-12", "HRW-28",
         "RepDQE-d3", "Petz_Ising3"]
METHODS = ["Stinespring", "VCBE", "VCBE+Qiskit opt3", "Raw", "Raw+Qiskit opt3", "Ours"]
SUCCESS = {"ok", "synthesis_ok_postprocessing_timeout"}
STATUSES = {
    "unavailable_in_source_tables": "No data",
    "not_scheduled_in_saved_vcbe_batch": "/",
    "not_run_queue_paused": "Paused",
    "preflight_memory_limit": "OOM",
    "fit_budget_exhausted": "Fail",
    "timeout": "Timeout",
    "memory_limit": "OOM",
    "worker_error": "Error",
    "not_run_user_assigned_memory_limit": "OOM",
    "not_run_user_assigned_fit_failed": "Fail",
    "not_run": "/",
    "no_ibm_measurement": "No IBM data",
}
COST_CAPTION = ""
BREAKDOWN_CAPTION = ""


def decimal_text(value: str, places: int) -> str:
    """Use fixed notation and discard numerical noise below display precision."""
    rounded = Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    result = format(rounded, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def measured(row: dict[str, str]) -> bool:
    return row["status"] in SUCCESS and bool(row["G"])


def status(row: dict[str, str]) -> str:
    return r"\status{" + STATUSES[row["status"]] + "}"


def case_name(case: str) -> str:
    return "Petz-Ising-3" if case == "Petz_Ising3" else case


def cost_width_table(rows: dict[tuple[str, str], dict[str, str]]) -> str:
    lines = [r"\begin{table}[!ht]", r"\centering", r"\small",
             r"\caption{" + COST_CAPTION + "}", r"\label{tab:costwidth}",
             r"\setlength{\tabcolsep}{2.5pt}", r"\renewcommand{\arraystretch}{1.13}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrr@{}}",
             r"\toprule",
             r"& & Dilation & \multicolumn{2}{c}{Matrix block encoding} & \multicolumn{3}{c}{Pauli channel-LCU}\\",
             r"\cmidrule(lr){3-3}\cmidrule(lr){4-5}\cmidrule(l){6-8}",
             r"Benchmark & Metric & Stinespring & VCBE & +opt3 & Raw & +opt3 & Ours\\",
             r"\midrule"]
    for i, case in enumerate(CASES):
        if i:
            lines.append(r"\midrule" if case in {"HRW-8", "RepDQE-d3", "Petz_Ising3"}
                         else r"\addlinespace[4pt]")
        best = min(Decimal(rows[case, method]["Lambda_G"]) for method in METHODS
                   if measured(rows[case, method]))
        costs, widths = [], []
        for method in METHODS:
            row = rows[case, method]
            if measured(row):
                value = r"\num{" + decimal_text(row["Lambda_G"], 2) + "}"
                if Decimal(row["Lambda_G"]) == best:
                    value = r"{\bfseries " + value + "}"
                costs.append(value)
                widths.append(row["W"])
            else:
                costs.append(r"\multirow{2}{*}{" + status(row) + "}")
                widths.append("")
        lines.append(r"\multirow{2}{*}{" + case_name(case) + r"} & $C$ & "
                     + " & ".join(costs) + r" \\")
        lines.append(r" & $W$ & " + " & ".join(widths) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    return "\n".join(lines)


def breakdown_table(rows: dict[tuple[str, str], dict[str, str]]) -> str:
    lines = [r"\begin{table}[!ht]", r"\centering", r"\small",
             r"\caption{" + BREAKDOWN_CAPTION + "}", r"\label{tab:breakdown}",
             r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.0}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrr@{}}",
             r"\toprule",
             r"Benchmark & Implementation & Gates $G$ & Scale $\Lambda$ & Qubits $W$\\",
             r"\midrule"]
    for i, case in enumerate(CASES):
        if i:
            lines.append(r"\addlinespace[4pt]")
        for j, method in enumerate(METHODS):
            prefix = r"\multirow{6}{*}{" + case_name(case) + "}" if j == 0 else ""
            row = rows[case, method]
            if measured(row):
                cells = [r"\num{" + row["G"] + "}",
                         r"\num{" + decimal_text(row["Lambda"], 8) + "}", row["W"]]
                lines.append(prefix + " & " + method + " & "
                             + " & ".join(cells) + r" \\")
            else:
                lines.append(prefix + " & " + method + r" & \multicolumn{3}{c}{"
                             + status(row) + r"} \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "Results/rq1_resource_layout_ibm_20261014_v2.csv")
    parser.add_argument("--layout", type=Path,
                        default=ROOT / "Results/rq1_resource_layout_preview.tex")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    all_rows = list(csv.DictReader(args.data.open()))
    rows = {(row["case"], row["method"]): row for row in all_rows
            if row["case"] in CASES and row["method"] in METHODS}
    if len(rows) != len(CASES) * len(METHODS):
        raise ValueError(f"The layout requires all {len(CASES) * len(METHODS)} benchmark/method slots")
    for row in rows.values():
        if measured(row):
            if row["basis"] != "IBM" or not row["W"]:
                raise ValueError("Measured cells must have IBM gate counts and widths")
            expected = Decimal(row["Lambda"]) * int(row["G"])
            if abs(expected - Decimal(row["Lambda_G"])) > max(abs(expected), 1) * Decimal("1e-12"):
                raise ValueError("The recorded cost disagrees with the unrounded scale")
        elif row["status"] not in STATUSES:
            raise ValueError(f"Unmapped result status: {row['status']}")
    source = args.layout.read_text()
    table_pattern = re.compile(r"\\begin\{table\}\[!ht\].*?\\end\{table\}", re.DOTALL)
    tables = [cost_width_table(rows), breakdown_table(rows)]
    if len(table_pattern.findall(source)) != len(tables):
        raise ValueError("Expected the supplied two-table layout")
    table_iter = iter(tables)
    updated = table_pattern.sub(lambda _: next(table_iter), source)
    updated = re.sub(
        r"\\sisetup\{[^\n]*\}",
        lambda _: r"\sisetup{group-separator={,},group-minimum-digits=4,group-digits=integer,detect-weight=true}",
        updated, count=1)
    updated = re.sub(r"\A(?:%[^\n]*\n)*", "", updated)
    header = ("% Two-table layout supplied by the user; measurements refreshed from the collected CSV.\n"
              f"% Data: {args.data.name}; SHA256: {sha256(args.data.read_bytes()).hexdigest()}\n")
    output = args.out or args.layout
    output.write_text(header + updated)
    print(f"Updated {output}: {len(rows)} slots, {sum(measured(row) for row in rows.values())} measured circuits")


if __name__ == "__main__":
    main()
