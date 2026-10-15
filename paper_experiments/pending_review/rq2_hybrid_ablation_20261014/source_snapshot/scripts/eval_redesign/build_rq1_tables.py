"""Generate RQ1 table fragments and a standalone preview from saved CSVs.

Run with ``python scripts/eval_redesign/build_rq1_tables.py`` from the repository.
Compile ``Results/rq1_tables_and_scaling_20261014.tex`` from its directory with
pdflatex (twice for cross-references). This script does not launch experiments.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Results"


def load(name):
    with (OUTPUT / name).open() as stream:
        return list(csv.DictReader(stream))


def number(value):
    return f"{int(value):,}"


def storage(byte_count):
    for power, unit in [(5, "PiB"), (4, "TiB"), (3, "GiB"), (2, "MiB"), (1, "KiB")]:
        if byte_count >= 1024 ** power:
            return f"{byte_count / 1024 ** power:g}\\,\\mathrm{{{unit}}}"
    return f"{byte_count}\\,B"


def circuit_cell(measurement):
    return (
        r"\shortstack[r]{$" + number(measurement["G"]) + r"$\\$("
        + f"{float(measurement['Lambda']):.3g},{measurement['W']}" + r")$}"
    )


def table1():
    rows = [r for r in load("rq1_table1_resources_20261013.csv") if r["table1_candidate"] == "True"]
    cases = list(dict.fromkeys(r["case"] for r in rows))
    text = [r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\caption{Resources for implementations of the same completely positive map in each row. Matrix cells show complex128 storage for a density matrix $\rho$ above a superoperator $S$. Circuit cells show gate count $G$ above $(\Lambda,W)$. With all auxiliary qubits initialized to $|0\rangle$, the circuit's unnormalized system map is obtained by postselecting the encoding's success ancillas to zero and tracing out the remaining auxiliary registers; multiplying this map by $\Lambda$ recovers the target map. $W$ is the total circuit qubit count, including auxiliary qubits. Gates use the NAM basis $\{R_z,H,X,\mathrm{CNOT}\}$. LCU denotes linear combinations of unitaries.}",
        r"\label{tab:rq1-map-resources}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Case & Dense $\rho$ / $S$ & Stinespring & Raw channel-LCU & Ours \\", r"\midrule"]
    for case in cases:
        measurements = {r["method"]: r for r in rows if r["case"] == case}
        matrix = measurements["Direct matrix"]
        cells = [case, r"\shortstack[r]{$" + storage(int(matrix["rho_bytes"])) +
                 r"$\\$" + storage(int(matrix["superoperator_bytes"])) + "$}"]
        for method in ["Stinespring", "Raw channel-LCU", "Ours"]:
            m = measurements[method]
            if m["status"] == "preflight_memory_limit":
                cell = r"\shortstack[r]{Mem.\\$(1,{\text{---}})$}"
            else:
                cell = circuit_cell(m)
            cells.append(cell)
        text.append(" & ".join(cells) + r" \\[2pt]")
    text += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"TFIM and HRW denote the transverse-field Ising model and hypercube random walk; their suffix is the system qubit count $n$. RepDQE-d3 is the distance-3 repetition-code correction and encoded weak-measurement/reset benchmark, with $n=3$ system qubits. Input definitions follow Appendix~D.1; TFIM uses a first-order step of length 0.1, unit coupling and field, and decay rate 0.1; RepDQE uses measurement strength 0.25. The matrix shapes are $2^n\!\times2^n$ and $4^n\!\times4^n$; storage is calculated, without measuring matrix action time or peak memory. Mem.: the HRW-12 isometry requires 8 GiB, exceeding the 3 GiB preflight budget. TFIM Stinespring adds a success flag with $\Lambda=\max\{1,\|\sum_j K_j^\dagger K_j\|_2\}$ for target Kraus operators $K_j$, evaluated with upward numerical rounding. Displayed $\Lambda$ values have three significant digits; checks use full precision. Stinespring input isometries $V$ for TFIM-8 and HRW-8 satisfied $\|V^\dagger V-I\|_F<10^{-10}$; their output circuits were not checked against the full Choi matrix. Full circuit Choi checks passed for all three circuit methods on TFIM-4, HRW-4, and RepDQE-d3, with relative Frobenius error below $10^{-10}$ after restoration by $\Lambda$. Raw/Ours full Choi checks were not run for TFIM-8, HRW-8, or HRW-12; Ours' Kraus rewrites passed coefficient-space map-equivalence checks in all six cases. Raw uses unoptimized channel-LCU assembly; Ours uses Kraus rewriting, Pauli placement, and control flattening. Final lowering is at optimization level 0, without a downstream optimizer.",
        r"\end{minipage}", r"\end{table}"]
    (OUTPUT / "rq1_table1_resources_20261014.tex").write_text("\n".join(text) + "\n")


def raw_optimizer_table():
    rows = load("rq1_raw_optimizer_20261014.csv")
    cases = ["TFIM-4", "HRW-4", "HRW-12", "Petz_Ising3"]
    methods = ["Raw", "Raw+VOQC", "Ours"]
    text = [r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Complete channel-LCU circuits optimized by the Verified Optimizer for Quantum Circuits (VOQC), compared with Raw channel-LCU and Ours, for four inputs with completed VOQC runs. Each row targets the same completely positive map. Raw+VOQC applies VOQC to the Raw circuit and retains its normalization factor $\Lambda$. Circuit cells show $G$ above $(\Lambda,W)$, using the NAM gate basis and success-map convention of Table~\ref{tab:rq1-map-resources}. The last column gives the normalization-weighted gate-count ratio $r=(\Lambda G)_{\rm Ours}/(\Lambda G)_{\rm Raw+VOQC}$.}",
        r"\label{tab:rq1-raw-optimizer}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Case & Raw channel-LCU & Raw+VOQC & Ours & $r$ \\", r"\midrule"]
    for case in cases:
        selected = {}
        for method in methods:
            matches = [r for r in rows if r["case"] == case and r["method"] == method]
            if len(matches) != 1:
                raise ValueError(f"Expected one saved {case}/{method} row, found {len(matches)}")
            measurement = matches[0]
            if measurement["status"] != "ok" or measurement["basis"] != "NAM":
                raise ValueError(f"Incomplete or non-NAM result: {case}/{method}")
            if not measurement["input_pairing"].startswith("passed"):
                raise ValueError(f"Unpaired input: {case}/{method}")
            if not math.isclose(float(measurement["Lambda_G"]),
                                float(measurement["Lambda"]) * int(measurement["G"]),
                                rel_tol=1e-12):
                raise ValueError(f"Inconsistent weighted count: {case}/{method}")
            selected[method] = measurement
        input_hashes = {m["canonical_input_sha256"] for m in selected.values()}
        if len(input_hashes) != 1 or "" in input_hashes:
            raise ValueError(f"Inconsistent canonical inputs: {case}")
        raw, optimized = selected["Raw"], selected["Raw+VOQC"]
        if (optimized["input_qasm_sha256"] != raw["qasm_sha256"]
                or optimized["W"] != raw["W"]
                or optimized["Lambda"] != raw["Lambda"]):
            raise ValueError(f"VOQC is not paired with the saved Raw circuit: {case}")
        ratio = float(selected["Ours"]["Lambda_G"]) / float(optimized["Lambda_G"])
        cells = [case.replace("Petz_Ising3", "Petz-Ising-3")]
        cells += [circuit_cell(selected[method]) for method in methods]
        cells.append(f"{ratio:.3g}")
        text.append(" & ".join(cells) + r" \\[2pt]")
    text += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"TFIM, HRW, and their suffixes follow Table~\ref{tab:rq1-map-resources}. Petz-Ising-3 is the three-qubit Petz recovery map specified by the Kraus operators in \path{../paper_experiments/pending_review/raw_baseline_20261012/inputs/Petz_Ising3.json}. Ours reuses the TFIM-4 and HRW-4 circuits reported in Table~\ref{tab:rq1-map-resources}; per-row rewrite configurations are recorded in the source CSV. All Ours circuits use the assembly and level-0 NAM lowering described in Table~\ref{tab:rq1-map-resources}. Displayed $\Lambda$ and $r$ values have three significant digits; ratios use full-precision data. All four Ours Kraus rewrites passed coefficient-space map-equivalence checks. Full Ours circuit Choi checks passed for TFIM-4, HRW-4, and Petz-Ising-3, with relative Frobenius error below $10^{-10}$ after restoration by $\Lambda$; HRW-12 was not checked against the full Choi matrix. VOQC input-file hashes match those of the saved Raw circuits; output gate counts, gate basis, and widths were checked from saved circuits. Full channel-equivalence simulation of the VOQC outputs was not additionally run. These candidate rows comprise all three completed TFIM/HRW comparisons and the Petz input. Six inputs in total have both successful VOQC runs and Ours circuits; all recorded outcomes are retained in \path{rq1_raw_optimizer_20261014.csv}. The remaining optimizer queue is paused.",
        r"\end{minipage}", r"\end{table}"]
    (OUTPUT / "rq1_raw_optimizer_20261014.tex").write_text("\n".join(text) + "\n")


def vcbe_table():
    rows = load("rq1_pauli_vcbe_20261014.csv")
    primary = ["TFIM-2", "TFIM-3", "TFIM-4", "HRW-4-rewritten"]
    selected = [r for r in rows if r["basis"] == "NAM" and r["final_optimization_level"] == "0"]
    text = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Complete channel circuits using Pauli linear combinations of unitaries (Pauli-LCU) or variational compilation of block encodings (VCBE). Each row fixes the Kraus representation and normalization $\Lambda$ for all methods. With all auxiliary qubits initialized to $|0\rangle$, the unnormalized system map is obtained by postselecting the success ancillas to zero and tracing out the remaining auxiliary registers; multiplying this map by $\Lambda$ recovers the target map. TFIM denotes the transverse-field Ising model and HRW the hypercube random walk; suffixes give system qubit counts. HRW uses the shared rewritten representation. $G$ counts gates in the NAM basis $\{R_z,H,X,\mathrm{CNOT}\}$; $W$ is the total circuit qubit count, including auxiliary qubits. VCBE+opt3 optimizes each fitted inner block at Qiskit level 3 before channel assembly. Final lowering is at level 0 for all columns.}",
        r"\label{tab:rq1-shared-rewrite-vcbe}",
        r"\begin{tabular}{@{}lr rr rr rr@{}}", r"\toprule",
        r"\multirow{2}{*}{Case} & \multirow{2}{*}{$\Lambda$} & \multicolumn{2}{c}{Pauli-LCU} & \multicolumn{2}{c}{VCBE} & \multicolumn{2}{c}{VCBE+opt3} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"& & $G$ & $W$ & $G$ & $W$ & $G$ & $W$ \\", r"\midrule"]
    for case in primary:
        methods = {r["method"]: r for r in selected if r["case"] == case}
        cells = [case.replace("-rewritten", " (rewritten)"), f"{float(next(iter(methods.values()))['Lambda']):.3g}"]
        for method in ["pauli_matrix_order_fstar", "vcbe_source_tree", "vcbe_preopt3_tree"]:
            m = methods[method]
            cells += [f"{int(m['G']):,}", m["W"]]
        text.append(" & ".join(cells) + r" \\")
    text += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"Inputs use the TFIM/HRW definitions and parameters of Table~\ref{tab:rq1-map-resources}. Pauli-LCU uses the Ours backend. Both routes use the same outer state preparation and Kraus-selection tree. Pauli-LCU controls only Pauli selection; VCBE controls each fitted unitary in full. VCBE uses cached blocks from our independent reproduction of the generic ansatz in Fig.~1 and Eq.~(7) of \href{https://arxiv.org/abs/2507.17658v2}{Rullk{\"o}tter et al., arXiv:2507.17658v2}. Displayed $\Lambda$ values are rounded to three significant digits; checks use full precision. All 12 displayed circuits passed full Choi comparisons: $\|J_{\rm circuit}-J_{\rm target}\|_F/\|J_{\rm target}\|_F<10^{-10}$, where $J_{\rm circuit}$ includes restoration by $\Lambda$; the maximum was $1.49\times10^{-11}$. The source data contain 60 validated circuits: five inputs (including HRW before Kraus rewriting), three methods, two gate bases (NAM and IBM $\{u_1,u_2,u_3,\mathrm{CNOT}\}$), and final optimization levels 0/3.",
        r"\end{minipage}", r"\end{table}"]
    (OUTPUT / "rq1_pauli_vcbe_20261014.tex").write_text("\n".join(text) + "\n")


def scaling_fragment():
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{rq1_scaling_20261014.pdf}
\caption{Measured transverse-field Ising model (TFIM) and hypercube random-walk (HRW) scaling with system qubit count $n$, using the input definitions and parameters of Table~\ref{tab:rq1-map-resources}. $G$ counts full-circuit gates in the NAM basis $\{R_z,H,X,\mathrm{CNOT}\}$; $\Lambda G$ weights this count by the factor $\Lambda$ that restores the unnormalized system map defined in Table~\ref{tab:rq1-map-resources}. Raw uses unoptimized channel-LCU assembly (LCU: linear combinations of unitaries); Ours adds Kraus rewriting, Pauli placement, and control flattening. Compilation time includes canonicalization, Kraus rewrite (Ours only), circuit assembly, and NAM lowering; imports, input construction, verification, and output I/O are excluded. M4 denotes the local Apple M4 host; EPYC denotes the remote dual AMD EPYC 9534 host. Remote Raw times are separate platform measurements. Lines connect single observations. Historical Raw measurements reused for TFIM $n=4,8,12$ and HRW $n=4,8,12,20,28$ used 600 s/4 GiB budgets; new local Raw jobs used 600 s/6 GiB, and remote Raw jobs used 3600 s/32 GiB. Ours had no imposed per-case time or resident-memory cap. The reported completion ranges are subject to these different operational budgets. Deferred jobs were not executed; failed/deferred jobs have no plotted gate count or compilation time.}
\label{fig:rq1-scaling}
\Description{Six panels show gate count, normalization-weighted gate count, and compilation time for TFIM and HRW. Local Raw results stop before the largest instances; two remote Raw points extend the resource comparison.}
\end{figure}
"""
    (OUTPUT / "rq1_scaling_20261014.tex").write_text(text)


def standalone_preview():
    fragments = [
        "rq1_table1_resources_20261014.tex",
        "rq1_raw_optimizer_20261014.tex",
        "rq1_pauli_vcbe_20261014.tex",
        "rq1_scaling_20261014.tex",
    ]
    text = [
        "% Generated by scripts/eval_redesign/build_rq1_tables.py.",
        "% Compile from this directory with pdflatex (twice for references).",
        r"\documentclass[acmsmall,screen,nonacm]{acmart}",
        r"\usepackage{multirow}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\graphicspath{{./}}",
        r"\title{RQ1 table and figure candidates}",
        r"\begin{document}",
    ]
    for fragment in fragments:
        text += [r"\input{" + fragment + "}", r"\clearpage"]
    text.append(r"\end{document}")
    path = OUTPUT / "rq1_tables_and_scaling_20261014.tex"
    path.write_text("\n".join(text) + "\n")
    print(f"Standalone TeX preview: {path}")


if __name__ == "__main__":
    table1()
    raw_optimizer_table()
    vcbe_table()
    scaling_fragment()
    standalone_preview()
