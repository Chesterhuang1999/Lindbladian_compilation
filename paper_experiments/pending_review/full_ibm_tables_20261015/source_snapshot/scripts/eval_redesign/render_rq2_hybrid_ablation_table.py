#!/usr/bin/env python3
"""Render the measured hybrid RQ2 run in the manuscript's Table 5 layout.

Random-instance percentages are ratios of arithmetic mean gate counts.
Compilation measurements are retained when only equivalence verification
reached a resource limit; the table marks those cells explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "paper_experiments/pending_review/rq2_hybrid_ablation_20261014"
DEFAULT_OUTPUT = ROOT / "Results/eval_redesign/tables/rq2_hybrid_ablation_20261014"
CONFIGS = ("Basic", "R", "O", "Fstar", "R_O", "R_Fstar", "O_Fstar", "All")
LABELS = ("Basic", "R", "O", "F*", "R+O", "R+F*", "O+F*", "All")
BENCHMARKS = {
    "TFIM-8": "TFIM_8",
    "HRW-20": "HRW_20",
    "Thermal2_HO": "Thermal2_HO",
    "Random-5K-4q-10-2": "Rand_Paulis_(5,4,10,2)",
    "Petz_TFIM2": "Petz_TFIM2",
    "RepDQE-d3": "Rep-DQE-d3",
}
METRICS = (
    "G", "NC", "W", "Lambda", "Lambda_G", "Lambda_NC", "M", "support",
    "rz", "h", "x", "cx", "canonicalization_s", "rewrite_s", "assembly_s",
    "lowering_s", "compile_time_s", "process_wall_s",
)

CAPTION = r"""Ablation of Kraus-representation rewriting (R), Pauli-sum SELECT ordering (O), and conditional flattening (F$^\ast$). The seven configuration columns report gate-count percentages relative to Basic in the $\{R_z,H,X,\mathrm{CNOT}\}$ gate set."""
NOTES = r"""Basic disables all three passes; All enables all three. F$^\ast$ uses a Kraus-branch selection tree and applies branch controls only to SELECT (the Pauli-term selection circuit). Rewrite and $\Delta($Subnorm$)$ give, respectively, the total Pauli coefficient count (magnitude $>10^{-12}$) and subnormalization factor $\Lambda$ before $\rightarrow$ after R. For Kraus operators $K_k=\sum_p a_{kp}P_p$, where $P_p$ are Pauli strings, $\Lambda=\sum_k(\sum_p|a_{kp}|)^2$. The displayed rewrite changes hold for both the basic outer structure and the F$^\ast$ selection tree.
TFIM\_8 and HRW\_20 denote the eight-qubit transverse-field Ising channel and twenty-qubit hypercube random-walk channel, respectively.
Random Pauli values are arithmetic means over seeds 0--4; each percentage divides the configuration's mean gate count by the Basic mean. $(5,4,10,2)$ specifies five Kraus operators on four qubits, with ten Pauli terms in the first operator and two in each remaining operator.
Thermal2\_HO uses $H=0.3Z_0+0.45Z_1+0.2X_0X_1$ and $L_{j,01}=(X_j+\mathrm{i}Y_j)/2$, $L_{j,10}=\sqrt{0.4}(X_j-\mathrm{i}Y_j)/2$ for $j\in\{0,1\}$ (with $j$ indexing qubits). Its higher-order construction uses time step $\delta=0.12$, maximum jump count $K=2$, $q=2$ Gauss--Legendre nodes per time integral, and drift Taylor degree $p=3$.
Counts exclude resets; lowering uses optimization level 0 with no downstream circuit optimizer. $\dagger$: Thermal2\_HO/Basic and Thermal2\_HO/O completed compilation and exported-circuit recounting; full Choi-matrix equivalence checks reached the 24\,GiB memory and 180\,s time limits, respectively. Full Choi checks are attempted only for at most four system qubits and twenty total qubits."""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_count(value: float) -> str:
    return f"{round(value):,}" if value == round(value) else f"{value:,.1f}"


def escape_tex(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def collect(run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load(run / "manifest.json")
    summary = load(run / "results.json")["rows"]
    indexed = {(r["case"], r["config"]): r for r in summary}
    cases = manifest["cases"]
    require(len(indexed) == len(summary) == len(cases) * len(CONFIGS),
            "Expected one measured result per case and configuration")
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["id"]
        case_dir = run / "cases" / case_id
        for suffix in ("json", "npz"):
            digest = hashlib.sha256((case_dir / f"input.{suffix}").read_bytes()).hexdigest()
            require(digest == case[f"input_{suffix}_sha256"], f"Input hash mismatch: {case_id}")
        inp = load(case_dir / "input.json")
        if case.get("group_id") == "Random-5K-4q-10-2":
            require(case["system_qubits"] == 4 and
                    [len(k) for k in inp["canonical_kraus"]] == [10, 2, 2, 2, 2] and
                    all(len(term[0]) == 4 for k in inp["canonical_kraus"] for term in k),
                    f"Incorrect random Pauli parameters: {case_id}")
        rewrites = {ctx: load(case_dir / "rewrites" / f"{ctx}.json")
                    for ctx in ("basic", "tree")}
        for ctx, rewrite in rewrites.items():
            require(rewrite["status"] == "ok" and rewrite["verification"]["passed"],
                    f"Rewrite verification did not pass: {case_id}/{ctx}")
            require(rewrite["canonical_input_sha256"] == case["canonical_input_sha256"],
                    f"Rewrite input mismatch: {case_id}/{ctx}")
        for slug, label in zip(CONFIGS, LABELS):
            s = indexed[case_id, slug]
            d = load(run / s["result"])
            settings = d["settings"]
            cfg = settings["configuration"]
            rewrite_on = slug in ("R", "R_O", "R_Fstar", "All")
            order_on = slug in ("O", "R_O", "O_Fstar", "All")
            fstar_on = slug in ("Fstar", "R_Fstar", "O_Fstar", "All")
            context = ("tree" if fstar_on else "basic") if rewrite_on else None
            require(cfg["rewrite"] == rewrite_on and cfg["order"] == order_on and
                    cfg["fstar"] == fstar_on and cfg["rewrite_context"] == context and
                    cfg["structure"] == ("tree" if fstar_on else "basic") and
                    cfg["opt"] == ("Matrix-order" if order_on else "No"),
                    f"Ablation flag mismatch: {case_id}/{slug}")
            require(settings["basis"] == "nam" and
                    set(settings["basis_gates"]) == {"rz", "h", "x", "cx"} and
                    settings["lowering_optimization_level"] == 0 and
                    settings["downstream_optimizer"] is None and
                    settings["seed_transpiler"] == 20261009 and
                    settings["qubits_initially_zero"] is False,
                    f"Lowering protocol mismatch: {case_id}/{slug}")
            require(d["canonical_input_sha256"] == case["canonical_input_sha256"],
                    f"Canonical input mismatch: {case_id}/{slug}")
            require(all(manifest["snapshot_sha256"].get(f"src/{name}") == digest
                        for name, digest in d["sources"].items()),
                    f"Measured compiler differs from frozen source: {case_id}/{slug}")
            require(d["metadata"]["structure"] == cfg["structure"] and
                    d["metadata"]["block_opt"] == cfg["opt"] and
                    d["metadata"]["fstar"] == fstar_on and
                    d["metadata"]["controlled_prepare"] ==
                    (d["representation"]["M"] > 1 and not fstar_on),
                    f"Assembled circuit differs from ablation flags: {case_id}/{slug}")
            expected_input = (rewrites[context]["compiled_input_sha256"] if rewrite_on
                              else case["canonical_input_sha256"])
            if rewrite_on:
                # The worker checks the input digest before assembly; the result's
                # digest is captured after assembly. Link the frozen rewrite cache
                # by its identity, retaining both digests in the instance records.
                require(settings["rewrite"]["cache_identity"] == rewrites[context]["cache_identity"] and
                        d["representation"]["M"] == rewrites[context]["metrics"]["M"] and
                        d["representation"]["support"] == rewrites[context]["metrics"]["support"],
                        f"Rewrite cache mismatch: {case_id}/{slug}")
            else:
                require(d["compiled_input_sha256"] == expected_input,
                        f"Compiled input mismatch: {case_id}/{slug}")
            for metric in ("G", "NC", "W", "Lambda"):
                require(d[metric] == s[metric] and math.isfinite(d[metric]),
                        f"Summary differs from result: {case_id}/{slug}/{metric}")
            recount = d["qasm_recount"]
            require(d["G"] == recount["metric_total"] ==
                    sum(d["counts"].get(gate, 0) for gate in ("rz", "h", "x", "cx")) and
                    d["NC"] == recount["clifford"]["non_clifford"] and
                    d["W"] == recount["num_qubits"],
                    f"Circuit recount mismatch: {case_id}/{slug}")
            require(d["rewrite_verification"]["passed"],
                    f"Representation equivalence failed: {case_id}/{slug}")
            v = d["verification"]
            verification_status = "passed" if v.get("passed") else v.get("status", "unknown")
            if s["status"] != "ok":
                require(case_id == "Thermal2_HO" and
                        (slug, s["status"]) in (("Basic", "memory_limit"),
                                               ("O", "verification_timeout")) and
                        verification_status == "not_completed",
                        f"Unexpected incomplete job: {case_id}/{slug}")
            elif verification_status != "passed":
                require((case["system_qubits"] > 4 or d["W"] > 20) and verification_status == "not_run",
                        f"Unexpected verification status: {case_id}/{slug}")
            record = {
                "group": case.get("group_id", case_id), "case": case_id,
                "seed": case.get("parameters", {}).get("seed", ""),
                "config": slug, "label": label, "basis": "nam",
                "rewrite": rewrite_on, "order": order_on, "fstar": fstar_on,
                "structure": cfg["structure"], "rewrite_context": context or "",
                **{m: d[m] for m in METRICS if m in d},
                "process_wall_s": s["process_wall_s"],
                "M": d["representation"]["M"], "support": d["representation"]["support"],
                **{g: d["counts"].get(g, 0) for g in ("rz", "h", "x", "cx")},
                "status": s["status"], "verification_status": verification_status,
                "verification_reason": v.get("reason", ""),
                "choi_relative_error": v.get("relative_frobenius_error", ""),
                "rewrite_equivalence_passed": d["rewrite_verification"]["passed"],
                "canonical_input_sha256": d["canonical_input_sha256"],
                "preassembly_input_sha256": expected_input,
                "compiled_input_sha256": d["compiled_input_sha256"],
                "qasm_sha256": d["qasm_sha256"], "qasm_path": recount["path"],
                "source_result": str((run / s["result"]).relative_to(ROOT)),
            }
            records.append(record)

    groups = manifest["config"]["case_groups"]
    group_ids = [g["id"] for g in groups]
    require(len(set(group_ids)) == len(group_ids) and
            all(group_id in BENCHMARKS for group_id in group_ids), "Unexpected benchmark groups")
    aggregate: list[dict[str, Any]] = []
    for group in groups:
        group_id = group["id"]
        selected = [r for r in records if r["group"] == group_id]
        basics = [r for r in selected if r["config"] == "Basic"]
        n = 5 if group_id == "Random-5K-4q-10-2" else 1
        require(len(basics) == n, f"Incomplete group: {group_id}")
        if n == 5:
            require(sorted(r["seed"] for r in basics) == list(range(5)), "Expected random seeds 0--4")
        basic_g = mean(r["G"] for r in basics)
        for slug, label in zip(CONFIGS, LABELS):
            selection = [r for r in selected if r["config"] == slug]
            require(len(selection) == n, f"Incomplete configuration: {group_id}/{slug}")
            aggregate.append({
                "benchmark": BENCHMARKS[group_id], "group": group_id,
                "config": slug, "label": label, "sample_count": n,
                "seeds": "0;1;2;3;4" if n == 5 else "",
                "basis": "nam", "denominator": "Basic",
                **{m: mean(r[m] for r in selection) for m in METRICS},
                "G_sample_sd": stdev(r["G"] for r in selection) if n > 1 else "",
                "G_over_Basic": mean(r["G"] for r in selection) / basic_g,
                "percent_of_Basic": 100 * mean(r["G"] for r in selection) / basic_g,
                "measurement_count": n,
                "choi_passed_count": sum(r["verification_status"] == "passed" for r in selection),
                "verification_skipped_count": sum(r["verification_status"] == "not_run" for r in selection),
                "verification_limited_count": sum(r["verification_status"] == "not_completed" for r in selection),
                "job_status_counts": json.dumps(dict(Counter(r["status"] for r in selection)), sort_keys=True),
                "source_cases": ";".join(r["case"] for r in selection),
            })
    return records, aggregate


def render(aggregate: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    indexed = {(r["group"], r["config"]): r for r in aggregate}
    table_rows: list[dict[str, Any]] = []
    tex_rows: list[str] = []
    groups = list(dict.fromkeys(r["group"] for r in aggregate))
    for group in groups:
        name = BENCHMARKS[group]
        basic, rewritten = indexed[group, "Basic"], indexed[group, "R"]
        rewritten_tree = indexed[group, "All"]
        require(rewritten["support"] == rewritten_tree["support"] and
                math.isclose(rewritten["Lambda"], rewritten_tree["Lambda"], abs_tol=1e-12),
                f"One rewrite-change column cannot describe both contexts: {group}")
        row = {
            "Benchmark": name, "Denom.": "Basic",
            "Rewrite": f"{basic['support']:g}->{rewritten['support']:g}",
            "Delta(Subnorm)": f"{basic['Lambda']:.2f}->{rewritten['Lambda']:.2f}",
            "Ref. gates": fmt_count(basic["G"]),
            **{label: f"{indexed[group, slug]['percent_of_Basic']:.2f}%"
               for slug, label in zip(CONFIGS[1:], LABELS[1:])},
        }
        cells = [escape_tex(row["Benchmark"]), row["Denom."],
                 "$" + row["Rewrite"].replace("->", r"\rightarrow") + "$",
                 "$" + row["Delta(Subnorm)"].replace("->", r"\rightarrow") + "$",
                 row["Ref. gates"],
                 *(escape_tex(row[label]) for label in LABELS[1:])]
        if group == "Thermal2_HO":
            cells[4] += r"$^\dagger$"
            cells[6] += r"$^\dagger$"
            row["Ref. gates"] += "†"
            row["O"] += "†"
        table_rows.append(row)
        tex_rows.append(" & ".join(cells) + r" \\")
    notes = NOTES
    additions = []
    if "Petz_TFIM2" in groups:
        additions.append(r"Petz\_TFIM2 applies the Petz recovery map after tracing out qubit 0, using the Gibbs state $\sigma=e^{-\beta H}/\operatorname{Tr}(e^{-\beta H})$ with $H=-Z_0Z_1-0.6(X_0+X_1)$ and $\beta=0.7$.")
    if "RepDQE-d3" in groups:
        additions.append(r"Rep-DQE-d3 denotes the three-qubit bit-flip repetition-code channel with stabilizers $S_1=Z_0Z_1$, $S_2=Z_0Z_2$ and logical operators $\bar X=X_0X_1X_2$, $\bar Z=Z_0$. For syndrome $s=(s_1,s_2)\in\{0,1\}^2$, let $\Pi_s=\prod_{j=1}^2\frac{I+(-1)^{s_j}S_j}{2}$ and $(R_{00},R_{01},R_{10},R_{11})=(I,X_2,X_1,X_0)$. Its eight Kraus operators are $\{M_0R_s\Pi_s,\bar ZM_1R_s\Pi_s\}_s$, where $M_0=(1-\epsilon/2)I+(\epsilon/2)\bar X$, $M_1=\sqrt{\epsilon(2-\epsilon)}(I-\bar X)/2$, and $\epsilon=0.25$; the $M_1$ outcome triggers logical-$Z$ feedback after the weak measurement.")
    if additions:
        notes = notes.replace("Counts exclude resets;", " ".join(additions) + "\nCounts exclude resets;")
    tex = r"""\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\caption{""" + CAPTION + r"""}
\label{tab:rq2-hybrid-ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllrrrrrrrr}
\toprule
Benchmark & Denom. & Rewrite & $\Delta($Subnorm$)$ & Ref. gates & R & O & F$^\ast$ & R+O & R+F$^\ast$ & O+F$^\ast$ & All \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}%
}
\vspace{0.6ex}
\begin{minipage}{0.98\textwidth}
\footnotesize\emph{Notes.} """ + notes + r"""
\end{minipage}
\vspace{-0.8em}
\end{table*}
"""
    return table_rows, tex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--supplement-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runs = [args.run_dir.resolve(), *(p.resolve() for p in args.supplement_run_dir)]
    require(len(set(runs)) == len(runs), "The same run was supplied more than once")
    reference = load(runs[0] / "manifest.json")
    protocol = {k: v for k, v in reference["config"].items() if k != "case_groups"}
    compilation_files = {name: digest for name, digest in reference["snapshot_sha256"].items()
                         if name.startswith("src/") or name in (
                             "scripts/eval_redesign/hybrid_rewrite.py",
                             "scripts/eval_redesign/hybrid_candidate_kernel.py",
                             "scripts/eval_redesign/selected_channel_inputs.py",
                             "scripts/eval_redesign/run_rq2_hybrid_ablation.py",
                             "Baseline_scripts/baseline_common.py")}
    records, aggregate = [], []
    seen_groups: set[str] = set()
    for run in runs:
        manifest = load(run / "manifest.json")
        for name, digest in manifest["snapshot_sha256"].items():
            require(hashlib.sha256((run / "source_snapshot" / name).read_bytes()).hexdigest() == digest,
                    f"Frozen source hash mismatch: {run}/{name}")
        require({k: v for k, v in manifest["config"].items() if k != "case_groups"} == protocol,
                f"Supplement uses a different measurement protocol: {run}")
        require(all(manifest["snapshot_sha256"].get(name) == digest
                    for name, digest in compilation_files.items()),
                f"Supplement uses a different compiler or rewrite implementation: {run}")
        group_ids = {g["id"] for g in manifest["config"]["case_groups"]}
        require(not seen_groups.intersection(group_ids), "Benchmark group appears in multiple runs")
        seen_groups.update(group_ids)
        new_records, new_aggregate = collect(run)
        records.extend(new_records)
        aggregate.extend(new_aggregate)
    table_rows, tex = render(aggregate)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "table5.csv", table_rows)
    write_csv(output / "ablation_means.csv", aggregate)
    write_csv(output / "ablation_instances.csv", records)
    (output / "table5.tex").write_text(tex)
    (output / "table5_preview.tex").write_text(r"""\documentclass[acmsmall,screen,review,anonymous,nonacm]{acmart}
\usepackage{graphicx}
\usepackage{booktabs}
\begin{document}
\pagestyle{empty}
\setcounter{table}{4}
\renewcommand{\topfraction}{1}
\renewcommand{\textfraction}{0}
\input{table5.tex}
\null
\clearpage
\end{document}
""")
    print(f"Validated {len(records)} measurements; generated {len(aggregate)} configuration means and {len(table_rows)} Table 5 rows.")
    print(json.dumps(table_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
