"""Merge the nine inputs in the three RQ1 table candidates into one table.

Run: python scripts/eval_redesign/build_rq1_merged_table.py
Outputs: Results/rq1_unified_resources_20261014.{csv,tex}
Compile the standalone TeX with pdflatex. No experiments are launched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Results"
STEM = "rq1_unified_resources_20261014"
SOURCES = [
    "rq1_table1_resources_20261013.csv",
    "rq1_raw_optimizer_20261014.csv",
    "rq1_pauli_vcbe_20261014.csv",
]
CASES = [
    "TFIM-2", "TFIM-3", "TFIM-4", "TFIM-8",
    "HRW-4", "HRW-8", "HRW-12", "RepDQE-d3", "Petz_Ising3",
]
METHODS = ["Direct matrix", "Stinespring", "Raw", "Raw+VOQC", "VCBE", "VCBE+opt3", "Ours"]
VCBE_METHODS = {
    "pauli_matrix_order_fstar": "Ours",
    "vcbe_source_tree": "VCBE",
    "vcbe_preopt3_tree": "VCBE+opt3",
}
SUCCESS = {"ok", "synthesis_ok_postprocessing_timeout"}


def read(path):
    return json.loads(Path(path).read_text())


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def encoded_hash(kraus):
    return sha256(json.dumps(kraus, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def coefficients(kraus):
    result = []
    for terms in kraus:
        row = Counter()
        for label, magnitude, real, imag in terms:
            row[label] += magnitude * complex(real, imag)
        result.append({p: value for p, value in row.items() if value != 0})
    return result


def map_error(left, right):
    def gram(kraus):
        values = Counter()
        for row in coefficients(kraus):
            for p, a in row.items():
                for q, b in row.items():
                    values[p, q] += a.conjugate() * b
        return values
    a, b = gram(left), gram(right)
    norm = math.sqrt(sum(abs(value) ** 2 for value in a.values()))
    return math.sqrt(sum(abs(b[key] - a[key]) ** 2 for key in a.keys() | b.keys())) / norm


def same_representation(left, right):
    a, b = coefficients(left), coefficients(right)
    return len(a) == len(b) and all(
        all(abs(x.get(p, 0) - y.get(p, 0)) < 1e-12 for p in x.keys() | y.keys())
        for x, y in zip(a, b)
    )


def collect():
    candidates = defaultdict(list)
    case_records = defaultdict(list)
    for filename in SOURCES:
        with (OUTPUT / filename).open() as stream:
            for original in csv.DictReader(stream):
                row = dict(original)
                case = row["case"].removesuffix("-rewritten")
                if case not in CASES:
                    continue
                method = row["method"]
                if filename == SOURCES[0]:
                    if row["table1_candidate"] != "True":
                        continue
                    method = "Raw" if method == "Raw channel-LCU" else method
                elif filename == SOURCES[1]:
                    if method not in {"Raw", "Raw+VOQC", "Ours"}:
                        continue
                else:
                    if (row["basis"] != "NAM" or row["final_optimization_level"] != "0"
                            or row["case"].endswith("-raw")):
                        continue
                    method = VCBE_METHODS[method]
                row["source_csv"] = filename
                candidates[case, method].append(row)
                case_records[case].append(row)

    targets = {}
    for case in CASES:
        records = case_records[case]
        ns = {int(r["n"]) for r in records}
        if len(ns) != 1:
            raise ValueError(f"Inconsistent system width: {case}")
        n = ns.pop()
        fixture_rows = [r for r in records if r.get("input_fixture_path")]
        canonical_hashes = {r["canonical_input_sha256"] for r in records if r.get("canonical_input_sha256")}
        if len(canonical_hashes) > 1:
            raise ValueError(f"Different target inputs were assigned the same case: {case}")
        if fixture_rows:
            source = fixture_rows[0]
            fixture = Path(source["input_fixture_path"])
            document = read(fixture)
            kraus = document["input_kraus"]
            if sha256(fixture.read_bytes()).hexdigest() != source["input_file_sha256"]:
                raise ValueError(f"Input fixture changed since collection: {case}")
            target_path = str(fixture)
            model = document.get("model", {})
        else:
            source = next(r for r in records if r["source_csv"] == SOURCES[2])
            document = read(source["source_result_path"])
            kraus = document["kraus"]
            if encoded_hash(kraus) != source["input_kraus_sha256"]:
                raise ValueError(f"VCBE input changed since collection: {case}")
            target_path = source["source_result_path"] + "#kraus"
            model = document["provenance"]
        if any(len(term[0]) != n for terms in kraus for term in terms):
            raise ValueError(f"Input Pauli width disagrees with n: {case}")
        targets[case] = dict(n=n, kraus=kraus, target_path=target_path,
                             target_kraus_sha256=encoded_hash(kraus),
                             model_parameters=packed(model))

    # Check the new joins between the original-resource and shared-Kraus tables.
    shared_errors = {}
    for case in CASES:
        vcbe_rows = [r for r in case_records[case] if r["source_csv"] == SOURCES[2]]
        if not vcbe_rows:
            continue
        record = read(vcbe_rows[0]["source_result_path"])
        shared = record["kraus"]
        if encoded_hash(shared) != vcbe_rows[0]["input_kraus_sha256"]:
            raise ValueError(f"Changed shared Kraus input: {case}")
        error = map_error(targets[case]["kraus"], shared)
        if error >= 1e-10:
            raise ValueError(f"VCBE and the original table target different maps: {case}: {error}")
        shared_errors[case] = error
        for ours in candidates[case, "Ours"]:
            if ours["source_csv"] == SOURCES[2]:
                continue
            compiled = read(ours["source_result_path"])["assembly"]["compiled_kraus"]
            if not same_representation(compiled, shared):
                raise ValueError(f"Ours and VCBE do not share the same Kraus representation: {case}")

    rows = []
    for case in CASES:
        target = targets[case]
        for method in METHODS:
            available = candidates[case, method]
            # Prefer the newer paired Ours record; never average historical runs.
            chosen = next((r for r in available if r["source_csv"] == SOURCES[1]),
                          available[0] if available else {})
            measured = [r for r in available if r["status"] in SUCCESS]
            for r in measured:
                if r.get("basis") != "NAM":
                    raise ValueError(f"Non-NAM result: {case}/{method}")
                if not math.isclose(float(r["Lambda_G"]), float(r["Lambda"]) * int(r["G"]), rel_tol=1e-12):
                    raise ValueError(f"Inconsistent weighted count: {case}/{method}")
            if measured:
                reference = measured[0]
                for other in measured[1:]:
                    if (other["G"] != reference["G"] or other["W"] != reference["W"]
                            or not math.isclose(float(other["Lambda"]), float(reference["Lambda"]), rel_tol=1e-12)):
                        raise ValueError(f"Conflicting duplicate resources: {case}/{method}")
                chosen = chosen if chosen in measured else reference
            status = chosen.get("status", "unavailable_in_source_tables")
            if method == "Direct matrix":
                status = "calculated_storage"
            row = dict(case=case, n=target["n"], method=method, status=status,
                       basis="NAM" if method != "Direct matrix" else "not_applicable",
                       G=chosen.get("G", "") if measured else "",
                       Lambda=chosen.get("Lambda", "") if method != "Direct matrix" else "",
                       Lambda_G=float(chosen["Lambda"]) * int(chosen["G"]) if measured else "",
                       W=chosen.get("W", "") if measured else "",
                       rho_bytes=16 * 4 ** target["n"] if method == "Direct matrix" else "",
                       superoperator_bytes=16 * 16 ** target["n"] if method == "Direct matrix" else "",
                       target_kraus_path=target["target_path"], target_kraus_sha256=target["target_kraus_sha256"],
                       model_parameters=target["model_parameters"],
                       canonical_input_sha256=chosen.get("canonical_input_sha256", ""),
                       shared_map_relative_error=shared_errors.get(case, ""),
                       full_choi_status=chosen.get("full_choi_status", "not_available"),
                       full_choi_relative_error=chosen.get("full_choi_relative_error", ""),
                       source_csv=chosen.get("source_csv", ""),
                       source_result_path=chosen.get("source_result_path", ""),
                       source_records=packed(available), duplicate_records=len(available))
            rows.append(row)
        by_method = {r["method"]: r for r in rows if r["case"] == case}
        raw, ours = by_method["Raw"], by_method["Ours"]
        optimizer = by_method["Raw+VOQC"]
        if optimizer["G"]:
            selected_optimizer = next(r for r in candidates[case, "Raw+VOQC"] if r["status"] == "ok")
            selected_raw = next(r for r in candidates[case, "Raw"] if r["source_csv"] == SOURCES[1])
            if (selected_optimizer["input_qasm_sha256"] != selected_raw["qasm_sha256"]
                    or optimizer["W"] != raw["W"]
                    or not math.isclose(float(optimizer["Lambda"]), float(raw["Lambda"]), rel_tol=1e-12)):
                raise ValueError(f"VOQC does not match the displayed Raw circuit: {case}")
        for r in by_method.values():
            r["Lambda_G_over_Ours"] = r["Lambda_G"] / ours["Lambda_G"] if r["Lambda_G"] != "" else ""
            r["Lambda_G_over_Raw"] = r["Lambda_G"] / raw["Lambda_G"] if r["Lambda_G"] != "" and raw["Lambda_G"] != "" else ""
    return rows


def storage(size):
    for power, unit in [(5, "PiB"), (4, "TiB"), (3, "GiB"), (2, "MiB"), (1, "KiB")]:
        if size >= 1024 ** power:
            return f"{size / 1024 ** power:g}\\,\\mathrm{{{unit}}}"
    return f"{size}\\,\\mathrm{{B}}"


def cost(value):
    rounded = f"{value:.3g}"
    if "e" in rounded:
        mantissa, exponent = rounded.split("e")
        return mantissa + r"\!\times\!10^{" + str(int(exponent)) + "}"
    return rounded


def cell(row, best_cost):
    if row["method"] == "Direct matrix":
        return r"\shortstack[r]{$" + storage(row["rho_bytes"]) + r"$\\$" + storage(row["superoperator_bytes"]) + "$}"
    if not row["G"]:
        if row["status"] == "preflight_memory_limit":
            return r"\shortstack[r]{Mem.\\---\\$(1,\text{---})$}"
        if row["status"] == "not_run_queue_paused":
            return r"\textit{Paused}"
        return "---"
    weighted = cost(row["Lambda_G"])
    if math.isclose(row["Lambda_G"], best_cost, rel_tol=1e-12):
        weighted = r"\mathbf{" + weighted + "}"
    if not row["full_choi_status"].startswith("passed"):
        weighted += r"^{\dagger}"
    return (r"\shortstack[r]{$" + weighted + r"$\\$" + f"{int(row['G']):,}"
            + r"$\\$(" + f"{float(row['Lambda']):.3g},{row['W']}" + ")$}")


def write_table(rows):
    text = [
        "% Generated by scripts/eval_redesign/build_rq1_merged_table.py.",
        "% Standalone local preview; no manuscript files are included.",
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[margin=1.5cm]{geometry}",
        r"\usepackage{booktabs,amsmath,graphicx}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{caption}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Unified resource comparison for nine benchmark inputs. Each row fixes a target completely positive map. Circuit cells show, from top to bottom, normalization-weighted gate count $C=\Lambda G$, gate count $G$, and $(\Lambda,W)$. $G$ counts full-circuit gates in the NAM basis $\{R_z,H,X,\mathrm{CNOT}\}$, and $W$ counts all circuit qubits, including auxiliary qubits. Starting with all auxiliaries in $|0\rangle$, postselecting the success ancillas to zero and tracing out the remaining auxiliary registers gives an unnormalized system map; multiplying it by $\Lambda$ recovers the target. The matrix column shows complex128 storage for the density matrix $\rho$ above the superoperator $S$. Bold $C$ marks the smallest measured value in each row.}",
        r"\label{tab:rq1-unified-resources}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
        r"\shortstack[l]{Benchmark} & \shortstack[r]{Dense matrix\\$\rho$ / $S$} & Stinespring & \shortstack[r]{Raw\\channel-LCU} & \shortstack[r]{Raw\\+VOQC} & VCBE & \shortstack[r]{VCBE\\+opt3} & \shortstack[r]{Ours\\(Pauli-LCU)} \\",
        r"\midrule",
    ]
    for case in CASES:
        if case in {"HRW-4", "RepDQE-d3", "Petz_Ising3"}:
            text.append(r"\midrule")
        measurements = {r["method"]: r for r in rows if r["case"] == case}
        best = min(r["Lambda_G"] for r in measurements.values() if r["Lambda_G"] != "")
        cells = [case.replace("Petz_Ising3", "Petz-Ising-3")]
        cells += [cell(measurements[method], best) for method in METHODS]
        text.append(" & ".join(cells) + r" \\[4pt]")
    text += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{4pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"TFIM is the transverse-field Ising model and HRW the hypercube random walk; suffixes give system qubit counts. RepDQE-d3 is the distance-3 repetition-code correction and encoded weak-measurement/reset input with three system qubits. Petz-Ising-3 is a three-qubit Petz recovery map specified by the input Kraus operators referenced in the source CSV. TFIM uses a first-order step of length 0.1, unit coupling and field, and decay rate 0.1; RepDQE uses measurement strength 0.25. Matrix sizes are calculated as $16\cdot4^n$ and $16\cdot16^n$ bytes for $n$ system qubits, and describe representation storage rather than peak process memory.",
        r"\par\smallskip",
        r"LCU denotes linear combinations of unitaries. Raw uses unoptimized channel-LCU assembly. Raw+VOQC applies the Verified Optimizer for Quantum Circuits to that Raw circuit and retains its $\Lambda$. Ours combines Kraus rewriting, Pauli placement, and control flattening; the former Pauli-LCU column is merged here after checking duplicate resources. Variational compilation of block encodings (VCBE) uses the same Kraus representation and $\Lambda$ as Ours on the four shared rows, including rewritten HRW-4. Both use the same outer state preparation and Kraus-selection tree; VCBE controls each fitted inner unitary in full, while Ours controls Pauli selection. VCBE+opt3 applies Qiskit level 3 to inner blocks before channel assembly. Final NAM lowering is at level 0 for every route. VCBE uses cached blocks from our independent reproduction of Fig.~1/Eq.~(7) in \href{https://arxiv.org/abs/2507.17658v2}{Rullk{\"o}tter et al., arXiv:2507.17658v2}. $C$ and $\Lambda$ are rounded to three significant digits; $G$ and $W$ are exact, and comparisons use full precision.",
        r"\par\smallskip",
        r"---: unavailable source measurement. Paused: the queued optimizer job has not run. Mem.: the HRW-12 Stinespring isometry requires 8 GiB, above its 3 GiB preflight budget. Stinespring uses $\Lambda=\max\{1,\|\sum_jK_j^\dagger K_j\|_2\}$ for target Kraus operators $K_j$, with upward numerical rounding when a success flag is needed. TFIM-8 Stinespring synthesis and gate counting completed; only subsequent non-Clifford classification timed out. $\dagger$: a full compiled-circuit Choi comparison is unavailable. All unmarked circuit entries passed relative Frobenius Choi checks below $10^{-10}$ after restoration by $\Lambda$. Ours rewrites and the joins to the VCBE inputs passed coefficient-space map-equivalence checks; the two 8-qubit Stinespring inputs passed isometry checks. VOQC verification covers exact Raw input hashes and saved NAM output gate counts and widths. Compilation times retain their original stage scopes in the source data and the separate scaling figure; cached VCBE fitting is excluded from its assembly times.",
        r"\par\smallskip",
        r"Source: \path{rq1_unified_resources_20261014.csv}, merged from the three saved table datasets. It retains original records, input identities, verification coverage, time scopes, and full-precision $C$, $C/C_{\rm Ours}$, and $C/C_{\rm Raw}$ where the denominator is available. Missing measurements have no numeric cost or ratio. This local preview includes the union of the nine previously displayed inputs; no new circuit jobs were launched.",
        r"\end{minipage}", r"\end{table}", r"\end{document}",
    ]
    path = OUTPUT / (STEM + ".tex")
    path.write_text("\n".join(text) + "\n")
    return path


if __name__ == "__main__":
    rows = collect()
    with (OUTPUT / (STEM + ".csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = write_table(rows)
    print(packed(dict(tex=str(path), cases=len(CASES), method_slots=len(rows),
                      measured_circuits=sum(bool(r["G"]) for r in rows),
                      duplicate_slots=sum(r["duplicate_records"] > 1 for r in rows))))
