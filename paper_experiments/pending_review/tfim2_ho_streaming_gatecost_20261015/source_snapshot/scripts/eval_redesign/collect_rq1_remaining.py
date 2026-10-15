"""Collect saved RQ1 data; never compile circuits or dispatch optimizer jobs.

The three CSVs retain configuration, input pairing, time scopes, platforms,
failed/deferred jobs, and the evidence available for each metric. Large remote
QASM files are not copied or simulated. Their local result mirrors retain the
original remote paths and saved recount evidence.
"""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PENDING = ROOT / "paper_experiments/pending_review"
RAW = PENDING / "raw_baseline_20261012"
REMOTE = PENDING / "raw_scaling_remote_20261012"
REMOTE_PROJECT = Path("/pub/data/huangqf/Lindbladian_compilation")
FULL = PENDING / "tfim_hrw_full_20261011/cases"
HYBRID = PENDING / "support_gatecost_hybrid_20261012/cases"
COUNTEREXAMPLES = PENDING / "support_gatecost_counterexamples_20261012/cases"
VCBE = ROOT / "Baseline_results/vcbe_channel_smoke_20261012"
TFIM = [4, 8, 12, 20, 32, 40, 54, 60]
HRW = [4, 8, 12, 20, 28, 48, 60, 128, 256, 512, 1024, 1440]
sys.path.insert(0, str(ROOT / "Baseline_scripts"))
from baseline_common import QREG2_RE, parse_qasm_operation


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def identity(encoded):
    return sha256(json.dumps(encoded, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def packed(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def coefficients(encoded):
    result = []
    for terms in encoded:
        row = Counter()
        for label, magnitude, real, imag in terms:
            row[label] += magnitude * complex(real, imag)
        result.append({p: c for p, c in row.items() if c != 0})
    return result


def array_coefficients(array, labels):
    return [{str(p): complex(c) for p, c in zip(labels, row) if c != 0}
            for row in array]


def sparse_gram_error(left, right):
    """Compare Kraus maps in Pauli coefficient space without dense matrices."""
    def gram(rows):
        output = Counter()
        for row in rows:
            for p, a in row.items():
                for q, b in row.items():
                    output[p, q] += a.conjugate() * b
        return output
    a, b = gram(left), gram(right)
    norm = math.sqrt(sum(abs(value) ** 2 for value in a.values()))
    return math.sqrt(sum(abs(b[key] - a[key]) ** 2 for key in a.keys() | b.keys())) / norm


def recount(path, basis="nam"):
    counts, width = Counter(), 0
    allowed = {"rz", "h", "x", "cx", "reset"} if basis == "nam" else {
        "u1", "u2", "u3", "cx", "reset"}
    with Path(path).open() as stream:
        for line in stream:
            match = QREG2_RE.search(line)
            if match:
                width += int(match.group(1))
            operation = parse_qasm_operation(line.strip())
            if operation:
                assert operation.name in allowed, (path, operation.name)
                counts[operation.name] += 1
    return sum(v for k, v in counts.items() if k != "reset"), width, dict(counts)


def validate_metrics(row, metrics):
    counts = metrics["counts"]
    assert set(counts).issubset({"rz", "h", "x", "cx", "reset"})
    assert sum(v for k, v in counts.items() if k != "reset") == metrics["G"]
    row.update(G=metrics["G"], W=metrics["W"], CX=counts.get("cx", 0),
               counts=packed(counts), metric_evidence="saved circuit count_ops and NAM basis check")


def verification_fields(verification):
    return {
        "full_choi_status": "passed" if verification.get("passed") else
                            verification.get("status", "not_recorded"),
        "full_choi_relative_error": verification.get("relative_frobenius_error", ""),
        "full_choi_scope": verification.get("method", verification.get("reason", "")),
    }


def failed_row(case, status, method, host, source, **extra):
    return dict(case=case, method=method, status=status, host=host,
                basis="NAM", G="", Lambda_G="", W="", reported_compile_s="",
                source_result_path=str(source), **extra)


def raw_row(entry, saved, host="local", remote=False):
    case = entry["case"]
    if saved["status"] not in {"ok", "reused"}:
        source = (REMOTE if remote else RAW) / "cases" / case / "resources.json"
        progress = source.parent / "progress.json"
        stage = read(progress).get("stage", "") if progress.exists() else "not_recorded"
        return failed_row(case, saved["status"], "Raw", host, source,
            Lambda=entry["Lambda_raw"], canonical_input_sha256=entry["canonicalization"]["canonical_input_sha256"],
            input_pairing="manifest matches selected input", stopped_stage=stage,
            process_wall_s=saved.get("elapsed_s", saved.get("wall_s", "")),
            peak_rss_bytes=saved.get("peak_rss_bytes", ""),
            timeout_s=3600 if remote else 600, rss_limit_bytes=(32 if remote else 6) * 1024 ** 3,
            notes=saved.get("reason", "No complete Raw circuit was produced."),
            predecessor=saved.get("predecessor", ""))
    source = Path(saved["result"])
    local_source = ROOT / source.relative_to(REMOTE_PROJECT) if remote else source
    measured = read(local_source)
    assert identity(measured["input_kraus"]) == measured["canonicalization"]["raw_input_sha256"]
    assert identity(measured["canonical_kraus"]) == measured["canonicalization"]["canonical_input_sha256"]
    assert measured["canonicalization"]["canonical_input_sha256"] == entry["canonicalization"]["canonical_input_sha256"]
    assert measured["metrics"]["G"] == saved["G"]
    assert measured["qasm_sha256"] == saved["qasm_sha256"]
    settings = measured["metrics"]["settings"]
    assert settings["basis"] == "nam" and settings["lowering_optimization_level"] == 0
    assert settings["qubits_initially_zero"] is False and settings["downstream_optimizer"] is None
    assert math.isclose(measured["Lambda"], entry["Lambda_raw"], rel_tol=1e-12)
    reused_suite = PENDING / "rq1_paper_suite_20261010/full_v2"
    if remote:
        budget_source = REMOTE / "manifest.json"
        timeout_s, rss_limit_bytes = 3600, 32 * 1024 ** 3
    elif local_source.is_relative_to(reused_suite):
        budget_source = reused_suite / "manifest.json"
        budget = read(budget_source)
        timeout_s = budget["per_job_timeout_s"]
        rss_limit_bytes = int(budget["per_job_rss_limit_gib"] * 1024 ** 3)
    else:
        budget_source = RAW / "manifest.json"
        budget = read(budget_source)
        timeout_s, rss_limit_bytes = budget["per_job_timeout_s"], budget["per_job_memory_bytes"]
    qasm = Path(measured["qasm"])
    row = dict(case=case, method="Raw", status="ok", original_run_status=saved["status"],
        host=host, basis="NAM", optimization_level=0, downstream_optimizer="none",
        configuration="R_off_O_off_Fstar_off", Lambda=measured["Lambda"],
        Lambda_G=measured["Lambda"] * measured["metrics"]["G"],
        non_clifford=measured.get("non_clifford", ""),
        reported_compile_s=measured["compile_time_s"], source_time_field="compile_time_s",
        time_scope="canonicalization + assembly + NAM lowering; excludes imports, input construction, verification and I/O",
        canonicalization_s=measured["canonicalization"]["time_s"],
        assembly_s=measured["assembly"]["assembly_time_s"],
        lowering_s=measured["metrics"]["lowering_time_s"],
        input_kraus_sha256=measured["canonicalization"]["raw_input_sha256"],
        canonical_input_sha256=measured["canonicalization"]["canonical_input_sha256"],
        input_pairing="passed identical canonical representation hash",
        qasm_path=str(qasm), qasm_sha256=measured["qasm_sha256"],
        qasm_hash_check="not_rechecked_remote_artifact" if remote else "not_rechecked_saved_hash",
        source_result_path=str(local_source), original_source_result_path=str(source),
        timeout_s=timeout_s, rss_limit_bytes=rss_limit_bytes, budget_source_path=str(budget_source),
        process_wall_s=saved.get("elapsed_s", saved.get("wall_s", "")),
        peak_rss_bytes=saved.get("peak_rss_bytes", ""),
        **verification_fields(measured.get("verification", {})))
    validate_metrics(row, measured["metrics"])
    return row, measured


def ours_historical(case, path, expected_hash=None):
    measured = read(path)
    if measured.get("status") != "ok":
        return failed_row(case, measured.get("status", "failed"), "Ours", "local", path)
    fixture = read(path.parent / "input.json")
    assert digest(path.parent / "input.json") == measured["input_sha256"]
    assert identity(fixture["input_kraus"]) == measured["canonicalization"]["raw_input_sha256"]
    if expected_hash:
        assert expected_hash == measured["canonicalization"]["canonical_input_sha256"]
    error = sparse_gram_error(coefficients(fixture["input_kraus"]),
                             coefficients(measured["assembly"]["compiled_kraus"]))
    assert error < 1e-10, (case, error)
    qasm = path.parent / "channel_nam.qasm"
    assert digest(qasm) == measured["qasm_sha256"]
    settings = measured["protocol"]
    assert settings["basis"] == "nam" and settings["lowering_optimization_level"] == 0
    assert settings["qubits_initially_zero"] is False and settings["downstream_optimizer"] is None
    row = dict(case=case, method="Ours", status="ok", host="local", basis="NAM",
        optimization_level=0, downstream_optimizer="none", configuration="tree_Fstar_Matrix_order",
        rewrite_policy=packed(measured["effective_rewrite_options"]),
        rewrite_gram_relative_error=error, Lambda=measured["Lambda"],
        Lambda_G=measured["Lambda"] * measured["G"], non_clifford=measured["non_clifford"],
        reported_compile_s=measured["compile_time_s"], source_time_field="compile_time_s",
        time_scope=measured["timing_scope"],
        input_kraus_sha256=measured["canonicalization"]["raw_input_sha256"],
        canonical_input_sha256=measured["canonicalization"]["canonical_input_sha256"],
        input_pairing="passed identical canonical representation hash and sparse Gram equivalence",
        qasm_path=str(qasm), qasm_sha256=measured["qasm_sha256"], qasm_hash_check="passed",
        source_result_path=str(path), **verification_fields(measured.get("full_choi", {})))
    for stage, seconds in measured["exclusive_stage_times_s"].items():
        row["stage_" + stage + "_s"] = seconds
    validate_metrics(row, measured)
    return row


def ours_saved_rewrite(case, path, raw_measured, method):
    directory = path.parent
    measured = read(path)
    search = read(directory / "results.json")
    manifest = read(directory / "manifest.json")
    with np.load(directory / (method + ".npz")) as arrays:
        before = array_coefficients(arrays["A_initial"], arrays["labels"])
        after = array_coefficients(arrays["A_final"], arrays["labels"])
        if method == "hybrid":
            assert sha256(arrays["A_initial"].tobytes()).hexdigest() == manifest["input_matrix_sha256"]
    input_error = sparse_gram_error(coefficients(raw_measured["canonical_kraus"]), before)
    rewrite_error = sparse_gram_error(before, after)
    assert max(input_error, rewrite_error) < 1e-10, (case, input_error, rewrite_error)
    stored_method = search["methods"][method] if "methods" in search else search[method]
    qasm_method = measured.get("reused_identical_output_of", method)
    qasm = directory / (qasm_method + "_nam.qasm")
    assert qasm.exists(), qasm
    G, W, counts = recount(qasm)
    assert (G, W) == (measured["G"], measured["W"])
    assert counts == measured["qasm_recount"]["ops"]
    rewrite_s = stored_method.get("wall_s", "")
    settings = measured["settings"]
    assert settings["basis"] == "nam" and settings["lowering_optimization_level"] == 0
    assert settings["qubits_initially_zero"] is False and settings["downstream_optimizer"] is None
    row = dict(case=case, method="Ours", status="ok", host="local", basis="NAM",
        configuration="tree_Fstar_Matrix_order", optimization_level=0, downstream_optimizer="none",
        rewrite_policy="support_frontier_then_gatecost_polish_hybrid" if method == "hybrid" else
                       "supplementary_gatecost_streaming_beam; not a final hybrid measurement",
        rewrite_settings=packed(manifest), rewrite_s=rewrite_s,
        reported_compile_s=measured["compile_s"], source_time_field="compile_s",
        time_scope="assembly + NAM lowering and in-memory metric collection; cached rewrite, verification and QASM I/O/recount excluded",
        combined_rewrite_compile_s=rewrite_s + measured["compile_s"] if isinstance(rewrite_s, (int, float)) else "",
        combined_time_scope="sum of separately timed rewrite and compilation; excludes canonicalization and JIT warmup" if rewrite_s != "" else "",
        assembly_s=measured["assembly_s"], lowering_s=measured["lowering_s"],
        G=G, W=W, CX=counts.get("cx", 0), counts=packed(counts),
        Lambda=measured["Lambda"], Lambda_G=measured["Lambda"] * G,
        non_clifford=measured.get("NC", measured["qasm_recount"]["clifford"]["non_clifford"]),
        canonical_input_sha256=raw_measured["canonicalization"]["canonical_input_sha256"],
        input_pairing="passed sparse coefficient Gram equivalence to the common input map",
        input_pairing_gram_relative_error=input_error, rewrite_gram_relative_error=rewrite_error,
        qasm_path=str(qasm), qasm_sha256=digest(qasm), qasm_hash_check="computed_current_saved_output",
        metric_evidence="saved NAM QASM recounted", source_result_path=str(path),
        rewrite_source_result_path=str(directory / "results.json"),
        **verification_fields(measured.get("verification", {})))
    return row


def optimizer_row(case, tool, raw_entry, raw_measured, manifest):
    path = RAW / "optimizers" / case / tool / "results.json"
    if not path.exists():
        status = "not_run_queue_paused" if raw_measured is not None else "not_run_raw_unavailable"
        return failed_row(case, status, "Raw+" + tool.upper(), "local", path,
            Lambda=raw_entry["Lambda_raw"], timeout_s=3600, rss_limit_bytes=6 * 1024 ** 3,
            notes="Queue remains paused; no new jobs dispatched by this collector.")
    measured = read(path)
    assert raw_measured is not None
    assert measured["input_sha256"] == raw_measured["qasm_sha256"]
    assert measured["input_G"] == raw_measured["metrics"]["G"]
    assert math.isclose(measured["Lambda"], raw_measured["Lambda"], rel_tol=1e-12)
    # This inexpensive hash check binds every completed optimizer job, including
    # timeouts, to the exact Raw file; results are not paired by case names alone.
    assert digest(raw_measured["qasm"]) == measured["input_sha256"]
    resource = measured["tool_resources"]
    row = failed_row(case, measured["status"], "Raw+" + tool.upper(), "local", path,
        Lambda=measured["Lambda"], tool=tool, input_G=measured["input_G"],
        input_qasm_sha256=measured["input_sha256"], tool_sha256=measured["tool_sha256"],
        input_pairing="passed exact Raw QASM hash", canonical_input_sha256=raw_measured["canonicalization"]["canonical_input_sha256"],
        optimizer_s=resource["elapsed_seconds"], process_wall_s=resource["elapsed_seconds"],
        time_scope="optimizer process only; excludes Raw generation and adapter conversion/recount",
        timeout_s=manifest["optimizer_stage"]["timeout_s"], rss_limit_bytes=manifest["per_job_memory_bytes"],
        peak_rss_bytes=resource["peak_rss_bytes"], stopped_stage=resource.get("resource_stop", ""),
        channel_equivalence_status="not_run", channel_equivalence_scope=measured["validation"])
    job = path.parent / "job.json"
    if job.exists():
        job_result = read(job)
        row.update(adapter_wall_s=job_result.get("wall_s", ""),
                   adapter_peak_rss_bytes=job_result.get("adapter_peak_rss_bytes", ""))
    if measured["status"] == "ok":
        adapter = read(measured["adapter_result"])
        assert adapter["input"] == raw_measured["qasm"]
        output = Path(adapter["output_stats"]["path"])
        G, W, counts = recount(output)
        assert G == measured["G"] == adapter["output_stats"]["metric_total"]
        assert W == adapter["output_stats"]["num_qubits"]
        assert counts == adapter["output_stats"]["ops"]
        row.update(G=G, W=W, CX=counts.get("cx", 0), Lambda_G=G * measured["Lambda"],
            non_clifford=adapter["output_stats"]["clifford"]["non_clifford"],
            qasm_path=str(output), qasm_sha256=digest(output), qasm_hash_check="computed_current_saved_output",
            metric_evidence="saved NAM output recounted", G_over_Raw=G / measured["input_G"],
            Lambda_G_over_Raw=G / measured["input_G"])
    else:
        assert measured["G"] is None
    return row


def collect_optimizers():
    manifest = read(RAW / "manifest.json")
    saved = {r["case"]: r for r in read(RAW / "results.json")}
    paused = read(RAW / "optimizer_status.json")
    assert paused["status"] == "dispatch_paused", "Inspect changed queue status before collecting"
    rows = []
    for entry in manifest["cases"]:
        case = entry["case"]
        first_row = len(rows)
        result = raw_row(entry, saved[case])
        raw, raw_measured = result if isinstance(result, tuple) else (result, None)
        rows.append(raw)
        hybrid = HYBRID / case / "hybrid_circuit.json"
        historical = Path(entry["source"]).parent / "results.json"
        supplemental = COUNTEREXAMPLES / case / "gatecost_circuit.json"
        if hybrid.exists():
            reference = raw_measured or {
                "canonical_kraus": read(entry["input"])["input_kraus"],
                "canonicalization": entry["canonicalization"],
            }
            ours = ours_saved_rewrite(case, hybrid, reference, "hybrid")
        elif historical.exists() and "input.json" == Path(entry["source"]).name:
            content = read(historical)
            ours = ours_historical(case, historical, entry["canonicalization"]["canonical_input_sha256"]) if "compile_time_s" in content else failed_row(
                case, "not_measured", "Ours", "local", historical,
                notes="Rewrite record exists; no historical full-pipeline result.")
        elif supplemental.exists() and raw_measured is not None:
            ours = ours_saved_rewrite(case, supplemental, raw_measured, "gatecost")
        else:
            ours = failed_row(case, "not_measured", "Ours", "local", entry["source"],
                notes="No paired complete Ours compilation in the inspected batches.")
        if raw_measured is not None and ours["status"] == "ok":
            ours.update(G_over_Raw=ours["G"] / raw["G"],
                        Lambda_G_over_Raw=ours["Lambda_G"] / raw["Lambda_G"])
        rows.append(ours)
        for tool in ("voqc", "feynman"):
            rows.append(optimizer_row(case, tool, entry, raw_measured, manifest))
        for row in rows[first_row:]:
            row.update(n=entry["system_qubits"], inventory_group=entry["group"],
                       input_fixture_path=entry["input"], input_file_sha256=entry["input_sha256"])
    completed = [r for r in rows if r["method"].startswith("Raw+") and
                 r["status"] not in {"not_run_queue_paused", "not_run_raw_unavailable"}]
    assert len(completed) == len(list((RAW / "optimizers").glob("*/*/results.json")))
    print(packed({"dataset": "raw_optimizer", "rows": len(rows), "cases": len(saved),
        "completed_optimizer_jobs": len(completed),
        "optimizer_statuses": dict(Counter(r["status"] for r in completed)),
        "available_ours": sum(r["method"] == "Ours" and r["status"] == "ok" for r in rows),
        "queue": paused["status"]}), flush=True)
    return rows


def collect_scaling():
    manifest = read(RAW / "manifest.json")
    entries = {e["case"]: e for e in manifest["cases"]}
    saved = {r["case"]: r for r in read(RAW / "results.json")}
    remote = {r["case"]: r for r in read(REMOTE / "results.json")}
    rows = []
    for family, sizes in [("TFIM", TFIM), ("HRW", HRW)]:
        for n in sizes:
            case = f"{family}-{n}"
            ours_path = (ROOT / "Baseline_results/table1_scaling_ours_20261013/cases/HRW-8/results.json"
                         if case == "HRW-8" else FULL / case / "results.json")
            ours = ours_historical(case, ours_path)
            assert ours["status"] == "ok"
            if case == "HRW-8":
                path = PENDING / "rq1_paper_suite_20261010/full_v2/runs/HRW-8/raw/results.json"
                measurement = read(path)
                entry = dict(case=case, Lambda_raw=measurement["Lambda"], canonicalization=measurement["canonicalization"])
                source = dict(case=case, status="reused", result=str(path), G=measurement["metrics"]["G"],
                              qasm_sha256=measurement["qasm_sha256"])
            else:
                entry, source = entries[case], saved[case]
            result = raw_row(entry, source)
            raw = result[0] if isinstance(result, tuple) else result
            assert raw["canonical_input_sha256"] == ours["canonical_input_sha256"]
            for row in [raw, ours]:
                row.update(family=family, n=n, repetitions=1, warmups=0,
                    numerical_threads=1, timing_statistic="single observed compilation",
                    cpu="Apple M4", host_memory_gib=16)
                if row["method"] == "Ours":
                    row.update(timeout_s="", rss_limit_bytes="",
                        time_budget_policy="no imposed per-case wall timeout",
                        rss_budget_policy="no imposed per-case RSS cap",
                        budget_source_path=str(ROOT / "Baseline_results/table1_scaling_ours_20261013/run_hrw8.py"
                            if case == "HRW-8" else FULL.parent / "source_snapshot/scripts/eval_redesign/run_selected_tfim_hrw.py"))
                rows.append(row)
            if case in remote:
                result = raw_row(entry, remote[case], host="remote", remote=True)
                row = result[0] if isinstance(result, tuple) else result
                row.update(family=family, n=n, repetitions=1, warmups=0, numerical_threads=1,
                    timing_statistic="single observed compilation", cpu="dual AMD EPYC 9534", host_memory_gib=503.4)
                rows.append(row)
    assert sum(r["method"] == "Ours" for r in rows) == 20
    print(packed({"dataset": "scaling", "rows": len(rows), "ours_successes": 20,
        "raw_by_host_status": dict(Counter(r["host"] + ":" + r["status"] for r in rows if r["method"] == "Raw"))}), flush=True)
    return rows


def collect_vcbe():
    batch = read(VCBE / "results.json")
    rows = []
    for case, saved in batch["cases"].items():
        path = VCBE / "cases" / case / "results.json"
        record = read(path)
        assert record == saved and record["status"] == "success"
        # Bind every route to the runner's exact serialized Kraus representation.
        encoded = record["kraus"]
        assert identity(encoded) == record["input_sha256"]
        alpha = [sum(abs(c) for c in row.values()) for row in coefficients(encoded)]
        assert np.allclose(alpha, record["alpha"], rtol=1e-12, atol=1e-14)
        lam = sum(a ** 2 for a in alpha)
        assert math.isclose(lam, record["Lambda"], rel_tol=1e-12)
        reference_error = ""
        provenance = record["provenance"]
        if isinstance(provenance, str):
            source = Path(provenance)
            if source.is_file():
                target = read(source)["input_kraus"]
            else:
                target = read(source / "results.json")["assembly"]["compiled_kraus"] if case.endswith("rewritten") else read(source / "input.json")["input_kraus"]
            reference_error = sparse_gram_error(coefficients(target), coefficients(encoded))
            assert reference_error < 1e-10
        unique_fits = {b["template"]: b["original_search_s"] for b in record["vcbe_blocks"]}
        assert math.isclose(sum(unique_fits.values()), record["cached_unique_fit_s"], rel_tol=1e-12)
        for method, measured in record["methods"].items():
            for setting, metric in measured["metrics"].items():
                basis, level = setting.split("_opt")
                qasm = path.parent / f"{method}_{setting}.qasm"
                assert digest(qasm) == metric["qasm_sha256"]
                G, W, counts = recount(qasm, basis)
                assert (G, W, counts) == (metric["G"], metric["width"], metric["counts"])
                assert math.isclose(metric["Lambda_G"], lam * G, rel_tol=1e-12)
                verification = metric["validation"]
                assert verification["passed"] and verification["relative_choi_frobenius_error"] < batch["tolerance"]
                row = dict(case=case, method=method, status="ok", n=record["n"], M=record["M"],
                    shared_rewrite="HRW common rewritten representation" if case.endswith("rewritten") else
                                   "raw HRW diagnostic" if case.endswith("raw") else "TFIM rewrite is neutral",
                    primary_table_candidate=basis == "nam" and level == "0" and not case.endswith("raw"),
                    host="local", basis=basis.upper(), final_optimization_level=int(level),
                    inner_preoptimization_level=3 if method == "vcbe_preopt3_tree" else 0,
                    input_kraus_sha256=record["input_sha256"], input_pairing="same Kraus representation and alpha for all methods",
                    source_pairing_gram_relative_error=reference_error,
                    Lambda=lam, G=G, W=W, CX=metric["CX"], non_clifford=metric["NC"], Lambda_G=lam * G,
                    alpha_per_kraus=packed(alpha), assembly_s=measured["assembly_s"],
                    lowering_s=metric["lowering_s"], reported_compile_s=metric["assembly_plus_lowering_s"],
                    time_scope="assembly + final lowering; cached block fit and Kraus rewrite excluded",
                    cached_unique_fit_s=record["cached_unique_fit_s"] if method.startswith("vcbe") else "",
                    fit_time_scope="historical deduplicated template fitting; not rerun or an end-to-end compilation" if method.startswith("vcbe") else "",
                    vcbe_implementation=batch["vcbe"] if method.startswith("vcbe") else "",
                    vcbe_paper_url="https://arxiv.org/abs/2507.17658v2" if method.startswith("vcbe") else "",
                    vcbe_reference_title="Resource-efficient Variational Compilation of Block-Encodings" if method.startswith("vcbe") else "",
                    control_policy="control entire synthesized unitary" if method.startswith("vcbe") else "Fstar: control SELECT only",
                    outer_policy="shared unary-iteration tree and outer PREPARE",
                    full_choi_status="passed", full_choi_relative_error=verification["relative_choi_frobenius_error"],
                    max_success_branch_error=verification["max_branch_frobenius_error"],
                    tree_ancilla_leakage_probability=verification["tree_ancilla_leakage_probability"],
                    tolerance=batch["tolerance"], qasm_path=str(qasm), qasm_sha256=metric["qasm_sha256"],
                    metric_evidence="saved QASM hash, basis, width and gate counts verified",
                    source_result_path=str(path), counts=packed(counts))
                rows.append(row)
    assert len(rows) == 60
    print(packed({"dataset": "vcbe", "rows": len(rows), "shared_inputs": len(batch["cases"]),
        "max_relative_choi_error": max(r["full_choi_relative_error"] for r in rows)}), flush=True)
    return rows


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


if __name__ == "__main__":
    raw_rows = collect_optimizers()
    scaling_rows = collect_scaling()
    vcbe_rows = collect_vcbe()
    for name, rows in [("raw_optimizer", raw_rows), ("scaling", scaling_rows), ("pauli_vcbe", vcbe_rows)]:
        write_csv(ROOT / "Results" / f"rq1_{name}_20261014.csv", rows)
