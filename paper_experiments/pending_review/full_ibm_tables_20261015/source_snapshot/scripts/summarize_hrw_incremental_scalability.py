#!/usr/bin/env python3
"""Aggregate HRW greedy_incremental attempts and audit their saved artifacts.

No compilation, simulation, dense channel construction, or circuit reload occurs.
NPZ terminal rewrites are checked numerically; circuit round trips are the checks
recorded by the runner, cross-checked against its independent program report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = ROOT / "Results/hrw_incremental_scalability_20260909"
BASIS = {"ibm": ["u1", "u2", "u3", "cx", "reset"],
         "nam": ["h", "x", "rz", "cx", "reset"]}
TIMING_DEFINITIONS = {
    "ibm_compile_time_s": "rewrite + circuit build + IBM lowering",
    "nam_compile_time_s": "rewrite + circuit build + NAM lowering",
    "compile_exclusions": "model, imports, provenance, exports, counts, and validation",
    "logical_compile_time_s": "model + rewrite + circuit build",
    "program_generation_time_s": "logical compile + IBM lowering + QPY/program export",
    "case_wall_time_s": "model start through exports/counts/validation; excludes imports/provenance",
    "process_wall_time_s": "worker launch through exit; includes imports/provenance",
    "case_wall_limit_s": "configured case-wall budget, not a completed compile time",
    "parent_watchdog_s": "configured parent watchdog, not a completed compile time",
    "program_check_limit_s": "configured executable-program validation budget",
    "peak_rss_mib": "worker peak resident memory, MiB = 2**20 bytes; null if unrecorded",
    "statevector_complex128_bytes": "16 * 2**width estimate; not an observed allocation or OOM",
    "progress_accepted_rewrite_steps": "last logged accepted Step index + 1; observed lower bound, not completed rewrite",
    "progress_support": "support after the last logged accepted rewrite; not final_support",
}


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def check(checks, name, passed, detail=None):
    checks[name] = {"status": "unavailable" if passed is None else "pass" if bool(passed) else "fail"}
    if detail is not None:
        checks[name]["detail"] = detail


def equality(checks, name, observed, expected):
    check(checks, name, None if observed is None else observed == expected,
          {"observed": observed, "expected": expected})


def close(checks, name, observed, expected, tol=1e-9):
    valid = (isinstance(observed, (int, float)) and not isinstance(observed, bool)
             and math.isfinite(observed) and math.isclose(observed, expected, rel_tol=tol, abs_tol=tol))
    check(checks, name, None if observed is None else valid,
          {"observed": observed, "expected": expected})


def audit_root(root):
    result = {"root": str(root), "checks": {}, "manifest": {}}
    checks = result["checks"]
    manifest_path = root / "source_manifest.json"
    archive_path = root / "compiler_sources.executed.tar.gz"
    try:
        manifest = read_json(manifest_path)
        result.update(manifest=manifest, manifest_sha256=sha256(manifest_path))
        check(checks, "manifest_readable", True)
    except (OSError, ValueError) as exc:
        check(checks, "manifest_readable", None, str(exc))
        return result
    hashes = manifest.get("source_sha256", {})
    check(checks, "manifest_source_hashes_present", bool(hashes))
    try:
        digest = sha256(archive_path)
        result["archive_sha256"] = digest
        equality(checks, "archive_hash_matches_manifest", digest, manifest.get("archive_sha256"))
        archived = {}
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    with archive.extractfile(member) as source:
                        if member.name in archived:
                            raise ValueError(f"Duplicate archive member: {member.name}")
                        archived[member.name] = hashlib.file_digest(source, "sha256").hexdigest()
        mismatches = sorted(name for name in hashes if archived.get(name) != hashes[name])
        result["archived_source_sha256"] = archived
        check(checks, "archived_sources_match_manifest", bool(hashes) and not mismatches,
              {"mismatched_or_missing_sources": mismatches})
    except (OSError, ValueError, tarfile.TarError) as exc:
        check(checks, "archive_readable", None, str(exc))
    return result


def audit_arrays(path, n, checks):
    import numpy as np

    if not path.is_file():
        check(checks, "rewrite_arrays_readable", None, "rewrite_arrays.npz is missing")
        return
    try:
        with np.load(path, allow_pickle=False) as arrays:
            a = arrays["A_final"]
            labels = arrays["labels"].tolist()
            pairs = arrays["pairs"]
            unitaries = arrays["unitaries"]
            trajectory = arrays["support_trajectory"]
        check(checks, "rewrite_arrays_readable", True)
        equality(checks, "terminal_array_shape", list(a.shape), [2 * n, 3 * n + 1])
        valid_labels = (isinstance(labels, list) and len(labels) == 3 * n + 1
                        and all(isinstance(label, str) and len(label) == n for label in labels)
                        and len(set(labels)) == len(labels))
        check(checks, "terminal_labels_unique_and_width_correct", valid_labels)
        check(checks, "consecutive_neighbor_pairs", np.array_equal(pairs, np.arange(2 * n).reshape(n, 2)))
        rotation = np.array([[1, -1], [1, 1]], dtype=complex) / math.sqrt(2)
        check(checks, "rotation_sequence", unitaries.shape == (n, 2, 2)
              and np.allclose(unitaries, rotation, rtol=0, atol=1e-12),
              "Each U = [[1,-1],[1,1]] / sqrt(2)")
        check(checks, "support_trajectory", np.array_equal(trajectory, np.arange(8 * n, 4 * n - 1, -4)))
        shape_valid = a.shape == (2 * n, 3 * n + 1)
        finite = bool(np.all(np.isfinite(a)))
        check(checks, "terminal_coefficients_finite", finite)
        if not shape_valid or not valid_labels or not finite:
            check(checks, "terminal_kraus_formula", False, "Invalid dimensions, labels, or nonfinite entries")
            return
        amplitude = 1 / (2 * math.sqrt(n))
        index = {label: j for j, label in enumerate(labels)}
        row_support_ok = magnitude_ok = formula_ok = True
        # Qiskit Pauli strings put system qubit zero at the rightmost position.
        for i in range(n):
            label = lambda p: "I" * (n - i - 1) + p + "I" * i
            expected = ({label("Z"): amplitude, label("Y"): -1j * amplitude},
                        {"I" * n: amplitude, label("X"): amplitude})
            for row, coefficients in zip((2 * i, 2 * i + 1), expected):
                nonzero = np.flatnonzero(np.abs(a[row]) > 1e-12)
                row_support_ok &= len(nonzero) == 2
                magnitude_ok &= bool(np.allclose(np.abs(a[row, nonzero]), amplitude, rtol=1e-10, atol=1e-12))
                expected_indices = {index[key] for key in coefficients if key in index}
                formula_ok &= len(expected_indices) == 2 and set(nonzero.tolist()) == expected_indices
                formula_ok &= all(key in index and np.isclose(a[row, index[key]], value, rtol=1e-10, atol=1e-12)
                                  for key, value in coefficients.items())
        check(checks, "two_nonzero_coefficients_per_kraus", row_support_ok)
        check(checks, "terminal_nonzero_magnitude", magnitude_ok, {"expected": amplitude})
        check(checks, "terminal_kraus_formula", formula_ok,
              "K[2i]=(Z_i-iY_i)/(2*sqrt(n)); K[2i+1]=(I+X_i)/(2*sqrt(n))")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        check(checks, "rewrite_arrays_readable", False, str(exc))


def artifact_checks(summary, source, manifest, checks):
    artifacts = {}
    names = ("rewrite_arrays.npz", "rewrite_result.json", "circuit_ibm.qpy",
             "circuit_ibm_opt0.qasm", "circuit_nam_opt0.qasm", "qiskit_program.py", "program_check.log")
    for name in names:
        path = source.parent / name
        try:
            size = path.stat().st_size
            artifacts[name] = {"path": str(path), "bytes": size, "sha256": sha256(path)}
            check(checks, f"artifact_nonempty:{name}", size > 0)
        except OSError as exc:
            artifacts[name] = {"path": str(path), "error": str(exc)}
            check(checks, f"artifact_nonempty:{name}", None, str(exc))
    program_hash = artifacts.get("qiskit_program.py", {}).get("sha256")
    template_hash = manifest.get("source_sha256", {}).get("scripts/scalability_qiskit_program.py")
    check(checks, "program_matches_archived_template", None if not program_hash or not template_hash
          else program_hash == template_hash)
    equality(checks, "runner_qpy_roundtrip_equal", summary.get("qpy_roundtrip_equal"), True)
    equality(checks, "runner_program_check_returncode", summary.get("program_check_returncode"), 0)
    for basis in BASIS:
        entry = summary.get("basis_results", {}).get(basis, {})
        equality(checks, f"runner_{basis}_qasm_roundtrip", entry.get("qasm_roundtrip_width_ops_equal"), True)
        stats, exported = entry.get("stats", {}), entry.get("export", {})
        equality(checks, f"{basis}_counted_width", stats.get("num_qubits"), summary.get("width"))
        equality(checks, f"{basis}_exported_width", exported.get("num_qubits"), summary.get("width"))
        check(checks, f"{basis}_count_and_export_ops_match", None if "ops" not in stats or "ops" not in exported
              else stats["ops"] == exported["ops"])
        if "ops" in stats:
            equality(checks, f"{basis}_native_gate_total", stats.get("metric_total"),
                     sum(count for gate, count in stats["ops"].items() if gate != "reset"))
            check(checks, f"{basis}_native_gates_only", set(stats["ops"]).issubset(BASIS[basis]))
        else:
            check(checks, f"{basis}_native_gate_total", None)
            check(checks, f"{basis}_native_gates_only", None)
        equality(checks, f"{basis}_unknown_gate_classifications", stats.get("clifford", {}).get("unknown"), 0)
    try:
        report = read_json(source.parent / "program_check.log")
        equality(checks, "program_report_status", report.get("status"), "ok")
        equality(checks, "program_report_action", report.get("action"), "check")
        equality(checks, "program_report_width", report.get("num_qubits"), summary.get("width"))
        ibm_export = summary.get("basis_results", {}).get("ibm", {}).get("export", {})
        for field, expected_field in (("operations", "ops"), ("size", "size"), ("depth", "depth")):
            observed, expected = report.get(field), ibm_export.get(expected_field)
            check(checks, f"program_report_{field}", None if observed is None or expected is None else observed == expected)
    except (OSError, ValueError) as exc:
        check(checks, "program_report_readable", None, str(exc))
    return artifacts


def rewrite_progress(summary, source):
    progress = dict.fromkeys(("progress_log_path", "progress_last_accepted_step_index",
                              "progress_accepted_rewrite_steps", "progress_support",
                              "progress_log_line_number", "progress_read_error"))
    if summary.get("status") not in ("timeout", "validation_timeout", "running"):
        return progress
    path = source.parent.parent / f"{source.parent.name}.log"
    if not path.is_file() and summary.get("log_path"):
        path = Path(summary["log_path"])
    progress["progress_log_path"] = str(path)
    accepted = re.compile(r"^\s*Step (\d+): apply U on pair \(\d+,\s*\d+\), delta=-?\d+, support=(\d+)\s*$")
    try:
        with path.open(encoding="utf-8", errors="replace") as log:
            for line_number, line in enumerate(log, 1):
                match = accepted.fullmatch(line)
                if match:
                    step, support = map(int, match.groups())
                    progress.update(progress_last_accepted_step_index=step,
                                    progress_accepted_rewrite_steps=step + 1,
                                    progress_support=support, progress_log_line_number=line_number)
    except OSError as exc:
        progress["progress_read_error"] = str(exc)
    return progress


def timeout_class(row):
    if row["status"] not in ("timeout", "validation_timeout"):
        return None
    stage = row.get("stage") or ""
    both_complete = all(row[f"{basis}_compile_status"] == "complete" for basis in BASIS)
    if row["status"] == "validation_timeout" or stage.startswith(("export_", "validate_", "count_")):
        return "post_compile_export_or_validation" if both_complete else "interleaved_export_or_validation"
    if both_complete:
        return "post_compile_other"
    if stage in ("model", "rewrite", "circuit_build", "lower_ibm", "lower_nam"):
        return "compile_incomplete"
    return "unknown"


def make_row(summary, source, n, role):
    config = summary.get("configuration", {})
    status = summary.get("status", "unknown")
    rss = summary.get("peak_rss_bytes")
    row = {
        "case": f"HRW-{n}", "system_qubits": n, "role": role,
        "rewrite_strategy": config.get("rewrite_strategy"), "status": status,
        "stage": summary.get("stage"),
        "failure_stage": summary.get("stage") if status != "ok" else None,
        "error": summary.get("error"), "timeout_source": summary.get("timeout_source"),
        "outcome_censored": status in ("timeout", "validation_timeout", "killed", "interrupted", "running"),
        "max_steps": config.get("max_steps"), "case_wall_limit_s": config.get("case_wall_limit_s"),
        "parent_watchdog_s": config.get("parent_watchdog_s"),
        "program_check_limit_s": config.get("program_check_limit_s"),
        "total_qubits": summary.get("width"), "ancilla_qubits": summary.get("ancilla_qubits"),
        "model_time_s": summary.get("model_time_s"), "rewrite_time_s": summary.get("rewrite_time_s"),
        "circuit_build_time_s": summary.get("circuit_build_time_s"),
        "logical_compile_time_s": summary.get("logical_compile_time_s"),
        "program_generation_time_s": summary.get("program_generation_time_s"),
        "case_wall_time_s": summary.get("case_wall_time_s"),
        "process_wall_time_s": summary.get("process_wall_time_s"),
        "failed_stage_elapsed_s": summary.get("failed_stage_elapsed_s"),
        "peak_rss_mib": None if rss is None else rss / 2**20,
        "statevector_complex128_bytes": summary.get("statevector_complex128_bytes"),
        "simulation_status": summary.get("simulation_status"),
        "process_returncode": summary.get("process_returncode"),
        "termination_signal": summary.get("termination_signal"),
        "rewrite_steps": summary.get("rewrite", {}).get("steps"),
        "initial_support": summary.get("rewrite", {}).get("initial_support"),
        "final_support": summary.get("rewrite", {}).get("final_support"),
        "stop_reason": (summary.get("rewrite", {}).get("termination") or {}).get("stop_reason"),
        "subnormalization": summary.get("subnormalization"),
        "started_at_utc": summary.get("started_at_utc"), "finished_at_utc": summary.get("finished_at_utc"),
        "source_path": str(source),
    }
    for basis in BASIS:
        entry = summary.get("basis_results", {}).get(basis, {})
        row[f"{basis}_compile_status"] = entry.get("compile_status")
        for field in ("compile_time_s", "frontend_inclusive_compile_time_s", "lowering_time_s",
                      "basis_export_time_s", "count_time_s"):
            row[f"{basis}_{field}"] = entry.get(field)
        row[f"{basis}_gates"] = entry.get("stats", {}).get("metric_total")
        row[f"{basis}_non_clifford"] = entry.get("stats", {}).get("clifford", {}).get("non_clifford")
    row["timeout_class"] = timeout_class(row)
    row.update(rewrite_progress(summary, source))
    return row


def audit_case(summary, source, n, root_audit):
    checks = {f"root:{key}": value for key, value in root_audit["checks"].items()}
    manifest = root_audit["manifest"]
    config = summary.get("configuration", {})
    for key, expected in (("scope", "full_channel"), ("rewrite_strategy", "greedy_incremental"),
                          ("optimization_level", 0), ("basis", BASIS)):
        equality(checks, f"configuration:{key}", config.get(key), expected)
    equality(checks, "simulation_not_run", summary.get("simulation_status"), "not_run_compile_only")
    equality(checks, "launch_source_check", summary.get("source_hashes_match_launch"), True)
    case_hashes, root_hashes = summary.get("source_sha256"), manifest.get("source_sha256")
    check(checks, "case_sources_match_manifest", None if not case_hashes or not root_hashes else case_hashes == root_hashes)
    equality(checks, "case_in_manifest", None if "cases" not in manifest else ["hrw", n] in manifest["cases"], True)
    expected_width = n + 2 * (2 * n - 1).bit_length() + 1
    equality(checks, "total_width", summary.get("width"), expected_width)
    equality(checks, "ancilla_width", summary.get("ancilla_qubits"), expected_width - n)
    rewrite = summary.get("rewrite", {})
    for key, expected in (("steps", n), ("initial_support", 8 * n), ("final_support", 4 * n)):
        equality(checks, f"rewrite:{key}", rewrite.get(key), expected)
    termination = rewrite.get("termination") or {}
    equality(checks, "rewrite:stop_reason", termination.get("stop_reason"), "no_improvement")
    equality(checks, "rewrite:iterations", termination.get("iterations"), n)
    equality(checks, "rewrite:summary_trajectory", termination.get("support_trajectory"), list(range(8 * n, 4 * n - 1, -4)))
    for section, terms, max_terms in (("raw_channel_metrics", 8 * n, 4), ("rewritten_channel_metrics", 4 * n, 2)):
        for key, expected in (("kraus_count", 2 * n), ("pauli_terms_total", terms),
                              ("max_terms_per_kraus", max_terms), ("system_qubits", n)):
            equality(checks, f"{section}:{key}", summary.get(section, {}).get(key), expected)
    close(checks, "subnormalization", summary.get("subnormalization"), 2)
    for basis in BASIS:
        entry = summary.get("basis_results", {}).get(basis, {})
        equality(checks, f"{basis}_compile_completed", entry.get("compile_status"), "complete")
        components = [summary.get("rewrite_time_s"), summary.get("circuit_build_time_s"), entry.get("lowering_time_s")]
        if all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in components):
            close(checks, f"{basis}_compile_time_definition", entry.get("compile_time_s"), sum(components))
        else:
            check(checks, f"{basis}_compile_time_definition", None)
    audit_arrays(source.parent / "rewrite_arrays.npz", n, checks)
    artifacts = artifact_checks(summary, source, manifest, checks)
    failed = [name for name, item in checks.items() if item["status"] == "fail"]
    unavailable = [name for name, item in checks.items() if item["status"] == "unavailable"]
    validation_status = "failed" if failed else "incomplete" if unavailable else "verified"
    return {"status": validation_status, "failed_checks": failed, "unavailable_checks": unavailable,
            "checks": checks, "artifacts": artifacts,
            "limits": ["No full semantic channel equivalence or simulation check.",
                       "QPY/QASM round trips were performed by the runner; the summarizer does not reload circuits.",
                       "Source hashes were recorded at launch; no end-of-run source hash was recorded."]}


def collect(roots, reference_sources):
    rows, details, root_reports = [], [], []
    seen = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Result root is not a directory: {root}")
        audit = audit_root(root)
        root_reports.append(audit)
        candidates = {path.parent.name: path for path in root.glob("hrw_*/summary.json")}
        for family, size in audit["manifest"].get("cases", []):
            if family == "hrw":
                candidates.setdefault(f"hrw_{size}", root / f"hrw_{size}/summary.json")
        for directory, source in sorted(candidates.items()):
            if source in seen:
                continue
            seen.add(source)
            try:
                n = int(directory.removeprefix("hrw_"))
                if n < 1:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid HRW case directory: {source.parent}") from None
            try:
                summary = read_json(source)
            except (OSError, ValueError) as exc:
                started = source.exists() or (root / f"{directory}.log").exists()
                summary = {"status": "unreadable_summary" if source.exists() else "missing_summary" if started else "not_started",
                           "error": str(exc)}
            strategy = summary.get("configuration", {}).get("rewrite_strategy", audit["manifest"].get("rewrite_strategy"))
            if strategy != "greedy_incremental":
                raise ValueError(f"Expected greedy_incremental, found {strategy!r}: {source}")
            if summary.get("n", n) != n or summary.get("family", "hrw") != "hrw":
                raise ValueError(f"Case identity disagrees with directory: {source}")
            row = make_row(summary, source, n, "incremental_scalability")
            validation = audit_case(summary, source, n, audit)
            row["validation_status"] = validation["status"]
            row["validated_success"] = row["status"] == "ok" and validation["status"] == "verified"
            rows.append(row)
            details.append({"source_path": str(source), "summary_sha256": sha256(source) if source.is_file() else None,
                            "configuration": summary.get("configuration", {}), "environment": summary.get("environment", {}),
                            "source_sha256": summary.get("source_sha256", {}), "git_head": summary.get("git_head"),
                            "runner_timing_definitions": summary.get("timing_definitions", {}), "validation": validation})
    if not rows:
        raise ValueError("No HRW incremental cases found in the supplied roots")
    for source in reference_sources:
        if source in seen:
            raise ValueError(f"Reference duplicates an incremental source: {source}")
        seen.add(source)
        summary = read_json(source)
        if (summary.get("family") != "hrw" or summary.get("n") != 128
                or summary.get("configuration", {}).get("rewrite_strategy") != "greedy"):
            raise ValueError(f"Historical reference must be original-greedy HRW-128: {source}")
        row = make_row(summary, source, 128, "historical_original_greedy_reference")
        row.update(validation_status="not_reaudited_historical_reference", validated_success=False)
        rows.append(row)
        details.append({"source_path": str(source), "summary_sha256": sha256(source),
                        "configuration": summary.get("configuration", {}), "environment": summary.get("environment", {}),
                        "source_sha256": summary.get("source_sha256", {}), "note": "Historical reference only; no speedup ratio."})
    rows.sort(key=lambda row: (row["role"] != "incremental_scalability", row["system_qubits"], row["source_path"]))
    return rows, details, root_reports


def observed_bounds(rows):
    incremental = [row for row in rows if row["role"] == "incremental_scalability"]
    largest = lambda selected: max((row["system_qubits"] for row in selected), default=None)
    smallest = lambda selected: min((row["system_qubits"] for row in selected), default=None)
    timed_out = [row for row in incremental if row["status"] in ("timeout", "validation_timeout")]
    result = {
        "largest_observed_reported_success_n": largest(row for row in incremental if row["status"] == "ok"),
        "largest_observed_validated_success_n": largest(row for row in incremental if row["validated_success"]),
        "largest_attempted_n": largest(row for row in incremental if row["status"] != "not_started"),
        "smallest_observed_timeout_n": smallest(timed_out),
        "smallest_observed_incomplete_compile_timeout_n": smallest(
            row for row in timed_out if row["timeout_class"] == "compile_incomplete"),
        "smallest_observed_postcompile_export_or_validation_timeout_n": smallest(
            row for row in timed_out if row["timeout_class"] == "post_compile_export_or_validation"),
        "smallest_observed_interleaved_export_or_validation_timeout_n": smallest(
            row for row in timed_out if row["timeout_class"] == "interleaved_export_or_validation"),
        "timeout_interpretation": "Only timeouts during model/rewrite/build/lowering count as incomplete compile-stage observations. Export/count/validation timeouts, including validation_timeout, do not establish a compile upper bound.",
        "interpretation": "Observed sizes under recorded budgets; no unmeasured scaling or memory limit is claimed.",
    }
    for basis in BASIS:
        result[f"largest_observed_completed_{basis}_compile_n"] = largest(
            row for row in incremental if row[f"{basis}_compile_status"] == "complete")
    return result


def plot_scalability(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = lambda value: isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    incremental = [row for row in rows if row["role"] == "incremental_scalability"
                   and row["rewrite_strategy"] == "greedy_incremental"]
    successful = sorted((row for row in incremental if row["validated_success"]),
                        key=lambda row: row["system_qubits"])
    timeouts = [row for row in incremental if row["timeout_class"] == "compile_incomplete"
                and positive(row["case_wall_limit_s"])]
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    values = [18000.0]
    xs = [row["system_qubits"] for row in successful]
    series = (("ibm_compile_time_s", "IBM total", "#222222", "o"),
              ("rewrite_time_s", "Rewrite", "#e69f00", "s"),
              ("circuit_build_time_s", "Circuit build", "#d6604d", "D"))
    for field, label, color, marker in series:
        ys = [row[field] if positive(row[field]) else math.nan for row in successful]
        values.extend(value for value in ys if positive(value))
        total = field == "ibm_compile_time_s"
        ax.plot(xs, ys, color=color, marker=marker, label=label,
                linestyle="--" if total else "-", linewidth=1.8 if total else 1.6,
                markersize=8 if total else 5, markerfacecolor="none" if total else color,
                markeredgewidth=1.6 if total else 0.8, zorder=4 if total else 3)
    if timeouts:
        limits = [row["case_wall_limit_s"] for row in timeouts]
        values.extend(limits)
        ax.scatter([row["system_qubits"] for row in timeouts], limits, marker="^", s=70,
                   color="#b2182b", zorder=5, label="Compile timeout: case-wall budget")
    ax.axhline(18000, color="#666666", linestyle="--", linewidth=1, label="5 h case-wall reference")
    ax.set(xlabel="System qubits", ylabel="Time (seconds, logarithmic scale)", yscale="log")
    ax.set_ylim(min(values) * 0.65, max(values) * 1.8)
    ticks = sorted(set(xs + [row["system_qubits"] for row in timeouts]))
    if ticks:
        ax.set_xticks(ticks)
        if len(ticks) == 1:
            ax.set_xlim(ticks[0] - max(1, ticks[0] * 0.05), ticks[0] + max(1, ticks[0] * 0.05))
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("HRW incremental rewrite scalability", fontsize=14, y=0.98)
    fig.text(0.5, 0.91, "Verified successes; timeout markers are budgets, not completed times; no fit", ha="center", fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.23)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def markdown(rows, bounds, outputs):
    def cell(value):
        if value is None:
            return "-"
        return f"{value:.3f}" if isinstance(value, float) else str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# HRW Incremental Rewrite Scalability", "",
        "Full-channel compilation using the explicitly selected `greedy_incremental` strategy.",
        "Every supplied attempt is retained, including duplicate sizes and failures; no attempt overrides another.",
        "These are individual runs without error bars or a fitted complexity model. No simulation was run.", "",
        "IBM/NAM compile time = rewrite + circuit build + lowering to that basis. Model construction, imports,",
        "provenance, export, gate counting, and validation are excluded. Completed basis times remain valid",
        "stage measurements when a later export or validation fails; a timeout budget is not a completed compile time.", "",
        f"Largest observed validated successful size: **{cell(bounds['largest_observed_validated_success_n'])}** system qubits.",
        f"Largest runner-reported successful size: **{cell(bounds['largest_observed_reported_success_n'])}** system qubits.",
        f"Smallest observed timeout size (any stage): **{cell(bounds['smallest_observed_timeout_n'])}** system qubits.",
        f"Smallest observed timeout during incomplete compilation: **{cell(bounds['smallest_observed_incomplete_compile_timeout_n'])}** system qubits.",
        "Export/count/validation timeouts are separate outcomes and do not establish a compile upper bound.",
        "These observations do not establish the largest compilable size or an OOM boundary.", "",
        "| Case | Status | Audit | Width | IBM (s) | Rewrite (s) | Build (s) | NAM (s) | RSS (MiB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    fields = ("case", "status", "validation_status", "total_qubits", "ibm_compile_time_s", "rewrite_time_s",
              "circuit_build_time_s", "nam_compile_time_s", "peak_rss_mib")
    for row in rows:
        if row["role"] == "incremental_scalability":
            lines.append("| " + " | ".join(cell(row[field]) for field in fields) + " |")
    lines.extend(["", "| Case / attempt root | Failure stage | Case limit (s) | Watchdog (s) | Program limit (s) | Case wall (s) | Process wall (s) | IBM gates / NC | NAM gates / NC |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        if row["role"] != "incremental_scalability":
            continue
        values = [f"{row['case']} / {Path(row['source_path']).parents[1].name}", row["failure_stage"],
                  row["case_wall_limit_s"], row["parent_watchdog_s"], row["program_check_limit_s"],
                  row["case_wall_time_s"], row["process_wall_time_s"],
                  f"{cell(row['ibm_gates'])} / {cell(row['ibm_non_clifford'])}",
                  f"{cell(row['nam_gates'])} / {cell(row['nam_non_clifford'])}"]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    progress_rows = [row for row in rows if row["role"] == "incremental_scalability"
                     and row["status"] in ("timeout", "validation_timeout", "running")]
    if progress_rows:
        lines.extend(["", "| Case / attempt root | Timeout class | Last logged accepted steps | Last logged support |",
                      "|---|---|---:|---:|"])
        for row in progress_rows:
            values = [f"{row['case']} / {Path(row['source_path']).parents[1].name}", row["timeout_class"],
                      row["progress_accepted_rewrite_steps"], row["progress_support"]]
            lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        lines.extend(["", "Logged progress is an observed lower bound on accepted rewrites; it does not fill completed rewrite fields."])
    lines.extend(["", "Audit checks: n consecutive-neighbor rotations `[[1,-1],[1,1]] / sqrt(2)`; support 8n -> 4n;",
                  "termination `no_improvement`; 2n final Kraus operators with two coefficients each;",
                  "`K[2i]=(Z_i-iY_i)/(2*sqrt(n))`, `K[2i+1]=(I+X_i)/(2*sqrt(n))`; subnormalization 2;",
                  "width `n + 2*ceil(log2(2n)) + 1`. Pauli strings place system qubit zero on the right.", "",
                  "Source provenance compares the saved source archive, its manifest, and each worker's launch hashes.",
                  "No source hashes were recorded at completion. Artifact hashes describe files at aggregation time.",
                  "QPY/QASM round trips are runner-recorded checks; this report also verifies artifact presence,",
                  "gate-count/export consistency, the generated program's archived template, and its independent check report.",
                  "This is structural artifact validation, not full semantic channel verification. No circuits are reloaded.",
                  "Statevector bytes in the CSV/JSON are the estimate `16 * 2**width`, not an observed allocation or OOM.", ""])
    references = [row for row in rows if row["role"] == "historical_original_greedy_reference"]
    if references:
        lines.extend(["## Historical Original-Greedy Reference", "",
                      "Different-size results are not a same-size speedup comparison. No speedup ratio is computed.", "",
                      "| Case | Status | IBM (s) | NAM (s) | Source |", "|---|---|---:|---:|---|"])
        for row in references:
            lines.append("| " + " | ".join(cell(row[key]) for key in
                         ("case", "status", "ibm_compile_time_s", "nam_compile_time_s", "source_path")) + " |")
    if "png" in outputs:
        lines.extend(["", f"![HRW incremental compilation timing trend]({outputs['png'].name})",
                      "Only verified incremental successes form the curves. Triangles mark compile-stage case-wall",
                      "timeout budgets, not completed times; the horizontal dashed line marks 5 h. No fitted model.", ""])
    lines.extend(["", f"[CSV]({outputs['csv'].name}) | [JSON with detailed checks]({outputs['json'].name})", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True, help="Incremental runner output roots; all attempts are retained")
    parser.add_argument("--reference-summaries", nargs="*", type=Path, default=[], help="Optional original-greedy HRW-128 summary.json references")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--plot", action="store_true", help="Also render a verified incremental timing trend to <out-prefix>.png")
    args = parser.parse_args()
    roots = list(dict.fromkeys(root.resolve() for root in args.roots))
    try:
        rows, details, root_reports = collect(roots, [path.resolve() for path in args.reference_summaries])
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    prefix = args.out_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {suffix: Path(f"{prefix}.{suffix}") for suffix in ("json", "csv", "md")}
    if args.plot:
        outputs["png"] = Path(f"{prefix}.png")
        plot_scalability(rows, outputs["png"])
    bounds = observed_bounds(rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full_channel", "rewrite_strategy": "greedy_incremental",
        "attempt_policy": "Retain all distinct source summaries; no duplicate-size overrides or speedup ratios.",
        "timing_definitions": TIMING_DEFINITIONS, "observed_bounds": bounds,
        "roots": root_reports, "rows": rows, "attempt_details": details,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["json"].write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with outputs["csv"].open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs["md"].write_text(markdown(rows, bounds, outputs), encoding="utf-8")
    invalid_successes = [row["source_path"] for row in rows if row["role"] == "incremental_scalability"
                         and row["status"] == "ok" and not row["validated_success"]]
    print(json.dumps({"rows": len(rows), "observed_bounds": bounds, "invalid_reported_successes": invalid_successes,
                      "outputs": payload["outputs"]}, indent=2))
    return 1 if invalid_successes else 0


if __name__ == "__main__":
    raise SystemExit(main())
