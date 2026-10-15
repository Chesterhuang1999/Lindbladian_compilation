"""Collect the revised seven-benchmark resource layout without launching jobs.

Historical measurements remain in their original CSVs. The active layout CSV
distinguishes measured failures from the user-assigned labels for unrun TFIM-12
VCBE and Stinespring entries.
"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
import math
from pathlib import Path

from build_rq1_merged_table import encoded_hash, map_error

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Results"
CASES = ["TFIM-4", "TFIM-8", "TFIM-12", "HRW-8", "HRW-12", "RepDQE-d3", "Petz_Ising3"]
METHODS = ["Stinespring", "VCBE", "VCBE+opt3", "Raw", "Raw+Qiskit opt3", "Ours"]
SUCCESS = {"ok", "synthesis_ok_postprocessing_timeout"}
QISKIT_FAILURES = {"timeout", "memory_limit", "worker_error"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_csv(name):
    return list(csv.DictReader((RESULTS / name).open()))


def collect():
    historical = load_csv("rq1_unified_resources_20261014.csv")
    raw_table = load_csv("rq1_raw_optimizer_20261014.csv")
    qiskit_batch_path = ROOT / "Baseline_results/rq1_raw_qiskit_opt3_20261014/results.json"
    qiskit_batch = read(qiskit_batch_path)
    if (qiskit_batch.get("status") == "paused_user_request"
            or qiskit_batch.get("configuration_audit", {}).get("status") == "basis_mismatch_not_for_active_table"):
        raise ValueError("The NAM-native Qiskit opt3 batch is paused and retained as a diagnostic; do not use it to refresh the active table")
    qiskit = load_csv("rq1_raw_qiskit_opt3_20261014.csv")
    rows = {(r["case"], r["method"]): dict(r, status_basis="saved_measurement_or_source_inventory")
            for r in historical if r["case"] in CASES and r["method"] in METHODS}
    raw12 = next(r for r in raw_table if r["case"] == "TFIM-12" and r["method"] == "Raw")
    ours12 = next(r for r in raw_table if r["case"] == "TFIM-12" and r["method"] == "Ours")
    fixture = read(raw12["input_fixture_path"])
    if digest(raw12["input_fixture_path"]) != raw12["input_file_sha256"]:
        raise ValueError("The TFIM-12 target fixture changed")
    ours_record = read(ours12["source_result_path"])
    error = map_error(fixture["input_kraus"], ours_record["assembly"]["compiled_kraus"])
    if error >= 1e-10 or raw12["canonical_input_sha256"] != ours12["canonical_input_sha256"]:
        raise ValueError("TFIM-12 Raw and Ours are not paired to the same map")
    target12 = dict(case="TFIM-12", n=raw12["n"], basis="NAM",
                    target_kraus_path=raw12["input_fixture_path"],
                    target_kraus_sha256=encoded_hash(fixture["input_kraus"]),
                    model_parameters=packed(fixture.get("model", {})),
                    canonical_input_sha256=raw12["canonical_input_sha256"],
                    shared_map_relative_error=error)
    for source in (raw12, ours12):
        if source["status"] != "ok" or digest(source["qasm_path"]) != source["qasm_sha256"]:
            raise ValueError("A saved TFIM-12 circuit changed or is incomplete")
        rows["TFIM-12", source["method"]] = dict(source, **{
            **target12, "source_csv": "rq1_raw_optimizer_20261014.csv",
            "source_records": packed([source]), "duplicate_records": 1,
            "status_basis": "saved_measurement"})
    for method in ("Stinespring", "VCBE", "VCBE+opt3"):
        kind = "memory_limit" if method == "Stinespring" else "fit_failed"
        rows["TFIM-12", method] = dict(target12, method=method,
            status="not_run_user_assigned_" + kind,
            status_basis="user_assigned_display_label_for_unrun_entry",
            display_status="Memory limit" if method == "Stinespring" else "Fit failed",
            measurement_attempted=False, G="", W="", Lambda="", Lambda_G="",
            full_choi_status="not_run", source_result_path="", source_csv="",
            source_records="[]", duplicate_records=0,
            absence_reason="User requested this display label and omitted the run")
    for case in CASES:
        raw = rows[case, "Raw"]
        candidate = next(r for r in qiskit if r["case"] == case)
        if candidate["status"] not in {"ok", *QISKIT_FAILURES}:
            raise ValueError(f"Unrecognized Qiskit opt3 status for {case}: {candidate['status']}")
        current_raw = read(raw["source_result_path"])
        plan_input = qiskit_batch["config"]["inputs"][case]
        pairing = candidate if candidate["status"] == "ok" else dict(plan_input,
            input_qasm_sha256=plan_input["qasm_sha256"], input_G=plan_input["G"])
        if (pairing["input_qasm_sha256"] != current_raw["qasm_sha256"]
                or pairing["canonical_input_sha256"] != raw["canonical_input_sha256"]
                or int(pairing["input_G"]) != int(raw["G"])
                or int(pairing["W"]) != int(raw["W"])
                or not math.isclose(float(pairing["Lambda"]), float(raw["Lambda"]), rel_tol=1e-12)
                or digest(plan_input["qasm_path"]) != pairing["input_qasm_sha256"]):
            raise ValueError(f"Qiskit opt3 was applied to a different Raw input: {case}")
        targets = {key: raw[key] for key in ("target_kraus_path", "target_kraus_sha256", "model_parameters")}
        if candidate["status"] == "ok":
            if digest(candidate["qasm_path"]) != candidate["qasm_sha256"]:
                raise ValueError(f"Optimized QASM changed: {case}")
            row = dict(candidate, **targets, source_records=packed([candidate]))
        else:
            actual = qiskit_batch["cases"][case]
            if actual["status"] != candidate["status"]:
                raise ValueError(f"CSV and supervisor disagree for {case}")
            progress_path = qiskit_batch_path.parent / "cases" / case / "progress.json"
            progress = read(progress_path) if progress_path.exists() else {}
            row = dict(candidate, **targets, n=raw["n"], basis="NAM",
                canonical_input_sha256=raw["canonical_input_sha256"],
                input_G=raw["G"], input_qasm_path=plan_input["qasm_path"],
                input_qasm_sha256=plan_input["qasm_sha256"],
                G="", W="", Lambda="", Lambda_G="", qasm_path="", qasm_sha256="",
                full_choi_status="not_run_no_output", measurement_attempted=True,
                source_result_path=str(qiskit_batch_path),
                stopped_stage=progress.get("stage", "unknown"),
                process_wall_s=actual["supervisor"]["elapsed_s"],
                timeout_s=actual["supervisor"]["timeout_s"],
                peak_rss_bytes=actual["supervisor"]["sampled_peak_rss_bytes"],
                source_records=packed([dict(actual, input_plan=plan_input, worker_progress=progress)]))
        rows[case, "Raw+Qiskit opt3"] = dict(row,
            source_csv="rq1_raw_qiskit_opt3_20261014.csv", duplicate_records=1,
            status_basis="new_measurement")
        for method in METHODS:
            row = rows[case, method]
            row["measurement_attempted"] = row.get("measurement_attempted",
                row["status"] in SUCCESS or row["status"] in {"fit_budget_exhausted", *QISKIT_FAILURES})
            if row["status"] in SUCCESS and row.get("G"):
                cost = float(row["Lambda_G"])
                if not math.isclose(cost, float(row["Lambda"]) * int(row["G"]), rel_tol=1e-12):
                    raise ValueError(f"Inconsistent cost: {case}/{method}")
                row["Lambda_G_over_Raw"] = cost / float(raw["Lambda_G"])
                row["Lambda_G_over_Ours"] = cost / float(rows[case, "Ours"]["Lambda_G"])
            else:
                row["Lambda_G_over_Raw"] = row["Lambda_G_over_Ours"] = ""
    return [rows[case, method] for case in CASES for method in METHODS]


def main():
    rows = collect()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = RESULTS / "rq1_resource_layout_preview.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected {len(rows)} slots with {sum(r['status'] in SUCCESS for r in rows)} measured circuits: {path}")


if __name__ == "__main__":
    main()
