"""Join the nine-benchmark IBM resource table from audited saved measurements.

Run with Python; no compilation, optimizer, fit, or experiment is launched.
Keep failure labels distinct from user-assigned labels and missing IBM data.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path

from build_rq1_merged_table import map_error

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Baseline_results"
SUITE = ROOT / "paper_experiments/pending_review"
CASES = ["TFIM-4", "TFIM-8", "TFIM-12", "TFIM-20", "HRW-8", "HRW-12", "HRW-28",
         "RepDQE-d3", "Petz_Ising3"]
METHODS = ["Stinespring", "VCBE", "VCBE+Qiskit opt3", "Raw", "Raw+Qiskit opt3", "Ours"]
IBM = {"u1", "u2", "u3", "cx"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            hasher.update(block)
    return hasher.hexdigest()


def identity(encoded):
    normalized = [[[str(p), float(c), float(re), float(im)] for p, c, re, im in row] for row in encoded]
    return sha256(json.dumps(normalized, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fixture_path(case):
    if case == "HRW-8":
        return BASE / "table1_scaling_ours_20261013/cases/HRW-8/input.json"
    if case == "RepDQE-d3":
        return SUITE / "remaining_full_20261011/cases" / case / "input.json"
    if case == "Petz_Ising3":
        return SUITE / "raw_baseline_20261012/inputs/Petz_Ising3.json"
    return SUITE / "tfim_hrw_full_20261011/cases" / case / "input.json"


def collect():
    targets = {}
    for case in CASES:
        path = fixture_path(case)
        document = read(path)
        targets[case] = dict(target_path=str(path), target_file_sha256=digest(path),
            target_kraus_sha256=identity(document["input_kraus"]),
            target=document["input_kraus"])
    rows = {}

    def measured(source, batch, **overrides):
        record = dict(source, **overrides)
        case, method = record["case"], record["method"]
        assert (case, method) not in rows and case in CASES and method in METHODS
        assert record["status"] == "ok" and record["basis"].lower() == "ibm"
        target = targets[case]
        if record.get("target_kraus_sha256"):
            assert record["target_kraus_sha256"] == target["target_kraus_sha256"]
        if record.get("target_file_sha256"):
            assert record["target_file_sha256"] == target["target_file_sha256"]
        if record.get("qasm_path"):
            assert digest(record["qasm_path"]) == record["qasm_sha256"]
        original = read(record["original_result_path"])
        if record.get("original_result_sha256"):
            assert digest(record["original_result_path"]) == record["original_result_sha256"]
        if method.startswith("VCBE"):
            assert identity(original["kraus"]) == target["target_kraus_sha256"]
            assert record.get("inner_preoptimization_level", 0) == 0
            if "rewrite" in original:
                assert original["rewrite"]["enabled"] is False
        if method.startswith("Raw"):
            if "canonicalization" in original:
                assert original["canonicalization"]["raw_input_sha256"] == target["target_kraus_sha256"]
                assert map_error(target["target"], original["canonical_kraus"]) < 1e-10
        if method == "Ours" and "assembly" in original:
            assert map_error(target["target"], original["assembly"]["compiled_kraus"]) < 1e-10
        if "+Qiskit opt3" in method:
            assert record["lowering_optimization_level"] == 3
            assert "whole assembled IBM opt0" in record["optimizer_scope"]
        else:
            assert record["lowering_optimization_level"] == 0
        operations = record.get("counts", record.get("counts_IBM"))
        assert set(operations) <= IBM | {"reset", "barrier"}
        G, W = int(record["G"]), int(record["W"])
        assert G == sum(operations.get(g, 0) for g in IBM) and G > 0 and W > 0
        lam = record.get("Lambda_display", str(record["Lambda"]))
        if case != "Petz_Ising3" and method != "Stinespring":
            lam = str(round(float(lam), 12))
        cost = Decimal(lam) * G
        assert math.isclose(float(cost), record["Lambda"] * G, rel_tol=1e-12)
        assert math.isclose(record["Lambda_G"], record["Lambda"] * G, rel_tol=1e-12)
        rows[case, method] = dict(case=case, method=method, status="ok", basis="IBM",
            G=G, Lambda=lam, Lambda_numeric=record["Lambda"], Lambda_G=format(cost, "f"), W=W,
            full_choi_status=record["full_choi_status"], CX=operations.get("cx", 0),
            non_clifford=record.get("non_clifford", record.get("NC_IBM", "")),
            optimizer_s=record.get("optimizer_s", 0), optimizer_scope=record.get("optimizer_scope", "none"),
            measurement_attempted=True, status_basis="saved_IBM_measurement",
            absence_reason="", source_result_path=str(batch), source_result_sha256=digest(batch),
            original_result_path=record["original_result_path"],
            original_result_sha256=digest(record["original_result_path"]),
            qasm_path=record.get("qasm_path", ""), qasm_sha256=record.get("qasm_sha256", ""),
            **{k:v for k,v in target.items() if k != "target"},
            source_record=json.dumps(record, separators=(",", ":"), allow_nan=False))

    for name in ("rq1_layout_ibm_completion_20261014", "rq1_hrw12_hrw28_ibm_opt3_20261014",
                 "rq1_tfim_dqe_petz_ibm_opt3_20261014"):
        batch = BASE / name / "results.json"
        document = read(batch)
        assert document["status"] == "complete"
        for row in document["rows"]:
            measured(row, batch)
    batch = BASE / "rq1_hrw8_ibm_backend_20261014/results.json"
    document = read(batch)
    assert document["status"] == "complete"
    manifest = read(document["manifest_path"])
    for row in document["methods"]:
        if row["method"] not in {"Stinespring", "Raw", "Raw+Qiskit opt3", "Ours"}:
            continue
        for path, expected in manifest["files"].items():
            if path.endswith("/input.json"):
                assert digest(path) == expected
        if row["method"] == "Stinespring":
            original = read(row["original_result_path"])
            assert original["input_kraus_sha256"] == targets["HRW-8"]["target_kraus_sha256"]
        measured(row, batch, G=row["G_IBM"], Lambda_G=row["C_IBM"],
            optimizer_scope="whole assembled IBM opt0 circuit" if "+Qiskit opt3" in row["method"] else "none")
    for case, mode in (("TFIM-4", "flagged_minimum_beta"), ("RepDQE-d3", "native")):
        batch = BASE / "table_a_stinespring_20261013/cases" / case / "results.json"
        original = read(batch)
        assert original["input_kraus_sha256"] == targets[case]["target_kraus_sha256"]
        assert original["input_file_sha256"] == targets[case]["target_file_sha256"]
        variant = next(v for v in original["variants"] if v["basis"] == "ibm" and v["mode"] == mode)
        assert variant["success_choi_relative_error"] < 1e-10
        measured(variant, batch, case=case, method="Stinespring", counts=variant["counts"]["ops"],
            original_result_path=str(batch), qasm_path=variant["qasm"],
            full_choi_status="passed_archived_ibm_opt0", lowering_optimization_level=0)

    rerun_failures = {}
    rerun_batch = BASE / "rq1_tfim8_petz_stinespring_ibm_20261014/results.json"
    if rerun_batch.exists():
        document = read(rerun_batch)
        assert document["status"] == "complete"
        manifest = read(document["manifest_path"])
        assert digest(manifest["runner_path"]) == manifest["runner_sha256"]
        assert manifest["basis_gates"] == ["u1", "u2", "u3", "cx"]
        for record in document["rows"]:
            case = record["case"]
            assert case in {"TFIM-8", "Petz_Ising3"} and record["method"] == "Stinespring"
            for path, expected in manifest["files"][case].items():
                assert digest(path) == expected
            assert record["input_kraus_sha256"] == targets[case]["target_kraus_sha256"]
            assert record["input_file_sha256"] == targets[case]["target_file_sha256"]
            if record["status"] != "ok":
                rerun_failures[case] = record
                continue
            assert record["seed_transpiler"] == 20261013
            assert record["approximation_degree"] == 1.0 and record["qubits_initially_zero"] is False
            assert record["full_choi_status"] == "skipped_user_request"
            assert digest(record["isometry_path"]) == record["isometry_sha256"]
            assert record["isometry_orthogonality_error"] < 1e-10
            assert record["isometry_success_kraus_max_error"] < 1e-10
            if case == "Petz_Ising3":
                assert record["compiled_isometry_validation_status"] == "passed_all_input_columns"
                assert record["compiled_isometry_column_max_error"] < 1e-10
            measured(record, rerun_batch)

    def missing(case, method, state, reason, original=None, attempted=False):
        assert (case, method) not in rows
        rows[case, method] = dict(case=case, method=method, status=state, basis="IBM", G="", W="",
            Lambda="", Lambda_numeric="", Lambda_G="", full_choi_status="not_run_no_IBM_output",
            CX="", non_clifford="", optimizer_s="", optimizer_scope="none",
            measurement_attempted=attempted,
            status_basis="user_assigned_unexecuted_label" if state.startswith("not_run_user_assigned") else "source_inventory",
            absence_reason=reason, source_result_path=str(original) if original else "",
            source_result_sha256=digest(original) if original else "",
            original_result_path=str(original) if original else "", original_result_sha256=digest(original) if original else "",
            qasm_path="", qasm_sha256="", source_record="",
            **{k:v for k,v in targets[case].items() if k != "target"})
    for case in CASES:
        if (case, "Stinespring") in rows:
            continue
        if case in rerun_failures:
            record = rerun_failures[case]
            missing(case, "Stinespring", record["status"],
                f"IBM rerun stopped at {record.get('stopped_stage', 'unknown stage')}",
                record["original_result_path"], True)
        elif case == "TFIM-8":
            source = BASE / "table1_scaling_stinespring_20261013/cases/TFIM-8/results.json"
            record = read(source)
            assert record["input_kraus_sha256"] == targets[case]["target_kraus_sha256"]
            assert record["variants"] and all(v["basis"] == "nam" for v in record["variants"])
            missing(case, "Stinespring", "no_ibm_measurement", "Archived synthesis has NAM counts only; no IBM synthesis was run", source)
        elif case == "TFIM-12":
            missing(case, "Stinespring", "not_run_user_assigned_memory_limit", "User specified Memory limit and requested omitting the experiment")
        elif case == "HRW-12":
            source = BASE / "table1_scaling_stinespring_20261013/cases/HRW-12/results.json"
            record = read(source)
            assert record["status"] == "preflight_memory_limit"
            assert record["input_kraus_sha256"] == targets[case]["target_kraus_sha256"]
            assert record["isometry_bytes"] > record["protocol"]["rss_limit_bytes"]
            missing(case, "Stinespring", "preflight_memory_limit", record["resource_reason"], source, True)
        else:
            missing(case, "Stinespring", "not_run", "No saved IBM Stinespring measurement; no experiment requested")
    for case in ("TFIM-8", "TFIM-12", "TFIM-20"):
        for method in ("VCBE", "VCBE+Qiskit opt3"):
            if case == "TFIM-8":
                source = BASE / "vcbe_scaling_20261014/cases/TFIM-8/results.json"
                record = read(source)
                assert record["status"] == "fit_budget_exhausted"
                assert identity(record["kraus"]) == targets[case]["target_kraus_sha256"]
                missing(case, method, "fit_budget_exhausted", "Recorded VCBE fit exhausted its search budget; no qualified circuit", source, method == "VCBE")
            elif case == "TFIM-12":
                missing(case, method, "not_run_user_assigned_fit_failed", "User specified Fit failed and requested omitting the experiment")
            else:
                missing(case, method, "not_run", "No VCBE fitting experiment was requested for TFIM-20")
    assert len(rows) == len(CASES) * len(METHODS) == 54
    for case in CASES:
        for method in ("Raw", "VCBE"):
            baseline = rows[case, method]
            optimized = rows[case, method + "+Qiskit opt3"]
            if optimized["status"] != "ok":
                continue
            record = json.loads(optimized["source_record"])
            assert record["input_qasm_sha256"] == baseline["qasm_sha256"]
            assert optimized["W"] == baseline["W"] and optimized["Lambda"] == baseline["Lambda"]
            if "input_G" in record:
                assert record["input_G"] == baseline["G"]
    return [rows[case, method] for case in CASES for method in METHODS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "Results/rq1_resource_layout_ibm_20261014_v2.csv")
    args = parser.parse_args()
    rows = collect()
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Collected {len(rows)} slots, {sum(r['status'] == 'ok' for r in rows)} measured IBM circuits: {args.out}")


if __name__ == "__main__":
    main()
