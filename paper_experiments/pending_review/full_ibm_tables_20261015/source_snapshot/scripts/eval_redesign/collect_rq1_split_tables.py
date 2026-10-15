"""Merge retained IBM table rows with the seed-0 RandCh batch.

No experiments are started. Pending output can be previewed explicitly, but
the default collection requires a finished batch. Historical Petz records
are preserved as provenance and excluded from the active benchmark list.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

from build_rq1_split_tables import CASES, METHODS, SUCCESS

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "Baseline_results/rq1_randch_ibm_20261015"
IBM = {"u1", "u2", "u3", "cx"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "Results/rq1_resource_tables_ibm_20261015.csv")
    args = parser.parse_args()
    inherited = ROOT / "Results/rq1_resource_layout_ibm_20261014_v2.csv"
    rows = {(r["case"], r["method"]): r for r in csv.DictReader(inherited.open()) if r["case"] in CASES}
    manifest = read(BATCH / "manifest.json")
    assert digest(manifest["runner_path"]) == manifest["runner_sha256"]
    for case in manifest["cases"]:
        for path, expected in manifest["files"][case].items():
            assert digest(path) == expected
    for path, expected in manifest["helpers"].items():
        assert digest(path) == expected
    if not args.allow_pending:
        assert read(BATCH / "status.json")["status"] == "complete"
        assert read(BATCH / "results.json")["status"] == "complete"
    for case in manifest["cases"]:
        for method in METHODS:
            path = BATCH / "cases" / case / (method.replace("+", "_").replace(" ", "_") + "_results.json")
            record = read(path) if path.exists() else dict(case=case, method=method, status="running", basis="IBM")
            assert record["case"] == case and record["method"] == method and record["basis"] == "IBM"
            state = record["status"]
            if state in SUCCESS:
                G, W = int(record["G"]), int(record["W"])
                operations = record["counts"]
                assert set(operations) <= IBM and G == sum(operations.values()) and G > 0 and W > 0
                lam = str(record["Lambda"])
                cost = format(Decimal(lam) * G, "f")
                assert abs(float(cost) - record["Lambda_G"]) < max(abs(float(cost)), 1) * 1e-12
                if record.get("qasm_path"):
                    assert digest(record["qasm_path"]) == record["qasm_sha256"]
                if record.get("original_result_sha256"):
                    assert digest(record["original_result_path"]) == record["original_result_sha256"]
                assert record["approximation_degree"] == 1.0 and record["qubits_initially_zero"] is False
                if method == "Stinespring":
                    assert digest(record["isometry_path"]) == record["isometry_sha256"]
                    assert record["isometry_orthogonality_error"] < 1e-10
                    assert record["isometry_success_kraus_max_error"] < 1e-10
                if method.startswith("VCBE"):
                    assert all(b["full_embedded_frobenius_block_error"] < 1e-10 for b in record["vcbe_blocks"])
                if "+Qiskit opt3" in method:
                    baseline = rows[case, method.split("+")[0]]
                    assert record["input_qasm_sha256"] == baseline["qasm_sha256"]
                    assert int(record["input_G"]) == int(baseline["G"]) and W == int(baseline["W"])
            else:
                G, W, lam, cost = "", "", "", ""
            if not args.allow_pending:
                assert state != "running"
            rows[case, method] = dict(case=case, method=method, status=state, basis="IBM", G=G, Lambda=lam,
                Lambda_numeric=record.get("Lambda", ""), Lambda_G=cost, W=W,
                full_choi_status=record.get("full_choi_status", "not_run"), CX=record.get("CX", ""),
                non_clifford=record.get("non_clifford", ""), optimizer_s=record.get("optimizer_s", ""),
                optimizer_scope=record.get("optimizer_scope", "none"),
                measurement_attempted=record.get("measurement_attempted", state != "not_run"),
                status_basis="saved_IBM_measurement" if state in SUCCESS else "explicit_user_omission" if state == "not_run" else "saved_fit_or_resource_failure",
                absence_reason=record.get("absence_reason", record.get("stopped_stage", "")),
                source_result_path=str(path) if path.exists() else "", source_result_sha256=digest(path) if path.exists() else "",
                original_result_path=record.get("original_result_path", ""), original_result_sha256=record.get("original_result_sha256", ""),
                qasm_path=record.get("qasm_path", ""), qasm_sha256=record.get("qasm_sha256", ""),
                target_path=record.get("target_path", ""), target_file_sha256=record.get("target_file_sha256", ""),
                target_kraus_sha256=record.get("target_kraus_sha256", ""),
                source_record=json.dumps(record, separators=(",", ":"), allow_nan=False))
    assert len(rows) == len(CASES) * len(METHODS) == 60
    columns = list(rows[CASES[0], METHODS[0]])
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([rows[c, m] for c in CASES for m in METHODS])
    print(f"Collected {len(rows)} slots; Petz-Ising3 excluded; {sum(r['status'] in SUCCESS for r in rows.values())} IBM measurements")


if __name__ == "__main__":
    main()
