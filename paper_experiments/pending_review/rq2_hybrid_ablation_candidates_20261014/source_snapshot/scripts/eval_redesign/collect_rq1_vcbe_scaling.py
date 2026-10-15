"""Audit the skipped-Choi VCBE scaling batch and export qualified resources."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from collect_rq1_remaining import (
    ROOT, coefficients, digest, identity, packed, read, recount, sparse_gram_error, write_csv,
)

BATCH = ROOT / "Baseline_results/vcbe_scaling_20261014"
OUTPUT = ROOT / "Results/rq1_vcbe_scaling_20261014.csv"
METHODS = ("vcbe_source_tree", "vcbe_preopt3_tree")


def collect():
    batch = read(BATCH / "results.json")
    assert batch["full_choi_status"] == "skipped_user_request"
    assert batch["production_sources_unchanged"]
    assert set(batch["cases"]) == {"TFIM-8", "HRW-8", "HRW-12"}
    assert batch["basis"] == "NAM" and batch["final_optimization_level"] == 0
    assert batch["qubits_initially_zero"] is False
    rows = []
    for case, record in batch["cases"].items():
        path = BATCH / "cases" / case / "results.json"
        assert read(path) == record
        encoded = record["kraus"]
        assert identity(encoded) == record["input_sha256"]
        provenance = record["provenance"]
        target_path = Path(provenance["target_path"])
        shared_path = Path(provenance["shared_kraus_path"])
        ours_path = Path(provenance["ours_result_path"])
        assert digest(target_path) == provenance["target_file_sha256"]
        assert digest(shared_path) == provenance["shared_kraus_file_sha256"]
        assert digest(ours_path) == provenance["ours_result_sha256"]
        target = read(target_path)["input_kraus"]
        map_error = sparse_gram_error(coefficients(target), coefficients(encoded))
        assert map_error < batch["tolerance"]
        if shared_path.suffix == ".npz":
            with np.load(shared_path) as arrays:
                shared = [{str(p): complex(c) for p, c in zip(arrays["labels"], row) if c != 0}
                          for row in arrays["A_final"] if np.any(row != 0)]
        else:
            shared = coefficients(read(shared_path)["assembly"]["compiled_kraus"])
        actual = coefficients(encoded)
        assert len(actual) == len(shared)
        assert all(all(abs(a.get(p, 0) - b.get(p, 0)) < 1e-12 for p in a.keys() | b.keys())
                   for a, b in zip(actual, shared))
        alpha = [sum(abs(c) for c in row.values()) for row in actual]
        assert np.allclose(alpha, record["alpha"], rtol=1e-12, atol=1e-14)
        lam = sum(a ** 2 for a in alpha)
        assert math.isclose(lam, record["Lambda"], rel_tol=1e-12)
        assert math.isclose(lam, read(ours_path)["Lambda"], rel_tol=1e-12)
        unique = {b["template"]: b for b in record.get("vcbe_blocks", [])}
        full_fit = next((b for b in unique.values() if not b["cached"]), {})
        for method in METHODS:
            measured = record["methods"].get(method, {})
            metric = measured.get("metrics", {}).get("nam_opt0", {})
            qualified = metric.get("block_qualified", False)
            G, W, counts = "", "", {}
            if metric:
                qasm = Path(metric["qasm_path"])
                assert digest(qasm) == metric["qasm_sha256"]
                actual_G, actual_W, counts = recount(qasm, "nam")
                assert (actual_G, actual_W, dict(counts)) == (metric["G"], metric["W"], metric["counts"])
                assert math.isclose(metric["Lambda_G"], lam * actual_G, rel_tol=1e-12)
                assert metric["full_choi_status"] == "skipped_user_request"
                assert qualified == all(v["qualified"] for v in measured["block_validation"])
                if qualified:
                    assert record["status"] == "success_skipped_full_choi"
                    assert all(v["full_embedded_frobenius_block_error"] < batch["tolerance"]
                               and v["preoptimization_unitary_frobenius_error"] < batch["tolerance"]
                               for v in measured["block_validation"])
                    G, W = actual_G, actual_W
            row = dict(case=case, method=method, status="ok" if qualified else record["status"],
                n=record["n"], M=record["M"], basis="NAM", final_optimization_level=0,
                inner_preoptimization_level=3 if method == METHODS[1] else 0,
                primary_table_candidate=True, host="local", Lambda=lam,
                G=G, W=W, Lambda_G=lam * G if G != "" else "",
                CX=metric.get("CX", "") if qualified else "",
                non_clifford=metric.get("NC", "") if qualified else "",
                diagnostic_G=metric.get("G", "") if not qualified else "",
                diagnostic_W=metric.get("W", "") if not qualified else "",
                diagnostic_Lambda_G=metric.get("Lambda_G", "") if not qualified else "",
                input_kraus_sha256=record["input_sha256"], alpha_per_kraus=packed(alpha),
                shared_rewrite="current table Ours compiled Kraus representation",
                input_pairing="same Kraus operators and alpha_j as the current Ours result",
                source_pairing_gram_relative_error=map_error,
                ours_result_path=str(ours_path), ours_result_sha256=provenance["ours_result_sha256"],
                shared_representation_check="passed coefficient-by-coefficient against current saved Ours",
                full_choi_status="skipped_user_request", full_choi_relative_error="",
                block_validation_scope=metric.get("validation_scope", "not_completed"),
                max_full_embedded_block_error=max((v["full_embedded_frobenius_block_error"]
                    for v in measured.get("block_validation", [])), default=""),
                fit_status=full_fit.get("status", "cached_local_templates"),
                fit_error=full_fit.get("fit_error", ""), fit_budget_s=batch["fit_budget_s"],
                per_trial_budget_s=batch["per_trial_budget_s"],
                newly_measured_unique_fit_s=record.get("newly_measured_unique_fit_s", ""),
                cached_unique_fit_s=record.get("cached_unique_fit_s", ""),
                assembly_s=metric.get("assembly_s", ""), lowering_s=metric.get("lowering_s", ""),
                reported_compile_s=metric.get("assembly_plus_lowering_s", ""),
                time_scope="block preparation/fit excluded; assembly includes inner opt3 and active-unitary checks; final NAM lowering",
                block_preparation_s=record.get("block_preparation_s", ""),
                total_elapsed_s=record.get("total_elapsed_s", ""),
                peak_rss_bytes=record.get("peak_rss_bytes", record["supervisor"]["sampled_peak_rss_bytes"]),
                supervisor=packed(record["supervisor"]),
                control_policy="control entire synthesized unitary", outer_policy="shared tree and outer PREPARE",
                vcbe_implementation=batch["vcbe"], vcbe_paper_url="https://arxiv.org/abs/2507.17658v2",
                tolerance=batch["tolerance"], qasm_path=metric.get("qasm_path", ""),
                qasm_sha256=metric.get("qasm_sha256", ""), counts=packed(counts),
                metric_evidence="saved QASM hash, NAM basis, width and gate recount verified" if metric else "no compiled circuit",
                source_result_path=str(path))
            rows.append(row)
    return rows


if __name__ == "__main__":
    rows = collect()
    write_csv(OUTPUT, rows)
    print(packed(dict(csv=str(OUTPUT), rows=len(rows), qualified=sum(r["status"] == "ok" for r in rows),
                      statuses={r["case"]: r["status"] for r in rows})))
