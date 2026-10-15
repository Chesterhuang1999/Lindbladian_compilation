"""Audit fresh raw-Kraus VCBE circuits and export the new comparison data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from qiskit import qasm2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
from baseline_common import count_qasm_text
from experiment_protocol import source_identity
from rq1_pipeline import deserialize_kraus, input_identity
from run_rq1_vcbe_scaling import digest


def collect(batch_path):
    batch = json.loads((batch_path / "results.json").read_text())
    assert batch["sources"] == source_identity()
    assert batch["rewrite_enabled"] is False and batch["historical_fit_reused"] is False
    for name, recorded in batch["implementation_sources"].items():
        assert digest(ROOT / name) == recorded
    with (ROOT / "Results/rq1_unified_resources_20261014.csv").open() as stream:
        ours = {r["case"]: r for r in csv.DictReader(stream) if r["method"] == "Ours"}
    rows = []
    for case, saved in batch["cases"].items():
        result_path = batch_path / "cases" / case / "results.json"
        record = json.loads(result_path.read_text())
        assert saved["worker_exit_code"] == 0 and record["status"] == "success"
        assert record["production_sources_unchanged"] and record["raw_kraus_unchanged"]
        provenance = record["provenance"]
        fixture = Path(provenance["input_path"])
        assert digest(fixture) == provenance["input_file_sha256"]
        original = json.loads(fixture.read_text())["input_kraus"]
        assert original == record["kraus"]
        kraus = deserialize_kraus(original)
        assert input_identity(kraus) == record["raw_kraus_sha256"]
        assert record["raw_kraus_sha256"] == ours[case]["target_kraus_sha256"]
        assert len(kraus) == record["M"]
        assert all(b["block_qualified"] and not b["historical_fit_reused"]
                   for b in record["vcbe_blocks"])
        maximum_error = max(b["full_embedded_frobenius_block_error"] for b in record["vcbe_blocks"])
        assert maximum_error < batch["tolerance"]
        for method, measured in record["methods"].items():
            metric = measured["metrics"]["nam_opt0"]
            qasm = Path(metric["qasm_path"])
            assert digest(qasm) == metric["qasm_sha256"]
            circuit = qasm2.loads(qasm.read_text())
            assert set(circuit.count_ops()) <= {"rz", "h", "x", "cx"}
            assert dict(circuit.count_ops()) == metric["counts"]
            count = count_qasm_text(qasm.read_text(), gate_set="nam")
            assert count["metric_total"] == metric["G"]
            assert count["clifford"]["non_clifford"] == metric["NC"]
            assert circuit.num_qubits == metric["W"]
            assert metric["Lambda"] == record["Lambda"]
            assert metric["Lambda_G"] == metric["Lambda"] * metric["G"]
            assert measured["metadata"]["outer_control_scope"] == "whole-block"
            assert measured["metadata"]["alpha_per_kraus"] == record["alpha"]
            assert metric["status"] == "ok"
            assert all(v["qualified"] for v in measured["block_validation"])
            compiled_block_error = max(v["full_embedded_frobenius_block_error"]
                                       for v in measured["block_validation"])
            if case not in ("HRW-8", "HRW-12"):
                assert measured["verification"]["passed"]
                assert metric["full_choi_status"] == "passed"
            rows.append(dict(case=case, method=method, status=metric["status"], n=record["n"],
                M=record["M"], basis="NAM", final_optimization_level=0,
                inner_preoptimization_level=3 if method == "vcbe_preopt3_tree" else 0,
                rewrite_enabled=False, shared_rewrite="none: original input Kraus operators",
                historical_fit_reused=False,
                normalization="Pauli 1-norm computed from each original Kraus",
                G=metric["G"], Lambda=metric["Lambda"], Lambda_G=metric["Lambda_G"], W=metric["W"],
                CX=metric["CX"], non_clifford=metric["NC"],
                alpha_per_kraus=json.dumps(record["alpha"]),
                input_kraus_sha256=record["raw_kraus_sha256"],
                input_fixture_path=str(fixture), input_file_sha256=provenance["input_file_sha256"],
                input_pairing="exact raw Kraus equality, original order and phases preserved",
                unique_raw_targets=record["unique_raw_targets"], fresh_unique_fit_s=record["fresh_unique_fit_s"],
                assembly_s=metric["assembly_s"], lowering_s=metric["lowering_s"],
                reported_compile_s=record["fresh_unique_fit_s"] + metric["assembly_plus_lowering_s"],
                compile_time_scope="sum of fresh fit wall time (including fit checkpoints and block checks) and assembly_plus_lowering_s (including preoptimization and block checks); excludes imports, full-channel verification and final circuit I/O",
                per_unique_raw_target_budget_s=batch["fitting_recipe"]["per_unique_raw_target_budget_s"],
                max_full_embedded_block_error=maximum_error,
                max_compiled_inner_block_error=compiled_block_error,
                full_choi_status=metric["full_choi_status"],
                full_choi_relative_error=metric["full_choi_relative_error"],
                validation_scope=metric["validation_scope"],
                control_policy="control entire fitted unitary; Fstar disabled",
                shared_outer_policy="outer PREPARE construction and Kraus unary-iteration tree",
                vcbe_implementation=batch["vcbe"],
                ours_G=int(ours[case]["G"]), ours_Lambda=float(ours[case]["Lambda"]),
                ours_Lambda_G=float(ours[case]["Lambda_G"]), ours_W=int(ours[case]["W"]),
                G_over_Ours=metric["G"] / int(ours[case]["G"]),
                Lambda_G_over_Ours=metric["Lambda_G"] / float(ours[case]["Lambda_G"]),
                ours_reference="existing complete Ours pipeline, including its own rewrite",
                ours_source_result_path=ours[case]["source_result_path"],
                worker_peak_rss_bytes=record["peak_rss_bytes"],
                qasm_path=str(qasm), qasm_sha256=metric["qasm_sha256"],
                metric_evidence="saved QASM hash, NAM basis, exact counts, width and normalization rechecked",
                source_result_path=str(result_path)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=ROOT / "Baseline_results/vcbe_no_rewrite_20261014")
    parser.add_argument("--output", type=Path, default=ROOT / "Results/rq1_vcbe_no_rewrite_20261014.csv")
    args = parser.parse_args()
    rows = collect(args.batch.resolve())
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(output=str(args.output), rows=len(rows), cases=len({r["case"] for r in rows}),
        max_block_error=max(r["max_compiled_inner_block_error"] for r in rows),
        max_verified_full_choi_error=max(r["full_choi_relative_error"] for r in rows
                                        if r["full_choi_status"] == "passed"))))


if __name__ == "__main__":
    main()
