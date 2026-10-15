#!/usr/bin/env python3
"""Collect the six selected Table 1 cases with input and resource evidence."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PENDING = ROOT / "paper_experiments/pending_review"
SMALL = ROOT / "Baseline_results/table_a_stinespring_20261013/cases"
SCALING = ROOT / "Baseline_results/table1_scaling_stinespring_20261013/cases"
FULL = PENDING / "tfim_hrw_full_20261011/cases"
SUITE = PENDING / "rq1_paper_suite_20261010/full_v2"
CASES = ["TFIM-4", "TFIM-8", "HRW-4", "HRW-8", "HRW-12", "RepDQE-d3"]
sys.path.insert(0, str(ROOT / "Baseline_scripts"))
from baseline_common import QREG2_RE, parse_qasm_operation, is_nam_operation_clifford


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoded_hash(encoded):
    return sha256(json.dumps(encoded, separators=(",", ":")).encode()).hexdigest()


def gram_error(left, right):
    labels = sorted({p for family in (left, right) for row in family for p, *_ in row})
    index = {p: i for i, p in enumerate(labels)}

    def gram(family):
        matrix = np.zeros((len(family), len(labels)), complex)
        for i, row in enumerate(family):
            for p, magnitude, real, imag in row:
                matrix[i, index[p]] += magnitude * complex(real, imag)
        return matrix.conj().T @ matrix

    reference = gram(left)
    return float(np.linalg.norm(gram(right) - reference) / np.linalg.norm(reference))


def recount(path):
    counts, width, nc = Counter(), 0, 0
    classifications = {}
    with Path(path).open() as stream:
        for line in stream:
            match = QREG2_RE.search(line)
            if match:
                width += int(match.group(1))
            operation = parse_qasm_operation(line.strip())
            if operation is None:
                continue
            assert operation.name in {"rz", "h", "x", "cx", "reset"}, operation.name
            counts[operation.name] += 1
            if operation.name == "rz":
                key = operation.params
                if key not in classifications:
                    classifications[key] = is_nam_operation_clifford(operation)
                assert classifications[key] is not None
                nc += not classifications[key]
    return {"G": sum(counts[k] for k in ("rz", "h", "x", "cx")), "W": width,
            "counts": dict(counts), "non_clifford": int(nc)}


def collect():
    state = read(SCALING.parent / "status.json")
    assert state["status"] == "finished", "Finish the bounded Stinespring run before collecting"
    rows = []
    for case in CASES:
        if case == "RepDQE-d3":
            ours_path = PENDING / "remaining_full_20261011/cases" / case / "results.json"
        elif case == "HRW-8":
            ours_path = ROOT / "Baseline_results/table1_scaling_ours_20261013/cases/HRW-8/results.json"
        else:
            ours_path = FULL / case / "results.json"
        ours = read(ours_path)
        fixture = ours_path.parent / "input.json"
        original = read(fixture)
        encoded = original["input_kraus"]
        n = len(encoded[0][0][0])
        dim = 1 << n
        raw_path = SUITE / "runs" / case / "raw/results.json"
        raw = read(raw_path)
        assert raw["input_kraus"] == encoded
        input_hash = encoded_hash(encoded)
        assert input_hash == raw["canonicalization"]["raw_input_sha256"] == ours["canonicalization"]["raw_input_sha256"]
        canonical_hash = raw["canonicalization"]["canonical_input_sha256"]
        assert canonical_hash == encoded_hash(raw["canonical_kraus"]) == ours["canonicalization"]["canonical_input_sha256"]
        assert ours["input_sha256"] == file_hash(fixture)
        rewrite_error = gram_error(encoded, ours["assembly"]["compiled_kraus"])
        assert rewrite_error < 1e-10 and gram_error(encoded, raw["canonical_kraus"]) < 1e-12
        stine_path = (SMALL if case in {"TFIM-4", "HRW-4", "RepDQE-d3"} else SCALING) / case / "results.json"
        stine = read(stine_path)
        assert stine["input_kraus_sha256"] == input_hash
        assert stine["input_file_sha256"] == file_hash(stine["input_source"])
        base = {
            "case": case, "n": n, "status": "ok", "table1_candidate": True,
            "input_fixture_path": str(fixture), "input_file_sha256": file_hash(fixture),
            "input_kraus_sha256": input_hash, "canonical_input_sha256": canonical_hash,
            "input_pairing": "passed", "input_kraus_count": len(encoded),
            "canonical_kraus_count": len(raw["canonical_kraus"]), "kraus_rank": stine["kraus_rank"],
            "native_cptp": stine["native_cptp"], "qiskit_version": "2.1.1", "host": "local",
            "model_parameters": json.dumps(original["model"], ensure_ascii=False, separators=(",", ":")),
        }
        classical = stine["classical_dense"]
        rows.append({**base, "method": "Direct matrix", "variant": "finite_CP_map_complex128",
            "configuration_status": "available", "G": "N/A", "Lambda": "N/A", "Lambda_G": "N/A", "W": "N/A",
            "rho_shape": str([dim, dim]), "rho_bytes": 16 * dim * dim,
            "raw_kraus_bytes": len(encoded) * 16 * dim * dim,
            "canonical_kraus_bytes": len(raw["canonical_kraus"]) * 16 * dim * dim,
            "rank_compressed_kraus_bytes": stine["kraus_rank"] * 16 * dim * dim,
            "superoperator_shape": str([dim * dim, dim * dim]), "superoperator_bytes": 16 * dim ** 4,
            "storage_evidence": classical.get("storage_status", "existing_small_dense_artifacts"),
            "superoperator_build_s": classical.get("superoperator_build_s", ""),
            "classical_action_relative_error": classical.get("action_relative_error", ""),
            "source_result_path": str(stine_path),
            "notes": "Exact complex128 representation sizes, not peak RSS; large matrices not allocated. Matrix action time not measured."})

        variants = [v for v in stine["variants"] if v["basis"] == "nam"]
        if not variants:
            assert stine["status"] in {"preflight_memory_limit", "memory_limit", "timeout", "failed"}
            resources = read(stine_path.parent / "resources.json")
            rows.append({**base, "method": "Stinespring", "variant": "native" if stine["native_cptp"] else "flagged_minimum_beta",
                "status": stine["status"], "configuration_status": "budget_exceeded", "basis": "NAM", "optimization_level": 0,
                "G": "N/A", "Lambda": stine.get("Lambda", "N/A"), "Lambda_G": "N/A", "W": "N/A",
                "planned_W": stine.get("planned_W"), "isometry_shape": str(stine["isometry_shape"]),
                "isometry_bytes": stine["isometry_bytes"], "rss_limit_bytes": stine["protocol"]["rss_limit_bytes"],
                "timeout_s": stine["protocol"]["timeout_s"], "process_wall_s": resources["wall_s"],
                "peak_rss_bytes": resources["peak_rss_bytes"], "stopped_stage": stine.get("stopped_stage"),
                "full_choi_status": "not_run", "source_result_path": str(stine_path),
                "notes": stine.get("resource_reason", "Fixed resource budget reached during the recorded stage; no circuit output.")})
        for variant in variants:
            matched = variant["mode"] == "flagged_matched_historical_ours_lambda"
            error = variant.get("success_choi_relative_error")
            postprocessing_timeout = (stine["status"] == "timeout" and
                                      stine.get("stopped_stage") == "gate_classification" and
                                      variant["status"] == "synthesized")
            record = {**base, "method": "Stinespring", "variant": variant["mode"], "table1_candidate": not matched,
                "status": "synthesis_ok_postprocessing_timeout" if postprocessing_timeout else stine["status"],
                "run_status": stine["status"], "synthesis_status": "ok",
                "postprocessing_status": "NC_classification_timeout" if postprocessing_timeout else "ok",
                "stopped_stage": stine.get("stopped_stage", ""),
                "configuration_status": "supplementary_matched_Lambda" if matched else "available_with_partial_postprocessing" if postprocessing_timeout else "available",
                "basis": "NAM", "optimization_level": 0, "downstream_optimizer": "none",
                "G": variant["G"], "Lambda": variant["Lambda"], "Lambda_G": variant["Lambda_G"], "W": variant["W"],
                "CX": variant["CX"], "reported_compile_s": variant["compile_time_s"], "source_time_field": "compile_time_s",
                "time_scope": variant.get("time_scope", "materialization + rank preprocessing + completion/wrapper + synthesis; verification and I/O excluded"),
                "circuit_s": variant["synthesis_s"], "transpiler_seed": stine["protocol"]["seed_transpiler"],
                "full_choi_status": "passed_existing_verification" if error is not None else "not_run_budget",
                "full_choi_relative_error": "" if error is None else error,
                "source_result_path": str(stine_path), "G_over_Raw": variant["G"] / raw["metrics"]["G"],
                "Lambda_G_over_Raw": variant["Lambda_G"] / raw["Lambda_G"]}
            if "qasm" in variant:
                checked = recount(variant["qasm"])
                assert checked["G"] == variant["G"] and checked["W"] == variant["W"]
                assert file_hash(variant["qasm"]) == variant["qasm_sha256"]
                record.update(non_clifford=checked["non_clifford"], qasm_path=variant["qasm"],
                              qasm_sha256=variant["qasm_sha256"], qasm_recount="passed")
            else:
                assert sum(variant["counts"].values()) == variant["G"]
                record.update(non_clifford=variant.get("non_clifford", ""),
                              gate_count_evidence="actual basis circuit count_ops and basis check",
                              qasm_recount="not_applicable_no_exported_QASM", notes=variant["circuit_artifact"])
                if postprocessing_timeout:
                    record["notes"] += "; synthesis and G/CX/W completed within the 300 s budget; subsequent optional non-Clifford classification timed out; NC is missing, not zero"
            if case in {"TFIM-8", "HRW-8", "HRW-12"}:
                resources = read(stine_path.parent / "resources.json")
                record.update(process_wall_s=resources["wall_s"], peak_rss_bytes=resources["peak_rss_bytes"],
                    timeout_s=stine["protocol"]["timeout_s"], rss_limit_bytes=stine["protocol"]["rss_limit_bytes"],
                    isometry_bytes=stine["isometry_bytes"], isometry_shape=str(stine["isometry_shape"]))
            rows.append(record)

        for method, measured, path, qasm in (
            ("Raw channel-LCU", raw, raw_path, Path(raw["qasm"])),
            ("Ours", ours, ours_path, ours_path.parent / "channel_nam.qasm")):
            metrics = measured["metrics"] if method == "Raw channel-LCU" else measured
            assert file_hash(qasm) == measured["qasm_sha256"]
            checked = recount(qasm)
            assert checked["G"] == metrics["G"] and checked["W"] == metrics["W"]
            assert checked["counts"] == metrics["counts"]
            assert checked["non_clifford"] == measured["non_clifford"]
            verification = measured.get("verification", measured.get("full_choi", {}))
            no_gain = case in {"TFIM-4", "TFIM-8", "RepDQE-d3"}
            if method == "Ours" and no_gain:
                assert measured["assembly"]["rewrite"]["steps"] == 0
            rows.append({**base, "method": method, "variant": "R_off_O_off_Fstar_off" if method == "Raw channel-LCU" else "tree_Fstar_Matrix_order",
                "configuration_status": "available" if method == "Raw channel-LCU" else "accepted_no_rewrite_gain" if no_gain else "accepted_equivalent_HRW_rewrite",
                "rewrite_policy": "none" if method == "Raw channel-LCU" else json.dumps(measured["effective_rewrite_options"], separators=(",", ":")),
                "basis": "NAM", "optimization_level": 0, "downstream_optimizer": "none",
                "G": metrics["G"], "Lambda": measured["Lambda"], "Lambda_G": measured["Lambda"] * metrics["G"], "W": metrics["W"],
                "CX": checked["counts"].get("cx", 0), "non_clifford": checked["non_clifford"],
                "reported_compile_s": measured["compile_time_s"], "source_time_field": "compile_time_s",
                "time_scope": measured.get("timing_scope", "canonicalization + assembly + NAM lowering; imports, input construction, verification and I/O excluded"),
                "rewrite_s": 0.0 if method == "Raw channel-LCU" else measured["exclusive_stage_times_s"]["rewrite"],
                "rewrite_equivalence": "not_applied" if method == "Raw channel-LCU" else "passed",
                "rewrite_gram_relative_error": "" if method == "Raw channel-LCU" else rewrite_error,
                "full_choi_status": "passed_existing_verification" if verification.get("passed") else "not_run_budget",
                "full_choi_relative_error": verification.get("relative_frobenius_error", ""),
                "qasm_path": str(qasm), "qasm_sha256": measured["qasm_sha256"], "qasm_recount": "passed",
                "source_result_path": str(path), "G_over_Raw": metrics["G"] / raw["metrics"]["G"],
                "Lambda_G_over_Raw": measured["Lambda"] * metrics["G"] / raw["Lambda_G"],
                "notes": "TFIM/RepDQE rewrite has zero steps; accepted HRW rewrite policies are retained with explicit configuration labels." if method == "Ours" else "Frozen Raw output; paired to the same input."})
        print(json.dumps({"case": case, "input_pairing": "passed", "rewrite_gram_relative_error": rewrite_error,
                          "Stinespring_status": stine["status"], "Ours_G": ours["G"]}), flush=True)
    assert {r["case"] for r in rows} == set(CASES)
    assert sum(r["table1_candidate"] for r in rows) == 24
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Results/rq1_table1_resources_20261013.csv")
    args = parser.parse_args()
    result = collect()
    columns = list(dict.fromkeys(k for row in result for k in row))
    temporary = args.output.with_suffix(".csv.writing")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "rows": len(result), "candidate_rows": 24}), flush=True)
