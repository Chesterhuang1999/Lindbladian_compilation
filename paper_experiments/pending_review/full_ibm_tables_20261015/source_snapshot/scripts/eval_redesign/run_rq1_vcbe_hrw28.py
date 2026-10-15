"""Fresh raw-Kraus HRW-28 VCBE synthesis and audited comparison to saved Ours.

Reuse the existing no-rewrite experiment implementation without changing its
source or invalidating the provenance of the completed smaller batch.
"""
from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
import run_rq1_vcbe_no_rewrite as experiment
from run_rq1_vcbe_scaling import digest, dump, gram_error
from baseline_common import count_qasm_text
from experiment_protocol import source_identity
from rq1_pipeline import deserialize_kraus, input_identity

OUT = ROOT / "Baseline_results/vcbe_no_rewrite_hrw28_20261014"
CASE = "HRW-28"
ORIGINAL_FIT = experiment.fit_target


def fit_with_precision_floor(target, spectator_factor, directory, args):
    # sqrt(2**27) amplifies local error in the full-system Frobenius norm.
    # Avoid a sub-double-precision fitting goal, while checking the actual
    # spectator factor and retaining the 1e-10 full-system acceptance threshold.
    goal_factor = min(spectator_factor, 1e-12 / args.active_goal_floor)
    circuit, info = ORIGINAL_FIT(target, goal_factor, directory, args)
    error = float(np.linalg.norm(Operator(circuit).data[:len(target), :len(target)] - target))
    embedded_error = error * spectator_factor
    info.update(spectator_frobenius_factor=spectator_factor,
        fitting_goal_spectator_factor=goal_factor, active_goal_floor=args.active_goal_floor,
        active_fitting_target=max(1e-12 / spectator_factor, args.active_goal_floor),
        actual_full_embedded_frobenius_error=embedded_error,
        status="success" if embedded_error < experiment.TOL else "block_tolerance_not_met",
        fitting_goal_policy="max(1e-12/sqrt(spectator dimension), 1e-15); original full-system acceptance unchanged")
    dump(directory / "results.json", info)
    return circuit, info


def audit_qasm(path, expected):
    assert digest(path) == expected["qasm_sha256"]
    circuit = qasm2.loads(path.read_text())
    assert set(circuit.count_ops()) <= {"rz", "h", "x", "cx"}
    assert dict(circuit.count_ops()) == expected["counts"]
    assert circuit.num_qubits == expected["W"]
    count = count_qasm_text(path.read_text(), gate_set="nam")
    assert count["metric_total"] == expected["G"]
    return count


def export_comparison(row, out_dir):
    original_dir = ROOT / "paper_experiments/pending_review/tfim_hrw_full_20261011/cases" / CASE
    original_path = original_dir / "results.json"
    ours = json.loads(original_path.read_text())
    fixture_path = original_dir / "input.json"
    assert digest(fixture_path) == ours["input_sha256"] == row["provenance"]["input_file_sha256"]
    assert ours["canonicalization"]["raw_input_sha256"] == row["raw_kraus_sha256"]
    assert ours["protocol"]["basis"] == "nam"
    assert ours["protocol"]["lowering_optimization_level"] == 0
    assert ours["protocol"]["qubits_initially_zero"] is False
    equivalence = gram_error(row["kraus"], ours["assembly"]["compiled_kraus"])
    assert equivalence < experiment.TOL
    ours_qasm = original_dir / "channel_nam.qasm"
    ours_count = audit_qasm(ours_qasm, ours)
    assert ours_count["clifford"]["non_clifford"] == ours["non_clifford"]
    ours_cost = ours["Lambda"] * ours["G"]
    common = dict(case=CASE, n=28, M=row["M"], basis="NAM", final_optimization_level=0,
        seed_transpiler=20261009, qubits_initially_zero=False,
        target_kraus_sha256=row["raw_kraus_sha256"],
        target_path=str(fixture_path), target_file_sha256=digest(fixture_path),
        model_parameters=json.dumps(row["provenance"]["model"], separators=(",", ":")),
        ours_G=ours["G"], ours_Lambda=ours["Lambda"], ours_Lambda_G=ours_cost, ours_W=ours["W"],
        ours_source_result_path=str(original_path), ours_source_result_sha256=digest(original_path),
        target_map_coefficient_gram_relative_error=equivalence,
        metric_evidence="saved QASM hash, NAM basis, G, width, exact operation counts and non-Clifford recount verified")
    rows = [dict(common, method="Ours", status="ok", G=ours["G"], Lambda=ours["Lambda"],
        Lambda_G=ours_cost, W=ours["W"], CX=ours["counts"]["cx"], non_clifford=ours["non_clifford"],
        rewrite_enabled=True, rewrite_policy=json.dumps(ours["effective_rewrite_options"]),
        inner_control_policy="Fstar: control SELECT only", fresh_unique_fit_s="",
        reported_compile_s=ours["compile_time_s"], compile_time_scope=ours["timing_scope"],
        historical_fit_reused="not_applicable", measurement_reused=True,
        full_choi_status=ours["full_choi"]["status"],
        validation_scope="Pauli coefficient Gram equivalence; archived full-circuit Choi skipped",
        max_source_embedded_block_error="", max_compiled_inner_block_error="",
        G_over_Ours=1.0, Lambda_G_over_Ours=1.0,
        qasm_path=str(ours_qasm), qasm_sha256=digest(ours_qasm), source_result_path=str(original_path))]
    for method, measured in row["methods"].items():
        metric = measured["metrics"]["nam_opt0"]
        assert metric["status"] == "ok" and all(v["qualified"] for v in measured["block_validation"])
        count = audit_qasm(Path(metric["qasm_path"]), metric)
        assert count["clifford"]["non_clifford"] == metric["NC"]
        assert metric["Lambda"] == row["Lambda"]
        assert metric["Lambda_G"] == metric["Lambda"] * metric["G"]
        assert measured["metadata"]["alpha_per_kraus"] == row["alpha"]
        assert measured["metadata"]["outer_control_scope"] == "whole-block"
        rows.append(dict(common, method="VCBE" if method == "vcbe_source_tree" else "VCBE+opt3",
            status=metric["status"], G=metric["G"], Lambda=metric["Lambda"], Lambda_G=metric["Lambda_G"],
            W=metric["W"], CX=metric["CX"], non_clifford=metric["NC"], rewrite_enabled=False,
            rewrite_policy="disabled; exact original Kraus order, phases and coefficients retained",
            inner_control_policy="control whole synthesized unitary",
            fresh_unique_fit_s=row["fresh_unique_fit_s"],
            reported_compile_s=row["fresh_unique_fit_s"] + metric["assembly_plus_lowering_s"],
            compile_time_scope="sum of fresh fit wall time (including fit checkpoints and block checks) and assembly/lowering (including inner preoptimization and block checks); excludes imports, final QASM I/O and full-channel verification",
            historical_fit_reused=False, measurement_reused=False,
            full_choi_status=metric["full_choi_status"], validation_scope=metric["validation_scope"],
            max_source_embedded_block_error=max(b["full_embedded_frobenius_block_error"] for b in row["vcbe_blocks"]),
            max_compiled_inner_block_error=max(v["full_embedded_frobenius_block_error"] for v in measured["block_validation"]),
            G_over_Ours=metric["G"] / ours["G"], Lambda_G_over_Ours=metric["Lambda_G"] / ours_cost,
            qasm_path=metric["qasm_path"], qasm_sha256=metric["qasm_sha256"],
            source_result_path=str(out_dir / "cases" / CASE / "results.json")))
    output = ROOT / "Results/rq1_vcbe_hrw28_20261014.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--fit-budget", type=float, default=600)
    parser.add_argument("--trial-budget", type=float, default=50)
    parser.add_argument("--polish-budget", type=float, default=20)
    parser.add_argument("--active-goal-floor", type=float, default=1e-15)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.case = CASE
    assert not (args.out_dir / "results.json").exists(), "Use a new directory for a fresh run"
    before = source_identity()
    gradient_checks = experiment.check_adjoint()
    experiment.fit_target = fit_with_precision_floor
    started = time.perf_counter()
    experiment.worker(args)
    case_path = args.out_dir / "cases" / CASE / "results.json"
    row = json.loads(case_path.read_text())
    assert row["status"] == "success" and row["raw_kraus_unchanged"]
    assert row["production_sources_unchanged"] and before == source_identity()
    assert input_identity(deserialize_kraus(row["kraus"])) == row["raw_kraus_sha256"]
    output, comparisons = export_comparison(row, args.out_dir)
    record = dict(status="success", sources=before, case=CASE, cases={CASE: row},
        runner_sha256=digest(__file__), base_runner_sha256=digest(Path(experiment.__file__)),
        gradient_checks=gradient_checks, vcbe="independent generic Fig.1/Eq.(7) reproduction, arXiv:2507.17658v2",
        rewrite_enabled=False, historical_fit_reused=False, active_goal_floor=args.active_goal_floor,
        full_embedded_block_acceptance=experiment.TOL,
        fitting_recipe=dict(per_unique_target_budget_s=args.fit_budget, per_bfgs_trial_budget_s=args.trial_budget,
                            per_polish_budget_s=args.polish_budget, layers=[1, 2, 4, 8, 16, 32], seeds=[0, 1]),
        elapsed_s=time.perf_counter() - started, comparison_csv=str(output), comparison_rows=comparisons,
        environment=dict(python=sys.version, platform=platform.platform(),
                         packages={p: version(p) for p in ("qiskit", "numpy", "scipy")}))
    dump(args.out_dir / "results.json", record)
    for comparison in comparisons:
        print("COMPARISON", json.dumps({k: comparison[k] for k in
            ("method", "G", "Lambda", "Lambda_G", "W", "G_over_Ours", "Lambda_G_over_Ours")}), flush=True)
    print("CSV", str(output), flush=True)


if __name__ == "__main__":
    main()
