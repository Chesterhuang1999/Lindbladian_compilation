"""Fill IBM layout gaps using frozen circuits and saved Kraus representations.

Run the archive smoke into OUT first, then execute with qiskit_simulate.
No Kraus rewrite, variational fit, Stinespring synthesis, or full Choi check.
Completed historical runners and their manifest-bound sources are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import gc
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Baseline_results/rq1_layout_ibm_completion_20261014"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["QISKIT_PARALLEL"] = "FALSE"
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / name) for name in ("src", "Baseline_scripts", "scripts/eval_redesign")]

import numpy as np
import qiskit
from qiskit import QuantumCircuit, qasm2, qpy
from block_encoding import BlockEncoding
from channel_IR import channel as ChannelIR, channel_ensemble
from channel_LCU import channel_to_LCU
from experiment_protocol import source_identity, transpile_channel_circuit
from rq1_pipeline import build_pipeline, deserialize_kraus, input_identity
from run_rq1_hrw_ibm_opt3 import digest, dump, counts, map_error, require_identity_layout, save_qpy
from run_rq1_tfim_dqe_petz_ibm_opt0 import read, local_checks, recount_qasm
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SUITE = ROOT / "paper_experiments/pending_review"
OURS = ("TFIM-4", "TFIM-8", "TFIM-12", "TFIM-20", "RepDQE-d3", "Petz_Ising3")
IBM, NAM, SEED = ["u1", "u2", "u3", "cx"], ["rz", "h", "x", "cx"], 20261009
TOL = 1e-10


def lower(source, basis=IBM, level=0):
    return transpile_channel_circuit(source, basis_gates=basis, optimization_level=level,
        seed_transpiler=SEED, approximation_degree=1.0, qubits_initially_zero=False, num_processes=1)


def fixture_path(case):
    if case == "HRW-8":
        return ROOT / "Baseline_results/table1_scaling_ours_20261013/cases/HRW-8/input.json"
    if case.startswith("TFIM-") or case.startswith("HRW-"):
        return SUITE / "tfim_hrw_full_20261011/cases" / case / "input.json"
    if case == "RepDQE-d3":
        return SUITE / "remaining_full_20261011/cases" / case / "input.json"
    return SUITE / "raw_baseline_20261012/inputs/Petz_Ising3.json"


def audit(path):
    plan = read(OUT / "manifest.json")
    assert source_identity() == plan["sources"]
    assert digest(__file__) == plan["runner_sha256"]
    assert digest(path) == plan["files"][str(path)], path


def save(case, method, circuit, lam, original, target_path, elapsed, **extra):
    require_identity_layout(circuit)
    G, operations = counts(circuit, IBM)
    flat = QuantumCircuit(circuit.num_qubits)
    flat.compose(circuit, inplace=True)
    directory = OUT / "cases" / case
    directory.mkdir(parents=True, exist_ok=True)
    slug = method.replace("+", "_").replace(" ", "_")
    path = directory / f"{slug}_ibm.qasm"
    qasm2.dump(flat, path)
    recount_G, recount_ops, W, nc = recount_qasm(path)
    assert (recount_G, recount_ops, W) == (G, operations, circuit.num_qubits)
    target = read(target_path)["input_kraus"]
    display_lam = str(lam) if case == "Petz_Ising3" else str(round(lam, 12))
    assert abs(lam - float(display_lam)) < 1e-12
    row = dict(case=case, method=method, status="ok", basis="IBM", G=G, W=W,
        Lambda=lam, Lambda_G=lam * G, Lambda_display=display_lam,
        Lambda_G_display=format(Decimal(display_lam) * G, "f"), counts=operations,
        CX=operations.get("cx", 0), non_clifford=nc, lowering_s=elapsed,
        optimizer_s=elapsed if "+Qiskit opt3" in method else 0.0,
        optimizer_scope="whole assembled IBM opt0 circuit" if "+Qiskit opt3" in method else "none",
        lowering_optimization_level=3 if "+Qiskit opt3" in method else 0,
        basis_gates=IBM, seed_transpiler=SEED, approximation_degree=1.0,
        qubits_initially_zero=False, numerical_threads=1, num_processes=1,
        target_path=str(target_path), target_file_sha256=digest(target_path),
        target_kraus_sha256=input_identity(deserialize_kraus(target)),
        original_result_path=str(original), original_result_sha256=digest(original),
        qasm_path=str(path), qasm_sha256=digest(path),
        global_phase_rad=float(circuit.global_phase),
        independent_qasm_recount="passed G, NC, counts and width",
        full_choi_status="skipped_user_request",
        validation_scope="frozen input hashes, sparse coefficient Gram, archived NAM histogram and independent IBM QASM recount; no new full Choi",
        **extra)
    assert source_identity() == read(OUT / "manifest.json")["sources"]
    dump(directory / f"{slug}_results.json", row)
    print("RESULT", case, method, "G", G, "C", row["Lambda_G_display"], "W", W, flush=True)
    return row


def ours(case):
    target_path = fixture_path(case)
    audit(target_path)
    target = read(target_path)["input_kraus"]
    if case == "Petz_Ising3":
        directory = SUITE / "support_gatecost_hybrid_20261012/cases" / case
        original = directory / "hybrid_circuit.json"
        audit(directory / "hybrid.npz")
        arrays = np.load(directory / "hybrid.npz")
        encoded = [[[str(p), float(abs(c)), float(c.real / abs(c)), float(c.imag / abs(c))]
                    for p, c in zip(arrays["labels"], row) if c != 0] for row in arrays["A_final"]]
        record = read(original)
        options = record["settings"]["matrix_order_options"]
        historical_qasm = directory / "support_nam.qasm"
        reference_counts = counts(qasm2.load(historical_qasm), NAM)
        assert reference_counts[0] == record["G"] and record["reused_identical_output_of"] == "support"
    else:
        family = "remaining_full_20261011" if case == "RepDQE-d3" else "tfim_hrw_full_20261011"
        directory = SUITE / family / "cases" / case
        original, historical_qasm = directory / "results.json", directory / "channel_nam.qasm"
        record = read(original)
        assert digest(target_path) == record["input_sha256"]
        assert digest(historical_qasm) == record["qasm_sha256"]
        encoded, options = record["assembly"]["compiled_kraus"], record["assembly"]["matrix_order_options"]
        reference_counts = (record["G"], record["counts"])
    audit(original)
    audit(historical_qasm)
    error = map_error(target, encoded)
    assert error < TOL
    def backend(ms):
        return BlockEncoding(ms, placement_options={}, matrix_order_options=options)
    source, registers = channel_to_LCU(channel_ensemble([deepcopy(deserialize_kraus(encoded))]),
        structure="tree", opt="Matrix-order", fstar=True, block_encoding_cls=backend)
    if case != "Petz_Ising3":
        assert registers == record["registers"]
    assert np.isclose(source.metadata["subnormalization_factor"], record["Lambda"], rtol=1e-12)
    assert counts(lower(source, NAM), NAM) == reference_counts
    save_qpy(source, OUT / "cases" / case / "Ours_source.qpy")
    started = time.perf_counter()
    circuit = lower(source)
    elapsed = time.perf_counter() - started
    assert circuit.num_qubits == record["W"]
    return save(case, "Ours", circuit, record["Lambda"], original, target_path, elapsed,
        reused_rewrite=True, rewrite_enabled=True, matrix_order_options=options,
        backend_change_scope="final lowering only; frozen rewrite and NAM placement options retained",
        reference_NAM_G=record["G"], reference_NAM_counts=reference_counts[1],
        nam_reproduction="passed exact archived histogram", target_map_coefficient_gram_relative_error=error)


def raw4():
    case = "TFIM-4"
    directory = SUITE / "rq1_paper_suite_20261010/full_v2/runs" / case / "raw"
    original, target_path = directory / "results.json", fixture_path(case)
    audit(original); audit(target_path); audit(directory / "channel_nam_opt0.qasm")
    record, target = read(original), read(target_path)["input_kraus"]
    assert input_identity(deserialize_kraus(target)) == record["canonicalization"]["raw_input_sha256"]
    assert digest(directory / "channel_nam_opt0.qasm") == record["qasm_sha256"]
    error = map_error(target, record["canonical_kraus"])
    assert error < TOL
    source, registers, _ = build_pipeline(deserialize_kraus(record["canonical_kraus"]), "Raw")
    assert registers == record["registers"]
    assert counts(lower(source, NAM), NAM) == (record["metrics"]["G"], record["metrics"]["counts"])
    started = time.perf_counter(); circuit = lower(source); elapsed = time.perf_counter() - started
    assert circuit.num_qubits == record["metrics"]["W"]
    return save(case, "Raw", circuit, record["Lambda"], original, target_path, elapsed,
        rewrite_enabled=False, reference_NAM_G=record["metrics"]["G"],
        nam_reproduction="passed exact archived histogram", target_map_coefficient_gram_relative_error=error)


def vcbe(case):
    batch = "vcbe_channel_smoke_20261012" if case == "TFIM-4" else "vcbe_no_rewrite_20261014"
    directory = ROOT / "Baseline_results" / batch / "cases" / case
    original, target_path = directory / "results.json", fixture_path(case)
    audit(original); audit(target_path)
    record, target = read(original), read(target_path)["input_kraus"]
    assert record["status"] == "success"
    assert map_error(target, record["kraus"]) < TOL
    assert input_identity(deserialize_kraus(target)) == input_identity(deserialize_kraus(record["kraus"]))
    metric = record["methods"]["vcbe_source_tree"]["metrics"]
    if case == "TFIM-4":
        path = directory / "vcbe_source_tree_ibm_opt0.qasm"
        audit(path)
        assert digest(path) == metric["ibm_opt0"]["qasm_sha256"]
        assert metric["ibm_opt0"]["validation"]["passed"]
        source = qasm2.load(path)
        assert counts(source, IBM) == (metric["ibm_opt0"]["G"], metric["ibm_opt0"]["counts"])
        elapsed, validation = 0.0, "passed_archived_ibm_opt0"
        assert record["provenance"] == str(target_path)
    else:
        assert record["raw_kraus_unchanged"] and not record["rewrite"]["enabled"]
        path = directory / "vcbe_source_tree_source.qpy"
        audit(path)
        with path.open("rb") as stream: source = qpy.load(stream)[0]
        assert source.metadata["outer_control_scope"] == "whole-block"
        assert counts(lower(source, NAM), NAM) == (metric["nam_opt0"]["G"], metric["nam_opt0"]["counts"])
        checks = local_checks(record, directory)
        dump(OUT / "cases" / case / "vcbe_local_validation.json", checks)
        started = time.perf_counter(); source = lower(source); elapsed = time.perf_counter() - started
        validation = "skipped_user_request"
    row = save(case, "VCBE", source, record["Lambda"], original, target_path, elapsed,
        rewrite_enabled=False, raw_kraus_unchanged=True, reused_vcbe_fit=True,
        inner_preoptimization_level=0, outer_control_scope="whole-block",
        reference_NAM_G=metric["nam_opt0"]["G"])
    row["full_choi_status"] = validation
    row["validation_scope"] = ("archived IBM opt0 full Choi and independent QASM recount"
                               if case == "TFIM-4" else "input hash, archived NAM histogram, local blocks and QASM recount; no full Choi")
    dump(OUT / "cases" / case / "VCBE_results.json", row)
    return row


def opt3(previous):
    assert digest(previous["qasm_path"]) == previous["qasm_sha256"]
    source = qasm2.load(previous["qasm_path"])
    assert counts(source, IBM) == (previous["G"], previous["counts"])
    started = time.perf_counter(); circuit = lower(source, level=3); elapsed = time.perf_counter() - started
    assert circuit.num_qubits == previous["W"]
    return save(previous["case"], previous["method"] + "+Qiskit opt3", circuit, previous["Lambda"],
        Path(previous["original_result_path"]), Path(previous["target_path"]), elapsed,
        input_G=previous["G"], input_qasm_path=previous["qasm_path"],
        input_qasm_sha256=previous["qasm_sha256"], rewrite_enabled=False,
        reused_vcbe_fit=previous["method"] == "VCBE", inner_preoptimization_level=0)


def main():
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and smoke["sources"] == source_identity()
    assert not (OUT / "manifest.json").exists(), "Use a fresh batch for new counts"
    paths = {fixture_path(c) for c in (*OURS, "HRW-8")}
    for case in OURS:
        family = "remaining_full_20261011" if case == "RepDQE-d3" else "tfim_hrw_full_20261011"
        if case == "Petz_Ising3":
            folder = SUITE / "support_gatecost_hybrid_20261012/cases" / case
            paths.update(folder / name for name in ("hybrid.npz", "hybrid_circuit.json", "support_nam.qasm"))
        else:
            folder = SUITE / family / "cases" / case
            paths.update(folder / name for name in ("results.json", "channel_nam.qasm"))
    folder = SUITE / "rq1_paper_suite_20261010/full_v2/runs/TFIM-4/raw"
    paths.update(folder / name for name in ("results.json", "channel_nam_opt0.qasm"))
    for batch, case, source in (("vcbe_channel_smoke_20261012", "TFIM-4", "vcbe_source_tree_ibm_opt0.qasm"),
                                 ("vcbe_no_rewrite_20261014", "HRW-8", "vcbe_source_tree_source.qpy")):
        folder = ROOT / "Baseline_results" / batch / "cases" / case
        paths.update(folder / name for name in ("results.json", source))
    for case in (*OURS, "HRW-8"): (OUT / "cases" / case).mkdir(parents=True, exist_ok=True)
    dump(OUT / "manifest.json", dict(sources=source_identity(), runner_sha256=digest(__file__),
        files={str(p): digest(p) for p in sorted(paths)}, basis_gates=IBM, seed_transpiler=SEED,
        approximation_degree=1.0, qubits_initially_zero=False, num_processes=1,
        numerical_threads=1, qiskit=qiskit.__version__, full_choi_status="skipped_user_request",
        ours_policy="frozen rewrite and NAM placement; IBM final lowering only",
        vcbe_policy="original Kraus, no inner opt3; qualified archived fits",
        opt3_scope="whole assembled IBM opt0 circuit"))
    def no_rewrite(*args, **kwargs): raise AssertionError("No new rewrite in this counting batch")
    for name in ("rewrite_search", "gatecost_rewrite_search", "apply_rewrite_result"):
        setattr(ChannelIR, name, no_rewrite)
    rows = []
    def collect(row):
        rows.append(row)
        dump(OUT / "results.json", dict(status="running", rows=rows, manifest_path=str(OUT / "manifest.json")))
        gc.collect()
    for case in OURS:
        print("COUNT", case, "Ours", flush=True); collect(ours(case))
    row = raw4(); collect(row); collect(opt3(row))
    for case in ("TFIM-4", "HRW-8"):
        row = vcbe(case); collect(row); collect(opt3(row))
    assert len(rows) == 12
    dump(OUT / "results.json", dict(status="complete", rows=rows, manifest_path=str(OUT / "manifest.json")))


if __name__ == "__main__":
    main()
