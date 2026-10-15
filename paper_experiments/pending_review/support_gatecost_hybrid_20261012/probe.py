"""Isolated support-frontier selection plus bounded gate-cost polishing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PENDING = HERE.parent
ROOT = HERE.parents[2]
PREVIOUS = PENDING / "streaming_support_beam_20261012"
sys.path.insert(0, str(PREVIOUS))
import fast_probe as support
import diagnostic_gatecost as gate
from support_archive import search as archived_search
from channel_IR import channel as ChannelIR, Matrixsum, PauliAtom, channel_ensemble
from rq1_pipeline import canonicalize, verify_channel
from channel_LCU import channel_to_LCU
from block_encoding import BlockEncoding
from experiment_protocol import lower_and_count
sys.path.insert(0, str(ROOT / "Baseline_scripts"))
from baseline_common import count_qasm_text
from qiskit import qasm2

TOL = 1e-12
CASES = ["Decay-HO", "HRW-12", "Color-n4q4-linear", "Petz_Ising3", "CPTP2_M4_seed4"]


def rows(A, labels):
    return [Matrixsum([(PauliAtom(str(p), phase=z / abs(z)), float(abs(z)))
                      for p, z in zip(labels, row) if abs(z) > TOL])
            for row in A if np.any(np.abs(row) > TOL)]


def input_data(name):
    base = PENDING / "new_rewrite_incremental_20261011"
    if (base / name / "reference_arrays.npz").exists():
        path, key = base / name / "reference_arrays.npz", "canonical"
    elif name == "HRW-12":
        path, key = PREVIOUS / "inputs/HRW-12.npz", "A_initial"
    elif name.startswith("CPTP"):
        path = PENDING / "support_gatecost_counterexamples_20261012/cases" / name / "input.npz"
        key = "A"
    else:
        path = PENDING / "remaining_full_20261011/cases" / name / "input.json"
        data = json.loads(path.read_text())
        raw = [Matrixsum([(PauliAtom(p, phase=complex(re, im)), mag) for p, mag, re, im in row])
               for row in data["input_kraus"]]
        canonical, info = canonicalize(raw)
        A, labels = ChannelIR(canonical)._coeff_matrix(TOL)
        return A, np.array(labels), {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "canonicalization": info}
    z = np.load(path)
    return z[key].copy(), z["labels"], {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "array": key}


def H(A):
    return support.metrics(A, TOL)["static_cost"]["total"]


def save_state(dest, method, A0, labels, state, termination, wall_s):
    verification, replay = support.verify(A0, state.A, state.steps, TOL)
    assert verification["passed"], verification
    np.savez_compressed(dest / (method + ".npz"), A_initial=A0, A_final=state.A,
                        A_replayed=replay, labels=labels)
    support.dump(dest / (method + "_steps.json"),
                 [{"pair": [i, j], "U_real": u.real.tolist(), "U_imag": u.imag.tolist()} for i, j, u in state.steps])
    return {"metrics": support.metrics(state.A, TOL), "verification": verification,
            "termination": termination, "wall_s": wall_s,
            "status": "ok" if termination["stop_reason"] == "frontier_exhausted" else "incomplete"}


def compile_output(dest, method, A0, A, labels, rewrite_record):
    path = dest / (method + "_circuit.json")
    if path.exists():
        return json.loads(path.read_text())
    support.dump(dest / "progress.json", {"stage": "compile", "method": method, "pid": os.getpid()})
    start = time.perf_counter()
    backend = lambda ms: BlockEncoding(ms, placement_options={}, matrix_order_options={})
    circuit, registers = channel_to_LCU(channel_ensemble([rows(A, labels)]), structure="tree", opt="Matrix-order",
                                       block_encoding_cls=backend, fstar=True)
    assembly_s = time.perf_counter() - start
    lowered, metrics = lower_and_count(circuit, rq="RQ1", basis="nam")
    compile_s = time.perf_counter() - start
    text = qasm2.dumps(lowered)
    (dest / (method + "_nam.qasm")).write_text(text)
    stats = count_qasm_text(text, gate_set="nam")
    assert stats["metric_total"] == metrics["G"]
    assert not (set(metrics["counts"]) - {"rz", "h", "x", "cx", "reset"})
    if len(registers[-1]) <= 4 and lowered.num_qubits <= 20:
        verification = verify_channel(lowered, registers, rows(A0, labels))
        assert verification["passed"], verification
    else:
        verification = {"status": "not_run", "reason": "full Choi probe budget: <=4 system and <=20 total qubits",
                        "rewrite_equivalence_passed": rewrite_record["verification"]["passed"]}
    settings = dict(metrics["settings"])
    settings["protocol_default_rewrite"] = settings.pop("rewrite")
    settings["rewrite"] = {"method": method, "source": str(dest / "results.json"), "applied_during_compilation": False}
    lam = float(circuit.metadata["subnormalization_factor"])
    result = {"G": metrics["G"], "W": metrics["W"], "Lambda": lam, "Lambda_G": lam * metrics["G"],
              "NC": stats["clifford"]["non_clifford"], "Lambda_NC": lam * stats["clifford"]["non_clifford"],
              "compile_s": compile_s, "assembly_s": assembly_s, "lowering_s": metrics["lowering_time_s"],
              "verification": verification, "qasm_recount": stats, "settings": settings,
              "sources": metrics["sources"], "metadata": circuit.metadata}
    support.dump(path, result)
    return result


def historical(name):
    for folder in ("remaining_full_20261011", "tfim_hrw_full_20261011"):
        path = PENDING / folder / "cases" / name / "results.json"
        if path.exists():
            r = json.loads(path.read_text())
            return {"path": str(path), "status": r["status"], "H": r["static_fstar_cost_audit"]["total"],
                    "S": r["static_fstar_cost_audit"]["total_support"], "M": r["static_fstar_cost_audit"]["kraus_count"],
                    "G": r["G"], "Lambda": r["Lambda"], "Lambda_G": r["Lambda"] * r["G"],
                    "rewrite_options": r["effective_rewrite_options"], "termination": r["rewrite_termination"]["stop_reason"]}
    return {"status": "no_successful_full_pipeline_reference"}


def main(name):
    if name not in CASES:
        raise ValueError("Fixture case retired from this legacy probe; use paper_experiments/experiments_new/fixture_benchmarks/run.py")
    dest = HERE / "cases" / name
    dest.mkdir(parents=True, exist_ok=True)
    A0, labels, source = input_data(name)
    max_steps = max(256, int(np.count_nonzero(np.abs(A0) > TOL)) + 1)
    timeout = 300
    manifest = {"case": name, "input": source, "input_matrix_sha256": hashlib.sha256(A0.tobytes()).hexdigest(),
                "shape": list(A0.shape), "beam_width": 8, "max_steps_per_search": max_steps, "timeout_per_search_s": timeout,
                "support_rank": "support_rank", "gate_rank": "admit_support", "keep_neutral": True,
                "hybrid": "Keep original support final and minimum-H committed frontier state; polish at most two distinct seeds; retain both seeds as fallbacks.",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), HERE / "support_archive.py", PREVIOUS / "fast_probe.py", PREVIOUS / "diagnostic_gatecost.py", PREVIOUS / "candidate_kernel.py"]}}
    support.dump(dest / "manifest.json", manifest)
    np.savez_compressed(dest / "input.npz", A=A0, labels=labels)
    # Warm the arithmetic kernels on a tiny independent matrix, outside timings.
    warm = np.array([[1, 1j], [1, -1j]], complex)
    warm_start = time.perf_counter()
    support.search(warm, None, max_steps=1, rank_mode="support_rank")
    gate.search(warm, None, max_steps=1, mode="admit_support", neutral=True)
    jit_s = time.perf_counter() - warm_start
    result = {"case": name, "input": support.metrics(A0, TOL), "historical": historical(name), "jit_warmup_s": jit_s, "methods": {}}
    archive = {"state": None, "key": None, "H": float("inf"), "layer": 0, "evaluated": 0, "time_s": 0}
    def observe(layer, beam):
        start = time.perf_counter()
        for s in beam:
            h = H(s.A)
            archive["evaluated"] += 1
            rank = (h, s.support, int(np.count_nonzero(np.any(np.abs(s.A) > TOL, axis=1))))
            old = archive["state"]
            if old is None or rank < (archive["H"], old.support, int(np.count_nonzero(np.any(np.abs(old.A) > TOL, axis=1)))):
                archive.update(state=s, key=support.state_key(s.keys), H=h, layer=layer)
        archive["time_s"] += time.perf_counter() - start
    start = time.perf_counter()
    pure, termination = archived_search(A0, dest, width=8, max_steps=max_steps, timeout=timeout,
                                        rank_mode="support_rank", neutral=True, on_layer=observe)
    support_s = time.perf_counter() - start
    result["methods"]["support"] = save_state(dest, "support", A0, labels, pure, termination, support_s)
    result["archive"] = {k: v for k, v in archive.items() if k not in ("state", "key")}
    chosen = archive["state"]
    result["methods"]["selected"] = save_state(dest, "selected", A0, labels, chosen, termination, support_s)
    # Confirm observer leaves the historical support trajectory/output unchanged.
    reference_name = "Thermal2_HO_support_rank" if name == "Thermal2_HO" else name
    reference = PREVIOUS / reference_name / "coefficients.npz"
    if reference.exists():
        old = np.load(reference)
        if old["A_initial"].shape == A0.shape and np.allclose(old["A_initial"], A0, atol=1e-14, rtol=1e-14):
            error = float(np.linalg.norm(old["A_final"] - pure.A))
            result["observer_reference_output_error"] = error
            assert error < 1e-10, error
    support.dump(dest / "results.json", result)
    seeds = [chosen]
    if support.state_key(pure.keys) != support.state_key(chosen.keys):
        seeds.append(pure)
    polished = [chosen, pure]
    records = []
    for i, seed in enumerate(seeds):
        support.dump(dest / "progress.json", {"stage": "polish", "seed": i, "pid": os.getpid()})
        start = time.perf_counter()
        state, term = gate.search(seed.A, None, width=8, max_steps=max_steps, timeout=timeout, mode="admit_support", neutral=True)
        elapsed = time.perf_counter() - start
        combined = support.State(state.A, state.support, state.keys, seed.steps + state.steps)
        record = save_state(dest, "polish_" + str(i), A0, labels, combined, term, elapsed)
        record["seed_H"] = H(seed.A)
        records.append(record)
        polished.append(combined)
    best = min(polished, key=lambda s: (H(s.A), s.support))
    hybrid_term = {"stop_reason": "frontier_exhausted" if termination["stop_reason"] == "frontier_exhausted" and all(r["status"] == "ok" for r in records) else "incomplete_stage",
                   "support_stop": termination["stop_reason"], "polish_stops": [r["termination"]["stop_reason"] for r in records],
                   "accepted_steps": len(best.steps)}
    result["polishes"] = records
    result["methods"]["hybrid"] = save_state(dest, "hybrid", A0, labels, best, hybrid_term, support_s + sum(r["wall_s"] for r in records))
    assert H(best.A) <= min(H(pure.A), H(A0)) * (1 + 1e-12)
    support.dump(dest / "results.json", result)
    print(json.dumps({"case": name, "phase": "hybrid_done", "methods": {k:v["metrics"] for k,v in result["methods"].items()}}), flush=True)
    # Independent gate search reference. Reuse validated same-mode Thermal run.
    if name == "Thermal2_HO":
        d = PREVIOUS / "diagnosis_admit_support"
        z = np.load(d / "coefficients.npz")
        input_difference = float(np.max(np.abs(z["A_initial"] - A0)))
        assert input_difference < 1e-14, input_difference
        r = json.loads((d / "results.json").read_text())
        seq = json.loads((d / "rewrite_steps.json").read_text())
        steps = tuple((s["pair"][0],s["pair"][1], np.array(s["U_real"])+1j*np.array(s["U_imag"])) for s in seq)
        replayed = A0.copy()
        for i, j, u in steps:
            replayed[[i, j]] = u @ replayed[[i, j]]
            replayed[np.abs(replayed) < TOL] = 0
        state = support.State(replayed,int(np.count_nonzero(np.abs(replayed)>TOL)),tuple(support.row_key(row,TOL) for row in replayed),steps)
        record = save_state(dest,"gate",A0,labels,state,r["termination"],r["termination"]["search_time_s"])
        record["replayed_cached_path_from"] = str(d / "results.json")
        record["cached_input_max_absolute_difference"] = input_difference
        record["search_time_note"] = "Historical search time; rotations replayed and verified on this run's exact input."
    else:
        support.dump(dest / "progress.json", {"stage": "gate_from_input", "pid": os.getpid()})
        start = time.perf_counter()
        state, term = gate.search(A0, dest, width=8, max_steps=max_steps, timeout=timeout, mode="admit_support", neutral=True)
        record = save_state(dest,"gate",A0,labels,state,term,time.perf_counter()-start)
    result["methods"]["gate"] = record
    support.dump(dest / "results.json", result)
    variants = {"support": pure.A, "hybrid": best.A, "gate": state.A}
    compiled = {}
    for method, array in variants.items():
        if result["methods"][method]["status"] != "ok":
            continue
        same = next((m for m in compiled if np.array_equal(array, variants[m])), None)
        if same:
            c = dict(compiled[same], reused_identical_output_of=same)
            support.dump(dest / (method + "_circuit.json"), c)
        else:
            c = compile_output(dest,method,A0,array,labels,result["methods"][method])
        compiled[method] = c
        result["methods"][method]["circuit"] = {k:c[k] for k in ["G","W","Lambda","Lambda_G","NC","Lambda_NC","compile_s","verification"]}
        support.dump(dest / "results.json", result)
    result["status"] = "ok" if all(v["status"] == "ok" for v in result["methods"].values()) else "incomplete"
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    support.dump(dest / "results.json", result)
    support.dump(dest / "progress.json", {"stage": "finished", "status": result["status"]})
    print(json.dumps({"case":name,"status":result["status"],"methods":{k:{"S":v["metrics"]["support"],"H":v["metrics"]["static_cost"]["total"],"circuit":v.get("circuit")} for k,v in result["methods"].items()}}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    main(parser.parse_args().case)
