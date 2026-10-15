"""Bounded VCBE scaling tests, without allocating a full channel Choi matrix.

Run in qiskit_simulate with --skip-full-choi. The old VCBE batches are read
only. Local templates retain the generic VCBE ansatz and the Pauli 1-norm.
Unqualified fitted circuits are saved as diagnostics, never valid baselines.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import psutil
from qiskit import QuantumCircuit, qasm2, qpy
from qiskit.quantum_info import Operator
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Baseline_results/vcbe_scaling_20261014"
OLD = ROOT / "Baseline_results/matrix_synthesis_hrw_tfim_20261012"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts"), str(OLD)]
from baseline_common import count_qasm_text
from channel_IR import channel_ensemble
from channel_LCU import channel_to_LCU
from experiment_protocol import source_identity, transpile_channel_circuit
from rq1_pipeline import deserialize_kraus, input_identity, serialize_kraus
from vcbe_probe import Ansatz, StopSearch
from least_squares_precision import residual_jacobian

TOL = 1e-10
SEED = 20261009
NAM = ["rz", "h", "x", "cx"]
IBM = ["u1", "u2", "u3", "cx"]
METHODS = ("vcbe_source_tree", "vcbe_preopt3_tree")
CASES = ("HRW-8", "HRW-12", "TFIM-8")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_circuit(path, qc):
    with Path(path).open("wb") as stream:
        qpy.dump(qc, stream)


def coefficients(encoded):
    rows = []
    for terms in encoded:
        row = Counter()
        for p, magnitude, real, imag in terms:
            row[p] += magnitude * complex(real, imag)
        rows.append({p: c for p, c in row.items() if c != 0})
    return rows


def gram_error(left, right):
    def gram(encoded):
        q = Counter()
        for row in coefficients(encoded):
            for p, a in row.items():
                for r, b in row.items():
                    q[p, r] += a.conjugate() * b
        return q
    a, b = gram(left), gram(right)
    return float(np.sqrt(sum(abs(a[k] - b[k]) ** 2 for k in a.keys() | b.keys())
                         / sum(abs(c) ** 2 for c in a.values())))


def load_case(name):
    if name == "HRW-8":
        directory = ROOT / "Baseline_results/table1_scaling_ours_20261013/cases" / name
    else:
        directory = ROOT / "paper_experiments/pending_review/tfim_hrw_full_20261011/cases" / name
    target_path = directory / "input.json"
    target = read(target_path)["input_kraus"]
    if name == "HRW-12":
        shared = ROOT / "paper_experiments/pending_review/support_gatecost_hybrid_20261012/cases" / name
        shared_path = shared / "hybrid.npz"
        with np.load(shared_path) as arrays:
            encoded = []
            for row in arrays["A_final"]:
                terms = [[str(p), float(abs(c)), float(c.real / abs(c)), float(c.imag / abs(c))]
                         for p, c in zip(arrays["labels"], row) if c != 0]
                if terms:
                    encoded.append(terms)
        shared_result_path = shared / "hybrid_circuit.json"
    else:
        shared_path = shared_result_path = directory / "results.json"
        encoded = read(shared_path)["assembly"]["compiled_kraus"]
    error = gram_error(target, encoded)
    assert error < TOL, (name, error)
    kraus = deserialize_kraus(encoded)
    encoded = serialize_kraus(kraus)
    lam = float(sum(ms.pauli_norm() ** 2 for ms in kraus))
    assert np.isclose(lam, read(shared_result_path)["Lambda"], rtol=1e-12, atol=1e-14)
    return kraus, {
        "target_path": str(target_path), "target_file_sha256": digest(target_path),
        "target_kraus_sha256": input_identity(deserialize_kraus(target)),
        "shared_kraus_path": str(shared_path), "shared_kraus_file_sha256": digest(shared_path),
        "ours_result_path": str(shared_result_path), "ours_result_sha256": digest(shared_result_path),
        "rewrite_coefficient_gram_relative_error": error,
        "shared_representation": "current table Ours compiled Kraus operators, with identical alpha_j",
    }


class ReversibleAdjointAnsatz(Ansatz):
    """Same gates/loss/gradient as Ansatz; recover forward states by inversion.

    This stores two block-column arrays instead of one for every gate. It does
    not construct a Choi matrix or a sensitivity tensor for the eight-qubit fit.
    """
    def evaluate(self, theta, target, gradient=True):
        state = np.eye(self.dim, self.system_dim, dtype=complex)
        matrices = []
        for gate in self.gates:
            if gate.kind == "cx":
                state = state[self.permutations[gate.wires]]
                matrices.append(None)
            else:
                md = self.matrix(gate.kind, theta[list(gate.params)])
                state = self.apply(md[0], gate.wires[0], state)
                matrices.append(md)
        diff = state[:self.system_dim] - target
        error = float(np.linalg.norm(diff))
        if not gradient:
            return error, state[:self.system_dim]
        adjoint = np.zeros_like(state)
        adjoint[:self.system_dim] = diff / max(error, 1e-30)
        grad = np.zeros(self.num_params)
        for gate, md in zip(self.gates[::-1], matrices[::-1]):
            if gate.kind == "cx":
                state = state[self.permutations[gate.wires]]
                adjoint = adjoint[self.permutations[gate.wires]]
            else:
                m, derivatives = md
                before = self.apply(m.conj().T, gate.wires[0], state)
                for p, derivative in zip(gate.params, derivatives):
                    grad[p] = float(np.vdot(adjoint, self.apply(derivative, gate.wires[0], before)).real)
                adjoint = self.apply(m.conj().T, gate.wires[0], adjoint)
                state = before
        return error, grad


def check_adjoint():
    checks = []
    rng = np.random.default_rng(20261014)
    for real in (False, True):
        reference = Ansatz(2, 4, real)
        bounded = ReversibleAdjointAnsatz(2, 4, real)
        theta = rng.normal(size=reference.num_params)
        target = rng.normal(size=(4, 4)) + (0 if real else 1j * rng.normal(size=(4, 4)))
        loss, grad = reference.evaluate(theta, target)
        other, other_grad = bounded.evaluate(theta, target)
        matrix = bounded.evaluate(theta, target, False)[1]
        circuit_error = float(np.max(np.abs(matrix - Operator(bounded.circuit(theta)).data[:4, :4])))
        gradient_error = float(np.max(np.abs(grad - other_grad)))
        assert abs(loss - other) < 1e-13 and gradient_error < 1e-12 and circuit_error < 1e-13
        checks.append(dict(real=real, loss_difference=abs(loss - other),
                           gradient_difference=gradient_error, circuit_error=circuit_error))
    return checks


def fit_full(ms, directory, budget, trial_budget, reuse_fit=False):
    a = np.asarray(ms.eff_op().to_matrix(), complex)
    alpha = float(ms.pauli_norm())
    target = a / alpha
    saved_path = directory / "fit_results.json"
    if reuse_fit and saved_path.exists():
        saved = read(saved_path)
        assert saved["n"] == ms.size and saved["alpha"] == alpha
        assert saved["budget_s"] == budget and saved["per_trial_budget_s"] == trial_budget
        assert digest(directory / "fit_target.npy") == saved["target_file_sha256"]
        assert np.array_equal(np.load(directory / "fit_target.npy"), target)
        qc = Ansatz(ms.size, saved["layers"], saved["real_ansatz"]).circuit(np.array(saved["parameters"]))
        error = float(np.linalg.norm(Operator(qc).data[:len(a), :len(a)] - target))
        assert abs(error - saved["fit_error"]) < 1e-12
        return qc, dict(saved, reused_fit_checkpoint=True), target
    np.save(directory / "fit_target.npy", target)
    real = bool(np.max(np.abs(target.imag)) < 1e-14)
    started = time.perf_counter()
    best = {"error": None}
    trials = []
    evaluations = 0
    for depth in (1, 2, 4, 8, 16, 32):
        ansatz = ReversibleAdjointAnsatz(ms.size, depth, real)
        for seed in (0, 1):
            if time.perf_counter() - started >= budget:
                break
            trial_started = time.perf_counter()
            local = {"error": None, "evaluations": 0}
            initial = np.random.default_rng(20261012 + 1000 * depth + seed).uniform(
                -np.pi, np.pi, ansatz.num_params)

            def objective(theta):
                nonlocal evaluations
                elapsed = time.perf_counter() - started
                if elapsed >= budget:
                    raise StopSearch("fit_time_budget")
                if time.perf_counter() - trial_started >= trial_budget:
                    raise StopSearch("trial_time_budget")
                value, gradient = ansatz.evaluate(theta, target)
                evaluations += 1
                local["evaluations"] += 1
                if local["error"] is None or value < local["error"]:
                    local["error"] = value
                if best["error"] is None or value < best["error"]:
                    best.update(error=value, layers=depth, seed=seed, parameters=theta.tolist())
                    dump(directory / "fit_best.json", {**best, "elapsed_s": elapsed, "evaluations": evaluations})
                if value <= 1e-11:
                    raise StopSearch("error_threshold_met")
                return value, gradient

            try:
                opt = minimize(objective, initial, method="BFGS", jac=True,
                               options={"gtol": 1e-8, "maxiter": 1500})
                stop, nit = str(opt.message), int(opt.nit)
            except StopSearch as exc:
                stop, nit = str(exc), None
            trial = dict(layers=depth, seed=seed, params=ansatz.num_params, best_error=local["error"],
                         evaluations=local["evaluations"], iterations=nit, stop=stop,
                         seconds=time.perf_counter() - trial_started)
            trials.append(trial)
            dump(directory / "fit_progress.json", dict(elapsed_s=time.perf_counter() - started,
                 best_error=best["error"], best_layers=best.get("layers"), trials=trials))
            print("FIT_TRIAL", json.dumps(trial), flush=True)
            if best["error"] is not None and best["error"] <= TOL:
                break
        if (best["error"] is not None and best["error"] <= TOL) or time.perf_counter() - started >= budget:
            break
    assert best["error"] is not None
    elapsed = time.perf_counter() - started
    ansatz = ReversibleAdjointAnsatz(ms.size, best["layers"], real)
    qc = ansatz.circuit(np.array(best["parameters"]))
    error = float(np.linalg.norm(Operator(qc).data[:len(a), :len(a)] - target))
    assert abs(error - best["error"]) < 1e-10
    qualified = error < TOL
    record = dict(**best, fit_error=error, n=ms.size, alpha=alpha, real_ansatz=real,
        status="success" if qualified else "fit_budget_exhausted" if elapsed >= budget else "fit_search_exhausted",
        tolerance_frobenius=TOL, fitting_target=1e-11, budget_s=budget, per_trial_budget_s=trial_budget,
        total_search_s=elapsed, newly_measured_fit_s=elapsed, trials=trials, evaluations=evaluations,
        template="TFIM8_raw_0_full_fixed_alpha", cached=False,
        gradient="exact reversible adjoint; checked against stored-forward adjoint on small complex/real ansatz",
        matrix_scope="512 by 256 unitary input-column block; no complete-channel matrix",
        target_file_sha256=digest(directory / "fit_target.npy"))
    dump(directory / "fit_results.json", record)
    save_circuit(directory / "fit_best_source.qpy", qc)
    return qc, record, target


def append_permutation(qc, destination):
    slots = list(range(len(destination)))
    for pos, token in enumerate(np.argsort(destination).tolist()):
        other = slots.index(token)
        if other != pos:
            qc.swap(pos, other)
            slots[pos], slots[other] = slots[other], slots[pos]


def materialize_layout(qc):
    if qc.layout is None:
        return qc
    out = QuantumCircuit(qc.num_qubits)
    append_permutation(out, qc.layout.initial_index_layout())
    out.compose(qc, inplace=True)
    append_permutation(out, np.argsort(qc.layout.final_index_layout()).tolist())
    out.metadata = deepcopy(qc.metadata)
    error = float(np.max(np.abs(Operator(out).data - Operator.from_circuit(qc).data)))
    assert error < TOL
    return out


class TemplateCache:
    def __init__(self):
        self.records = read(OLD / "vcbe_refined.json")["cases"]
        self.records.update(read(OLD / "vcbe_precision.json")["cases"])
        self.loaded = {}

    def local(self, target, system_n):
        right_gates = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]])}
        for name, record in self.records.items():
            if "local_fixed_alpha" not in name:
                continue
            path = OLD / "vcbe" / name
            template = np.load(path / "matrix.npy") / record["alpha"]
            for right, matrix in right_gates.items():
                if np.max(np.abs(template @ matrix - target)) >= 1e-12:
                    continue
                if name not in self.loaded:
                    assert record["status"] == "success"
                    original = read(path / "results.json")
                    ansatz = Ansatz(1, record["layers"], original["real_ansatz"])
                    theta = np.array(record["parameters"])
                    error = ansatz.evaluate(theta, template, False)[0]
                    before = error
                    started = time.perf_counter()
                    precision_s = 0.0
                    if error * np.sqrt(2 ** (system_n - 1)) >= TOL:
                        best = {"error": error, "theta": theta}
                        def evaluate(x):
                            residual, jac = residual_jacobian(ansatz, x, template)
                            loss = float(np.linalg.norm(residual))
                            if loss < best["error"]:
                                best.update(error=loss, theta=x.copy())
                            return residual, jac
                        least_squares(lambda x: evaluate(x)[0], theta, jac=lambda x: evaluate(x)[1],
                                      ftol=None, gtol=None, xtol=1e-15, max_nfev=30)
                        theta = best["theta"]
                        precision_s = time.perf_counter() - started
                    source = ansatz.circuit(theta)
                    error = float(np.linalg.norm(Operator(source).data[:2, :2] - template))
                    info = dict(template=name, cached=True, fit_error=error, cached_fit_error=before,
                        cached_unique_fit_s=record["total_search_s"], newly_measured_fit_s=precision_s,
                        precision_method="local analytic-Jacobian least squares" if precision_s else "none",
                        n=1, layers=record["layers"], real_ansatz=original["real_ansatz"],
                        parameters=theta.tolist(), template_matrix_sha256=digest(path / "matrix.npy"),
                        cached_results_sha256=digest(OLD / "vcbe_refined.json"),
                        precision_results_sha256=digest(OLD / "vcbe_precision.json"))
                    self.loaded[name] = source, info
                source, info = self.loaded[name]
                qc = QuantumCircuit(2)
                if right == "X":
                    qc.x(0)
                qc.compose(source, inplace=True)
                return qc, dict(info, right_pauli=right)
        raise ValueError("No normalized local VCBE template matches the target")


def blocks_for(kraus, directory, fit_budget, trial_budget, reuse_fit=False):
    cache = TemplateCache()
    blocks, details = [], []
    for j, ms in enumerate(kraus):
        n, alpha = ms.size, float(ms.pauli_norm())
        active = [q for q in range(n) if any(atom.expr[n - 1 - q] != "I" for atom, _ in ms.instances)]
        if len(active) == 1:
            q = active[0]
            # Strip spectator identities before constructing any matrix.
            local_terms = [[label[n - 1 - q], magnitude, real, imag]
                           for label, magnitude, real, imag in serialize_kraus([ms])[0]]
            local_ms = deserialize_kraus([local_terms])[0]
            target = np.asarray(local_ms.eff_op().to_matrix(), complex) / alpha
            qc, info = cache.local(target, n)
            wires = [q, n]
            spectator_factor = float(np.sqrt(2 ** (n - 1)))
            preprocessing = "exact spectator identity removal in Pauli coefficient space"
        else:
            assert n == 8 and j == 0
            qc, info, target = fit_full(ms, directory, fit_budget, trial_budget, reuse_fit)
            wires = list(range(n + 1))
            spectator_factor = 1.0
            preprocessing = "full eight-qubit Kraus target, fixed Pauli 1-norm"
        d = len(target)
        error = float(np.linalg.norm(Operator(qc).data[:d, :d] - target))
        detail = dict(info, index=j, alpha=alpha, active_wires=active, embedding_wires=wires,
                      preprocessing=preprocessing, active_frobenius_block_error=error,
                      full_embedded_frobenius_block_error=error * spectator_factor,
                      spectator_frobenius_factor=spectator_factor,
                      block_qualified=error * spectator_factor < TOL)
        blocks.append((qc, wires, target))
        details.append(detail)
        save_circuit(directory / f"vcbe_active_block_{j}.qpy", qc)
    return blocks, details


def compile_method(kraus, blocks, details, method, directory, lam):
    started = time.perf_counter()
    prepared = {}
    validations = []
    preoptimization_s = 0.0
    for j, (ms, (active_qc, wires, target), detail) in enumerate(zip(kraus, blocks, details)):
        block = active_qc.copy()
        if method == "vcbe_preopt3_tree":
            t = time.perf_counter()
            block = materialize_layout(transpile_channel_circuit(block, basis_gates=IBM,
                      optimization_level=3, seed_transpiler=SEED))
            preoptimization_s += time.perf_counter() - t
        unitary_error = float(np.linalg.norm(Operator(block).data - Operator(active_qc).data))
        d = len(target)
        error = float(np.linalg.norm(Operator(block).data[:d, :d] - target)) * detail["spectator_frobenius_factor"]
        validations.append(dict(index=j, active_width=block.num_qubits,
            preoptimization_unitary_frobenius_error=unitary_error,
            full_embedded_frobenius_block_error=error,
            qualified=error < TOL and unitary_error < TOL))
        embedded = QuantumCircuit(ms.size + 1)
        embedded.compose(block, qubits=wires, inplace=True)
        embedded.metadata = dict(system_qubits=list(range(ms.size)), workspace_qubits=[ms.size],
                                 alpha=float(ms.pauli_norm()), vcbe_source=detail["template"])
        prepared[input_identity([ms])] = embedded

    class Backend:
        def __init__(self, ms):
            self.block = prepared[input_identity([ms])]
        def circuit(self, opt=None):
            return self.block.copy()

    qc, registers = channel_to_LCU(channel_ensemble([deepcopy(kraus)]), structure="tree",
                                   opt="VCBE", block_encoding_cls=Backend, fstar=False)
    assembly = time.perf_counter() - started
    assert np.isclose(qc.metadata["subnormalization_factor"], lam, rtol=1e-12, atol=1e-14)
    assert qc.metadata["outer_control_scope"] == "whole-block"
    save_circuit(directory / f"{method}_source.qpy", qc)
    t = time.perf_counter()
    lowered = transpile_channel_circuit(qc, basis_gates=NAM, optimization_level=0, seed_transpiler=SEED)
    lowering = time.perf_counter() - t
    assert set(lowered.count_ops()) <= set(NAM)
    qasm = qasm2.dumps(lowered)
    qasm_path = directory / f"{method}_nam_opt0.qasm"
    qasm_path.write_text(qasm)
    save_circuit(directory / f"{method}_nam_opt0.qpy", lowered)
    t = time.perf_counter()
    count = count_qasm_text(qasm, gate_set="nam")
    recount_s = time.perf_counter() - t
    qualified = all(v["qualified"] for v in validations)
    metric = dict(G=count["metric_total"], W=lowered.num_qubits,
        CX=int(lowered.count_ops().get("cx", 0)), NC=count["clifford"]["non_clifford"],
        Lambda=lam, Lambda_G=lam * count["metric_total"], counts=dict(lowered.count_ops()),
        assembly_s=assembly, inner_preoptimization_s=preoptimization_s, lowering_s=lowering,
        assembly_plus_lowering_s=assembly + lowering, qasm_recount_s=recount_s,
        qasm_sha256=digest(qasm_path), qasm_path=str(qasm_path),
        block_qualified=qualified,
        full_choi_status="skipped_user_request", full_choi_relative_error=None,
        validation_scope="each active block and exact spectator embedding; no full-channel Choi or basis-column simulation",
        status="ok" if qualified else "diagnostic_unqualified_fit")
    print("CIRCUIT", directory.name, method, json.dumps(metric), flush=True)
    return dict(status=metric["status"], registers=registers, metadata=qc.metadata,
                assembly_s=assembly, block_validation=validations, metrics={"nam_opt0": metric})


def worker(args):
    directory = OUT / "cases" / args.case
    directory.mkdir(parents=True, exist_ok=True)
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and smoke["sources"] == source_identity()
    started = time.perf_counter()
    kraus, provenance = load_case(args.case)
    row = dict(case=args.case, status="started", n=kraus[0].size, M=len(kraus),
        kraus=serialize_kraus(kraus), input_sha256=input_identity(kraus), provenance=provenance,
        sources=source_identity(), alpha=[float(ms.pauli_norm()) for ms in kraus],
        Lambda=float(sum(ms.pauli_norm() ** 2 for ms in kraus)), methods={},
        full_choi_status="skipped_user_request", tolerance=TOL)
    dump(directory / "results.json", row)
    t = time.perf_counter()
    blocks, details = blocks_for(kraus, directory, args.fit_budget, args.trial_budget, args.reuse_fit)
    row.update(vcbe_blocks=details, block_preparation_s=time.perf_counter() - t, status="blocks_ready")
    unique = {b["template"]: b for b in details}
    row["cached_unique_fit_s"] = sum(b.get("cached_unique_fit_s", 0) for b in unique.values())
    row["newly_measured_unique_fit_s"] = sum(b["newly_measured_fit_s"] for b in unique.values())
    dump(directory / "results.json", row)
    for method in METHODS:
        row["methods"][method] = compile_method(kraus, blocks, details, method, directory, row["Lambda"])
        dump(directory / "results.json", row)
    row["status"] = "success_skipped_full_choi" if all(b["block_qualified"] for b in details) else next(
        (b["status"] for b in details if not b["block_qualified"] and not b["cached"]), "block_tolerance_not_met")
    row["total_elapsed_s"] = time.perf_counter() - started
    row["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    assert row["sources"] == source_identity()
    row["production_sources_unchanged"] = True
    dump(directory / "results.json", row)
    print("DONE", args.case, row["status"], row["total_elapsed_s"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-full-choi", action="store_true", required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--fit-budget", type=float, default=600)
    parser.add_argument("--trial-budget", type=float, default=50)
    parser.add_argument("--rss-budget-gib", type=float, default=6)
    parser.add_argument("--reuse-fit", action="store_true", help="Reassemble from the saved, hash-checked fit checkpoint")
    parser.add_argument("--resume", action="store_true", help="Keep completed cases and recover unfinished postprocessing")
    args = parser.parse_args()
    if args.case:
        worker(args)
        return
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and smoke["sources"] == source_identity()
    record = dict(sources=source_identity(), runner_sha256=digest(__file__),
        vcbe="independent generic VCBE Fig.1/Eq.(7), arXiv:2507.17658v2; local template reuse plus bounded TFIM8 fitting",
        full_choi_status="skipped_user_request", tolerance=TOL,
        basis="NAM", final_optimization_level=0, seed_transpiler=SEED, qubits_initially_zero=False,
        fixed=["current Ours Kraus representation", "alpha_j=Pauli 1-norm", "Lambda", "outer PREPARE", "outer tree"],
        fit_budget_s=args.fit_budget, per_trial_budget_s=args.trial_budget,
        rss_budget_bytes=int(args.rss_budget_gib * 1024 ** 3),
        fitting_recipe=dict(layers=[1, 2, 4, 8, 16, 32], seeds=[0, 1], method="BFGS", gtol=1e-8, maxiter=1500),
        gradient_checks=check_adjoint(),
        environment=dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                         physical_memory_bytes=psutil.virtual_memory().total), cases={})
    if args.resume:
        previous = read(OUT / "results.json")
        assert previous["sources"] == record["sources"]
        for key in ("fit_budget_s", "per_trial_budget_s", "rss_budget_bytes", "fitting_recipe"):
            assert previous[key] == record[key]
        record = previous
        record["postprocess_runner_sha256"] = digest(__file__)
    for package in ("qiskit", "numpy", "scipy"):
        from importlib.metadata import version
        record["environment"][package] = version(package)
    dump(OUT / "results.json", record)
    for case in CASES:
        if args.resume and record["cases"].get(case, {}).get("status") == "success_skipped_full_choi":
            continue
        prior = record["cases"].get(case, {}) if args.resume else {}
        directory = OUT / "cases" / case
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ("run.log", "results.json"):
            previous = directory / filename
            if previous.exists():
                attempt = 1
                while previous.with_name(f"setup_attempt_{attempt:02d}_{filename}").exists():
                    attempt += 1
                previous.rename(previous.with_name(f"setup_attempt_{attempt:02d}_{filename}"))
        command = [sys.executable, str(Path(__file__).resolve()), "--skip-full-choi", "--case", case,
                   "--fit-budget", str(args.fit_budget), "--trial-budget", str(args.trial_budget)]
        if args.resume:
            command.append("--reuse-fit")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MPLCONFIGDIR": str(OUT / "matplotlib_cache")}
        started = time.perf_counter()
        hard_budget = args.fit_budget + 300 if case == "TFIM-8" else 300
        reason, peak, rss_monitor_error = None, 0, None
        with (directory / "run.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            process = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    rss = process.memory_info().rss
                    peak = max(peak, rss)
                except (psutil.Error, OSError) as exc:
                    rss_monitor_error = str(exc)
                if peak > record["rss_budget_bytes"]:
                    reason = "runner_rss_budget_exhausted"
                if time.perf_counter() - started > hard_budget:
                    reason = "runner_wall_time_budget_exhausted"
                if reason:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(0.25)
            code = proc.wait()
        path = directory / "results.json"
        row = read(path) if path.exists() else dict(case=case, methods={})
        row["supervisor"] = dict(exit_code=code, termination_reason=reason, sampled_peak_rss_bytes=peak,
            hard_wall_budget_s=hard_budget, rss_budget_bytes=record["rss_budget_bytes"],
            rss_monitor_scope="single worker process; no child processes used", rss_monitor_error=rss_monitor_error,
            elapsed_s=time.perf_counter() - started)
        if prior:
            row["prior_supervisor_attempts"] = prior.get("prior_supervisor_attempts", []) + [prior["supervisor"]]
            row["postprocessing_recovery"] = "hash-checked saved fit; fitting not rerun; inner unitary deviations retained as diagnostic qualification flags"
            row["total_attempt_elapsed_s"] = sum(x["elapsed_s"] for x in row["prior_supervisor_attempts"]) + row["supervisor"]["elapsed_s"]
            row["peak_rss_bytes"] = max(prior.get("peak_rss_bytes", 0), prior["supervisor"]["sampled_peak_rss_bytes"], row.get("peak_rss_bytes", 0))
        if reason or code != 0:
            row["status"] = reason or "worker_failed"
        dump(path, row)
        record["cases"][case] = row
        dump(OUT / "results.json", record)
        print("CASE_RESULT", case, row["status"], row["supervisor"], flush=True)
    assert record["sources"] == source_identity()
    record["production_sources_unchanged"] = True
    dump(OUT / "results.json", record)


if __name__ == "__main__":
    main()
