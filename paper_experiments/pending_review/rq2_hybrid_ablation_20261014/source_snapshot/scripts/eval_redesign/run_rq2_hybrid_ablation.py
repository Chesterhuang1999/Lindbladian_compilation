#!/usr/bin/env python3
"""Freeze shared channel inputs and run the eight R/O/F* NAM ablations.

Prepare locally, transfer the directory, then launch this script from its
source_snapshot. Workers are isolated and resource limits retain per-job
failure/timeout states. This runner never dispatches circuit optimizers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("src", "scripts", "Baseline_scripts")]
CONFIG_PATH = Path("paper_experiments/configs/rq2_hybrid_ablation_20261014.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def configurations():
    from experiment_protocol import CONFIGS
    labels = {
        (False, False, False): ("Basic", "Basic"),
        (True, False, False): ("R", "R"),
        (False, False, True): ("O", "O"),
        (False, True, False): ("Fstar", "F*"),
        (True, False, True): ("R_O", "R+O"),
        (True, True, False): ("R_Fstar", "R+F*"),
        (False, True, True): ("O_Fstar", "O+F*"),
        (True, True, True): ("All", "All"),
    }
    result = {}
    for config in CONFIGS:
        slug, label = labels[(config.rewrite, config.flat, config.order)]
        result[slug] = {**asdict(config), "slug": slug, "label": label,
                        "fstar": config.flat,
                        "rewrite_context": config.structure if config.rewrite else None}
        assert config.structure == ("tree" if config.flat else "basic")
        assert config.opt == ("Matrix-order" if config.order else "No")
    assert len(result) == 8
    return result


def expand_cases(groups):
    result = []
    for original in groups:
        seeds = original.get("parameters", {}).get("seeds")
        if seeds is None:
            result.append(original)
            continue
        for seed in seeds:
            case = json.loads(json.dumps(original))
            case["group_id"] = original["id"]
            case["id"] += f"-seed{seed}"
            case["parameters"].pop("seeds")
            case["parameters"]["seed"] = seed
            result.append(case)
    return result


def matrix_rows(matrix, labels, tol):
    import numpy as np
    from channel_IR import Matrixsum, PauliAtom
    return [Matrixsum([(PauliAtom(str(label), phase=value / abs(value)), float(abs(value)))
                       for label, value in zip(labels, row) if abs(value) > tol])
            for row in matrix if np.any(np.abs(row) > tol)]


def prepare(out, config):
    import numpy as np
    from channel_IR import channel as ChannelIR
    from rq1_pipeline import canonicalize, serialize_kraus
    from run_selected_tfim_hrw import sparse_gram_check
    from selected_channel_inputs import build_selected_case

    if (out / "manifest.json").exists():
        raise ValueError("Choose a fresh output directory.")
    snapshot = out / "source_snapshot"
    identities = {}
    for folder in ("src", "scripts", "Baseline_scripts"):
        for source in sorted((ROOT / folder).rglob("*.py")):
            relative = source.relative_to(ROOT)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            identities[str(relative)] = file_hash(destination)
    dump(snapshot / CONFIG_PATH, config)
    identities[str(CONFIG_PATH)] = file_hash(snapshot / CONFIG_PATH)
    cases = expand_cases(config["case_groups"])
    for case in cases:
        if "fixture" in case:
            relative = Path(case["fixture"])
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
            identities[str(relative)] = file_hash(target)
        directory = out / "cases" / case["id"]
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        raw, model = build_selected_case(case)
        build_s = time.perf_counter() - started
        canonical, canonicalization = canonicalize(raw)
        raw_rows, canonical_rows = serialize_kraus(raw), serialize_kraus(canonical)
        check = sparse_gram_check(raw_rows, canonical_rows)
        if not check["passed"]:
            raise AssertionError(check)
        matrix, labels = ChannelIR(canonical)._coeff_matrix(tol=0.0)
        tol = config["rewrite"]["tol"]
        cleanup = matrix.copy()
        cleanup[np.abs(cleanup) <= tol] = 0
        gram = matrix.conj().T @ matrix
        cleanup_error = float(np.linalg.norm(cleanup.conj().T @ cleanup - gram)
                              / max(np.linalg.norm(gram), 1e-30))
        payload = {
            "case": case, "model": model, "raw_kraus": raw_rows,
            "canonical_kraus": canonical_rows, "canonicalization": canonicalization,
            "canonicalization_verification": check, "input_construction_s": build_s,
            "matrix_sha256": sha256(matrix.tobytes()).hexdigest(),
            "matrix_shape": list(matrix.shape),
            "term_counts_raw": [len(row) for row in raw_rows],
            "term_counts_canonical": [len(row) for row in canonical_rows],
            "numerical_cleanup_audit": {
                "tol": tol, "canonical_zero_policy": "exact",
                "terms_at_or_below_tol": int(np.count_nonzero(matrix) - np.count_nonzero(cleanup)),
                "cleanup_gram_relative_error": cleanup_error,
                "applied_to_R_off_input": False,
                "note": "R reconstruction removes terms at or below tol; this numerical cleanup is reported separately from rotations."
            },
        }
        dump(directory / "input.json", payload)
        np.savez_compressed(directory / "input.npz", A=matrix, labels=np.asarray(labels))
        case["input_json_sha256"] = file_hash(directory / "input.json")
        case["input_npz_sha256"] = file_hash(directory / "input.npz")
        case["canonical_input_sha256"] = canonicalization["canonical_input_sha256"]
    manifest = {
        "created_at_utc": now(), "status": "prepared", "research_question": "RQ2",
        "config": config, "cases": cases, "configurations": configurations(),
        "source_snapshot": "source_snapshot", "snapshot_sha256": identities,
        "input_instances": len(cases), "configuration_runs": len(cases) * 8,
        "rewrite_contexts": ["basic", "tree"],
        "support_search_sharing": "One identical support beam; each lowering context observes committed frontiers and polishes its own seeds.",
        "timing_scope": "Fresh process; input construction, imports, JIT warmup, verification and export excluded from compilation time. Shared canonicalization and per-context rewrite times are charged to each applicable configuration.",
    }
    dump(out / "manifest.json", manifest)
    dump(out / "status.json", {"status": "prepared", "updated_at_utc": now(),
                              "configuration_runs": len(cases) * 8})


def worker(out, case, slug, manifest):
    directory = out / "cases" / case["id"]
    job_dir = directory / ("rewrites" if slug == "rewrite" else "configs/" + slug)
    job_dir.mkdir(parents=True, exist_ok=True)

    def progress(stage, **extra):
        data = {"case": case["id"], "config": slug, "stage": stage,
                "updated_at_utc": now(), "stage_started_monotonic": time.monotonic(),
                "pid": os.getpid(), **extra}
        dump(job_dir / "progress.json", data)
        print(json.dumps(data), flush=True)

    progress("importing")
    import numpy as np
    from channel_IR import channel as ChannelIR, channel_ensemble
    from channel_rewrite_cost import CostSettings
    from hybrid_rewrite import hybrid_search, metrics as representation_metrics, verify, warmup
    from rq1_pipeline import deserialize_kraus, input_identity, serialize_kraus, verify_channel
    assert file_hash(directory / "input.json") == case["input_json_sha256"]
    assert file_hash(directory / "input.npz") == case["input_npz_sha256"]
    input_data = json.loads((directory / "input.json").read_text())
    arrays = np.load(directory / "input.npz")
    initial, labels = arrays["A"], arrays["labels"]
    assert sha256(initial.tobytes()).hexdigest() == input_data["matrix_sha256"]
    config = manifest["config"]
    rewrite_options = config["rewrite"]
    tol = rewrite_options["tol"]
    if slug == "rewrite":
        progress("jit_warmup")
        jit_s = warmup()
        progress("rewrite", compilation_started_monotonic=time.monotonic())
        hybrid = hybrid_search(
            initial, out=job_dir / "search", width=rewrite_options["beam_width"], tol=tol,
            timeout=rewrite_options["timeout_per_search_s"],
            max_stagnant=rewrite_options["max_stagnant_support_layers"])
        records = {}
        for context, value in hybrid["contexts"].items():
            state = value["state"]
            final = state.A.copy()
            final[np.abs(final) <= tol] = 0
            check, replay = verify(initial, final, state.steps, tol)
            if not check["passed"]:
                raise AssertionError(check)
            rows = matrix_rows(final, labels, tol)
            identity_parameters = {
                "canonical_input_sha256": case["canonical_input_sha256"],
                "method": rewrite_options["method"], "settings": rewrite_options,
                "context": {"structure": context, "fstar": context != "basic"},
                "sources": manifest["snapshot_sha256"],
            }
            record = {
                **value["record"], "method": rewrite_options["method"],
                "cache_identity": sha256(json.dumps(identity_parameters, sort_keys=True).encode()).hexdigest(),
                "canonical_input_sha256": case["canonical_input_sha256"],
                "compiled_input_sha256": input_identity(rows),
                "compiled_kraus": serialize_kraus(rows), "verification": check,
                "numerical_cleanup": input_data["numerical_cleanup_audit"],
                "jit_warmup_s": jit_s,
                "initial_metrics": representation_metrics(initial, tol, CostSettings(context, context != "basic")),
            }
            np.savez_compressed(job_dir / (context + ".npz"), A_initial=initial, A_final=final,
                                A_replayed=replay, labels=labels)
            dump(job_dir / (context + "_steps.json"), [
                {"pair": [i, j], "U_real": unitary.real.tolist(), "U_imag": unitary.imag.tolist()}
                for i, j, unitary in state.steps])
            dump(job_dir / (context + ".json"), record)
            records[context] = {key: record[key] for key in (
                "status", "termination", "metrics", "rewrite_time_s", "verification", "cache_identity")}
        result = {"case": case["id"], "status": "ok" if all(r["status"] == "ok" for r in records.values()) else "rewrite_incomplete",
                  "contexts": records, "jit_warmup_s": jit_s}
        dump(job_dir / "results.json", result)
        progress(result["status"])
        return

    from block_encoding import BlockEncoding
    from experiment_protocol import lower_and_count
    from matrix_order_pipeline import MatrixOrderOptions
    from qiskit import qasm2
    from baseline_common import count_qasm_file
    cfg = manifest["configurations"][slug]
    rewrite = None
    if cfg["rewrite"]:
        rewrite_path = directory / "rewrites" / (cfg["rewrite_context"] + ".json")
        rewrite = json.loads(rewrite_path.read_text())
        assert rewrite["canonical_input_sha256"] == case["canonical_input_sha256"]
        kraus = deserialize_kraus(rewrite["compiled_kraus"])
        assert input_identity(kraus) == rewrite["compiled_input_sha256"]
    else:
        kraus = deserialize_kraus(input_data["canonical_kraus"])
        assert input_identity(kraus) == case["canonical_input_sha256"]
    matrix, _ = ChannelIR(kraus)._coeff_matrix(tol=0.0)
    static_metrics = representation_metrics(matrix, tol, CostSettings(cfg["structure"], cfg["fstar"]))
    started_clock = time.monotonic()
    progress("assembly", compilation_started_monotonic=started_clock)
    started = time.perf_counter()
    backend = lambda ms: BlockEncoding(ms, placement_options={}, matrix_order_options=MatrixOrderOptions())
    circuit, registers = __import__("channel_LCU").channel_to_LCU(
        channel_ensemble([kraus]), structure=cfg["structure"], opt=cfg["opt"],
        block_encoding_cls=backend, fstar=cfg["fstar"])
    assembly_s = time.perf_counter() - started
    for key, expected in (("structure", cfg["structure"]), ("block_opt", cfg["opt"]), ("fstar", cfg["fstar"])):
        assert circuit.metadata[key] == expected, (key, circuit.metadata[key], expected)
    assert circuit.metadata["controlled_prepare"] == (len(kraus) > 1 and not cfg["fstar"])
    lam = float(circuit.metadata["subnormalization_factor"])
    assert abs(lam - static_metrics["static_cost"]["Lambda"]) <= 1e-10 * max(lam, 1.0)
    progress("lowering", compilation_started_monotonic=started_clock, assembly_s=assembly_s)
    lowered, measured = lower_and_count(circuit, rq="RQ2", basis=config["lowering"]["basis"],
                                        seed_transpiler=config["lowering"]["seed_transpiler"])
    assembly_lowering_s = time.perf_counter() - started
    policy = dict(measured["settings"])
    policy["configuration"] = cfg
    policy["matrix_order_options"] = asdict(MatrixOrderOptions()) if cfg["order"] else None
    policy["rewrite"] = ({"method": "none", "time_s": 0.0} if rewrite is None else {
        "method": rewrite["method"], "settings": rewrite["settings"],
        "cache_identity": rewrite["cache_identity"], "status": rewrite["status"],
        "termination": rewrite["termination"], "time_s": rewrite["rewrite_time_s"],
        "record": "../../rewrites/" + cfg["rewrite_context"] + ".json"})
    rewrite_s = rewrite["rewrite_time_s"] if rewrite else 0.0
    canonical_s = input_data["canonicalization"]["time_s"]
    result = {
        "case": case["id"], "group_id": case.get("group_id", case["id"]),
        "config": slug, "label": cfg["label"], "status": "compiled",
        "G": measured["G"], "W": measured["W"], "Lambda": lam, "Lambda_G": lam * measured["G"],
        "counts": measured["counts"], "canonical_input_sha256": case["canonical_input_sha256"],
        "compiled_input_sha256": input_identity(kraus),
        "rewrite_verification": rewrite["verification"] if rewrite else {"passed": True, "method": "identical canonical input"},
        "representation": static_metrics, "canonicalization_s": canonical_s,
        "rewrite_s": rewrite_s, "assembly_s": assembly_s, "lowering_s": measured["lowering_time_s"],
        "assembly_lowering_s": assembly_lowering_s,
        "compile_time_s": canonical_s + rewrite_s + assembly_lowering_s,
        "settings": policy, "metadata": circuit.metadata, "registers": registers,
        "versions": measured["versions"], "sources": measured["sources"],
        "verification": {"status": "pending"}, "qasm": "channel_nam.qasm",
    }
    dump(job_dir / "compiled.json", result)
    progress("export", G=result["G"], W=result["W"])
    qasm_path = job_dir / "channel_nam.qasm"
    qasm2.dump(lowered, qasm_path)
    recount = count_qasm_file(qasm_path, gate_set="nam")
    assert recount["metric_total"] == result["G"] and recount["num_qubits"] == result["W"]
    assert recount["clifford"]["unknown"] == 0
    result.update(NC=recount["clifford"]["non_clifford"], qasm_recount=recount,
                  qasm_sha256=file_hash(qasm_path))
    result["Lambda_NC"] = lam * result["NC"]
    dump(job_dir / "compiled.json", result)
    resources = config["resources"]
    if case["system_qubits"] <= resources["choi_max_system_qubits"] and lowered.num_qubits <= resources["choi_max_total_qubits"]:
        progress("verification", G=result["G"])
        started = time.perf_counter()
        result["verification"] = verify_channel(lowered, registers, deserialize_kraus(input_data["canonical_kraus"]))
        result["verification_s"] = time.perf_counter() - started
        if not result["verification"]["passed"]:
            dump(job_dir / "failed_verification.json", result)
            raise AssertionError(result["verification"])
    else:
        result["verification"] = {"status": "not_run", "reason": "full Choi budget: <=4 system and <=20 total qubits",
                                  "rewrite_equivalence_passed": result["rewrite_verification"]["passed"]}
    result["status"] = "rewrite_incomplete" if rewrite and rewrite["status"] != "ok" else "ok"
    dump(job_dir / "results.json", result)
    progress(result["status"], G=result["G"], W=result["W"])


def job_directory(out, record):
    return out / "cases" / record["case"] / ("rewrites" if record["config"] == "rewrite" else "configs/" + record["config"])


def stop_child(child):
    import psutil
    try:
        process = psutil.Process(child.pid)
        for descendant in process.children(recursive=True):
            try:
                descendant.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
    except psutil.NoSuchProcess:
        pass
    child.wait()


def coordinator(out):
    import psutil
    manifest = json.loads((out / "manifest.json").read_text())
    snapshot = out / "source_snapshot"
    for name, expected in manifest["snapshot_sha256"].items():
        assert file_hash(snapshot / name) == expected, name
    if (out / "launch.json").exists():
        raise ValueError("This run has already been launched; choose a new run for retries.")
    dump(out / "launch.json", {"started_at_utc": now(), "coordinator_pid": os.getpid(),
                               "python": sys.executable, "out_dir": str(out), "host": __import__("socket").gethostname()})
    limits = manifest["config"]["resources"]
    cases = {case["id"]: case for case in manifest["cases"]}
    records = [{"case": case, "config": "rewrite", "status": "queued", "peak_rss_bytes": 0} for case in cases]
    records += [{"case": case, "config": slug, "status": "queued", "peak_rss_bytes": 0}
                for slug in manifest["config"]["configuration_order"] for case in cases]
    active = {}
    started = now()

    def save(state="running"):
        dump(out / "status.json", {"status": state, "started_at_utc": started, "updated_at_utc": now(),
                                  "coordinator_pid": os.getpid(), "jobs": records})
        rows = []
        for record in records:
            if record["config"] == "rewrite":
                continue
            path = job_directory(out, record) / "results.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            rows.append({**record, **{key: data[key] for key in (
                "G", "W", "Lambda", "Lambda_G", "NC", "Lambda_NC", "compile_time_s",
                "rewrite_s", "assembly_s", "lowering_s", "verification", "canonical_input_sha256") if key in data},
                "result": str(path.relative_to(out)) if path.exists() else None})
        for row in rows:
            basic = next((r for r in rows if r["case"] == row["case"] and r["config"] == "Basic"), {})
            for metric in ("G", "Lambda_G"):
                row[metric + "_over_Basic"] = row[metric] / basic[metric] if metric in row and basic.get(metric) else None
        dump(out / "results.json", {"status": state, "updated_at_utc": now(), "rows": rows})

    save()
    try:
        while any(record["status"] in ("queued", "running") for record in records):
            for record in records:
                if len(active) >= limits["jobs"]:
                    break
                if record["status"] != "queued":
                    continue
                cfg = manifest["configurations"].get(record["config"])
                if cfg and cfg["rewrite"]:
                    dependency = next(r for r in records if r["case"] == record["case"] and r["config"] == "rewrite")
                    if dependency["status"] in ("queued", "running"):
                        continue
                    if not (out / "cases" / record["case"] / "rewrites" / (cfg["rewrite_context"] + ".json")).exists():
                        record.update(status="blocked_rewrite_failure")
                        continue
                directory = job_directory(out, record)
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-B", str(snapshot / "scripts/eval_redesign/run_rq2_hybrid_ablation.py"),
                           "--out-dir", str(out), "--worker", record["case"], record["config"]]
                log = (directory / "run.log").open("w")
                child = subprocess.Popen(command, cwd=snapshot, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                record.update(status="running", pid=child.pid, started_at_utc=now())
                active[child.pid] = (child, record, log, time.monotonic())
                save()
            for pid, (child, record, log, worker_start) in list(active.items()):
                directory = job_directory(out, record)
                progress_path = directory / "progress.json"
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                record["stage"] = progress.get("stage", "importing")
                resource_stop = None
                if child.poll() is None:
                    try:
                        process = psutil.Process(pid)
                        rss = process.memory_info().rss
                        for descendant in process.children(recursive=True):
                            try:
                                rss += descendant.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                        record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss)
                        if rss > limits["memory_limit_gib_per_worker"] * 1024 ** 3:
                            resource_stop = "memory_limit"
                    except psutil.NoSuchProcess:
                        pass
                    stage = record["stage"]
                    elapsed = time.monotonic() - progress.get("stage_started_monotonic", worker_start)
                    if record["config"] == "rewrite":
                        if time.monotonic() - worker_start > limits["rewrite_worker_timeout_s"]:
                            resource_stop = resource_stop or "rewrite_worker_timeout"
                    elif stage in ("assembly", "lowering"):
                        if time.monotonic() - progress["compilation_started_monotonic"] > limits["compilation_timeout_s"]:
                            resource_stop = resource_stop or "compilation_timeout"
                    elif stage == "verification" and elapsed > limits["verification_timeout_s"]:
                        resource_stop = resource_stop or "verification_timeout"
                    elif stage == "export" and elapsed > limits["export_timeout_s"]:
                        resource_stop = resource_stop or "export_timeout"
                    elif stage == "importing" and elapsed > limits["import_timeout_s"]:
                        resource_stop = resource_stop or "import_timeout"
                    if resource_stop:
                        stop_child(child)
                if child.poll() is None:
                    continue
                log.close()
                record.update(exit_code=child.returncode, finished_at_utc=now(),
                              process_wall_s=time.monotonic() - worker_start, resource_stop=resource_stop)
                path = directory / "results.json"
                if child.returncode == 0 and path.exists():
                    record["status"] = json.loads(path.read_text())["status"]
                else:
                    record["status"] = resource_stop or "failed"
                    checkpoint = directory / "compiled.json"
                    if checkpoint.exists() and not (directory / "failed_verification.json").exists():
                        result = json.loads(checkpoint.read_text())
                        result.update(status=record["status"], verification={"status": "not_completed", "reason": record["status"]})
                        dump(path, result)
                dump(directory / "resources.json", {key: record.get(key) for key in (
                    "pid", "peak_rss_bytes", "process_wall_s", "resource_stop", "exit_code")})
                del active[pid]
                save()
            if active:
                time.sleep(2)
            save()
    finally:
        for child, record, log, _ in active.values():
            stop_child(child)
            log.close()
            record["status"] = "coordinator_interrupted"
    state = "complete" if all(r["status"] == "ok" for r in records) else "complete_with_issues"
    save(state)


def configure_environment(out):
    runtime = out / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "QISKIT_NUM_PROCS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    for name in ("matplotlib", "numba", "tmp"):
        (runtime / name).mkdir(exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(runtime / "matplotlib"),
                      NUMBA_CACHE_DIR=str(runtime / "numba"), TMPDIR=str(runtime / "tmp"))
    dependencies = runtime / "site-packages"
    if dependencies.exists():
        sys.path.insert(0, str(dependencies))
        # Fresh workers inherit the same isolated additions.
        os.environ["PYTHONPATH"] = str(dependencies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("CASE", "CONFIG"))
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    configure_environment(out)
    if args.prepare_only:
        prepare(out, json.loads((ROOT / CONFIG_PATH).read_text()))
    elif args.worker:
        manifest = json.loads((out / "manifest.json").read_text())
        case = next(c for c in manifest["cases"] if c["id"] == args.worker[0])
        try:
            worker(out, case, args.worker[1], manifest)
        except Exception:
            directory = out / "cases" / case["id"] / ("rewrites" if args.worker[1] == "rewrite" else "configs/" + args.worker[1])
            dump(directory / "failure.json", {"time_utc": now(), "traceback": traceback.format_exc()})
            raise
    elif args.launch:
        if (out / "launch.json").exists():
            raise ValueError("This run has already been launched.")
        script = out / "source_snapshot/scripts/eval_redesign/run_rq2_hybrid_ablation.py"
        with (out / "coordinator.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-B", str(script), "--out-dir", str(out)],
                                     cwd=out / "source_snapshot", stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"coordinator_pid": child.pid, "out_dir": str(out), "started_at_utc": now()}), flush=True)
    else:
        coordinator(out)


if __name__ == "__main__":
    main()
