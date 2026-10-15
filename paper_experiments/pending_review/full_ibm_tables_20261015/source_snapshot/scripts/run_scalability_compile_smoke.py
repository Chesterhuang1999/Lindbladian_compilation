#!/usr/bin/env python3
"""Compile scalability cases using each family's established RQ1 route.

Each case runs in a fresh process, without simulation or cached circuits.
The default cases are TFIM-20 and HRW-36; sizes can be extended explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASIS = {"ibm": ("u1", "u2", "u3", "cx", "reset"),
         "nam": ("h", "x", "rz", "cx", "reset")}
SOURCE_PATHS = [
    Path(__file__), ROOT / "scripts/scalability_qiskit_program.py",
    ROOT / "scripts/run_tfim4_channel_lcu_ours_nam.py",
    ROOT / "scripts/run_hypercube_n4_ours_nam.py",
    ROOT / "scripts/build_graph_cluster_state_channel_lcu.py",
    ROOT / "scripts/build_toom_2d_channel_lcu.py",
    *sorted((ROOT / "src").glob("*.py")),
    ROOT / "Baseline_scripts/baseline_common.py",
]


class CompilationTimeout(TimeoutError):
    pass


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_worker_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The group may still contain the validation child after its leader exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def interrupt_parent(_signum, _frame):
    raise KeyboardInterrupt


def peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run_case(args):
    import numpy as np
    import psutil
    import qiskit
    import qiskit_aer
    from qiskit import qasm2, qpy, transpile

    for path in (ROOT / "src", ROOT / "Baseline_scripts"):
        sys.path.insert(0, str(path))
    from baseline_common import count_qasm_file
    from qasm_export import export_openqasm2_quartz_input

    family, size = args.worker.split(":")
    if family == "gc_grid":
        grid_rows, grid_cols = map(int, size.split("x"))
        n = grid_rows * grid_cols
    else:
        n = int(size) ** 2 if family == "toom" else int(size)
    case_dir = args.out_root / f"{family}_{size}"
    case_dir.mkdir(parents=True, exist_ok=False)
    module_names = {
        "tfim": "run_tfim4_channel_lcu_ours_nam",
        "hrw": "run_hypercube_n4_ours_nam",
        "gc_grid": "build_graph_cluster_state_channel_lcu",
        "gc_random": "build_graph_cluster_state_channel_lcu",
        "toom": "build_toom_2d_channel_lcu",
    }
    mod = importlib.import_module(module_names[family])
    if family in ("gc_grid", "gc_random", "toom"):
        for key in tuple(os.environ):
            if key.startswith("CHANNEL_LCU_"):
                del os.environ[key]
        if family == "toom":
            os.environ["CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH"] = "1"
    rewrite_method = "gatecost_rewrite_search" if family == "toom" else "rewrite_search"
    rewrite_strategy = "beam" if family == "toom" else args.rewrite_strategy
    beam_width = 8 if family == "toom" else 3
    case_name = f"TOOM-L{size}" if family == "toom" else f"{family.upper()}-{size}"
    summary = {
        "case": case_name, "family": family, "n": n,
        "status": "running", "stage": "model", "simulation_status": "not_run_compile_only",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__, "numpy": np.__version__,
            "physical_memory_bytes": psutil.virtual_memory().total,
            "thread_environment": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "QISKIT_PARALLEL", "RAYON_NUM_THREADS")},
            "compiler_environment": {k: v for k, v in os.environ.items() if k.startswith("CHANNEL_LCU_")},
        },
        "configuration": {
            "scope": "full_channel", "rewrite_method": rewrite_method,
            "rewrite_strategy": rewrite_strategy, "beam_width": beam_width,
            "max_steps": args.max_steps, "tol": 1e-12,
            "rewrite_progress": args.rewrite_progress and family in ("tfim", "hrw"),
            "case_wall_limit_s": args.timeout_s,
            "parent_watchdog_s": args.timeout_s + 120,
            "program_check_limit_s": args.program_check_timeout_s,
            "outer_select": "mulplex_U_opt", "block_encoding": "block_encoding_new",
            "block_opt": "Matrix-order", "optimization_level": 0,
            "seed_transpiler": 7, "basis": BASIS,
            "reference_runner": "Baseline_scripts/rerun_table1_ours_matrixsum_phase_refresh.py",
            "reused_cached_inputs": False,
        },
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE_PATHS
        },
        "timing_definitions": {
            "compile_time_s": "rewrite + circuit build + basis lowering; matches historical RQ1",
            "frontend_inclusive_compile_time_s": "model construction + compile_time_s",
            "program_generation_time_s": "model + rewrite + circuit build + IBM lowering + QPY/program export",
            "case_wall_time_s": "model start through all exports/counts/validation; excludes imports/provenance",
            "case_wall_limit_s": "wall limit for model through exports/counts/validation; excludes imports/provenance",
            "basis_export_time_s": "QASM serialization only; excluded from compile_time_s",
        },
    }
    if family == "toom":
        summary["configuration"].update(
            keep_neutral=False, max_neutral_candidates=3, skip_disjoint_pairs=True,
        )
    summary_path = case_dir / "summary.json"
    write_json(summary_path, summary)
    started = time.perf_counter()
    stage_started = started

    def check_deadline():
        if time.perf_counter() - started >= args.timeout_s:
            timeout_handler(None, None)

    def stage(name):
        nonlocal stage_started
        check_deadline()
        stage_started = time.perf_counter()
        summary["stage"] = name
        summary["stage_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, summary)
        print(f"[{summary['case']}] {name}; peak RSS {peak_rss_bytes() / 2**20:.1f} MiB", flush=True)

    def timeout_handler(_signum, _frame):
        raise CompilationTimeout(
            f"{summary['case']} exceeded {args.timeout_s:g}s case wall limit during {summary['stage']}"
        )

    previous_alarm_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        launch = json.loads((args.out_root / "source_manifest.json").read_text())
        summary["source_hashes_match_launch"] = summary["source_sha256"] == launch["source_sha256"]
        if not summary["source_hashes_match_launch"]:
            raise RuntimeError("Compiler sources changed after the batch snapshot")
        # Repeat in case a dependency swallows an exception in a GC callback.
        signal.setitimer(signal.ITIMER_REAL, args.timeout_s, 1.0)
        model_started = time.perf_counter()
        if family == "tfim":
            kraus_ops, model = mod.build_periodic_tfim_channel(n, args.delta_t)
            model.update({"boundary": "periodic", "J": 1.0, "h": 1.0, "decay_rate": 0.1})
            model["jump_pauli_coefficient"] = model["gamma"]
        elif family == "hrw":
            kraus_ops = mod.build_hypercube_channel(n)
            model = {"generator": "build_hypercube_section52_channel", "n": n}
        elif family in ("gc_grid", "gc_random"):
            spec = (
                mod.grid_2d_cluster_graph(grid_rows, grid_cols)
                if family == "gc_grid"
                else mod.random_bounded_degree_graph(n, args.random_degree, args.random_seed)
            )
            kraus_ops, model = mod.build_channel_for_spec(spec, 0.1, args.delta_t)
            if family == "gc_random":
                model.update(random_degree=args.random_degree, random_seed=args.random_seed)
        else:
            kraus_ops, model = mod.build_toom_channel(int(size), 0.1, args.delta_t)
        summary["model_time_s"] = time.perf_counter() - model_started
        summary["model"] = model
        summary["raw_channel_metrics"] = mod.channel_metrics(kraus_ops)

        stage("rewrite")
        rewrite_args = dict(strategy=rewrite_strategy, beam_width=beam_width,
                            max_steps=args.max_steps, tol=1e-12)
        if family in ("tfim", "hrw"):
            rewrite_args["verbose"] = args.rewrite_progress
        if family == "toom":
            rewritten, result, rewrite_time = mod.gatecost_rewrite_channel(
                kraus_ops, **rewrite_args, keep_neutral=False,
                max_neutral_candidates=3, skip_disjoint_pairs=True, verbose=False,
            )
        else:
            rewritten, result, rewrite_time = mod.rewrite_channel(kraus_ops, **rewrite_args)
        summary["rewrite_time_s"] = rewrite_time
        summary["rewrite"] = {
            "initial_support": result.get("initial_support"),
            "final_support": result.get("final_support"),
            "steps": len(result.get("steps", [])),
            "termination": result.get("termination"),
        }
        stage("export_rewrite")
        write_json(case_dir / "rewrite_result.json", result)
        if rewrite_strategy == "greedy_incremental":
            np.savez_compressed(
                case_dir / "rewrite_arrays.npz", A_final=result["A_final"],
                labels=np.asarray(result["labels"]),
                pairs=np.asarray([entry[0] for entry in result["steps"]], dtype=np.int64).reshape(-1, 2),
                unitaries=np.asarray([entry[1] for entry in result["steps"]], dtype=complex).reshape(-1, 2, 2),
                support_trajectory=np.asarray(result["termination"]["support_trajectory"], dtype=np.int64),
            )
        summary["rewritten_channel_metrics"] = mod.channel_metrics(rewritten)
        summary["subnormalization"] = float(sum(ms.pauli_norm() ** 2 for ms in rewritten))

        stage("circuit_build")
        qc, qubit_indexes, build_time = mod.build_rewrite_opt_matrix_order_circuit(rewritten)
        summary["circuit_build_time_s"] = build_time
        summary["qubit_indexes"] = dict(zip(("select", "workspace", "block", "system"), qubit_indexes))
        summary["logical_compile_time_s"] = summary["model_time_s"] + rewrite_time + build_time
        summary["width"] = qc.num_qubits
        summary["ancilla_qubits"] = qc.num_qubits - n
        summary["statevector_complex128_bytes"] = 16 * (1 << qc.num_qubits)
        summary["basis_results"] = {}

        for basis_name, basis_gates in BASIS.items():
            stage(f"lower_{basis_name}")
            lower_started = time.perf_counter()
            lowered = transpile(qc, basis_gates=list(basis_gates),
                                optimization_level=0, seed_transpiler=7)
            lowering_time = time.perf_counter() - lower_started
            unsupported = set(lowered.count_ops()) - set(basis_gates)
            if unsupported:
                raise ValueError(f"Unsupported gates after lowering: {unsupported}")
            compile_time = rewrite_time + build_time + lowering_time
            summary["basis_results"][basis_name] = {
                "compile_status": "complete", "compile_time_s": compile_time,
                "frontend_inclusive_compile_time_s": summary["model_time_s"] + compile_time,
                "lowering_time_s": lowering_time,
            }
            write_json(summary_path, summary)

            if basis_name == "ibm":
                stage("export_qpy_program")
                export_started = time.perf_counter()
                with (case_dir / "circuit_ibm.qpy").open("wb") as output:
                    qpy.dump(lowered, output)
                shutil.copyfile(ROOT / "scripts/scalability_qiskit_program.py", case_dir / "qiskit_program.py")
                summary["program_export_time_s"] = time.perf_counter() - export_started
                summary["program_generation_time_s"] = (
                    summary["logical_compile_time_s"] + lowering_time + summary["program_export_time_s"]
                )
                stage("validate_qpy")
                with (case_dir / "circuit_ibm.qpy").open("rb") as input_file:
                    restored = qpy.load(input_file)[0]
                if restored != lowered:
                    raise ValueError("QPY round-trip changed the circuit")
                summary["qpy_roundtrip_equal"] = True
                del restored

            qasm_path = case_dir / f"circuit_{basis_name}_opt0.qasm"
            stage(f"export_{basis_name}")
            export_started = time.perf_counter()
            export_meta = export_openqasm2_quartz_input(lowered, qasm_path, basis_gates=basis_gates)
            export_time = time.perf_counter() - export_started
            summary["basis_results"][basis_name].update(basis_export_time_s=export_time, qasm=str(qasm_path))
            stage(f"count_{basis_name}")
            count_started = time.perf_counter()
            stats = count_qasm_file(qasm_path, gate_set=basis_name)
            count_time = time.perf_counter() - count_started
            summary["basis_results"][basis_name].update(count_time_s=count_time, stats=stats, export=export_meta)
            stage(f"validate_qasm_{basis_name}")
            loaded = qasm2.load(qasm_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            if loaded.num_qubits != lowered.num_qubits or loaded.count_ops() != lowered.count_ops():
                raise ValueError("QASM round-trip changed width or gate counts")
            summary["basis_results"][basis_name]["qasm_roundtrip_width_ops_equal"] = True
            del loaded, lowered

        stage("validate_program")
        check = subprocess.run([sys.executable, str(case_dir / "qiskit_program.py"), "--check"],
                               capture_output=True, text=True, timeout=args.program_check_timeout_s)
        (case_dir / "program_check.log").write_text(check.stdout + check.stderr, encoding="utf-8")
        summary["program_check_returncode"] = check.returncode
        if check.returncode:
            raise RuntimeError(f"Generated program check failed: {check.stderr}")
        check_deadline()
        summary["status"] = "ok"
        summary["stage"] = "complete"
    except Exception as exc:
        summary["status"] = (
            "timeout" if isinstance(exc, CompilationTimeout)
            else "validation_timeout" if isinstance(exc, subprocess.TimeoutExpired)
            else "memory_error" if isinstance(exc, MemoryError) else "error"
        )
        summary["failed_stage_elapsed_s"] = time.perf_counter() - stage_started
        if isinstance(exc, CompilationTimeout):
            summary["timeout_source"] = "case_wall_alarm"
        elif isinstance(exc, subprocess.TimeoutExpired):
            summary["timeout_source"] = "program_check"
            output = exc.stdout or b""
            errors = exc.stderr or b""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            if isinstance(errors, bytes):
                errors = errors.decode("utf-8", errors="replace")
            (case_dir / "program_check.log").write_text(output + errors, encoding="utf-8")
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        summary["case_wall_time_s"] = time.perf_counter() - started
        summary["peak_rss_bytes"] = peak_rss_bytes()
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, summary)
    print(json.dumps({key: summary.get(key) for key in (
        "case", "status", "width", "logical_compile_time_s", "program_generation_time_s", "peak_rss_bytes")}), flush=True)
    return 0 if summary["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfim-sizes", nargs="*", type=int, default=[20])
    parser.add_argument("--hrw-sizes", nargs="*", type=int, default=[36])
    parser.add_argument("--gc-grids", nargs="*", default=[], metavar="ROWSxCOLS")
    parser.add_argument("--gc-random-sizes", nargs="*", type=int, default=[])
    parser.add_argument("--toom-lattice-sizes", nargs="*", type=int, default=[])
    parser.add_argument("--random-degree", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--rewrite-strategy", choices=("greedy", "greedy_incremental"), default="greedy",
                        help="Support-search strategy; Toom retains its original gatecost beam route")
    parser.add_argument("--rewrite-progress", action="store_true")
    parser.add_argument("--program-check-timeout-s", type=float, default=60)
    parser.add_argument("--timeout-s", type=float, default=18000,
                        help="Per-case wall limit in seconds, excluding imports/provenance (default: 18000)")
    parser.add_argument("--out-root", type=Path, default=ROOT / "Baseline_results" / "scalability_compile_smoke_20260909")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    if args.max_steps <= 0 or args.delta_t <= 0 or any(n < 2 for n in args.tfim_sizes + args.hrw_sizes):
        parser.error("Sizes must be >= 2, and max steps / delta t must be positive")
    if not 0 < args.timeout_s < float("inf"):
        parser.error("--timeout-s must be a finite positive value")
    if not 0 < args.program_check_timeout_s < float("inf"):
        parser.error("--program-check-timeout-s must be finite and positive")
    try:
        grids = [tuple(map(int, shape.split("x"))) for shape in args.gc_grids]
        if any(len(shape) != 2 or min(shape) < 2 for shape in grids):
            raise ValueError
    except ValueError:
        parser.error("--gc-grids entries must have form ROWSxCOLS with dimensions >= 2")
    if any(n < 3 for n in args.gc_random_sizes) or args.random_degree < 2:
        parser.error("Random graphs require >= 3 vertices and degree cap >= 2")
    if any(size < 2 for size in args.toom_lattice_sizes):
        parser.error("Toom lattice sizes must be >= 2")
    if args.worker:
        return run_case(args)

    cases = [(family, n) for family, sizes in (("tfim", args.tfim_sizes), ("hrw", args.hrw_sizes)) for n in sizes]
    cases.extend(("gc_grid", f"{r}x{c}") for r, c in grids)
    cases.extend(("gc_random", n) for n in args.gc_random_sizes)
    cases.extend(("toom", size) for size in args.toom_lattice_sizes)
    if not cases or len(set(cases)) != len(cases):
        parser.error("Provide at least one case, with no duplicate sizes")
    args.out_root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.out_root / "compiler_sources.executed.tar.gz", "w:gz") as archive:
        for source in SOURCE_PATHS:
            archive.add(source, arcname=str(source.relative_to(ROOT)))
    write_json(args.out_root / "source_manifest.json", {
        "cases": cases,
        "rewrite_strategy": args.rewrite_strategy,
        "max_steps": args.max_steps,
        "timeout_s": args.timeout_s,
        "parent_watchdog_s": args.timeout_s + 120,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE_PATHS
        },
        "archive_sha256": hashlib.sha256(
            (args.out_root / "compiler_sources.executed.tar.gz").read_bytes()
        ).hexdigest(),
    })
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", QISKIT_PARALLEL="FALSE", RAYON_NUM_THREADS="1")
    rows = []
    failed = False
    for family, n in cases:
        log_path = args.out_root / f"{family}_{n}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", f"{family}:{n}",
                   "--out-root", str(args.out_root), "--max-steps", str(args.max_steps),
                   "--delta-t", str(args.delta_t), "--timeout-s", str(args.timeout_s),
                   "--random-degree", str(args.random_degree), "--random-seed", str(args.random_seed),
                   "--rewrite-strategy", args.rewrite_strategy,
                   "--program-check-timeout-s", str(args.program_check_timeout_s)]
        if args.rewrite_progress:
            command.append("--rewrite-progress")
        print(f"Compiling {family.upper()}-{n}; log: {log_path}", flush=True)
        process_started = time.perf_counter()
        watchdog_expired = False
        interrupted = None
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            previous_term_handler = signal.signal(signal.SIGTERM, interrupt_parent)
            try:
                process.wait(timeout=args.timeout_s + 120)
            except subprocess.TimeoutExpired:
                watchdog_expired = True
                stop_worker_group(process)
            except BaseException as exc:
                interrupted = exc
                stop_worker_group(process)
            finally:
                signal.signal(signal.SIGTERM, previous_term_handler)
            if process.returncode < 0:
                stop_worker_group(process)
        process_wall = time.perf_counter() - process_started
        summary_path = args.out_root / f"{family}_{n}" / "summary.json"
        try:
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            summary = {"status": "unreadable_summary", "summary_read_error": repr(exc)}
        if interrupted is not None:
            summary.update(status="interrupted", interruption=repr(interrupted))
            failed = True
        elif watchdog_expired:
            summary.update(status="timeout", timeout_source="parent_watchdog")
            failed = True
        elif process.returncode:
            if process.returncode < 0:
                summary["status"] = "killed"
            elif summary.get("status") in (None, "running", "ok"):
                summary["status"] = "error" if summary else "initialization_error"
            failed = True
        elif summary.get("status") != "ok":
            summary["status"] = "incomplete"
            failed = True
        summary["log_path"] = str(log_path)
        if process.returncode < 0:
            summary["termination_signal"] = -process.returncode
        summary.update(process_returncode=process.returncode, process_wall_time_s=process_wall)
        write_json(summary_path, summary)
        row = {"case": summary.get("case", f"{family.upper()}-{n}"), "status": summary["status"],
               "system_qubits": summary.get("n"), "width": summary.get("width"),
               "model_time_s": summary.get("model_time_s"),
               "rewrite_time_s": summary.get("rewrite_time_s"),
               "circuit_build_time_s": summary.get("circuit_build_time_s"),
               "logical_compile_time_s": summary.get("logical_compile_time_s"),
               "program_generation_time_s": summary.get("program_generation_time_s"),
               "peak_rss_mib": None if summary.get("peak_rss_bytes") is None else summary["peak_rss_bytes"] / 2**20,
               "process_wall_time_s": process_wall, "summary": str(summary_path)}
        for basis in BASIS:
            entry = summary.get("basis_results", {}).get(basis, {})
            row[f"{basis}_compile_time_s"] = entry.get("compile_time_s")
            row[f"{basis}_gates"] = entry.get("stats", {}).get("metric_total")
        rows.append(row)
        with (args.out_root / "summary.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        write_json(args.out_root / "summary.json", rows)
        print(json.dumps(row), flush=True)
        if interrupted is not None:
            raise interrupted
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
