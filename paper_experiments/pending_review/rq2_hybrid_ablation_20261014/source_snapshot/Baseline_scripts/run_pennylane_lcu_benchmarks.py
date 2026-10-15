#!/usr/bin/env python3
"""Replay the current Qrisp benchmark registry through PennyLane's native LCU APIs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Baseline_results" / "pennylane_lcu_v1"
REFERENCE_PATHS = (
    ROOT / "Baseline_results/table_main_select_only_ours_qrisp_v1/summary.json",
    ROOT / "Baseline_results/table_main_select_only_random4q100_v1/summary.json",
    ROOT / "Baseline_results/table_main_select_only_random4q164_v1/summary.json",
)
PROFILES = ("legacy_raw", "legacy_compile", "graph_compile", "graph_work", "graph_work_raw")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references():
    result = {}
    for path in REFERENCE_PATHS:
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for group in ("rewritten_kraus", "single", "random"):
            for case in summary.get(group, {}).get("cases", []):
                result[case["case"]["name"]] = (case, str(path))
    return result


def historical_check(terms, old, num_qubits):
    metrics = old.get("metrics", old)
    preview = metrics.get("terms_preview", [])
    errors = []
    if metrics.get("term_count") != len(terms):
        errors.append("term_count")
    if metrics.get("system_qubits", num_qubits) != num_qubits:
        errors.append("system_qubits")
    alpha = sum(abs(value) for _, value in terms)
    old_alpha = metrics.get("l1_norm", metrics.get("pauli_norm"))
    if old_alpha is not None and not math.isclose(alpha, float(old_alpha), rel_tol=1e-10, abs_tol=1e-12):
        errors.append("l1_norm")
    for (label, coefficient), saved in zip(terms, preview):
        old_coefficient = complex(saved.get("effective_coeff", saved.get("coeff")))
        if label != saved["pauli"] or abs(coefficient - old_coefficient) > 1e-10:
            errors.append("ordered_terms_preview")
            break
    if not preview:
        errors.append("missing_preview")
    return {
        "status": "mismatch" if errors else "matched_recorded_fields",
        "errors": errors,
        "preview_terms": len(preview),
        "total_terms": len(terms),
        "historical_full_input_available": len(preview) == len(terms),
    }


def prepare_inputs(args):
    for directory in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
        sys.path.insert(0, str(directory))
    from build_select_only_random_trend_and_main_table import COMPACT_CASES
    from run_table_main_select_only_ours_qrisp import (
        MAIN_ROW_ORDER, RANDOM_CASES, REWRITTEN_CASES, SINGLE_CASES,
        build_benchmark_current, build_random_matrixsum, build_single_matrixsum,
        kraus_effective_terms, rewrite_channel,
    )

    specs = {spec["name"]: spec for spec in (*REWRITTEN_CASES, *SINGLE_CASES, *RANDOM_CASES)}
    names = args.cases or list(MAIN_ROW_ORDER if args.suite == "extended" else COMPACT_CASES)
    unknown = set(names) - set(specs)
    if unknown:
        raise ValueError(f"Unknown benchmark names: {sorted(unknown)}")
    old_cases = references()
    manifest = {
        "schema_version": 1,
        "suite": args.suite,
        "policy": "Current SELECT benchmark inputs; full blocks use these same inputs. No full-channel circuit is constructed.",
        "historical_input_caveat": "Historical JSON stores previews, not complete inputs in most cases. Matching recorded fields is not proof of exact historical input equality.",
        "source_sha256": {
            str(path.relative_to(ROOT)): source_hash(path)
            for path in (
                ROOT / "Baseline_scripts/run_table_main_select_only_ours_qrisp.py",
                ROOT / "Baseline_scripts/run_block_encoding_new_qrisp_comparison.py",
                ROOT / "Baseline_scripts/run_dense_random_pauli4_block_encoding.py",
                ROOT / "Baseline_scripts/run_random_pauli_kraus_block_encoding.py",
                ROOT / "Baseline_scripts/run_table3_extra_block_encoding_cases.py",
                ROOT / "Baseline_scripts/sum_rewritten_kraus_block_counts_qrisp_vs_ours.py",
                ROOT / "scripts/run_random_coloring_channel_lcu_ours_nam.py",
                ROOT / "scripts/run_tfim4_channel_lcu_ours_nam.py",
                ROOT / "scripts/build_repetition_code_dqe_channel.py",
                ROOT / "src/hypercube_channel_pauli.py",
                ROOT / "src/channel_IR.py",
            )
        },
        "cases": [],
    }
    for name in names:
        spec = specs[name]
        old_case, old_path = old_cases.get(name, ({}, None))
        case = {"name": name, "spec": spec, "reference_path": old_path, "inputs": []}
        started = time.perf_counter()
        if spec in REWRITTEN_CASES:
            raw, model = build_benchmark_current(name)
            operators, rewrite, elapsed = rewrite_channel(raw, max_steps=50, beam_width=3, tol=1e-12)
            case.update({
                "aggregation": "sum_kraus", "model": model,
                "rewrite": {"max_steps": 50, "beam_width": 3, "tol": 1e-12,
                            "initial_support": rewrite.get("initial_support"),
                            "final_support": rewrite.get("final_support"), "seconds": elapsed},
            })
            old_inputs = {entry["kraus_index"]: entry for entry in old_case.get("kraus_results", [])}
            inputs = [(f"kraus{index:04d}", ms, old_inputs.get(index, {}), None)
                      for index, ms in enumerate(operators) if ms.instances]
        elif spec in RANDOM_CASES:
            case["aggregation"] = "sample_mean"
            samples = spec["samples"] if args.samples is None else min(spec["samples"], args.samples)
            old_inputs = {entry["seed"]: entry for entry in old_case.get("samples", [])}
            inputs = []
            for index in range(samples):
                seed = spec["seed_base"] + index
                inputs.append((f"sample{index:02d}", build_random_matrixsum(spec, seed), old_inputs.get(seed, {}), seed))
        else:
            case["aggregation"] = "single"
            inputs = [("single", build_single_matrixsum(spec), old_case, spec.get("seed"))]

        for input_id, ms, old, seed in inputs:
            terms = kraus_effective_terms(ms)
            serialized = [{"pauli": label, "real": value.real, "imag": value.imag} for label, value in terms]
            check = historical_check(terms, old, int(ms.size)) if old else {"status": "unavailable"}
            item = {
                "id": input_id, "seed": seed, "num_system_qubits": int(ms.size),
                "terms": serialized, "input_sha256": digest(serialized),
                "historical_check": check,
                "qrisp_select_reference": old.get("qrisp_select_only_q_switch", {}).get("basis_summary", {}),
                "ours_select_reference": old.get("ours_select_only_matrix_order", {}).get("basis_summary", {}),
            }
            case["inputs"].append(item)
        case["input_preparation_seconds"] = time.perf_counter() - started
        manifest["cases"].append(case)
        save_json(args.out_dir / "inputs.partial.json", manifest)
        print(f"prepared {name}: {len(inputs)} inputs", flush=True)
    save_json(args.out_dir / "inputs.json", manifest)
    return manifest


def run_worker(args):
    sys.path.insert(0, str(ROOT / "Baseline_scripts"))
    from pennylane_lcu_adapter import build_lcu_tape, compile_lcu_tape, tape_to_qiskit
    from run_mcg_tool_raw_channel_lcu_benchmarks import recount_qiskit_circuit

    manifest = json.loads((args.out_dir / "inputs.json").read_text())
    case = next(case for case in manifest["cases"] if case["name"] == args.worker_case)
    profile, scope = args.worker_profile, args.worker_scope
    directory = args.out_dir / case["name"] / scope / profile
    directory.mkdir(parents=True, exist_ok=True)
    result = {"case": case["name"], "scope": scope, "profile": profile,
              "aggregation": case["aggregation"], "expected_inputs": len(case["inputs"]),
              "max_no_work_terms": args.max_no_work_terms,
              "status": "running", "inputs": []}
    cache = {}
    save_json(directory / "result.partial.json", result)
    for item in case["inputs"]:
        if not profile.startswith("graph_work") and len(item["terms"]) > args.max_no_work_terms:
            result["inputs"].append({
                "id": item["id"], "status": "resource_limit", "cache_hit": False,
                "input_sha256": item["input_sha256"], "elapsed_seconds": 0.0,
                "error": f"No-workspace profile limited to {args.max_no_work_terms} terms; input has {len(item['terms'])}. Use a graph_work profile or explicitly increase --max-no-work-terms.",
            })
            save_json(directory / "result.partial.json", result)
            continue
        key = (item["num_system_qubits"], item["input_sha256"])
        if key in cache:
            entry = dict(cache[key], id=item["id"], cache_hit=True)
        else:
            entry = {"id": item["id"], "input_sha256": item["input_sha256"], "cache_hit": False}
            started = time.perf_counter()
            try:
                terms = [(term["pauli"], complex(term["real"], term["imag"])) for term in item["terms"]]
                work = max(0, (len(terms) - 1).bit_length() - 1) if profile.startswith("graph_work") else 0
                tape, wires, build_meta = build_lcu_tape(terms, item["num_system_qubits"], scope=scope, work_wires=work)
                lowered, compile_meta = compile_lcu_tape(
                    tape, wires, optimize=profile not in {"legacy_raw", "graph_work_raw"},
                    decomposition="graph" if profile.startswith("graph") else "legacy",
                )
                export_started = time.perf_counter()
                circuit = tape_to_qiskit(lowered, wires)
                entry.update({"build": build_meta, "compile": compile_meta,
                              "export_seconds": time.perf_counter() - export_started,
                              "logical_global_phase": float(circuit.global_phase),
                              "basis_summary": {}})
                for basis in ("nam", "ibm"):
                    lowered_result = recount_qiskit_circuit(
                        circuit, basis_name=basis,
                        out_path=directory / f"{item['id']}_{basis}_opt0.qasm", optimization_level=0,
                    )
                    stats = lowered_result["stats"]
                    entry["basis_summary"][basis] = {
                        "gate_count": int(stats["metric_total"]),
                        "non_clifford": int(stats["clifford"]["non_clifford"]),
                        "width": int(stats["num_qubits"]), "ops": stats["ops"],
                        "transpile_seconds": lowered_result["transpile_time_s"],
                        "qasm": lowered_result["qasm"],
                    }
                entry["status"] = "ok"
            except Exception:
                entry.update(status="error", error=traceback.format_exc())
            entry["elapsed_seconds"] = time.perf_counter() - started
            cache[key] = entry
        result["inputs"].append(entry)
        save_json(directory / "result.partial.json", result)
        print(f"{case['name']} {scope} {profile} {item['id']}: {entry['status']}", flush=True)
    result["status"] = "ok" if all(entry["status"] == "ok" for entry in result["inputs"]) else "error"
    if any(entry["status"] == "resource_limit" for entry in result["inputs"]):
        result["status"] = "resource_limit"
    save_json(directory / "result.json", result)


def dispatch(args, name, scope, profile):
    directory = args.out_dir / name / scope / profile
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(args.out_dir),
               "--worker-case", name, "--worker-scope", scope, "--worker-profile", profile,
               "--max-no-work-terms", str(args.max_no_work_terms)]
    started = time.perf_counter()
    status = None
    save_json(directory / "result.partial.json", {
        "case": name, "scope": scope, "profile": profile, "status": "starting",
        "input_manifest_sha256": source_hash(args.out_dir / "inputs.json"), "inputs": [],
    })
    with (directory / "run.log").open("w") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            if completed.returncode:
                status = "worker_error"
        except subprocess.TimeoutExpired:
            status = "timeout"
    path = directory / ("result.partial.json" if status else "result.json")
    result = json.loads(path.read_text()) if path.exists() else {
        "case": name, "scope": scope, "profile": profile, "inputs": [],
    }
    if status:
        result["status"] = status
    result["wall_seconds"] = time.perf_counter() - started
    result["timeout_seconds"] = args.timeout
    result["log"] = str(directory / "run.log")
    save_json(directory / "result.json", result)
    return result


def summarize(manifest, results):
    rows = []
    cases = {case["name"]: case for case in manifest["cases"]}
    for result in results:
        case = cases[result["case"]]
        entries = result["inputs"]
        expected = len(case["inputs"])
        for basis in ("nam", "ibm"):
            successful = [entry for entry in entries if entry["status"] == "ok" and basis in entry.get("basis_summary", {})]
            complete = len(successful) == expected and result["status"] == "ok"
            row = {key: result[key] for key in ("case", "scope", "profile", "status")}
            row.update(basis=basis, aggregation=case["aggregation"], expected_inputs=expected,
                       completed_inputs=len(successful), wall_seconds=result["wall_seconds"],
                       elapsed_unique_seconds=sum(entry.get("elapsed_seconds", 0) for entry in entries if not entry.get("cache_hit")))
            for metric in ("gate_count", "non_clifford", "width"):
                values = [entry["basis_summary"][basis][metric] for entry in successful]
                if not complete:
                    row[metric] = None
                elif case["aggregation"] == "sample_mean":
                    row[metric] = statistics.fmean(values)
                    row[metric + "_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
                else:
                    row[metric] = max(values) if metric == "width" else sum(values)
            row["pennylane_over_qrisp"] = None
            for method in ("qrisp", "ours"):
                valid = result["scope"] == "select" and all(
                    item["historical_check"]["status"] == "matched_recorded_fields"
                    and item[method + "_select_reference"].get(basis) for item in case["inputs"]
                )
                for metric in ("gate_count", "non_clifford", "width"):
                    values = [item[method + "_select_reference"][basis][metric] for item in case["inputs"]] if valid else []
                    row[method + "_" + metric] = (
                        statistics.fmean(values) if case["aggregation"] == "sample_mean"
                        else max(values) if metric == "width" else sum(values)
                    ) if values else None
            if row["gate_count"] is not None and row["qrisp_gate_count"]:
                row["pennylane_over_qrisp"] = row["gate_count"] / row["qrisp_gate_count"]
            rows.append(row)
    return rows


def write_report(path, summary):
    def number(value):
        return "--" if value is None else f"{value:,.2f}".rstrip("0").rstrip(".")

    lines = ["# PennyLane LCU benchmark results", "",
             f"PennyLane {summary['environment']['pennylane']} native LCU decomposition; Qiskit optimization_level=0 only for NAM/IBM recount.",
             "SELECT and complete PREPARE-SELECT-PREPARE-dagger blocks are separate workloads. No full-channel circuit or QSVT polynomial is included in these costs.",
             "Historical Qrisp/Ours numbers are reused only for SELECT after checking recorded input fields. Most historical inputs contain only previews; exact full historical equality is unverified.",
             "Channel cases sum Kraus costs and report maximum width. Random cases report sample means (standard deviations are in summary.json).", "",
             "legacy_raw = native recursive decomposition; legacy_compile adds the default qml.compile pipeline (one pass).",
             "graph_compile = graph decomposition plus that pipeline; graph_work additionally provides clean work wires for unary iteration.",
             "graph_work_raw = graph decomposition with the same clean work wires, without the subsequent local compile pipeline.",
             "Timeout/error/resource_limit rows have no complete cost; partial results and logs remain under the case directory.", ""]
    for scope in dict.fromkeys(row["scope"] for row in summary["rows"]):
        for basis in ("nam", "ibm"):
            lines += [f"## {scope} / {basis.upper()}", "",
                      "| Case | Profile | PL gates | PL NC | PL width | Qrisp gates | Ours gates | PL/Qrisp | Completed | Status | Seconds |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
            for row in summary["rows"]:
                if row["scope"] != scope or row["basis"] != basis:
                    continue
                cells = [row["case"], row["profile"], number(row["gate_count"]), number(row["non_clifford"]),
                         number(row["width"]), number(row["qrisp_gate_count"]), number(row["ours_gate_count"]),
                         number(row["pennylane_over_qrisp"]), f"{row['completed_inputs']}/{row['expected_inputs']}",
                         row["status"], number(row["wall_seconds"])]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("compact", "extended"), default="compact")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--scopes", choices=("select", "block"), nargs="+", default=["select", "block"])
    parser.add_argument("--profiles", choices=PROFILES, nargs="+", default=["legacy_compile", "graph_work"])
    parser.add_argument("--samples", type=int, help="Limit samples per random case; omission preserves original sample counts")
    parser.add_argument("--timeout", type=float, default=300, help="Seconds per case/scope/profile, including all its inputs")
    parser.add_argument("--max-no-work-terms", type=int, default=256,
                        help="Explicit term budget for profiles without work wires; graph_work profiles are exempt")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-inputs", action="store_true", help="Use saved complete input manifest without regenerating it")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--worker-profile", choices=PROFILES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-scope", choices=("select", "block"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / "matplotlib"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.jobs < 1 or args.timeout <= 0 or args.max_no_work_terms < 1 or (args.samples is not None and args.samples < 1):
        parser.error("jobs, timeout, max-no-work-terms and samples must be positive")
    if args.worker_case:
        run_worker(args)
        return
    manifest = json.loads((args.out_dir / "inputs.json").read_text()) if args.reuse_inputs else prepare_inputs(args)
    available = {case["name"] for case in manifest["cases"]}
    if args.cases and set(args.cases) - available:
        parser.error(f"Cases absent from input manifest: {sorted(set(args.cases) - available)}")
    if args.reuse_inputs and args.samples is not None:
        parser.error("--samples applies when preparing inputs; omit it with --reuse-inputs")
    if args.prepare_only:
        return
    summary = {
        "experiment": "pennylane_native_lcu", "input_manifest": str(args.out_dir / "inputs.json"),
        "input_manifest_sha256": source_hash(args.out_dir / "inputs.json"),
        "environment": {name: importlib.metadata.version(name) for name in ("pennylane", "qiskit", "numpy", "scipy")},
        "python": sys.version, "executable": sys.executable,
        "source_sha256": {str(path.relative_to(ROOT)): source_hash(path) for path in (
            Path(__file__).resolve(), ROOT / "Baseline_scripts/pennylane_lcu_adapter.py",
            ROOT / "Baseline_scripts/baseline_common.py",
            ROOT / "Baseline_scripts/run_mcg_tool_raw_channel_lcu_benchmarks.py",
        )},
        "qasm_policy": "OpenQASM2 artifacts are for gate counting. They omit global phase; saved logical_global_phase refers to the circuit before basis lowering, not a phase-restoration value for lowered QASM.",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "results": [], "rows": [],
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(dispatch, args, case["name"], scope, profile)
                   for case in manifest["cases"] if not args.cases or case["name"] in args.cases
                   for scope in args.scopes for profile in args.profiles]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            summary["results"].append(result)
            print(f"finished {result['case']} {result['scope']} {result['profile']}: {result['status']}", flush=True)
            summary["rows"] = summarize(manifest, summary["results"])
            save_json(args.out_dir / "summary.partial.json", summary)
    order = {case["name"]: index for index, case in enumerate(manifest["cases"])}
    summary["results"].sort(key=lambda row: (order[row["case"]], row["scope"], row["profile"]))
    summary["rows"] = summarize(manifest, summary["results"])
    save_json(args.out_dir / "summary.json", summary)
    write_report(args.report or args.out_dir / "report.md", summary)
    print(f"wrote {args.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
