#!/usr/bin/env python3
"""Rerun Table 1 appendix in-house channel-LCU rows after Matrixsum phase updates.

The runner intentionally rebuilds the in-house ``ours`` circuits from the model
definitions.  It does not load previously generated QASM files as inputs.  The
output CSV has the same overlay schema consumed by
``generate_channel_lcu_appendix_full_table.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402

from experiment_protocol import transpile_channel_circuit as transpile


DEFAULT_OUT_ROOT = (
    ROOT / "Baseline_results" / "eval_redesign" / "matrixsum_phase_refresh_20260706"
)
DEFAULT_RESULTS_CSV = ROOT / "Results" / "channel_lcu_matrixsum_phase_refresh_all.csv"
DEFAULT_RESULTS_MD = ROOT / "Results" / "channel_lcu_matrixsum_phase_refresh_all.md"

NAM_BASIS = ("h", "x", "rz", "cx", "reset")
IBM_BASIS = ("u1", "u2", "u3", "cx", "reset")

STRUCTURED_MODULE = (
    ROOT
    / "Baseline_results"
    / "eval_redesign"
    / "channel_lcu_structure_opt_refresh_20260705"
    / "structured"
    / "rerun_structured_families_refresh.py"
)


def metric_total(stats: dict[str, Any], basis: str) -> int:
    if stats.get("metric_total") is not None:
        return int(stats["metric_total"])
    if basis == "NAM":
        return int(stats["nam"]["rz_clifford_total"])
    if basis == "IBM":
        return int(stats["ibm"]["cx_u_total"])
    raise ValueError(basis)


def non_clifford(stats: dict[str, Any]) -> int:
    return int(stats["clifford"]["non_clifford"])


def dump_high_level_qasm(qc: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        qasm2.dump(qc, out_file)


def export_basis_count(
    qc: QuantumCircuit,
    path: Path,
    *,
    basis: str,
    optimization_level: int = 0,
) -> tuple[dict[str, Any], float]:
    basis_gates = NAM_BASIS if basis == "NAM" else IBM_BASIS
    started = time.perf_counter()
    tqc = transpile(qc, basis_gates=list(basis_gates), optimization_level=optimization_level)
    elapsed = time.perf_counter() - started
    export_openqasm2_quartz_input(tqc, path, basis_gates=basis_gates)
    stats = count_qasm_file(path, gate_set=basis.lower())
    return stats, elapsed


def basis_summary(
    *,
    stats: dict[str, Any],
    qasm: Path | str,
    transpile_time_s: float,
    compile_time_s: float,
    basis: str,
) -> dict[str, Any]:
    return {
        "gate_count": metric_total(stats, basis),
        "non_clifford": non_clifford(stats),
        "compile_time_s": float(compile_time_s),
        "transpile_time_s": float(transpile_time_s),
        "qasm": str(qasm),
        "stats": stats,
    }


def row_from_summary(
    *,
    family: str,
    benchmark: str,
    basis: str,
    summary_path: Path | str,
    basis_result: dict[str, Any],
    status: str = "ok",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "family": family,
        "benchmark": benchmark,
        "basis": basis,
        "before_gate_count": "",
        "after_gate_count": basis_result.get("gate_count", ""),
        "delta_gate_count": "",
        "before_non_clifford": "",
        "after_non_clifford": basis_result.get("non_clifford", ""),
        "delta_non_clifford": "",
        "before_compile_time_s": "",
        "after_compile_time_s": basis_result.get("compile_time_s", ""),
        "delta_compile_time_s": "",
        "before_status": "",
        "after_status": status,
        "source_before": "",
        "source_after": str(summary_path),
        "notes": notes,
    }


def write_case_summary(case_dir: Path, benchmark: str, payload: dict[str, Any]) -> Path:
    summary_path = case_dir / f"{benchmark}_matrixsum_phase_ours_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return summary_path


def run_tfim_case(n: int, out_root: Path) -> list[dict[str, Any]]:
    mod = importlib.import_module("run_tfim4_channel_lcu_ours_nam")
    benchmark = f"periodic_tfim_n{n}_dt0p1"
    case_dir = out_root / "main" / "tfim" / benchmark
    qasm_dir = case_dir / "qasm"
    raw_kraus_ops, model_info = mod.build_periodic_tfim_channel(n, 0.1)
    rewritten_kraus_ops, rewrite_result, rewrite_time = mod.rewrite_channel(
        raw_kraus_ops,
        strategy="greedy",
        beam_width=3,
        max_steps=50,
        tol=1e-12,
    )
    qc, qubit_indexes, build_time = mod.build_rewrite_opt_matrix_order_circuit(
        rewritten_kraus_ops
    )
    high_qasm = qasm_dir / f"{benchmark}_ours_rewrite_opt_matrix_order.qasm"
    dump_high_level_qasm(qc, high_qasm)

    basis_results: dict[str, dict[str, Any]] = {}
    for basis in ("NAM", "IBM"):
        qasm_path = qasm_dir / f"{benchmark}_ours_rewrite_opt_matrix_order_{basis.lower()}_opt0.qasm"
        stats, transpile_time = export_basis_count(qc, qasm_path, basis=basis)
        compile_time = float(rewrite_time + build_time + transpile_time)
        basis_results[basis.lower()] = basis_summary(
            stats=stats,
            qasm=qasm_path,
            transpile_time_s=transpile_time,
            compile_time_s=compile_time,
            basis=basis,
        )

    summary = {
        "case": benchmark,
        "family": "TFIM",
        "status": "ok",
        "stage": "matrixsum_phase_refresh_20260706",
        "policy": "rebuilt from model; no previously generated QASM input",
        "model": model_info,
        "raw_channel_metrics": mod.channel_metrics(raw_kraus_ops),
        "rewritten_channel_metrics": mod.channel_metrics(rewritten_kraus_ops),
        "rewrite": {
            "method": "rewrite_search",
            "time_s": float(rewrite_time),
            "initial_support": rewrite_result.get("initial_support"),
            "final_support": rewrite_result.get("final_support"),
            "steps": len(rewrite_result.get("steps", [])),
            "termination": rewrite_result.get("termination"),
        },
        "build_time_s": float(build_time),
        "qubit_indexes": qubit_indexes,
        "high_level_qasm": str(high_qasm),
        "basis_results": basis_results,
        "circuit": {
            "num_qubits": int(qc.num_qubits),
            "num_clbits": int(qc.num_clbits),
            "depth": int(qc.depth()),
            "size": int(qc.size()),
            "ops": {str(gate): int(count) for gate, count in qc.count_ops().items()},
        },
    }
    summary_path = write_case_summary(case_dir, benchmark, summary)
    return [
        row_from_summary(
            family="TFIM",
            benchmark=benchmark,
            basis=basis,
            summary_path=summary_path,
            basis_result=basis_results[basis.lower()],
            notes="fresh model rebuild via scripts/run_tfim4_channel_lcu_ours_nam.py helpers",
        )
        for basis in ("NAM", "IBM")
    ]


def run_hrw_case(n: int, out_root: Path) -> list[dict[str, Any]]:
    mod = importlib.import_module("run_hypercube_n4_ours_nam")
    benchmark = f"hypercube_random_walk_n{n}"
    case_dir = out_root / "main" / "hrw" / benchmark
    qasm_dir = case_dir / "qasm"
    raw_kraus_ops = mod.build_hypercube_channel(n)
    rewritten_kraus_ops, rewrite_result, rewrite_time = mod.rewrite_channel(
        raw_kraus_ops,
        strategy="greedy",
        beam_width=3,
        max_steps=50,
        tol=1e-12,
    )
    qc, qubit_indexes, build_time = mod.build_rewrite_opt_matrix_order_circuit(
        rewritten_kraus_ops
    )
    high_qasm = qasm_dir / f"{benchmark}_ours_rewrite_opt_matrix_order.qasm"
    dump_high_level_qasm(qc, high_qasm)

    basis_results: dict[str, dict[str, Any]] = {}
    for basis in ("NAM", "IBM"):
        qasm_path = qasm_dir / f"{benchmark}_ours_rewrite_opt_matrix_order_{basis.lower()}_opt0.qasm"
        stats, transpile_time = export_basis_count(qc, qasm_path, basis=basis)
        compile_time = float(rewrite_time + build_time + transpile_time)
        basis_results[basis.lower()] = basis_summary(
            stats=stats,
            qasm=qasm_path,
            transpile_time_s=transpile_time,
            compile_time_s=compile_time,
            basis=basis,
        )

    summary = {
        "case": benchmark,
        "family": "HRW",
        "status": "ok",
        "stage": "matrixsum_phase_refresh_20260706",
        "policy": "rebuilt from model; no previously generated QASM input",
        "raw_channel_metrics": mod.channel_metrics(raw_kraus_ops),
        "rewritten_channel_metrics": mod.channel_metrics(rewritten_kraus_ops),
        "rewrite": {
            "method": "rewrite_search",
            "time_s": float(rewrite_time),
            "initial_support": rewrite_result.get("initial_support"),
            "final_support": rewrite_result.get("final_support"),
            "steps": len(rewrite_result.get("steps", [])),
            "termination": rewrite_result.get("termination"),
        },
        "build_time_s": float(build_time),
        "qubit_indexes": qubit_indexes,
        "high_level_qasm": str(high_qasm),
        "basis_results": basis_results,
        "circuit": {
            "num_qubits": int(qc.num_qubits),
            "num_clbits": int(qc.num_clbits),
            "depth": int(qc.depth()),
            "size": int(qc.size()),
            "ops": {str(gate): int(count) for gate, count in qc.count_ops().items()},
        },
    }
    summary_path = write_case_summary(case_dir, benchmark, summary)
    return [
        row_from_summary(
            family="HRW",
            benchmark=benchmark,
            basis=basis,
            summary_path=summary_path,
            basis_result=basis_results[basis.lower()],
            notes="fresh model rebuild via scripts/run_hypercube_n4_ours_nam.py helpers",
        )
        for basis in ("NAM", "IBM")
    ]


def run_random_coloring_case(out_root: Path) -> list[dict[str, Any]]:
    mod = importlib.import_module("run_random_coloring_channel_lcu_ours_nam")
    case = mod.CaseSpec(n=4, q=4, graph="linear")
    args = SimpleNamespace(
        out_root=out_root / "main" / "random_coloring",
        rewrite_objective="gatecost",
        rewrite_strategy="beam",
        beam_width=8,
        max_steps=50,
        tol=1e-12,
        transpile_optimization_level=0,
        metadata_only=False,
        raw_only=False,
        ours_only=True,
        color_mixing_preprocess=False,
        no_color_mixing=True,
        synthesis_timeout_sec=0.0,
        max_total_pauli_terms=250_000,
        max_terms_per_kraus=50_000,
    )
    summary = mod.run_case(args, case)
    benchmark = str(summary["case"])
    status = str(summary.get("status", "unknown"))
    summary_path = summary.get("summary_path", "")
    rows = []
    for basis in ("NAM", "IBM"):
        output = summary.get("ours", {}).get("basis_outputs", {}).get(basis.lower(), {})
        stats = output.get("stats")
        if not stats:
            rows.append(
                {
                    "family": "random coloring",
                    "benchmark": benchmark,
                    "basis": basis,
                    "before_gate_count": "",
                    "after_gate_count": "",
                    "delta_gate_count": "",
                    "before_non_clifford": "",
                    "after_non_clifford": "",
                    "delta_non_clifford": "",
                    "before_compile_time_s": "",
                    "after_compile_time_s": "",
                    "delta_compile_time_s": "",
                    "before_status": "",
                    "after_status": status,
                    "source_before": "",
                    "source_after": summary_path,
                    "notes": summary.get("error_msg", ""),
                }
            )
            continue
        compile_time = (
            float(summary["rewrite"]["time_s"])
            + float(summary["ours"]["build_time_s"])
            + float(summary["ours"]["basis_transpile_times_s"][basis.lower()])
        )
        rows.append(
            row_from_summary(
                family="random coloring",
                benchmark=benchmark,
                basis=basis,
                summary_path=summary_path,
                basis_result={
                    "gate_count": metric_total(stats, basis),
                    "non_clifford": non_clifford(stats),
                    "compile_time_s": compile_time,
                },
                status=status,
                notes="fresh model rebuild; ours-only path; no color-mixing preprocess",
            )
        )
    return rows


def run_random_channel_cases(out_root: Path, *, seeds_n5: list[int], seeds_n9: list[int]) -> list[dict[str, Any]]:
    mod = importlib.import_module("build_random_channel_lcu_benchmark")
    out_dir = out_root / "main" / "random_channel"
    circuits_dir = out_dir / "circuits"
    rows: list[dict[str, Any]] = []
    for spec in [(5, 4, 10, 2), (9, 8, 20, 4)]:
        seeds = seeds_n5 if spec[0] == 5 else seeds_n9
        args = SimpleNamespace(
            out_root=out_dir,
            circuits_root=circuits_dir,
            results_md=out_dir / "random_channel_summary.md",
            results_csv=out_dir / "random_channel_summary.csv",
            transpile_optimization_level=0,
            rewrite_strategy="greedy",
            beam_width=3,
            max_steps=50,
            tol=1e-12,
            force_build=True,
            run_tools=False,
            build_only=True,
            tools=[],
            timeout=3600.0,
            workflow_timeout=0.0,
            runner_python=sys.executable,
            specs=[spec],
            seeds=seeds,
            force_tools=False,
            skip_final_recount=True,
            quartz_max_candidates=64,
            quartz_checkpoint_interval=300.0,
            quartz_no_increase=False,
            quartz_no_nop=False,
            quartz_use_available_xfers=False,
            bqskit_optimization_level=2,
            bqskit_workflow_mode="skip-retarget-mapping",
            bqskit_checkpoint=False,
        )
        for case in mod.selected_cases(args):
            summary = mod.build_case(args, case)
            benchmark = str(summary["case"])
            summary_path = (
                Path(summary["ours"]["basis_outputs"]["nam"]["qasm"]).parent
                / f"{benchmark}_build_summary.json"
            )
            for basis in ("NAM", "IBM"):
                output = summary["ours"]["basis_outputs"][basis.lower()]
                stats = output["stats"]
                compile_time = (
                    float(summary["rewrite"]["time_s"])
                    + float(summary["ours"]["build_time_s"])
                    + float(output.get("transpile_time_s", 0.0))
                )
                rows.append(
                    row_from_summary(
                        family="random channel",
                        benchmark=benchmark,
                        basis=basis,
                        summary_path=summary_path,
                        basis_result={
                            "gate_count": metric_total(stats, basis),
                            "non_clifford": non_clifford(stats),
                            "compile_time_s": compile_time,
                        },
                        status=str(summary.get("status", "ok")),
                        notes="fresh random-channel rebuild from seed parameters",
                    )
                )
    return rows


def load_structured_module() -> Any:
    spec = importlib.util.spec_from_file_location("matrixsum_phase_structured_refresh", STRUCTURED_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import structured refresh module: {STRUCTURED_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_structured_cases(out_root: Path, only: set[str] | None = None) -> list[dict[str, Any]]:
    module = load_structured_module()
    out_dir = out_root / "structured"
    rows: list[dict[str, Any]] = []
    selected = [case for case in module.CASES if only is None or case.example in only]
    for case in selected:
        print(f"[structured] {case.family}: {case.example}", flush=True)
        if case.kind.startswith("graph"):
            summary = module.rerun_graph(case, out_dir)
        elif case.kind == "toom":
            summary = module.rerun_toom(case, out_dir)
        elif case.kind == "decay":
            summary = module.rerun_decay(case, out_dir)
        elif case.kind == "dqe":
            summary = module.rerun_dqe(case, out_dir)
        else:
            raise ValueError(f"unknown structured case kind: {case.kind}")
        summary_path = summary["summary_path"]
        for basis in ("NAM", "IBM"):
            basis_result = summary["basis_results"][basis.lower()]
            rows.append(
                row_from_summary(
                    family=case.family,
                    benchmark=case.example,
                    basis=basis,
                    summary_path=summary_path,
                    basis_result=basis_result,
                    status=str(summary.get("status", "ok")),
                    notes=f"fresh model rebuild via structured runner; {case.notes}".strip("; "),
                )
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "benchmark",
        "basis",
        "before_gate_count",
        "after_gate_count",
        "delta_gate_count",
        "before_non_clifford",
        "after_non_clifford",
        "delta_non_clifford",
        "before_compile_time_s",
        "after_compile_time_s",
        "delta_compile_time_s",
        "before_status",
        "after_status",
        "source_before",
        "source_after",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def write_md(rows: list[dict[str, Any]], path: Path, csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Table 1 Ours Matrixsum Phase Refresh",
        "",
        f"CSV: `{csv_path}`",
        "",
        "All after rows were rebuilt from model definitions; no previously generated QASM files were used as inputs.",
        "",
        "| Family | Benchmark | Basis | Gates | Non-Clifford | Compile time (s) | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {benchmark} | {basis} | {after_gate_count} | "
            "{after_non_clifford} | {after_compile_time_s} | {after_status} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    lines.extend(["", "## Notes", ""])
    for row in rows:
        if row.get("notes"):
            lines.append(f"- `{row['benchmark']}` `{row['basis']}`: {row['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_generated_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        benchmark = str(item.get("benchmark", ""))
        source_after = str(item.get("source_after", ""))
        if (
            benchmark.startswith("random_channel_")
            and source_after.endswith("_build_summary.json")
            and f"/cases/{benchmark}/" not in source_after
        ):
            prefix = source_after[: -len(f"{benchmark}_build_summary.json")]
            item["source_after"] = f"{prefix}{benchmark}/{benchmark}_build_summary.json"
        normalized.append(item)
    return normalized


def run_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    jobs: list[tuple[str, Any]] = []
    if "tfim" in args.families:
        jobs.extend((f"TFIM n={n}", lambda n=n: run_tfim_case(n, out_root)) for n in (4, 8, 12))
    if "hrw" in args.families:
        jobs.extend((f"HRW n={n}", lambda n=n: run_hrw_case(n, out_root)) for n in (4, 8, 12, 20, 28))
    if "structured" in args.families:
        only = set(args.structured_only) if args.structured_only else None
        jobs.append(("structured cases", lambda: run_structured_cases(out_root, only=only)))
    if "random_coloring" in args.families:
        jobs.append(("random coloring n4 q4 linear", lambda: run_random_coloring_case(out_root)))
    if "random_channel" in args.families:
        jobs.append(
            (
                "random channel N5/N9",
                lambda: run_random_channel_cases(out_root, seeds_n5=args.n5_seeds, seeds_n9=args.n9_seeds),
            )
        )

    for label, fn in jobs:
        started = time.perf_counter()
        print(f"[run] {label}", flush=True)
        try:
            produced = fn()
            rows.extend(produced)
            print(f"[ok] {label}: {len(produced)} rows in {time.perf_counter() - started:.3f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"label": label, "error": error, "traceback": traceback.format_exc()})
            print(f"[failed] {label}: {error}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "failures.json").write_text(
        json.dumps(failures, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS_MD)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("tfim", "hrw", "structured", "random_coloring", "random_channel"),
        default=["tfim", "hrw", "structured", "random_coloring", "random_channel"],
    )
    parser.add_argument("--structured-only", nargs="*", default=None)
    parser.add_argument("--n5-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n9-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--collect-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    after_rows_path = out_root / "after_rows.json"
    if args.collect_only:
        rows = normalize_generated_rows(json.loads(after_rows_path.read_text(encoding="utf-8")))
    else:
        rows = normalize_generated_rows(run_jobs(args))
        out_root.mkdir(parents=True, exist_ok=True)
        after_rows_path.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    write_csv(rows, Path(args.results_csv))
    write_md(rows, Path(args.results_md), Path(args.results_csv))
    print(f"[wrote] {args.results_csv}")
    print(f"[wrote] {args.results_md}")
    print(f"[wrote] {after_rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
