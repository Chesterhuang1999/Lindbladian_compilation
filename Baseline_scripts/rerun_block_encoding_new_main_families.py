#!/usr/bin/env python3
"""Rerun current in-house channel-LCU results affected by block_encoding_new.

This worker covers the main-agent slice:

- periodic TFIM n=4,8,12
- hypercube random walk n=4,8,12,20,28
- random coloring n=4,q=4,linear
- random Pauli-channel cases N5/N9 with existing seeds

It writes a before/after table without modifying the historical result roots.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qiskit import qasm2


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402

from experiment_protocol import transpile_channel_circuit as transpile


DEFAULT_OUT_ROOT = (
    ROOT
    / "Baseline_results"
    / "eval_redesign"
    / "block_encoding_new_ours_rerun_20260704"
    / "main_families"
)
DEFAULT_RESULTS_CSV = ROOT / "Results" / "block_encoding_new_ours_rerun_main_families.csv"
DEFAULT_RESULTS_MD = ROOT / "Results" / "block_encoding_new_ours_rerun_main_families.md"

CP5_ALL_ROWS = ROOT / "Baseline_results" / "eval_redesign" / "cp5_aggregation" / "cp1_cp5_all_rows.csv"
IBM_DIAG = (
    ROOT
    / "Baseline_results"
    / "eval_redesign"
    / "ours_ibm_and_random_coloring_diagnostic"
    / "ours_full_ibm_metrics.csv"
)
RANDOM_CHANNEL_ROOT = (
    ROOT / "Baseline_results" / "eval_redesign" / "random_channel_benchmarks"
)

NAM_BASIS = ("h", "x", "rz", "cx", "reset")
IBM_BASIS = ("u1", "u2", "u3", "cx", "reset")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as in_file:
        return list(csv.DictReader(in_file))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    value_f = as_float(value)
    if value_f is None:
        return None
    return int(round(value_f))


def metric_total(stats: dict[str, Any], basis: str) -> int:
    if basis == "NAM":
        return int(stats.get("metric_total", stats["nam"]["rz_clifford_total"]))
    if basis == "IBM":
        return int(stats.get("metric_total", stats["ibm"]["cx_u_total"]))
    raise ValueError(basis)


def non_clifford(stats: dict[str, Any]) -> int:
    return int(stats["clifford"]["non_clifford"])


def existing_ours_lookup() -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(CP5_ALL_ROWS):
        if row.get("method") != "ours_full":
            continue
        benchmark = row.get("benchmark_name", "")
        basis = row.get("basis_group", "")
        if basis not in {"NAM", "IBM"}:
            continue
        if benchmark in {
            "periodic_tfim_n4_dt0p1",
            "periodic_tfim_n8_dt0p1",
            "periodic_tfim_n12_dt0p1",
            "hypercube_random_walk_n4",
            "hypercube_random_walk_n8",
            "hypercube_random_walk_n12",
            "hypercube_random_walk_n20",
            "hypercube_random_walk_n28",
            "random_coloring_n4_q4_linear",
        }:
            current = lookup.get((benchmark, basis))
            if current is None or row_priority(row) > row_priority(current):
                lookup[(benchmark, basis)] = row
    return lookup


def row_priority(row: dict[str, str]) -> tuple[int, int, int]:
    status_ok = 1 if row.get("status") in {"ok", "ours_ok"} else 0
    has_gates = 1 if row.get("opt_gates") not in (None, "") else 0
    has_time = 1 if row.get("total_time_sec") not in (None, "") else 0
    return (status_ok, has_gates, has_time)


def ibm_diag_lookup() -> dict[str, dict[str, str]]:
    if not IBM_DIAG.exists():
        return {}
    return {row["benchmark_name"]: row for row in read_csv(IBM_DIAG)}


def before_row_from_cp5(
    *,
    family: str,
    benchmark: str,
    basis: str,
    cp5_lookup: dict[tuple[str, str], dict[str, str]],
    ibm_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    row = cp5_lookup.get((benchmark, basis))
    before: dict[str, Any] = {
        "family": family,
        "benchmark": benchmark,
        "basis": basis,
        "before_gate_count": None,
        "before_non_clifford": None,
        "before_compile_time_s": None,
        "source_before": None,
        "before_status": "missing_before",
    }
    if row is not None:
        before.update(
            {
                "before_gate_count": as_int(row.get("opt_gates")),
                "before_non_clifford": as_int(row.get("non_clifford_count")),
                "before_compile_time_s": as_float(row.get("total_time_sec")),
                "source_before": row.get("source_path"),
                "before_status": row.get("status") or "unknown",
            }
        )

    # Reconstruct IBM compile time for TFIM/HRW from the historical NAM summary
    # plus the old IBM lowering time when possible.  The CP5 table kept the
    # shorter diagnostic wall time; this field is closer to the in-house compile
    # path used for fresh reruns.
    if basis == "IBM" and benchmark in ibm_lookup and row is not None:
        source = row.get("source_path")
        diag = ibm_lookup[benchmark]
        source_path = Path(source) if source else None
        if source_path and not source_path.is_absolute():
            source_path = ROOT / source_path
        if source_path and source_path.exists() and source_path.suffix == ".json":
            try:
                summary = load_json(source_path)
                rewrite_time = float(summary.get("rewrite", {}).get("time_s", 0.0))
                build_time = float(summary.get("ours", {}).get("build_time_s", 0.0))
                ibm_transpile = as_float(diag.get("transpile_time_sec")) or 0.0
                before["before_compile_time_s"] = rewrite_time + build_time + ibm_transpile
                before["before_status"] = f"{before['before_status']};ibm_time_reconstructed"
            except Exception:
                pass
    return before


def random_channel_before_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((RANDOM_CHANNEL_ROOT / "cases").glob("*/*_build_summary.json")):
        summary = load_json(path)
        case = str(summary.get("case", path.parent.name))
        model = summary.get("model", {})
        if not (case.startswith("random_channel_N5_") or case.startswith("random_channel_N9_")):
            continue
        for basis in ("NAM", "IBM"):
            key = basis.lower()
            output = summary["ours"]["basis_outputs"][key]
            stats = output["stats"]
            compile_time = (
                float(summary["rewrite"]["time_s"])
                + float(summary["ours"]["build_time_s"])
                + float(output.get("transpile_time_s", 0.0))
            )
            rows.append(
                {
                    "family": "random channel",
                    "benchmark": case,
                    "basis": basis,
                    "before_gate_count": metric_total(stats, basis),
                    "before_non_clifford": non_clifford(stats),
                    "before_compile_time_s": compile_time,
                    "source_before": str(path),
                    "before_status": "ok",
                    "extra": {
                        "parameter_set": (
                            f"N{model.get('kraus_count')}_dim{model.get('hilbert_dim')}"
                            f"_N1_{model.get('first_kraus_terms')}"
                            f"_N2_{model.get('other_kraus_terms')}"
                        ),
                        "seed": model.get("seed"),
                    },
                }
            )
    return rows


def recount_logical_qasm(
    source_qasm: Path,
    out_qasm: Path,
    *,
    basis: str,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    qc = qasm2.load(source_qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    gate_basis = NAM_BASIS if basis == "NAM" else IBM_BASIS
    tqc = transpile(qc, basis_gates=list(gate_basis), optimization_level=0)
    export_openqasm2_quartz_input(tqc, out_qasm, basis_gates=gate_basis)
    elapsed = time.perf_counter() - started
    stats = count_qasm_file(out_qasm, gate_set=basis.lower())
    return stats, elapsed


def after_entries_from_tfim_summary(
    summary: dict[str, Any],
    *,
    family: str,
    benchmark: str,
    out_dir: Path,
    nam_note: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    nam_stats = summary["ours"]["stats"]
    entries.append(
        after_entry(
            family=family,
            benchmark=benchmark,
            basis="NAM",
            stats=nam_stats,
            compile_time=float(summary["ours"]["compile_time_s"]),
            source_after=summary["summary_path"],
            status="ok",
            notes=nam_note,
        )
    )
    ibm_qasm = out_dir / f"{benchmark}_ours_full_ibm_opt0.qasm"
    ibm_stats, ibm_time = recount_logical_qasm(
        Path(summary["ours"]["qasm"]),
        ibm_qasm,
        basis="IBM",
    )
    entries.append(
        after_entry(
            family=family,
            benchmark=benchmark,
            basis="IBM",
            stats=ibm_stats,
            compile_time=float(summary["rewrite"]["time_s"])
            + float(summary["ours"]["build_time_s"])
            + ibm_time,
            source_after=str(ibm_qasm),
            status="ok",
            notes="IBM opt0 recount from freshly rerun logical ours QASM",
        )
    )
    return entries


def after_entry(
    *,
    family: str,
    benchmark: str,
    basis: str,
    stats: dict[str, Any],
    compile_time: float | None,
    source_after: str,
    status: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "family": family,
        "benchmark": benchmark,
        "basis": basis,
        "after_gate_count": metric_total(stats, basis),
        "after_non_clifford": non_clifford(stats),
        "after_compile_time_s": compile_time,
        "source_after": source_after,
        "after_status": status,
        "notes": notes,
    }


def run_tfim(n: int, out_root: Path) -> list[dict[str, Any]]:
    mod = importlib.import_module("run_tfim4_channel_lcu_ours_nam")
    out_dir = out_root / "tfim" / f"periodic_tfim_n{n}_dt0p1"
    args = SimpleNamespace(
        num_qubits=n,
        delta_t=0.1,
        out_dir=out_dir,
        rewrite_strategy="greedy",
        beam_width=3,
        max_steps=50,
        tol=1e-12,
        nam_optimization_level=0,
    )
    summary = mod.run(args)
    benchmark = f"periodic_tfim_n{n}_dt0p1"
    return after_entries_from_tfim_summary(
        summary,
        family="TFIM",
        benchmark=benchmark,
        out_dir=out_dir,
        nam_note="fresh rerun via scripts/run_tfim4_channel_lcu_ours_nam.py",
    )


def run_hrw(n: int, out_root: Path) -> list[dict[str, Any]]:
    mod = importlib.import_module("run_hypercube_n4_ours_nam")
    benchmark = f"hypercube_random_walk_n{n}"
    table1_raw = ROOT / "circuits" / "Table1" / benchmark / f"{benchmark}_basic_raw_nam_opt0.qasm"
    out_dir = out_root / "hrw" / benchmark
    args = SimpleNamespace(
        num_qubits=n,
        out_dir=out_dir,
        table1_raw_qasm=table1_raw,
        rewrite_strategy="greedy",
        beam_width=3,
        max_steps=50,
        tol=1e-12,
        nam_optimization_level=0,
    )
    summary = mod.run(args)
    return after_entries_from_tfim_summary(
        summary,
        family="HRW",
        benchmark=benchmark,
        out_dir=out_dir,
        nam_note="fresh rerun via scripts/run_hypercube_n4_ours_nam.py",
    )


def run_random_coloring(out_root: Path) -> list[dict[str, Any]]:
    mod = importlib.import_module("run_random_coloring_channel_lcu_ours_nam")
    out_dir = out_root / "random_coloring"
    case = mod.CaseSpec(n=4, q=4, graph="linear")
    args = SimpleNamespace(
        out_root=out_dir,
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
    entries: list[dict[str, Any]] = []
    benchmark = summary["case"]
    status = str(summary.get("status", "unknown"))
    if status not in {"ok", "ours_ok"}:
        for basis in ("NAM", "IBM"):
            entries.append(
                {
                    "family": "random coloring",
                    "benchmark": benchmark,
                    "basis": basis,
                    "after_gate_count": None,
                    "after_non_clifford": None,
                    "after_compile_time_s": None,
                    "source_after": summary.get("summary_path"),
                    "after_status": status,
                    "notes": summary.get("error_msg", ""),
                }
            )
        return entries

    for basis in ("NAM", "IBM"):
        key = basis.lower()
        output = summary["ours"]["basis_outputs"][key]
        stats = output["stats"]
        compile_time = (
            float(summary["rewrite"]["time_s"])
            + float(summary["ours"]["build_time_s"])
            + float(summary["ours"]["basis_transpile_times_s"][key])
        )
        entries.append(
            after_entry(
                family="random coloring",
                benchmark=benchmark,
                basis=basis,
                stats=stats,
                compile_time=compile_time,
                source_after=output["qasm"],
                status=status,
                notes="ours-only rerun; raw channel-LCU synthesis intentionally skipped",
            )
        )
    return entries


def run_random_channels(out_root: Path, *, seeds_n5: list[int], seeds_n9: list[int]) -> list[dict[str, Any]]:
    mod = importlib.import_module("build_random_channel_lcu_benchmark")
    out_dir = out_root / "random_channel"
    circuits_dir = out_dir / "circuits"
    results_md = out_dir / "random_channel_summary.md"
    results_csv = out_dir / "random_channel_summary.csv"
    specs = [(5, 4, 10, 2), (9, 8, 20, 4)]
    entries: list[dict[str, Any]] = []

    for spec in specs:
        seeds = seeds_n5 if spec[0] == 5 else seeds_n9
        args = SimpleNamespace(
            out_root=out_dir,
            circuits_root=circuits_dir,
            results_md=results_md,
            results_csv=results_csv,
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
            for basis in ("NAM", "IBM"):
                key = basis.lower()
                output = summary["ours"]["basis_outputs"][key]
                stats = output["stats"]
                compile_time = (
                    float(summary["rewrite"]["time_s"])
                    + float(summary["ours"]["build_time_s"])
                    + float(output.get("transpile_time_s", 0.0))
                )
                entries.append(
                    after_entry(
                        family="random channel",
                        benchmark=summary["case"],
                        basis=basis,
                        stats=stats,
                        compile_time=compile_time,
                        source_after=output["qasm"],
                        status=str(summary.get("status", "ok")),
                        notes="fresh rerun via scripts/build_random_channel_lcu_benchmark.py",
                    )
                )
    return entries


def run_main_slice(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    after_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    jobs: list[tuple[str, Any]] = []
    if "tfim" in args.families:
        jobs.extend([(f"TFIM n={n}", lambda n=n: run_tfim(n, out_root)) for n in (4, 8, 12)])
    if "hrw" in args.families:
        jobs.extend([(f"HRW n={n}", lambda n=n: run_hrw(n, out_root)) for n in (4, 8, 12, 20, 28)])
    if "random_coloring" in args.families:
        jobs.append(("random coloring n4 q4 linear", lambda: run_random_coloring(out_root)))
    if "random_channel" in args.families:
        jobs.append(
            (
                "random channel N5/N9",
                lambda: run_random_channels(out_root, seeds_n5=args.n5_seeds, seeds_n9=args.n9_seeds),
            )
        )

    for label, fn in jobs:
        print(f"[rerun] {label}", flush=True)
        started = time.perf_counter()
        try:
            produced = fn()
            after_rows.extend(produced)
            print(f"[done] {label}: {len(produced)} basis rows in {time.perf_counter() - started:.3f}s", flush=True)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[failed] {label}: {exc}", flush=True)
            failures.append({"label": label, "error": str(exc), "traceback": tb})

    (out_root / "main_failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )
    return after_rows


def combine_before_after(after_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cp5 = existing_ours_lookup()
    ibm = ibm_diag_lookup()
    before_rows: list[dict[str, Any]] = []

    for family, benchmarks in {
        "TFIM": [
            "periodic_tfim_n4_dt0p1",
            "periodic_tfim_n8_dt0p1",
            "periodic_tfim_n12_dt0p1",
        ],
        "HRW": [
            "hypercube_random_walk_n4",
            "hypercube_random_walk_n8",
            "hypercube_random_walk_n12",
            "hypercube_random_walk_n20",
            "hypercube_random_walk_n28",
        ],
        "random coloring": ["random_coloring_n4_q4_linear"],
    }.items():
        for benchmark in benchmarks:
            for basis in ("NAM", "IBM"):
                before_rows.append(
                    before_row_from_cp5(
                        family=family,
                        benchmark=benchmark,
                        basis=basis,
                        cp5_lookup=cp5,
                        ibm_lookup=ibm,
                    )
                )
    before_rows.extend(random_channel_before_rows())

    before_lookup = {
        (row["benchmark"], row["basis"]): row
        for row in before_rows
    }
    after_lookup = {
        (row["benchmark"], row["basis"]): row
        for row in after_rows
    }
    keys = sorted(set(before_lookup) | set(after_lookup))
    combined: list[dict[str, Any]] = []
    for benchmark, basis in keys:
        before = before_lookup.get((benchmark, basis), {})
        after = after_lookup.get((benchmark, basis), {})
        family = before.get("family") or after.get("family") or ""
        row = {
            "family": family,
            "benchmark": benchmark,
            "basis": basis,
            "before_gate_count": before.get("before_gate_count"),
            "after_gate_count": after.get("after_gate_count"),
            "delta_gate_count": delta(before.get("before_gate_count"), after.get("after_gate_count")),
            "before_non_clifford": before.get("before_non_clifford"),
            "after_non_clifford": after.get("after_non_clifford"),
            "delta_non_clifford": delta(before.get("before_non_clifford"), after.get("after_non_clifford")),
            "before_compile_time_s": before.get("before_compile_time_s"),
            "after_compile_time_s": after.get("after_compile_time_s"),
            "delta_compile_time_s": delta(before.get("before_compile_time_s"), after.get("after_compile_time_s")),
            "before_status": before.get("before_status"),
            "after_status": after.get("after_status"),
            "source_before": before.get("source_before"),
            "source_after": after.get("source_after"),
            "notes": after.get("notes", ""),
        }
        combined.append(row)
    return combined


def delta(before: Any, after: Any) -> float | int | None:
    before_f = as_float(before)
    after_f = as_float(after)
    if before_f is None or after_f is None:
        return None
    value = after_f - before_f
    if float(value).is_integer():
        return int(value)
    return float(value)


def write_outputs(rows: list[dict[str, Any]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
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
    with csv_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    lines = [
        "# block_encoding_new rerun: main families",
        "",
        f"CSV: `{csv_path}`",
        "",
        "| Family | Benchmark | Basis | Gates before -> after | Non-Clifford before -> after | Compile time before -> after (s) | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {benchmark} | {basis} | {g0} -> {g1} ({dg}) | "
            "{n0} -> {n1} ({dn}) | {t0} -> {t1} ({dt}) | {st0} -> {st1} |".format(
                family=row.get("family", ""),
                benchmark=row.get("benchmark", ""),
                basis=row.get("basis", ""),
                g0=fmt_int(row.get("before_gate_count")),
                g1=fmt_int(row.get("after_gate_count")),
                dg=fmt_signed(row.get("delta_gate_count")),
                n0=fmt_int(row.get("before_non_clifford")),
                n1=fmt_int(row.get("after_non_clifford")),
                dn=fmt_signed(row.get("delta_non_clifford")),
                t0=fmt_float(row.get("before_compile_time_s")),
                t1=fmt_float(row.get("after_compile_time_s")),
                dt=fmt_signed_float(row.get("delta_compile_time_s")),
                st0=row.get("before_status") or "-",
                st1=row.get("after_status") or "-",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_int(value: Any) -> str:
    value_i = as_int(value)
    return "-" if value_i is None else f"{value_i:,}"


def fmt_float(value: Any) -> str:
    value_f = as_float(value)
    return "-" if value_f is None else f"{value_f:.3f}"


def fmt_signed(value: Any) -> str:
    value_i = as_int(value)
    if value_i is None:
        return "-"
    return f"{value_i:+,}"


def fmt_signed_float(value: Any) -> str:
    value_f = as_float(value)
    if value_f is None:
        return "-"
    return f"{value_f:+.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS_MD)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("tfim", "hrw", "random_coloring", "random_channel"),
        default=["tfim", "hrw", "random_coloring", "random_channel"],
    )
    parser.add_argument("--n5-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--n9-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--collect-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    after_path = out_root / "after_rows_main.json"
    if args.collect_only:
        after_rows = load_json(after_path) if after_path.exists() else []
    else:
        after_rows = run_main_slice(args)
        out_root.mkdir(parents=True, exist_ok=True)
        after_path.write_text(json.dumps(after_rows, indent=2, default=str), encoding="utf-8")
    rows = combine_before_after(after_rows)
    write_outputs(rows, Path(args.results_csv), Path(args.results_md))
    (out_root / "combined_rows_main.json").write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {args.results_csv}")
    print(f"wrote markdown to {args.results_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
