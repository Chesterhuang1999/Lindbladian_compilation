#!/usr/bin/env python3
"""Build and run random Pauli-Kraus channel-LCU benchmarks.

Each benchmark has N Kraus operators on log2(dim) system qubits.  The first
Kraus operator has N1 random Pauli terms and every remaining Kraus operator has
N2 random Pauli terms.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
for path in (SRC_DIR, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import Matrixsum, PauliAtom, channel as ChannelIR, channel_ensemble  # type: ignore  # noqa: E402
import channel_LCU  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402
from experiment_protocol import EXTERNAL_NAM_OPTIMIZERS

from experiment_protocol import transpile_channel_circuit as transpile


OUT_ROOT = ROOT / "Baseline_results" / "eval_redesign" / "random_channel_benchmarks"
CIRCUITS_ROOT = ROOT / "circuits" / "random_channel_benchmarks"
RESULTS_MD = ROOT / "Results" / "random_channel_benchmarks_summary.md"
RESULTS_CSV = ROOT / "Results" / "random_channel_benchmarks_summary.csv"
UNIFIED_RUNNER = ROOT / "Baseline_scripts" / "run_table1_unified_ibm_count_workflow.py"

NAM_BASIS = ("h", "x", "rz", "cx")
NAM_TRANSPILER_BASIS = ("h", "x", "rz", "cx", "reset")
IBM_BASIS = ("u1", "u2", "u3", "cx")
IBM_TRANSPILER_BASIS = ("u1", "u2", "u3", "cx", "reset")

NAM_TOOLS = ("voqc", "feynman", "wisq", "quartz")
IBM_TOOLS = ("qiskit_opt3", "pytket", "bqskit")
DEFAULT_EXTERNAL_TOOLS = EXTERNAL_NAM_OPTIMIZERS
DEFAULT_SPECS = ((5, 4, 10, 2), (9, 8, 20, 4))
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
OK_STATUSES = {"ok", "ok_recount_only", "raw_ok", "ours_ok"}


@dataclass(frozen=True)
class RandomChannelSpec:
    kraus_count: int
    hilbert_dim: int
    first_kraus_terms: int
    other_kraus_terms: int

    @classmethod
    def from_qubits(cls, *, kraus_count, num_qubits, first_kraus_terms, other_kraus_terms):
        from experiment_protocol import RandomChannelInput
        value = RandomChannelInput(kraus_count=kraus_count, num_qubits=num_qubits,
                                   first_kraus_terms=first_kraus_terms,
                                   other_kraus_terms=other_kraus_terms, seed=0)
        return cls(value.kraus_count, value.hilbert_dim,
                   value.first_kraus_terms, value.other_kraus_terms)

    @property
    def system_qubits(self) -> int:
        log_dim = math.log2(self.hilbert_dim)
        if abs(log_dim - round(log_dim)) > 1e-12:
            raise ValueError(f"hilbert_dim must be a power of two: {self.hilbert_dim}")
        return int(round(log_dim))

    @property
    def label(self) -> str:
        return (
            f"N{self.kraus_count}_dim{self.hilbert_dim}_"
            f"N1_{self.first_kraus_terms}_N2_{self.other_kraus_terms}"
        )


@dataclass(frozen=True)
class RandomChannelCase:
    spec: RandomChannelSpec
    seed: int

    @property
    def name(self) -> str:
        return f"random_channel_{self.spec.label}_seed{self.seed}"


def all_pauli_labels(num_qubits: int, *, include_identity: bool = False) -> list[str]:
    labels = [""]
    for _ in range(num_qubits):
        labels = [prefix + pauli for prefix in labels for pauli in "IXYZ"]
    if include_identity:
        return labels
    identity = "I" * num_qubits
    return [label for label in labels if label != identity]


def build_random_matrixsum(
    *,
    num_qubits: int,
    term_count: int,
    rng: np.random.Generator,
) -> Matrixsum:
    labels = all_pauli_labels(num_qubits, include_identity=False)
    if term_count > len(labels):
        raise ValueError(
            f"Cannot draw {term_count} unique non-identity Pauli labels "
            f"on {num_qubits} qubits; only {len(labels)} are available."
        )
    chosen = rng.choice(np.array(labels, dtype=object), size=term_count, replace=False)
    phases = np.array([1.0 + 0.0j, -1.0 + 0.0j, 1.0j, -1.0j], dtype=complex)
    instances = []
    for label in chosen:
        phase = complex(rng.choice(phases))
        magnitude = float(rng.uniform(0.5, 1.5))
        instances.append((PauliAtom(str(label), phase=phase), magnitude))
    return Matrixsum(instances).simplify()


def build_random_kraus_ops(case: RandomChannelCase) -> tuple[list[Matrixsum], dict[str, Any]]:
    rng = np.random.default_rng(case.seed)
    q = case.spec.system_qubits
    kraus_ops = []
    term_counts = []
    for index in range(case.spec.kraus_count):
        term_count = (
            case.spec.first_kraus_terms if index == 0 else case.spec.other_kraus_terms
        )
        kraus_ops.append(build_random_matrixsum(num_qubits=q, term_count=term_count, rng=rng))
        term_counts.append(term_count)
    model = {
        "model": "random_pauli_kraus_channel",
        "kraus_count": int(case.spec.kraus_count),
        "hilbert_dim": int(case.spec.hilbert_dim),
        "system_qubits": int(q),
        "first_kraus_terms": int(case.spec.first_kraus_terms),
        "other_kraus_terms": int(case.spec.other_kraus_terms),
        "seed": int(case.seed),
        "term_counts": [int(x) for x in term_counts],
        "coefficient_magnitude_distribution": "uniform[0.5,1.5)",
        "coefficient_phase_distribution": "uniform over {1,-1,i,-i}",
        "pauli_label_policy": "unique non-identity labels per Kraus operator",
    }
    return kraus_ops, model


def channel_metrics(kraus_ops: list[Matrixsum]) -> dict[str, int]:
    nonzero = [op for op in kraus_ops if len(op.instances) > 0]
    return {
        "kraus_count": int(len(nonzero)),
        "pauli_terms_total": int(sum(len(op.instances) for op in nonzero)),
        "max_terms_per_kraus": int(max((len(op.instances) for op in nonzero), default=0)),
        "system_qubits": int(max((op.size for op in nonzero), default=0)),
    }


def rewrite_kraus_ops(
    kraus_ops: list[Matrixsum],
    *,
    strategy: str,
    beam_width: int,
    max_steps: int,
    tol: float,
) -> tuple[list[Matrixsum], dict[str, Any], float]:
    started = time.perf_counter()
    rewrite_source = ChannelIR(copy.deepcopy(kraus_ops))
    rewrite_source.zero_elim()
    result = rewrite_source.rewrite_search(
        strategy=strategy,
        beam_width=beam_width,
        max_steps=max_steps,
        tol=tol,
    )
    rewritten = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten.apply_rewrite_result(result, tol=tol)
    rewritten.zero_elim()
    return rewritten.kraus_ops, result, time.perf_counter() - started


def with_new_block_encoding(func):
    old_block_encoding = channel_LCU.BlockEncoding
    channel_LCU.BlockEncoding = NewBlockEncoding
    try:
        return func()
    finally:
        channel_LCU.BlockEncoding = old_block_encoding


def build_basic_channel_lcu_circuit(
    kraus_ops: list[Matrixsum],
) -> tuple[QuantumCircuit, Any, float]:
    started = time.perf_counter()

    def _build():
        return channel_LCU.channel_to_LCU(
            channel_ensemble([copy.deepcopy(kraus_ops)]),
            structure="basic",
            opt="No",
        )

    qc, qubit_indexes = with_new_block_encoding(_build)
    return qc, qubit_indexes, time.perf_counter() - started


def compact_matrix_order_block_metadata(
    kraus_index: int,
    matrixsum: Matrixsum,
    block_encoding: Any,
    circuit: QuantumCircuit,
    *,
    selected_crit_override: str | None = None,
) -> dict[str, Any]:
    placement_debug = getattr(block_encoding, "placement_search_debug", {}) or {}
    return {
        "kraus_index": int(kraus_index),
        "pauli_terms": int(len(matrixsum.instances)),
        "system_qubits": int(matrixsum.size),
        "block_qubits": int(circuit.num_qubits),
        "selected_k": int(getattr(block_encoding, "selected_k", -1)),
        "selected_crit": selected_crit_override
        or str(getattr(block_encoding, "selected_crit", "")),
        "candidate_costs": getattr(block_encoding, "candidate_costs", []),
        "width_candidate_costs": getattr(block_encoding, "width_candidate_costs", []),
        "placement": {
            "initial_strategy": placement_debug.get("initial_strategy"),
            "shortcut": placement_debug.get("shortcut"),
            "shortcut_axis": placement_debug.get("shortcut_axis"),
            "shortcut_rank": placement_debug.get("shortcut_rank"),
            "pre_refinement_cost": placement_debug.get("pre_refinement_cost"),
            "best_cost": placement_debug.get("best_cost"),
            "padded_search_cost": placement_debug.get("padded_search_cost"),
            "padding_count": placement_debug.get("padding_count"),
            "search_profile": placement_debug.get("search_profile"),
            "candidate_width": placement_debug.get("candidate_width"),
            "candidate_widths": placement_debug.get("candidate_widths"),
        },
    }


def build_ours_matrix_order_circuit(
    kraus_ops: list[Matrixsum],
) -> tuple[QuantumCircuit, Any, float]:
    """Use the shared corrected backend and explicit tree outer selection."""
    from channel_IR import channel_ensemble
    import channel_LCU

    started = time.perf_counter()
    qc, qubit_indexes = channel_LCU.channel_to_LCU(
        channel_ensemble([copy.deepcopy(kraus_ops)]), structure="tree", opt="Matrix-order",
    )
    return qc, qubit_indexes, time.perf_counter() - started


def dump_qasm2(qc: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        qasm2.dump(qc, out_file)


def export_basis_count_qasm(
    qc: QuantumCircuit,
    path: Path,
    *,
    gate_set: str,
    optimization_level: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if gate_set == "nam":
        basis = NAM_TRANSPILER_BASIS
    elif gate_set == "ibm":
        basis = IBM_TRANSPILER_BASIS
    else:
        raise ValueError(gate_set)
    tqc = transpile(qc, basis_gates=list(basis), optimization_level=optimization_level)
    elapsed = time.perf_counter() - started
    export_openqasm2_quartz_input(tqc, path, basis_gates=basis)
    return count_qasm_file(path, gate_set=gate_set), elapsed


def metric_total(stats: dict[str, Any], basis: str) -> int:
    if basis == "nam":
        return int(stats["nam"]["rz_clifford_total"])
    if basis == "ibm":
        return int(stats["ibm"]["cx_u_total"])
    raise ValueError(basis)


def row_from_stats(
    *,
    case: RandomChannelCase,
    method: str,
    category: str,
    basis: str,
    stats: dict[str, Any],
    status: str,
    reference_total: int | None,
    compile_time_s: float | None,
    summary_path: str | None,
    qasm_path: str | None,
) -> dict[str, Any]:
    total = metric_total(stats, basis)
    return {
        "benchmark": case.name,
        "parameter_set": case.spec.label,
        "N": int(case.spec.kraus_count),
        "hilbert_dim": int(case.spec.hilbert_dim),
        "N1": int(case.spec.first_kraus_terms),
        "N2": int(case.spec.other_kraus_terms),
        "system_qubits": int(case.spec.system_qubits),
        "seed": int(case.seed),
        "category": category,
        "method": method,
        "basis": basis,
        "status": status,
        "metric_total": int(total),
        "ratio_to_raw": None if reference_total in (None, 0) else float(total / reference_total),
        "non_clifford": int(stats["clifford"].get("non_clifford", 0)),
        "cx": int(stats["ops"].get("cx", 0)),
        "qubits": int(stats.get("num_qubits", 0)),
        "elapsed_seconds": None,
        "compile_time_s": compile_time_s,
        "summary_path": summary_path,
        "qasm_path": qasm_path,
    }


def copy_canonical_inputs(case_dir: Path, circuits_dir: Path, build_summary: dict[str, Any]) -> dict[str, str]:
    circuits_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for basis in ("nam", "ibm"):
        source = Path(build_summary["raw"]["basis_outputs"][basis]["qasm"])
        destination = circuits_dir / f"{build_summary['case']}_basic_raw_{basis}_opt0.qasm"
        shutil.copyfile(source, destination)
        copied[f"raw_{basis}_qasm"] = str(destination)
    copied["case_dir"] = str(case_dir)
    copied["circuits_dir"] = str(circuits_dir)
    return copied


def build_case(args: argparse.Namespace, case: RandomChannelCase) -> dict[str, Any]:
    case_dir = Path(args.out_root) / "cases" / case.name
    summary_path = case_dir / f"{case.name}_build_summary.json"
    if summary_path.exists() and not args.force_build:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["summary_path"] = str(summary_path)
        return summary

    started = time.perf_counter()
    case_dir.mkdir(parents=True, exist_ok=True)
    kraus_ops, model = build_random_kraus_ops(case)
    raw_channel_metrics = channel_metrics(kraus_ops)

    raw_qc, raw_qubit_indexes, raw_build_time = build_basic_channel_lcu_circuit(kraus_ops)
    raw_qasm = case_dir / f"{case.name}_basic_raw.qasm"
    dump_qasm2(raw_qc, raw_qasm)

    raw_basis_outputs: dict[str, Any] = {}
    raw_metric_totals: dict[str, int] = {}
    for basis in ("nam", "ibm"):
        basis_qasm = case_dir / f"{case.name}_basic_raw_{basis}_opt0.qasm"
        stats, elapsed = export_basis_count_qasm(
            raw_qc,
            basis_qasm,
            gate_set=basis,
            optimization_level=args.transpile_optimization_level,
        )
        raw_basis_outputs[basis] = {
            "qasm": str(basis_qasm),
            "stats": stats,
            "transpile_time_s": float(elapsed),
        }
        raw_metric_totals[basis] = metric_total(stats, basis)

    rewritten_kraus_ops, rewrite_result, rewrite_time = rewrite_kraus_ops(
        kraus_ops,
        strategy=args.rewrite_strategy,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        tol=args.tol,
    )
    rewritten_metrics = channel_metrics(rewritten_kraus_ops)

    ours_qc, ours_qubit_indexes, ours_build_time = build_ours_matrix_order_circuit(
        rewritten_kraus_ops
    )
    ours_qasm = case_dir / f"{case.name}_rewrite_opt_matrix_order.qasm"
    dump_qasm2(ours_qc, ours_qasm)

    ours_basis_outputs: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for basis in ("nam", "ibm"):
        basis_qasm = case_dir / f"{case.name}_rewrite_opt_matrix_order_{basis}_opt0.qasm"
        stats, elapsed = export_basis_count_qasm(
            ours_qc,
            basis_qasm,
            gate_set=basis,
            optimization_level=args.transpile_optimization_level,
        )
        total = metric_total(stats, basis)
        ours_basis_outputs[basis] = {
            "qasm": str(basis_qasm),
            "stats": stats,
            "transpile_time_s": float(elapsed),
        }
        comparisons[basis] = {
            "raw_metric_total": int(raw_metric_totals[basis]),
            "ours_metric_total": int(total),
            "ours_to_raw_ratio": float(total / raw_metric_totals[basis]),
        }

    summary = {
        "experiment": "random_channel_lcu_benchmark",
        "case": case.name,
        "status": "ok",
        "model": model,
        "basis_counted": {"nam": list(NAM_BASIS), "ibm": list(IBM_BASIS)},
        "transpiler_basis": {
            "nam": list(NAM_TRANSPILER_BASIS),
            "ibm": list(IBM_TRANSPILER_BASIS),
        },
        "transpile_optimization_level": int(args.transpile_optimization_level),
        "raw": {
            "channel_metrics": raw_channel_metrics,
            "build_time_s": float(raw_build_time),
            "qasm": str(raw_qasm),
            "basis_outputs": raw_basis_outputs,
            "qubit_indexes": raw_qubit_indexes,
        },
        "rewrite": {
            "channel_metrics": rewritten_metrics,
            "time_s": float(rewrite_time),
            "initial_support": int(rewrite_result["initial_support"]),
            "final_support": int(rewrite_result["final_support"]),
            "steps": int(len(rewrite_result["steps"])),
            "termination": rewrite_result.get("termination", {}),
        },
        "ours": {
            "structure": "tree",
            "block_backend": "block_encoding.BlockEncoding",
            "qubits_initially_zero": False,
            "opt": "Matrix-order",
            "build_time_s": float(ours_build_time),
            "qasm": str(ours_qasm),
            "basis_outputs": ours_basis_outputs,
            "qubit_indexes": ours_qubit_indexes,
            "matrix_order_metadata": getattr(ours_qc, "metadata", {}) or {},
        },
        "comparison": comparisons,
        "total_time_s": float(time.perf_counter() - started),
    }
    canonical = copy_canonical_inputs(
        case_dir,
        Path(args.circuits_root) / case.name,
        {"case": case.name, "raw": {"basis_outputs": raw_basis_outputs}},
    )
    summary["canonical_external_inputs"] = canonical
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_unified_tools(args: argparse.Namespace, build_summary: dict[str, Any]) -> dict[str, Any]:
    bench = build_summary["case"]
    tool_dir = Path(args.out_root) / "cases" / bench / "generic_tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    command_result_path = tool_dir / "workflow_command_result.json"
    native_summary_path = tool_dir / f"{bench}_native_basis_summary.json"
    if native_summary_path.exists() and not args.force_tools:
        return {
            "benchmark": bench,
            "skipped_existing": True,
            "native_summary": str(native_summary_path),
            "command_result": str(command_result_path) if command_result_path.exists() else None,
        }

    canonical = build_summary["canonical_external_inputs"]
    command = [
        str(args.runner_python),
        str(UNIFIED_RUNNER),
        "--benchmark-name",
        bench,
        "--nam-qasm",
        canonical["raw_nam_qasm"],
        "--ibm-qasm",
        canonical["raw_ibm_qasm"],
        "--result-dir",
        str(tool_dir),
        "--tools",
        *args.tools,
        "--timeout",
        str(float(args.timeout)),
        "--final-count-basis",
        "ibm",
        "--qiskit-opt3-basis",
        "ibm",
        "--qiskit-opt3-timeout",
        str(float(args.timeout)),
        "--wisq-optimization-objective",
        "TOTAL",
        "--quartz-max-candidates",
        str(int(args.quartz_max_candidates)),
        "--quartz-checkpoint-interval",
        str(float(args.quartz_checkpoint_interval)),
        "--bqskit-optimization-level",
        str(int(args.bqskit_optimization_level)),
        "--bqskit-workflow-mode",
        str(args.bqskit_workflow_mode),
        "--pytket-strategy",
        "best-of",
        "--pytket-best-of-objective",
        "metric_total",
        "--keep-intermediate",
    ]
    if args.skip_final_recount:
        command.append("--skip-final-recount")
    if args.quartz_no_increase:
        command.append("--quartz-no-increase")
    if args.quartz_no_nop:
        command.append("--quartz-no-nop")
    if args.quartz_use_available_xfers:
        command.append("--quartz-use-available-xfers")
    if args.bqskit_checkpoint:
        command.append("--bqskit-checkpoint")

    stdout_path = tool_dir / "workflow_stdout.log"
    stderr_path = tool_dir / "workflow_stderr.log"
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=None if args.workflow_timeout <= 0 else float(args.workflow_timeout),
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

    elapsed = time.perf_counter() - started
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    result = {
        "benchmark": bench,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": float(elapsed),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "native_summary": str(native_summary_path) if native_summary_path.exists() else None,
    }
    command_result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["command_result"] = str(command_result_path)
    return result


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def case_from_summary(summary: dict[str, Any]) -> RandomChannelCase:
    model = summary["model"]
    spec = RandomChannelSpec(
        kraus_count=int(model["kraus_count"]),
        hilbert_dim=int(model["hilbert_dim"]),
        first_kraus_terms=int(model["first_kraus_terms"]),
        other_kraus_terms=int(model["other_kraus_terms"]),
    )
    return RandomChannelCase(spec=spec, seed=int(model["seed"]))


def build_rows_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    case = case_from_summary(summary)
    rows: list[dict[str, Any]] = []
    raw_refs: dict[str, int] = {}
    for basis in ("nam", "ibm"):
        raw_output = summary["raw"]["basis_outputs"][basis]
        raw_total = metric_total(raw_output["stats"], basis)
        raw_refs[basis] = raw_total
        rows.append(
            row_from_stats(
                case=case,
                method="raw",
                category="raw",
                basis=basis,
                stats=raw_output["stats"],
                status="ok",
                reference_total=raw_total,
                compile_time_s=float(summary["raw"]["build_time_s"])
                + float(raw_output.get("transpile_time_s", 0.0)),
                summary_path=summary.get("summary_path"),
                qasm_path=raw_output["qasm"],
            )
        )
    for basis in ("nam", "ibm"):
        ours_output = summary["ours"]["basis_outputs"][basis]
        rows.append(
            row_from_stats(
                case=case,
                method="ours",
                category="ours",
                basis=basis,
                stats=ours_output["stats"],
                status="ok",
                reference_total=raw_refs[basis],
                compile_time_s=float(summary["rewrite"]["time_s"])
                + float(summary["ours"]["build_time_s"])
                + float(ours_output.get("transpile_time_s", 0.0)),
                summary_path=summary.get("summary_path"),
                qasm_path=ours_output["qasm"],
            )
        )
    return rows


def native_gate_total(gate_counts: dict[str, Any] | None) -> int | None:
    if not isinstance(gate_counts, dict):
        return None
    if gate_counts.get("total") is None:
        return None
    return int(gate_counts["total"])


def build_generic_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    case = case_from_summary(summary)
    raw_refs = {
        basis: metric_total(summary["raw"]["basis_outputs"][basis]["stats"], basis)
        for basis in ("nam", "ibm")
    }
    bench = summary["case"]
    case_dir = Path(summary["summary_path"]).parent
    tool_dirs = [case_dir / "generic_tools"]
    tool_dirs.extend(sorted(case_dir.glob("generic_tools_*")))
    rows_by_method_basis: dict[tuple[str, str], tuple[dict[str, Any], Path, dict[str, Any]]] = {}
    command_results = []

    def row_score(row: dict[str, Any]) -> tuple[int, int]:
        counts = native_gate_total(row.get("gate_counts"))
        status = str(row.get("status", ""))
        return (1 if counts is not None else 0, 1 if status in OK_STATUSES else 0)

    for tool_dir in tool_dirs:
        native_summary_path = tool_dir / f"{bench}_native_basis_summary.json"
        native_summary = load_json(native_summary_path)
        command_result = load_json(tool_dir / "workflow_command_result.json") or {}
        if command_result:
            command_results.append(command_result)
        if not native_summary:
            continue
        tool_results = {}
        for result in native_summary.get("tool_results", []):
            if isinstance(result, dict) and result.get("tool"):
                tool_results[str(result["tool"])] = result
        for native_row in native_summary.get("rows", []):
            method = native_row.get("method")
            basis = native_row.get("basis")
            if method == "qiskit_opt3_ibm":
                key = ("qiskit_opt3", "ibm")
            elif method in set(NAM_TOOLS) | {"pytket", "bqskit"}:
                key = (str(method), str(basis))
            else:
                continue
            candidate = (native_row, native_summary_path, tool_results.get(key[0], {}))
            previous = rows_by_method_basis.get(key)
            if previous is None or row_score(native_row) > row_score(previous[0]):
                rows_by_method_basis[key] = candidate

    expected = [(tool, "nam") for tool in NAM_TOOLS] + [(tool, "ibm") for tool in IBM_TOOLS]
    rows: list[dict[str, Any]] = []
    workflow_failed = any(result.get("returncode") not in (0, None) for result in command_results)
    for method, basis in expected:
        selected = rows_by_method_basis.get((method, basis))
        if selected is None:
            status = "workflow_failed" if workflow_failed else "not_run"
            rows.append(
                {
                    "benchmark": case.name,
                    "parameter_set": case.spec.label,
                    "N": int(case.spec.kraus_count),
                    "hilbert_dim": int(case.spec.hilbert_dim),
                    "N1": int(case.spec.first_kraus_terms),
                    "N2": int(case.spec.other_kraus_terms),
                    "system_qubits": int(case.spec.system_qubits),
                    "seed": int(case.seed),
                    "category": "generic",
                    "method": method,
                    "basis": basis,
                    "status": status,
                    "metric_total": None,
                    "ratio_to_raw": None,
                    "non_clifford": None,
                    "cx": None,
                    "qubits": None,
                    "elapsed_seconds": None,
                    "compile_time_s": None,
                    "summary_path": None,
                    "qasm_path": None,
                }
            )
            continue

        native_row, native_summary_path, tool_result = selected
        gate_counts = native_row.get("gate_counts")
        total = native_gate_total(gate_counts)
        if method == "qiskit_opt3":
            qasm_path = None
        else:
            qasm_path = tool_result.get("native_output")
        rows.append(
            {
                "benchmark": case.name,
                "parameter_set": case.spec.label,
                "N": int(case.spec.kraus_count),
                "hilbert_dim": int(case.spec.hilbert_dim),
                "N1": int(case.spec.first_kraus_terms),
                "N2": int(case.spec.other_kraus_terms),
                "system_qubits": int(case.spec.system_qubits),
                "seed": int(case.seed),
                "category": "generic",
                "method": method,
                "basis": basis,
                "status": native_row.get("status"),
                "metric_total": total,
                "ratio_to_raw": None if total is None else float(total / raw_refs[basis]),
                "non_clifford": None
                if not isinstance(gate_counts, dict)
                else gate_counts.get("non_clifford"),
                "cx": None if not isinstance(gate_counts, dict) else gate_counts.get("cx"),
                "qubits": None,
                "elapsed_seconds": native_row.get("elapsed_seconds"),
                "compile_time_s": None,
                "summary_path": str(native_summary_path),
                "qasm_path": qasm_path,
            }
        )
    return rows


def collect_case_summaries(out_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((out_root / "cases").glob("*/*_build_summary.json")):
        summary = load_json(path)
        if summary is None:
            continue
        summary["summary_path"] = str(path)
        summaries.append(summary)
    return summaries


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "parameter_set",
        "N",
        "hilbert_dim",
        "N1",
        "N2",
        "system_qubits",
        "seed",
        "category",
        "method",
        "basis",
        "status",
        "metric_total",
        "ratio_to_raw",
        "non_clifford",
        "cx",
        "qubits",
        "elapsed_seconds",
        "compile_time_s",
        "summary_path",
        "qasm_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_int(value: float | int | None) -> str:
    if value is None:
        return "-"
    return str(int(round(float(value))))


def status_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    ok = 0
    timeout = 0
    failed = 0
    not_run = 0
    for row in rows:
        status = str(row.get("status"))
        if status in OK_STATUSES:
            ok += 1
        elif "timeout" in status:
            timeout += 1
        elif status in {"not_run", "skipped"}:
            not_run += 1
        else:
            failed += 1
    return ok, timeout, failed, not_run


def method_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    order = {
        ("raw", "nam"): 0,
        ("raw", "ibm"): 1,
        ("ours", "nam"): 2,
        ("ours", "ibm"): 3,
        ("voqc", "nam"): 4,
        ("feynman", "nam"): 5,
        ("wisq", "nam"): 6,
        ("quartz", "nam"): 7,
        ("qiskit_opt3", "ibm"): 8,
        ("pytket", "ibm"): 9,
        ("bqskit", "ibm"): 10,
    }
    return (order.get((str(row["method"]), str(row["basis"])), 99), str(row["method"]))


def write_markdown(rows: list[dict[str, Any]], path: Path, out_root: Path, csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["parameter_set"], row["method"], row["basis"]), []).append(row)

    mean_lines = [
        "| Parameter set | Method | Basis | ok/timeout/failed/not-run | metrics | Mean gates | Mean G/G0 | Mean non-Clifford |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(grouped, key=lambda item: (item[0], method_sort_key({"method": item[1], "basis": item[2]}))):
        parameter_set, method, basis = key
        group_rows = grouped[key]
        ok, timeout, failed, not_run = status_counts(group_rows)
        metrics = [int(row["metric_total"]) for row in group_rows if row.get("metric_total") not in (None, "")]
        ratios = [float(row["ratio_to_raw"]) for row in group_rows if row.get("ratio_to_raw") not in (None, "")]
        non_cliffords = [
            int(row["non_clifford"])
            for row in group_rows
            if row.get("non_clifford") not in (None, "")
        ]
        mean_lines.append(
            f"| {parameter_set} | {method} | {basis.upper()} | "
            f"{ok}/{timeout}/{failed}/{not_run} | {len(metrics)}/{len(group_rows)} | "
            f"{fmt_int(mean(metrics) if metrics else None)} | "
            f"{fmt_float(mean(ratios) if ratios else None, 6)} | "
            f"{fmt_int(mean(non_cliffords) if non_cliffords else None)} |"
        )

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["benchmark"], []).append(row)

    per_seed_lines = [
        "| Parameter set | Seed | Raw NAM | Raw IBM | Ours NAM | Ours IBM | VOQC | Feynman | WISQ | Quartz | Qiskit opt3 | pytket | BQSKit |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    cell_methods = [
        ("raw", "nam"),
        ("raw", "ibm"),
        ("ours", "nam"),
        ("ours", "ibm"),
        ("voqc", "nam"),
        ("feynman", "nam"),
        ("wisq", "nam"),
        ("quartz", "nam"),
        ("qiskit_opt3", "ibm"),
        ("pytket", "ibm"),
        ("bqskit", "ibm"),
    ]
    for benchmark in sorted(by_case):
        case_rows = by_case[benchmark]
        first = case_rows[0]
        row_map = {(row["method"], row["basis"]): row for row in case_rows}
        cells = []
        for method, basis in cell_methods:
            row = row_map.get((method, basis))
            if row is None:
                cells.append("-")
                continue
            total = fmt_int(row.get("metric_total"))
            status = row.get("status") or "-"
            cells.append(f"{total} ({status})")
        per_seed_lines.append(
            f"| {first['parameter_set']} | {first['seed']} | " + " | ".join(cells) + " |"
        )

    lines = [
        "# Random channel benchmarks",
        "",
        "Random Pauli-Kraus channel-LCU benchmarks with five seeds per parameter set.",
        "QuiZX is skipped following the current mainline convention.",
        "",
        f"- Output root: `{out_root}`",
        f"- CSV: `{csv_path}`",
        "- Timeout: 3600 s per external-tool invocation",
        "- NAM tools: VOQC, Feynman, WISQ, Quartz",
        "- IBM tools: Qiskit opt3, pytket, BQSKit",
        "",
        "## Mean gate counts",
        "",
        *mean_lines,
        "",
        "## Per-seed results",
        "",
        *per_seed_lines,
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(args: argparse.Namespace, build_summaries: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> None:
    manifest = {
        "experiment": "random_channel_lcu_benchmark",
        "out_root": str(args.out_root),
        "circuits_root": str(args.circuits_root),
        "results_markdown": str(args.results_md),
        "results_csv": str(args.results_csv),
        "timeout": float(args.timeout),
        "seeds": [int(seed) for seed in args.seeds],
        "specs": [list(map(int, spec)) for spec in args.specs],
        "tools": list(args.tools),
        "quizx": "skipped_by_current_mainline_convention",
        "build_summaries": [summary.get("summary_path") for summary in build_summaries],
        "tool_results": tool_results,
    }
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def parse_specs(values: list[str] | None) -> list[tuple[int, int, int, int]]:
    if not values:
        return [tuple(spec) for spec in DEFAULT_SPECS]
    specs = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Spec must have form N,dim,N1,N2: {value}")
        specs.append(tuple(int(part) for part in parts))
    return specs


def selected_cases(args: argparse.Namespace) -> list[RandomChannelCase]:
    cases = []
    for spec_tuple in args.specs:
        spec = RandomChannelSpec(*map(int, spec_tuple))
        for seed in args.seeds:
            cases.append(RandomChannelCase(spec=spec, seed=int(seed)))
    return cases


def summarize(args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = collect_case_summaries(Path(args.out_root))
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.extend(build_rows_from_summary(summary))
        rows.extend(build_generic_rows(summary))
    rows.sort(
        key=lambda row: (
            str(row["parameter_set"]),
            int(row["seed"]),
            method_sort_key(row),
        )
    )
    write_csv(rows, Path(args.results_csv))
    write_markdown(rows, Path(args.results_md), Path(args.out_root), Path(args.results_csv))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    specs = parser.add_mutually_exclusive_group()
    specs.add_argument("--spec", dest="spec_values", action="append", help="Legacy: N,hilbert_dim,N1,N2")
    specs.add_argument("--qubit-spec", dest="qubit_spec_values", action="append", help="N,num_qubits,N1,N2")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--circuits-root", type=Path, default=CIRCUITS_ROOT)
    parser.add_argument("--results-md", type=Path, default=RESULTS_MD)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--transpile-optimization-level", type=int, default=0)
    parser.add_argument("--rewrite-strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--run-tools", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-tools", action="store_true")
    parser.add_argument("--tools", nargs="+", default=list(DEFAULT_EXTERNAL_TOOLS))
    parser.add_argument("--tool-jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--qiskit-opt3-timeout",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility knob; qiskit opt3 uses --timeout when "
            "dispatching the unified workflow."
        ),
    )
    parser.add_argument("--workflow-timeout", type=float, default=0.0)
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument("--skip-final-recount", action="store_true")
    parser.add_argument("--quartz-max-candidates", type=int, default=64)
    parser.add_argument("--quartz-checkpoint-interval", type=float, default=300.0)
    parser.add_argument("--quartz-no-increase", action="store_true")
    parser.add_argument("--quartz-no-nop", action="store_true")
    parser.add_argument("--quartz-use-available-xfers", action="store_true")
    parser.add_argument("--bqskit-optimization-level", type=int, default=2)
    parser.add_argument(
        "--bqskit-workflow-mode",
        choices=("standard", "skip-retarget-mapping"),
        default="skip-retarget-mapping",
    )
    parser.add_argument("--bqskit-checkpoint", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.specs = parse_specs(args.spec_values)
    if args.qubit_spec_values:
        args.specs = []
        for value in args.qubit_spec_values:
            n, qubits, first, other = (int(part.strip()) for part in value.split(","))
            spec = RandomChannelSpec.from_qubits(kraus_count=n, num_qubits=qubits,
                                                first_kraus_terms=first, other_kraus_terms=other)
            args.specs.append((spec.kraus_count, spec.hilbert_dim,
                               spec.first_kraus_terms, spec.other_kraus_terms))
    args.out_root = Path(args.out_root).resolve()
    args.circuits_root = Path(args.circuits_root).resolve()
    args.results_md = Path(args.results_md).resolve()
    args.results_csv = Path(args.results_csv).resolve()

    if args.summarize_only:
        rows = summarize(args)
        print(f"wrote {len(rows)} rows to {args.results_csv}")
        print(f"wrote summary to {args.results_md}")
        return 0

    cases = selected_cases(args)
    build_summaries = []
    for case in cases:
        print(f"[build] {case.name}", flush=True)
        build_summaries.append(build_case(args, case))

    tool_results: list[dict[str, Any]] = []
    if args.run_tools and not args.build_only:
        jobs = max(1, int(args.tool_jobs))
        if jobs == 1:
            for summary in build_summaries:
                print(f"[tools] {summary['case']}", flush=True)
                tool_results.append(run_unified_tools(args, summary))
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_map = {
                    executor.submit(run_unified_tools, args, summary): summary["case"]
                    for summary in build_summaries
                }
                for future in as_completed(future_map):
                    bench = future_map[future]
                    print(f"[tools done] {bench}", flush=True)
                    tool_results.append(future.result())

    write_manifest(args, build_summaries, tool_results)
    rows = summarize(args)
    print(f"wrote {len(rows)} rows to {args.results_csv}")
    print(f"wrote summary to {args.results_md}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
