#!/usr/bin/env python3
"""Rerun current block_encoding_new ours data for structured channel-LCU cases.

This script is intentionally scoped to the subagent assignment:
graph cluster, Toom, decay higher-order, and repetition-code DQE.
It preserves old artifacts and writes only under this output directory plus
an intermediate Results markdown/csv.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from qiskit import qasm2, transpile
from qiskit.circuit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
BASELINE_DIR = ROOT / "Baseline_scripts"
for path in (SRC_DIR, SCRIPTS_DIR, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from channel_IR import channel as ChannelIR  # type: ignore  # noqa: E402
import channel_LCU  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402

import build_decay_higher_order_expansion as decay_builder  # type: ignore  # noqa: E402
import build_graph_cluster_state_channel_lcu as graph_builder  # type: ignore  # noqa: E402
import build_repetition_code_dqe_channel as dqe_builder  # type: ignore  # noqa: E402
import build_toom_2d_channel_lcu as toom_builder  # type: ignore  # noqa: E402


NAM_BASIS = ("h", "x", "rz", "cx", "reset")
IBM_BASIS = ("u1", "u2", "u3", "cx", "reset")
RESULTS_MD = ROOT / "Results" / "block_encoding_new_ours_rerun_subagent_structured.md"
RESULTS_CSV = ROOT / "Results" / "block_encoding_new_ours_rerun_subagent_structured.csv"


@dataclass(frozen=True)
class CaseSpec:
    family: str
    example: str
    before_summary: Path
    kind: str
    params: dict[str, Any]
    before_high_qasm: Path | None = None
    notes: str = ""


CASES = [
    CaseSpec(
        family="graph cluster",
        example="graph_cluster_2d_grid_2x4",
        kind="graph_grid",
        params={"rows": 2, "cols": 4, "gamma": 0.1, "delta_t": 0.1},
        before_summary=ROOT
        / "Baseline_results"
        / "graph_cluster_state_channel_lcu"
        / "graph_cluster_2d_grid_2x4"
        / "graph_cluster_2d_grid_2x4_summary.json",
        before_high_qasm=ROOT
        / "circuits"
        / "Table1"
        / "graph_cluster_2d_grid_2x4"
        / "graph_cluster_2d_grid_2x4_rewrite_opt_matrix_order.qasm",
    ),
    CaseSpec(
        family="graph cluster",
        example="graph_cluster_random_bd3_n12_seed7",
        kind="graph_random",
        params={
            "random_n": 12,
            "random_degree": 3,
            "random_seed": 7,
            "gamma": 0.1,
            "delta_t": 0.1,
        },
        before_summary=ROOT
        / "Baseline_results"
        / "graph_cluster_state_channel_lcu"
        / "graph_cluster_random_bd3_n12_seed7"
        / "graph_cluster_random_bd3_n12_seed7_summary.json",
        before_high_qasm=ROOT
        / "circuits"
        / "Table1"
        / "graph_cluster_random_bd3_n12_seed7"
        / "graph_cluster_random_bd3_n12_seed7_rewrite_opt_matrix_order.qasm",
    ),
    CaseSpec(
        family="Toom",
        example="toom_2d_periodic_L2_dt0p1",
        kind="toom",
        params={"lattice_L": 2, "gamma": 0.1, "delta_t": 0.1, "rank_width_disabled": True},
        before_summary=ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "toom_l2_rank_width_disabled"
        / "summaries"
        / "toom_2d_periodic_L2_dt0p1"
        / "toom_2d_periodic_L2_dt0p1_summary.json",
        before_high_qasm=ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "toom_l2_rank_width_disabled"
        / "circuits"
        / "toom_2d_periodic_L2_dt0p1"
        / "toom_2d_periodic_L2_dt0p1_rewrite_opt_matrix_order.qasm",
        notes="before source is the rank-width-disabled corrected Toom L2 run",
    ),
    CaseSpec(
        family="Toom",
        example="toom_2d_periodic_L3_dt0p1",
        kind="toom",
        params={"lattice_L": 3, "gamma": 0.1, "delta_t": 0.1, "force_original_ctrl_width": True},
        before_summary=ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "toom_l3_force_original_ctrl_width"
        / "summaries"
        / "toom_2d_periodic_L3_dt0p1"
        / "toom_2d_periodic_L3_dt0p1_summary.json",
        before_high_qasm=ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "toom_l3_force_original_ctrl_width"
        / "circuits"
        / "toom_2d_periodic_L3_dt0p1"
        / "toom_2d_periodic_L3_dt0p1_rewrite_opt_matrix_order.qasm",
        notes="before source is the force-original-control-width corrected Toom L3 run",
    ),
    CaseSpec(
        family="decay-higher-order",
        example="decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K1_4",
        kind="decay",
        params={
            "lambda_value": 1.0,
            "v": 0.5,
            "scaling": 4.01,
            "total_time": 0.1,
            "k_order": 2,
            "sampling_points": 4,
            "coherent_order": 4,
            "structure": "optimized",
            "coherent_opt": "Matrix-order",
            "jump_opt": "Matrix-order",
        },
        before_summary=ROOT
        / "Baseline_results"
        / "decay_higher_order_expansion"
        / "decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K14_optimized_coh_matrix-order_jump_matrix-order"
        / "decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K14_optimized_coh_matrix-order_jump_matrix-order_summary.json",
        before_high_qasm=ROOT
        / "circuits"
        / "Table1"
        / "decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K14_optimized_coh_matrix-order_jump_matrix-order"
        / "decay_only_lambda1_v0p5_scale4p01_t0p1_higher_order_K2_q4_K14_optimized_coh_matrix-order_jump_matrix-order.qasm",
        notes="K=2, q=4, K1=4 optimized coherent/jump Matrix-order source",
    ),
    CaseSpec(
        family="repetition DQE",
        example="repetition_dqe_d3_eps0p25",
        kind="dqe",
        params={"d": 3, "epsilon": 0.25},
        before_summary=ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "repetition_dqe_raw_ours_probe"
        / "repetition_dqe_d3_eps0p25"
        / "repetition_dqe_d3_eps0p25_ours_summary.json",
        before_high_qasm=ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "repetition_dqe_raw_ours_probe"
        / "repetition_dqe_d3_eps0p25"
        / "repetition_dqe_d3_eps0p25_ours_rewrite_opt_matrix_order.qasm",
    ),
    CaseSpec(
        family="repetition DQE",
        example="repetition_dqe_d5_eps0p25",
        kind="dqe",
        params={"d": 5, "epsilon": 0.25},
        before_summary=ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "repetition_dqe_d5_ours_main"
        / "repetition_dqe_d5_eps0p25"
        / "repetition_dqe_d5_eps0p25_ours_summary.json",
        before_high_qasm=ROOT
        / "Baseline_results"
        / "eval_redesign"
        / "repetition_dqe_d5_ours_main"
        / "repetition_dqe_d5_eps0p25"
        / "repetition_dqe_d5_eps0p25_ours_rewrite_opt_matrix_order.qasm",
        notes="complete d5 ours source with NAM/IBM exports",
    ),
]


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as in_file:
        return json.load(in_file)


def metric_from_stats(stats: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not stats:
        return None, None
    return int(stats["metric_total"]), int(stats["clifford"]["non_clifford"])


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
    basis_gates = NAM_BASIS if basis == "nam" else IBM_BASIS
    started = time.perf_counter()
    tqc = transpile(qc, basis_gates=list(basis_gates), optimization_level=optimization_level)
    elapsed = time.perf_counter() - started
    export_openqasm2_quartz_input(tqc, path, basis_gates=basis_gates)
    return count_qasm_file(path, gate_set=basis), elapsed


def derive_basis_from_high_qasm(
    high_qasm: Path,
    out_path: Path,
    *,
    basis: str,
) -> tuple[dict[str, Any], float, str]:
    qc = qasm2.load(str(high_qasm), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    stats, transpile_time = export_basis_count(qc, out_path, basis=basis)
    return stats, transpile_time, str(out_path)


def basis_summary(
    stats: dict[str, Any],
    qasm_path: Path | str,
    transpile_time_s: float,
    compile_time_s: float,
) -> dict[str, Any]:
    gates, non_cliff = metric_from_stats(stats)
    return {
        "gate_count": gates,
        "non_clifford": non_cliff,
        "compile_time_s": float(compile_time_s),
        "transpile_time_s": float(transpile_time_s),
        "qasm": str(qasm_path),
        "stats": stats,
    }


def graph_spec(case: CaseSpec) -> graph_builder.GraphSpec:
    if case.kind == "graph_grid":
        return graph_builder.grid_2d_cluster_graph(case.params["rows"], case.params["cols"])
    return graph_builder.random_bounded_degree_graph(
        case.params["random_n"],
        case.params["random_degree"],
        case.params["random_seed"],
    )


def build_generic_opt_circuit(kraus_ops: list[Any]) -> tuple[QuantumCircuit, Any, float]:
    started = time.perf_counter()
    sys_size = max(ms.size for ms in kraus_ops)
    select_size = int(__import__("numpy").ceil(__import__("numpy").log2(len(kraus_ops))))
    coeff_sums = [ms.pauli_norm() for ms in kraus_ops]

    lcu_ini = channel_LCU.prep_sup_state(coeff_sums)
    circuits = [NewBlockEncoding(copy.deepcopy(ms)).circuit(opt="Matrix-order") for ms in kraus_ops]
    ctrl_size = max(circ.num_qubits - sys_size for circ in circuits)

    qc = QuantumCircuit(2 * select_size + ctrl_size + sys_size)
    select_indexes = [2 * j for j in range(select_size)]
    anc_indexes = [2 * j + 1 for j in range(select_size)]
    ctrl_indexes = list(range(2 * select_size, 2 * select_size + ctrl_size))
    sys_indexes = list(range(2 * select_size + ctrl_size, 2 * select_size + ctrl_size + sys_size))

    qc.compose(lcu_ini, qubits=select_indexes, inplace=True)
    qc, _tcount, _mccount, _cxcount = channel_LCU.mulplex_U_opt(
        qc,
        circuits,
        select_size,
        ctrl_size,
        sys_size,
    )
    return qc, [select_indexes, anc_indexes, ctrl_indexes, sys_indexes], time.perf_counter() - started


@contextlib.contextmanager
def temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def rerun_graph(case: CaseSpec, out_dir: Path) -> dict[str, Any]:
    spec = graph_spec(case)
    raw_kraus_ops, model_info = graph_builder.build_channel_for_spec(
        spec,
        case.params["gamma"],
        case.params["delta_t"],
    )
    rewritten_kraus_ops, rewrite_result, rewrite_time = graph_builder.rewrite_channel(
        raw_kraus_ops,
        strategy="greedy",
        beam_width=3,
        max_steps=50,
        tol=1e-12,
    )
    qc, qubit_indexes, build_time = build_generic_opt_circuit(rewritten_kraus_ops)
    return write_after_case(
        case,
        out_dir,
        qc,
        model_info=model_info,
        channel_metrics=graph_builder.channel_metrics(rewritten_kraus_ops),
        rewrite={"method": "rewrite_search", "time_s": rewrite_time, "result": rewrite_result},
        build_time_s=build_time,
        qubit_indexes=qubit_indexes,
    )


def rerun_toom(case: CaseSpec, out_dir: Path) -> dict[str, Any]:
    env = {
        "CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC": None,
        "CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH": "1"
        if case.params.get("force_original_ctrl_width")
        else None,
    }
    with temporary_env(env):
        raw_kraus_ops, model_info = toom_builder.build_toom_channel(
            case.params["lattice_L"],
            case.params["gamma"],
            case.params["delta_t"],
        )
        rewritten_kraus_ops, rewrite_result, rewrite_time = toom_builder.gatecost_rewrite_channel(
            raw_kraus_ops,
            strategy="beam",
            beam_width=8,
            max_steps=50,
            tol=1e-12,
            keep_neutral=False,
            max_neutral_candidates=3,
            skip_disjoint_pairs=True,
            verbose=False,
        )
        qc, qubit_indexes, build_time = build_generic_opt_circuit(rewritten_kraus_ops)
    return write_after_case(
        case,
        out_dir,
        qc,
        model_info=model_info,
        channel_metrics=toom_builder.channel_metrics(rewritten_kraus_ops),
        rewrite={"method": "gatecost_rewrite_search", "time_s": rewrite_time, "result": rewrite_result},
        build_time_s=build_time,
        qubit_indexes=qubit_indexes,
        env=env,
    )


def rerun_decay(case: CaseSpec, out_dir: Path) -> dict[str, Any]:
    args = argparse.Namespace(
        lambda_value=case.params["lambda_value"],
        v=case.params["v"],
        scaling=case.params["scaling"],
        total_time=case.params["total_time"],
        k_order=case.params["k_order"],
        sampling_points=case.params["sampling_points"],
        coherent_order=case.params["coherent_order"],
        coherent_opt=case.params["coherent_opt"],
        jump_opt=case.params["jump_opt"],
        structure=case.params["structure"],
        tol=1e-12,
    )
    qc, model_info, build_time = decay_builder.build_decay_higher_order_circuit(args)
    return write_after_case(
        case,
        out_dir,
        qc,
        model_info=model_info,
        channel_metrics={
            "logical_kraus_count": model_info.get("logical_kraus_count"),
            "selection_state_nonzero": model_info.get("selection_state_nonzero"),
        },
        rewrite={"method": "none", "time_s": 0.0, "result": {}},
        build_time_s=build_time,
        qubit_indexes=model_info.get("qubit_indexes"),
    )


def rerun_dqe(case: CaseSpec, out_dir: Path) -> dict[str, Any]:
    spec = dqe_builder.DqeSpec(d=case.params["d"], epsilon=case.params["epsilon"])
    channel, metadata = dqe_builder.build_repetition_dqe_channel(spec)
    matrixsum_ops = [dqe_builder.pauli_dict_to_matrixsum(kraus) for kraus in channel]
    started = time.perf_counter()
    source = ChannelIR(copy.deepcopy(matrixsum_ops))
    source.zero_elim()
    rewrite_result = source.rewrite_search(strategy="greedy", beam_width=3, max_steps=50, tol=1e-12)
    rewritten = ChannelIR(copy.deepcopy(matrixsum_ops))
    rewritten.apply_rewrite_result(rewrite_result, tol=1e-12)
    rewritten.zero_elim()
    rewrite_time = time.perf_counter() - started
    qc, qubit_indexes, build_time = build_generic_opt_circuit(rewritten.kraus_ops)
    return write_after_case(
        case,
        out_dir,
        qc,
        model_info=metadata,
        channel_metrics=dqe_builder.channel_metrics(channel),
        rewrite={"method": "rewrite_search", "time_s": rewrite_time, "result": rewrite_result},
        build_time_s=build_time,
        qubit_indexes=qubit_indexes,
        matrixsum_metrics={
            "kraus_count": len(rewritten.kraus_ops),
            "pauli_terms_total": sum(len(op.instances) for op in rewritten.kraus_ops),
            "max_terms_per_kraus": max((len(op.instances) for op in rewritten.kraus_ops), default=0),
            "min_terms_per_kraus": min((len(op.instances) for op in rewritten.kraus_ops), default=0),
            "system_qubits": max((op.size for op in rewritten.kraus_ops), default=0),
        },
    )


def write_after_case(
    case: CaseSpec,
    out_dir: Path,
    qc: QuantumCircuit,
    *,
    model_info: dict[str, Any],
    channel_metrics: dict[str, Any],
    rewrite: dict[str, Any],
    build_time_s: float,
    qubit_indexes: Any,
    env: dict[str, str | None] | None = None,
    matrixsum_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = out_dir / "cases" / case.example
    qasm_dir = case_dir / "qasm"
    qasm_dir.mkdir(parents=True, exist_ok=True)
    high_qasm = qasm_dir / f"{case.example}_after_ours_rewrite_opt_matrix_order.qasm"
    dump_high_level_qasm(qc, high_qasm)

    basis_results = {}
    for basis in ("nam", "ibm"):
        qasm_path = qasm_dir / f"{case.example}_after_ours_rewrite_opt_matrix_order_{basis}_opt0.qasm"
        stats, transpile_time = export_basis_count(qc, qasm_path, basis=basis)
        compile_time = float(rewrite.get("time_s", 0.0) + build_time_s + transpile_time)
        basis_results[basis] = basis_summary(stats, qasm_path, transpile_time, compile_time)

    summary = {
        "case": case.example,
        "family": case.family,
        "status": "ok",
        "stage": "after_current_block_encoding_new",
        "model": model_info,
        "channel_metrics": channel_metrics,
        "matrixsum_metrics": matrixsum_metrics,
        "rewrite": {
            "method": rewrite.get("method"),
            "time_s": rewrite.get("time_s"),
            "initial_support": rewrite.get("result", {}).get("initial_support"),
            "final_support": rewrite.get("result", {}).get("final_support"),
            "steps": len(rewrite.get("result", {}).get("steps", []))
            if isinstance(rewrite.get("result", {}).get("steps", []), list)
            else None,
            "termination": rewrite.get("result", {}).get("termination"),
            "initial_cost": rewrite.get("result", {}).get("initial_cost"),
            "final_cost": rewrite.get("result", {}).get("final_cost"),
        },
        "build_time_s": float(build_time_s),
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
        "environment": {
            "CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC": os.environ.get(
                "CHANNEL_LCU_ENABLE_RANK_WIDTH_HEURISTIC"
            ),
            "CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH": os.environ.get(
                "CHANNEL_LCU_FORCE_ORIGINAL_CTRL_WIDTH"
            ),
            "case_env_override": env or {},
        },
    }
    summary_path = case_dir / f"{case.example}_after_ours_summary.json"
    with summary_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, default=str)
    summary["summary_path"] = str(summary_path)
    return summary


def extract_existing_basis(summary: dict[str, Any], basis: str) -> dict[str, Any] | None:
    if "exports" in summary and isinstance(summary["exports"], dict):
        entry = summary["exports"].get(basis)
        if entry and entry.get("stats"):
            stats = entry["stats"]
            compile_time = (
                summary.get("rewrite", {}).get("time_s", 0.0)
                + summary.get("build_time_s", 0.0)
                + entry.get("export_time_s", 0.0)
            )
            return basis_summary(stats, entry.get("qasm", ""), entry.get("export_time_s", 0.0), compile_time)

    if basis == "nam":
        if summary.get("ours", {}).get("stats"):
            ours = summary["ours"]
            compile_time = ours.get("compile_time_s")
            if compile_time is None:
                compile_time = (
                    summary.get("rewrite", {}).get("time_s", 0.0)
                    + ours.get("build_time_s", 0.0)
                    + ours.get("nam_transpile_time_s", 0.0)
                )
            return basis_summary(
                ours["stats"],
                ours.get("nam_qasm", "") if isinstance(ours.get("nam_qasm"), str) else ours.get("nam_qasm", {}).get("path", ""),
                ours.get("nam_transpile_time_s", 0.0),
                compile_time,
            )
        if summary.get("nam_stats"):
            compile_time = summary.get("build_time_s", 0.0) + summary.get("nam_transpile_time_s", 0.0)
            return basis_summary(
                summary["nam_stats"],
                summary.get("nam_qasm", ""),
                summary.get("nam_transpile_time_s", 0.0),
                compile_time,
            )
    elif summary.get("ibm_stats"):
        compile_time = summary.get("build_time_s", 0.0) + summary.get("ibm_transpile_time_s", 0.0)
        return basis_summary(
            summary["ibm_stats"],
            summary.get("ibm_qasm", ""),
            summary.get("ibm_transpile_time_s", 0.0),
            compile_time,
        )
    return None


def source_base_time(summary: dict[str, Any]) -> float:
    if summary.get("ours"):
        return float(summary.get("rewrite", {}).get("time_s", 0.0) + summary["ours"].get("build_time_s", 0.0))
    if summary.get("build_time_s") is not None:
        return float(summary.get("build_time_s", 0.0))
    if summary.get("exports"):
        return float(summary.get("rewrite", {}).get("time_s", 0.0) + summary.get("build_time_s", 0.0))
    return 0.0


def before_for_case(case: CaseSpec, out_dir: Path) -> dict[str, Any]:
    summary = load_json(case.before_summary)
    basis_results: dict[str, Any] = {}
    notes: dict[str, str] = {}
    for basis in ("nam", "ibm"):
        existing = extract_existing_basis(summary, basis)
        if existing is not None:
            existing["status"] = "ok_existing"
            existing["source"] = str(case.before_summary)
            basis_results[basis] = existing
            continue
        if case.before_high_qasm and case.before_high_qasm.exists():
            try:
                derive_dir = out_dir / "before_derived" / case.example
                derive_dir.mkdir(parents=True, exist_ok=True)
                derive_qasm = derive_dir / f"{case.example}_before_derived_{basis}_opt0.qasm"
                stats, transpile_time, qasm_path = derive_basis_from_high_qasm(
                    case.before_high_qasm,
                    derive_qasm,
                    basis=basis,
                )
                existing = basis_summary(
                    stats,
                    qasm_path,
                    transpile_time,
                    source_base_time(summary) + transpile_time,
                )
                existing["status"] = "ok_derived_from_existing_high_level_qasm"
                existing["source"] = str(case.before_high_qasm)
                basis_results[basis] = existing
                notes[basis] = "basis count derived by opt0 transpiling the preserved before high-level ours QASM"
            except Exception as exc:  # noqa: BLE001
                basis_results[basis] = {
                    "status": "missing_or_derivation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "source": str(case.before_summary),
                }
        else:
            basis_results[basis] = {"status": "missing", "source": str(case.before_summary)}
    return {
        "case": case.example,
        "family": case.family,
        "summary_path": str(case.before_summary),
        "high_level_qasm": str(case.before_high_qasm) if case.before_high_qasm else None,
        "basis_results": basis_results,
        "notes": notes,
        "case_notes": case.notes,
    }


def make_rows(before: dict[str, Any], after: dict[str, Any], case: CaseSpec) -> list[dict[str, Any]]:
    rows = []
    for basis in ("nam", "ibm"):
        b = before["basis_results"].get(basis, {})
        a = after.get("basis_results", {}).get(basis, {})
        before_gate = b.get("gate_count")
        after_gate = a.get("gate_count")
        before_non = b.get("non_clifford")
        after_non = a.get("non_clifford")
        row_status = "ok"
        if before_gate is None and after_gate is None:
            row_status = "before_after_missing"
        elif before_gate is None:
            row_status = "before_missing"
        elif after_gate is None:
            row_status = "after_missing"
        notes = "; ".join(
            part
            for part in [
                case.notes,
                before.get("notes", {}).get(basis),
                f"before_status={b.get('status')}" if b.get("status") else "",
                f"after_summary={rel(after.get('summary_path'))}",
            ]
            if part
        )
        rows.append(
            {
                "family": case.family,
                "example": case.example,
                "basis": basis,
                "before_gate_count": before_gate,
                "before_non_clifford": before_non,
                "before_compile_time_s": b.get("compile_time_s"),
                "after_gate_count": after_gate,
                "after_non_clifford": after_non,
                "after_compile_time_s": a.get("compile_time_s"),
                "delta_gate_count": None
                if before_gate is None or after_gate is None
                else int(after_gate) - int(before_gate),
                "delta_non_clifford": None
                if before_non is None or after_non is None
                else int(after_non) - int(before_non),
                "source_before": rel(b.get("source") or before.get("summary_path")),
                "source_after": rel(after.get("summary_path")),
                "status": row_status,
                "notes": notes,
            }
        )
    return rows


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# block_encoding_new rerun: structured-family ours rows",
        "",
        "Scope: graph cluster, Toom, decay higher-order, and repetition-code DQE. Gate counts are metric totals in the stated basis; non-Clifford counts use `baseline_common.py`.",
        "",
        "| Family | Example | Basis | Before gates | After gates | Delta | Before NC | After NC | Before time (s) | After time (s) | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {example} | {basis} | {before_gate_count} | {after_gate_count} | "
            "{delta_gate_count} | {before_non_clifford} | {after_non_clifford} | "
            "{before_compile_time_s} | {after_compile_time_s} | {status} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for row in rows:
        if row.get("notes"):
            lines.append(f"- `{row['example']}` `{row['basis']}`: {row['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "family",
        "example",
        "basis",
        "before_gate_count",
        "before_non_clifford",
        "before_compile_time_s",
        "after_gate_count",
        "after_non_clifford",
        "after_compile_time_s",
        "delta_gate_count",
        "delta_non_clifford",
        "source_before",
        "source_after",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(case: CaseSpec, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = before_for_case(case, out_dir)
    if case.kind.startswith("graph"):
        after = rerun_graph(case, out_dir)
    elif case.kind == "toom":
        after = rerun_toom(case, out_dir)
    elif case.kind == "decay":
        after = rerun_decay(case, out_dir)
    elif case.kind == "dqe":
        after = rerun_dqe(case, out_dir)
    else:
        raise ValueError(f"unknown case kind: {case.kind}")
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--only", nargs="*", default=None, help="Optional example-name filter.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [case for case in CASES if args.only is None or case.example in set(args.only)]

    all_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for case in selected:
        started = time.perf_counter()
        print(f"[run] {case.family}: {case.example}", flush=True)
        try:
            before, after = run_case(case, out_dir)
            rows = make_rows(before, after, case)
            all_rows.extend(rows)
            run_summaries.append(
                {
                    "case": case.example,
                    "family": case.family,
                    "status": "ok",
                    "elapsed_s": time.perf_counter() - started,
                    "before": before,
                    "after_summary": after,
                }
            )
            print(f"[ok] {case.example} elapsed={time.perf_counter() - started:.3f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            run_summaries.append(
                {
                    "case": case.example,
                    "family": case.family,
                    "status": "failed",
                    "elapsed_s": time.perf_counter() - started,
                    "error": error,
                }
            )
            for basis in ("nam", "ibm"):
                all_rows.append(
                    {
                        "family": case.family,
                        "example": case.example,
                        "basis": basis,
                        "before_gate_count": None,
                        "before_non_clifford": None,
                        "before_compile_time_s": None,
                        "after_gate_count": None,
                        "after_non_clifford": None,
                        "after_compile_time_s": None,
                        "delta_gate_count": None,
                        "delta_non_clifford": None,
                        "source_before": rel(case.before_summary),
                        "source_after": "",
                        "status": "failed",
                        "notes": error,
                    }
                )
            print(f"[failed] {case.example}: {error}", flush=True)

    csv_path = out_dir / "structured_ours_before_after.csv"
    md_path = out_dir / "structured_ours_before_after.md"
    json_path = out_dir / "structured_ours_before_after.json"
    write_csv(all_rows, csv_path)
    write_markdown(all_rows, md_path)
    json_path.write_text(
        json.dumps({"rows": all_rows, "runs": run_summaries}, indent=2, default=str),
        encoding="utf-8",
    )
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(all_rows, RESULTS_MD)
    write_csv(all_rows, RESULTS_CSV)

    print(f"[wrote] {csv_path}")
    print(f"[wrote] {md_path}")
    print(f"[wrote] {json_path}")
    print(f"[wrote] {RESULTS_MD}")
    print(f"[wrote] {RESULTS_CSV}")


if __name__ == "__main__":
    main()
