#!/usr/bin/env python3
"""Main-table SELECT-only comparison for Ours vs Qrisp.

This runner removes PREPARE/PREPARE-dagger from every example in the main
Ours/Qrisp block-encoding table.  The comparison target is only the SELECT
oracle:

* Ours: current ``block_encoding.BlockEncoding.mulplex_U_opt_order()``
  followed by the same large-controlled-gate lowering used by the full
  block-encoding path.
* Qrisp: direct ``q_switch(index, unitaries, system)`` using the same
  phase-bearing Pauli unitaries used by the legal ``from_lcu`` input.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding import (  # type: ignore  # noqa: E402
    BlockEncoding as CurrentBlockEncoding,
    expand_large_controlled_gates,
)
from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: E402
from qiskit import QuantumCircuit  # type: ignore  # noqa: E402
from qiskit.quantum_info import Pauli  # type: ignore  # noqa: E402
from run_block_encoding_new_qrisp_comparison import (  # type: ignore  # noqa: E402
    build_benchmark_current,
)
from run_dense_random_pauli4_block_encoding import (  # type: ignore  # noqa: E402
    full_matrixsum as dense_full_matrixsum,
    random_matrixsum as dense_random_matrixsum,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    install_aer_stub,
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from run_random_pauli_kraus_block_encoding import (  # type: ignore  # noqa: E402
    random_pauli_matrixsum,
)
from run_table3_extra_block_encoding_cases import (  # type: ignore  # noqa: E402
    build_tfim_a0,
    qrisp_pauli_unitary_with_phase,
    term_to_label_phase,
)
from sum_rewritten_kraus_block_counts_qrisp_vs_ours import (  # type: ignore  # noqa: E402
    channel_metrics,
    kraus_cache_key,
    kraus_effective_terms,
    kraus_terms,
    natural_width,
    rewrite_channel,
)


RESULT_ROOT = ROOT / "Baseline_results" / "table_main_select_only_ours_qrisp_v1"
RESULTS_MD = ROOT / "Results" / "Table_main_select_only_ours_qrisp.md"
RESULTS_TEX = ROOT / "Results" / "Table_main_select_only_ours_qrisp.tex"


REWRITTEN_CASES = (
    {
        "name": "HRW_8",
        "label": "HRW-8",
        "num_qubits": 8,
    },
    {
        "name": "HRW_28",
        "label": "HRW-28",
        "num_qubits": 28,
    },
    {
        "name": "random_coloring_linear_n4_q4",
        "label": "Coloring-linear",
        "num_qubits": 8,
    },
    {
        "name": "random_coloring_cycle_n4_q4",
        "label": "Coloring-cycle",
        "num_qubits": 8,
    },
    {
        "name": "repetition_dqe_d3",
        "label": "Repetition DQE d=3",
        "num_qubits": 3,
    },
)

SINGLE_CASES = (
    {
        "name": "TFIM_3_A0",
        "label": "TFIM-3 A0",
        "num_qubits": 3,
        "num_terms": 10,
        "builder": "tfim",
    },
    {
        "name": "TFIM_8_A0",
        "label": "TFIM-8 A0",
        "num_qubits": 8,
        "num_terms": 25,
        "builder": "tfim",
    },
    {
        "name": "full_pauli_n4",
        "label": "Full Pauli n=4",
        "num_qubits": 4,
        "num_terms": 256,
        "builder": "full_pauli_dense4_random_positive",
        "seed": 20560706,
    },
    {
        "name": "full_pauli_n6",
        "label": "Full Pauli n=6",
        "num_qubits": 6,
        "num_terms": 4096,
        "builder": "full_pauli_uniform",
    },
)

RANDOM_CASES = (
    {
        "name": "random_pauli_4q_20t",
        "label": "Random Pauli 4q/20t",
        "num_qubits": 4,
        "num_terms": 20,
        "samples": 10,
        "seed_base": 20260703,
        "builder": "dense_random_positive",
    },
    {
        "name": "random_pauli_4q_64t",
        "label": "Random Pauli 4q/64t",
        "num_qubits": 4,
        "num_terms": 64,
        "samples": 5,
        "seed_base": 20260706,
        "builder": "dense_random_positive",
    },
    {
        "name": "random_pauli_4q_100t",
        "label": "Random Pauli 4q/100t",
        "num_qubits": 4,
        "num_terms": 100,
        "samples": 5,
        "seed_base": 20290706,
        "builder": "dense_random_positive",
    },
    {
        "name": "random_pauli_4q_129t",
        "label": "Random Pauli 4q/129t",
        "num_qubits": 4,
        "num_terms": 129,
        "samples": 5,
        "seed_base": 20360706,
        "builder": "dense_random_positive",
    },
    {
        "name": "random_pauli_4q_164t",
        "label": "Random Pauli 4q/164t",
        "num_qubits": 4,
        "num_terms": 164,
        "samples": 5,
        "seed_base": 20390706,
        "builder": "dense_random_positive",
    },
    {
        "name": "random_pauli_4q_200t",
        "label": "Random Pauli 4q/200t",
        "num_qubits": 4,
        "num_terms": 200,
        "samples": 5,
        "seed_base": 20460706,
        "builder": "dense_random_positive",
    },
    {
        "name": "random_pauli_8q_80t",
        "label": "Random Pauli 8q/80t",
        "num_qubits": 8,
        "num_terms": 80,
        "samples": 10,
        "seed_base": 20360703,
        "builder": "legacy_uniform_support",
    },
)

MAIN_ROW_ORDER = (
    "HRW_8",
    "HRW_28",
    "random_coloring_linear_n4_q4",
    "random_coloring_cycle_n4_q4",
    "repetition_dqe_d3",
    "TFIM_3_A0",
    "TFIM_8_A0",
    "random_pauli_4q_20t",
    "random_pauli_4q_64t",
    "random_pauli_4q_100t",
    "random_pauli_4q_129t",
    "random_pauli_4q_164t",
    "random_pauli_4q_200t",
    "full_pauli_n4",
    "random_pauli_8q_80t",
    "full_pauli_n6",
)


def summarize_basis(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry or entry.get("status") != "ok" or entry.get("result") is None:
        return None
    stats = entry["result"]["stats"]
    return {
        "gate_count": int(stats["metric_total"]),
        "width": int(stats["num_qubits"]),
        "non_clifford": int(stats["clifford"]["non_clifford"]),
        "ops": stats["ops"],
        "qasm": entry["result"]["qasm"],
        "elapsed_seconds": float(entry["elapsed_seconds"]),
        "transpile_time_s": float(entry["result"]["transpile_time_s"]),
    }


def recount_bases(qc: QuantumCircuit, out_dir: Path, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for basis in ("nam", "ibm"):
        status, result, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                qc,
                basis_name=basis,
                out_path=out_dir / f"{prefix}_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        rows[basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": result,
            "error": error,
        }
    return rows


def matrixsum_metrics(ms: Matrixsum) -> dict[str, Any]:
    terms = kraus_effective_terms(ms)
    labels = [label for label, _coeff in terms]
    identity = "I" * ms.size
    return {
        "system_qubits": int(ms.size),
        "term_count": int(len(terms)),
        "unique_terms": int(len(set(labels))),
        "has_identity": bool(identity in labels),
        "l1_norm": float(sum(abs(coeff) for _label, coeff in terms)),
        "natural_width": int(natural_width(ms)),
        "terms_preview": [
            {"pauli": label, "coeff": repr(coeff)}
            for label, coeff in terms[: min(8, len(terms))]
        ],
    }


def full_pauli_matrixsum(num_qubits: int) -> Matrixsum:
    labels = ["".join(chars) for chars in itertools.product("IXYZ", repeat=num_qubits)]
    coeff = 1.0 / float(len(labels))
    return Matrixsum([(PauliAtom(label, phase=1.0), coeff) for label in labels])


def qiskit_pauli_unitary(label: str, phase: complex) -> QuantumCircuit:
    qc = QuantumCircuit(len(label), name=f"select_{label}")
    angle = math.atan2(phase.imag, phase.real)
    if abs(angle) > 1e-14:
        qc.global_phase += angle
    if any(char != "I" for char in label):
        qc.append(Pauli(label), range(len(label)))
    return qc.decompose()


def single_term_select_circuit(ms: Matrixsum) -> QuantumCircuit:
    terms = kraus_effective_terms(ms)
    if not terms:
        return QuantumCircuit(max(0, int(ms.size)), name="empty_select")
    label, coeff = terms[0]
    magnitude = abs(coeff)
    phase = 1.0 + 0.0j if magnitude == 0 else coeff / magnitude
    return qiskit_pauli_unitary(label, phase)


def count_ours_select_only(ms: Matrixsum, out_dir: Path, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "ours_select_only_matrix_order",
        "term_count": int(len(ms.instances)),
        "natural_width": int(natural_width(ms)),
    }
    started = time.perf_counter()
    try:
        if len(ms.instances) <= 1:
            qc = single_term_select_circuit(ms)
            raw_select = qc
            analytical = {
                "tcount": 0,
                "mcx_like": 0,
                "cx": 0,
            }
            ctrl_size = 0
            be = None
        else:
            be = CurrentBlockEncoding(copy.deepcopy(ms))
            raw_select, tcount, mccount, cxcount, ctrl_size = be.mulplex_U_opt_order()
            qc = expand_large_controlled_gates(raw_select, if_optimize=True)
            analytical = {
                "tcount": int(tcount),
                "mcx_like": int(mccount),
                "cx": int(cxcount),
            }
    except Exception:
        result["status"] = "build_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        result["error"] = traceback.format_exc()
        return result

    result.update(
        {
            "status": "ok",
            "ctrl_size": int(ctrl_size),
            "selected_k": int(getattr(be, "selected_k", ctrl_size)) if be is not None else 0,
            "selected_crit": str(getattr(be, "selected_crit", "")) if be is not None else "single-term",
            "placement_search_debug": getattr(be, "placement_search_debug", {}) if be is not None else {},
            "placement_width_analysis": getattr(be, "placement_width_analysis", {}) if be is not None else {},
            "width_candidate_costs": getattr(be, "width_candidate_costs", []) if be is not None else [],
            "analytical": analytical,
            "raw_select_metrics": qiskit_metrics(raw_select),
            "qiskit_metrics": qiskit_metrics(qc),
            "qasm": write_qasm2(qc, out_dir / f"{prefix}_logical.qasm"),
        }
    )
    result["basis_recounts"] = recount_bases(qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def qrisp_terms(ms: Matrixsum) -> list[tuple[str, complex]]:
    terms = []
    for instance, coeff in copy.deepcopy(ms).instances:
        label, atom_phase = term_to_label_phase(instance)
        effective = complex(coeff) * atom_phase
        magnitude = abs(effective)
        phase = 1.0 + 0.0j if magnitude == 0 else effective / magnitude
        terms.append((label, phase))
    return terms


def count_qrisp_select_only(
    ms: Matrixsum,
    out_dir: Path,
    prefix: str,
    *,
    workspace: int,
) -> dict[str, Any]:
    import qrisp
    from qrisp import QuantumFloat, QuantumVariable
    from qrisp.jasp import q_switch

    terms = qrisp_terms(ms)
    index_width = int(math.ceil(math.log2(max(1, len(terms)))))
    result: dict[str, Any] = {
        "method": "qrisp_select_only_q_switch",
        "term_count": int(len(terms)),
        "index_width": int(index_width),
        "workspace": int(workspace),
        "qrisp_version": getattr(qrisp, "__version__", None),
        "qrisp_file": getattr(qrisp, "__file__", None),
    }

    started = time.perf_counter()
    try:
        sys_qv = QuantumVariable(ms.size, name="sys")
        unitaries = [
            qrisp_pauli_unitary_with_phase(label, phase)
            for label, phase in terms
        ]
        if len(unitaries) == 0:
            pass
        elif len(unitaries) == 1:
            unitaries[0](sys_qv)
        else:
            idx = QuantumFloat(index_width, name="idx")
            q_switch(idx, unitaries, sys_qv)
        compiled = sys_qv.qs.compile(workspace=workspace)
        qc = compiled.to_qiskit()
    except Exception:
        result["status"] = "build_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        result["error"] = traceback.format_exc()
        return result

    result["status"] = "ok"
    result["compiled_qrisp_metrics"] = {
        "num_qubits": int(compiled.num_qubits()),
        "depth": int(compiled.depth()),
        "ops": {str(k): int(v) for k, v in compiled.count_ops().items()},
    }
    result["qiskit_metrics"] = qiskit_metrics(qc)
    result["qasm"] = write_qasm2(qc, out_dir / f"{prefix}_qiskit.qasm")
    result["basis_recounts"] = recount_bases(qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def empty_totals() -> dict[str, dict[str, Any]]:
    return {
        basis: {
            "gate_count": 0,
            "non_clifford": 0,
            "max_width": 0,
            "completed": 0,
            "missing": 0,
            "elapsed_seconds_unique": 0.0,
            "elapsed_seconds_per_occurrence_sum": 0.0,
        }
        for basis in ("nam", "ibm")
    }


def add_result_to_totals(totals: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
    for basis in ("nam", "ibm"):
        elapsed = float(result.get("elapsed_seconds", 0.0))
        totals[basis]["elapsed_seconds_per_occurrence_sum"] += elapsed
        if not result.get("cache_hit"):
            totals[basis]["elapsed_seconds_unique"] += elapsed
        basis_summary = result.get("basis_summary", {}).get(basis)
        if result.get("status") != "ok" or basis_summary is None:
            totals[basis]["missing"] += 1
            continue
        totals[basis]["gate_count"] += int(basis_summary["gate_count"])
        totals[basis]["non_clifford"] += int(basis_summary["non_clifford"])
        totals[basis]["max_width"] = max(totals[basis]["max_width"], int(basis_summary["width"]))
        totals[basis]["completed"] += 1


def row_from_totals(
    spec: dict[str, Any],
    term_count: int,
    totals: dict[str, dict[str, dict[str, Any]]],
    basis: str,
) -> dict[str, Any]:
    ours = totals["ours_select_only_matrix_order"][basis]
    qrisp = totals["qrisp_select_only_q_switch"][basis]
    row = {
        "case": spec["name"],
        "label": spec["label"],
        "basis": basis,
        "num_qubits": int(spec["num_qubits"]),
        "term_count": int(term_count),
        "scope": "sum rewritten Kraus SELECT",
        "sampled": False,
        "ours": {
            "gate_count": int(ours["gate_count"]) if ours["completed"] > 0 else None,
            "non_clifford": int(ours["non_clifford"]) if ours["completed"] > 0 else None,
            "width": int(ours["max_width"]) if ours["completed"] > 0 else None,
            "completed": int(ours["completed"]),
            "missing": int(ours["missing"]),
            "elapsed_seconds": float(ours["elapsed_seconds_unique"]),
        },
        "qrisp": {
            "gate_count": int(qrisp["gate_count"]) if qrisp["completed"] > 0 else None,
            "non_clifford": int(qrisp["non_clifford"]) if qrisp["completed"] > 0 else None,
            "width": int(qrisp["max_width"]) if qrisp["completed"] > 0 else None,
            "completed": int(qrisp["completed"]),
            "missing": int(qrisp["missing"]),
            "elapsed_seconds": float(qrisp["elapsed_seconds_unique"]),
        },
    }
    attach_ratios(row)
    return row


def attach_ratios(row: dict[str, Any]) -> None:
    ours_gate = row["ours"].get("gate_count")
    qrisp_gate = row["qrisp"].get("gate_count")
    ours_nc = row["ours"].get("non_clifford")
    qrisp_nc = row["qrisp"].get("non_clifford")
    row["ratio_gate"] = (
        None if ours_gate in (None, 0) or qrisp_gate is None else float(qrisp_gate / ours_gate)
    )
    row["ratio_non_clifford"] = (
        None if ours_nc in (None, 0) or qrisp_nc is None else float(qrisp_nc / ours_nc)
    )


def run_rewritten_case(
    spec: dict[str, Any],
    out_dir: Path,
    *,
    workspace: int,
    rewrite_max_steps: int,
    rewrite_beam_width: int,
    rewrite_tol: float,
) -> dict[str, Any]:
    bench_dir = out_dir / "rewritten_kraus" / spec["name"]
    bench_dir.mkdir(parents=True, exist_ok=True)

    raw_kraus_ops, model_info = build_benchmark_current(spec["name"])
    rewritten_kraus_ops, rewrite_result, rewrite_seconds = rewrite_channel(
        raw_kraus_ops,
        max_steps=rewrite_max_steps,
        beam_width=rewrite_beam_width,
        tol=rewrite_tol,
    )
    nonzero = [
        (index, ms)
        for index, ms in enumerate(rewritten_kraus_ops)
        if len(ms.instances) > 0
    ]
    totals = {
        "ours_select_only_matrix_order": empty_totals(),
        "qrisp_select_only_q_switch": empty_totals(),
    }
    summary: dict[str, Any] = {
        "case": spec,
        "model_info": model_info,
        "raw_channel_metrics": channel_metrics(raw_kraus_ops),
        "rewritten_channel_metrics": channel_metrics(rewritten_kraus_ops),
        "rewrite": {
            "elapsed_seconds": float(rewrite_seconds),
            "result": rewrite_result,
        },
        "kraus_results": [],
        "totals": totals,
        "rows": [],
    }

    ours_cache: dict[str, dict[str, Any]] = {}
    qrisp_cache: dict[str, dict[str, Any]] = {}
    for ordinal, (kraus_index, ms) in enumerate(nonzero):
        key = kraus_cache_key(ms)
        prefix = f"{spec['name']}_kraus{kraus_index:04d}"
        if key in ours_cache:
            ours = copy.deepcopy(ours_cache[key])
            ours["cache_hit"] = True
        else:
            ours = count_ours_select_only(ms, bench_dir, prefix + "_ours_select")
            ours["cache_hit"] = False
            ours_cache[key] = copy.deepcopy(ours)
        if key in qrisp_cache:
            qrisp = copy.deepcopy(qrisp_cache[key])
            qrisp["cache_hit"] = True
        else:
            qrisp = count_qrisp_select_only(
                ms,
                bench_dir,
                prefix + "_qrisp_select",
                workspace=workspace,
            )
            qrisp["cache_hit"] = False
            qrisp_cache[key] = copy.deepcopy(qrisp)

        add_result_to_totals(totals["ours_select_only_matrix_order"], ours)
        add_result_to_totals(totals["qrisp_select_only_q_switch"], qrisp)
        summary["kraus_results"].append(
            {
                "ordinal": int(ordinal),
                "kraus_index": int(kraus_index),
                "term_count": int(len(ms.instances)),
                "natural_width": int(natural_width(ms)),
                "pauli_norm": float(ms.pauli_norm()),
                "terms_preview": kraus_terms(ms)[:8],
                "ours_select_only_matrix_order": ours,
                "qrisp_select_only_q_switch": qrisp,
            }
        )

    term_count = int(summary["rewritten_channel_metrics"]["pauli_terms_total"])
    summary["rows"] = [
        row_from_totals(spec, term_count, totals, basis)
        for basis in ("nam", "ibm")
    ]
    return summary


def build_single_matrixsum(spec: dict[str, Any]) -> Matrixsum:
    if spec["builder"] == "tfim":
        num_qubits = int(spec["num_qubits"])
        _name, ms, _metadata = build_tfim_a0(num_qubits)
        return ms
    if spec["builder"] == "full_pauli_dense4_random_positive":
        return dense_full_matrixsum(
            int(spec["num_qubits"]),
            int(spec["seed"]),
            coeff_mode="random-positive",
        )
    if spec["builder"] == "full_pauli_uniform":
        return full_pauli_matrixsum(int(spec["num_qubits"]))
    raise ValueError(f"unknown single-case builder: {spec['builder']!r}")


def row_from_result(
    spec: dict[str, Any],
    result_ours: dict[str, Any],
    result_qrisp: dict[str, Any],
    basis: str,
    *,
    term_count: int,
    scope: str,
) -> dict[str, Any]:
    ours_summary = result_ours.get("basis_summary", {}).get(basis)
    qrisp_summary = result_qrisp.get("basis_summary", {}).get(basis)
    row = {
        "case": spec["name"],
        "label": spec["label"],
        "basis": basis,
        "num_qubits": int(spec["num_qubits"]),
        "term_count": int(term_count),
        "scope": scope,
        "sampled": False,
        "ours": {
            "gate_count": None if ours_summary is None else int(ours_summary["gate_count"]),
            "non_clifford": None if ours_summary is None else int(ours_summary["non_clifford"]),
            "width": None if ours_summary is None else int(ours_summary["width"]),
            "elapsed_seconds": float(result_ours.get("elapsed_seconds", 0.0)),
            "status": result_ours.get("status"),
        },
        "qrisp": {
            "gate_count": None if qrisp_summary is None else int(qrisp_summary["gate_count"]),
            "non_clifford": None if qrisp_summary is None else int(qrisp_summary["non_clifford"]),
            "width": None if qrisp_summary is None else int(qrisp_summary["width"]),
            "elapsed_seconds": float(result_qrisp.get("elapsed_seconds", 0.0)),
            "status": result_qrisp.get("status"),
        },
    }
    attach_ratios(row)
    return row


def run_single_case(spec: dict[str, Any], out_dir: Path, *, workspace: int) -> dict[str, Any]:
    case_dir = out_dir / "single" / spec["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    ms = build_single_matrixsum(spec)
    metrics = matrixsum_metrics(ms)
    ours = count_ours_select_only(ms, case_dir, f"{spec['name']}_ours_select")
    qrisp = count_qrisp_select_only(
        ms,
        case_dir,
        f"{spec['name']}_qrisp_select",
        workspace=workspace,
    )
    return {
        "case": spec,
        "metrics": metrics,
        "ours_select_only_matrix_order": ours,
        "qrisp_select_only_q_switch": qrisp,
        "rows": [
            row_from_result(
                spec,
                ours,
                qrisp,
                basis,
                term_count=int(metrics["term_count"]),
                scope="single SELECT",
            )
            for basis in ("nam", "ibm")
        ],
    }


def build_random_matrixsum(spec: dict[str, Any], seed: int) -> Matrixsum:
    if spec["builder"] == "dense_random_positive":
        return dense_random_matrixsum(
            int(spec["num_qubits"]),
            int(spec["num_terms"]),
            seed,
            coeff_mode="random-positive",
        )
    if spec["builder"] == "legacy_uniform_support":
        return random_pauli_matrixsum(
            int(spec["num_qubits"]),
            int(spec["num_terms"]),
            seed,
        )
    raise ValueError(f"unknown random-case builder: {spec['builder']!r}")


def aggregate(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(statistics.fmean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_random_rows(spec: dict[str, Any], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for basis in ("nam", "ibm"):
        buckets: dict[str, dict[str, list[float]]] = {
            "ours": {"gate_count": [], "non_clifford": [], "width": [], "elapsed_seconds": []},
            "qrisp": {"gate_count": [], "non_clifford": [], "width": [], "elapsed_seconds": []},
        }
        for sample in samples:
            for target, key in (
                ("ours", "ours_select_only_matrix_order"),
                ("qrisp", "qrisp_select_only_q_switch"),
            ):
                result = sample[key]
                summary = result.get("basis_summary", {}).get(basis)
                if result.get("status") != "ok" or summary is None:
                    continue
                buckets[target]["gate_count"].append(float(summary["gate_count"]))
                buckets[target]["non_clifford"].append(float(summary["non_clifford"]))
                buckets[target]["width"].append(float(summary["width"]))
                buckets[target]["elapsed_seconds"].append(float(result.get("elapsed_seconds", 0.0)))
        row = {
            "case": spec["name"],
            "label": spec["label"],
            "basis": basis,
            "num_qubits": int(spec["num_qubits"]),
            "term_count": int(spec["num_terms"]),
            "scope": f"avg. {spec['samples']} random SELECTs",
            "sampled": True,
            "samples": int(spec["samples"]),
            "ours": {
                name: aggregate(vals)
                for name, vals in buckets["ours"].items()
            },
            "qrisp": {
                name: aggregate(vals)
                for name, vals in buckets["qrisp"].items()
            },
        }
        ours_gate = row["ours"]["gate_count"]["mean"]
        qrisp_gate = row["qrisp"]["gate_count"]["mean"]
        ours_nc = row["ours"]["non_clifford"]["mean"]
        qrisp_nc = row["qrisp"]["non_clifford"]["mean"]
        row["ratio_gate"] = None if not ours_gate or qrisp_gate is None else float(qrisp_gate / ours_gate)
        row["ratio_non_clifford"] = None if not ours_nc or qrisp_nc is None else float(qrisp_nc / ours_nc)
        row["completed_samples"] = {
            "ours": int(row["ours"]["gate_count"]["count"]),
            "qrisp": int(row["qrisp"]["gate_count"]["count"]),
        }
        rows.append(row)
    return rows


def run_random_case(spec: dict[str, Any], out_dir: Path, *, workspace: int) -> dict[str, Any]:
    case_dir = out_dir / "random" / spec["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for sample_index in range(int(spec["samples"])):
        seed = int(spec["seed_base"] + sample_index)
        sample_dir = case_dir / f"sample{sample_index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        ms = build_random_matrixsum(spec, seed)
        metrics = matrixsum_metrics(ms)
        if metrics["unique_terms"] != metrics["term_count"]:
            raise RuntimeError(f"duplicate Pauli term in {spec['name']} seed {seed}: {metrics}")
        if metrics["has_identity"]:
            raise RuntimeError(f"non-full random case contains identity in {spec['name']} seed {seed}")
        prefix = f"{spec['name']}_sample{sample_index:02d}_seed{seed}"
        ours = count_ours_select_only(ms, sample_dir, prefix + "_ours_select")
        qrisp = count_qrisp_select_only(
            ms,
            sample_dir,
            prefix + "_qrisp_select",
            workspace=workspace,
        )
        samples.append(
            {
                "sample_index": int(sample_index),
                "seed": int(seed),
                "metrics": metrics,
                "ours_select_only_matrix_order": ours,
                "qrisp_select_only_q_switch": qrisp,
            }
        )
    return {
        "case": spec,
        "samples": samples,
        "rows": aggregate_random_rows(spec, samples),
    }


def collect_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_case_basis = {}
    for group_name in ("rewritten_kraus", "single", "random"):
        for case_summary in summary[group_name]["cases"]:
            for row in case_summary["rows"]:
                by_case_basis[(row["case"], row["basis"])] = row
    rows = []
    for case_name in MAIN_ROW_ORDER:
        for basis in ("nam", "ibm"):
            row = by_case_basis.get((case_name, basis))
            if row is not None:
                rows.append(row)
    return rows


def scalar_metric(row: dict[str, Any], method: str, metric: str) -> Any:
    value = row[method][metric]
    if isinstance(value, dict):
        return value["mean"]
    return value


def fmt_number(value: Any, *, digits: int = 1) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):,}"
    return f"{numeric:,.{digits}f}"


def fmt_metric(row: dict[str, Any], method: str, metric: str) -> str:
    value = row[method][metric]
    if isinstance(value, dict):
        return fmt_number(value["mean"], digits=1)
    return fmt_number(value, digits=1)


def fmt_ratio(value: Any) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if numeric >= 100:
        return f"{numeric:.3g}"
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def write_report(summary: dict[str, Any], path: Path) -> None:
    rows = collect_table_rows(summary)
    sections = [
        "# Main-Table SELECT-Only Ours vs Qrisp",
        "",
        "All rows remove PREPARE/PREPARE-dagger and count only the SELECT oracle.",
        "HRW, coloring, and repetition rows sum SELECT costs over all nonempty rewritten Kraus operators.",
        "Random rows report sample averages using the same supports as the full block-encoding runs.",
        "",
    ]
    for basis in ("nam", "ibm"):
        basis_rows = []
        for row in rows:
            if row["basis"] != basis:
                continue
            basis_rows.append(
                [
                    row["label"],
                    row["num_qubits"],
                    f"{int(row['term_count']):,}",
                    fmt_metric(row, "ours", "gate_count"),
                    fmt_metric(row, "ours", "non_clifford"),
                    fmt_metric(row, "ours", "width"),
                    fmt_metric(row, "qrisp", "gate_count"),
                    fmt_metric(row, "qrisp", "non_clifford"),
                    fmt_metric(row, "qrisp", "width"),
                    fmt_ratio(row["ratio_gate"]),
                    fmt_ratio(row["ratio_non_clifford"]),
                ]
            )
        sections.extend(
            [
                f"## {basis.upper()}",
                "",
                markdown_table(
                    [
                        "Example",
                        "n",
                        "Terms",
                        "Ours Total",
                        "Ours NC",
                        "Ours Width",
                        "Qrisp Total",
                        "Qrisp NC",
                        "Qrisp Width",
                        "Ratio Total",
                        "Ratio NC",
                    ],
                    basis_rows,
                ),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")


def write_tex(summary: dict[str, Any], path: Path) -> None:
    rows = [row for row in collect_table_rows(summary) if row["basis"] == "nam"]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{NAM-basis SELECT-only cost comparison. PREPARE and PREPARE$^{\\dagger}$ are removed from every row. HRW, coloring, and repetition rows sum over all nonempty rewritten Kraus operators; random rows report sample averages. Ratio is Qrisp/Ours. NC denotes non-Clifford NAM gates.}",
        "\\label{tab:nam-select-only-ours-qrisp-main}",
        "\\begin{tabular}{lrrrrrrrrrr}",
        "\\toprule",
        "& & & \\multicolumn{3}{c}{Ours} & \\multicolumn{3}{c}{Qrisp} & \\multicolumn{2}{c}{Ratio} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}",
        "Example & $n$ & Terms",
        "& Total & NC & Width",
        "& Total & NC & Width",
        "& Total & NC \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['label'])} & {row['num_qubits']} & {int(row['term_count']):,} & "
            f"{fmt_metric(row, 'ours', 'gate_count')} & "
            f"{fmt_metric(row, 'ours', 'non_clifford')} & "
            f"{fmt_metric(row, 'ours', 'width')} & "
            f"{fmt_metric(row, 'qrisp', 'gate_count')} & "
            f"{fmt_metric(row, 'qrisp', 'non_clifford')} & "
            f"{fmt_metric(row, 'qrisp', 'width')} & "
            f"{fmt_ratio(row['ratio_gate'])} & "
            f"{fmt_ratio(row['ratio_non_clifford'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_partial(summary: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "summary.partial.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.partial.md")


def case_allowed(name: str, only: set[str] | None) -> bool:
    if only is None:
        return True
    return name in only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--workspace", type=int, default=0)
    parser.add_argument("--rewrite-max-steps", type=int, default=50)
    parser.add_argument("--rewrite-beam-width", type=int, default=3)
    parser.add_argument("--rewrite-tol", type=float, default=1e-12)
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional subset of case names, e.g. TFIM_3_A0 HRW_8 full_pauli_n4.",
    )
    parser.add_argument(
        "--skip-results-write",
        action="store_true",
        help="Only write into --out-dir; do not update paper-facing Results files.",
    )
    args = parser.parse_args()

    install_aer_stub()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only) if args.only else None
    summary: dict[str, Any] = {
        "experiment": "table_main_select_only_ours_qrisp",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "policy": {
            "scope": "SELECT only for every example; no coefficient state preparation",
            "ours": "block_encoding.BlockEncoding.mulplex_U_opt_order + expand_large_controlled_gates",
            "qrisp": "qrisp.jasp.q_switch with phase-bearing Pauli unitaries",
        },
        "rewritten_kraus": {"cases": []},
        "single": {"cases": []},
        "random": {"cases": []},
    }

    for spec in REWRITTEN_CASES:
        if not case_allowed(spec["name"], only):
            continue
        summary["rewritten_kraus"]["cases"].append(
            run_rewritten_case(
                spec,
                out_dir,
                workspace=args.workspace,
                rewrite_max_steps=args.rewrite_max_steps,
                rewrite_beam_width=args.rewrite_beam_width,
                rewrite_tol=args.rewrite_tol,
            )
        )
        write_partial(summary, out_dir)

    for spec in SINGLE_CASES:
        if not case_allowed(spec["name"], only):
            continue
        summary["single"]["cases"].append(
            run_single_case(spec, out_dir, workspace=args.workspace)
        )
        write_partial(summary, out_dir)

    for spec in RANDOM_CASES:
        if not case_allowed(spec["name"], only):
            continue
        summary["random"]["cases"].append(
            run_random_case(spec, out_dir, workspace=args.workspace)
        )
        write_partial(summary, out_dir)

    summary["table_rows"] = collect_table_rows(summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(summary, out_dir / "report.md")
    if not args.skip_results_write:
        write_report(summary, RESULTS_MD)
        write_tex(summary, RESULTS_TEX)
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"report: {out_dir / 'report.md'}")
    if not args.skip_results_write:
        print(f"results md: {RESULTS_MD}")
        print(f"results tex: {RESULTS_TEX}")
    print(json.dumps(summary["table_rows"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
