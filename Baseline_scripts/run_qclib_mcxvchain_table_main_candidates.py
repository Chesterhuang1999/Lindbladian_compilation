#!/usr/bin/env python3
"""Run QClib McxVchainDirty baselines for Table_main candidates.

The experiment records both the row-wise QClib SELECT multiplexor and the
shared Qrisp PREPARE/PREPARE^dag component for traceability. The paper-facing
QClib table reports the McxVchainDirty SELECT object only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from qiskit import QuantumCircuit


ROOT = Path(__file__).resolve().parents[1]
QCLIB_SITE = Path(
    "/Users/chester_hqf/miniconda3/envs/ChannelIR_test/lib/python3.12/site-packages"
)
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts", QCLIB_SITE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from block_encoding_new import BlockEncoding as CurrentBlockEncoding  # type: ignore  # noqa: E402
from compare_tfim4_a0_select_qclib_mcxvchain_mode_a import (  # type: ignore  # noqa: E402
    append_row_phase_sidecar_qclib,
    apply_pauli_basis_sidecar,
    ctrl_state_for_index,
    qiskit_pauli_targets,
    required_work_qubits,
)
from run_block_encoding_new_qrisp_comparison import (  # type: ignore  # noqa: E402
    build_benchmark_current,
    build_tfim_a0,
    count_ours_block_encoding_new,
    rewrite_channel,
)
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    timed,
    write_qasm2,
)
from run_random_pauli_kraus_block_encoding import (  # type: ignore  # noqa: E402
    CASES as RANDOM_PAULI_CASES,
    random_pauli_matrixsum,
    validate_matrixsum,
)
from run_table3_extra_block_encoding_cases import (  # type: ignore  # noqa: E402
    count_qrisp_from_lcu_phase_unitaries,
    full_pauli_matrixsum,
)
from sum_rewritten_kraus_block_counts_qrisp_vs_ours import (  # type: ignore  # noqa: E402
    channel_metrics,
    kraus_cache_key,
    kraus_terms,
    natural_width,
)


OUT_DIR = ROOT / "Baseline_results" / "qclib_mcxvchain_table_main_candidates_v1"
RESULT_NAM_MD = ROOT / "Results" / "Table_main_nam_ours_qclib_ratio.md"
RESULT_NAM_TEX = ROOT / "Results" / "Table_main_nam_ours_qclib_ratio.tex"
RESULT_IBM_MD = ROOT / "Results" / "Table_main_ibm_ours_qclib_ratio.md"
RESULT_IBM_TEX = ROOT / "Results" / "Table_main_ibm_ours_qclib_ratio.tex"
SELECT_ONLY_OURS_QRISP_MD = ROOT / "Results" / "Table_main_select_only_ours_qrisp.md"


MAIN_ROWS = [
    {
        "key": "HRW_8",
        "label": "HRW-8",
        "n": 8,
        "terms": 32,
        "scope": "sum Kraus",
        "kind": "rewritten",
    },
    {
        "key": "HRW_28",
        "label": "HRW-28",
        "n": 28,
        "terms": 112,
        "scope": "sum Kraus",
        "kind": "rewritten",
    },
    {
        "key": "random_coloring_linear_n4_q4",
        "label": "Coloring-linear",
        "n": 8,
        "terms": 640,
        "scope": "sum Kraus",
        "kind": "rewritten",
    },
    {
        "key": "random_coloring_cycle_n4_q4",
        "label": "Coloring-cycle",
        "n": 8,
        "terms": 1024,
        "scope": "sum Kraus",
        "kind": "rewritten",
    },
    {
        "key": "repetition_dqe_d3",
        "label": "Repetition DQE d=3",
        "n": 3,
        "terms": 64,
        "scope": "sum Kraus",
        "kind": "rewritten",
    },
    {
        "key": "TFIM_3_A0",
        "label": "TFIM-3 A0",
        "n": 3,
        "terms": 10,
        "scope": "A0",
        "kind": "tfim",
    },
    {
        "key": "TFIM_8_A0",
        "label": "TFIM-8 A0",
        "n": 8,
        "terms": 25,
        "scope": "A0",
        "kind": "tfim",
    },
    {
        "key": "random_pauli_4q_20t",
        "label": "Random Pauli 4q/20t",
        "n": 4,
        "terms": 20,
        "scope": "avg. 10",
        "kind": "random_pauli",
    },
    {
        "key": "random_pauli_8q_80t",
        "label": "Random Pauli 8q/80t",
        "n": 8,
        "terms": 80,
        "scope": "avg. 10",
        "kind": "random_pauli",
    },
    {
        "key": "full_pauli_n6",
        "label": "Full Pauli n=6",
        "n": 6,
        "terms": 4096,
        "scope": "single",
        "kind": "full_pauli",
    },
]


TABLE_EXCLUDED_KEYS = {
    "HRW_8",
    "TFIM_3_A0",
    "random_coloring_linear_n4_q4",
    "random_pauli_4q_20t",
}
TABLE_ROWS = [row for row in MAIN_ROWS if row["key"] not in TABLE_EXCLUDED_KEYS]


def effective_terms(ms: Any) -> list[tuple[str, float, complex, complex]]:
    terms: list[tuple[str, float, complex, complex]] = []
    for instance, coeff in copy.deepcopy(ms).instances:
        if hasattr(instance, "expr") and hasattr(instance, "phase"):
            label = str(instance.expr)
            atom_phase = complex(instance.phase)
        else:
            label_method = getattr(instance, "to_label", None)
            label = str(label_method()) if callable(label_method) else repr(instance)
            atom_phase = 1.0 + 0.0j
        effective = complex(coeff) * atom_phase
        magnitude = abs(effective)
        phase = 1.0 + 0.0j if magnitude <= 1e-15 else effective / magnitude
        terms.append((label, float(magnitude), phase, effective))
    return terms


def address_size(num_rows: int) -> int:
    if num_rows < 1:
        raise ValueError("num_rows must be positive")
    return 0 if num_rows == 1 else int(math.ceil(math.log2(num_rows)))


def complex_record(value: complex, tol: float = 1e-12) -> dict[str, Any]:
    z = complex(value)
    real = 0.0 if abs(z.real) < tol else float(z.real)
    imag = 0.0 if abs(z.imag) < tol else float(z.imag)
    return {
        "real": real,
        "imag": imag,
        "repr": repr(complex(real, imag)),
        "angle_pi": float(np.angle(z) / np.pi),
    }


def build_qclib_mode_a_select(ms: Any) -> tuple[QuantumCircuit, dict[str, Any], list[dict[str, Any]]]:
    from qclib.gates.mcx import McxVchainDirty

    terms = effective_terms(ms)
    if not terms:
        raise ValueError("empty Matrixsum is not supported")
    labels = [label for label, _magnitude, _phase, _effective in terms]
    m = int(ms.size)
    k = address_size(len(terms))
    a = required_work_qubits(k)

    qc = QuantumCircuit(k + m + a, name=f"qclib_mode_a_select_{m}q_{len(terms)}t")
    address_qubits = list(range(k))
    system_qubits = list(range(k, k + m))
    work_qubits = list(range(k + m, k + m + a))
    stats = {
        "num_rows": int(len(terms)),
        "address_qubits": int(k),
        "system_qubits": int(m),
        "work_qubits_mode_a_clean": int(a),
        "full_width_mode_a_clean": int(k + m + a),
        "identity_rows": 0,
        "mcxvchain_rows": 0,
        "qclib_mcg_phase_sidecars": 0,
        "single_control_phase_sidecars": 0,
        "global_phase_only": 0,
        "phase_open_control_x_conjugations": 0,
        "basis_h_pairs": 0,
        "basis_sdg_s_pairs": 0,
        "total_x_like_targets": 0,
        "max_row_weight": 0,
        "num_X": int(sum(p.count("X") for p in labels)),
        "num_Y": int(sum(p.count("Y") for p in labels)),
        "num_Z": int(sum(p.count("Z") for p in labels)),
        "num_I": int(sum(p.count("I") for p in labels)),
    }
    rows = []
    for row_index, (pauli, magnitude, phase, effective) in enumerate(terms):
        ctrl_state = ctrl_state_for_index(row_index, k, address_lsb_first=True)
        targets = qiskit_pauli_targets(pauli, k)
        weight = len(targets)
        append_row_phase_sidecar_qclib(qc, address_qubits, ctrl_state, phase, stats)
        row_info = {
            "address": int(row_index),
            "ctrl_state": ctrl_state,
            "pauli": pauli,
            "magnitude": float(magnitude),
            "phase": complex_record(phase),
            "effective": complex_record(effective),
            "weight": int(weight),
            "targets": targets,
            "mcxvchain_definition_width": None,
        }
        if weight == 0:
            stats["identity_rows"] += 1
            rows.append(row_info)
            continue
        for target in targets:
            letter = str(target["pauli"])
            if letter == "Z":
                stats["basis_h_pairs"] += 1
            elif letter == "Y":
                stats["basis_sdg_s_pairs"] += 1
        apply_pauli_basis_sidecar(qc, system_qubits, targets, inverse=False)
        sub = McxVchainDirty(
            num_controls=k,
            num_target_qubit=weight,
            ctrl_state=ctrl_state,
            relative_phase=False,
            action_only=False,
        ).definition
        row_info["mcxvchain_definition_width"] = int(sub.num_qubits)
        qc.compose(
            sub,
            list(address_qubits)
            + list(work_qubits[:a])
            + [system_qubits[int(t["sys_qubit"])] for t in targets],
            inplace=True,
        )
        apply_pauli_basis_sidecar(qc, system_qubits, targets, inverse=True)
        stats["mcxvchain_rows"] += 1
        stats["total_x_like_targets"] += weight
        stats["max_row_weight"] = max(stats["max_row_weight"], weight)
        rows.append(row_info)

    return qc, {"layout": {"address": address_qubits, "system": system_qubits, "work": work_qubits}, "stats": stats}, rows


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


def count_qclib_select(ms: Any, out_dir: Path, prefix: str) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "method": "qclib_mcxvchain_mode_a_select_only",
        "term_count": int(len(ms.instances)),
        "natural_width": int(natural_width(ms)),
    }
    try:
        qc, metadata, rows = build_qclib_mode_a_select(ms)
    except Exception:
        result["status"] = "build_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        result["error"] = traceback.format_exc()
        return result
    result["status"] = "ok"
    result["build_seconds"] = float(time.perf_counter() - started)
    result["metadata"] = metadata
    result["rows_preview"] = rows[: min(16, len(rows))]
    result["qiskit_metrics"] = qiskit_metrics(qc)
    result["qasm"] = write_qasm2(qc, out_dir / f"{prefix}_logical.qasm")
    result["basis_recounts"] = recount_bases(qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def count_qrisp_prepare_pair(ms: Any, out_dir: Path, prefix: str, workspace: int) -> dict[str, Any]:
    from qrisp import QuantumFloat
    from qrisp.alg_primitives.state_preparation import prepare
    from qrisp.environments import invert

    terms = effective_terms(ms)
    coeffs = [magnitude for _label, magnitude, _phase, _effective in terms]
    m = len(coeffs)
    k = address_size(m)
    padded = np.pad(np.asarray(coeffs, dtype=float), (0, (1 << k) - m))
    alpha = float(np.sum(padded))
    amps = np.sqrt(padded / alpha) if alpha > 0 else padded

    result: dict[str, Any] = {
        "method": "qrisp_prepare_pair_only",
        "workspace": int(workspace),
        "term_count": int(m),
        "index_width": int(k),
        "padded_slots": int(1 << k),
        "alpha": float(alpha),
    }
    started = time.perf_counter()
    try:
        idx = QuantumFloat(k, name="idx")
        prepare(idx, amps)
        with invert():
            prepare(idx, amps)
    except Exception:
        result["status"] = "build_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        result["error"] = traceback.format_exc()
        return result

    status, compiled, elapsed, error = timed(lambda: idx.qs.compile(workspace=workspace))
    result["compile"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "compile_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        return result

    status, qiskit_qc, elapsed, error = timed(compiled.to_qiskit)
    result["to_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "to_qiskit_error"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        return result

    result["qiskit_metrics"] = qiskit_metrics(qiskit_qc)
    result["qasm"] = write_qasm2(qiskit_qc, out_dir / f"{prefix}_qiskit.qasm")
    result["basis_recounts"] = recount_bases(qiskit_qc, out_dir, prefix)
    result["basis_summary"] = {
        basis: summarize_basis(entry)
        for basis, entry in result["basis_recounts"].items()
    }
    result["status"] = "ok"
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def combine_select_prepare(select: dict[str, Any], prep_pair: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": "qclib_mcxvchain_select_plus_qrisp_prepare_pair",
        "status": "ok",
        "assumption": "QClib supplies SELECT; Qrisp PREPARE/PREPARE^dag is reused for coefficient state preparation.",
        "basis_summary": {},
        "components": {
            "qclib_select": select,
            "qrisp_prepare_pair": prep_pair,
        },
    }
    for basis in ("nam", "ibm"):
        s = select.get("basis_summary", {}).get(basis)
        p = prep_pair.get("basis_summary", {}).get(basis)
        if not s or not p:
            out["basis_summary"][basis] = None
            out["status"] = "partial"
            continue
        out["basis_summary"][basis] = {
            "gate_count": int(s["gate_count"]) + int(p["gate_count"]),
            "non_clifford": int(s["non_clifford"]) + int(p["non_clifford"]),
            "width": max(int(s["width"]), int(p["width"])),
            "components": {
                "qclib_select": {
                    "gate_count": int(s["gate_count"]),
                    "non_clifford": int(s["non_clifford"]),
                    "width": int(s["width"]),
                },
                "qrisp_prepare_pair": {
                    "gate_count": int(p["gate_count"]),
                    "non_clifford": int(p["non_clifford"]),
                    "width": int(p["width"]),
                },
            },
        }
    return out


def add_basis_totals(totals: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
    for basis in ("nam", "ibm"):
        summary = result.get("basis_summary", {}).get(basis)
        elapsed = float(result.get("elapsed_seconds", 0.0))
        totals[basis]["elapsed_seconds_per_occurrence_sum"] += elapsed
        if not result.get("cache_hit"):
            totals[basis]["elapsed_seconds_unique"] += elapsed
        if summary is None:
            totals[basis]["missing"] += 1
            continue
        totals[basis]["gate_count"] += int(summary["gate_count"])
        totals[basis]["non_clifford"] += int(summary["non_clifford"])
        totals[basis]["max_width"] = max(totals[basis]["max_width"], int(summary["width"]))
        totals[basis]["completed"] += 1


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


def run_one_matrixsum(ms: Any, case_dir: Path, prefix: str, workspace: int) -> dict[str, Any]:
    select = count_qclib_select(copy.deepcopy(ms), case_dir, prefix + "_qclib_select")
    prep_pair = count_qrisp_prepare_pair(copy.deepcopy(ms), case_dir, prefix + "_qrisp_prep_pair", workspace)
    full = combine_select_prepare(select, prep_pair)
    full["elapsed_seconds"] = float(select.get("elapsed_seconds", 0.0)) + float(
        prep_pair.get("elapsed_seconds", 0.0)
    )
    full["term_count"] = int(len(ms.instances))
    full["natural_width"] = int(natural_width(ms))
    full["terms_preview"] = kraus_terms(ms)[:8]
    return full


def run_rewritten_case(
    name: str,
    out_dir: Path,
    *,
    workspace: int,
    rewrite_max_steps: int,
    rewrite_beam_width: int,
    rewrite_tol: float,
) -> dict[str, Any]:
    case_dir = out_dir / "rewritten_kraus" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_kraus_ops, model_info = build_benchmark_current(name)
    rewritten_kraus_ops, rewrite_result, rewrite_seconds = rewrite_channel(
        raw_kraus_ops,
        max_steps=rewrite_max_steps,
        beam_width=rewrite_beam_width,
        tol=rewrite_tol,
    )
    rewritten_nonzero = [
        (index, op)
        for index, op in enumerate(rewritten_kraus_ops)
        if len(op.instances) > 0
    ]
    summary: dict[str, Any] = {
        "case": name,
        "model_info": model_info,
        "raw_channel_metrics": channel_metrics(raw_kraus_ops),
        "rewritten_channel_metrics": channel_metrics(rewritten_kraus_ops),
        "rewrite": {"elapsed_seconds": float(rewrite_seconds), "result": rewrite_result},
        "kraus_results": [],
        "totals": empty_totals(),
    }
    cache: dict[str, dict[str, Any]] = {}
    for ordinal, (kraus_index, ms) in enumerate(rewritten_nonzero):
        key = kraus_cache_key(ms)
        if key in cache:
            result = copy.deepcopy(cache[key])
            result["cache_hit"] = True
        else:
            prefix = f"{name}_kraus{kraus_index:04d}"
            result = run_one_matrixsum(ms, case_dir, prefix, workspace)
            result["cache_hit"] = False
            cache[key] = copy.deepcopy(result)
        add_basis_totals(summary["totals"], result)
        summary["kraus_results"].append(
            {
                "ordinal": int(ordinal),
                "kraus_index": int(kraus_index),
                "term_count": int(len(ms.instances)),
                "natural_width": int(natural_width(ms)),
                "qclib_full": result,
            }
        )
    return summary


def run_single_case(name: str, ms: Any, metadata: dict[str, Any], out_dir: Path, workspace: int) -> dict[str, Any]:
    case_dir = out_dir / "single" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    result = run_one_matrixsum(ms, case_dir, name, workspace)
    return {
        "case": name,
        "metadata": metadata,
        "metrics": {
            "system_qubits": int(ms.size),
            "term_count": int(len(ms.instances)),
            "natural_width": int(natural_width(ms)),
        },
        "qclib_full": result,
    }


def aggregate_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(statistics.fmean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def run_random_case(
    case: dict[str, Any],
    out_dir: Path,
    *,
    samples: int,
    seed_base: int,
    case_offset: int,
    workspace: int,
) -> dict[str, Any]:
    case_dir = out_dir / "random_pauli" / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = {"case": case, "num_samples": int(samples), "samples": [], "aggregate": {}}
    for sample_index in range(samples):
        seed = int(seed_base + 100000 * case_offset + sample_index)
        ms = random_pauli_matrixsum(case["num_qubits"], case["num_terms"], seed)
        validation = validate_matrixsum(ms)
        if validation["unique_terms"] != validation["term_count"] or validation["has_identity"]:
            raise RuntimeError(f"invalid random Pauli sample: {validation}")
        sample_dir = case_dir / f"sample{sample_index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        result = run_one_matrixsum(ms, sample_dir, f"{case['name']}_sample{sample_index:02d}_seed{seed}", workspace)
        summary["samples"].append(
            {
                "sample_index": int(sample_index),
                "seed": int(seed),
                "validation": validation,
                "qclib_full": result,
            }
        )

    for basis in ("nam", "ibm"):
        buckets = {"gate_count": [], "non_clifford": [], "width": [], "elapsed_seconds": []}
        for sample in summary["samples"]:
            result = sample["qclib_full"]
            basis_summary = result.get("basis_summary", {}).get(basis)
            if basis_summary is None:
                continue
            buckets["gate_count"].append(float(basis_summary["gate_count"]))
            buckets["non_clifford"].append(float(basis_summary["non_clifford"]))
            buckets["width"].append(float(basis_summary["width"]))
            buckets["elapsed_seconds"].append(float(result.get("elapsed_seconds", 0.0)))
        summary["aggregate"][basis] = {
            name: aggregate_values(values)
            for name, values in buckets.items()
        }
    return summary


def build_full_pauli_case(num_qubits: int) -> tuple[str, Any, dict[str, Any]]:
    return f"full_pauli_n{num_qubits}", full_pauli_matrixsum(num_qubits), {
        "family": "full_pauli_group_uniform",
        "scope": "single Pauli-sum over all n-qubit Paulis including identity",
        "num_qubits": int(num_qubits),
    }


def load_main_table(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or "Example" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 12:
            continue
        rows[cells[0]] = {
            "n": cells[1],
            "terms": cells[2],
            "scope": cells[3],
            "ours_total": cells[4],
            "ours_nc": cells[5],
            "ours_width": cells[6],
            "qrisp_total": cells[7],
            "qrisp_nc": cells[8],
            "qrisp_width": cells[9],
        }
    return rows


def load_select_only_ours_table(path: Path, basis: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    in_basis_section = False
    header: dict[str, int] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_basis_section = stripped[3:].strip().lower() == basis.lower()
            header = None
            continue
        if not in_basis_section:
            continue
        if not stripped.startswith("| "):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and cells[0] == "Example":
            header = {name: index for index, name in enumerate(cells)}
            continue
        if stripped.startswith("| ---") or header is None:
            continue
        rows[cells[header["Example"]]] = {
            "n": cells[header["n"]],
            "terms": cells[header["Terms"]],
            "ours_total": cells[header["Ours Total"]],
            "ours_nc": cells[header["Ours NC"]],
            "ours_width": cells[header["Ours Width"]],
        }
    return rows


def qclib_select_basis_summary(result: dict[str, Any], basis: str) -> dict[str, Any] | None:
    return (
        result.get("components", {})
        .get("qclib_select", {})
        .get("basis_summary", {})
        .get(basis)
    )


def qclib_select_totals_for_rewritten(case: dict[str, Any], basis: str) -> dict[str, Any] | None:
    total = {
        "gate_count": 0,
        "non_clifford": 0,
        "width": 0,
        "completed": 0,
        "missing": 0,
    }
    for kraus in case.get("kraus_results", []):
        summary = qclib_select_basis_summary(kraus.get("qclib_full", {}), basis)
        if summary is None:
            total["missing"] += 1
            continue
        total["gate_count"] += int(summary["gate_count"])
        total["non_clifford"] += int(summary["non_clifford"])
        total["width"] = max(total["width"], int(summary["width"]))
        total["completed"] += 1
    if total["completed"] == 0:
        return None
    return total


def qclib_select_aggregate_for_random(case: dict[str, Any], basis: str) -> dict[str, Any] | None:
    buckets = {"gate_count": [], "non_clifford": [], "width": []}
    for sample in case.get("samples", []):
        summary = qclib_select_basis_summary(sample.get("qclib_full", {}), basis)
        if summary is None:
            continue
        buckets["gate_count"].append(float(summary["gate_count"]))
        buckets["non_clifford"].append(float(summary["non_clifford"]))
        buckets["width"].append(float(summary["width"]))
    if not buckets["gate_count"]:
        return None
    return {
        "gate_count": float(statistics.fmean(buckets["gate_count"])),
        "non_clifford": float(statistics.fmean(buckets["non_clifford"])),
        "width": float(statistics.fmean(buckets["width"])),
        "stdev_gate_count": float(statistics.stdev(buckets["gate_count"])) if len(buckets["gate_count"]) > 1 else 0.0,
        "stdev_non_clifford": float(statistics.stdev(buckets["non_clifford"])) if len(buckets["non_clifford"]) > 1 else 0.0,
        "stdev_width": float(statistics.stdev(buckets["width"])) if len(buckets["width"]) > 1 else 0.0,
    }


def qclib_row_for(summary: dict[str, Any], key: str, kind: str, basis: str) -> dict[str, Any] | None:
    if kind == "rewritten":
        case = summary["rewritten_kraus"]["cases"].get(key)
        if not case:
            return None
        return qclib_select_totals_for_rewritten(case, basis)
    if kind in {"tfim", "full_pauli"}:
        case = summary["single"]["cases"].get(key)
        if not case:
            return None
        return qclib_select_basis_summary(case["qclib_full"], basis)
    if kind == "random_pauli":
        case = summary["random_pauli"]["cases"].get(key)
        if not case:
            return None
        return qclib_select_aggregate_for_random(case, basis)
    return None


def parse_number(text: str | None) -> float | None:
    if not text or text in {"--", "-"}:
        return None
    cleaned = text.replace(",", "").replace("e", "E")
    try:
        return float(cleaned)
    except ValueError:
        return None


def fmt_value(value: int | float | None, *, random_row: bool = False) -> str:
    if value is None:
        return "--"
    if random_row:
        return f"{float(value):,.1f}"
    return f"{int(round(float(value))):,}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "--"
    if value >= 100:
        return f"{value:.3g}"
    return f"{value:.2f}"


def ratio_from_text(numerator: int | float | None, denominator_text: str | None) -> str:
    denominator = parse_number(denominator_text)
    if numerator is None or denominator is None or denominator == 0:
        return "--"
    return fmt_ratio(float(numerator) / denominator)


def write_basis_tables(summary: dict[str, Any], basis: str, md_path: Path, tex_path: Path) -> None:
    existing = load_select_only_ours_table(SELECT_ONLY_OURS_QRISP_MD, basis)
    if not existing:
        existing = load_main_table(
            ROOT / "Results" / f"Table_main_{basis}_ours_qrisp.md"
        )
    basis_upper = basis.upper()
    md_lines = [
        f"# {basis_upper}-Basis Main-Text Candidates: SELECT-Only Ours vs QClib",
        "",
        "Both methods report SELECT-only costs. QClib uses the McxVchainDirty Mode A SELECT object; coefficient PREPARE/PREPARE^dagger cost is excluded.",
        "",
        "| Example | Ours Total | Ours NC | Ours W | QClib Total | QClib NC | QClib W | QClib/Ours Total | QClib/Ours NC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    tex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\caption{{{basis_upper}-basis SELECT-only cost comparison between our compiler and QClib McxVchainDirty on the Table~\\ref{{tab:nam-ours-qrisp-main}} candidate examples. Both methods exclude coefficient PREPARE/PREPARE^\\dagger. The last two columns report QClib/Ours ratios. NC denotes non-Clifford gates.}}",
        f"\\label{{tab:{basis}-ours-qclib-ratio-main}}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "& \\multicolumn{3}{c}{Ours} & \\multicolumn{3}{c}{QClib} & \\multicolumn{2}{c}{QClib/Ours} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
        "Example & Total & NC & W & Total & NC & W & Total & NC \\\\",
        "\\midrule",
    ]
    for row in TABLE_ROWS:
        base = existing.get(row["label"], {})
        qclib = qclib_row_for(summary, row["key"], row["kind"], basis)
        random_row = row["kind"] == "random_pauli"
        q_total = qclib.get("gate_count") if qclib else None
        q_nc = qclib.get("non_clifford") if qclib else None
        q_width = qclib.get("width") if qclib else None
        total_ratio = ratio_from_text(q_total, base.get("ours_total"))
        nc_ratio = ratio_from_text(q_nc, base.get("ours_nc"))
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    base.get("ours_total", "--"),
                    base.get("ours_nc", "--"),
                    base.get("ours_width", "--"),
                    fmt_value(q_total, random_row=random_row),
                    fmt_value(q_nc, random_row=random_row),
                    fmt_value(q_width, random_row=random_row),
                    total_ratio,
                    nc_ratio,
                ]
            )
            + " |"
        )
        tex_lines.append(
            f"{row['label'].replace('_', '\\_')} & "
            f"{base.get('ours_total', '--')} & {base.get('ours_nc', '--')} & {base.get('ours_width', '--')} & "
            f"{fmt_value(q_total, random_row=random_row)} & "
            f"{fmt_value(q_nc, random_row=random_row)} & {fmt_value(q_width, random_row=random_row)} & "
            f"{total_ratio} & {nc_ratio} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_partial(summary: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "summary.partial.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_reports(summary, out_dir / "report.partial.md")


def write_reports(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# QClib McxVchainDirty Table-Main Candidate Baseline",
        "",
        "Reported QClib cost is the McxVchainDirty SELECT object only; Qrisp PREPARE/PREPARE^dagger is excluded.",
        "",
        "| case | basis | gates | NC | width | completed/missing |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in TABLE_ROWS:
        for basis in ("nam", "ibm"):
            q = qclib_row_for(summary, row["key"], row["kind"], basis)
            if not q:
                lines.append(f"| {row['label']} | {basis} | -- | -- | -- | -- |")
                continue
            completed = q.get("completed", 1)
            missing = q.get("missing", 0)
            lines.append(
                f"| {row['label']} | {basis} | {fmt_value(q['gate_count'], random_row=row['kind'] == 'random_pauli')} | "
                f"{fmt_value(q['non_clifford'], random_row=row['kind'] == 'random_pauli')} | "
                f"{fmt_value(q['width'], random_row=row['kind'] == 'random_pauli')} | {completed}/{missing} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workspace", type=int, default=0)
    parser.add_argument("--rewrite-max-steps", type=int, default=50)
    parser.add_argument("--rewrite-beam-width", type=int, default=3)
    parser.add_argument("--rewrite-tol", type=float, default=1e-12)
    parser.add_argument("--random-pauli-samples", type=int, default=10)
    parser.add_argument("--random-pauli-seed", type=int, default=20260703)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or [row["key"] for row in MAIN_ROWS])
    summary: dict[str, Any] = {
        "experiment": "qclib_mcxvchain_table_main_candidates",
        "result_dir": str(out_dir),
        "arguments": vars(args),
        "policy": {
            "qclib": "McxVchainDirty Mode A row-wise SELECT",
            "table_metric": "QClib table/report metrics use McxVchainDirty SELECT only",
            "state_preparation": "Qrisp PREPARE + PREPARE^dagger pair recorded separately but excluded from QClib table metrics",
            "basis": "NAM and IBM opt0 recounts",
        },
        "rewritten_kraus": {"cases": {}},
        "single": {"cases": {}},
        "random_pauli": {"cases": {}},
    }

    for row in MAIN_ROWS:
        if row["key"] not in selected or row["kind"] != "rewritten":
            continue
        print(f"[run] rewritten {row['key']}", flush=True)
        summary["rewritten_kraus"]["cases"][row["key"]] = run_rewritten_case(
            row["key"],
            out_dir,
            workspace=args.workspace,
            rewrite_max_steps=args.rewrite_max_steps,
            rewrite_beam_width=args.rewrite_beam_width,
            rewrite_tol=args.rewrite_tol,
        )
        write_partial(summary, out_dir)

    tfim_builders: dict[str, Callable[[], tuple[str, Any, dict[str, Any]]]] = {
        "TFIM_3_A0": lambda: build_tfim_a0(3),
        "TFIM_8_A0": lambda: build_tfim_a0(8),
    }
    for row in MAIN_ROWS:
        if row["key"] not in selected or row["kind"] != "tfim":
            continue
        print(f"[run] single {row['key']}", flush=True)
        name, ms, metadata = tfim_builders[row["key"]]()
        summary["single"]["cases"][row["key"]] = run_single_case(
            name,
            ms,
            metadata,
            out_dir,
            args.workspace,
        )
        write_partial(summary, out_dir)

    for case_offset, case in enumerate(RANDOM_PAULI_CASES):
        if case["name"] not in selected:
            continue
        print(f"[run] random {case['name']}", flush=True)
        summary["random_pauli"]["cases"][case["name"]] = run_random_case(
            case,
            out_dir,
            samples=args.random_pauli_samples,
            seed_base=args.random_pauli_seed,
            case_offset=case_offset,
            workspace=args.workspace,
        )
        write_partial(summary, out_dir)

    if "full_pauli_n6" in selected:
        print("[run] single full_pauli_n6", flush=True)
        name, ms, metadata = build_full_pauli_case(6)
        summary["single"]["cases"]["full_pauli_n6"] = run_single_case(
            name,
            ms,
            metadata,
            out_dir,
            args.workspace,
        )
        write_partial(summary, out_dir)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_reports(summary, out_dir / "report.md")
    write_basis_tables(summary, "nam", RESULT_NAM_MD, RESULT_NAM_TEX)
    write_basis_tables(summary, "ibm", RESULT_IBM_MD, RESULT_IBM_TEX)
    print(f"wrote {summary_path}")
    print(f"wrote {RESULT_NAM_MD}")
    print(f"wrote {RESULT_NAM_TEX}")
    print(f"wrote {RESULT_IBM_MD}")
    print(f"wrote {RESULT_IBM_TEX}")


if __name__ == "__main__":
    main()
