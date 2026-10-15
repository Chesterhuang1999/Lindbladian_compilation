#!/usr/bin/env python3
"""Inspect the SELECT module for the full n=6 Pauli block encoding."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from qiskit import transpile  # noqa: E402

from block_encoding_new import (  # type: ignore  # noqa: E402
    BlockEncoding,
    expand_large_controlled_gates,
)
from rerun_block_encoding_new_ours_refresh import full_pauli_matrixsum  # type: ignore  # noqa: E402
from run_mcg_tool_raw_channel_lcu_benchmarks import (  # type: ignore  # noqa: E402
    json_default,
    qiskit_metrics,
    recount_qiskit_circuit,
    write_qasm2,
)


RESULT_DIR = ROOT / "Baseline_results" / "full_pauli_n6_select_clifford_t_v1"


CLIFFORD_T_BASIS = [
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "x",
    "y",
    "z",
    "cx",
]


def op_counter(qc: Any) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(qc.count_ops().items())}


def clifford_t_stats(qc: Any) -> dict[str, Any]:
    ops = Counter({str(k): int(v) for k, v in qc.count_ops().items()})
    t_count = int(ops.get("t", 0) + ops.get("tdg", 0))
    clifford_gate_names = {"h", "s", "sdg", "x", "y", "z", "cx"}
    clifford_count = int(sum(ops.get(g, 0) for g in clifford_gate_names))
    return {
        "total_ops": int(sum(ops.values())),
        "t_count": t_count,
        "t": int(ops.get("t", 0)),
        "tdg": int(ops.get("tdg", 0)),
        "clifford_count": clifford_count,
        "cx": int(ops.get("cx", 0)),
        "h": int(ops.get("h", 0)),
        "s": int(ops.get("s", 0)),
        "sdg": int(ops.get("sdg", 0)),
        "x": int(ops.get("x", 0)),
        "y": int(ops.get("y", 0)),
        "z": int(ops.get("z", 0)),
        "other_ops": {
            gate: int(count)
            for gate, count in sorted(ops.items())
            if gate not in clifford_gate_names and gate not in {"t", "tdg"}
        },
    }


def basis_recount(qc: Any, basis: str, path_prefix: str) -> dict[str, Any]:
    return recount_qiskit_circuit(
        qc,
        basis_name=basis,
        out_path=RESULT_DIR / f"{path_prefix}_{basis}_opt0.qasm",
        optimization_level=0,
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    ms = full_pauli_matrixsum(6)
    be = BlockEncoding(ms)
    raw_select, analytical_t, analytical_mcx_like, analytical_cx, ctrl_size = be.mulplex_U_opt_order()
    expanded_select = expand_large_controlled_gates(raw_select, if_optimize=True)

    clifford_t = transpile(
        expanded_select,
        basis_gates=CLIFFORD_T_BASIS,
        optimization_level=0,
    )

    nam_recount = basis_recount(expanded_select, "nam", "full_pauli_n6_select_expanded")
    ibm_recount = basis_recount(expanded_select, "ibm", "full_pauli_n6_select_expanded")

    raw_qasm = write_qasm2(raw_select, RESULT_DIR / "full_pauli_n6_select_raw.qasm")
    expanded_qasm = write_qasm2(expanded_select, RESULT_DIR / "full_pauli_n6_select_expanded.qasm")
    clifford_t_qasm = write_qasm2(clifford_t, RESULT_DIR / "full_pauli_n6_select_clifford_t.qasm")

    summary = {
        "experiment": "full_pauli_n6_select_clifford_t",
        "term_count": int(len(ms.instances)),
        "system_qubits": 6,
        "ctrl_size": int(ctrl_size),
        "selected_k": int(getattr(be, "selected_k", ctrl_size)),
        "selected_crit": str(getattr(be, "selected_crit", "")),
        "width_candidate_costs": getattr(be, "width_candidate_costs", []),
        "placement_width_analysis": getattr(be, "placement_width_analysis", {}),
        "analytical_from_mulplex_U_opt_order": {
            "tcount": int(analytical_t),
            "mccount": int(analytical_mcx_like),
            "cxcount": int(analytical_cx),
        },
        "raw_select": {
            "num_qubits": int(raw_select.num_qubits),
            "depth": int(raw_select.depth()),
            "ops": op_counter(raw_select),
            "qasm": raw_qasm,
            "qiskit_metrics": qiskit_metrics(raw_select),
        },
        "expanded_select": {
            "num_qubits": int(expanded_select.num_qubits),
            "depth": int(expanded_select.depth()),
            "ops": op_counter(expanded_select),
            "qasm": expanded_qasm,
            "qiskit_metrics": qiskit_metrics(expanded_select),
        },
        "clifford_t_select": {
            "basis_gates": CLIFFORD_T_BASIS,
            "num_qubits": int(clifford_t.num_qubits),
            "depth": int(clifford_t.depth()),
            "ops": op_counter(clifford_t),
            "stats": clifford_t_stats(clifford_t),
            "qasm": clifford_t_qasm,
            "qiskit_metrics": qiskit_metrics(clifford_t),
        },
        "nam_recount_from_expanded_select": nam_recount,
        "ibm_recount_from_expanded_select": ibm_recount,
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )

    lines = [
        "# Full Pauli n=6 SELECT Clifford+T Probe",
        "",
        f"- ctrl_size: {ctrl_size}",
        f"- raw SELECT width/depth: {raw_select.num_qubits} / {raw_select.depth()}",
        f"- expanded SELECT width/depth: {expanded_select.num_qubits} / {expanded_select.depth()}",
        f"- Clifford+T width/depth: {clifford_t.num_qubits} / {clifford_t.depth()}",
        "",
        "## Clifford+T ops",
        "",
        "| gate | count |",
        "| --- | ---: |",
    ]
    for gate, count in op_counter(clifford_t).items():
        lines.append(f"| {gate} | {count} |")
    stats = clifford_t_stats(clifford_t)
    lines.extend(
        [
            "",
            "## Cost",
            "",
            f"- total Clifford+T basis ops: {stats['total_ops']}",
            f"- T count (`t + tdg`): {stats['t_count']}",
            f"- Clifford count: {stats['clifford_count']}",
            f"- CX count: {stats['cx']}",
            "",
            "## NAM/IBM opt0 recount from expanded SELECT",
            "",
            f"- NAM: {nam_recount['stats']['metric_total']} gates, {nam_recount['stats']['clifford']['non_clifford']} non-Clifford, width {nam_recount['stats']['num_qubits']}",
            f"- IBM: {ibm_recount['stats']['metric_total']} gates, {ibm_recount['stats']['clifford']['non_clifford']} non-Clifford, width {ibm_recount['stats']['num_qubits']}",
        ]
    )
    (RESULT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary": str(RESULT_DIR / "summary.json"),
        "report": str(RESULT_DIR / "report.md"),
        "ctrl_size": int(ctrl_size),
        "expanded_width": int(expanded_select.num_qubits),
        "clifford_t": summary["clifford_t_select"]["stats"],
        "nam": {
            "gates": nam_recount["stats"]["metric_total"],
            "non_clifford": nam_recount["stats"]["clifford"]["non_clifford"],
            "width": nam_recount["stats"]["num_qubits"],
        },
        "ibm": {
            "gates": ibm_recount["stats"]["metric_total"],
            "non_clifford": ibm_recount["stats"]["clifford"]["non_clifford"],
            "width": ibm_recount["stats"]["num_qubits"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
