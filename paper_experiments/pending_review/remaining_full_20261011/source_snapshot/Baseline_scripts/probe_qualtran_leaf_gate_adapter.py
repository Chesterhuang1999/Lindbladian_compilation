#!/usr/bin/env python3
"""Prototype Qiskit/NAM/IBM adapter for Qualtran sparse-alias leaf gates.

This script is intentionally conservative: when Qualtran is available it tries
to reconstruct ``SparseStatePreparationAliasSampling`` for the requested
coefficient row and collect the call-graph leaf bloqs.  When Qualtran is not
available, it can still consume the existing sparse-alias summary produced by
``probe_qualtran_sparse_alias_tfim8_a0.py``.

The current adapter lowers the *leaf gate multiset* to Qiskit-transpilable
operations.  It does not claim to reconstruct the exact Qualtran circuit order
or register wiring unless a future Qualtran/Cirq export path is added.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "Baseline_results" / "qualtran_sparse_alias_tfim8_a0_v1"
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "qualtran_leaf_gate_adapter_v1"


def add_project_paths() -> None:
    for candidate in (ROOT / "Baseline_scripts", ROOT / "src", ROOT / "scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def sympy_to_json(value: Any) -> Any:
    try:
        if hasattr(value, "is_integer") and value.is_integer:
            return int(value)
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return str(value)


def bloq_count_rows(counts: dict[Any, Any]) -> list[dict[str, Any]]:
    rows = []
    for bloq, count in counts.items():
        rows.append(
            {
                "bloq_type": type(bloq).__name__,
                "bloq": str(bloq),
                "count": sympy_to_json(count),
            }
        )
    rows.sort(key=lambda item: (str(item["bloq_type"]), str(item["bloq"])))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_existing_qualtran_row(source_dir: Path, width: int, epsilon: float) -> dict[str, Any] | None:
    path = source_dir / "qualtran_sparse_alias_summary.json"
    if not path.exists():
        return None
    summary = load_json(path)
    for row in summary.get("rows", []):
        if int(row.get("w", -1)) == width and math.isclose(
            float(row.get("epsilon_normalized", -1.0)),
            float(epsilon),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            return dict(row)
    return None


def find_coefficients_row(source_dir: Path, width: int) -> dict[str, Any] | None:
    summary_path = source_dir / "coefficients_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        for row in summary.get("rows", []):
            if int(row.get("ctrl_size", row.get("forced_w", -1))) == width:
                return dict(row)

    row_path = source_dir / f"tfim8_a0_matrix_order_w{width}_coefficients.json"
    if row_path.exists():
        return load_json(row_path)
    return None


def try_reconstruct_qualtran_row(
    source_dir: Path,
    width: int,
    epsilon: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Reconstruct a Qualtran sparse-alias bloq and return a row-like summary."""

    diagnostic: dict[str, Any] = {
        "attempted": True,
        "status": "not_started",
        "error": None,
        "qualtran_module": None,
    }
    coeff_row = find_coefficients_row(source_dir, width)
    if coeff_row is None:
        diagnostic.update(
            {
                "status": "blocked",
                "error": f"No coefficients row for w={width} under {source_dir}",
            }
        )
        return None, diagnostic

    try:
        import importlib

        from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
            SparseStatePreparationAliasSampling,
        )
        from qualtran.resource_counting import BloqCount, QubitCount, get_cost_value

        diagnostic["qualtran_module"] = importlib.import_module("qualtran").__file__
    except Exception as exc:
        diagnostic.update({"status": "blocked", "error": repr(exc)})
        return None, diagnostic

    weights = {
        int(item["index"]): float(item["weight"])
        for item in coeff_row.get("nonzero_coefficients", [])
        if float(item["weight"]) > 0
    }
    if not weights:
        diagnostic.update({"status": "blocked", "error": "Coefficient row has no nonzero weights"})
        return None, diagnostic

    started = time.perf_counter()
    slot_count = int(coeff_row["slot_count"])
    precision = float(epsilon) * float(sum(weights.values()))
    try:
        bloq = SparseStatePreparationAliasSampling.from_sparse_dict(
            weights,
            slot_count,
            precision=precision,
        )
        call_graph, _sigma = bloq.call_graph()
        leaf_counts = get_cost_value(bloq, BloqCount.for_call_graph_leaf_bloqs(call_graph))
        qubit_count = get_cost_value(bloq, QubitCount())
        row = {
            "benchmark": coeff_row.get("benchmark", "TFIM_8"),
            "kraus_index": int(coeff_row.get("kraus_index", 0)),
            "w": width,
            "slot_count": slot_count,
            "support": int(coeff_row["support"]),
            "epsilon_normalized": float(epsilon),
            "precision_unnormalized": precision,
            "mu": int(bloq.mu),
            "selection_bitsize": int(bloq.selection_bitsize),
            "sparse_index_bitsize": int(bloq.sparse_index_bitsize),
            "sum_weight": float(sum(weights.values())),
            "qualtran_signature_qubits": int(bloq.signature.n_qubits()),
            "qualtran_qubit_count": sympy_to_json(qubit_count),
            "leaf_gate_counts": bloq_count_rows(leaf_counts),
            "call_graph": {
                "nodes": int(len(call_graph.nodes)),
                "edges": int(len(call_graph.edges)),
            },
        }
        diagnostic.update(
            {
                "status": "ok",
                "construct_seconds": float(time.perf_counter() - started),
                "leaf_source": "reconstructed_qualtran_call_graph",
            }
        )
        return row, diagnostic
    except Exception as exc:
        diagnostic.update(
            {
                "status": "blocked",
                "error": repr(exc),
                "construct_seconds": float(time.perf_counter() - started),
            }
        )
        return None, diagnostic


def parse_rz_angle(bloq: str) -> float:
    match = re.search(r"Rz\(([^)]+)\)", bloq)
    if match is None:
        raise ValueError(f"Cannot parse Rz angle from {bloq!r}")
    return float(match.group(1))


class WireCursor:
    def __init__(self, width: int):
        self.width = int(width)
        self.index = 0

    def take(self, arity: int) -> list[int]:
        if arity > self.width:
            raise ValueError(f"Need {arity} distinct qubits but width is {self.width}")
        start = self.index
        self.index = (self.index + arity) % self.width
        return [(start + offset) % self.width for offset in range(arity)]


def aggregate_leaf_counts(leaf_rows: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in leaf_rows:
        count = int(row["count"])
        if count <= 0:
            continue
        counts[(str(row["bloq_type"]), str(row["bloq"]))] += count
    return counts


def adapter_rules() -> dict[str, dict[str, Any]]:
    return {
        "Hadamard/H": {
            "qiskit": "QuantumCircuit.h(q)",
            "semantics": "direct one-qubit H",
            "status": "implemented",
        },
        "XGate/X": {
            "qiskit": "QuantumCircuit.x(q)",
            "semantics": "direct one-qubit X",
            "status": "implemented",
        },
        "CNOT": {
            "qiskit": "QuantumCircuit.cx(control, target)",
            "semantics": "direct two-qubit CNOT",
            "status": "implemented",
        },
        "Rz(theta)": {
            "qiskit": "QuantumCircuit.rz(theta, q)",
            "semantics": "angle parsed from Qualtran leaf string",
            "status": "implemented",
        },
        "And": {
            "qiskit": "QuantumCircuit.ccx(a, b, target)",
            "semantics": (
                "unitary Toffoli compute used as a prototype for Qualtran And; "
                "does not model measurement-assisted clean/dirty ancilla details"
            ),
            "status": "prototype",
        },
        "And†": {
            "qiskit": "QuantumCircuit.ccx(a, b, target)",
            "semantics": (
                "unitary Toffoli uncompute surrogate; Qualtran's adjoint And "
                "may include measurement/classical-control in fault-tolerant decompositions"
            ),
            "status": "prototype",
        },
        "TwoBitCSwap": {
            "qiskit": "two QuantumCircuit.cswap(control, a_i, b_i) operations",
            "semantics": "controlled swap between two 2-bit registers",
            "status": "implemented",
        },
        "Allocate": {
            "qiskit": "reserved by choosing circuit width; no operation emitted",
            "semantics": "bookkeeping allocation",
            "status": "bookkeeping",
        },
        "Free": {
            "qiskit": "no operation emitted",
            "semantics": "bookkeeping deallocation; reset/assert-zero not modeled",
            "status": "bookkeeping",
        },
        "Split": {
            "qiskit": "no operation emitted",
            "semantics": "register reinterpretation",
            "status": "bookkeeping",
        },
        "Join": {
            "qiskit": "no operation emitted",
            "semantics": "register reinterpretation",
            "status": "bookkeeping",
        },
    }


def emit_leaf_multiset_circuit(row: dict[str, Any], out_dir: Path):
    from qiskit import QuantumCircuit

    width_raw = row.get("qualtran_qubit_count") or row.get("qualtran_signature_qubits")
    try:
        width = int(width_raw)
    except Exception:
        width = int(row.get("qualtran_signature_qubits", 1))
    width = max(width, 5)
    qc = QuantumCircuit(width, name="qualtran_leaf_multiset_adapter")
    cursor = WireCursor(width)
    counts = aggregate_leaf_counts(row["leaf_gate_counts"])
    lowered: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    unsupported: list[dict[str, Any]] = []

    def lower_once(bloq_type: str, bloq: str) -> None:
        if bloq_type == "Hadamard" or bloq == "H":
            (q,) = cursor.take(1)
            qc.h(q)
            lowered["h"] += 1
        elif bloq_type == "XGate" or bloq == "X":
            (q,) = cursor.take(1)
            qc.x(q)
            lowered["x"] += 1
        elif bloq_type == "CNOT" or bloq == "CNOT":
            control, target = cursor.take(2)
            qc.cx(control, target)
            lowered["cx"] += 1
        elif bloq_type == "Rz" or bloq.startswith("Rz("):
            (q,) = cursor.take(1)
            qc.rz(parse_rz_angle(bloq), q)
            lowered["rz"] += 1
        elif bloq_type == "And" and bloq == "And":
            a, b, target = cursor.take(3)
            qc.ccx(a, b, target)
            lowered["and_as_ccx"] += 1
        elif bloq_type == "And" and bloq in {"And†", "And^†", "And.adjoint()"}:
            a, b, target = cursor.take(3)
            qc.ccx(a, b, target)
            lowered["and_adjoint_as_ccx"] += 1
        elif bloq_type == "TwoBitCSwap" or bloq == "TwoBitCSwap":
            control, a0, a1, b0, b1 = cursor.take(5)
            qc.cswap(control, a0, b0)
            qc.cswap(control, a1, b1)
            lowered["twobitcswap_as_2_cswap"] += 1
        elif bloq_type in {"Allocate", "Free", "Split", "Join"}:
            skipped[bloq_type] += 1
        else:
            unsupported.append({"bloq_type": bloq_type, "bloq": bloq})

    for (bloq_type, bloq), count in sorted(counts.items()):
        for _ in range(int(count)):
            lower_once(bloq_type, bloq)

    if unsupported:
        return None, {
            "status": "blocked",
            "width": width,
            "lowered_operations": dict(lowered),
            "bookkeeping_skipped": dict(skipped),
            "unsupported_leaf_gates": unsupported,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    qpy_path = out_dir / "w8_eps1e-3_leaf_multiset_adapter.qpy"
    try:
        import qiskit.qpy

        with qpy_path.open("wb") as handle:
            qiskit.qpy.dump(qc, handle)
    except Exception:
        qpy_path = None

    return qc, {
        "status": "ok",
        "width": width,
        "input_leaf_gate_total_including_bookkeeping": int(
            sum(counts.values())
        ),
        "input_leaf_gate_counts": [
            {"bloq_type": key[0], "bloq": key[1], "count": int(value)}
            for key, value in sorted(counts.items())
        ],
        "lowered_operations": dict(lowered),
        "bookkeeping_skipped": dict(skipped),
        "qpy_path": None if qpy_path is None else str(qpy_path),
        "semantic_scope": (
            "synthetic leaf-multiset circuit for adapter cost probing; "
            "not an ordered/register-faithful implementation of SparseStatePreparationAliasSampling"
        ),
    }


def qasm2_dumps(circuit: Any) -> str:
    import qiskit.qasm2 as qasm2

    return qasm2.dumps(circuit)


def transpile_and_count(circuit: Any, basis: list[str], gate_set: str, out_path: Path) -> dict[str, Any]:
    add_project_paths()
    from baseline_common import count_qasm_text
    from qiskit import transpile

    started = time.perf_counter()
    transpiled = transpile(circuit, basis_gates=basis, optimization_level=0)
    elapsed = time.perf_counter() - started
    qasm_text = qasm2_dumps(transpiled)
    out_path.write_text(qasm_text, encoding="utf-8")
    stats = count_qasm_text(qasm_text, path=str(out_path), gate_set=gate_set)
    return {
        "status": "ok",
        "basis": basis,
        "qasm_path": str(out_path),
        "transpile_seconds": float(elapsed),
        "depth": int(transpiled.depth()),
        "raw_count_ops": {str(k): int(v) for k, v in transpiled.count_ops().items()},
        "stats": stats,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qualtran Leaf-Gate Adapter v1",
        "",
        "## Scope",
        "",
        summary["scope_note"],
        "",
        "## Adapter Rules",
        "",
        "| Qualtran leaf | Qiskit lowering | status |",
        "| --- | --- | --- |",
    ]
    for name, rule in summary["adapter_rules"].items():
        lines.append(f"| `{name}` | `{rule['qiskit']}` | {rule['status']} |")

    lines.extend(["", "## w=8 eps=1e-3 Result", ""])
    result = summary.get("result", {})
    if result.get("status") == "ok":
        nam = result["nam"]["stats"]
        ibm = result["ibm"]["stats"]
        lines.extend(
            [
                f"- Source leaf row: {summary['leaf_source']}",
                f"- Width: {result['adapter']['width']} qubits",
                f"- NAM basis `{result['nam']['basis']}`: gates={nam['metric_total']}, "
                f"non_clifford={nam['clifford']['non_clifford']}, ops={nam['ops']}",
                f"- IBM basis `{result['ibm']['basis']}`: gates={ibm['metric_total']}, "
                f"non_clifford={ibm['clifford']['non_clifford']}, ops={ibm['ops']}",
                "",
                "These counts are for the synthetic leaf-multiset adapter circuit, not a complete "
                "ordered Qualtran sparse-alias circuit.",
            ]
        )
    else:
        lines.append(f"- Blocked: {result.get('error', result)}")

    lines.extend(["", "## Complete-Circuit Status", ""])
    for item in summary["complete_circuit_blockers"]:
        lines.append(f"- {item}")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--w", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument(
        "--force-reconstruct",
        action="store_true",
        help="Try Qualtran reconstruction before using existing qualtran_sparse_alias_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_diag: dict[str, Any]
    row: dict[str, Any] | None = None
    leaf_source = "none"
    if args.force_reconstruct:
        row, reconstruction_diag = try_reconstruct_qualtran_row(source_dir, args.w, args.eps)
        if row is not None:
            leaf_source = "reconstructed_qualtran_call_graph"
    else:
        reconstruction_diag = {"attempted": False, "status": "skipped"}

    if row is None:
        row = find_existing_qualtran_row(source_dir, args.w, args.eps)
        if row is not None:
            leaf_source = str(source_dir / "qualtran_sparse_alias_summary.json")

    if row is None and not args.force_reconstruct:
        row, reconstruction_diag = try_reconstruct_qualtran_row(source_dir, args.w, args.eps)
        if row is not None:
            leaf_source = "reconstructed_qualtran_call_graph"

    summary: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "source_dir": str(source_dir),
        "result_dir": str(out_dir),
        "w": int(args.w),
        "epsilon_normalized": float(args.eps),
        "adapter_rules": adapter_rules(),
        "leaf_source": leaf_source,
        "qualtran_reconstruction": reconstruction_diag,
        "scope_note": (
            "Adapter v1 lowers Qualtran sparse-alias leaf gate categories into Qiskit "
            "operations and measures NAM/IBM transpilation cost. It is a lowering-rule "
            "prototype because the available JSON has leaf counts but not operation order "
            "or register bindings."
        ),
        "complete_circuit_blockers": [],
        "result": {},
    }

    if row is None:
        summary["result"] = {
            "status": "blocked",
            "error": (
                "No matching leaf-count row found and Qualtran reconstruction failed. "
                f"Reconstruction diagnostic: {reconstruction_diag}"
            ),
        }
    else:
        (out_dir / "w8_eps1e-3_source_leaf_row.json").write_text(
            json.dumps(row, indent=2, default=json_default),
            encoding="utf-8",
        )
        circuit, adapter_diag = emit_leaf_multiset_circuit(row, out_dir)
        if circuit is None:
            summary["result"] = {"status": "blocked", "adapter": adapter_diag}
        else:
            raw_qasm = qasm2_dumps(circuit)
            raw_qasm_path = out_dir / "w8_eps1e-3_leaf_multiset_adapter_raw.qasm"
            raw_qasm_path.write_text(raw_qasm, encoding="utf-8")
            nam = transpile_and_count(
                circuit,
                ["rz", "h", "x", "cx"],
                "nam",
                out_dir / "w8_eps1e-3_leaf_multiset_adapter_nam.qasm",
            )
            ibm = transpile_and_count(
                circuit,
                ["u1", "u2", "u3", "cx"],
                "ibm",
                out_dir / "w8_eps1e-3_leaf_multiset_adapter_ibm.qasm",
            )
            summary["result"] = {
                "status": "ok",
                "adapter": adapter_diag,
                "raw_qasm_path": str(raw_qasm_path),
                "nam": nam,
                "ibm": ibm,
            }

        summary["complete_circuit_blockers"] = [
            "Existing summary provides leaf counts, not a topologically ordered decomposition with register bindings.",
            "Qualtran/Cirq export was not available in this local environment unless --force-reconstruct succeeds in an environment with Qualtran installed.",
            "And/And† are represented by unitary CCX surrogates; an exact adapter needs Qualtran's measurement-assisted adjoint-And rule if that decomposition is required.",
            "Allocate/Free/Split/Join are bookkeeping-only in this prototype; exact allocation lifetime and zero-state assertions need the full composite bloq decomposition.",
        ]

    summary_path = out_dir / "leaf_gate_adapter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    write_report(out_dir / "leaf_gate_adapter_report.md", summary)
    print(json.dumps({"summary_path": str(summary_path), "status": summary["result"].get("status")}))


if __name__ == "__main__":
    main()
