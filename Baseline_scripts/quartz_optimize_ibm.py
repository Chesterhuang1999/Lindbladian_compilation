from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "circuits" / "test_table1_m_no_n4_ibm_select.qasm"
DEFAULT_OUTPUT = ROOT / "circuits" / "test_table1_m_no_n4_ibm_select_quartz_optimized.qasm"
DEFAULT_STATS = ROOT / "circuits" / "test_table1_m_no_n4_ibm_select_quartz_stats.json"
DEFAULT_ECC = ROOT / "external_tools" / "quartz" / "eccset" / "IBM_3_3_complete_ECC_set.json"
IBM_GATE_SET = ("u1", "u2", "u3", "cx")
IBM_QUARTZ_CONTEXT_GATES = IBM_GATE_SET + ("add",)
COUNTED_GATES = IBM_GATE_SET + ("reset",)

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _import_quartz():
    try:
        import quartz  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Could not import Quartz. Run this script with the channelIR_test "
            "environment, e.g. `conda run -n channelIR_test python "
            "Baseline_scripts/quartz_optimize_ibm.py`."
        ) from exc
    return quartz


def qasm_text_gate_stats(path: Path) -> dict[str, Any]:
    ops: dict[str, int] = {}
    num_qubits = 0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("qubit[") or line.startswith("qreg "):
            left_bracket = line.find("[")
            right_bracket = line.find("]")
            if left_bracket != -1 and right_bracket != -1:
                try:
                    num_qubits += int(line[left_bracket + 1 : right_bracket])
                except ValueError:
                    pass
            continue
        if line.startswith(("OPENQASM", "include", "creg ", "bit[")):
            continue
        gate = line.split("(", 1)[0].split()[0].rstrip(";")
        ops[gate] = ops.get(gate, 0) + 1

    return {
        "path": str(path),
        "num_qubits": int(num_qubits),
        "size": int(sum(ops.values())),
        "ops": ops,
        "ibm_gate_count": int(sum(ops.get(gate, 0) for gate in IBM_GATE_SET)),
        "cx": int(ops.get("cx", 0)),
        "u1": int(ops.get("u1", 0)),
        "u2": int(ops.get("u2", 0)),
        "u3": int(ops.get("u3", 0)),
        "reset": int(ops.get("reset", 0)),
    }


def quartz_gate_stats(graph) -> dict[str, int]:
    return {
        "gate_count": int(graph.gate_count),
        "cx_count": int(graph.cx_count),
        "depth": int(graph.depth),
    }


def optimize_with_quartz(
    context,
    init_graph,
    timeout_sec: float,
    max_candidates: int,
    upper_limit: float,
    progress_every: int,
):
    start = time.time()
    original_gate_count = int(init_graph.gate_count)
    best_graph = init_graph
    best_gate_count = original_gate_count
    candidates = [(original_gate_count, 0, init_graph)]
    seen = {init_graph.hash()}
    invoke_count = 0
    candidate_serial = 1

    while candidates:
        if time.time() - start >= timeout_sec:
            break
        if len(candidates) > max_candidates:
            candidates = heapq.nsmallest(max_candidates // 2, candidates)
            heapq.heapify(candidates)

        _, _, graph = heapq.heappop(candidates)
        nodes = graph.all_nodes()

        for xfer in context.get_xfers():
            for node in nodes:
                if time.time() - start >= timeout_sec:
                    return best_graph, {
                        "invoke_count": invoke_count,
                        "seen_circuits": len(seen),
                        "elapsed_sec": time.time() - start,
                    }

                invoke_count += 1
                new_graph = graph.apply_xfer(
                    xfer=xfer,
                    node=node,
                    eliminate_rotation=True,
                )
                if new_graph is None:
                    continue

                new_hash = new_graph.hash()
                if new_hash in seen:
                    continue
                seen.add(new_hash)

                new_gate_count = int(new_graph.gate_count)
                if new_gate_count > int(original_gate_count * upper_limit):
                    continue

                if new_gate_count < best_gate_count:
                    best_gate_count = new_gate_count
                    best_graph = new_graph
                    print(
                        "Quartz improved gate_count "
                        f"{original_gate_count} -> {best_gate_count} "
                        f"after {invoke_count} rewrite attempts.",
                        flush=True,
                    )

                heapq.heappush(candidates, (new_gate_count, candidate_serial, new_graph))
                candidate_serial += 1

                if progress_every > 0 and invoke_count % progress_every == 0:
                    print(
                        f"progress: attempts={invoke_count}, "
                        f"seen={len(seen)}, best_gate_count={best_gate_count}",
                        flush=True,
                    )

    return best_graph, {
        "invoke_count": invoke_count,
        "seen_circuits": len(seen),
        "elapsed_sec": time.time() - start,
    }


def default_quartz_input_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_quartz_input{input_path.suffix}")


def prepare_quartz_input(input_path: Path, quartz_input_path: Path) -> dict[str, Any]:
    if quartz_input_path.exists():
        return {"path": str(quartz_input_path), "generated": False}

    from qasm_export import export_openqasm2_quartz_input  # noqa: PLC0415

    circuit = QuantumCircuit.from_qasm_file(str(input_path))
    result = export_openqasm2_quartz_input(
        circuit,
        quartz_input_path,
        basis_gates=IBM_GATE_SET,
    )
    result["generated"] = True
    return result


def verify_qasm_equivalence(
    input_path: Path,
    output_path: Path,
    max_qubits: int,
) -> dict[str, Any]:
    input_circuit = QuantumCircuit.from_qasm_file(str(input_path))
    output_circuit = QuantumCircuit.from_qasm_file(str(output_path))

    if input_circuit.num_qubits != output_circuit.num_qubits:
        return {
            "checked": True,
            "equivalent": False,
            "reason": "different_num_qubits",
            "input_num_qubits": int(input_circuit.num_qubits),
            "output_num_qubits": int(output_circuit.num_qubits),
        }

    if input_circuit.num_qubits > max_qubits:
        raise RuntimeError(
            "Refusing dense Operator equivalence check for "
            f"{input_circuit.num_qubits} qubits. Increase --max-equivalence-qubits "
            "or pass --skip-equivalence-check if this is intentional."
        )

    equivalent = Operator(input_circuit).equiv(Operator(output_circuit))
    return {
        "checked": True,
        "equivalent": bool(equivalent),
        "method": "qiskit.quantum_info.Operator.equiv",
        "input_num_qubits": int(input_circuit.num_qubits),
        "output_num_qubits": int(output_circuit.num_qubits),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    quartz = _import_quartz()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    stats_path = Path(args.stats).resolve()
    ecc_path = Path(args.ecc).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input QASM not found: {input_path}")
    if not ecc_path.exists():
        raise FileNotFoundError(f"Quartz ECC set not found: {ecc_path}")

    quartz_input_path = (
        Path(args.quartz_input).resolve()
        if args.quartz_input
        else default_quartz_input_path(input_path)
    )
    quartz_input = prepare_quartz_input(input_path, quartz_input_path)

    context = quartz.QuartzContext(
        gate_set=list(IBM_QUARTZ_CONTEXT_GATES),
        filename=str(ecc_path),
        no_increase=False,
        include_nop=True,
    )
    num_xfers = int(context.num_xfers)
    if num_xfers <= 1 and not args.allow_trivial_xfer_set:
        raise RuntimeError(
            "Quartz loaded only "
            f"{num_xfers} xfer(s) from {ecc_path}. This usually means the ECC "
            "file format or gate set is incompatible with the current Quartz "
            "build. Refusing to optimize without a real rewrite set."
        )

    init_graph = quartz.PyGraph.from_qasm(context=context, filename=str(quartz_input_path))
    best_graph, search_stats = optimize_with_quartz(
        context=context,
        init_graph=init_graph,
        timeout_sec=float(args.timeout),
        max_candidates=int(args.max_candidates),
        upper_limit=float(args.upper_limit),
        progress_every=int(args.progress_every),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    best_graph.to_qasm(filename=str(candidate_output_path))

    if args.skip_equivalence_check:
        equivalence = {"checked": False, "equivalent": None}
    else:
        equivalence = verify_qasm_equivalence(
            quartz_input_path,
            candidate_output_path,
            max_qubits=int(args.max_equivalence_qubits),
        )
        if not equivalence["equivalent"]:
            raise RuntimeError(
                "Quartz output failed independent Qiskit Operator equivalence "
                f"check: {equivalence}"
            )

    candidate_output_path.replace(output_path)

    result = {
        "gate_set": list(IBM_GATE_SET),
        "quartz_context_gate_set": list(IBM_QUARTZ_CONTEXT_GATES),
        "input": {
            "qasm_text": qasm_text_gate_stats(input_path),
            "quartz_input": {
                "normalization": quartz_input,
                "qasm_text": qasm_text_gate_stats(quartz_input_path),
            },
            "quartz": quartz_gate_stats(init_graph),
        },
        "output": {
            "qasm_text": qasm_text_gate_stats(output_path),
            "quartz": quartz_gate_stats(best_graph),
            "equivalence": equivalence,
        },
        "quartz": {
            "ecc": str(ecc_path),
            "num_xfers": num_xfers,
            "search": search_stats,
        },
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize an IBM-basis QASM circuit with Quartz and report gate counts. "
            "Run with the channelIR_test conda environment."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input QASM file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Optimized output QASM file.")
    parser.add_argument("--stats", default=str(DEFAULT_STATS), help="JSON gate-count report path.")
    parser.add_argument("--ecc", default=str(DEFAULT_ECC), help="Quartz ECC set JSON.")
    parser.add_argument(
        "--quartz-input",
        default=None,
        help="Optional flattened QASM2 input path used internally by Quartz.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Quartz search timeout in seconds.")
    parser.add_argument("--max-candidates", type=int, default=10000, help="Maximum candidate queue size.")
    parser.add_argument("--upper-limit", type=float, default=1.05, help="Candidate gate-count upper limit factor.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N rewrite attempts.")
    parser.add_argument(
        "--max-equivalence-qubits",
        type=int,
        default=10,
        help="Maximum qubit count for dense Qiskit Operator equivalence checking.",
    )
    parser.add_argument(
        "--skip-equivalence-check",
        action="store_true",
        help="Write the Quartz output without independent Qiskit equivalence checking.",
    )
    parser.add_argument(
        "--allow-trivial-xfer-set",
        action="store_true",
        help="Allow optimization to run even if Quartz loads one or zero xfers.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
