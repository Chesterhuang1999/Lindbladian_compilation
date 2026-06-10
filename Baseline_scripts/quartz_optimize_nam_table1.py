from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


NAM_BASIS_GATES = ("h", "x", "rz", "cx")
NAM_QUARTZ_CONTEXT_GATES = ("h", "x", "rz", "cx", "add")
DEFAULT_ECC = ROOT / "external_tools" / "quartz" / "eccset" / "Nam_5_3_complete_ECC_set.json"
DEFAULT_OUTPUT_DIR = ROOT / "circuits"
DEFAULT_METRICS = ("m_no",)


def _import_quartz():
    try:
        import quartz  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Could not import Quartz. Run with the ChannelIR_test environment, "
            "e.g. `conda run -n ChannelIR_test python "
            "Baseline_scripts/quartz_optimize_nam_table1.py`."
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
        "nam_gate_count": int(sum(ops.get(gate, 0) for gate in NAM_BASIS_GATES)),
        "h": int(ops.get("h", 0)),
        "x": int(ops.get("x", 0)),
        "rz": int(ops.get("rz", 0)),
        "cx": int(ops.get("cx", 0)),
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
                        "timed_out": True,
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
        "timed_out": False,
    }


def run_one(args: argparse.Namespace, context, metric: str) -> dict[str, Any]:
    quartz = _import_quartz()
    output_dir = Path(args.output_dir).resolve()
    qasm3_path = output_dir / f"test_table1_{metric}_n{args.num_qubits}_nam.qasm"
    quartz_input_path = output_dir / f"test_table1_{metric}_n{args.num_qubits}_nam_quartz_input.qasm"
    optimized_path = output_dir / f"test_table1_{metric}_n{args.num_qubits}_nam_quartz_optimized.qasm"

    if args.skip_generate:
        if not qasm3_path.exists():
            raise FileNotFoundError(f"Generated Nam QASM3 not found: {qasm3_path}")
        if not quartz_input_path.exists():
            raise FileNotFoundError(f"Generated Nam Quartz input not found: {quartz_input_path}")
        export_info: dict[str, Any] = {
            "path": str(qasm3_path),
            "basis_gates": list(NAM_BASIS_GATES),
            "optimization_level": 0,
            "quartz_input": {"path": str(quartz_input_path), "written": True},
            "generated_in_this_process": False,
        }
    else:
        from test_table1 import export_tfim_lcu_nam_openqasm3  # noqa: PLC0415

        export_info = export_tfim_lcu_nam_openqasm3(
            num_qubits=int(args.num_qubits),
            metrics=metric,
            out_path=qasm3_path,
            delta_t=float(args.delta_t),
        )
        quartz_input = export_info.get("quartz_input")
        if not isinstance(quartz_input, dict) or not quartz_input.get("written"):
            raise RuntimeError(f"Failed to write Quartz input for {metric}: {quartz_input}")

        generated_quartz_input_path = Path(str(quartz_input["path"]))
        if generated_quartz_input_path != quartz_input_path:
            generated_quartz_input_path.replace(quartz_input_path)
            quartz_input["path"] = str(quartz_input_path)
        export_info["generated_in_this_process"] = True

    init_graph = quartz.PyGraph.from_qasm(
        context=context,
        filename=str(quartz_input_path),
    )
    best_graph, search_stats = optimize_with_quartz(
        context=context,
        init_graph=init_graph,
        timeout_sec=float(args.timeout),
        max_candidates=int(args.max_candidates),
        upper_limit=float(args.upper_limit),
        progress_every=int(args.progress_every),
    )

    optimized_path.parent.mkdir(parents=True, exist_ok=True)
    best_graph.to_qasm(filename=str(optimized_path))

    return {
        "metric": metric,
        "export": export_info,
        "input": {
            "qasm3": qasm_text_gate_stats(qasm3_path),
            "quartz_input": qasm_text_gate_stats(quartz_input_path),
            "quartz": quartz_gate_stats(init_graph),
        },
        "output": {
            "qasm": qasm_text_gate_stats(optimized_path),
            "quartz": quartz_gate_stats(best_graph),
        },
        "quartz": {
            "search": search_stats,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    quartz = _import_quartz()
    ecc_path = Path(args.ecc).resolve()
    if not ecc_path.exists():
        raise FileNotFoundError(f"Nam ECC set not found: {ecc_path}")

    context = quartz.QuartzContext(
        gate_set=list(NAM_QUARTZ_CONTEXT_GATES),
        filename=str(ecc_path),
        no_increase=False,
        include_nop=True,
    )
    num_xfers = int(context.num_xfers)
    if num_xfers <= 1:
        raise RuntimeError(f"Loaded only {num_xfers} Quartz xfer(s) from {ecc_path}")

    results = [run_one(args, context, metric) for metric in args.metrics]
    summary = {
        "basis_gates": list(NAM_BASIS_GATES),
        "optimization_level": 0,
        "ecc": str(ecc_path),
        "num_xfers": num_xfers,
        "results": results,
        "comparison": {
            item["metric"]: {
                "transpiled_qasm3_size": item["input"]["qasm3"]["size"],
                "quartz_input_gate_count": item["input"]["quartz"]["gate_count"],
                "quartz_optimized_gate_count": item["output"]["quartz"]["gate_count"],
                "delta_vs_quartz_input": (
                    item["output"]["quartz"]["gate_count"]
                    - item["input"]["quartz"]["gate_count"]
                ),
                "cx_before": item["input"]["quartz"]["cx_count"],
                "cx_after": item["output"]["quartz"]["cx_count"],
            }
            for item in results
        },
    }

    stats_path = Path(args.stats).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate test_table1 circuits in Nam basis and optimize them with "
            "Quartz Nam ECC rules. Run with ChannelIR_test."
        )
    )
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ecc", default=str(DEFAULT_ECC))
    parser.add_argument(
        "--stats",
        default=str(DEFAULT_OUTPUT_DIR / "test_table1_n4_nam_quartz_summary.json"),
    )
    parser.add_argument("--timeout", type=float, default=10800.0)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--upper-limit", type=float, default=1.05)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help=(
            "Do not import Qiskit/test_table1 or generate QASM; optimize existing "
            "test_table1_*_nam.qasm and *_nam_quartz_input.qasm files."
        ),
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
