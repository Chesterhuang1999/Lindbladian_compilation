#!/usr/bin/env python3
"""Unified Quartz baseline runner for IBM/NAM QASM inputs."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = ROOT / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_common import (  # noqa: E402
    compare_gate_stats,
    count_qasm_file,
    require_input_qasm,
    write_summary,
)


GATESETS = {
    "ibm": {
        "basis": ("u1", "u2", "u3", "cx"),
        "context": ("u1", "u2", "u3", "cx", "add"),
        "ecc": ROOT / "external_tools" / "quartz" / "eccset" / "IBM_3_3_complete_ECC_set.json",
    },
    "nam": {
        "basis": ("h", "x", "rz", "cx"),
        "context": ("h", "x", "rz", "cx", "add"),
        "ecc": ROOT / "external_tools" / "quartz" / "eccset" / "Nam_5_3_complete_ECC_set.json",
    },
}


def import_quartz():
    try:
        import quartz  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Could not import Quartz. Run with ChannelIR_test, for example: "
            "conda run -n ChannelIR_test python Baseline_scripts/run_quartz_baseline.py ..."
        ) from exc
    return quartz


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
    checkpoint_qasm_path: Path | None,
    checkpoint_summary_path: Path | None,
    checkpoint_interval_sec: float,
    use_available_xfers: bool,
    prune_relative_to_best: bool,
    eliminate_rotation: bool,
):
    start = time.perf_counter()
    original_gate_count = int(init_graph.gate_count)
    best_graph = init_graph
    best_gate_count = original_gate_count
    candidates = [(original_gate_count, 0, init_graph)]
    seen = {init_graph.hash()}
    invoke_count = 0
    candidate_serial = 1
    checkpoint_count = 0
    next_checkpoint_at = (
        start + checkpoint_interval_sec if checkpoint_interval_sec > 0 else float("inf")
    )

    def search_stats(timed_out: bool) -> dict[str, Any]:
        return {
            "candidate_queue_len": len(candidates),
            "checkpoint_count": checkpoint_count,
            "elapsed_seconds": time.perf_counter() - start,
            "invoke_count": invoke_count,
            "seen_circuits": len(seen),
            "timed_out": timed_out,
        }

    def maybe_checkpoint(force: bool = False) -> None:
        nonlocal checkpoint_count, next_checkpoint_at
        now = time.perf_counter()
        if checkpoint_qasm_path is None or (not force and now < next_checkpoint_at):
            return

        checkpoint_qasm_path.parent.mkdir(parents=True, exist_ok=True)
        best_graph.to_qasm(filename=str(checkpoint_qasm_path))
        checkpoint_count += 1
        stats = {
            "best_gate_count": best_gate_count,
            "checkpoint_count": checkpoint_count,
            "elapsed_seconds": now - start,
            "invoke_count": invoke_count,
            "path": str(checkpoint_qasm_path),
            "seen_circuits": len(seen),
        }
        if checkpoint_summary_path is not None:
            checkpoint_summary_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_summary_path.write_text(
                json.dumps(stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        next_checkpoint_at = now + checkpoint_interval_sec

    while candidates:
        if time.perf_counter() - start >= timeout_sec:
            break
        if len(candidates) > max_candidates:
            candidates = heapq.nsmallest(max_candidates // 2, candidates)
            heapq.heapify(candidates)

        _, _, graph = heapq.heappop(candidates)
        nodes = graph.all_nodes()

        node_xfers = (
            (
                node,
                [
                    context.get_xfer_from_id(id=int(xfer_id))
                    for xfer_id in graph.available_xfers_parallel(context=context, node=node)
                ],
            )
            for node in nodes
        ) if use_available_xfers else ((node, context.get_xfers()) for node in nodes)

        for node, xfers in node_xfers:
            for xfer in xfers:
                if xfer is None:
                    continue
                if time.perf_counter() - start >= timeout_sec:
                    maybe_checkpoint(force=True)
                    return best_graph, search_stats(timed_out=True)

                invoke_count += 1
                new_graph = graph.apply_xfer(
                    xfer=xfer,
                    node=node,
                    eliminate_rotation=eliminate_rotation,
                )
                if new_graph is None:
                    continue

                new_hash = new_graph.hash()
                if new_hash in seen:
                    continue
                seen.add(new_hash)

                new_gate_count = int(new_graph.gate_count)
                limit_base = best_gate_count if prune_relative_to_best else original_gate_count
                if new_gate_count > int(limit_base * upper_limit):
                    continue

                if new_gate_count < best_gate_count:
                    best_gate_count = new_gate_count
                    best_graph = new_graph
                    maybe_checkpoint(force=True)
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
                maybe_checkpoint()

    maybe_checkpoint(force=True)
    return best_graph, search_stats(timed_out=False)


def prepare_quartz_input(input_path: Path, output_path: Path, basis_gates: tuple[str, ...]) -> dict[str, Any]:
    from qasm_export import export_openqasm2_quartz_input  # noqa: PLC0415
    from qiskit import QuantumCircuit  # noqa: PLC0415

    circuit = QuantumCircuit.from_qasm_file(str(input_path))
    result = export_openqasm2_quartz_input(circuit, output_path, basis_gates=basis_gates)
    result["generated"] = True
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    quartz = import_quartz()
    input_path = require_input_qasm(Path(args.input))
    input_stats = count_qasm_file(input_path)
    gate_set = str(input_stats["detected_gate_set"])
    config = GATESETS[gate_set]

    ecc_path = Path(args.ecc).resolve() if args.ecc else Path(config["ecc"]).resolve()
    if not ecc_path.exists():
        raise FileNotFoundError(f"Quartz ECC set not found: {ecc_path}")

    output_path = Path(args.output).resolve() if args.output else None
    checkpoint_qasm_path = Path(args.checkpoint_output).resolve() if args.checkpoint_output else None
    checkpoint_summary_path = (
        Path(args.checkpoint_summary).resolve()
        if args.checkpoint_summary
        else (
            checkpoint_qasm_path.with_suffix(".json")
            if checkpoint_qasm_path is not None
            else None
        )
    )
    with tempfile.TemporaryDirectory(prefix="quartz_baseline_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        quartz_input_path = tmp_dir_path / "quartz_input.qasm"
        kept_output = output_path is not None
        actual_output_path = output_path or (tmp_dir_path / "optimized.qasm")

        quartz_input_info = prepare_quartz_input(
            input_path=input_path,
            output_path=quartz_input_path,
            basis_gates=tuple(config["basis"]),
        )
        quartz_input_stats = count_qasm_file(quartz_input_path, gate_set=gate_set)

        context = quartz.QuartzContext(
            gate_set=list(config["context"]),
            filename=str(ecc_path),
            no_increase=bool(args.no_increase),
            include_nop=not bool(args.no_nop),
        )
        num_xfers = int(context.num_xfers)
        if num_xfers <= 1 and not args.allow_trivial_xfer_set:
            raise RuntimeError(
                f"Quartz loaded only {num_xfers} xfer(s) from {ecc_path}; "
                "pass --allow-trivial-xfer-set to run anyway."
            )

        init_graph = quartz.PyGraph.from_qasm(context=context, filename=str(quartz_input_path))
        best_graph, search_stats = optimize_with_quartz(
            context=context,
            init_graph=init_graph,
            timeout_sec=float(args.timeout),
            max_candidates=int(args.max_candidates),
            upper_limit=float(args.upper_limit),
            progress_every=int(args.progress_every),
            checkpoint_qasm_path=checkpoint_qasm_path,
            checkpoint_summary_path=checkpoint_summary_path,
            checkpoint_interval_sec=float(args.checkpoint_interval),
            use_available_xfers=bool(args.use_available_xfers),
            prune_relative_to_best=bool(args.prune_relative_to_best),
            eliminate_rotation=not bool(args.no_eliminate_rotation),
        )

        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        best_graph.to_qasm(filename=str(actual_output_path))
        output_stats = count_qasm_file(actual_output_path, gate_set=gate_set)
        if not kept_output:
            output_stats["path"] = None
            quartz_input_info["path"] = None
            quartz_input_stats["path"] = None

        summary = {
            "tool": "quartz",
            "input": str(input_path),
            "detected_gate_set": gate_set,
            "basis_gates": list(config["basis"]),
            "context_gate_set": list(config["context"]),
            "ecc": str(ecc_path),
            "num_xfers": num_xfers,
            "quartz_options": {
                "checkpoint_interval": float(args.checkpoint_interval),
                "checkpoint_output": (
                    str(checkpoint_qasm_path) if checkpoint_qasm_path is not None else None
                ),
                "checkpoint_summary": (
                    str(checkpoint_summary_path)
                    if checkpoint_summary_path is not None
                    else None
                ),
                "eliminate_rotation": not bool(args.no_eliminate_rotation),
                "include_nop": not bool(args.no_nop),
                "max_candidates": int(args.max_candidates),
                "no_increase": bool(args.no_increase),
                "prune_relative_to_best": bool(args.prune_relative_to_best),
                "progress_every": int(args.progress_every),
                "upper_limit": float(args.upper_limit),
                "use_available_xfers": bool(args.use_available_xfers),
            },
            "kept_output": kept_output,
            "output_qasm": str(actual_output_path) if kept_output else None,
            "input_stats": input_stats,
            "quartz_input": {
                "normalization": quartz_input_info,
                "stats": quartz_input_stats,
            },
            "output_stats": output_stats,
            "comparison": compare_gate_stats(input_stats, output_stats, gate_set),
            "quartz_stats": {
                "input_graph": quartz_gate_stats(init_graph),
                "output_graph": quartz_gate_stats(best_graph),
                "search": search_stats,
            },
        }

    write_summary(summary, Path(args.summary).resolve() if args.summary else None)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Quartz on one QASM input, infer IBM/NAM, and report gate counts."
    )
    parser.add_argument("--input", required=True, help="Input OpenQASM2 file.")
    parser.add_argument("--output", default=None, help="Optional path to keep optimized QASM.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument("--ecc", default=None, help="Override Quartz ECC JSON.")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Quartz search timeout in seconds.")
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--upper-limit", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=0.0,
        help="Write the current best graph every N seconds; 0 disables periodic checkpoints.",
    )
    parser.add_argument("--checkpoint-output", default=None, help="Checkpoint QASM path.")
    parser.add_argument("--checkpoint-summary", default=None, help="Checkpoint JSON summary path.")
    parser.add_argument(
        "--no-increase",
        action="store_true",
        help="Load only Quartz transfers that do not increase gate count.",
    )
    parser.add_argument("--no-nop", action="store_true", help="Do not include Quartz nop transfer.")
    parser.add_argument(
        "--use-available-xfers",
        action="store_true",
        help="Ask Quartz for applicable transfers per node instead of trying every transfer.",
    )
    parser.add_argument(
        "--prune-relative-to-best",
        action="store_true",
        help="Apply upper-limit relative to the current best count instead of the original count.",
    )
    parser.add_argument(
        "--no-eliminate-rotation",
        action="store_true",
        help="Disable Quartz eliminate_rotation during transfer application.",
    )
    parser.add_argument("--allow-trivial-xfer-set", action="store_true")
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
