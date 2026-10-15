#!/usr/bin/env python3
"""Unified BQSKit baseline runner for IBM/NAM QASM inputs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MODULE_NAME = "run_bqskit_baseline"
sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

from baseline_common import (  # noqa: E402
    compare_gate_stats,
    count_qasm_file,
    load_qasm_stats_or_count,
    require_input_qasm,
    write_summary,
)

try:
    from bqskit.compiler.basepass import BasePass
except ImportError:  # pragma: no cover - parser help in envs without bqskit.
    BasePass = object  # type: ignore[assignment,misc]


class CheckpointSavePass(BasePass):  # type: ignore[misc,valid-type]
    """Serializable BQSKit pass that saves the current circuit."""

    def __init__(self, checkpoint_dir: str, step_index: int, pass_name: str) -> None:
        self.checkpoint_dir = str(checkpoint_dir)
        self.step_index = int(step_index)
        self.pass_name = str(pass_name)

    @property
    def name(self) -> str:
        return f"CheckpointSavePass[{self.step_index}]"

    async def run(self, circuit, data) -> None:  # noqa: ANN001
        checkpoint_root = Path(self.checkpoint_dir)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_root / f"checkpoint_{self.step_index:04d}.qasm"
        circuit.save(str(checkpoint_path))
        (checkpoint_root / "latest.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "step_index": self.step_index,
                    "after_pass": self.pass_name,
                    "num_qudits": int(circuit.num_qudits),
                    "num_operations": int(circuit.num_operations),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


CheckpointSavePass.__module__ = MODULE_NAME


def build_machine_model(num_qudits: int, gate_set: str):
    from bqskit import MachineModel  # noqa: PLC0415
    from bqskit.ir.gates import CNOTGate, HGate, RZGate, U1Gate, U2Gate, U3Gate, XGate  # noqa: PLC0415

    if gate_set == "ibm":
        gates = {U1Gate(), U2Gate(), U3Gate(), CNOTGate()}
    elif gate_set == "nam":
        gates = {RZGate(), HGate(), XGate(), CNOTGate()}
    else:
        raise ValueError(f"Unsupported gate set: {gate_set}")
    return MachineModel(num_qudits, gate_set=gates)


def workflow_pass_names(workflow) -> list[str]:  # noqa: ANN001
    return [str(pass_obj.name) for pass_obj in workflow]


def select_workflow_passes(workflow, workflow_mode: str):  # noqa: ANN001
    from bqskit.compiler.workflow import Workflow  # noqa: PLC0415

    if workflow_mode == "standard":
        return workflow, []
    if workflow_mode != "skip-retarget-mapping":
        raise ValueError(f"Unsupported BQSKit workflow mode: {workflow_mode}")

    skipped_names = {"Multi Qudit Retargeting", "SABRE Mapping"}
    selected_passes = []
    skipped_passes = []
    for pass_obj in workflow:
        if str(pass_obj.name) in skipped_names:
            skipped_passes.append(str(pass_obj.name))
        else:
            selected_passes.append(pass_obj)
    return Workflow(selected_passes, name=f"{workflow.name} [{workflow_mode}]"), skipped_passes


def compile_once(
    input_path: str,
    output_path: str,
    gate_set: str,
    optimization_level: int,
    max_synthesis_size: int,
    synthesis_epsilon: float,
    seed: int | None,
    workflow_mode: str,
    checkpoint_dir: str | None,
    result_queue: mp.Queue,
) -> None:
    try:
        from bqskit import Circuit, compile as bqskit_compile  # noqa: PLC0415
        from bqskit.compiler import Compiler  # noqa: PLC0415
        from bqskit.compiler.compile import build_workflow  # noqa: PLC0415
        from bqskit.compiler.workflow import Workflow  # noqa: PLC0415

        start = time.perf_counter()
        circuit = Circuit.from_file(input_path)
        model = build_machine_model(circuit.num_qudits, gate_set)
        if checkpoint_dir is None and workflow_mode == "standard":
            compiled = bqskit_compile(
                circuit,
                model=model,
                optimization_level=int(optimization_level),
                max_synthesis_size=int(max_synthesis_size),
                synthesis_epsilon=float(synthesis_epsilon),
                seed=seed,
            )
            checkpoint_count = 0
            workflow_passes = None
            selected_workflow_passes = None
            skipped_workflow_passes = []
        else:
            checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
            if checkpoint_dir is not None:
                checkpoint_root.mkdir(parents=True, exist_ok=True)

            workflow = build_workflow(
                circuit,
                model,
                int(optimization_level),
                float(synthesis_epsilon),
                int(max_synthesis_size),
                None,
                8,
                seed,
            )
            workflow_passes = workflow_pass_names(workflow)
            workflow, skipped_workflow_passes = select_workflow_passes(workflow, workflow_mode)
            selected_workflow_passes = workflow_pass_names(workflow)
            checkpointed_passes = []
            checkpoint_count = 0
            if checkpoint_dir is None:
                workflow_to_run = workflow
            else:
                for pass_index, pass_obj in enumerate(workflow):
                    checkpointed_passes.append(pass_obj)
                    checkpointed_passes.append(
                        CheckpointSavePass(str(checkpoint_root), pass_index, pass_obj.name)
                    )
                    checkpoint_count += 1
                workflow_to_run = Workflow(
                    checkpointed_passes,
                    name=f"{workflow.name} with checkpoints",
                )
            with Compiler() as compiler:
                compiled = compiler.compile(circuit, workflow_to_run)
        compiled.save(output_path)
        result_queue.put(
            {
                "ok": True,
                "elapsed_seconds": time.perf_counter() - start,
                "input_num_qudits": int(circuit.num_qudits),
                "input_num_operations": int(circuit.num_operations),
                "output_num_qudits": int(compiled.num_qudits),
                "output_num_operations": int(compiled.num_operations),
                "checkpoint_count": checkpoint_count,
                "workflow_mode": workflow_mode,
                "workflow_passes": workflow_passes,
                "selected_workflow_passes": selected_workflow_passes,
                "skipped_workflow_passes": skipped_workflow_passes,
            }
        )
    except Exception as exc:  # pragma: no cover - subprocess diagnostic path
        result_queue.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_bqskit_subprocess(
    input_path: Path,
    output_path: Path,
    gate_set: str,
    optimization_level: int,
    max_synthesis_size: int,
    synthesis_epsilon: float,
    seed: int | None,
    workflow_mode: str,
    checkpoint_dir: Path | None,
    timeout: float | None,
) -> dict[str, Any]:
    result_queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=compile_once,
        args=(
            str(input_path),
            str(output_path),
            gate_set,
            int(optimization_level),
            int(max_synthesis_size),
            float(synthesis_epsilon),
            seed,
            workflow_mode,
            None if checkpoint_dir is None else str(checkpoint_dir),
            result_queue,
        ),
    )

    start = time.perf_counter()
    process.start()
    process.join(timeout)
    elapsed = time.perf_counter() - start
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            "ok": False,
            "timed_out": True,
            "returncode": process.exitcode,
            "elapsed_seconds": elapsed,
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "workflow_mode": workflow_mode,
        }

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {
            "ok": process.exitcode == 0,
            "error": None if process.exitcode == 0 else "BQSKit worker exited without a result.",
        }

    result.update(
        {
            "timed_out": False,
            "returncode": process.exitcode,
            "elapsed_seconds": result.get("elapsed_seconds", elapsed),
        }
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = require_input_qasm(Path(args.input))
    input_stats = load_qasm_stats_or_count(
        input_path,
        stats_path=Path(args.input_stats).resolve() if args.input_stats else None,
    )
    gate_set = str(args.gate_set or input_stats["detected_gate_set"])
    output_path = Path(args.output).resolve() if args.output else None

    with tempfile.TemporaryDirectory(prefix="bqskit_baseline_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        actual_output_path = output_path or (Path(tmp_dir) / "optimized.qasm")
        checkpoint_dir = tmp_dir_path / "checkpoints" if args.checkpoint else None
        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        process = run_bqskit_subprocess(
            input_path=input_path,
            output_path=actual_output_path,
            gate_set=gate_set,
            optimization_level=int(args.optimization_level),
            max_synthesis_size=int(args.max_synthesis_size),
            synthesis_epsilon=float(args.synthesis_epsilon),
            seed=args.seed,
            workflow_mode=str(args.workflow_mode),
            checkpoint_dir=checkpoint_dir,
            timeout=args.timeout,
        )

        checkpoint_metadata = None
        checkpoint_output_used = False
        fallback_output_used = False
        if checkpoint_dir is not None and (checkpoint_dir / "latest.json").exists():
            import json

            checkpoint_metadata = json.loads(
                (checkpoint_dir / "latest.json").read_text(encoding="utf-8")
            )
            checkpoint_path = Path(str(checkpoint_metadata["checkpoint"]))
            if (bool(process.get("timed_out")) or not bool(process.get("ok"))) and checkpoint_path.exists():
                shutil.copy2(checkpoint_path, actual_output_path)
                checkpoint_output_used = True

        if (bool(process.get("timed_out")) or not bool(process.get("ok"))) and not actual_output_path.exists():
            shutil.copy2(input_path, actual_output_path)
            fallback_output_used = True

        output_exists = actual_output_path.exists()
        output_stats = (
            count_qasm_file(actual_output_path, gate_set=gate_set)
            if output_exists
            else None
        )
        if output_stats is not None and output_path is None:
            output_stats["path"] = None

        summary: dict[str, Any] = {
            "tool": "bqskit",
            "input": str(input_path),
            "detected_gate_set": input_stats["detected_gate_set"],
            "target_gate_set": gate_set,
            "optimization_level": int(args.optimization_level),
            "max_synthesis_size": int(args.max_synthesis_size),
            "synthesis_epsilon": float(args.synthesis_epsilon),
            "seed": args.seed,
            "workflow_mode": str(args.workflow_mode),
            "checkpoint_enabled": bool(args.checkpoint),
            "checkpoint_output_used": checkpoint_output_used,
            "checkpoint_metadata": checkpoint_metadata,
            "fallback_output_used": fallback_output_used,
            "kept_output": output_path is not None,
            "output_qasm": str(actual_output_path) if output_path is not None else None,
            "input_stats": input_stats,
            "output_stats": output_stats,
            "comparison": (
                compare_gate_stats(input_stats, output_stats, gate_set)
                if output_stats is not None
                else None
            ),
            "process": {
                **process,
                "output_written": output_exists,
            },
        }

    write_summary(summary, Path(args.summary).resolve() if args.summary else None)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BQSKit on one QASM input, infer IBM/NAM, and report gate counts."
    )
    parser.add_argument("--input", required=True, help="Input OpenQASM2 file.")
    parser.add_argument("--input-stats", default=None, help="Optional cached input stats JSON.")
    parser.add_argument("--output", default=None, help="Optional path to keep optimized QASM.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument("--gate-set", choices=["ibm", "nam"], default=None)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-synthesis-size", type=int, default=2)
    parser.add_argument("--synthesis-epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workflow-mode",
        choices=["standard", "skip-retarget-mapping"],
        default="standard",
        help=(
            "standard uses BQSKit's off-the-shelf workflow. "
            "skip-retarget-mapping removes top-level Multi Qudit Retargeting "
            "and SABRE Mapping passes before running the remaining workflow."
        ),
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help=(
            "Insert pass-level checkpoint saves into the standard BQSKit workflow. "
            "If the worker times out, the latest checkpoint is used as output; "
            "if no checkpoint exists, the input circuit is copied as fallback."
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
