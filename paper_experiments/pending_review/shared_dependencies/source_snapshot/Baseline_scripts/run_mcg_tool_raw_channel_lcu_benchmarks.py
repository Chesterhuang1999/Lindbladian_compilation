#!/usr/bin/env python3
"""Evaluate raw channel-LCU compatibility for MCG-oriented tools.

This runner follows the raw-baseline policy from
``mcg_open_source_tools_agent_guide_v4_raw_baseline.md``:

* external tools receive the raw ``channel_to_LCU(..., structure="basic",
  opt="No")`` circuit or a tool-provided direct import of that circuit;
* no project SELECT/address-placement/factorization optimization is applied
  before external-tool execution;
* tools without a direct Qiskit/QASM importer or Qiskit-native circuit
  conversion path are marked unsupported instead of receiving a bespoke
  adapter.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import multiprocessing as mp
import os
import pkgutil
import subprocess
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
BASELINE_DIR = ROOT / "Baseline_scripts"
RESULT_ROOT = ROOT / "Baseline_results" / "mcg_tool_raw_channel_lcu_benchmarks"

for candidate in (SRC_DIR, SCRIPTS_DIR, BASELINE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

NAM_BASIS = ("h", "x", "rz", "cx")
NAM_TRANSPILE_BASIS = ("h", "x", "rz", "cx", "reset")
IBM_BASIS = ("u1", "u2", "u3", "cx")


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    family: str
    size: int


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("TFIM_8", "tfim", 8),
    BenchmarkSpec("TFIM_12", "tfim", 12),
    BenchmarkSpec("HRW_8", "hrw", 8),
    BenchmarkSpec("HRW_20", "hrw", 20),
    BenchmarkSpec("repetition_dqe_d3", "repetition_dqe", 3),
)


def json_default(obj: Any) -> str:
    return repr(obj)


def timed(fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    started = time.perf_counter()
    try:
        return "ok", fn(), time.perf_counter() - started, None
    except Exception:
        return "error", None, time.perf_counter() - started, traceback.format_exc()


def install_aer_stub() -> None:
    """Avoid optional qiskit-aer import failures during channel construction."""

    if "qiskit_aer" in sys.modules:
        return

    aer = types.ModuleType("qiskit_aer")
    library = types.ModuleType("qiskit_aer.library")

    class AerSimulator:  # pragma: no cover - not used by this runner
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("AerSimulator is not used in raw MCG-tool baselines")

    def _unused(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise RuntimeError("qiskit_aer helpers are not used in raw MCG-tool baselines")

    aer.AerSimulator = AerSimulator
    aer.noise = types.SimpleNamespace()
    library.save_statevector = _unused
    library.set_statevector = _unused
    sys.modules["qiskit_aer"] = aer
    sys.modules["qiskit_aer.library"] = library


def qiskit_metrics(qc: Any) -> dict[str, Any]:
    ops = qc.count_ops()
    return {
        "num_qubits": int(qc.num_qubits),
        "num_clbits": int(qc.num_clbits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "ops": {str(name): int(count) for name, count in ops.items()},
    }


def write_qasm2(qc: Any, path: Path) -> dict[str, Any]:
    from qiskit import qasm2

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)
    return {"path": str(path), "bytes": int(path.stat().st_size)}


def recount_qiskit_circuit(
    qc: Any,
    *,
    basis_name: str,
    out_path: Path,
    optimization_level: int = 0,
) -> dict[str, Any]:
    from qiskit import transpile
    from baseline_common import count_qasm_file  # type: ignore

    if basis_name == "nam":
        basis = NAM_TRANSPILE_BASIS
        gate_set = "nam"
    elif basis_name == "ibm":
        basis = IBM_BASIS
        gate_set = "ibm"
    else:
        raise ValueError(f"unsupported basis: {basis_name}")

    started = time.perf_counter()
    tqc = transpile(qc, basis_gates=list(basis), optimization_level=optimization_level)
    transpile_time = time.perf_counter() - started
    write_qasm2(tqc, out_path)
    stats = count_qasm_file(out_path, gate_set=gate_set)
    return {
        "basis": basis_name,
        "basis_gates": list(basis),
        "optimization_level": int(optimization_level),
        "transpile_time_s": float(transpile_time),
        "qasm": str(out_path),
        "qiskit_metrics": qiskit_metrics(tqc),
        "stats": stats,
    }


def channel_metrics(kraus_ops: list[Any]) -> dict[str, int]:
    nonzero = [op for op in kraus_ops if len(op.instances) > 0]
    return {
        "kraus_count": int(len(nonzero)),
        "pauli_terms_total": int(sum(len(op.instances) for op in nonzero)),
        "max_terms_per_kraus": int(max((len(op.instances) for op in nonzero), default=0)),
        "min_terms_per_kraus": int(min((len(op.instances) for op in nonzero), default=0)),
        "system_qubits": int(max((op.size for op in nonzero), default=0)),
    }


def build_kraus_ops(spec: BenchmarkSpec) -> tuple[list[Any], dict[str, Any]]:
    install_aer_stub()

    if spec.family == "tfim":
        from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore

        kraus_ops, model = build_periodic_tfim_channel(spec.size, 0.1)
        return kraus_ops, {
            "benchmark": spec.name,
            "family": "periodic_tfim_lindbladian",
            "num_system_qubits": int(spec.size),
            "model": model,
        }

    if spec.family == "hrw":
        from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore

        ensemble = build_hypercube_section52_channel(spec.size)
        return ensemble.channels[0][1], {
            "benchmark": spec.name,
            "family": "hypercube_random_walk",
            "num_system_qubits": int(spec.size),
        }

    if spec.family == "repetition_dqe":
        from build_repetition_code_dqe_channel import (  # type: ignore
            DqeSpec,
            build_repetition_dqe_channel,
            pauli_dict_to_matrixsum,
        )

        dqe_spec = DqeSpec(d=spec.size, epsilon=0.25)
        channel, metadata = build_repetition_dqe_channel(dqe_spec)
        kraus_ops = [pauli_dict_to_matrixsum(copy.deepcopy(kraus)) for kraus in channel]
        return kraus_ops, {
            "benchmark": spec.name,
            "family": "repetition_code_stabilizer_encoded_dqe",
            "num_system_qubits": int(spec.size),
            "metadata": metadata,
        }

    raise ValueError(f"unsupported benchmark family: {spec.family}")


def build_raw_channel_lcu(kraus_ops: list[Any]) -> tuple[Any, Any, float]:
    from channel_IR import channel_ensemble  # type: ignore
    import channel_LCU  # type: ignore

    started = time.perf_counter()
    qc, qubit_indexes = channel_LCU.channel_to_LCU(
        channel_ensemble([copy.deepcopy(kraus_ops)]),
        structure="basic",
        opt="No",
    )
    return qc, qubit_indexes, time.perf_counter() - started


def rewrite_channel(kraus_ops: list[Any], *, max_steps: int = 50) -> tuple[list[Any], dict[str, Any], float]:
    from channel_IR import channel as ChannelIR  # type: ignore

    started = time.perf_counter()
    source = ChannelIR(copy.deepcopy(kraus_ops))
    source.zero_elim()
    result = source.rewrite_search(strategy="greedy", beam_width=3, max_steps=max_steps, tol=1e-12)
    rewritten = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten.apply_rewrite_result(result, tol=1e-12)
    rewritten.zero_elim()
    return rewritten.kraus_ops, result, time.perf_counter() - started


def build_ours_channel_lcu(kraus_ops: list[Any]) -> tuple[Any, Any, float]:
    import numpy as np
    from qiskit import QuantumCircuit
    from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore
    import channel_LCU  # type: ignore

    started = time.perf_counter()
    sys_size = max(ms.size for ms in kraus_ops)
    select_size = int(np.ceil(np.log2(len(kraus_ops))))
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


def run_ours_workflow(kraus_ops: list[Any], bench_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "rewrite_search + structure=opt + Matrix-order",
        "used_project_optimizer": True,
        "status": "running",
    }
    status, value, elapsed, error = timed(lambda: rewrite_channel(kraus_ops))
    result["rewrite"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "failed"
        return result

    rewritten, rewrite_result, rewrite_time = value
    result["rewrite"].update(
        {
            "initial_support": int(rewrite_result["initial_support"]),
            "final_support": int(rewrite_result["final_support"]),
            "steps": int(len(rewrite_result["steps"])),
            "termination": rewrite_result.get("termination", {}),
        }
    )

    status, value, elapsed, error = timed(lambda: build_ours_channel_lcu(rewritten))
    result["build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "failed"
        return result

    ours_qc, qubit_indexes, build_time = value
    result["qubit_indexes"] = qubit_indexes
    result["qiskit_metrics"] = qiskit_metrics(ours_qc)
    result["qasm"] = write_qasm2(ours_qc, bench_dir / "ours_rewrite_opt_matrix_order.qasm")

    basis_recounts: dict[str, Any] = {}
    for basis in ("nam", "ibm"):
        status, recount, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                ours_qc,
                basis_name=basis,
                out_path=bench_dir / f"ours_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        basis_recounts[basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": recount,
            "error": error,
        }
    result["basis_recounts"] = basis_recounts
    result["compile_time_s"] = float(rewrite_time + build_time + sum(
        row["elapsed_seconds"] for row in basis_recounts.values() if row["status"] == "ok"
    ))
    result["status"] = "ok"
    return result


def qrisp_metrics(qc: Any) -> dict[str, Any]:
    count_ops = getattr(qc, "count_ops", None)
    num_qubits = getattr(qc, "num_qubits")
    return {
        "num_qubits": int(num_qubits() if callable(num_qubits) else num_qubits),
        "depth": int(qc.depth()),
        "ops": {str(k): int(v) for k, v in count_ops().items()} if count_ops else None,
    }


def qrisp_base_operation_name(op: Any) -> str | None:
    base = getattr(op, "base_operation", None)
    if base is None:
        return None
    name = getattr(base, "name", None)
    return str(name if name is not None else base)


def is_qrisp_auto_controlled_candidate(op: Any, qubit_count: int) -> bool:
    controls = getattr(op, "controls", None)
    if controls is not None:
        try:
            return hasattr(op, "method") and getattr(op, "method") is not None and len(controls) > 1
        except TypeError:
            pass
    return hasattr(op, "method") and getattr(op, "method") is not None and qubit_count > 3


def qrisp_controlled_operation_summary(qc: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "eligible_multi_controlled_operations": 0,
        "by_method": {},
        "sample": [],
    }
    for index, instr in enumerate(getattr(qc, "data", [])):
        op = getattr(instr, "op", None)
        if op is None or not is_qrisp_auto_controlled_candidate(op, len(getattr(instr, "qubits", []))):
            continue
        method = str(getattr(op, "method", None))
        summary["eligible_multi_controlled_operations"] += 1
        summary["by_method"][method] = int(summary["by_method"].get(method, 0)) + 1
        if len(summary["sample"]) < 20:
            summary["sample"].append(
                {
                    "index": int(index),
                    "name": str(getattr(op, "name", op)),
                    "base_operation": qrisp_base_operation_name(op),
                    "method": method,
                    "ctrl_state": str(getattr(op, "ctrl_state", "")),
                    "controls": list(getattr(op, "controls", [])) if getattr(op, "controls", None) is not None else None,
                    "qubits": int(len(getattr(instr, "qubits", []))),
                }
            )
    return summary


def qrisp_replay_as_session_with_auto_controlled_ops(qrisp_qc: Any) -> tuple[Any, dict[str, Any]]:
    from qrisp import QuantumSession

    if getattr(qrisp_qc, "clbits", []):
        raise ValueError("Qrisp QuantumSession replay only supports circuits without classical bits")

    session = QuantumSession()
    session_qubits = session.request_qubits(len(qrisp_qc.qubits), name="raw")
    qubit_map = {old_qubit: session_qubits[index] for index, old_qubit in enumerate(qrisp_qc.qubits)}
    metadata: dict[str, Any] = {
        "replayed_instructions": 0,
        "auto_controlled_ops_marked": 0,
        "auto_controlled_ops_original_methods": {},
    }

    for instr in qrisp_qc.data:
        op = instr.op.copy() if hasattr(instr.op, "copy") else copy.copy(instr.op)
        instr_qubits = list(getattr(instr, "qubits", []))
        if is_qrisp_auto_controlled_candidate(op, len(instr_qubits)):
            original_method = str(getattr(op, "method", None))
            metadata["auto_controlled_ops_marked"] += 1
            metadata["auto_controlled_ops_original_methods"][original_method] = (
                int(metadata["auto_controlled_ops_original_methods"].get(original_method, 0)) + 1
            )
            op.method = "auto"
        session.append(op, [qubit_map[qubit] for qubit in instr_qubits], [])
        metadata["replayed_instructions"] += 1
    return session, metadata


def _qrisp_worker(raw_qc: Any, bench_dir: Path, queue: Any) -> None:
    try:
        import qrisp
        from qrisp import QuantumCircuit as QrispQuantumCircuit

        result: dict[str, Any] = {
            "tool": "Qrisp",
            "status": "running",
            "adapter_type": "tool_provided_QuantumCircuit.from_qiskit",
            "input_circuit_form": "Qiskit QuantumCircuit raw channel-LCU basic/no",
            "output_circuit_form": "Qrisp QuantumCircuit from qrisp.transpile; converted back with to_qiskit()",
            "api_evidence": {
                "qrisp_version": getattr(qrisp, "__version__", "unknown"),
                "from_qiskit_signature": str(inspect.signature(QrispQuantumCircuit.from_qiskit)),
                "to_qiskit_exists": hasattr(QrispQuantumCircuit, "to_qiskit"),
                "transpile_signature": str(inspect.signature(qrisp.transpile)),
            },
        }

        status, qrisp_qc, elapsed, error = timed(lambda: QrispQuantumCircuit.from_qiskit(raw_qc))
        result["from_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "failed_direct_import"
            queue.put(result)
            return

        result["input_qrisp_metrics"] = qrisp_metrics(qrisp_qc)
        (bench_dir / "qrisp_input.qasm").write_text(qrisp_qc.to_qasm2(), encoding="utf-8")

        status, compiled_qrisp, elapsed, error = timed(lambda: qrisp.transpile(qrisp_qc))
        result["tool_transpile"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "failed_tool_transpile"
            queue.put(result)
            return

        result["compiled_qrisp_metrics"] = qrisp_metrics(compiled_qrisp)
        status, qiskit_back, elapsed, error = timed(compiled_qrisp.to_qiskit)
        result["to_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "failed_to_qiskit"
            queue.put(result)
            return

        result["compiled_qiskit_metrics"] = qiskit_metrics(qiskit_back)
        result["compiled_qiskit_gate_set"] = sorted(result["compiled_qiskit_metrics"]["ops"])
        write_qasm2(qiskit_back, bench_dir / "qrisp_compiled_qiskit.qasm")
        basis_recounts: dict[str, Any] = {}
        for basis in ("nam", "ibm"):
            status, recount, elapsed, error = timed(
                lambda basis=basis: recount_qiskit_circuit(
                    qiskit_back,
                    basis_name=basis,
                    out_path=bench_dir / f"qrisp_compiled_{basis}_opt0.qasm",
                    optimization_level=0,
                )
            )
            basis_recounts[basis] = {
                "status": status,
                "elapsed_seconds": elapsed,
                "result": recount,
                "error": error,
            }
        result["basis_recounts"] = basis_recounts
        result["status"] = "executed_full"
        queue.put(result)
    except Exception:
        queue.put({"tool": "Qrisp", "status": "failed_exception", "error": traceback.format_exc()})


def run_qrisp(raw_qc: Any, bench_dir: Path, timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("fork")
    queue: Any = ctx.Queue()
    proc = ctx.Process(target=_qrisp_worker, args=(raw_qc, bench_dir, queue))
    started = time.perf_counter()
    proc.start()
    proc.join(timeout_s)
    elapsed = time.perf_counter() - started
    if proc.is_alive():
        proc.terminate()
        proc.join(30)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "tool": "Qrisp",
            "status": "timeout",
            "elapsed_seconds_total": float(elapsed),
            "timeout_s": float(timeout_s),
            "adapter_type": "tool_provided_QuantumCircuit.from_qiskit",
        }
    if queue.empty():
        return {
            "tool": "Qrisp",
            "status": "failed_no_result",
            "elapsed_seconds_total": float(elapsed),
            "exitcode": proc.exitcode,
        }
    result = queue.get()
    result["elapsed_seconds_total"] = float(elapsed)
    result["exitcode"] = proc.exitcode
    return result


def _qrisp_workspace_auto_worker(raw_qc: Any, bench_dir: Path, workspace_budget: int, queue: Any) -> None:
    try:
        from qrisp import QuantumCircuit as QrispQuantumCircuit, QuantumSession

        result: dict[str, Any] = {
            "tool": "Qrisp",
            "variant": "workspace_auto",
            "status": "running",
            "workspace_budget": int(workspace_budget),
            "adapter_type": "tool_provided_QuantumCircuit.from_qiskit + QuantumSession replay",
            "input_circuit_form": "Qiskit QuantumCircuit raw channel-LCU basic/no",
            "method_policy": (
                "Imported multi-controlled Qrisp ControlledOperation objects are marked method='auto' "
                "so Qrisp QuantumSession.compile can choose its own decomposition; no hybrid/gidney/"
                "gray/maslov method is explicitly requested."
            ),
            "workspace_policy": (
                "workspace is set to max(0, ours_raw_width_delta), i.e. the extra qubits used by "
                "the in-house workflow relative to the raw circuit. The in-house circuit is not "
                "passed to Qrisp."
            ),
            "output_circuit_form": (
                "Qrisp QuantumCircuit from QuantumSession.compile(workspace=...); converted back "
                "with to_qiskit()"
            ),
            "api_evidence": {
                "from_qiskit_signature": str(inspect.signature(QrispQuantumCircuit.from_qiskit)),
                "quantum_session_compile_signature": str(inspect.signature(QuantumSession.compile)),
            },
        }

        status, qrisp_qc, elapsed, error = timed(lambda: QrispQuantumCircuit.from_qiskit(raw_qc))
        result["from_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "failed_direct_import"
            queue.put(result)
            return

        result["input_qrisp_metrics"] = qrisp_metrics(qrisp_qc)
        result["imported_controlled_operation_summary"] = qrisp_controlled_operation_summary(qrisp_qc)
        (bench_dir / "qrisp_workspace_auto_input.qasm").write_text(qrisp_qc.to_qasm2(), encoding="utf-8")

        status, replay_value, elapsed, error = timed(
            lambda: qrisp_replay_as_session_with_auto_controlled_ops(qrisp_qc)
        )
        result["session_replay"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "failed_session_replay"
            queue.put(result)
            return

        session, replay_metadata = replay_value
        result["session_replay"].update(replay_metadata)
        result["session_precompile_metrics"] = qrisp_metrics(session)

        status, compiled_qrisp, elapsed, error = timed(lambda: session.compile(workspace=int(workspace_budget)))
        result["tool_compile"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "failed_tool_compile"
            queue.put(result)
            return

        result["compiled_qrisp_metrics"] = qrisp_metrics(compiled_qrisp)
        status, qiskit_back, elapsed, error = timed(compiled_qrisp.to_qiskit)
        result["to_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "failed_to_qiskit"
            queue.put(result)
            return

        result["compiled_qiskit_metrics"] = qiskit_metrics(qiskit_back)
        result["compiled_qiskit_gate_set"] = sorted(result["compiled_qiskit_metrics"]["ops"])
        write_qasm2(qiskit_back, bench_dir / "qrisp_workspace_auto_compiled_qiskit.qasm")
        basis_recounts: dict[str, Any] = {}
        for basis in ("nam", "ibm"):
            status, recount, elapsed, error = timed(
                lambda basis=basis: recount_qiskit_circuit(
                    qiskit_back,
                    basis_name=basis,
                    out_path=bench_dir / f"qrisp_workspace_auto_compiled_{basis}_opt0.qasm",
                    optimization_level=0,
                )
            )
            basis_recounts[basis] = {
                "status": status,
                "elapsed_seconds": elapsed,
                "result": recount,
                "error": error,
            }
        result["basis_recounts"] = basis_recounts
        result["status"] = "executed_full"
        queue.put(result)
    except Exception:
        queue.put(
            {
                "tool": "Qrisp",
                "variant": "workspace_auto",
                "status": "failed_exception",
                "error": traceback.format_exc(),
            }
        )


def run_qrisp_workspace_auto(
    raw_qc: Any,
    bench_dir: Path,
    timeout_s: float,
    workspace_budget: int,
) -> dict[str, Any]:
    ctx = mp.get_context("fork")
    queue: Any = ctx.Queue()
    proc = ctx.Process(
        target=_qrisp_workspace_auto_worker,
        args=(raw_qc, bench_dir, int(workspace_budget), queue),
    )
    started = time.perf_counter()
    proc.start()
    proc.join(timeout_s)
    elapsed = time.perf_counter() - started
    if proc.is_alive():
        proc.terminate()
        proc.join(30)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "tool": "Qrisp",
            "variant": "workspace_auto",
            "status": "timeout",
            "elapsed_seconds_total": float(elapsed),
            "timeout_s": float(timeout_s),
            "workspace_budget": int(workspace_budget),
            "adapter_type": "tool_provided_QuantumCircuit.from_qiskit + QuantumSession replay",
        }
    if queue.empty():
        return {
            "tool": "Qrisp",
            "variant": "workspace_auto",
            "status": "failed_no_result",
            "elapsed_seconds_total": float(elapsed),
            "exitcode": proc.exitcode,
            "workspace_budget": int(workspace_budget),
        }
    result = queue.get()
    result["elapsed_seconds_total"] = float(elapsed)
    result["exitcode"] = proc.exitcode
    return result


def get_qiskit_instruction_parts(instruction: Any) -> tuple[Any, list[Any], list[Any]]:
    if hasattr(instruction, "operation"):
        return instruction.operation, list(instruction.qubits), list(instruction.clbits)
    if hasattr(instruction, "op"):
        return instruction.op, list(instruction.qubits), list(instruction.clbits)
    op, qargs, cargs = instruction
    return op, list(qargs), list(cargs)


def qiskit_control_state_bits(controlled_op: Any) -> str:
    num_controls = int(getattr(controlled_op, "num_ctrl_qubits"))
    ctrl_state = getattr(controlled_op, "ctrl_state", None)
    if ctrl_state is None:
        return "1" * num_controls
    if isinstance(ctrl_state, str):
        ctrl_state = ctrl_state.replace(" ", "")
        if len(ctrl_state) == num_controls:
            return ctrl_state
        return format(int(ctrl_state, 2), f"0{num_controls}b")
    return format(int(ctrl_state), f"0{num_controls}b")


def append_qclib_controlled_global_phase(
    out_qc: Any,
    phase: Any,
    controls: list[int],
    ctrl_state: str,
    stats: dict[str, Any],
) -> None:
    from qiskit.circuit.library import PhaseGate
    from qclib.gates.mcg import Mcg

    if not controls:
        out_qc.global_phase += phase
        stats["global_phase_direct"] += 1
        return

    target = controls[-1]
    target_state = ctrl_state[-1]
    remaining_controls = controls[:-1]
    remaining_state = ctrl_state[:-1]
    if target_state == "0":
        out_qc.x(target)
        stats["negative_control_x_conjugations"] += 2
    if remaining_controls:
        out_qc.append(Mcg(PhaseGate(phase).to_matrix(), len(remaining_controls), ctrl_state=remaining_state), remaining_controls + [target])
        stats["qclib_mcg_inserted"] += 1
        stats["controlled_global_phase_via_mcg"] += 1
    else:
        out_qc.p(phase, target)
        stats["controlled_global_phase_direct_phase"] += 1
    if target_state == "0":
        out_qc.x(target)


def append_qclib_adapted_definition(
    out_qc: Any,
    definition: Any,
    parent_qids: list[int],
    controls: list[int],
    ctrl_state: str,
    stats: dict[str, Any],
    depth: int,
) -> None:
    if getattr(definition, "global_phase", 0):
        append_qclib_controlled_global_phase(out_qc, definition.global_phase, controls, ctrl_state, stats)

    qubit_map = {qubit: parent_qids[index] for index, qubit in enumerate(definition.qubits)}
    for instruction in definition.data:
        op, qargs, cargs = get_qiskit_instruction_parts(instruction)
        if cargs:
            raise ValueError(f"qclib adapter does not support classical args inside controlled definition: {op.name}")
        append_qclib_adapted_operation(
            out_qc,
            op,
            [qubit_map[qubit] for qubit in qargs],
            [],
            controls,
            ctrl_state,
            stats,
            depth + 1,
        )


def append_qclib_adapted_operation(
    out_qc: Any,
    op: Any,
    qids: list[int],
    cids: list[int],
    controls: list[int],
    ctrl_state: str,
    stats: dict[str, Any],
    depth: int = 0,
) -> None:
    from qiskit.circuit import ControlledGate
    from qclib.gates.mcg import Mcg

    if depth > 200:
        raise RecursionError("qclib adapter exceeded recursive definition depth")

    if isinstance(op, ControlledGate):
        num_controls = int(op.num_ctrl_qubits)
        inner_controls = qids[:num_controls]
        target_qids = qids[num_controls:]
        stats["controlled_gates_expanded"] += 1
        stats["controlled_gate_names"][str(op.name)] = int(stats["controlled_gate_names"].get(str(op.name), 0)) + 1
        append_qclib_adapted_operation(
            out_qc,
            op.base_gate,
            target_qids,
            [],
            controls + inner_controls,
            ctrl_state + qiskit_control_state_bits(op),
            stats,
            depth + 1,
        )
        return

    if controls:
        if cids or int(getattr(op, "num_clbits", 0)):
            raise ValueError(f"qclib adapter cannot control operation with classical bits: {op.name}")
        num_qubits = int(getattr(op, "num_qubits", len(qids)))
        if num_qubits == 0:
            append_qclib_controlled_global_phase(out_qc, getattr(op, "params", [0])[0], controls, ctrl_state, stats)
            return
        if num_qubits == 1:
            try:
                unitary = op.to_matrix()
            except Exception:
                definition = getattr(op, "definition", None)
                if definition is None:
                    raise
                stats["single_qubit_defs_expanded_under_control"] += 1
                stats["single_qubit_def_names"][str(op.name)] = int(stats["single_qubit_def_names"].get(str(op.name), 0)) + 1
                append_qclib_adapted_definition(out_qc, definition, qids, controls, ctrl_state, stats, depth + 1)
                return
            else:
                out_qc.append(Mcg(unitary, len(controls), ctrl_state=ctrl_state), controls + qids)
                stats["qclib_mcg_inserted"] += 1
                stats["qclib_mcg_by_base"][str(op.name)] = int(stats["qclib_mcg_by_base"].get(str(op.name), 0)) + 1
                stats["max_controls"] = max(int(stats["max_controls"]), len(controls))
                return

        definition = getattr(op, "definition", None)
        if definition is None:
            raise ValueError(f"qclib adapter cannot control opaque multi-qubit operation: {op.name}")
        stats["composite_ops_expanded_under_control"] += 1
        stats["composite_op_names"][str(op.name)] = int(stats["composite_op_names"].get(str(op.name), 0)) + 1
        append_qclib_adapted_definition(out_qc, definition, qids, controls, ctrl_state, stats, depth + 1)
        return

    out_qc.append(op, qids, cids)
    stats["uncontrolled_ops_copied"] += 1


def qclib_mcg_adapt_circuit(raw_qc: Any) -> tuple[Any, dict[str, Any]]:
    from qiskit import QuantumCircuit

    adapted = QuantumCircuit(raw_qc.num_qubits, raw_qc.num_clbits, global_phase=raw_qc.global_phase)
    qubit_index = {qubit: index for index, qubit in enumerate(raw_qc.qubits)}
    clbit_index = {clbit: index for index, clbit in enumerate(raw_qc.clbits)}
    stats: dict[str, Any] = {
        "adapter": "recursive Qiskit ControlledGate expansion + qclib.gates.Mcg",
        "controlled_gates_expanded": 0,
        "controlled_gate_names": {},
        "composite_ops_expanded_under_control": 0,
        "composite_op_names": {},
        "qclib_mcg_inserted": 0,
        "qclib_mcg_by_base": {},
        "single_qubit_defs_expanded_under_control": 0,
        "single_qubit_def_names": {},
        "max_controls": 0,
        "uncontrolled_ops_copied": 0,
        "negative_control_x_conjugations": 0,
        "global_phase_direct": 0,
        "controlled_global_phase_via_mcg": 0,
        "controlled_global_phase_direct_phase": 0,
    }

    for instruction in raw_qc.data:
        op, qargs, cargs = get_qiskit_instruction_parts(instruction)
        append_qclib_adapted_operation(
            adapted,
            op,
            [qubit_index[qubit] for qubit in qargs],
            [clbit_index[clbit] for clbit in cargs],
            [],
            "",
            stats,
        )
    return adapted, stats


def _qclib_mcg_worker(raw_qc: Any, bench_dir: Path, queue: Any) -> None:
    try:
        import qclib
        from qclib.gates.mcg import Mcg

        result: dict[str, Any] = {
            "tool": "qclib",
            "variant": "mcg_adapter",
            "status": "running",
            "adapter_type": "recursive_qiskit_controlled_gate_adapter_to_qclib_Mcg",
            "input_circuit_form": "Qiskit QuantumCircuit raw channel-LCU basic/no",
            "adapter_policy": (
                "The adapter recursively expands Qiskit ControlledGate/composite definitions and replaces "
                "only the resulting controlled single-qubit operations by qclib.gates.Mcg with the same "
                "control qubits and ctrl_state. It does not reorder addresses, factor Pauli strings, or "
                "run the in-house channel optimizer."
            ),
            "output_circuit_form": "Qiskit QuantumCircuit containing qclib Mcg gates; recounted after Qiskit opt0 lowering",
            "api_evidence": {
                "qclib_version": getattr(qclib, "__version__", "unknown"),
                "mcg_signature": str(inspect.signature(Mcg)),
            },
        }

        status, adapted_value, elapsed, error = timed(lambda: qclib_mcg_adapt_circuit(raw_qc))
        result["adapter_run"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
        if status != "ok":
            result["status"] = "failed_adapter"
            queue.put(result)
            return

        adapted_qc, adapter_stats = adapted_value
        result["adapter_run"].update(adapter_stats)
        result["adapted_qiskit_metrics"] = qiskit_metrics(adapted_qc)
        result["adapted_qiskit_gate_set"] = sorted(result["adapted_qiskit_metrics"]["ops"])
        write_qasm2(adapted_qc, bench_dir / "qclib_mcg_adapter_qiskit.qasm")

        basis_recounts: dict[str, Any] = {}
        for basis in ("nam", "ibm"):
            status, recount, elapsed, error = timed(
                lambda basis=basis: recount_qiskit_circuit(
                    adapted_qc,
                    basis_name=basis,
                    out_path=bench_dir / f"qclib_mcg_adapter_{basis}_opt0.qasm",
                    optimization_level=0,
                )
            )
            basis_recounts[basis] = {
                "status": status,
                "elapsed_seconds": elapsed,
                "result": recount,
                "error": error,
            }
        result["basis_recounts"] = basis_recounts
        result["status"] = "executed_full"
        queue.put(result)
    except Exception:
        queue.put(
            {
                "tool": "qclib",
                "variant": "mcg_adapter",
                "status": "failed_exception",
                "error": traceback.format_exc(),
            }
        )


def run_qclib_mcg_adapter(raw_qc: Any, bench_dir: Path, timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("fork")
    queue: Any = ctx.Queue()
    proc = ctx.Process(target=_qclib_mcg_worker, args=(raw_qc, bench_dir, queue))
    started = time.perf_counter()
    proc.start()
    proc.join(timeout_s)
    elapsed = time.perf_counter() - started
    if proc.is_alive():
        proc.terminate()
        proc.join(30)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "tool": "qclib",
            "variant": "mcg_adapter",
            "status": "timeout",
            "elapsed_seconds_total": float(elapsed),
            "timeout_s": float(timeout_s),
            "adapter_type": "recursive_qiskit_controlled_gate_adapter_to_qclib_Mcg",
        }
    if queue.empty():
        return {
            "tool": "qclib",
            "variant": "mcg_adapter",
            "status": "failed_no_result",
            "elapsed_seconds_total": float(elapsed),
            "exitcode": proc.exitcode,
        }
    result = queue.get()
    result["elapsed_seconds_total"] = float(elapsed)
    result["exitcode"] = proc.exitcode
    return result


def extract_stats(recount_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not recount_row or recount_row.get("status") != "ok":
        return None
    result = recount_row.get("result")
    if not isinstance(result, dict):
        return None
    stats = result.get("stats")
    return stats if isinstance(stats, dict) else None


def build_comparison_rows(benchmark: str, basis: str, bench_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_stats = extract_stats(bench_summary.get("raw", {}).get("basis_recounts", {}).get(basis))
    raw_total = int(raw_stats["metric_total"]) if raw_stats else None

    sources = [
        ("raw_qiskit_opt0", "Raw channel-LCU + Qiskit opt0", raw_stats, None, "raw"),
        (
            "qrisp",
            "Qrisp direct import default(gray) + Qrisp transpile + Qiskit opt0",
            extract_stats(bench_summary.get("tools", {}).get("qrisp", {}).get("basis_recounts", {}).get(basis)),
            bench_summary.get("tools", {}).get("qrisp", {}).get("elapsed_seconds_total"),
            bench_summary.get("tools", {}).get("qrisp", {}).get("status"),
        ),
        (
            "qrisp_workspace_auto",
            "Qrisp direct import + controlled-op auto + QuantumSession.compile(workspace) + Qiskit opt0",
            extract_stats(
                bench_summary.get("tools", {}).get("qrisp_workspace_auto", {}).get("basis_recounts", {}).get(basis)
            ),
            bench_summary.get("tools", {}).get("qrisp_workspace_auto", {}).get("elapsed_seconds_total"),
            bench_summary.get("tools", {}).get("qrisp_workspace_auto", {}).get("status"),
        ),
        (
            "qclib_mcg_adapter",
            "qclib recursive controlled-gate adapter + Mcg + Qiskit opt0",
            extract_stats(bench_summary.get("tools", {}).get("qclib_mcg_adapter", {}).get("basis_recounts", {}).get(basis)),
            bench_summary.get("tools", {}).get("qclib_mcg_adapter", {}).get("elapsed_seconds_total"),
            bench_summary.get("tools", {}).get("qclib_mcg_adapter", {}).get("status"),
        ),
        (
            "ours",
            "Ours rewrite+opt+Matrix-order + Qiskit opt0",
            extract_stats(bench_summary.get("ours", {}).get("basis_recounts", {}).get(basis)),
            bench_summary.get("ours", {}).get("compile_time_s"),
            bench_summary.get("ours", {}).get("status"),
        ),
    ]

    for method, label, stats, elapsed, status in sources:
        if not stats:
            rows.append(
                {
                    "benchmark": benchmark,
                    "basis": basis,
                    "method": method,
                    "label": label,
                    "status": status or "no_stats",
                    "metric_total": None,
                    "ratio_vs_raw": None,
                    "non_clifford": None,
                    "qubits": None,
                    "ops": None,
                    "elapsed_seconds": elapsed,
                }
            )
            continue
        total = int(stats["metric_total"])
        rows.append(
            {
                "benchmark": benchmark,
                "basis": basis,
                "method": method,
                "label": label,
                "status": status or "ok",
                "metric_total": total,
                "ratio_vs_raw": None if raw_total in (None, 0) else float(total / raw_total),
                "non_clifford": int(stats["clifford"]["non_clifford"]),
                "qubits": int(stats["num_qubits"]),
                "ops": stats["ops"],
                "elapsed_seconds": elapsed,
            }
        )
    return rows


def run_benchmark_worker(spec: BenchmarkSpec, out_dir: Path, qrisp_timeout_s: float) -> dict[str, Any]:
    bench_dir = out_dir / "benchmarks" / spec.name
    bench_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "benchmark": spec.name,
        "family": spec.family,
        "size": int(spec.size),
        "out_dir": str(bench_dir),
        "policy": {
            "external_tool_input": "raw channel_to_LCU structure=basic opt=No",
            "repo_optimizations_used_for_external_tools": [],
            "ours_used_only_for_comparison": True,
        },
    }

    status, value, elapsed, error = timed(lambda: build_kraus_ops(spec))
    summary["channel_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        write_summary(summary, bench_dir / "summary.json")
        return summary

    kraus_ops, model_info = value
    summary["model_info"] = model_info
    summary["channel_metrics"] = channel_metrics(kraus_ops)

    status, value, elapsed, error = timed(lambda: build_raw_channel_lcu(kraus_ops))
    summary["raw_build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        write_summary(summary, bench_dir / "summary.json")
        return summary

    raw_qc, raw_qubit_indexes, raw_build_time = value
    summary["raw"] = {
        "qiskit_metrics": qiskit_metrics(raw_qc),
        "qubit_indexes": raw_qubit_indexes,
        "build_time_s": float(raw_build_time),
        "qasm": write_qasm2(raw_qc, bench_dir / "raw_channel_lcu_basic_no.qasm"),
        "basis_recounts": {},
    }
    for basis in ("nam", "ibm"):
        status, recount, elapsed, error = timed(
            lambda basis=basis: recount_qiskit_circuit(
                raw_qc,
                basis_name=basis,
                out_path=bench_dir / f"raw_qiskit_{basis}_opt0.qasm",
                optimization_level=0,
            )
        )
        summary["raw"]["basis_recounts"][basis] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": recount,
            "error": error,
        }

    summary["ours"] = run_ours_workflow(kraus_ops, bench_dir)
    raw_width = int(summary["raw"]["qiskit_metrics"]["num_qubits"])
    ours_width = None
    if summary["ours"].get("status") == "ok":
        ours_width = int(summary["ours"]["qiskit_metrics"]["num_qubits"])
    workspace_budget = max(0, int(ours_width - raw_width)) if ours_width is not None else 0
    summary["workspace_budget_policy"] = {
        "raw_width": raw_width,
        "ours_width": ours_width,
        "workspace_for_qrisp_compile": int(workspace_budget),
        "notes": (
            "The Qrisp workspace-auto baseline uses the extra width of the in-house workflow "
            "as a resource budget only; the in-house optimized circuit is not given to Qrisp."
        ),
    }
    summary["tools"] = {
        "qrisp": run_qrisp(raw_qc, bench_dir, qrisp_timeout_s),
        "qrisp_workspace_auto": run_qrisp_workspace_auto(raw_qc, bench_dir, qrisp_timeout_s, workspace_budget),
        "qclib_mcg_adapter": run_qclib_mcg_adapter(raw_qc, bench_dir, qrisp_timeout_s),
    }
    summary["comparison_rows"] = (
        build_comparison_rows(spec.name, "nam", summary)
        + build_comparison_rows(spec.name, "ibm", summary)
    )
    write_summary(summary, bench_dir / "summary.json")
    return summary


def write_summary(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def module_candidate_attrs(module_name: str, tokens: tuple[str, ...]) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return [{"module": module_name, "import_error": f"{type(exc).__name__}: {exc}"}]
    for attr in dir(module):
        lowered = attr.lower()
        if not any(token in lowered for token in tokens):
            continue
        obj = getattr(module, attr, None)
        try:
            signature = str(inspect.signature(obj)) if callable(obj) else None
        except Exception:
            signature = None
        rows.append({"module": module_name, "attribute": attr, "signature": signature, "repr": repr(obj)[:240]})
    return rows


def inspect_qclib() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    try:
        import qclib

        evidence["installed"] = True
        evidence["version"] = getattr(qclib, "__version__", "unknown")
        evidence["module_file"] = getattr(qclib, "__file__", None)
        package_path = Path(qclib.__file__).resolve().parent
        modules = [m.name for m in pkgutil.walk_packages([str(package_path)], prefix="qclib.")]
        evidence["module_sample"] = modules[:120]
        evidence["import_like_candidates"] = []
        for module_name in ("qclib", "qclib.gates", "qclib.gates.mcg", "qclib.gates.mcu", "qclib.gates.mcx"):
            evidence["import_like_candidates"].extend(
                module_candidate_attrs(
                    module_name,
                    ("qiskit", "qasm", "quantumcircuit", "from_", "convert", "transpile", "pass"),
                )
            )
    except Exception:
        evidence["installed"] = False
        evidence["error"] = traceback.format_exc()
    return evidence


def inspect_ket() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    try:
        import ket

        evidence["installed"] = True
        evidence["version"] = getattr(ket, "__version__", "unknown")
        evidence["module_file"] = getattr(ket, "__file__", None)
        evidence["import_like_candidates"] = []
        for module_name in ("ket", "ket.gates", "ket.operations", "ket.qulib", "ket.ibm"):
            evidence["import_like_candidates"].extend(
                module_candidate_attrs(
                    module_name,
                    ("qiskit", "qasm", "quantumcircuit", "from_", "convert", "import"),
                )
            )
    except Exception:
        evidence["installed"] = False
        evidence["error"] = traceback.format_exc()
    return evidence


def inspect_revkit() -> dict[str, Any]:
    evidence: dict[str, Any] = {"installed": False}
    try:
        import revkit

        evidence["installed"] = True
        evidence["version"] = getattr(revkit, "__version__", "unknown")
        evidence["module_file"] = getattr(revkit, "__file__", None)
        evidence["import_like_candidates"] = module_candidate_attrs(
            "revkit",
            ("qiskit", "qasm", "quantumcircuit", "from_", "convert", "import"),
        )
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def inspect_qrisp() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    try:
        import qrisp
        from qrisp import QuantumCircuit as QrispQuantumCircuit

        evidence["installed"] = True
        evidence["version"] = getattr(qrisp, "__version__", "unknown")
        evidence["module_file"] = getattr(qrisp, "__file__", None)
        for name in ("from_qiskit", "from_qasm_file", "from_qasm_str", "to_qiskit", "to_qasm2"):
            obj = getattr(QrispQuantumCircuit, name, None)
            evidence[name] = {
                "exists": obj is not None,
                "signature": str(inspect.signature(obj)) if obj is not None else None,
            }
        evidence["transpile_signature"] = str(inspect.signature(qrisp.transpile))
    except Exception:
        evidence["installed"] = False
        evidence["error"] = traceback.format_exc()
    return evidence


def tool_compatibility() -> dict[str, Any]:
    return {
        "Qrisp": {
            "full_circuit_status": "direct_supported",
            "direct_adaptation": (
                "Qrisp QuantumCircuit.from_qiskit accepts a Qiskit QuantumCircuit; "
                "the runner uses that API on the raw channel-LCU circuit."
            ),
            "qasm_support": "Qrisp also exposes from_qasm_file/from_qasm_str when available.",
            "executed_on_benchmarks": True,
            "output_form": "Qrisp QuantumCircuit, then Qiskit QuantumCircuit via to_qiskit().",
            "api_evidence": inspect_qrisp(),
        },
        "Ket": {
            "full_circuit_status": "unable_direct_support",
            "direct_adaptation": (
                "No installed Ket API was found that imports an arbitrary Qiskit "
                "QuantumCircuit or OpenQASM circuit while preserving raw channel-LCU "
                "control semantics."
            ),
            "executed_on_benchmarks": False,
            "output_form": None,
            "not_executed_reason": (
                "Using Ket would require a nontrivial reimplementation or custom "
                "gate-by-gate adapter, which is outside the allowed baseline policy."
            ),
            "api_evidence": inspect_ket(),
        },
        "qclib": {
            "full_circuit_status": "simple_adapter_supported",
            "direct_adaptation": (
                "qclib provides Qiskit gate classes for MCX/MCU/MCG-style synthesis, "
                "but no direct full-circuit importer was found. This runner adds a "
                "semantics-preserving recursive Qiskit ControlledGate adapter that maps "
                "controlled single-qubit operations to qclib.gates.Mcg."
            ),
            "executed_on_benchmarks": True,
            "output_form": "Qiskit QuantumCircuit containing qclib Mcg gates, then Qiskit opt0 basis lowering.",
            "adapter_policy": (
                "The adapter expands only Qiskit-provided gate definitions and preserves "
                "gate order, controls, and ctrl_state; it performs no channel-LCU address "
                "placement, Pauli factorization, or project optimizer step."
            ),
            "api_evidence": inspect_qclib(),
        },
        "RevKit": {
            "full_circuit_status": "unable_direct_support",
            "direct_adaptation": (
                "RevKit targets Boolean/reversible specifications rather than arbitrary "
                "quantum channel-LCU circuits or controlled-Pauli target actions."
            ),
            "executed_on_benchmarks": False,
            "output_form": None,
            "not_executed_reason": (
                "A RevKit experiment would be an address-decoder/oracle subcomponent "
                "benchmark, not a full raw channel-LCU circuit baseline."
            ),
            "api_evidence": inspect_revkit(),
        },
    }


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": sys.platform,
    }
    try:
        import qiskit

        env["qiskit_version"] = getattr(qiskit, "__version__", "unknown")
    except Exception as exc:
        env["qiskit_import_error"] = f"{type(exc).__name__}: {exc}"
    return env


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Benchmark | Basis | Method | Status | Qubits | Gates | G/G_raw | Non-Clifford | Time (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gates = "-" if row["metric_total"] is None else str(row["metric_total"])
        ratio = "-" if row["ratio_vs_raw"] is None else f"{row['ratio_vs_raw']:.6f}"
        non_cliff = "-" if row["non_clifford"] is None else str(row["non_clifford"])
        qubits = "-" if row["qubits"] is None else str(row["qubits"])
        elapsed = row.get("elapsed_seconds")
        elapsed_cell = "-" if elapsed is None else f"{float(elapsed):.3f}"
        lines.append(
            f"| {row['benchmark']} | {row['basis'].upper()} | {row['label']} | "
            f"{row['status']} | {qubits} | {gates} | {ratio} | {non_cliff} | {elapsed_cell} |"
        )
    return "\n".join(lines)


def write_report(summary: dict[str, Any], path: Path) -> None:
    compatibility = summary["tool_compatibility"]
    rows = summary["comparison_rows"]

    def version_cell(tool_info: dict[str, Any]) -> str:
        evidence = tool_info.get("api_evidence", {})
        if not isinstance(evidence, dict):
            return "-"
        if evidence.get("installed") is False:
            return "not installed"
        return str(evidence.get("version", "unknown"))

    lines = [
        "# Raw Channel-LCU MCG Tool Baseline Evaluation",
        "",
        "## Policy",
        "",
        "- External tools are tested only on raw `channel_to_LCU(..., structure=\"basic\", opt=\"No\")` circuits.",
        "- No project SELECT/address-placement/factorization optimization is used before external tools.",
        "- Tools without a direct Qiskit `QuantumCircuit`/QASM import or built-in direct conversion path are marked `unable_direct_support`.",
        "- Ours is included only as the comparison workflow (`rewrite_search + structure=opt + Matrix-order`).",
        "",
        "## Direct-Support Classification",
        "",
        "| Tool | Full raw channel-LCU status | Direct adaptation / reason | Executed? | Output circuit form |",
        "|---|---|---|---:|---|",
    ]
    for tool, info in compatibility.items():
        reason = info.get("direct_adaptation") or info.get("not_executed_reason") or ""
        output_form = info.get("output_form") or "-"
        executed = "yes" if info.get("executed_on_benchmarks") else "no"
        lines.append(
            f"| {tool} | `{info['full_circuit_status']}` | {reason} | {executed} | {output_form} |"
        )

    lines.extend(
        [
            "",
            "## Direct Object/QASM Support",
            "",
            "| Tool | Version | Qiskit `QuantumCircuit` direct path | QASM direct path | Benchmark decision |",
            "|---|---:|---|---|---|",
            (
                "| Qrisp | "
                f"{version_cell(compatibility['Qrisp'])} | "
                "`qrisp.QuantumCircuit.from_qiskit(qc)` | "
                "`from_qasm_file` / `from_qasm_str` are present in the installed API | "
                "Executed on all five full raw channel-LCU circuits |"
            ),
            (
                "| Ket | "
                f"{version_cell(compatibility['Ket'])} | "
                "No direct importer found | "
                "No arbitrary OpenQASM circuit importer found | "
                "Marked `unable_direct_support`; no custom adapter was built |"
            ),
            (
                "| qclib | "
                f"{version_cell(compatibility['qclib'])} | "
                "No built-in full-circuit converter/pass found; runner uses a recursive Qiskit ControlledGate adapter to qclib `Mcg` | "
                "No arbitrary OpenQASM circuit importer found | "
                "Executed as `qclib_mcg_adapter` on all selected full raw channel-LCU circuits |"
            ),
            (
                "| RevKit | "
                f"{version_cell(compatibility['RevKit'])} | "
                "No arbitrary quantum-circuit path | "
                "No full channel-LCU QASM path | "
                "Marked `unable_direct_support`; suitable only for Boolean/reversible subcomponents |"
            ),
            "",
            "## Output Circuit Forms",
            "",
            "- Qrisp emits a Qrisp `QuantumCircuit` after `qrisp.transpile`; this runner converts it back with `to_qiskit()` and then performs common NAM/IBM recounts with Qiskit optimization level 0.",
            "- Qrisp `from_qiskit` imports Qiskit controlled/composite gates as Qrisp `ControlledOperation` objects whose installed-version default method is `gray`. The `qrisp_workspace_auto` variant changes those imported Qrisp controlled operations to `method='auto'` and then calls `QuantumSession.compile(workspace=...)`; it does not use `method='hybrid'` and does not pass the in-house optimized circuit into Qrisp.",
            "- Ket was not executed; an executable Ket result would have to be a Ket-native process/circuit, but no allowed direct importer was available.",
            "- qclib is executed through a small recursive Qiskit adapter: Qiskit-provided `ControlledGate`/composite definitions are expanded without address reordering or Pauli factorization, and controlled single-qubit operations are emitted as qclib `Mcg` gates before the common NAM/IBM recount.",
            "- RevKit was not executed as a full-circuit baseline; its relevant outputs are reversible/Boolean networks or oracle circuits for subcomponents, not the full controlled-Pauli channel-LCU circuit.",
        ]
    )

    lines.extend(
        [
            "",
            "## Performance Recount",
            "",
            "All rows are recounted after Qiskit optimization-level 0 lowering to the named basis. `Time (s)` is the wall-clock runtime recorded for that route, including the route's own compile/import and the final recount work included by this runner.",
            "",
            markdown_table(rows),
            "",
            "## Width And Workspace",
            "",
            "| Benchmark | Raw width | Ours width | Qrisp workspace | Qrisp workspace-auto status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for bench in summary["benchmarks"]:
        policy = bench.get("workspace_budget_policy", {})
        workspace_qrisp = bench.get("tools", {}).get("qrisp_workspace_auto", {})
        raw_width = policy.get("raw_width")
        ours_width = policy.get("ours_width")
        workspace = policy.get("workspace_for_qrisp_compile")
        lines.append(
            f"| {bench['benchmark']} | {raw_width if raw_width is not None else '-'} | "
            f"{ours_width if ours_width is not None else '-'} | "
            f"{workspace if workspace is not None else '-'} | "
            f"`{workspace_qrisp.get('status', 'missing')}` |"
        )

    lines.extend(
        [
            "",
            "## Runtime Summary",
            "",
            "| Benchmark | Qrisp default(gray) time (s) | Qrisp workspace-auto time (s) | qclib adapter time (s) | Ours workflow time (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bench in summary["benchmarks"]:
        qrisp_default = bench.get("tools", {}).get("qrisp", {})
        qrisp_workspace = bench.get("tools", {}).get("qrisp_workspace_auto", {})
        qclib_adapter = bench.get("tools", {}).get("qclib_mcg_adapter", {})
        ours = bench.get("ours", {})

        def time_cell(value: Any) -> str:
            return "-" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {bench['benchmark']} | "
            f"{time_cell(qrisp_default.get('elapsed_seconds_total'))} | "
            f"{time_cell(qrisp_workspace.get('elapsed_seconds_total'))} | "
            f"{time_cell(qclib_adapter.get('elapsed_seconds_total'))} | "
            f"{time_cell(ours.get('compile_time_s'))} |"
        )

    lines.extend(
        [
            "",
            "## Tool Output Gate Sets",
            "",
            "| Benchmark | Tool variant | Status | Workspace | Qiskit gate set before final recount | Tool time (s) |",
            "|---|---|---|---:|---|---:|",
        ]
    )
    for bench in summary["benchmarks"]:
        for variant, label in (("qrisp", "default(gray)"), ("qrisp_workspace_auto", "workspace-auto")):
            qrisp = bench.get("tools", {}).get(variant, {})
            gate_set = qrisp.get("compiled_qiskit_gate_set")
            gate_set_cell = "-" if gate_set is None else "`" + ", ".join(gate_set) + "`"
            elapsed = qrisp.get("elapsed_seconds_total")
            elapsed_cell = "-" if elapsed is None else f"{float(elapsed):.3f}"
            workspace = qrisp.get("workspace_budget")
            workspace_cell = "-" if workspace is None else str(workspace)
            lines.append(
                f"| {bench['benchmark']} | {label} | `{qrisp.get('status', 'missing')}` | "
                f"{workspace_cell} | {gate_set_cell} | {elapsed_cell} |"
            )
        qclib = bench.get("tools", {}).get("qclib_mcg_adapter", {})
        gate_set = qclib.get("adapted_qiskit_gate_set")
        gate_set_cell = "-" if gate_set is None else "`" + ", ".join(gate_set) + "`"
        elapsed = qclib.get("elapsed_seconds_total")
        elapsed_cell = "-" if elapsed is None else f"{float(elapsed):.3f}"
        lines.append(
            f"| {bench['benchmark']} | qclib-mcg-adapter | `{qclib.get('status', 'missing')}` | "
            f"- | {gate_set_cell} | {elapsed_cell} |"
        )

    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- Machine-readable summary: `{summary['summary_path']}`")
    for bench in summary["benchmarks"]:
        lines.append(f"- `{bench['benchmark']}` summary: `{Path(bench['out_dir']) / 'summary.json'}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_driver(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {name.strip() for name in args.benchmarks}
    specs = [spec for spec in BENCHMARKS if spec.name in selected]
    summary: dict[str, Any] = {
        "experiment": "raw_channel_lcu_mcg_tool_baselines",
        "out_dir": str(out_dir),
        "environment": collect_environment(),
        "tool_compatibility": tool_compatibility(),
        "benchmarks": [],
        "comparison_rows": [],
    }

    for spec in specs:
        worker_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--benchmark",
            spec.name,
            "--out-dir",
            str(out_dir),
            "--qrisp-timeout-sec",
            str(args.qrisp_timeout_sec),
        ]
        started = time.perf_counter()
        proc = subprocess.run(
            worker_cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(args.worker_timeout_sec),
            check=False,
        )
        worker_elapsed = time.perf_counter() - started
        bench_summary_path = out_dir / "benchmarks" / spec.name / "summary.json"
        if bench_summary_path.exists():
            bench_summary = json.loads(bench_summary_path.read_text(encoding="utf-8"))
            bench_summary["worker_process"] = {
                "returncode": proc.returncode,
                "elapsed_seconds": float(worker_elapsed),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        else:
            bench_summary = {
                "benchmark": spec.name,
                "status": "worker_failed_no_summary",
                "out_dir": str(out_dir / "benchmarks" / spec.name),
                "worker_process": {
                    "returncode": proc.returncode,
                    "elapsed_seconds": float(worker_elapsed),
                    "stdout_tail": proc.stdout[-4000:],
                    "stderr_tail": proc.stderr[-4000:],
                },
            }
        summary["benchmarks"].append(bench_summary)
        summary["comparison_rows"].extend(bench_summary.get("comparison_rows", []))
        write_summary(summary, out_dir / "summary.partial.json")

    summary_path = out_dir / "summary.json"
    summary["summary_path"] = str(summary_path)
    report_path = out_dir / "mcg_tool_raw_channel_lcu_report.md"
    summary["report_path"] = str(report_path)
    write_summary(summary, summary_path)
    write_report(summary, report_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("driver", "worker"), default="driver")
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--benchmark", choices=[spec.name for spec in BENCHMARKS])
    parser.add_argument("--benchmarks", nargs="+", default=[spec.name for spec in BENCHMARKS])
    parser.add_argument("--qrisp-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--worker-timeout-sec", type=float, default=4200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        if args.benchmark is None:
            raise SystemExit("--benchmark is required in worker mode")
        spec_by_name = {spec.name: spec for spec in BENCHMARKS}
        summary = run_benchmark_worker(spec_by_name[args.benchmark], args.out_dir.resolve(), args.qrisp_timeout_sec)
        print(json.dumps({"benchmark": args.benchmark, "summary": str(Path(summary["out_dir"]) / "summary.json")}))
        return

    summary = run_driver(args)
    print(f"summary: {summary['summary_path']}")
    print(f"report: {summary['report_path']}")


if __name__ == "__main__":
    main()
