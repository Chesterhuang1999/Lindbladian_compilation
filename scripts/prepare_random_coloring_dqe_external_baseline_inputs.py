#!/usr/bin/env python3
"""Prepare basis-level RandomColoring and repetition-DQE QASM inputs."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SRC_DIR, BASELINE_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from build_repetition_code_dqe_channel import (  # type: ignore  # noqa: E402
    DqeSpec,
    build_repetition_dqe_channel,
    channel_metrics as dqe_channel_metrics,
    pauli_dict_to_matrixsum,
)
from run_random_coloring_channel_lcu_ours_nam import (  # type: ignore  # noqa: E402
    CaseSpec as RandomColoringSpec,
    build_random_coloring_channel,
    build_channel_lcu_circuit,
    channel_metrics,
    dump_untranspiled_qasm,
    export_basis_count_qasm,
    graph_edges,
    neighbor_sets,
    random_coloring_model_info,
)


DEFAULT_OUT_ROOT = ROOT / "circuits" / "Table1"
DEFAULT_SUMMARY_ROOT = ROOT / "Baseline_results" / "eval_redesign" / "prepared_external_inputs"


class StageTimeout(RuntimeError):
    pass


def write_progress(target_dir: Path, case: str, stage: str, extra: dict[str, Any] | None = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "case": case,
        "stage": stage,
        "time_s": time.time(),
    }
    if extra:
        payload.update(extra)
    (target_dir / f"{case}_raw_prepare_progress.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def dump_qpy(qc: Any, path: Path) -> None:
    from qiskit import qpy

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out_file:
        qpy.dump(qc, out_file)


def _read_worker_message(parent_conn: Any, proc: mp.Process, timeout_s: float, stage: str) -> Any:
    deadline = time.monotonic() + timeout_s
    while True:
        if parent_conn.poll(0.05):
            message = parent_conn.recv()
            proc.join(5)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
            return message
        if not proc.is_alive():
            proc.join()
            if parent_conn.poll():
                return parent_conn.recv()
            raise RuntimeError(f"{stage} subprocess exited without returning a result.")
        if time.monotonic() >= deadline:
            proc.terminate()
            proc.join(5)
            raise StageTimeout(f"{stage} exceeded timeout {timeout_s}s.")


def _export_basis_worker(conn: Any, qc: Any, path: Path, gate_set: str, optimization_level: int) -> None:
    try:
        stats, elapsed = export_basis_count_qasm(
            qc,
            path,
            gate_set=gate_set,
            optimization_level=optimization_level,
        )
        conn.send(("ok", stats, elapsed))
    except Exception as exc:
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


def export_basis_with_timeout(
    qc: Any,
    path: Path,
    *,
    gate_set: str,
    optimization_level: int,
    timeout_s: float,
) -> tuple[dict[str, Any], float]:
    if timeout_s <= 0:
        return export_basis_count_qasm(
            qc,
            path,
            gate_set=gate_set,
            optimization_level=optimization_level,
        )

    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(
        target=_export_basis_worker,
        args=(child_conn, qc, path, gate_set, optimization_level),
    )
    proc.start()
    child_conn.close()
    message = _read_worker_message(parent_conn, proc, timeout_s, f"{gate_set} export")
    parent_conn.close()
    if message[0] == "ok":
        return message[1], message[2]
    raise RuntimeError(f"{gate_set} export failed: {message[1]}")


def _raw_prepare_worker(
    conn: Any,
    kraus_ops: list[Any],
    case: str,
    target_dir: Path,
    metadata: dict[str, Any],
    channel_stats: dict[str, Any],
    export_timeout_s: float,
    dump_raw_qasm_file: bool,
    dump_raw_qpy_file: bool,
    raw_build_only: bool,
) -> None:
    try:
        write_progress(target_dir, case, "building_raw_channel_lcu")
        qc, qubit_indexes, synthesis_time_s = build_channel_lcu_circuit(
            kraus_ops,
            structure="basic",
            opt="No",
        )
        write_progress(
            target_dir,
            case,
            "raw_channel_lcu_built",
            {
                "synthesis_time_s": synthesis_time_s,
                "num_qubits": int(qc.num_qubits),
                "num_instructions": int(len(qc.data)),
            },
        )
        if raw_build_only:
            qpy_path = target_dir / f"{case}_basic_raw.qpy"
            dump_qpy(qc, qpy_path)
            result = {
                "case": case,
                "status": "raw_built",
                "metadata": metadata,
                "channel_metrics": channel_stats,
                "raw_qasm": None,
                "raw_qasm_dumped": False,
                "raw_qpy": str(qpy_path),
                "raw_qpy_dumped": True,
                "basis_outputs": {},
                "synthesis_time_s": synthesis_time_s,
                "circuit_qubits": int(qc.num_qubits),
                "circuit_instructions": int(len(qc.data)),
                "qubit_indexes": qubit_indexes,
                "note": "--raw-build-only requested; basis QASM export was skipped.",
            }
            conn.send(("ok", result))
            return
        result = prepare_basis_qasm(
            case=case,
            qc=qc,
            target_dir=target_dir,
            synthesis_time_s=synthesis_time_s,
            export_timeout_s=0,
            dump_raw_qasm_file=dump_raw_qasm_file,
            dump_raw_qpy_file=dump_raw_qpy_file,
            metadata=metadata,
            channel_stats=channel_stats,
        )
        result["qubit_indexes"] = qubit_indexes
        conn.send(("ok", result))
    except Exception as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def prepare_raw_basis_qasm_with_timeout(
    *,
    case: str,
    kraus_ops: list[Any],
    target_dir: Path,
    metadata: dict[str, Any],
    channel_stats: dict[str, Any],
    synthesis_timeout_s: float,
    export_timeout_s: float,
    dump_raw_qasm_file: bool,
    dump_raw_qpy_file: bool,
    raw_build_only: bool,
) -> dict[str, Any]:
    if synthesis_timeout_s <= 0:
        qc, qubit_indexes, synthesis_time_s = build_channel_lcu_circuit(
            kraus_ops,
            structure="basic",
            opt="No",
        )
        result = prepare_basis_qasm(
            case=case,
            qc=qc,
            target_dir=target_dir,
            synthesis_time_s=synthesis_time_s,
            export_timeout_s=export_timeout_s,
            dump_raw_qasm_file=dump_raw_qasm_file,
            dump_raw_qpy_file=dump_raw_qpy_file,
            metadata=metadata,
            channel_stats=channel_stats,
        )
        result["qubit_indexes"] = qubit_indexes
        return result

    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(
        target=_raw_prepare_worker,
        args=(
            child_conn,
            kraus_ops,
            case,
            target_dir,
            metadata,
            channel_stats,
            export_timeout_s,
            dump_raw_qasm_file,
            dump_raw_qpy_file,
            raw_build_only,
        ),
    )
    proc.start()
    child_conn.close()
    message = _read_worker_message(parent_conn, proc, synthesis_timeout_s, f"{case} raw basis preparation")
    parent_conn.close()
    if message[0] == "ok":
        return message[1]
    raise RuntimeError(f"{case} raw basis preparation failed: {message[1]}")


def write_case_summary(out_dir: Path, summary: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary['case']}_input_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def prepare_basis_qasm(
    *,
    case: str,
    qc: Any,
    target_dir: Path,
    synthesis_time_s: float,
    export_timeout_s: float,
    dump_raw_qasm_file: bool,
    dump_raw_qpy_file: bool,
    metadata: dict[str, Any],
    channel_stats: dict[str, Any],
) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_qasm = target_dir / f"{case}_basic_raw.qasm"
    raw_qpy = target_dir / f"{case}_basic_raw.qpy"
    nam_qasm = target_dir / f"{case}_basic_raw_nam_opt0.qasm"
    ibm_qasm = target_dir / f"{case}_basic_raw_ibm_opt0.qasm"
    if dump_raw_qasm_file:
        write_progress(target_dir, case, "dumping_high_level_raw_qasm")
        dump_untranspiled_qasm(qc, raw_qasm)
    if dump_raw_qpy_file:
        write_progress(target_dir, case, "dumping_raw_qpy")
        dump_qpy(qc, raw_qpy)

    basis_outputs: dict[str, Any] = {}
    for gate_set, path in (("nam", nam_qasm), ("ibm", ibm_qasm)):
        write_progress(target_dir, case, f"exporting_{gate_set}_basis_qasm")
        stats, elapsed = export_basis_with_timeout(
            qc,
            path,
            gate_set=gate_set,
            optimization_level=0,
            timeout_s=export_timeout_s,
        )
        basis_outputs[gate_set] = {
            "qasm": str(path),
            "stats": stats,
            "export_time_s": elapsed,
        }

    return {
        "case": case,
        "status": "ok",
        "metadata": metadata,
        "channel_metrics": channel_stats,
        "raw_qasm": str(raw_qasm) if dump_raw_qasm_file else None,
        "raw_qasm_dumped": bool(dump_raw_qasm_file),
        "raw_qpy": str(raw_qpy) if dump_raw_qpy_file else None,
        "raw_qpy_dumped": bool(dump_raw_qpy_file),
        "basis_outputs": basis_outputs,
        "synthesis_time_s": synthesis_time_s,
    }


def random_coloring_metadata(spec: RandomColoringSpec) -> dict[str, Any]:
    edges = graph_edges(spec.graph, spec.n)
    neighbors = neighbor_sets(spec.n, edges)
    max_degree = max((len(nbrs) for nbrs in neighbors), default=0)
    metadata = random_coloring_model_info(spec, edges, max_degree)
    metadata["kraus_color_mixing"] = "walsh_hadamard_by_vertex"
    return metadata


def prepare_random_coloring_case(args: argparse.Namespace, spec: RandomColoringSpec) -> dict[str, Any]:
    case = spec.name
    target_dir = Path(args.out_root) / case
    summary_dir = Path(args.summary_root) / case
    started = time.perf_counter()
    metadata = random_coloring_metadata(spec)
    try:
        kraus_ops, metadata = build_random_coloring_channel(
            spec.n,
            spec.graph,
            q=spec.q,
            tol=args.tol,
            max_total_pauli_terms=args.max_total_pauli_terms,
            max_terms_per_kraus=args.max_terms_per_kraus,
            mix_colors=True,
        )
        stats = channel_metrics(kraus_ops)
        result = prepare_raw_basis_qasm_with_timeout(
            case=case,
            target_dir=target_dir,
            kraus_ops=kraus_ops,
            synthesis_timeout_s=args.synthesis_timeout_sec,
            export_timeout_s=args.export_timeout_sec,
            dump_raw_qasm_file=not args.skip_high_level_raw_qasm,
            dump_raw_qpy_file=args.dump_raw_qpy,
            raw_build_only=args.raw_build_only,
            metadata=metadata,
            channel_stats=stats,
        )
    except Exception as exc:
        result = {
            "case": case,
            "status": "failed",
            "error_msg": f"{type(exc).__name__}: {exc}",
            "metadata": metadata,
            "target_dir": str(target_dir),
            "total_time_s": float(time.perf_counter() - started),
        }
    result["summary_path"] = str(write_case_summary(summary_dir, result))
    return result


def prepare_dqe_case(args: argparse.Namespace, spec: DqeSpec) -> dict[str, Any]:
    case = spec.name
    target_dir = Path(args.out_root) / case
    summary_dir = Path(args.summary_root) / case
    started = time.perf_counter()
    channel, metadata = build_repetition_dqe_channel(spec)
    stats = dqe_channel_metrics(channel)
    try:
        kraus_ops = [pauli_dict_to_matrixsum(copy.deepcopy(kraus)) for kraus in channel]
        result = prepare_raw_basis_qasm_with_timeout(
            case=case,
            target_dir=target_dir,
            kraus_ops=kraus_ops,
            synthesis_timeout_s=args.synthesis_timeout_sec,
            export_timeout_s=args.export_timeout_sec,
            dump_raw_qasm_file=not args.skip_high_level_raw_qasm,
            dump_raw_qpy_file=args.dump_raw_qpy,
            raw_build_only=args.raw_build_only,
            metadata=metadata,
            channel_stats=stats,
        )
    except Exception as exc:
        result = {
            "case": case,
            "status": "failed",
            "error_msg": f"{type(exc).__name__}: {exc}",
            "metadata": metadata,
            "channel_metrics": stats,
            "target_dir": str(target_dir),
            "total_time_s": float(time.perf_counter() - started),
        }
    result["summary_path"] = str(write_case_summary(summary_dir, result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=("random_coloring", "dqe"), default=["random_coloring", "dqe"])
    parser.add_argument("--random-n", nargs="+", type=int, default=[4])
    parser.add_argument("--random-graphs", nargs="+", choices=("cycle", "complete"), default=["cycle", "complete"])
    parser.add_argument("--dqe-d", nargs="+", type=int, default=[5])
    parser.add_argument("--dqe-epsilon", type=float, default=0.25)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max-total-pauli-terms", type=int, default=250_000)
    parser.add_argument("--max-terms-per-kraus", type=int, default=50_000)
    parser.add_argument("--synthesis-timeout-sec", type=float, default=300.0)
    parser.add_argument("--export-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--skip-high-level-raw-qasm",
        action="store_true",
        help="Skip dumping the composite raw channel-LCU QASM and export only basis-level NAM/IBM QASM.",
    )
    parser.add_argument(
        "--dump-raw-qpy",
        action="store_true",
        help="Write the raw QuantumCircuit as QPY after construction.",
    )
    parser.add_argument(
        "--raw-build-only",
        action="store_true",
        help="Build and optionally QPY-dump the raw circuit, but skip basis QASM export.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    if "random_coloring" in args.benchmarks:
        for n in args.random_n:
            for graph in args.random_graphs:
                spec = RandomColoringSpec(n=n, q=n, graph=graph)
                print(f"[prepare-random-coloring] {spec.name}", flush=True)
                result = prepare_random_coloring_case(args, spec)
                results.append(result)
                print(f"[prepare-random-coloring] {spec.name}: {result['status']}", flush=True)
                if result["status"] != "ok":
                    print(result.get("error_msg", ""), flush=True)

    if "dqe" in args.benchmarks:
        for d in args.dqe_d:
            spec = DqeSpec(d=d, epsilon=args.dqe_epsilon)
            print(f"[prepare-dqe] {spec.name}", flush=True)
            result = prepare_dqe_case(args, spec)
            results.append(result)
            print(f"[prepare-dqe] {spec.name}: {result['status']}", flush=True)
            if result["status"] != "ok":
                print(result.get("error_msg", ""), flush=True)

    args.summary_root.mkdir(parents=True, exist_ok=True)
    index_path = args.summary_root / "prepared_random_coloring_dqe_inputs_index.json"
    index_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
