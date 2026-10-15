from __future__ import annotations

import csv
import json
import os
import platform
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import Kraus, Stinespring


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore  # noqa: E402
from scripts.generate_table1_channel_lcu_qasm import build_periodic_tfim_example  # type: ignore  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
LOG_DIR = OUT_DIR / "logs"
RAW_JSONL = OUT_DIR / "raw_runs.jsonl"
SUMMARY_CSV = OUT_DIR / "summary.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"
MANIFEST = OUT_DIR / "manifest.json"
NOTES = OUT_DIR / "notes.md"

IBM_BASIS = ["u1", "u2", "u3", "cx"]
SUMMARY_COLUMNS = [
    "benchmark_family",
    "benchmark_name",
    "n_or_size",
    "method",
    "construction_time_sec",
    "transpile_time_sec",
    "total_time_sec",
    "gates",
    "depth",
    "width",
    "memory_peak_mb",
    "timeout",
    "status",
    "error_msg",
    "source_path",
    "source_type",
    "comment",
]


class TimeoutExpired(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    def handler(_signum: int, _frame: object) -> None:
        raise TimeoutExpired(f"timeout after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def memory_peak_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for mod_name in ("qiskit", "numpy"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            versions[mod_name] = f"unavailable:{type(exc).__name__}"
    return versions


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def matrixsum_kraus_to_mats(kraus_ops: list[Any]) -> list[np.ndarray]:
    return [np.asarray(op.eff_op().to_matrix(), dtype=complex) for op in kraus_ops]


def single_qubit_thermal_relaxation(gamma: float = 0.1) -> list[np.ndarray]:
    # Amplitude damping is the smallest Lindblad-like relaxation sanity case.
    return [
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex),
        np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
    ]


def build_case(case: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    start = time.perf_counter()
    if case["kind"] == "hypercube":
        ensemble = build_hypercube_section52_channel(case["n"])
        kraus_ops = ensemble.channels[0][1]
        kraus_mats = matrixsum_kraus_to_mats(kraus_ops)
        source_path = "src/hypercube_channel_pauli.py:build_hypercube_section52_channel"
        source_type = "existing_repo_constructor"
        channel_meta = {
            "kraus_count": len(kraus_mats),
            "system_qubits": case["n"],
            "pauli_terms_total": sum(len(op.instances) for op in kraus_ops),
        }
    elif case["kind"] == "tfim":
        example = build_periodic_tfim_example(case["n"], case["delta_t"])
        kraus_ops = example.kraus_ops
        kraus_mats = matrixsum_kraus_to_mats(kraus_ops)
        source_path = "scripts/generate_table1_channel_lcu_qasm.py:build_periodic_tfim_example"
        source_type = "existing_repo_constructor"
        channel_meta = {
            "kraus_count": len(kraus_mats),
            "system_qubits": case["n"],
            "pauli_terms_total": sum(len(op.instances) for op in kraus_ops),
        }
    elif case["kind"] == "thermal_relaxation":
        kraus_mats = single_qubit_thermal_relaxation(case["gamma"])
        source_path = "Baseline_results/eval_redesign/cp1_stinespring_sanity/run_cp1c_stinespring_sanity.py:single_qubit_thermal_relaxation"
        source_type = "tiny_sanity_wrapper"
        channel_meta = {
            "kraus_count": len(kraus_mats),
            "system_qubits": 1,
            "gamma": case["gamma"],
        }
    else:
        raise ValueError(f"unknown case kind: {case['kind']}")

    channel_meta["target_channel_construction_time_sec"] = time.perf_counter() - start
    channel_meta["source_path"] = source_path
    channel_meta["source_type"] = source_type
    return kraus_mats, channel_meta


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    total_start = time.perf_counter()
    record: dict[str, Any] = {
        "benchmark_family": case["family"],
        "benchmark_name": case["name"],
        "n_or_size": case["n_or_size"],
        "method": "qiskit_stinespring_isometry_ibm_opt0",
        "start_time": now_iso(),
        "timeout_seconds": case["timeout"],
        "status": "running",
        "timeout": False,
        "error_msg": "",
        "source_path": "",
        "source_type": "",
        "comment": "",
    }

    try:
        with time_limit(case["timeout"]):
            kraus_mats, channel_meta = build_case(case)
            record.update(channel_meta)
            construct_start = time.perf_counter()
            stinespring = Stinespring(Kraus(kraus_mats))
            isometry = Isometry(stinespring.data, 0, 0)
            qc = QuantumCircuit(isometry.num_qubits, name=f"{case['name']}_stinespring")
            qc.append(isometry, range(isometry.num_qubits))
            record["construction_time_sec"] = time.perf_counter() - construct_start
            record["stinespring_matrix_shape"] = list(stinespring.data.shape)
            record["pre_transpile_width"] = int(qc.num_qubits)
            record["pre_transpile_ops"] = {str(k): int(v) for k, v in qc.count_ops().items()}

            transpile_start = time.perf_counter()
            tqc = transpile(qc, basis_gates=IBM_BASIS, optimization_level=0)
            record["transpile_time_sec"] = time.perf_counter() - transpile_start
            record["gates"] = int(tqc.size())
            record["depth"] = int(tqc.depth())
            record["width"] = int(tqc.num_qubits)
            record["basis_gates"] = IBM_BASIS
            record["transpile_optimization_level"] = 0
            record["ops"] = {str(k): int(v) for k, v in tqc.count_ops().items()}
            record["status"] = "ok"
            record["comment"] = "Completed Qiskit Stinespring-to-Isometry synthesis and IBM opt0 transpilation."
    except TimeoutExpired as exc:
        record["status"] = "timeout"
        record["timeout"] = True
        record["error_msg"] = str(exc)
        record["comment"] = "Stopped by strict CP1C timeout; failure row intentionally preserved."
    except Exception as exc:
        record["status"] = "failed"
        record["error_msg"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["comment"] = "Qiskit Stinespring/Isometry construction or transpilation failed; failure row intentionally preserved."
    finally:
        record["total_time_sec"] = time.perf_counter() - total_start
        record["memory_peak_mb"] = memory_peak_mb()
        record["end_time"] = now_iso()

    return record


def row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        column: record.get(column, "")
        for column in SUMMARY_COLUMNS
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_notes(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    timeout_rows = [row for row in rows if row["timeout"]]
    failed_rows = [row for row in rows if row["status"] == "failed"]

    largest = max(ok_rows, key=lambda row: int(row.get("gates") or 0), default=None)
    lines = [
        "# CP1C Stinespring Sanity Notes",
        "",
        "This checkpoint is appendix-only sanity evidence. It is not a main RQ1 baseline and should not be placed in the main RQ1 comparison table against channel-first or generic post-hoc optimizers.",
        "",
        "## Method",
        "",
        "Each row builds a target channel/superoperator from existing repository constructors where possible, forms Qiskit's Stinespring representation, maps the Stinespring isometry through `qiskit.circuit.library.Isometry`, and transpiles to the IBM circuit-first basis `{u1,u2,u3,cx}` with Qiskit optimization level 0. No external optimizer receives high-level QASM.",
        "",
        "## Results",
        "",
        "| Benchmark | status | construction s | transpile s | total s | gates | depth | width | timeout | comment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["benchmark_name"],
                    row["status"],
                    fmt(row.get("construction_time_sec", "")),
                    fmt(row.get("transpile_time_sec", "")),
                    fmt(row.get("total_time_sec", "")),
                    fmt(row.get("gates", "")),
                    fmt(row.get("depth", "")),
                    fmt(row.get("width", "")),
                    str(row.get("timeout", "")),
                    str(row.get("comment", "")).replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if largest:
        lines.append(
            f"The largest completed row is `{largest['benchmark_name']}` with {largest['gates']} IBM-basis gates, depth {largest['depth']}, width {largest['width']}, and total time {largest['total_time_sec']:.3f}s."
        )
    if timeout_rows:
        names = ", ".join(row["benchmark_name"] for row in timeout_rows)
        lines.append(
            f"Strict timeout rows were preserved for: {names}. These rows support a scaling-boundary statement for this benchmark setting."
        )
    if failed_rows:
        names = ", ".join(row["benchmark_name"] for row in failed_rows)
        lines.append(
            f"Failure rows were preserved for: {names}. Inspect `raw_runs.jsonl` and `logs/run.log` before using them in an appendix table."
        )
    if ok_rows and (timeout_rows or largest and int(largest.get("gates") or 0) > 100000):
        lines.append(
            "The data support the careful conclusion that circuit-first Stinespring synthesis is not suitable as a main baseline in this benchmark setting: it either produces very large small-instance circuits or reaches the strict scaling timeout."
        )
    else:
        lines.append(
            "The completed rows are appendix sanity evidence only. They should be used cautiously and not overgeneralized beyond these small benchmark instances."
        )

    lines.extend(
        [
            "",
            "This does not claim Stinespring synthesis is universally unusable. It only documents why this circuit-first dilation path is not used as the main RQ1 baseline for these channel-LCU/Lindbladian benchmarks.",
            "",
            "## CP5 Blockers",
            "",
            "- CP5 can aggregate `summary.csv` directly with the required schema.",
            "- Any appendix text should keep the appendix-only label and avoid mixing these rows into the main RQ1 ratio table.",
            f"- Manifest success flag: `{manifest['success']}`; known failures: {json.dumps(manifest['known_failures'])}.",
            "",
        ]
    )
    NOTES.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start_time = now_iso()
    RAW_JSONL.write_text("", encoding="utf-8")

    cases = [
        {
            "family": "Lindblad",
            "name": "single_qubit_thermal_relaxation_gamma0p1",
            "n_or_size": "1q",
            "kind": "thermal_relaxation",
            "gamma": 0.1,
            "timeout": 60,
        },
        {
            "family": "Algorithm",
            "name": "hypercube_random_walk_n4",
            "n_or_size": "n=4",
            "kind": "hypercube",
            "n": 4,
            "timeout": 300,
        },
        {
            "family": "Lindblad",
            "name": "periodic_tfim_n3_dt0p1",
            "n_or_size": "n=3,dt=0.1",
            "kind": "tfim",
            "n": 3,
            "delta_t": 0.1,
            "timeout": 300,
        },
        {
            "family": "Algorithm",
            "name": "hypercube_random_walk_n8",
            "n_or_size": "n=8",
            "kind": "hypercube",
            "n": 8,
            "timeout": 300,
        },
    ]

    rows: list[dict[str, Any]] = []
    with (LOG_DIR / "run.log").open("w", encoding="utf-8") as log:
        log.write(f"CP1C Stinespring sanity start: {start_time}\n")
        for index, case in enumerate(cases, start=1):
            log.write(f"[{now_iso()}] START {index}/{len(cases)} {case['name']} timeout={case['timeout']}s\n")
            log.flush()
            record = run_case(case)
            append_jsonl(RAW_JSONL, record)
            rows.append(record)
            log.write(
                f"[{now_iso()}] END {case['name']} status={record['status']} "
                f"total={record.get('total_time_sec'):.6g}s gates={record.get('gates', '')} "
                f"error={record.get('error_msg', '')}\n"
            )
            if record.get("traceback"):
                log.write(record["traceback"] + "\n")
            log.flush()

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(row_from_record(row) for row in rows)

    SUMMARY_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    known_failures = [
        {
            "benchmark_name": row["benchmark_name"],
            "status": row["status"],
            "timeout": row["timeout"],
            "error_msg": row["error_msg"],
        }
        for row in rows
        if row["status"] != "ok"
    ]
    manifest = {
        "checkpoint": "cp1_stinespring_sanity",
        "agent": "Agent RQ1-Stinespring",
        "repo_commit": repo_commit(),
        "start_time": start_time,
        "end_time": now_iso(),
        "machine": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "package_versions": package_versions(),
        "tools_available": {
            "qiskit_stinespring": hasattr(Stinespring, "to_instruction"),
            "qiskit_isometry": True,
        },
        "timeouts": {case["name"]: case["timeout"] for case in cases},
        "success": len(known_failures) == 0 or any(row["status"] == "ok" for row in rows),
        "known_failures": known_failures,
        "output_dir": str(OUT_DIR),
        "basis": IBM_BASIS,
        "summary_rows": len(rows),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_notes(rows, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
