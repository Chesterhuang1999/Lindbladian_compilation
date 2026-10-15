#!/usr/bin/env python3
"""Probe direct Ket/Qrisp support for the raw HRW_8 channel-LCU circuit.

This script intentionally builds only the raw channel-LCU circuit:

    channel_to_LCU(..., structure="basic", opt="No")

It does not call Channel-IR rewrite search, structure-opt, Matrix-order, or any
other optimization from this repository. Tool-side compilation/transpilation is
recorded separately as the external tool action under test.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
RESULT_ROOT = ROOT / "Baseline_results" / "mcx_direct_support_hrw8"


def _json_default(obj: Any) -> str:
    return repr(obj)


def _now() -> float:
    return time.time()


def timed(label: str, fn: Callable[[], Any]) -> tuple[str, Any, float, str | None]:
    start = _now()
    try:
        value = fn()
        return "ok", value, _now() - start, None
    except Exception:
        return "error", None, _now() - start, traceback.format_exc()


def install_aer_stub() -> None:
    """Let HRW channel construction import without qiskit-aer.

    The HRW Pauli-sum construction path used here does not invoke Aer. The
    source module imports Aer at top level for separate simulation helpers, and
    the server qiskit-aer installation is incompatible with Qiskit 2.1.1. A
    tiny stub keeps this script from touching project source just to avoid that
    optional import.
    """

    if "qiskit_aer" in sys.modules:
        return

    aer = types.ModuleType("qiskit_aer")
    library = types.ModuleType("qiskit_aer.library")

    class AerSimulator:  # pragma: no cover - should never be used in this probe
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("AerSimulator is not used by this direct-support probe")

    def _unused(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise RuntimeError("qiskit_aer.library helpers are not used by this probe")

    aer.AerSimulator = AerSimulator
    aer.noise = types.SimpleNamespace()
    library.save_statevector = _unused
    library.set_statevector = _unused
    sys.modules["qiskit_aer"] = aer
    sys.modules["qiskit_aer.library"] = library


def build_hrw8_raw_channel_lcu() -> tuple[Any, Any, dict[str, Any]]:
    install_aer_stub()
    for path in (SRC_DIR, ROOT / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from channel_LCU import channel_to_LCU  # type: ignore
    from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore

    ensemble = build_hypercube_section52_channel(8)
    kraus_ops = ensemble.channels[0][1]
    qc, qubit_indexes = channel_to_LCU(ensemble, structure="basic", opt="No")
    meta = {
        "benchmark": "HRW_8",
        "source": "build_hypercube_section52_channel(8)",
        "channel_lcu_call": {
            "function": "channel_to_LCU",
            "structure": "basic",
            "opt": "No",
        },
        "repo_optimizations_used": [],
        "kraus_count": len(kraus_ops),
        "pauli_terms_total": sum(len(ms.instances) for ms in kraus_ops),
        "max_terms_per_kraus": max(len(ms.instances) for ms in kraus_ops),
        "qubit_indexes": qubit_indexes,
    }
    return qc, qubit_indexes, meta


def qiskit_metrics(qc: Any) -> dict[str, Any]:
    ops = qc.count_ops()
    return {
        "num_qubits": int(qc.num_qubits),
        "num_clbits": int(qc.num_clbits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "ops": {str(k): int(v) for k, v in ops.items()},
    }


def qrisp_metrics(qc: Any) -> dict[str, Any]:
    count_ops = getattr(qc, "count_ops", None)
    num_qubits = getattr(qc, "num_qubits")
    return {
        "num_qubits": int(num_qubits() if callable(num_qubits) else num_qubits),
        "depth": int(qc.depth()),
        "ops": {str(k): int(v) for k, v in count_ops().items()} if count_ops else None,
    }


def try_write_qpy(qc: Any, path: Path) -> dict[str, Any]:
    from qiskit import qpy

    with path.open("wb") as handle:
        qpy.dump(qc, handle)
    return {"path": str(path), "bytes": path.stat().st_size}


def try_write_qasm2(qc: Any, path: Path) -> dict[str, Any]:
    from qiskit import qasm2

    text = qasm2.dumps(qc)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size}


def try_write_qasm3(qc: Any, path: Path) -> dict[str, Any]:
    from qiskit import qasm3

    text = qasm3.dumps(qc)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size}


def collect_qrisp_api_evidence() -> dict[str, Any]:
    import qrisp
    from qrisp import QuantumCircuit as QrispQuantumCircuit

    evidence: dict[str, Any] = {
        "module_file": getattr(qrisp, "__file__", None),
        "version": getattr(qrisp, "__version__", "unknown"),
    }
    for name in ("from_qiskit", "from_qasm_file", "from_qasm_str", "to_qiskit", "to_qasm2"):
        obj = getattr(QrispQuantumCircuit, name, None)
        evidence[name] = {
            "exists": obj is not None,
            "signature": str(inspect.signature(obj)) if obj is not None else None,
        }
    evidence["transpile_signature"] = str(inspect.signature(qrisp.transpile))
    return evidence


def probe_qrisp(qc: Any, out_dir: Path) -> dict[str, Any]:
    import qrisp
    from qrisp import QuantumCircuit as QrispQuantumCircuit

    result: dict[str, Any] = {
        "tool": "Qrisp",
        "status": "running",
        "direct_support_rule": "tool provides QuantumCircuit.from_qiskit(qiskit_qc)",
        "api_evidence": collect_qrisp_api_evidence(),
    }

    status, qrisp_qc, elapsed, error = timed(
        "qrisp_from_qiskit", lambda: QrispQuantumCircuit.from_qiskit(qc)
    )
    result["from_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "failed"
        return result

    result["input_qrisp_metrics"] = qrisp_metrics(qrisp_qc)
    status, qasm_info, elapsed, error = timed(
        "qrisp_input_qasm2",
        lambda: _write_text_return(qrisp_qc.to_qasm2(), out_dir / "hrw8_qrisp_input.qasm"),
    )
    result["input_qasm2"] = {
        "status": status,
        "elapsed_seconds": elapsed,
        "result": qasm_info,
        "error": error,
    }

    status, compiled_qc, elapsed, error = timed(
        "qrisp_transpile", lambda: qrisp.transpile(qrisp_qc)
    )
    result["tool_transpile"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "failed"
        return result

    result["compiled_qrisp_metrics"] = qrisp_metrics(compiled_qc)
    status, qiskit_back, elapsed, error = timed("qrisp_to_qiskit", compiled_qc.to_qiskit)
    result["compiled_to_qiskit"] = {
        "status": status,
        "elapsed_seconds": elapsed,
        "error": error,
    }
    if status == "ok":
        result["compiled_qiskit_metrics"] = qiskit_metrics(qiskit_back)
        status2, qasm_info2, elapsed2, error2 = timed(
            "qrisp_compiled_qasm2",
            lambda: try_write_qasm2(qiskit_back, out_dir / "hrw8_qrisp_compiled_qiskit.qasm"),
        )
        result["compiled_qiskit_qasm2"] = {
            "status": status2,
            "elapsed_seconds": elapsed2,
            "result": qasm_info2,
            "error": error2,
        }

    result["status"] = "direct_supported"
    return result


def _write_text_return(text: str, path: Path) -> dict[str, Any]:
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size}


def collect_ket_api_evidence() -> dict[str, Any]:
    import ket

    package_path = Path(ket.__file__).resolve().parent
    candidate_names: list[str] = []
    for modinfo in pkgutil.walk_packages([str(package_path)], prefix="ket."):
        name = modinfo.name
        if name.count(".") <= 2:
            candidate_names.append(name)

    import_like: list[dict[str, str]] = []
    for module_name in ["ket", "ket.gates", "ket.operations", "ket.qulib", "ket.ibm"]:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            import_like.append(
                {"module": module_name, "import_error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for attr in dir(module):
            low = attr.lower()
            if any(tok in low for tok in ("qiskit", "qasm", "quantumcircuit", "import", "convert")):
                obj = getattr(module, attr, None)
                try:
                    sig = str(inspect.signature(obj)) if callable(obj) else None
                except Exception:
                    sig = None
                import_like.append(
                    {
                        "module": module_name,
                        "attribute": attr,
                        "repr": repr(obj),
                        "signature": sig,
                    }
                )

    return {
        "module_file": getattr(ket, "__file__", None),
        "version": getattr(ket, "__version__", "unknown"),
        "direct_import_api_candidates": import_like,
        "walked_modules_sample": candidate_names[:80],
    }


def probe_ket() -> dict[str, Any]:
    evidence = collect_ket_api_evidence()
    direct_candidates = [
        row
        for row in evidence["direct_import_api_candidates"]
        if row.get("attribute", "").lower()
        in {"from_qiskit", "from_qasm", "from_qasm_file", "from_qasm_str", "import_qasm"}
    ]
    return {
        "tool": "Ket",
        "status": "unable_direct_support",
        "reason": (
            "Installed ket-lang exposes a Python circuit-building API and backends, "
            "but no package API was found that directly imports a Qiskit QuantumCircuit "
            "or OpenQASM circuit. Building a nontrivial manual adapter is outside the "
            "allowed scope for this probe."
        ),
        "api_evidence": evidence,
        "matching_direct_import_candidates": direct_candidates,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    qiskit_raw = summary["qiskit_raw"]
    qrisp = summary["tools"]["qrisp"]
    ket = summary["tools"]["ket"]

    lines = [
        "# HRW_8 MCX Tool Direct-Support Probe",
        "",
        "## Input Circuit",
        "",
        "- Benchmark: HRW_8 hypercube random walk channel.",
        "- Construction: `channel_to_LCU(ensemble, structure=\"basic\", opt=\"No\")`.",
        "- Repository optimizations used: none.",
        f"- Qubits: {qiskit_raw['metrics']['num_qubits']}; depth: {qiskit_raw['metrics']['depth']}; size: {qiskit_raw['metrics']['size']}.",
        f"- Raw operation counts: `{qiskit_raw['metrics']['ops']}`.",
        "",
        "## Qrisp",
        "",
        f"- Status: `{qrisp['status']}`.",
        f"- Direct support evidence: `{qrisp['direct_support_rule']}`.",
    ]
    if qrisp.get("compiled_qiskit_metrics"):
        m = qrisp["compiled_qiskit_metrics"]
        lines.extend(
            [
                f"- Compiled-back Qiskit qubits/depth/size: {m['num_qubits']} / {m['depth']} / {m['size']}.",
                f"- Compiled-back operation counts: `{m['ops']}`.",
            ]
        )
    if qrisp.get("tool_transpile"):
        lines.append(
            f"- Qrisp transpile time: {qrisp['tool_transpile']['elapsed_seconds']:.6f}s."
        )

    lines.extend(
        [
            "",
            "## Ket",
            "",
            f"- Status: `{ket['status']}`.",
            f"- Reason: {ket['reason']}",
            "",
            "## Artifacts",
            "",
        ]
    )
    for item in summary.get("artifacts", []):
        lines.append(f"- `{item}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "root": str(ROOT),
        "out_dir": str(out_dir),
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
        },
        "policy": {
            "repo_optimizations_allowed": False,
            "repo_optimizations_used": [],
            "direct_support_rule": (
                "Only use Qiskit built-in lowering or a tool-provided direct "
                "QuantumCircuit/QASM import. No bespoke nontrivial adapter."
            ),
        },
        "artifacts": [],
    }

    status, build_value, elapsed, error = timed("build_hrw8_raw_channel_lcu", build_hrw8_raw_channel_lcu)
    summary["build"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1)

    qc, _qubit_indexes, circuit_meta = build_value
    summary["circuit_meta"] = circuit_meta
    summary["qiskit_raw"] = {"metrics": qiskit_metrics(qc)}

    for name, writer in [
        ("hrw8_raw_qiskit.qpy", lambda p: try_write_qpy(qc, p)),
        ("hrw8_raw_qiskit.qasm", lambda p: try_write_qasm2(qc, p)),
        ("hrw8_raw_qiskit.qasm3", lambda p: try_write_qasm3(qc, p)),
    ]:
        status, result, elapsed, error = timed(name, lambda name=name, writer=writer: writer(out_dir / name))
        summary.setdefault("qiskit_raw_exports", {})[name] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "result": result,
            "error": error,
        }
        if status == "ok":
            summary["artifacts"].append(str((out_dir / name).resolve()))

    tools: dict[str, Any] = {}
    status, result, elapsed, error = timed("probe_qrisp", lambda: probe_qrisp(qc, out_dir))
    tools["qrisp"] = result or {"tool": "Qrisp", "status": "failed", "error": error}
    tools["qrisp"]["elapsed_seconds_total"] = elapsed
    if error and "error" not in tools["qrisp"]:
        tools["qrisp"]["error"] = error

    status, result, elapsed, error = timed("probe_ket", probe_ket)
    tools["ket"] = result or {"tool": "Ket", "status": "failed", "error": error}
    tools["ket"]["elapsed_seconds_total"] = elapsed
    if error and "error" not in tools["ket"]:
        tools["ket"]["error"] = error

    summary["tools"] = tools

    for candidate in [
        out_dir / "hrw8_qrisp_input.qasm",
        out_dir / "hrw8_qrisp_compiled_qiskit.qasm",
    ]:
        if candidate.exists():
            summary["artifacts"].append(str(candidate.resolve()))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, out_dir / "summary.md")
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
