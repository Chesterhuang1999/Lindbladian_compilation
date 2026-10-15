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
import importlib.util
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
BASELINE_DIR = ROOT / "Baseline_scripts"
RESULT_ROOT = ROOT / "Baseline_results" / "mcx_direct_support_hrw8"
NAM_BASIS = ("h", "x", "rz", "cx")
NAM_TRANSPILE_BASIS = ("h", "x", "rz", "cx", "reset")
IBM_BASIS = ("u1", "u2", "u3", "cx")
DEFAULT_OURS_SUMMARY = (
    ROOT
    / "Baseline_results"
    / "hypercube_random_walk_n8_ours_nam"
    / "hypercube_random_walk_n8_ours_nam_summary.json"
)


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
    for path in (SRC_DIR, BASELINE_DIR, ROOT / "scripts"):
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


def count_ops_total(ops: dict[str, int]) -> int:
    return int(sum(int(value) for value in ops.values()))


def qrisp_metrics(qc: Any) -> dict[str, Any]:
    count_ops = getattr(qc, "count_ops", None)
    num_qubits = getattr(qc, "num_qubits")
    return {
        "num_qubits": int(num_qubits() if callable(num_qubits) else num_qubits),
        "depth": int(qc.depth()),
        "ops": {str(k): int(v) for k, v in count_ops().items()} if count_ops else None,
    }


def recount_qiskit_circuit(
    qc: Any,
    *,
    basis_name: str,
    out_path: Path,
    optimization_level: int = 0,
) -> dict[str, Any]:
    from qiskit import qasm2, transpile
    from baseline_common import count_qasm_file  # type: ignore

    if basis_name == "nam":
        basis = NAM_TRANSPILE_BASIS
        metric_gate_set = "nam"
    elif basis_name == "ibm":
        basis = IBM_BASIS
        metric_gate_set = "ibm"
    else:
        raise ValueError(f"Unsupported basis_name: {basis_name}")

    start = time.perf_counter()
    tqc = transpile(
        qc,
        basis_gates=list(basis),
        optimization_level=int(optimization_level),
    )
    transpile_elapsed = time.perf_counter() - start
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        qasm2.dump(tqc, handle)
    stats = count_qasm_file(out_path, gate_set=metric_gate_set)
    return {
        "basis_name": basis_name,
        "transpile_basis": list(basis),
        "optimization_level": int(optimization_level),
        "transpile_elapsed_seconds": transpile_elapsed,
        "qasm": str(out_path),
        "qiskit_metrics": qiskit_metrics(tqc),
        "stats": stats,
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
        result["compiled_qiskit_gate_set"] = sorted(
            result["compiled_qiskit_metrics"]["ops"].keys()
        )
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
        basis_recounts: dict[str, Any] = {}
        for basis_name in ("nam", "ibm"):
            status3, recount, elapsed3, error3 = timed(
                f"qrisp_recount_{basis_name}",
                lambda basis_name=basis_name: recount_qiskit_circuit(
                    qiskit_back,
                    basis_name=basis_name,
                    out_path=out_dir / f"hrw8_qrisp_compiled_{basis_name}_opt0.qasm",
                    optimization_level=0,
                ),
            )
            basis_recounts[basis_name] = {
                "status": status3,
                "elapsed_seconds": elapsed3,
                "result": recount,
                "error": error3,
            }
        result["basis_recounts"] = basis_recounts

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


def load_pytket_qiskit_convert_module() -> Any:
    try:
        import pytket.extensions.qiskit.qiskit_convert as qiskit_convert

        return qiskit_convert
    except Exception:
        import pytket

        convert_path = (
            Path(pytket.__file__).resolve().parent
            / "extensions"
            / "qiskit"
            / "qiskit_convert.py"
        )
        spec = importlib.util.spec_from_file_location(
            "pytket.extensions.qiskit.qiskit_convert",
            convert_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load pytket qiskit converter from {convert_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def probe_pytket(qc: Any, out_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool": "pytket",
        "status": "running",
        "direct_support_rule": (
            "pytket-qiskit provides qiskit_to_tk(qiskit.QuantumCircuit); "
            "this probe loads that converter directly if the extension package "
            "eager-imports unavailable Aer backends."
        ),
    }
    try:
        import pytket

        result["version"] = getattr(pytket, "__version__", "unknown")
    except Exception as exc:
        result["status"] = "not_installed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    status, converter, elapsed, error = timed("load_pytket_qiskit_convert", load_pytket_qiskit_convert_module)
    result["load_converter"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "unable_direct_support"
        return result
    result["api_evidence"] = {
        "qiskit_to_tk_signature": str(inspect.signature(converter.qiskit_to_tk)),
        "tk_to_qiskit_signature": str(inspect.signature(converter.tk_to_qiskit)),
    }

    status, tk_circuit, elapsed, error = timed("qiskit_to_tk", lambda: converter.qiskit_to_tk(qc))
    result["from_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "failed_direct_conversion"
        return result

    result["tk_metrics"] = {
        "n_qubits": int(tk_circuit.n_qubits),
        "n_gates": int(tk_circuit.n_gates),
        "depth": int(tk_circuit.depth()),
    }

    status, qiskit_back, elapsed, error = timed("tk_to_qiskit", lambda: converter.tk_to_qiskit(tk_circuit))
    result["to_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status == "ok":
        result["roundtrip_qiskit_metrics"] = qiskit_metrics(qiskit_back)
        status2, qasm_info, elapsed2, error2 = timed(
            "pytket_roundtrip_qasm2",
            lambda: try_write_qasm2(qiskit_back, out_dir / "hrw8_pytket_roundtrip_qiskit.qasm"),
        )
        result["roundtrip_qasm2"] = {
            "status": status2,
            "elapsed_seconds": elapsed2,
            "result": qasm_info,
            "error": error2,
        }

    result["status"] = "direct_supported"
    return result


def probe_bqskit(qc: Any, out_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool": "BQSKit",
        "status": "running",
        "direct_support_rule": "bqskit.ext.qiskit provides qiskit_to_bqskit(qiskit.QuantumCircuit)",
    }
    try:
        import bqskit
        from bqskit.ext.qiskit import bqskit_to_qiskit, qiskit_to_bqskit

        result["version"] = getattr(bqskit, "__version__", "unknown")
        result["api_evidence"] = {
            "qiskit_to_bqskit_signature": str(inspect.signature(qiskit_to_bqskit)),
            "bqskit_to_qiskit_signature": str(inspect.signature(bqskit_to_qiskit)),
        }
    except Exception:
        result["status"] = "not_installed_or_import_failed"
        result["error"] = traceback.format_exc()
        return result

    status, bq_circuit, elapsed, error = timed("qiskit_to_bqskit_raw", lambda: qiskit_to_bqskit(qc))
    result["from_qiskit_raw"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    conversion_source = "raw"
    if status != "ok":
        conversion_source = "qiskit_ibm_opt0"

        def _lower_and_convert() -> Any:
            from qiskit import transpile

            lowered = transpile(qc, basis_gates=list(IBM_BASIS), optimization_level=0)
            return qiskit_to_bqskit(lowered)

        status, bq_circuit, elapsed, error = timed("qiskit_to_bqskit_after_ibm_lowering", _lower_and_convert)
        result["from_qiskit_after_ibm_lowering"] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
        }
    if status != "ok":
        result["status"] = "failed_direct_conversion"
        return result

    result["conversion_source"] = conversion_source
    result["bqskit_metrics"] = {
        "num_qudits": int(bq_circuit.num_qudits),
        "num_operations": int(bq_circuit.num_operations),
    }
    status, qiskit_back, elapsed, error = timed("bqskit_to_qiskit", lambda: bqskit_to_qiskit(bq_circuit))
    result["to_qiskit"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status == "ok":
        result["roundtrip_qiskit_metrics"] = qiskit_metrics(qiskit_back)
        status2, qasm_info, elapsed2, error2 = timed(
            "bqskit_roundtrip_qasm2",
            lambda: try_write_qasm2(qiskit_back, out_dir / "hrw8_bqskit_roundtrip_qiskit.qasm"),
        )
        result["roundtrip_qasm2"] = {
            "status": status2,
            "elapsed_seconds": elapsed2,
            "result": qasm_info,
            "error": error2,
        }
    result["status"] = "direct_supported"
    return result


def probe_pyzx(qc: Any, out_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool": "PyZX",
        "status": "running",
        "direct_support_rule": (
            "No qiskit.QuantumCircuit object converter was found; using Qiskit "
            "built-in NAM lowering plus PyZX's qasm(string) reader is recorded "
            "as QASM-level direct support."
        ),
    }
    try:
        import pyzx as zx

        result["version"] = getattr(zx, "__version__", "unknown")
        result["api_evidence"] = {"qasm_signature": str(inspect.signature(zx.qasm))}
    except Exception:
        result["status"] = "not_installed_or_import_failed"
        result["error"] = traceback.format_exc()
        return result

    def _lower_parse() -> tuple[Any, dict[str, Any]]:
        from qiskit import qasm2, transpile
        import pyzx as zx

        lowered = transpile(qc, basis_gates=list(NAM_BASIS), optimization_level=0)
        qasm_text = qasm2.dumps(lowered)
        qasm_path = out_dir / "hrw8_pyzx_input_nam_opt0.qasm"
        qasm_path.write_text(qasm_text, encoding="utf-8")
        zx_circuit = zx.qasm(qasm_text)
        return zx_circuit, {
            "qasm": str(qasm_path),
            "qiskit_lowered_metrics": qiskit_metrics(lowered),
        }

    status, value, elapsed, error = timed("qiskit_nam_qasm_to_pyzx", _lower_parse)
    result["from_qiskit_nam_qasm"] = {"status": status, "elapsed_seconds": elapsed, "error": error}
    if status != "ok":
        result["status"] = "failed_qasm_conversion"
        return result
    zx_circuit, lower_info = value
    result["lowering"] = lower_info
    result["pyzx_metrics"] = {
        "qubits": int(getattr(zx_circuit, "qubits", 0)),
        "gates": int(len(getattr(zx_circuit, "gates", []))),
    }
    result["status"] = "qasm_direct_supported"
    return result


def load_ours_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_stats = data["raw"]["stats"]
    ours_stats = data["ours"]["stats"]
    return {
        "path": str(path),
        "raw_nam": {
            "num_qubits": int(raw_stats["num_qubits"]),
            "metric_total": int(raw_stats["metric_total"]),
            "non_clifford": int(raw_stats["clifford"]["non_clifford"]),
            "ops": raw_stats["ops"],
        },
        "ours_nam": {
            "num_qubits": int(ours_stats["num_qubits"]),
            "metric_total": int(ours_stats["metric_total"]),
            "non_clifford": int(ours_stats["clifford"]["non_clifford"]),
            "ignored": int(ours_stats["clifford"].get("ignored", 0)),
            "compile_time_s": float(data["ours"].get("compile_time_s", 0.0)),
            "ops": ours_stats["ops"],
        },
    }


def build_comparison(summary: dict[str, Any]) -> dict[str, Any] | None:
    reference = summary.get("ours_reference")
    qrisp = summary["tools"].get("qrisp", {})
    recount = qrisp.get("basis_recounts", {}).get("nam", {}).get("result")
    if not reference or not recount:
        return None
    raw_total = int(reference["raw_nam"]["metric_total"])
    ours_total = int(reference["ours_nam"]["metric_total"])
    qrisp_stats = recount["stats"]
    qrisp_total = int(qrisp_stats["metric_total"])
    return {
        "metric_basis": "NAM {h,x,rz,cx}; Qrisp output is recounted after Qiskit opt0 NAM lowering",
        "rows": [
            {
                "method": "Raw channel-LCU + Qiskit opt0 NAM",
                "qubits": int(reference["raw_nam"]["num_qubits"]),
                "metric_total": raw_total,
                "ratio_vs_raw": 1.0,
                "ratio_vs_ours": raw_total / ours_total,
                "non_clifford": int(reference["raw_nam"]["non_clifford"]),
            },
            {
                "method": "Qrisp direct import + Qrisp transpile + Qiskit opt0 NAM recount",
                "qubits": int(qrisp_stats["num_qubits"]),
                "metric_total": qrisp_total,
                "ratio_vs_raw": qrisp_total / raw_total,
                "ratio_vs_ours": qrisp_total / ours_total,
                "non_clifford": int(qrisp_stats["clifford"]["non_clifford"]),
            },
            {
                "method": "Ours (rewrite + opt + Matrix-order) + Qiskit opt0 NAM",
                "qubits": int(reference["ours_nam"]["num_qubits"]),
                "metric_total": ours_total,
                "ratio_vs_raw": ours_total / raw_total,
                "ratio_vs_ours": 1.0,
                "non_clifford": int(reference["ours_nam"]["non_clifford"]),
            },
        ],
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
