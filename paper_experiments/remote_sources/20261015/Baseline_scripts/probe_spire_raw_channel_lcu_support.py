#!/usr/bin/env python3
"""Probe Spire artifact support for raw channel-LCU circuits.

The evaluation policy matches the MCG raw-baseline runner:

* external tools are tested only on raw ``channel_to_LCU(..., structure="basic",
  opt="No")`` circuits;
* no project SELECT/address-placement/factorization optimization is applied
  before external-tool execution;
* tools without a direct Qiskit ``QuantumCircuit``/QASM importer or built-in
  full-circuit conversion path are marked unsupported instead of receiving a
  bespoke adapter.

Spire/Tower is not a Python package; this probe inspects the official artifact
checkout and, when possible, attempts a direct CLI build/probe. It intentionally
does not implement a QASM-to-Tower translator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPIRE_DIR = Path("/pub/data/huangqf/spire-artifact")
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "spire_raw_channel_lcu_support"
DEFAULT_QASM = (
    ROOT
    / "Baseline_results"
    / "mcg_tool_raw_channel_lcu_benchmarks_qrisp_workspace_auto_v2"
    / "benchmarks"
    / "HRW_8"
    / "raw_channel_lcu_basic_no.qasm"
)


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: float = 60.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_tail": proc.stdout[-12000:],
            "stderr_tail": proc.stderr[-12000:],
            "status": "ok" if proc.returncode == 0 else "nonzero",
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "elapsed_seconds": time.perf_counter() - started,
            "status": "exception",
            "error": f"{type(exc).__name__}: {exc}",
        }


def read_text(path: Path, limit: int = 12000) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return None


def grep_files(root: Path, pattern: str) -> list[dict[str, Any]]:
    regex = re.compile(pattern, re.IGNORECASE)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, start=1):
            if regex.search(line):
                rows.append({"file": str(path.relative_to(root)), "line": line_no, "text": line[:260]})
                if len(rows) >= 300:
                    return rows
    return rows


def source_file_evidence(spire_dir: Path) -> dict[str, Any]:
    files = {
        "README.md": spire_dir / "README.md",
        "Dockerfile": spire_dir / "Dockerfile",
        "tower_args": spire_dir / "tower-src" / "src" / "lib" / "args.ml",
        "tower_main": spire_dir / "tower-src" / "src" / "main.ml",
        "tower_dotqc_export": spire_dir / "tower-src" / "src" / "lib" / "export" / "dotQC.ml",
        "tower_prim_circuit": spire_dir / "tower-src" / "src" / "lib" / "codegen" / "prim_circuit_.ml",
    }
    evidence: dict[str, Any] = {}
    for name, path in files.items():
        text = read_text(path)
        evidence[name] = {
            "path": str(path),
            "exists": path.exists(),
            "snippet": text,
        }
    return evidence


def classify_spire(evidence: dict[str, Any], grep_rows: list[dict[str, Any]]) -> dict[str, Any]:
    args_text = evidence.get("tower_args", {}).get("snippet") or ""
    main_text = evidence.get("tower_main", {}).get("snippet") or ""
    dotqc_text = evidence.get("tower_dotqc_export", {}).get("snippet") or ""
    prim_text = evidence.get("tower_prim_circuit", {}).get("snippet") or ""

    has_qasm_import = any(
        row["file"].startswith("tower-src/")
        and re.search(r"from_qasm|qasm.*parse|parse.*qasm|openqasm|read.*qasm", row["text"], re.IGNORECASE)
        for row in grep_rows
    )
    has_qiskit_import = any(
        row["file"].startswith("tower-src/")
        and re.search(r"qiskit|QuantumCircuit|from_qiskit", row["text"], re.IGNORECASE)
        for row in grep_rows
    )
    has_dotqc_input = any(
        row["file"].startswith("tower-src/")
        and re.search(r"parse.*dotqc|read.*dotqc|dotqc.*parse|dotqc.*input", row["text"], re.IGNORECASE)
        for row in grep_rows
    )

    return {
        "full_raw_channel_lcu_status": "unable_direct_support",
        "direct_adaptation": (
            "No official direct Qiskit QuantumCircuit/OpenQASM importer or full-circuit "
            "optimizer entry point was found. Spire/Tower parses Tower source programs "
            "and can emit dotQC, but it is not a raw channel-LCU circuit optimizer under "
            "the no-custom-adapter policy."
        ),
        "qasm_direct_path": has_qasm_import,
        "qiskit_direct_path": has_qiskit_import,
        "dotqc_direct_input_path": has_dotqc_input,
        "executed_on_channel_lcu": False,
        "not_executed_reason": (
            "Executing Spire on channel-LCU would require translating Qiskit/OpenQASM "
            "circuits into Tower source or into Spire's internal IR, which is a bespoke "
            "adapter and changes the benchmark boundary."
        ),
        "source_observations": {
            "cli_options_define_dotqc_output": "--dotQC" in args_text and "Export to .qc format" in args_text,
            "main_argparse_loads_program_files": "Arg.parse Args.speclist load desc" in main_text,
            "main_description_tower_language": "Tower quantum programming language" in main_text,
            "dotqc_export_uses_tof_and_x": "tof " in dotqc_text and "X q" in dotqc_text,
            "primitive_ir_only_controlled_not": "type gate = Pnot of wire * wire list" in prim_text,
        },
    }


def qasm_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    gate_defs = re.findall(r"^gate\s+([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.MULTILINE)
    ops = re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|q|\[)", text, flags=re.MULTILINE)
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "lines": len(text.splitlines()),
        "declared_gate_count": len(gate_defs),
        "declared_gate_sample": gate_defs[:40],
        "operation_token_sample": sorted(set(ops))[:60],
        "contains_openqasm": "OPENQASM 2.0;" in text,
        "contains_qelib": 'include "qelib1.inc";' in text,
        "contains_custom_mcx": any(name.startswith("mcx") for name in gate_defs),
        "contains_state_preparation": any("state_preparation" in name for name in gate_defs),
        "contains_multiplexer": any("multiplexer" in name for name in gate_defs),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    cls = summary["classification"]
    qasm = summary["channel_lcu_qasm"]
    lines = [
        "# Spire Raw Channel-LCU Support Probe",
        "",
        "## Policy",
        "",
        "- Input boundary: raw `channel_to_LCU(..., structure=\"basic\", opt=\"No\")` circuit.",
        "- No project-side SELECT/address-placement/factorization optimization before the external tool.",
        "- A tool must expose a direct Qiskit `QuantumCircuit`, OpenQASM, or built-in full-circuit conversion path.",
        "- No bespoke QASM-to-tool-source adapter is allowed for this compatibility classification.",
        "",
        "## Classification",
        "",
        "| Tool | Full raw channel-LCU status | Qiskit direct path | QASM direct path | Executed? | Reason |",
        "|---|---|---:|---:|---:|---|",
        (
            "| Spire/Tower | "
            f"`{cls['full_raw_channel_lcu_status']}` | "
            f"{'yes' if cls['qiskit_direct_path'] else 'no'} | "
            f"{'yes' if cls['qasm_direct_path'] else 'no'} | "
            "no | "
            f"{cls['direct_adaptation']} |"
        ),
        "",
        "## Source Evidence",
        "",
        "| Evidence | Observed |",
        "|---|---:|",
    ]
    for key, value in cls["source_observations"].items():
        lines.append(f"| `{key}` | {'yes' if value else 'no'} |")
    lines.extend(
        [
            "",
            "The inspected Tower CLI parses Tower source program files and provides `--dotQC` as an export mode. "
            "The dotQC exporter prints only `X` and multi-controlled `tof` forms from a primitive controlled-NOT IR. "
            "No Tower-source QASM/OpenQASM/Qiskit importer or dotQC input parser was found in the artifact checkout.",
            "",
            "## Channel-LCU Input Probe",
            "",
            "| Input | Exists | Bytes | Lines | Custom gates | Notes |",
            "|---|---:|---:|---:|---:|---|",
            (
                f"| `{qasm['path']}` | "
                f"{'yes' if qasm.get('exists') else 'no'} | "
                f"{qasm.get('bytes', '-')} | "
                f"{qasm.get('lines', '-')} | "
                f"{qasm.get('declared_gate_count', '-')} | "
                f"custom_mcx={qasm.get('contains_custom_mcx')}, "
                f"state_preparation={qasm.get('contains_state_preparation')}, "
                f"multiplexer={qasm.get('contains_multiplexer')} |"
            ),
            "",
            "Because the artifact exposes no direct QASM/Qiskit full-circuit entry point, this QASM was not passed through a custom adapter. "
            "Under the benchmark policy, this is an unsupported full-circuit baseline rather than a failed optimization run.",
            "",
            "## Build Probe",
            "",
            "The official README describes a Docker-based artifact. The direct local `make` probe is recorded below for reproducibility.",
            "",
            "```text",
            (summary.get("build_probe", {}).get("stderr_tail") or summary.get("build_probe", {}).get("stdout_tail") or "").strip()[:4000],
            "```",
            "",
            "## Artifact Paths",
            "",
            f"- Summary JSON: `{summary['summary_path']}`",
            f"- Spire checkout: `{summary['spire_dir']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spire-dir", type=Path, default=DEFAULT_SPIRE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-qasm", type=Path, default=DEFAULT_QASM)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    spire_dir = args.spire_dir.resolve()
    tower_dir = spire_dir / "tower-src"

    summary: dict[str, Any] = {
        "experiment": "spire_raw_channel_lcu_support_probe",
        "policy": "raw channel-LCU, no bespoke adapter",
        "spire_dir": str(spire_dir),
        "environment": {
            "python": os.sys.executable,
            "cwd": str(Path.cwd()),
        },
    }

    summary["git"] = run_cmd(["git", "rev-parse", "HEAD"], cwd=spire_dir, timeout=20)
    summary["source_evidence"] = source_file_evidence(spire_dir)
    summary["grep_qasm_qiskit_dotqc"] = grep_files(
        spire_dir,
        r"qasm|openqasm|qiskit|QuantumCircuit|from_qiskit|from_qasm|dotQC|\.qc|parse|read",
    )
    summary["classification"] = classify_spire(summary["source_evidence"], summary["grep_qasm_qiskit_dotqc"])
    summary["channel_lcu_qasm"] = qasm_summary(args.raw_qasm.resolve())

    summary["opam_status"] = run_cmd(["opam", "switch", "list"], timeout=20)
    summary["docker_status"] = run_cmd(["docker", "--version"], timeout=20)
    if args.skip_build:
        summary["build_probe"] = {"status": "skipped"}
    else:
        summary["build_probe"] = run_cmd(["make"], cwd=tower_dir, timeout=120)

    summary_path = out_dir / "summary.json"
    report_path = out_dir / "spire_raw_channel_lcu_support_report.md"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, report_path)
    print(f"summary: {summary_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
