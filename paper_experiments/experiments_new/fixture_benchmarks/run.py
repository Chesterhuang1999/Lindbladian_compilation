#!/usr/bin/env python3
"""Check or explicitly prepare/launch the five fixture-channel experiments."""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PAPER_EXPERIMENTS = HERE.parents[1]
ROOT = PAPER_EXPERIMENTS.parent
CONFIG = HERE.relative_to(ROOT) / "config.json"
DRIVER = Path("scripts/eval_redesign/run_rq2_hybrid_ablation.py")
CASES = ("Collective4", "Local_QSW3", "Petz_TFIM2", "TFIM3_HO", "Thermal2_HO")


def read(path):
    return json.loads(Path(path).read_text())


def check_runtime():
    """Use only standard-library checks; do not import the compiler or JIT."""
    manifest = read(HERE / "runtime_manifest.json")
    if manifest["case_ids"] != list(CASES):
        raise ValueError("The runtime must select exactly the five fixture channels")
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing runtime dependency: {relative}")
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Runtime dependency hash changed: {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    config = read(ROOT / CONFIG)
    if tuple(case["id"] for case in config["case_groups"]) != CASES:
        raise ValueError("The configuration must select exactly the five fixture channels")
    plan = []
    for case in config["case_groups"]:
        if case["family"] != "new_benchmark_fixture":
            raise ValueError(f"Unexpected input family for {case['id']}")
        fixture = (ROOT / case["fixture"]).resolve()
        if not fixture.is_relative_to(HERE / "inputs"):
            raise ValueError("Fixture input must be inside this experiment's inputs directory")
        payload = read(fixture)
        width = case["system_qubits"]
        if payload["n"] != width or payload["name"] != case["id"]:
            raise ValueError(f"Fixture identity mismatch for {case['id']}")
        rows = payload["stages"]["raw"]["kraus"]
        if not rows or set(payload["stages"]) != {"raw"}:
            raise ValueError("The runtime must contain nonempty raw input only")
        for row in rows:
            for label, pair in row.items():
                if len(label) != width or set(label) - set("IXYZ"):
                    raise ValueError(f"Invalid Pauli label in {case['id']}")
                if len(pair) != 2 or not all(math.isfinite(value) for value in pair):
                    raise ValueError(f"Invalid coefficient in {case['id']}")
        plan.append((case["id"], width, len(rows), sum(map(len, rows))))
    missing = []
    packages = {}
    for package in manifest["required_packages"]:
        if util.find_spec(package) is None:
            missing.append(package)
            packages[package] = "missing"
            continue
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "available (version metadata unavailable)"
    return manifest, config, plan, packages, missing


def validate_output(path, *, prepared):
    out = path.expanduser().resolve()
    if not out.is_relative_to(PAPER_EXPERIMENTS):
        raise ValueError("Run outputs must be inside paper_experiments")
    protected = (HERE / "inputs", ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts")
    if (out == HERE or HERE.is_relative_to(out)
            or any(out.is_relative_to(source) or source.is_relative_to(out) for source in protected)):
        raise ValueError("Choose a separate run output directory")
    if not prepared:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Preparation requires a new or empty output directory")
        return out
    run_manifest = read(out / "manifest.json")
    if tuple(case["id"] for case in run_manifest["cases"]) != CASES:
        raise ValueError("The prepared run must select exactly the five fixture channels")
    if run_manifest["config"] != read(ROOT / CONFIG):
        raise ValueError("The prepared run configuration differs from this entry point")
    snapshot = out / "source_snapshot"
    for relative, expected in run_manifest["snapshot_sha256"].items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot) or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared snapshot hash changed: {relative}")
    if (out / "launch.json").exists():
        raise ValueError("This run was already launched; choose a new run for retries")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Check dependencies and print the plan (default)")
    action.add_argument("--prepare", action="store_true", help="Materialize fresh canonical inputs and freeze a new run; no rewrite or compilation")
    action.add_argument("--launch", action="store_true", help="Explicitly launch an already prepared run")
    parser.add_argument("--out-dir", type=Path, help="Separate run directory inside paper_experiments")
    args = parser.parse_args()
    if (args.prepare or args.launch) and args.out_dir is None:
        parser.error("--prepare and --launch require --out-dir")
    if not (args.prepare or args.launch) and args.out_dir is not None:
        parser.error("--out-dir is used only with --prepare or --launch")
    _, config, plan, packages, missing = check_runtime()
    print("Five fixture channels: raw input -> fresh canonicalization -> fresh hybrid rewrite -> NAM opt0 compilation")
    print("Case         Qubits  Raw Kraus  Raw Pauli terms")
    for name, width, rows, terms in plan:
        print(f"{name:<12} {width:>6} {rows:>10} {terms:>16}")
    print("Configurations:", ", ".join(config["configuration_order"]))
    print("Rewrite:", config["rewrite"]["method"])
    print("Packages:", "; ".join(f"{name}={value}" for name, value in packages.items()))
    if missing:
        raise ValueError("Missing packages in this Python environment: " + ", ".join(missing))
    if not (args.prepare or args.launch):
        print("Dependency checks passed. No inputs were prepared and no experiments were launched.")
        return 0
    out = validate_output(args.out_dir, prepared=args.launch)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Resolve shared sources explicitly and do not inherit unrelated modules.
    environment.pop("PYTHONPATH", None)
    command = [sys.executable, "-B", str(ROOT / DRIVER), "--out-dir", str(out)]
    if args.prepare:
        command += ["--prepare-only", "--config", str(ROOT / CONFIG),
                    "--source-manifest", str(HERE / "runtime_manifest.json")]
    else:
        command += ["--launch"]
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError) as error:
        print(f"Fixture experiment entry: {error}", file=sys.stderr)
        raise SystemExit(2)
