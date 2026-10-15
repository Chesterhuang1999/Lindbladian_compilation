#!/usr/bin/env python3
"""Print resolved evaluation settings and audit live block-encoding imports.

Read-only: does not run experiments or create reports. Historical copies,
source snapshots and exploratory workspaces are not formal rerun entrypoints.
"""

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from experiment_protocol import CONFIGS, RQ1_PIPELINES, settings, source_identity


def audit():
    imports = []
    syntax_errors = []
    for folder in ("scripts", "Baseline_scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if path.is_symlink():
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError as error:
                syntax_errors.append({"file": str(path.relative_to(ROOT)), "line": error.lineno,
                                      "error": error.msg})
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in (
                    "block_encoding", "block_encoding_new", "block_encoding_new_fixed"
                ) and any(item.name == "BlockEncoding" for item in node.names):
                    imports.append({"file": str(path.relative_to(ROOT)), "line": node.lineno,
                                    "module": node.module,
                                    "canonical_module": "block_encoding",
                                    "role": "diagnostic" if path.name.startswith(("test_", "probe_", "compare_", "experiment_", "diagnose_")) else "runner"})
    return {"block_imports": imports, "syntax_errors": syntax_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq", choices=("RQ1", "RQ2", "RQ3.1", "RQ3.2", "RQ4"), default="RQ1")
    parser.add_argument("--basis", choices=("nam", "ibm"), default="nam")
    args = parser.parse_args()
    result = {"settings": settings(args.rq, basis=args.basis), "rq1_pipelines": RQ1_PIPELINES,
              "ablation": [asdict(cfg) for cfg in CONFIGS], "sources": source_identity(), **audit()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["syntax_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
