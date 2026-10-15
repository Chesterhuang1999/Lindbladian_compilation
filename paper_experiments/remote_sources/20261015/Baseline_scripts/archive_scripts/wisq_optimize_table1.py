from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "circuits" / "select_only" / "test_table1_m_no_n4_ibm_qasm2.qasm"
DEFAULT_OUTPUT = ROOT / "Baseline_results" / "wisq" / "ibm" / "test_table1_m_no_n4_ibm_wisq_optimized.qasm"


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"WISQ input QASM not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wisq_input_") as tmp_dir:
        tmp_input = Path(tmp_dir) / input_path.name
        shutil.copy2(input_path, tmp_input)
        command = [
            "wisq",
            str(tmp_input),
            "--mode",
            "opt",
            "--target_gateset",
            args.target_gateset,
            "--optimization_objective",
            args.optimization_objective,
            "--opt_timeout",
            str(int(args.timeout)),
            "--output_path",
            str(output_path),
        ]
        if args.approx_epsilon is not None:
            command.extend(["--approx_epsilon", str(args.approx_epsilon)])
        if args.verbose:
            command.append("--verbose")

        completed = subprocess.run(command, check=False)
        return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize the table1 SELECT-only QASM2 input with WISQ/GUOQ."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-gateset", default="IBMO", choices=["NAM", "CLIFFORDT", "IBMO", "IBMN", "ION"])
    parser.add_argument(
        "--optimization-objective",
        default="TOTAL",
        choices=["TWO_Q", "FIDELITY", "FT", "TOTAL", "T"],
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--approx-epsilon", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
