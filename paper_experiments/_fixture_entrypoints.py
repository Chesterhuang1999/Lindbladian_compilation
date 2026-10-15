"""Keep retired exploratory CLIs from dispatching the maintained paper fixtures."""
from pathlib import Path
import argparse
import sys

RETIRED_CASES = frozenset({
    "Collective4", "Local_QSW3", "Petz_TFIM2", "TFIM3_HO", "Thermal2_HO",
})
ENTRYPOINT = Path(__file__).resolve().parent / "experiments_new/fixture_benchmarks/run.py"


def check_legacy_arguments(arguments, default=None):
    if any(arg in ("--help", "-h", "--self-test") for arg in arguments):
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--case", default=default)
    selected, _ = parser.parse_known_args(arguments)
    case = selected.case
    if case in RETIRED_CASES:
        raise SystemExit(f"Legacy entrypoint retired for {case}; use {ENTRYPOINT}")


if __name__ == "__main__":
    check_legacy_arguments(sys.argv[1:], globals().get("default_case"))
