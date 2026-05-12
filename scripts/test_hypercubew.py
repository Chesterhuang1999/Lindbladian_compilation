from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean

_CACHE_DIR = Path(tempfile.gettempdir()) / "lindbladian_compilation_cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

from qiskit import QuantumCircuit, transpile

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from block_encoding import BlockEncoding
from hypercube_channel_pauli import build_hypercube_section52_channel


DEFAULT_DIMS = (8, 12, 20, 28)
DEFAULT_OPTS = ("No", "Ctrl-line", "Matrix-order")


def _count_metrics(
    qc: QuantumCircuit,
    basis_gates: tuple[str, ...],
    optimization_level: int,
) -> dict[str, int]:
    tqc = transpile(
        qc,
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
    )
    ops = tqc.count_ops()
    return {
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx": int(ops.get("cx", 0)),
        "u3": int(ops.get("u3", 0)),
        "num_qubits": int(tqc.num_qubits),
    }


def _format_values(values: list[int]) -> str:
    if not values:
        return "n/a"
    if min(values) == max(values):
        return str(values[0])
    return f"{min(values)}..{max(values)} (mean={mean(values):.2f})"


def _summarize(rows: list[dict[str, int]]) -> dict[str, str]:
    keys = ("num_qubits", "depth", "size", "cx", "u3", "tcount", "mccount", "cxcount")
    return {key: _format_values([int(row[key]) for row in rows]) for key in keys}


def _compile_kraus_operator(
    ms,
    opt: str,
    basis_gates: tuple[str, ...],
    optimization_level: int,
) -> dict[str, int]:
    be = BlockEncoding(ms)
    qc = be.circuit(opt=opt)
    metrics = _count_metrics(qc, basis_gates=basis_gates, optimization_level=optimization_level)
    metrics.update(
        {
            "tcount": int(getattr(be, "tcount", 0)),
            "mccount": int(getattr(be, "mccount", 0)),
            "cxcount": int(getattr(be, "cxcount", 0)),
        }
    )
    return metrics


def run_hypercube_case(
    n: int,
    opts: tuple[str, ...],
    basis_gates: tuple[str, ...],
    optimization_level: int,
    representative_only: bool,
) -> dict[str, dict[str, str]]:
    ensemble = build_hypercube_section52_channel(n)
    kraus_ops = ensemble.channels[0][1]
    selected_kraus = kraus_ops[:1] if representative_only else kraus_ops

    print(f"=== Hypercube random walk n = {n} ===")
    print(
        f"total_kraus={len(kraus_ops)}, compiled_kraus={len(selected_kraus)}, "
        f"terms_per_kraus={kraus_ops[0].length}"
    )

    results = {}
    for opt in opts:
        rows = [
            _compile_kraus_operator(
                ms,
                opt=opt,
                basis_gates=basis_gates,
                optimization_level=optimization_level,
            )
            for ms in selected_kraus
        ]
        summary = _summarize(rows)
        results[opt] = summary
        print(
            f"({opt}): qubits={summary['num_qubits']}, depth={summary['depth']}, "
            f"size={summary['size']}, cx={summary['cx']}, u3={summary['u3']}, "
            f"tcount={summary['tcount']}, mccount={summary['mccount']}, "
            f"cxcount={summary['cxcount']}"
        )

    print("-----------------------------")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the Pauli-sum Kraus block encodings for the Section 5.2 "
            "hypercube random-walk channel."
        )
    )
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--opts", nargs="+", default=list(DEFAULT_OPTS), choices=list(DEFAULT_OPTS))
    parser.add_argument("--basis-gates", nargs="+", default=["cx", "u3"])
    parser.add_argument("--optimization-level", type=int, default=3, choices=range(4))
    parser.add_argument(
        "--representative-only",
        action="store_true",
        help="Compile only K_{0,0}; useful because all Kraus operators have the same Pauli-term shape.",
    )
    args = parser.parse_args()

    for n in args.dims:
        run_hypercube_case(
            n=n,
            opts=tuple(args.opts),
            basis_gates=tuple(args.basis_gates),
            optimization_level=args.optimization_level,
            representative_only=bool(args.representative_only),
        )


if __name__ == "__main__":
    main()
