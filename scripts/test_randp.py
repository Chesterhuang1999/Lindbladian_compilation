from __future__ import annotations

import argparse
import os
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np

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
from channel_IR import Matrixsum, PauliAtom


DEFAULT_QUBITS = (4, 8, 10)
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


def _ratio(curr: int, base: int) -> float:
    if base == 0:
        return float("nan")
    return 1.0 - (curr / base)


def _label_from_product_index(index: int, num_qubits: int) -> str:
    alphabet = "IXYZ"
    chars = ["I"] * num_qubits
    for pos in range(num_qubits - 1, -1, -1):
        index, digit = divmod(index, len(alphabet))
        chars[pos] = alphabet[digit]
    return "".join(chars)


def build_random_pauli_matrixsum(num_qubits: int, seed: int = 20260314) -> Matrixsum:
    """Build the random Pauli Matrixsum from evaluate.ipynb cell f2deabc9."""
    num_terms = max(num_qubits * num_qubits - num_qubits, 2 ** (num_qubits - 1) - num_qubits)
    total_labels = 4**num_qubits

    if num_terms > total_labels:
        raise ValueError(
            f"Requested num_terms={num_terms} exceeds total Pauli strings={total_labels} "
            f"for N={num_qubits}."
        )

    rng = np.random.default_rng(seed + num_qubits)
    picked_indices = rng.choice(total_labels, size=num_terms, replace=False)
    picked_labels = [_label_from_product_index(int(i), num_qubits) for i in picked_indices]
    instances = [(PauliAtom(label, phase=1.0), 1.0) for label in picked_labels]
    return Matrixsum(instances)


def build_random_pauli_matrixsum_reference(num_qubits: int, seed: int = 20260314) -> Matrixsum:
    """Literal notebook implementation. Kept for small-N reproducibility checks."""
    num_terms = max(num_qubits * num_qubits - num_qubits, 2 ** (num_qubits - 1) - num_qubits)
    all_labels = ["".join(p) for p in product("IXYZ", repeat=num_qubits)]

    if num_terms > len(all_labels):
        raise ValueError(
            f"Requested num_terms={num_terms} exceeds total Pauli strings={len(all_labels)} "
            f"for N={num_qubits}."
        )

    rng = np.random.default_rng(seed + num_qubits)
    idx = rng.choice(len(all_labels), size=num_terms, replace=False)
    picked_labels = [all_labels[int(i)] for i in idx]
    instances = [(PauliAtom(label, phase=1.0), 1.0) for label in picked_labels]
    return Matrixsum(instances)


def _compile_matrixsum(
    ms: Matrixsum,
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


def run_random_pauli_case(
    num_qubits: int,
    seed: int,
    opts: tuple[str, ...],
    basis_gates: tuple[str, ...],
    optimization_level: int,
) -> dict[str, dict[str, int]]:
    ms = build_random_pauli_matrixsum(num_qubits=num_qubits, seed=seed)

    print(f"=== Random-Pauli Matrixsum metrics for N = {num_qubits} ===")
    print(
        f"num_terms={ms.length}, seed={seed}, "
        f"basis_gates={list(basis_gates)}, opt_level={optimization_level}"
    )

    all_metrics = {
        opt: _compile_matrixsum(
            ms,
            opt=opt,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )
        for opt in opts
    }

    base = all_metrics.get("No")
    for opt, values in all_metrics.items():
        if base is None:
            ratio_text = ""
        else:
            ratio_text = (
                f", depth_ratio={_ratio(values['depth'], base['depth']):.4f}, "
                f"size_ratio={_ratio(values['size'], base['size']):.4f}, "
                f"cx_ratio={_ratio(values['cx'], base['cx']):.4f}, "
                f"u3_ratio={_ratio(values['u3'], base['u3']):.4f}"
            )

        print(
            f"({opt}): qubits={values['num_qubits']}, depth={values['depth']}, "
            f"size={values['size']}, cx={values['cx']}, u3={values['u3']}, "
            f"tcount={values['tcount']}, mccount={values['mccount']}, "
            f"cxcount={values['cxcount']}{ratio_text}"
        )

    print("-----------------------------")
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the random Pauli Matrixsum benchmark from evaluate.ipynb cell f2deabc9."
    )
    parser.add_argument("--qubits", type=int, nargs="+", default=list(DEFAULT_QUBITS))
    parser.add_argument("--seed", type=int, default=20260314)
    parser.add_argument("--opts", nargs="+", default=list(DEFAULT_OPTS), choices=list(DEFAULT_OPTS))
    parser.add_argument("--basis-gates", nargs="+", default=["cx", "u3"])
    parser.add_argument("--optimization-level", type=int, default=3, choices=range(4))
    args = parser.parse_args()

    for num_qubits in args.qubits:
        run_random_pauli_case(
            num_qubits=num_qubits,
            seed=args.seed,
            opts=tuple(args.opts),
            basis_gates=tuple(args.basis_gates),
            optimization_level=args.optimization_level,
        )


if __name__ == "__main__":
    main()
