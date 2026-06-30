from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import transpile


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from channel_IR import Lindbladian, channel as ChannelIR, channel_ensemble  # type: ignore  # noqa: E402
from channel_LCU import Lindblad_to_channel, channel_to_LCU  # type: ignore  # noqa: E402
from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore  # noqa: E402
from qasm_export import export_openqasm2_quartz_input  # type: ignore  # noqa: E402


DEFAULT_OUT_ROOT = REPO_ROOT / "circuits" / "Table1"
IBM_BASIS = ("u1", "u2", "u3", "cx")
NAM_BASIS = ("h", "x", "rz", "cx")


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    kraus_ops: list


def _float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _channel_metrics(kraus_ops: list) -> dict[str, int]:
    nonzero_ops = [op for op in kraus_ops if len(op.instances) > 0]
    return {
        "kraus_count": len(nonzero_ops),
        "pauli_terms_total": sum(len(op.instances) for op in nonzero_ops),
        "max_terms_per_kraus": max((len(op.instances) for op in nonzero_ops), default=0),
        "system_qubits": max((op.size for op in nonzero_ops), default=0),
    }


def _print_metric_delta(example_name: str, raw: dict[str, int], rewritten: dict[str, int]) -> None:
    kraus_delta = raw["kraus_count"] - rewritten["kraus_count"]
    pauli_delta = raw["pauli_terms_total"] - rewritten["pauli_terms_total"]
    kraus_status = "improved" if kraus_delta > 0 else "unchanged" if kraus_delta == 0 else "worse"
    pauli_status = "improved" if pauli_delta > 0 else "unchanged" if pauli_delta == 0 else "worse"

    print(f"[{example_name}] raw metrics: {raw}")
    print(f"[{example_name}] rewrite metrics: {rewritten}")
    print(
        f"[{example_name}] kraus_count {kraus_status}: "
        f"{raw['kraus_count']} -> {rewritten['kraus_count']} (delta={kraus_delta})"
    )
    print(
        f"[{example_name}] pauli_terms_total {pauli_status}: "
        f"{raw['pauli_terms_total']} -> {rewritten['pauli_terms_total']} "
        f"(delta={pauli_delta})"
    )


def build_hypercube_random_walk_example(n: int) -> ExampleSpec:
    ensemble = build_hypercube_section52_channel(n)
    return ExampleSpec(
        name=f"hypercube_random_walk_n{n}",
        kraus_ops=ensemble.channels[0][1],
    )


def build_periodic_tfim_example(num_qubits: int, delta_t: float) -> ExampleSpec:
    h_terms = []
    l_terms = []
    gamma = np.sqrt(0.1) / 2

    for i in range(num_qubits):
        z_ind = [i, (i + 1) % num_qubits]
        z_str = "".join("Z" if j in z_ind else "I" for j in range(num_qubits))
        x_str = "".join("X" if j == i else "I" for j in range(num_qubits))
        y_str = "".join("Y" if j == i else "I" for j in range(num_qubits))
        h_terms.append((z_str, -1))
        h_terms.append((x_str, -1))
        l_terms.append([(x_str, gamma), (y_str, -1j * gamma)])

    tfim_lind = Lindbladian(h_terms, l_terms)
    channel_lind, _, _ = Lindblad_to_channel(tfim_lind, delta_t)
    return ExampleSpec(
        name=f"periodic_tfim_n{num_qubits}_dt{_float_tag(delta_t)}",
        kraus_ops=channel_lind.channels[0][1],
    )


def rewrite_kraus_ops(
    kraus_ops: list,
    *,
    strategy: str,
    beam_width: int,
    max_steps: int,
    tol: float,
    verbose: bool,
) -> tuple[list, dict]:
    rewrite_channel = ChannelIR(copy.deepcopy(kraus_ops))
    rewrite_channel.zero_elim()
    result = rewrite_channel.rewrite_search(
        strategy=strategy,
        beam_width=beam_width,
        max_steps=max_steps,
        tol=tol,
        verbose=verbose,
    )

    rewritten_channel = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten_channel.apply_rewrite_result(result, tol=tol)
    rewritten_channel.zero_elim()
    return rewritten_channel.kraus_ops, result


def build_basic_channel_lcu_circuit(kraus_ops: list):
    ensemble = channel_ensemble([copy.deepcopy(kraus_ops)])
    qc, _ = channel_to_LCU(ensemble, structure="basic", opt="No")
    return qc


def export_basic_basis_qasm2(
    kraus_ops: list,
    out_prefix: Path,
) -> dict[str, object]:
    qc = build_basic_channel_lcu_circuit(kraus_ops)
    nam_qc = transpile(qc, basis_gates=list(NAM_BASIS), optimization_level=0)
    ibm_qc = transpile(qc, basis_gates=list(IBM_BASIS), optimization_level=0)

    nam_path = out_prefix.with_name(f"{out_prefix.name}_nam_opt0.qasm")
    ibm_path = out_prefix.with_name(f"{out_prefix.name}_ibm_opt0.qasm")
    nam_export = export_openqasm2_quartz_input(nam_qc, nam_path, basis_gates=NAM_BASIS)
    ibm_export = export_openqasm2_quartz_input(ibm_qc, ibm_path, basis_gates=IBM_BASIS)

    ops = {str(gate): int(count) for gate, count in qc.count_ops().items()}
    return {
        "nam_path": str(nam_path),
        "ibm_path": str(ibm_path),
        "format": "openqasm2",
        "structure": "basic",
        "opt": "No",
        "transpiled_to_basis": True,
        "transpile_optimization_level": 0,
        "num_qubits": int(qc.num_qubits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "ops": ops,
        "nam_export": nam_export,
        "ibm_export": ibm_export,
    }


def process_example(
    example: ExampleSpec,
    *,
    out_root: Path,
    strategy: str,
    beam_width: int,
    max_steps: int,
    tol: float,
    verbose_rewrite: bool,
) -> dict[str, object]:
    print(f"\n=== {example.name} ===")
    raw_kraus_ops = copy.deepcopy(example.kraus_ops)
    raw_metrics = _channel_metrics(raw_kraus_ops)

    rewritten_kraus_ops, rewrite_result = rewrite_kraus_ops(
        raw_kraus_ops,
        strategy=strategy,
        beam_width=beam_width,
        max_steps=max_steps,
        tol=tol,
        verbose=verbose_rewrite,
    )
    rewritten_metrics = _channel_metrics(rewritten_kraus_ops)
    _print_metric_delta(example.name, raw_metrics, rewritten_metrics)

    example_dir = out_root / example.name
    raw_prefix = example_dir / f"{example.name}_basic_raw"
    rewrite_prefix = example_dir / f"{example.name}_basic_rewrite"

    raw_export = export_basic_basis_qasm2(
        raw_kraus_ops,
        raw_prefix,
    )
    print(f"[{example.name}] wrote raw NAM QASM2: {raw_export['nam_path']}")
    print(f"[{example.name}] wrote raw IBM QASM2: {raw_export['ibm_path']}")

    rewrite_export = export_basic_basis_qasm2(
        rewritten_kraus_ops,
        rewrite_prefix,
    )
    print(f"[{example.name}] wrote rewrite NAM QASM2: {rewrite_export['nam_path']}")
    print(f"[{example.name}] wrote rewrite IBM QASM2: {rewrite_export['ibm_path']}")

    termination = rewrite_result.get("termination", {})
    print(
        f"[{example.name}] rewrite_search: "
        f"support {rewrite_result['initial_support']} -> {rewrite_result['final_support']}, "
        f"steps={len(rewrite_result['steps'])}, "
        f"stop={termination.get('stop_reason', 'unknown')}"
    )

    return {
        "example": example.name,
        "raw_metrics": raw_metrics,
        "rewrite_metrics": rewritten_metrics,
        "rewrite_search": {
            "initial_support": int(rewrite_result["initial_support"]),
            "final_support": int(rewrite_result["final_support"]),
            "steps": len(rewrite_result["steps"]),
            "termination": termination,
        },
        "raw_qasm": raw_export,
        "rewrite_qasm": rewrite_export,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate basic channel-LCU OpenQASM2 circuits for Table1 benchmark "
            "examples before and after Channel-IR rewrite_search."
        )
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        choices=("all", "hypercube", "tfim"),
        default=["all"],
        help="Examples to generate.",
    )
    parser.add_argument("--hypercube-n", type=int, default=4)
    parser.add_argument("--tfim-n", type=int, default=4)
    parser.add_argument("--tfim-delta-t", type=float, default=0.1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--rewrite-strategy",
        choices=("greedy", "beam"),
        default="greedy",
    )
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--verbose-rewrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.examples)
    if "all" in selected:
        selected = {"hypercube", "tfim"}

    examples: list[ExampleSpec] = []
    if "hypercube" in selected:
        examples.append(build_hypercube_random_walk_example(args.hypercube_n))
    if "tfim" in selected:
        examples.append(build_periodic_tfim_example(args.tfim_n, args.tfim_delta_t))

    if not examples:
        raise ValueError("No examples selected.")

    results = []
    for example in examples:
        results.append(
            process_example(
                example,
                out_root=args.out_root,
                strategy=args.rewrite_strategy,
                beam_width=args.beam_width,
                max_steps=args.max_steps,
                tol=args.tol,
                verbose_rewrite=args.verbose_rewrite,
            )
        )

    print("\n=== Summary ===")
    for result in results:
        raw = result["raw_metrics"]
        rewritten = result["rewrite_metrics"]
        print(
            f"{result['example']}: "
            f"kraus {raw['kraus_count']} -> {rewritten['kraus_count']}, "
            f"pauli {raw['pauli_terms_total']} -> {rewritten['pauli_terms_total']}"
        )


if __name__ == "__main__":
    main()
