#!/usr/bin/env python3
"""Build the repetition-code stabilizer-encoded DQE benchmark channel."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: E402


DEFAULT_OUT_ROOT = ROOT / "Baseline_results" / "eval_redesign" / "surface_repetition_dqe_channel"
PAULI_MUL_TABLE = {
    ("I", "I"): ("I", 1.0),
    ("I", "X"): ("X", 1.0),
    ("I", "Y"): ("Y", 1.0),
    ("I", "Z"): ("Z", 1.0),
    ("X", "I"): ("X", 1.0),
    ("Y", "I"): ("Y", 1.0),
    ("Z", "I"): ("Z", 1.0),
    ("X", "X"): ("I", 1.0),
    ("Y", "Y"): ("I", 1.0),
    ("Z", "Z"): ("I", 1.0),
    ("X", "Y"): ("Z", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "X"): ("Y", 1j),
    ("X", "Z"): ("Y", -1j),
}
SINGLE_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
PauliDict = dict[str, complex]


@dataclass(frozen=True)
class DqeSpec:
    d: int
    epsilon: float
    symmetrized: bool = False

    @property
    def t(self) -> int:
        return (self.d - 1) // 2

    @property
    def name(self) -> str:
        suffix = "_sym" if self.symmetrized else ""
        eps_label = str(self.epsilon).replace(".", "p")
        return f"repetition_dqe_d{self.d}_eps{eps_label}{suffix}"


def clean(terms: PauliDict, tol: float = 1e-14) -> PauliDict:
    return {label: coeff for label, coeff in terms.items() if abs(coeff) > tol}


def add(left: PauliDict, right: PauliDict) -> PauliDict:
    out = dict(left)
    for label, coeff in right.items():
        out[label] = out.get(label, 0.0) + coeff
    return clean(out)


def scale(terms: PauliDict, factor: complex) -> PauliDict:
    if factor == 0:
        return {}
    return clean({label: coeff * factor for label, coeff in terms.items()})


def multiply_label(left: str, right: str) -> tuple[str, complex]:
    label: list[str] = []
    phase: complex = 1.0
    for a, b in zip(left, right):
        out, local_phase = PAULI_MUL_TABLE[(a, b)]
        label.append(out)
        phase *= local_phase
    return "".join(label), phase


def mul(left: PauliDict, right: PauliDict) -> PauliDict:
    out: PauliDict = {}
    for left_label, left_coeff in left.items():
        for right_label, right_coeff in right.items():
            label, phase = multiply_label(left_label, right_label)
            out[label] = out.get(label, 0.0) + left_coeff * right_coeff * phase
    return clean(out)


def label_with(d: int, pauli: str, positions: list[int]) -> str:
    label = ["I"] * d
    for pos in positions:
        label[pos] = pauli
    return "".join(label)


def singleton(label: str, coeff: complex = 1.0) -> PauliDict:
    return {label: coeff}


def stabilizer_label(d: int, check_index: int) -> str:
    return label_with(d, "Z", [0, check_index + 1])


def stabilizer_projector(d: int, check_index: int, syndrome_bit: int) -> PauliDict:
    sign = 1.0 if syndrome_bit == 0 else -1.0
    return {
        "I" * d: 0.5,
        stabilizer_label(d, check_index): 0.5 * sign,
    }


def syndrome_projector(d: int, syndrome: tuple[int, ...]) -> PauliDict:
    out = singleton("I" * d)
    for check_index, bit in enumerate(syndrome):
        out = mul(out, stabilizer_projector(d, check_index, bit))
    return out


def syndrome_bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((index >> shift) & 1 for shift in range(width - 1, -1, -1))


def decoder_recovery_bits(spec: DqeSpec, syndrome: tuple[int, ...]) -> tuple[int, ...]:
    weight = sum(syndrome)
    f1 = 0 if weight <= spec.t else 1
    return tuple([f1] + [bit ^ f1 for bit in syndrome])


def recovery_label(spec: DqeSpec, syndrome: tuple[int, ...]) -> str:
    bits = decoder_recovery_bits(spec, syndrome)
    label = ["X" if bit else "I" for bit in bits]
    return "".join(label)


def correction_kraus(spec: DqeSpec, syndrome: tuple[int, ...]) -> PauliDict:
    recovery = singleton(recovery_label(spec, syndrome))
    return mul(recovery, syndrome_projector(spec.d, syndrome))


def logical_x_label(d: int) -> str:
    return "X" * d


def logical_z_label(d: int) -> str:
    return label_with(d, "Z", [0])


def encoded_success_operator(spec: DqeSpec) -> PauliDict:
    # M0 = Pi_X+ + (1-eps) Pi_X- = (1 - eps/2) I + (eps/2) Xbar.
    return {
        "I" * spec.d: 1.0 - spec.epsilon / 2.0,
        logical_x_label(spec.d): spec.epsilon / 2.0,
    }


def encoded_failure_reset_operator(spec: DqeSpec) -> PauliDict:
    # Zbar M1 = Z_1 * sqrt(eps(2-eps)) * (I - Xbar) / 2.
    prefactor = math.sqrt(spec.epsilon * (2.0 - spec.epsilon)) / 2.0
    m1 = {
        "I" * spec.d: prefactor,
        logical_x_label(spec.d): -prefactor,
    }
    return mul(singleton(logical_z_label(spec.d)), m1)


def compose_channels(left: list[PauliDict], right: list[PauliDict]) -> list[PauliDict]:
    return [mul(a, b) for a in left for b in right]


def build_repetition_dqe_channel(spec: DqeSpec) -> tuple[list[PauliDict], dict[str, Any]]:
    if spec.d < 3 or spec.d % 2 != 1:
        raise ValueError("d must be an odd integer >= 3.")
    if not (0.0 < spec.epsilon < 1.0):
        raise ValueError("epsilon must satisfy 0 < epsilon < 1.")

    correction = [
        correction_kraus(spec, syndrome_bits(index, spec.d - 1))
        for index in range(2 ** (spec.d - 1))
    ]
    measurement = [encoded_success_operator(spec), encoded_failure_reset_operator(spec)]
    channel = compose_channels(measurement, correction)
    if spec.symmetrized:
        channel = compose_channels(correction, channel)

    metadata = {
        "model": "repetition_code_stabilizer_encoded_dqe",
        "source_note": "Benchmark_guide/surface-code-dqe-channel.md",
        "d": spec.d,
        "t": spec.t,
        "epsilon": spec.epsilon,
        "symmetrized": spec.symmetrized,
        "physical_qubits": spec.d,
        "stabilizer_generators": [stabilizer_label(spec.d, j) for j in range(spec.d - 1)],
        "logical_x": logical_x_label(spec.d),
        "logical_z": logical_z_label(spec.d),
        "syndrome_branches": 2 ** (spec.d - 1),
        "encoded_measurement_branches": 2,
        "kraus_count": len(channel),
        "composition": "M_enc o C_rep" if not spec.symmetrized else "C_rep o M_enc o C_rep",
    }
    return channel, metadata


def pauli_dict_to_matrixsum(terms: PauliDict) -> Matrixsum:
    instances = []
    for label, coeff in terms.items():
        if abs(coeff) <= 1e-14:
            continue
        instances.append((PauliAtom(label, phase=coeff / abs(coeff)), float(abs(coeff))))
    return Matrixsum(instances)


def channel_metrics(channel: list[PauliDict]) -> dict[str, int]:
    return {
        "kraus_count": len(channel),
        "pauli_terms_total": sum(len(kraus) for kraus in channel),
        "max_terms_per_kraus": max((len(kraus) for kraus in channel), default=0),
        "min_terms_per_kraus": min((len(kraus) for kraus in channel), default=0),
        "nonzero_kraus_count": sum(1 for kraus in channel if kraus),
    }


def pauli_matrix(label: str) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for char in label:
        out = np.kron(out, SINGLE_PAULI[char])
    return out


def kraus_dense(terms: PauliDict) -> np.ndarray:
    size = 2 ** len(next(iter(terms)))
    out = np.zeros((size, size), dtype=complex)
    for label, coeff in terms.items():
        out += coeff * pauli_matrix(label)
    return out


def dense_completeness_error(channel: list[PauliDict]) -> float:
    if not channel:
        return 0.0
    size = 2 ** len(next(iter(channel[0])))
    acc = np.zeros((size, size), dtype=complex)
    for kraus in channel:
        mat = kraus_dense(kraus)
        acc += mat.conj().T @ mat
    return float(np.max(np.abs(acc - np.eye(size))))


def summarize_kraus(channel: list[PauliDict], max_preview: int) -> list[dict[str, Any]]:
    preview = []
    for index, kraus in enumerate(channel[:max_preview]):
        preview.append(
            {
                "index": index,
                "terms": len(kraus),
                "pauli_terms": [
                    {"label": label, "coeff_real": float(np.real(coeff)), "coeff_imag": float(np.imag(coeff))}
                    for label, coeff in sorted(kraus.items())
                ],
            }
        )
    return preview


def write_outputs(args: argparse.Namespace, spec: DqeSpec) -> dict[str, Any]:
    started = time.perf_counter()
    channel, metadata = build_repetition_dqe_channel(spec)
    metrics = channel_metrics(channel)

    completeness_error = None
    if spec.d <= args.dense_check_max_d:
        completeness_error = dense_completeness_error(channel)

    out_dir = Path(args.out_root) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "surface_code_dqe_repetition_code_channel_builder",
        "case": spec.name,
        "status": "ok",
        "metadata": metadata,
        "channel_metrics": metrics,
        "dense_completeness_max_abs_error": completeness_error,
        "kraus_preview": summarize_kraus(channel, args.preview_kraus),
        "total_time_s": float(time.perf_counter() - started),
    }
    summary_path = out_dir / f"{spec.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if args.write_matrixsum_json:
        matrixsum_payload = []
        for index, kraus in enumerate(channel):
            ms = pauli_dict_to_matrixsum(kraus)
            matrixsum_payload.append(
                {
                    "index": index,
                    "terms": [
                        {
                            "label": inst.expr,
                            "coeff": coeff,
                            "phase_real": float(np.real(inst.phase)),
                            "phase_imag": float(np.imag(inst.phase)),
                        }
                        for inst, coeff in ms.instances
                    ],
                }
            )
        (out_dir / f"{spec.name}_matrixsum_terms.json").write_text(
            json.dumps(matrixsum_payload, indent=2, default=str),
            encoding="utf-8",
        )

    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", nargs="+", type=int, default=[5, 7])
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--symmetrized", action="store_true")
    parser.add_argument("--dense-check-max-d", type=int, default=5)
    parser.add_argument("--preview-kraus", type=int, default=2)
    parser.add_argument("--write-matrixsum-json", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for d in args.d:
        spec = DqeSpec(d=d, epsilon=args.epsilon, symmetrized=args.symmetrized)
        summary = write_outputs(args, spec)
        summaries.append(summary)
        metrics = summary["channel_metrics"]
        print(
            f"{spec.name}: status={summary['status']}, "
            f"kraus={metrics['kraus_count']}, terms={metrics['pauli_terms_total']}, "
            f"max_terms={metrics['max_terms_per_kraus']}, "
            f"dense_error={summary['dense_completeness_max_abs_error']}"
        )
        print(f"summary: {summary['summary_path']}")
    index_path = Path(args.out_root) / "repetition_dqe_channel_index.json"
    index_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
