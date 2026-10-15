"""Generate CP4A/RQ3 frontend approximation accuracy data.

This script intentionally stays at the numerical channel level. It does not
export QASM or call external optimizers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "lindbladian_cp4a_mpl"))

from channel_IR import Lindbladian  # noqa: E402
from channel_LCU import Lindblad_to_channel  # noqa: E402


CHECKPOINT = "cp4_rq3_frontend_accuracy"
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "eval_redesign" / CHECKPOINT
DEFAULT_FIGURE = ROOT / "Results" / "eval_redesign" / "figures" / "rq3_accuracy_main.pdf"
DEFAULT_LEFT_FIGURE = ROOT / "Results" / "eval_redesign" / "figures" / "rq3_accuracy_first_order.pdf"
DEFAULT_RIGHT_FIGURE = ROOT / "Results" / "eval_redesign" / "figures" / "rq3_accuracy_higher_order.svg"
DEFAULT_COHERENT_KQK1_CSV = (
    ROOT
    / "Baseline_results"
    / "eval_redesign"
    / "cp4_rq3_coherent_kqk1_accuracy"
    / "coherent_kqk1_accuracy.csv"
)
CP0_DIR = ROOT / "codex_results" / "eval_redesign" / "cp0_infra"
SEED = 4313
FIRST_ORDER_REFERENCE = "dense_superoperator_expm_from_lindfront_generator"
HIGHER_ORDER_REFERENCE = "dense_superoperator_expm_closed_form_equivalent"
HIGHER_ORDER_THEORY_SOURCE = "arXiv:2212.02051v2 Eq.(75), unit hidden O-constant"


@dataclass(frozen=True)
class StateSet:
    states: list[np.ndarray]
    labels: list[str]


def build_periodic_tfim_lindbladian_pauli(
    num_qubits: int,
    gamma: float,
) -> tuple[list[tuple[str, complex]], list[list[tuple[str, complex]]]]:
    h_terms: list[tuple[str, complex]] = []
    l_terms: list[list[tuple[str, complex]]] = []

    for i in range(num_qubits):
        z_neighbors = [i, (i + 1) % num_qubits]
        zz_str = "".join("Z" if k in z_neighbors else "I" for k in range(num_qubits))
        x_str = "".join("X" if k == i else "I" for k in range(num_qubits))
        y_str = "".join("Y" if k == i else "I" for k in range(num_qubits))
        h_terms.append((zz_str, -1.0))
        h_terms.append((x_str, -1.0))
        l_terms.append([(x_str, gamma), (y_str, -1j * gamma)])

    return h_terms, l_terms


def matrixsum_to_dense(ms: Any, dim: int) -> np.ndarray:
    dense = np.zeros((dim, dim), dtype=complex)
    for inst, coeff in ms.instances:
        dense += complex(coeff) * np.asarray(inst.to_operator().data, dtype=complex)
    return dense


def lindbladian_dense_parts(lind: Lindbladian) -> tuple[np.ndarray, list[np.ndarray]]:
    n = lind.__size__()
    dim = 2**n
    h = matrixsum_to_dense(lind.H, dim) if lind.H.length else np.zeros((dim, dim), dtype=complex)
    jumps = [matrixsum_to_dense(l_op, dim) for l_op in lind.L_list]
    return h, jumps


def liouvillian_superop(h: np.ndarray, jumps: list[np.ndarray]) -> np.ndarray:
    dim = h.shape[0]
    eye = np.eye(dim, dtype=complex)
    superop = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for jump in jumps:
        ldag_l = jump.conj().T @ jump
        superop += np.kron(jump.conj(), jump)
        superop += -0.5 * np.kron(eye, ldag_l)
        superop += -0.5 * np.kron(ldag_l.T, eye)
    return superop


def kraus_superop(kraus: list[np.ndarray]) -> np.ndarray:
    dim = kraus[0].shape[0]
    out = np.zeros((dim * dim, dim * dim), dtype=complex)
    for k_op in kraus:
        out += np.kron(k_op.conj(), k_op)
    return out


def apply_superop(superop: np.ndarray, rho: np.ndarray) -> np.ndarray:
    dim = rho.shape[0]
    return (superop @ rho.reshape(dim * dim, order="F")).reshape((dim, dim), order="F")


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left - right, dtype=complex)
    diff = 0.5 * (diff + diff.conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff))))


def pure_density(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex)
    vec = vec / np.linalg.norm(vec)
    return np.outer(vec, vec.conj())


def state_set(num_qubits: int, random_count: int, seed: int) -> StateSet:
    rng = np.random.default_rng(seed + 97 * num_qubits)
    dim = 2**num_qubits
    states: list[np.ndarray] = []
    labels: list[str] = []

    basis0 = np.zeros(dim, dtype=complex)
    basis0[0] = 1.0
    states.append(pure_density(basis0))
    labels.append("zero")

    basis_last = np.zeros(dim, dtype=complex)
    basis_last[-1] = 1.0
    states.append(pure_density(basis_last))
    labels.append("one")

    plus = np.ones(dim, dtype=complex) / math.sqrt(dim)
    states.append(pure_density(plus))
    labels.append("uniform_plus")

    phase = np.exp(2j * np.pi * np.arange(dim) / dim) / math.sqrt(dim)
    states.append(pure_density(phase))
    labels.append("phase_ramp")

    for idx in range(random_count):
        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        states.append(pure_density(vec))
        labels.append(f"haar_like_{idx}")

    return StateSet(states=states, labels=labels)


def fit_log_slope(xs: list[float], ys: list[float]) -> float | None:
    valid = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0 and np.isfinite(y)]
    if len(valid) < 2:
        return None
    x_log = np.log([p[0] for p in valid])
    y_log = np.log([p[1] for p in valid])
    slope, _ = np.polyfit(x_log, y_log, deg=1)
    return float(slope)


def first_order_bound(delta: float, h: np.ndarray, jumps: list[np.ndarray], l_super: np.ndarray) -> float:
    dim = h.shape[0]
    dissipator = sum((jump.conj().T @ jump for jump in jumps), start=np.zeros_like(h))
    a_op = -1j * h - 0.5 * dissipator
    a_norm = float(np.linalg.norm(a_op, ord=2))
    l_norm = float(np.linalg.norm(l_super, ord=2))
    frob_to_trace = math.sqrt(dim)
    exact_remainder = 0.5 * (delta * l_norm) ** 2 * math.exp(delta * l_norm)
    explicit_kraus_second_order = (delta**2) * (a_norm**2)
    return float(0.5 * frob_to_trace * (exact_remainder + explicit_kraus_second_order))


def single_qubit_relaxation_be_norm(lambda0: float) -> float:
    """Block-encoding norm proxy for sqrt(lambda0) sigma_-.

    The paper defines ||L||_be = alpha_0 + 1/2 sum_j alpha_j^2.
    For sigma_- = (X + iY) / 2, the Pauli LCU norm of sqrt(lambda0) sigma_-
    is sqrt(lambda0), so ||L||_be = lambda0 / 2 when H=0.
    """
    return 0.5 * float(lambda0)


def eq75_higher_order_bound(
    *,
    delta: float,
    segments: int,
    order: int,
    quadrature_points: int,
    l_be_norm: float,
    hidden_constant: float,
) -> dict[str, float]:
    """Coarse per-point estimate from Eq. (75) of arXiv:2212.02051v2.

    Eq. (75) is stated for a single segment and keeps a hidden O constant in
    the Gaussian-quadrature term.  We use the segment time delta=T/M, the
    Lemma-5 time-dependent truncation term, ||J|| <= ||L||_be from Eq. (76),
    and a triangle/union bound over M repeated segments.
    """
    k_order = int(order)
    q_points = int(quadrature_points)
    l_norm = float(l_be_norm)
    j_norm_bound = l_norm
    constant = float(hidden_constant)
    segment_time = float(delta)

    truncation = (2.0 * l_norm * segment_time) ** (k_order + 1) / math.factorial(k_order + 1)
    quadrature_eq72 = 0.0
    for k in range(1, k_order + 1):
        quadrature_eq72 += (
            (2.0 * segment_time) ** (k - 1)
            * (2.0 ** (k + 1))
            * (l_norm**k)
            * (j_norm_bound ** (2 * q_points))
            * (segment_time ** (2 * q_points + 1))
            * q_points
            / (math.factorial(k - 1) * math.factorial(2 * q_points))
        )
    quadrature_eq72 *= constant
    quadrature_eq75 = (
        constant
        * (l_norm ** (2 * q_points))
        * (segment_time ** (2 * q_points + 1))
        * q_points
        * math.exp(4.0 * segment_time * l_norm)
        / math.factorial(2 * q_points)
    )
    segment_bound_eq72 = truncation + quadrature_eq72
    segment_bound_eq75 = truncation + quadrature_eq75

    return {
        "theory_lindblad_be_norm": l_norm,
        "theory_j_norm_bound": j_norm_bound,
        "theory_segment_truncation_bound": truncation,
        "theory_segment_quadrature_bound_eq72": quadrature_eq72,
        "theory_segment_quadrature_bound_eq75": quadrature_eq75,
        "theory_segment_bound_eq72": segment_bound_eq72,
        "theory_segment_bound_eq75": segment_bound_eq75,
        "theory_total_bound_eq72": float(segments) * segment_bound_eq72,
        "theory_total_bound_eq75": float(segments) * segment_bound_eq75,
    }


def lindfront_generator_from_kraus_derivative(
    lind: Lindbladian,
    dim: int,
    eps: float = 1.0e-7,
) -> np.ndarray:
    """Construct the dense Liouvillian in the same matrix convention as Lindblad_to_channel.

    The derivative is evaluated by a small finite difference of the same dense
    superoperator builder used for M_delta. This avoids mixing Qiskit/qutip
    endianness conventions in this CP4A numerical proxy.
    """
    channel_ensemble, _, _ = Lindblad_to_channel(lind, eps)
    kraus_ms = channel_ensemble.channels[0][1]
    kraus = [matrixsum_to_dense(ms, dim) for ms in kraus_ms]
    superop = kraus_superop(kraus)
    return (superop - np.eye(dim * dim, dtype=complex)) / eps


def run_first_order(args: argparse.Namespace, raw_fh: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    n_values = [3]
    if args.try_n5:
        n_values.append(5)

    for num_qubits in n_values:
        start = time.perf_counter()
        try:
            gamma = math.sqrt(0.1) / 2.0
            h_terms, l_terms = build_periodic_tfim_lindbladian_pauli(num_qubits, gamma)
            lind = Lindbladian(h_terms, l_terms)
            h, jumps = lindbladian_dense_parts(lind)
            dim = 2**num_qubits
            l_super = lindfront_generator_from_kraus_derivative(lind, dim)
            states = state_set(num_qubits, args.first_order_random_states, args.seed)
            deltas = list(args.first_order_deltas)

            if num_qubits == 5 and args.n5_max_points > 0:
                deltas = deltas[-args.n5_max_points :]

            avg_errors: list[float] = []
            for delta in deltas:
                channel_ensemble, _, coeff_sums = Lindblad_to_channel(lind, float(delta))
                kraus_ms = channel_ensemble.channels[0][1]
                kraus = [matrixsum_to_dense(ms, dim) for ms in kraus_ms]
                approx = kraus_superop(kraus)
                reference = expm(float(delta) * l_super)
                errors = []
                trace_devs = []
                for label, rho in zip(states.labels, states.states):
                    approx_rho = apply_superop(approx, rho)
                    ref_rho = apply_superop(reference, rho)
                    errors.append(trace_distance(approx_rho, ref_rho))
                    trace_devs.append(abs(float(np.real(np.trace(approx_rho))) - 1.0))
                    raw_fh.write(
                        json.dumps(
                            {
                        "part": "first_order",
                        "benchmark": "periodic_tfim_local_decay",
                        "n_qubits": num_qubits,
                        "appendix_only": bool(num_qubits == 5),
                        "delta": float(delta),
                        "state_label": label,
                        "trace_distance": errors[-1],
                                "approx_trace": float(np.real(np.trace(approx_rho))),
                                "reference_trace": float(np.real(np.trace(ref_rho))),
                                "status": "ok",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

                avg_error = float(np.mean(errors))
                avg_errors.append(avg_error)
                rows.append(
                    {
                        "checkpoint": CHECKPOINT,
                        "part": "first_order",
                        "benchmark": "periodic_tfim_local_decay",
                        "n_qubits": num_qubits,
                        "appendix_only": bool(num_qubits == 5),
                        "delta": float(delta),
                        "state_count": len(states.states),
                        "seed": args.seed,
                        "reference_method": FIRST_ORDER_REFERENCE,
                        "error_metric": "average_state_trace_distance",
                        "avg_trace_distance": avg_error,
                        "max_trace_distance": float(np.max(errors)),
                        "min_trace_distance": float(np.min(errors)),
                        "std_trace_distance": float(np.std(errors)),
                        "theoretical_bound_channel_upper": first_order_bound(float(delta), h, jumps, l_super),
                        "legacy_pauli_bound": float(5.0 * (float(delta) * lind.operator_norm()) ** 2),
                        "approx_trace_mean_abs_deviation": float(np.mean(trace_devs)),
                        "kraus_count": len(kraus),
                        "coeff_sum_l1_per_kraus": json.dumps([float(x) for x in coeff_sums]),
                        "status": "ok",
                        "failure_reason": "",
                    }
                )

            slope = fit_log_slope([float(x) for x in deltas], avg_errors)
            summary.append(
                {
                    "checkpoint": CHECKPOINT,
                    "part": "first_order",
                    "benchmark": f"periodic_tfim_local_decay_n{num_qubits}",
                    "appendix_only": bool(num_qubits == 5),
                    "order": 1,
                    "slope_variable": "delta",
                    "log_log_slope": "" if slope is None else slope,
                    "points": len(deltas),
                    "state_count": len(states.states),
                    "seed": args.seed,
                    "reference_method": FIRST_ORDER_REFERENCE,
                    "error_metric": "average_state_trace_distance",
                    "status": "ok",
                    "failure_reason": "",
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )
        except Exception as exc:  # noqa: BLE001
            raw_fh.write(
                json.dumps(
                    {
                        "part": "first_order",
                        "benchmark": "periodic_tfim_local_decay",
                        "n_qubits": num_qubits,
                        "appendix_only": bool(num_qubits == 5),
                        "status": "failed",
                        "failure_reason": repr(exc),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            summary.append(
                {
                    "checkpoint": CHECKPOINT,
                    "part": "first_order",
                    "benchmark": f"periodic_tfim_local_decay_n{num_qubits}",
                    "appendix_only": bool(num_qubits == 5),
                    "order": 1,
                    "slope_variable": "delta",
                    "log_log_slope": "",
                    "points": 0,
                    "state_count": 0,
                    "seed": args.seed,
                    "reference_method": FIRST_ORDER_REFERENCE,
                    "error_metric": "average_state_trace_distance",
                    "status": "failed",
                    "failure_reason": repr(exc),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )

    if not args.try_n5:
        summary.append(
            {
                "checkpoint": CHECKPOINT,
                "part": "first_order",
                "benchmark": "periodic_tfim_local_decay_n5",
                "appendix_only": True,
                "order": 1,
                "slope_variable": "delta",
                "log_log_slope": "",
                "points": 0,
                "state_count": 0,
                "seed": args.seed,
                "reference_method": FIRST_ORDER_REFERENCE,
                "error_metric": "average_state_trace_distance",
                "status": "not_run",
                "failure_reason": "N=5 exact dense superoperator sweep is optional for CP4A; main run used preferred N=3 only.",
                "elapsed_seconds": 0.0,
            }
        )
        raw_fh.write(
            json.dumps(
                {
                    "part": "first_order",
                    "benchmark": "periodic_tfim_local_decay",
                    "n_qubits": 5,
                    "appendix_only": True,
                    "status": "not_run",
                    "failure_reason": "Optional N=5 not requested; see reproduce.sh for --try-n5.",
                },
                sort_keys=True,
            )
            + "\n"
        )

    return rows, summary


def single_qubit_relaxation_superop(lambda0: float) -> np.ndarray:
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jump = math.sqrt(lambda0) * sigma_minus
    return liouvillian_superop(np.zeros((2, 2), dtype=complex), [jump])


def run_higher_order(args: argparse.Namespace, raw_fh: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    start = time.perf_counter()
    lambda0 = args.lambda0
    total_time = args.total_time
    l_super = single_qubit_relaxation_superop(lambda0)
    l_be_norm = single_qubit_relaxation_be_norm(lambda0)
    exact = expm(total_time * l_super)
    states = state_set(1, args.higher_order_random_states, args.seed + 1000)

    for order in args.orders:
        deltas: list[float] = []
        avg_errors: list[float] = []
        avg_bound_passes: list[bool] = []
        max_bound_passes: list[bool] = []
        for segments in args.segments:
            delta = total_time / int(segments)
            quadrature_points = int(args.higher_order_theory_q or order)
            theory = eq75_higher_order_bound(
                delta=float(delta),
                segments=int(segments),
                order=int(order),
                quadrature_points=quadrature_points,
                l_be_norm=l_be_norm,
                hidden_constant=args.theory_bound_constant,
            )
            poly = np.eye(l_super.shape[0], dtype=complex)
            term = np.eye(l_super.shape[0], dtype=complex)
            for k in range(1, int(order) + 1):
                term = term @ (delta * l_super) / k
                poly = poly + term
            approx = np.linalg.matrix_power(poly, int(segments))
            errors = []
            trace_devs = []
            for label, rho in zip(states.labels, states.states):
                approx_rho = apply_superop(approx, rho)
                exact_rho = apply_superop(exact, rho)
                errors.append(trace_distance(approx_rho, exact_rho))
                trace_devs.append(abs(float(np.real(np.trace(approx_rho))) - 1.0))
                raw_fh.write(
                    json.dumps(
                        {
                            "part": "higher_order",
                            "benchmark": "single_qubit_amplitude_decay",
                            "lambda0": lambda0,
                            "total_time": total_time,
                            "segments": int(segments),
                            "delta": float(delta),
                            "order": int(order),
                            "state_label": label,
                            "trace_distance": errors[-1],
                            "approx_trace": float(np.real(np.trace(approx_rho))),
                            "reference_trace": float(np.real(np.trace(exact_rho))),
                            "status": "ok",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

            avg_error = float(np.mean(errors))
            max_error = float(np.max(errors))
            avg_le_bound = bool(avg_error <= theory["theory_total_bound_eq75"])
            max_le_bound = bool(max_error <= theory["theory_total_bound_eq75"])
            deltas.append(float(delta))
            avg_errors.append(avg_error)
            avg_bound_passes.append(avg_le_bound)
            max_bound_passes.append(max_le_bound)
            rows.append(
                {
                    "checkpoint": CHECKPOINT,
                    "part": "higher_order",
                    "benchmark": "single_qubit_amplitude_decay",
                    "lambda0": lambda0,
                    "total_time": total_time,
                    "segments": int(segments),
                    "delta": float(delta),
                    "order": int(order),
                    "state_count": len(states.states),
                    "seed": args.seed + 1000,
                    "reference_method": HIGHER_ORDER_REFERENCE,
                    "error_metric": "average_state_trace_distance",
                    "avg_trace_distance": avg_error,
                    "max_trace_distance": max_error,
                    "min_trace_distance": float(np.min(errors)),
                    "std_trace_distance": float(np.std(errors)),
                    "approx_trace_mean_abs_deviation": float(np.mean(trace_devs)),
                    "theory_bound_source": HIGHER_ORDER_THEORY_SOURCE,
                    "theory_quadrature_points": quadrature_points,
                    "theory_hidden_constant": float(args.theory_bound_constant),
                    **theory,
                    "theory_avg_error_le_total_bound_eq75": avg_le_bound,
                    "theory_max_error_le_total_bound_eq75": max_le_bound,
                    "plot_theoretical_upper_bound": avg_le_bound,
                    "status": "ok",
                    "failure_reason": "",
                }
            )

        slope = fit_log_slope(deltas, avg_errors)
        summary.append(
            {
                "checkpoint": CHECKPOINT,
                "part": "higher_order",
                "benchmark": "single_qubit_amplitude_decay",
                "order": int(order),
                "slope_variable": "delta=T/M",
                "log_log_slope": "" if slope is None else slope,
                "points": len(deltas),
                "state_count": len(states.states),
                "seed": args.seed + 1000,
                "reference_method": HIGHER_ORDER_REFERENCE,
                "error_metric": "average_state_trace_distance",
                "theory_bound_source": HIGHER_ORDER_THEORY_SOURCE,
                "theory_quadrature_points": int(args.higher_order_theory_q or order),
                "theory_hidden_constant": float(args.theory_bound_constant),
                "theory_bound_all_avg_pass": bool(all(avg_bound_passes)),
                "theory_bound_all_max_pass": bool(all(max_bound_passes)),
                "status": "ok",
                "failure_reason": "",
                "elapsed_seconds": float(time.perf_counter() - start),
            }
        )

    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def higher_order_theory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "checkpoint",
        "benchmark",
        "lambda0",
        "total_time",
        "segments",
        "delta",
        "order",
        "avg_trace_distance",
        "max_trace_distance",
        "theory_bound_source",
        "theory_quadrature_points",
        "theory_hidden_constant",
        "theory_lindblad_be_norm",
        "theory_j_norm_bound",
        "theory_segment_truncation_bound",
        "theory_segment_quadrature_bound_eq72",
        "theory_segment_quadrature_bound_eq75",
        "theory_total_bound_eq72",
        "theory_total_bound_eq75",
        "theory_avg_error_le_total_bound_eq75",
        "theory_max_error_le_total_bound_eq75",
        "plot_theoretical_upper_bound",
    ]
    return [{field: row.get(field, "") for field in fields} for row in rows if row.get("part") == "higher_order"]


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod_name in ["numpy", "scipy", "matplotlib", "qiskit", "qutip"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            versions[mod_name] = f"unavailable:{type(exc).__name__}"
    return versions


def git_status_short() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return [f"git_status_failed:{exc!r}"]
    return proc.stdout.splitlines()


def write_reproduce(path: Path, args: argparse.Namespace) -> None:
    conda = os.environ.get("CONDA_EXE", "/Users/chester_hqf/miniconda3/bin/conda")
    command = [
        conda,
        "run",
        "-n",
        args.conda_env,
        "python",
        "scripts/run_cp4a_rq3_frontend_accuracy.py",
        "--out-dir",
        str(args.out_dir),
        "--figure",
        str(args.figure),
        "--figure-left",
        str(args.figure_left),
        "--figure-right",
        str(args.figure_right),
        "--seed",
        str(args.seed),
        "--lambda0",
        str(args.lambda0),
        "--total-time",
        str(args.total_time),
    ]
    if args.try_n5:
        command.append("--try-n5")
    command.extend(["--n5-max-points", str(args.n5_max_points)])
    command.extend(["--first-order-deltas", *[str(x) for x in args.first_order_deltas]])
    command.extend(["--first-order-random-states", str(args.first_order_random_states)])
    command.extend(["--higher-order-random-states", str(args.higher_order_random_states)])
    command.extend(["--segments", *[str(x) for x in args.segments]])
    command.extend(["--orders", *[str(x) for x in args.orders]])
    command.extend(["--higher-order-theory-q", str(args.higher_order_theory_q)])
    command.extend(["--theory-bound-constant", str(args.theory_bound_constant)])
    command.extend(["--coherent-kqk1-csv", str(args.coherent_kqk1_csv)])
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {ROOT!s}",
                (
                    f"if [ {str(args.coherent_kqk1_csv)!r} = {str(DEFAULT_COHERENT_KQK1_CSV)!r} ]; then\n"
                    f"  {conda} run -n {args.conda_env} python scripts/run_cp4a_rq3_coherent_kqk1_accuracy.py "
                    f"--total-time {args.total_time} "
                    f"--segments {' '.join(str(x) for x in args.segments)} "
                    f"--theory-bound-constant {args.theory_bound_constant}\n"
                    "fi"
                ),
                " ".join(command),
                (
                    f"{conda} run -n {args.conda_env} python - <<'PY'\n"
                    "from pathlib import Path\n"
                    f"required = {[str(p) for p in required_outputs(args.out_dir, args.figure, args.figure_left, args.figure_right)]!r}\n"
                    "missing = [p for p in required if not Path(p).exists()]\n"
                    "if missing:\n"
                    "    raise SystemExit('missing outputs: ' + ', '.join(missing))\n"
                    "print('CP4A outputs present:', len(required))\n"
                    "PY"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def required_outputs(out_dir: Path, figure: Path, figure_left: Path, figure_right: Path) -> list[Path]:
    return [
        out_dir / "manifest.json",
        out_dir / "first_order_accuracy.csv",
        out_dir / "higher_order_accuracy.csv",
        out_dir / "higher_order_theory_bounds.csv",
        out_dir / "summary.csv",
        out_dir / "raw_runs.jsonl",
        out_dir / "notes.md",
        out_dir / "reproduce.sh",
        figure,
        figure_left,
        figure_right,
    ]


def plot_first_order_panel(ax: Any, first_rows: list[dict[str, Any]], *, title: str) -> None:
    n3 = []
    for row in first_rows:
        try:
            if int(row["n_qubits"]) == 3 and row["status"] == "ok":
                n3.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    if n3:
        x = np.asarray([r["delta"] for r in n3], dtype=float)
        y = np.asarray([r["avg_trace_distance"] for r in n3], dtype=float)
        b = np.asarray([r["theoretical_bound_channel_upper"] for r in n3], dtype=float)
        order = np.argsort(x)
        ax.loglog(x[order], y[order], marker="o", label="TFIM N=3 measured")
        ax.loglog(x[order], b[order], linestyle="--", label="channel upper bound")
    ax.set_xlabel("timestep delta")
    ax.set_ylabel("average state trace distance")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)


def plot_higher_order_panel(
    ax: Any,
    higher_rows: list[dict[str, Any]],
    *,
    title: str,
    coherent_rows: list[dict[str, str]] | None = None,
) -> None:
    def inverse_delta(row: dict[str, Any]) -> float:
        if row.get("segments") not in ("", None):
            return float(row["segments"])
        return 1.0 / float(row["delta"])

    coherent_case_labels = {(2, 3, 3): "case 3"}
    case_colors = {
        "case 1": "tab:blue",
        "case 2": "tab:orange",
        "case 3": "tab:green",
    }
    for order_value, label in [(1, "case 1"), (2, "case 2")]:
        subset = []
        for row in higher_rows:
            try:
                if int(row["order"]) == order_value and row["status"] == "ok":
                    subset.append(row)
            except (KeyError, TypeError, ValueError):
                continue
        if not subset:
            continue
        x = np.asarray([inverse_delta(r) for r in subset], dtype=float)
        y = np.asarray([r["avg_trace_distance"] for r in subset], dtype=float)
        idx = np.argsort(x)
        ax.loglog(x[idx], y[idx], marker="o", color=case_colors[label], label=label)
        bound_subset = [
            r
            for r in subset
            if r.get("plot_theoretical_upper_bound") in (True, "True", "true", "1", 1)
            and r.get("theory_total_bound_eq75") not in ("", None)
        ]
        if bound_subset:
            bx = np.asarray([inverse_delta(r) for r in bound_subset], dtype=float)
            by = np.asarray([r["theory_total_bound_eq75"] for r in bound_subset], dtype=float)
            bidx = np.argsort(bx)
            ax.loglog(bx[bidx], by[bidx], linestyle="--", color=case_colors[label], label=f"{label} bound")
    if coherent_rows:
        coherent_markers = ["X", "D", "P", "v"]
        grouped: dict[tuple[int, int, int], list[dict[str, str]]] = {}
        for row in coherent_rows:
            if row.get("status") != "ok":
                continue
            try:
                key = (
                    int(row["duhamel_order_K"]),
                    int(row["sampling_points_q"]),
                    int(row["coherent_taylor_order_K1"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if key not in coherent_case_labels:
                continue
            grouped.setdefault(key, []).append(row)
        for idx, ((k_order, q_points, k1_order), rows) in enumerate(sorted(grouped.items())):
            x_values: list[float] = []
            y_values: list[float] = []
            for row in rows:
                try:
                    x_values.append(inverse_delta(row))
                    y_values.append(float(row["avg_trace_distance"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if not x_values:
                continue
            order = np.argsort(np.asarray(x_values, dtype=float))
            x_arr = np.asarray(x_values, dtype=float)[order]
            y_arr = np.asarray(y_values, dtype=float)[order]
            label = coherent_case_labels[(k_order, q_points, k1_order)]
            color = case_colors[label]
            ax.loglog(
                x_arr,
                y_arr,
                linestyle="-.",
                marker=coherent_markers[idx % len(coherent_markers)],
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
                color=color,
                label=label,
                zorder=6,
            )
            bound_values: list[float] = []
            bound_x_values: list[float] = []
            for row in rows:
                try:
                    if row.get("theory_avg_error_le_total_bound_eq75") not in ("True", "true", True):
                        continue
                    bound_x_values.append(inverse_delta(row))
                    bound_values.append(float(row["theory_total_bound_eq75"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if bound_values:
                bound_order = np.argsort(np.asarray(bound_x_values, dtype=float))
                bx_arr = np.asarray(bound_x_values, dtype=float)[bound_order]
                by_arr = np.asarray(bound_values, dtype=float)[bound_order]
                ax.loglog(
                    bx_arr,
                    by_arr,
                    linestyle=":",
                    color=color,
                    label=f"{label} bound",
                    zorder=5,
                )
    ax.set_xlabel(r"$1/\Delta(t)$", fontsize=14)
    ax.set_ylabel("average state trace distance", fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)


def make_figure(first_rows: list[dict[str, Any]], higher_rows: list[dict[str, Any]], figure: Path) -> str:
    import matplotlib.pyplot as plt

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    plot_first_order_panel(axes[0], first_rows, title="(a) First-order LindFront")
    plot_higher_order_panel(
        axes[1],
        higher_rows,
        title="(b) Error Rate vs Single-segment Evolution Duration",
    )
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    return str(figure)


def make_first_order_figure(first_rows: list[dict[str, Any]], figure: Path) -> str:
    import matplotlib.pyplot as plt

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), constrained_layout=True)
    plot_first_order_panel(ax, first_rows, title="First-order LindFront")
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    return str(figure)


def make_higher_order_figure(
    higher_rows: list[dict[str, Any]],
    figure: Path,
    coherent_rows: list[dict[str, str]] | None = None,
) -> str:
    import matplotlib.pyplot as plt

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), constrained_layout=True)
    plot_higher_order_panel(
        ax,
        higher_rows,
        title="Error Rate vs Single-segment Evolution Duration",
        coherent_rows=coherent_rows,
    )
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    return str(figure)


def write_notes(
    path: Path,
    summary_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    coherent_rows: list[dict[str, str]] | None = None,
) -> None:
    first = [r for r in summary_rows if r["part"] == "first_order" and r["status"] == "ok"]
    higher = [r for r in summary_rows if r["part"] == "higher_order" and r["status"] == "ok"]
    n5 = [r for r in summary_rows if r["benchmark"] == "periodic_tfim_local_decay_n5"]

    lines = [
        "# CP4A RQ3 frontend approximation accuracy",
        "",
        "## Scope",
        "",
        "This checkpoint generated only frontend numerical accuracy data. It did not run CP4B, external optimizers, or large-scale benchmark suites, and it did not pass high-level QASM to any external tool.",
        "",
        "## Metrics and references",
        "",
        "- First-order panel: periodic TFIM with local decay, preferred main size N=3. The first-order LindFront channel M_delta is built with `Lindblad_to_channel`; the reference is dense superoperator `expm(delta * L)`.",
        "- Higher-order panel: single-qubit amplitude decay with lambda0=2 by default, total time T=1. The order-p frontend is the pth-order Taylor polynomial of the Liouvillian segment map, repeated M=T/delta times; the reference is dense `expm(T * L)`, equivalent to the closed-form amplitude-damping evolution.",
        "- Empirical errors are state trace distance / average state trace distance over fixed structured states and seeded random pure states. They are not diamond norms.",
        "- The first-order plotted bound is a conservative channel-level upper bound derived from superoperator 2-norm/Frobenius-to-trace conversion. It is shown only as an upper bound on state-level error, not as an empirical diamond-norm measurement.",
        "- Higher-order theoretical estimates use the coarse Eq. (75) bound from arXiv:2212.02051v2 with the hidden `O` constant set by `--theory-bound-constant` (default 1). The estimate uses segment time `delta=T/M`, `||L||_be=lambda0/2`, `||J||<=||L||_be`, `q=order` unless `--higher-order-theory-q` overrides it, and multiplies the per-segment bound by M.",
        "- The Eq. (75) estimate is a diamond-norm upper-bound proxy for the paper's scaled-Gaussian-quadrature construction; the simulation data remain average state trace distances for the dense Taylor-polynomial proxy.",
        "",
        "## Slopes",
        "",
    ]

    for row in first + higher:
        slope = row["log_log_slope"]
        slope_text = "n/a" if slope == "" else f"{float(slope):.4f}"
        lines.append(
            f"- {row['part']} {row['benchmark']} order {row['order']}: slope vs {row['slope_variable']} = {slope_text}, points={row['points']}, states={row['state_count']}."
        )

    higher_theory = [r for r in summary_rows if r["part"] == "higher_order" and r["status"] == "ok"]
    if higher_theory:
        lines.extend(["", "## Higher-order Eq. (75) checks", ""])
        for row in higher_theory:
            lines.append(
                f"- order {row['order']}: avg-error bound check={row.get('theory_bound_all_avg_pass')}; "
                f"max-error bound check={row.get('theory_bound_all_max_pass')}; "
                f"q={row.get('theory_quadrature_points')}; hidden constant={row.get('theory_hidden_constant')}."
            )

    if coherent_rows:
        lines.extend(["", "## Coherent K/q/K1 overlay", ""])
        lines.append(
            f"- The split right SVG overlays the selected coherent-circuit segment-sweep curve from `{args.coherent_kqk1_csv}` as case 3."
        )
        coherent_case_labels = {(2, 3, 3): "case 3"}
        grouped: dict[tuple[int, int, int], list[dict[str, str]]] = {}
        for row in coherent_rows:
            if row.get("status") != "ok":
                continue
            try:
                key = (
                    int(row["duhamel_order_K"]),
                    int(row["sampling_points_q"]),
                    int(row["coherent_taylor_order_K1"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if key not in coherent_case_labels:
                continue
            grouped.setdefault(key, []).append(row)
        for (k_order, q_points, k1_order), rows in sorted(grouped.items()):
            deltas = [float(row["delta"]) for row in rows]
            inverse_deltas = [
                float(row["segments"]) if row.get("segments") not in ("", None) else 1.0 / float(row["delta"])
                for row in rows
            ]
            errors = [float(row["avg_trace_distance"]) for row in rows]
            slope = fit_log_slope(deltas, errors)
            slope_text = "n/a" if slope is None else f"{slope:.4f}"
            eq75_bounds = [
                float(row["theory_total_bound_eq75"])
                for row in rows
                if row.get("theory_total_bound_eq75") not in ("", None)
            ]
            avg_pass = all(row.get("theory_avg_error_le_total_bound_eq75") in ("True", "true", True) for row in rows)
            max_pass = all(row.get("theory_max_error_le_total_bound_eq75") in ("True", "true", True) for row in rows)
            bound_text = (
                "n/a"
                if not eq75_bounds
                else f"{min(eq75_bounds):.6g}..{max(eq75_bounds):.6g}"
            )
            lines.append(
                "- "
                f"{coherent_case_labels[(k_order, q_points, k1_order)]}: "
                f"points={len(rows)}, 1/Delta(t) range={min(inverse_deltas):.6g}..{max(inverse_deltas):.6g}, "
                f"avg error range={min(errors):.6g}..{max(errors):.6g}, slope vs delta={slope_text}, "
                f"Eq.(75) bound range={bound_text}, avg/max pass={avg_pass}/{max_pass}."
            )

    if n5:
        lines.extend(["", "## Optional N=5 status", ""])
        for row in n5:
            if row["status"] == "ok":
                slope = row["log_log_slope"]
                slope_text = "n/a" if slope == "" else f"{float(slope):.4f}"
                lines.append(
                    f"- N=5 appendix-only status: ok; appendix_only=true; points={row['points']}; states={row['state_count']}; slope vs {row['slope_variable']} = {slope_text}; not included in `rq3_accuracy_main.pdf`."
                )
                if args.n5_max_points > 0 and args.n5_max_points < len(args.first_order_deltas):
                    lines.append(
                        f"- N=5 sweep used the largest {args.n5_max_points} delta points from the first-order grid to bound dense-superoperator runtime."
                    )
            else:
                lines.append(f"- N=5 status: {row['status']}; reason: {row['failure_reason']}")

    lines.extend(
        [
            "",
            "## Manual checks",
            "",
            "- Confirm the figure axis labels say state trace distance, not diamond norm.",
            "- Confirm first-order rows compare M_delta against `expm(delta * L)` and higher-order rows use total time T=1.",
            "- Confirm the optional channel-level bound is visually above the measured state-level TFIM N=3 curve.",
            "- Confirm the combined figure and split figures render as expected.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--figure-left", type=Path, default=DEFAULT_LEFT_FIGURE)
    parser.add_argument("--figure-right", type=Path, default=DEFAULT_RIGHT_FIGURE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--conda-env", default="qiskit_simulate")
    parser.add_argument("--try-n5", action="store_true")
    parser.add_argument("--n5-max-points", type=int, default=4)
    parser.add_argument(
        "--first-order-deltas",
        type=float,
        nargs="+",
        default=[0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1],
    )
    parser.add_argument("--first-order-random-states", type=int, default=8)
    parser.add_argument("--higher-order-random-states", type=int, default=8)
    parser.add_argument("--lambda0", type=float, default=2.0)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--segments", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--higher-order-theory-q", type=int, default=0)
    parser.add_argument("--theory-bound-constant", type=float, default=1.0)
    parser.add_argument("--coherent-kqk1-csv", type=Path, default=DEFAULT_COHERENT_KQK1_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.figure_left.parent.mkdir(parents=True, exist_ok=True)
    args.figure_right.parent.mkdir(parents=True, exist_ok=True)

    raw_path = args.out_dir / "raw_runs.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_fh:
        raw_fh.write(
            json.dumps(
                {
                    "event": "start",
                    "checkpoint": CHECKPOINT,
                    "argv": sys.argv,
                    "seed": args.seed,
                    "cp0_manifest": str(CP0_DIR / "manifest.json"),
                    "status": "ok",
                },
                sort_keys=True,
            )
            + "\n"
        )
        first_rows, first_summary = run_first_order(args, raw_fh)
        higher_rows, higher_summary = run_higher_order(args, raw_fh)
        raw_fh.write(
            json.dumps(
                {
                    "event": "end",
                    "checkpoint": CHECKPOINT,
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "status": "ok",
                },
                sort_keys=True,
            )
            + "\n"
        )

    summary_rows = first_summary + higher_summary
    coherent_rows = read_csv(args.coherent_kqk1_csv)
    write_csv(args.out_dir / "first_order_accuracy.csv", first_rows)
    write_csv(args.out_dir / "higher_order_accuracy.csv", higher_rows)
    write_csv(args.out_dir / "higher_order_theory_bounds.csv", higher_order_theory_rows(higher_rows))
    write_csv(args.out_dir / "summary.csv", summary_rows)
    figure_status = "ok"
    figure_error = ""
    try:
        make_figure(first_rows, higher_rows, args.figure)
        make_first_order_figure(first_rows, args.figure_left)
        make_higher_order_figure(higher_rows, args.figure_right, coherent_rows)
    except Exception as exc:  # noqa: BLE001
        figure_status = "failed"
        figure_error = repr(exc)
        with raw_path.open("a", encoding="utf-8") as raw_fh:
            raw_fh.write(
                json.dumps(
                    {
                        "event": "figure",
                        "status": "failed",
                        "failure_reason": figure_error,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    write_notes(args.out_dir / "notes.md", summary_rows, args, coherent_rows)
    write_reproduce(args.out_dir / "reproduce.sh", args)

    manifest = {
        "checkpoint": CHECKPOINT,
        "agent": "CP4A-RQ3-Frontend-Accuracy",
        "created_at_epoch": time.time(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "scope": "CP4A only; frontend approximation accuracy",
        "basis_policy": {
            "external_tools_called": False,
            "qasm_exported": False,
            "high_level_qasm_passed_to_external_tools": False,
            "nam_metric_basis": ["rz", "h", "x", "cx"],
            "ibm_metric_basis": ["u1", "u2", "u3", "cx"],
        },
        "cp0_inputs": {
            "manifest": str(CP0_DIR / "manifest.json"),
            "benchmark_inventory": str(CP0_DIR / "benchmark_inventory.csv"),
            "tool_inventory": str(CP0_DIR / "tool_inventory.csv"),
            "existing_results_inventory": str(CP0_DIR / "existing_results_inventory.csv"),
        },
        "methods": {
            "first_order": {
                "benchmark": "periodic_tfim_local_decay",
                "n_qubits_main": 3,
                "n_qubits_appendix_only": [5],
                "n5_appendix_only": True,
                "n5_attempted": bool(args.try_n5),
                "n5_max_points": int(args.n5_max_points),
                "deltas": [float(x) for x in args.first_order_deltas],
                "state_count": int(4 + args.first_order_random_states),
                "seed": args.seed,
                "reference_method": FIRST_ORDER_REFERENCE,
                "error_metric": "average_state_trace_distance",
                "theoretical_bound": "channel-level upper bound from superoperator 2-norm/Frobenius-to-trace conversion",
            },
            "higher_order": {
                "benchmark": "single_qubit_amplitude_decay",
                "lambda0": args.lambda0,
                "total_time": args.total_time,
                "segments": [int(x) for x in args.segments],
                "orders": [int(x) for x in args.orders],
                "theory_bound_source": HIGHER_ORDER_THEORY_SOURCE,
                "theory_quadrature_points": "order" if args.higher_order_theory_q == 0 else int(args.higher_order_theory_q),
                "theory_hidden_constant": float(args.theory_bound_constant),
                "theory_lindblad_be_norm": single_qubit_relaxation_be_norm(args.lambda0),
                "state_count": int(4 + args.higher_order_random_states),
                "seed": args.seed + 1000,
                "reference_method": HIGHER_ORDER_REFERENCE,
                "error_metric": "average_state_trace_distance",
            },
            "coherent_kqk1_overlay": {
                "source_csv": str(args.coherent_kqk1_csv),
                "row_count": len(coherent_rows),
                "plotted_in_split_right_svg": bool(coherent_rows),
                "note": "These coherent-circuit K/q/K1 points are overlaid only on the split higher-order SVG; the dense proxy right-panel rows have no K1 parameter.",
            },
        },
        "outputs": {p.name: str(p) for p in required_outputs(args.out_dir, args.figure, args.figure_left, args.figure_right)},
        "figure_status": figure_status,
        "figure_error": figure_error,
        "summary_status": summary_rows,
        "package_versions": package_versions(),
        "git_status_short": git_status_short(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(args.out_dir),
                "figure": str(args.figure),
                "figure_left": str(args.figure_left),
                "figure_right": str(args.figure_right),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
