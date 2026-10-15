#!/usr/bin/env python3
"""Run focused CP4/RQ3 coherent higher-order K/q/K1 accuracy checks.

This runner complements ``run_cp4a_rq3_frontend_accuracy.py``.  The CP4A RQ3
right-panel proxy is a dense Taylor-polynomial experiment and has no coherent
Taylor order K1.  This script uses the actual coherent higher-order circuit
constructor from ``src/series_expansion.py`` so the K, q, and K1 parameters are
all explicit in the output.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "lindbladian_cp4a_mpl"))

from channel_IR import Lindbladian  # noqa: E402
from series_expansion import (  # noqa: E402
    construct_higher_order_circuit,
    create_index_set,
    simulate_circuit_statevec,
)


CHECKPOINT = "cp4_rq3_coherent_kqk1_accuracy"
DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "eval_redesign" / CHECKPOINT
DEFAULT_CASES = ("1,4,4", "2,3,3")
DEFAULT_SEGMENTS = (4, 8, 16, 32, 64, 128)
THEORY_BOUND_SOURCE = "arXiv:2212.02051v2 Eq.(75), unit hidden O-constant"
K1_TAYLOR_BOUND_SOURCE = "coarse coherent J Taylor remainder estimate"
SEED = 6313


def parse_case(raw: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in raw.replace(":", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"case must be K,q,K1; got {raw!r}")
    try:
        k_order, q_points, k1_order = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"case must contain integer K,q,K1; got {raw!r}") from exc
    if min(k_order, q_points, k1_order) < 0:
        raise argparse.ArgumentTypeError(f"case values must be nonnegative; got {raw!r}")
    if q_points == 0 and k_order > 0:
        raise argparse.ArgumentTypeError(f"q must be positive when K > 0; got {raw!r}")
    return k_order, q_points, k1_order


def build_decay_lindbladian(lambda_value: float, v: float, scaling: float) -> Lindbladian:
    h_terms: list[tuple[str, complex]] = []
    l_terms = [
        [
            ("X", np.sqrt(lambda_value * (v + 1.0)) / scaling),
            ("Y", 1j * np.sqrt(lambda_value * (v + 1.0)) / scaling),
        ],
        [
            ("X", np.sqrt(lambda_value * v) / scaling),
            ("Y", -1j * np.sqrt(lambda_value * v) / scaling),
        ],
    ]
    return Lindbladian(h_terms, l_terms)


def liouvillian_superop(h_op: np.ndarray, jumps: list[np.ndarray]) -> np.ndarray:
    dim = h_op.shape[0]
    eye = np.eye(dim, dtype=complex)
    superop = -1j * (np.kron(eye, h_op) - np.kron(h_op.T, eye))
    for jump in jumps:
        ldag_l = jump.conj().T @ jump
        superop += np.kron(jump.conj(), jump)
        superop += -0.5 * np.kron(eye, ldag_l)
        superop += -0.5 * np.kron(ldag_l.T, eye)
    return superop


def decay_liouvillian_superop(lambda_value: float, v: float, scaling: float) -> np.ndarray:
    x_op = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y_op = np.asarray([[0.0, -1j], [1j, 0.0]], dtype=complex)
    jumps = [
        np.sqrt(lambda_value * (v + 1.0)) / scaling * (x_op + 1j * y_op),
        np.sqrt(lambda_value * v) / scaling * (x_op - 1j * y_op),
    ]
    return liouvillian_superop(np.zeros((2, 2), dtype=complex), jumps)


def decay_lindblad_be_norm(lambda_value: float, v: float, scaling: float) -> float:
    alpha_down = 2.0 * np.sqrt(lambda_value * (v + 1.0)) / scaling
    alpha_up = 2.0 * np.sqrt(lambda_value * v) / scaling
    return float(0.5 * (alpha_down**2 + alpha_up**2))


def eq75_higher_order_bound(
    *,
    delta: float,
    segments: int,
    order: int,
    quadrature_points: int,
    l_be_norm: float,
    hidden_constant: float,
) -> dict[str, float]:
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


def coherent_taylor_remainder_bound(
    *,
    delta: float,
    segments: int,
    order: int,
    coherent_order: int,
    j_norm_bound: float,
) -> dict[str, float]:
    z = float(j_norm_bound) * float(delta)
    single_block = math.exp(z) * (z ** (int(coherent_order) + 1)) / math.factorial(int(coherent_order) + 1)
    segment_estimate = (int(order) + 1) * single_block
    return {
        "theory_k1_taylor_bound_source": K1_TAYLOR_BOUND_SOURCE,
        "theory_k1_single_coherent_block_remainder_bound": single_block,
        "theory_k1_segment_remainder_bound_estimate": segment_estimate,
        "theory_k1_total_remainder_bound_estimate": float(segments) * segment_estimate,
    }


def pure_state(vector: np.ndarray) -> Statevector:
    vector = np.asarray(vector, dtype=complex)
    vector = vector / np.linalg.norm(vector)
    return Statevector(vector)


def state_set(random_count: int, seed: int) -> list[tuple[str, Statevector]]:
    rng = np.random.default_rng(seed)
    states: list[tuple[str, Statevector]] = [
        ("zero", Statevector.from_label("0")),
        ("one", Statevector.from_label("1")),
        ("plus", pure_state(np.asarray([1.0, 1.0], dtype=complex))),
        ("minus", pure_state(np.asarray([1.0, -1.0], dtype=complex))),
    ]
    for idx in range(random_count):
        vec = rng.normal(size=2) + 1j * rng.normal(size=2)
        states.append((f"haar_like_{idx}", pure_state(vec)))
    return states


def pure_density(state: Statevector) -> np.ndarray:
    vec = np.asarray(state.data, dtype=complex).reshape(-1)
    return np.outer(vec, vec.conj())


def apply_superop(superop: np.ndarray, rho: np.ndarray) -> np.ndarray:
    dim = rho.shape[0]
    return (superop @ rho.reshape(dim * dim, order="F")).reshape((dim, dim), order="F")


def as_density_array(value: Any) -> np.ndarray:
    if hasattr(value, "data"):
        return np.asarray(value.data, dtype=complex)
    return np.asarray(value, dtype=complex)


def density_to_vec(rho: np.ndarray) -> np.ndarray:
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")


def vec_to_density(vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=complex).reshape((2, 2), order="F")


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left - right, dtype=complex)
    diff = 0.5 * (diff + diff.conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff))))


def logical_kraus_count(num_jumps: int, k_order: int, q_points: int) -> int:
    return 1 + sum(len(create_index_set(k, q_points, num_jumps)) for k in range(1, k_order + 1))


def fit_log_slope(xs: list[float], ys: list[float]) -> float | None:
    valid = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0 and np.isfinite(y)]
    if len(valid) < 2:
        return None
    x_log = np.log([x for x, _y in valid])
    y_log = np.log([y for _x, y in valid])
    slope, _intercept = np.polyfit(x_log, y_log, deg=1)
    return float(slope)


def channel_probe_states() -> list[tuple[str, Statevector]]:
    return [
        ("zero", Statevector.from_label("0")),
        ("one", Statevector.from_label("1")),
        ("plus", pure_state(np.asarray([1.0, 1.0], dtype=complex))),
        ("plus_i", pure_state(np.asarray([1.0, 1j], dtype=complex))),
    ]


def reconstruct_single_qubit_superop(
    qc: Any,
    sel_state: np.ndarray,
    reg_sizes: list[int],
    coeff_sum_sq: float,
) -> tuple[np.ndarray, float]:
    input_columns: list[np.ndarray] = []
    output_columns: list[np.ndarray] = []
    trace_devs: list[float] = []
    for _label, state in channel_probe_states():
        rho_in = pure_density(state)
        with contextlib.redirect_stdout(io.StringIO()):
            approx_dm = coeff_sum_sq * simulate_circuit_statevec(qc, state, sel_state, reg_sizes)
        approx = as_density_array(approx_dm)
        approx_trace = complex(np.trace(approx))
        trace_devs.append(abs(float(np.real(approx_trace)) - 1.0))
        input_columns.append(density_to_vec(rho_in))
        output_columns.append(density_to_vec(approx))

    input_basis = np.column_stack(input_columns)
    output_basis = np.column_stack(output_columns)
    superop = output_basis @ np.linalg.inv(input_basis)
    return superop, float(np.mean(trace_devs))


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod_name in ("numpy", "qiskit", "qiskit_aer", "qutip"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001
            versions[mod_name] = f"unavailable:{type(exc).__name__}"
    return versions


def git_status_short() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return [f"git_status_failed:{exc!r}"]
    return proc.stdout.splitlines()


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
    with path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    *,
    lind: Lindbladian,
    states: list[tuple[str, Statevector]],
    reference_superop: np.ndarray,
    k_order: int,
    q_points: int,
    k1_order: int,
    segments: int,
    total_time: float,
    lindblad_be_norm: float,
    theory_bound_constant: float,
    coherent_opt: str,
    raw_fh: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    delta = float(total_time) / int(segments)
    qc, reg_sizes, coeff_sum_sq, sel_state = construct_higher_order_circuit(
        lind,
        k_order,
        q_points,
        delta,
        k1_order,
        opt=coherent_opt,
    )
    build_elapsed = time.perf_counter() - started
    theory = eq75_higher_order_bound(
        delta=float(delta),
        segments=int(segments),
        order=int(k_order),
        quadrature_points=int(q_points),
        l_be_norm=float(lindblad_be_norm),
        hidden_constant=float(theory_bound_constant),
    )
    k1_theory = coherent_taylor_remainder_bound(
        delta=float(delta),
        segments=int(segments),
        order=int(k_order),
        coherent_order=int(k1_order),
        j_norm_bound=float(theory["theory_j_norm_bound"]),
    )
    sim_started = time.perf_counter()
    segment_superop, probe_trace_dev = reconstruct_single_qubit_superop(qc, sel_state, reg_sizes, coeff_sum_sq)
    approx_superop = np.linalg.matrix_power(segment_superop, int(segments))
    sim_elapsed_total = time.perf_counter() - sim_started

    state_rows: list[dict[str, Any]] = []
    distances: list[float] = []
    raw_distances: list[float] = []
    trace_devs: list[float] = []

    for label, state in states:
        rho_in = pure_density(state)
        approx = apply_superop(approx_superop, rho_in)
        approx_trace = complex(np.trace(approx))
        approx_normalized = approx / approx_trace
        reference = apply_superop(reference_superop, rho_in)
        distance = trace_distance(approx, reference)
        normalized_distance = trace_distance(approx_normalized, reference)
        distances.append(normalized_distance)
        raw_distances.append(distance)
        trace_devs.append(abs(float(np.real(approx_trace)) - 1.0))

        raw_row = {
            "checkpoint": CHECKPOINT,
            "part": "coherent_higher_order",
            "benchmark": "decay_only_lindbladian_higher_order_expansion",
            "duhamel_order_K": int(k_order),
            "sampling_points_q": int(q_points),
            "coherent_taylor_order_K1": int(k1_order),
            "total_time": float(total_time),
            "segments": int(segments),
            "delta": float(delta),
            "state_label": label,
            "trace_distance": float(distance),
            "trace_distance_after_trace_normalization": float(normalized_distance),
            "approx_trace_real_before_normalization": float(np.real(approx_trace)),
            "approx_trace_imag_before_normalization": float(np.imag(approx_trace)),
            "per_state_simulation_elapsed_seconds": 0.0,
            "status": "ok",
        }
        raw_fh.write(json.dumps(raw_row, sort_keys=True) + "\n")
        state_rows.append(raw_row)

    summary = {
        "checkpoint": CHECKPOINT,
        "part": "coherent_higher_order",
        "benchmark": "decay_only_lindbladian_higher_order_expansion",
        "duhamel_order_K": int(k_order),
        "sampling_points_q": int(q_points),
        "coherent_taylor_order_K1": int(k1_order),
        "total_time": float(total_time),
        "segments": int(segments),
        "delta": float(delta),
        "coherent_opt": coherent_opt,
        "state_count": len(states),
        "logical_kraus_count": logical_kraus_count(len(lind.L_list), k_order, q_points),
        "select_register_size": int(reg_sizes[0]),
        "ctrl_register_size": int(reg_sizes[1]),
        "system_register_size": int(reg_sizes[2]),
        "circuit_num_qubits": int(qc.num_qubits),
        "raw_circuit_depth": int(qc.depth()),
        "raw_circuit_size": int(qc.size()),
        "raw_circuit_ops": json.dumps({str(k): int(v) for k, v in qc.count_ops().items()}, sort_keys=True),
        "coeff_sum_square": float(coeff_sum_sq),
        "select_state_nonzero": int(np.count_nonzero(np.abs(sel_state) > 1.0e-12)),
        "avg_trace_distance": float(np.mean(distances)),
        "max_trace_distance": float(np.max(distances)),
        "min_trace_distance": float(np.min(distances)),
        "std_trace_distance": float(np.std(distances)),
        "avg_trace_distance_before_trace_normalization": float(np.mean(raw_distances)),
        "max_trace_distance_before_trace_normalization": float(np.max(raw_distances)),
        "segment_probe_trace_mean_abs_deviation": float(probe_trace_dev),
        "approx_trace_mean_abs_deviation_before_normalization": float(np.mean(trace_devs)),
        "theory_bound_source": THEORY_BOUND_SOURCE,
        "theory_quadrature_points": int(q_points),
        "theory_hidden_constant": float(theory_bound_constant),
        **theory,
        **k1_theory,
        "theory_avg_error_le_total_bound_eq75": bool(float(np.mean(distances)) <= theory["theory_total_bound_eq75"]),
        "theory_max_error_le_total_bound_eq75": bool(float(np.max(distances)) <= theory["theory_total_bound_eq75"]),
        "theory_avg_error_le_total_bound_eq75_plus_k1": bool(
            float(np.mean(distances))
            <= theory["theory_total_bound_eq75"] + k1_theory["theory_k1_total_remainder_bound_estimate"]
        ),
        "theory_max_error_le_total_bound_eq75_plus_k1": bool(
            float(np.max(distances))
            <= theory["theory_total_bound_eq75"] + k1_theory["theory_k1_total_remainder_bound_estimate"]
        ),
        "error_metric": "state_trace_distance_after_repeated_segment_superop_and_final_trace_normalization",
        "reference_method": "dense_liouvillian_expm_total_time",
        "build_elapsed_seconds": float(build_elapsed),
        "simulation_elapsed_seconds": float(sim_elapsed_total),
        "elapsed_seconds": float(time.perf_counter() - started),
        "status": "ok",
        "failure_reason": "",
    }
    return summary, state_rows


def write_notes(path: Path, summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# CP4/RQ3 coherent K/q/K1 accuracy checks",
        "",
        "This checkpoint uses the actual coherent higher-order circuit constructor, not the dense Taylor-polynomial proxy used by `run_cp4a_rq3_frontend_accuracy.py`.",
        "",
        "## Model",
        "",
        f"- Benchmark: one-qubit decay/thermalization higher-order expansion.",
        f"- Parameters: lambda={args.lambda_value}, v={args.v}, scaling={args.scaling}, T={args.total_time}.",
        f"- Segments: {', '.join(str(x) for x in args.segments)}.",
        f"- Coherent block option: `{args.coherent_opt}`.",
        f"- Eq. (75) hidden constant: {args.theory_bound_constant}.",
        f"- States: 4 structured states plus {args.random_states} seeded Haar-like states.",
        "- Each row constructs the coherent circuit for one segment of duration `delta=T/M`, reconstructs its one-qubit superoperator from four probe states, applies it `M` times, and compares against dense `expm(TL)`.",
        "- Theory columns use the same coarse Eq. (75) total bound as the dense proxy plot, with this model's block-encoding norm. The K1 Taylor-remainder columns are a separate diagnostic estimate, not part of Eq. (75).",
        "",
        "## Results",
        "",
    ]
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in summaries:
        key = (
            int(row["duhamel_order_K"]),
            int(row["sampling_points_q"]),
            int(row["coherent_taylor_order_K1"]),
        )
        grouped.setdefault(key, []).append(row)

    for (k_order, q_points, k1_order), rows in grouped.items():
        rows = sorted(rows, key=lambda item: float(item["delta"]))
        slope = fit_log_slope(
            [float(row["delta"]) for row in rows],
            [float(row["avg_trace_distance"]) for row in rows],
        )
        slope_text = "n/a" if slope is None else f"{slope:.4f}"
        avg_pass = all(bool(row.get("theory_avg_error_le_total_bound_eq75")) for row in rows)
        max_pass = all(bool(row.get("theory_max_error_le_total_bound_eq75")) for row in rows)
        largest_eq75 = max(float(row["theory_total_bound_eq75"]) for row in rows)
        largest_k1 = max(float(row["theory_k1_total_remainder_bound_estimate"]) for row in rows)
        lines.append(
            "- "
            f"K={k_order}, q={q_points}, K1={k1_order}: "
            f"slope vs delta={slope_text}, points={len(rows)}, "
            f"avg error range={float(rows[0]['avg_trace_distance']):.6g}..{float(rows[-1]['avg_trace_distance']):.6g}, "
            f"Eq.(75) avg/max pass={avg_pass}/{max_pass}, "
            f"max Eq.(75) bound={largest_eq75:.6g}, max K1 diagnostic={largest_k1:.6g}, "
            f"qubits={rows[-1]['circuit_num_qubits']}, logical Kraus={rows[-1]['logical_kraus_count']}."
        )
    lines.extend(
        [
            "",
            "## Parameter note",
            "",
            "`K1` is the Taylor truncation order for the coherent `J` block in `series_expansion.construct_circuit_coherent`. The dense CP4A right-panel proxy does not have this parameter.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cases", type=parse_case, nargs="+", default=[parse_case(c) for c in DEFAULT_CASES])
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--scaling", type=float, default=4.01)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--segments", type=int, nargs="+", default=list(DEFAULT_SEGMENTS))
    parser.add_argument("--random-states", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--theory-bound-constant", type=float, default=1.0)
    parser.add_argument("--coherent-opt", choices=("No", "Matrix-order"), default="No")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lind = build_decay_lindbladian(args.lambda_value, args.v, args.scaling)
    generator = decay_liouvillian_superop(args.lambda_value, args.v, args.scaling)
    reference_superop = expm(float(args.total_time) * generator)
    lindblad_be_norm = decay_lindblad_be_norm(args.lambda_value, args.v, args.scaling)
    states = state_set(args.random_states, args.seed)

    summary_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    raw_path = args.out_dir / "coherent_kqk1_raw_runs.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_fh:
        raw_fh.write(
            json.dumps(
                {
                    "event": "start",
                    "checkpoint": CHECKPOINT,
                    "argv": sys.argv,
                    "seed": int(args.seed),
                    "status": "ok",
                },
                sort_keys=True,
            )
            + "\n"
        )
        for k_order, q_points, k1_order in args.cases:
            for segments in args.segments:
                summary, per_state = run_case(
                    lind=lind,
                    states=states,
                    reference_superop=reference_superop,
                    k_order=k_order,
                    q_points=q_points,
                    k1_order=k1_order,
                    segments=int(segments),
                    total_time=args.total_time,
                    lindblad_be_norm=lindblad_be_norm,
                    theory_bound_constant=args.theory_bound_constant,
                    coherent_opt=args.coherent_opt,
                    raw_fh=raw_fh,
                )
                summary_rows.append(summary)
                state_rows.extend(per_state)
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

    write_csv(args.out_dir / "coherent_kqk1_accuracy.csv", summary_rows)
    write_csv(args.out_dir / "coherent_kqk1_state_errors.csv", state_rows)
    write_notes(args.out_dir / "notes.md", summary_rows, args)

    manifest = {
        "checkpoint": CHECKPOINT,
        "created_at_epoch": time.time(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "scope": "Focused coherent higher-order K/q/K1 accuracy checks for CP4/RQ3",
        "model": {
            "benchmark": "decay_only_lindbladian_higher_order_expansion",
            "lambda_value": float(args.lambda_value),
            "v": float(args.v),
            "scaling": float(args.scaling),
            "total_time": float(args.total_time),
            "num_jump_operators": int(len(lind.L_list)),
            "lindblad_be_norm": float(lindblad_be_norm),
        },
        "theory": {
            "bound_source": THEORY_BOUND_SOURCE,
            "hidden_constant": float(args.theory_bound_constant),
            "k1_taylor_bound_source": K1_TAYLOR_BOUND_SOURCE,
        },
        "segments": [int(x) for x in args.segments],
        "cases": [
            {
                "duhamel_order_K": int(k_order),
                "sampling_points_q": int(q_points),
                "coherent_taylor_order_K1": int(k1_order),
            }
            for k_order, q_points, k1_order in args.cases
        ],
        "outputs": {
            "summary_csv": str(args.out_dir / "coherent_kqk1_accuracy.csv"),
            "state_errors_csv": str(args.out_dir / "coherent_kqk1_state_errors.csv"),
            "raw_runs_jsonl": str(raw_path),
            "notes": str(args.out_dir / "notes.md"),
            "manifest": str(args.out_dir / "manifest.json"),
        },
        "summaries": summary_rows,
        "package_versions": package_versions(),
        "git_status_short": git_status_short(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(args.out_dir),
                "summary_csv": str(args.out_dir / "coherent_kqk1_accuracy.csv"),
                "cases": [
                    {
                        "K": row["duhamel_order_K"],
                        "q": row["sampling_points_q"],
                        "K1": row["coherent_taylor_order_K1"],
                        "segments": row["segments"],
                        "delta": row["delta"],
                        "avg_trace_distance": row["avg_trace_distance"],
                        "max_trace_distance": row["max_trace_distance"],
                    }
                    for row in summary_rows
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
