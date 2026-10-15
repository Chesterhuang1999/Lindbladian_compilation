"""Generic QSVT construction for functions split into even and odd parts.

The input Matrixsum ``J`` is block-encoded as ``J / alpha`` with
``alpha = sum_j |c_j|`` inherited from ``BlockEncoding``. Therefore the branch
functions below are functions of the normalized scalar variable ``x in [-1, 1]``.
For example, to approximate ``exp(-i J t)``, use
``cos(alpha * t * x)`` and ``sin(alpha * t * x)`` with odd coefficient ``-1j``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebval
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from qiskit import QuantumCircuit, QuantumRegister

from block_encoding import BlockEncoding
from channel_IR import Matrixsum
from subroutine import apply_poly_phases


ScalarFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class QSVTBranchInfo:
    """Metadata for one parity branch in the generic QSVT construction."""

    name: str
    degree: int
    parity: int
    input_coeff: complex
    polynomial_scale: float
    effective_coeff: complex
    max_polynomial_value: float
    phases: tuple[float, ...]


def _as_real_values(values, name: str, atol: float) -> np.ndarray:
    arr = np.asarray(values)
    if np.max(np.abs(np.imag(arr))) > atol:
        raise ValueError(
            f"{name} must be real-valued for this QSVT phase generator. "
            "Put complex factors into the branch coefficient instead."
        )
    return np.real(arr)


def _validate_degree(degree: int, parity: int, name: str) -> None:
    if degree < 0:
        raise ValueError(f"{name} degree must be non-negative.")
    if degree % 2 != parity:
        expected = "even" if parity == 0 else "odd"
        raise ValueError(f"{name} degree must be {expected}; got {degree}.")


def _phase_adjustment(phi_values: np.ndarray, parity: int) -> np.ndarray:
    """Match the phase convention used by ``Hamiltonian.qsvt_Hamiltonian``."""
    phi = np.array(phi_values, dtype=float, copy=True)
    phi[0] -= np.pi / 4
    if parity == 0:
        phi[1:-1] += np.pi / 2
    else:
        phi[1:] += np.pi / 2
    phi[-1] += 3 * np.pi / 4
    return phi


def _qsp_phases_for_real_parity_function(
    func: ScalarFunction,
    degree: int,
    parity: int,
    name: str,
    sample_points: int,
    atol: float,
    normalize_over_one: bool,
) -> tuple[np.ndarray, float, float]:
    """Interpolate a parity function and compute QSP phases.

    Returns ``(phases, max_value, polynomial_scale)``. If the interpolated
    polynomial exceeds one in magnitude, it is scaled down before phase
    synthesis and the caller should absorb ``polynomial_scale`` into the LCU
    branch coefficient.
    """
    _validate_degree(degree, parity, name)

    def real_func(x):
        return _as_real_values(func(x), name, atol)

    poly = Chebyshev.interpolate(real_func, degree)
    coeffs = np.array(poly.coef, dtype=float, copy=True)
    if parity == 0:
        coeffs[1::2] = 0.0
    else:
        coeffs[0::2] = 0.0

    x_list = np.linspace(-1, 1, sample_points)
    values = chebval(x_list, coeffs)
    max_value = float(np.max(np.abs(values)))
    if max_value <= atol:
        return np.array([], dtype=float), max_value, 0.0

    polynomial_scale = max_value if normalize_over_one and max_value > 1.0 else 1.0
    qsp_coeffs = coeffs / polynomial_scale
    if degree == 0:
        constant = float(np.clip(qsp_coeffs[0], -1.0, 1.0))
        return np.array([np.arccos(constant)], dtype=float), max_value, polynomial_scale

    phases, _, _ = QuantumSignalProcessingPhases(
        qsp_coeffs,
        signal_operator="Wx",
        method="sym_qsp",
        chebyshev_basis=True,
    )
    return _phase_adjustment(np.asarray(phases), parity), max_value, polynomial_scale


def _build_branch_circuit(
    qc_basic: QuantumCircuit,
    sys_size: int,
    phases: np.ndarray,
    label: str,
) -> QuantumCircuit:
    ctrl_size = qc_basic.num_qubits - sys_size
    anc = QuantumRegister(1, "a")
    sys = QuantumRegister(sys_size, "s")
    gadget = qc_basic.to_gate(label="QSVT_basic_gadget")

    if ctrl_size == 0:
        qc_branch = QuantumCircuit(anc, sys, name=label)
        return apply_poly_phases(phases, gadget, qc_branch, anc, None)

    ctrl = QuantumRegister(ctrl_size, "c")
    qc_branch = QuantumCircuit(anc, ctrl, sys, name=label)
    return apply_poly_phases(phases, gadget, qc_branch, anc, ctrl)


def _combine_two_branches(
    even_circuit: QuantumCircuit,
    odd_circuit: QuantumCircuit,
    sys_size: int,
    even_coeff: complex,
    odd_coeff: complex,
) -> tuple[QuantumCircuit, float]:
    """LCU-combine two parity QSVT branches with arbitrary complex weights."""
    even_abs = abs(even_coeff)
    odd_abs = abs(odd_coeff)
    lcu_norm = even_abs + odd_abs
    if lcu_norm == 0:
        raise ValueError("At least one QSVT branch must have non-zero coefficient.")

    ctrl_size = even_circuit.num_qubits - sys_size - 1
    weight_even = even_abs / lcu_norm
    theta = 2 * np.arccos(np.sqrt(weight_even))

    sel = QuantumRegister(1, "sel")
    anc = QuantumRegister(1, "a")
    sys = QuantumRegister(sys_size, "s")

    if ctrl_size == 0:
        qc_main = QuantumCircuit(sel, anc, sys, name="qsvt_even_odd")
    else:
        ctrl = QuantumRegister(ctrl_size, "c")
        qc_main = QuantumCircuit(sel, anc, ctrl, sys, name="qsvt_even_odd")

    even_gate = even_circuit.to_gate(label="QSVT_even").control(1, ctrl_state="0")
    odd_gate = odd_circuit.to_gate(label="QSVT_odd").control(1, ctrl_state="1")

    even_phase = np.angle(even_coeff)
    odd_phase = np.angle(odd_coeff)

    qc_main.ry(theta, sel[0])
    qc_main.append(even_gate, qc_main.qubits)
    qc_main.append(odd_gate, qc_main.qubits)
    qc_main.global_phase += even_phase
    qc_main.p(odd_phase - even_phase, sel[0])
    qc_main.ry(-theta, sel[0])

    return qc_main, float(lcu_norm)


def qsvt_even_odd_function(
    J: Matrixsum,
    even_func: ScalarFunction | None,
    odd_func: ScalarFunction | None,
    even_degree: int,
    odd_degree: int,
    *,
    even_coeff: complex = 1.0,
    odd_coeff: complex = 1.0,
    opt: str = "No",
    normalize_over_one: bool = True,
    sample_points: int = 1000,
    atol: float = 1e-12,
) -> tuple[QuantumCircuit, QuantumCircuit, dict]:
    """Build QSVT for ``c_e f_e(J/alpha) + c_o f_o(J/alpha)``.

    ``f_e`` must be real-valued and even, and ``f_o`` must be real-valued and
    odd. Complex phases such as the ``-i`` in ``cos(x) - i sin(x)`` should be
    supplied through ``even_coeff`` and ``odd_coeff``.

    The returned circuit block-encodes the requested combination divided by
    ``metadata["lcu_norm"]``. This mirrors the selector-qubit LCU normalization
    in ``qsvt_Hamiltonian``.
    """
    be = BlockEncoding(J)
    qc_basic = be.circuit(opt=opt)
    subnorm_fac = float(sum(abs(coeff) for _, coeff in J.instances))

    active: list[tuple[str, QuantumCircuit, QSVTBranchInfo]] = []
    branch_info: dict[str, QSVTBranchInfo | None] = {"even": None, "odd": None}

    branch_specs = [
        ("even", even_func, even_degree, 0, complex(even_coeff)),
        ("odd", odd_func, odd_degree, 1, complex(odd_coeff)),
    ]

    for name, func, degree, parity, coeff in branch_specs:
        if func is None or abs(coeff) <= atol:
            continue

        phases, max_value, polynomial_scale = _qsp_phases_for_real_parity_function(
            func=func,
            degree=degree,
            parity=parity,
            name=name,
            sample_points=sample_points,
            atol=atol,
            normalize_over_one=normalize_over_one,
        )
        effective_coeff = coeff * polynomial_scale
        info = QSVTBranchInfo(
            name=name,
            degree=degree,
            parity=parity,
            input_coeff=coeff,
            polynomial_scale=float(polynomial_scale),
            effective_coeff=effective_coeff,
            max_polynomial_value=float(max_value),
            phases=tuple(float(phi) for phi in phases),
        )
        branch_info[name] = info

        if polynomial_scale <= atol:
            continue

        qc_branch = _build_branch_circuit(
            qc_basic=qc_basic,
            sys_size=J.size,
            phases=phases,
            label=f"qsvt_{name}",
        )
        active.append((name, qc_branch, info))

    if len(active) == 0:
        raise ValueError("Both QSVT branches are zero or disabled.")

    if len(active) == 1:
        _, qc_main, info = active[0]
        lcu_norm = abs(info.effective_coeff)
        qc_main = qc_main.copy(name=f"qsvt_{info.name}_only")
        qc_main.global_phase += np.angle(info.effective_coeff)
    else:
        even_branch = next(item for item in active if item[0] == "even")
        odd_branch = next(item for item in active if item[0] == "odd")
        qc_main, lcu_norm = _combine_two_branches(
            even_circuit=even_branch[1],
            odd_circuit=odd_branch[1],
            sys_size=J.size,
            even_coeff=even_branch[2].effective_coeff,
            odd_coeff=odd_branch[2].effective_coeff,
        )

    metadata = {
        "block_encoding_norm": subnorm_fac,
        "lcu_norm": float(lcu_norm),
        "success_amplitude_scale": float(1.0 / lcu_norm) if lcu_norm != 0 else 0.0,
        "branches": branch_info,
    }
    return qc_basic, qc_main, metadata


def qsvt_hamiltonian_via_even_odd(
    J: Matrixsum,
    t: float,
    deg: int = 4,
    opt: str = "No",
    **kwargs,
) -> tuple[QuantumCircuit, QuantumCircuit, dict]:
    """Reproduce ``qsvt_Hamiltonian`` through the generic even/odd interface."""
    alpha = float(sum(abs(coeff) for _, coeff in J.instances))
    return qsvt_even_odd_function(
        J,
        even_func=lambda x: np.cos(alpha * t * x),
        odd_func=lambda x: np.sin(alpha * t * x),
        even_degree=deg,
        odd_degree=deg + 1,
        even_coeff=1.0,
        odd_coeff=-1j,
        opt=opt,
        **kwargs,
    )
