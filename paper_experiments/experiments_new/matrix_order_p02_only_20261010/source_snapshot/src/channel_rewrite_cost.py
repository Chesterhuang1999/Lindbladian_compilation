"""Static non-Clifford proxy for Kraus rewriting; no circuit/placement imports.

H = Lambda * (outer PREPARE + flag tree + inner PREPAREs + SELECT + phases).
PREPARE uses the manuscript weight 7*(N-1), with N the padded address space.
Each Pauli string is one branch primitive, irrespective of system width or
Pauli weight. O's later placement/ANF savings are deliberately not predicted.
These are heuristic NC units (T and arbitrary non-Clifford rotations cost 1),
not measured NAM counts or precision-dependent Clifford+T synthesis costs.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np


STATIC_COST_MODEL = "H_length_static_nc_v1"
LEGACY_COST_MODEL = "H_length_generic_stateprep"
PHASE_TOL = 1e-10


def ceil_log2(n: int) -> int:
    return max(0, (int(n) - 1).bit_length()) if n > 1 else 0


def stateprep_nc(n: int) -> int:
    return 7 * max(0, n - 1)


def stateprep_total(n: int) -> int:
    return 10 * n - ceil_log2(n) - 10 if n > 1 else 0


def multi_control_t_count(c: int) -> int:
    return 7 * (2 * c - 3) if c >= 2 else 0


def outer_flag_ccx_count(m: int, structure: str = "tree") -> int:
    """Exact CCX count of the current outer flag shell, excluding branches.

    For tree, L=ceil(log2 M), q=ceil(M/2)-1:
      C = 2(L-1) + 2(q-popcount(q)) + q - floor(q/2**(L-2)).
    Initial down/final up contribute 2(L-1); transition t contributes
    2*v2(t) and one bounce CCX unless v2(t)=L-2. M<=2 costs zero.
    This counts the emitted standard CCX gates, not the old 4-T return proxy.
    """
    if structure not in ("basic", "opt", "tree"):
        raise ValueError("structure must be basic, opt, or tree")
    if m <= 2 or structure == "basic":
        return 0
    width = ceil_log2(m)
    if structure == "opt":
        q = m - 1
        return 2 * (width - 1) + 2 * (q - q.bit_count())
    q = (m - 1) // 2
    return 2 * (width - 1) + 2 * (q - q.bit_count()) + q - (q >> (width - 2))


@dataclass(frozen=True)
class CostSettings:
    structure: str = "tree"
    fstar: bool | None = None

    def __post_init__(self):
        if self.structure not in ("basic", "opt", "tree"):
            raise ValueError("structure must be basic, opt, or tree")
        if self.fstar is not None and type(self.fstar) is not bool:
            raise ValueError("fstar must be bool or None")
        if self.fstar is None:
            object.__setattr__(self, "fstar", self.structure != "basic")

    def metadata(self):
        return {"cost_model": STATIC_COST_MODEL, **asdict(self),
                "phase_model": "min(parity rotations, clean AND ladder, pi-phase MCZ)",
                "phase_gauge": "most frequent coefficient phase; first column breaks ties",
                "phase_tolerance": PHASE_TOL,
                "generic_rotation_nc_weight": 1,
                "ccx_nc_weight": 7, "stateprep_nc": "7*(padded_length-1)",
                "inner_select": "fixed direct Pauli branches; no placement search",
                "pauli_weight_model": "one primitive per nonzero coefficient"}


@dataclass(frozen=True)
class RowSummary:
    support: int
    alpha_squared: float
    # Counts after removal of ONE common row phase: pi, +/-pi/2, other.
    phases: tuple[int, int, int]


def summarize_row(row, tol: float = 1e-12) -> RowSummary:
    values = np.asarray(row)[np.abs(row) > tol]
    if not len(values):
        return RowSummary(0, 0.0, (0, 0, 0))
    angles = np.angle(values)
    # Same modal gauge as channel_ensemble, with circular phase comparisons.
    # Sorted windows avoid a quadratic all-pairs phase probe for large rows.
    ordered = np.sort(angles)
    extended = np.concatenate((ordered - 2 * np.pi, ordered, ordered + 2 * np.pi))
    counts = (np.searchsorted(extended, angles + PHASE_TOL, side="right")
              - np.searchsorted(extended, angles - PHASE_TOL, side="left"))
    relative = np.angle(np.exp(1j * (angles - angles[int(np.argmax(counts))])))
    distance = np.abs(relative)
    zero = distance <= PHASE_TOL
    pi = np.abs(distance - np.pi) <= PHASE_TOL
    half_pi = np.abs(distance - np.pi / 2) <= PHASE_TOL
    phases = (int(pi.sum()), int(half_pi.sum()), int((~(zero | pi | half_pi)).sum()))
    return RowSummary(len(values), float(np.sum(np.abs(values))) ** 2, phases)


@lru_cache(maxsize=256)
def phase_costs(c: int) -> tuple[int, int, int]:
    """Price pi, +/-pi/2 and generic coefficient phases on c address bits.

    A phase on a conjunction of c bits has a fixed parity expansion with
    2**c-1 rotations of angle theta/2**(c-1). Alternatively compute the AND
    using c-1 clean CCX, apply P(theta), then uncompute. For theta=pi, a
    (c-1)-controlled Z is a third template. Take their static minimum.
    All these templates are identity when the condition is false; clean
    scratch is assumed available, as in the MCX proxy. No circuit is built.
    """
    if c == 0:
        return (0, 0, 0)  # Entire-block global phase.
    # A parity estimate bigger than the linear template cannot be selected.
    parity = (1 << c) - 1 if c < 64 else 14 * (c - 1) + 2
    pi = min(0 if c <= 2 else parity, 14 * (c - 1), multi_control_t_count(c - 1))
    half_pi = min(0 if c == 1 else parity, 14 * (c - 1))
    generic = min(parity, 14 * (c - 1) + 1)
    return (pi, half_pi, generic)


@lru_cache(maxsize=4096)
def _row_cost(support: int, phases: tuple[int, int, int], outer: int, fstar: bool):
    inner = ceil_log2(support)
    length = 1 << inner
    if fstar or outer == 0:
        prep = 2 * stateprep_nc(length)
    else:
        # Whole-block control still charges every PREPARE gate. Generic
        # controlled one-qubit gates need a nonzero weight for outer=1.
        prep = 2 * stateprep_total(length) * multi_control_t_count(outer + 1)
    select = support * multi_control_t_count(outer + inner)
    phase = sum(n * cost for n, cost in zip(phases, phase_costs(outer + inner)))
    return prep, select, phase


def components_from_rows(rows, settings: CostSettings) -> dict:
    rows = [row for row in rows if row.support]
    m = len(rows)
    outer_width = ceil_log2(m)
    branch_controls = outer_width if settings.structure == "basic" else int(m > 1)
    prep = select = phase = 0
    for row in rows:
        p, s, f = _row_cost(row.support, row.phases, branch_controls, settings.fstar)
        prep += p
        select += s
        phase += f
    outer_prep = stateprep_nc(1 << outer_width)
    flag_ccx = outer_flag_ccx_count(m, settings.structure)
    outer_flags = 7 * flag_ccx
    unweighted = outer_prep + outer_flags + prep + select + phase
    normalization = math.fsum(row.alpha_squared for row in rows)
    return {"total": normalization * unweighted, "Lambda": normalization,
            "unweighted_nc": unweighted, "outer_prep": outer_prep,
            "outer_flags": outer_flags, "outer_flag_ccx": flag_ccx,
            "block_prep": prep, "block_select": select, "phase": phase,
            "kraus_count": m, "total_support": sum(row.support for row in rows),
            "outer_controls": outer_width, "branch_controls": branch_controls}


def h_length_components(A, tol: float = 1e-12, *, structure="tree", fstar=None) -> dict:
    """Evaluate equations (1)/(3), or the configured direct/whole-block route."""
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or not np.all(np.isfinite(A)) or not math.isfinite(tol) or tol < 0:
        raise ValueError("Expected a finite coefficient matrix and nonnegative tolerance")
    settings = CostSettings(structure, fstar)
    return components_from_rows([summarize_row(row, tol) for row in A], settings)
