"""Shared settings and counting interface for the four-RQ evaluation.

These settings describe the current executable backend. F* controls only
SELECT; calibration of the rewrite cost model is still pending. Using the
interface alone does not certify a formal rerun as complete.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import time


PROTOCOL_VERSION = "oopsla-four-rq-matrix-order-anf-pipeline-nam-opt0-20261010"
DEFAULT_TRANSPILE_SEED = 20261009
EXTERNAL_NAM_OPTIMIZERS = ("voqc", "feynman")
GATE_BASES = {"nam": ("rz", "h", "x", "cx"), "ibm": ("u1", "u2", "u3", "cx")}
OUTER_STRUCTURES = {"basic": "independent-controls", "opt": "ctrl-line", "tree": "tree"}
FORMAL_PREREQUISITES = (
    "F* rewrite cost-model calibration (legacy H_length still used)",
    "Final per-RQ instance lists and runtime budgets",
    "PennyLane QSVT operators, polynomials and shared phase sequences",
)


@dataclass(frozen=True)
class RewriteSettings:
    method: str = "gatecost_rewrite_search"
    strategy: str = "beam"
    beam_width: int = 8
    max_steps: int = 50
    tol: float = 1e-12
    keep_neutral: bool = False
    max_neutral_candidates: int = 3
    skip_disjoint_pairs: bool = True


REWRITE_DEFAULTS = RewriteSettings()


def transpile_channel_circuit(circuits, *args, **kwargs):
    """Lower an operator on arbitrary inputs, including externally supplied states.

    Qiskit's default can borrow untouched system wires as clean ancillas. That
    is valid for all-zero state preparation, but changes a channel's action on
    general inputs. Explicit True remains available for historical diagnostics.
    """
    from qiskit import transpile
    kwargs.setdefault("qubits_initially_zero", False)
    return transpile(circuits, *args, **kwargs)


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    rewrite: bool
    flat: bool
    order: bool
    structure: str
    opt: str

    @property
    def slug(self):
        return self.name.lower().replace("+", "_").replace(" ", "_").replace("__", "_")


CONFIGS = (
    ConfigSpec("Basic", False, False, False, "basic", "No"),
    ConfigSpec("Rewrite only", True, False, False, "basic", "No"),
    ConfigSpec("Flat only", False, True, False, "tree", "No"),
    ConfigSpec("Order only", False, False, True, "basic", "Matrix-order"),
    ConfigSpec("Rewrite + Flat", True, True, False, "tree", "No"),
    ConfigSpec("Rewrite + Order", True, False, True, "basic", "Matrix-order"),
    ConfigSpec("Flat + Order", False, True, True, "tree", "Matrix-order"),
    ConfigSpec("Rewrite + Flat + Order", True, True, True, "tree", "Matrix-order"),
)
CONFIG_BY_NAME = {cfg.name: cfg for cfg in CONFIGS}
CONFIG_BY_SLUG = {cfg.slug: cfg for cfg in CONFIGS}
RQ1_PIPELINES = {
    "Raw": {"rewrite": False, "inner": "No", "structure": "basic", "fstar": False},
    "Qrisp": {"rewrite": False, "inner": "Qrisp", "structure": "tree", "fstar": True},
    "Ours": {"rewrite": True, "inner": "Matrix-order", "structure": "tree", "fstar": True},
}


@dataclass(frozen=True, kw_only=True)
class RandomChannelInput:
    """Public input schema: size is always a qubit count, never a dimension."""
    kraus_count: int
    num_qubits: int
    first_kraus_terms: int
    other_kraus_terms: int
    seed: int

    def __post_init__(self):
        for key in ("kraus_count", "num_qubits", "first_kraus_terms", "other_kraus_terms"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if max(self.first_kraus_terms, self.other_kraus_terms) > 4 ** self.num_qubits - 1:
            raise ValueError("The random generator samples distinct nonidentity Pauli terms.")

    @property
    def hilbert_dim(self):
        return 1 << self.num_qubits

    def as_dict(self):
        return {**asdict(self), "hilbert_dim": self.hilbert_dim}


def source_identity():
    root = Path(__file__).resolve().parent
    files = ("experiment_protocol.py", "block_encoding_new.py", "block_encoding_new_fixed.py",
             "mobius_residual_fixed.py", "block_encoding.py", "placement_search.py",
             "channel_IR.py", "channelIR_gatecost_rewrite_search.py", "channel_LCU.py", "subroutine.py",
             "qrisp_channel_adapter.py", "rq1_pipeline.py", "matrix_order_pipeline.py")
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in files}


def cache_identity(parameters):
    """Include source and resolved settings so old results cannot impersonate reruns."""
    payload = {"protocol": PROTOCOL_VERSION, "parameters": parameters, "sources": source_identity()}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run_rewrite_search(channel, *, strategy="beam", beam_width=8, max_steps=50,
                       tol=1e-12, verbose=False, keep_neutral=False,
                       max_neutral_candidates=3, skip_disjoint_pairs=True):
    """Common R entrypoint; retain explicitly requested historical diagnostics."""
    if strategy == "greedy_incremental":
        return channel.rewrite_search(strategy=strategy, beam_width=beam_width,
                                      max_steps=max_steps, tol=tol, verbose=verbose)
    if strategy not in ("beam", "greedy"):
        raise ValueError(f"Unsupported rewrite strategy: {strategy}")
    return channel.gatecost_rewrite_search(
        strategy=strategy, beam_width=beam_width, max_steps=max_steps, tol=tol,
        verbose=verbose, keep_neutral=keep_neutral,
        max_neutral_candidates=max_neutral_candidates, skip_disjoint_pairs=skip_disjoint_pairs,
    )


def settings(rq, *, basis="nam", seed_transpiler=DEFAULT_TRANSPILE_SEED):
    from matrix_order_pipeline import resolve_options, options_dict
    if rq not in ("RQ1", "RQ2", "RQ3.1", "RQ3.2", "RQ4"):
        raise ValueError(f"Unknown research question: {rq}")
    if basis not in GATE_BASES:
        raise ValueError(f"Unknown gate basis: {basis}")
    return {
        "protocol": PROTOCOL_VERSION, "rq": rq, "basis": basis,
        "basis_gates": list(GATE_BASES[basis]),
        "lowering_optimization_level": 0,
        "downstream_optimizer": None,
        "downstream_optimization_level": None,
        "seed_transpiler": seed_transpiler,
        "qubits_initially_zero": False,
        "rewrite": asdict(REWRITE_DEFAULTS),
        "outer_structures": dict(OUTER_STRUCTURES),
        "supplementary_external_optimizers": list(EXTERNAL_NAM_OPTIMIZERS),
        "current_outer_control_scope": {"basic": "whole-block", "opt": "select-only", "tree": "select-only"},
        "matrix_order_fstar_lowering": "add outer controls to symbolic residuals, then expand_large_controlled_gates(if_optimize=True)",
        "matrix_order_options": options_dict(resolve_options()),
        "fstar_design_status": "implemented; rewrite cost-model calibration pending",
        "canonical_input": "shared A_can; retain A_raw and A_R separately",
        "canonical_zero_policy": "exact; report any numerical cleanup separately",
        "proportional_merge_tolerance": 1e-12,
        "metrics": ["G", "Lambda", "Lambda*G", "W", "compile_time_s"],
        "formal_prerequisites": list(FORMAL_PREREQUISITES),
    }


def lower_and_count(circuit, *, rq, basis="nam", seed_transpiler=DEFAULT_TRANSPILE_SEED):
    """Apply only basis lowering (opt0), uniformly to every pipeline.

    Compilation time here covers lowering only. The caller must also record
    canonicalization, rewriting, placement and assembly before reporting total
    compilation time. Lambda comes from the actual backend, not gate counts.
    """
    transpile = transpile_channel_circuit
    policy = settings(rq, basis=basis, seed_transpiler=seed_transpiler)
    started = time.perf_counter()
    lowered = transpile(circuit, basis_gates=list(GATE_BASES[basis]), optimization_level=0,
                        seed_transpiler=seed_transpiler)
    lowering_seconds = time.perf_counter() - started
    final = lowered
    optimizer_seconds = 0.0
    counts = {str(gate): int(count) for gate, count in final.count_ops().items()}
    unexpected = set(counts) - set(GATE_BASES[basis]) - {"barrier", "reset"}
    if unexpected:
        raise ValueError(f"Circuit contains uncounted operations: {sorted(unexpected)}")
    packages = {}
    for package in ("qiskit", "qrisp", "numpy", "pennylane"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    metrics = {
        "G": sum(counts.get(gate, 0) for gate in GATE_BASES[basis]),
        "W": final.num_qubits, "counts": counts,
        "lowering_time_s": lowering_seconds, "optimizer_time_s": optimizer_seconds,
        "settings": policy, "versions": packages, "sources": source_identity(),
        "input_circuit_metadata": circuit.metadata,
    }
    return final, metrics
