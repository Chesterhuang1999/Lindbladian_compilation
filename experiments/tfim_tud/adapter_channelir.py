"""Thin adapter to the existing complete Pauli block-encoding backend."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qiskit import QuantumCircuit
from block_encoding_new import BlockEncoding
from channel_IR import Matrixsum, PauliAtom


@dataclass
class BlockEncodingResult:
    circuit: QuantumCircuit
    normalization: float
    system_qubits: tuple[int, ...]
    success_qubits: tuple[int, ...]
    clean_work_qubits: tuple[int, ...]
    metadata: dict


def source_record(function):
    path = Path(inspect.getsourcefile(function)).resolve()
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "path": str(path),
        "line": inspect.getsourcelines(function)[1],
        "signature": str(inspect.signature(function)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def canonical_terms(terms, n):
    merged = {}
    for label, coefficient in terms:
        if len(label) != n or any(p not in "IXYZ" for p in label):
            raise ValueError(f"Invalid Pauli label: {label!r}")
        coefficient = complex(coefficient)
        if not (math.isfinite(coefficient.real) and math.isfinite(coefficient.imag)):
            raise ValueError("Nonfinite coefficient")
        merged[label] = merged.get(label, 0j) + coefficient
    return [(label, value) for label, value in sorted(merged.items()) if value != 0]


def code_provenance():
    if (ROOT / '.git').exists():
        return {'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'commit_source': 'local_git_HEAD'}
    manifest_path = ROOT / 'remote_source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    commit = manifest['local_source_base_commit']
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Deployment manifest lacks a valid local source base commit')
    return {'commit': commit, 'commit_source': 'deployment_manifest_local_source_base_commit',
            'deployment_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


def compile_existing_pauli_block(terms, n, *, placement_enabled=True, backend_class=None):
    terms = canonical_terms(terms, n)
    if not terms:
        raise ValueError("Zero branch must be handled before block compilation")
    atoms = []
    for label, coefficient in terms:
        magnitude = abs(coefficient)
        # Matrixsum expects positive weights and phases in atoms. Avoid its
        # complex-coefficient constructor path, which leaves tuple weights stale.
        atom = PauliAtom(label)
        atom.phase = coefficient / magnitude
        atoms.append((atom, magnitude))
    matrixsum = Matrixsum(atoms)
    selected_backend = BlockEncoding if backend_class is None else backend_class
    backend = selected_backend(matrixsum)
    opt = "Matrix-order" if placement_enabled else "No"
    circuit = backend.circuit(opt=opt)
    roles = {reg.name: tuple(circuit.find_bit(q).index for q in reg)
             for reg in circuit.qregs}
    if set(roles) - {"sys", "ctrl", "anc"} or len(roles.get("sys", ())) != n:
        raise RuntimeError(f"Unknown backend register contract: {roles}")
    system = roles["sys"]
    success = roles.get("ctrl", ())
    work = roles.get("anc", ())
    if set(system + success + work) != set(range(circuit.num_qubits)):
        raise RuntimeError("Incomplete or overlapping register roles")
    scale = float(sum(abs(value) for value in backend.coeff_list))
    if placement_enabled:
        prepare_scale = float(sum(abs(value) for value in backend.coeff_list_ordered))
        if not math.isclose(scale, prepare_scale, rel_tol=1e-14, abs_tol=1e-14):
            raise RuntimeError("Backend PREPARE scale disagrees with input weights")
    else:
        prepare_scale = scale
    return BlockEncodingResult(
        circuit, scale, system, success, work,
        {
            "backend": source_record(selected_backend.circuit),
            "prepare": source_record(selected_backend.mulplex_B_opt_order),
            "placement": source_record(selected_backend.find_optimal_order_matrices),
            **code_provenance(),
            "parameters": {"opt": opt, "placement_enabled": placement_enabled},
            "normalization_source": "sum(abs(backend.coeff_list)); cross-checked against PREPARE coeff_list_ordered",
            "prepare_scale": prepare_scale,
            "backend_succ_prob_field": float(backend.succ_prob),
            "succ_prob_field_semantics": "L1 scale, not success probability",
            "phase_gauge": "input coefficient phases preserved; no fitted global phase",
            "registers": roles,
            "success_contract": "input ctrl/anc zero; output ctrl/anc zero; system order is sys[0..n-1]",
            "clean_work_contract": "anc returns zero for all system inputs with ctrl initially zero, including failure branches",
            "selected_crit": getattr(backend, "selected_crit", None),
            "placement_search_debug": getattr(backend, "placement_search_debug", {}),
            "width_candidate_costs": getattr(backend, "width_candidate_costs", []),
            "coefficient_pruning": "none; exact duplicate merge and exact zero removal only",
        },
    )
