#!/usr/bin/env python3
"""TFIM_4 A0 probe using the canonical Tower template_selector_unit shape.

Tower sees only X/I masks: every non-identity Pauli character is emitted as an
X target.  The real Pauli string and phase are kept in a sidecar and restored in
the Qiskit circuit with H/S basis changes and address-state phases.
"""

from __future__ import annotations

import argparse
import cmath
import copy
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from baseline_common import count_qasm_file  # type: ignore  # noqa: E402
from block_encoding_new import BlockEncoding as NewBlockEncoding  # type: ignore  # noqa: E402
from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel  # type: ignore  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "Baseline_results" / "spire_tfim4_a0_template_selector_unit_v1"
DEFAULT_TOWER = Path("/private/tmp/spire-src-local/spire-src/tower")
IBM_BASIS = ("u1", "u2", "u3", "cx")
NAM_BASIS = ("h", "x", "rz", "cx")


@dataclass(frozen=True)
class TermData:
    index: int
    ctrl_state: str
    pauli: str
    x_mask: str
    atom_phase: complex
    coefficient: complex


@dataclass(frozen=True)
class DotQCGate:
    target: int
    controls: tuple[int, ...]


def complex_payload(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def term_payload(term: TermData) -> dict[str, Any]:
    return {
        "index": int(term.index),
        "ctrl_state": term.ctrl_state,
        "pauli": term.pauli,
        "x_mask": term.x_mask,
        "atom_phase": complex_payload(term.atom_phase),
        "atom_phase_angle": float(cmath.phase(term.atom_phase)),
        "coefficient": complex_payload(term.coefficient),
        "coefficient_abs": float(abs(term.coefficient)),
    }


def estimate_template_t_count(terms: list[TermData], ctrl_size: int) -> dict[str, Any]:
    """Estimate T-count with the markdown surface template model.

    For k >= 2 and w X targets:
      template_selector_unit surface cost = 14(k - 1) + 21w
      repeated direct C^kX cost        = 7w(2k - 3)

    Phase-only branches are recorded separately because they are sidecar phase
    restoration, not part of the controlled-PX template body.
    """
    rows: list[dict[str, Any]] = []
    surface_total = 0
    direct_repeated_total = 0
    core_patched_total = 0
    phase_only_count = 0
    active_branch_count = 0
    x_target_total = 0
    x_target_histogram: dict[str, int] = {}
    for term in terms:
        w = term.x_mask.count("X")
        x_target_histogram[str(w)] = x_target_histogram.get(str(w), 0) + 1
        x_target_total += w
        if w == 0:
            surface = 0
            direct = 0
            core_patched = 0
            phase_only_count += int(abs(cmath.phase(term.atom_phase)) > 1e-12)
        else:
            active_branch_count += 1
            surface = 14 * (ctrl_size - 1) + 21 * w
            direct = 7 * w * (2 * ctrl_size - 3)
            core_patched = 14 * (ctrl_size - 1)
        surface_total += surface
        direct_repeated_total += direct
        core_patched_total += core_patched
        rows.append(
            {
                "index": int(term.index),
                "ctrl_state": term.ctrl_state,
                "pauli": term.pauli,
                "x_mask": term.x_mask,
                "x_target_count": int(w),
                "template_surface_t": int(surface),
                "repeated_direct_cnx_t": int(direct),
                "core_patched_selector_t": int(core_patched),
                "phase_angle": float(cmath.phase(term.atom_phase)),
            }
        )

    return {
        "model": "tower_controlled_px_template.md surface model",
        "ctrl_size": int(ctrl_size),
        "term_count": int(len(terms)),
        "active_x_mask_branch_count": int(active_branch_count),
        "x_target_total": int(x_target_total),
        "x_target_count_histogram": {key: x_target_histogram[key] for key in sorted(x_target_histogram, key=int)},
        "paper_mcx_formula": "MCX with c>=2 controls -> 7*(2*(c-2)+1) T; Toffoli -> 7 T; X/CNOT/Clifford -> 0 T",
        "formula_template_surface": "14*(k-1) + 21*w for each controlled-PX branch with w>0",
        "formula_repeated_direct": "7*w*(2*k-3) for repeated direct C^kX branches",
        "formula_core_patched": "14*(k-1) for each controlled-PX branch with w>0; requires core-style direct XOR target lowering",
        "template_surface_t_total": int(surface_total),
        "repeated_direct_cnx_t_total": int(direct_repeated_total),
        "core_patched_selector_t_total": int(core_patched_total),
        "phase_only_branch_count": int(phase_only_count),
        "phase_restoration_t": "not included; branch phase uses arbitrary controlled phase in Qiskit/NAM/IBM counts",
        "per_term": rows,
    }


def build_tfim4_a0_terms() -> tuple[list[TermData], dict[str, Any]]:
    kraus_ops, model_info = build_periodic_tfim_channel(4, 0.1)
    matrixsum = kraus_ops[0]
    ctrl_size = int(math.ceil(math.log2(len(matrixsum.instances))))
    terms: list[TermData] = []
    for index, (atom, coeff) in enumerate(matrixsum.instances):
        pauli = str(atom.expr)
        atom_phase = complex(atom.phase)
        coefficient = complex(coeff)
        terms.append(
            TermData(
                index=index,
                ctrl_state=format(index, f"0{ctrl_size}b"),
                pauli=pauli,
                x_mask="".join("I" if char == "I" else "X" for char in pauli),
                atom_phase=atom_phase,
                coefficient=coefficient,
            )
        )
    metadata = {
        "model_info": model_info,
        "matrixsum_size": int(matrixsum.size),
        "term_count": int(len(matrixsum.instances)),
        "ctrl_size": int(ctrl_size),
        "pauli_norm": float(matrixsum.pauli_norm()),
        "target_convention": "template: pauli[j] maps to system q_j",
    }
    return terms, metadata


def emit_and_tree(lines: list[str], literals: list[str], indent: str, prefix: str) -> str:
    current = list(literals)
    level = 0
    while len(current) > 1:
        nxt: list[str] = []
        pairs = [(idx, current[idx], current[idx + 1]) for idx in range(0, len(current) - 1, 2)]
        has_carry = len(current) % 2 == 1
        final_pair = len(pairs) == 1 and not has_carry
        for pair_idx, lhs, rhs in pairs:
            out = f"{prefix}_sel" if final_pair else f"{prefix}_s_{level}_{pair_idx // 2}"
            lines.append(f"{indent}let {out} <- {lhs} && {rhs};")
            nxt.append(out)
        if has_carry:
            nxt.append(current[-1])
        current = nxt
        level += 1
    return current[0]


def append_template_block_source(
    lines: list[str],
    *,
    term: TermData,
    controls: list[str],
    targets: list[str],
) -> None:
    x_targets = [
        (target, f"t{term.index}_n{target}")
        for target, mask_char in zip(targets, term.x_mask, strict=True)
        if mask_char == "X"
    ]
    if not x_targets:
        lines.append(f"  /* addr {term.ctrl_state}: {term.pauli}, phase-only/identity in sidecar */")
        return

    prefix = f"t{term.index}"
    lines.extend(
        [
            f"  /* addr {term.ctrl_state}: real {term.pauli}, Tower X-mask {term.x_mask} */",
            "  with {",
        ]
    )
    literals: list[str] = []
    for control, bit in zip(controls, term.ctrl_state, strict=True):
        if bit == "1":
            literals.append(control)
        else:
            neg = f"{prefix}_n{control}"
            lines.append(f"    let {neg} <- not {control};")
            literals.append(neg)

    selector = emit_and_tree(lines, literals, "    ", prefix)
    lines.append("")
    for _target, complement in x_targets:
        lines.append(f"    let {complement} <- not {_target};")

    lines.extend(["  } do {", f"    if {selector} {{"])
    for target, complement in x_targets:
        lines.append(f"      {target} <-> {complement};")
    lines.extend(["    }", "  }", ""])


def tower_source(function_name: str, terms: list[TermData], ctrl_size: int, sys_size: int) -> str:
    controls = [f"c{i}" for i in range(ctrl_size)]
    targets = [f"q{i}" for i in range(sys_size)]
    params = ", ".join(f"{name}: bool" for name in [*controls, *targets])
    lines = [
        "/*",
        "  TFIM_4 A0 SELECT probe, canonical template_selector_unit form.",
        "  Tower sees only X/I masks.  Real Pauli strings/phases are restored",
        "  from the generated sidecar.",
        "*/",
        f"fun {function_name}({params}) -> unit {{",
        "",
    ]
    for term in terms:
        append_template_block_source(lines, term=term, controls=controls, targets=targets)
    lines.extend(["  return ();", "}", ""])
    return "\n".join(lines)


def run_tower(tower_bin: Path, source: Path, function_name: str, *, optimize_lir: bool) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(tower_bin),
        "-c",
        "--word_size",
        "1",
        "--mem_size",
        "0",
        "-b",
        f"{function_name}:1",
        "--dotQC",
    ]
    if optimize_lir:
        cmd.append("--optimize_lir")
    cmd.append(str(source))
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def parse_dotqc(text: str) -> tuple[list[str], list[DotQCGate]]:
    wires: list[str] = []
    gates: list[DotQCGate] = []
    in_body = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(".v "):
            wires = line.split()[1:]
            continue
        if line == "BEGIN":
            in_body = True
            continue
        if line == "END":
            in_body = False
            continue
        if not in_body:
            continue
        parts = line.split()
        if parts[0] == "X":
            gates.append(DotQCGate(int(parts[1][1:]), ()))
        elif parts[0] == "tof":
            wire_ids = tuple(int(token[1:]) for token in parts[1:])
            gates.append(DotQCGate(wire_ids[-1], wire_ids[:-1]))
        else:
            raise ValueError(f"unsupported dotQC line: {line}")
    return wires, gates


def dotqc_stats(text: str) -> dict[str, Any]:
    wires, gates = parse_dotqc(text)
    hist: dict[str, int] = {}
    for gate in gates:
        key = "X/1" if not gate.controls else f"tof/{len(gate.controls) + 1}"
        hist[key] = hist.get(key, 0) + 1
    return {"wire_count": len(wires), "gate_count": len(gates), "gate_histogram": hist}


def address_phase(qc: QuantumCircuit, controls: list[int], ctrl_state: str, angle: float) -> None:
    if abs(angle) < 1e-12:
        return
    zero_controls = [wire for wire, bit in zip(controls, ctrl_state, strict=True) if bit == "0"]
    for wire in zero_controls:
        qc.x(wire)
    if len(controls) == 1:
        qc.p(angle, controls[0])
    else:
        qc.mcp(angle, controls[:-1], controls[-1])
    for wire in reversed(zero_controls):
        qc.x(wire)


def pre_basis(qc: QuantumCircuit, wire: int, pauli: str) -> None:
    if pauli == "Y":
        qc.sdg(wire)
    elif pauli == "Z":
        qc.h(wire)


def post_basis(qc: QuantumCircuit, wire: int, pauli: str) -> None:
    if pauli == "Y":
        qc.s(wire)
    elif pauli == "Z":
        qc.h(wire)


def controlled_x_on_state(qc: QuantumCircuit, controls: list[int], target: int, ctrl_state: str) -> None:
    zero_controls = [wire for wire, bit in zip(controls, ctrl_state, strict=True) if bit == "0"]
    for wire in zero_controls:
        qc.x(wire)
    qc.mcx(controls, target)
    for wire in reversed(zero_controls):
        qc.x(wire)


def scratch_need(term: TermData, ctrl_size: int) -> int:
    if "X" not in term.x_mask:
        return 0
    neg_count = term.ctrl_state.count("0")
    and_count = ctrl_size - 1
    complement_count = term.x_mask.count("X")
    return neg_count + and_count + complement_count


def allocate_template_wires(
    *,
    term: TermData,
    controls: list[int],
    sys_wires: list[int],
    scratch_base: int,
) -> tuple[dict[str, int], list[tuple[int, int, int]], list[tuple[int, int]]]:
    cursor = scratch_base
    literal_wires: list[int] = []
    neg_uncompute: list[tuple[int, int]] = []
    for control, bit in zip(controls, term.ctrl_state, strict=True):
        if bit == "1":
            literal_wires.append(control)
        else:
            anc = cursor
            cursor += 1
            literal_wires.append(anc)
            neg_uncompute.append((control, anc))

    and_gates: list[tuple[int, int, int]] = []
    current = list(literal_wires)
    while len(current) > 1:
        nxt: list[int] = []
        for idx in range(0, len(current), 2):
            if idx + 1 == len(current):
                nxt.append(current[idx])
            else:
                out = cursor
                cursor += 1
                and_gates.append((current[idx], current[idx + 1], out))
                nxt.append(out)
        current = nxt
    selector = current[0]

    complements: dict[str, int] = {}
    for sys_index, mask_char in enumerate(term.x_mask):
        if mask_char == "X":
            complements[f"q{sys_index}"] = cursor
            cursor += 1

    return {"selector": selector, **complements}, and_gates, neg_uncompute


def apply_template_x_mask(
    qc: QuantumCircuit,
    *,
    term: TermData,
    controls: list[int],
    sys_wires: list[int],
    scratch_base: int,
) -> None:
    if "X" not in term.x_mask:
        return

    allocated, and_gates, neg_uncompute = allocate_template_wires(
        term=term,
        controls=controls,
        sys_wires=sys_wires,
        scratch_base=scratch_base,
    )
    selector = allocated["selector"]

    for control, anc in neg_uncompute:
        qc.x(anc)
        qc.cx(control, anc)
    for lhs, rhs, out in and_gates:
        qc.ccx(lhs, rhs, out)

    complement_order: list[tuple[int, int]] = []
    for sys_index, mask_char in enumerate(term.x_mask):
        if mask_char == "X":
            target = sys_wires[sys_index]
            complement = allocated[f"q{sys_index}"]
            qc.x(complement)
            qc.cx(target, complement)
            complement_order.append((target, complement))

    for target, complement in complement_order:
        qc.ccx(selector, complement, target)
        qc.ccx(selector, target, complement)
        qc.ccx(selector, complement, target)

    for target, complement in reversed(complement_order):
        qc.cx(target, complement)
        qc.x(complement)
    for lhs, rhs, out in reversed(and_gates):
        qc.ccx(lhs, rhs, out)
    for control, anc in reversed(neg_uncompute):
        qc.cx(control, anc)
        qc.x(anc)


def build_direct_reference(terms: list[TermData], ctrl_size: int, sys_size: int) -> QuantumCircuit:
    qc = QuantumCircuit(ctrl_size + sys_size, name="tfim4_a0_direct_reference")
    controls = list(range(ctrl_size))
    sys_wires = list(range(ctrl_size, ctrl_size + sys_size))
    for term in terms:
        address_phase(qc, controls, term.ctrl_state, cmath.phase(term.atom_phase))
        for sys_index, pauli in enumerate(term.pauli):
            if pauli == "I":
                continue
            target = sys_wires[sys_index]
            pre_basis(qc, target, pauli)
            controlled_x_on_state(qc, controls, target, term.ctrl_state)
            post_basis(qc, target, pauli)
    return qc


def build_template_restored(terms: list[TermData], ctrl_size: int, sys_size: int) -> QuantumCircuit:
    max_scratch = max((scratch_need(term, ctrl_size) for term in terms), default=0)
    scratch_base = ctrl_size + sys_size
    qc = QuantumCircuit(ctrl_size + sys_size + max_scratch, name="tfim4_a0_template_restored")
    controls = list(range(ctrl_size))
    sys_wires = list(range(ctrl_size, ctrl_size + sys_size))
    for term in terms:
        address_phase(qc, controls, term.ctrl_state, cmath.phase(term.atom_phase))
        for sys_index, pauli in enumerate(term.pauli):
            if pauli != "I":
                pre_basis(qc, sys_wires[sys_index], pauli)
        apply_template_x_mask(
            qc,
            term=term,
            controls=controls,
            sys_wires=sys_wires,
            scratch_base=scratch_base,
        )
        for sys_index in reversed(range(sys_size)):
            pauli = term.pauli[sys_index]
            if pauli != "I":
                post_basis(qc, sys_wires[sys_index], pauli)
    return qc


def qiskit_metrics(qc: QuantumCircuit) -> dict[str, Any]:
    return {
        "num_qubits": int(qc.num_qubits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "ops": {str(name): int(count) for name, count in qc.count_ops().items()},
    }


def write_qasm(qc: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        qasm2.dump(qc, handle)


def lower_and_count(qc: QuantumCircuit, path: Path, *, basis_name: str) -> dict[str, Any]:
    if basis_name == "ibm":
        basis = IBM_BASIS
        gate_set = "ibm"
    elif basis_name == "nam":
        basis = NAM_BASIS
        gate_set = "nam"
    else:
        raise ValueError(basis_name)
    started = time.perf_counter()
    lowered = transpile(qc, basis_gates=list(basis), optimization_level=0)
    elapsed = time.perf_counter() - started
    write_qasm(lowered, path)
    return {
        "basis": basis_name,
        "basis_gates": list(basis),
        "qasm": str(path),
        "transpile_time_s": float(elapsed),
        "qiskit_metrics": qiskit_metrics(lowered),
        "count": count_qasm_file(path, gate_set=gate_set),
    }


def semantic_action(term: TermData, bits: list[int], ctrl_size: int, sys_size: int) -> tuple[list[int], complex] | None:
    ctrl_bits = "".join(str(bit) for bit in bits[:ctrl_size])
    if ctrl_bits != term.ctrl_state:
        return None

    out = list(bits)
    phase = term.atom_phase
    for sys_index, pauli in enumerate(term.pauli):
        wire = ctrl_size + sys_index
        bit = out[wire]
        if pauli == "I":
            continue
        if pauli == "X":
            out[wire] ^= 1
        elif pauli == "Y":
            phase *= 1j if bit == 0 else -1j
            out[wire] ^= 1
        elif pauli == "Z":
            if bit == 1:
                phase *= -1
        else:
            raise ValueError(f"unsupported Pauli: {pauli}")
    return out, phase


def expected_sidecar_action(terms: list[TermData], basis: int, ctrl_size: int, sys_size: int) -> tuple[int, complex]:
    logical_width = ctrl_size + sys_size
    bits = [(basis >> index) & 1 for index in range(logical_width)]
    for term in terms:
        result = semantic_action(term, bits, ctrl_size, sys_size)
        if result is not None:
            out_bits, phase = result
            out_index = sum(bit << index for index, bit in enumerate(out_bits))
            return out_index, phase
    return basis, 1.0 + 0.0j


def statevector_semantic_equivalence(
    qc: QuantumCircuit,
    terms: list[TermData],
    *,
    ctrl_size: int,
    sys_size: int,
) -> dict[str, Any]:
    logical_width = ctrl_size + sys_size
    physical_width = qc.num_qubits
    dim = 1 << physical_width
    max_expected_error = 0.0
    max_leakage = 0.0
    mismatches: list[dict[str, Any]] = []

    for basis in range(1 << logical_width):
        expected_logical, expected_phase = expected_sidecar_action(terms, basis, ctrl_size, sys_size)
        expected_physical = expected_logical
        state = Statevector.from_int(basis, dim).evolve(qc).data
        expected_error = abs(state[expected_physical] - expected_phase)
        leakage_sq = max(0.0, float(np.vdot(state, state).real - abs(state[expected_physical]) ** 2))
        leakage = math.sqrt(leakage_sq)
        max_expected_error = max(max_expected_error, float(expected_error))
        max_leakage = max(max_leakage, float(leakage))
        if expected_error > 1e-8 or leakage > 1e-8:
            mismatches.append(
                {
                    "input_basis": int(basis),
                    "expected_basis": int(expected_physical),
                    "expected_phase": complex_payload(expected_phase),
                    "observed_expected_amplitude": complex_payload(complex(state[expected_physical])),
                    "expected_error": float(expected_error),
                    "leakage": float(leakage),
                }
            )
            if len(mismatches) >= 8:
                break

    return {
        "logical_width": int(logical_width),
        "physical_width": int(physical_width),
        "checked_basis_states": int(1 << logical_width),
        "max_expected_error": float(max_expected_error),
        "max_leakage": float(max_leakage),
        "passed": bool(max_expected_error < 1e-8 and max_leakage < 1e-8 and not mismatches),
        "mismatches": mismatches,
    }


def build_project_select(ms: Any, opt: str) -> QuantumCircuit:
    return NewBlockEncoding(copy.deepcopy(ms)).select_circuit(opt=opt)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    terms, metadata = build_tfim4_a0_terms()
    ctrl_size = int(metadata["ctrl_size"])
    sys_size = int(metadata["matrixsum_size"])
    function_name = "tfim4_a0_template_selector_unit"

    source_path = out_dir / f"{function_name}.twr"
    source_path.write_text(tower_source(function_name, terms, ctrl_size, sys_size), encoding="utf-8")
    sidecar_path = out_dir / f"{function_name}.sidecar.json"
    sidecar_path.write_text(
        json.dumps(
            {
                **metadata,
                "terms": [term_payload(term) for term in terms],
                "tower_policy": "I stays I; X/Y/Z are emitted as X targets in Tower",
                "restore_policy": {
                    "X": "no basis change",
                    "Y": "pre Sdg, post S around X-mask template",
                    "Z": "pre H, post H around X-mask template",
                    "phase": "address-state mcp(angle(atom.phase))",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "experiment": "tfim4_a0_template_selector_unit",
        "out_dir": str(out_dir),
        "tower_bin": str(args.tower_bin),
        "source": str(source_path),
        "sidecar": str(sidecar_path),
        "metadata": metadata,
        "t_count_estimate": estimate_template_t_count(terms, ctrl_size),
        "tower_variants": {},
        "circuits": {},
    }

    for label, optimize in (("orig", False), ("optimized_lir", True)):
        proc = run_tower(Path(args.tower_bin), source_path, function_name, optimize_lir=optimize)
        qc_path = out_dir / f"{function_name}.{label}.qc"
        stderr_path = out_dir / f"{function_name}.{label}.stderr.txt"
        qc_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        row: dict[str, Any] = {
            "returncode": int(proc.returncode),
            "qc": str(qc_path),
            "stderr": str(stderr_path),
        }
        if proc.returncode == 0:
            row["dotqc"] = dotqc_stats(proc.stdout)
            tof3_count = int(row["dotqc"]["gate_histogram"].get("tof/3", 0))
            row["markdown_t_model_from_dotqc"] = {
                "tof3_count": tof3_count,
                "t_per_tof3": 7,
                "t_total": int(7 * tof3_count),
            }
        summary["tower_variants"][label] = row

    direct = build_direct_reference(terms, ctrl_size, sys_size)
    template = build_template_restored(terms, ctrl_size, sys_size)
    direct_qasm = out_dir / "tfim4_a0_direct_reference.qasm"
    template_qasm = out_dir / "tfim4_a0_template_restored.qasm"
    write_qasm(direct, direct_qasm)
    write_qasm(template, template_qasm)

    summary["circuits"]["direct_reference"] = {
        "description": "serial controlled Pauli terms with the same sidecar target convention",
        "qasm": str(direct_qasm),
        "qiskit_metrics": qiskit_metrics(direct),
        "ibm": lower_and_count(direct, out_dir / "tfim4_a0_direct_reference.ibm_opt0.qasm", basis_name="ibm"),
        "nam": lower_and_count(direct, out_dir / "tfim4_a0_direct_reference.nam_opt0.qasm", basis_name="nam"),
    }
    summary["circuits"]["template_restored"] = {
        "description": "template-selector-unit X-mask circuit with H/S/phase sidecar restoration",
        "qasm": str(template_qasm),
        "qiskit_metrics": qiskit_metrics(template),
        "ibm": lower_and_count(template, out_dir / "tfim4_a0_template_restored.ibm_opt0.qasm", basis_name="ibm"),
        "nam": lower_and_count(template, out_dir / "tfim4_a0_template_restored.nam_opt0.qasm", basis_name="nam"),
    }
    summary["equivalence"] = {
        "template_restored_vs_sidecar_semantics": statevector_semantic_equivalence(
            template,
            terms,
            ctrl_size=ctrl_size,
            sys_size=sys_size,
        )
    }

    # Keep the project-local SELECT counts for orientation.  These use the
    # repository's Qiskit/Pauli endian convention and are not used as the
    # sidecar equivalence oracle above.
    kraus_ops, _model_info = build_periodic_tfim_channel(4, 0.1)
    ms = kraus_ops[0]
    for opt in ("No", "Ctrl-line", "Tree", "Matrix-order"):
        try:
            project_qc = build_project_select(ms, opt)
            key = f"project_select_{opt}"
            summary["circuits"][key] = {
                "description": f"NewBlockEncoding(A0).select_circuit(opt={opt!r}) orientation count only",
                "qiskit_metrics": qiskit_metrics(project_qc),
                "ibm": lower_and_count(project_qc, out_dir / f"tfim4_a0_project_select_{opt.lower().replace('-', '_')}.ibm_opt0.qasm", basis_name="ibm"),
                "nam": lower_and_count(project_qc, out_dir / f"tfim4_a0_project_select_{opt.lower().replace('-', '_')}.nam_opt0.qasm", basis_name="nam"),
            }
        except Exception as exc:
            summary["circuits"][f"project_select_{opt}"] = {"status": "error", "error": repr(exc)}

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(summary, out_dir / "report.md")
    return summary


def _ibm_ops(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    count = row.get("ibm", {}).get("count", {})
    ops = count.get("ops", {})
    return count.get("metric_total", "-"), ops.get("cx", "-"), ops.get("u3", "-")


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# TFIM_4 A0 Template-Selector-Unit Probe",
        "",
        f"- Tower source: `{summary['source']}`",
        f"- Sidecar: `{summary['sidecar']}`",
        "",
        "## Tower X-mask skeleton",
        "",
        "| variant | wires | .qc gates | histogram |",
        "|---|---:|---:|---|",
    ]
    for name, row in summary["tower_variants"].items():
        dotqc = row.get("dotqc") or {}
        hist = ", ".join(f"{key}:{value}" for key, value in sorted((dotqc.get("gate_histogram") or {}).items()))
        lines.append(f"| `{name}` | {dotqc.get('wire_count', '-')} | {dotqc.get('gate_count', '-')} | `{hist}` |")

    t_est = summary.get("t_count_estimate", {})
    if t_est:
        lines.extend(
            [
                "",
                "## T-count Estimate",
                "",
                "Model: MCX with `c >= 2` controls costs `7 * (2 * (c - 2) + 1)` T gates; Toffoli costs `7T`; X/CNOT/Clifford gates cost `0T`.",
                f"For this SELECT skeleton, `k = {t_est.get('ctrl_size', '-')}`, active X-mask branches = {t_est.get('active_x_mask_branch_count', '-')}, total X targets = {t_est.get('x_target_total', '-')}.",
                "",
                "| estimate | T-count | formula / note |",
                "|---|---:|---|",
                f"| current surface template | {t_est.get('template_surface_t_total', '-')} | `14*(k-1)+21*w` per nonempty branch |",
                f"| repeated direct `C^kX` | {t_est.get('repeated_direct_cnx_t_total', '-')} | `7*w*(2*k-3)` per nonempty branch |",
                f"| core/patched direct-XOR template | {t_est.get('core_patched_selector_t_total', '-')} | `14*(k-1)` per nonempty branch; not current surface syntax |",
            ]
        )
        for variant_name, variant_row in summary.get("tower_variants", {}).items():
            dotqc_t = variant_row.get("markdown_t_model_from_dotqc")
            if dotqc_t:
                lines.append(
                    f"| `.qc` cross-check `{variant_name}` | {dotqc_t.get('t_total', '-')} | "
                    f"`tof/3={dotqc_t.get('tof3_count', '-')}` times `7T` |"
                )
        hist = t_est.get("x_target_count_histogram") or {}
        hist_text = ", ".join(f"`w={key}`: {value}" for key, value in hist.items()) or "-"
        lines.extend(
            [
                "",
                f"X-target distribution: {hist_text}.",
                f"Phase restoration T-count is not included here: {t_est.get('phase_restoration_t', '-')}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Restored Qiskit circuits",
            "",
            "| circuit | qubits | logical gates | IBM total | CX | U3 | NAM total | equiv note |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    equiv = summary.get("equivalence", {}).get("template_restored_vs_sidecar_semantics", {})
    for name in ("direct_reference", "template_restored"):
        row = summary["circuits"][name]
        metrics = row["qiskit_metrics"]
        total, cx, u3 = _ibm_ops(row)
        nam_total = row.get("nam", {}).get("count", {}).get("metric_total", "-")
        note = "-"
        if name == "template_restored":
            note = (
                f"amp err {equiv.get('max_expected_error', '-')}, "
                f"leak {equiv.get('max_leakage', '-')}, passed={equiv.get('passed', '-')}"
            )
        lines.append(
            f"| `{name}` | {metrics['num_qubits']} | {metrics['size']} | {total} | {cx} | {u3} | {nam_total} | {note} |"
        )

    lines.extend(["", "## Project SELECT Orientation Counts", ""])
    lines.extend(["| opt | qubits | IBM total | CX | U3 | NAM total |", "|---|---:|---:|---:|---:|---:|"])
    for key, row in summary["circuits"].items():
        if not key.startswith("project_select_") or row.get("status") == "error":
            continue
        opt = key.removeprefix("project_select_")
        metrics = row["qiskit_metrics"]
        total, cx, u3 = _ibm_ops(row)
        nam_total = row.get("nam", {}).get("count", {}).get("metric_total", "-")
        lines.append(f"| `{opt}` | {metrics['num_qubits']} | {total} | {cx} | {u3} | {nam_total} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tower-bin", type=Path, default=DEFAULT_TOWER)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
