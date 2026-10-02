#!/usr/bin/env python3
"""Run pytket CnX pairwise decomposition on channelIR Matrix-order modes.

The script is intentionally separate from ``src/block_encoding_new.py``.  It
accepts either an already exported ``g_modes_with_phase`` JSON payload or a
Pauli-term Matrixsum JSON payload, builds a pytket circuit for the Matrix-order
controlled-Pauli SELECT block, applies ``CnXPairwiseDecomposition``, rebases to
the NAM gate set ``{rz, h, x, cx}``, and writes QASM plus a small JSON summary.
Nonzero controlled global phases are rejected by default; use
``--phase-policy synthesize`` to implement them with a reusable clean phase
ancilla and a ``CnX - U1 - CnX`` pattern before the pairwise pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BASELINE_DIR = ROOT / "Baseline_scripts"
for candidate in (SRC_DIR, BASELINE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from baseline_common import count_qasm_text, write_summary  # type: ignore  # noqa: E402


NAM_OPTYPE_NAMES = ("Rz", "H", "X", "CX")
IBM_OPTYPE_NAMES = ("U1", "U2", "U3", "CX")


@dataclass(frozen=True)
class Mode:
    pauli: str
    ctrl: str
    phase: float


def parse_complex(value: Any) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, str):
        return complex(value.replace("i", "j"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, dict):
        real = value.get("real", value.get("re", value.get("coeff_real", 0.0)))
        imag = value.get("imag", value.get("im", value.get("coeff_imag", 0.0)))
        return complex(float(real), float(imag))
    raise ValueError(f"Cannot parse complex value: {value!r}")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.complexfloating):
        complex_value = complex(value)
        return {"real": float(complex_value.real), "imag": float(complex_value.imag)}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def clean_pauli_label(label: Any) -> str:
    cleaned = str(label).strip().replace(" ", "").upper()
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    if len(cleaned) == 0 or any(char not in {"I", "X", "Y", "Z"} for char in cleaned):
        raise ValueError(f"Invalid Pauli label: {label!r}")
    return cleaned


def normalize_ctrl(ctrl: Any) -> str:
    cleaned = str(ctrl).strip().replace(" ", "")
    if len(cleaned) == 0 or any(bit not in {"0", "1"} for bit in cleaned):
        raise ValueError(f"Invalid control bit string: {ctrl!r}")
    return cleaned


def normalize_phase(phase: Any) -> float:
    if phase is None:
        return 0.0
    if isinstance(phase, str):
        return float(phase)
    return float(phase)


def mode_from_row(row: Any) -> Mode:
    if isinstance(row, dict):
        pauli = row.get("pauli", row.get("pauli_label", row.get("label")))
        ctrl = row.get("ctrl", row.get("ctrl_value", row.get("control", row.get("controls"))))
        phase = row.get("phase", row.get("phase_pi_over_2", row.get("phase_units", 0.0)))
        if pauli is None or ctrl is None:
            raise ValueError(f"Mode row must contain pauli/label and ctrl fields: {row!r}")
        return Mode(clean_pauli_label(pauli), normalize_ctrl(ctrl), normalize_phase(phase))

    if isinstance(row, (list, tuple)) and len(row) >= 2:
        phase = row[2] if len(row) >= 3 else 0.0
        return Mode(clean_pauli_label(row[0]), normalize_ctrl(row[1]), normalize_phase(phase))

    raise ValueError(f"Unsupported mode row: {row!r}")


def load_modes_json(path: Path) -> tuple[list[Mode], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {"source": str(path), "source_kind": "modes_json"}

    if isinstance(data, dict):
        modes_payload = (
            data.get("g_modes_with_phase")
            or data.get("modes")
            or data.get("modes_with_phase")
            or data.get("mobius_phase_result", {}).get("g_modes_with_phase")
            or data.get("input", {}).get("mobius_phase_result", {}).get("g_modes_with_phase")
        )
        if modes_payload is None:
            raise ValueError(
                "Modes JSON must be a list or contain g_modes_with_phase, modes, "
                "modes_with_phase, or mobius_phase_result.g_modes_with_phase."
            )
        for key in ("ctrl_size", "selected_k", "sys_size", "case", "name"):
            if key in data:
                metadata[key] = data[key]
    elif isinstance(data, list):
        modes_payload = data
    else:
        raise ValueError(f"Unsupported modes JSON root type: {type(data).__name__}")

    modes = [mode_from_row(row) for row in modes_payload]
    return modes, metadata


def matrixsum_terms_from_payload(data: Any, kraus_index: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {"source_kind": "matrixsum_json"}

    if isinstance(data, dict) and isinstance(data.get("terms"), list):
        metadata["kraus_index"] = data.get("index", kraus_index)
        return list(data["terms"]), metadata

    if isinstance(data, list) and all(isinstance(item, dict) and "terms" in item for item in data):
        selected = None
        if kraus_index is not None:
            for item in data:
                if int(item.get("index", -1)) == kraus_index:
                    selected = item
                    break
            if selected is None:
                raise ValueError(f"No Kraus entry with index={kraus_index} in Matrixsum JSON.")
        else:
            selected = next((item for item in data if len(item.get("terms", [])) > 0), None)
        if selected is None:
            raise ValueError("Matrixsum JSON does not contain any nonempty Kraus terms.")
        metadata["kraus_index"] = selected.get("index")
        return list(selected["terms"]), metadata

    if isinstance(data, list):
        metadata["kraus_index"] = kraus_index
        return list(data), metadata

    raise ValueError(
        "Matrixsum JSON must be a term list, a dict with terms, or a list of "
        "{index, terms} Kraus entries."
    )


def term_amplitude(term: dict[str, Any]) -> complex:
    if "phase_real" in term or "phase_imag" in term:
        phase = complex(float(term.get("phase_real", 1.0)), float(term.get("phase_imag", 0.0)))
        coeff = float(term.get("coeff", term.get("coefficient", 1.0)))
        return phase * coeff

    if "coeff_real" in term or "coeff_imag" in term:
        return complex(float(term.get("coeff_real", 0.0)), float(term.get("coeff_imag", 0.0)))

    coeff = term.get("coeff", term.get("coefficient", term.get("amplitude", 1.0)))
    return parse_complex(coeff)


def load_modes_from_matrixsum_json(path: Path, kraus_index: int | None) -> tuple[list[Mode], dict[str, Any]]:
    from block_encoding_new import BlockEncoding  # type: ignore  # noqa: PLC0415
    from channel_IR import Matrixsum, PauliAtom  # type: ignore  # noqa: PLC0415

    data = json.loads(path.read_text(encoding="utf-8"))
    terms, metadata = matrixsum_terms_from_payload(data, kraus_index)
    metadata["source"] = str(path)
    instances = []

    for term in terms:
        if not isinstance(term, dict):
            raise ValueError(f"Matrixsum term must be a dict: {term!r}")
        label = clean_pauli_label(term.get("label", term.get("pauli", term.get("pauli_label"))))
        amplitude = term_amplitude(term)
        if abs(amplitude) <= 1e-14:
            continue
        instances.append((PauliAtom(label, phase=amplitude / abs(amplitude)), float(abs(amplitude))))

    if len(instances) == 0:
        raise ValueError("Matrixsum JSON contains no nonzero Pauli terms.")

    matrixsum = Matrixsum(instances)
    block_encoding = BlockEncoding(matrixsum)
    _coeff_mode_dict, ctrl_size = block_encoding.find_optimal_order_matrices()
    mobius = getattr(block_encoding, "mobius_phase_result", {})
    modes = [mode_from_row(row) for row in mobius.get("g_modes_with_phase", [])]
    metadata.update(
        {
            "term_count": len(instances),
            "sys_size": matrixsum.size,
            "ctrl_size": int(ctrl_size),
            "selected_k": getattr(block_encoding, "selected_k", ctrl_size),
            "selected_crit": getattr(block_encoding, "selected_crit", None),
            "candidate_costs": getattr(block_encoding, "candidate_costs", None),
            "width_candidate_costs": getattr(block_encoding, "width_candidate_costs", None),
            "mobius_phase_result": mobius,
        }
    )
    return modes, metadata


def demo_modes() -> tuple[list[Mode], dict[str, Any]]:
    return (
        [
            Mode("XYZI", "111", 0.0),
            Mode("ZYXI", "110", 0.0),
            Mode("IXYZ", "101", 0.0),
        ],
        {"source_kind": "demo", "ctrl_size": 3, "sys_size": 4},
    )


def pauli_ops_from_label(label: str) -> list[tuple[str, int]]:
    return [
        (char, len(label) - 1 - label_index)
        for label_index, char in enumerate(label)
        if char != "I"
    ]


def is_zero_phase_units(phase: float) -> bool:
    return abs((phase % 4.0)) <= 1e-9 or abs((phase % 4.0) - 4.0) <= 1e-9


def basis_change_to_x(circuit: Any, pauli: str, target: int) -> str | None:
    if pauli == "X":
        return None
    if pauli == "Y":
        circuit.Sdg(target)
        return "Y"
    if pauli == "Z":
        circuit.H(target)
        return "Z"
    raise ValueError(f"Unsupported Pauli component: {pauli}")


def undo_basis_change_from_x(circuit: Any, marker: str | None, target: int) -> None:
    if marker is None:
        return
    if marker == "Y":
        circuit.S(target)
    elif marker == "Z":
        circuit.H(target)
    else:
        raise ValueError(f"Unsupported basis-change marker: {marker}")


def append_controlled_pauli_as_cnx(circuit: Any, mode: Mode, ctrl_size: int, sys_size: int) -> dict[str, int]:
    from pytket import OpType  # type: ignore  # noqa: PLC0415

    active_controls = [index for index, bit in enumerate(mode.ctrl[::-1]) if bit == "1"]
    basis_changes: list[tuple[str | None, int]] = []
    cnx_count = 0
    cx_count = 0
    x_count = 0

    for pauli, target_offset in pauli_ops_from_label(mode.pauli):
        if target_offset >= sys_size:
            raise ValueError(f"Target offset {target_offset} exceeds sys_size={sys_size}.")
        target = ctrl_size + target_offset
        marker = basis_change_to_x(circuit, pauli, target)
        basis_changes.append((marker, target))

    for _pauli, target_offset in pauli_ops_from_label(mode.pauli):
        target = ctrl_size + target_offset
        if len(active_controls) == 0:
            circuit.X(target)
            x_count += 1
        elif len(active_controls) == 1:
            circuit.CX(active_controls[0], target)
            cx_count += 1
        else:
            circuit.add_gate(OpType.CnX, active_controls + [target])
            cnx_count += 1

    for marker, target in reversed(basis_changes):
        undo_basis_change_from_x(circuit, marker, target)

    return {
        "active_controls": len(active_controls),
        "pauli_weight": len(pauli_ops_from_label(mode.pauli)),
        "cnx": cnx_count,
        "cx": cx_count,
        "x": x_count,
    }


def append_controlled_phase_as_cnx(
    circuit: Any,
    active_controls: list[int],
    phase_units_pi_over_2: float,
    phase_ancilla: int | None,
) -> dict[str, int]:
    from pytket import OpType  # type: ignore  # noqa: PLC0415

    if len(active_controls) == 0:
        return {"u1": 0, "cnx": 0}

    u1_half_turns = phase_units_pi_over_2 / 2.0
    if len(active_controls) == 1:
        circuit.U1(u1_half_turns, active_controls[0])
        return {"u1": 1, "cnx": 0}

    if phase_ancilla is None:
        raise ValueError("A phase ancilla is required for multi-controlled global phases.")

    circuit.add_gate(OpType.CnX, active_controls + [phase_ancilla])
    circuit.U1(u1_half_turns, phase_ancilla)
    circuit.add_gate(OpType.CnX, active_controls + [phase_ancilla])
    return {"u1": 1, "cnx": 2}


def build_pytket_circuit(
    modes: list[Mode],
    *,
    ctrl_size: int | None,
    sys_size: int | None,
    phase_policy: str,
) -> tuple[Any, dict[str, Any]]:
    from pytket import Circuit  # type: ignore  # noqa: PLC0415

    if len(modes) == 0:
        raise ValueError("No modes to compile.")

    inferred_ctrl_size = max(len(mode.ctrl) for mode in modes)
    inferred_sys_size = max(len(mode.pauli) for mode in modes)
    ctrl_size = inferred_ctrl_size if ctrl_size is None else int(ctrl_size)
    sys_size = inferred_sys_size if sys_size is None else int(sys_size)

    if any(len(mode.ctrl) != ctrl_size for mode in modes):
        raise ValueError("All ctrl bit strings must have length ctrl_size.")
    if any(len(mode.pauli) != sys_size for mode in modes):
        raise ValueError("All Pauli labels must have length sys_size.")

    needs_phase_ancilla = (
        phase_policy == "synthesize"
        and any(
            len([index for index, bit in enumerate(mode.ctrl[::-1]) if bit == "1"]) >= 2
            and not is_zero_phase_units(mode.phase)
            for mode in modes
        )
    )
    phase_ancilla = ctrl_size + sys_size if needs_phase_ancilla else None
    phase_ancilla_count = 1 if needs_phase_ancilla else 0
    circuit = Circuit(ctrl_size + sys_size + phase_ancilla_count)
    component_stats = {
        "modes": len(modes),
        "identity_modes": 0,
        "nonzero_controlled_phase_modes": 0,
        "ignored_uncontrolled_global_phase_modes": 0,
        "ignored_controlled_phase_modes": 0,
        "synthesized_controlled_phase_modes": 0,
        "phase_ancilla_count": phase_ancilla_count,
        "phase_u1_gates_before_pairwise": 0,
        "phase_cnx_gates_before_pairwise": 0,
        "pauli_components": 0,
        "cnx_gates_before_pairwise": 0,
        "cx_gates_before_pairwise": 0,
        "x_gates_before_pairwise": 0,
    }

    for mode in modes:
        active_controls = [index for index, bit in enumerate(mode.ctrl[::-1]) if bit == "1"]
        pauli_weight = len(pauli_ops_from_label(mode.pauli))
        has_nonzero_phase = not is_zero_phase_units(mode.phase)
        if has_nonzero_phase and len(active_controls) > 0:
            component_stats["nonzero_controlled_phase_modes"] += 1
            if phase_policy == "reject":
                raise ValueError(
                    "Encountered a controlled Pauli mode with nonzero global phase "
                    f"(pauli={mode.pauli}, ctrl={mode.ctrl}, phase={mode.phase} * pi/2). "
                    "Use --phase-policy synthesize for an exact phase-ancilla "
                    "implementation, or --phase-policy ignore only for gate-count "
                    "ablations where dropping controlled global phases is acceptable."
                )
            if phase_policy == "ignore":
                component_stats["ignored_controlled_phase_modes"] += 1
            elif phase_policy == "synthesize":
                phase_added = append_controlled_phase_as_cnx(
                    circuit,
                    active_controls,
                    mode.phase,
                    phase_ancilla,
                )
                component_stats["synthesized_controlled_phase_modes"] += 1
                component_stats["phase_u1_gates_before_pairwise"] += phase_added["u1"]
                component_stats["phase_cnx_gates_before_pairwise"] += phase_added["cnx"]
            else:
                raise ValueError(f"Unsupported phase policy: {phase_policy}")
        elif has_nonzero_phase:
            component_stats["ignored_uncontrolled_global_phase_modes"] += 1

        if pauli_weight == 0:
            component_stats["identity_modes"] += 1
            continue

        added = append_controlled_pauli_as_cnx(circuit, mode, ctrl_size, sys_size)
        component_stats["pauli_components"] += added["pauli_weight"]
        component_stats["cnx_gates_before_pairwise"] += added["cnx"]
        component_stats["cx_gates_before_pairwise"] += added["cx"]
        component_stats["x_gates_before_pairwise"] += added["x"]

    component_stats["ctrl_size"] = ctrl_size
    component_stats["sys_size"] = sys_size
    component_stats["num_qubits"] = ctrl_size + sys_size + phase_ancilla_count
    return circuit, component_stats


def optype_counts(circuit: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for command in circuit.get_commands():
        name = command.op.type.name
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def run_pytket_passes(circuit: Any, gate_set: str = "nam") -> tuple[Any, dict[str, Any]]:
    from pytket import OpType  # type: ignore  # noqa: PLC0415
    from pytket.passes import (  # type: ignore  # noqa: PLC0415
        AutoRebase,
        CnXPairwiseDecomposition,
        DecomposeMultiQubitsCX,
    )

    if gate_set == "nam":
        target_basis = {OpType.Rz, OpType.H, OpType.X, OpType.CX}
        requested_basis = list(NAM_OPTYPE_NAMES)
    elif gate_set == "ibm":
        target_basis = {OpType.U1, OpType.U2, OpType.U3, OpType.CX}
        requested_basis = list(IBM_OPTYPE_NAMES)
    else:
        raise ValueError(f"Unsupported gate set: {gate_set}")

    optimized = circuit.copy()
    pass_results: dict[str, Any] = {
        "gate_set": gate_set,
        "optype_counts_initial": optype_counts(optimized),
    }
    pass_results["cnx_pairwise_changed"] = bool(CnXPairwiseDecomposition().apply(optimized))
    pass_results["optype_counts_after_cnx_pairwise"] = optype_counts(optimized)
    pass_results["decompose_multi_qubits_cx_changed"] = bool(DecomposeMultiQubitsCX().apply(optimized))
    pass_results["optype_counts_after_decompose_multi"] = optype_counts(optimized)
    pass_results["target_rebase_changed"] = bool(AutoRebase(target_basis).apply(optimized))
    pass_results["optype_counts_after_target_rebase"] = optype_counts(optimized)
    pass_results["requested_basis"] = requested_basis
    return optimized, pass_results


def write_qasm(circuit: Any, path: Path) -> str:
    from pytket.qasm import circuit_to_qasm_str  # type: ignore  # noqa: PLC0415

    text = circuit_to_qasm_str(circuit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def load_input_modes(args: argparse.Namespace) -> tuple[list[Mode], dict[str, Any]]:
    selected = [
        args.modes_json is not None,
        args.matrixsum_json is not None,
        bool(args.demo),
    ]
    if sum(selected) != 1:
        raise ValueError("Choose exactly one input: --modes-json, --matrixsum-json, or --demo.")
    if args.modes_json is not None:
        return load_modes_json(args.modes_json)
    if args.matrixsum_json is not None:
        return load_modes_from_matrixsum_json(args.matrixsum_json, args.kraus_index)
    return demo_modes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes-json", type=Path, help="JSON containing Matrix-order g_modes_with_phase rows.")
    parser.add_argument("--matrixsum-json", type=Path, help="JSON containing channelIR Pauli Matrixsum terms.")
    parser.add_argument("--kraus-index", type=int, help="Kraus index to select when matrixsum JSON contains a channel list.")
    parser.add_argument("--demo", action="store_true", help="Run a small built-in phase-free controlled-Pauli demo.")
    parser.add_argument("--ctrl-size", type=int, help="Override control register width.")
    parser.add_argument("--sys-size", type=int, help="Override system register width.")
    parser.add_argument(
        "--phase-policy",
        choices=("reject", "ignore", "synthesize"),
        default="reject",
        help="How to handle nonzero controlled global phases in modes.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Baseline_results" / "pytket_cnx_pairwise_channelir",
    )
    parser.add_argument(
        "--gate-set",
        choices=("nam", "ibm"),
        default="nam",
        help="Target gate set used for final pytket rebase and metric counting.",
    )
    parser.add_argument("--output-qasm", type=Path, help="Output QASM path. Defaults under --out-dir.")
    parser.add_argument("--summary-json", type=Path, help="Summary JSON path. Defaults next to output QASM.")
    return parser.parse_args()


def default_stem(args: argparse.Namespace, metadata: dict[str, Any]) -> str:
    if args.demo:
        return "demo_pytket_cnx_pairwise_nam"
    source = metadata.get("source")
    if source:
        return f"{Path(str(source)).stem}_pytket_cnx_pairwise_nam"
    return "channelir_pytket_cnx_pairwise_nam"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    modes, metadata = load_input_modes(args)
    stem = default_stem(args, metadata)
    output_qasm = args.output_qasm or (args.out_dir / f"{stem}.qasm")
    summary_json = args.summary_json or output_qasm.with_suffix(".summary.json")

    initial_circuit, component_stats = build_pytket_circuit(
        modes,
        ctrl_size=args.ctrl_size or metadata.get("ctrl_size"),
        sys_size=args.sys_size or metadata.get("sys_size"),
        phase_policy=args.phase_policy,
    )
    optimized_circuit, pass_results = run_pytket_passes(initial_circuit, gate_set=args.gate_set)
    qasm_text = write_qasm(optimized_circuit, output_qasm)
    qasm_stats = count_qasm_text(qasm_text, path=str(output_qasm), gate_set=args.gate_set)

    summary = {
        "experiment": "pytket_cnx_pairwise_channelir",
        "status": "ok",
        "input": metadata,
        "arguments": {
            "modes_json": None if args.modes_json is None else str(args.modes_json),
            "matrixsum_json": None if args.matrixsum_json is None else str(args.matrixsum_json),
            "kraus_index": args.kraus_index,
            "demo": bool(args.demo),
            "ctrl_size": args.ctrl_size,
            "sys_size": args.sys_size,
            "phase_policy": args.phase_policy,
            "gate_set": args.gate_set,
            "out_dir": str(args.out_dir),
            "output_qasm": str(output_qasm),
            "summary_json": str(summary_json),
        },
        "component_stats": component_stats,
        "pytket_passes": pass_results,
        "qasm_stats": qasm_stats,
        "output_qasm": str(output_qasm),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_summary(jsonable(summary), summary_json)


if __name__ == "__main__":
    main()
