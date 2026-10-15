"""Apply shared conjugation control rules to the two verified diagonal cases.

The standalone SELECT experiments are unchanged. Ours restores their saved
placement and residuals; controlled SELECT is lowered structurally without
placement search. PREPARE is the same H^5 for all three methods. The common
two-branch QSVT and projector helpers are shared with the seven small cases.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "QISKIT_NUM_PROCS"):
    os.environ[variable] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/codex-rq3-numba")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-rq3-mpl")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]

import numpy as np
from qiskit import QuantumCircuit, qpy

from block_encoding import BlockEncodingComponents
from experiment_protocol import source_identity
from matrix_order_pipeline import resolve_options, symbolic_select
import run_rq3_frozen_select_qsvt as frozen
from verify_rq3_diagonal_counterexamples import verify_logical

BATCH = ROOT / "Baseline_results/rq3_qrisp_aligned_qsvt_20261014"
OUT = BATCH / "counterexamples"
SMOKE = BATCH / "interface_smoke/smoke.json"


def dump(path, data):
    frozen.dump(path, data)


def record(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}


def canonical_select(circuit, system, address, scratch, signal=None):
    order = system + address + scratch + ([] if signal is None else [signal])
    assert sorted(order) == list(range(circuit.num_qubits))
    mapping = {old: new for new, old in enumerate(order)}
    output = QuantumCircuit(circuit.num_qubits)
    output.compose(circuit, qubits=[mapping[q] for q in range(circuit.num_qubits)], inplace=True)
    return output


def restore_ours(case):
    metadata = case["methods"]["ours"]["metadata"]
    path = frozen.SELECTS / "cases" / case["id"] / "ours/select.qpy"
    with path.open("rb") as stream:
        select = qpy.load(stream)[0]
    prepare = QuantumCircuit(select.num_qubits)
    for q in metadata["address_qubits"]:
        prepare.h(q)
    block = QuantumCircuit(select.num_qubits)
    block.compose(prepare, inplace=True)
    block.compose(select, inplace=True)
    block.compose(prepare.inverse(), inplace=True)
    block.metadata = deepcopy(metadata["canonical"])
    options = resolve_options(block.metadata["matrix_order"]["options"])
    symbolic = symbolic_select(metadata["residuals"], 5, case["n"],
        phase_on_address=block.metadata["matrix_order"]["selected_candidate"].endswith("/phase-on-address"))
    parts = BlockEncodingComponents(block, prepare, select, symbolic_select=symbolic,
        matrix_order_options=options, placement_candidates=())
    active = parts.with_outer_control(1)
    assert active.placement_candidates == ()
    before = block.metadata["matrix_order"]["placement"]
    assert active.circuit.metadata["matrix_order"]["placement"] == before
    n = case["n"]
    width = active.circuit.num_qubits
    address = list(range(5))
    system = active.circuit.metadata["system_qubits"]
    scratch = [q for q in range(width) if q not in system + address]
    controlled = canonical_select(active.expanded_controlled_select,
        [q + 1 for q in system], [q + 1 for q in address],
        [q + 1 for q in scratch], signal=0)
    plain = canonical_select(select, metadata["system_qubits"], metadata["address_qubits"],
        metadata["clean_scratch_qubits"])
    original_scratch = plain.num_qubits - n - 5
    if len(scratch) > original_scratch:
        padded = QuantumCircuit(n + 5 + len(scratch))
        padded.compose(plain, inplace=True)
        plain = padded
    return plain, controlled, metadata["placement"], {
        "source_select": record(path), "source_standalone_metrics": case["methods"]["ours"]["basis"],
        "source_selected_candidate": block.metadata["matrix_order"]["selected_candidate"],
        "standalone_placement": before, "controlled_placement": active.circuit.metadata["matrix_order"]["placement"],
        "placement_reselection": False, "new_placement_search": False,
        "symbolic_residuals_restored": metadata["residuals"],
        "controlled_select_lowering": active.circuit.metadata["controlled_select_expansion"],
        "scratch_size": len(scratch)}


def build_qrisp_select(case, controlled):
    from run_rq3_qrisp_aligned_qsvt import build_qrisp_select as shared_builder

    adapted = {**case, "terms": [(label, real) for label, real, imag in case["terms"]]}
    assert all(imag == 0 for _, _, imag in case["terms"])
    plain, controls, metadata = shared_builder(adapted)
    return controls[1] if controlled else plain, metadata


def build_pl_select(case, controlled):
    from run_rq3_qrisp_aligned_qsvt import build_pl_select as shared_builder

    adapted = {**case, "terms": [(label, real) for label, real, imag in case["terms"]]}
    assert all(imag == 0 for _, _, imag in case["terms"])
    plain, controls, metadata = shared_builder(adapted, work_wires=4)
    return controls[1] if controlled else plain, metadata


def pad_plain_and_controlled(plain, controlled, n):
    scratch = max(plain.num_qubits - n - 5, controlled.num_qubits - n - 6)
    padded_plain = QuantumCircuit(n + 5 + scratch)
    padded_plain.compose(plain, inplace=True)
    padded_controlled = QuantumCircuit(n + 6 + scratch)
    mapping = list(range(controlled.num_qubits))
    mapping[-1] = padded_controlled.num_qubits - 1
    padded_controlled.compose(controlled, qubits=mapping, inplace=True)
    return padded_plain, padded_controlled, scratch


def diagonal_system_fragments(circuit, system):
    """Regroup exact phase-polynomial CNOT fragments for symbolic verification.

    Native controlled Z decompositions can temporarily XOR a workspace bit
    into a system bit. Regroup the closed XOR/phase fragment, verifying its
    exact local matrix is diagonal before using the Z-word proof. This changes
    only the verification representation, never the circuit that is measured.
    """
    from qiskit.quantum_info import Operator

    # Native graph decompositions also insert H/H pairs on a system wire,
    # separated solely by workspace operations. H^2=I exactly, and the
    # intervening gates have disjoint support. Cancel only these proven pairs
    # in the verification copy; measured circuits retain every instruction.
    last_touch = {}
    remove = set()
    for position, instruction in enumerate(circuit.data):
        bits = [circuit.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "h" and len(bits) == 1 and bits[0] in system:
            previous = last_touch.get(bits[0])
            if previous is not None and circuit.data[previous].operation.name == "h":
                remove.update((previous, position))
                last_touch.pop(bits[0])
                continue
        for bit in bits:
            last_touch[bit] = position
    verification = circuit.copy_empty_like()
    verification.global_phase = circuit.global_phase
    for position, instruction in enumerate(circuit.data):
        if position not in remove:
            verification.append(instruction.operation,
                [circuit.find_bit(q).index for q in instruction.qubits], instruction.clbits)
    circuit = verification
    diagonal_names = {"p", "rz", "t", "tdg", "s", "sdg", "z", "id"}
    output = circuit.copy_empty_like()
    output.global_phase = circuit.global_phase
    index = 0
    system = set(system)
    while index < len(circuit.data):
        item = circuit.data[index]
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name != "cx" or wires[-1] not in system:
            output.append(item.operation, wires, item.clbits)
            index += 1
            continue
        target = wires[-1]
        assert wires[0] not in system
        parity = {wires[0]}
        collected = [item]
        cursor = index + 1
        while cursor < len(circuit.data):
            current = circuit.data[cursor]
            bits = [circuit.find_bit(q).index for q in current.qubits]
            if current.operation.name in diagonal_names and not (set(bits) & (system - {target})):
                collected.append(current)
                cursor += 1
                continue
            if parity and current.operation.name == "cx" and bits[-1] == target and bits[0] not in system:
                parity.symmetric_difference_update({bits[0]})
                collected.append(current)
                cursor += 1
                continue
            assert not parity, ("System XOR phase fragment did not close", index, cursor)
            break
        assert not parity
        local = sorted({circuit.find_bit(q).index for instruction in collected for q in instruction.qubits})
        assert len(local) <= 9
        positions = {q: i for i, q in enumerate(local)}
        fragment = QuantumCircuit(len(local))
        for instruction in collected:
            fragment.append(instruction.operation,
                [positions[circuit.find_bit(q).index] for q in instruction.qubits])
        matrix = Operator(fragment).data
        assert np.max(abs(matrix - np.diag(np.diag(matrix)))) < 1e-12
        output.append(fragment.to_gate(label="ExactDiagonalPhaseFragment"), local)
        index = cursor
    return output


def validate_selects(case, plain, controlled, placement, scratch):
    n, m = case["n"], 5
    system = list(range(n))
    address = list(range(n, n + m))
    work = list(range(n + m, n + m + scratch))
    symbolic_plain = diagonal_system_fragments(plain, system)
    symbolic_controlled = diagonal_system_fragments(controlled, system)
    proof_plain, (words, phases) = verify_logical(symbolic_plain, system, address, work, case["terms"], placement)
    active = {label: format((1 << m) + int(bits, 2), "06b") for label, bits in placement.items()}
    proof_controlled, (c_words, c_phases) = verify_logical(symbolic_controlled, system,
        address + [controlled.num_qubits - 1], work, case["terms"], active)
    error = float(max(np.max(abs(c_phases[:1 << m] - 1)),
        np.max(abs(c_phases[1 << m:] - phases))))
    assert not np.any(c_words[:1 << m])
    assert np.array_equal(c_words[1 << m:], words)
    assert error < frozen.TOL
    return {"plain": proof_plain, "controlled": proof_controlled,
        "active_control_matches_saved_select": True, "inactive_control_exact_identity": True,
        "active_and_inactive_phase_max_error": error,
        "native_system_phase_fragments": "verification-only cancellation of H/H across disjoint support, followed by closed XOR/phase fragments with exact local diagonal-matrix checks"}


def run_case(case, method, directory, sources):
    from run_rq3_qrisp_aligned_qsvt import assemble_aligned_outer, compose_controlled_query

    started = time.perf_counter()
    assert case["K"] == 32 and (case["K"] - 1).bit_length() == 5
    magnitudes = [abs(complex(real, imag)) for _, real, imag in case["terms"]]
    assert max(magnitudes) - min(magnitudes) < 1e-14
    directory.mkdir(parents=True, exist_ok=True)
    if method == "ours":
        plain, controlled, placement, metadata = restore_ours(case)
    else:
        builder = build_pl_select if method == "pl" else build_qrisp_select
        plain, plain_meta = builder(case, False)
        controlled, control_meta = builder(case, True)
        placement = {label: format(i, "05b") for i, (label, _, _) in enumerate(case["terms"])}
        metadata = {"plain": plain_meta, "controlled": control_meta, "placement_reselection": False}
    plain, controlled, scratch = pad_plain_and_controlled(plain, controlled, case["n"])
    proof = validate_selects(case, plain, controlled, placement, scratch)
    n = case["n"]
    prepare = QuantumCircuit(n + 5 + scratch)
    for q in range(n, n + 5):
        prepare.h(q)
    query = compose_controlled_query(prepare, controlled,
        system_size=n, address_size=5, scratch_size=scratch)
    full, outer_metadata = assemble_aligned_outer(query, case["angles"],
        system_size=n, address_size=5, scratch_size=scratch, optimize=True)
    for filename, circuit in (("plain_select.qpy", plain), ("controlled_select.qpy", controlled),
                              ("controlled_query.qpy", query), ("full_qsvt.qpy", full)):
        with (directory / filename).open("wb") as handle:
            qpy.dump(circuit, handle)
    row = {"case": case["id"], "method": method, "input": {k: v for k, v in case.items() if k != "methods"},
        "source_records": sources,
        "placement": placement, "query_metadata": metadata, "controlled_select_validation": proof,
        "shared_outer": outer_metadata, "prepare": "identical unconditioned H^5",
        "placement_reselection": False, "basis": {}}
    for basis in frozen.BASES:
        lowered = frozen.lower(full, basis)
        saved, metric = frozen.write_circuit(directory / basis / "full_qsvt.qasm", lowered, basis)
        print(json.dumps({"case": case["id"], "method": method, "basis": basis,
            "stage": "counts", "G": metric["G"], "NC": metric["NC"], "W": metric["W"]}), flush=True)
        validation = frozen.verify_saved(saved, case, placement, case["angles"], directory / basis)
        validation["entire_system_domain_SELECT_proof"] = "all 64 logical address/outer sectors; symbolic system Z-word proof with clean scratch"
        metric["validation"] = validation
        row["basis"][basis] = metric
        dump(directory / "result.partial.json", row)
    row["passed"] = all(item["validation"]["passed"] for item in row["basis"].values())
    row["seconds"] = time.perf_counter() - started
    assert all(record(ROOT / item["path"]) == item for item in sources if not Path(item["path"]).is_absolute())
    assert all(sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
        for item in sources if Path(item["path"]).is_absolute())
    dump(directory / "result.json", row)
    return row


def main(args):
    smoke = json.loads(SMOKE.read_text())
    assert smoke["passed"] and smoke["sources"] == source_identity()
    paths = [Path(__file__).resolve(), SMOKE, frozen.PHASES]
    paths += [ROOT / "Baseline_scripts" / name for name in (
        "run_rq3_qrisp_aligned_qsvt.py", "run_rq3_frozen_select_qsvt.py",
        "verify_rq3_diagonal_counterexamples.py", "pennylane_lcu_adapter.py",
        "run_rq3_qrisp_qsvt.py", "baseline_common.py", "run_rq3_pennylane_qsvt.py",
        "run_rq3_common_pl_qsvt.py", "run_rq3_common_qrisp_qsvt.py")]
    paths += [ROOT / "src" / name for name in source_identity()]
    paths += [Path(importlib.import_module(name).__file__) for name in (
        "qrisp.alg_primitives.program_control.quantum_switch", "qrisp.environments.conjugation_environment",
        "qrisp.environments.custom_control_environment", "qrisp.environments.control_environment",
        "qrisp.block_encodings.constructors.from_lcu", "qrisp.interface.converter.qiskit_converter",
        "pennylane.templates.subroutines.select", "pennylane.templates.subroutines.prepselprep")]
    sources = []
    for path in dict.fromkeys(paths):
        item = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "sha256": sha256(path.read_bytes()).hexdigest()}
        sources.append(item)
        suffix = (path.relative_to(ROOT) if path.is_relative_to(ROOT) else
                  Path("package", *path.parts[path.parts.index("site-packages") + 1:]))
        snapshot = OUT / "source_snapshot" / suffix
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    phase_source = json.loads(frozen.PHASES.read_text())["cases"][0]
    calibration = frozen.calibrate()
    rows = []
    for case_id in args.cases:
        path = frozen.SELECTS / "cases" / case_id / "result.json"
        original = json.loads(path.read_text())
        assert original["sources"] == source_identity()
        prior_validation = json.loads(path.with_name("verification.json").read_text())
        assert all(prior_validation["methods"][method]["logical"]["passed"] for method in args.methods)
        case = {**original, "poly": phase_source["poly"], "angles": phase_source["angles"],
                "source_result": record(path), "source_validation": record(path.with_name("verification.json"))}
        for method in args.methods:
            directory = OUT / "cases" / case_id / method
            result_path = directory / "result.json"
            if result_path.exists() and not args.force:
                row = json.loads(result_path.read_text())
                assert row["passed"] and row["input"]["source_result"] == case["source_result"]
                assert row["source_records"] == sources
            else:
                row = run_case(case, method, directory, sources)
            rows.append(row)
            assert smoke["sources"] == source_identity()
            summary = {"scope": "Qrisp-aligned local controls in the common two-branch QSVT on fixed standalone placements",
                "runner": record(Path(__file__)), "phase_source": record(frozen.PHASES), "smoke": record(SMOKE),
                "sources": smoke["sources"], "policy": {
                    "outer": "H-C1(U_phi)-C0(U_phi.adjoint)-H", "query_count": 10, "projector_count": 12,
                    "PREPARE": "identical H^5, no outer control", "projector": "shared independent clean flag; unconditioned compute/uncompute and controlled RZ",
                    "placement_reselection": False, "new_placement_search": False,
                    "upper_compile": "same PennyLane basic compile for every route; NAM/IBM opt0"},
                "source_records": sources,
                "calibration": calibration, "rows": rows, "passed": all(row["passed"] for row in rows)}
            dump(OUT / "summary.partial.json", summary)
    dump(OUT / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=frozen.CASES, default=list(frozen.CASES))
    parser.add_argument("--methods", nargs="+", choices=("ours", "pl", "qrisp"), default=["ours", "pl", "qrisp"])
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
