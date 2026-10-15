"""Refresh only the seven retained PennyLane QSVT + canonical Ours cases.

The official two-branch outer circuit and archived phases are unchanged.
The custom PennyLane query delegates controlled block synthesis to the
canonical components API before importing its already-controlled leaves.
No matrix is used for synthesis, and PREPARE remains uncontrolled.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]

import numpy as np
import pennylane as qml
import qiskit
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from baseline_common import count_qasm_file
from block_encoding import BlockEncoding
from channel_IR import Matrixsum, PauliAtom
from pennylane_lcu_adapter import (
    DEFAULT_PIPELINE, LEAF_GATE_SET, _audit_tape, _decomposition_mode, tape_to_qiskit,
)

BASIS = {"nam": ["rz", "h", "x", "cx"], "ibm": ["u1", "u2", "u3", "cx"]}
BRIDGE = {
    "id": qml.Identity, "x": qml.X, "y": qml.Y, "z": qml.Z,
    "h": qml.Hadamard, "s": qml.S, "t": qml.T, "sx": qml.SX,
    "rx": qml.RX, "ry": qml.RY, "rz": qml.RZ, "p": qml.PhaseShift,
    "cx": qml.CNOT, "cy": qml.CY, "cz": qml.CZ, "swap": qml.SWAP,
}
ATOL = 1e-8
ADAPTERS = {}
SIMULATOR = AerSimulator(method="statevector", max_parallel_threads=1,
                         max_parallel_experiments=1)


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_record(path):
    path = Path(path)
    return {"path": relative(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def canonicalize(circuit, system, *, controlled=False):
    """Put system on low Qiskit bits and the outer signal on the high bit."""
    offset = int(controlled)
    workspace = [q for q in range(offset, circuit.num_qubits) if q not in system]
    order = list(system) + workspace + ([0] if controlled else [])
    destinations = {old: new for new, old in enumerate(order)}
    output = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    output.compose(circuit, qubits=[destinations[q] for q in range(circuit.num_qubits)], inplace=True)
    output.metadata = dict(circuit.metadata or {})
    return output


def leaf_operations(circuit, *, controlled=False):
    lowered = transpile(circuit, basis_gates=list(BRIDGE), optimization_level=0,
                        seed_transpiler=7, qubits_initially_zero=False)
    width = circuit.num_qubits
    # Template wire 0 is the signal; block wires are 1,...,width.
    highest_wire = width - int(controlled)
    with qml.QueuingManager.stop_recording():
        operations = [qml.GlobalPhase(-float(lowered.global_phase), wires=highest_wire)]
        for instruction in lowered.data:
            wires = [(width - lowered.find_bit(qubit).index - int(controlled))
                     for qubit in instruction.qubits]
            operations.append(BRIDGE[instruction.operation.name](
                *[float(p) for p in instruction.operation.params], wires=wires))
    return tuple(operations), lowered


def resources_for(operations):
    counts = Counter(qml.resource_rep(type(op), **op.resource_params) for op in operations)
    return dict(counts)


class CanonicalBlockQuery(qml.operation.Operation):
    """A query whose controlled decomposition is the canonical F-star export."""

    num_params = 0
    grad_method = None
    resource_keys = {"token", "inverse"}

    def __init__(self, token, wires, inverse=False, id=None):
        self._hyperparameters = {"token": token, "inverse": bool(inverse)}
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return dict(self.hyperparameters)

    def _flatten(self):
        return (), (self.hyperparameters["token"], tuple(self.wires), self.hyperparameters["inverse"])

    @classmethod
    def _unflatten(cls, data, metadata):
        token, wires, inverse = metadata
        return cls(token, wires, inverse)

    def adjoint(self):
        return CanonicalBlockQuery(self.hyperparameters["token"], self.wires,
                                   not self.hyperparameters["inverse"])

    @staticmethod
    def compute_decomposition(*, wires, token, inverse):
        operations = ADAPTERS[token]["leaves"][(bool(inverse), None)]
        return [qml.map_wires(op, dict(zip(range(1, len(wires) + 1), wires))) for op in operations]


def plain_resources(token, inverse):
    return resources_for(ADAPTERS[token]["leaves"][(bool(inverse), None)])


@qml.register_resources(plain_resources)
def plain_decomposition(*, wires, token, inverse):
    mapping = dict(zip(range(1, len(wires) + 1), wires))
    for op in ADAPTERS[token]["leaves"][(bool(inverse), None)]:
        qml.apply(qml.map_wires(op, mapping))


def controlled_resources(base_params, num_control_wires, num_zero_control_values, **_):
    if num_control_wires != 1:
        raise ValueError("The retained official wrapper has exactly one signal control")
    key = (base_params["inverse"], int(not num_zero_control_values))
    return resources_for(ADAPTERS[base_params["token"]]["leaves"][key])


@qml.register_resources(controlled_resources)
def controlled_decomposition(*, wires, base, control_wires, control_values, **_):
    if len(control_wires) != 1:
        raise ValueError("Unexpected multi-signal control")
    key = (base.hyperparameters["inverse"], int(control_values[0]))
    mapping = {0: control_wires[0], **dict(zip(range(1, len(base.wires) + 1), base.wires))}
    for op in ADAPTERS[base.hyperparameters["token"]]["leaves"][key]:
        qml.apply(qml.map_wires(op, mapping))


qml.add_decomps(CanonicalBlockQuery, plain_decomposition)
qml.add_decomps("C(CanonicalBlockQuery)", controlled_decomposition)


def build_adapter(case, log_file):
    started = time.perf_counter()
    terms = [(label, complex(coefficient)) for label, coefficient in case["terms"]]
    operator = Matrixsum([(PauliAtom(label, phase=coefficient / abs(coefficient)), abs(coefficient))
                          for label, coefficient in terms])
    with log_file.open("w") as log, redirect_stdout(log):
        encoder = BlockEncoding(operator, placement_options={})
        plain = encoder.circuit_components("Matrix-order")
        parts = plain.with_outer_control(1)
    system = parts.circuit.metadata["system_qubits"]
    source = canonicalize(parts.circuit, system)
    leaves = {}
    lowered_queries = {}
    for inverse in (False, True):
        circuit = source.inverse() if inverse else source
        leaves[(inverse, None)], lowered_queries[(inverse, None)] = leaf_operations(circuit)
        for state in (0, 1):
            controlled = parts.controlled(1, ctrl_state=state)
            controlled = canonicalize(controlled, [q + 1 for q in system], controlled=True)
            if inverse:
                controlled = controlled.inverse()
            leaves[(inverse, state)], lowered_queries[(inverse, state)] = leaf_operations(
                controlled, controlled=True)
    ADAPTERS[case["id"]] = {"leaves": leaves, "lowered_queries": lowered_queries}
    block_width = source.num_qubits
    with qml.QueuingManager.stop_recording():
        encoding = CanonicalBlockQuery(case["id"], list(range(1, block_width + 1)))
    metadata = {
        "backend": "block_encoding.BlockEncoding(placement_options={}).circuit_components('Matrix-order')",
        "control_export": "BlockEncodingComponents.with_outer_control(1).controlled(1, ctrl_state)",
        "prepare_has_outer_control": False,
        "symbolic_select_controlled_before_expansion": True,
        "uncontrolled_metadata": plain.circuit.metadata,
        "controlled_context_metadata": parts.circuit.metadata,
        "construction_seconds": time.perf_counter() - started,
        "block_width": block_width,
        "leaf_counts": {str(key): dict(Counter(op.name for op in ops)) for key, ops in leaves.items()},
        "matrix_synthesis_used": False,
    }
    return source, encoding, metadata


def load_official_wrapper(archive):
    member = "Baseline_results/pennylane_official_qsvt_sources/tutorial_qsvt_hardware.py"
    with tarfile.open(archive) as bundle:
        text = bundle.extractfile(member).read().decode()
    tree = ast.parse(text)
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "circuit")
    code = "\n".join(text.splitlines()[min(n.lineno for n in wrapper.decorator_list) - 1:wrapper.end_lineno]) + "\n"
    return wrapper, {"archive_member": member, "source_sha256": sha256(text.encode()).hexdigest(),
                     "outer_function_verbatim": code, "outer_function_sha256": sha256(code.encode()).hexdigest()}


def official_tape(wrapper, encoding, angles, system_size):
    with qml.QueuingManager.stop_recording():
        projectors = [qml.PCPhase(phi, dim=2**system_size, wires=encoding.wires) for phi in angles]
    scope = {"qp": qml, "dev": qml.device("default.qubit"),
             "block_encode": encoding, "projectors": projectors}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "official_hardware_wrapper", "exec"), scope)
    tape = qml.workflow.construct_tape(scope["circuit"])().copy(measurements=[])
    names = [op.name for op in tape.operations]
    if names != ["Hadamard", "C(QSVT)", "C(Adjoint(QSVT))", "Hadamard"]:
        raise ValueError(f"Official wrapper changed: {names}")
    if list(tape.operations[1].control_values) != [True] or list(tape.operations[2].control_values) != [False]:
        raise ValueError("Official branch controls changed")
    return tape


def compile_official(tape, wires):
    started = time.perf_counter()
    fixed = {CanonicalBlockQuery: plain_decomposition,
             "C(CanonicalBlockQuery)": controlled_decomposition}
    with _decomposition_mode("graph"):
        [expanded], _ = qml.transforms.decompose(tape, gate_set=LEAF_GATE_SET,
                                                fixed_decomps=fixed, num_work_wires=0)
        _audit_tape(expanded, wires)
        decomposed_counts = dict(Counter(op.name for op in expanded.operations))
        decomposition_seconds = time.perf_counter() - started
        optimizing = time.perf_counter()
        [expanded], _ = qml.compile(expanded, basis_set=sorted(LEAF_GATE_SET))
        optimization_seconds = time.perf_counter() - optimizing
        _audit_tape(expanded, wires)
    metadata = {"decomposition": "graph", "pipeline": list(DEFAULT_PIPELINE), "num_passes": 1,
                "dynamic_work_wire_budget": 0, "leaf_gate_set": sorted(LEAF_GATE_SET),
                "decomposition_seconds": decomposition_seconds, "optimization_seconds": optimization_seconds,
                "compile_seconds": time.perf_counter() - started,
                "fixed_decompositions": ["CanonicalBlockQuery", "C(CanonicalBlockQuery)"],
                "decomposed_operation_counts": decomposed_counts,
                "output_operation_counts": dict(Counter(op.name for op in expanded.operations))}
    return expanded, metadata


def columns(circuit, dimension):
    """Check all system basis inputs with zero auxiliary registers."""
    compiled = transpile(circuit, SIMULATOR, optimization_level=0, seed_transpiler=7,
                         qubits_initially_zero=False)
    experiments = []
    for column in range(dimension):
        experiment = QuantumCircuit(compiled.num_qubits)
        if column:
            for bit in range(column.bit_length()):
                if (column >> bit) & 1:
                    experiment.x(bit)
        experiment.compose(compiled, inplace=True)
        experiment.save_statevector()
        experiments.append(experiment)
    result = SIMULATOR.run(experiments).result()
    return np.column_stack([np.asarray(result.get_statevector(i)) for i in range(dimension)])


def source_reference(source, angles, system_size):
    """Independently evolve queries and reverse-adjoint projector sequences."""
    forward = transpile(source, basis_gates=list(BRIDGE), optimization_level=0,
                        seed_transpiler=7, qubits_initially_zero=False)
    inverse = forward.inverse()
    dimension = 1 << system_size

    def evolve(vector, circuit):
        return np.asarray(Statevector(np.array(vector, dtype=complex, copy=True)).evolve(circuit).data)

    def project(vector, phi):
        output = vector * np.exp(-1j * phi)
        output[:dimension] = vector[:dimension] * np.exp(1j * phi)
        return output

    def branch(vector, adjoint=False):
        if not adjoint:
            for index, phi in enumerate(angles):
                vector = project(vector, phi)
                if index + 1 < len(angles):
                    vector = evolve(vector, forward if index % 2 == 0 else inverse)
        else:
            for index in reversed(range(len(angles))):
                vector = project(vector, -angles[index])
                if index:
                    vector = evolve(vector, inverse if (index - 1) % 2 == 0 else forward)
        return vector

    block_columns, expected = [], []
    for column in range(dimension):
        initial = np.zeros(1 << forward.num_qubits, dtype=complex)
        initial[column] = 1.0
        block_columns.append(evolve(initial, forward))
        positive, negative = branch(initial), branch(initial, True)
        expected.append(np.concatenate(((positive + negative) / 2, (negative - positive) / 2)))
    return np.column_stack(block_columns), np.column_stack(expected)


def polynomial(matrix, coefficients):
    result = np.zeros_like(matrix, dtype=complex)
    power = np.eye(len(matrix), dtype=complex)
    for coefficient in coefficients:
        result += coefficient * power
        power = power @ matrix
    return result


def run_case(case, wrapper, out_dir):
    started = time.perf_counter()
    case_dir = out_dir / "cases" / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / "input.json", case)
    print(f"Building {case['id']}", flush=True)
    source, encoding, build = build_adapter(case, case_dir / "placement.log")
    angles = np.asarray(case["angles"], dtype=float)
    system_size = len(case["terms"][0][0])
    dimension = 1 << system_size
    normalized_h = SparsePauliOp.from_list(case["terms"]).to_matrix() / case["alpha"]
    target = polynomial(normalized_h, case["poly"])
    tape = official_tape(wrapper, encoding, angles, system_size)
    full_wires = [0] + list(encoding.wires)
    print(f"Compiling {case['id']}, width={len(full_wires)}", flush=True)
    lowered_tape, compile_meta = compile_official(tape, full_wires)
    exporting = time.perf_counter()
    circuit = tape_to_qiskit(lowered_tape, full_wires)
    export_seconds = time.perf_counter() - exporting
    validation_start = time.perf_counter()
    print(f"Verifying {case['id']}, all {dimension} system basis inputs", flush=True)
    block_columns, expected = source_reference(source, angles, system_size)
    native = columns(circuit, dimension)
    scratch = int(build["controlled_context_metadata"].get("scratch_qubits", 0))
    address = int(build["controlled_context_metadata"].get("shared_address_qubits", source.num_qubits - system_size - scratch))
    scratch_mask = sum(1 << (system_size + address + bit) for bit in range(scratch))
    bad_scratch = (np.arange(1 << circuit.num_qubits) & scratch_mask) != 0
    checks = {
        "source_block_encoding_error": max_error(block_columns[:dimension], normalized_h),
        "phase_sensitive_valid_input_isometry_error": max_error(native, expected),
        "physical_polynomial_block_error": max_error(native[:dimension], target),
        "reference_polynomial_block_error": max_error(expected[:dimension], target),
        "checked_columns_orthogonality_error": max_error(native.conj().T @ native, np.eye(dimension)),
        "clean_scratch_output_error": float(np.max(np.abs(native[bad_scratch]))) if np.any(bad_scratch) else 0.0,
    }
    result = {"id": case["id"], "name": case["name"], "terms": case["terms"],
              "alpha": case["alpha"], "poly": case["poly"], "angles": case["angles"],
              "pl_native_rerun": False, "archived_pl_metrics": case["pl_metrics"],
              "build": build, "compile": compile_meta, "export_seconds": export_seconds,
              "queries": 2 * (len(angles) - 1), "projectors": 2 * len(angles),
              "outer_function_preserved": True, "ours_qsvt_branch_engine_used": False,
              "physical_polynomial_imaginary_max": float(np.max(np.abs(native[:dimension].imag))),
              "target_polynomial_imaginary_max": float(np.max(np.abs(target.imag))),
              "checks": checks, "basis": {},
              "verification": {"method": "all clean-auxiliary system basis inputs; all complex output amplitudes",
                               "system_basis_inputs": dimension, "all_output_amplitudes_checked": True,
                               "physical_block_compared_without_real_projection": True,
                               "independent_reference": "source queries and reverse-adjoint projector sequence",
                               "all_nonzero_auxiliary_inputs_checked": False,
                               "reference_and_native_seconds": time.perf_counter() - validation_start}}
    for basis, gates in BASIS.items():
        lowering_start = time.perf_counter()
        lowered = transpile(circuit, basis_gates=gates, optimization_level=0,
                            seed_transpiler=7, qubits_initially_zero=False)
        lowering_seconds = time.perf_counter() - lowering_start
        compiled_dir = case_dir / "compiled"
        compiled_dir.mkdir(parents=True, exist_ok=True)
        qasm_path = compiled_dir / f"ours_injected_{basis}_opt0.qasm"
        qasm_path.write_text(qasm2.dumps(lowered) + "\n")
        phase = float(lowered.global_phase)
        phase_path = qasm_path.with_suffix(".global_phase.json")
        write_json(phase_path, {"global_phase_radians": phase,
                                "restore_policy": "Restore circuit.global_phase after importing OpenQASM 2"})
        validation_start = time.perf_counter()
        lowered_columns = columns(lowered, dimension)
        # Verify the actual saved QASM with its saved phase, including IBM u1/u2/u3.
        roundtrip = qasm2.load(qasm_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        roundtrip.global_phase = json.loads(phase_path.read_text())["global_phase_radians"]
        roundtrip_columns = columns(roundtrip, dimension)
        stats = count_qasm_file(qasm_path, gate_set=basis)
        metric = {"gate_count": int(stats["metric_total"]),
                  "non_clifford": int(stats["clifford"]["non_clifford"]),
                  "cx_count": int(stats["ops"].get("cx", 0)), "width": lowered.num_qubits,
                  "depth": lowered.depth(), "ops": stats["ops"],
                  "basis_lowering_seconds": lowering_seconds,
                  "total_compile_seconds": build["construction_seconds"] + compile_meta["compile_seconds"]
                    + export_seconds + lowering_seconds,
                  "phase_sensitive_lowering_error": max_error(lowered_columns, native),
                  "phase_sensitive_reference_error": max_error(lowered_columns, expected),
                  "physical_polynomial_block_error": max_error(lowered_columns[:dimension], target),
                  "saved_qasm_roundtrip_error": max_error(roundtrip_columns, lowered_columns),
                  "verification_seconds": time.perf_counter() - validation_start,
                  "global_phase_radians": phase, "qasm": relative(qasm_path),
                  "global_phase_sidecar": relative(phase_path)}
        metric["pl_over_ours_gate_ratio"] = case["pl_metrics"][basis]["gate_count"] / metric["gate_count"]
        result["basis"][basis] = metric
        write_json(case_dir / "summary.json", result)
        print(json.dumps({"case": case["id"], "basis": basis, "G": metric["gate_count"],
                          "NC": metric["non_clifford"], "W": metric["width"],
                          "lowering_error": metric["phase_sensitive_lowering_error"]}), flush=True)
    errors = list(checks.values()) + [metric[key] for metric in result["basis"].values()
             for key in ("phase_sensitive_lowering_error", "phase_sensitive_reference_error",
                         "physical_polynomial_block_error", "saved_qasm_roundtrip_error")]
    result["passed"] = max(errors) < ATOL
    result["status"] = "ok" if result["passed"] else "semantic_failure"
    result["wall_seconds"] = time.perf_counter() - started
    write_json(case_dir / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "Baseline_results/rq3_refresh_20261013/qsvt_inputs.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Baseline_results/rq3_refresh_20261013/pennylane_qsvt")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = ROOT / "Baseline_results/rq3_refresh_20261013/smoke.json"
    smoke = json.loads(smoke_path.read_text())
    if smoke.get("passed") is not True:
        raise SystemExit("Canonical block encoding archive smoke test must pass before execution")
    for name in ("block_encoding.py", "placement_search.py", "matrix_order_pipeline.py",
                 "channel_IR.py", "subroutine.py"):
        if sha256((ROOT / "src" / name).read_bytes()).hexdigest() != smoke["sources"][name]:
            raise SystemExit(f"Canonical source changed since the passing smoke test: {name}")
    manifest = json.loads(args.inputs.read_text())
    archive = ROOT / manifest["archive"]
    if sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise SystemExit("The source archive fingerprint differs from the frozen inputs")
    wrapper, official = load_official_wrapper(archive)
    sources = [source_record(path) for path in (
        Path(__file__), args.inputs, smoke_path, ROOT / "Baseline_scripts/pennylane_lcu_adapter.py",
        ROOT / "Baseline_scripts/baseline_common.py", ROOT / "src/block_encoding.py",
        ROOT / "src/placement_search.py", ROOT / "src/matrix_order_pipeline.py",
        ROOT / "src/channel_IR.py", ROOT / "src/subroutine.py",
        Path(inspect.getfile(qml.QSVT)), Path(inspect.getfile(qml.PCPhase)),
    )]
    result = {"measured_at": datetime.now(timezone.utc).isoformat(),
              "versions": {"pennylane": qml.__version__, "qiskit": qiskit.__version__, "python": sys.version},
              "source_fingerprints": sources, "official": official,
              "basis_policy": {"bases": BASIS, "optimization_level": 0, "seed_transpiler": 7,
                               "qubits_initially_zero": False, "metric": "baseline_common.count_qasm_file"},
              "only_ours_injected_routes_executed": True, "runs": []}
    cases = [case for case in manifest["cases"] if not args.cases or case["id"] in args.cases]
    if args.cases and {case["id"] for case in cases} != set(args.cases):
        raise SystemExit("Unknown retained case id")
    for case in cases:
        summary_path = args.out_dir / "cases" / case["id"] / "summary.json"
        if args.resume and summary_path.exists():
            cached = json.loads(summary_path.read_text())
            if cached.get("passed") and cached.get("source_fingerprints") == sources:
                result["runs"].append(cached)
                continue
        try:
            row = run_case(case, wrapper, args.out_dir)
        except Exception as exc:
            row = {"id": case["id"], "status": "exception", "passed": False,
                   "error": repr(exc), "traceback": traceback.format_exc()}
            write_json(summary_path, row)
            print(row["traceback"], flush=True)
        row["source_fingerprints"] = sources
        row["input_source_member"] = case["source_member"]
        write_json(summary_path, row)
        result["runs"].append(row)
        result["passed"] = all(row.get("passed") for row in result["runs"])
        write_json(args.out_dir / "summary.json", result)
    result["passed"] = bool(result["runs"]) and all(row.get("passed") for row in result["runs"])
    write_json(args.out_dir / "summary.json", result)
    print(json.dumps({"passed": result["passed"], "summary": relative(args.out_dir / "summary.json")}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
