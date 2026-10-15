"""Check the saved native PL QSVT circuits against the frozen complex target."""

from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
OUT = ROOT / "Baseline_results/rq3_refresh_20261013"


def main():
    import numpy as np
    from qiskit import qasm2
    from qiskit.quantum_info import SparsePauliOp
    from experiment_protocol import source_identity
    from run_rq3_pennylane_qsvt import columns, polynomial

    select = json.loads((OUT / "select/summary.json").read_text())
    assert select["completed"] and len(select["rows"]) == 20, "Finish all SELECT experiments first."
    smoke = json.loads((OUT / "smoke.json").read_text())
    assert smoke["passed"] and source_identity() == smoke["sources"]
    manifest_path = OUT / "qsvt_inputs.json"
    inputs = json.loads(manifest_path.read_text())
    pl = json.loads((OUT / "pennylane_qsvt/summary.json").read_text())
    qr = json.loads((OUT / "qrisp_qsvt/summary.json").read_text())
    assert pl["passed"] and len(pl["runs"]) == len(qr["cases"]) == len(inputs["cases"]) == 7
    assert all(case["status"] == "passed" for case in qr["cases"])
    archive = ROOT / inputs["archive"]
    assert sha256(archive.read_bytes()).hexdigest() == inputs["archive_sha256"]
    result = {"runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "inputs_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
              "sources": smoke["sources"], "archive_sha256": inputs["archive_sha256"],
              "native_pennylane_compilation_rerun": False,
              "target": "P(H/sum(abs(coefficients))), complete complex projected block",
              "scope": "all system basis inputs, zero auxiliary inputs; saved NAM/IBM circuits",
              "cases": [], "completed": False}

    def save():
        path = OUT / "archived_pl_qsvt_correctness.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    with tarfile.open(archive) as saved:
        for case in inputs["cases"]:
            started = time.perf_counter()
            n = len(case["terms"][0][0])
            dimension = 1 << n
            h = SparsePauliOp.from_list(case["terms"]).to_matrix() / case["alpha"]
            target = polynomial(h, case["poly"])
            outputs, checks = {}, {}
            for basis in ("nam", "ibm"):
                metric = case["pl_metrics"][basis]
                member = metric["qasm"]
                text = saved.extractfile(member).read().decode()
                circuit = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
                phase = metric.get("global_phase_radians", metric.get("global_phase"))
                assert phase is not None, (case["id"], basis)
                if "global_phase_sidecar" in metric:
                    sidecar = json.loads(saved.extractfile(metric["global_phase_sidecar"]).read())
                    assert abs(sidecar["global_phase_radians"] - phase) < 1e-12
                circuit.global_phase = float(phase)
                assert circuit.num_qubits == metric["width"]
                outputs[basis] = columns(circuit, dimension)
                checks[basis] = {
                    "target_max_abs_error": float(np.max(np.abs(outputs[basis][:dimension] - target))),
                    "column_orthogonality_max_abs_error": float(np.max(np.abs(
                        outputs[basis].conj().T @ outputs[basis] - np.eye(dimension)))),
                    "qasm_member": member, "qasm_sha256": sha256(text.encode()).hexdigest(),
                    "global_phase_radians": phase, "width": circuit.num_qubits,
                    "system_basis_inputs": dimension,
                }
            basis_error = float(np.max(np.abs(outputs["nam"] - outputs["ibm"])))
            worst = max(basis_error, *(entry[key] for entry in checks.values()
                                      for key in ("target_max_abs_error", "column_orthogonality_max_abs_error")))
            row = {"id": case["id"], "basis": checks, "basis_full_output_max_abs_error": basis_error,
                   "target_imaginary_max": float(np.max(np.abs(target.imag))),
                   "passed": worst < 2e-8, "seconds": time.perf_counter() - started}
            result["cases"].append(row)
            save()
            print(json.dumps({"id": case["id"], "passed": row["passed"], "max_error": worst}), flush=True)
    result["completed"] = True
    result["passed"] = all(row["passed"] for row in result["cases"])
    save()
    assert result["passed"], "A saved native PL QSVT circuit failed semantic verification."


if __name__ == "__main__":
    main()
