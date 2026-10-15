"""Compile selected rewrite outcomes through the same tree/F*/Matrix-order path."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "streaming_support_beam_20261012"
SNAPSHOT = HERE.parent / "thermal2_ho_remote_20261011/source_snapshot"
sys.path.insert(0, str(PREVIOUS))
from fast_probe import dump
from channel_IR import Matrixsum, PauliAtom, channel_ensemble
from channel_LCU import channel_to_LCU
from block_encoding import BlockEncoding
from experiment_protocol import lower_and_count
from rq1_pipeline import verify_channel
sys.path.insert(0, str(ROOT / "Baseline_scripts"))
from baseline_common import count_qasm_text
from qiskit import qasm2


def kraus_from_array(A, labels):
    rows = []
    for row in A:
        terms = [(PauliAtom(str(label), phase=value / abs(value)), float(abs(value)))
                 for label, value in zip(labels, row) if abs(value) > 1e-12]
        if terms:
            rows.append(Matrixsum(terms))
    return rows


def effective_settings(case, method, protocol_settings):
    manifest = json.loads((case / "manifest.json").read_text())
    settings = dict(protocol_settings)
    settings["protocol_default_rewrite"] = settings.pop("rewrite")
    settings["rewrite"] = {
        "implementation": "isolated fast_probe.py" if method == "support" else "isolated diagnostic_gatecost.py",
        "strategy": "streaming_beam",
        "objective": "Pauli support" if method == "support" else "H_length_static_nc_v1",
        "rank_mode": "support_rank" if method == "support" else manifest["gate_mode"],
        "beam_width": manifest["width"], "max_steps": manifest["max_steps"],
        "keep_neutral": manifest["neutral"], "tol": 1e-12,
        "timeout_s": manifest["per_search_timeout_s"],
        "applied_during_this_compilation": False,
        "input": "saved verified rewrite output",
    }
    return settings


def main():
    names = ["CPTP2_M4_seed0", "CPTP2_M4_seed4", "CPTP2_M3_seed6"]
    for name in names:
        case = HERE / "cases" / name
        for method in ("support", "gatecost"):
            output = case / (method + "_circuit.json")
            if output.exists():
                continue
            z = np.load(case / (method + ".npz"))
            rows = kraus_from_array(z["A_final"], z["labels"])
            original_rows = kraus_from_array(z["A_initial"], z["labels"])
            start = time.perf_counter()
            backend = lambda ms: BlockEncoding(ms, placement_options={}, matrix_order_options={})
            circuit, registers = channel_to_LCU(channel_ensemble([rows]), structure="tree", opt="Matrix-order",
                                               block_encoding_cls=backend, fstar=True)
            assembly_s = time.perf_counter() - start
            lowered, metrics = lower_and_count(circuit, rq="RQ1", basis="nam")
            compile_s = time.perf_counter() - start
            qasm = qasm2.dumps(lowered)
            (case / (method + "_nam.qasm")).write_text(qasm)
            stats = count_qasm_text(qasm, gate_set="nam")
            assert stats["metric_total"] == metrics["G"]
            validation = verify_channel(lowered, registers, original_rows)
            assert validation["passed"]
            lam = float(circuit.metadata["subnormalization_factor"])
            result = {"case": name, "method": method, "G": metrics["G"], "W": metrics["W"],
                      "Lambda": lam, "Lambda_G": lam * metrics["G"], "compile_s": compile_s,
                      "assembly_s": assembly_s, "lowering_s": metrics["lowering_time_s"],
                      "counts": metrics["counts"], "qasm_recount": stats,
                      "verification": validation, "settings": effective_settings(case, method, metrics["settings"]),
                      "sources": metrics["sources"], "metadata": circuit.metadata,
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            dump(output, result)
            print(json.dumps({k: result[k] for k in ("case", "method", "G", "W", "Lambda", "Lambda_G", "compile_s", "verification")}), flush=True)


if __name__ == "__main__":
    main()
