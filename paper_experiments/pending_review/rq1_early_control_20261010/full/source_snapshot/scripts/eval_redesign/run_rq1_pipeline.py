#!/usr/bin/env python3
"""Run Raw, Qrisp+F*, and Ours with shared inputs and NAM opt0 lowering only.

Examples:
  python scripts/eval_redesign/run_rq1_pipeline.py --pipeline all --example TFIM-3 --verify
  python scripts/eval_redesign/run_rq1_pipeline.py --pipeline qrisp --example HRW-4 --verify
  python scripts/eval_redesign/run_rq1_pipeline.py --pipeline ours --input-json channel.json

Input JSON contains input_kraus (or canonical_kraus), each row a list of
[Pauli_label, magnitude, phase_real, phase_imag] terms. An existing validation
results.json using this schema is also accepted. Output is JSON and NAM QASM.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / folder) for folder in ("src", "scripts", "Baseline_scripts")]
from experiment_protocol import REWRITE_DEFAULTS, DEFAULT_TRANSPILE_SEED


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", choices=("raw", "qrisp", "ours", "all"), default="all")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--example", help="TFIM-N or HRW-N; N is the system qubit count")
    inputs.add_argument("--input-json", type=Path)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_experiments/pending_review/rq1_nam_opt0")
    parser.add_argument("--verify", action="store_true", help="Verify the full unnormalized Choi matrix (small cases)")
    parser.add_argument("--verify-auto", action="store_true", help="Full Choi check when system <=4 and total <=18; record skips explicitly")
    parser.add_argument("--rewrite-method", choices=("gatecost_rewrite_search", "rewrite_search"), default=REWRITE_DEFAULTS.method)
    parser.add_argument("--rewrite-strategy", choices=("beam", "greedy"), default=REWRITE_DEFAULTS.strategy)
    parser.add_argument("--beam-width", type=int, default=REWRITE_DEFAULTS.beam_width)
    parser.add_argument("--max-steps", type=int, default=REWRITE_DEFAULTS.max_steps)
    parser.add_argument("--tol", type=float, default=REWRITE_DEFAULTS.tol)
    parser.add_argument("--seed-transpiler", type=int, default=DEFAULT_TRANSPILE_SEED)
    args = parser.parse_args()
    # Configure optional-package caches before importing benchmark modules.
    # A remote run therefore keeps temporary files under its output workspace.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-", dir=args.out_dir.resolve()) as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(cache) / "numba"))
        run(args, parser)


def run(args, parser):
    from qiskit import qasm2
    from baseline_common import count_qasm_file
    from experiment_protocol import lower_and_count
    from rq1_pipeline import canonicalize, deserialize_kraus, serialize_kraus, build_pipeline, verify_channel
    if args.example:
        try:
            model_name, size = args.example.upper().split("-")
            size = int(size)
            if size < 1 or model_name not in ("TFIM", "HRW"):
                raise ValueError
        except ValueError:
            parser.error("--example must be TFIM-N or HRW-N with positive integer N")
        if model_name == "TFIM":
            from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel
            raw, model = build_periodic_tfim_channel(size, args.delta_t)
        else:
            from run_hypercube_n4_ours_nam import build_hypercube_channel
            raw, model = build_hypercube_channel(size), {"family": "HRW section 5.2", "num_qubits": size}
        name = f"{model_name}-{size}"
    else:
        payload = json.loads(args.input_json.read_text())
        rows = payload.get("input_kraus", payload.get("canonical_kraus"))
        if rows is None:
            parser.error("Input JSON must contain input_kraus or canonical_kraus")
        raw = deserialize_kraus(rows)
        name = args.input_json.stem
        model = {**payload.get("model", {}), "input_json": str(args.input_json.resolve()),
                 "input_json_sha256": sha256(args.input_json.read_bytes()).hexdigest()}
    canonical, canonical_info = canonicalize(raw)
    selected = ("Raw", "Qrisp", "Ours") if args.pipeline == "all" else ({"raw":"Raw", "qrisp":"Qrisp", "ours":"Ours"}[args.pipeline],)
    output = args.out_dir.resolve() / name
    for pipeline in selected:
        if (output / pipeline.lower() / "results.json").exists():
            parser.error(f"Results already exist for {name}/{pipeline}; choose a fresh --out-dir.")
    for pipeline in selected:
        print(f"Building {name}: {pipeline}", flush=True)
        circuit, registers, assembly = build_pipeline(
            canonical, pipeline, rewrite_method=args.rewrite_method, strategy=args.rewrite_strategy,
            beam_width=args.beam_width, max_steps=args.max_steps, tol=args.tol,
        )
        final, metrics = lower_and_count(circuit, rq="RQ1", basis="nam", seed_transpiler=args.seed_transpiler)
        metrics["settings"]["rewrite"] = assembly["rewrite"]
        lam = float(circuit.metadata["subnormalization_factor"])
        directory = output / pipeline.lower()
        directory.mkdir(parents=True, exist_ok=True)
        qasm_path = directory / "channel_nam_opt0.qasm"
        qasm2.dump(final, qasm_path)
        recount = count_qasm_file(qasm_path, gate_set="nam")
        assert recount["metric_total"] == metrics["G"]
        do_verify = args.verify or (args.verify_auto and len(registers[-1]) <= 4 and final.num_qubits <= 18)
        verification = (verify_channel(final, registers, raw) if do_verify else
                        {"status": "skipped", "reason": "automatic full Choi budget: <=4 system / <=18 total qubits"}
                        if args.verify_auto else {"status": "not requested"})
        data = {"example": name, "model": model, "pipeline": pipeline,
                "input_kraus": serialize_kraus(raw), "canonical_kraus": serialize_kraus(canonical),
                "canonicalization": canonical_info, "assembly": assembly, "metrics": metrics,
                "Lambda": lam, "Lambda_G": lam * metrics["G"],
                "compile_time_s": canonical_info["time_s"] + assembly["rewrite"]["time_s"] + assembly["assembly_time_s"] + metrics["lowering_time_s"],
                "registers": registers, "non_clifford": recount["clifford"]["non_clifford"],
                "qasm": str(qasm_path), "qasm_sha256": sha256(qasm_path.read_bytes()).hexdigest(),
                "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "verification": verification}
        (directory / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"example":name,"pipeline":pipeline,"G":metrics["G"],"W":metrics["W"],"Lambda":lam,
                          "verification":data["verification"],"results":str(directory / "results.json")}), flush=True)
        if do_verify and not data["verification"]["passed"]:
            raise SystemExit(f"Channel verification failed for {name}/{pipeline}")


if __name__ == "__main__":
    main()
