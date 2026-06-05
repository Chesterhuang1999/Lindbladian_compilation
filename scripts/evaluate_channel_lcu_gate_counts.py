"""Channel-LCU gate-count comparison split from ``evaluate.ipynb``."""

from __future__ import annotations

import argparse
import csv

import numpy as np
from qiskit import transpile

from evaluate_common import (
    RESULTS_DIR,
    Lindblad_to_channel,
    Lindbladian,
    build_periodic_tfim_lindbladian_pauli,
    channel_ensemble,
    channel_to_LCU,
    compression_ratio,
    remove_save_instructions,
)


def normalize_setting(structure: str, opt: str):
    structure_map = {"basic": "basic", "opt": "opt", "no": "basic"}
    opt_map = {"no": "No", "matrix-order": "Matrix-order", "ctrl-line": "Ctrl-line"}

    structure_l = structure.strip().lower()
    opt_l = opt.strip().lower()
    if structure_l not in structure_map:
        raise ValueError(f"Unsupported structure: {structure}")
    if opt_l not in opt_map:
        raise ValueError(f"Unsupported opt: {opt}")
    return structure_map[structure_l], opt_map[opt_l]


def gate_count_channel_lcu_four_settings(num_qubits: int, delta_t: float = 0.1):
    gamma = np.sqrt(0.1) / 2
    h_terms, l_terms = build_periodic_tfim_lindbladian_pauli(num_qubits, gamma)
    tfim_lind = Lindbladian(h_terms, l_terms)
    channel_lind, _, _ = Lindblad_to_channel(tfim_lind, float(delta_t))
    single_channel = channel_lind.channels[0][1]

    raw_settings = [
        ("basic", "No"),
        ("opt", "No"),
        ("basic", "matrix-order"),
        ("opt", "matrix-order"),
    ]
    settings = [normalize_setting(structure, opt) for structure, opt in raw_settings]

    counts = {}
    for structure, opt in settings:
        tag = f"({structure}, {opt})"
        qc, _ = channel_to_LCU(channel_ensemble([single_channel]), structure=structure, opt=opt)
        qc_clean = remove_save_instructions(qc)
        qc_t = transpile(qc_clean, basis_gates=["cx", "u3"], optimization_level=2)
        ops = qc_t.count_ops()
        counts[tag] = {
            "cx": int(ops.get("cx", 0)),
            "u3": int(ops.get("u3", 0)),
            "depth": int(qc_t.depth()),
            "size": int(qc_t.size()),
            "num_qubits": int(qc_t.num_qubits),
        }

    base = counts["(basic, No)"]
    for values in counts.values():
        values["cx_compression_ratio"] = round(compression_ratio(values["cx"], base["cx"]), 4)
        values["u3_compression_ratio"] = round(compression_ratio(values["u3"], base["u3"]), 4)
        values["depth_compression_ratio"] = round(compression_ratio(values["depth"], base["depth"]), 4)
        values["size_compression_ratio"] = round(compression_ratio(values["size"], base["size"]), 4)

    print(f"=== Transpiled metrics when N = {num_qubits} (basis_gates=['cx','u3']) ===")
    for tag, values in counts.items():
        print(
            f"{tag}: qubits={values['num_qubits']}, depth={values['depth']}, "
            f"size={values['size']}, cx={values['cx']}, u3={values['u3']}, "
            f"depth_ratio={values['depth_compression_ratio']:.4f}, "
            f"size_ratio={values['size_compression_ratio']:.4f}, "
            f"cx_ratio={values['cx_compression_ratio']:.4f}, "
            f"u3_ratio={values['u3_compression_ratio']:.4f}"
        )
    return counts


def export_gate_counts_new(
    all_counts: dict,
    out_path=None,
):
    if out_path is None:
        out_path = RESULTS_DIR / "TFIM_cost" / "TFIM_gate_counts_N_new.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "N",
        "option",
        "num_qubits",
        "depth",
        "size",
        "cx",
        "u3",
        "depth_compression_ratio",
        "size_compression_ratio",
        "cx_compression_ratio",
        "u3_compression_ratio",
    ]
    rows = []
    for n in sorted(all_counts.keys()):
        for option, values in all_counts[n].items():
            rows.append({
                "N": int(n),
                "option": option,
                "num_qubits": int(values["num_qubits"]),
                "depth": int(values["depth"]),
                "size": int(values["size"]),
                "cx": int(values["cx"]),
                "u3": int(values["u3"]),
                "depth_compression_ratio": round(float(values["depth_compression_ratio"]), 4),
                "size_compression_ratio": round(float(values["size_compression_ratio"]), 4),
                "cx_compression_ratio": round(float(values["cx_compression_ratio"]), 4),
                "u3_compression_ratio": round(float(values["u3_compression_ratio"]), 4),
            })

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")
    return {"rows": len(rows), "path": str(out_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", type=int, nargs="*", default=[4, 8, 12])
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    all_gate_counts = {
        n: gate_count_channel_lcu_four_settings(n, delta_t=args.delta_t)
        for n in args.n_list
    }
    if not args.no_export:
        export_gate_counts_new(all_gate_counts)
    return all_gate_counts


if __name__ == "__main__":
    main()

