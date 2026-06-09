import os
import sys
from dataclasses import dataclass
from numbers import Real
import time
from qiskit import transpile

# Make src importable when running this file from repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hypercube_channel_pauli import build_hypercube_section52_channel  # type: ignore
from channel_LCU import channel_to_LCU  # type: ignore
from qasm_export import export_openqasm3_baseline  # type: ignore


DATA_DIR = os.path.join(REPO_ROOT, "Data")


@dataclass(frozen=True)
class CaseConfig:
    name: str
    structure: str
    opt: str


# N_POINTS = [8, 12]

N_POINTS = [4,8, 12, 16, 20, 28]
CASE_CONFIGS = [
    CaseConfig(name="basic+No", structure="basic", opt="No"),
    CaseConfig(name="opt+No", structure="opt", opt="No"),
    # User requested ('no', 'matrix-order'); map 'no' -> 'basic' as intended baseline.
    CaseConfig(name="basic+matrix-order", structure="basic", opt="Matrix-order"),
    CaseConfig(name="opt+matrix-order", structure="opt", opt="Matrix-order"),
]


def _gate_metrics(qc) -> dict[str, int|float]:
    start = time.time()
    tqc = transpile(qc, basis_gates=["cx", "u3"], optimization_level=3)
    ops_raw = tqc.count_ops()
    ops = {str(k): int(v) for k, v in ops_raw.items()}
    end = time.time()
    return {
        "num_qubits": int(tqc.num_qubits),
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx": int(ops.get("cx", 0)),
        "u3": int(ops.get("u3", 0)),
        "elapsed_time_sec": round(end - start, 6),
    }


def collect_batch_stats(
    n_points: list[int] | None = None,
) -> list[dict[str, int | str | float]]:
    if n_points is None:
        n_points = N_POINTS

    rows: list[dict[str, int | str | float]] = []
    for n in n_points:
        ensemble = build_hypercube_section52_channel(n)
        kraus_ops = ensemble.channels[0][1]
        total_pauli_terms = int(sum(len(ms.instances) for ms in kraus_ops))

        n_rows: list[dict[str, int | str | float]] = []
        for case in CASE_CONFIGS:
            start_compile = time.time()
            qc, _ = channel_to_LCU(ensemble, structure=case.structure, opt=case.opt)
            end_compile = time.time()

            metrics = _gate_metrics(qc)

            row: dict[str, int | str | float] = {
                "n": n,
                "case": case.name,
                "structure": case.structure,
                "opt": case.opt,
                "kraus_count": len(kraus_ops),
                "pauli_terms_total": total_pauli_terms,
                "compile_time_sec": round(end_compile - start_compile, 6),
                **metrics,
            }

            n_rows.append(row)

        base = next(r for r in n_rows if r["case"] == "basic+No")
        base_depth = float(base["depth"])
        base_size = float(base["size"])
        base_cx = float(base["cx"])
        base_u3 = float(base["u3"])
        base_elapsed_time_sec = float(base["elapsed_time_sec"])
        for row in n_rows:
            row["depth_ratio_vs_basic_no"] = float(row["depth"]) / base_depth if base_depth > 0 else float("nan")
            row["size_ratio_vs_basic_no"] = float(row["size"]) / base_size if base_size > 0 else float("nan")
            row["cx_ratio_vs_basic_no"] = float(row["cx"]) / base_cx if base_cx > 0 else float("nan")
            row["u3_ratio_vs_basic_no"] = float(row["u3"]) / base_u3 if base_u3 > 0 else float("nan")

        rows.extend(n_rows)
    return rows


def build_hypercube_channel_lcu_baseline_circuit(n: int):
    ensemble = build_hypercube_section52_channel(n)
    qc, _ = channel_to_LCU(ensemble, structure="basic", opt="No")
    return qc


def export_hypercube_channel_lcu_baseline_openqasm3(
    n: int = 4,
    out_path: str | os.PathLike | None = None,
) -> dict[str, object]:
    if out_path is None:
        out_path = os.path.join(DATA_DIR, f"test_hypercube_channel_lcu_basic_no_n{n}_baseline.qasm")
    qc = build_hypercube_channel_lcu_baseline_circuit(n)
    return export_openqasm3_baseline(qc, out_path)


def print_table(rows: list[dict[str, int | str | float]]) -> None:
    # headers = [
    #     "n", "case", "structure", "opt", "kraus_count", "pauli_terms_total", "compile_time_sec",
    #     "num_qubits", "depth", "size", "cx", "u3", "elapsed_time_sec",
    #     "depth_ratio_vs_basic_no", "size_ratio_vs_basic_no", "cx_ratio_vs_basic_no", "u3_ratio_vs_basic_no",
    # ]
    headers = [
        "n", "case", "structure", "opt", "compile_time_sec", "elapsed_time_sec",
    ]
    def format_cell(h: str, v: int | str | float) -> str:
        if "_ratio_vs_basic_no" in h and isinstance(v, Real):
            return f"{float(v):.4f}"
        return str(v)

    # Print one table per N to make per-case comparison easier.
    n_values = sorted({int(r["n"]) for r in rows})
    for n in n_values:
        rows_n = [r for r in rows if int(r["n"]) == n]
        widths = {h: len(h) for h in headers}
        for row in rows_n:
            for h in headers:
                txt = format_cell(h, row[h])
                widths[h] = max(widths[h], len(txt))

        def fmt_row(row_vals: dict[str, int | str | float]) -> str:
            items = []
            for h in headers:
                txt = format_cell(h, row_vals[h])
                items.append(txt.rjust(widths[h]))
            return " | ".join(items)

        print(f"\n=== N={n} ===")
        print(fmt_row({h: h for h in headers}))
        print("-+-".join("-" * widths[h] for h in headers))
        for row in rows_n:
            print(fmt_row(row))


def main() -> int:
    rows = collect_batch_stats(n_points=N_POINTS)
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
