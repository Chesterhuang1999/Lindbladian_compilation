#!/usr/bin/env python3
"""Inspect the first random-coloring Kraus operator before/after rewrite."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts", ROOT / "Baseline_scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from channel_IR import channel as ChannelIR  # type: ignore  # noqa: E402
from run_random_coloring_channel_lcu_ours_nam import (  # type: ignore  # noqa: E402
    build_random_coloring_channel,
)


RESULT_ROOT = ROOT / "Baseline_results" / "random_coloring_first_kraus_forms_v1"


def effective_terms(ms: Any) -> list[tuple[str, complex]]:
    rows = []
    for instance, coeff in copy.deepcopy(ms).instances:
        label = getattr(instance, "expr", str(instance))
        phase = complex(getattr(instance, "phase", 1.0))
        rows.append((str(label), complex(coeff) * phase))
    rows.sort(key=lambda item: (item[0].count("I"), item[0]))
    return rows


def format_complex(value: complex) -> str:
    real = 0.0 if abs(value.real) < 1e-14 else value.real
    imag = 0.0 if abs(value.imag) < 1e-14 else value.imag
    if imag == 0.0:
        return f"{real:.12g}"
    if real == 0.0:
        return f"{imag:.12g}j"
    sign = "+" if imag >= 0.0 else ""
    return f"{real:.12g}{sign}{imag:.12g}j"


def pauli_expression(rows: list[tuple[str, complex]], *, max_terms: int | None = None) -> str:
    selected = rows if max_terms is None else rows[:max_terms]
    return " + ".join(f"({format_complex(coeff)}) {label}" for label, coeff in selected)


def rewrite_channel(kraus_ops: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    source = ChannelIR(copy.deepcopy(kraus_ops))
    source.zero_elim()
    result = source.rewrite_search(strategy="greedy", beam_width=3, max_steps=50, tol=1e-12)
    rewritten = ChannelIR(copy.deepcopy(kraus_ops))
    rewritten.apply_rewrite_result(result, tol=1e-12)
    rewritten.zero_elim()
    return rewritten.kraus_ops, result


def inspect_graph(graph: str) -> dict[str, Any]:
    kraus_ops, model_info = build_random_coloring_channel(
        4,
        graph,
        q=4,
        tol=1e-12,
        max_total_pauli_terms=None,
        max_terms_per_kraus=None,
        mix_colors=False,
    )
    rewritten_kraus_ops, rewrite_result = rewrite_channel(kraus_ops)
    raw = effective_terms(kraus_ops[0])
    rewritten = effective_terms(rewritten_kraus_ops[0])
    return {
        "graph": graph,
        "first_kraus_index": 0,
        "first_kraus_parameters": {"vertex": 0, "proposed_color": 0},
        "model_info": model_info,
        "raw_term_count": len(raw),
        "rewritten_term_count": len(rewritten),
        "raw_terms": [{"pauli": label, "coeff": format_complex(coeff)} for label, coeff in raw],
        "rewritten_terms": [{"pauli": label, "coeff": format_complex(coeff)} for label, coeff in rewritten],
        "raw_expression": pauli_expression(raw),
        "rewritten_expression": pauli_expression(rewritten),
        "rewrite": {
            "initial_support": int(rewrite_result["initial_support"]),
            "final_support": int(rewrite_result["final_support"]),
            "steps": len(rewrite_result["steps"]),
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = ["# Random Coloring First Kraus Forms", ""]
    for graph, data in summary.items():
        lines.extend(
            [
                f"## {graph}",
                "",
                f"- first Kraus: index `0`, `(vertex=0, proposed_color=0)`",
                f"- graph edges: `{data['model_info']['edges']}`",
                f"- raw terms: `{data['raw_term_count']}`",
                f"- rewritten terms: `{data['rewritten_term_count']}`",
                f"- rewrite support: `{data['rewrite']['initial_support']} -> {data['rewrite']['final_support']}`",
                "",
                "### Raw first Kraus",
                "",
                "| coeff | Pauli |",
                "| ---: | --- |",
            ]
        )
        for row in data["raw_terms"]:
            lines.append(f"| `{row['coeff']}` | `{row['pauli']}` |")
        lines.extend(["", "### Rewritten first Kraus", "", "| coeff | Pauli |", "| ---: | --- |"])
        for row in data["rewritten_terms"]:
            lines.append(f"| `{row['coeff']}` | `{row['pauli']}` |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {graph: inspect_graph(graph) for graph in ("linear", "cycle")}
    (RESULT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, RESULT_ROOT / "report.md")
    compact = {
        graph: {
            "raw_term_count": data["raw_term_count"],
            "rewritten_term_count": data["rewritten_term_count"],
            "raw_expression": data["raw_expression"],
            "rewritten_expression": data["rewritten_expression"],
        }
        for graph, data in summary.items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
