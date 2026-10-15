"""Recount and publish the SELECT and common-outer QSVT comparison."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]

import build_rq3_comparison_tables as previous
from baseline_common import count_qasm_file

OUT = ROOT / "Baseline_results/rq3_common_outer_20261014"
CASE_IDS = ["resource_three_term", "h2_molecular_api", "hubbard_chain2_api",
            "heisenberg_square4_api", "maxcut_triangle3", "cluster3_weighted", "maxcut_cycle4"]
BASES = {"nam": {"rz", "h", "x", "cx"}, "ibm": {"u1", "u2", "u3", "cx"}}
ROUTES = ("pl_native", "qrisp_native", "ours")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_sources(records):
    if isinstance(records, list):
        pairs = [(record["path"], record["sha256"]) for record in records]
    else:
        pairs = [(path, value if isinstance(value, str) else value["sha256"])
                 for path, value in records.items()]
    assert pairs, "Source fingerprints must be recorded."
    for source, expected in pairs:
        path = ROOT / source
        assert digest(path) == expected, f"Source changed: {path}"


def error_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (float, int)) and not isinstance(item, bool) and any(
                    marker in key for marker in ("error", "max_amplitude", "max_abs")):
                assert math.isfinite(item), (key, item)
                yield float(item)
            elif isinstance(item, (dict, list)):
                yield from error_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from error_values(item)


def validate_metric(metric, basis, artifacts):
    path = ROOT / metric["qasm"]
    stats = count_qasm_file(path, gate_set=basis)
    actual = (int(stats["metric_total"]), int(stats["clifford"]["non_clifford"]), int(stats["num_qubits"]))
    expected = (metric["G"], metric["NC"], metric["W"])
    assert actual == expected, (path, actual, expected)
    assert set(stats["ops"]) <= BASES[basis] and stats["clifford"]["unknown"] == 0
    for short, long in (("G", "gate_count"), ("NC", "non_clifford"), ("W", "width")):
        if long in metric:
            assert metric[short] == metric[long]
    phase_path = ROOT / metric["global_phase_sidecar"]
    phase = read(phase_path)["global_phase_radians"]
    assert math.isfinite(phase) and math.isclose(phase, metric["global_phase_radians"], abs_tol=1e-12)
    if "qasm_sha256" in metric:
        assert digest(path) == metric["qasm_sha256"]
    if "global_phase_sidecar_sha256" in metric:
        assert digest(phase_path) == metric["global_phase_sidecar_sha256"]
    artifacts.append({"path": relative(path), "sha256": digest(path), "basis": basis,
                      "G": actual[0], "NC": actual[1], "W": actual[2],
                      "global_phase_sidecar": relative(phase_path), "phase_sha256": digest(phase_path)})


def validate_qsvt(inputs, smoke, artifacts):
    pl = read(OUT / "pennylane_qsvt/summary.json")
    qr = read(OUT / "qrisp_qsvt/summary.json")
    assert [case["id"] for case in inputs["cases"]] == CASE_IDS
    assert [case["id"] for case in pl["cases"]] == CASE_IDS and pl["passed"]
    assert pl["completed"]
    assert [case["id"] for case in qr["cases"]] == CASE_IDS and qr["passed"]
    inherited = read(ROOT / inputs["inherited_inputs"])
    assert digest(ROOT / inputs["inherited_inputs"]) == inputs["inherited_inputs_sha256"]
    assert inputs["cases"][:6] == inherited["cases"][:6]
    assert digest(ROOT / inputs["archive"]) == inputs["archive_sha256"]
    assert inputs["policy"]["query_count"] == 10 and inputs["policy"]["projector_count"] == 12
    assert inputs["policy"]["retired_active_case"] == "fullpauli2_nonseparable"
    for policy in (pl["policy"], qr["basis_policy"]):
        assert policy["optimization_level"] == 0 and policy["seed_transpiler"] == 7
        assert policy["qubits_initially_zero"] is False
    native_audit = read(previous.OUT / "archived_pl_qsvt_correctness.json")
    assert native_audit["completed"] and native_audit["passed"]
    assert native_audit["inputs_sha256"] == inputs["inherited_inputs_sha256"]
    assert native_audit["archive_sha256"] == inputs["archive_sha256"]
    rows, worst = [], 0.0
    route_count = 0
    for case, pl_case, qr_case in zip(inputs["cases"], pl["cases"], qr["cases"]):
        results = {**pl_case["routes"], **qr_case["routes"]}
        assert set(results) == set(ROUTES)
        for route, result in results.items():
            assert result["passed"], (case["id"], route)
            assert result["terms"] == case["terms"]
            assert result["alpha"] == case["alpha"] and result["poly"] == case["poly"]
            assert result["angles"] == case["angles"]
            assert result["queries"] == 10 and result["projectors"] == 12
            check_sources(result["source_fingerprints"])
            source_qpy = result["source_qpy"]
            assert digest(ROOT / source_qpy["path"]) == source_qpy["sha256"]
            if route == "pl_native" and case["id"] != "maxcut_cycle4":
                assert result["reused"] and result["reuse_checks"]["original_projector_equivalent"]
                audit = next(row for row in native_audit["cases"] if row["id"] == case["id"])
                assert audit["passed"]
                for basis, metric in result["basis"].items():
                    saved = audit["basis"][basis]
                    original = case["pl_metrics"][basis]
                    assert metric["archive_member"] == saved["qasm_member"] == original["qasm"]
                    assert digest(ROOT / metric["qasm"]) == saved["qasm_sha256"]
                    assert metric["global_phase_radians"] == saved["global_phase_radians"]
                    assert (metric["G"], metric["NC"], metric["W"]) == (
                        original["gate_count"], original["non_clifford"], original["width"])
            else:
                validations = ([result["query_domain_validation"]] if route != "qrisp_native" else
                               [result["query_validation"]["forward"]])
                if route == "qrisp_native":
                    assert result["query_validation"]["inverse"]["passed"]
                    releases = result["release_validation"]
                    assert releases["passed"] and releases["checkpoint_count"] > 0
                    assert releases["basis_checkpoint_checks"] > 0 and releases["max_error"] < 1e-8
                n = len(case["terms"][0][0])
                for domain in validations:
                    assert domain["passed"] and domain["projector_excludes_scratch_authorized_by_validation"]
                    assert domain["system_basis_inputs"] == 1 << n
                    assert domain["source_basis_inputs_checked"] == (1 << n) * domain["all_logical_addresses"]
                    assert domain["controlled_basis_inputs_checked"] == 4 * domain["source_basis_inputs_checked"]
                    assert set(domain["controlled"]) == {"0", "1"}
                assert result["checks"]["physical_polynomial_block_error"] < 1e-8
                assert result["checks"]["clean_scratch_output_error"] < 1e-8
            assert set(result["basis"]) == set(BASES)
            assert result["basis"]["nam"]["W"] == result["basis"]["ibm"]["W"]
            evidence = [result.get(key, {}) for key in (
                "checks", "basis", "query_domain_validation", "query_validation", "release_validation", "reuse_checks")]
            errors = [error for item in evidence for error in error_values(item)]
            assert errors and max(errors) < 1e-8, (case["id"], route, max(errors, default=None))
            worst = max(worst, *errors)
            for basis in BASES:
                validate_metric(result["basis"][basis], basis, artifacts)
            route_count += 1
        metrics = {route: results[route]["basis"] for route in ROUTES}
        ratios = {basis: {"pl_over_ours": metrics["pl_native"][basis]["G"] / metrics["ours"][basis]["G"],
                          "qrisp_over_ours": metrics["qrisp_native"][basis]["G"] / metrics["ours"][basis]["G"]}
                  for basis in BASES}
        rows.append({"id": case["id"], "name": case["name"], "metrics": metrics, "ratios": ratios})
    cycle = inputs["cases"][-1]
    assert cycle["alpha"] == 4 and len(cycle["terms"]) == 5
    cut = cycle["case_definition"]["basis_state_cut_validation"]
    assert cut["checked_states"] == 16 and cut["diagonal_equals_negative_cut_value"]
    assert cut["max_abs_error"] == 0.0
    assert smoke["sources"] == read(previous.OUT / "smoke.json")["sources"]
    return rows, {"cases": 7, "routes": route_count, "max_error": worst,
                  "query_count_per_route": 10, "projector_count_per_route": 12,
                  "cycle4_basis_state_cut_check": cut}


def triple(metric):
    return " / ".join(f"{metric[key]:,}" for key in ("G", "NC", "W"))


def select_document(summary, checks):
    text = previous.select_document(summary, checks)
    return text.replace(
        f"表格由 [统一生成器]({ROOT / 'Baseline_scripts/build_rq3_comparison_tables.py'}) "
        f"从完整结果重新计数后生成，验证清单见 [汇总数据]({previous.OUT / 'comparison.json'})。",
        f"当前两表由 [统一生成器]({ROOT / 'Baseline_scripts/build_rq3_common_comparison_tables.py'}) "
        f"重新计数后维护，验证清单见 [汇总数据]({OUT / 'comparison.json'})。")


def qsvt_document(rows, checks):
    lines = [
        "# PennyLane / Qrisp / Ours QSVT 比较（2026-10-14 更新）", "",
        "七个输入统一采用相同的 QSVT 相位序列及 `H–C₁(U_phi)–C₀(U_phi†)–H` 外层；"
        "PL `PrepSelPrep`、Qrisp `BlockEncoding.from_lcu` 和 Ours 分别提供底层块查询。"
        "每条完整电路均有 10 次块查询、12 个投影相位，差异来自块编码与多控制门合成采用的优化。", "",
        "目标为 `P(H/alpha)`，其中 `P(x)=-x+0.5x³+0.5x⁵`、`alpha=sum(abs(coefficients))`。"
        f"完整有序输入和六个时间顺序相位见 [冻结输入]({OUT / 'qsvt_inputs.json'})。"
        "当前多项式只有奇项；伴随分支用于实系数多项式的相干提取。投影相位只检查逻辑地址位，"
        "临时工作位在查询后归零，其宽度仍计入 W。", "",
        "G 为总门数，NC 为非 Clifford 门数，W 为含工作位的总比特数。"
        "NAM=`{rz,h,x,cx}`，IBM=`{u1,u2,u3,cx}`；最终均使用 Qiskit opt0、seed 7，"
        "不假定系统输入为零。比例按总门数计算，大于 1 表示 Ours 门数较少。", "",
        "| 例子 | NAM PL G / NC / W | NAM Qrisp G / NC / W | NAM Ours G / NC / W | NAM PL/Ours | NAM Qrisp/Ours | IBM PL G / NC / W | IBM Qrisp G / NC / W | IBM Ours G / NC / W | IBM PL/Ours | IBM Qrisp/Ours |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = [row["name"]]
        for basis in BASES:
            cells += [triple(row["metrics"][route][basis]) for route in ROUTES]
            cells += [f"{row['ratios'][basis][key]:.3f}" for key in ("pl_over_ours", "qrisp_over_ours")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["",
        "MaxCut 四节点环使用官方 API 的 `H=-2I+0.5*sum_edges(Z_u Z_v)`、`alpha=4`；"
        "全部 16 个基状态的能量均等于割边数的负值。", "",
        "PL 前六例复用已验证的原生电路存档，重新核对输入、构造与 QASM 计数；"
        "Ours 七例、Qrisp 七例及四节点环的 PL 路线均实际运行。新电路检查全部系统基输入的完整复输出、"
        "目标多项式、查询工作位归零、受控查询的未激活恒等行为，以及 NAM/IBM 保存 QASM 的相位恢复。"
        f"全部路线验证通过，最大记录误差为 `{checks['max_error']:.3g}`。", "",
        f"实验前 [接口 smoke test]({OUT / 'smoke.json'}) 与 10 月 10 日、12 日存档一致。"
        f"逐例结果：[PL 与 Ours]({OUT / 'pennylane_qsvt/summary.json'})、"
        f"[Qrisp]({OUT / 'qrisp_qsvt/summary.json'})；"
        f"[统一生成器]({ROOT / 'Baseline_scripts/build_rq3_common_comparison_tables.py'}) "
        f"重新计数全部导出，来源与验证证据见 [汇总数据]({OUT / 'comparison.json'})。", "",
    ]
    return "\n".join(lines)


def main():
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and len(smoke["checks"]) == 4
    for source, expected in smoke["sources"].items():
        assert digest(ROOT / "src" / source) == expected, source
    for fixture in smoke["checks"]:
        assert fixture["passed"] and fixture["archive_count_matches"]
        assert digest(ROOT / fixture["reference"]) == fixture["reference_sha256"]
    inputs = read(OUT / "qsvt_inputs.json")
    assert digest(ROOT / inputs["generator"]) == inputs["generator_sha256"]
    artifacts = []
    select, select_checks = previous.validate_select(smoke, artifacts)
    select_path = ROOT / "Results/PennyLane_comparison.md"
    previous_comparison = read(previous.OUT / "comparison.json")
    original_select = previous.select_document(select, select_checks)
    assert sha256(original_select.encode()).hexdigest() == previous_comparison["documents"][relative(select_path)]
    select_text = select_document(select, select_checks)
    assert select_path.read_text() in (original_select, select_text)
    rows, qsvt_checks = validate_qsvt(inputs, smoke, artifacts)
    qsvt_path = ROOT / "Results/PennyLane_QSVT_tables.md"
    document = qsvt_document(rows, qsvt_checks)
    previous_path = OUT / "previous_tables.json"
    if not previous_path.exists():
        write_json(previous_path, {relative(path): {"sha256": digest(path), "text": path.read_text()}
                                   for path in (select_path, qsvt_path)})
    snapshots = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__),
                 ROOT / "Baseline_scripts/prepare_rq3_common_inputs.py",
                 ROOT / "Baseline_scripts/run_rq3_common_pl_qsvt.py",
                 ROOT / "Baseline_scripts/run_rq3_common_qrisp_qsvt.py",
                 ROOT / "Baseline_scripts/run_rq3_pennylane_qsvt.py",
                 ROOT / "Baseline_scripts/run_rq3_qrisp_qsvt.py",
                 ROOT / "Baseline_scripts/build_rq3_comparison_tables.py",
                 ROOT / "Baseline_scripts/pennylane_lcu_adapter.py",
                 ROOT / "Baseline_scripts/baseline_common.py",
                 ROOT / "Baseline_scripts/smoke_block_encoding_archive.py",
                 ROOT / "Benchmark_guide/pennylane_lcu_benchmarks.md",
                 OUT / "qsvt_inputs.json", OUT / "smoke.json"]
    source_manifest = {relative(path): digest(path) for path in snapshots}
    with tarfile.open(OUT / "source_snapshot.tar.gz", "w:gz") as saved:
        for path in snapshots:
            saved.add(path, arcname=relative(path), recursive=False)
    write_json(OUT / "source_manifest.json", source_manifest)
    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "sources": smoke["sources"], "policy": inputs["policy"],
        "select_rows": select["rows"], "qsvt_rows": rows,
        "validation": {"select": select_checks, "qsvt": qsvt_checks,
                       "qasm_artifacts_recounted": len(artifacts)},
        "artifacts": artifacts, "source_manifest": relative(OUT / "source_manifest.json"),
        "documents": {relative(select_path): sha256(select_text.encode()).hexdigest(),
                      relative(qsvt_path): sha256(document.encode()).hexdigest()},
        "retained_correctness_audits": {
            "rq1": "Baseline_results/rq1_correctness_audit_20261013/summary.json",
            "tfim3": "Baseline_results/rq3_refresh_20261013/tfim3_phase_audit/summary.json",
        },
    }
    write_json(OUT / "comparison.json", comparison)
    select_path.write_text(select_text)
    qsvt_path.write_text(document)
    print(json.dumps({"passed": True, "tables": [relative(select_path), relative(qsvt_path)],
                      "validation": comparison["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
