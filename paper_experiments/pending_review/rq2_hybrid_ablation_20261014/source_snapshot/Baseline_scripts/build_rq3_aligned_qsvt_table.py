"""Publish QSVT results after checking the shared controls and saved exports.

SELECT and random-density experiments remain frozen. This publisher updates
the existing QSVT table using the seven retained cases and two diagonal
counterexamples, with source hashes and a fresh recount of every NAM/IBM file.
"""

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

from baseline_common import count_qasm_file
from experiment_protocol import source_identity

OUT = ROOT / "Baseline_results/rq3_qrisp_aligned_qsvt_20261014"
INPUTS = ROOT / "Baseline_results/rq3_common_outer_20261014/qsvt_inputs.json"
SELECTS = ROOT / "Baseline_results/rq3_placement_counterexamples_20261014"
SMALL_IDS = ("resource_three_term", "h2_molecular_api", "hubbard_chain2_api",
             "heisenberg_square4_api", "maxcut_triangle3", "cluster3_weighted", "maxcut_cycle4")
LARGE_NAMES = {"qrisp_magnetization_32q": "32 比特磁化算子（官方例子扩展）",
               "qrisp_ising_zero_field_chain33": "33 比特零场 Ising 链（官方例子变体）"}
ROUTES = ("pl_native", "qrisp_native", "ours")
BASES = {"nam": {"rz", "h", "x", "cx"}, "ibm": {"u1", "u2", "u3", "cx"}}
OUTER = "H-C1(U_phi)-C0(U_phi.adjoint)-H"
TOL = 1e-8


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def file_record(path):
    return {"path": relative(path), "sha256": digest(path)}


def check_record(record):
    path = ROOT / record["path"]
    assert digest(path) == record["sha256"], path
    return path


def errors(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if "error" in key and isinstance(item, (int, float)) and not isinstance(item, bool):
                assert math.isfinite(item) and item >= 0, (key, item)
                yield float(item)
            elif isinstance(item, (dict, list)):
                yield from errors(item)
    elif isinstance(value, list):
        for item in value:
            yield from errors(item)


def check_sources(records, *, large=False):
    assert records
    if large:
        for record in records:
            path = check_record(record)
            suffix = (path.relative_to(ROOT) if path.is_relative_to(ROOT) else
                      Path("package", *path.parts[path.parts.index("site-packages") + 1:]))
            assert digest(OUT / "counterexamples/source_snapshot" / suffix) == record["sha256"]
    else:
        with tarfile.open(OUT / "source_snapshot.tar.gz") as archive:
            for record in records:
                check_record(record)
                saved = archive.extractfile(record["path"].lstrip("/"))
                assert saved is not None
                assert sha256(saved.read()).hexdigest() == record["sha256"]


def check_upper(upper):
    assert upper["outer"] == OUTER
    assert upper["queries"] == 10 and upper["projectors"] == 12
    assert upper["PREPARE_has_outer_control"] is False
    assert upper["projector_compute_has_outer_control"] is False
    assert upper["projector_uncompute_has_outer_control"] is False
    assert upper["projector_excludes_query_scratch"] is True
    assert upper["matrix_synthesis_used"] is False
    assert upper["shared_basic_compile"]["pipeline"] == [
        "commute_controlled", "cancel_inverses", "merge_rotations", "remove_barrier"]
    assert upper["shared_basic_compile"]["num_passes"] == 1


def recount(metric, basis, artifacts):
    qasm = metric["qasm"]
    if isinstance(qasm, dict):
        path = check_record(qasm)
        phase_path = check_record(metric["phase_sidecar"])
    else:
        path = ROOT / qasm
        assert digest(path) == metric["qasm_sha256"]
        phase_path = ROOT / metric["global_phase_sidecar"]
        assert digest(phase_path) == metric["global_phase_sidecar_sha256"]
    phase = read(phase_path)["global_phase_radians"]
    assert math.isfinite(phase)
    if "global_phase_radians" in metric:
        assert math.isclose(phase, metric["global_phase_radians"], abs_tol=1e-12)
    count = count_qasm_file(path, gate_set=basis)
    assert (count["metric_total"], count["clifford"]["non_clifford"], count["num_qubits"]) == (
        metric["G"], metric["NC"], metric["W"]), path
    assert count["ops"] == metric["ops"] and count["ops"].get("cx", 0) == metric["CX"]
    assert set(count["ops"]) <= BASES[basis] and count["clifford"]["unknown"] == 0
    artifacts.append({**file_record(path), "basis": basis,
        **{key: metric[key] for key in ("G", "NC", "W", "CX")},
        "phase_sidecar": file_record(phase_path)})


def make_row(case_id, name, results):
    metrics = {route: {basis: {key: result["basis"][basis][key] for key in ("G", "NC", "W", "CX")}
                          for basis in BASES} for route, result in results.items()}
    ratios = {basis: {"pl_over_ours": metrics["pl_native"][basis]["G"] / metrics["ours"][basis]["G"],
                      "qrisp_over_ours": metrics["qrisp_native"][basis]["G"] / metrics["ours"][basis]["G"]}
              for basis in BASES}
    return {"id": case_id, "name": name, "metrics": metrics, "ratios": ratios}


def validate_small(inputs, artifacts):
    summary = read(OUT / "summary.json")
    assert summary["completed"] and summary["passed"]
    assert summary["sources"] == source_identity()
    check_record(summary["smoke"])
    assert tuple(case["id"] for case in summary["cases"]) == SMALL_IDS
    policy = summary["policy"]
    assert policy["outer"] == OUTER and policy["queries"] == 10 and policy["projectors"] == 12
    assert policy["prepare_outer_control"] is False and policy["projector_predicate_outer_control"] is False
    assert policy["ours_placement_reselection"] is False and policy["extra_independent_clean_flags"] == 1
    assert policy["Qrisp_SELECT_native_custom_control"] and policy["PL_SELECT_native_graph_control"]
    assert policy["public_qrisp_QSVT_called"] is False
    assert policy["optimization_level"] == 0 and policy["seed_transpiler"] == 7
    assert policy["qubits_initially_zero"] is False
    check_sources(summary["source_fingerprints"])
    rows, worst, queries, releases = [], 0.0, 0, 0
    for case, saved in zip(inputs["cases"], summary["cases"], strict=True):
        results = saved["routes"]
        assert set(results) == set(ROUTES)
        for route, result in results.items():
            assert result == read(OUT / "cases" / case["id"] / route / "summary.json")
            assert result["passed"] and result["all_complex_output_amplitudes_checked"]
            assert result["source_fingerprints"] == summary["source_fingerprints"]
            for key in ("terms", "alpha", "poly", "angles"):
                assert result[key] == case[key]
            check_upper(result["upper"])
            build = result["build"]
            assert build["prepare_has_outer_control"] is False and build["placement_reselection"] is False
            assert build["matrix_synthesis_used"] is False
            if route == "ours":
                assert build["candidate_count_after_freezing"] == 0 and build["standalone_placement_preserved"]
                assert build["fixed_placement"] == build["standalone_metadata"]["matrix_order"]["placement"]
                assert build["fixed_placement"] == build["controlled_context_metadata"]["matrix_order"]["placement"]
                assert build["standalone_selected_candidate"] == build["controlled_context_metadata"]["matrix_order"]["selected_candidate"]
            elif route == "pl_native":
                assert build["controlled_select_native_graph_before_leaf_expansion"]
            else:
                assert build["native_q_switch_custom_control"] and build["control_enters_tree_root"]
                release = result["release_validation"]
                assert release["passed"] and release["basis_checkpoint_checks"] > 0
                releases += release["basis_checkpoint_checks"]
            domain = result["query_domain_validation"]
            dimension = 1 << len(case["terms"][0][0])
            assert domain["passed"] and domain["system_basis_inputs"] == dimension
            assert domain["source_basis_inputs_checked"] == dimension * domain["all_logical_addresses"]
            assert domain["controlled_basis_inputs_checked"] == 4 * domain["source_basis_inputs_checked"]
            assert set(domain["controlled"]) == {"0", "1"}
            queries += domain["source_basis_inputs_checked"] + domain["controlled_basis_inputs_checked"]
            assert result["projector_validation"]["passed"]
            check_record(result["source_qpy"])
            for record in result["query_artifacts"].values():
                check_record(record)
            recorded_errors = list(errors(result))
            assert recorded_errors and max(recorded_errors) < TOL
            worst = max(worst, *recorded_errors)
            assert result["basis"]["nam"]["W"] == result["basis"]["ibm"]["W"]
            for basis in BASES:
                recount(result["basis"][basis], basis, artifacts)
        rows.append(make_row(case["id"], case["name"], results))
    return rows, {"cases": 7, "routes": 21, "query_basis_inputs": queries,
                  "qrisp_release_checks": releases, "max_error": worst}


def validate_large(inputs, artifacts):
    summary = read(OUT / "counterexamples/summary.json")
    assert summary["passed"] and summary["sources"] == source_identity()
    assert len(summary["rows"]) == 6
    assert {(row["case"], row["method"]) for row in summary["rows"]} == {
        (case, method) for case in LARGE_NAMES for method in ("pl", "qrisp", "ours")}
    check_sources(summary["source_records"], large=True)
    check_record(summary["smoke"])
    check_record(summary["phase_source"])
    assert summary["policy"]["outer"] == OUTER
    assert summary["policy"]["placement_reselection"] is False
    assert summary["policy"]["new_placement_search"] is False
    assert summary["policy"]["query_count"] == 10 and summary["policy"]["projector_count"] == 12
    assert summary["calibration"]["passed"]
    rows, worst, probes = [], 0.0, 0
    for case_id, name in LARGE_NAMES.items():
        source = read(SELECTS / "cases" / case_id / "result.json")
        results = {}
        for method in ("pl", "qrisp", "ours"):
            result = next(row for row in summary["rows"] if (row["case"], row["method"]) == (case_id, method))
            directory = OUT / "counterexamples/cases" / case_id / method
            assert result == read(directory / "result.json") and result["passed"]
            assert result["source_records"] == summary["source_records"]
            case = result["input"]
            assert case["terms"] == source["terms"] and case["K"] == 32 and case["alpha"] == source["alpha"]
            assert case["poly"] == inputs["cases"][0]["poly"] and case["angles"] == inputs["cases"][0]["angles"]
            check_record(case["source_result"])
            check_record(case["source_validation"])
            assert result["placement_reselection"] is False
            check_upper(result["shared_outer"])
            proof = result["controlled_select_validation"]
            assert proof["plain"]["passed"] and proof["controlled"]["passed"]
            assert proof["plain"]["logical_addresses_checked"] == 32
            assert proof["controlled"]["logical_addresses_checked"] == 64
            assert proof["active_control_matches_saved_select"] and proof["inactive_control_exact_identity"]
            assert proof["controlled"]["system_dimension_covered_symbolically"] == 1 << case["n"]
            metadata = result["query_metadata"]
            assert metadata["placement_reselection"] is False
            if method == "ours":
                assert metadata["new_placement_search"] is False
                assert metadata["standalone_placement"] == metadata["controlled_placement"]
                assert result["placement"] == source["methods"]["ours"]["metadata"]["placement"]
                check_record(metadata["source_select"])
            recorded_errors = list(errors(result))
            assert recorded_errors and max(recorded_errors) < TOL
            worst = max(worst, *recorded_errors)
            for basis in BASES:
                metric = result["basis"][basis]
                validation = metric["validation"]
                assert validation["passed"] and validation["basis_representatives"] == 33
                assert validation["all_distinct_spectral_values_covered"]
                assert validation["all_address_signal_amplitudes_per_probe"] == 64
                assert validation["full_system_basis_exhaustively_simulated"] is False
                assert len(validation["records"]) == 33
                probes += 33
                recount(metric, basis, artifacts)
            assert result["basis"]["nam"]["W"] == result["basis"]["ibm"]["W"]
            for filename in ("plain_select.qpy", "controlled_select.qpy", "controlled_query.qpy", "full_qsvt.qpy"):
                assert (directory / filename).exists()
                artifacts.append({**file_record(directory / filename), "kind": "qpy"})
            results[{"pl": "pl_native", "qrisp": "qrisp_native", "ours": "ours"}[method]] = result
        rows.append(make_row(case_id, name, results))
    return rows, {"cases": 2, "routes": 6, "spectral_probes": probes,
                  "address_signal_amplitudes_per_probe": 64, "max_error": worst}


def triple(metric):
    return " / ".join(f"{metric[key]:,}" for key in ("G", "NC", "W"))


def document(rows, checks):
    lines = ["# PennyLane / Qrisp / Ours QSVT 比较（2026-10-14 更新）", "",
        "九个输入统一采用 `H–C₁(U_phi)–C₀(U_phi†)–H`，其中 `C₁`、`C₀` 分别在信号位为 1、0 时激活，"
        "`U_phi` 为给定相位序列的 QSVT 电路。每条完整电路包含 10 次块查询和 12 个投影相位。", "",
        "三条路线采用相同的局部控制规则：PREPARE 及其逆不加外层控制；投影相位用独立的干净标志位计算"
        "逻辑地址是否为零，只控制中间的 `RZ(2φ)`，再精确反计算。Qrisp 使用原生 `from_lcu` / `q_switch` "
        "的树入口控制，PL 在原生 graph SELECT 分解前加入外控，Ours 使用正式 `src/block_encoding.py` 接口"
        "导出并固定独立块编码选定的 placement，外控加入后不重选。三条路线均运行同一组基本编译变换，"
        "包括控制门交换、逆门消除和相邻旋转合并。", "",
        "目标为 `P(H/alpha)`，其中 `P(x)=-x+0.5x³+0.5x⁵`、`alpha=sum(abs(coefficients))`。"
        f"七个保留输入及六个时间顺序相位见 [冻结输入]({INPUTS})；两个大例复用同一多项式与相位。"
        "当前多项式只有奇项，伴随分支用于提取实部。临时工作位和投影标志位在输出时归零。", "",
        "G 为总门数，NC 为非 Clifford 门数，W 为包含工作位、投影标志位及信号位的总比特数。"
        "NAM=`{rz,h,x,cx}`，IBM=`{u1,u2,u3,cx}`；最终均使用 Qiskit opt0、seed 7，不假定系统输入为零。"
        "比例按总门数计算，大于 1 表示 Ours 门数较少。", "",
        "| 例子 | NAM PL G / NC / W | NAM Qrisp G / NC / W | NAM Ours G / NC / W | NAM PL/Ours | NAM Qrisp/Ours | IBM PL G / NC / W | IBM Qrisp G / NC / W | IBM Ours G / NC / W | IBM PL/Ours | IBM Qrisp/Ours |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        cells = [row["name"]]
        for basis in BASES:
            cells.extend(triple(row["metrics"][route][basis]) for route in ROUTES)
            cells.extend(f"{row['ratios'][basis][key]:.4f}" for key in ("pl_over_ours", "qrisp_over_ours"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["",
        "MaxCut 四节点环使用官方 API 的 `H=-2I+0.5*sum_edges(Z_u Z_v)`、`alpha=4`；"
        "全部 16 个基状态的能量均等于割边数的负值。", "",
        "两个大例来自 [Qrisp 官方 QSP Hamiltonian-simulation 文档]"
        "(https://qrisp.eu/reference/Algorithms/generated/qrisp.gqsp.hamiltonian_simulation.html)："
        "磁化算子扩展为 `M₃₂=(1/32)*sum_j Z_j`；Ising 链扩展为 33 比特并取零场，"
        "`H=-0.25*sum_(j=0)^31 Z_j Z_(j+1)`、`alpha=8`。本表将它们作为 QSVT 输入。", "",
        "本轮九例的 PL、Qrisp、Ours 全部重新构造与执行。七个小例检查全部系统基输入的完整复输出、"
        "目标多项式、受控查询的未激活恒等行为和工作位归零；两个大例先对 SELECT 的 32 个地址及 64 个"
        "地址／外控分支作符号 Pauli-Z 证明，覆盖任意系统输入；再对每例全部 33 个不同谱值的代表输入"
        "检查完整 QSVT 的 64 个地址／信号复振幅、目标多项式及归零。"
        "所有 NAM/IBM 保存 QASM 恢复全局相位后均通过验证。"
        f"小例最大记录误差 `{checks['small']['max_error']:.3g}`，大例 `{checks['large']['max_error']:.3g}`。", "",
        f"实验前 [接口 smoke test]({OUT / 'interface_smoke/smoke.json'}) 与 10 月 10 日、12 日存档一致。"
        f"逐例结果：[七个保留例子]({OUT / 'summary.json'})、[两个大例]({OUT / 'counterexamples/summary.json'})；"
        f"[表格生成器]({Path(__file__).resolve()}) 重新计数全部 54 份 NAM/IBM 导出，"
        f"来源指纹与验证数据见 [汇总数据]({OUT / 'comparison.json'})。", ""]
    return "\n".join(lines)


def main():
    smoke = read(OUT / "interface_smoke/smoke.json")
    assert smoke["passed"] and smoke["sources"] == source_identity() and len(smoke["checks"]) == 4
    for fixture in smoke["checks"]:
        assert fixture["passed"] and fixture["archive_count_matches"]
        assert digest(ROOT / fixture["reference"]) == fixture["reference_sha256"]
    inputs = read(INPUTS)
    assert tuple(case["id"] for case in inputs["cases"]) == SMALL_IDS
    artifacts = []
    small_rows, small_checks = validate_small(inputs, artifacts)
    large_rows, large_checks = validate_large(inputs, artifacts)
    rows = small_rows + large_rows
    checks = {"small": small_checks, "large": large_checks,
              "qasm_artifacts_recounted": sum(item.get("basis") in BASES for item in artifacts)}
    assert checks["qasm_artifacts_recounted"] == 54
    text = document(rows, checks)
    qsvt_path = ROOT / "Results/PennyLane_QSVT_tables.md"
    select_path = ROOT / "Results/PennyLane_comparison.md"
    select_text = select_path.read_text()
    old_reference = (f"当前两表由 [统一生成器]({ROOT / 'Baseline_scripts/build_rq3_common_comparison_tables.py'}) "
                     f"重新计数后维护，验证清单见 [汇总数据]({ROOT / 'Baseline_results/rq3_common_outer_20261014/comparison.json'})。")
    new_reference = (f"SELECT 表沿用已验证的结果，计数与验证清单见 [汇总数据]({ROOT / 'Baseline_results/rq3_refresh_20261013/comparison.json'})。"
                     f"QSVT 表的当前生成器为 [统一控制规则生成器]({Path(__file__).resolve()})。")
    assert old_reference in select_text or new_reference in select_text
    select_text = select_text.replace(old_reference, new_reference)
    comparison = {"generated_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "sources": source_identity(), "generator": file_record(Path(__file__)),
        "small_summary": file_record(OUT / "summary.json"),
        "large_summary": file_record(OUT / "counterexamples/summary.json"),
        "smoke": file_record(OUT / "interface_smoke/smoke.json"), "inputs": file_record(INPUTS),
        "policy": read(OUT / "summary.json")["policy"], "qsvt_rows": rows,
        "validation": checks, "artifacts": artifacts,
        "select_results_reused": file_record(ROOT / "Baseline_results/rq3_refresh_20261013/comparison.json"),
        "documents": {relative(qsvt_path): sha256(text.encode()).hexdigest(),
                      relative(select_path): sha256(select_text.encode()).hexdigest()}}
    destination = OUT / "comparison.json"
    temporary = destination.with_suffix(".json.partial")
    temporary.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    qsvt_path.write_text(text)
    select_path.write_text(select_text)
    print(json.dumps({"passed": True, "tables": [relative(select_path), relative(qsvt_path)],
                      "validation": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
