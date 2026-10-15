"""Publish the two RQ3 tables only from complete, validated refresh results."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "Baseline_scripts")]
OUT = ROOT / "Baseline_results/rq3_refresh_20261013"
BASES = {"nam": {"rz", "h", "x", "cx"}, "ibm": {"u1", "u2", "u3", "cx"}}
SELECT_CASES = ["HRW_8", "HRW_28", "random_coloring_linear_n4_q4",
                "random_coloring_cycle_n4_q4", "repetition_dqe_d3", "TFIM_3_A0",
                "TFIM_8_A0", "random_pauli_4q_20t", "random_pauli_8q_80t", "full_pauli_n6"]
QSVT_CASES = ["resource_three_term", "h2_molecular_api", "hubbard_chain2_api",
              "heisenberg_square4_api", "maxcut_triangle3", "cluster3_weighted",
              "fullpauli2_nonseparable"]


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def same_number(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def source_matches(path, expected):
    if digest(path) == expected:
        return
    # The strengthened inactive-state smoke check preserved its original input.
    if path == OUT / "smoke.json" and digest(OUT / "smoke_initial.json") == expected:
        return
    raise ValueError(f"Source fingerprint changed: {path}")


def recount(text, basis, metric, path, artifacts):
    from baseline_common import count_qasm_text

    stats = count_qasm_text(text, gate_set=basis)
    expected = (metric.get("G", metric.get("gate_count")),
                metric.get("NC", metric.get("non_clifford")),
                metric.get("W", metric.get("width")))
    actual = (stats["metric_total"], stats["clifford"]["non_clifford"], stats["num_qubits"])
    assert actual == expected, (path, actual, expected)
    assert set(stats["ops"]) <= BASES[basis], (path, stats["ops"])
    assert stats["clifford"]["unknown"] == 0, path
    artifacts.append({"path": path, "sha256": sha256(text.encode()).hexdigest(),
                      "basis": basis, "G": actual[0], "NC": actual[1], "W": actual[2]})


def metric_artifact(metric, basis, artifacts):
    path = ROOT / metric["qasm"]
    recount(path.read_text(), basis, metric, rel(path), artifacts)
    if "global_phase_sidecar" in metric:
        sidecar = ROOT / metric["global_phase_sidecar"]
        same_number(read(sidecar)["global_phase_radians"], metric["global_phase_radians"])
    else:
        assert math.isfinite(metric["global_phase"])


def validate_select(smoke, artifacts):
    inputs = read(ROOT / "Baseline_results/pennylane_lcu_v1/inputs.json")
    summary = read(OUT / "select/summary.json")
    historical = read(ROOT / "Baseline_results/pennylane_lcu_comparison/comparison.json")
    assert [case["name"] for case in inputs["cases"]] == SELECT_CASES
    assert sum(len(case["inputs"]) for case in inputs["cases"]) == 135
    assert summary["completed"] is True
    assert summary["policy"]["sources"] == smoke["sources"]
    assert summary["policy"]["qubits_initially_zero"] is False
    assert summary["policy"]["qiskit_optimization_level"] == 0
    assert summary["policy"]["input_manifest_sha256"] == digest(
        ROOT / "Baseline_results/pennylane_lcu_v1/inputs.json")
    assert len(summary["rows"]) == 20
    assert {(row["case"], row["basis"]) for row in summary["rows"]} == {
        (case, basis) for case in SELECT_CASES for basis in BASES}
    max_symbolic_error, max_lowering_error, checked_addresses = 0.0, 0.0, 0
    for case in inputs["cases"]:
        result = read(OUT / "select" / case["name"] / "result.json")
        assert result["passed"] and result["sources"] == smoke["sources"]
        assert result["runner_sha256"] == summary["policy"]["runner_sha256"]
        assert [(row["id"], row["input_sha256"]) for row in result["inputs"]] == [
            (row["id"], row["input_sha256"]) for row in case["inputs"]]
        for entry in result["inputs"]:
            assert entry["passed"]
            checks = entry["verification"]
            max_symbolic_error = max(max_symbolic_error, checks["symbolic_max_error"])
            max_lowering_error = max([max_lowering_error, *checks.get("lowering_max_errors", {}).values()])
            checked_addresses += checks["occupied_addresses_checked"]
            for basis in BASES:
                metric_artifact(entry["basis"][basis], basis, artifacts)
        for basis in BASES:
            row = next(row for row in summary["rows"] if row["case"] == case["name"] and row["basis"] == basis)
            assert row["status"] == "ok" and row["inputs"] == len(case["inputs"])
            prior = next(row for row in historical["main_rows"] if row["case"] == case["name"]
                         and row["basis"] == basis and row["scope"] == "select" and row["profile"] == "graph_work")
            assert row["pl"] == prior["pennylane"] and row["qrisp"] == prior["references"]["qrisp"]
            metrics = [entry["basis"][basis] for entry in result["inputs"]]
            for key in ("G", "NC", "W", "CX"):
                expected = (statistics.fmean(item[key] for item in metrics) if case["aggregation"] == "sample_mean"
                            else max(item[key] for item in metrics) if key == "W"
                            else sum(item[key] for item in metrics))
                same_number(row["ours"][key], expected)
            for method in ("pl", "qrisp"):
                same_number(row[f"{method}_over_ours"], row[method]["gate_count"] / row["ours"]["G"])
    return summary, {"cases": 10, "inputs": 135, "occupied_addresses_checked": checked_addresses,
                     "max_symbolic_error": max_symbolic_error, "representative_max_lowering_error": max_lowering_error}


def validate_qsvt(smoke, artifacts):
    inputs = read(OUT / "qsvt_inputs.json")
    pl = read(OUT / "pennylane_qsvt/summary.json")
    qr = read(OUT / "qrisp_qsvt/summary.json")
    assert [case["id"] for case in inputs["cases"]] == QSVT_CASES
    assert [case["id"] for case in pl["runs"]] == QSVT_CASES and pl["passed"]
    assert [case["id"] for case in qr["cases"]] == QSVT_CASES
    assert pl["only_ours_injected_routes_executed"] is True
    assert pl["basis_policy"]["optimization_level"] == 0 and pl["basis_policy"]["qubits_initially_zero"] is False
    assert qr["downstream"]["optimization_level"] == 0 and qr["downstream"]["qubits_initially_zero"] is False
    for source in pl["source_fingerprints"]:
        source_matches(ROOT / source["path"], source["sha256"])
    for path, value in qr["source_hashes"].items():
        source_matches(ROOT / path, value)
    for name, value in smoke["sources"].items():
        source_matches(ROOT / "src" / name, value)
    archive = ROOT / inputs["archive"]
    assert digest(archive) == inputs["archive_sha256"]
    rows, worst = [], 0.0
    with tarfile.open(archive) as saved:
        for case, pl_case, qr_case in zip(inputs["cases"], pl["runs"], qr["cases"]):
            assert pl_case["status"] == "ok" and pl_case["passed"]
            assert qr_case["status"] == "passed"
            for result in (pl_case, qr_case):
                assert result["terms"] == case["terms"] and result["alpha"] == case["alpha"] and result["poly"] == case["poly"]
            assert pl_case["angles"] == case["angles"] and pl_case["queries"] == 10 and pl_case["projectors"] == 12
            assert pl_case["archived_pl_metrics"] == case["pl_metrics"] and not pl_case["pl_native_rerun"]
            assert pl_case["outer_function_preserved"] and pl_case["verification"]["physical_block_compared_without_real_projection"]
            assert pl_case["build"]["prepare_has_outer_control"] is False
            assert pl_case["build"]["symbolic_select_controlled_before_expansion"] is True
            assert pl_case["compile"]["decomposition"] == "graph" and pl_case["compile"]["num_passes"] == 1
            worst = max(worst, *pl_case["checks"].values())
            for basis in BASES:
                archived = case["pl_metrics"][basis]
                member = archived["qasm"]
                handle = saved.extractfile(member)
                assert handle is not None, member
                recount(handle.read().decode(), basis, archived,
                        f"{rel(archive)}::{member}", artifacts)
                ours = pl_case["basis"][basis]
                metric_artifact(ours, basis, artifacts)
                worst = max(worst, *(ours[key] for key in (
                    "phase_sensitive_lowering_error", "phase_sensitive_reference_error",
                    "physical_polynomial_block_error", "saved_qasm_roundtrip_error")))
            rows.append({"id": case["id"], "name": case["name"], "framework": "PennyLane",
                         "baseline": case["pl_metrics"], "ours": pl_case["basis"],
                         "ratios": {basis: case["pl_metrics"][basis]["gate_count"] / pl_case["basis"][basis]["gate_count"] for basis in BASES}})
            for route in ("qrisp_native", "ours"):
                result = qr_case["routes"][route]
                assert result["status"] == "passed"
                metadata = result["metadata"]
                assert metadata["query_count"] == 5 and metadata["rescale"] is False
                assert metadata["qrisp_angles"] == case["angles"][::-1]
                same_number(metadata["input_alpha"], case["alpha"])
                same_number(metadata["output_alpha"], 1.0)
                assert metadata["ports"]["quantum_session_compile_used"] is False
                checks = result["verification"]
                assert checks["system_basis_inputs"] == 2 ** len(metadata["ports"]["system_qubits"])
                if route == "ours":
                    worst = max(worst, metadata["block"]["qiskit_to_qrisp_bridge_max_abs_error"])
                worst = max(worst, *(checks[key] for key in (
                    "target_max_abs_error", "final_clean_scratch_max_amplitude", "temporary_release_max_amplitude")))
                for basis in BASES:
                    metric = result["metrics"][basis]
                    metric_artifact(metric, basis, artifacts)
                    worst = max(worst, metric["phase_sensitive_lowering_error"], metric["target_max_abs_error"])
            native, ours = (qr_case["routes"][route]["metrics"] for route in ("qrisp_native", "ours"))
            ratios = {basis: native[basis]["gate_count"] / ours[basis]["gate_count"] for basis in BASES}
            for basis in BASES:
                same_number(ratios[basis], qr_case["qrisp_over_ours"][basis])
            assert read(OUT / "qrisp_qsvt/cases" / case["id"] / "source_hashes.json") == qr["source_hashes"]
            rows.append({"id": case["id"], "name": case["name"], "framework": "Qrisp",
                         "baseline": native, "ours": ours, "ratios": ratios})
    assert worst < 1e-8, worst
    return rows, {"cases": 7, "framework_rows": 14, "max_error": worst}


def number(value):
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def triple(metric):
    return " / ".join(number(metric[key]) for key in ("gate_count", "non_clifford", "width"))


def select_document(summary, checks):
    lines = ["# LCU SELECT 比较（2026-10-13 更新）", "",
             "Ours 使用当前 `src/block_encoding.py` 的 `circuit_components(\"Matrix-order\").select` 重跑原有 10 个例子、135 个冻结输入；PennyLane（PL）和 Qrisp 列复用已保存的 SELECT 基线。制备部分采用共同成本口径，本表只比较 SELECT。", "",
             "G 为总门数，NC 为非 Clifford 门数，W 为总量子比特数。主表统一为 NAM `{rz,h,x,cx}`、Qiskit opt0；NC 使用项目分类，并非 T 门数。两列比例分别为 Qrisp/Ours、PL/Ours 的总门数之比，大于 1 表示 Ours 门数较少。", "",
             "通道实例对非空 Kraus 的 G、NC 求和，W 取最大；随机实例取相同种子的 10 个样本均值。W 包含工作位。", "",
             "| Benchmark | PL G / NC / W | Qrisp G / NC / W | Ours G / NC / W | Qrisp/Ours | PL/Ours |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in SELECT_CASES:
        row = next(row for row in summary["rows"] if row["case"] == name and row["basis"] == "nam")
        ours = " / ".join(number(row["ours"][key]) for key in ("G", "NC", "W"))
        lines.append(f"| {name} | {triple(row['pl'])} | {triple(row['qrisp'])} | {ours} | {row['qrisp_over_ours']:.3f} | {row['pl_over_ours']:.3f} |")
    lines += ["", "Ours 的 NAM、IBM 导出均已完成，逐输入计数、QASM、全局相位和验证证据保存于 "
              f"[SELECT 原始结果]({OUT / 'select/summary.json'})。全部占用地址的含相位 Pauli 乘积验证通过；小规模代表输入另验证编码块和实际降门。", "",
              "PL 基线使用 `graph_work` 配置。历史 Qrisp 输入多数只保存前 8 项；已保留项数、范数和有序预览核对证据，该核对不等于完整历史输入一致性证明。", "",
              f"TFIM-3 的 [独立相位审计]({OUT / 'tfim3_phase_audit/summary.json'}) "
              "确认当前 A0 是物理无跳跃项的 −i 整体相位规范，历史 Qrisp SELECT 和当前 Ours SELECT 的全部 80 个有效输入均通过。旧完整 block 记录的差异可由中间加法过早消除整体相位的机制复现。", "",
              f"实验前的 [接口 smoke test]({OUT / 'smoke_initial.json'}) 已通过 10 月 10 日、12 日存档校准；"
              f"[加强复核]({OUT / 'smoke.json'}) 同样通过。表格由 [统一生成器]({ROOT / 'Baseline_scripts/build_rq3_comparison_tables.py'}) "
              f"从完整结果重新计数后生成，验证清单见 [汇总数据]({OUT / 'comparison.json'})。", ""]
    return "\n".join(lines)


def qsvt_document(rows, checks):
    lines = ["# PennyLane / Qrisp QSVT 比较（2026-10-13 更新）", "",
             "仅保留以下 7 个有效比较输入。每个框架的原生块编码与 Ours 路线都实现 `P(H/alpha)`，其中 `P(x)=-x+0.5x³+0.5x⁵`、`alpha=sum(abs(coefficients))`；每个输入的有序项、系数、多项式和六个相位冻结在 "
             f"[输入清单]({OUT / 'qsvt_inputs.json'})。Ours 的底层统一由当前 `src/block_encoding.py` 正式导出接口生成。", "",
             "PennyLane 行保留官方 H–受控 QSVT–受控伴随 QSVT–H 外层，共 10 次块查询、12 个投影相位；本次只重跑接入 Ours 的电路，PL `PrepSelPrep` 基线复用并重新核对存档 QASM 计数。Ours 通过 `with_outer_control(1).controlled(...)` 在实际受控查询上下文选择 placement，将外控加入符号 SELECT 后展开，PREP 与 PREP† 保持无外控。", "",
             "Qrisp 0.9.9 提供 `qrisp.gqsp.QSVT(A, p, kind=\"Polynomial\", parity=\"odd\", rescale=False)`。Qrisp 行分别采用原生 `BlockEncoding.from_lcu` 和 Ours，保留 Qrisp 的信号位、反射和 5 次块查询。该接口没有公开相位参数，因此只临时替换相位求解器，输入反序的已冻结 PL 相位；两路线使用完全相同的相位和归一化。", "",
             "G 为总门数，NC 为非 Clifford 门数，W 为含工作位的总比特数。NAM=`{rz,h,x,cx}`，IBM=`{u1,u2,u3,cx}`，均最终 Qiskit opt0。箭头为各框架原生块编码 → Ours；比例为对应总门数的 PL/Ours 或 Qrisp/Ours，大于 1 表示 Ours 门数较少。", "",
             "| 例子 | QSVT 框架 | NAM G / NC：基线 → Ours | NAM 基线/Ours | IBM G / NC：基线 → Ours | IBM 基线/Ours | W：基线 → Ours |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        cells = []
        for basis in BASES:
            baseline, ours = row["baseline"][basis], row["ours"][basis]
            cells += [f"{number(baseline['gate_count'])} / {number(baseline['non_clifford'])} → {number(ours['gate_count'])} / {number(ours['non_clifford'])}",
                      f"{row['ratios'][basis]:.3f}"]
        assert row["baseline"]["nam"]["width"] == row["baseline"]["ibm"]["width"]
        assert row["ours"]["nam"]["width"] == row["ours"]["ibm"]["width"]
        width = f"{number(row['baseline']['nam']['width'])} → {number(row['ours']['nam']['width'])}"
        lines.append("| " + " | ".join([row["name"], row["framework"], *cells, width]) + " |")
    lines += ["", "两框架保留各自的 QSVT 外层和编译流程，比例用于各框架内替换块编码的比较。Qrisp 导出只按生命周期复用已释放的临时工作位，系统位与投影位始终保留；各释放点均验证工作位归零。", "",
              "所有例子均检查全部系统计算基输入、完整复输出幅度、目标多项式块和工作位归零；NAM、IBM 保存的 QASM 恢复全局相位后也通过验证。复数目标按完整复振幅比较。", "",
              f"全部实验完成后，另从 [PL 基线存档电路]({OUT / 'archived_pl_qsvt_correctness.json'}) "
              "复核全部系统计算基输入，7 例的两个基门导出均通过独立目标多项式检查，并保留 NAM/IBM 完整输出一致性证据。", "",
              f"逐例结果与源码快照：[PennyLane]({OUT / 'pennylane_qsvt/summary.json'})、[Qrisp]({OUT / 'qrisp_qsvt/summary.json'})；"
              f"[统一生成器]({ROOT / 'Baseline_scripts/build_rq3_comparison_tables.py'}) 重新计数全部导出和 PL 存档 QASM，"
              f"完整来源与验证清单见 [汇总数据]({OUT / 'comparison.json'})。", ""]
    return "\n".join(lines)


def main():
    smoke = read(OUT / "smoke.json")
    assert smoke["passed"] and read(OUT / "smoke_initial.json")["passed"]
    artifacts = []
    select, select_checks = validate_select(smoke, artifacts)
    qsvt_rows, qsvt_checks = validate_qsvt(smoke, artifacts)
    native_audit = read(OUT / "archived_pl_qsvt_correctness.json")
    phase_audit = read(OUT / "tfim3_phase_audit/summary.json")
    assert native_audit["completed"] and native_audit["passed"] and phase_audit["passed"]
    assert [case["id"] for case in native_audit["cases"]] == QSVT_CASES
    assert native_audit["sources"] == phase_audit["sources"] == smoke["sources"]
    documents = {ROOT / "Results/PennyLane_comparison.md": select_document(select, select_checks),
                 ROOT / "Results/PennyLane_QSVT_tables.md": qsvt_document(qsvt_rows, qsvt_checks)}
    previous_path = OUT / "previous_tables.json"
    if not previous_path.exists():
        dump(previous_path, {rel(path): {"sha256": digest(path), "text": path.read_text()}
                             for path in documents})
    snapshots = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__),
                 ROOT / "Baseline_scripts/run_rq3_select_refresh.py",
                 ROOT / "Baseline_scripts/smoke_block_encoding_archive.py",
                 ROOT / "Baseline_scripts/verify_rq3_archived_pl_qsvt.py",
                 ROOT / "Baseline_scripts/audit_tfim3_phases.py",
                 ROOT / "Baseline_scripts/baseline_common.py",
                 ROOT / "Baseline_results/pennylane_lcu_v1/inputs.json",
                 ROOT / "Baseline_results/pennylane_lcu_comparison/comparison.json",
                 OUT / "qsvt_inputs.json", OUT / "smoke.json", OUT / "smoke_initial.json"]
    source_manifest = {rel(path): digest(path) for path in snapshots}
    with tarfile.open(OUT / "select_source_snapshot.tar.gz", "w:gz") as saved:
        for path in snapshots:
            saved.add(path, arcname=rel(path), recursive=False)
    dump(OUT / "select_source_manifest.json", source_manifest)
    comparison = {"generated_at": datetime.now(timezone.utc).isoformat(), "passed": True,
                  "sources": smoke["sources"], "select_rows": select["rows"], "qsvt_rows": qsvt_rows,
                  "validation": {"select": select_checks, "qsvt": qsvt_checks,
                                 "qasm_artifacts_recounted": len(artifacts)},
                  "post_experiment_validation": {
                      "archived_native_pl_qsvt": rel(OUT / "archived_pl_qsvt_correctness.json"),
                      "tfim3_phase_audit": rel(OUT / "tfim3_phase_audit/summary.json")},
                  "artifacts": artifacts, "source_manifest": rel(OUT / "select_source_manifest.json"),
                  "documents": {rel(path): sha256(text.encode()).hexdigest() for path, text in documents.items()}}
    dump(OUT / "comparison.json", comparison)
    for path, text in documents.items():
        path.write_text(text)
    print(json.dumps({"passed": True, "tables": [rel(path) for path in documents],
                      "validation": comparison["validation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
