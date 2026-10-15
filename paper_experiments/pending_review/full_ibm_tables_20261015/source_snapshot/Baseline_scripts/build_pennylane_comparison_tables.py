#!/usr/bin/env python3
"""Build provenance-checked PennyLane/Qrisp/Ours comparison tables."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from run_pennylane_lcu_benchmarks import historical_check, save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN = ROOT / "Baseline_results/pennylane_lcu_v1/summary.json"
DEFAULT_SMOKE = ROOT / "Baseline_results/pennylane_lcu_profiles_smoke/summary.json"
DEFAULT_RAW = ROOT / "Baseline_results/pennylane_lcu_full_pauli_graph_raw/summary.json"
DEFAULT_HISTORY = ROOT / "Baseline_results/block_encoding_new_qrisp_comparison_v1/summary.json"
DEFAULT_REPORT = ROOT / "Results/PennyLane_comparison.md"
DEFAULT_OUTPUT = ROOT / "Baseline_results/pennylane_lcu_comparison"
DEFAULT_PREPARE = ROOT / "Baseline_results/pennylane_prepare_cost_all/summary.json"
METRICS = ("gate_count", "non_clifford", "width")
PROFILES = ("legacy_raw", "legacy_compile", "graph_compile", "graph_work")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_run(path, *, allow_partial=False):
    if "partial" in path.name and not allow_partial:
        raise ValueError(f"Refusing partial summary: {path}")
    summary = load(path)
    manifest_path = Path(summary["input_manifest"])
    manifest = load(manifest_path)
    if summary["input_manifest_sha256"] != sha256(manifest_path):
        raise ValueError(f"Input manifest has changed since the run: {manifest_path}")
    arguments = summary["arguments"]
    names = [case["name"] for case in manifest["cases"]
             if not arguments.get("cases") or case["name"] in arguments["cases"]]
    expected = {(name, scope, profile) for name in names
                for scope in arguments["scopes"] for profile in arguments["profiles"]}
    actual = {(result["case"], result["scope"], result["profile"])
              for result in summary["results"] if result["status"] != "running"}
    complete = expected == actual and "partial" not in path.name
    if not complete and not allow_partial:
        raise ValueError(f"Run is not a completed batch: {path}; missing {sorted(expected - actual)}")
    return {
        "summary": summary, "manifest": manifest,
        "source": {
            "path": str(path), "sha256": sha256(path), "batch_complete": complete,
            "partial_source": "partial" in path.name, "missing_jobs": sorted(expected - actual),
            "input_manifest": str(manifest_path), "input_manifest_sha256": sha256(manifest_path),
            "environment": summary.get("environment"), "arguments": arguments,
            "run_source_sha256": summary.get("source_sha256", {}),
            "input_source_sha256": manifest.get("source_sha256", {}),
        },
    }


def old_cases(history):
    result = {}
    for case in history.get("rewritten_kraus", {}).get("benchmarks", []):
        result[case["benchmark"]] = ("sum_kraus", case)
    for case in history.get("tfim", {}).get("cases", []):
        result[case["case"]] = ("single", case)
    for case in history.get("random_pauli", {}).get("cases", []):
        result[case["case"]["name"]] = ("sample_mean", case)
    return result


def aggregate(values, aggregation, metric):
    if not values:
        return None
    if aggregation == "sample_mean":
        return statistics.fmean(values)
    return max(values) if metric == "width" else sum(values)


def full_references(case, history_index, history_path):
    evidence = {"source_path": str(history_path), "inputs": [], "scope": "block"}
    previous = history_index.get(case["name"])
    if previous is None:
        evidence.update(status="unavailable", reason="case_absent_from_selected_historical_full_summary")
        return evidence, []
    aggregation, old = previous
    if aggregation != case["aggregation"]:
        evidence.update(status="mismatch", reason="aggregation")
        return evidence, []
    if aggregation == "sum_kraus":
        indexed = {f"kraus{item['kraus_index']:04d}": item for item in old["kraus_results"]}
        membership_match = set(indexed) == {item["id"] for item in case["inputs"]}
    elif aggregation == "sample_mean":
        indexed = {item["seed"]: item for item in old["samples"]}
        membership_match = all(item["seed"] in indexed for item in case["inputs"])
    else:
        indexed = {"single": old}
        membership_match = len(case["inputs"]) == 1
    matched = []
    for item in case["inputs"]:
        key = item["seed"] if aggregation == "sample_mean" else item["id"]
        saved = indexed.get(key)
        details = {}
        if saved is None:
            check = {"status": "unavailable", "errors": ["historical_input_missing"]}
        else:
            metrics = dict(saved.get("validation", saved.get("metrics", saved)))
            if "num_qubits" in metrics:
                metrics["system_qubits"] = metrics["num_qubits"]
            terms = [(term["pauli"], complex(term["real"], term["imag"])) for term in item["terms"]]
            check = historical_check(terms, {"metrics": metrics}, item["num_system_qubits"])
            differences = []
            for (label, value), old_term in zip(terms, metrics.get("terms_preview", [])):
                old_value = complex(old_term.get("effective_coeff", old_term.get("coeff")))
                if label != old_term["pauli"] or abs(value - old_value) > 1e-10:
                    differences.append({
                        "current": {"pauli": label, "real": value.real, "imag": value.imag},
                        "historical": {"pauli": old_term["pauli"], "real": old_value.real, "imag": old_value.imag},
                    })
            details = {
                "current_l1_norm": sum(abs(value) for _, value in terms),
                "historical_l1_norm": metrics.get("l1_norm", metrics.get("pauli_norm")),
                "current_term_count": len(terms), "historical_term_count": metrics.get("term_count"),
                "preview_mismatch_examples": differences[:3],
            }
        evidence["inputs"].append({
            "id": item["id"], "seed": item["seed"], "input_sha256": item["input_sha256"],
            "historical_locator": key, "validation": check, "comparison_details": details,
        })
        matched.append(saved)
    evidence["membership_match"] = membership_match
    valid = membership_match and all(item["validation"]["status"] == "matched_recorded_fields"
                                     for item in evidence["inputs"])
    evidence["status"] = "matched_recorded_fields" if valid else "mismatch"
    evidence["historical_full_input_available"] = all(
        item["validation"].get("historical_full_input_available", False) for item in evidence["inputs"]
    )
    return evidence, matched if valid else []


def reference_metrics(case, scope, basis, history_index, history_path):
    if scope == "block":
        evidence, saved_inputs = full_references(case, history_index, history_path)
        keys = {"qrisp": "qrisp_from_lcu", "ours": "ours_block_encoding_new_current"}
        available = {
            method: [saved.get(key, {}).get("basis_summary", {}).get(basis)
                     if saved.get(key, {}).get("status") == "ok" else None for saved in saved_inputs]
            for method, key in keys.items()
        }
    else:
        valid = all(item["historical_check"]["status"] == "matched_recorded_fields" for item in case["inputs"])
        evidence = {
            "scope": "select", "source_path": case.get("reference_path"),
            "status": "matched_recorded_fields" if valid else "mismatch",
            "inputs": [{"id": item["id"], "seed": item["seed"], "input_sha256": item["input_sha256"],
                        "validation": item["historical_check"]} for item in case["inputs"]],
        }
        available = {method: [item.get(method + "_select_reference", {}).get(basis) for item in case["inputs"]]
                     if valid else [] for method in ("qrisp", "ours")}
    references = {}
    for method, values in available.items():
        valid = len(values) == len(case["inputs"]) and all(values)
        references[method] = {
            metric: aggregate([value[metric] for value in values], case["aggregation"], metric) if valid else None
            for metric in METRICS
        }
    return references, evidence


def comparison_row(run, row, history_index, history_path, *, alternative=False):
    case = next(case for case in run["manifest"]["cases"] if case["name"] == row["case"])
    references, evidence = reference_metrics(case, row["scope"], row["basis"], history_index, history_path)
    result = next((result for result in run["summary"]["results"]
                   if all(result[key] == row[key] for key in ("case", "scope", "profile"))), {})
    complete = row.get("status") == "ok" and row.get("completed_inputs") == len(case["inputs"])
    output = {key: row.get(key) for key in ("case", "scope", "basis", "profile", "status", "aggregation",
                                          "completed_inputs", "expected_inputs", "wall_seconds", "elapsed_unique_seconds")}
    output.update({
        "alternative_profile": alternative, "source_path": run["source"]["path"],
        "source_batch_complete": run["source"]["batch_complete"],
        "timeout_seconds": result.get("timeout_seconds", run["summary"]["arguments"].get("timeout")),
        "log": result.get("log"), "input_count": len(case["inputs"]),
        "pennylane": {metric: row.get(metric) if complete else None for metric in METRICS},
        "references": references, "reference_evidence": evidence,
        "sample_seeds": [item["seed"] for item in case["inputs"]] if case["aggregation"] == "sample_mean" else [],
    })
    for method in ("qrisp", "ours"):
        denominator = references[method]["gate_count"]
        numerator = output["pennylane"]["gate_count"]
        output["pennylane_over_" + method] = numerator / denominator if numerator is not None and denominator else None
    return output


def main_rows(main, raw, history_index, history_path):
    rows = []
    arguments = main["summary"]["arguments"]
    indexed = {(row["case"], row["scope"], row["basis"], row["profile"]): row for row in main["summary"]["rows"]}
    for case in main["manifest"]["cases"]:
        if arguments.get("cases") and case["name"] not in arguments["cases"]:
            continue
        for scope in ("select", "block"):
            if scope not in arguments["scopes"]:
                continue
            for basis in ("nam", "ibm"):
                row = indexed.get((case["name"], scope, basis, "graph_work"), {
                    "case": case["name"], "scope": scope, "basis": basis, "profile": "graph_work",
                    "status": "pending", "aggregation": case["aggregation"], "completed_inputs": 0,
                    "expected_inputs": len(case["inputs"]),
                })
                rows.append(comparison_row(main, row, history_index, history_path))
                if case["name"] == "full_pauli_n6" and row["status"] != "ok":
                    raw_case = next(other for other in raw["manifest"]["cases"] if other["name"] == case["name"])
                    if [item["input_sha256"] for item in raw_case["inputs"]] != [item["input_sha256"] for item in case["inputs"]]:
                        raise ValueError("Alternative Full Pauli run uses different complete inputs")
                    alternatives = [other for other in raw["summary"]["rows"]
                                    if other["case"] == case["name"] and other["scope"] == scope
                                    and other["basis"] == basis and other["profile"] == "graph_work_raw"]
                    for alternative in alternatives:
                        rows.append(comparison_row(raw, alternative, history_index, history_path, alternative=True))
    return rows


def additive_block_rows(rows):
    """Apply a common PL full-minus-SELECT residual to the three SELECT costs."""
    indexed = {(row["case"], row["basis"], row["profile"]): row
               for row in rows if row["scope"] == "select"}
    derived = []
    for block in rows:
        if block["scope"] != "block" or block["alternative_profile"]:
            continue
        select = indexed.get((block["case"], block["basis"], block["profile"]))
        valid = select is not None and block["status"] == select["status"] == "ok"
        residual = {
            metric: block["pennylane"][metric] - select["pennylane"][metric]
            if valid and block["pennylane"][metric] is not None and select["pennylane"][metric] is not None else None
            for metric in ("gate_count", "non_clifford")
        }
        methods = {}
        for method in ("qrisp", "ours"):
            values = select["references"][method] if select else {}
            methods[method] = {
                metric: values[metric] + residual[metric]
                if values.get(metric) is not None and residual[metric] is not None else None
                for metric in ("gate_count", "non_clifford")
            }
            # Counts do not establish compatible selector layouts or a width.
            methods[method].update(width=None, depth=None, compile_seconds=None)
        item = {
            "case": block["case"], "basis": block["basis"], "profile": block["profile"],
            "scope": "full_block_additive_estimate", "aggregation": block["aggregation"],
            "input_count": block["input_count"], "sample_seeds": block["sample_seeds"],
            "donor": "pennylane_same_current_inputs",
            "formula": "estimate(method) = SELECT(method) + full(PL) - SELECT(PL)",
            "residual_is_isolated_prepare_cost": False,
            "residual": residual, "pennylane_measured": block["pennylane"],
            "estimates": methods,
            "historical_full_references": block["references"],
            "select_reference_evidence": select["reference_evidence"] if select else None,
            "source_path": block["source_path"],
            "status": "estimated" if valid and all(
                methods[method]["gate_count"] is not None for method in methods
            ) else "unavailable",
        }
        for method in methods:
            denominator = methods[method]["gate_count"]
            numerator = block["pennylane"]["gate_count"]
            item["pennylane_over_" + method] = (
                numerator / denominator if numerator is not None and denominator else None
            )
        derived.append(item)
    return derived


def calibrated_block_rows(run, rows, preparations):
    if preparations["status"] != "ok":
        raise ValueError("The independent preparation batch must be complete")
    if preparations["input_manifest_sha256"] != run["source"]["input_manifest_sha256"]:
        raise ValueError("Independent preparations use a different input manifest")
    cases = {case["case"]: case for case in preparations["cases"]}
    originals = {case["name"]: case for case in run["manifest"]["cases"]}
    derived = []
    for select in rows:
        if select["scope"] != "select" or select["alternative_profile"]:
            continue
        case = cases[select["case"]]
        expected = {item["id"]: item["input_sha256"] for item in originals[select["case"]]["inputs"]}
        actual = {item["id"]: item["input_sha256"] for item in case["inputs"] if item["status"] == "ok"}
        if actual != expected:
            raise ValueError(f"Independent preparation inputs mismatch: {select['case']}")
        pair = case["basis_summary"][select["basis"]]
        methods = {}
        for method in ("pennylane", "qrisp", "ours"):
            source = select["pennylane"] if method == "pennylane" else select["references"][method]
            methods[method] = {
                metric: source[metric] + pair[metric] if source[metric] is not None else None
                for metric in ("gate_count", "non_clifford")
            }
        item = {
            "case": select["case"], "basis": select["basis"], "profile": select["profile"],
            "scope": "modular_full_block_with_measured_common_prepare",
            "status": "cost_model", "aggregation": select["aggregation"],
            "input_count": case["input_count"], "preparation_pair": pair,
            "formula": "cost(method) = measured SELECT(method) + measured PL PREP + measured PL PREP_dagger",
            "composed_costs": methods, "hybrid_circuit_compiled": False,
            "width": None, "depth": None, "compile_seconds": None,
        }
        for method in ("qrisp", "ours"):
            numerator, denominator = methods["pennylane"]["gate_count"], methods[method]["gate_count"]
            item["pennylane_over_" + method] = numerator / denominator if numerator is not None and denominator else None
        derived.append(item)
    return derived


def number(value):
    return "--" if value is None else f"{value:,.3f}".rstrip("0").rstrip(".")


def triplet(values):
    return " / ".join(number(values[metric]) for metric in METRICS)


def table(rows):
    lines = ["| Benchmark | Profile | PL G / NC / W | Qrisp G / NC / W | Ours G / NC / W | PL/Qrisp | PL/Ours | 状态 | 完成 | 秒 / 预算 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |"]
    for row in rows:
        profile = row["profile"] + (" (独立替代配置)" if row["alternative_profile"] else "")
        cells = [row["case"], profile, triplet(row["pennylane"]), triplet(row["references"]["qrisp"]),
                 triplet(row["references"]["ours"]), number(row["pennylane_over_qrisp"]),
                 number(row["pennylane_over_ours"]), row["status"],
                 f"{row['completed_inputs']}/{row['expected_inputs']}",
                 f"{number(row['wall_seconds'])} / {number(row['timeout_seconds'])}"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def additive_table(rows):
    lines = [
        "| Benchmark | PL 实测 G / NC / W | 公共差额 G / NC | Qrisp 估算 G / NC | Ours 估算 G / NC | PL/Qrisp估算 | PL/Ours估算 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pair = lambda values: " / ".join(number(values[key]) for key in ("gate_count", "non_clifford"))
        cells = [row["case"], triplet(row["pennylane_measured"]), pair(row["residual"]),
                 pair(row["estimates"]["qrisp"]), pair(row["estimates"]["ours"]),
                 number(row["pennylane_over_qrisp"]), number(row["pennylane_over_ours"])]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def calibrated_table(rows):
    def pair(values):
        return " / ".join(number(values[key]) for key in ("gate_count", "non_clifford"))

    lines = [
        "| Benchmark | 独立 PREP+逆 G / NC | PL 模块和 G / NC | Qrisp 模块和 G / NC | Ours 模块和 G / NC | PL/Qrisp | PL/Ours |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = [row["case"], pair(row["preparation_pair"])]
        cells += [pair(row["composed_costs"][method]) for method in ("pennylane", "qrisp", "ours")]
        cells += [number(row["pennylane_over_" + method]) for method in ("qrisp", "ours")]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(path, data):
    primary = [row for row in data["main_rows"] if not row["alternative_profile"]]
    graph_complete = bool(primary) and all(row["status"] == "ok" for row in primary)
    completion_text = (
        "本次主表的图编译配置全部完成，包含 Full Pauli n=6 的 SELECT 和完整 block。graph_work_raw 的独立结果另行保存，用于观察省略局部编译阶段的成本。"
        if graph_complete else
        "若 Full Pauli n=6 的 graph_work 配置未完成，保留其状态，并另列 graph_work_raw；后者省略 qml.compile 优化，不是同配置完成结果。"
    )
    lines = ["# PennyLane、Qrisp 与 Ours 的 LCU 比较", "",
             "G 为总门数，NC 为非 Clifford 门数，W 为量子比特数。所有方法在相同 NAM 或 IBM 基础门集合下计数；PennyLane 输出仅使用 Qiskit opt0 做统一降门。比例为 PL/基线，小于 1 表示 PennyLane 门数较少。", "",
             "SELECT 与完整 PREPARE-SELECT-PREPARE† 分开统计。通道类实例逐个非空 Kraus 求和，W 取最大值；随机实例使用相同种子的样本均值。这里未构造外层 channel-LCU，也未将 QSVT 多项式查询计入成本。", "",
             "主表明确使用 graph_work。" + completion_text + "超时预算按整个 case/scope/profile 计算，含全部输入；超时或错误的部分成本不作为总成本。", "",
             "主批次全部配置的状态统计：" + ", ".join(f"{key}={value}" for key, value in data["main_run_status_counts"].items()) + "。未完成配置的日志和局部结果保留在原始运行目录中。", "",
             "Qrisp/Ours 数值来自保存的历史运行。本次核对项数、系数范数和有序项预览；多数历史文件只保存前 8 项，不能据此证明完整历史输入完全一致。JSON 保留逐输入证据、源路径、文件摘要、种子、时间及状态。历史 Qrisp 的系统比特标签顺序存在整体反序，门数比较不受该重标记影响。", "",
             "完整 block 主表现按共同制备开销的加法模型更新：Delta = full(PL) - SELECT(PL)，Qrisp/Ours 的估算完整成本 = 各自 SELECT 成本 + Delta。统一选 PL 为 donor，因为它覆盖当前全部输入，不需要混用系数不同的旧完整 block。NC 也按同样公式给出估算。", "",
             "Delta 是剩余成本，不保证等于单独 PREPARE+PREPARE† 的实际门数：PL 完整 block 的内部 SELECT 使用 partial=True，独立 SELECT 使用 partial=False，且编译可能跨模块抵消门。因此这些是用户指定公式下的成本估算，没有构建或验证混合电路，不推算其宽度、深度和编译时间，也不假定不同地址布局可直接拼接。", "",
             "旧完整 block 的原始对照保留在后面的审计表中。Random4q20 的系数分布不同；Coloring、Repetition DQE、TFIM 的旧记录也未全部匹配；例如 TFIM-3 的 identity 系数由旧记录 -0.0075-1j 变为当前 -0.9925j。加法估算不使用这些不匹配的旧完整 block 数值。", ""]
    if data["preview"]:
        lines[2:2] = ["**这是进行中批次的预览，不是完成后的比较报告。**", ""]
    for scope, title in (("select", "SELECT 主表"),):
        for basis in ("nam", "ibm"):
            lines += [f"## {title}：{basis.upper()}", ""]
            lines += table([row for row in data["main_rows"] if row["scope"] == scope and row["basis"] == basis]) + [""]
    for basis in ("nam", "ibm"):
        lines += [f"## 完整 Block-Encoding：统一 PL 差额估算，{basis.upper()}", ""]
        lines += additive_table([row for row in data["additive_block_rows"] if row["basis"] == basis]) + [""]
    lines += ["## 独立制备计数校准", "",
              "为排除 partial SELECT 对差额的影响，另对全部 135 个输入单独编译 PREP 和 PREP†，将两者门数相加。两段分开编译，避免在缺少中间 SELECT 时互相抵消。这里三者都采用自己的完整地址 SELECT 成本加同一实测制备成本，所以 PL 的模块和可能大于上一表的实际完整 block。", "",
              "这是共同制备模块下的成本模型，仍未验证不同 SELECT 地址布局的混合电路拼接，故不推算组合电路的宽度、深度或时间。例：TFIM-3 NAM 的独立制备成本为 298 门、30 NC，而差额为 276 门、22 NC；相差的 22 门、8 NC 来自 partial SELECT。", ""]
    for basis in ("nam", "ibm"):
        lines += [f"### 共用实测 PREP 成本：{basis.upper()}", ""]
        lines += calibrated_table([row for row in data["calibrated_block_rows"] if row["basis"] == basis]) + [""]
    for basis in ("nam", "ibm"):
        lines += [f"## 原始完整 Block 实测与历史对照：{basis.upper()}", ""]
        lines += table([row for row in data["main_rows"] if row["scope"] == "block" and row["basis"] == basis]) + [""]
    lines += ["## 完整 Block 历史输入核对", "",
              "这里只在所有对应输入的已记录字段都匹配时引用基线。matched_recorded_fields 仍不代表完整历史输入的精确等同性。", "",
              "| Benchmark | 核对状态 | 失败字段或缺失原因 |", "| --- | --- | --- |"]
    for row in data["main_rows"]:
        if row["scope"] != "block" or row["basis"] != "nam" or row["alternative_profile"]:
            continue
        evidence = row["reference_evidence"]
        reasons = sorted({error for item in evidence["inputs"] for error in item["validation"].get("errors", [])})
        if evidence.get("reason"):
            reasons.append(evidence["reason"])
        if evidence.get("membership_match") is False:
            reasons.append("Kraus/sample membership")
        lines.append(f"| {row['case']} | {evidence['status']} | {', '.join(reasons) or '--'} |")
    lines.append("")
    lines += ["## 四种配置的小规模对照", "",
              "此组仅含 TFIM-3 A0 单例和 Random Pauli 4q/20t 的一个样本，后者 seed=20260703；不能视为 10 样本平均。legacy_raw 为原生递归分解；legacy_compile 增加默认编译流水线；graph_compile 使用图分解；graph_work 进一步显式提供 clean work wires。配置的准确 pass 与环境元数据以源 JSON 为准。", ""]
    for scope in ("select", "block"):
        for basis in ("nam", "ibm"):
            lines += [f"### {scope} / {basis.upper()}", ""]
            lines += table([row for row in data["profile_rows"] if row["scope"] == scope and row["basis"] == basis]) + [""]
    lines += ["## 数据与复现", ""]
    for source in data["sources"]:
        lines.append(f"- `{source['path']}`；批次完成：{source['batch_complete']}。")
    lines += [f"- 历史完整 block：`{data['historical_full_source']['path']}`。", "",
              f"- 独立制备计数：`{data['independent_prepare_source']['path']}`。", "",
              "主运行完成并生成 summary.json 后，在仓库目录执行：", "", "```bash",
              "/Users/chester_hqf/miniconda3/envs/qiskit_simulate/bin/python Baseline_scripts/build_pennylane_comparison_tables.py",
              "```", "",
              "生成器默认拒绝缺失或未完成的最终汇总。重新运行 benchmark 的具体参数记录在各 summary.json 的 arguments 字段；主比较表不会用正在更新的 partial 文件冒充最终结果。", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-summary", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--smoke-summary", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--raw-summary", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--history-summary", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--prepare-summary", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--allow-partial", action="store_true", help="Preview only; requires non-default output and report paths")
    args = parser.parse_args()
    if args.allow_partial and (args.out_dir.resolve() == DEFAULT_OUTPUT or args.report.resolve() == DEFAULT_REPORT):
        parser.error("Partial previews require separate --out-dir and --report paths")
    main_run = read_run(args.main_summary.resolve(), allow_partial=args.allow_partial)
    smoke = read_run(args.smoke_summary.resolve())
    raw = read_run(args.raw_summary.resolve())
    history_path = args.history_summary.resolve()
    history_index = old_cases(load(history_path))
    preparations_path = args.prepare_summary.resolve()
    preparations = load(preparations_path)
    rows = main_rows(main_run, raw, history_index, history_path)
    profile_rows = [comparison_row(smoke, row, history_index, history_path) for row in smoke["summary"]["rows"]]
    profile_rows.sort(key=lambda row: (row["case"], row["scope"], row["basis"], PROFILES.index(row["profile"])))
    data = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "main_run_status_counts": dict(Counter(result["status"] for result in main_run["summary"]["results"])),
        "preview": args.allow_partial, "builder_sha256": sha256(Path(__file__)),
        "policy": "Same-basis gate counts; historical references require matching recorded fields, not claimed exact full-input equality.",
        "sources": [run["source"] for run in (main_run, smoke, raw)],
        "historical_select_sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in sorted({Path(case["reference_path"]) for run in (main_run, smoke, raw)
                                for case in run["manifest"]["cases"] if case.get("reference_path")})
        ],
        "historical_full_source": {"path": str(history_path), "sha256": sha256(history_path)},
        "main_rows": rows, "profile_rows": profile_rows,
        "additive_block_rows": additive_block_rows(rows),
        "calibrated_block_rows": calibrated_block_rows(main_run, rows, preparations),
        "independent_prepare_source": {"path": str(preparations_path), "sha256": sha256(preparations_path)},
    }
    save_json(args.out_dir / "comparison.json", data)
    write_report(args.report, data)
    print(f"wrote {args.out_dir / 'comparison.json'}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
