"""Prepare review-only D16 table data and an appendix patch after the missing run."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent
DATA = ROOT / 'Baseline_results/eval_redesign/artifact_review_20260917/D16_missing_n12_rerun'
result = json.loads((DATA / 'result.json').read_text())
assert result['status'] == 'ok' and all(result['source_hashes_unchanged'].values())
history = json.loads((REVIEW / 'D09_D16_notebook_values.json').read_text())
lookup = {r['n']: r for r in history['timing_rows']}
rows = []
for n in [3, 5, 7, 8, 10, 12]:
    old = lookup.get(n)
    reference_killed = n in [8, 10, 12]
    rows.append({'n': n,
                 'reference_status': 'oom_killed' if reference_killed else 'ok',
                 'reference_time_s': None if reference_killed else old['reference_mean'],
                 'reference_timeout_seconds': None,
                 'reference_status_source': 'latest user clarification: no time budget; killed due to memory exhaustion' if reference_killed else 'src/evaluate.ipynb cell10 stdout',
                 'ours_status': 'ok',
                 'ours_time_s': result['simulation_time_s'] if n == 12 else old['lcu_mean'],
                 'ours_source': str((DATA / 'result.json').relative_to(ROOT)) if n == 12 else 'src/evaluate.ipynb cell10 stdout',
                 'ours_batch': '2026-09-17 fresh rerun' if n == 12 else 'historical recorded output',
                 'aggregation': 'mean over 10 delta values' if n < 7 else 'one delta=0.1 run',
                 'delta_note': 'historical delta schedule inferred from saved notebook source; n=12 explicitly configured and logged'})
candidate = {'rows': rows, 'original_paper_n12_s': 188.03,
             'original_paper_n12_traceable': False,
             'reference_timeout_budget': None,
             'future_optional_timeout_s': 3600,
             'reference_termination': 'memory exhaustion / killed, not a wall-clock timeout',
             'paper_modified': False,
             'new_n12_timing_scope': result['timing_scope']}
(REVIEW / 'D16_updated_timing_candidate.json').write_text(json.dumps(candidate, indent=2) + '\n')

paper = ROOT.parent / 'nonudymics/Appendix/evaluation_ap.tex'
original = paper.read_text()
new = original
replacements = [
    ('the classical method can only handle small scale systems with $n < 8$ qubits; for larger systems the classical method will run out of memory and crash, see~\\Cref{tab:sim_time_clcu_ap}.',
     'the recorded reference runs at $n=8,10,12$ exhausted available memory and were killed before completion, as indicated by OOM in~\\Cref{tab:sim_time_clcu_ap}.'),
    ('3 & $0.0013s$ & $0.09s$', '3 & $0.0012s$ & $0.09s$'),
    ('7 & $904.49s$ & $0.89s$', '7 & $934.15s$ & $0.88s$'),
    ('8 & $N/A$  &  $3.03s$', '8 & OOM & $3.03s$'),
    ('10 &  $N/A$ & $17.77s$', '10 & OOM & $17.77s$'),
    ('12 & $N/A$ & $188.03s$', f"12 & OOM & ${result['simulation_time_s']:.2f}s$"),
    ('\\caption{Simulation time cost for mesolve and our method.}',
     '\\caption{Simulation time cost for the reference solver and channel-LCU. '
     'OOM denotes termination due to memory exhaustion; no wall-clock timeout was imposed. '
     'The $n=12$ channel-LCU value is a new single-run measurement at $\\Delta t=0.1$; '
     'the remaining values are retained historical measurements.}'),
]
for old, replacement in replacements:
    assert new.count(old) == 1, (old, new.count(old))
    new = new.replace(old, replacement)
patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True), new.splitlines(keepends=True),
                                   fromfile='a/Appendix/evaluation_ap.tex', tofile='b/Appendix/evaluation_ap.tex'))
(REVIEW / 'D16_appendix_candidate.patch').write_text(patch)
paper_hash = hashlib.sha256(paper.read_bytes()).hexdigest()
(REVIEW / 'D16_appendix_patch_source.json').write_text(json.dumps({'source': str(paper), 'sha256': paper_hash,
                                                                 'applied': False}, indent=2) + '\n')

lines = ['# D16：原数据复查、n=12补跑与附录修改建议', '',
         '本次没有修改论文或覆盖原始输出。以下数据是待审核候选；用户最新澄清reference空数组来自内存溢出后被killed，历史运行没有设置时间上限。', '',
         '## 原数据搜索', '',
         '检查了编译仓库（含忽略文件）、父目录的Baseline_results、论文仓库、另一份工作树notebook、Results/accuracy_cLCU，以及src/evaluate.ipynb的全部21个Git历史版本。仍未找到n=12的188.03s原始计时。', '',
         '`Outdated/test1.py`确实包含n=12运行入口，但其Δt=0.05且没有保存计时输出，不能用它证明188.03s。其余表中行可由现有notebook数组核算。搜索记录见`D16_original_data_search.json`。', '',
         'reference在n=8、10、12的空记录按用户最新确认记为OOM / killed，不能写成按时间预算触发的Timeout。3600s仅是用户允许在今后重跑时采用的可选预算，不是历史设置。本次没有重跑reference。', '',
         '## n=12当前轮补跑', '',
         f"实际测得 **{result['simulation_time_s']:.9f}s**，附录显示 **{result['simulation_time_s']:.2f}s**。本次为一次运行，不是重复平均。", '',
         '周期TFIM，n=12，Δt=0.1，γ=√0.1/2，输入为全零态；沿用notebook的`channel_to_LCU(...,structure="basic",opt="No")`和`channel_LCU.simulate_circuit`，使用当前`block_encoding.BlockEncoding`。D11的Matrix-order残差修复不涉及此No-order计时入口。', '',
         '电路共22 qubits，寄存器大小为[6,4,12]。计时范围包含模拟函数内部的电路复制、Aer转译opt1、模拟、幅度阈值1e-6处理及后选择密度矩阵提取；模型/电路构造、初态分配、导出和验证不计入。', '',
         f"电路构造另计{result['build_time_s']:.6f}s。零态输出与直接Kraus计算的支撑子矩阵最大差为{result['validation']['max_error_on_reference_support']:.3e}，支撑外对角质量为{result['validation']['outside_reference_support_diagonal_mass']:.3e}，通过1e-6检查。", '',
         '当前运行环境：'+', '.join(f'{k} {v}' for k,v in result['versions'].items())+'；'+result['platform']+'。硬件型号读取被环境限制，protocol保存了这一事实。', '',
         '没有重跑reference，也没有将新数据冒充为188.03s的原始记录。原补跑protocol/result写入时用户尚未澄清killed原因，其中historical_timeout_user_confirmed是当时的临时元数据；保留这些原记录，并以`D16_reference_status_clarification.json`和本候选覆盖该状态解释，计时数值不受影响。', '',
         '## 表格候选', '',
         '| n | Reference | Ours | 来源 |', '|---:|---:|---:|---|']
for row in rows:
    n = row['n']
    ref = 'OOM / killed' if row['reference_status'] == 'oom_killed' else (f"{row['reference_time_s']:.4f}s" if n == 3 else f"{row['reference_time_s']:.2f}s")
    lines.append(f"| {n} | {ref} | {row['ours_time_s']:.2f}s | {'本轮补跑；reference为历史OOM' if n == 12 else '原始notebook数组'} |")
lines += ['', '## 附录修改建议', '',
          '1. `Appendix/evaluation_ap.tex`中原表n=3 reference改0.0012s；n=7改934.15s / 0.88s；n=12 ours替换成本轮实测值。',
          '2. n=8、10、12的reference统一用OOM或OOM/killed，注释说明内存耗尽后被终止、无完成时间、未设置时间上限；不能标成TO或>3600s。',
          '3. 将“经典方法只能处理n<8”的泛化结论改成“本次记录中的reference在n=8、10、12耗尽内存而被终止”，限定于实际实验配置。',
          '4. caption明确n=12来自本轮单次补跑，其余为历史数据。n=3/5原值是10个Δt点的平均；n≥7按现存notebook逻辑是Δt=0.1单点。历史执行时源码/硬件没有完整锁定，不能将混合批次描述成同一轮运行。',
          '5. 保留simulation timing的实际范围说明，尤其是simulate_circuit内部转译也计时；不要把它误称为纯Aer内核时间。若后续要求同一硬件下的严格性能比较，应另行统一重跑整表；本次只补缺失n=12。', '',
          '`D16_appendix_candidate.patch`已给出可审阅补丁，未应用。`D16_updated_timing_candidate.json`保存完整精度及逐行来源。原始补跑记录、QPY电路、稀疏密度矩阵和日志位于`Baseline_results/eval_redesign/artifact_review_20260917/D16_missing_n12_rerun/`。', '',
          '运行过程曾因可选硬件型号读取受限而在模拟开始前退出；修复记录器后才开始有效实验。第一次日志保留为`initial_metadata_failure.log`，不计作计时样本。']
(REVIEW / 'D16_update_review_zh.md').write_text('\n'.join(lines) + '\n')
assert hashlib.sha256(paper.read_bytes()).hexdigest() == paper_hash
print('Prepared D16 review, full-precision candidate data and unapplied appendix patch.')
