# Lean formalization of main_v2.pdf

本库对应 `paper-source.json` 固定的 54 页论文版本，正文编号结论为 4.1–4.6、5.1、6.1、6.2，另包含直接依赖的附录 C.1、C.2、C.4。统一入口为 `import ChannelFormalization`；逐项进度及核验状态见 `THEOREMS.md`。

## 环境与运行

- Lean 4.34.1；Lake 5.0.0-src+5045d00。
- Physlib/QuantumInfo 固定到 `a4ef46125b32bb2599ab42d9c3a95e1d88ca75ff`。
- mathlib 固定到 `d13f23b723b8a846827a245b89c10fc7d3f11612`（v4.34.1）；全部传递依赖记录在 `lake-manifest.json`。

本机安装、依赖和缓存均位于本目录内的 `.toolchain/`、`.lake/`、`.cache/`。`lakew` 优先使用该本地工具链；解压源码包到其他机器时，先安装 `lean-toolchain` 指定的 Lean 版本，`lakew` 也可使用 PATH 中的 Lake。

```sh
./lakew build
./lakew env lean scripts/audit.lean
python3 scripts/check_source.py
```

最后一条命令检查原始 PDF 的 SHA-256；也可传入 PDF 新路径。源码包不包含 PDF、工具链或依赖缓存。首次在新机器构建会获取锁定版本的依赖，并需要网络连接。

## 结论入口

下表名称均位于 `ChannelFormalization` 命名空间。

| 论文 | 文件 | 主要声明 |
|---|---|---|
| Theorem 4.1 | `ChannelIR.lean` | `ChannelIR.theorem_4_1` |
| Lemma 4.2 | `GuardedRewrites.lean` | `ChannelIR.Guarded.Rewrite.wellTyped`、`Rewrites.wellTyped` |
| Theorem 4.3 | `GuardedRewrites.lean` | `ChannelIR.Guarded.Rewrite.sound`、`Rewrites.sound` |
| Proposition 4.4 | `GuardedCompleteness.lean` | `ChannelIR.Guarded.proposition_4_4_guarded` |
| Proposition 4.5 | `ChannelLCU.lean` | `ChannelLCU.proposition_4_5`、`proposition_4_5_with_unused_addresses`、`success_probability`、`success_probability_TP` |
| Proposition 4.6 | `KrausReduction.lean` | `proposition_4_6` |
| Proposition 5.1 | `PauliCompiler.lean` | `PauliCompiler.proposition_5_1` |
| Proposition 6.1 | `FrontendCleanup.lean` | `FrontendCleanup.proposition_6_1_guarded`、`exists_sampled_normalForm_rank` |
| Theorem 6.2 | `GuardedCompiler.lean` | `GuardedCompiler.theorem_6_2_guarded` |
| Lemma C.1 | `AddressPlacement.lean` | `AddressPlacement.lemma_C_1` |
| Lemma C.2 | `Placement.lean` | `Placement.lemma_C_2`、`lemma_C_2_unitary` |
| Lemma C.4 | `Placement.lean`、`PauliPlacement.lean`、`PauliCost.lean` | 残差唯一性、增量更新、实际矩阵相位恢复、局部代价公式 |
| Section 4.2 终止性 | `GuardedRewrites.lean` | `ChannelIR.Guarded.deterministic_terminates`、`no_infinite_sequence`、`exists_normalForm` |
| Example 5.3 的下界 | `PauliOptimality.lean` | `PauliCost.example_5_3_circuit_lower_bound` |

旧版 Theorem 2.2 已不属于当前编号结论。`LCU.theorem_2_2` 及其未使用地址版本保留为通用近似 LCU 引理，兼容已有调用；当前背景中的 exact LCU 对应其零误差特例。

`Smoke.lean` 使用 Physlib 的任意 Kraus 家族完全正性及实际 Pauli 矩阵酉性。`scripts/audit.lean` 遍历统一入口导入的全部本库定理，包含私有定理，并检查传递公理依赖；只允许 Lean/mathlib 标准公理 `propext`、`Classical.choice`、`Quot.sound`。

## 表示约定

- `ChannelIR.QState n = Fin n → Fin 2`，状态空间基数证明为 `2^n`。通用矩阵引理允许任意有限寄存器，因此也覆盖各个量子位寄存器。
- ChannelIR 的类型判断允许任意复系数（包括零）与空和。`SurfaceKrausCanonical` 的非零条件是规范化后的额外性质。旧 `collectFinite` 的符号收集已在 `GuardedCollection` 中展开为真实的逐项合并和删零路径。新版 Proposition 4.4 在共同基的固定顺序规范形式上给出 `Guarded.Rewrites` 路径，包含酉混合、逐项收集和零分支删除，覆盖原文“up to term and branch order”。
- `ChannelIR.Guarded` 使用论文的展平项表。`zp` 计入 primitive 的 Pauli 相位，只计严格正实数有效系数；R-Phase 严格增加 `zp`，R-MergeKraus 要求至少两个分支。五种确定性规则按“分支数、项数、项数减 zp”词典序严格下降，因而终止且存在可达正常形。排列只调整实质规则的表示，不作为独立重写步骤；正常形定义为没有可用确定性规则，不声称唯一。
- 完整 `Guarded.Rewrite` 另含实际展平的酉混合，`Guarded.PaddedRewrites` 允许零 Kraus padding。`ChannelIR.Rewrite` 保留为更一般的代数接口；其无守卫相位规则及结构收集不用于终止性结论。
- Block-encoding 使用谱/算子范数。外层成功操作从初始化密度矩阵、实际电路共轭、块寄存器投影和 Kraus 寄存器部分迹推导，没有把 Φ/Λ 定义成成功操作。
- `diamondNorm` 是所有有限辅助维度上迹范数单位球的上确界，取扩展非负实数值以避免无界实数上确界的陷阱。`diamondNorm_le_iff` 证明有限非负误差界与这一单位球上的逐点迹范数界等价；未另行形式化它与论文密度态最大值写法之间的等价性。
- Proposition 5.1 对任何合法注入 placement 成立，显式使用同一 PREPARE 的平方根幅度，证明 α 不变与未使用地址零振幅。定义 2.1 要求 α>0；全零 Pauli 输入不属于这一 block-encoding 接口。
- Proposition 6.1 从实际采样轨迹构造 Pauli 语法输入，并直接接受新版确定性重写路径。输出 Pauli 系数秩保持由 CP 语义与 Choi Gram 关系推出；不要求调用方另行假设矩阵清理证书。正常形存在性也已连接到秩界。
- Theorem 6.2 直接接受新版六类规则及补零路径，调用 Proposition 5.1 和实际 channel-LCU 电路证明。保留论文已有的前端误差界、精确状态制备和精确门实现假设。它证明未归一化成功操作及其乘以 Λ 后的误差界；并非条件归一化后的输出误差。
- `PauliCompiler` 使用真实 Physlib Pauli 张量矩阵、二进制残差、计算得到的相位补偿以及实际单调受控门乘积。`active` 可包含零系数 padding 标签；其他地址的系数和准备重叠为零。外层注入地址表显式补单位操作与零振幅。

Example 5.3 的下界采用修正后的论证：零地址 residual 只贡献仿射偏移，完整 Pauli 覆盖迫使非零地址 residual 张成全部 2n 维二进制空间，故 weighted Pauli control cost 至少为 2n。这一结论已连接实际 phased-Pauli 乘积，但未证明 one-hot 构造达到下界，也不把该成本解释为包含纯相位门的全部电路门数。PDF 中“每个非平凡 residual 都至少成本一”的原句仍需修正。

本库验证论文中的数学模型和电路矩阵构造。它不包含对仓库 Python 编译器、搜索返回值、硬件门实现或实验结果的代码级验证；Example 5.2 的最优性、Example 5.3 的完整最优性、附录 B.1 的分组加强界及前端更高阶误差分析也不在上述结论覆盖范围内。
