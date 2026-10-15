# main_v2.pdf 定理待办清单

来源固定于 `paper-source.json` 的 SHA-256；页码为 PDF 页码。正文含 9 项编号结论。勾选只表示对应的完整结论已通过 Lean 编译及公理检查；局部引理或附加假设版本不会计为完成。两个独立审查者的语义核验完成前，不能将“可编译”称为“与原文完全一致”。

## 前置验证

- [x] 检查常见本机安装位置：未发现已有 Lean/Lake/Elan。
- [x] 在隔离目录安装 Lean 4.34.1 / Lake，验证官方安装包 SHA-256。
- [x] 固定 Physlib a4ef46125b32bb2599ab42d9c3a95e1d88ca75ff、mathlib v4.34.1 和传递依赖。
- [x] Quantum smoke：任意 Kraus 家族完全正性、Pauli 酉性；无 `sorryAx`，仅标准公理。

## 正文结论

- [x] **Theorem 4.1**（p. 7）：良类型 ChannelIR 的 Kraus 语义是作用于 n-qubit 算子的完全正线性映射；不把 TP 当作自动结论。
- [x] **Lemma 4.2**（p. 8）：新版六类重写及有限序列保持维度类型；T-Kraus 允许零系数，T-Zero 允许空和。入口为 `Guarded.Rewrite.wellTyped` / `Rewrites.wellTyped`。
- [x] **Theorem 4.3**（p. 8）：新版 R-Phase 的 `zp` guard、R-MergeKraus 的至少两分支限制，以及全部六类规则均保持整个线性映射。入口为 `Guarded.Rewrite.sound` / `Rewrites.sound`。
- [x] **Proposition 4.4**（pp. 8–9）：相同 CP map 的两个共同基规范 Kraus families，r ≤ m 时，可经酉混合、真实逐项合并/删零及零分支删除到达目标。`GuardedCollection` 展开批量收集；`Guarded.proposition_4_4_guarded` 直接给出新版六规则的有限路径。
- [x] **Proposition 4.5**（p. 9）：从实际外层电路的后选择与 partial trace 推出 I_succ(ρ)=Φ(ρ)/Λ，成功概率 tr(Φ(ρ))/Λ；TP 时为 1/Λ。
- [x] **Proposition 4.6**（p. 10）：rank(A)=r≥1 时，构造行酉混合和零行删除，得到 r 个非零行且 pc(B)≤rN−r(r−1)/2；推出严格减少条件。
- [x] **Proposition 5.1**（p. 14）：原始系数/Pauli 项经注入 placement、可选零系数 padding 标签、共享 PREPARE 和实际残差相位恢复电路后，仍为同一原始和的精确 α-block-encoding，α 不变，未使用地址振幅为零。入口 `PauliCompiler.proposition_5_1`；按定义 2.1 保留 α>0。
- [x] **Proposition 6.1**（p. 16）：由实际有限阶漂移/跳跃积推导采样系数矩阵秩 ≤ min(M,N,choose((k+1)p+k,k))，直接覆盖新版确定性规则（包括相位归一化）及其正常形。入口 `FrontendCleanup.proposition_6_1_guarded` / `exists_sampled_normalForm_rank`。
- [x] **Theorem 6.2**（p. 17）：新版六类规则及补零路径直接连接 Proposition 5.1、实际 SELECT 和 channel-LCU，得到 I_succ=Φδ/Λ、成功概率及 rescaled successful map 的 diamond error ≤ ε_front；前端误差保留为论文原有假设。入口 `GuardedCompiler.theorem_6_2_guarded`。

旧版编号 2.2 已撤下；`LCU.theorem_2_2` 及未使用地址版本保留为通用辅助定理，不计入当前 9 项结论。

## 直接依赖与交付核验

- [x] 附录 C.1（p. 36）：注入地址重标记保持 LCU 的系数–算子配对和 postselected block。
- [x] 附录 C.2 / C.4（pp. 36–37）：实际 monotone-control 门乘积、Pauli XOR 残差、矩阵相位恢复、增量更新及 weighted-control 局部代价公式。
- [x] Section 4.2（p. 8）：证明新版五种确定性规则终止、不存在无限重写序列，且存在可达的无规则可用正常形；不声称正常形唯一或任意矩阵原语下的更强语义形状刻画。
- [x] Example 5.3（p. 15）的修正下界：非零地址残差张成全部 2n 维空间，weighted Pauli control cost ≥ 2n；已连接实际 phased-Pauli 乘积。未证明匹配该下界的构造或完整最优性。
- [x] 最终全库编译通过（3291 个构建任务），632 个项目定理常量（包含私有及生成定理）的传递公理审计通过，仅使用 propext、Classical.choice、Quot.sound；未使用 sorryAx 或额外公理。
- [x] 第一个新的无对话上下文审查者：独立核对当前 PDF、源码、定义和覆盖范围，重新构建/审计；确认正文九项及本次新增结果在 README 明示范围内无剩余实质问题。
- [x] 第二个新的无对话上下文审查者：独立核对当前 PDF、源码、定义和覆盖范围，重新构建/审计；确认正文九项及本次新增结果在 README 明示范围内无剩余实质问题。
- [x] 修复审查发现的 Proposition 4.4 路径缺口；更新当前 54 页 PDF 指纹，来源检查通过；重新打包并核对 Lean_formalization-src.tar.gz 内全部文件与交付源码逐字节一致。
