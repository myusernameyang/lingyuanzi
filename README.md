# 灵元子 (Lingyuanzi)

**一种基于「先验方向 + 内容表」互补注意力的持久记忆语言模型**

> A brain-inspired byte-level language model with persistent memory — replacing KV cache with a fixed-length "direction state".

---

## 一句话

用「持久方向状态 **g** + 内容表 **contents**」替代 Transformer 的 KV 缓存，实现**跨序列（cross-batch）的持久记忆**；这套机制天然是「先验 + 预测误差 + 纠偏」的**预测编码（predictive coding）**——一个**状态级的世界模型**，而不是像素级的生成模型。

---

## 核心思想

```
g（快权重矩阵 [d,d]）     ← 持久状态，跨 batch 携带记忆，不随序列长度增长
contents（知识矩阵 [d,d]） ← 可学习的知识权重
Rh = R @ h（偏转）         ← 「选择头」（激活锚点）
query = normalize(g @ Rh)      ← g 投影 Rh → 锚点方向
content = contents @ query      ← 知识权重投影锚点 → 检索记忆
error = (h - h.mean) - content  ← 具体 − 记忆 = 知识冲突
g = normalize(g + error ⊗ Rh)   ← 外积快权重累积（连加，无衰减）
```

**核心一句**：`content = contents @ normalize(g @ (R @ h))` —— g（快权重矩阵）把「当前偏转 Rh」投影成「锚点」，contents（知识矩阵）再把锚点投影成「检索出的记忆」；预测误差（`h - h.mean - content`）以**外积**（`error ⊗ Rh`，即 Wqkv 的「梯度 = 误差 ⊗ 输入」）写回 g。

**g 连续累积（快权重）**：`g_j = normalize(g_{j-1} + error_j ⊗ Rh_j)` —— 外积连加（无衰减，外层 normalize 钳幅），等价于「连加 + 并行 cumsum」，O(S·D²) 线性。

---

## g 更新版本演进（结果都标了在哪个版本下测得）

| 版本 | g 更新方式 | 该版本下测得的结果 |
|---|---|---|
| **v67 并行 g** | 逐位置独立，取最后位置 carry | 短程 toy 4.0x~10.4x、长程 1.0x、补全 58%（窗口 KV，g 消融无贡献） |
| **v68 content-attends-to-h** | g 仍并行，纠偏改成 content 定位相关 token | 近因偏置修正 4.7x、自然数据 ce 1.195 |
| 顺序/加性/hybrid/cumprod g | 各种「连续累积」尝试 | 长程都 1.0x（被稀释/洗没） |
| **v69 cumprod 连续 g + query 门控 read** | g_j = g_init ⊙ ∏(1+normalize(query_k))，read = query ⊙ contents | 正在从零训练，长程待测 |
| **v70 快权重 g（矩阵）+ 锚点检索** | g=[d,d] 矩阵，query=normalize(g@Rh)，content=contents@query，g=normalize(g + error⊗Rh) | **correct-vs-wrong 5/5 全对**（合成 5-fact） |

## 关键结果（2026-10-29）

| 实验 | 结果 |
|---|---|
| g 跨 batch 记忆（正确 g vs 错误 g 的 P(答案) 比值，短事实） | **4.0x~10.4x** |
| 近因偏置修正（答案在句中） | 1.0x → **4.7x** |
| 相近因果补全（滑动窗口 8-32 字节，自然数据） | **58%** 字节准确率（窗口 KV 局部补全；g 消融显示有/无 g 都是 58%，非 g 贡献） |
| 自然数据训练（20GB，60000 步） | ce 2.57 → **1.195** |
| **快权重 g（矩阵）+ 锚点检索，correct-vs-wrong** | **5/5 全对**（Alice/Bob/Carol/David/Eve 各自正确 g 给最高 P(答案)）；新名字能写入（Frank/Rome → P(R)=0.0495） |

---

## 定位：状态级世界模型

```
预测（g 先验）→ 观测（h）→ 预测误差（expl_err）→ 纠偏（更新 g）
```

完整的**预测编码回路**，与 2026 年世界模型前沿（PhiZero 的「物理语言」、V-JEPA 2 的「潜空间预测」、自由能原理）同一方向。

**差异化**：

| | Transformer | 灵元子 |
|---|---|---|
| 长上下文记忆 | KV 缓存（O(S)） | g（定长，跨 batch 携带） |
| 表示 | 符号/像素级 | 连续状态级（因果可解释） |
| 可解释性 | — | 因果干预（换 g → 观测概率变化） |

---

## 当前状态

- ✅ 短程跨 batch 回忆（合成数据 4.0x~10.4x），近因偏置修正 4.7x；
- ✅ 自然数据训练 ce 1.195（60000 步）；
- ✅ **快权重 g（矩阵）+ 锚点检索：correct-vs-wrong 5/5 全对**（记忆版本第一次成功，2026-10-31）；
- ⚠️ 相近因果补全 58%（窗口 KV 局部补全，有 g 无 g 一样，**非 g 实现**）；
- 🔄 覆盖（同结构替换）半成品：无衰减加法是「混合」不是「替换」，待加指数衰减 `g = λ·g + outer`；
- 🔄 提速：外积物化 [S,d,d] 太慢，待改「顺序累加进 [d,d]」；
- 🔄 源码暂未开源（整理中）。

## 作者

**yangtianpeng** · 2026-10-31

*本文档用于记录研究优先权（priority），完整源码与训练细节后续开放。*
