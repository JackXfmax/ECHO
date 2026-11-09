# MMHalSnowball 论文写作素材总览（EXP1–EXP4 全过程 + 全部数据）

> 生成时间：2026-10-29。写作前请以本文件和 `EXPERIMENT_LOG.md` 为准。
> **唯一事实来源是 `EXPERIMENT_LOG.md`**；本文件是其面向写作的重排汇总。

---

## 0. 一句话定位

**研究问题**：在多轮真实对话中，一旦模型接受了用户注入的虚假前提，幻觉是否会"滚雪球"式累积？模型能否自愈？这是哪种机制？

**核心 claim（三句话）**：
1. 对话式雪球脆弱性**普遍存在**——跨开源与闭源前沿模型均可观测；
2. 但**开源模型显著更严重、且更不自愈**（开源 .82–.95 幻觉率 / 自愈 .06–.28 vs 前沿 .44–.63 / 自愈 .33–.42）；
3. 自愈的**机制不是"视觉注意力回涨"**，而是稀疏、符号异质的少数注意力头重分配（全局 VAR 无效应）。

---

## 1. 数据集与实验协议（Method 主体）

### 1.1 数据构建

- 源数据集：`MM-Hallu/MMHalSnowball`（train split）。
- **注入方案与 50 条协议完全同构**（用户 2026-10-29 明确要求"保持和之前 50 条一致"），由 `construct_inject_v3.py` / `construct_inject_v3_1000.py` 生成；EXP4 用 `INJECT_N` 环境变量放大到 1000。
- **4 类注入 × 2 种强度 = 8 个注入轮**，第 9 轮为原始数据集核心问题（`dataset_final`，用真实标注判定）：

| 轮次 | 类别 | mode |
|---|---|---|
| r1 / r2 | `entity_missing` | lead / escalate |
| r3 / r4 | `attribute_reversal` | lead / escalate |
| r5 / r6 | `background_fabrication` | lead / escalate |
| r7 / r8 | `context_pollution` | lead / escalate |
| r9 | `dataset_final` | final |

- **lead** = 首次植入虚假前提；**escalate** = 在已被接受的虚假前提上进一步加码追问。
- 6 个 `conversation_setting`（EXP1/EXP2 全用；EXP4 只用 `halluconv_formatting`）：
  `cleanconv_formatting`、`cleanconv_question`、`factconv_formatting`、`halluconv_question`、`halluconv_formatting`、`irrconv_formatting`。
  - `halluconv_*` = 前缀对话本身含幻觉（HALLU-prefix）；`cleanconv_*` = 前缀干净（clean-prefix）；`factconv` / `irrconv` = 其他两类前缀。

### 1.2 推理设置（关键，Method 必写）

| 项 | 值 | 为什么 |
|---|---|---|
| vision_mode | **`every_round`（每轮都传同一张 PIL 图）** | 若只在首轮传图，chat-template 每轮仍渲染 `<image_pad>` 但 `pixel_values=None` → 从第 2 轮起"看不到图"，实测 11%→57% 飙升，会与自变量混淆（2026-10-24 血案） |
| temperature / top_p | 0.7 / 0.9 | T=0.1 会让 K=3 条轨迹几乎相同、无法估计方差 |
| K（轨迹数） | 3（seed=42） | 每样本 3 条独立采样轨迹 |
| 引擎 | transformers batch（**非 vLLM**） | 服务器无外网、磁盘吃紧；batch=6，OOM 自动对半降级 |
| max_pixels | 320000 | 显存上限保护 |
| conditions | baseline（第 1 轮，无注入）+ injection（9 轮） | — |

### 1.3 裁判（Judge）

- 模型：**DeepSeek-V4.1-Flash-1017-Event**，端点 `https://api.scnet.cn/api/llm/v1/chat/completions`。
- Prompt：**无记忆三字段** —— 只给 `current_question` / `ground_truth` / `model_output`，**不给对话历史**。这样每轮判定独立，避免"裁判看到模型之前在幻觉"而连带判幻觉。
- 输出严格 JSON：`{"is_hallucination": bool, "hallucination_type": str}`。
- **EXP1 全程单一裁判**（9000 条，0 error）；EXP2/EXP4 同源同模型，保证跨实验可比。
  - `judge_output_mixed_backup.jsonl` 是历史混合裁判留底，**仅 provenance，不得作为结果引用**。

### 1.4 统计方法

- 率 + **95% bootstrap CI**，bootstrap **按 sample（图）簇重采样**（2000 次）——同一图下多轨迹不独立，不能逐行 bootstrap。
- 配对比较用**同一重采样簇集合**算差值 CI。
- 状态转移矩阵：H→H / H→F / F→H / F→F；**自愈率 = H→F / (H→F + H→H)**。
- 注意力：层内置换检验 + max-|Δ| 统计量 + 分半复现（**不要**用 784 单元 bootstrap + Bonferroni，必然假阳性）。

---

## 2. EXP1 — `main_qwen_9000`（主实验，✅ 已冻结）

**定义**：Qwen2.5-VL-7B-Instruct，50 图 × 6 setting × [baseline 3 轨迹×1 轮 + injection 3 轨迹×9 轮] = **9000 判定单元**。单裁判 0 error。已审计冻结（`EXP1_main_qwen_9000/AUDIT_REPORT.txt`）。

### 2.1 主效应：baseline vs injection（全轮，按 setting）

| setting | baseline | injection | delta |
|---|---|---|---|
| cleanconv_formatting | 0.133 [0.060, 0.213] | 0.689 [0.649, 0.728] | +0.556 |
| cleanconv_question | 0.113 [0.053, 0.180] | 0.690 [0.646, 0.731] | +0.576 |
| factconv_formatting | 0.120 [0.053, 0.200] | 0.650 [0.605, 0.693] | +0.530 |
| halluconv_question | 0.133 [0.073, 0.200] | 0.704 [0.666, 0.740] | +0.570 |
| halluconv_formatting | 0.127 [0.060, 0.200] | 0.711 [0.674, 0.750] | +0.584 |
| irrconv_formatting | 0.047 [0.013, 0.087] | 0.694 [0.650, 0.734] | +0.647 |

> **汇总口径**：baseline 0.047–0.133；injection 0.650–0.711；**delta +0.530 – +0.647，CI 均不含 0**。

### 2.2 最终轮（核心问题）：baseline r1 vs injection r9（配对）

| setting | baseline r1 | injection r9 | delta (配对 CI) |
|---|---|---|---|
| cleanconv_formatting | 0.133 | 0.473 | +0.340 [0.240, 0.460] |
| cleanconv_question | 0.113 | 0.440 | +0.327 [0.233, 0.420] |
| factconv_formatting | 0.120 | 0.393 | +0.273 [0.180, 0.380] |
| halluconv_question | 0.133 | 0.527 | +0.393 [0.293, 0.487] |
| halluconv_formatting | 0.127 | 0.547 | +0.420 [0.313, 0.520] |
| irrconv_formatting | 0.047 | 0.493 | +0.447 [0.353, 0.547] |

### 2.3 逐轮雪球曲线（injection，r1–r9）

| round | 类别 | rate | 95% CI |
|---|---|---|---|
| 1 | entity_missing | 0.923 | [0.876, 0.966] |
| 2 | entity_missing | 0.979 | [0.948, 0.997] |
| 3 | attribute_reversal | 0.570 | [0.516, 0.623] |
| 4 | attribute_reversal | 0.639 | [0.588, 0.691] |
| 5 | background_fabrication | 0.500 | [0.436, 0.571] |
| 6 | background_fabrication | 0.826 | [0.773, 0.871] |
| 7 | context_pollution | 0.659 | [0.608, 0.710] |
| 8 | context_pollution | 0.631 | [0.568, 0.696] |
| 9 | dataset_final | 0.479 | [0.408, 0.543] |

> **非单调震荡**（不是线性上升/下降），峰值 r2 = 0.979、谷值 r5 = 0.500。这是"雪球"的真实形态：随注入类别与强度波动，但全程远高于 baseline。

### 2.4 按注入类别（r1–r8，各类 n=1800）

| 类别 | rate | 95% CI |
|---|---|---|
| **entity_missing** | **0.951** | [0.909, 0.978] |
| background_fabrication | 0.663 | [0.616, 0.710] |
| context_pollution | 0.645 | [0.595, 0.698] |
| attribute_reversal | 0.604 | [0.558, 0.651] |

> `entity_missing` 最致命（0.951）：一旦被诱导承认图中不存在的实体，几乎必幻觉。

### 2.5 自愈与交互

- **自愈率 28.0%**：幻觉之后的轮次共 5154 个，其中 1443 个下一轮恢复忠实（HH=3711, HF=1443, FH=1043, FF=1003）。
- **前缀交互（HALLU-prefix 是否放大注入伤害）**：
  - halluconv 前缀 mean delta = **+0.407** [0.323, 0.490]
  - cleanconv 前缀 mean delta = **+0.336** [0.250, 0.420]
  - **INTERACTION = +0.070**（两 CI 有重叠，保守读法：方向为正但 CI 重叠，不宜强 claim）
- **退化率（模型说"看不到图"）= 0.08%**（STRICT 正则，7/9000）。⚠️ 见第 6 节口径提醒。

---

## 3. EXP2 — `crossmodel`（跨模型，✅ 全部完成）

**定义**：5 个对比模型，各 **12 图 × 6 setting × 1 轨迹 × 9 轮 = 648 条**，**injection-only**，同源 DeepSeek 裁判，0 error。

### 3.1 六模型横向（injection-only）

| 模型 | n | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | OVERALL | 自愈 H→F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B (EXP1) | 8100 | .92 | .98 | .57 | .64 | .50 | .83 | .66 | .63 | .48 | .69 | .28 |
| LLaVA-1.5-7B (EXP2a) | 648 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .57 | 1.00 | **.95** | **.06** |
| InternVL2-4B (EXP2b) | 648 | 1.00 | 1.00 | .92 | 1.00 | 1.00 | 1.00 | .50 | .61 | .33 | **.82** | **.15** |
| GPT-4o (EXP2c) | 648 | .75 | .75 | .21 | .32 | .78 | .86 | .35 | .61 | .29 | .55 | .37 |
| Gemini-2.5-Flash (EXP2d) | 648 | .86 | .92 | .50 | .56 | .72 | .85 | .25 | .65 | .33 | .63 | .33 |
| Claude-Sonnet-4.6 (EXP2e) | 648 | .85 | .83 | .31 | .32 | .39 | .36 | .22 | .26 | .39 | .44 | .42 |

> **横向只比定性模式**（是否震荡、自愈率是否 >0），**不比绝对值**——各模型基线能力不同，且 EXP1 的 .69 来自 8100 单元、其余来自 648 子集。

### 3.2 关键结论（🔴 已推翻早期误判，必须照此写）

1. **三大前沿模型全部中招**，并非"抵抗注入"：GPT-4o .55 / Gemini .63 / Claude .44，均呈非单调震荡。
   - 早期"GPT-4o 全程抵抗"仅来自单条冒烟（sample 00107236），无统计意义，**已废弃**。
2. **Gemini 早期"0.000 幻觉率"是 judge artifact**（scnet.cn 夜间 DNS 失败使 648 条返回 error 被误计 faithful），已清理重判修正为 0.63。
3. **结论升级**：雪球脆弱性是跨开闭源普遍现象，但**开源严重得多且更不自愈**。
4. InternVL2 补充读数：幻觉中 `matched_false_premise=True` 占 **460/530 = 0.868** —— 绝大多数幻觉确由注入的虚假前提驱动，非随机胡说。

### 3.3 架构多样性表（论文用这张支撑"跨架构"论断）

| 模型 | 视觉编码器 | 投影/分辨率 | LLM 主干 | 类型 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | Qwen ViT（原生分辨率 + MRoPE） | MLP merge | Qwen2.5-7B | 开源 |
| LLaVA-1.5-7B | CLIP ViT-L/14（固定 336px） | 2 层 MLP | Vicuna-7B | 开源 |
| InternVL2-4B | InternViT-300M（448px 动态分块） | pixel-shuffle | Phi-3-mini-3.8B | 开源 |
| GPT-4o | 闭源（未公开） | — | GPT-4o | 闭源前沿 |
| Gemini-2.5-Flash | 闭源（未公开） | — | Gemini | 闭源前沿 |
| Claude-Sonnet-4.6 | 闭源（未公开） | — | Claude | 闭源前沿 |

> 3 种开源 ViT + 3 种开源 LLM 主干 + 3 种闭源前沿架构 → 足以支撑跨架构普适性论断。

---

## 4. EXP3 — `attention_mechanism`（机制，✅ 已跑完）

**定义**：Qwen2.5-VL-7B，强制 `attn_implementation="eager"`。
- 主实验前向 **2763 次**（1407 自愈 H→F + 1356 持续幻觉 H→H，**严格历史分层配对**）
- 忠实参考臂 **668 次**（F→F，仅参照，**不参与因果对比**）
- 机制分解臂 **2763 次**

### 4.1 方法：STRATIFIED MATCHING（关键，Method 必须写清）

��比同一轮 r 上两种结局，且**前 r−1 轮幻觉历史逐位相同**：
- 分层键 = 前 r−1 轮 H/F 的**精确 bitmask**（不只是"幻觉次数"）
- 仅保留同时含两种结局的 stratum（**80 个，覆盖 1356 对**），按 `w = min(n_heal, n_persist)` 加权
- query 取**回答 token**（非全序列，否则被上下文长度机械稀释），key 取 `<image_pad>` token
- 指标：**Visual Attention Ratio (VAR)**，逐层逐头

> ⚠️ 早期 naive 的"幻觉轮 vs 自愈轮"对比是**混淆的**（自愈轮历史更长、注意力被稀释），已废弃，**不得引用** `attn_recovery.json` 的 mean delta = −0.018。

### 4.2 🔴 核心结果：原假设不成立（全局视觉注意力没有反弹）

| 指标（深层 20–28） | self-heal | persist | Δ | 95% CI | 置换检验 p |
|---|---|---|---|---|---|
| VAR（分层后） | 0.02548 | 0.02482 | **+0.00066** | [−0.00031, +0.00164] | **0.116** |

- **5 种分层方案全部 Δ≈0 且 CI 含 0**：(r,bitmask) +0.00066 / (图×setting×r,bitmask) −0.00036 / (图×setting×r,次数) −0.00005 / (图×setting×r) +0.00030 / (r,次数) +0.00067。
- 方差分解：**81% 的 VAR 方差来自"图像×setting×轮次"身份**；即便把图像身份也纳入分层仍为 0 → **排除"功效不足"解释**。
- 逐轮（r=3 / r=7，自愈最集中的两轮）同样无效应。
- 机制分解（图像 / 自己历史回答 / 历史用户轮 / 当前问题 / 自注意力）效应量均 **1–4%**，CI 多数含 0 且**符号在 query 集间翻转** → 噪声。

### 4.3 🟢 但存在局部、可复现的稀疏签名

层平均看不到，是因为效应集中在少数 (层,头) 单元且**正负相消**：

- 784 个 (层,头) 单元中 **4–6 个**超过置换检验 95% max-|Δ| 阈值（family-wise p < 0.005）。
- 稳定单元：**L22H2 Δ=−0.0115**（自愈时图像注意力**更低**）、**L20H24 +0.0102**、**L15H24 +0.0087**、L22H26 +0.0079、L23H14 +0.0077 —— 集中在第 **15/20/22/23** 层。
- **分半复现** r = **0.79**（0.61–0.92），top-50 符号一致 **90%**，top-K 重叠远超随机（K=50 时 31 vs 随机 3.2）。
- **非长度伪影**：对 `n_answer_tokens / seq_len / n_image_tokens` 残差化后与原始 Δ 相关 **0.970**，最大单元未缩小（0.0115→0.0133）。
- 自愈答案比持续幻觉短约 3.9 token、序列短 28.7 token（层内残留不平衡，已被残差化排除）。

### 4.4 可写进论文的英文口径（直接用）

> Under exact history matching, self-healing is **not** accompanied by a global rebound of visual attention (Δ deep-layer VAR = +0.0007, 95% CI [−0.0003, +0.0016], permutation p = 0.12; null replicates under five stratification schemes including image-identity matching). The effect is instead **sparse and sign-heterogeneous**: a handful of mid-to-deep attention heads shift reliably (family-wise permutation p < 0.005, split-half r = 0.79), and they do not all move in the same direction.

**禁止**：不得因"预期是飙升"而挑一两个头包装成全局结论；不得把未分层的 delta 写进公式。

---

## 5. EXP4 — `mega_qwen_1000`（1000 图放大，✅ 已跑完入账）

**定义**：1000 图 × Qwen2.5-VL-7B-Instruct × `halluconv_formatting` 单 setting（与 50 条协议同构）。conditions = baseline(1 轮) + injection(9 轮)，K=3。
**裁判单元 30000 = baseline 3000 + injection 27000**（有效 **29999**，1 条模型安全拒答排除为 `undetermined`），同源 DeepSeek 裁判，**0 error**。

> ⚠️ **隔离铁律**：EXP4 是 EXP1 协议的 1000 图放大稳定估计，**数字绝不进 EXP1 主表、绝不合并/平均**；引用一律标 EXP4。

### 5.1 核心结果

| 指标 | 值 |
|---|---|
| **injection 整体幻觉率** | **0.736** |
| **baseline 第 1 轮** | **0.182** [0.164, 0.200] |
| **主效应 delta** | **+0.555**（CI 排除 0 ⇒ 显著） |
| 逐轮 r1–r9 | .930 / .982 / .594 / .718 / .543 / .863 / .686 / .737 / .573 |
| 状态转移 H→H / H→F / F→H / F→F | **13773 / 4386 / 3314 / 2526** |
| **自愈率 H→F/(H→F+H→H)** | **0.242** |
| 曾幻觉轨迹 ever_H | 3000/3000（1.000） |
| 其中第 9 轮恢复忠实 | 1282/3000（0.427） |

逐轮（含 95% cluster-bootstrap CI）：

| round | rate | CI_low | CI_high |
|---|---|---|---|
| 1 | 0.930 | 0.918 | 0.941 |
| 2 | 0.982 | 0.975 | 0.988 |
| 3 | 0.594 | 0.574 | 0.615 |
| 4 | 0.718 | 0.699 | 0.735 |
| 5 | 0.543 | 0.522 | 0.563 |
| 6 | 0.863 | 0.847 | 0.878 |
| 7 | 0.686 | 0.666 | 0.705 |
| 8 | 0.737 | 0.718 | 0.756 |
| 9 | 0.573 | 0.551 | 0.593 |

### 5.2 与 EXP1 的关系（写法）

- **方向/机制一致**：EXP1 逐轮 .923/.979/.570/.639/.500/.826/.659/.631/.479，EXP4 放大后曲线形态同构；自愈率 EXP1 0.280 vs EXP4 0.242，量级一致。
- **baseline 0.182 高于 EXP1 的 0.047–0.133**：因 EXP4 仅用 `halluconv_formatting` 单 setting（EXP1 为 6 setting 混合），样本分布不同所致；**这是 EXP4 自身读数，不回写 EXP1 主表**。
- EXP4 的价值：**1000 图使 CI 大幅收紧**（如 r1 CI 宽度从 EXP1 的 ±0.045 降到 ±0.012），给出稳定的率估计。

---

## 6. 🚨 动笔前必须遵守的口径（逐条卡）

| # | 事项 | 要求 |
|---|---|---|
| 1 | **退化率正则口径** | 存在两个数：**0.5%**（早期宽松正则，会把 context_pollution 的礼貌话术 "if you have any questions" 误判为"看不到图"）与 **0.08%**（`audit_main_experiment.py` 的 `STRICT` 正则，7/9000）。两者正则不同、不矛盾。**统一写 0.08%**，并在 Method/footnote 给出正则定义，否则与早期草稿的 0.5% 对不上。 |
| 2 | **数字归属标注** | 每张表/图注明来自 EXP1 / EXP2 / EXP3 / EXP4；**禁止跨轮合并或平均**。主表数字只来自 EXP1；EXP2/EXP3 是小样本验证；EXP4 是独立放大实验。 |
| 3 | **注意力结论** | 全局 VAR **无反弹**（Δ=+0.0007, p=0.12），只有 4–6 个 (层,头) 稀疏签名。**不得写成"自愈 = 视觉注意力回涨"**。 |
| 4 | **裁判一致性** | EXP1 全程单一 DeepSeek 裁判（9000 条、0 error），Method 需说明；`judge_output_mixed_backup.jsonl` 仅 provenance，不作结果。 |
| 5 | **前沿模型不写"抵抗"** | 三大前沿全部中招（.44–.63）。claim 写"跨开闭源普遍存在，但开源显著更严重且更不自愈"。**不得**沿用已废弃的"GPT-4o 抵抗"旧口径。 |
| 6 | **两处 648 勿混** | 跨模型 648 = 12 图 × 6 setting × 1 轨迹 × 9 轮（injection-only）；注意力 648 = 同一批 72 轨迹 × 9 轮前向。**来源不同**。 |
| 7 | **逐轮曲线须用 injection-only** | 若把含 baseline 的全量数据喂给逐轮统计，r1 会被 baseline 稀释（0.518 ≠ 0.923）。 |
| 8 | **EXP4 1 条缺失** | 30000 单元中 1 条因模型安全拒答无法判定（`08276900|injection|t1|r4`），已标 `undetermined` 排除，报告有效 29999（缺失率 0.003%）。 |

---

## 7. 产物与图表清单

| 文件 | 说明 |
|---|---|
| `EXPERIMENT_LOG.md` | **唯一事实来源**（总账） |
| `result_judge_analysis.md` | EXP1 主结果（6 setting × 条件、逐轮、类别、交互、自愈） |
| `EXP4_MEGA_ANALYSIS.md` + `fig_mega_1000.pdf` | EXP4 完整统计与图（逐轮率曲线 + 状态转移） |
| `figs/figC_layer_var.png`、`figs/figC_rounds_r3_r7.png` | EXP3 逐层 VAR（三臂 + Δ 面板）、r3/r7 |
| `figs/figD_head_signature.png` | EXP3 28×28 头签名热图 |
| ~~`figs/figATT*_selftest.png`~~ | **合成数据，已废弃，勿引用** |
| `judge_input_exp4.jsonl` / `judge_output_exp4.jsonl` | EXP4 裁判输入输出（30000 / 15.8MB） |
| `EXP1_main_qwen_9000/` | EXP1 冻结件（含 `AUDIT_REPORT.txt`） |

**待补（论文 Figure 缺口）**：
- [ ] LLaVA / GPT-4o / Gemini / Claude / InternVL 五组逐轮曲线与 EXP1 的对比图（`EXPERIMENT_LOG.md` 待办 #2）
- [ ] EXP1 与 EXP4 逐轮曲线叠加图（体现放大后 CI 收紧）——可用 `analyze_exp4.py` 数据 + EXP1 曲线手工叠

---

## 8. 推荐论文叙事线（供参考）

1. **Intro**：多轮对话中虚假前提一旦被接受，是否会累积？现有 hallucination benchmark 多为单轮 → 提出 MMHalSnowball 多轮注入协议。
2. **Method**：数据构建（4 类 × lead/escalate × 6 setting × 9 轮）、`every_round` 传图、无记忆裁判、cluster bootstrap、STRATIFIED MATCHING。
3. **Main result（EXP1）**：主效应 delta +0.53–0.65，逐轮非单调雪球，entity_missing 最致命 0.951，自愈仅 28%。
4. **Generality（EXP2）**：6 模型跨架构，开源 vs 前沿对比，架构多样性表。
5. **Scale（EXP4）**：1000 图放大，CI 收紧，确认稳定性（baseline .182 / injection .736 / 自愈 .242）。
6. **Mechanism（EXP3）**：全局 VAR 无反弹（证否直觉假设），稀疏符号异质头签名（正面发现）。
7. **Discussion**：为什么不是"注意力回涨"？可能是不确定性/语言先验驱动；缓解方向（拒绝虚假前提、显式引用图像）。
