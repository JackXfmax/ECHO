# MMHalSnowball 实验版本总账

> **唯一事实来源**。每新增一轮实验，在此追加一条；任何数字引用前先查本文件确认归属。
> 目的：防止后续补充实验与第一轮结果混淆。

## 命名与隔离规则（硬约束）

1. **每轮实验一个 `EXP<n>_<codename>` 目录**，冻结件放该目录下，不再改动。
2. **产物文件名必须带本轮前缀**，禁止复用他轮文件名：
   - EXP1：无前缀（历史既定，见下表）
   - EXP2：`cross_model_*`
   - EXP3：`attn_*`
3. **判定数据不合并**：每轮有自己的 `*_judge_input.jsonl` / `*_judge_output.jsonl`。
   EXP1 的 9000 条是**唯一**的主结果来源，其他轮是小样本验证，**不得**混入主表。
4. **引用必标来源**：写作/投稿时写成
   `EXP1 (Qwen2.5-VL-7B, 50 images, 9000 units)`、
   `EXP2 (LLaVA-1.5-7B, 12 images, 648 units)` 等。
5. **冻结前必须跑审计**（`audit_main_experiment.py` 或本轮等价脚本），审计通过才允许冻结。
6. 冻结用**复制**不用移动，现网文件继续供在跑的实验使用。

---

## EXP1 — `main_qwen_9000` ✅ 已冻结

| 项 | 值 |
|---|---|
| 目录 | `EXP1_main_qwen_9000/` |
| 被测模型 | Qwen2.5-VL-7B-Instruct |
| 规模 | 50 图 × 6 setting × 2 条件 × 3 轨迹 → **9000 判定单元** |
| 裁判 | DeepSeek-V4.1-Flash-1017-Event（单裁判，0 error） |
| 冻结时间 | 2026-10-26 23:26 |
| 审计 | 全项通过（`EXP1_main_qwen_9000/AUDIT_REPORT.txt`） |
| 主结果 | `result_judge_analysis.md` |

**EXP1 权威数字（引用时以此为准）**

| 指标 | 值 |
|---|---|
| baseline 幻觉率 | 0.047 – 0.133 |
| injection 幻觉率 | 0.650 – 0.711 |
| injection − baseline delta | **+0.530 – +0.647**（CI 均不含 0） |
| 最终轮 r1 vs r9 配对 delta | +0.273 – +0.447 |
| entity_missing | **0.951** |
| background_fabrication | 0.663 |
| context_pollution | 0.645 |
| attribute_reversal | 0.604 |
| **自愈率 H→F** | **28.0%**（HH=3711 HF=1443 FH=1043 FF=1003） |
| 退化率（看不到图） | 0.08%（严格正则） |
| 逐轮曲线 | 0.923 / 0.979 / 0.570 / 0.639 / 0.500 / 0.826 / 0.659 / 0.631 / 0.479 |

**EXP1 已确认不存在的风险**：6 个 setting 各覆盖全部 50 图（排除去重类 bug）、
无缺失/孤儿 uid、判定单裁判无混用。

---

## EXP2 — `crossmodel` ✅ 全部完成（5 对比模型，覆盖 3 开源 ViT+LLM + 3 闭源前沿架构；EXP1 为主参照）

用户 2026-10-26 拍板：**只跑一个对比模型太薄，撑不起 "universal"**，追加 InternVL2-4B 与三大前沿 API 模型。
截至 2026-10-28：LLaVA-1.5-7B + InternVL2-4B + GPT-4o + Gemini-2.5-Flash + Claude-Sonnet-4.6 **全部完成并判定（各 648 条，0 error）**。

### EXP2 子实验索引

| 子实验 | 模型 | 规模 | 裁判 | 状态 | 产物前缀 |
|---|---|---|---|---|---|
| EXP2a | LLaVA-1.5-7B | 12 图 × 6 setting × 1 轨迹 × 9 轮 = **648 条** | DeepSeek（同源） | ✅ 完成 | `cross_model_*` |
| EXP2b | InternVL2-4B | 同左 = 648 条 | 同左 | ✅ 完成 | `cross_model_*_internvl` |
| EXP2c | GPT-4o | 同左 = 648 条 | 同左 | ✅ 完成 | `cross_model_*` |
| EXP2d | Gemini-2.5-Flash | 同左 = 648 条 | 同左 | ✅ 完成 | `cross_model_*` |
| EXP2e | Claude-Sonnet-4.6 | 同左 = 648 条 | 同左 | ✅ 完成 | `cross_model_*` |

> 全部 injection-only、同裁判 DeepSeek-V4.1-Flash-1017-Event，保证跨模型可比。
> 本地 frontier 探测走 `https://<REDACTED-PROXY-HOST>/v1` 中转（本地可达、4090 不可达），图片导出到 `cross_images/`（12 张）。

### 🔴 关键修正（推翻早期误判，必须同步到论文）

1. **三大前沿模型全部中招，并非"抵抗注入"**：GPT-4o / Gemini / Claude OVERALL = 0.55 / 0.63 / 0.44，且均呈非单调雪球震荡。
   早期"GPT-4o 全程抵抗"的结论仅来自单条冒烟（sample 00107236），不具统计意义，已废弃。
2. **Gemini 早期"0.000 幻觉率"是 judge artifact，非真实结论**：scnet.cn 夜间 DNS 解析失败导致 648 条 judge 返回 `error`，被错误计入 faithful。
   已清理 error 行并重判，修正为 0.63。当前四个 cross-model 文件均 0 残留 error（见 `verify_ledger.py` 输出）。
3. **结论升级**：对话式雪球脆弱性是**跨开闭源普遍现象**，但**开源模型严重得多、且自愈能力更弱**（见下方横向表）。
   —— 论文 claim 不再写"前沿模型抵抗"，改为"开源 LVLM 漏洞显著重于前沿闭源模型，但雪球效应在两类模型上均可观测"。

### 六模型横向对比（injection-only，同源裁判，已独立核验）

| 模型 | n | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | OVERALL | 自愈 H→F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B (EXP1) | 8100 | .92 | .98 | .57 | .64 | .50 | .83 | .66 | .63 | .48 | .69 | .28 |
| LLaVA-1.5-7B (EXP2a) | 648 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .57 | 1.00 | .95 | .06 |
| **InternVL2-4B (EXP2b)** | 648 | 1.00 | 1.00 | .92 | 1.00 | 1.00 | 1.00 | .50 | .61 | .33 | **.82** | **.15** |
| GPT-4o (EXP2c) | 648 | .75 | .75 | .21 | .32 | .78 | .86 | .35 | .61 | .29 | .55 | .37 |
| Gemini-2.5-Flash (EXP2d) | 648 | .86 | .92 | .50 | .56 | .72 | .85 | .25 | .65 | .33 | .63 | .33 |
| Claude-Sonnet-4.6 (EXP2e) | 648 | .85 | .83 | .31 | .32 | .39 | .36 | .22 | .26 | .39 | .44 | .42 |

> EXP2b 关键读数（同源 DeepSeek 裁判，648 条 0 error）：**OVERALL 幻觉率 0.818**、**自愈率 H→F = 0.150**（HH=430 HF=76 FH=28 FF=42）。
> 逐轮模式：r1–r6 几乎全中招（1.00/1.00/.92/1.00/1.00/1.00），r7 起骤降（.50/.61/.33）——即"长期饱和幻觉 + 末轮部分自愈"，与 Qwen/前沿的"非单调震荡"形态不同，但**同样证明对话式雪球在开源模型上最严重**。
> 幻觉中 `matched_false_premise=True` 占 460/530 = 0.868（即绝大多数幻觉确由注入的虚假前提驱动，而非随机胡说）。

> 横向只比**定性模式**（是否震荡、自愈率是否 >0），**不比绝对值**（各模型基线能力不同）。
> EXP1 的 .69 来自 8100 单元（50 图），其余来自 648 子集（12 图），样本量不同，绝对值不可直接横比。

### 架构多样性表（论文用这张）

| 模型 | 视觉编码器 | 投影/分辨率策略 | LLM 主干 | 类型 |
|---|---|---|---|---|
| Qwen2.5-VL-7B（EXP1） | Qwen ViT（原生分辨率 + MRoPE） | MLP merge | Qwen2.5-7B | 开源 |
| LLaVA-1.5-7B（EXP2a） | CLIP ViT-L/14（固定 336px） | 2 层 MLP | Vicuna-7B | 开源 |
| InternVL2-4B（EXP2b，已完成） | InternViT-300M（448px 动态分块） | pixel-shuffle | Phi-3-mini-3.8B | 开源 |
| GPT-4o（EXP2c） | 闭源多模态（未公开） | — | GPT-4o | 闭源前沿 |
| Gemini-2.5-Flash（EXP2d） | 闭源多模态（未公开） | — | Gemini | 闭源前沿 |
| Claude-Sonnet-4.6（EXP2e） | 闭源多模态（未公开） | — | Claude | 闭源前沿 |

→ **三种不同开源 ViT + 三种不同开源 LLM 主干 + 三种闭源前沿架构**，足以支撑跨架构普适性论断。

### 技术修复记录（排障留底）

- **LLaVA 多轮生成崩溃** `ValueError: Image features and image tokens do not match, tokens: 0, features: 2359296`：
  根因 transformers 5.20 的 LLaVA chat-template 多轮渲染缺陷（r2+ 渲染出 0 个 image token 且吞掉历史文本）。
  修复：放弃 `apply_chat_template`，改用官方手动 prompt 拼接（`USER: <image>\n{q} ASSISTANT:` 累积式），每轮仍传图。648/648 成功（`run_llava3.out`）。
- **judge 重判**：`run_glm_judge.py` 已有断点续跑 + 12 次重试；清理 `verdict` 非 faithful/hallucination 的 error 行后在 scnet.cn 恢复（200）后重判，四个文件最终 0 残留 error。
- **EXP2b InternVL2-4B 在 transformers 5.20.2 下生成崩溃 → 改用 4.51.3 影子环境（关键根因修复）**：服务器现网 transformers 5.20.2 与 InternVL 自定义代码（针对 4.x 写就）不兼容，报错链为 `Phi3ForCausalLM` 缺 `.generate`（5.20 把 `GenerationMixin` 移出 `PreTrainedModel`）→ 缺 `generation_config` → `DynamicCache` 缺 `seen_tokens`/`get_usable_length`（5.20 缓存 API 重构）。逐处 patch 会陷入 whack-a-mole。**最终方案**：本地下载 `transformers==4.51.3`（纯 py wheel）→ SCP 解压到 `/home/<SERVER_USER>/old_tf` → 跑生成时 `PYTHONPATH=/home/<SERVER_USER>/old_tf` 前置遮蔽，现网 5.20.2 完全不动（保 EXP1/EXP2a 不受影响）。4.51.3 对 `tokenizers<0.22`/`huggingface-hub<1.0` 的保守上限检查在影子副本里改为 try/except 跳过（本机依赖均更新，仅上限触发、API 实质兼容）。配套 `server_cross_model_internvl.py` 顶部 `import dyncache_shim`（`dyncache_shim.py` 在 5.20 下补 `seen_tokens`/`get_usable_length`，在 4.51 下为空操作）。此前在缓存自定义代码里为 5.20 加的 `GenerationMixin` 继承、`generation_config` 赋值、`.seen_tokens→.get_seq_length` 三处补丁在 4.51 下均惰性无害。**648/648 成功，0 error。**

### 待办

- [x] EXP2b InternVL2-4B ✅ 已完成（648/648 生成 + 同源 DeepSeek 判定，0 error；见上方六模型横向对比）。
- [ ] LLaVA/GPT-4o/Gemini/Claude/InternVL 五组逐轮曲线与 EXP1 的逐轮对比图（论文 Figure）。
- [x] EXP3 注意力机制已完成（2763+668+2763 前向，0 失败）。

---

## EXP3 — `attention_mechanism` ✅ 已跑完（2026-10-27）

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-VL-7B-Instruct，强制 `attn_implementation="eager"` |
| 主实验前向 | **2763 次**（= 1407 自愈 H→F + 1356 持续幻觉 H→H，严格历史分层配对） |
| 忠实参考臂 | **668 次**（F→F；仅作参照水平，**不参与因果对比**） |
| 机制分解臂 | **2763 次**（同一批样本，注意力按 key 组分解） |
| 服务器 | 4090 **cuda:1**（GPU0 被他人占用）；`CUDA_VISIBLE_DEVICES=1` + `--device cuda:0`（逻辑编号会重排，写 cuda:1 会 invalid device ordinal） |
| 耗时 | 主 11.3 min / 参考 2.9 min / 分解 11.5 min，失败 0 |
| 产物 | `attn_perlayer_hf.jsonl`、`attn_perlayer_ff.jsonl`、`attn_decompose_hf.jsonl` |

### 实验设计（STRATIFIED MATCHING，用户 2026-10-27 拍板）

对比的是同一轮 r 上两种结局，且**前 r−1 轮的幻觉历史逐位相同**：

- **self-heal（自愈）**：r−1 轮幻觉 → r 轮忠实（H→F）
- **persist（持续幻觉）**：r−1 轮幻觉 → r 轮仍幻觉（H→H）
- **分层键 = 前 r−1 轮 H/F 的精确 bitmask**（不只是"幻觉次数"）
- 仅保留**同时含两种结局**的 stratum（80 个，覆盖 1356 对）
- 跨层按 `w = min(n_heal, n_persist)` 加权

测量：复刻 EXP1 的原始 prompt（system + 原数据集前缀对话 + 前 r−1 轮真实 Q/A + 第 r 轮问题），
teacher-force 模型自己的第 r 轮回答，取**回答 token 作为 query**、**`<|image_pad|>` token 作为 key**，
算每层每头的 **Visual Attention Ratio (VAR)**。注意力图用 hook 逐层消费后立即释放（否则 28 层全保留会爆显存）。

选回答 token 而非全序列，是为了避免"对话越长、图像占比被机械稀释"这一混淆（旧脚本 `server_attn_per_round.py` 正是此问题）。

### 🔴 核心结果：原假设**不成立**（全局视觉注意力没有反弹）

| 指标（深层 20–28） | self-heal | persist | Δ | 95% CI | 置换检验 p |
|---|---|---|---|---|---|
| VAR（分层后） | 0.02548 | 0.02482 | **+0.00066** | [−0.00031, +0.00164] | **0.116** |

- **5 种分层方案全部 Δ≈0 且 CI 含 0**：(r,bitmask) +0.00066 / (图×setting×r,bitmask) −0.00036 / (图×setting×r,次数) −0.00005 / (图×setting×r) +0.00030 / (r,次数) +0.00067。
- 方差分解显示 **81% 的 VAR 方差来自"图像×setting×轮次"身份**，因此又把图像身份也纳入分层——即便如此仍为 0，**排除"功效不足"解释**。
- 逐轮（r=3 / r=7，自愈最集中的两轮）同样无效应。
- 机制分解（图像 / 自己历史回答 / 历史用户轮 / 当前问题 / 自注意力）效应量均 1–4%，CI 多数含 0 且**符号在 query 集之间翻转** → 噪声。
- **结论：预期中"自愈轮深层视觉注意力飙升"现象不存在。** 雪球/自愈的动态**不能**表述为"整体视觉注意力回涨"。

### 🟢 但存在一个**局部、可复现**的注意力签名

层平均看不到，是因为效应集中在少数 (层, 头) 单元且**正负相消**：

- 784 个 (层, 头) 单元中，**4–6 个**超过置换检验 95% max-|Δ| 阈值（family-wise p < 0.005）。
- 稳定单元：**L22H2 Δ=−0.0115**（自愈时图像注意力**更低**）、**L20H24 Δ=+0.0102**、**L15H24 Δ=+0.0087**、L22H26 +0.0079、L23H14 +0.0077 —— 集中在第 15/20/22/23 层。
- **分半复现**：两半独立估计的 per-cell Δ 相关 **r=0.79**（0.61–0.92），top-50 符号一致 **90%**，top-K 重叠远超随机（K=50 时 31 vs 随机 3.2）。
- **非长度伪影**：对 `n_answer_tokens / seq_len / n_image_tokens` 残差化后，与原始 Δ 相关 **0.970**，最大单元未缩小（0.0115→0.0133）。
- 注意：自愈答案比持续幻觉短约 3.9 token、序列短 28.7 token，这是层内残留不平衡，已在上一条中排除。

### 可写进论文的口径

> Under exact history matching, self-healing is **not** accompanied by a global
> rebound of visual attention (Δ deep-layer VAR = +0.0007, 95% CI [−0.0003, +0.0016],
> permutation p = 0.12; null replicates under five stratification schemes including
> image-identity matching). The effect is instead **sparse and sign-heterogeneous**:
> a handful of mid-to-deep attention heads shift reliably (family-wise permutation
> p < 0.005, split-half r = 0.79), and they do not all move in the same direction.

**禁止**：不得因"预期是飙升"而挑选某一两个头包装成全局结论；不得把未分层的 delta 写进公式。

### 产物与脚本

| 文件 | 说明 |
|---|---|
| `build_attn_selection_hf.py` | 从 EXP1 裁决构建 1443 条自愈事件的严格 bitmask 配对选择集（自愈 1407 + 持续 1356） |
| `build_attn_selection_ff.py` | 忠实参考臂（668 条），**仅参照、非因果对比** |
| `server_attn_perlayer.py` | 服务器：eager 注意力 + hook 逐层消费，输出 28 层 VAR（含 per-head） |
| `server_attn_decompose.py` | 服务器：注意力按 img/hist_asst/hist_user/cur_q/self 分解 |
| `analyze_attn_perlayer.py` / `make_figC.py` | 分层分析 + Figure C（逐层曲线 / Δ / r=3,r=7） |
| `analyze_decompose.py` / `make_figD_heads.py` | 机制分解 + Figure D（28×28 头签名热图） |
| `diag_permutation.py` / `diag_splithalf.py` / `diag_residualize.py` | 置换检验、分半复现、协变量残差化三重稳健性 |
| `sshx.py` | paramiko 远程助手（本环境可直接连服务器跑任务，无需手动 SSH） |

Figures：`figs/figC_layer_var.png`（三臂：persist / self-heal / faithful-ref + Δ 面板）、
`figs/figC_rounds_r3_r7.png`、`figs/figD_head_signature.png`、已废弃的 `figs/figATT*_selftest.png`（合成数据，勿引用）。

---

## EXP4 — `mega_qwen_1000` ✅ 已跑完（2026-10-29）

> **隔离铁律**：EXP4 是 EXP1（50 图 / 9000 单元）协议放大到 **1000 图**的稳定估计，**数字绝不进 EXP1 主表、绝不合并/平均**；引用一律标 EXP4。两者仅方向/机制可比。

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-VL-7B-Instruct（与 EXP1 同模型，本地 2×4090 transformers batch 推理，非 vLLM——服务器无外网） |
| 数据 | `MM-Hallu/MMHalSnowball` train 过滤 `halluconv_formatting` 单 setting，1000 唯一样本；注入方案与 50 条协议同构（8 类 × lead/escalate、每轮传图 `every_round`、T=0.7、top_p=0.9、K=3、seed=42） |
| conditions | baseline（第 1 轮，无注入）+ injection（第 1–9 轮，注入 + 第 9 轮 dataset_final） |
| 裁判单元 | **30000**（baseline 3000 + injection 27000）；有效 **29999**，1 条模型安全拒答（返回非 JSON）已排除为 `undetermined` |
| 裁判 | 同源 **DeepSeek-V4.1-Flash-1017-Event**（scnet.cn，无记忆三字段 prompt），与 EXP1/EXP2 完全一致；0 error |
| 产物 | `fused_1000_hallu_a/b.jsonl`、`judge_input_exp4.jsonl`、`judge_output_exp4.jsonl`、`fig_mega_1000.pdf`、`EXP4_MEGA_ANALYSIS.md` |

### 🔴 核心结果

| 指标 | 值 |
|---|---|
| **injection 整体幻觉率** | **0.736**（CI 见逐轮表） |
| **baseline 第 1 轮幻觉率** | **0.182**（CI [0.164, 0.200]） |
| **主效应 delta（injection − baseline）** | **+0.555**（CI 排除 0 ⇒ 显著） |
| 逐轮 r1–r9 | 0.930 / 0.982 / 0.594 / 0.718 / 0.543 / 0.863 / 0.686 / 0.737 / 0.573（**非单调雪球震荡**，与 EXP1 同构） |
| 状态转移 H→H / H→F / F→H / F→F | 13773 / 4386 / 3314 / 2526 |
| **自愈率 H→F/(H→F+H→H)** | **0.242** |
| 曾幻觉轨迹（ever_H） | 3000/3000（1.000） |
| 其中第 9 轮恢复到忠实 | 1282/3000（0.427） |

- **雪球效应确认**：注入条件下幻觉率远高于 baseline 控制，且随轮次震荡累积（sycophantic agreement with injected false premises）。
- **自愈弱**：仅 0.242 的幻觉态在下一轮翻回忠实；多数曾幻觉轨迹贯穿第 9 轮仍是幻觉。
- **与 EXP1 方向/机制一致**：EXP1 逐轮曲线 r1..r9 = .923/.979/.570/.639/.500/.826/.659/.631/.479，EXP4 放大到 1000 图后曲线形态同构；自愈率 EXP1=0.280、EXP4=0.242，量级一致。
- **baseline 0.182 高于 EXP1 的 0.047–0.133**：因 EXP4 仅用 `halluconv_formatting` 单 setting（EXP1 为 6 setting 混合），样本分布不同所致；**这是 EXP4 自身读数，不回写 EXP1 主表**。

### 产物与脚本

| 文件 | 说明 |
|---|---|
| `construct_inject_v3_1000.py` | 复用 50 条 builder，输出 `constructed_1000_halluconv.jsonl`（1000 样本，halluconv 两 cell） |
| `run_qwen_fused_batch.py`（服务器） | 2 分片（cuda:0 样本0–499 / cuda:1 样本500–999），`every_round` 每轮传图防丢图伪影 |
| `build_judge_input_exp4.py` | 展开为每 (样本,条件,轨迹,轮次) 一条裁判单元（30000） |
| `run_glm_judge.py` | 复用 EXP1/EXP2 同源 scnet.cn 裁判（断点续跑 + 12 次重试） |
| `analyze_exp4.py` | 逐轮率 + 95% cluster-bootstrap CI（按 sample 簇重采样 2000 次）+ H/F 转移矩阵 + 自愈率 → `fig_mega_1000.pdf` + `EXP4_MEGA_ANALYSIS.md` |

---

## 🔔 写论文前的待办提醒（未完成，动笔时必须处理）

| # | 事项 | 说明 |
|---|---|---|
| 1 | **退化率正则口径** | 两个口径：**0.5%**（早期宽松正则，会把 context_pollution 的礼貌话术误判为"看不到图"）vs **0.08%**（`audit_main_experiment.py` 的 `STRICT` 正则，7/9000）。**写作统一用 0.08% 并在 Method/footnote 给出正则定义**，否则与早期草稿的 0.5% 对不上。 |
| 2 | **数字归属标注** | 每张表/图注明来自 EXP1 还是 EXP2/EXP3；禁止跨轮合并或平均。 |
| 3 | **注意力结论（EXP3 已出结果）** | 全局 VAR 无反弹（Δ=+0.0007, p=0.12），只有 4–6 个 (层,头) 单元的稀疏签名。论文口径见 EXP3 段的『可写进论文的口径』；**不得**写成『自愈 = 视觉注意力回涨』。 |
| 4 | **裁判一致性说明** | EXP1 全程单一 DeepSeek 裁判（9000 条、0 error），需在 Method 中说明；`judge_output_mixed_backup.jsonl` 是历史混合裁判留底，仅作 provenance，不用作结果。 |

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-26 | EXP1 审计通过并冻结（`EXP1_main_qwen_9000/`），建立本账本与命名规则 |
| 2026-11-07 | **总冻结**：EXP2/EXP3/EXP4 各建独立冻结目录（复制不移动），EXP1 原地校验不改动；99 个文件 MD5 基线入 `frozen_registry.json`；四轮定义/权威数字/口径红线/代码溯源统一收进 **`FROZEN_MANIFEST.md`**（唯一入口）；一键校验 `verify_frozen.py` 全项通过（7 组判定 0 error）。本账本继续作为版本总账，引用数字前先跑 `python verify_frozen.py`。 |
