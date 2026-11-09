# MMHalSnowball 实验总冻结清单 🔒

> **冻结日期**：2026-11-07（精确时间见 `frozen_registry.json` 的 `frozen_at`）　|　**状态**：✅ 全部完成、审计通过、已冻结
> **本文件是「有哪些实验、各自数字是多少、产物在哪」的唯一入口。**
> 引用任何数字前：`python verify_frozen.py` → 必须 `ALL CHECKS PASSED`。
> 明细文件清单见 `FROZEN_FILES.md`；机器可读 MD5 基线见 `frozen_registry.json`。

---

## 0. 一句话规则

**四轮实验的数字互相隔离，绝不合并、绝不平均，引用一律标 EXP 编号。**
任何产物被改动 → `verify_frozen.py` 立刻报 `MD5 CHANGED`。

| 命令 | 作用 |
|---|---|
| `python verify_frozen.py` | 校验 `frozen_registry.json` 登记的全部文件 MD5/大小/行数 + **现网文件漂移检测** + 7 组判定 0 error；退出码 0 = 安全 |
| `python freeze_all_experiments.py` | 重新冻结（**仅在新增实验/主动重冻时使用**，会覆盖基线） |

已实测三种情况：改现网文件 → `LIVE FILE DRIFT` 报警；改冻结件 → `MD5 CHANGED` 报警；恢复 → `ALL CHECKS PASSED`。

---

## 1. 实验索引（四轮，全部完成）

| 编号 | codename | 被测模型 | 规模 | 裁判 | 冻结位置 |
|---|---|---|---|---|---|
| **EXP1** | `main_qwen_9000` | Qwen2.5-VL-7B-Instruct | 50 图 × 6 setting × 2 条件 × 3 轨迹 = **9000 单元** | DeepSeek-V4.1-Flash-1017-Event，0 error | `EXP1_main_qwen_9000/`（2026-10-26 已冻结，本次只校验不改动） |
| **EXP2** | `crossmodel` | LLaVA-1.5-7B / InternVL2-4B / GPT-4o / Gemini-2.5-Flash / Claude-Sonnet-4.6 | 各 12 图 × 6 setting × 1 轨迹 × 9 轮 = **648 条**（injection-only） | 同源 DeepSeek，各 0 error | `EXP2_crossmodel/` |
| **EXP3** | `attention_mechanism` | Qwen2.5-VL-7B-Instruct（eager attention） | **2763** 主前向 + 668 忠实参考臂 + 2763 机制分解 | — | `EXP3_attention_mechanism/` |
| **EXP4** | `mega_qwen_1000` | Qwen2.5-VL-7B-Instruct | 1000 图 × `halluconv_formatting` 单 setting = **30000 单元**（有效 29999） | 同源 DeepSeek，0 error | `EXP4_mega_qwen_1000/` |

> 裁判统一：`DeepSeek-V4.1-Flash-1017-Event`（scnet.cn）、无记忆三字段 prompt
> （`current_question` / `ground_truth` / `model_output`），保证跨实验可比。

---

## 2. 权威数字（引用以此为准，勿重算）

### EXP1 — 主结果（唯一主表来源）

| 指标 | 值 |
|---|---|
| baseline / injection 幻觉率 | 0.047–0.133 / 0.650–0.711 |
| injection − baseline delta | **+0.530 – +0.647**（CI 均不含 0） |
| 最终轮 r1 vs r9 配对 delta | +0.273 – +0.447 |
| 分类型：entity_missing / background_fabrication / context_pollution / attribute_reversal | **0.951** / 0.663 / 0.645 / 0.604 |
| **自愈率 H→F** | **28.0%**（HH=3711 HF=1443 FH=1043 FF=1003） |
| 逐轮 r1..r9 | 0.923 / 0.979 / 0.570 / 0.639 / 0.500 / 0.826 / 0.659 / 0.631 / 0.479 |
| 退化率 | **0.08%**（严格正则，7/9000） |

### EXP2 — 六模型横向（injection-only，同源裁判）

| 模型 | n | OVERALL | 自愈 H→F |
|---|---|---:|---:|
| Qwen2.5-VL-7B (EXP1) | 8100 | .69 | .28 |
| LLaVA-1.5-7B (EXP2a) | 648 | .95 | .06 |
| InternVL2-4B (EXP2b) | 648 | .82 | .15 |
| GPT-4o (EXP2c) | 648 | .55 | .37 |
| Gemini-2.5-Flash (EXP2d) | 648 | .63 | .33 |
| Claude-Sonnet-4.6 (EXP2e) | 648 | .44 | .42 |

> 横向只比**定性模式**，不比绝对值（EXP1 是 8100 单元/50 图，其余是 648 子集/12 图）。

### EXP3 — 注意力机制

- 严格分层匹配后，深层 20–28 **ΔVAR = +0.00066，95% CI [−0.00031, +0.00164]，p = 0.116**；5 种分层方案（含图像身份）全部 Δ≈0。
- 只有 **4–6 个 (层,头) 单元**有真实签名：L22H2 −0.0115、L20H24 +0.0102、L15H24 +0.0087、L22H26 +0.0079、L23H14 +0.0077；分半复现 **r = 0.79**，残差化后与原始 Δ 相关 0.970。

### EXP4 — 1000 图放大版

| 指标 | 值 |
|---|---|
| injection 整体幻觉率 | **0.736** |
| 逐轮 r1..r9 | .930 / .982 / .594 / .718 / .543 / .863 / .686 / .737 / .573 |
| baseline 第 1 轮 | **0.182**（CI [0.164, 0.200]） |
| 主效应 delta | **+0.555**（CI 排除 0） |
| 状态转移 H→H / H→F / F→H / F→F | 13773 / 4386 / 3314 / 2526 |
| **自愈率 H→F** | **0.242** |
| ever_H | 3000/3000；第 9 轮恢复忠实 1282/3000（0.427） |

---

## 3. 🔴 口径红线（写论文逐条卡）

1. **数字隔离**：EXP1 的 9000 是唯一主表来源；EXP2/EXP3/EXP4 是小样本验证或放大复测，**不得混入主表、不得合并/平均**。引用格式：`EXP1 (Qwen2.5-VL-7B, 50 images, 9000 units)`。
2. **退化率统一 0.08%**（`audit_main_experiment.py` 的 STRICT 正则，7/9000）。早期 0.5% 是宽松正则、会把礼貌话术误判为"看不到图"，两者不矛盾但**不得混用**；Method/footnote 给出正则定义。
3. **不得写"前沿模型抵抗"**：三大前沿全部中招（.44–.63）。已被推翻的旧口径（"GPT-4o 全程抵抗"仅单条冒烟；Gemini 0.000 是 judge DNS artifact）**禁止沿用**。正确 claim：雪球跨开闭源普遍，但开源更严重、自愈更弱。
4. **不得写"自愈 = 视觉注意力回涨"**（EXP3 结论与预期相反）。只能写稀疏、符号异质的头部重分配。
5. **两处 648 来源不同**：EXP2 跨模型 648 = 12 图 × 6 setting × 1 轨迹 × 9 轮；EXP3 注意力 648 是另一批前向。**勿混为一谈**。
6. **逐轮曲线必须用 injection-only**；把含 baseline 的全量喂进逐轮统计会把 r1 稀释（0.518 ≠ 0.923）。
7. **逐轮指标措辞**：写 "the j-th generated response under the accumulating dialogue history"，**不要**写 "the answer to the j-th question"；r1 不赋予"回答实体缺失诱导问"、r9 不赋予"撤除注入净测试"的解释（生成器 `run_sample` 在生成后才把该轮 Q/A 追加进历史，逐轮标签滞后一拍；该行为对所有组/模型系统一致，不影响主效应、跨模型对比与定性结论）。
8. **裁判一致性**：EXP1 全程单一 DeepSeek 裁判（9000 条 0 error），Method 中说明；`judge_output_mixed_backup.jsonl` 仅作 provenance，**不作结果来源**。

---

## 4. 代码溯源（已核实，勿再置疑）

- **实际生成引擎 = `run_qwen_fused_batch.py`**。证据链：
  1. `EXP1_main_qwen_9000/MANIFEST.md` 复现顺序第 2 步明确写 `run_qwen_fused_batch.py → fused_50_out_v3.jsonl`；
  2. MD5 三方一致：冻结件 code 副本 = 工作区副本 = MANIFEST 记录值 `373e260ee9ed47e6df2097c537b23c39`；产物 `fused_50_out_v3.jsonl` = `ebf5e0ed376eb3aba6386a726675a191`；
  3. 服务器 7 个分片脚本 `run_v3_t0.sh`…`t42.sh` 全部调用 `python run_qwen_fused_batch.py --conditions baseline,injection`；
  4. 服务器现存副本（18308 B, 10/24 18:37）与冻结件**去 CR 后逐字节相同**，仅 LF vs CRLF 行尾差异（md5 不同只因此），已存为 `_server_run_qwen_fused_batch.py`。
- **注入协议**：`construct_inject_v3_server.py` 的 `build_injections()` —— 8 轮固定顺序
  r1–r2 entity_missing (lead/escalate) → r3–r4 attribute_reversal → r5–r6 background_fabrication → r7–r8 context_pollution → r9 `dataset_final`（原数据集最后一问，GT=truth_token）；
  `prefix = original_conversations[setting][:-1]` + system prompt；baseline 1 轮、injection 9 轮；每轮传图、K=3、T=0.7/top_p=0.9/max_new_tokens=256、seed=42+k·977+j。
- **历史版本（未产出冻结件，勿引用）**：`run_qwen_fused.py`（非 batch 单例版）、`run_qwen_test.py`（冒烟）、`merge_pull.py`/`merge_v2.py`（v1/v2 合并）、`fused_50_out.jsonl`/`fused_50_out_v2.jsonl`。
- ⚠️ `export_judge_input.py` 文件头 "baseline cells reused from the v1 run" 是 **v2 时代遗留的过时注释**；v3 同批生成（冻结产物中 baseline r1 与 injection r1 输出 150/150 完全相同可证）。
- ⚠️ 空文件 `cross_model_judge_input.jsonl`（0 字节，根目录）是残留 stub，**勿引用**；
  `figs/figATT*_selftest.png`、`*_smoke.png` 为合成/冒烟图，**勿引用**。

---

## 5. 产物定位

- EXP1：`EXP1_main_qwen_9000/{code,data,results}/` + `MANIFEST.md` + `AUDIT_REPORT.txt`
- EXP2/3/4：各自目录下 `data/`（判定/生成/注意力原始）、`results/`（分析 MD + 图）、`code/`（该轮脚本）；EXP2 另有 `assets/`（12 张 `cross_images`）
- 论文素材：`PAPER_WRITING_KIT.md`（EXP1–4 全过程与数据总览）、`EXPERIMENT_LOG.md`（版本总账）
- 完整逐文件 MD5 表：`FROZEN_FILES.md`；机器基线：`frozen_registry.json`

---

## 6. 冻结纪律

1. 冻结用**复制**不用移动——根目录现网文件保持原样，可继续用于后续工作。
2. 冻结件**不再改动**；若确需新增实验，另建 `EXP<n>_<codename>/`，先跑审计再重冻，并更新本清单与 `EXPERIMENT_LOG.md`。
3. 每次引用数字前跑 `verify_frozen.py`；出现 `MD5 CHANGED` 必须查清原因再引用。
