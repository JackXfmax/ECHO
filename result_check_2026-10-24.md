# 融合实验结果检查报告（2026-10-24）

被测模型：Qwen2.5-VL-7B-Instruct（服务器 <REDACTED-SERVER-IP>，双 4090）
数据：`constructed_50.jsonl`（50 样本）｜采样 T=0.7, top_p=0.9, K=3 独立轨迹

## 1. 第一批全量结果：结构完整，但存在致命混淆

已跑完并拉回：`fused_50_out.jsonl`（50 样本 × 12 cell × 3 轨迹 = **9000 次问答**）

| 检查项 | 结果 |
|---|---|
| 样本数 / cell 数 | 50 / 每样本 12（6 setting × 2 条件），无缺失 |
| 轨迹数 | 每 cell 3 条，共 1800 条 |
| 问答总数 | 9000（baseline 900 + injection 8100） |
| 空回答 | 0 |
| 截断率（无终止标点） | 6.5%（max_new_tokens=256） |
| 前置语境 | 已进 history，不单独产生被测回答（符合"1 轮 = 1 次问答"口径） |

轮次设计落地正确：injection = entity_missing×2 → attribute_reversal×2 →
background_fabrication×2 → context_pollution×2 → 第 9 轮数据集核心题。

## 2. 致命问题：注入条件第 2 轮起图像特征未进入模型

**根因**：每轮都重新渲染 chat template，第 2 轮起文本仍含 `<|image_pad|>`，
但 `processor(images=None)` 返回 `pixel_values=None` → 图像占位符无视觉特征可替换。
服务器实测：第 1 轮 `pixel_values=(720,1176)`；第 2 轮不传图 → `None`；传图 → 正常。

**后果**：35.9% 的注入回答退化成"请上传图片"，且随轮次飙升；baseline 为 0%（它只有 1 轮，有图）。

| 轮次 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| 退化率 | 0.0% | 11.3% | 32.0% | 41.2% | 46.9% | 41.9% | 47.4% | **56.9%** | 44.7% |

这使 baseline 与 injection 之间混入"能否看图"这一变量，6×2 表的差异无法干净归因。

## 3. 修复与验证

- 改为**每轮都传图**（`--vision_mode every_round`），并用 **batch=6**（6 个 setting 同轮一起解码）
- 速度：6.3 s/次 → 2.2 s/次（约 2.9×）｜显存约 23 GB
- 批量正确性自检（greedy，token 级）：cleanconv_* 100% 一致；factconv 前 44 token 一致后
  因 bf16 噪声提前 EOS，但"放 batch 首位/末位"结果完全一致 → padding 与位置编码正确
- **退化率 35.9% → 7.4%**，且模型在能看图的前提下仍顺从假前提：
  - r3 直接编造 "the car is parked along the side of the road next to the bus"
  - r5/r6 接受虚构的 "large window"，r9 仍在延续该虚构
  → 雪球效应在视觉信息完整时依然存在，结论更有说服力

## 4. 当前状态（2026-10-24 16:43 启动）

- 只重跑 injection（baseline 仅 1 轮且该轮带图，已复用，不重跑）
- 4 个分片 + GPU 守卫调度（GPU1 被他人 SFT 占用，释放后自动接管）
- 服务端 `finish_inj_v2.py` 会在 4 个分片全部完成后自动合并并做结构校验
- 单卡预计约 4.6 h；双卡约 2.5 h

## 5. 待重跑完成后执行

1. `merge_v2.py` → `fused_50_out_v2.jsonl`（baseline + 新 injection）
2. `export_judge_input.py` → 9000 条 Judge 输入（三字段，元数据分离）
3. `run_glm_judge.py`（需智谱 key）→ GLM 判定，含 Rule 6 顺从惩罚
4. `analyze_results.py` → 6×2 幻觉率表、末轮 r1 vs r9 配对比较、逐轮崩溃曲线、
   halluconv vs cleanconv 交互效应、自愈率、分注入类别统计
   （簇 bootstrap 按 sample_id，B=2000；配对比较用同一重采样样本集算差值 CI）

## 6. 粗信号（非判定结果，仅供直觉，等 Judge 才算数）

末轮核心题回答中提及 false token 的比例：baseline 22.2% → injection 46.2%。
该指标为字符串包含，未经过 Judge，不能作为结论。
