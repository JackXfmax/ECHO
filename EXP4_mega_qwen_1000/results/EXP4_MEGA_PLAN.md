# EXP4 — 1000 图 Qwen2.5-VL-7B 多轮幻觉注入：执行状态与取舍留痕

> 用户硬约束：**构建必须与之前 50 条一致**。本文记录每一步的实际做法、与原 spec 的偏离及理由。
> 生成时间 2026-10-29 00:2x（沙箱时区）

---

## Step 1 数据准备 ✅ 完成

### 做法：复用构建器，不重写
- 构建器：`construct_inject_v3.py`（服务器上产出 `constructed_50.jsonl` 的**同一支**）
- 改动：仅插入 1 行 `sys.exit(0)` 跳过 HTML 预览（逐样本扫 parquet 取图，N=1000 时代价不可接受）
- 验证：`diff` 确认**仅此一行**不同 → jsonl 构建逻辑与 50 条逐字节相同
- 命令：`INJECT_N=1000 python construct_inject_v3_1000.py`

### 产物与校验
| 项 | 结果 |
|---|---|
| 文件 | 服务器 `constructed_1000.jsonl`；本地 `test_set_1000.json` / `.jsonl` |
| 条数 | 1000（唯一 sample_id 1000） |
| top-level 字段名 | 与 `constructed_50` **完全相同** |
| injection 字段名 | **完全相同**（round/category_key/category_label/current_question/ground_truth/mode/premise_false/truth） |
| 每样本注入轮数 | 8（4 类别 × lead/escalate） |
| 空 `ground_truth` / `current_question` | 0 / 0 |
| 分层 | existence / imagination / attribute / relation 各 250（同 50 条策略） |
| 与 `constructed_50` 重叠 | **50**（旧 50 条为子集 → 可直接对照，无泄漏） |

### 数据集事实
`MM-Hallu/MMHalSnowball` train = 29838 行，**唯一 sample_id = 4973**，6 个 `conversation_setting` 各覆盖全部 4973 → 抽 1000 充裕。

### 关键口径
构建器产出是 **8 轮**；EXP1 用的 **9 轮** 由推理引擎追加原数据集 `dataset_final` 轮（第 9 轮）而来。

---

## Step 2 推理 ✅ 已启动（引擎替换 + 协议校正）

### 阻塞：vLLM 无法按原样执行（已核实）
| 证据 | 结论 |
|---|---|
| 服务器 `import vllm` → ModuleNotFoundError | 未安装 |
| 服务器无对外网络（校园 ePortal，本会话已证实） | pip 装不了 |
| 根盘 100% 满，仅 **6.4G** 可用 | vLLM 数 GB 装不下 |
| 本机 RTX 4060 Laptop **8GB**，torch **2.14.1+cpu** | 跑不了 7B |

✅ 替代：项目既有引擎 `run_qwen_fused_batch.py`（就是原来跑 50 条那支），本地权重 `/home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct` 已就位；OOM 时自带 `split batch 6 -> 3+3` 自动降级（满足 spec 的 OOM 要求）。

### 协议校正（原 spec 三处会毁结论，按"与 50 条一致"校正）
| 原 spec | 实际采用 | 理由 |
|---|---|---|
| 图**只在第 1 轮**传 | **`vision_mode = every_round`** | `fused_50_out_v3.jsonl` 元数据明写 `every_round`；USER 记忆 2026-10-24 血案：不每轮传图 → 第 2 轮起丢图（11%→57%），与自变量混淆、结论不可归因，且与 EXP1 不可比 |
| `temperature = 0.1` 配 K=3 | **`temperature 0.7 / top_p 0.9`** | 50 条用 0.7 正是让 3 个 seed 产生真实差异；0.1 下三次重复近乎相同，"capture variance" 为空 |
| vLLM | `run_qwen_fused_batch.py` | 唯一可执行路径（见上） |

> 另：**保留 baseline 对照**（每样本仅多 1 轮 × 3 traj，成本约 +10%），以保住 `injection − baseline` 这个 EXP1 的主效应 delta。

### 范围裁剪
引擎无 `--settings` 参数 → 改为**过滤输入**：把每条 `original_conversations` 只保留 `halluconv_formatting`（注入内容不动），得 `constructed_1000_halluconv.jsonl` → 每样本 2 cell（baseline + injection）。

### 冒烟（双验证，均通过）
| 版本 | 耗时 | 产出校验 |
|---|---|---|
| 全 6 setting | 2 样本 **12.4 min** | 12 cell、injection 1–9 轮、180 条记录、空输出 0、top keys 与 `fused_50_out_v3` **逐字段一致**、类别映射与 EXP1 完全一致 |
| halluconv-only | 2 样本 **3.2 min** | 2 cell、baseline 1 轮 / injection 1–9 轮、30 条记录、空输出 0、`SMOKE_OK=True` |

### 全量已启动（detached，两片并行）
```
run_qwen_fused_batch.py --data constructed_1000_halluconv.jsonl \
  --temperature 0.7 --top_p 0.9 --num_samples 3 --seed 42 \
  --conditions baseline,injection --batch_size 6 --max_pixels 320000
```
| 分片 | GPU | 显存 | 首样本 | ETA |
|---|---|---|---|---|
| a：`--start 0 --limit 500` | cuda:0 | 16.7G | 1.7 min | ~836 min（~14h） |
| b：`--start 500 --limit 500` | cuda:1 | 17.0G | 1.1 min | ~531 min（~8.9h） |

---

## Step 3 判定 🔴 阻塞（需用户提供 key）

- spec 要求 `base_url="https://api.deepseek.com"`（官方）→ **环境中无该 key**
- 项目现成的是 scnet.cn 中转：`https://api.scnet.cn/api/llm/v1/chat/completions`，模型 `DeepSeek-V4.1-Flash-1017-Event`（EXP1/EXP2 全程同源裁判）
- 量级：约 **30000 条**待判定（1000 × 30），成本/耗时尚未估，需先定方案

**需要你二选一**：① 给官方 DeepSeek key（走 `api.deepseek.com`）；② 沿用 scnet.cn 同源裁判。

---

## Step 4 统计与作图 ⏸ 待 Step 3

计划产出：
- 逐轮（1–9）幻觉率 + bootstrap 95% CI
- 状态转移矩阵 H→H / H→F / F→H / F→F 与自愈率（H→F）
- `fig_mega_1000.pdf`（折线 + CI 阴影带）
- `EXP4_MEGA_ANALYSIS.md`

---

## 记账约定（勿违反）
本轮为 **EXP4**，产物前缀须独立，**不得**与 EXP1 的 9000 条主结果合并或平均；引用时标注 `EXP4 (Qwen2.5-VL-7B, 1000 images, halluconv_formatting)`。结果出来后再补 `EXPERIMENT_LOG.md` 的 EXP4 条目。
