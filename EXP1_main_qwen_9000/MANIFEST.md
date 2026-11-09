# EXPERIMENT 1 (EXP1) -- MMHalSnowball main study [FROZEN]

> **这是「第一轮实验」的最终冻结件。** 后续任何补充实验（EXP2/EXP3/...）都
> 必须另建独立目录，**严禁**与本目录文件混用、覆盖或交叉引用数字。

- **实验编号**：**EXP1**
- **代号**：`main_qwen_9000`（Qwen2.5-VL-7B 主实验，50 图 × 9000 判定单元）
- **冻结时间**：2026-10-26 23:26:08
- **裁判**：单一 DeepSeek-V4.1-Flash-1017-Event（全程 9000 条，0 error）
- **状态**：✅ 已审计通过、已冻结、不再修改

## 本轮「不包含」什么（防止与后续混淆）

以下均属后续实验，**本目录内没有任何它们的产物**：

| 后续编号 | 内容 | 状态 | 产物前缀 |
|---|---|---|---|
| EXP2 | LLaVA-1.5-7B 跨模型验证（12 图 × 648 条） | 进行中 | `cross_model_` |
| EXP3 | Qwen 同轮次注意力机制检验（72 轨迹 × 648 前向） | 待跑 | `attn_` |

**引用规则**：来自 EXP1 的数字，写作时标注为
*"EXP1 (Qwen2.5-VL-7B, 50 images, 9000 units)"*；
EXP2/EXP3 的数字单独标注，**不得**与 EXP1 主表数字合并、平均或互相代入。

Frozen after `audit_main_experiment.py` passed all checks: 9000/9000 rows,
0 errors, no missing/orphan uids, all 6 settings x 50 images x 3 trajectories,
degradation < 2%, published numbers reproduce (entity_missing 0.951,
self-recovery 28.0%, injection-baseline delta +0.53..+0.65).

| group | file | bytes | md5 |
|---|---|---:|---|
| code | construct_inject_v3_server.py | 26,206 | `03eb1876427b3d05671396758fc850ed` |
| code | run_qwen_fused_batch.py | 18,699 | `373e260ee9ed47e6df2097c537b23c39` |
| code | export_judge_input.py | 2,312 | `447aa527b73ce9fb828485bda3f9a218` |
| code | judge_prompt.py | 6,394 | `4ea7bb320d0eb7ce36429895e9c1f90a` |
| code | run_glm_judge.py | 4,979 | `c5f9c486358781520b431ddc6e5b9e09` |
| code | analyze_results_fast.py | 9,477 | `f73c72c2a15f39a93df4cf778d00be76` |
| code | fig_make.py | 12,425 | `09bccd2672eac0112f38354cf2fa4190` |
| code | audit_main_experiment.py | 6,923 | `188fd30ce3036f9b6522b5363948d208` |
| code | freeze_snapshot.py | 5,803 | `494be86efaf6ad7be018df88816e4a06` |
| data | fused_50_out_v3.jsonl | 8,098,130 | `ebf5e0ed376eb3aba6386a726675a191` |
| data | judge_input.jsonl | 9,482,947 | `8eaa8285f6bbbe5cbefb9c58efbb2f34` |
| data | judge_output.jsonl | 4,701,163 | `e0ddba3903605f586cfaac467bb6eaae` |
| data | judge_output_mixed_backup.jsonl | 4,943,196 | `e8a9e2a8e5247dd3ceac3446f2976609` |
| results | result_judge_analysis.md | 3,721 | `a4d667d4a464d920fff611cbcbad0a8a` |
| results | self_recovery_analysis.md | 4,157 | `d71a477f1d72b599dedf0c2bbd86cbcf` |
| results | self_recovery_sample20.json | 7,201 | `9859b6891c43581d2f8c70c6a1068205` |
| results | figs/fig1_round_curve.png | 326,311 | `810f2806fe0812a6cf039e9658ac7d0d` |
| results | figs/fig2_category_bar.png | 133,020 | `db4c35468e4955b399e7ab5865ea9d83` |
| results | figs/fig3_recovery_sankey.png | 209,363 | `09ae8a38f30de12bae1eff3be2479094` |

## Audit trail

`AUDIT_REPORT.txt` (77 lines) is the verbatim output of `audit_main_experiment.py`
run against these exact files. Final line: `ALL CHECKS PASSED -- safe to freeze`.

Headline numbers reproduced inside that report:

- injection vs baseline delta, per setting: **+0.530 .. +0.647**
- attack-category hallucination rate: entity_missing **0.951**,
  background_fabrication 0.663, context_pollution 0.645, attribute_reversal 0.604
- self-recovery transition HH=3711 HF=1443 FH=1043 FF=1003 -> **H->F = 28.0%**
- strict-pattern "cannot see the image" degradation: 7/9000 = **0.08%**
- design balance: every (setting, condition, round) cell = 150 rows;
  all 6 settings cover all 50 images (explicitly rules out the dedup bug
  found later in `server_cross_model_probe.py`)

## How to verify nothing changed later

Re-compute MD5 of the live file and compare with the table above, e.g.:

```bash
md5sum EXP1_main_qwen_9000/data/*.jsonl
```

If any hash differs from this table, the live file has been modified since the
freeze and the published numbers must be re-audited before use.

## Reproduction order (main study)

1. `construct_inject_v3_server.py` -- build the 6x2 x 3-trajectory protocol
2. `run_qwen_fused_batch.py` -- Qwen2.5-VL-7B generation, image EVERY round -> `fused_50_out_v3.jsonl`
3. `export_judge_input.py` -- -> `judge_input.jsonl` (9000)
4. `run_glm_judge.py --model DeepSeek-V4.1-Flash-1017-Event --max_tokens 4096` -> `judge_output.jsonl`
5. `analyze_results_fast.py` -> `result_judge_analysis.md`
6. `fig_make.py` -> `figs/*.png`

> **注**：`freeze_snapshot.py` 是**工具脚本**（非实验产物）。EXP1 冻结后为支持
> EXP2/EXP3 而通用化（新增 `--tag` / `--spec` 参数），故其 MD5 于 2026-10-26 23:33 更新；
> **实验数据与结果文件的 MD5 全部未变**，本轮结论不受影响。
