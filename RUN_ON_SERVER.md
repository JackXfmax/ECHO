# 第二步(b) & 第三步：服务器端执行手册（在 <REDACTED-SERVER-IP> / `train` 环境跑）

本地沙箱**没有 GPU、模型权重、图片**，所以下面两个脚本只能在 4090 服务器执行。
它们都从本目录已有的 `fused_50_out_v3.jsonl` 抽取攻击协议（脚本化的注入问题，
与具体模型无关），只需要补上**图片目录**和**模型权重路径**。

## 0. 前置（一次性）
```bash
ssh <SERVER_USER>@<REDACTED-SERVER-IP>          # 密码 <REDACTED-SERVER-PASSWORD>
conda activate train
cd <本目录在服务器上的同步路径>
# 确认图片目录：每张图文件名 stem == sample_id（8位数字），扩展名 .jpg/.jpeg/.png/.webp
# 若布局不同，改下面命令里的 --image_dir
```

## 1. 第二步(b)：注意力提取（机制的决定性检验）
目标：对比「幻觉轮」与「自愈轮」的 attention 在图像 token vs 文本 token 上的质量。

```bash
python server_extract_attention.py \
    --fused fused_50_out_v3.jsonl \
    --sample self_recovery_sample20.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --image_dir /path/to/snowball_images \
    --out attn_recovery.json
```
- 需要 `~16GB` 显存（`output_attentions=True` + eager 实现）。
- 输出每个样本的 `src_attn_image / rec_attn_image / delta`；脚本末尾打印
  **mean delta（rec−src）attention-on-image** 与配对 t 值。
- **判读**：若 mean delta 显著 > 0，说明自愈轮 attention 确实「跳回图像」，
  立住「视觉-文本信噪比动态方程」。

## 2. 第三步：跨模型迷你验证
目标：把同一套注入协议打到 LLaVA-1.5-7B（或 InternVL2-2B），证明震荡/自愈是架构共性。

```bash
python server_cross_model_probe.py \
    --fused fused_50_out_v3.jsonl \
    --model llava-hf/llava-1.5-7b-hf \
    --image_dir /path/to/snowball_images \
    --n_images 10 --out cross_model_judge_input.jsonl
```
- 产出 `cross_model_judge_input.jsonl`（judge-ready）。
- 判定 + 出曲线（**可本地用 API 跑**，不占服务器）：
```bash
python run_glm_judge.py --key <KEY> --model DeepSeek-V4.1-Flash-1017-Event \
    --inp cross_model_judge_input.jsonl --out cross_model_judge_output.jsonl
python analyze_results_fast.py --jin cross_model_judge_input.jsonl \
    --jout cross_model_judge_output.jsonl
```
- 看输出的 `round 1..9` 行：若 LLaVA 也出现非单调震荡（哪怕数值不同），
  即可在论文写 *"a universal phenomenon across LVLM architectures"*。

## 3. 关键假设已被数据推翻（先读这个）
`self_recovery_analysis.md` 显示：自愈轮问题**并不更短**（92 vs 91 字符），
视觉引导词率**更低**（10% vs 21%）。即自愈是模型**自发**纠错，非用户引导触发。
因此第二步(b)的注意力证据是机制的唯一决定性检验。
