"""
run_qwen_test.py — 用 Qwen2.5-VL-7B-Instruct 跑 MMHalSnowball 注入测试集（Test Set）

实验协议（来自改进点 2）：
  1. 每张图 = 一个全新对话：先发 [Image]（只在第 1 轮 user 消息里带图）。
  2. 按序把 8 轮 current_question 发给 Qwen；每发完一轮，把 [Q, A] 追加进对话历史。
  3. 图与图之间零污染（每条样本新建 messages），单图内 8 轮累积，测雪球效应。

科研采样设置（2026-10-23 与用户敲定）：
  - 温度 T=0.7（保留有意义的采样随机性，贴合真实部署，不报单点数字）。
  - 每样本跑 K 条 *独立* 完整轨迹（默认 K=3），每轮在这 K 条轨迹上分别判定后聚合
    （均值 + bootstrap 95% CI），满足"不报单点、给区间"的纪律。
  - 消除视觉前向非确定性：torch.backends.cuda.enable_flash_sdp(False)。
    这是 GPU 浮点归约噪声（不是模型行为），保留它=不可复现；
    消除它对"测试真实度"零损失，唯一随机源只剩温度采样，从而同 seed 完全可复现。
  - seed 仅用于固定数据顺序 / 视觉确定性 / 温度采样 RNG，使"同 seed + 同温度"可复现。

Judge（GLM）在另一脚本 judge_prompt.py 中独立进行，只读三字段。

用法示例（服务器 train 环境）：
  python run_qwen_test.py \
      --model_path /home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct \
      --data /home/<SERVER_USER>/mmhal_inject_v3/constructed_50.jsonl \
      --out /home/<SERVER_USER>/mmhal_inject_v3/qwen_50_out.jsonl \
      --parquet_dir /home/<SERVER_USER>/MMHalSnowball \
      --temperature 0.7 --num_samples 3 --seed 42 \
      --limit 50 --device cuda:0
"""

import argparse, json, glob, os, io, sys

def get_bytes(img):
    if isinstance(img, dict):
        return img.get("bytes")
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    return None


class ImageLoader:
    """Lazily map sample_id -> PIL image from the 30 MMHalSnowball parquet files."""
    def __init__(self, parquet_dir):
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not self.files:
            raise FileNotFoundError(f"no parquet in {parquet_dir}")
        import pyarrow.parquet as pq
        self.idx = {}
        for fp in self.files:
            t = pq.read_table(fp, columns=["sample_id"])
            for i, sid in enumerate(t.column("sample_id").to_pylist()):
                if sid is not None:
                    self.idx.setdefault(sid, (fp, i))
        self._cache = {}

    def __getitem__(self, sid):
        if sid not in self.idx:
            return None
        fp, row = self.idx[sid]
        if fp not in self._cache:
            import pyarrow.parquet as pq
            self._cache[fp] = pq.read_table(fp, columns=["sample_id", "image"])
        tbl = self._cache[fp]
        if tbl.column("sample_id")[row].as_py() != sid:
            sids = tbl.column("sample_id").to_pylist()
            if sid in sids:
                row = sids.index(sid)
            else:
                return None
        raw = tbl.column("image")[row].as_py()
        b = get_bytes(raw)
        if not b:
            return None
        from PIL import Image
        return Image.open(io.BytesIO(b)).convert("RGB")


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        return data
    except json.JSONDecodeError:
        pass
    data = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def run_one_trajectory(model, processor, image_loader, sample, temperature, top_p, max_new_tokens):
    """Run ONE full 8-round trajectory for a sample (fresh conversation, image only in round 1)."""
    import torch
    sid = sample.get("sample_id")
    pil = image_loader[sid]
    if pil is None:
        return None, "image-missing"

    injections = sample.get("injections") or sample.get("test_set", {}).get("rounds") or []
    if not injections:
        return None, "no-injections"

    device = next(model.parameters()).device
    messages = [{"role": "system", "content": "You are a helpful vision-language assistant."}]
    rounds = []
    for it in injections:
        q = it.get("current_question", "")
        if len(messages) == 1:
            user_content = [{"type": "image", "image": pil}, {"type": "text", "text": q}]
        else:
            user_content = [{"type": "text", "text": q}]
        messages.append({"role": "user", "content": user_content})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Avoid qwen_vl_utils (broken torchvision build in train env). Exactly one image
        # (first user turn); later turns must NOT pass images -> unpredictable behaviour.
        images = [pil] if len(messages) == 1 else None
        inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=top_p)
        ans = processor.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        rounds.append({
            "category_key": it.get("category_key"),
            "round": it.get("round"),
            "mode": it.get("mode"),
            "current_question": q,
            "ground_truth": it.get("ground_truth"),
            "premise_false": it.get("premise_false"),
            "model_output": ans,
        })
        # accumulate [Q, A] into history for the next round (within THIS trajectory only)
        messages.append({"role": "assistant", "content": ans})
    return rounds, None


def run_sample_trajectories(model, processor, image_loader, sample,
                            temperature, top_p, max_new_tokens, num_samples, base_seed, sample_idx):
    """K independent trajectories; each gets a deterministic per-(sample,k) seed."""
    import torch
    trajs = []
    for k in range(num_samples):
        seed_k = base_seed + sample_idx * 10000 + k
        torch.manual_seed(seed_k)
        torch.cuda.manual_seed_all(seed_k)
        rounds, err = run_one_trajectory(
            model, processor, image_loader, sample,
            temperature, top_p, max_new_tokens)
        if rounds is None:
            return None, err
        trajs.append({"trajectory_id": k, "seed": seed_k, "rounds": rounds})
    return trajs, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data", required=True, help="constructed.json (list) or .jsonl")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--parquet_dir", default="/home/<SERVER_USER>/MMHalSnowball")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature (kept non-zero for realistic, reproducible-under-seed randomness)")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--num_samples", type=int, default=3,
                    help="K independent trajectories per sample (aggregated later with bootstrap CI)")
    ap.add_argument("--seed", type=int, default=42, help="base seed for reproducibility")
    args = ap.parse_args()

    # ---- reproducibility: fix all seeds + deterministic backends -----------
    import random
    try:
        import numpy as np
    except Exception:
        np = None
    SEED = args.seed
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    if np is not None:
        np.random.seed(SEED)
    import torch
    # KEY: eliminate visual-forward non-determinism (flash-attn + bf16 float reduction noise).
    # This is GPU numeric noise, NOT model behaviour; removing it loses nothing in realism
    # (temperature sampling randomness stays) yet makes runs reproducible under the same seed.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    print(f"[seed] SEED={SEED} T={args.temperature} top_p={args.top_p} K={args.num_samples} "
          f"(flash_sdp=mem_efficient_sdp=False -> visual det)", flush=True)
    # ----------------------------------------------------------------------

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("loading model ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()

    image_loader = ImageLoader(args.parquet_dir)
    data = load_data(args.data)
    print(f"data samples: {len(data)}; image index size: {len(image_loader.idx)}", flush=True)

    subset = data[args.start: (args.start + args.limit) if args.limit else None]
    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(subset):
            sid = sample.get("sample_id")
            trajs, err = run_sample_trajectories(
                model, processor, image_loader, sample,
                args.temperature, args.top_p, args.max_new_tokens,
                args.num_samples, SEED, i)
            if trajs is None:
                print(f"[{i}] skip sid={sid}: {err}", flush=True)
                continue
            rec = {
                "sample_id": sid,
                "hallucination_type": sample.get("hallucination_type"),
                "truth_noun": sample.get("truth_noun"),
                "false_adj": sample.get("false_adj"),
                "false_noun": sample.get("false_noun"),
                "image_description": sample.get("image_description"),
                "modified_description": sample.get("modified_description"),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_samples": args.num_samples,
                "seed": SEED,
                "trajectories": trajs,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            print(f"[{i}] sid={sid} done (K={len(trajs)} trajectories x "
                  f"{len(trajs[0]['rounds'])} rounds)", flush=True)
    print(f"DONE. wrote {n_ok} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
