"""
run_qwen_fused.py — 融合实验：前置语境(6) x 是否注入(2) 因子设计

实验设计（2026-10-23 与用户敲定）：
  因子 A（前置语境，取自原数据集 6 种 conversation_setting 的对话前缀）：
      cleanconv_*        = 无前置（发图直接问）
      factconv_*         = 真实描述前置
      halluconv_*        = 幻觉描述前置   <-- 关键
      irrconv_*          = 无关问题前置
  因子 B（是否注入）：
      baseline  = 只回答该 setting 的最终问题（数据集原生核心题）
      injection = 前置 + 我们注入的 8 轮渐进逼问 + 最终问题(第9轮)
  交互效应（核心卖点）：幻觉前置是否显著放大后续注入的诱导成功率 = snowballing 本质假设

每张图轮次：
  基线条件：cleanconv 1 轮；其余 2 轮（含 1 条数据集自带前置问答）
  注入条件：cleanconv 9 轮（8注入+1核心题）；其余 10 轮（1前置+8注入+1核心题）
  单轨迹 Qwen 生成 60 次；K=3 -> 180 次/图；50 图 -> 9000 次

采样：T=0.7 top_p=0.9，每条件 K 条独立轨迹；消除视觉前向非确定(flash_sdp=False)。
"""

import argparse, json, glob, os, io, sys

def get_bytes(img):
    if isinstance(img, dict):
        return img.get("bytes")
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    return None


class ImageLoader:
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
        b = get_bytes(tbl.column("image")[row].as_py())
        if not b:
            return None
        from PIL import Image
        return Image.open(io.BytesIO(b)).convert("RGB")


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        return [data] if isinstance(data, dict) else data
    except json.JSONDecodeError:
        pass
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def build_prefix_messages(bubbles, pil):
    """Convert dataset bubbles (role/type/content) into Qwen chat messages.
    - the image bubble becomes a real PIL image part
    - consecutive same-role messages are merged (Qwen expects image+text in one user turn)"""
    msgs = []
    for role, ctype, content in bubbles:
        r = "user" if role == "user" else "assistant"
        if ctype == "image":
            part = {"type": "image", "image": pil}
        else:
            part = {"type": "text", "text": content}
        if msgs and msgs[-1]["role"] == r:
            if isinstance(msgs[-1]["content"], list):
                msgs[-1]["content"].append(part)
            else:
                msgs[-1]["content"] = [{"type": "text", "text": msgs[-1]["content"]}, part]
        else:
            msgs.append({"role": r, "content": [part]})
    return msgs


def generate_once(model, processor, messages, device, temperature, top_p, max_new_tokens, pil=None):
    import torch
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [pil] if pil is not None else None
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=temperature, top_p=top_p)
    return processor.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_condition(model, processor, sample, pil, device, setting, cond,
                  temperature, top_p, max_new_tokens, K, seed, vision_mode="every_round"):
    """One (setting, condition) cell: run K independent trajectories."""
    import torch
    bubbles = sample["original_conversations"][setting]
    prefix = bubbles[:-1]                      # everything except the final question
    final_b = bubbles[-1]                      # the dataset's core question
    final_q = final_b[2]
    final_gt = sample.get("truth_token") or sample.get("answer") or ""

    inj = sample["injections"]
    if cond == "baseline":
        plan = [{"round": 1, "category_key": "dataset_final", "mode": "final",
                 "current_question": final_q, "ground_truth": final_gt,
                 "premise_false": None}]
    else:
        plan = [{"round": i + 1,
                 "category_key": it.get("category_key"),
                 "mode": it.get("mode"),
                 "current_question": it.get("current_question"),
                 "ground_truth": it.get("ground_truth"),
                 "premise_false": it.get("premise_false")} for i, it in enumerate(inj)]
        plan.append({"round": len(inj) + 1, "category_key": "dataset_final", "mode": "final",
                     "current_question": final_q, "ground_truth": final_gt,
                     "premise_false": None})

    trajs = []
    for k in range(K):
        seed_k = seed + k
        torch.manual_seed(seed_k)
        torch.cuda.manual_seed_all(seed_k)
        messages = [{"role": "system", "content": "You are a helpful vision-language assistant."}]
        messages += build_prefix_messages(prefix, pil)
        rounds = []
        for j, p in enumerate(plan):
            # vision_mode:
            #   every_round = pass the image at every round (correct: <|image_pad|> is in the
            #                 template at every round, so pixel_values must be supplied every time)
            #   first_only  = legacy bug: rounds >=2 get pixel_values=None -> image is lost
            pil_arg = pil if (vision_mode == "every_round" or j == 0) else None
            ans = generate_once(model, processor, messages, device,
                                temperature, top_p, max_new_tokens, pil_arg)
            rec = dict(p)
            rec["model_output"] = ans
            rounds.append(rec)
            messages.append({"role": "user", "content": [{"type": "text", "text": p["current_question"]}]})
            messages.append({"role": "assistant", "content": ans})
        trajs.append({"trajectory_id": k, "seed": seed_k, "rounds": rounds})
    return trajs


SETTINGS = ["cleanconv_formatting", "cleanconv_question",
            "factconv_formatting", "halluconv_question",
            "halluconv_formatting", "irrconv_formatting"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet_dir", default="/home/<SERVER_USER>/MMHalSnowball")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conditions", default="baseline,injection")
    ap.add_argument("--vision_mode", default="every_round",
                    choices=["every_round", "first_only"])
    args = ap.parse_args()

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
    conds = [c.strip() for c in args.conditions.split(",")]
    print(f"[fused] SEED={SEED} T={args.temperature} top_p={args.top_p} K={args.num_samples} "
          f"conds={conds} device={args.device}", flush=True)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("loading model ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    device = next(model.parameters()).device

    image_loader = ImageLoader(args.parquet_dir)
    data = load_data(args.data)
    print(f"data samples: {len(data)}", flush=True)

    subset = data[args.start: (args.start + args.limit) if args.limit else None]
    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(subset):
            sid = sample.get("sample_id")
            pil = image_loader[sid]
            if pil is None:
                print(f"[{i}] skip {sid}: image-missing", flush=True)
                continue
            cells = []
            for si, setting in enumerate(SETTINGS):
                if setting not in sample.get("original_conversations", {}):
                    continue
                for ci, cond in enumerate(conds):
                    # absolute sample index so the two GPU shards never share seeds
                    seed_cell = SEED + (args.start + i) * 1000000 + si * 10000 + ci * 1000
                    trajs = run_condition(model, processor, sample, pil, device,
                                          setting, cond, args.temperature, args.top_p,
                                          args.max_new_tokens, args.num_samples, seed_cell,
                                          vision_mode=args.vision_mode)
                    cells.append({"setting": setting, "condition": cond,
                                  "n_rounds_per_traj": len(trajs[0]["rounds"]),
                                  "trajectories": trajs})
            rec = {
                "sample_id": sid,
                "hallucination_type": sample.get("hallucination_type"),
                "truth_token": sample.get("truth_token"),
                "false_token": sample.get("false_token"),
                "truth_noun": sample.get("truth_noun"),
                "fact": sample.get("fact"),
                "hallucinatory_fact": sample.get("hallucinatory_fact"),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_samples": args.num_samples,
                "seed": SEED,
                "cells": cells,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            print(f"[{i}] sid={sid} done ({len(cells)} cells)", flush=True)
    print(f"DONE. wrote {n_ok} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
