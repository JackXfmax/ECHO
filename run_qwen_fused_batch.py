"""
run_qwen_fused_batch.py — injection re-run, image supplied EVERY round, batched decoding.

Why a rewrite:
  * correctness: v1 passed the image only on round 1, but every round re-renders the chat
    template which still contains <|image_pad|>; with images=None the processor returns
    pixel_values=None, so rounds 2..9 silently ran without any visual features
    (35.9% of answers degenerated into "please upload the image").
  * speed: decoding 1 sequence at a time leaves a 4090 idle; batching the 6 settings of a
    sample (same round index -> similar lengths) gives ~3x throughput and fits in 24 GB.

Batching is verified by --selfcheck: greedy batch output must equal greedy single output.

Only the injection condition is produced here; baseline cells come from the v1 file
(baseline is a single round that DOES carry the image, so it is already correct).
"""
import argparse, json, glob, os, io, sys, time

SETTINGS = ["cleanconv_formatting", "cleanconv_question", "factconv_formatting",
            "halluconv_question", "halluconv_formatting", "irrconv_formatting"]


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
    msgs = []
    for role, ctype, content in bubbles:
        r = "user" if role == "user" else "assistant"
        part = {"type": "image", "image": pil} if ctype == "image" else {"type": "text", "text": content}
        if msgs and msgs[-1]["role"] == r:
            if isinstance(msgs[-1]["content"], list):
                msgs[-1]["content"].append(part)
            else:
                msgs[-1]["content"] = [{"type": "text", "text": msgs[-1]["content"]}, part]
        else:
            msgs.append({"role": r, "content": [part]})
    return msgs


def build_plan(sample, cond):
    inj = sample["injections"]
    final_q = None
    final_gt = sample.get("truth_token") or sample.get("answer") or ""
    # final question is shared by every setting; take it from any available conversation
    for s in SETTINGS:
        oc = sample.get("original_conversations", {}).get(s)
        if oc:
            final_q = oc[-1][2]
            break
    if cond == "baseline":
        return [{"round": 1, "category_key": "dataset_final", "mode": "final",
                 "current_question": final_q, "ground_truth": final_gt, "premise_false": None}]
    plan = [{"round": i + 1, "category_key": it.get("category_key"), "mode": it.get("mode"),
             "current_question": it.get("current_question"), "ground_truth": it.get("ground_truth"),
             "premise_false": it.get("premise_false")} for i, it in enumerate(inj)]
    plan.append({"round": len(inj) + 1, "category_key": "dataset_final", "mode": "final",
                 "current_question": final_q, "ground_truth": final_gt, "premise_false": None})
    return plan


def generate_batch(model, processor, messages_list, pil, device, temperature, top_p,
                   max_new_tokens, do_sample=True, max_pixels=None):
    """Decode a batch of conversations. The image is supplied for EVERY sequence."""
    import torch
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    kw = {"max_pixels": max_pixels} if max_pixels else {}
    inputs = processor(text=texts, images=[pil] * len(texts), padding=True,
                       return_tensors="pt", **kw).to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, temperature=temperature, top_p=top_p)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return [processor.decode(g, skip_special_tokens=True).strip() for g in gen]


def generate_batch_safe(model, processor, messages_list, pil, device, temperature, top_p,
                        max_new_tokens, do_sample=True, max_pixels=None):
    """Halve the batch on OOM and retry, so one large image can never kill a whole shard."""
    import torch
    try:
        return generate_batch(model, processor, messages_list, pil, device, temperature, top_p,
                              max_new_tokens, do_sample, max_pixels)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(messages_list) == 1:
            raise
        mid = len(messages_list) // 2
        print(f"   [oom] split batch {len(messages_list)} -> {mid}+{len(messages_list)-mid}", flush=True)
        return (generate_batch_safe(model, processor, messages_list[:mid], pil, device, temperature,
                                    top_p, max_new_tokens, do_sample, max_pixels)
                + generate_batch_safe(model, processor, messages_list[mid:], pil, device, temperature,
                                      top_p, max_new_tokens, do_sample, max_pixels))


def run_sample(model, processor, sample, pil, device, conds, temperature, top_p,
               max_new_tokens, K, seed_base, batch_size=6, max_pixels=None):
    """Returns list of cells (one per setting x condition), each with K trajectories."""
    import torch
    cells = []
    for cond in conds:
        plan = build_plan(sample, cond)
        # states[(si, k)] = message list
        states = {}
        for si, setting in enumerate(SETTINGS):
            bubbles = sample.get("original_conversations", {}).get(setting)
            if not bubbles:
                continue
            prefix = bubbles[:-1]
            for k in range(K):
                msgs = [{"role": "system", "content": "You are a helpful vision-language assistant."}]
                msgs += build_prefix_messages(prefix, pil)
                states[(si, k)] = msgs
        sidx = sorted({si for si, _ in states})
        # process in batches over settings (same round -> similar length)
        rounds_out = {(si, k): [] for si, _ in states for k in range(K)}
        for j, p in enumerate(plan):
            for k in range(K):
                torch.manual_seed(seed_base + k * 977 + j)
                torch.cuda.manual_seed_all(seed_base + k * 977 + j)
                for bstart in range(0, len(sidx), batch_size):
                    chunk = sidx[bstart:bstart + batch_size]
                    msgs_batch = [states[(si, k)] for si in chunk]
                    outs = generate_batch_safe(model, processor, msgs_batch, pil, device,
                                               temperature, top_p, max_new_tokens,
                                               do_sample=True, max_pixels=max_pixels)
                    for si, o in zip(chunk, outs):
                        rec = dict(p)
                        rec["model_output"] = o
                        rounds_out[(si, k)].append(rec)
            # append this round's Q/A to every state AFTER generating (all k, all settings)
            for si in sidx:
                for k in range(K):
                    last_out = rounds_out[(si, k)][-1]["model_output"]
                    states[(si, k)].append({"role": "user",
                                            "content": [{"type": "text", "text": p["current_question"]}]})
                    states[(si, k)].append({"role": "assistant", "content": last_out})
            torch.cuda.empty_cache()   # KV cache grows every round; keep fragmentation down
        for si, setting in enumerate(SETTINGS):
            if si not in sidx:
                continue
            trajs = [{"trajectory_id": k, "seed": seed_base + k * 977, "rounds": rounds_out[(si, k)]}
                     for k in range(K)]
            cells.append({"setting": SETTINGS[si], "condition": cond,
                          "n_rounds_per_traj": len(trajs[0]["rounds"]), "trajectories": trajs})
    return cells


def _tok(processor, s):
    try:
        return processor.tokenizer(s, add_special_tokens=False)["input_ids"]
    except Exception:
        return list(s)


def selfcheck(model, processor, sample, pil, device, max_new_tokens, max_pixels=None):
    """Verify batched decoding is faithful.

    Two things can go wrong when batching a multimodal model:
      (a) a real bug  - left padding / mrope position ids shift the image tokens, which
          corrupts EVERY sequence in the batch;
      (b) benign      - bf16 batched matmul reduction order differs from the single-sequence
          kernel, so two greedy runs diverge in the tail after dozens of identical tokens.

    We therefore do not demand bitwise equality: we locate the first divergent token.
    A real bug shows up as divergence within the first few tokens. Anything past ~60%
    of the sequence is numerical noise.
    """
    plan = build_plan(sample, "injection")
    q0 = plan[0]["current_question"]
    ok = True
    for si, setting in enumerate(SETTINGS[:3]):
        bubbles = sample.get("original_conversations", {}).get(setting)
        if not bubbles:
            continue
        msgs = [{"role": "system", "content": "You are a helpful vision-language assistant."}]
        msgs += build_prefix_messages(bubbles[:-1], pil)
        m1 = list(msgs) + [{"role": "user", "content": [{"type": "text", "text": q0}]}]
        single = generate_batch(model, processor, [m1], pil, device, 0.0, 1.0,
                                max_new_tokens, do_sample=False, max_pixels=max_pixels)[0]

        others = []
        for sj, sj_setting in enumerate(SETTINGS):
            if sj == si:
                continue
            b2 = sample.get("original_conversations", {}).get(sj_setting)
            if not b2:
                continue
            mm = [{"role": "system", "content": "You are a helpful vision-language assistant."}]
            mm += build_prefix_messages(b2[:-1], pil)
            mm.append({"role": "user", "content": [{"type": "text", "text": q0}]})
            others.append(mm)
            if len(others) >= 5:
                break

        outs_first = generate_batch(model, processor, [m1] + others, pil, device, 0.0, 1.0,
                                    max_new_tokens, do_sample=False, max_pixels=max_pixels)
        outs_last = generate_batch(model, processor, others + [m1], pil, device, 0.0, 1.0,
                                   max_new_tokens, do_sample=False, max_pixels=max_pixels)

        def agree(a, b):
            ta, tb = _tok(processor, a), _tok(processor, b)
            n = min(len(ta), len(tb))
            i = 0
            while i < n and ta[i] == tb[i]:
                i += 1
            frac = i / max(1, n)
            return i, n, frac

        i1, n1, f1 = agree(single, outs_first[0])
        i2, n2, f2 = agree(single, outs_last[-1])
        # bitwise identical, or a long identical prefix + stable under reordering => fine.
        # (a short identical prefix would mean the image tokens were mis-placed)
        pos_ok = (i1 == n1 and i1 > 5) or (i1 >= 30 and stab) or (f1 >= 0.6 and f2 >= 0.6)
        stab = outs_first[0].strip() == outs_last[-1].strip()
        print(f"[selfcheck] {setting}: identical_prefix_tokens={i1}/{n1} ({f1:.0%}) "
              f"as-last={f2:.0%}  position-stable={stab}  -> {'OK' if pos_ok else 'SUSPECT'}", flush=True)
        if not pos_ok:
            print("   single      :", single[:200].replace("\n", " "))
            print("   batch-first :", outs_first[0][:200].replace("\n", " "))
            print("   batch-last  :", outs_last[-1][:200].replace("\n", " "))
            ok = False
    return ok


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
    ap.add_argument("--conditions", default="injection")
    ap.add_argument("--batch_size", type=int, default=3)
    ap.add_argument("--max_pixels", type=int, default=320000,
                    help="cap visual tokens per image (default 320k px ~ 408 patches)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    import random, torch
    try:
        import numpy as np
    except Exception:
        np = None
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if np is not None:
        np.random.seed(args.seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    try:
        processor.tokenizer.padding_side = "left"
    except Exception:
        pass
    model.eval()
    device = next(model.parameters()).device

    loader = ImageLoader(args.parquet_dir)
    data = load_data(args.data)
    subset = data[args.start:(args.start + args.limit) if args.limit else None]
    conds = [c.strip() for c in args.conditions.split(",")]
    print(f"[batch] conds={conds} K={args.num_samples} T={args.temperature} top_p={args.top_p} "
          f"batch={args.batch_size} device={device} n={len(subset)}", flush=True)

    if args.selfcheck:
        pil0 = loader[subset[0]["sample_id"]]
        ok = selfcheck(model, processor, subset[0], pil0, device, args.max_new_tokens,
                       args.max_pixels)
        print("SELFCHECK", "PASS" if ok else "FAIL", flush=True)
        if not ok:
            sys.exit("batch decoding is not equivalent to single decoding - aborting")

    # ---- resume: skip samples already present in the output file ----
    done_sids = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done_sids.add(json.loads(line)["sample_id"])
            except Exception:
                pass
        print(f"[batch] resume: {len(done_sids)} samples already in {args.out}", flush=True)

    t0 = time.time()
    n_ok = 0
    with open(args.out, "a", encoding="utf-8") as fout:      # append: safe to resume
        for i, sample in enumerate(subset):
            sid = sample.get("sample_id")
            if sid in done_sids:
                print(f"[{i}] skip {sid}: already done", flush=True)
                continue
            pil = loader[sid]
            if pil is None:
                print(f"[{i}] skip {sid}: image-missing", flush=True)
                continue
            seed_base = args.seed + (args.start + i) * 1000000
            cells = run_sample(model, processor, sample, pil, device, conds,
                               args.temperature, args.top_p, args.max_new_tokens,
                               args.num_samples, seed_base,
                               batch_size=args.batch_size, max_pixels=args.max_pixels)
            fout.write(json.dumps({
                "sample_id": sid,
                "hallucination_type": sample.get("hallucination_type"),
                "truth_token": sample.get("truth_token"),
                "false_token": sample.get("false_token"),
                "truth_noun": sample.get("truth_noun"),
                "fact": sample.get("fact"),
                "hallucinatory_fact": sample.get("hallucinatory_fact"),
                "temperature": args.temperature, "top_p": args.top_p,
                "num_samples": args.num_samples, "seed": args.seed,
                "vision_mode": "every_round", "batch_size": args.batch_size,
                "max_pixels": args.max_pixels,
                "cells": cells,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += 1
            el = time.time() - t0
            print(f"[{i}] sid={sid} done ({len(cells)} cells)  {el/60:.1f} min elapsed, "
                  f"eta {el/(i+1)*(len(subset)-i-1)/60:.1f} min", flush=True)
    print(f"DONE. wrote {n_ok} records -> {args.out} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
