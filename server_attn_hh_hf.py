"""
server_attn_hh_hf.py -- REAL-trajectory H->H vs H->F attention comparison
(CVPR figure, EXP4 sample 131064738 / trajectory t0, verdicts r1-r9 = HHHHFHHFF).

Faithful teacher-forced replay of the frozen EXP4 protocol
(_server_run_qwen_fused_batch.py::run_sample):
  context = system + prefix(user[image]+"Please describe...") + pairs(1..k-1)
  answer  = recorded O_k  (generated BEFORE Q_k was appended -- frozen semantics)
For target rounds k, forward prompt+O_k with attn_implementation="eager",
output_attentions=True; for every requested (layer, head) take rows at O_k
token positions, slice image_pad columns -> per-token vectors; save full
(n_ans, n_img) row matrices so the local side can aggregate over any token
subset (full turn / entity tokens).

Run:
  CUDA_VISIBLE_DEVICES=0 /home/<SERVER_USER>/miniconda3/envs/train/bin/python \
      server_attn_hh_hf.py --image elephant.jpg --traj traj_131064738_t0.json \
      --heads 20:24,22:2,15:24 --out attn_HH_HF_multi.npz
"""
import argparse, json, torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--image", required=True)
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--heads", default="20:24", help="comma list of layer:head")
ap.add_argument("--target_rounds", default="4,5")   # 4=H->H, 5=H->F
ap.add_argument("--scale", type=float, default=1.0,
                help="upscale image before processing -> finer attention grid (display only)")
args = ap.parse_args()

HEADS = []
for spec in args.heads.split(","):
    li, hi = spec.split(":")
    HEADS.append((int(li), int(hi)))
print("[config] heads =", HEADS)

payload = json.load(open(args.traj, encoding="utf-8"))
img = Image.open(args.image).convert("RGB")
orig_w, orig_h = img.size
if args.scale != 1.0:
    img = img.resize((int(orig_w * args.scale), int(orig_h * args.scale)),
                     Image.LANCZOS)
print(f"[smoke] image {orig_w}x{orig_h} -> processed {img.size[0]}x{img.size[1]} (scale={args.scale})")

processor = AutoProcessor.from_pretrained(args.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()
tok = processor.tokenizer
IMG_PAD = tok.convert_tokens_to_ids("<|image_pad|>")
IM_END = tok.convert_tokens_to_ids("<|im_end|>")

SYSTEM = {"role": "system", "content": "You are a helpful vision-language assistant."}

def prefix_msgs():
    msgs = []
    for b in payload["prefix"]:
        role, ctype, content = b[0], b[1], b[2]
        r = "user" if role == "user" else "assistant"
        part = {"type": "image"} if ctype == "image" else {"type": "text", "text": content}
        if msgs and msgs[-1]["role"] == r:
            msgs[-1]["content"].append(part)
        else:
            msgs.append({"role": r, "content": [part]})
    return msgs

def build_history(upto_pair):
    msgs = [SYSTEM] + prefix_msgs()
    for rd in payload["rounds"][:upto_pair]:
        msgs.append({"role": "user", "content": [{"type": "text", "text": rd["q"]}]})
        msgs.append({"role": "assistant", "content": rd["a"]})
    return msgs

@torch.no_grad()
def extract(k):
    """k = 1-based target round -> dict per head."""
    hist = build_history(k - 1)
    base_text = processor.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
    full_msgs = hist + [{"role": "assistant", "content": payload["rounds"][k - 1]["a"]}]
    full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    inp_base = processor(text=[base_text], images=[img], max_pixels=320000,
                         return_tensors="pt").to(model.device)
    inp_full = processor(text=[full_text], images=[img], max_pixels=320000,
                         return_tensors="pt").to(model.device)
    base_ids = inp_base["input_ids"][0]
    full_ids = inp_full["input_ids"][0]
    Lb, Lf = base_ids.numel(), full_ids.numel()
    pref_ok = bool((base_ids == full_ids[:Lb]).all().item())
    print(f"[smoke] round {k}: base={Lb} full={Lf} prefix_ok={pref_ok}")
    assert pref_ok, "base ids are not a prefix of full ids -- template mismatch"

    end_pos = Lf
    for i in range(Lb, Lf):
        if full_ids[i].item() == IM_END:
            end_pos = i
            break
    ans_positions = torch.arange(Lb, end_pos, device=full_ids.device)
    img_positions = (full_ids == IMG_PAD).nonzero(as_tuple=True)[0]
    thw = inp_full["image_grid_thw"][0].tolist()
    h_p, w_p = int(thw[1]), int(thw[2])
    n_img = img_positions.numel()
    assert n_img == (h_p // 2) * (w_p // 2) * int(thw[0]), "grid mismatch"
    print(f"[smoke] answer tokens={ans_positions.numel()}, img tokens={n_img}, "
          f"grid {h_p//2}x{w_p//2}, seq_len={Lf}")

    out = model(**inp_full, output_attentions=True, return_dict=True)
    per_head = {}
    max_layer = max(li for li, _ in HEADS)
    for li, hi in HEADS:
        att = out.attentions[li - 1][0]          # (heads, Q, K)
        assert att.shape[1] == att.shape[2] == Lf
        a = att[hi]
        rows = a[ans_positions]                  # (n_ans, K)
        rows_img = rows[:, img_positions].detach().float()  # (n_ans, n_img)
        img_attn = rows_img.mean(dim=0)
        attn2d = img_attn.reshape(h_p // 2, w_p // 2).cpu().numpy()
        per_head[f"rows_L{li}H{hi}_r{k}"] = rows_img.cpu().numpy().astype(np.float16)
        per_head[f"map_L{li}H{hi}_r{k}"] = attn2d
        per_head[f"pertok_L{li}H{hi}_r{k}"] = rows_img.mean(dim=1).cpu().numpy().astype(np.float32)
        print(f"[done] r{k} L{li}H{hi}: map {attn2d.shape}, "
              f"max={attn2d.max():.4f}, img_attn_sum={attn2d.sum():.4f}")
        del att, a, rows
    ans_tokens = [tok.decode([full_ids[i]]) for i in ans_positions]
    del out
    torch.cuda.empty_cache()
    info = {"round": k, "seq_len": Lf, "n_ans": ans_positions.numel(),
            "n_img": n_img, "grid": [h_p // 2, w_p // 2]}
    return per_head, ans_tokens, info

targets = [int(x) for x in args.target_rounds.split(",")]
allnpz, meta = {}, {"heads": HEADS, "sample": payload["sample_id"], "traj": "t0",
                    "verdict_seq": "HHHHFHHFF"}
for k in targets:
    ph, at, info = extract(k)
    allnpz.update(ph)
    meta[f"r{k}"] = {"info": info, "answer": payload["rounds"][k - 1]["a"],
                     "question": payload["rounds"][k - 1]["q"],
                     "ans_tokens": at}

np.savez_compressed(args.out, **allnpz,
                    grid=np.array(meta[f"r{targets[0]}"]["info"]["grid"]),
                    orig_size=np.array([orig_w, orig_h]),
                    proc_size=np.array([img.size[0], img.size[1]]))
with open(args.out.replace(".npz", "_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)
print(f"[saved] {args.out} + meta")
