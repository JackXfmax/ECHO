"""
server_attn_heatmap.py -- extract L20H24 attention over image tokens for the
"self-recovery" style single-turn prompt (CVPR figure).

Run:
  CUDA_VISIBLE_DEVICES=0 conda run -n train python server_attn_heatmap.py \
      --image /home/<SERVER_USER>/attn_viz/elephant.jpg \
      --out /home/<SERVER_USER>/attn_viz/attn_L20H24.npz

Pipeline:
  1) greedy-generate an answer for the single-turn prompt (max 80 tokens)
  2) ONE re-forward of prompt+answer with attn_implementation="eager",
     output_attentions=True
  3) take layer-20 head-24 matrix (Q,K); rows = answer-token positions,
     cols = <|image_pad|> positions
  4) average rows -> 1D vector over image tokens; reshape to
     (h_p/2, w_p/2) via image_grid_thw (Qwen2.5-VL merged-token grid)
  5) save .npz: attn2d, grid, generated text/tokens, image size
"""
import argparse, json, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--image", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default="Based on the visual evidence, describe the actual animals in the image.")
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--head", type=int, default=24)
ap.add_argument("--max_new_tokens", type=int, default=80)
ap.add_argument("--max_side", type=int, default=768)
args = ap.parse_args()

# ---- smoke test: image load + resize ----
img = Image.open(args.image).convert("RGB")
orig_w, orig_h = img.size
if max(img.size) > args.max_side:
    s = args.max_side / max(img.size)
    img = img.resize((int(orig_w * s), int(orig_h * s)))
print(f"[smoke] original image {orig_w}x{orig_h} -> resized {img.size[0]}x{img.size[1]}")

processor = AutoProcessor.from_pretrained(args.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()
tok = processor.tokenizer
IMG_PAD = tok.convert_tokens_to_ids("<|image_pad|>")
print(f"[smoke] IMG_PAD id = {IMG_PAD}")

messages = [{"role": "user", "content": [
    {"type": "image"}, {"type": "text", "text": args.prompt}]}]

@torch.no_grad()
def forward(inputs, gen=False, **kw):
    return (model.generate(**inputs, **kw) if gen
            else model(**inputs, output_attentions=True, return_dict=True))

# ---- step 1: greedy generation ----
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
prompt_len = inputs["input_ids"].shape[1]
thw = inputs["image_grid_thw"][0].tolist()          # (t, h_p, w_p)
t_p, h_p, w_p = int(thw[0]), int(thw[1]), int(thw[2])
expected_img_tokens = (h_p // 2) * (w_p // 2) * t_p
n_img = int((inputs["input_ids"][0] == IMG_PAD).sum())
print(f"[smoke] grid_thw={thw} -> merged grid {h_p//2}x{w_p//2} = {expected_img_tokens} tokens; "
      f"input has {n_img} image_pad; prompt_len={prompt_len}")
assert n_img == expected_img_tokens, "image token count mismatch with merged grid"

gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
gen_new = gen_ids[0, prompt_len:].tolist()
gen_text = tok.decode(gen_new, skip_special_tokens=True)
print(f"[gen] answer: {gen_text!r}")

# ---- step 2: re-forward prompt+answer with attentions ----
full_ids = gen_ids[:, :prompt_len + len(gen_new)]
inputs_full = {
    "input_ids": full_ids,
    "attention_mask": torch.ones_like(full_ids),
    "pixel_values": inputs["pixel_values"],
    "image_grid_thw": inputs["image_grid_thw"],
}
out = model(**inputs_full, output_attentions=True, return_dict=True)

# ---- step 3: layer/head slice ----
L = len(out.attentions)
li = args.layer - 1
assert 0 <= li < L, f"layer {args.layer} out of range (L={L})"
att = out.attentions[li][0]                    # (heads, Q, K)
print(f"[smoke] attention tensor layer{args.layer}: {tuple(att.shape)} "
      f"(heads={att.shape[0]}, Q={att.shape[1]}, K={att.shape[2]})")
assert att.shape[1] == att.shape[2], "attention matrix not square"
a = att[args.head]                             # (Q, K)

img_positions = (full_ids[0] == IMG_PAD).nonzero(as_tuple=True)[0]
ans_positions = torch.arange(prompt_len, full_ids.shape[1], device=a.device)
print(f"[smoke] answer tokens: {len(ans_positions)}, image positions: {img_positions.numel()}")

# ---- step 4: average rows over answer tokens -> map over image tokens ----
rows = a[ans_positions]                        # (n_ans, K)
img_attn = rows[:, img_positions].mean(dim=0)  # (n_img,)
n_img_final = img_attn.numel()
attn2d = img_attn.reshape(h_p // 2, w_p // 2).float().cpu().numpy()
print(f"[smoke] 2D attention map shape: {attn2d.shape}, "
      f"min={attn2d.min():.2e} max={attn2d.max():.2e} sum(img_attn)={img_attn.sum().item():.4f}")
assert attn2d.ndim == 2

# per-token breakdown (top attended image token + answer token table)
ans_tokens = [tok.decode([i]) for i in gen_new]
per_tok = rows[:, img_positions].mean(dim=1).float().cpu().tolist()

import numpy as np
np.savez_compressed(
    args.out,
    attn2d=attn2d,
    grid=np.array([h_p // 2, w_p // 2]),
    orig_size=np.array([orig_w, orig_h]),
    resized_size=np.array(img.size),
    layer_head=np.array([args.layer, args.head]),
    gen_ids=np.array(gen_new),
    per_tok=np.array(per_tok),
)
meta = {"gen_text": gen_text, "ans_tokens": ans_tokens,
        "prompt": args.prompt, "layer": args.layer, "head": args.head,
        "grid": [h_p // 2, w_p // 2], "orig_size": [orig_w, orig_h],
        "n_img_tokens": n_img_final, "model": args.model}
with open(args.out.replace(".npz", "_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[done] saved {args.out} + meta")
