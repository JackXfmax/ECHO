"""
server_attn_per_round.py -- clean mechanism probe (run on 4090 server)

The naive "src(hallucination round) vs recovery round" attention comparison is
CONFOUNDED: the recovery round always comes AFTER the hallucination round, so its
context is longer and the normalized image-attention fraction is mechanically
smaller -- independent of any mechanism. This script fixes that by measuring the
image-attention fraction at EVERY round 1..9 for the same trajectories, so we can
compare hallucinated vs faithful states at MATCHED round length (controlling for
context length). Output is joined with the Qwen verdicts locally in
analyze_attn_controlled.py.

Run:
  CUDA_VISIBLE_DEVICES=0 conda run -n train python server_attn_per_round.py \
      --fused fused_50_out_v3.jsonl --sample cross_model_selection.json \
      --parquet_dir /home/<SERVER_USER>/MMHalSnowball --out attn_per_round.json
"""
import argparse, json, os, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import server_common as sc

ap = argparse.ArgumentParser()
ap.add_argument("--fused", required=True)
ap.add_argument("--sample", required=True)
ap.add_argument("--parquet_dir", default="/home/<SERVER_USER>/MMHalSnowball")
ap.add_argument("--model", default="/home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--out", default="attn_per_round.json")
ap.add_argument("--max_new_tokens", type=int, default=8)
args = ap.parse_args()

fused = sc.load_fused_index(args.fused)
samples = json.load(open(args.sample, encoding="utf-8"))
processor = AutoProcessor.from_pretrained(args.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
model.eval()
tok = processor.tokenizer
IMG_PAD = tok.convert_tokens_to_ids("<|image_pad|>")
loader = sc.ImageLoader(args.parquet_dir)


def load_image(sample_id):
    """return the RAW image (resizing is deferred to attn_frac so we can
    shrink it on OOM retry)"""
    img = loader[sample_id]
    if img is None:
        raise FileNotFoundError(sample_id)
    return img


def resize(img, max_side):
    if max(img.size) <= max_side:
        return img
    s = max_side / max(img.size)
    return img.resize((int(img.size[0] * s), int(img.size[1] * s)))


def build_messages(traj, upto_round, image):
    msgs = []
    for rd in sorted(traj["rounds"], key=lambda r: r["round"]):
        if rd["round"] > upto_round:
            break
        if rd["round"] == 1:
            msgs.append({"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": rd["current_question"]}]})
        else:
            msgs.append({"role": "user", "content": rd["current_question"]})
        if rd["model_output"]:
            msgs.append({"role": "assistant", "content": rd["model_output"]})
    return msgs


@torch.no_grad()
def attn_frac(messages, image, max_side=512):
    image = resize(image, max_side)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0]
    img_positions = (input_ids == IMG_PAD).nonzero(as_tuple=True)[0]
    n_img = img_positions.numel()
    if n_img == 0:
        return None, 0
    out = model(**inputs, output_attentions=True, return_dict=True)
    L = len(out.attentions)
    fracs = []
    for li in range(L):
        att = out.attentions[li][0]          # (heads, Q, K)
        for h in range(att.shape[0]):
            a = att[h]
            img_mass = a[:, img_positions].sum(dim=1)
            denom = a.sum(dim=1) + 1e-9
            fracs.append((img_mass / denom).mean().item())
    return float(torch.tensor(fracs).mean()), n_img


def attn_frac_robust(messages, image):
    """Full-sequence attention tensors for late rounds can be ~7GB on top of the
    15GB bf16 model, which OOMs a 24GB card. Shrink the image (fewer image tokens)
    and retry instead of silently dropping the trajectory."""
    for max_side in (512, 384, 288, 224):
        try:
            frac, nimg = attn_frac(messages, image, max_side=max_side)
            torch.cuda.empty_cache()
            return frac, nimg, max_side
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at max_side={max_side}; shrinking")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print("  ERR", e)
            torch.cuda.empty_cache()
            return None, 0, max_side
    return None, 0, 0


results = []
for s in samples:
    key = (s["sample_id"], s["setting"], s["traj"])
    inj = fused.get(key, {}).get("injection")
    if inj is None:
        print("skip (no injection):", key); continue
    image = load_image(s["sample_id"])
    for rd in sorted(inj["rounds"], key=lambda r: r["round"]):
        r = rd["round"]
        frac, nimg, used_side = attn_frac_robust(build_messages(inj, r, image), image)
        if frac is None:
            print("skip (no img tokens / failed):", key, "r", r); continue
        results.append({"sample_id": s["sample_id"], "setting": s["setting"],
                        "trajectory_id": s["traj"], "round": r,
                        "image_attn_frac": frac, "n_image_tokens": int(nimg),
                        "max_side": used_side})
        print(f"{s['sample_id']} r{r} frac={frac:.3f} nimg={nimg} side={used_side}")

json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2)
print(f"saved {args.out}  ({len(results)} round-records)")
