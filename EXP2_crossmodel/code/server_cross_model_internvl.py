"""
server_cross_model_internvl.py  --  EXP2b : InternVL2-4B cross-architecture probe
(run on the 4090 server, NOT locally)

Same protocol as EXP2a (LLaVA-1.5-7B): the SAME 12 images, the SAME 6 injection
settings, the SAME 9 scripted attack rounds. If the non-monotonic oscillation also
appears here, the snowball effect is architectural rather than a Qwen/LLaVA quirk.

WHY THIS FILE IS NOT A COPY OF THE LLaVA ONE
--------------------------------------------
InternVL2 is loaded with trust_remote_code and exposes its own `model.chat()`.
Two traps in that implementation (verified by reading the cached
modeling_internvl_chat.py on this server):

1. `chat()` only auto-prepends `<image>` when `history is None`. On round >= 2 the
   image placeholder would be missing entirely -> pixel_values would have no
   IMG_CONTEXT slots to fill -> the model effectively answers WITHOUT the image.
   That is the exact confound we must avoid (see project rule: every round must
   receive the image).
2. `query.replace('<image>', image_tokens, 1)` replaces only the FIRST occurrence.
   If earlier turns keep their `<image>` marker, the replacement lands on round 1
   forever and the current round silently gets the literal string "<image>".

Fix: pass `'<image>\n' + question` on EVERY turn, then strip the marker from the
history entry that `chat()` just stored, so the single live occurrence is always
the CURRENT turn.

Run:
  cd /home/<SERVER_USER>/mmhal_inject_v3
  CUDA_VISIBLE_DEVICES=1 python server_cross_model_internvl.py \
      --fused fused_50_out_v3.jsonl \
      --parquet_dir /home/<SERVER_USER>/MMHalSnowball \
      --model OpenGVLab/InternVL2-4B \
      --include_file cross_model_selection_ids.json \
      --n_images 12 --first_traj_only --resume \
      --out cross_model_judge_input_internvl.jsonl

Then judge locally with the SAME DeepSeek judge used for every other model:
  python run_glm_judge.py --key <KEY> --model DeepSeek-V4.1-Flash-1017-Event \
      --inp cross_model_judge_input_internvl.jsonl \
      --out cross_model_judge_output_internvl.jsonl
"""
import argparse, json, os, random
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import server_common as sc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios,
                                                    orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=448, max_num=12):
    """image: PIL.Image. returns [n_tiles, 3, 448, 448] float tensor."""
    image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size,
                                use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(im) for im in images])


ap = argparse.ArgumentParser()
ap.add_argument("--fused", required=True)
ap.add_argument("--parquet_dir", default="/home/<SERVER_USER>/MMHalSnowball")
ap.add_argument("--model", default="OpenGVLab/InternVL2-4B")
ap.add_argument("--include_file", default=None)
ap.add_argument("--n_images", type=int, default=12)
ap.add_argument("--first_traj_only", action="store_true")
ap.add_argument("--out", default="cross_model_judge_input_internvl.jsonl")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--max_new_tokens", type=int, default=128)
ap.add_argument("--max_num", type=int, default=12, help="max dynamic tiles per image")
ap.add_argument("--device", default="cuda")
ap.add_argument("--limit", type=int, default=0, help="smoke: stop after N new rows")
ap.add_argument("--no_resume", action="store_true")
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

by_sample = sc.load_by_sample(args.fused)
all_ids = sorted(by_sample)
rng = random.Random(args.seed)

include = []
if args.include_file:
    include = json.load(open(args.include_file, encoding="utf-8"))
    include = sorted({s for s in include if s in by_sample})
remaining = [s for s in all_ids if s not in set(include)]
fill = rng.sample(remaining, max(0, args.n_images - len(include)))
chosen = (include + fill)[:args.n_images]
json.dump(chosen, open("cross_model_chosen_internvl.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
n_traj_total = sum(len(by_sample[s]) for s in chosen)
if args.first_traj_only:
    n_traj_total = sum(len({t[0] for t in by_sample[s]}) for s in chosen)
print(f"chosen {len(chosen)} images (forced={len(include)}); "
      f"trajectories = {n_traj_total} (first_traj_only={args.first_traj_only})", flush=True)

# ---- resume: read back OUR OWN outputs so rebuilt history is model-consistent ----
done = {}
if (not args.no_resume) and os.path.exists(args.out):
    for l in open(args.out, encoding="utf-8"):
        try:
            r = json.loads(l)
            done[r["uid"]] = r.get("model_output", "")
        except Exception:
            pass
    print(f"resume: {len(done)} rows already done, skipping", flush=True)

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                  trust_remote_code=True).to(args.device).eval()
print("model loaded:", type(model).__name__, "num_image_token=", getattr(model, "num_image_token", "?"),
      flush=True)
loader = sc.ImageLoader(args.parquet_dir)

GEN = dict(max_new_tokens=args.max_new_tokens, do_sample=False)


def chat_once(pixels, question, history, max_num):
    resp, hist = model.chat(tokenizer, pixels, question, GEN,
                            history=history, return_history=True)
    return resp, hist


out = open(args.out, "a", encoding="utf-8")
n_write = 0
for sid in chosen:
    try:
        pil = loader[sid]
    except Exception as e:
        print("skip (no image):", sid, e, flush=True)
        continue
    pixels = load_image(pil, max_num=args.max_num).to(torch.bfloat16).to(args.device)
    trajs = by_sample[sid]
    if args.first_traj_only:
        seen, keep = set(), []
        for setting, tj in trajs:
            if setting in seen:
                continue
            seen.add(setting)
            keep.append((setting, tj))
        trajs = keep
    for setting, tj in trajs:
        traj_id = tj["trajectory_id"]
        history = None
        for rd in sorted(tj["rounds"], key=lambda r: r["round"]):
            uid = f"{sid}_{setting}_{traj_id}_r{rd['round']}"
            q = (rd["current_question"] or "").strip()
            if uid in done:
                history = (history or []) + [(q, (done[uid] or "").replace("\n", " ").strip())]
                continue
            question = "<image>\n" + q
            try:
                ans, history = chat_once(pixels, question, history, args.max_num)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM at {uid}: retry with max_num=1", flush=True)
                small = load_image(pil, max_num=1).to(torch.bfloat16).to(args.device)
                ans, history = chat_once(small, question, history, 1)
                del small
            # strip the marker so the NEXT turn's single replace() hits the CURRENT turn
            if history:
                history[-1] = (history[-1][0].replace("<image>\n", "", 1), history[-1][1])
            ans = (ans or "").replace("\n", " ").strip()
            rec = {
                "uid": uid, "sample_id": sid, "setting": setting, "condition": "injection",
                "trajectory_id": traj_id, "round": rd["round"], "category_key": rd["category_key"],
                "current_question": rd["current_question"], "ground_truth": rd["ground_truth"],
                "model_output": ans, "mode": rd.get("mode"),
                "hallucination_type": tj.get("hallucination_type") or rd.get("hallucination_type"),
                "truth_token": rd.get("truth_token"), "false_token": rd.get("false_token"),
                "judge_model": "InternVL2-4B",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_write += 1
            torch.cuda.empty_cache()
            if args.limit and n_write >= args.limit:
                print("smoke limit reached", flush=True)
                out.close()
                raise SystemExit(0)
        print(f"  done {sid} {setting} {traj_id}", flush=True)
    del pixels
    torch.cuda.empty_cache()
out.close()
print(f"saved/append {args.out}  (+{n_write} new rows) -> judge with run_glm_judge.py", flush=True)
