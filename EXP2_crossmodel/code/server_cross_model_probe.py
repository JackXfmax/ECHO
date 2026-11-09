"""
server_cross_model_probe.py  -- STEP 3 (run on the 4090 server, NOT locally)

Defensive cross-architecture check. Reviewer will ask: "is the self-recovery /
oscillation just a Qwen quirk?" Here we replay the SAME injection protocol (the
scripted attack questions, which are model-agnostic) through a DIFFERENT LVLM
(LLaVA-1.5-7B, or InternVL2-2B) and emit a judge-ready input file. If the same
non-monotonic oscillation appears, the phenomenon is architectural, not a Qwen bug.

Images are read from the MMHalSnowball parquet by sample_id (see server_common).

Run:
  conda activate train
  python server_cross_model_probe.py \
      --fused fused_50_out_v3.jsonl \
      --parquet_dir /home/<SERVER_USER>/MMHalSnowball \
      --model llava-hf/llava-1.5-7b-hf \
      --include_file cross_model_selection_ids.json \
      --n_images 15 \
      --resume \
      --out cross_model_judge_input.jsonl

Follow-up (judge + per-round curve), can run locally via API:
  python run_glm_judge.py --key <KEY> --model DeepSeek-V4.1-Flash-1017-Event \
      --inp cross_model_judge_input.jsonl --out cross_model_judge_output.jsonl
"""
import argparse, json, os, random
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import server_common as sc

ap = argparse.ArgumentParser()
ap.add_argument("--fused", required=True)
ap.add_argument("--parquet_dir", default="/home/<SERVER_USER>/MMHalSnowball")
ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
ap.add_argument("--include_file", default=None,
                help="JSON list of sample_ids that MUST be included (recovery set)")
ap.add_argument("--n_images", type=int, default=12)
ap.add_argument("--first_traj_only", action="store_true",
                help="only run the first trajectory per (sample,setting) to bound GPU time")
ap.add_argument("--out", default="cross_model_judge_input.jsonl")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--max_new_tokens", type=int, default=128)
ap.add_argument("--no_resume", action="store_true", help="ignore existing --out and overwrite")
ap.add_argument("--resume", action="store_true", help="resume (default; explicit)")
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
json.dump(chosen, open("cross_model_chosen.json", "w", encoding="utf-8"), indent=2)
n_traj_total = sum(len(by_sample[s]) for s in chosen)
if args.first_traj_only:
    # one trajectory per (sample, setting) -- dedup by SETTING, not by trajectory_id:
    # trajectory_id is 0/1/2 in EVERY setting, so deduping on it globally would
    # collapse 6 settings down to 3.
    n_traj_total = sum(len({t[0] for t in by_sample[s]}) for s in chosen)
print(f"chosen {len(chosen)} images (forced-include={len(include)}); "
      f"trajectories to run = {n_traj_total} (first_traj_only={args.first_traj_only})")

# ---- resume support ----
done = set()
if (not args.no_resume) and os.path.exists(args.out):
    for l in open(args.out, encoding="utf-8"):
        try:
            done.add(json.loads(l)["uid"])
        except Exception:
            pass
    print(f"resume: {len(done)} rows already done, skipping")

processor = AutoProcessor.from_pretrained(args.model)
model = LlavaForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map="auto")
model.eval()
loader = sc.ImageLoader(args.parquet_dir)


def generate(text, image):
    # LLaVA-1.5 official manual prompt. The <image> placeholder is already
    # embedded in `text` (only on the first user turn). We pass the SAME image on
    # EVERY call so the single image token always has features.
    # DO NOT use apply_chat_template: this LLaVA build's template silently drops
    # multi-turn assistant/user text, and yields 0 image tokens on round>1.
    inp = processor(text=text, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
    return processor.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


out = open(args.out, "a", encoding="utf-8")
n_write = 0
for sid in chosen:
    try:
        image = loader[sid]
    except Exception as e:
        print("skip (no image):", sid, e); continue
    trajs = by_sample[sid]
    if args.first_traj_only:
        seen = set()
        keep = []
        for setting, tj in trajs:
            if setting in seen:
                continue
            seen.add(setting)
            keep.append((setting, tj))
        trajs = keep
    for setting, tj in trajs:
        traj_id = tj["trajectory_id"]
        prompt = None  # growing LLaVA manual prompt; <image> only on first user turn
        for rd in sorted(tj["rounds"], key=lambda r: r["round"]):
            uid = f"{sid}_{setting}_{traj_id}_r{rd['round']}"
            q = rd["current_question"]
            if uid in done:
                # rebuild history so later rounds see the full conversation
                a = (rd["model_output"] or "").replace("\n", " ").strip()
                if prompt is None:
                    prompt = f"USER: <image>\n{q} ASSISTANT: {a}"
                else:
                    prompt = f"{prompt} {q} ASSISTANT: {a}"
                continue
            if prompt is None:
                text = f"USER: <image>\n{q} ASSISTANT:"
            else:
                text = f"{prompt} {q} ASSISTANT:"
            ans = generate(text, image)
            a = ans.replace("\n", " ").strip()
            if prompt is None:
                prompt = f"USER: <image>\n{q} ASSISTANT: {a}"
            else:
                prompt = f"{prompt} {q} ASSISTANT: {a}"
            rec = {
                "uid": uid, "sample_id": sid, "setting": setting, "condition": "injection",
                "trajectory_id": traj_id, "round": rd["round"], "category_key": rd["category_key"],
                "current_question": rd["current_question"], "ground_truth": rd["ground_truth"],
                "model_output": ans, "mode": rd.get("mode"),
                "hallucination_type": tj.get("hallucination_type") or rd.get("hallucination_type"),
                "truth_token": rd.get("truth_token"), "false_token": rd.get("false_token"),
                "judge_model": "LLaVA-1.5-7B",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush(); n_write += 1
        print(f"  done {sid} {setting} {traj_id}", flush=True)
out.close()
print(f"saved/append {args.out}  (+{n_write} new rows) -> feed to run_glm_judge.py next")
