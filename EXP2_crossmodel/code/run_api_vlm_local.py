"""
run_api_vlm_local.py -- replay the MMHalSnowball injection protocol through
proprietary frontier VLMs via an OpenAI-compatible relay.

Runs LOCALLY (not on the 4090 server) because the relay is reachable from here but
NOT from the server (server -> <REDACTED-PROXY-HOST> returns 000). Images were exported
from the server parquet into ./cross_images/.

One invocation per model. Output uses the SAME uid scheme as the local-model runs
({sample_id}_{setting}_{traj_id}_r{round}) so every model can be joined round-by-round.

  python run_api_vlm_local.py --model gpt-4o --tag gpt4o
"""
import argparse, json, os, random, base64, io, time, threading
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image

RELAY_KEY = "sk-REDACTED"
RELAY_BASE = "https://<REDACTED-PROXY-HOST>/v1"

ap = argparse.ArgumentParser()
ap.add_argument("--fused", default="fused_50_out_v3.jsonl")
ap.add_argument("--image_dir", default="cross_images")
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--base_url", default=RELAY_BASE)
ap.add_argument("--key", default=RELAY_KEY)
ap.add_argument("--include_file", default="cross_model_selection_ids.json")
ap.add_argument("--n_images", type=int, default=12)
ap.add_argument("--first_traj_only", action="store_true", default=True)
ap.add_argument("--out", default=None)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--max_tokens", type=int, default=300)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--image_detail", default="high")
ap.add_argument("--max_side", type=int, default=768)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--no_resume", action="store_true")
args = ap.parse_args()

args.out = args.out or f"cross_model_judge_input_{args.tag}.jsonl"
endpoint = args.base_url.rstrip("/") + "/chat/completions"
LOCK = threading.Lock()

# ---------------- selection (identical to the local-model probes) ----------------
by_sample = {}
for line in open(args.fused, encoding="utf-8"):
    rec = json.loads(line)
    for c in rec["cells"]:
        if c["condition"] != "injection":
            continue
        for tj in c["trajectories"]:
            by_sample.setdefault(rec["sample_id"], []).append((c["setting"], tj))

all_ids = sorted(by_sample)
rng = random.Random(args.seed)
include = []
if args.include_file:
    include = json.load(open(args.include_file, encoding="utf-8"))
    include = sorted({s for s in include if s in by_sample})
remaining = [s for s in all_ids if s not in set(include)]
fill = rng.sample(remaining, max(0, args.n_images - len(include)))
chosen = (include + fill)[:args.n_images]

jobs = []
for sid in chosen:
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
        jobs.append((sid, setting, tj))
n_rounds = sum(len(tj["rounds"]) for _, _, tj in jobs)
print(f"[{args.tag}] images={len(chosen)} trajectories={len(jobs)} rounds={n_rounds}")

done = set()
if (not args.no_resume) and os.path.exists(args.out):
    for l in open(args.out, encoding="utf-8"):
        try:
            done.add(json.loads(l)["uid"])
        except Exception:
            pass
    print(f"[{args.tag}] resume: {len(done)} rows already done")

_img_cache = {}
def b64_image(sid):
    if sid in _img_cache:
        return _img_cache[sid]
    p = os.path.join(args.image_dir, f"{sid}.jpg")
    img = Image.open(p).convert("RGB")
    if max(img.size) > args.max_side:
        s = args.max_side / max(img.size)
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    v = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    _img_cache[sid] = v
    return v


def call_api(messages, retries=5):
    payload = {"model": args.model, "messages": messages,
               "temperature": args.temperature, "max_tokens": args.max_tokens}
    last = None
    for k in range(retries):
        try:
            r = requests.post(endpoint,
                              headers={"Authorization": f"Bearer {args.key}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=180)
            if r.status_code == 200:
                ch = r.json()["choices"][0]["message"]
                return (ch.get("content") or "").strip()
            last = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(2 + 3 * k)
    print(f"   [{args.tag}] api fail: {last}")
    return None


out = open(args.out, "a", encoding="utf-8")
counter = {"n": 0}

def run_one(job):
    sid, setting, tj = job
    data_url = b64_image(sid)
    traj_id = tj["trajectory_id"]
    history = []
    for rd in sorted(tj["rounds"], key=lambda r: r["round"]):
        uid = f"{sid}_{setting}_{traj_id}_r{rd['round']}"
        q = rd["current_question"]
        if not history:
            msg = {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": data_url, "detail": args.image_detail}},
                {"type": "text", "text": q}]}
        else:
            msg = {"role": "user", "content": q}
        msgs = history + [msg]
        if uid in done:
            history = msgs + [{"role": "assistant",
                               "content": rd.get("model_output", "")}]
            continue
        resp = call_api(msgs)
        if resp is None:
            resp = ""
        history = msgs + [{"role": "assistant", "content": resp}]
        rec = {"uid": uid, "current_question": rd["current_question"],
               "ground_truth": rd.get("ground_truth", ""), "model_output": resp,
               "sample_id": sid, "setting": setting, "condition": "injection",
               "trajectory_id": traj_id, "round": rd["round"],
               "category_key": rd.get("category_key", ""), "model": args.model}
        with LOCK:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            counter["n"] += 1
            if counter["n"] % 50 == 0:
                print(f"   [{args.tag}] wrote {counter['n']}", flush=True)


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    list(ex.map(run_one, jobs))

print(f"[{args.tag}] DONE. wrote {counter['n']} new rows -> {args.out}")
out.close()
