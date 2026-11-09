"""Run the GLM judge over judge_input.jsonl (concurrent, resumable, cached).

Usage:
  set ZHIPU_API_KEY=...            (Windows)  /  export ZHIPU_API_KEY=...
  python run_glm_judge.py --workers 16 --model glm-4.6-flash

Writes judge_output.jsonl with one line per uid; already-done uids are skipped on
re-run, so it is safe to interrupt and resume.
"""
import json, os, sys, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_prompt import build_judge_messages, parse_verdict

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=r"<WORKSPACE>\judge_input.jsonl")
ap.add_argument("--out", default=r"<WORKSPACE>\judge_output.jsonl")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--model", default="GLM-5.3-Flash")
ap.add_argument("--base", default="https://api.scnet.cn/api/llm/v1/chat/completions")
ap.add_argument("--key", default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--max_tokens", type=int, default=4096,
                help="initial output token budget; escalated on empty-content (reasoning) cuts")
args = ap.parse_args()

KEY = (args.key or os.environ.get("ZHIPU_API_KEY", "")).strip()
if not KEY:
    sys.exit("API key not set: pass --key or set ZHIPU_API_KEY")

import requests
from requests.adapters import HTTPAdapter

BASE = args.base
sess_local = threading.local()

def get_session():
    s = getattr(sess_local, "s", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers))
        sess_local.s = s
    return s

items = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
if args.limit:
    items = items[: args.limit]

done = {}
if os.path.exists(args.out):
    for l in open(args.out, encoding="utf-8"):
        try:
            d = json.loads(l)
            done[d["uid"]] = d
        except Exception:
            pass
todo = [it for it in items if it["uid"] not in done]
print(f"total={len(items)} done={len(done)} todo={len(todo)} model={args.model} workers={args.workers}")

lock = threading.Lock()
fout = open(args.out, "a", encoding="utf-8")
t0 = time.time()
cnt = [0]

def work(it):
    msgs = build_judge_messages(it["current_question"], it["ground_truth"], it["model_output"])
    last_err = None
    # escalate token budget on empty-content (reasoning-model) cuts so they self-heal
    mtries = [args.max_tokens, args.max_tokens * 2, args.max_tokens * 4]
    for attempt in range(6):
        mt = mtries[min(attempt, len(mtries) - 1)]
        try:
            r = get_session().post(
                BASE,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                json={"model": args.model, "messages": msgs, "temperature": 0.0, "max_tokens": mt},
                timeout=120,
            )
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:160]}"
                time.sleep(2 + 3 * attempt)
                continue
            content = (r.json()["choices"][0]["message"].get("content") or "").strip()
            # GLM reasoning models occasionally spend the whole token budget on
            # thinking and emit an empty content -> retry instead of recording error
            if not content:
                last_err = "empty content (likely reasoning budget exhausted)"
                time.sleep(2 + 3 * attempt)
                continue
            v = parse_verdict(content)
            if v["verdict"] == "error":
                # malformed JSON; retry a couple times before giving up
                last_err = f"parse: {v['reason'][:120]}"
                time.sleep(2 + 3 * attempt)
                continue
            return {"uid": it["uid"], "verdict": v["verdict"],
                    "matched_false_premise": v["matched_false_premise"],
                    "reason": v["reason"], "raw": content[:400]}
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            time.sleep(2 + 3 * attempt)
    return {"uid": it["uid"], "verdict": "error", "matched_false_premise": None,
            "reason": f"call failed: {last_err}", "raw": ""}

with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(work, it): it for it in todo}
    for fu in as_completed(futs):
        res = fu.result()
        with lock:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            cnt[0] += 1
            if cnt[0] % 200 == 0:
                el = time.time() - t0
                print(f"  {cnt[0]}/{len(todo)}  {el/60:.1f} min  eta {el/cnt[0]*(len(todo)-cnt[0])/60:.1f} min", flush=True)
fout.close()
print(f"DONE {cnt[0]} new verdicts in {(time.time()-t0)/60:.1f} min -> {args.out}")
