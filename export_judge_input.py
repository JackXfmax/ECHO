"""Export every (trajectory, round) answer as a Judge item.

Unit of analysis = (trajectory, round); one round = one Q/A turn.

Reads the merged v2 file (baseline cells reused from the v1 run + injection cells
re-run with vision_mode=every_round) and writes a flat JSONL with ONLY the three
fields the judge sees, plus metadata kept aside for later aggregation.
"""
import json, os, argparse, hashlib

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=r"<WORKSPACE>\fused_50_out_v3.jsonl")
ap.add_argument("--out", default=r"<WORKSPACE>\judge_input.jsonl")
args = ap.parse_args()

recs = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
n = 0
seen = set()
with open(args.out, "w", encoding="utf-8") as fo:
    for r in recs:
        sid = r["sample_id"]
        for c in r["cells"]:
            for t in c["trajectories"]:
                for rd in t["rounds"]:
                    q = rd["current_question"]
                    gt = rd["ground_truth"]
                    mo = rd["model_output"]
                    uid = hashlib.md5(
                        f"{sid}|{c['setting']}|{c['condition']}|{t['trajectory_id']}|{rd['round']}".encode()
                    ).hexdigest()[:16]
                    assert uid not in seen, uid
                    seen.add(uid)
                    fo.write(json.dumps({
                        "uid": uid,
                        "current_question": q,
                        "ground_truth": gt,
                        "model_output": mo,
                        # metadata: NEVER sent to the judge
                        "sample_id": sid,
                        "setting": c["setting"],
                        "condition": c["condition"],
                        "trajectory_id": t["trajectory_id"],
                        "round": rd["round"],
                        "category_key": rd.get("category_key"),
                        "mode": rd.get("mode"),
                        "hallucination_type": r.get("hallucination_type"),
                        "truth_token": r.get("truth_token"),
                        "false_token": r.get("false_token"),
                    }, ensure_ascii=False) + "\n")
                    n += 1
print("wrote", n, "judge items ->", args.out)
