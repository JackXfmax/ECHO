import json
from collections import defaultdict

# Frontier models: injection-only, 12 images x 6 settings x 1 traj x 9 rounds = 648 units
MODELS = {
    "GPT-4o":            "cross_model_judge_output_gpt4o.jsonl",
    "Gemini-2.5-Flash":  "cross_model_judge_output_gemini25flash.jsonl",
    "Claude-Sonnet-4.6": "cross_model_judge_output_claudesonnet46.jsonl",
}

def load_dedup(fn):
    rows = {}
    for l in open(fn, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        rows[r["uid"]] = r  # dedup by uid (resume may duplicate a few)
    return rows

def get_round(uid):
    # uid like "..._r4" at the end
    return int(uid.rsplit("_r", 1)[-1])

def analyze(name, rows):
    by_round = defaultdict(lambda: [0, 0])          # round -> [hallu, total]
    trans = defaultdict(int)                          # "H->F" etc -> count
    paired = 0
    for r in rows.values():
        rd = get_round(r["uid"])
        is_h = (r.get("verdict") == "hallucination")
        by_round[rd][1] += 1
        if is_h:
            by_round[rd][0] += 1
    # recovery: round r hallucination -> round r+1 faithful (within same uid prefix)
    by_uid_round = defaultdict(dict)
    for r in rows.values():
        uid = r["uid"].rsplit("_r", 1)[0]  # strip _rN
        by_uid_round[uid][get_round(r["uid"])] = (r.get("verdict") == "hallucination")
    for uid, d in by_uid_round.items():
        for rd in sorted(d):
            if rd + 1 in d:
                paired += 1
                a, b = d[rd], d[rd + 1]
                trans[f"{'H' if a else 'F'}->{ 'H' if b else 'F'}"] += 1
    print(f"\n=== {name}  (n_units={len(rows)}, recovery_pairs={paired}) ===")
    curve = []
    for rd in sorted(by_round):
        h, t = by_round[rd]
        rate = h / t if t else 0
        curve.append(rate)
        print(f"  r{rd}: {h:3d}/{t:3d}  halluc_rate={rate:.3f}")
    oh = sum(v[0] for v in by_round.values())
    ot = sum(v[1] for v in by_round.values())
    print(f"  OVERALL injection halluc_rate = {oh}/{ot} = {oh/ot:.3f}")
    hh = trans.get("H->H", 0); hf = trans.get("H->F", 0)
    fh = trans.get("F->H", 0); ff = trans.get("F->F", 0)
    recov = hf / (hh + hf) if (hh + hf) else 0
    print(f"  recovery H->F / (H->H+H->F) = {hf}/{hh+hf} = {recov:.3f}  (self-heal rate)")
    return curve, oh / ot, recov

results = {}
for name, fn in MODELS.items():
    rows = load_dedup(fn)
    results[name] = analyze(name, rows)

print("\n=== 横向对比（injection-only 逐轮幻觉率）===")
print("round :  " + "  ".join(f"r{rd}" for rd in range(1, 10)))
for name, (curve, _, _) in results.items():
    print(f"{name:18s}: " + "  ".join(f"{v:.2f}" for v in curve))
print("\n(对照 EXP1 Qwen2.5-VL-7B injection 逐轮: 0.92 0.98 0.57 0.64 0.50 0.83 0.66 0.63 0.48)")
print("引用口径: frontier = EXP2c/d/e (API, 12 images, 648 units, DeepSeek-V4.1 judge)")
