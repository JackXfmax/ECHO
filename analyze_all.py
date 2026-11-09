import json
from collections import defaultdict

def load(inp, out, condition_filter=None):
    inmap = {}
    for l in open(inp, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if condition_filter and r.get("condition") != condition_filter:
            continue
        inmap[r["uid"]] = (r.get("round"), r.get("category_key"))
    outmap = {}
    for l in open(out, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if r.get("verdict") in ("faithful", "hallucination"):
            outmap[r["uid"]] = r["verdict"]
    data = []
    for uid, (rd, cat) in inmap.items():
        if uid in outmap:
            data.append((uid, rd, cat, outmap[uid]))
    return data

def analyze(name, data):
    by_round = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    traj = defaultdict(dict)
    for uid, rd, cat, verd in data:
        is_h = (verd == "hallucination")
        by_round[rd][1] += 1
        if is_h:
            by_round[rd][0] += 1
        if cat:
            by_cat[cat][1] += 1
            if is_h:
                by_cat[cat][0] += 1
        traj[uid.rsplit("_r", 1)[0]][rd] = is_h
    curve = [by_round[rd][0] / by_round[rd][1] if by_round[rd][1] else 0 for rd in range(1, 10)]
    hh = hf = 0
    for t, d in traj.items():
        for rd in sorted(d):
            if rd + 1 in d:
                a, b = d[rd], d[rd + 1]
                if a and not b:
                    hf += 1
                elif a and b:
                    hh += 1
    recov = hf / (hh + hf) if (hh + hf) else 0
    oh = sum(v[0] for v in by_round.values()); ot = sum(v[1] for v in by_round.values())
    print(f"\n=== {name}  (n={len(data)}) ===")
    for rd in range(1, 10):
        h, t = by_round[rd]
        print(f"  r{rd}: {h:4d}/{t:4d} = {h/t:.3f}")
    print(f"  OVERALL injection = {oh/ot:.3f}   recovery(H->F) = {recov:.3f}")
    return curve, oh/ot, recov

models = [
    ("Qwen2.5-VL-7B (EXP1)", "EXP1_main_qwen_9000/data/judge_input.jsonl",
     "EXP1_main_qwen_9000/data/judge_output.jsonl", "injection"),
    ("LLaVA-1.5-7B (EXP2a)", "cross_model_judge_input_llava.jsonl",
     "cross_model_judge_output_llava.jsonl", None),
    ("InternVL2-4B (EXP2b)", "cross_model_judge_input_internvl.jsonl",
     "cross_model_judge_output_internvl.jsonl", None),
    ("GPT-4o (EXP2c)", "cross_model_judge_input_gpt4o.jsonl",
     "cross_model_judge_output_gpt4o.jsonl", None),
    ("Gemini-2.5-Flash (EXP2d)", "cross_model_judge_input_gemini25flash.jsonl",
     "cross_model_judge_output_gemini25flash.jsonl", None),
    ("Claude-Sonnet-4.6 (EXP2e)", "cross_model_judge_input_claudesonnet46.jsonl",
     "cross_model_judge_output_claudesonnet46.jsonl", None),
]
results = {}
for name, inp, out, cf in models:
    data = load(inp, out, cf)
    results[name] = analyze(name, data)

print("\n================ 逐轮幻觉率对比 (injection-only, round 1..9) ================")
print(f"{'model':28s} " + " ".join(f"r{rd}" for rd in range(1, 10)) + "  OVERALL  RECOV")
for name, (curve, ov, rec) in results.items():
    print(f"{name:28s} " + " ".join(f"{v:.2f}" for v in curve) + f"   {ov:.2f}    {rec:.2f}")

print("\n(口径: 全部 injection-only, 同裁判 DeepSeek-V4.1-Flash-1017-Event; EXP1=n=8100, 其余 n=648)")
print("EXP1 权威逐轮参照: 0.92 0.98 0.57 0.64 0.50 0.83 0.66 0.63 0.48  self-heal 0.28")
