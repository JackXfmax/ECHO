"""
analyze_cross_model.py -- local, post-judge.

Reads the LLaVA (or any 2nd-model) probe outputs and produces the per-round
oscillation curve + self-recovery transition, then places them side-by-side with
Qwen's reference numbers from result_judge_analysis.md. This is the defensive
"universality" evidence for reviewers.

Usage:
  python analyze_cross_model.py \
      --jin cross_model_judge_input.jsonl \
      --jout cross_model_judge_output.jsonl \
      --out cross_model_analysis.md
"""
import argparse, json, collections, random
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--jin", required=True)
ap.add_argument("--jout", required=True)
ap.add_argument("--out", default="cross_model_analysis.md")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--b", type=int, default=2000)
args = ap.parse_args()
rng = random.Random(args.seed)

# Qwen reference (single DeepSeek judge, result_judge_analysis.md / fig_make.py)
QWEN_PER_ROUND = [0.923, 0.979, 0.570, 0.639, 0.500, 0.826, 0.659, 0.631, 0.479]
QWEN_RECOVERY = 0.280

meta = {}
for l in open(args.jin, encoding="utf-8"):
    d = json.loads(l); meta[d["uid"]] = d
verd = {}
for l in open(args.jout, encoding="utf-8"):
    d = json.loads(l)
    if d["verdict"] != "error":
        verd[d["uid"]] = d

rows = []
for uid, m in meta.items():
    v = verd.get(uid)
    if v is None:
        continue
    rows.append({"sample_id": m["sample_id"], "round": m["round"],
                 "hallu": 1 if v["verdict"] == "hallucination" else 0})

samples = sorted({r["sample_id"] for r in rows})


def boot_mean(getter):
    base = [getter(s) for s in samples]
    base = [b for b in base if b is not None]
    if not base:
        return (float("nan"), float("nan"), float("nan"))
    pt = float(np.mean(base))
    vals = np.empty(args.b)
    for bi in range(args.b):
        idx = [rng.randrange(len(samples)) for _ in samples]
        vals[bi] = np.mean([getter(samples[i]) for i in idx])
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# per-round rate
round_rates = {}
for r in range(1, 10):
    def g(s, r=r):
        v = [x["hallu"] for x in rows if x["sample_id"] == s and x["round"] == r]
        return sum(v) / len(v) if v else None
    round_rates[r] = boot_mean(g)

# 2x2 transition on consecutive rounds within each (sample,setting,traj)
seq = collections.defaultdict(list)
for l in open(args.jin, encoding="utf-8"):
    d = json.loads(l)
    if d["uid"] in verd:
        seq[(d["sample_id"], d["setting"], d["trajectory_id"])].append(
            (d["round"], 1 if verd[d["uid"]]["verdict"] == "hallucination" else 0))
trans = {(1, 1): 0, (1, 0): 0, (0, 1): 0, (0, 0): 0}
for k, s in seq.items():
    s = sorted(s, key=lambda x: x[0])
    for a, b in zip(s, s[1:]):
        trans[(a[1], b[1])] += 1
HH, HF, FH, FF = trans[(1, 1)], trans[(1, 0)], trans[(0, 1)], trans[(0, 0)]
recovery = HF / (HH + HF) if (HH + HF) else float("nan")

# ---- write report ----
L = []
L.append(f"# Cross-Model Validation: LLaVA-1.5-7B vs Qwen2.5-VL\n")
L.append(f"rows judged: {len(rows)}   samples: {len(samples)}   errors skipped: "
         f"{len(meta) - len(rows)}\n")
L.append("## 1. Per-round hallucination rate (injection, rounds 1-9)\n")
L.append("| round | LLaVA (this run) | Qwen (reference) |")
L.append("|---|---|---|")
cat = {1: "entity_missing", 2: "entity_missing", 3: "attribute_reversal",
       4: "attribute_reversal", 5: "background_fabrication", 6: "background_fabrication",
       7: "context_pollution", 8: "context_pollution", 9: "dataset_final"}
for r in range(1, 10):
    pt, lo, hi = round_rates[r]
    q = QWEN_PER_ROUND[r - 1]
    L.append(f"| {r} ({cat[r]}) | {pt:.3f} [{lo:.3f},{hi:.3f}] | {q:.3f} |")
L.append("")
L.append("## 2. Self-recovery transition (per consecutive round)\n")
L.append(f"HH={HH}  HF={HF}  FH={FH}  FF={FF}   (H=hallucination, F=faithful)")
L.append(f"\n**Self-recovery rate H->F = {HF}/{HH+HF} = {recovery:.1%}**")
L.append(f"\nQwen reference self-recovery rate = {QWEN_RECOVERY:.1%}\n")
L.append("## 3. Interpretation\n")
L.append("- If LLaVA's per-round curve ALSO shows non-monotonic oscillation "
         "(e.g. a spike at the entity-missing rounds 1-2 and a dip/recovery later), "
         "the snowball+self-recovery phenomenon is architectural, not a Qwen quirk.")
L.append("- The absolute rates differ across architectures (expected); the "
         "STRUCTURAL pattern (oscillation + partial self-recovery) is the claim.")
L.append("- Note: LLaVA here runs only the first trajectory per (sample,setting) "
         "to bound compute; Qwen's reference aggregates all trajectories.")
open(args.out, "w", encoding="utf-8").write("\n".join(L))
print("saved", args.out)
print("LLaVA per-round:", {r: round(round_rates[r][0], 3) for r in range(1, 10)})
print(f"LLaVA recovery H->F = {recovery:.1%}  (Qwen {QWEN_RECOVERY:.1%})")
