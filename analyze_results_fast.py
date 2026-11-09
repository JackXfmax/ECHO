"""Analyse judged results for the 6-setting x 2-condition fused experiment.

OPTIMIZED version: precomputes per-cell value lists so the cluster bootstrap
is O(1) per lookup instead of O(n_rows) per call. Output is identical to
analyze_results.py.

Unit of analysis: (trajectory, round). One round = one Q/A turn.
  * hallucination rate = hallucination / (hallucination + faithful)
  * CLUSTER bootstrap over sample_id (50 images), B=2000, percentile 95% CI
  * paired contrasts use the SAME resampled image set for both arms
"""
import json, argparse, collections, random
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--jin", default=r"<WORKSPACE>\judge_input.jsonl")
ap.add_argument("--jout", default=r"<WORKSPACE>\judge_output.jsonl")
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

meta = {}
order = []
for l in open(args.jin, encoding="utf-8"):
    d = json.loads(l)
    meta[d["uid"]] = d
    order.append(d["uid"])

verd = {}
for l in open(args.jout, encoding="utf-8"):
    d = json.loads(l)
    verd[d["uid"]] = d

rows = []
nerr = 0
for uid in order:
    m = meta[uid]
    v = verd.get(uid)
    if v is None:
        continue
    if v["verdict"] == "error":
        nerr += 1
        continue
    rows.append({
        "sample_id": m["sample_id"], "setting": m["setting"], "condition": m["condition"],
        "traj": m["trajectory_id"], "round": m["round"], "cat": m["category_key"],
        "hallu": 1 if v["verdict"] == "hallucination" else 0,
        "mfp": v.get("matched_false_premise"),
    })
print(f"judged rows: {len(rows)}   errors: {nerr} ({nerr/max(1,len(rows)+nerr):.1%})")
if not rows:
    raise SystemExit("no judged rows")

SETTINGS = ["cleanconv_formatting", "cleanconv_question", "factconv_formatting",
            "halluconv_question", "halluconv_formatting", "irrconv_formatting"]
CONDS = ["baseline", "injection"]

# ---- precompute O(1) lookups ----
cell_lists = collections.defaultdict(list)        # (sid, setting, cond) -> [hallu]
cell_round_lists = collections.defaultdict(list)  # (sid, setting, cond, round) -> [hallu]
cat_lists = collections.defaultdict(list)         # (sid, cat) -> [hallu]
seq_map = collections.defaultdict(list)           # (sid, setting, traj) -> [(round, hallu)]
for r in rows:
    cell_lists[(r["sample_id"], r["setting"], r["condition"])].append(r["hallu"])
    cell_round_lists[(r["sample_id"], r["setting"], r["condition"], r["round"])].append(r["hallu"])
    cat_lists[(r["sample_id"], r["cat"])].append(r["hallu"])
    seq_map[(r["sample_id"], r["setting"], r["traj"])].append((r["round"], r["hallu"]))

def cell_value(sid, setting, cond, rounds=None):
    if rounds is None:
        vals = cell_lists.get((sid, setting, cond))
    else:
        vals = []
        for rd in rounds:
            v = cell_round_lists.get((sid, setting, cond, rd))
            if v:
                vals.extend(v)
    return sum(vals) / len(vals) if vals else None

def cat_value(sid, cat):
    vals = cat_lists.get((sid, cat))
    return sum(vals) / len(vals) if vals else None

samples = sorted({r["sample_id"] for r in rows})
rng = random.Random(args.seed)

def boot_mean(getter, B=args.B):
    """Cluster bootstrap over samples; getter(sample_id) -> value or None."""
    base = [getter(s) for s in samples]
    base = [b for b in base if b is not None]
    if not base:
        return (float("nan"), float("nan"), float("nan"))
    pt = float(np.mean(base))
    idxs = [[rng.randrange(len(samples)) for _ in samples] for _ in range(B)]
    vals = np.empty(B)
    for bi, idx in enumerate(idxs):
        vals[bi] = np.mean([getter(samples[i]) for i in idx])
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

print("\n================ 1. Hallucination rate by setting x condition (ALL rounds) ================")
print(f"{'setting':<22}{'baseline':>16}{'injection':>16}{'delta':>16}")
res_all = {}
for s in SETTINGS:
    out = {}
    for c in CONDS:
        out[c] = boot_mean(lambda sid, s=s, c=c: cell_value(sid, s, c))
    res_all[s] = out
    print(f"{s:<22}{out['baseline'][0]:>8.3f}[{out['baseline'][1]:.3f},{out['baseline'][2]:.3f}]"
          f"{out['injection'][0]:>8.3f}[{out['injection'][1]:.3f},{out['injection'][2]:.3f}]"
          f"{out['injection'][0]-out['baseline'][0]:>16.3f}")

print("\n========= 2. FINAL round (dataset core question): baseline r1 vs injection r9 =============")
print(f"{'setting':<22}{'baseline r1':>16}{'injection r9':>16}{'delta (paired CI)':>26}")
res_fin = {}
for s in SETTINGS:
    b = boot_mean(lambda sid, s=s: cell_value(sid, s, "baseline", rounds={1}))
    i = boot_mean(lambda sid, s=s: cell_value(sid, s, "injection", rounds={9}))
    def paired(B=args.B):
        d = []
        for _ in range(B):
            idx = [rng.randrange(len(samples)) for _ in samples]
            dv = []
            for i2 in idx:
                sid = samples[i2]
                a = cell_value(sid, s, "baseline", rounds={1})
                c = cell_value(sid, s, "injection", rounds={9})
                if a is not None and c is not None:
                    dv.append(c - a)
            if dv:
                d.append(np.mean(dv))
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    lo, hi = paired(400)
    res_fin[s] = (b, i, (i[0] - b[0], lo, hi))
    print(f"{s:<22}{b[0]:>8.3f}[{b[1]:.3f},{b[2]:.3f}]{i[0]:>8.3f}[{i[1]:.3f},{i[2]:.3f}]"
          f"{i[0]-b[0]:>12.3f} [{lo:.3f},{hi:.3f}]")

print("\n=============== 3. Injection collapse curve: hallucination rate by round 1..9 ==============")
for r in range(1, 10):
    def getter(sid, r=r):
        vals = [cell_round_lists[(sid, s, "injection", r)]
                for s in SETTINGS if (sid, s, "injection", r) in cell_round_lists]
        if not vals:
            return None
        flat = [x for sub in vals for x in sub]
        return sum(flat) / len(flat) if flat else None
    pt, lo, hi = boot_mean(getter)
    cats = collections.Counter(x["cat"] for x in rows if x["condition"] == "injection" and x["round"] == r)
    cat = list(cats)[0] if cats else "?"
    print(f"  round {r} ({cat:<22}): {pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\n===== 4. Interaction: does a hallucinated prefix (halluconv) amplify injection damage? =====")
def delta_for(setting, B=400):
    d = []
    for _ in range(B):
        idx = [rng.randrange(len(samples)) for _ in samples]
        dv = []
        for i2 in idx:
            sid = samples[i2]
            a = cell_value(sid, setting, "baseline", rounds={1})
            c = cell_value(sid, setting, "injection", rounds={9})
            if a is not None and c is not None:
                dv.append(c - a)
        if dv:
            d.append(np.mean(dv))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

hallu_settings = [s for s in SETTINGS if s.startswith("halluconv")]
clean_settings = [s for s in SETTINGS if s.startswith("cleanconv")]
for s in SETTINGS:
    m, lo, hi = delta_for(s)
    tag = "HALLU-prefix" if s in hallu_settings else ("clean-prefix" if s in clean_settings else "other")
    print(f"  {s:<22} {tag:<13} delta={m:+.3f} [{lo:+.3f},{hi:+.3f}]")

def group_delta(group, B=400):
    d = []
    for _ in range(B):
        idx = [rng.randrange(len(samples)) for _ in samples]
        dv = []
        for i2 in idx:
            sid = samples[i2]
            vals = []
            for s in group:
                a = cell_value(sid, s, "baseline", rounds={1})
                c = cell_value(sid, s, "injection", rounds={9})
                if a is not None and c is not None:
                    vals.append(c - a)
            if vals:
                dv.append(np.mean(vals))
        if dv:
            d.append(np.mean(dv))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

mh, loh, hih = group_delta(hallu_settings)
mc, loc, hic = group_delta(clean_settings)
print(f"\n  halluconv prefix mean delta = {mh:+.3f} [{loh:+.3f},{hih:+.3f}]")
print(f"  cleanconv prefix mean delta = {mc:+.3f} [{loc:+.3f},{hic:+.3f}]")
print(f"  INTERACTION (hallu - clean) = {mh-mc:+.3f}  (CI from separate bootstraps; "
      f"overlap of [{loh:.3f},{hih:.3f}] and [{loc:.3f},{hic:.3f}] is the conservative read)")

print("\n=================== 5. Self-recovery inside injected trajectories ===================")
inj_seq = collections.defaultdict(list)
for r in rows:
    if r["condition"] == "injection":
        inj_seq[(r["sample_id"], r["setting"], r["traj"])].append((r["round"], r["hallu"]))
tot_count = 0; rec_count = 0
for key, seq in inj_seq.items():
    seq = sorted(seq, key=lambda x: x[0])
    for a, b in zip(seq, seq[1:]):
        if a[1] == 1:
            tot_count += 1
            if b[1] == 0:
                rec_count += 1
print(f"  rounds following a hallucination: {tot_count}   recovered next round: {rec_count} "
      f"({rec_count/max(1,tot_count):.1%})")

print("\n=================== 6. By injection category (rounds 1-8) ===================")
for cat in ["entity_missing", "attribute_reversal", "background_fabrication", "context_pollution"]:
    def getter(sid, cat=cat):
        return cat_value(sid, cat)
    pt, lo, hi = boot_mean(getter)
    n = sum(1 for x in rows if x["cat"] == cat)
    print(f"  {cat:<24} n={n:<5} rate={pt:.3f} [{lo:.3f},{hi:.3f}]")
