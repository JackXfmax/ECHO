"""
analyze_decompose.py -- EXP3b: if not the image, then what?

Reads attn_decompose_hf.jsonl and tests, under the SAME stratified matching, whether
self-healing shows up as a change in where the answer's attention goes:

    img_a        -> image tokens                       (visual grounding)
    hist_asst_a  -> the model's OWN earlier answers     (the snowball carrier)
    hist_user_a  -> earlier user turns (injected premises) + prefix
    cur_q_a      -> the current question
    self_a_a     -> the answer's own earlier tokens
    grounding    = img_a / (img_a + hist_asst_a)        (image vs own history)

query sets: _a = answer tokens, _q = current-question tokens, _a8 = first 8 answer tokens
"""
import json, os, random
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = "attn_decompose_hf.jsonl"
rows = [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()]
print(f"records {len(rows)}")

METRICS = [f"{k}_{q}" for k in ("img", "hist_asst", "hist_user", "cur_q", "self_a")
           for q in ("a", "q", "a8")]


def gname(g):
    return "heal" if g == "selfheal" else g


def strat():
    st = defaultdict(lambda: {"heal": [], "persist": []})
    for r in rows:
        st[(r["round"], r["stratum"])][gname(r["group"])].append(
            ((r["sample_id"], r["setting"], r["trajectory_id"]), r))
    return st


def est(st, key):
    ds, ws, hs, ps = [], [], [], []
    for k, d in st.items():
        nh, npp = len(d["heal"]), len(d["persist"])
        if nh == 0 or npp == 0:
            continue
        h = np.mean([x[1][key] for x in d["heal"]], 0)
        p = np.mean([x[1][key] for x in d["persist"]], 0)
        w = min(nh, npp)
        hs.append(h); ps.append(p); ds.append(h - p); ws.append(w)
    if not hs:
        return None
    ws = np.asarray(ws, float)
    f = lambda a: (np.asarray(a) * ws[:, None]).sum(0) / ws.sum()
    return f(hs), f(ps), f(ds), len(hs), int(ws.sum())


def boot(st, key, nboot=600, seed=2):
    flat = []
    for k, d in st.items():
        for g in ("heal", "persist"):
            for tk, rec in d[g]:
                flat.append((tk, g, k, rec[key]))
    idx = defaultdict(list)
    for i, f in enumerate(flat):
        idx[f[0]].append(i)
    trajs = list(idx.keys())
    rng = random.Random(seed)
    out = []
    for _ in range(nboot):
        cnt = defaultdict(int)
        for _t in range(len(trajs)):
            cnt[rng.choice(trajs)] += 1
        acc = defaultdict(lambda: {"heal": [], "persist": []})
        for t, c in cnt.items():
            for i in idx[t]:
                tk, g, k, arr = flat[i]
                for _ in range(c):
                    acc[k][g].append(arr)
        ds, ws = [], []
        for k, d in acc.items():
            if d["heal"] and d["persist"]:
                ds.append(np.mean(d["heal"], 0) - np.mean(d["persist"], 0))
                ws.append(min(len(d["heal"]), len(d["persist"])))
        if not ds:
            continue
        ds = np.asarray(ds); ws = np.asarray(ws, float)
        out.append((ds * ws[:, None]).sum(0) / ws.sum())
    a = np.asarray(out)
    return np.percentile(a, 2.5, 0), np.percentile(a, 97.5, 0)


# derived grounding index
for r in rows:
    r["grounding_a"] = [i / (i + h + 1e-9) for i, h in zip(r["img_a"], r["hist_asst_a"])]
    r["grounding_q"] = [i / (i + h + 1e-9) for i, h in zip(r["img_q"], r["hist_asst_q"])]

st = strat()
print(f"strata with both arms: {sum(1 for k,d in st.items() if d['heal'] and d['persist'])}")

print(f"\n{'metric':<16} {'deep(20-28) heal':>16} {'persist':>10} {'delta':>10} {'95% CI':>22} {'rel':>8}")
summary = {}
for m in METRICS + ["grounding_a", "grounding_q"]:
    res = est(st, m)
    if res is None:
        continue
    h, p, d, ns, npz = res
    lo, hi = boot(st, m, 400)
    dl = d[19:].mean()
    base = p[19:].mean()
    print(f"{m:<16} {h[19:].mean():>16.5f} {p[19:].mean():>10.5f} {dl:>+10.5f} "
          f"  [{lo[19:].mean():+.5f},{hi[19:].mean():+.5f}] {dl/max(base,1e-9):>+8.2%}")
    summary[m] = {"heal": h.tolist(), "persist": p.tolist(), "delta": d.tolist(),
                  "ci": [lo.tolist(), hi.tolist()], "deep_delta": float(dl)}

json.dump(summary, open("exp3_decompose_numbers.json", "w"), indent=2)

# ---- figure: the two candidate mechanisms side by side -----------------------
os.makedirs("figs", exist_ok=True)
xs = np.arange(1, 29)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), dpi=180)
for ax, m, ttl in zip(axes, ("img_a", "hist_asst_a"),
                      ("(a) attention to the IMAGE", "(b) attention to OWN earlier answers")):
    res = est(st, m)
    h, p, d, ns, npz = res
    lo, hi = boot(st, m, 400)
    ax.plot(xs, p, "-o", ms=3, lw=2, color="#c0392b", label="persist")
    ax.fill_between(xs, lo, hi, color="#888", alpha=.0)
    ax.plot(xs, h, "-o", ms=3, lw=2, color="#1f77b4", label="self-heal")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("attention mass")
    ax.set_title(f"{ttl}\n$\\Delta$deep = {d[19:].mean():+.5f}  "
                 f"[{lo[19:].mean():+.5f}, {hi[19:].mean():+.5f}]")
    ax.grid(alpha=.25, ls=":")
    ax.legend(frameon=False, fontsize=9)
fig.suptitle("EXP3b — where the answer's attention goes (history-matched, deep layers 20-28)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figs/figD_mechanism_decompose.png")
plt.close(fig)
print("wrote figs/figD_mechanism_decompose.png")
