"""
Figure D -- the localized attention signature of self-healing.

Layer-averaged visual attention shows nothing (Delta = +0.0007, permutation p = 0.12)
because the effect is concentrated in a handful of (layer, head) cells with OPPOSITE
signs that cancel when you average over heads.

Here we plot the full 28 x 28 map of the history-matched Delta of visual attention
(self-heal - persist) and mark the cells that exceed the permutation 95% max-|delta|
threshold (family-wise 5%, calibrated by diag_permutation.py).
"""
import json, random
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open("attn_perlayer_hf.jsonl", encoding="utf-8") if l.strip()]
V = np.array([np.array(r["heads"], dtype=float).reshape(-1) for r in rows])
sid_list = [(r["round"], r["stratum"]) for r in rows]
grp = np.array([1 if r["group"] == "selfheal" else 0 for r in rows])
uniq_s = sorted(set(sid_list), key=str)
sidx = np.array([uniq_s.index(s) for s in sid_list])
by_s = defaultdict(list)
for i, s in enumerate(sidx):
    by_s[s].append(i)


def delta_vec():
    dc = np.zeros(V.shape[1])
    wsum = 0.0
    for s, idxs in by_s.items():
        idxs = np.asarray(idxs)
        h = idxs[grp[idxs] == 1]
        p = idxs[grp[idxs] == 0]
        if len(h) == 0 or len(p) == 0:
            continue
        w = min(len(h), len(p))
        dc += (V[h].mean(0) - V[p].mean(0)) * w
        wsum += w
    return dc / wsum


d = delta_vec().reshape(28, 28)

# permutation threshold (recomputed here so the figure is self-contained)
rng = random.Random(7)
nulls = []
for _ in range(300):
    lab = grp.copy()
    for s, idxs in by_s.items():
        idxs = np.asarray(idxs)
        v = lab[idxs].copy()
        rng.shuffle(v)
        lab[idxs] = v
    dc = np.zeros(V.shape[1])
    wsum = 0.0
    for s, idxs in by_s.items():
        idxs = np.asarray(idxs)
        h = idxs[lab[idxs] == 1]
        p = idxs[lab[idxs] == 0]
        if len(h) == 0 or len(p) == 0:
            continue
        w = min(len(h), len(p))
        dc += (V[h].mean(0) - V[p].mean(0)) * w
        wsum += w
    nulls.append(np.abs(dc / wsum).max())
thr = float(np.percentile(nulls, 95))
print(f"permutation 95% max-|delta| threshold = {thr:.5f}")
sig = np.abs(d) >= thr
print(f"cells above threshold: {int(sig.sum())} / 784")
print(f"  positive (self-heal > persist): {int((sig & (d > 0)).sum())}")
print(f"  negative (self-heal < persist): {int((sig & (d < 0)).sum())}")
print(f"  mean |delta| over significant cells = {np.abs(d[sig]).mean():.5f}")
li = np.where(sig)
print("  layers involved:", sorted(set((li[0] + 1).tolist())))
print("  top positive cells:", [(int(a) + 1, int(b) + 1, round(float(d[a, b]), 5))
                                for a, b in sorted(zip(*np.where(sig & (d > 0)), ),
                                                   key=lambda t: -abs(d[t[0], t[1]]))[:10]])
print("  top negative cells:", [(int(a) + 1, int(b) + 1, round(float(d[a, b]), 5))
                                for a, b in sorted(zip(*np.where(sig & (d < 0)), ),
                                                   key=lambda t: -abs(d[t[0], t[1]]))[:10]])

fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=180)
lim = float(np.percentile(np.abs(d), 99))
im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", origin="lower")
ax.set_xlabel("attention head")
ax.set_ylabel("transformer layer")
ax.set_title("$\\Delta$ visual attention (self-heal $-$ persist), history-matched\n"
             f"outlined = exceeds permutation 95% threshold ($\\pm${thr:.4f}); "
             f"{int(sig.sum())}/784 cells")
for a, b in zip(*np.where(sig)):
    ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False,
                               edgecolor="k", lw=.8))
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
cb.set_label("$\\Delta$ attention mass on image tokens")
fig.tight_layout()
fig.savefig("figs/figD_head_signature.png")
plt.close(fig)
print("wrote figs/figD_head_signature.png")

json.dump({"threshold": thr, "delta": d.tolist(), "sig": sig.tolist()},
          open("exp3_head_signature.json", "w"), indent=2)
