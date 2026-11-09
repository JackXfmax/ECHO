"""
Permutation test for EXP3.

The Bonferroni-on-bootstrap approach is invalid: with 300 bootstrap draws you cannot
estimate a 0.0032% percentile. Instead we PERMUTE the heal/persist labels INSIDE each
stratum (which preserves the history matching and the stratum sizes exactly) and use
the max-|delta| over all 784 (layer, head) cells as the test statistic. That gives a
correct family-wise p-value for "is there ANY head that moves".

We also permute-test the pre-registered statistic: mean delta over deep layers 20-28.
"""
import json, random
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for l in open("attn_perlayer_hf.jsonl", encoding="utf-8") if l.strip()]
V = np.array([np.array(r["heads"], dtype=float).reshape(-1) for r in rows])   # (N, 784)
Lm = np.array([r["layers"] for r in rows], dtype=float)                       # (N, 28)
sid_list = [(r["round"], r["stratum"]) for r in rows]
tkey = [(r["sample_id"], r["setting"], r["trajectory_id"]) for r in rows]
grp = np.array([1 if r["group"] == "selfheal" else 0 for r in rows])
uniq_s = sorted(set(sid_list), key=str)
sidx = np.array([uniq_s.index(s) for s in sid_list])
S = len(uniq_s)
print(f"N={len(rows)} strata={S} cells={V.shape[1]}")

order_by_s = defaultdict(list)
for i, s in enumerate(sidx):
    order_by_s[s].append(i)


def delta_for_labels(labels):
    """labels: 0/1 array. returns (per-cell delta 784, per-layer delta 28)"""
    dc = np.zeros(V.shape[1])
    dl = np.zeros(28)
    wsum = 0.0
    for s, idxs in order_by_s.items():
        idxs = np.asarray(idxs)
        h = idxs[labels[idxs] == 1]
        p = idxs[labels[idxs] == 0]
        if len(h) == 0 or len(p) == 0:
            continue
        w = min(len(h), len(p))
        dc += (V[h].mean(0) - V[p].mean(0)) * w
        dl += (Lm[h].mean(0) - Lm[p].mean(0)) * w
        wsum += w
    return dc / wsum, dl / wsum


obs_cell, obs_layer = delta_for_labels(grp)
stat_deep = obs_layer[19:].mean()
stat_max = np.abs(obs_cell).max()
print(f"\nOBSERVED  deep-layer(20-28) delta = {stat_deep:+.6f}")
print(f"OBSERVED  max |delta| over 784 (layer,head) cells = {stat_max:.6f} "
      f"@ L{int(np.abs(obs_cell).argmax())//28+1} H{int(np.abs(obs_cell).argmax())%28+1}")

rng = random.Random(7)
NPERM = 500
null_deep = []
null_max = []
for _ in range(NPERM):
    lab = grp.copy()
    for s, idxs in order_by_s.items():
        idxs = np.asarray(idxs)
        v = lab[idxs].copy()
        rng.shuffle(v)
        lab[idxs] = v
    dc, dl = delta_for_labels(lab)
    null_deep.append(dl[19:].mean())
    null_max.append(np.abs(dc).max())

null_deep = np.asarray(null_deep)
null_max = np.asarray(null_max)
p_deep = (np.abs(null_deep) >= abs(stat_deep)).mean()
p_max = (null_max >= stat_max).mean()
print(f"\nPERMUTATION (n={NPERM}, labels shuffled within stratum)")
print(f"  deep-layer delta : null mean={null_deep.mean():+.6f} sd={null_deep.std():.6f} "
      f"  p(two-sided) = {p_deep:.3f}")
print(f"  max |delta|      : null mean={null_max.mean():.6f} sd={null_max.std():.6f} "
      f"  p(family-wise) = {p_max:.3f}")
print(f"  null 95% range of max|delta| = [{np.percentile(null_max,2.5):.6f}, "
      f"{np.percentile(null_max,97.5):.6f}]")
print("\nverdict:", "signal" if p_max < 0.05 else "NO signal anywhere in the attention maps")
