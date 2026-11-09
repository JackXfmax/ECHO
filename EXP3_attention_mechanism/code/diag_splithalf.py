"""Split-half validation of the localized (layer, head) attention signature.

The family-wise permutation test says SOME (layer, head) cell moves more than chance.
Before believing it, check that the same cells move in two INDEPENDENT halves of the
trajectories. A noise artifact will not replicate.
"""
import json, random
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for l in open("attn_perlayer_hf.jsonl", encoding="utf-8") if l.strip()]
V = np.array([np.array(r["heads"], dtype=float).reshape(-1) for r in rows])
sid_list = [(r["round"], r["stratum"]) for r in rows]
tkey = [(r["sample_id"], r["setting"], r["trajectory_id"]) for r in rows]
grp = np.array([1 if r["group"] == "selfheal" else 0 for r in rows])
uniq_s = sorted(set(sid_list), key=str)
sidx = np.array([uniq_s.index(s) for s in sid_list])

by_s = defaultdict(list)
for i, s in enumerate(sidx):
    by_s[s].append(i)


def delta(mask):
    dc = np.zeros(V.shape[1])
    wsum = 0.0
    for s, idxs in by_s.items():
        idxs = np.asarray(idxs)
        idxs = idxs[mask[idxs]]
        h = idxs[grp[idxs] == 1]
        p = idxs[grp[idxs] == 0]
        if len(h) == 0 or len(p) == 0:
            continue
        w = min(len(h), len(p))
        dc += (V[h].mean(0) - V[p].mean(0)) * w
        wsum += w
    return dc / wsum


trajs = sorted(set(tkey))
rng = random.Random(11)

agree = []
corrs = []
for rep in range(5):
    rng.shuffle(trajs)
    half = set(trajs[:len(trajs) // 2])
    mA = np.array([t in half for t in tkey])
    mB = ~mA
    dA, dB = delta(mA), delta(mB)
    corr = np.corrcoef(dA, dB)[0, 1]
    corrs.append(corr)
    for K in (10, 25, 50):
        topA = set(np.argsort(-np.abs(dA))[:K].tolist())
        topB = set(np.argsort(-np.abs(dB))[:K].tolist())
        agree.append((K, len(topA & topB)))
    if rep == 0:
        # sign agreement on top cells
        topA = np.argsort(-np.abs(dA))[:50]
        sign_ok = np.mean(np.sign(dA[topA]) == np.sign(dB[topA]))
        print(f"rep0: corr of per-cell delta between halves = {corr:.3f}")
        print(f"rep0: sign agreement on top-50 cells = {sign_ok:.1%}")
        print(f"rep0: top-10 cells A: {[(int(i)//28+1, int(i)%28+1, round(float(dA[i]),5)) for i in np.argsort(-np.abs(dA))[:10]]}")
        print(f"rep0: same cells in B: {[round(float(dB[i]),5) for i in np.argsort(-np.abs(dA))[:10]]}")

print(f"\ncorrelation across 5 random splits: mean={np.mean(corrs):.3f} "
      f"min={min(corrs):.3f} max={max(corrs):.3f}")
print("top-K overlap (chance overlap for K=10 is 10*10/784=0.13, i.e. ~1.3 cells):")
for K in (10, 25, 50):
    v = [n for k, n in agree if k == K]
    print(f"  K={K:>3}: mean overlap {np.mean(v):.1f} / {K}   "
          f"(chance {K*K/784:.1f})")
