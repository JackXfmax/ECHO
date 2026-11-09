"""
Robustness: is the head signature just an answer-length artifact?

Self-healed answers are ~3.9 tokens shorter than persisting ones, and a shorter
answer leaves less of the attention mass on the answer's own tokens, mechanically
raising everything else (including the image). We therefore residualise every
(layer, head) measurement on (n_answer_tokens, seq_len, n_image_tokens) AFTER
removing stratum fixed effects, then redo the delta and the permutation test.

Note: answer length is a post-treatment variable, so this is a robustness check,
not the primary estimate.
"""
import json, random
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for l in open("attn_perlayer_hf.jsonl", encoding="utf-8") if l.strip()]
V = np.array([np.array(r["heads"], dtype=float).reshape(-1) for r in rows])
X = np.array([[r["n_answer_tokens"], r["seq_len"], r["n_image_tokens"]]
              for r in rows], dtype=float)
sid_list = [(r["round"], r["stratum"]) for r in rows]
grp = np.array([1 if r["group"] == "selfheal" else 0 for r in rows])
uniq_s = sorted(set(sid_list), key=str)
sidx = np.array([uniq_s.index(s) for s in sid_list])
by_s = defaultdict(list)
for i, s in enumerate(sidx):
    by_s[s].append(i)


def demean(A, lab):
    B = A.copy()
    for li, idxs in by_s.items():
        idxs = np.asarray(idxs)
        B[idxs] -= B[idxs].mean(0) if lab is None else B[idxs].mean(0)
    return B


# 1. remove stratum fixed effects
Vd = V.copy()
Xd = X.copy()
for li, idxs in by_s.items():
    idxs = np.asarray(idxs)
    Vd[idxs] -= Vd[idxs].mean(0)
    Xd[idxs] -= Xd[idxs].mean(0)
# 2. regress out covariates
beta, *_ = np.linalg.lstsq(Xd, Vd, rcond=None)
R = Vd - Xd @ beta
print("residualised on n_answer_tokens, seq_len, n_image_tokens")


def delta_from(M, labels):
    dc = np.zeros(M.shape[1])
    wsum = 0.0
    for li, idxs in by_s.items():
        idxs = np.asarray(idxs)
        h = idxs[labels[idxs] == 1]
        p = idxs[labels[idxs] == 0]
        if len(h) == 0 or len(p) == 0:
            continue
        w = min(len(h), len(p))
        dc += (M[h].mean(0) - M[p].mean(0)) * w
        wsum += w
    return dc / wsum


for name, M in (("raw", V), ("residualised", R)):
    d = delta_from(M, grp)
    rng = random.Random(3)
    nulls = []
    for _ in range(300):
        lab = grp.copy()
        for li, idxs in by_s.items():
            idxs = np.asarray(idxs)
            v = lab[idxs].copy()
            rng.shuffle(v)
            lab[idxs] = v
        nulls.append(np.abs(delta_from(M, lab)).max())
    thr = float(np.percentile(nulls, 95))
    sig = np.abs(d) >= thr
    print(f"\n[{name}] threshold={thr:.5f}  cells above={int(sig.sum())}/784  "
          f"pos={int((sig&(d>0)).sum())} neg={int((sig&(d<0)).sum())}")
    print(f"   max |delta| = {np.abs(d).max():.5f} @ L{int(np.abs(d).argmax())//28+1} "
          f"H{int(np.abs(d).argmax())%28+1}")
    top = np.argsort(-np.abs(d))[:6]
    print("   top cells:", [(int(i)//28+1, int(i)%28+1, round(float(d[i]), 5)) for i in top])

# how much of the raw signature do the covariates explain?
d_raw = delta_from(V, grp)
d_res = delta_from(R, grp)
print(f"\ncorr(raw delta, residualised delta) over 784 cells = "
      f"{np.corrcoef(d_raw, d_res)[0,1]:.3f}")
print(f"shrinkage of the max cell: {np.abs(d_raw).max():.5f} -> {np.abs(d_res).max():.5f}")
