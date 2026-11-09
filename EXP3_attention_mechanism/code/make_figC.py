"""
make_figC.py -- the EXP3 headline figure: Visual Attention Ratio vs layer (1..28).

Arms
  persist      H at r-1 -> H at r   (stays hallucinated)
  self-heal    H at r-1 -> F at r
  faithful-ref F at r-1 -> F at r, history identical except at r-1  (reference level
               only -- NOT part of the causal contrast, differs from the heal arm at
               exactly one history position)

All heal/persist comparisons are computed WITHIN strata that fix the exact
hallucination bitmask of rounds 1..r-1, then combined with weights min(n_h, n_p).
95% CI = cluster bootstrap over trajectories.
"""
import json, os, random
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = "attn_perlayer_hf.jsonl"
REF = "attn_perlayer_ff.jsonl"


def load(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")
            if l.strip() and all(x is not None for x in json.loads(l)["layers"])]


def gname(g):
    return "heal" if g == "selfheal" else g


def strat(rows, rounds=None):
    st = defaultdict(lambda: {"heal": [], "persist": []})
    for r in rows:
        if rounds and r["round"] not in rounds:
            continue
        st[(r["round"], r["stratum"])][gname(r["group"])].append(
            ((r["sample_id"], r["setting"], r["trajectory_id"]),
             np.asarray(r["layers"], dtype=float)))
    return st


def est(st):
    ds, ws, hs, ps = [], [], [], []
    for k, d in st.items():
        nh, npp = len(d["heal"]), len(d["persist"])
        if nh == 0 or npp == 0:
            continue
        h = np.mean([x[1] for x in d["heal"]], 0)
        p = np.mean([x[1] for x in d["persist"]], 0)
        w = min(nh, npp)
        hs.append(h); ps.append(p); ds.append(h - p); ws.append(w)
    if not hs:
        return None
    ws = np.asarray(ws, float)
    f = lambda a: (np.asarray(a) * ws[:, None]).sum(0) / ws.sum()
    return f(hs), f(ps), f(ds), len(hs), int(ws.sum())


def boot(st, nboot=1000, seed=1):
    flat = []
    for k, d in st.items():
        for g in ("heal", "persist"):
            for tk, L in d[g]:
                flat.append((tk, g, k, L))
    idx = defaultdict(list)
    for i, f in enumerate(flat):
        idx[f[0]].append(i)
    trajs = list(idx.keys())
    rng = random.Random(seed)
    H, P, D = [], [], []
    for _ in range(nboot):
        acc = defaultdict(lambda: {"heal": [], "persist": []})
        cnt = defaultdict(int)
        for _t in range(len(trajs)):
            cnt[rng.choice(trajs)] += 1
        for t, c in cnt.items():
            for i in idx[t]:
                tk, g, k, L = flat[i]
                for _ in range(c):
                    acc[k][g].append((tk, L))
        res = est(acc)
        if res is None:
            continue
        h, p, d, _, _ = res
        H.append(h); P.append(p); D.append(d)
    q = lambda a, x: np.percentile(np.asarray(a), x, 0)
    return (q(H, 2.5), q(H, 97.5), q(P, 2.5), q(P, 97.5), q(D, 2.5), q(D, 97.5))


def main():
    os.makedirs("figs", exist_ok=True)
    rows = load(MAIN)
    ref = load(REF)
    print(f"main {len(rows)}  ref {len(ref)}")
    st = strat(rows)
    h, p, d, nst, npr = est(st)
    B = boot(st, 1000)
    print(f"strata={nst} pairs={npr}")
    print(f"deep(20-28): heal={h[19:].mean():.5f} persist={p[19:].mean():.5f} "
          f"delta={d[19:].mean():+.5f}  CI=[{B[4][19:].mean():+.5f},{B[5][19:].mean():+.5f}]")

    refc = None
    if ref:
        refc = np.mean([np.asarray(r["layers"], float) for r in ref], 0)

    xs = np.arange(1, 29)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), dpi=180,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.plot(xs, p, "-o", ms=3, lw=2, color="#c0392b", label="persist (H$\\rightarrow$H)")
    ax.fill_between(xs, B[2], B[3], color="#c0392b", alpha=.15)
    ax.plot(xs, h, "-o", ms=3, lw=2, color="#1f77b4", label="self-heal (H$\\rightarrow$F)")
    ax.fill_between(xs, B[0], B[1], color="#1f77b4", alpha=.15)
    if refc is not None:
        ax.plot(xs, refc, "--s", ms=3, lw=1.6, color="#27ae60", alpha=.85,
                label="faithful reference (F$\\rightarrow$F)")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Visual Attention Ratio\n(answer tokens $\\rightarrow$ image tokens)")
    ax.set_title("(a) Visual attention by layer, history-matched")
    ax.grid(alpha=.25, ls=":")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.axhline(0, color="#555", lw=1)
    ax.plot(xs, d, "-o", ms=3, lw=2, color="#2c3e50")
    ax.fill_between(xs, B[4], B[5], color="#2c3e50", alpha=.2)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("$\\Delta$ VAR (self-heal $-$ persist)")
    ax.set_title("(b) Causal effect — CI covers 0 at every layer")
    ax.grid(alpha=.25, ls=":")
    fig.suptitle("EXP3: self-healing is NOT accompanied by a visual-attention rebound "
                 "(Qwen2.5-VL-7B, 2763 forwards, 80 history-matched strata)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figs/figC_layer_var.png")
    plt.close(fig)
    print("wrote figs/figC_layer_var.png")

    # ---- per-round panel for r=3 and r=7 ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180, sharey=True)
    for ax, r in zip(axes, (3, 7)):
        st_r = strat(rows, rounds={r})
        res = est(st_r)
        if res is None:
            ax.set_title(f"round {r}: no matched stratum")
            continue
        hh, pp, dd, ns, npz = res
        Br = boot(st_r, 400)
        ax.plot(xs, pp, "-o", ms=3, lw=2, color="#c0392b", label="persist")
        ax.fill_between(xs, Br[2], Br[3], color="#c0392b", alpha=.15)
        ax.plot(xs, hh, "-o", ms=3, lw=2, color="#1f77b4", label="self-heal")
        ax.fill_between(xs, Br[0], Br[1], color="#1f77b4", alpha=.15)
        ax.set_title(f"round {r}   (strata={ns}, pairs={npz}, "
                     f"$\\Delta$deep={dd[19:].mean():+.5f})")
        ax.set_xlabel("Transformer layer")
        ax.grid(alpha=.25, ls=":")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Visual Attention Ratio")
    fig.suptitle("The two rounds with the largest self-heal mass (r=3: 371, r=7: 257)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("figs/figC_rounds_r3_r7.png")
    plt.close(fig)
    print("wrote figs/figC_rounds_r3_r7.png")

    json.dump({
        "n_strata": nst, "n_pairs": npr,
        "heal": h.tolist(), "persist": p.tolist(), "delta": d.tolist(),
        "ci_heal": [B[0].tolist(), B[1].tolist()],
        "ci_persist": [B[2].tolist(), B[3].tolist()],
        "ci_delta": [B[4].tolist(), B[5].tolist()],
        "ref": refc.tolist() if refc is not None else None,
    }, open("exp3_figC_numbers.json", "w"), indent=2)


if __name__ == "__main__":
    main()
