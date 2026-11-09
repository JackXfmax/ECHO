"""
analyze_attn_perlayer.py -- EXP3 stratified causal analysis + Figure C (layer 1..28)

INPUT   attn_perlayer_hf.jsonl  (produced by server_attn_perlayer.py)
        each record: sample_id, setting, trajectory_id, round, group(selfheal|persist),
                     stratum (exact H/F bitmask of rounds 1..r-1), layers[28], heads,
                     n_image_tokens, n_answer_tokens, seq_len, max_pixels

ESTIMAND
    For every stratum s = (round r, exact history bitmask) that contains BOTH a
    self-heal member (H->F) and a persist member (H->H):

        delta_s[l] = mean_{selfheal in s} VAR_l  -  mean_{persist in s} VAR_l

    and the stratified estimate

        Delta[l] = sum_s w_s * delta_s[l] / sum_s w_s ,  w_s = min(n_heal, n_persist)

    Because the bitmask fixes the entire hallucination history, Delta[l] is the
    effect of the CURRENT round's outcome, not of "the model answered differently
    earlier".

OUTPUTS
    figs/figC_layer_var.png          layer 1..28, self-heal vs persist (+95% CI)
    figs/figC_layer_delta.png        Delta[l] with CI, per layer
    figs/figC_rounds_r3_r7.png       same contrast restricted to r=3 and r=7
    exp3_stratified_result.json      the numbers behind the figures

SECONDARY METRIC (robustness)
    lift = VAR * seq_len / n_image_tokens   -- chance-normalised. Raw VAR grows with
    the number of image tokens; lift removes that (and the sequence length) so the
    two groups stay comparable even if their images differ in resolution.

USAGE
    python analyze_attn_perlayer.py --attn attn_perlayer_hf.jsonl
    python analyze_attn_perlayer.py --attn attn_smoke.jsonl --boot 200   # plumbing test
"""
import argparse, json, os, random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("layers") and all(x is not None for x in r["layers"]):
            rows.append(r)
    return rows


def traj_key(r):
    return (r["sample_id"], r["setting"], r["trajectory_id"])


def build_strata(rows, metric="raw"):
    """(r, stratum) -> {'heal': [curve...], 'persist': [curve...]}"""
    st = defaultdict(lambda: {"heal": [], "persist": []})
    for r in rows:
        L = np.asarray(r["layers"], dtype=float)
        if metric == "lift":
            L = L * r["seq_len"] / max(1, r["n_image_tokens"])
        key = (r["round"], r["stratum"])
        g = r["group"]
        if g in ("heal", "selfheal"):
            st[key]["heal"].append((traj_key(r), L))
        elif g == "persist":
            st[key]["persist"].append((traj_key(r), L))
    return st


def strat_estimate(st, weights=None):
    """returns (heal_curve, persist_curve, delta_curve, n_strata, n_pairs)"""
    hs, ps, ds, ws = [], [], [], []
    n_pairs = 0
    for key, d in st.items():
        nh, npp = len(d["heal"]), len(d["persist"])
        if nh == 0 or npp == 0:
            continue
        h = np.mean([x[1] for x in d["heal"]], axis=0)
        p = np.mean([x[1] for x in d["persist"]], axis=0)
        w = min(nh, npp)
        hs.append(h)
        ps.append(p)
        ds.append(h - p)
        ws.append(w)
        n_pairs += w
    if not hs:
        return None
    ws = np.asarray(ws, dtype=float)
    if weights is not None:
        ws = np.asarray([weights.get(i, w) for i, w in enumerate(ws)], dtype=float)
    hs = np.asarray(hs)
    ps = np.asarray(ps)
    ds = np.asarray(ds)
    hc = (hs * ws[:, None]).sum(0) / ws.sum()
    pc = (ps * ws[:, None]).sum(0) / ws.sum()
    dc = (ds * ws[:, None]).sum(0) / ws.sum()
    return hc, pc, dc, len(hs), int(n_pairs)


def bootstrap(st, nboot, seed=0, metric="raw"):
    """cluster bootstrap over trajectories; returns per-layer 2.5/97.5 for heal,
    persist and delta"""
    rng = random.Random(seed)
    idx_by_traj = defaultdict(list)
    # flat list of (trajkey, group, stratumkey, curve)
    flat = []
    for key, d in st.items():
        for g in ("heal", "persist"):
            for tk, L in d[g]:
                flat.append((tk, g, key, L))
    for i, f in enumerate(flat):
        idx_by_traj[f[0]].append(i)
    tl = list(idx_by_traj.keys())
    dh, dp, dd = [], [], []
    for b in range(nboot):
        pick = [rng.choice(tl) for _ in range(len(tl))]
        cnt = defaultdict(int)
        for t in pick:
            cnt[t] += 1
        st_b = defaultdict(lambda: {"heal": [], "persist": []})
        for t, c in cnt.items():
            for i in idx_by_traj[t]:
                tk, g, key, L = flat[i]
                for _ in range(c):
                    st_b[key][g].append((tk, L))
        res = strat_estimate(st_b)
        if res is None:
            continue
        hc, pc, dc, _, _ = res
        dh.append(hc)
        dp.append(pc)
        dd.append(dc)
    q = lambda a, p: np.percentile(np.asarray(a), p, axis=0)
    return (q(dh, 2.5), q(dh, 97.5), q(dp, 2.5), q(dp, 97.5),
            q(dd, 2.5), q(dd, 97.5), len(dh))


def plot_layers(curve_h, curve_p, ci_h, ci_p, n_img_layers, out, title):
    xs = np.arange(1, len(curve_h) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    ax.plot(xs, curve_p, "-o", ms=3.2, lw=2.0, color="#c0392b",
            label="persist  (H at r-1 $\\rightarrow$ H at r)")
    ax.fill_between(xs, ci_p[0], ci_p[1], color="#c0392b", alpha=.16)
    ax.plot(xs, curve_h, "-o", ms=3.2, lw=2.0, color="#1f77b4",
            label="self-heal (H at r-1 $\\rightarrow$ F at r)")
    ax.fill_between(xs, ci_h[0], ci_h[1], color="#1f77b4", alpha=.16)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Visual Attention Ratio (answer tokens $\\rightarrow$ image tokens)")
    ax.set_title(title)
    ax.grid(alpha=.25, ls=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_delta(curve_d, ci_d, out, title):
    xs = np.arange(1, len(curve_d) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=180)
    ax.axhline(0, color="#555", lw=1)
    ax.plot(xs, curve_d, "-o", ms=3.2, lw=2.0, color="#2c3e50")
    ax.fill_between(xs, ci_d[0], ci_d[1], color="#2c3e50", alpha=.18)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("$\\Delta$ VAR  (self-heal  $-$  persist)")
    ax.set_title(title)
    ax.grid(alpha=.25, ls=":")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="attn_perlayer_hf.jsonl")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--rounds", default="", help="e.g. 3,7  restrict to these rounds")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load(args.attn)
    print(f"loaded {len(rows)} usable records")
    if not rows:
        return
    if args.rounds:
        keep = set(int(x) for x in args.rounds.split(","))
        rows = [r for r in rows if r["round"] in keep]

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or ("_r" + args.rounds.replace(",", "") if args.rounds else "")

    # sanity: are the two groups comparable in image-token count / seq length?
    for g in ("selfheal", "persist"):
        sub = [r for r in rows if r["group"] == g]
        if sub:
            print(f"  {g:9s} n={len(sub):5d}  n_img mean={np.mean([r['n_image_tokens'] for r in sub]):.1f}"
                  f"  seq mean={np.mean([r['seq_len'] for r in sub]):.0f}"
                  f"  n_ans mean={np.mean([r['n_answer_tokens'] for r in sub]):.0f}")

    out = {}
    for metric in ("raw", "lift"):
        st = build_strata(rows, metric=metric)
        res = strat_estimate(st)
        if res is None:
            print(f"[{metric}] no stratum with both groups -- nothing to plot")
            continue
        hc, pc, dc, nst, npairs = res
        B = bootstrap(st, args.boot)
        ci_h = (B[0], B[1])
        ci_p = (B[2], B[3])
        ci_d = (B[4], B[5])
        print(f"\n[{metric}] strata={nst} pairs={npairs} boot={B[6]}")
        print(f"  Delta VAR: overall mean={dc.mean():+.4f}  "
              f"shallow(1-9)={dc[:9].mean():+.4f}  deep(20-28)={dc[19:].mean():+.4f}  "
              f"max={dc.max():+.4f}@L{int(dc.argmax())+1}")
        out[metric] = {
            "n_strata": nst, "n_pairs": npairs,
            "heal": hc.tolist(), "persist": pc.tolist(), "delta": dc.tolist(),
            "ci_heal": [ci_h[0].tolist(), ci_h[1].tolist()],
            "ci_persist": [ci_p[0].tolist(), ci_p[1].tolist()],
            "ci_delta": [ci_d[0].tolist(), ci_d[1].tolist()],
        }
        if metric == "raw":
            plot_layers(hc, pc, ci_h, ci_p, len(hc),
                        f"{args.outdir}/figC_layer_var{tag}.png",
                        "Visual Attention Ratio by layer — history-matched "
                        "(identical hallucination history, rounds 1..r-1)")
            plot_delta(dc, ci_d, f"{args.outdir}/figC_layer_delta{tag}.png",
                       "Causal effect of self-healing on visual attention "
                       "(stratified, history held fixed)")
        else:
            plot_delta(dc, ci_d, f"{args.outdir}/figC_layer_delta_lift{tag}.png",
                       "Robustness: chance-normalised $\\Delta$ visual-attention lift")

    json.dump(out, open(f"exp3_stratified_result{tag}.json", "w"), indent=2)
    print(f"\nwrote exp3_stratified_result{tag}.json")


if __name__ == "__main__":
    main()
