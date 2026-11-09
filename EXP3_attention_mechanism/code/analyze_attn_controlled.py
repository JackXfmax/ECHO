"""
analyze_attn_controlled.py -- EXP3 mechanism probe, STRATIFIED (history-controlled)

This is the "keystone" analysis. The naive hallucination-round vs recovery-round
attention comparison is confounded by context length AND by history *content*
(the hallucinated vs faithful groups at round r have different prior-round
answers). Here we control for history content by STRATIFIED MATCHING:

  At each round r, we split trajectories into strata by
  k = number of hallucinations in rounds 1..r-1  (their shared history).
  Within each stratum we compare the image-attention fraction of the
  HALLUCINATED state vs the FAITHFUL state. Only strata that contain BOTH
  states contribute a matched contrast. Pooling across strata (weighted by
  matched size) yields the PURE causal effect of the *current round's state*
  on image attention, holding history constant.

Two decisive figures:
  ATT-1: image-attention fraction for Faithful vs Hallucinated across rounds 1..9
         (history-matched). Shows faithful states attend back to the image.
  ATT-2: per-round matched delta (Faithful - Hallucinated) with 95% cluster-bootstrap CI.
         Positive delta = attention recovers to image when faithful, even at matched history.

Usage:
  python analyze_attn_controlled.py --dummy            # synthetic end-to-end selftest
  python analyze_attn_controlled.py --attn attn_per_round.json   # real data
"""
import argparse, json, os, random
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(BASE, "figs")
os.makedirs(FIGDIR, exist_ok=True)
random.seed(7)


# ---------------- verdict map from EXP1 (injection only) ----------------
def load_verdict_map():
    inp = os.path.join(BASE, "EXP1_main_qwen_9000/data/judge_input.jsonl")
    out = os.path.join(BASE, "EXP1_main_qwen_9000/data/judge_output.jsonl")
    verd = {}
    for l in open(out, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        v = r.get("verdict")
        if v in ("faithful", "hallucination"):
            verd[r["uid"]] = (v == "hallucination")
    m = {}
    for l in open(inp, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        if r.get("condition") != "injection":
            continue
        uid = r["uid"]
        if uid in verd:
            key = (r["sample_id"], r["setting"], r["trajectory_id"], r["round"])
            m[key] = verd[uid]
    return m


# ---------------- load attention records ----------------
def load_attn(path):
    recs = json.load(open(path, encoding="utf-8"))
    # accept either {"records":[...]} or a bare list
    if isinstance(recs, dict):
        recs = recs.get("records", [])
    return recs


# ---------------- join + history-count + stratified contrast ----------------
def build_joined(attn_recs, verdict_map):
    joined = []
    skipped = 0
    for rec in attn_recs:
        key = (rec["sample_id"], rec["setting"], rec["trajectory_id"], rec["round"])
        if key not in verdict_map:
            skipped += 1
            continue
        joined.append({
            "traj": (rec["sample_id"], rec["setting"], rec["trajectory_id"]),
            "round": rec["round"],
            "frac": float(rec["image_attn_frac"]),
            "hallu": verdict_map[key],
        })
    return joined, skipped


def compute_history(joined):
    """group by trajectory, sort by round, attach history_hallu_count (preceding rounds)."""
    by_traj = defaultdict(dict)
    for j in joined:
        by_traj[j["traj"]][j["round"]] = j
    for traj, d in by_traj.items():
        for r in sorted(d):
            hc = sum(1 for rr in d if rr < r and d[rr]["hallu"])
            d[r]["hist"] = hc
    return by_traj


def pooled_contrast(by_traj):
    """per round r: stratify by hist count; within stratum compare faithful vs hallucinated frac.
    Returns per-round pooled mean_F, mean_H, delta, plus record counts."""
    # collect per-round records
    per_round = defaultdict(list)
    for traj, d in by_traj.items():
        for r, j in d.items():
            per_round[r].append(j)
    rounds = sorted(per_round)
    out = {}
    for r in rounds:
        strata = defaultdict(lambda: {"F": [], "H": []})
        for j in per_round[r]:
            strata[j["hist"]]["H" if j["hallu"] else "F"].append(j["frac"])
        mF_list, mH_list, w_list = [], [], []
        for s, grp in strata.items():
            if grp["F"] and grp["H"]:
                mF = sum(grp["F"]) / len(grp["F"])
                mH = sum(grp["H"]) / len(grp["H"])
                w = min(len(grp["F"]), len(grp["H"]))  # matched size
                mF_list.append((mF, w)); mH_list.append((mH, w)); w_list.append(w)
        if w_list:
            W = sum(w_list)
            pF = sum(mF * w for mF, w in mF_list) / W
            pH = sum(mH * w for mH, w in mH_list) / W
            out[r] = {"mean_F": pF, "mean_H": pH, "delta": pF - pH,
                      "n_strata": len(w_list), "W": W,
                      "nF": sum(len(strata[s]["F"]) for s in strata),
                      "nH": sum(len(strata[s]["H"]) for s in strata)}
        else:
            out[r] = None
    return out, per_round


def bootstrap(by_traj, per_round, B=2000):
    """cluster bootstrap over trajectories; return per-round (mean_F, mean_H, delta) distributions."""
    trajs = list(by_traj.keys())
    R = max(per_round.keys())
    store = {r: {"F": [], "H": [], "D": []} for r in per_round}
    for _ in range(B):
        # resample trajectories with replacement
        pick = [random.choice(trajs) for _ in trajs]
        # rebuild per_round from picked trajectories (only rounds they have)
        pr = defaultdict(list)
        for t in pick:
            for r, j in by_traj[t].items():
                pr[r].append(j)
        for r in pr:
            strata = defaultdict(lambda: {"F": [], "H": []})
            for j in pr[r]:
                strata[j["hist"]]["H" if j["hallu"] else "F"].append(j["frac"])
            mF_list, mH_list, w_list = [], [], []
            for s, grp in strata.items():
                if grp["F"] and grp["H"]:
                    mF = sum(grp["F"]) / len(grp["F"]); mH = sum(grp["H"]) / len(grp["H"])
                    w = min(len(grp["F"]), len(grp["H"]))
                    mF_list.append((mF, w)); mH_list.append((mH, w)); w_list.append(w)
            if w_list:
                W = sum(w_list)
                pF = sum(mF * w for mF, w in mF_list) / W
                pH = sum(mH * w for mH, w in mH_list) / W
                store[r]["F"].append(pF); store[r]["H"].append(pH); store[r]["D"].append(pF - pH)
    ci = {}
    for r in store:
        if store[r]["F"]:
            ci[r] = {
                "F": (pct(store[r]["F"], 2.5), pct(store[r]["F"], 97.5)),
                "H": (pct(store[r]["H"], 2.5), pct(store[r]["H"], 97.5)),
                "D": (pct(store[r]["D"], 2.5), pct(store[r]["D"], 97.5)),
            }
    return ci


def pct(arr, q):
    arr = sorted(arr)
    i = min(len(arr) - 1, max(0, int(round(q / 100 * (len(arr) - 1)))))
    return arr[i]


# ---------------- figures ----------------
def make_figures(contrast, ci, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.dpi": 200, "axes.linewidth": 1.2})
    rounds = sorted([r for r in contrast if contrast[r]])
    x = np.array(rounds, float)

    # ATT-1: faithful vs hallucinated image-attention, history-matched
    fF = [contrast[r]["mean_F"] for r in rounds]
    fH = [contrast[r]["mean_H"] for r in rounds]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(x, fF, "-o", color="#1E8449", lw=2.6, ms=7, label="Faithful state (matched history)")
    ax.plot(x, fH, "-s", color="#C0392B", lw=2.6, ms=7, label="Hallucinated state (matched history)")
    if ci:
        fFl = [ci[r]["F"][0] for r in rounds]; fFu = [ci[r]["F"][1] for r in rounds]
        fHl = [ci[r]["H"][0] for r in rounds]; fHu = [ci[r]["H"][1] for r in rounds]
        ax.fill_between(x, fFl, fFu, color="#1E8449", alpha=0.12)
        ax.fill_between(x, fHl, fHu, color="#C0392B", alpha=0.12)
    ax.set_xlabel("Conversation round (1 → 9)")
    ax.set_ylabel("Image-attention fraction (matched on history)")
    ax.set_title("Faithful answers re-attend the image; hallucinated answers drift away\n(history-content controlled)", fontsize=13, fontweight="bold")
    ax.set_xticks(rounds); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    p1 = os.path.join(FIGDIR, f"figATT1_attn_faithful_vs_hallu{tag}.png")
    fig.savefig(p1, bbox_inches="tight"); plt.close(fig)

    # ATT-2: per-round matched delta with 95% CI
    d = [contrast[r]["delta"] for r in rounds]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    dl = [ci[r]["D"][0] for r in rounds]; du = [ci[r]["D"][1] for r in rounds]
    ax.fill_between(x, dl, du, color="#2471A3", alpha=0.15)
    ax.plot(x, d, "-o", color="#2471A3", lw=2.6, ms=7)
    ax.axhline(0, color="gray", ls="--", lw=1.2)
    for xi, yi, lo, hi in zip(x, d, dl, du):
        ax.plot([xi, xi], [lo, hi], color="#2471A3", lw=1.4)
    ax.set_xlabel("Conversation round (1 → 9)")
    ax.set_ylabel("Δ image-attention  (Faithful − Hallucinated, history-matched)")
    ax.set_title("Pure causal effect of the current-round state on image attention\n(positive = faithful re-attends image, holding history constant)", fontsize=12.5, fontweight="bold")
    ax.set_xticks(rounds)
    fig.tight_layout()
    p2 = os.path.join(FIGDIR, f"figATT2_causal_delta{tag}.png")
    fig.savefig(p2, bbox_inches="tight"); plt.close(fig)
    return p1, p2


# ---------------- synthetic selftest ----------------
def dummy_data():
    recs = []
    for ti in range(72):
        # random history "true difficulty": some trajectories hallucinate more
        base = random.uniform(-0.1, 0.5)
        seq = []
        for r in range(1, 10):
            # hallucination depends on base + round (snowball) + noise
            p = 0.3 + 0.4 * base + 0.05 * (r - 1) + random.uniform(-0.15, 0.15)
            hallu = random.random() < min(0.95, max(0.05, p))
            seq.append(hallu)
        # also build a 'history' prior to r
        for r in range(1, 10):
            hist = sum(1 for rr in range(1, r) if seq[rr-1])
            # attention: faithful higher than hallucinated; ALSO history affects it (confounder)
            frac = 0.55 + (0.0 if seq[r-1] else 0.15) - 0.03 * hist + random.uniform(-0.04, 0.04)
            frac = min(0.98, max(0.05, frac))
            recs.append({"sample_id": f"img{ti//6}", "setting": f"set{ti%6}",
                         "trajectory_id": ti, "traj": (f"img{ti//6}", f"set{ti%6}", ti),
                         "round": r, "image_attn_frac": frac, "frac": frac,
                         "hallu": seq[r-1]})
    return recs


def run(joined):
    by_traj = compute_history(joined)
    contrast, per_round = pooled_contrast(by_traj)
    ci = bootstrap(by_traj, per_round, B=2000)
    return contrast, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default=None)
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    if args.dummy:
        recs = dummy_data()
        # build a fake verdict_map from the hallu flag
        verdict_map = {}
        for rec in recs:
            verdict_map[(rec["sample_id"], rec["setting"], rec["trajectory_id"], rec["round"])] = rec["hallu"]
        joined, skipped = recs, 0
        contrast, ci = run(joined)
        tag = "_selftest"
    else:
        assert args.attn, "need --attn attn_per_round.json or --dummy"
        verdict_map = load_verdict_map()
        attn = load_attn(args.attn)
        joined, skipped = build_joined(attn, verdict_map)
        print(f"joined {len(joined)} records, skipped {skipped} (no verdict)")
        assert joined, "no joined records -- check the join key / verdict map"
        contrast, ci = run(joined)
        tag = ""

    print("\n=== Stratified (history-matched) contrast ===")
    print(f"{'r':>2} {'mean_F':>7} {'mean_H':>7} {'delta':>7}  {'nF':>4} {'nH':>4} {'strata':>6}")
    for r in sorted(contrast):
        c = contrast[r]
        if not c:
            print(f"{r:>2}  (no matched stratum)"); continue
        print(f"{r:>2} {c['mean_F']:7.3f} {c['mean_H']:7.3f} {c['delta']:7.3f}  {c['nF']:>4} {c['nH']:>4} {c['n_strata']:>6}")
    p1, p2 = make_figures(contrast, ci, tag)
    print("saved", p1, p2)
    print("DONE")


if __name__ == "__main__":
    main()
