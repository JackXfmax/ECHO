"""
build_attn_selection_hf.py -- build the EXP3 attention-replay selection from EXP1 verdicts.

GOAL (the causal question):
    At round r, compare trajectories that SELF-HEAL (H at r-1 -> F at r) against
    trajectories that PERSIST (H at r-1 -> H at r), where BOTH groups have an
    IDENTICAL hallucination history over rounds 1..r-1.

    Because the history is held fixed, any difference in visual attention at
    round r cannot be blamed on "the model simply answered differently earlier" --
    it is attributable to the current round's visual re-grounding.

STRATA
    --strata bitmask (default, strictest): the exact H/F pattern of rounds 1..r-1,
       e.g. r=3 with history "HF" and "FH" are DIFFERENT strata.
    --strata count  : only the number of prior hallucinations (0..r-1) is matched,
       e.g. "HF" and "FH" collapse into the same stratum. Larger strata, weaker
       control. Used only as a fallback / robustness check.

OUTPUT
    attn_selection_hf.jsonl  one record per (trajectory, round) forward pass.
    stdout: stratum diagnostics (how many strata are usable, i.e. contain BOTH
    a self-heal and a persist member) and the total forward-pass budget.
"""
import argparse, json, random
from collections import defaultdict, Counter

INP = "EXP1_main_qwen_9000/data/judge_input.jsonl"
OUTP = "EXP1_main_qwen_9000/data/judge_output.jsonl"


def load():
    meta = {}
    for line in open(INP, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("condition") != "injection":
            continue
        meta[r["uid"]] = (r["sample_id"], r["setting"], r["trajectory_id"], r["round"])
    verd = {}
    for line in open(OUTP, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        v = r.get("verdict")
        if v in ("hallucination", "faithful"):
            verd[r["uid"]] = v
    traj = defaultdict(dict)
    for uid, (sid, st, tid, rd) in meta.items():
        if uid in verd:
            traj[(sid, st, tid)][rd] = (verd[uid] == "hallucination")
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", choices=["bitmask", "count"], default="bitmask")
    ap.add_argument("--fallback_count", action="store_true", default=True,
                    help="for self-heal events whose bitmask stratum has no persist "
                         "member, retry at count-level matching")
    ap.add_argument("--max_per_stratum", type=int, default=999,
                    help="cap persist controls per stratum (keep selection balanced)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="attn_selection_hf.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    traj = load()
    print(f"trajectories: {len(traj)}   (expect 900)")

    def key_of(r, hist):
        """hist = list of bools for rounds 1..r-1"""
        if args.strata == "bitmask":
            return "".join("H" if h else "F" for h in hist)
        return str(sum(hist))

    def key_count(hist):
        return str(sum(hist))

    # ---- collect events -------------------------------------------------
    heal = defaultdict(list)     # (r, stratum) -> [(trajkey, hist)]
    persist = defaultdict(list)
    counts = Counter()
    for tk, rd_map in sorted(traj.items()):
        rds = sorted(rd_map)
        for r in rds:
            if r == 1 or (r - 1) not in rd_map:
                continue
            prev, cur = rd_map[r - 1], rd_map[r]
            hist = [rd_map[q] for q in range(1, r) if q in rd_map]
            if len(hist) != r - 1:
                continue          # incomplete trajectory, skip
            if prev and not cur:
                counts[("heal", r)] += 1
                heal[(r, key_of(r, hist))].append((tk, hist))
            elif prev and cur:
                counts[("persist", r)] += 1
                persist[(r, key_of(r, hist))].append((tk, hist))

    print("\n-- transitions by round (H at r-1) --")
    print(f"{'r':>3} {'selfheal(H->F)':>15} {'persist(H->H)':>14}")
    tot_h = tot_p = 0
    for r in range(2, 10):
        h = counts[("heal", r)]
        p = counts[("persist", r)]
        tot_h += h
        tot_p += p
        print(f"{r:>3} {h:>15} {p:>14}")
    print(f"{'ALL':>3} {tot_h:>15} {tot_p:>14}   (ledger: HF=1443 HH=3711)")

    # ---- match ----------------------------------------------------------
    sel = []
    used_persist = defaultdict(int)
    diag = []
    n_unmatched = 0
    n_fallback = 0
    for (r, stratum), hs in sorted(heal.items()):
        pool = persist.get((r, stratum), [])
        matched = False
        fb = False
        if not pool and args.fallback_count and args.strata == "bitmask":
            # widen to count-level: gather persist events at same r with same prior count
            cnt = str(sum(1 for c in stratum if c == "H"))
            pool = []
            for (r2, s2), ps in persist.items():
                if r2 == r and s2 == cnt:
                    pool += ps
            fb = True
        if not pool:
            n_unmatched += len(hs)
            continue
        matched = True
        random.shuffle(pool)
        take = min(len(hs), len(pool), args.max_per_stratum)
        chosen = pool[:take]
        for i in range(take):
            used_persist[(r, stratum)] += 1
        for tk, hist in hs:
            sel.append(mk(tk, r, "selfheal", stratum, hist, fb))
        for tk, hist in chosen:
            sel.append(mk(tk, r, "persist", stratum, hist, fb))
        if fb:
            n_fallback += len(hs)
        diag.append((r, stratum, len(hs), len(chosen)))

    print("\n-- matched strata (top 25 by size) --")
    print(f"{'r':>3} {'stratum':>12} {'heal':>6} {'persist':>7}")
    for r, s, nh, np_ in sorted(diag, key=lambda x: -x[2])[:25]:
        print(f"{r:>3} {s:>12} {nh:>6} {np_:>7}")
    print(f"\nusable strata: {len(diag)}")
    print(f"self-heal events unmatched (no control available): {n_unmatched}")
    print(f"self-heal events matched via count-level fallback: {n_fallback}")

    ns = sum(1 for x in sel if x["group"] == "selfheal")
    npp = sum(1 for x in sel if x["group"] == "persist")
    print(f"\nSELECTION: selfheal={ns} persist={npp} TOTAL FORWARD PASSES={len(sel)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for x in sel:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


def mk(tk, r, group, stratum, hist, fb):
    sid, st, tid = tk
    return {
        "sample_id": sid,
        "setting": st,
        "trajectory_id": tid,
        "round": r,
        "group": group,
        "stratum": stratum,
        "hist_bitmask": "".join("H" if h else "F" for h in hist),
        "hist_count": int(sum(hist)),
        "matched_by_count_fallback": bool(fb),
    }


if __name__ == "__main__":
    main()
