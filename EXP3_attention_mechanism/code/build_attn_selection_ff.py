"""
build_attn_selection_ff.py -- build the "faithful reference" arm of EXP3.

The main contrast (self-heal vs persist, identical history bitmask) tells us whether
self-healing COMES WITH a visual-attention rebound. It does not tell us whether the
rebound reaches the level of a model that never derailed.

This script selects, for every heal stratum (r, b) with b ending in "H", trajectories
whose history is b' = b[:-1] + "F"  (i.e. identical up to r-2, and FAITHFUL at r-1)
and that are also faithful at r  ->  group "faithful_ref".

CAVEAT (must be stated in the paper): this arm is NOT fully history-matched -- it
differs from the heal arm at exactly one position (round r-1). It is a reference
level, not part of the causal contrast.
"""
import json, random
from collections import defaultdict, Counter

INP = "EXP1_main_qwen_9000/data/judge_input.jsonl"
OUTP = "EXP1_main_qwen_9000/data/judge_output.jsonl"


def load():
    meta, verd = {}, {}
    for line in open(INP, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("condition") == "injection":
            meta[r["uid"]] = (r["sample_id"], r["setting"], r["trajectory_id"], r["round"])
    for line in open(OUTP, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("verdict") in ("hallucination", "faithful"):
            verd[r["uid"]] = r["verdict"]
    traj = defaultdict(dict)
    for uid, (sid, st, tid, rd) in meta.items():
        if uid in verd:
            traj[(sid, st, tid)][rd] = (verd[uid] == "hallucination")
    return traj


def main():
    random.seed(0)
    traj = load()
    heal = json.loads  # noqa
    heal_rows = [json.loads(l) for l in open("attn_selection_hf.jsonl", encoding="utf-8") if l.strip()]
    n_by = Counter((r["round"], r["stratum"]) for r in heal_rows if r["group"] == "selfheal")

    idx = defaultdict(list)
    for tk, rd_map in traj.items():
        rds = sorted(rd_map)
        for r in rds:
            if r == 1 or (r - 1) not in rd_map:
                continue
            hist = [rd_map[q] for q in range(1, r) if q in rd_map]
            if len(hist) != r - 1:
                continue
            b = "".join("H" if h else "F" for h in hist)
            prev, cur = rd_map[r - 1], rd_map[r]
            if (not prev) and (not cur):
                idx[(r, b)].append(tk)

    out = []
    miss = 0
    for (r, b), n in sorted(n_by.items()):
        if not b.endswith("H"):
            continue
        b2 = b[:-1] + "F"
        pool = idx.get((r, b2), [])
        if not pool:
            miss += n
            continue
        random.shuffle(pool)
        for tk in pool[:n]:
            sid, st, tid = tk
            out.append({
                "sample_id": sid, "setting": st, "trajectory_id": tid,
                "round": r, "group": "faithful_ref",
                "stratum": b2, "hist_bitmask": b2,
                "hist_count": b2.count("H"),
                "matched_by_count_fallback": False,
            })
    print(f"heal strata: {len(n_by)}   strata with no faithful reference: {miss}")
    print(f"faithful_ref records: {len(out)}")
    with open("attn_selection_ff.jsonl", "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("wrote attn_selection_ff.jsonl")


if __name__ == "__main__":
    main()
