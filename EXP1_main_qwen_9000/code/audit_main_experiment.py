"""
audit_main_experiment.py -- pre-freeze integrity audit of the MAIN study (9000 rows).

Checks, in order:
  A. Coverage: judge_input vs judge_output (missing / orphan / duplicate uids)
  B. Errors: verdict == 'error' rows
  C. Design balance: counts per setting x condition x round (the 6x2 design)
  D. Source completeness: fused_50_out_v3 covers 50 samples x 6 settings x 3 traj
  E. Degradation: fraction of model outputs that look like "I cannot see the image"
  F. Published-number reproducibility: recompute the headline figures and compare
     against the ones written in result_judge_analysis.md
"""
import json, collections, re, sys, os

LOCAL = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(LOCAL, *a)

fails, warns = [], []
def check(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)

print("=" * 78)
print("A. COVERAGE  (judge_input vs judge_output)")
print("=" * 78)
inp = [json.loads(l) for l in open(P("judge_input.jsonl"), encoding="utf-8")]
out = [json.loads(l) for l in open(P("judge_output.jsonl"), encoding="utf-8")]
iu = [d["uid"] for d in inp]
ou = [d["uid"] for d in out]
print(f"  input rows={len(inp)}  output rows={len(out)}")
check(len(inp) == len(out) == 9000, "both files have exactly 9000 rows")
check(len(set(iu)) == len(iu), f"input uids unique ({len(set(iu))}/{len(iu)})")
check(len(set(ou)) == len(ou), f"output uids unique ({len(set(ou))}/{len(ou)})")
check(set(iu) == set(ou), "input uid set == output uid set (no missing, no orphan)")

print()
print("=" * 78)
print("B. ERRORS")
print("=" * 78)
vc = collections.Counter(d["verdict"] for d in out)
print("  verdicts:", dict(vc))
check(vc.get("error", 0) == 0, "zero error verdicts")
check(set(vc) <= {"hallucination", "faithful", "error"}, "no unexpected verdict labels")
verd = {d["uid"]: (1 if d["verdict"] == "hallucination" else 0) for d in out}

print()
print("=" * 78)
print("C. DESIGN BALANCE  (6 settings x 2 conditions x rounds)")
print("=" * 78)
cell = collections.Counter((d["setting"], d["condition"], d["round"]) for d in inp)
settings = sorted({d["setting"] for d in inp})
conds = sorted({d["condition"] for d in inp})
print(f"  settings({len(settings)}): {settings}")
print(f"  conditions({len(conds)}): {conds}")
check(len(settings) == 6, "exactly 6 settings")
check(len(conds) == 2, "exactly 2 conditions")
bad = {k: v for k, v in cell.items() if k[1] == "injection" and v != 150}
check(not bad, "every injection (setting,round) cell has 150 rows (50 img x 3 traj)"
      + (f" -- offenders {bad}" if bad else ""))
bad2 = {k: v for k, v in cell.items() if k[1] == "baseline" and v != 150}
check(not bad2, "every baseline (setting,round) cell has 150 rows"
      + (f" -- offenders {bad2}" if bad2 else ""))
# trajectory count
traj = {(d["sample_id"], d["setting"], d["trajectory_id"]) for d in inp}
print(f"  unique (sample,setting,traj) combos = {len(traj)}")
check(len(traj) == 900, "900 unique trajectories = 50 img x 6 setting x 3 traj")
# per-setting sample coverage (guards against the dedup bug seen in the probe script)
per_set = collections.defaultdict(set)
for d in inp:
    per_set[d["setting"]].add(d["sample_id"])
for s in settings:
    check(len(per_set[s]) == 50, f"setting '{s}' covers all 50 images (got {len(per_set[s])})")

print()
print("=" * 78)
print("D. SOURCE COMPLETENESS  (fused_50_out_v3)")
print("=" * 78)
nrec = 0
samp = set()
cells = collections.Counter()
for l in open(P("fused_50_out_v3.jsonl"), encoding="utf-8"):
    rec = json.loads(l)
    nrec += 1
    samp.add(rec["sample_id"])
    for c in rec["cells"]:
        cells[(c["condition"], c["setting"])] += 1
print(f"  fused records={nrec}  unique samples={len(samp)}")
check(len(samp) == 50, "fused has 50 unique samples")
check(nrec == 50, "one fused record per sample")
missing = [k for k in {(cd, s) for cd in conds for s in settings} if cells.get(k, 0) != 50]
check(not missing, f"all 6x2 condition x setting cells present for all 50 samples ({missing})")

print()
print("=" * 78)
print("E. DEGRADATION  (model says it cannot see the image)")
print("=" * 78)
STRICT = re.compile(
    r"(i\s*(?:can(?:no|')t|cannot|am unable to|'m unable to)\s*(?:see|view|access|look at)"
    r"|no image|without (?:the |an )?image|please (?:upload|provide|share) (?:the |an )?image"
    r"|image (?:is |was )?not (?:provided|available|visible)"
    r"|i (?:don't|do not) (?:have|see) (?:access to )?(?:the |any )?image)",
    re.I)
deg = sum(1 for d in inp if STRICT.search(d["model_output"] or ""))
print(f"  strict-pattern degraded outputs = {deg} / {len(inp)} = {deg/len(inp):.2%}")
check(deg / len(inp) < 0.02, "degradation rate < 2%")

print()
print("=" * 78)
print("F. PUBLISHED-NUMBER REPRODUCIBILITY")
print("=" * 78)
# rebuild rows
rows = []
for d in inp:
    rows.append({"setting": d["setting"], "condition": d["condition"], "round": d["round"],
                 "sample_id": d["sample_id"], "traj": d["trajectory_id"],
                 "cat": d.get("category_key") or d.get("hallucination_type", "?"),
                 "hallu": verd[d["uid"]]})

def rate(sel):
    v = [r["hallu"] for r in rows if sel(r)]
    return sum(v) / len(v) if v else float("nan")

# main effect per setting
print("  -- injection vs baseline by setting --")
for s in settings:
    b = rate(lambda r, s=s: r["setting"] == s and r["condition"] == "baseline")
    i = rate(lambda r, s=s: r["setting"] == s and r["condition"] == "injection")
    print(f"    {s:<22} baseline={b:.3f}  injection={i:.3f}  delta={i-b:+.3f}")
    check(i - b > 0.3, f"{s}: injection >> baseline (delta {i-b:+.3f})")

# category rates
print("  -- hallucination rate by attack category (injection) --")
cats = collections.Counter()
cs = collections.Counter()
for r in rows:
    if r["condition"] == "injection" and r["cat"] not in ("?", None):
        cats[r["cat"]] += r["hallu"]; cs[r["cat"]] += 1
for c in sorted(cats, key=lambda k: -cats[k] / cs[k]):
    print(f"    {c:<24} {cats[c]/cs[c]:.3f}  (n={cs[c]})")

# self-recovery
seq = collections.defaultdict(list)
for r in rows:
    if r["condition"] == "injection":
        seq[(r["sample_id"], r["setting"], r["traj"])].append((r["round"], r["hallu"]))
HH = HF = FH = FF = 0
for k, s in seq.items():
    s = sorted(s)
    for a, b in zip(s, s[1:]):
        if a[1] and b[1]: HH += 1
        elif a[1]: HF += 1
        elif b[1]: FH += 1
        else: FF += 1
rec = HF / max(1, HH + HF)
print(f"  transition HH={HH} HF={HF} FH={FH} FF={FF}")
print(f"  ** self-recovery H->F = {HF}/{HH+HF} = {rec:.1%} **")
check(abs(rec - 0.28) < 0.02, f"self-recovery reproduces published 28.0% (got {rec:.1%})")

print()
print("=" * 78)
if fails:
    print(f"AUDIT RESULT: {len(fails)} CHECK(S) FAILED -- DO NOT FREEZE")
    for f in fails:
        print("   x", f)
    sys.exit(1)
print("AUDIT RESULT: ALL CHECKS PASSED -- safe to freeze")
print("=" * 78)
