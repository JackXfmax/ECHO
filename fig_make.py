"""Generate the three paper figures from the single-judge results.

All aggregate numbers are recomputed with the SAME method + seed (7) as
analyze_results_fast.py, so the figures are numerically identical to
result_judge_analysis.md. No model runs; pure post-hoc plotting.
"""
import json, collections, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JIN = os.path.join(HERE, "judge_input.jsonl")
JOUT = os.path.join(HERE, "judge_output.jsonl")
FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

SEED = 7
B = 2000
SETTINGS = ["cleanconv_formatting", "cleanconv_question", "factconv_formatting",
            "halluconv_question", "halluconv_formatting", "irrconv_formatting"]

# ---------- load ----------
meta = {}
order = []
for l in open(JIN, encoding="utf-8"):
    d = json.loads(l); meta[d["uid"]] = d; order.append(d["uid"])
verd = {}
for l in open(JOUT, encoding="utf-8"):
    d = json.loads(l); verd[d["uid"]] = d

rows = []
for uid in order:
    m = meta[uid]; v = verd.get(uid)
    if v is None or v["verdict"] == "error":
        continue
    rows.append({
        "sample_id": m["sample_id"], "setting": m["setting"], "condition": m["condition"],
        "traj": m["trajectory_id"], "round": m["round"], "cat": m["category_key"],
        "hallu": 1 if v["verdict"] == "hallucination" else 0,
    })

cell_round_lists = collections.defaultdict(list)
cat_lists = collections.defaultdict(list)
seq_map = collections.defaultdict(list)
for r in rows:
    cell_round_lists[(r["sample_id"], r["setting"], r["condition"], r["round"])].append(r["hallu"])
    cat_lists[(r["sample_id"], r["cat"])].append(r["hallu"])
    if r["condition"] == "injection":
        seq_map[(r["sample_id"], r["setting"], r["traj"])].append((r["round"], r["hallu"]))

samples = sorted({r["sample_id"] for r in rows})
rng = random.Random(SEED)

def boot_mean(getter, B=B):
    base = [getter(s) for s in samples]
    base = [b for b in base if b is not None]
    if not base:
        return (float("nan"), float("nan"), float("nan"))
    pt = float(np.mean(base))
    idxs = [[rng.randrange(len(samples)) for _ in samples] for _ in range(B)]
    vals = np.empty(B)
    for bi, idx in enumerate(idxs):
        vals[bi] = np.mean([getter(samples[i]) for i in idx])
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

# ---------- per-round injection curve (Fig 1) ----------
round_data = {}
for r in range(1, 10):
    def getter(sid, r=r):
        vals = [cell_round_lists[(sid, s, "injection", r)]
                for s in SETTINGS if (sid, s, "injection", r) in cell_round_lists]
        if not vals:
            return None
        flat = [x for sub in vals for x in sub]
        return sum(flat) / len(flat) if flat else None
    round_data[r] = boot_mean(getter)

# ---------- per-category (Fig 2) ----------
cat_data = {}
for cat in ["entity_missing", "attribute_reversal", "background_fabrication", "context_pollution"]:
    cat_data[cat] = boot_mean(lambda sid, cat=cat: (lambda v: sum(v)/len(v) if v else None)(cat_lists.get((sid, cat))))

# ---------- 2x2 transition (Fig 3) ----------
trans = {(1, 1): 0, (1, 0): 0, (0, 1): 0, (0, 0): 0}  # (src_hallu, dst_hallu)
for key, seq in seq_map.items():
    s = sorted(seq, key=lambda x: x[0])
    for a, b in zip(s, s[1:]):
        trans[(a[1], b[1])] += 1
HH, HF, FH, FF = trans[(1, 1)], trans[(1, 0)], trans[(0, 1)], trans[(0, 0)]
print("transition 2x2:", trans, "total", sum(trans.values()))

# ============================ FIGURE 1 ============================
rounds = list(range(1, 10))
rates = [round_data[r][0] for r in rounds]
lo = [round_data[r][1] for r in rounds]
hi = [round_data[r][2] for r in rounds]
err_lo = [rates[i] - lo[i] for i in range(9)]
err_hi = [hi[i] - rates[i] for i in range(9)]

cat_per_round = {1: "entity_missing", 2: "entity_missing",
                 3: "attribute_reversal", 4: "attribute_reversal",
                 5: "background_fabrication", 6: "background_fabrication",
                 7: "context_pollution", 8: "context_pollution",
                 9: "dataset_final"}
span_colors = {
    "entity_missing": "#fde0dd", "attribute_reversal": "#f5f5f5",
    "background_fabrication": "#e8f3ec", "context_pollution": "#eef0f8",
    "dataset_final": "#fff4e0"}

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 11, "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(9.2, 5.4))
# background shading by attack category
for r in rounds:
    c = cat_per_round[r]
    ax.axvspan(r - 0.5, r + 0.5, color=span_colors[c], zorder=0, alpha=0.9)
# CI band + line
ax.fill_between(rounds, lo, hi, color="#c0392b", alpha=0.18, zorder=1, label="95% CI")
ax.plot(rounds, rates, "-o", color="#c0392b", lw=2.4, ms=7, zorder=3, label="Hallucination rate")
# annotations
ax.annotate("Entity-Missing Cliff\n(≈0.95)", xy=(1.5, 0.95), xytext=(1.5, 0.80),
            ha="center", fontsize=11, color="#a93226", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#a93226", lw=1.4))
ax.annotate("Background-Fabrication\nSpike (r5→r6)", xy=(6, 0.826), xytext=(6, 0.66),
            ha="center", fontsize=11, color="#1e7d4f", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1e7d4f", lw=1.4))
ax.annotate("Dataset core\nquestion (r9)", xy=(9, 0.479), xytext=(8.2, 0.30),
            ha="center", fontsize=10.5, color="#b9770e",
            arrowprops=dict(arrowstyle="->", color="#b9770e", lw=1.2))
ax.set_xlim(0.5, 9.5)
ax.set_ylim(0, 1.02)
ax.set_xticks(rounds)
ax.set_xlabel("Injection Round")
ax.set_ylabel("Hallucination Rate")
ax.set_title("Contextual Snowballing: Non-Monotonic Oscillation Across 9 Rounds")
ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
# category legend (shaded)
from matplotlib.patches import Patch
handles = [Patch(facecolor=span_colors[k], label=k.replace("_", " ")) for k in
           ["entity_missing", "attribute_reversal", "background_fabrication", "context_pollution", "dataset_final"]]
h2, l2 = ax.get_legend_handles_labels()
ax.legend(handles=handles + h2, loc="upper right", framealpha=0.92, fontsize=9.5, ncol=1)
fig.tight_layout()
p1 = os.path.join(FIGDIR, "fig1_round_curve.png")
fig.savefig(p1, bbox_inches="tight"); plt.close(fig)
print("saved", p1)

# ============================ FIGURE 2 ============================
cats = ["entity_missing", "attribute_reversal", "background_fabrication", "context_pollution"]
labels = ["Entity\nMissing", "Attribute\nReversal", "Background\nFabrication", "Context\nPollution"]
vals = [cat_data[c][0] for c in cats]
clo = [cat_data[c][1] for c in cats]
chi = [cat_data[c][2] for c in cats]
eci = [[vals[i] - clo[i] for i in range(4)], [chi[i] - vals[i] for i in range(4)]]
colors = ["#c0392b", "#7f8c8d", "#7f8c8d", "#7f8c8d"]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
bars = ax.bar(labels, vals, color=colors, yerr=eci, capsize=6,
              edgecolor="black", linewidth=0.8, width=0.62, zorder=3)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Hallucination Rate (Rounds 1–8)")
ax.set_title("Hallucination Rate by Injection Category")
for b, v, ch in zip(bars, vals, chi):
    ax.text(b.get_x() + b.get_width() / 2, ch + 0.025, f"{v:.3f}",
            ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.annotate("Entity-Missing = the killer:\ndestroys the visual anchor",
            xy=(0, vals[0]), xytext=(0.6, 0.90), fontsize=10.5, color="#a93226",
            arrowprops=dict(arrowstyle="->", color="#a93226", lw=1.3))
ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
fig.tight_layout()
p2 = os.path.join(FIGDIR, "fig2_category_bar.png")
fig.savefig(p2, bbox_inches="tight"); plt.close(fig)
print("saved", p2)

# ============================ FIGURE 3 (Sankey + matrix) ============================
def draw_sankey(ax, flows, left_labels, right_labels, node_colors, flow_colors):
    L = len(left_labels); R = len(right_labels)
    xL, xR, node_w = 0.10, 0.88, 0.03
    gap = 0.05
    ytop, ybot = 0.97, 0.06
    avail = ytop - ybot
    left_tot = [0] * L; right_tot = [0] * R
    for li, ri, val in flows:
        left_tot[li] += val; right_tot[ri] += val
    maxnode = max(max(left_tot), max(right_tot))
    k = (avail * 0.82) / maxnode
    ly = [0] * L; cursor = ytop
    for i in range(L):
        h = left_tot[i] * k; cursor -= h; ly[i] = cursor + h / 2; cursor -= gap
    ry = [0] * R; cursor = ytop
    for i in range(R):
        h = right_tot[i] * k; cursor -= h; ry[i] = cursor + h / 2; cursor -= gap
    left_used = [0] * L; right_used = [0] * R
    bands = []
    for li, ri, val in sorted(flows, key=lambda f: (f[0], f[1])):
        sh = left_used[li] * k
        node_top = ly[li] + left_tot[li] * k / 2
        s0 = node_top - sh; s1 = s0 - val * k; left_used[li] += val
        th = right_used[ri] * k
        tnode_top = ry[ri] + right_tot[ri] * k / 2
        t0 = tnode_top - th; t1 = t0 - val * k; right_used[ri] += val
        bands.append((s0, s1, t0, t1, flow_colors[li]))
    for i in range(L):
        h = left_tot[i] * k
        ax.add_patch(Rectangle((xL, ly[i] - h / 2), node_w, h, color=node_colors["L"][i], ec="black", lw=0.8))
    for i in range(R):
        h = right_tot[i] * k
        ax.add_patch(Rectangle((xR, ry[i] - h / 2), node_w, h, color=node_colors["R"][i], ec="black", lw=0.8))
    xc = (xL + node_w + xR) / 2
    for s0, s1, t0, t1, c in bands:
        verts = [(xL + node_w, s0), (xc, s0), (xc, t0), (xR, t0),
                 (xR, t1), (xc, t1), (xc, s1), (xL + node_w, s1), (xL + node_w, s0)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=c, edgecolor="none", alpha=0.55))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # labels
    ax.text(xL - 0.02, ly[0], left_labels[0], ha="right", va="center", fontsize=12, fontweight="bold")
    ax.text(xL - 0.02, ly[1], left_labels[1], ha="right", va="center", fontsize=12, fontweight="bold")
    ax.text(xR + node_w + 0.02, ry[0], right_labels[0], ha="left", va="center", fontsize=12, fontweight="bold")
    ax.text(xR + node_w + 0.02, ry[1], right_labels[1], ha="left", va="center", fontsize=12, fontweight="bold")
    return ly, ry, left_tot, right_tot, k

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 0.95]})
# Sankey
flows = [(1, 1, HH), (1, 0, HF), (0, 1, FH), (0, 0, FF)]
node_colors = {"L": ["#e67e22", "#27ae60"], "R": ["#e67e22", "#27ae60"]}
draw_sankey(axL, flows, ["Hallucination\n(prev round)", "Faithful\n(prev round)"],
            ["Hallucination\n(next round)", "Faithful\n(next round)"], node_colors, ["#e67e22", "#27ae60"])
axL.set_title("State Transition (Sankey)", fontsize=13)
# annotate recovery
axL.text(0.5, 0.015, f"Self-recovery H→F = {HF}/{HH+HF} = {HF/(HH+HF):.1%}",
         ha="center", fontsize=11, color="#1e7d4f", fontweight="bold", transform=axL.transAxes)

# Matrix heatmap
M = np.array([[FF, FH], [HF, HH]], dtype=float)  # rows=src(F,H), cols=dst(F,H)
Mp = M / M.sum(axis=1, keepdims=True)
axR.imshow(Mp, cmap="YlOrRd", vmin=0, vmax=1)
axR.set_xticks([0, 1]); axR.set_yticks([0, 1])
axR.set_xticklabels(["Faithful", "Halluc."])
axR.set_yticklabels(["Faithful\n(src)", "Halluc.\n(src)"])
for i in range(2):
    for j in range(2):
        cnt = int(M[i, j]); pct = Mp[i, j]
        axR.text(j, i, f"{cnt}\n{pct:.1%}", ha="center", va="center", fontsize=12, fontweight="bold",
                 color="black" if pct < 0.6 else "white")
axR.set_title("Transition Matrix", fontsize=13)
axR.set_xlabel("next round"); axR.set_ylabel("prev round")
fig.suptitle("Self-Recovery Inside Injected Trajectories (per consecutive round)", fontsize=14, y=1.02)
fig.tight_layout()
p3 = os.path.join(FIGDIR, "fig3_recovery_sankey.png")
fig.savefig(p3, bbox_inches="tight"); plt.close(fig)
print("saved", p3)
print("FIG1 rounds:", [round(round_data[r][0], 3) for r in rounds])
print("FIG2 cats:", {c: round(cat_data[c][0], 3) for c in cats})
print("transition:", dict(trans))
