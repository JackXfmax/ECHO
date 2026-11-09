# -*- coding: utf-8 -*-
"""
plot_attn_hh_hf_final.py -- CVPR 1x3 figure, entity-token aggregation.

Data: real EXP4 trajectory (sample 131064738 / injection t0, verdicts HHHHFHHFF).
  (b) Hallucination turn (r7, H): L{H} attention at the ' adult' token emission
      ("the adult elephant is positioned on the left side of the frame" -- judged H)
  (c) Self-recovery turn (r8, F): attention at the correction's entity tokens
      (adult/larger/baby/smaller -- "I apologize... Let me clarify..." -- judged F)
Both overlays share ONE color scale; mass annotated in-panel.

Two versions generated:
  attention_hh_hf_L15H24.png/pdf  (main; spatially structured head)
  attention_hh_hf_L20H24.png/pdf  (companion; the originally requested head)
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for cand in [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf"]:
    try:
        font_manager.fontManager.addfont(cand)
    except Exception:
        pass
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

NPZ = "attn_viz/attn_HH_HF_r78.npz"
IMG = "bus_images/131064738.jpg"
ALPHA = 0.5
SEL_H7 = [10]                                   # ' adult'
SEL_F8 = [16, 21, 43, 57, 65, 74, 77]           # adult/larger/baby/smaller
BLUR_SIGMA = 0.9                                # display-only smoothing (grid cells)

d = np.load(NPZ, allow_pickle=True)
orig_w, orig_h = (int(x) for x in d["orig_size"])
img = cv2.imread(IMG)
img = cv2.resize(img, (orig_w, orig_h)) if img.shape[1] != orig_w else img
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def entity_map(head, rows_key, sel):
    rows = d[f"rows_L{head[0]}H{head[1]}_{rows_key}"].astype(np.float32)
    return rows[sel].mean(0).reshape(13, 18)

def to_overlay(m, vmax):
    sm = cv2.GaussianBlur(m, (0, 0), BLUR_SIGMA)
    a8 = (np.clip(sm / vmax, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(a8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    ov = cv2.addWeighted(img, 1 - ALPHA, heat, ALPHA, 0)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)

for head, tag in [((15, 24), "L15H24"), ((20, 24), "L20H24")]:
    mH = entity_map(head, "r7", SEL_H7)
    mF = entity_map(head, "r8", SEL_F8)
    vmax = max(mH.max(), mF.max())               # shared scale
    massH, massF = mH.sum(), mF.sum()

    panels = [rgb, to_overlay(mH, vmax), to_overlay(mF, vmax)]
    labels = ["(a) Input Image",
              "(b) Hallucination Turn (H$\\rightarrow$H)",
              "(c) Self-Recovery Turn (H$\\rightarrow$F)"]
    notes = [None,
             f"{tag} @ \"adult\" token    $\\Sigma_{{img}}$={massH:.3f}",
             f"{tag} @ entity tokens    $\\Sigma_{{img}}$={massF:.3f}"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
    for ax, im, lab, nt in zip(axes, panels, labels, notes):
        ax.imshow(im)
        ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(lab, fontsize=18, fontweight="bold", labelpad=10)
        if nt:
            ax.text(0.5, -0.13, nt, transform=ax.transAxes, ha="center",
                    fontsize=14, color="0.25")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.20, wspace=0.03)
    pdf = f"attn_viz/attention_hh_hf_{tag}.pdf"
    png = f"attn_viz/attention_hh_hf_{tag}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[done] {png} | vmax={vmax:.4f} massH={massH:.4f} massF={massF:.4f}")
