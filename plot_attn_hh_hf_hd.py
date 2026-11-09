# -*- coding: utf-8 -*-
"""
plot_attn_hh_hf_hd.py -- publication-quality H->H vs H->F attention figure.

Rendering upgrades vs v1:
  1. fine grid: re-extracted at 2x image scale (17x23 = 391 tokens, was 13x18)
  2. attention-modulated alpha: unattended background stays a clean image,
     only attended regions receive JET color (no global blue tint)
  3. shared contrast stretch at the 99th percentile of both maps
  4. thin white contour around the high-attention region
  5. Times New Roman, in-panel mass annotation

Data: EXP4 sample 131064738 / injection t0 (verdicts HHHHFHHFF);
  (b) r7 hallucination turn @ ' adult' token
  (c) r8 self-recovery turn @ 7 entity tokens (adult/larger/baby/smaller)
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

NPZ = "attn_viz/attn_HH_HF_hd.npz"
IMG = "bus_images/131064738.jpg"
SEL_H7 = [10]
SEL_F8 = [16, 21, 43, 57, 65, 74, 77]
BLUR_SIGMA = 1.1          # grid cells
CONTOUR_LEVEL = 0.55      # fraction of vmax for the white outline
A_MAX = 0.85              # max per-pixel blend weight

d = np.load(NPZ, allow_pickle=True)
pw, ph = (int(x) for x in d["proc_size"])
gh, gw = (int(x) for x in d["grid"])

# upscale the source photo to the processing resolution (LANCZOS)
img0 = cv2.imread(IMG)
img = cv2.resize(img0, (pw, ph), interpolation=cv2.INTER_LANCZOS4)

def entity_map(head, rnd, sel):
    rows = d[f"rows_L{head[0]}H{head[1]}_{rnd}"].astype(np.float32)
    return rows[sel].mean(0).reshape(gh, gw)

def render(m, vmax):
    sm = cv2.GaussianBlur(m, (0, 0), BLUR_SIGMA)
    n = np.clip(sm / vmax, 0, 1)
    a8 = (n * 255).astype(np.uint8)
    heat = cv2.applyColorMap(a8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (pw, ph), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(n, (pw, ph), interpolation=cv2.INTER_CUBIC)[..., None] * A_MAX
    ov = (img.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    # thin white contour around the high-attention region
    mask = (cv2.resize(n, (pw, ph), interpolation=cv2.INTER_CUBIC) > CONTOUR_LEVEL).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (255, 255, 255), 2)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)

for head, tag in [((15, 24), "L15H24"), ((20, 24), "L20H24")]:
    mH = entity_map(head, "r7", SEL_H7)
    mF = entity_map(head, "r8", SEL_F8)
    # shared scale at the smoothed peak of the stronger map: (b) saturates,
    # (c) keeps its true relative intensity (difference carried by the figure,
    # cross-checked by the annotated Sigma_img values)
    sH = cv2.GaussianBlur(mH, (0, 0), BLUR_SIGMA)
    sF = cv2.GaussianBlur(mF, (0, 0), BLUR_SIGMA)
    vmax = max(sH.max(), sF.max())
    massH, massF = mH.sum(), mF.sum()

    panels = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), render(mH, vmax), render(mF, vmax)]
    labels = ["(a) Input Image",
              "(b) Hallucination Turn (H$\\rightarrow$H)",
              "(c) Self-Recovery Turn (H$\\rightarrow$F)"]
    notes = [None,
             f"{tag} @ \"adult\" token   $\\Sigma_{{img}}$={massH:.3f}",
             f"{tag} @ entity tokens   $\\Sigma_{{img}}$={massF:.3f}"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), dpi=300)
    for ax, im, lab, nt in zip(axes, panels, labels, notes):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(lab, fontsize=19, fontweight="bold", labelpad=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        if nt:
            ax.text(0.5, -0.155, nt, transform=ax.transAxes, ha="center",
                    fontsize=14.5, color="0.2")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.18, wspace=0.02)
    fig.savefig(f"attn_viz/attention_hh_hf_{tag}_hd.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"attn_viz/attention_hh_hf_{tag}_hd.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[done] {tag}: vmax={vmax:.5f} massH={massH:.4f} massF={massF:.4f}")
