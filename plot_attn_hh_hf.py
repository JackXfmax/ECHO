# -*- coding: utf-8 -*-
"""
plot_attn_hh_hf.py -- CVPR 1x3 figure: real-trajectory H->H vs H->F attention
of L20H24 (Qwen2.5-VL-7B, EXP4 sample 131064738 / traj t0, verdicts HHHHFHHFF,
r4 = hallucination turn, r5 = self-recovery turn).

Honest-visualization note: both overlays share ONE color scale (both divided
by the global max) so the reader can compare attention mass across turns --
the recovery turn does NOT show a visual-attention rebound (EXP3-consistent).
"""
import json
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

NPZ = "attn_viz/attn_HH_HF.npz"
IMG = "bus_images/131064738.jpg"
OUT_PDF = "attn_viz/attention_hh_hf_comparison.pdf"
OUT_PNG = "attn_viz/attention_hh_hf_comparison.png"
ALPHA = 0.5

d = np.load(NPZ)
orig_w, orig_h = (int(x) for x in d["orig_size"])
mH, mF = d["r4"].astype(np.float32), d["r5"].astype(np.float32)

# smoke: 2D check
assert mH.ndim == 2 and mF.ndim == 2, (mH.shape, mF.shape)
print(f"[smoke] H map {mH.shape}, F map {mF.shape}")

# shared scale: normalize by the GLOBAL max so panels are comparable
gmax = max(mH.max(), mF.max())
mass_H, mass_F = float(mH.sum()), float(mF.sum())
print(f"[info] img-attn mass: H->H turn {mass_H:.4f} | H->F turn {mass_F:.4f} "
      f"(ratio {mass_H / mass_F:.2f}x)")

img = cv2.imread(IMG)
img = cv2.resize(img, (orig_w, orig_h)) if img.shape[1] != orig_w else img

def to_overlay(m):
    a8 = (np.clip(m / (gmax + 1e-12), 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(a8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    ov = cv2.addWeighted(img, 1 - ALPHA, heat, ALPHA, 0)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
panels = [rgb, to_overlay(mH), to_overlay(mF)]
labels = ["(a) Input Image",
          "(b) Hallucination Turn (H$\\rightarrow$H)",
          "(c) Self-Recovery Turn (H$\\rightarrow$F)"]
masses = [None, mass_H, mass_F]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
for ax, im, lab, ms in zip(axes, panels, labels, masses):
    ax.imshow(im)
    ax.axis("off")
    ax.set_xlabel(lab, fontsize=18, fontweight="bold", labelpad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if ms is not None:
        ax.text(0.03, 0.06, f"$\\Sigma_{{img}}$ = {ms:.3f}", transform=ax.transAxes,
                fontsize=15, color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none",
                          boxstyle="round,pad=0.25"))

fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.15, wspace=0.03)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"[done] {OUT_PDF} + {OUT_PNG}")
