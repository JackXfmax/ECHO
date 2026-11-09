# -*- coding: utf-8 -*-
"""
plot_attn_hh_hf_l22h2.py -- L22H2 (NEGATIVE routing head) H->H vs H->F figure.
Identical rendering pipeline / format to plot_attn_hh_hf_hd.py (L20H24 figure):
  same image, same trajectory (EXP4 131064738 / injection t0), same rounds
  (b) r7 hallucination turn @ ' adult' token
  (c) r8 self-recovery turn  @ 7 entity tokens (adult/larger/baby/smaller)
Same 2x-scale grid (17x23 = 391 image tokens), attention-modulated alpha,
shared peak color scale, white contour, Times New Roman, Sigma_img annotation.
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

NPZ = "attn_viz/attn_HH_HF_L22H2.npz"
IMG = "bus_images/131064738.jpg"
HEAD = (22, 2)
TAG = "L22H2"
SEL_H7 = [10]
SEL_F8 = [16, 21, 43, 57, 65, 74, 77]
BLUR_SIGMA = 1.1
CONTOUR_LEVEL = 0.55
A_MAX = 0.85

d = np.load(NPZ, allow_pickle=True)
pw, ph = (int(x) for x in d["proc_size"])
gh, gw = (int(x) for x in d["grid"])

img0 = cv2.imread(IMG)
img = cv2.resize(img0, (pw, ph), interpolation=cv2.INTER_LANCZOS4)

def entity_map(rnd, sel):
    rows = d[f"rows_L{HEAD[0]}H{HEAD[1]}_{rnd}"].astype(np.float32)
    return rows[sel].mean(0).reshape(gh, gw)

def render(m, vmax):
    sm = cv2.GaussianBlur(m, (0, 0), BLUR_SIGMA)
    n = np.clip(sm / vmax, 0, 1)
    a8 = (n * 255).astype(np.uint8)
    heat = cv2.applyColorMap(a8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (pw, ph), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(n, (pw, ph), interpolation=cv2.INTER_CUBIC)[..., None] * A_MAX
    ov = (img.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    mask = (cv2.resize(n, (pw, ph), interpolation=cv2.INTER_CUBIC) > CONTOUR_LEVEL).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (255, 255, 255), 2)
    return cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)

mH = entity_map("r7", SEL_H7)
mF = entity_map("r8", SEL_F8)
sH = cv2.GaussianBlur(mH, (0, 0), BLUR_SIGMA)
sF = cv2.GaussianBlur(mF, (0, 0), BLUR_SIGMA)
vmax = max(sH.max(), sF.max())
massH, massF = mH.sum(), mF.sum()

panels = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), render(mH, vmax), render(mF, vmax)]
labels = ["(a) Input Image",
          "(b) Hallucination Turn (H$\\rightarrow$H)",
          "(c) Self-Recovery Turn (H$\\rightarrow$F)"]
notes = [None,
         f"{TAG} @ \"adult\" token   $\\Sigma_{{img}}$={massH:.3f}",
         f"{TAG} @ entity tokens   $\\Sigma_{{img}}$={massF:.3f}"]

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
fig.savefig(f"attn_viz/attention_hh_hf_{TAG}_hd.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"attn_viz/attention_hh_hf_{TAG}_hd.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"[done] {TAG}: vmax={vmax:.6f} massH={massH:.4f} massF={massF:.4f}")
