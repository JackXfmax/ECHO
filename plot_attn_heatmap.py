# -*- coding: utf-8 -*-
"""
plot_attn_heatmap.py -- CVPR-style 1x3 figure:
  (a) Input Image  (b) Attention Map (JET)  (c) Overlay (alpha=0.5)
Data: attn_viz/attn_L20H24.npz from server_attn_heatmap.py (Qwen2.5-VL-7B,
layer 20, head 24, attention averaged over 73 generated answer tokens across
234 merged image tokens, 13x18 grid from image_grid_thw).
"""
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- fonts: Times New Roman, >=16 ----
for cand in [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf"]:
    try:
        font_manager.fontManager.addfont(cand)
    except Exception:
        pass
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

NPZ = "attn_viz/attn_L20H24.npz"
IMG = "bus_images/131064738.jpg"
OUT_PDF = "attn_viz/attention_visualization.pdf"
OUT_PNG = "attn_viz/attention_visualization.png"
ALPHA = 0.5

# ---- load data ----
d = np.load(NPZ)
attn2d = d["attn2d"].astype(np.float32)          # (13, 18)
orig_w, orig_h = (int(x) for x in d["orig_size"])
lh = d["layer_head"].tolist()
meta = json.load(open("attn_viz/attn_L20H24_meta.json", encoding="utf-8"))

# ---- smoke test: 2D check before colormap ----
assert attn2d.ndim == 2, f"attention map must be 2D, got {attn2d.shape}"
print(f"[smoke] attention map shape: {attn2d.shape} (layer {lh[0]}, head {lh[1]})")

# ---- original image ----
img = cv2.imread(IMG)
assert img is not None
img = cv2.resize(img, (orig_w, orig_h)) if img.shape[1] != orig_w else img
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---- normalize to 0-255 ----
amap = attn2d - attn2d.min()
amap = amap / (amap.max() + 1e-12)
amap8 = (amap * 255).astype(np.uint8)

# ---- (b) JET heatmap resized to image dims ----
heat = cv2.applyColorMap(amap8, cv2.COLORMAP_JET)
heat = cv2.resize(heat, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

# ---- (c) overlay ----
overlay = cv2.addWeighted(img, 1 - ALPHA, heat, ALPHA, 0)
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ---- plot ----
panels = [rgb, heat_rgb, overlay_rgb]
labels = ["(a) Input Image", "(b) Attention Map", "(c) Overlay"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
for ax, im, lab in zip(axes, panels, labels):
    ax.imshow(im)
    ax.axis("off")
    ax.set_xlabel(lab, fontsize=20, fontweight="bold", labelpad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.14, wspace=0.03)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"[done] {OUT_PDF} + {OUT_PNG}")
print(f"[info] answer: {meta['gen_text'][:90]}...")
print(f"[info] img_attn sum over 234 tokens = "
      f"{float(d['attn2d'].sum()):.4f}; peak = {attn2d.max():.4f}")
