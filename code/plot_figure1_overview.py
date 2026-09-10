"""Figure 1 -- conceptual overview of cross-lingual steering in Devanagari LLMs.

Three panels:
  (a) Surface layer & shared tokenization: shared Devanagari tokens -> prompt ->
      Layer-0 distinct structural spaces; steering vector v = mu_src - mu_tgt.
  (b) The hourglass geometry & steering: per-language trajectories vs. transformer
      depth (distinct -> merged mid-network core -> re-separated), with the L20
      steering point; t-SNE and adherence-bar insets.
  (c) Steered output generation: unsteered (Hindi) vs. steered (Nepali) pathways
      with native-Devanagari generations.

Native Devanagari via a system font; bar-chart values are our real Llama-L20
language-adherence scores (peak adherence / 5).

  uv run --with matplotlib --with numpy python code/plot_figure1_overview.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "method_overview.png"

# ---- fonts ----
_deva_paths = ['/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc',
               '/System/Library/Fonts/Kohinoor.ttc',
               '/System/Library/Fonts/Supplemental/DevanagariMT.ttc']
DEVA = fm.FontProperties(fname=next(p for p in _deva_paths if os.path.exists(p)))

# ---- palette ----
HI, NEP, MAI = "#1f6fb2", "#2e8b3d", "#e08a1e"          # Hindi / Nepali / Maithili
GRAY, GRAYE = "#e9e9e9", "#888888"
RED = "#c0392b"
BLUEBOX, GREENBOX = "#eaf2fb", "#eaf7ee"
BLUEBOXE, GREENBOXE = "#2e6da4", "#3a9d4e"


def box(ax, x, y, w, h, fc, ec, lw=1.4, rounding=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                 fc=fc, ec=ec, lw=lw, zorder=3))


def arrow(ax, x1, y1, x2, y2, color="#333", lw=2.0, ms=16, style="-|>",
          conn="arc3,rad=0", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=ms, lw=lw, color=color, connectionstyle=conn,
                 zorder=6, linestyle=ls))


def t(ax, x, y, s, fs=11, w="normal", c="black", ha="center", va="center",
      deva=False, rot=0, style="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va,
            rotation=rot, style=style, zorder=7,
            fontproperties=(DEVA if deva else None))


def main():
    fig, ax = plt.subplots(figsize=(16.5, 8.6))
    ax.set_xlim(0, 300); ax.set_ylim(0, 100); ax.axis("off")

    # ===== title =====
    ax.add_patch(FancyBboxPatch((3, 92.5), 294, 6.2, boxstyle="round,pad=0,rounding_size=1",
                 fc="#f0f0f0", ec="#555", lw=1.4, zorder=2))
    t(ax, 150, 95.6, "Mechanistic Insights into Cross-Lingual Steering in Devanagari LLMs",
      fs=20, w="bold")

    # =========================================================
    # (a) SURFACE LAYER & SHARED TOKENIZATION
    # =========================================================
    box(ax, 4, 60, 86, 28, "white", "#555", 1.6)        # Fixed Tokenizer Context
    t(ax, 47, 84.5, "Fixed Tokenizer Context", fs=13, w="bold")
    box(ax, 9, 74.5, 76, 7, GRAY, GRAYE, 1.2)
    t(ax, 47, 78, "Shared Devanagari Script Tokens", fs=11.5)
    arrow(ax, 47, 74.3, 47, 70.8, lw=1.8)
    box(ax, 9, 62.5, 76, 7.5, GRAY, GRAYE, 1.2)
    t(ax, 30, 66.2, "Input Prompt:", fs=11)
    t(ax, 64, 66.2, "आज मौसम कैसा है?", fs=13, deva=True)
    arrow(ax, 47, 60, 47, 55.5, lw=2.0)

    box(ax, 4, 8, 86, 45, "white", "#555", 1.6)         # Layer 0 distinct spaces
    t(ax, 47, 49, "Layer 0: Distinct Structural Spaces", fs=12.5, w="bold")
    t(ax, 16, 41, "Baseline:", fs=11.5)
    ax.scatter([40], [41], s=260, c=GRAY, edgecolors=GRAYE, linewidths=1.5, zorder=5)
    t(ax, 66, 42.5, r"$v=\mu_{\mathrm{src}}-\mu_{\mathrm{tgt}}$", fs=13)
    langs = [(18, HI, "Hindi", "(HI)"), (47, NEP, "Nepali", "(NEP)"),
             (76, MAI, "Maithili", "(MAI)")]
    for lx, col, nm, tag in langs:
        arrow(ax, 40, 38.5, lx, 22.5, color=col, lw=2.2,
              conn=f"arc3,rad={0.0 if lx==47 else (0.18 if lx>47 else -0.18)}")
        ax.scatter([lx], [19], s=300, c=col, edgecolors="black", linewidths=0.8, zorder=5)
        t(ax, lx, 14, nm, fs=12.5, w="bold", c=col)
        t(ax, lx, 10.5, tag, fs=10.5, c=col)
    t(ax, 47, 1.8, "(a)  Surface Layer & Shared Tokenization", fs=12.5, w="bold")

    # =========================================================
    # (b) THE HOURGLASS GEOMETRY & STEERING
    # =========================================================
    X0 = 150.0                       # centre of the depth axis
    AX_X = 108                       # vertical axis position
    yT, yB = 88.0, 30.0              # depth 0 (top) .. 32 (bottom); bottom raised
    #                                  to leave room for Output Logits + insets

    def y_of(d):                    # depth 0..32 -> y
        return yT - (d / 32.0) * (yT - yB)

    # depth axis
    arrow(ax, AX_X, yB - 2, AX_X, yT + 3, color="black", lw=1.8)
    t(ax, AX_X - 5.5, (yT + yB) / 2, "Transformer Layer Depth ($L$)", fs=12, rot=90)
    for d in (0, 12, 20, 32):
        ax.plot([AX_X - 1, AX_X + 1], [y_of(d)] * 2, color="black", lw=1.2, zorder=4)
        t(ax, AX_X - 2.2, y_of(d), str(d), fs=11, ha="right")
    # dashed guide lines at 0 / 12 / 20 / 32
    for d in (0, 12, 20, 32):
        ax.plot([AX_X + 1, 196], [y_of(d)] * 2, color="#bbb", lw=0.8, ls=(0, (4, 4)), zorder=1)

    # shared mid-network core band (depth 12..20)
    ax.add_patch(plt.Rectangle((118, y_of(20)), 60, y_of(12) - y_of(20),
                 fc="#e2e2e2", ec="none", alpha=0.7, zorder=1))
    t(ax, 185, (y_of(12) + y_of(20)) / 2 + 3, "Shared\nMid-Network\nSemantic Core",
      fs=10.5, ha="center")

    # trajectories: offset(depth) -> hourglass. converge by ~depth 16, steer at 20.
    depths = np.linspace(0, 32, 200)

    def traj(side):
        """side: -1 Hindi(left), 0 Nepali(centre), +1 Maithili(right).
        Distinct at the surface, fully merged through the 12--20 core band,
        re-separated after the L20 steering point."""
        off = np.empty_like(depths)
        for i, d in enumerate(depths):
            if d <= 12:                                   # funnel into the core
                conv = ((12 - d) / 12.0) ** 1.25
                off[i] = side * 15.0 * conv
            elif d <= 20:                                 # merged core band
                off[i] = side * 1.0                       # ~overlapping
            else:                                         # after steering @20
                if side == 0:
                    off[i] = 0.0                          # Nepali stays -> output
                else:
                    off[i] = side * 19.0 * ((d - 20) / 12.0) ** 1.25
        return off

    for side, col in [(-1, HI), (0, NEP), (1, MAI)]:
        off = traj(side)
        xs = X0 + off
        ys = np.array([y_of(d) for d in depths])
        pre = depths <= 20
        ax.plot(xs[pre], ys[pre], color=col, lw=3.4, solid_capstyle="round", zorder=4)
        if side == 0:
            ax.plot(xs[~pre], ys[~pre], color=col, lw=3.4, zorder=4)
            arrow(ax, xs[-1], ys[-1] + 1.5, xs[-1], yB - 3, color=col, lw=3.4, ms=20)
        else:
            ax.plot(xs[~pre], ys[~pre], color=col, lw=3.0, ls=(0, (5, 3)), zorder=4)
            arrow(ax, xs[-1], ys[-1] + 1.5, xs[-1], yB - 3, color=col, lw=3.0, ms=18)
        ax.scatter([X0 + side * 20], [yT], s=140, c=col, edgecolors="white",
                   linewidths=1.2, zorder=6)

    # steering point @ L20 -- arrow shows the APPLIED hook (alpha<0 => -v dir,
    # source->target), labelled as the intervention, not the raw vector v
    # (v=mu_src-mu_tgt is defined in panel (a)).
    arrow(ax, 176, y_of(20) - 1, X0 + 4, y_of(20) + 0.5, color=RED, lw=2.6, ms=18)
    t(ax, 184, y_of(20) + 3.0, "Steering Point\n(Layer 20)", fs=10.5, w="bold", c=RED)
    t(ax, 167, y_of(20) - 6.5, r"$h_\ell \leftarrow h_\ell + \alpha v,\ \alpha<0$",
      fs=11, c=RED)

    t(ax, X0, 25.5, "Output Logits", fs=11.5, w="bold")

    # --- compact insets along the bottom, titles placed ABOVE them (no clipping) ---
    # PCA-2D inset (schematic), bottom-left -- matches the layer-wise PCA in
    # the geometry section (NOT t-SNE)
    t(ax, 125, 19.5, "PCA-2D (layer-wise)", fs=8)
    axt = ax.inset_axes([113, 4.5, 24, 13], transform=ax.transData)
    rng = np.random.default_rng(0)
    for cx, cy, col in [(0.22, 0.62, HI), (0.6, 0.66, NEP), (0.66, 0.34, MAI)]:
        axt.scatter(cx + rng.normal(0, 0.05, 40), cy + rng.normal(0, 0.05, 40),
                    s=5, c=col, alpha=0.7, linewidths=0)
    axt.annotate("", xy=(0.55, 0.5), xytext=(0.32, 0.56),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    axt.set_xticks([]); axt.set_yticks([])
    for s in axt.spines.values():
        s.set_edgecolor("#aaa")

    # adherence bar inset (real Llama-L20 peak adherence / 5), bottom-right
    t(ax, 178, 19.5, "Language Adherence ($\\div5$)", fs=8)
    axb = ax.inset_axes([168, 4.5, 22, 13], transform=ax.transData)
    vals = [1.47 / 5, 3.23 / 5, 1.80 / 5]                 # Hindi-ctrl, Nepali, Maithili
    errs = [0.05, 0.08, 0.07]
    axb.bar([0, 1, 2], vals, yerr=errs, capsize=2,
            color=[HI, NEP, MAI], edgecolor="black", linewidth=0.5)
    axb.set_xticks([0, 1, 2]); axb.set_xticklabels(["Base", "NEP", "MAI"], fontsize=6.5)
    axb.set_ylim(0, 0.8); axb.set_yticks([0, 0.4, 0.8])
    axb.tick_params(labelsize=6, length=2)
    for s in axb.spines.values():
        s.set_edgecolor("#aaa")

    t(ax, X0, 0.6, "(b)  The Hourglass Geometry & Steering", fs=12.5, w="bold")

    # =========================================================
    # (c) STEERED OUTPUT GENERATION
    # =========================================================
    box(ax, 210, 58, 86, 30, BLUEBOX, BLUEBOXE, 1.8)
    t(ax, 253, 84.5, "Unsteered Pathway (Baseline)", fs=12.5, w="bold", c=BLUEBOXE)
    t(ax, 253, 77, "Output Context: Hindi (HI)", fs=11.5)
    ax.plot([216, 290], [73, 73], color=BLUEBOXE, lw=0.8, alpha=0.5)
    t(ax, 253, 68.5, "Generation:", fs=10.5)
    t(ax, 253, 63, "आज मौसम बहुत अच्छा है।", fs=14, deva=True)

    box(ax, 210, 14, 86, 32, GREENBOX, GREENBOXE, 1.8)
    t(ax, 253, 42.5, "Steered Pathway (Layer 20 Shift)", fs=12.5, w="bold", c=GREENBOXE)
    t(ax, 253, 35.5, "Output Context: Nepali (NEP) Adherence", fs=11)
    ax.plot([216, 290], [31, 31], color=GREENBOXE, lw=0.8, alpha=0.5)
    t(ax, 253, 26.5, "Generation:", fs=10.5)
    t(ax, 253, 21, "आज मौसम कस्तो छ?", fs=14, deva=True)

    # connectors from panel (b) outputs to (c)
    arrow(ax, 197, 70, 209, 73, color=BLUEBOXE, lw=1.8, conn="arc3,rad=-0.15")
    arrow(ax, 197, 40, 209, 30, color=GREENBOXE, lw=1.8, conn="arc3,rad=0.15")
    t(ax, 253, 1.8, "(c)  Steered Output Generation", fs=12.5, w="bold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
