"""Three-step method-overview schematic for the paper.

Step 1  Preparation & Extraction  : run parallel source/target passages through
        the frozen LLM, pull the per-layer residual-stream states h_l.
Step 2  Vector Computation         : per-layer centroids, v_l = mu_src - mu_tgt
        averaged over the N parallel pairs.
Step 3  Inference-Time Steering    : source-language prompt; a forward hook on
        block l adds alpha*v_l (alpha<0) -> target-language output.

Pure schematic (no data); matches the notation in main.tex.

  uv run --with matplotlib --with numpy python code/plot_method_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "method_overview.png"

# palette
C_LLM = "#fdf0d5"      # warm LLM body
C_LLM_E = "#e0a96d"
C_SRC = "#cfe8ff"      # source (Hindi anchor) blue
C_SRC_E = "#3d7fb5"
C_TGT = "#ffd6d6"      # target red
C_TGT_E = "#c0504d"
C_VEC = "#f4c542"      # language-vector gold
C_VEC_E = "#b8860b"
C_STEP = "#e8eef7"     # vector-computation panel boxes
C_STEP_E = "#9aa9c4"
C_OUT = "#d7f0d7"      # steered-output green
C_OUT_E = "#4a9e4a"
C_HL = "#fff6b0"       # highlighted "Layer l"


def box(ax, x, y, w, h, fc, ec, text="", fs=9, weight="normal", rounded=0.02,
        ha="center", va="center", lw=1.3, tcol="black", style="round"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rounded}",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2 if ha == "center" else x + 0.4, y + h / 2, text,
                ha=ha, va=va, fontsize=fs, fontweight=weight, color=tcol, zorder=4)
    return p


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.6, style="-|>", ms=14,
          conn="arc3,rad=0", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        lw=lw, color=color, connectionstyle=conn, zorder=5, ls=ls)
    ax.add_patch(a)


def llm_block(ax, x, y, w, h, label="LLM", n_layers=4, hl_idx=2):
    """Draw an LLM body with stacked layer bars, one highlighted."""
    box(ax, x, y, w, h, C_LLM, C_LLM_E, "", rounded=0.02, lw=1.6)
    ax.text(x + w / 2, y + h - 1.6, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", zorder=4)
    bw, bh = w * 0.74, (h - 4.2) / (n_layers + 0.6) * 0.7
    bx = x + (w - bw) / 2
    gap = (h - 4.2 - n_layers * bh) / (n_layers + 1)
    for i in range(n_layers):
        by = y + gap + i * (bh + gap)
        hl = (i == hl_idx)
        box(ax, bx, by, bw, bh, C_HL if hl else "#fbe7c2",
            C_VEC_E if hl else C_LLM_E, "", rounded=0.01,
            lw=1.5 if hl else 1.0)
        if hl:
            ax.text(bx + bw / 2, by + bh / 2, r"Layer $\ell$", ha="center",
                    va="center", fontsize=8.5, fontweight="bold", zorder=6)
    return bx, bw, gap, bh


def main():
    fig, ax = plt.subplots(figsize=(13.6, 6.0))
    ax.set_xlim(0, 138)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # ---- panel separators + titles ----
    for sx in (45.5, 88):
        ax.axvline(sx, color="#cccccc", lw=1.0, ls=(0, (4, 4)))
    titles = [(22, "Step 1: Preparation & Extraction"),
              (66.5, "Step 2: Vector Computation"),
              (113, "Step 3: Inference-Time Steering")]
    for tx, tt in titles:
        ax.text(tx, 60, tt, ha="center", va="center", fontsize=12.5,
                fontweight="bold")

    # =========================================================
    # STEP 1 : extraction
    # =========================================================
    bx, bw, gap, bh = llm_block(ax, 13, 18, 20, 26, "LLM", 4, 2)
    ax.text(23, 46.5, "Extract  $h_\\ell$  states", ha="center", va="center",
            fontsize=9.5, style="italic")
    # round-trip extraction arrows over the top
    arrow(ax, 12.5, 40, 7, 50, color="#333", conn="arc3,rad=0.35", lw=1.5)
    arrow(ax, 39, 50, 33.5, 40, color="#333", conn="arc3,rad=0.35", lw=1.5)
    # mini activation bars top-left/right
    for j, (mx, col, ce) in enumerate([(5.5, C_SRC, C_SRC_E), (34.5, C_TGT, C_TGT_E)]):
        for k in range(4):
            box(ax, mx + k * 1.4, 50.5, 1.1, 4.0, col, ce, "", rounded=0.0, lw=0.8)

    # source / target data boxes feeding in
    box(ax, 3.5, 4, 17.5, 10.5, C_SRC, C_SRC_E, "", rounded=0.02)
    ax.text(12.2, 12.0, "Source (Hindi, $N$ pairs)", ha="center", fontsize=8.6,
            fontweight="bold", color="#1f4d6b")
    ax.text(12.2, 8.1, "$Q^{\\rm src}_1\\;A^{\\rm src}_1$\n$Q^{\\rm src}_2\\;A^{\\rm src}_2$  $\\cdots$",
            ha="center", va="center", fontsize=8.0, color="#1f4d6b")
    box(ax, 25, 4, 17.5, 10.5, C_TGT, C_TGT_E, "", rounded=0.02)
    ax.text(33.7, 12.0, "Target ($N$ pairs)", ha="center", fontsize=8.6,
            fontweight="bold", color="#7a2e2c")
    ax.text(33.7, 8.1, "$Q^{\\rm tgt}_1\\;A^{\\rm tgt}_1$\n$Q^{\\rm tgt}_2\\;A^{\\rm tgt}_2$  $\\cdots$",
            ha="center", va="center", fontsize=8.0, color="#7a2e2c")
    ax.text(23, 1.6, "Parallel data pairs", ha="center", fontsize=8.6, style="italic")
    arrow(ax, 12.2, 14.5, 17.5, 18.3, color=C_SRC_E, lw=1.6)
    arrow(ax, 33.7, 14.5, 28.5, 18.3, color=C_TGT_E, lw=1.6)

    # =========================================================
    # STEP 2 : vector computation
    # =========================================================
    # two rows of (mu_src - mu_tgt = delta)
    def actbars(x, y, col, ce, n=4):
        for k in range(n):
            box(ax, x + k * 1.25, y, 1.0, 4.2, col, ce, "", rounded=0.0, lw=0.8)
    for r, yy in enumerate([47, 38]):
        actbars(49, yy, C_SRC, C_SRC_E)
        ax.text(58.0, yy + 2.1, "$-$", ha="center", va="center", fontsize=13)
        actbars(60, yy, C_TGT, C_TGT_E)
        ax.text(69.0, yy + 2.1, "$=$", ha="center", va="center", fontsize=13)
        actbars(71, yy, "#d9c7f0", "#7b5ea7")
        ax.text(49 - 1.0, yy + 2.1,
                f"$\\mu^{{(i)}}_{{\\rm src}}$" if r == 0 else "", ha="right",
                va="center", fontsize=9)
    ax.text(54.2, 53.0, "per-sample, per-layer hidden states", ha="center",
            fontsize=8.2, style="italic")
    ax.text(76.5, 49.1, "$\\Delta_1$", ha="left", va="center", fontsize=10)
    ax.text(76.5, 40.1, "$\\Delta_2$", ha="left", va="center", fontsize=10)
    ax.text(67, 33.3, "$\\vdots$", ha="center", fontsize=13)

    # aggregate box
    ax.text(66.5, 28.5, "Aggregate $N$ samples", ha="center", fontsize=8.8,
            style="italic")
    box(ax, 50, 19.5, 33, 7.2, C_STEP, C_STEP_E,
        "$\\Delta^{(i)}_\\ell=\\mu^{(i)}_{{\\rm src},\\ell}-\\mu^{(i)}_{{\\rm tgt},\\ell}$",
        fs=10.5, rounded=0.03)
    box(ax, 50, 9.5, 33, 7.2, C_STEP, C_STEP_E,
        "$v_\\ell=\\dfrac{1}{N}\\sum_i \\Delta^{(i)}_\\ell$", fs=11.5, rounded=0.03)
    arrow(ax, 66.5, 19.3, 66.5, 16.9, color="#444", lw=1.6)
    arrow(ax, 66.5, 36.5, 66.5, 27.0, color="#444", lw=1.4, style="-|>")

    # the language vector chip
    box(ax, 57, 2.0, 19, 5.2, C_VEC, C_VEC_E, "", rounded=0.02, lw=1.6)
    for k in range(7):
        box(ax, 58.4 + k * 2.3, 3.0, 1.9, 3.2, "#ffe89a", C_VEC_E, "", rounded=0.0, lw=0.7)
    ax.text(66.5, 0.4, "Language vector  $v_\\ell$", ha="center", fontsize=8.8,
            fontweight="bold")
    arrow(ax, 66.5, 9.3, 66.5, 7.4, color=C_VEC_E, lw=1.8)

    # =========================================================
    # STEP 3 : inference-time steering
    # =========================================================
    # prompt box
    box(ax, 92, 50, 42, 8.0, "white", "#333", "", rounded=0.02, lw=1.4)
    ax.text(113, 54.0,
            "Prompt:  [Hindi few-shot]  +  [Target question]",
            ha="center", va="center", fontsize=8.8)
    ax.text(99.5, 47.3, "Forward pass", ha="center", fontsize=8.2, style="italic")
    arrow(ax, 99.5, 50, 99.5, 44.3, color="#333", lw=1.6)

    # LLM with hooked layer
    sbx, sbw, sgap, sbh = llm_block(ax, 92, 17, 21, 27, "LLM", 4, 2)
    # injection equation by the highlighted layer
    hl_y = 17 + sgap + 2 * (sbh + sgap) + sbh / 2
    box(ax, 114.5, hl_y - 2.7, 22, 5.4, C_OUT, C_OUT_E,
        "$h'_\\ell=h_\\ell+\\alpha\\, v_\\ell$", fs=10.5, rounded=0.04, lw=1.4)
    ax.text(125.5, hl_y - 3.9, "$\\alpha<0$:  src $\\rightarrow$ tgt", ha="center",
            fontsize=7.8, style="italic")

    # language vector chip on the right feeding the hook
    box(ax, 117, hl_y + 6.5, 17, 5.0, C_VEC, C_VEC_E, "", rounded=0.02, lw=1.4)
    for k in range(6):
        box(ax, 118.2 + k * 2.4, hl_y + 7.4, 2.0, 3.2, "#ffe89a", C_VEC_E, "",
            rounded=0.0, lw=0.7)
    ax.text(125.5, hl_y + 12.3, "Language vector $v_\\ell$", ha="center",
            fontsize=8.0, fontweight="bold")
    arrow(ax, 125.5, hl_y + 6.3, 117.2, hl_y + 0.2, color=C_VEC_E, lw=1.6,
          conn="arc3,rad=0.25")
    # arrow from injection eq into the highlighted layer bar
    arrow(ax, 114.3, hl_y, sbx + sbw + 0.3, hl_y, color=C_OUT_E, lw=1.6)

    # steered output
    box(ax, 96.5, 4.5, 33, 7.0, C_OUT, C_OUT_E, "Steered output  (target language)",
        fs=9.5, rounded=0.03, lw=1.4)
    arrow(ax, 102.5, 16.8, 102.5, 11.7, color=C_OUT_E, lw=1.8)

    fig.suptitle("Language-vector steering: extract parallel residual states $\\to$ "
                 "centroid-difference vector $v_\\ell=\\mu_{\\rm src}-\\mu_{\\rm tgt}$ $\\to$ "
                 "inject at inference",
                 fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
