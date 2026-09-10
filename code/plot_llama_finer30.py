"""Plots for the Llama L20 finer-α sweep (task 1, 2026-06-01).

Produces two figures:
  1. results/figures/llama/llama_pareto_l20_finer30.png
     2×3: top row = lang-adherence vs fluency Pareto (one panel per target);
     bottom row = all score dims vs α. Mirrors code/plot_pareto.py (Aya) but
     with adherence y-limits widened because Llama npi reaches 3.23.
  2. results/figures/llama/llama_vs_aya_adherence_l20vL22.png
     1×3: language-adherence vs α for Llama (L20) and Aya (L22) overlaid per
     target, with each model's ~ceiling annotated. This is the headline
     cross-model figure (finding #15).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEN = PROJECT_ROOT / "results" / "generations"
LLAMA = GEN / "llama_all_l20_finer30_scored.parquet"
AYA = GEN / "aya_all_l22_finer30_scored.parquet"
OUTDIR = PROJECT_ROOT / "results" / "figures" / "llama"

LANG_LABELS = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}
LANG_COLOURS = {"npi": "#2ca02c", "mai": "#1f77b4", "bho": "#ff7f0e"}
DIMS = ["language_adherence", "fluency", "faithfulness", "coherence", "overall_quality"]
TARGETS = ["npi", "mai", "bho"]


def load_means(path):
    df = pd.read_parquet(path)
    for d in DIMS:
        df[d] = pd.to_numeric(df[d], errors="coerce")
    return df.groupby(["target_lang", "alpha"])[DIMS].mean().reset_index()


def plot_pareto(means, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    adh_max = means["language_adherence"].max()

    for col, tgt in enumerate(TARGETS):
        ax = axes[0, col]
        sub = means[means["target_lang"] == tgt].sort_values("alpha", ascending=False)
        alphas, x, y = sub["alpha"].values, sub["fluency"].values, sub["language_adherence"].values
        a_norm = (alphas - alphas.min()) / max(alphas.max() - alphas.min(), 1e-6)
        colors = plt.get_cmap("plasma")(1 - a_norm)
        ax.plot(x, y, "-", color="gray", alpha=0.5, zorder=2)
        for xi, yi, ci, ai in zip(x, y, colors, alphas):
            ax.scatter(xi, yi, color=ci, s=180, edgecolor="black", linewidths=1.0, zorder=4)
            ax.annotate(f"α={ai:+.2f}", (xi, yi), xytext=(8, 6), textcoords="offset points",
                        fontsize=9, zorder=5)
        bi = int(np.argmax(y))
        ax.scatter(x[bi], y[bi], marker="*", s=380, facecolor="gold",
                   edgecolor="black", linewidths=1.4, zorder=6)
        ci_ = int(np.argmax(np.isclose(alphas, 0.0)))
        ax.scatter(x[ci_], y[ci_], marker="s", s=380, facecolor="none",
                   edgecolor="black", linewidths=1.4, zorder=5)
        ax.set_xlabel("Fluency (mean, 1-5)")
        if col == 0:
            ax.set_ylabel("Language adherence (mean, 1-5)")
        ax.set_title(f"{LANG_LABELS[tgt]} (target={tgt})", fontsize=11, color=LANG_COLOURS[tgt])
        ax.set_xlim(0.8, 5)
        ax.set_ylim(0.9, max(2.3, adh_max + 0.35))
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.98, "★ peak adherence    ☐ α=0 control", transform=ax.transAxes,
                fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    for col, tgt in enumerate(TARGETS):
        ax = axes[1, col]
        sub = means[means["target_lang"] == tgt].sort_values("alpha")
        alphas = sub["alpha"].values
        for cs, label, lw in [("language_adherence", "Lang adherence", 2.5),
                              ("fluency", "Fluency", 2.0), ("overall_quality", "Overall", 2.0),
                              ("faithfulness", "Faithfulness", 1.2), ("coherence", "Coherence", 1.2)]:
            ax.plot(alphas, sub[cs].values, marker="o", markersize=5, linewidth=lw, label=label)
        ax.axvline(0, color="black", alpha=0.3, linewidth=0.8, linestyle="--")
        ax.set_xlabel("α")
        if col == 0:
            ax.set_ylabel("Mean score (1-5)")
        ax.set_title(f"{LANG_LABELS[tgt]}: scores vs α", fontsize=10)
        ax.set_ylim(0.9, 5.0)
        ax.grid(True, alpha=0.25)
        if col == 2:
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

    fig.suptitle("Llama-3.1-8B steering at L20 (n=30 eval prompts × 5 α): "
                 "language-adherence vs fluency Pareto frontier", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_cross_model(llama, aya, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for col, tgt in enumerate(TARGETS):
        ax = axes[col]
        for means, name, color, marker in [(llama, "Llama-3.1 (L20)", "#d62728", "o"),
                                           (aya, "Aya-23 (L22)", "#1f77b4", "s")]:
            sub = means[means["target_lang"] == tgt].sort_values("alpha")
            ax.plot(sub["alpha"].values, sub["language_adherence"].values,
                    marker=marker, markersize=6, linewidth=2.2, color=color, label=name)
            peak = sub["language_adherence"].max()
            ax.axhline(peak, color=color, alpha=0.35, linewidth=1.0, linestyle=":")
        ax.axhline(1.57, color="gray", alpha=0.6, linewidth=1.0, linestyle="--")
        ax.text(ax.get_xlim()[0], 1.6, "Aya ~1.57 ceiling", fontsize=8, color="gray", va="bottom")
        ax.set_xlabel("α (steering coefficient)")
        if col == 0:
            ax.set_ylabel("Language adherence (mean, 1-5)")
        ax.set_title(f"{LANG_LABELS[tgt]} (target={tgt})", fontsize=11, color=LANG_COLOURS[tgt])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    fig.suptitle("Cross-model: single-layer adherence ceiling is model-dependent — "
                 "Llama breaks it for Nepali (3.23) while Aya caps at ~1.57",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    llama = load_means(LLAMA)
    plot_pareto(llama, OUTDIR / "llama_pareto_l20_finer30.png")
    if AYA.exists():
        aya = load_means(AYA)
        plot_cross_model(llama, aya, OUTDIR / "llama_vs_aya_adherence_l20vL22.png")
    else:
        print(f"(skipping cross-model fig — {AYA} not found)")


if __name__ == "__main__":
    main()
