"""Plot the language-adherence vs fluency Pareto frontier from the finer α sweep.

Reads `results/generations/aya_all_l22_finer30_scored.parquet` (5 alphas × 3 targets
× 30 prompts × 5 score dims) and produces:

  results/figures/aya/aya_pareto_l22_finer30.png

Layout: 1×3 panels (one per target). X = mean fluency, Y = mean language adherence,
points coloured by α with α=0 control marked. The best α per target is starred.
A second panel row shows mean overall_quality vs α for direct comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORED = PROJECT_ROOT / "results" / "generations" / "aya_all_l22_finer30_scored.parquet"
OUT = PROJECT_ROOT / "results" / "figures" / "aya" / "aya_pareto_l22_finer30.png"

LANG_LABELS = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}
LANG_COLOURS = {"npi": "#2ca02c", "mai": "#1f77b4", "bho": "#ff7f0e"}


def main():
    df = pd.read_parquet(SCORED)
    means = (df.groupby(["target_lang", "alpha"])
               [["language_adherence", "fluency", "faithfulness",
                 "coherence", "overall_quality"]]
               .mean().reset_index())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    targets = ["npi", "mai", "bho"]

    # Top row: lang_adherence vs fluency Pareto
    for col, tgt in enumerate(targets):
        ax = axes[0, col]
        sub = means[means["target_lang"] == tgt].sort_values("alpha", ascending=False)
        alphas = sub["alpha"].values
        x = sub["fluency"].values
        y = sub["language_adherence"].values

        # Colour by α — viridis from 0 (light) to most-negative (dark).
        a_norm = (alphas - alphas.min()) / max(alphas.max() - alphas.min(), 1e-6)
        colors = plt.get_cmap("plasma")(1 - a_norm)

        ax.plot(x, y, "-", color="gray", alpha=0.5, zorder=2)
        for xi, yi, ci, ai in zip(x, y, colors, alphas):
            ax.scatter(xi, yi, color=ci, s=180, edgecolor="black",
                       linewidths=1.0, zorder=4)
            ax.annotate(f"α={ai:+.2f}", (xi, yi), xytext=(8, 6),
                        textcoords="offset points", fontsize=9,
                        color="black", zorder=5)

        # Star the alpha with highest language_adherence.
        best_idx = int(np.argmax(y))
        ax.scatter(x[best_idx], y[best_idx], marker="*", s=380,
                   facecolor="gold", edgecolor="black", linewidths=1.4, zorder=6)

        # Mark control (α=0) with a black square outline.
        ctrl_idx = int(np.argmax(np.isclose(alphas, 0.0)))
        ax.scatter(x[ctrl_idx], y[ctrl_idx], marker="s", s=380,
                   facecolor="none", edgecolor="black", linewidths=1.4, zorder=5)

        ax.set_xlabel("Fluency (mean, 1-5)")
        if col == 0:
            ax.set_ylabel("Language adherence (mean, 1-5)")
        ax.set_title(f"{LANG_LABELS[tgt]} (target={tgt})", fontsize=11,
                     color=LANG_COLOURS[tgt])
        ax.set_xlim(0.5, 5)
        ax.set_ylim(0.9, 2.2)
        ax.grid(True, alpha=0.25)
        # legend annotations
        ax.text(0.02, 0.98, "★ peak adherence    ☐ α=0 control",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # Bottom row: each score-dim vs α
    for col, tgt in enumerate(targets):
        ax = axes[1, col]
        sub = means[means["target_lang"] == tgt].sort_values("alpha")
        alphas = sub["alpha"].values
        for col_score, label, lw in [
            ("language_adherence", "Lang adherence", 2.5),
            ("fluency", "Fluency", 2.0),
            ("overall_quality", "Overall", 2.0),
            ("faithfulness", "Faithfulness", 1.2),
            ("coherence", "Coherence", 1.2),
        ]:
            ax.plot(alphas, sub[col_score].values, marker="o", markersize=5,
                    linewidth=lw, label=label)
        ax.axvline(0, color="black", alpha=0.3, linewidth=0.8, linestyle="--")
        ax.set_xlabel("α")
        if col == 0:
            ax.set_ylabel("Mean score (1-5)")
        ax.set_title(f"{LANG_LABELS[tgt]}: scores vs α", fontsize=10)
        ax.set_ylim(0.9, 4.2)
        ax.grid(True, alpha=0.25)
        if col == 2:
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    fig.suptitle(
        "Aya-23-8B steering at L22 (n=30 eval prompts × 5 α): "
        "language-adherence vs fluency Pareto frontier",
        fontsize=12.5, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")

    # Print compact summary
    print("\n=== Best α per target (by language_adherence) ===")
    for tgt in targets:
        sub = means[means["target_lang"] == tgt].sort_values("alpha", ascending=False)
        best = sub.loc[sub["language_adherence"].idxmax()]
        ctrl = sub.loc[np.isclose(sub["alpha"], 0.0)].iloc[0]
        d_adh = best["language_adherence"] - ctrl["language_adherence"]
        d_flu = best["fluency"] - ctrl["fluency"]
        d_ovr = best["overall_quality"] - ctrl["overall_quality"]
        print(f"  {LANG_LABELS[tgt]:>10s}  α*={best['alpha']:+.2f}   "
              f"adh={best['language_adherence']:.2f} ({d_adh:+.2f})   "
              f"flu={best['fluency']:.2f} ({d_flu:+.2f})   "
              f"ovr={best['overall_quality']:.2f} ({d_ovr:+.2f})")


if __name__ == "__main__":
    main()
