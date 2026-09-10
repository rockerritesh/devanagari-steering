"""Task 2 figure: real steering vector vs random directions of equal magnitude.

Shows, for Llama npi at L20 and alpha in {-1.5, -2.0}:
  - language adherence: 10 random-seed means (strip) + their mean, vs the REAL
    vector's value, vs the alpha=0 control. Random sits at control; real spikes.
  - fluency: random stays near control (~3.9) while the real vector collapses —
    the real vector trades fluency for language identity; random does neither.

  uv run --with pandas --with pyarrow --with matplotlib \
      python code/plot_random_baseline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
PANEL = ROOT / "results" / "judge_panel"
OUT = ROOT / "results" / "figures" / "llama" / "llama_npi_random_vs_real.png"
ALPHAS = [-1.5, -2.0]
DIMS = ["language_adherence", "fluency"]


def num(df):
    for d in DIMS:
        df[d] = pd.to_numeric(df[d], errors="coerce")
    return df


def load_panel(path):
    """Read a multi_judge panel parquet; expose panel-mean scores under DIMS names."""
    df = pd.read_parquet(path)
    for d in DIMS:
        df[d] = pd.to_numeric(df[f"panel_mean__{d}"], errors="coerce")
    return df


MET = ROOT / "results" / "robust_metrics"
PEAK = -2.0   # the adherence-peak operating point


def metric_val(df, alpha, col, agg="mean"):
    s = pd.to_numeric(df[df.alpha == alpha][col], errors="coerce")
    return float(s.median() if agg == "median" else s.mean())


def main():
    """Multi-metric direction-specificity profile at the adherence peak (alpha=-2.0):
    control vs random-direction vs the real vector, across the full metric suite. The
    real vector shifts every language metric; random directions move none of them."""
    # judge panel (adherence, fluency)
    rp = load_panel(PANEL / "llama_random_panel.parquet")
    fp = load_panel(PANEL / "llama_finer30_panel.parquet"); fp = fp[fp.target_lang == "npi"]
    # reference-free metrics
    rm = pd.read_parquet(MET / "llama_random_metrics.parquet")
    fm = pd.read_parquet(MET / "llama_finer30_metrics.parquet"); fm = fm[fm.target_lang == "npi"]

    # (label, control, random, real) at PEAK; control taken at alpha=0
    specs = [
        ("Adherence\n(1--5)$\\uparrow$", metric_val(fp, 0.0, "language_adherence"),
         metric_val(rp, PEAK, "language_adherence"), metric_val(fp, PEAK, "language_adherence")),
        ("Fluency\n(1--5)", metric_val(fp, 0.0, "fluency"),
         metric_val(rp, PEAK, "fluency"), metric_val(fp, PEAK, "fluency")),
        ("Semantic\n(LaBSE cos)", metric_val(rm, 0.0, "sem_labse"),
         metric_val(rm, PEAK, "sem_labse"), metric_val(fm, PEAK, "sem_labse")),
        ("Pseudo-PPL\n(IndicBERT)$\\uparrow$", metric_val(rm, 0.0, "indicbert_ppl", "median"),
         metric_val(rm, PEAK, "indicbert_ppl", "median"), metric_val(fm, PEAK, "indicbert_ppl", "median")),
        ("Distinct-2\n(diversity)", metric_val(rm, 0.0, "distinct2"),
         metric_val(rm, PEAK, "distinct2"), metric_val(fm, PEAK, "distinct2")),
    ]
    COLORS = ["#1f77b4", "#9a9a9a", "#d62728"]; NAMES = ["control", "random", "real vector"]
    fig, axes = plt.subplots(1, len(specs), figsize=(12.5, 3.1))
    for ax, (lab, c, rnd, real) in zip(axes, specs):
        vals = [c, rnd, real]
        bars = ax.bar([0, 1, 2], vals, color=COLORS, edgecolor="black", linewidth=0.6, width=0.72)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=8.5, xytext=(0, 1), textcoords="offset points")
        ax.set_title(lab, fontsize=9.5)
        ax.set_xticks([]); ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(0, max(vals) * 1.22)
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=COLORS[i], edgecolor="black") for i in range(3)], NAMES,
               loc="upper center", ncol=3, fontsize=9.5, framealpha=0.95, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
