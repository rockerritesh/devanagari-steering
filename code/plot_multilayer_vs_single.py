"""Task 3 figure: multi-layer composite vs single-layer L20 steering (Llama).

Grouped bars of PEAK language adherence (and fluency at that peak) per target,
single-layer L20 vs multi-layer L18+L20+L22 (norm-eq). Shows multi-layer does
NOT break the ~1.8-2.0 mai/bho wall, and a dashed line marks the "mostly target"
(>=3) threshold neither method crosses for mai/bho.

  uv run --with pandas --with pyarrow --with matplotlib \
      python code/plot_multilayer_vs_single.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
OUT = ROOT / "results" / "figures" / "llama" / "llama_multilayer_vs_single.png"
TARGETS = ["npi", "mai", "bho"]
LABELS = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}


def peak(path):
    df = pd.read_parquet(path)
    for d in ("language_adherence", "fluency"):
        df[d] = pd.to_numeric(df[d], errors="coerce")
    out = {}
    for t in TARGETS:
        g = df[df.target_lang == t].groupby("alpha")[["language_adherence", "fluency"]].mean()
        a = g["language_adherence"].idxmax()
        out[t] = (g.loc[a, "language_adherence"], g.loc[a, "fluency"], a)
    return out


def main():
    single = peak(GEN / "llama_all_l20_finer30_scored.parquet")
    multi = peak(GEN / "llama_all_multiL182022_eval_scored.parquet")

    x = np.arange(len(TARGETS)); w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, idx, title, thr in [(axes[0], 0, "Peak language adherence (1–5)", 3.0),
                                (axes[1], 1, "Fluency at adherence-peak (1–5)", None)]:
        s = [single[t][idx] for t in TARGETS]
        m = [multi[t][idx] for t in TARGETS]
        b1 = ax.bar(x - w/2, s, w, label="single-layer L20", color="#1f77b4")
        b2 = ax.bar(x + w/2, m, w, label="multi-layer L18+20+22", color="#d62728")
        for bars in (b1, b2):
            for b in bars:
                ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                            ha="center", va="bottom", fontsize=9)
        if thr is not None:
            ax.axhline(thr, color="green", linestyle="--", linewidth=1.3, alpha=0.7)
            ax.text(len(TARGETS)-0.5, thr+0.05, "“mostly target” (≥3)", color="green",
                    fontsize=9, ha="right")
        ax.set_xticks(x); ax.set_xticklabels([LABELS[t] for t in TARGETS])
        ax.set_ylim(0, 5); ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9.5)

    fig.suptitle("Task 3 — multi-layer composite does NOT break the single-layer wall (Llama)\n"
                 "mai/bho stay ~1.8–2.0 (below “mostly target”); npi unchanged ≈3.2 — the L20 onset "
                 "is already near-optimal",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    for t in TARGETS:
        print(f"  {t}: single adh={single[t][0]:.2f}@a{single[t][2]}  multi adh={multi[t][0]:.2f}@a{multi[t][2]}")


if __name__ == "__main__":
    main()
