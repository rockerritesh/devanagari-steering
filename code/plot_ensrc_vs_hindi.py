"""Task 5 figure: Hindi-source vs EN-source steering (Llama, L20).

Left: peak language adherence per target, Hindi-source vs EN-source, with the
"mostly target" (>=3) line. Right: adherence vs alpha for Nepali — Hindi-source
rises to 3.23 while EN-source stays flat (~1.4) then degrades. English, the
dominant pretraining language, fails as a springboard; Hindi (script/linguistic
proximity) succeeds.

  uv run --with pandas --with pyarrow --with matplotlib \
      python code/plot_ensrc_vs_hindi.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
PANEL = ROOT / "results" / "judge_panel"
OUT = ROOT / "results" / "figures" / "llama" / "llama_ensrc_vs_hindi.png"
TARGETS = ["npi", "mai", "bho"]
LABELS = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}


def load(path):
    """Panel parquet; expose panel-mean adherence under the plain column name."""
    df = pd.read_parquet(path)
    df["language_adherence"] = pd.to_numeric(df["panel_mean__language_adherence"], errors="coerce")
    return df


def main():
    hi = load(PANEL / "llama_finer30_panel.parquet")
    en = load(PANEL / "llama_ensrc_panel.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: grouped bars of peak adherence ---
    ax = axes[0]
    hp = [hi[hi.target_lang == t].groupby("alpha")["language_adherence"].mean().max() for t in TARGETS]
    ep = [en[en.target_lang == t].groupby("alpha")["language_adherence"].mean().max() for t in TARGETS]
    x = np.arange(len(TARGETS)); w = 0.36
    b1 = ax.bar(x - w/2, hp, w, label="Hindi-source", color="#d62728")
    b2 = ax.bar(x + w/2, ep, w, label="English-source", color="#7f7f7f")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=10)
    ax.axhline(3.0, color="green", linestyle="--", linewidth=1.3, alpha=0.7)
    ax.text(len(TARGETS)-0.5, 3.05, "“mostly target” (≥3)", color="green", fontsize=9, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[t] for t in TARGETS])
    ax.set_ylim(0, 5); ax.set_ylabel("Peak language adherence (1–5)")
    ax.set_title("Peak adherence: Hindi-source vs English-source", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25); ax.legend(loc="upper right", fontsize=10)

    # --- right: adherence vs alpha for Nepali ---
    ax = axes[1]
    for df, name, c in [(hi, "Hindi-source", "#d62728"), (en, "English-source", "#7f7f7f")]:
        g = df[df.target_lang == "npi"].groupby("alpha")["language_adherence"].mean().sort_index()
        ax.plot(g.index, g.values, marker="o", markersize=6, linewidth=2.3, color=c, label=name)
    ax.axhline(3.0, color="green", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("α (steering coefficient)"); ax.set_ylabel("Language adherence (1–5)")
    ax.set_title("Nepali: adherence vs α", fontsize=12)
    ax.set_ylim(0.8, 3.6); ax.grid(True, alpha=0.25); ax.legend(loc="upper center", fontsize=10)

    # in-image overall title removed: the LaTeX \caption covers it (journal requirement)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    for t in TARGETS:
        print(f"  {t}: Hindi {hp[TARGETS.index(t)]:.2f}  EN {ep[TARGETS.index(t)]:.2f}")


if __name__ == "__main__":
    main()
