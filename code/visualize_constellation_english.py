"""English-inclusive full-depth centroid constellation (Llama, L0-L32).

Same convention as visualize_constellations.py's full-depth figure, but with
English added as a 5th language (Llama only — Aya has no EN memory bank). One
shared UMAP (cosine) over all (lang x layer x sample) points; per-language
polyline through the layer-centroids; square at L0; star per layer; dashed
gray gap lines from Hindi to each other language. English sits apart from the
Devanagari cluster across depth — the geometry behind Hindi >> English (finding #22).

  uv run --with torch --with numpy --with matplotlib --with umap-learn \
      python code/visualize_constellation_english.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

ROOT = Path(__file__).resolve().parent.parent
MODEL = "llama"
LANGS = ["hin", "mai", "npi", "bho", "en"]
COL = {"hin": "#d62728", "mai": "#1f77b4", "npi": "#2ca02c", "bho": "#ff7f0e", "en": "#9467bd"}
LAB = {"hin": "Hindi (source)", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri", "en": "English"}


def load():
    out = {}
    for lg in LANGS:
        out[lg] = torch.load(ROOT / "results" / "memory_banks" / MODEL / f"{lg}.pt",
                             map_location="cpu", weights_only=False)["vectors"].float().numpy()
    return out


def main():
    data = load()
    layers = list(range(33))
    n = data["hin"].shape[0]

    chunks = [data[lg][:, l, :] for lg in LANGS for l in layers]
    combined = np.concatenate(chunks, axis=0)
    print(f"fitting UMAP on {combined.shape[0]} points ({len(LANGS)} langs x {len(layers)} layers x {n}) ...")
    coords = umap.UMAP(n_neighbors=25, min_dist=0.15, n_components=2,
                       metric="cosine", random_state=0).fit_transform(combined)

    pts = {}; cur = 0
    for lg in LANGS:
        for l in layers:
            pts[(lg, l)] = coords[cur:cur + n]; cur += n

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor("white")
    # background samples
    for (lg, _l), p in pts.items():
        ax.scatter(p[:, 0], p[:, 1], s=2.0, alpha=0.05, color=COL[lg], linewidths=0)

    cents = {lg: np.stack([pts[(lg, l)].mean(0) for l in layers]) for lg in LANGS}
    # dashed Hindi->other gaps
    for lg in ["mai", "npi", "bho", "en"]:
        for li in range(len(layers)):
            hp, tp = cents["hin"][li], cents[lg][li]
            ax.plot([hp[0], tp[0]], [hp[1], tp[1]], linestyle=(0, (3, 3)),
                    color="gray", linewidth=0.5, alpha=0.30, zorder=3)
    # per-language trajectory
    for lg in LANGS:
        c = cents[lg]
        ax.plot(c[:, 0], c[:, 1], color=COL[lg], linewidth=2.0, alpha=0.95, zorder=5)
        ax.scatter(c[:, 0], c[:, 1], color=COL[lg], s=110, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=6, label=LAB[lg])
        ax.scatter(c[0, 0], c[0, 1], color=COL[lg], s=150, marker="s",
                   edgecolors="black", linewidths=0.9, zorder=7)
    # layer labels on Hindi + English paths
    for li, l in enumerate(layers):
        for lg, dy in [("hin", 4), ("en", -10)]:
            p = cents[lg][li]
            ax.annotate(f"L{l}", (p[0], p[1]), xytext=(4, dy), textcoords="offset points",
                        fontsize=6.5, color=COL[lg], fontweight="bold", alpha=0.8, zorder=8)

    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    # in-image title removed: the LaTeX \caption covers it (journal requirement)
    ax.grid(True, alpha=0.2); ax.legend(loc="best", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "llama" / "llama_constellation_english_full_L0-L32.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
