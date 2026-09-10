"""English-inclusive constellation (Llama) — visualizes WHY Hindi > English source.

Two figures:
  llama_constellation_english_pca.png — per-sample PCA-2D at selected layers with
      ENGLISH added as a 5th language. English forms a separate cluster; Hindi sits
      among the Devanagari sisters.
  llama_english_vs_hindi_proximity.png — per-layer cosine of each anchor (Hindi,
      English) to each Devanagari target. Hindi is consistently closer, especially
      at the steering layer (L20) — the geometric reason Hindi transfers and English
      doesn't (finding #22).

  uv run --with torch --with numpy --with scikit-learn --with matplotlib \
      python code/visualize_english_constellation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
MODEL = "llama"
STEER = 20
LANGS = ["en", "hin", "mai", "npi", "bho"]
LABELS = {"en": "English", "hin": "Hindi", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri"}
COL = {"en": "#9467bd", "hin": "#e6194B", "mai": "#4363d8", "npi": "#3cb44b", "bho": "#f58231"}
TARGETS = ["npi", "mai", "bho"]


def load():
    banks = {}
    for lg in LANGS:
        path = ROOT / "results" / "memory_banks" / MODEL / f"{lg}.pt"
        if path.exists():
            banks[lg] = torch.load(path, map_location="cpu", weights_only=False)["vectors"].float().numpy()
        else:
            banks[lg] = None  # e.g. en.pt not pulled yet; centroid recovered below
    return banks, banks["hin"].shape[1]


def en_centroid_recovered():
    """Recover mu_en per layer without en.pt: mu_en = v_{en->tgt} + mu_tgt."""
    v = torch.load(ROOT / "results" / "constellations_en" / MODEL / "npi.pt",
                   map_location="cpu", weights_only=False)["steering"].float().numpy()
    mu_npi = torch.load(ROOT / "results" / "memory_banks" / MODEL / "npi.pt",
                        map_location="cpu", weights_only=False)["vectors"].float().numpy().mean(0)
    return v + mu_npi                                   # (L+1, H)


def fig_pca(banks, nL, out):
    layers = [0, 14, STEER, nL - 1]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.3 * len(layers), 3.6))
    for ax, l in zip(axes, layers):
        X = np.concatenate([banks[lg][:, l] for lg in LANGS], axis=0)
        pts = PCA(n_components=2, random_state=0).fit_transform(X)
        n = banks["hin"].shape[0]
        for i, lg in enumerate(LANGS):
            s = pts[i * n:(i + 1) * n]
            ax.scatter(s[:, 0], s[:, 1], s=7, alpha=0.6, color=COL[lg],
                       label=LABELS[lg], linewidths=0)
        tag = {0: "L0 · input", 14: "L14 · merged", STEER: f"L{STEER} · steer",
               nL - 1: f"L{nL-1} · output"}.get(l, f"L{l}")
        ax.set_title(tag, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if l == 0:
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9, markerscale=1.6)
    fig.suptitle("Llama-3.1 with English added: per-sample PCA by layer. English (purple) clusters "
                 "apart;\nHindi (red) sits among the Devanagari sisters — why Hindi is the better steering source",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


_EN_REC = None
def centroid(banks, lg):
    global _EN_REC
    if lg == "en" and banks["en"] is None:
        if _EN_REC is None:
            _EN_REC = en_centroid_recovered()
        return _EN_REC
    return banks[lg].mean(axis=0)                      # (L+1, H)


def cos_curve(banks, a, b):
    ca, cb = centroid(banks, a), centroid(banks, b)
    return (ca * cb).sum(1) / (np.linalg.norm(ca, axis=1) * np.linalg.norm(cb, axis=1) + 1e-8)


def fig_proximity(banks, nL, out):
    x = np.arange(nL)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, t in zip(axes, TARGETS):
        ax.plot(x, cos_curve(banks, "hin", t), color="#e6194B", linewidth=2.4,
                marker="o", markersize=3, label=f"cos(Hindi, {LABELS[t]})")
        ax.plot(x, cos_curve(banks, "en", t), color="#9467bd", linewidth=2.4,
                marker="s", markersize=3, label=f"cos(English, {LABELS[t]})")
        ax.axvline(STEER, color="black", linestyle="--", linewidth=1.3, alpha=0.6)
        ax.text(STEER - 0.4, ax.get_ylim()[0] if False else 0.62, f"steer L{STEER}",
                rotation=90, va="bottom", ha="right", fontsize=8)
        ax.set_title(f"Anchor proximity to {LABELS[t]}", fontsize=11, color=COL[t])
        ax.set_xlabel("Layer (0=embed … 32=final)")
        ax.grid(True, alpha=0.25); ax.legend(loc="lower left", fontsize=8.5)
    axes[0].set_ylabel("cosine of centroids (higher = closer)")
    # in-image overall title removed: the LaTeX \caption covers it (journal requirement)
    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    # text summary at steer layer
    print(f"  at L{STEER}: " + "  ".join(
        f"{t}: hin={cos_curve(banks,'hin',t)[STEER]:.3f} en={cos_curve(banks,'en',t)[STEER]:.3f}"
        for t in TARGETS))


def main():
    banks, nL = load()
    d = ROOT / "results" / "figures" / "llama"
    fig_proximity(banks, nL, d / "llama_english_vs_hindi_proximity.png")  # centroid-based, always works
    if banks["en"] is not None:
        fig_pca(banks, nL, d / "llama_constellation_english_pca.png")     # needs en.pt samples
    else:
        print("  (skipping per-sample PCA constellation — en.pt not local yet; "
              "pull it from the VM after re-auth to enable)")


if __name__ == "__main__":
    main()
