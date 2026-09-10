"""Combined hourglass for BOTH models in one image (Llama top, Aya bottom).

Per-sample PCA-2D snapshots at fixed depths (L0, 8, 16, 24, 32) for each model;
both rows share the same layer columns so the distinct -> merged -> distinct
"hourglass" is directly comparable across the two architectures.

  uv run --with torch --with numpy --with scikit-learn --with matplotlib \
      python code/visualize_hourglass_combined.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("llama", "Llama-3.1-8B", 20), ("aya", "Aya-23-8B", 22)]
LAYERS = [0, 8, 16, 24, 32]
LANGS = ["hin", "mai", "npi", "bho"]
LAB = {"hin": "Hindi", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri"}
COL = {"hin": "#d62728", "mai": "#1f77b4", "npi": "#2ca02c", "bho": "#f58231"}


def load(model):
    out = {}
    for lg in LANGS:
        out[lg] = torch.load(ROOT / "results" / "memory_banks" / model / f"{lg}.pt",
                             map_location="cpu", weights_only=False)["vectors"].float().numpy()
    return out


def main():
    fig, axes = plt.subplots(len(MODELS), len(LAYERS), figsize=(3.0 * len(LAYERS), 6.2))
    for r, (model, label, steer) in enumerate(MODELS):
        data = load(model)
        n = data["hin"].shape[0]
        for c, l in enumerate(LAYERS):
            ax = axes[r, c]
            X = np.concatenate([data[lg][:, l] for lg in LANGS], axis=0)
            pts = PCA(n_components=2, random_state=0).fit_transform(X)
            for i, lg in enumerate(LANGS):
                s = pts[i * n:(i + 1) * n]
                ax.scatter(s[:, 0], s[:, 1], s=5, alpha=0.55, color=COL[lg],
                           label=LAB[lg], linewidths=0)
            ax.set_xticks([]); ax.set_yticks([])
            mark = "  (steer)" if abs(l - steer) <= 2 else ("  (input)" if l == 0 else ("  (output)" if l == 32 else ""))
            if r == 0:
                ax.set_title(f"L{l}{mark}", fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9, markerscale=1.6,
                          handletextpad=0.2, borderpad=0.3)
    # in-image overall title removed: the LaTeX \caption covers it (journal requirement)
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "hourglass_combined_llama_aya.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
