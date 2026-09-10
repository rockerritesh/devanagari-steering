"""Per-sample, per-layer language-separation visualisations from the memory banks.

Loads results/memory_banks/<model>/{hin,mai,npi,bho}.pt — each (N, L+1, H) of
mean-pooled per-layer activations for 250 parallel samples — and produces:

  results/figures/<model>/<model>_persample_pca_layers.png
      6-panel PCA-2D scatter (1000 points = 250×4 langs) at selected layers,
      showing clusters dissolve in the mid valley and re-form deep.

  results/figures/<model>/<model>_separability_curve.png
      Per-layer language separability across all 33 layers:
        (a) 4-way linear-probe accuracy (logistic regression, 5-fold CV on PCA-50)
        (b) silhouette score (geometric cluster separation, PCA-50)
      with the steering layer marked. Probe stays high (info is decodable) while
      silhouette dips to ~0 in the middle (geometrically merged) — the hourglass.

Runs offline (no GPU):
  uv run --with torch --with numpy --with scikit-learn --with matplotlib \
      python code/visualize_per_sample_layers.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
LANGS = ["hin", "mai", "npi", "bho"]
LANG_LABELS = {"hin": "Hindi", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri"}
LANG_COLOURS = {"hin": "#d62728", "mai": "#1f77b4", "npi": "#2ca02c", "bho": "#ff7f0e"}
STEER_LAYER = {"llama": 20, "aya": 22}


def load_bank(model):
    """Return X (n_layers+1, N*4, H) float32 and labels (N*4,)."""
    mats, labels = [], []
    for li, lg in enumerate(LANGS):
        p = torch.load(ROOT / "results" / "memory_banks" / model / f"{lg}.pt",
                       map_location="cpu", weights_only=False)
        v = p["vectors"].float().numpy()           # (N, L+1, H)
        mats.append(v)
        labels.append(np.full(v.shape[0], li))
    X = np.concatenate(mats, axis=0)               # (N*4, L+1, H)
    y = np.concatenate(labels)                     # (N*4,)
    return np.transpose(X, (1, 0, 2)), y           # (L+1, N*4, H), (N*4,)


def fig_pca_panels(model, Xl, y, out):
    nL = Xl.shape[0]
    sl = STEER_LAYER[model]
    layers = [0, 8, 14, sl, 26, nL - 1]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.1 * len(layers), 3.4))
    for ax, l in zip(axes, layers):
        pts = PCA(n_components=2, random_state=0).fit_transform(Xl[l])
        for li, lg in enumerate(LANGS):
            m = y == li
            ax.scatter(pts[m, 0], pts[m, 1], s=7, alpha=0.6,
                       color=LANG_COLOURS[lg], label=LANG_LABELS[lg], linewidths=0)
        tag = " (steer)" if l == sl else (" (embed)" if l == 0 else (" (final)" if l == nL - 1 else ""))
        ax.set_title(f"L{l}{tag}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9, markerscale=1.6)
    fig.suptitle(f"{model}: per-sample PCA-2D by layer (250 samples × 4 languages). "
                 f"Clusters merge in the mid valley, re-form deep.", fontsize=11.5, y=1.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_separability(model, Xl, y, out):
    nL = Xl.shape[0]
    sl = STEER_LAYER[model]
    probe_acc, sil = [], []
    for l in range(nL):
        Z = StandardScaler().fit_transform(Xl[l])
        Z = PCA(n_components=50, random_state=0).fit_transform(Z)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        acc = cross_val_score(clf, Z, y, cv=5, scoring="accuracy").mean()
        probe_acc.append(acc)
        sil.append(silhouette_score(Z, y))
    probe_acc, sil = np.array(probe_acc), np.array(sil)
    x = np.arange(nL)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(x, probe_acc, marker="o", markersize=4, linewidth=2.3, color="#6a3d9a",
             label="4-way linear-probe accuracy (CV)")
    ax1.axhline(0.25, color="#6a3d9a", alpha=0.4, linestyle=":", linewidth=1.0)
    ax1.text(0.5, 0.27, "chance (0.25)", color="#6a3d9a", fontsize=8)
    ax1.set_ylabel("Probe accuracy (4-way)", color="#6a3d9a")
    ax1.set_ylim(0.0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#6a3d9a")

    ax2 = ax1.twinx()
    ax2.plot(x, sil, marker="s", markersize=4, linewidth=2.3, color="#e31a1c",
             label="silhouette (geometric separation)")
    ax2.set_ylabel("Silhouette score", color="#e31a1c")
    ax2.tick_params(axis="y", labelcolor="#e31a1c")

    ax1.axvline(sl, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax1.text(sl - 0.4, 0.5, f"steer L{sl}", rotation=90, va="center", ha="right", fontsize=9)
    ax1.set_xlabel("Layer (0 = embedding … 32 = final)")
    ax1.set_title(f"{model}: per-layer language separability — info stays decodable "
                  f"(probe high) but geometry merges mid-network (silhouette dips)",
                  fontsize=11)
    ax1.grid(True, alpha=0.2)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    # compact text summary
    print(f"  {model}: silhouette min={sil.min():.3f} @L{sil.argmin()}, "
          f"max={sil.max():.3f} @L{sil.argmax()}; probe min={probe_acc.min():.3f} "
          f"@L{probe_acc.argmin()}, @steer L{sl}: probe={probe_acc[sl]:.3f} sil={sil[sl]:.3f}")


def main():
    for model in ["llama", "aya"]:
        Xl, y = load_bank(model)
        d = ROOT / "results" / "figures" / model
        fig_pca_panels(model, Xl, y, d / f"{model}_persample_pca_layers.png")
        fig_separability(model, Xl, y, d / f"{model}_separability_curve.png")


if __name__ == "__main__":
    main()
