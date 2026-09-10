"""Combined probe-vs-geometry separability curve for BOTH models, stacked
vertically (Llama top, Aya bottom) sharing the layer axis so the two
probe+silhouette pairs never overlap. Shows the dissociation replicates across
architectures.

  uv run --with pandas --with pyarrow --with torch --with numpy \
         --with scikit-learn --with matplotlib \
         python code/plot_separability_both.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
LANGS = ["hin", "mai", "npi", "bho"]
STEER = {"llama": 20, "aya": 22}
OUT = ROOT / "results" / "figures" / "separability_both.png"
PROBE_C, SIL_C = "#6a3d9a", "#c44e52"


def load_bank(model):
    mats, lab = [], []
    for li, lg in enumerate(LANGS):
        v = torch.load(ROOT / "results/memory_banks" / model / f"{lg}.pt",
                       map_location="cpu", weights_only=False)["vectors"].float().numpy()
        mats.append(v); lab.append(np.full(v.shape[0], li))
    X = np.concatenate(mats); y = np.concatenate(lab)
    return np.transpose(X, (1, 0, 2)), y          # (L+1, N*4, H), (N*4,)


def curve(Xl, y):
    pr, si = [], []
    for l in range(Xl.shape[0]):
        Z = StandardScaler().fit_transform(Xl[l])
        Z = PCA(n_components=50, random_state=0).fit_transform(Z)
        pr.append(cross_val_score(LogisticRegression(max_iter=2000), Z, y, cv=5).mean())
        si.append(silhouette_score(Z, y))
    return np.array(pr), np.array(si)


def panel(ax, model, pr, si):
    x = np.arange(len(pr))
    ax2 = ax.twinx()
    ax.plot(x, pr, color=PROBE_C, lw=2, marker="o", ms=2.5, label="4-way probe acc (CV)")
    ax2.plot(x, si, color=SIL_C, lw=2, marker="s", ms=2.5, label="silhouette")
    ax.axhline(0.25, color=PROBE_C, lw=0.7, ls=":", alpha=0.6)
    sl = STEER[model]
    ax.axvline(sl, color="#555", lw=1, ls="--")
    ax.text(sl + 0.4, 0.45, f"steer L{sl}", rotation=90, fontsize=8, color="#555")
    ax.set_ylim(0, 1.05); ax.set_ylabel("probe acc", color=PROBE_C, fontsize=9)
    ax2.set_ylabel("silhouette", color=SIL_C, fontsize=9)
    ax.tick_params(axis="y", labelcolor=PROBE_C); ax2.tick_params(axis="y", labelcolor=SIL_C)
    ax.set_title(f"{'Llama-3.1-8B' if model=='llama' else 'Aya-23-8B'}",
                 fontsize=10, loc="left")
    return ax2


def main():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True)
    for ax, model in [(a1, "llama"), (a2, "aya")]:
        X, y = load_bank(model)
        pr, si = curve(X, y)
        ax2 = panel(ax, model, pr, si)
        print(f"{model}: probe mean {pr.mean():.3f}; silhouette min {si.min():.3f}@L{si.argmin()}, "
              f"max {si.max():.3f}@L{si.argmax()}")
        if model == "llama":
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right", framealpha=0.9)
    a2.set_xlabel("Layer (0 = embedding … final)", fontsize=9)
    # in-image overall title removed: the LaTeX \caption covers it (journal requirement)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
