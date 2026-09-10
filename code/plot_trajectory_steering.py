"""Illustrative 2-D figure: what depth-trajectory steering does to the representation path.

Projects the per-layer centroid *trajectories* into one shared UMAP embedding and overlays:
  - the unsteered Hindi centroid path (source), layer 0 -> final,
  - the Nepali centroid path (target),
  - the STEERED path = mu_hin,l + alpha * v_l applied at every layer of the steerable
    band (v_l = mu_hin,l - mu_npi,l; alpha<0 moves Hindi toward Nepali),
with per-layer steering arrows on the band. This is a geometric illustration of the
intervention (centroid + alpha*v per layer), not the model's realized post-steer states.

  uv run --with torch --with numpy --with umap-learn --with scikit-learn --with matplotlib \
      python code/plot_trajectory_steering.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "results" / "memory_banks" / "llama"
OUT = ROOT / "results" / "figures" / "llama" / "trajectory_steering_pca.png"
BAND = list(range(8, 21))          # steerable band L8-L20 (the trajectory)
ALPHA = -0.7                        # illustrative magnitude (raw v); alpha<0: Hindi->Nepali
SUB = 120                           # samples/lang/layer for the UMAP fit (speed)
SEED = 0


def load(lg):
    return torch.load(BANK / f"{lg}.pt", map_location="cpu", weights_only=False)["vectors"].float().numpy()


def main():
    from sklearn.preprocessing import StandardScaler
    Xh, Xn = load("hin"), load("npi")          # (N, L+1, H)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(Xh.shape[0], size=min(SUB, Xh.shape[0]), replace=False)

    # Focus on the steerable band L8-L20 (where the trajectory intervention is applied);
    # the final layer has huge norm and would dominate a full-depth projection.
    mu_h = Xh.mean(0); mu_n = Xn.mean(0)         # (L+1, H)
    v = mu_h - mu_n                              # alpha<0 -> Hindi toward Nepali
    steered = {l: mu_h[l] + ALPHA * v[l] for l in BAND}

    # Fit PCA (on standardized features) using band-layer sample clouds only.
    fit_pts = np.concatenate(
        [Xh[idx][:, l, :] for l in BAND] + [Xn[idx][:, l, :] for l in BAND], axis=0)
    scaler = StandardScaler().fit(fit_pts)
    reducer = PCA(n_components=2, random_state=SEED).fit(scaler.transform(fit_pts))
    ev = reducer.explained_variance_ratio_ * 100
    tf = lambda M: reducer.transform(scaler.transform(np.atleast_2d(M)))

    Ph = tf(np.stack([mu_h[l] for l in BAND]))
    Pn = tf(np.stack([mu_n[l] for l in BAND]))
    Ps = tf(np.stack([steered[l] for l in BAND]))

    fig, ax = plt.subplots(figsize=(9, 7))
    bg = reducer.transform(scaler.transform(fit_pts)); nfit = len(idx) * len(BAND)
    ax.scatter(bg[:nfit, 0], bg[:nfit, 1], s=3, alpha=0.05, color="#d62728", linewidths=0)
    ax.scatter(bg[nfit:, 0], bg[nfit:, 1], s=3, alpha=0.05, color="#2ca02c", linewidths=0)

    def path(P, color, label, ls="-", lw=2.4, z=5):
        ax.plot(P[:, 0], P[:, 1], ls, color=color, lw=lw, alpha=0.95, zorder=z, label=label)
        ax.scatter(P[:, 0], P[:, 1], color=color, s=34, zorder=z + 1, edgecolors="white", linewidths=0.5)

    path(Ph, "#d62728", "Hindi trajectory (unsteered)")
    path(Pn, "#2ca02c", "Nepali trajectory (target)")
    path(Ps, "#1f1f1f", f"Steered trajectory (Hindi $+\\,\\alpha v_\\ell$, $\\alpha{{=}}{ALPHA}$)", ls="--", lw=2.6, z=7)

    for k, l in enumerate(BAND):                 # arrow Hindi->steered at each band layer
        ax.annotate("", xy=(Ps[k, 0], Ps[k, 1]), xytext=(Ph[k, 0], Ph[k, 1]),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.0, alpha=0.75), zorder=6)
    for k, l in enumerate(BAND):                 # layer labels on Hindi + a couple on the paths
        if l in (BAND[0], 14, BAND[-1]):
            ax.annotate(f"L{l}", (Ph[k, 0], Ph[k, 1]), xytext=(5, 5), textcoords="offset points",
                        fontsize=9, color="#d62728", fontweight="bold", zorder=8)
    for P, c in [(Ph, "#d62728"), (Pn, "#2ca02c"), (Ps, "#1f1f1f")]:   # square = band start (L8)
        ax.scatter(P[0, 0], P[0, 1], color=c, s=95, marker="s", edgecolors="black", linewidths=0.8, zorder=9)

    ax.set_xlabel(f"PC 1 ({ev[0]:.0f}% var)"); ax.set_ylabel(f"PC 2 ({ev[1]:.0f}% var)")
    # in-image title removed: the LaTeX \caption covers it (journal requirement)
    ax.legend(loc="best", fontsize=9, framealpha=0.95); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
