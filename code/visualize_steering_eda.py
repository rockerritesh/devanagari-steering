"""Six new EDA figures for the steering paper.

Reads `results/memory_banks/{model}/{lang}.pt` (each = (N, L+1, H) tensor) and
writes paper-grade plots that go beyond the existing UMAP / cosine views.

Lenses covered:
  1. PCA-trajectory (2D)       — joint-basis avg-class trajectory through depth
  2. PCA-trajectory (3D)       — same, in 3D for cleaner separation
  3. Layer dynamics (2-panel)  — per-layer velocity + angular curvature
  4. Effective rank / PR       — participation ratio of activations across depth
  5. CKA(L_i, L_j) heatmap     — within-language layer-vs-layer similarity
                                 (block-recurrent test, BRH paper)
  6. Steering-vector geometry  — cos(v_l, v_l') heatmap + per-layer rank-1
                                 dominance (sigma_1 / sigma_2 of the residual)

Output: results/figures/{model}_eda_<name>.png

Usage:
    .venv/bin/python code/visualize_steering_eda.py --model llama
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMBANK = PROJECT_ROOT / "results" / "memory_banks"
DEFAULT_FIG_DIR = PROJECT_ROOT / "results" / "figures"

LANGS = ["hin", "mai", "npi", "bho"]
TARGETS = ["mai", "npi", "bho"]
LANG_COLOURS = {"hin": "#d62728", "mai": "#1f77b4", "npi": "#2ca02c", "bho": "#ff7f0e"}
LANG_LABELS = {"hin": "Hindi (source)", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri"}


# ---------------------------------------------------------------------------
# Loaders / helpers
# ---------------------------------------------------------------------------

def load_all(model_dir: Path) -> dict[str, torch.Tensor]:
    out = {}
    for lang in LANGS:
        payload = torch.load(model_dir / f"{lang}.pt", map_location="cpu",
                             weights_only=False)
        out[lang] = payload["vectors"].float()      # (N, L+1, H)
    return out


def per_layer_centroids(data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Return per-language (L+1, H) centroid tensors as numpy arrays."""
    return {lang: t.mean(dim=0).numpy() for lang, t in data.items()}


def gradient_polyline(ax, xy: np.ndarray, cmap, *, lw=2.4, alpha=1.0, zorder=4):
    """Draw a polyline coloured by a gradient along its length (2D)."""
    n = xy.shape[0]
    segs = np.stack([xy[:-1], xy[1:]], axis=1)              # (n-1, 2, 2)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(np.linspace(0, 1, n - 1))
    ax.add_collection(lc)
    return lc


def gradient_polyline_3d(ax, xyz: np.ndarray, cmap, *, lw=2.2, alpha=1.0, zorder=4):
    n = xyz.shape[0]
    segs = np.stack([xyz[:-1], xyz[1:]], axis=1)            # (n-1, 2, 3)
    lc = Line3DCollection(segs, cmap=cmap, linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(np.linspace(0, 1, n - 1))
    ax.add_collection3d(lc)
    return lc


def lang_cmap(lang: str):
    """Gradient that goes from a desaturated tint of the language colour
    to the saturated colour — keeps the early portion of trajectories visible."""
    base = mcolors.to_rgb(LANG_COLOURS[lang])
    # 50% blend with white = visible but lighter
    light = tuple(0.5 * c + 0.5 for c in base)
    dark = base
    return mcolors.LinearSegmentedColormap.from_list(f"{lang}_grad", [light, dark])


# ---------------------------------------------------------------------------
# Figure 1 — joint-basis 2D PCA trajectory of per-layer class centroids
# ---------------------------------------------------------------------------

def _draw_pca_panel(ax, proj: dict[str, np.ndarray], layers: np.ndarray,
                    title: str, ev: np.ndarray, *,
                    annotate_layers: list[int]):
    """One 2D PCA panel given pre-projected coordinates."""
    L = len(layers)
    for lang in LANGS:
        xy = proj[lang]
        gradient_polyline(ax, xy, cmap=lang_cmap(lang), lw=2.6, alpha=0.95)
        ax.scatter(xy[0, 0], xy[0, 1], s=100, marker="s",
                   facecolor=LANG_COLOURS[lang], edgecolor="black",
                   linewidths=0.8, zorder=6)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=110, marker="o",
                   facecolor=LANG_COLOURS[lang], edgecolor="black",
                   linewidths=0.8, zorder=6, label=LANG_LABELS[lang])
        # locate L20 within this panel's layer slice if present
        if 20 in layers:
            idx20 = int(np.where(layers == 20)[0][0])
            ax.scatter(xy[idx20, 0], xy[idx20, 1], s=140, marker="X",
                       facecolor=LANG_COLOURS[lang], edgecolor="black",
                       linewidths=1.0, zorder=7)

    # annotate Hindi path with layer indices
    hin_xy = proj["hin"]
    for li in annotate_layers:
        if li in layers:
            idx = int(np.where(layers == li)[0][0])
            ax.annotate(
                f"L{li}",
                xy=hin_xy[idx],
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=LANG_COLOURS["hin"], fontweight="bold",
                alpha=0.85, zorder=8,
            )
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}% var)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)


def fig_pca_trajectory_2d(data, fig_dir: Path, model_name: str):
    """Three-panel 2D PCA:
        (a) full depth — hourglass, depth-dominated PC1
        (b) late layers L17-L32 — language identity emerges
        (c) per-layer-mean-subtracted — language-only axes (depth removed)
    """
    cents = per_layer_centroids(data)
    L_plus_1 = cents["hin"].shape[0]

    # (a) full-depth joint PCA
    stack_a = np.concatenate([cents[l] for l in LANGS], axis=0)
    pca_a = PCA(n_components=2, random_state=0).fit(stack_a)
    proj_a = {l: pca_a.transform(cents[l]) for l in LANGS}
    layers_a = np.arange(L_plus_1)

    # (b) late layers only, joint PCA
    late = np.arange(17, L_plus_1)
    stack_b = np.concatenate([cents[l][late] for l in LANGS], axis=0)
    pca_b = PCA(n_components=2, random_state=0).fit(stack_b)
    proj_b = {l: pca_b.transform(cents[l][late]) for l in LANGS}

    # (c) per-layer-mean-subtracted: remove the depth axis entirely
    pooled = np.stack([cents[l] for l in LANGS]).mean(axis=0)        # (L+1, H)
    cents_c = {l: cents[l] - pooled for l in LANGS}                   # (L+1, H)
    stack_c = np.concatenate([cents_c[l] for l in LANGS], axis=0)
    pca_c = PCA(n_components=2, random_state=0).fit(stack_c)
    proj_c = {l: pca_c.transform(cents_c[l]) for l in LANGS}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))

    _draw_pca_panel(
        axes[0], proj_a, layers_a,
        title="(a) Full depth — joint PCA  (PC1 = depth axis, the 'hourglass')",
        ev=pca_a.explained_variance_ratio_,
        annotate_layers=[0, 8, 16, 20, 24, 32],
    )
    _draw_pca_panel(
        axes[1], proj_b, late,
        title=f"(b) Late layers L17–L{L_plus_1-1} only — joint PCA  (language identity emerges)",
        ev=pca_b.explained_variance_ratio_,
        annotate_layers=[17, 20, 24, 28, 32],
    )
    _draw_pca_panel(
        axes[2], proj_c, layers_a,
        title="(c) Per-layer-mean subtracted — language-only PCA  (depth axis removed)",
        ev=pca_c.explained_variance_ratio_,
        annotate_layers=[0, 8, 16, 20, 24, 32],
    )
    axes[0].legend(loc="best", framealpha=0.95, fontsize=9)
    fig.suptitle(
        f"{model_name}: 2D-PCA of avg. class trajectory  "
        f"(■ = first, X = L20, ● = last)",
        fontsize=12.5, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = fig_dir / f"{model_name}_eda_pca_trajectory_2d.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — joint-basis 3D PCA trajectory
# ---------------------------------------------------------------------------

def fig_pca_trajectory_3d(data, fig_dir: Path, model_name: str):
    cents = per_layer_centroids(data)
    L_plus_1 = cents["hin"].shape[0]
    stack = np.concatenate([cents[l] for l in LANGS], axis=0)

    pca = PCA(n_components=3, random_state=0).fit(stack)
    proj = {l: pca.transform(cents[l]) for l in LANGS}

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    for lang in LANGS:
        xyz = proj[lang]
        gradient_polyline_3d(ax, xyz, cmap=lang_cmap(lang), lw=2.4, alpha=0.95)
        ax.scatter(*xyz[0], s=80, marker="s", color=LANG_COLOURS[lang],
                   edgecolor="black", linewidths=0.6, zorder=6)
        ax.scatter(*xyz[-1], s=90, marker="o", color=LANG_COLOURS[lang],
                   edgecolor="black", linewidths=0.6, zorder=6,
                   label=LANG_LABELS[lang])
        if 20 < L_plus_1:
            ax.scatter(*xyz[20], s=130, marker="X", color=LANG_COLOURS[lang],
                       edgecolor="black", linewidths=0.9, zorder=7)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3  ({ev[2]*100:.1f}%)")
    ax.set_title(
        f"{model_name}: 3D-PCA of avg. class trajectory  "
        f"(■ = L0, X = L20, ● = L{L_plus_1-1})"
    )
    ax.legend(loc="upper left", framealpha=0.95)
    ax.view_init(elev=22, azim=45)
    fig.tight_layout()
    out = fig_dir / f"{model_name}_eda_pca_trajectory_3d.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — layer dynamics: velocity (||Δ||) and angular curvature
# ---------------------------------------------------------------------------

def fig_layer_dynamics(data, fig_dir: Path, model_name: str):
    cents = per_layer_centroids(data)             # {lang: (L+1, H)}
    L_plus_1 = cents["hin"].shape[0]

    velocity = {}
    curvature = {}
    for lang in LANGS:
        c = cents[lang]                                              # (L+1, H)
        d = np.diff(c, axis=0)                                        # (L, H)
        velocity[lang] = np.linalg.norm(d, axis=1)                    # (L,)
        # angle between consecutive deltas
        d1 = d[:-1]
        d2 = d[1:]
        n1 = np.linalg.norm(d1, axis=1) + 1e-12
        n2 = np.linalg.norm(d2, axis=1) + 1e-12
        cos = (d1 * d2).sum(axis=1) / (n1 * n2)
        cos = np.clip(cos, -1, 1)
        curvature[lang] = np.degrees(np.arccos(cos))                  # (L-1,)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    layers = np.arange(1, L_plus_1)
    for lang in LANGS:
        ax.plot(layers, velocity[lang], marker="o", markersize=3.5,
                linewidth=1.8, color=LANG_COLOURS[lang],
                label=LANG_LABELS[lang])
    ax.set_yscale("log")
    ax.axvline(20, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ymin, ymax = ax.get_ylim()
    ax.text(20.2, ymax / 1.4, "L20", fontsize=9,
            color="black", alpha=0.75, ha="left", va="top")
    ax.set_xlabel("Layer transition (L → L+1)")
    ax.set_ylabel(r"Velocity $\|\mu_{l+1} - \mu_{l}\|_2$  (log)")
    ax.set_title("Per-layer velocity — log scale (final RMSNorm dominates linearly)")
    ax.set_xticks(np.arange(0, L_plus_1, 2))
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)

    ax = axes[1]
    layers = np.arange(2, L_plus_1)
    for lang in LANGS:
        ax.plot(layers, curvature[lang], marker="o", markersize=3.5,
                linewidth=1.8, color=LANG_COLOURS[lang],
                label=LANG_LABELS[lang])
    ax.axvline(20, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Layer (turn at L)")
    ax.set_ylabel(r"Angular curvature  $\angle(\Delta_{l-1}, \Delta_{l})$  [deg]")
    ax.set_title("Trajectory bending — where the path turns")
    ax.set_xticks(np.arange(0, L_plus_1, 2))
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"{model_name}: layer-wise dynamics of class centroids", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = fig_dir / f"{model_name}_eda_layer_dynamics.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4 — effective rank / participation ratio across depth
# ---------------------------------------------------------------------------

def participation_ratio(X: np.ndarray) -> float:
    """PR = (sum eig)^2 / sum(eig^2). X is (N, H), centred internally."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # Use SVD on centred X; eigenvalues of cov are sigma_i^2 / (N-1).
    s = np.linalg.svd(Xc, compute_uv=False)
    eig = (s ** 2) / max(Xc.shape[0] - 1, 1)
    num = eig.sum() ** 2
    den = (eig ** 2).sum() + 1e-18
    return float(num / den)


def fig_effective_rank(data, fig_dir: Path, model_name: str):
    L_plus_1 = data["hin"].shape[1]
    pr = {lang: np.zeros(L_plus_1) for lang in LANGS}
    for lang in LANGS:
        X_all = data[lang].numpy()                                    # (N, L+1, H)
        for layer in range(L_plus_1):
            pr[lang][layer] = participation_ratio(X_all[:, layer, :])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    layers = np.arange(L_plus_1)
    for lang in LANGS:
        ax.plot(layers, pr[lang], marker="o", markersize=3.5,
                linewidth=1.8, color=LANG_COLOURS[lang],
                label=LANG_LABELS[lang])
    ax.axvline(20, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Participation ratio  $(\\sum \\lambda_i)^2 / \\sum \\lambda_i^2$")
    ax.set_title(
        f"{model_name}: effective rank / participation ratio across depth\n"
        "(BRH-style low-rank-collapse test — lower = activations live in fewer dims)"
    )
    ax.set_xticks(np.arange(0, L_plus_1, 2))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    out = fig_dir / f"{model_name}_eda_effective_rank.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5 — within-language CKA(L_i, L_j) heatmap (block-recurrence test)
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (N, H) representations."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    den = (np.linalg.norm(Xc.T @ Xc, ord="fro")
           * np.linalg.norm(Yc.T @ Yc, ord="fro") + 1e-18)
    return float(num / den)


def fig_cka_heatmap(data, fig_dir: Path, model_name: str):
    L_plus_1 = data["hin"].shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, lang in zip(axes.reshape(-1), LANGS):
        X = data[lang].numpy()                                        # (N, L+1, H)
        M = np.zeros((L_plus_1, L_plus_1))
        for i in range(L_plus_1):
            for j in range(i, L_plus_1):
                v = linear_cka(X[:, i, :], X[:, j, :])
                M[i, j] = M[j, i] = v
        im = ax.imshow(M, cmap="magma", vmin=0.4, vmax=1, origin="lower")
        ax.set_title(f"{LANG_LABELS[lang]}  (linear CKA, vmin=0.4)", fontsize=11)
        ax.set_xlabel("Layer $j$")
        ax.set_ylabel("Layer $i$")
        ax.set_xticks(np.arange(0, L_plus_1, 4))
        ax.set_yticks(np.arange(0, L_plus_1, 4))
        # mark L20 grid lines
        ax.axvline(20, color="cyan", alpha=0.4, linewidth=0.6)
        ax.axhline(20, color="cyan", alpha=0.4, linewidth=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{model_name}: layer-vs-layer linear CKA, per language\n"
        "(block-diagonal blocks = stretches of recurrent reuse — BRH signature)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = fig_dir / f"{model_name}_eda_cka_layer_layer.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 6 — steering-vector geometry: cos(v_l, v_l') heatmap + rank-1 dominance
# ---------------------------------------------------------------------------

def fig_steering_geometry(data, fig_dir: Path, model_name: str):
    cents = per_layer_centroids(data)
    L_plus_1 = cents["hin"].shape[0]

    # v_l(target) = mu_hin_l - mu_target_l, layer-stacked
    V = {tgt: cents["hin"] - cents[tgt] for tgt in TARGETS}            # (L+1, H)

    # Per-target layer-vs-layer cosine of steering direction
    def cos_matrix(M: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-18
        Mn = M / nrm
        return Mn @ Mn.T

    # Per-layer rank-1 dominance: SVD of paired residuals (h_hin_i - h_tgt_i),
    # i in samples, on a per-layer matrix (N, H). Ratio sigma_1 / sigma_2.
    def rank1_curve(tgt: str) -> np.ndarray:
        N = data["hin"].shape[0]
        out = np.zeros(L_plus_1)
        H = data["hin"].numpy()
        T = data[tgt].numpy()
        D = H - T                                                       # (N, L+1, H)
        for layer in range(L_plus_1):
            s = np.linalg.svd(D[:, layer, :], compute_uv=False)
            out[layer] = s[0] / (s[1] + 1e-12)
        return out

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.28)

    # Top row: per-target cosine heatmaps
    for col, tgt in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, col])
        M = cos_matrix(V[tgt])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.set_title(f"$\\cos(v_l, v_{{l'}})$  —  hin vs {LANG_LABELS[tgt]}",
                     fontsize=10.5)
        ax.set_xlabel("Layer $l'$")
        ax.set_ylabel("Layer $l$" if col == 0 else "")
        ax.set_xticks(np.arange(0, L_plus_1, 4))
        ax.set_yticks(np.arange(0, L_plus_1, 4))
        ax.axvline(20, color="black", alpha=0.5, linewidth=0.6, linestyle="--")
        ax.axhline(20, color="black", alpha=0.5, linewidth=0.6, linestyle="--")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: rank-1 dominance line plot (full width)
    ax = fig.add_subplot(gs[1, :])
    layers = np.arange(L_plus_1)
    for tgt in TARGETS:
        r = rank1_curve(tgt)
        ax.plot(layers, r, marker="o", markersize=3.5, linewidth=1.8,
                color=LANG_COLOURS[tgt], label=f"hin → {LANG_LABELS[tgt]}")
    ax.axvline(20, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ymin, ymax = ax.get_ylim()
    ax.text(20.2, ymin + 0.92 * (ymax - ymin), "L20", fontsize=9,
            color="black", alpha=0.85, ha="left", va="top")
    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"Rank-1 dominance  $\sigma_1 / \sigma_2$  of $(h_{hin}-h_{tgt})$")
    ax.set_title(
        "Per-layer rank-1 dominance of the residual (high = clean single-direction "
        "steering signal; supports 'single L20 wins multi-layer' claim)",
        fontsize=11,
    )
    ax.set_xticks(np.arange(0, L_plus_1, 2))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.95)

    fig.suptitle(
        f"{model_name}: steering-vector geometry — direction stability + rank-1 dominance",
        fontsize=12.5, y=0.995,
    )
    out = fig_dir / f"{model_name}_eda_steering_vector_geometry.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--membank-dir", type=Path, default=DEFAULT_MEMBANK)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset: pca2, pca3, dynamics, rank, cka, steering")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.membank_dir / args.model
    print(f"loading memory bank: {model_dir}")
    data = load_all(model_dir)
    n_per, L_plus_1, H = data["hin"].shape
    print(f"  shapes: N={n_per}, layers={L_plus_1}, H={H}")

    selected = set(args.only) if args.only else {
        "pca2", "pca3", "dynamics", "rank", "cka", "steering"
    }

    if "pca2" in selected:
        print("\n[1/6] joint-basis 2D PCA trajectory")
        fig_pca_trajectory_2d(data, args.fig_dir, args.model)
    if "pca3" in selected:
        print("\n[2/6] joint-basis 3D PCA trajectory")
        fig_pca_trajectory_3d(data, args.fig_dir, args.model)
    if "dynamics" in selected:
        print("\n[3/6] layer-wise velocity + angular curvature")
        fig_layer_dynamics(data, args.fig_dir, args.model)
    if "rank" in selected:
        print("\n[4/6] effective rank / participation ratio")
        fig_effective_rank(data, args.fig_dir, args.model)
    if "cka" in selected:
        print("\n[5/6] within-language CKA(L_i, L_j) heatmaps")
        fig_cka_heatmap(data, args.fig_dir, args.model)
    if "steering" in selected:
        print("\n[6/6] steering-vector geometry")
        fig_steering_geometry(data, args.fig_dir, args.model)


if __name__ == "__main__":
    main()
