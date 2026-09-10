"""Two visualisations of the per-layer language geometry for the steering paper.

Figure 1 — per-layer cosine separation between Hindi and each sister language
plus a random-vector baseline. Reads centroid_analysis.json (or recomputes from
the .pt files if missing) and writes a single line plot.

Figure 2 — UMAP 2D projection at every layer, one panel per layer. For each
layer we stack the 250 samples × 4 languages = 1000 hidden states and fit UMAP
to 2D, colouring points by language. The grid view shows how the four language
clouds separate / merge across the depth of the network. We also save a
curated-layers view (a small subset of "interesting" layers) suitable for a
paper figure.

Usage:
    .venv/bin/python code/visualize_layers.py --model llama
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMBANK = PROJECT_ROOT / "results" / "memory_banks"
DEFAULT_FIG_DIR = PROJECT_ROOT / "results" / "figures"
LANGS = ["hin", "mai", "npi", "bho"]
LANG_COLOURS = {"hin": "#d62728", "mai": "#1f77b4", "npi": "#2ca02c", "bho": "#ff7f0e"}
LANG_LABELS = {"hin": "Hindi (source)", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri"}


def load_all(model_dir: Path) -> dict[str, torch.Tensor]:
    out = {}
    for lang in LANGS:
        payload = torch.load(model_dir / f"{lang}.pt", map_location="cpu",
                             weights_only=False)
        out[lang] = payload["vectors"].float()       # (N, L+1, H)
    return out


def fig_cosine_curves(model_dir: Path, fig_dir: Path, model_name: str):
    """One-panel line plot of per-layer cos(hin-centroid, lang-centroid)."""
    data = load_all(model_dir)
    L_plus_1 = data["hin"].shape[1]

    cents = {lang: t.mean(dim=0) for lang, t in data.items()}            # (L+1, H)

    g = torch.Generator().manual_seed(0)
    rand = torch.randn(L_plus_1, cents["hin"].shape[1], generator=g)
    rand = rand / rand.norm(dim=-1, keepdim=True)

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1).numpy()

    layers = np.arange(L_plus_1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for target in ["mai", "npi", "bho"]:
        ax.plot(layers, cos(cents["hin"], cents[target]),
                marker="o", markersize=4, linewidth=1.8,
                color=LANG_COLOURS[target], label=f"hin ↔ {target}")
    ax.plot(layers, cos(cents["hin"], rand),
            linestyle="--", linewidth=1.2, color="gray",
            label="hin ↔ random unit")
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Layer index (0 = embedding output, 32 = final residual)")
    ax.set_ylabel("Cosine similarity of per-layer centroids")
    ax.set_title(f"{model_name}: per-layer Hindi-vs-sister centroid alignment")
    ax.set_xticks(np.arange(0, L_plus_1, 2))
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = fig_dir / f"{model_name}_layer_cosine_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def umap_layer(layer_acts: dict[str, np.ndarray], seed: int, n_neighbors: int,
               min_dist: float) -> dict[str, np.ndarray]:
    """Fit UMAP to all 4 languages stacked at one layer; return per-lang coords."""
    n_per = layer_acts["hin"].shape[0]
    stacked = np.concatenate([layer_acts[l] for l in LANGS], axis=0)     # (4N, H)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, metric="cosine", random_state=seed)
    coords = reducer.fit_transform(stacked)                              # (4N, 2)
    out = {}
    for i, lang in enumerate(LANGS):
        out[lang] = coords[i * n_per : (i + 1) * n_per]
    return out


def fig_umap_grid(
    model_dir: Path,
    fig_dir: Path,
    model_name: str,
    layers: list[int] | None,
    *,
    n_neighbors: int = 25,
    min_dist: float = 0.15,
    seed: int = 0,
    point_size: float = 6.0,
    alpha: float = 0.65,
    suffix: str = "",
):
    """Grid of UMAP scatters, one per requested layer."""
    data = load_all(model_dir)
    L_plus_1 = data["hin"].shape[1]
    if layers is None:
        layers = list(range(L_plus_1))

    n = len(layers)
    n_cols = 6 if n > 12 else (3 if n <= 9 else 4)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for panel_idx, layer in enumerate(layers):
        ax = axes[panel_idx]
        layer_acts = {l: data[l][:, layer, :].numpy() for l in LANGS}
        coords = umap_layer(layer_acts, seed=seed, n_neighbors=n_neighbors,
                            min_dist=min_dist)
        for lang in LANGS:
            ax.scatter(coords[lang][:, 0], coords[lang][:, 1],
                       s=point_size, alpha=alpha,
                       color=LANG_COLOURS[lang], linewidths=0,
                       label=LANG_LABELS[lang])
        ax.set_title(f"L{layer}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.3)
        print(f"  layer {layer:>2}: UMAP done")

    for ax in axes[len(layers):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=11, markerscale=2.5)
    fig.suptitle(
        f"{model_name}: per-layer UMAP of language centroids (250 parallel samples × 4 langs)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    out = fig_dir / f"{model_name}_umap_layers{suffix}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--membank-dir", type=Path, default=DEFAULT_MEMBANK)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-neighbors", type=int, default=25)
    ap.add_argument("--min-dist", type=float, default=0.15)
    ap.add_argument("--skip-cosine", action="store_true")
    ap.add_argument("--skip-umap-all", action="store_true")
    ap.add_argument("--skip-umap-curated", action="store_true")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.membank_dir / args.model

    if not args.skip_cosine:
        fig_cosine_curves(model_dir, args.fig_dir, args.model)

    # All-layers UMAP grid
    if not args.skip_umap_all:
        print(f"\n[UMAP all 33 layers]")
        fig_umap_grid(model_dir, args.fig_dir, args.model, layers=None,
                      n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                      seed=args.seed, suffix="_all")

    # Curated 9-layer grid for the paper
    if not args.skip_umap_curated:
        print(f"\n[UMAP curated layers]")
        curated = [0, 4, 8, 12, 16, 20, 24, 28, 32]
        fig_umap_grid(model_dir, args.fig_dir, args.model, layers=curated,
                      n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                      seed=args.seed, suffix="_curated")


if __name__ == "__main__":
    main()
