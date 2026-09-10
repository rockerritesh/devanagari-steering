"""Constellation-style trajectory plots — same convention as the SafeConstellations
figures (centroid trajectories per category, faded individual points behind).

For each requested layer band we fit ONE shared UMAP on all (lang × layer ×
sample) points in the band, then trace four polygons (one per language)
connecting the per-language layer-centroids in layer order. The shared UMAP
makes intra-panel trajectories spatially comparable; we deliberately do NOT
share UMAPs across panels because UMAP is not stable under different inputs.

Three figures are produced:

  llama_constellation_bands.png      — 6-panel grid by layer bands
                                       (0-5, 6-10, 11-15, 16-20, 21-25, 26-32)
  llama_constellation_headline.png   — single big panel for L17-L30 (the
                                       language-identity band, where the
                                       hin-vs-sister cosine drops most)
  llama_constellation_full.png       — single big panel for all layers L0-L32

Each plot draws:
  - background : individual hidden-state points (s=2, alpha=0.08)
  - star markers at each (lang, layer) centroid
  - a polyline connecting the stars in layer order per language
  - a square at the FIRST layer of each language's path (the "start")
  - dashed gray lines between hin-centroid and each sister-centroid at the
    same layer, so the per-layer "steering gap" is visible as line length.
  - a per-star layer-index label so readers can read direction

Usage:
    .venv/bin/python code/visualize_constellations.py --model llama
"""

from __future__ import annotations

import argparse
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


def constellation_panel(
    ax,
    data: dict[str, torch.Tensor],
    layers: list[int],
    *,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    show_legend: bool = False,
    show_layer_labels: bool = True,
    show_steering_dashes: bool = True,
    title: str | None = None,
) -> None:
    """Render one constellation panel for the given list of layer indices."""
    n_per = data["hin"].shape[0]

    # 1. Stack everything for a single shared UMAP fit on this band.
    chunks = []
    for lang in LANGS:
        for layer in layers:
            chunks.append(data[lang][:, layer, :].numpy())
    combined = np.concatenate(chunks, axis=0)             # (4 * n_layers * N, H)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, metric="cosine", random_state=seed,
    )
    coords = reducer.fit_transform(combined)              # (4 * n_layers * N, 2)

    # 2. Slice back into per (lang, layer) point clouds.
    pts: dict[tuple[str, int], np.ndarray] = {}
    cur = 0
    for lang in LANGS:
        for layer in layers:
            pts[(lang, layer)] = coords[cur : cur + n_per]
            cur += n_per

    # 3. Background: faded individual points.
    for (lang, _layer), p in pts.items():
        ax.scatter(p[:, 0], p[:, 1], s=2.5, alpha=0.07,
                   color=LANG_COLOURS[lang], linewidths=0)

    # 4. Per-language constellation: line + stars + start square.
    centroids: dict[str, np.ndarray] = {}
    for lang in LANGS:
        cents = np.stack([pts[(lang, layer)].mean(axis=0) for layer in layers])  # (n_layers, 2)
        centroids[lang] = cents
        ax.plot(
            cents[:, 0], cents[:, 1],
            color=LANG_COLOURS[lang], linewidth=2.0, alpha=0.95, zorder=5,
        )
        ax.scatter(
            cents[:, 0], cents[:, 1],
            color=LANG_COLOURS[lang], s=110, marker="*",
            edgecolors="black", linewidths=0.6, zorder=6,
            label=LANG_LABELS[lang],
        )
        ax.scatter(
            cents[0, 0], cents[0, 1],
            color=LANG_COLOURS[lang], s=140, marker="s",
            edgecolors="black", linewidths=0.9, zorder=7,
        )

    # 5. Per-layer dashed gray "steering gap" between Hindi and each sister.
    if show_steering_dashes:
        for lang in ["mai", "npi", "bho"]:
            for li in range(len(layers)):
                hp = centroids["hin"][li]
                tp = centroids[lang][li]
                ax.plot([hp[0], tp[0]], [hp[1], tp[1]],
                        linestyle=(0, (3, 3)), color="gray",
                        linewidth=0.6, alpha=0.4, zorder=3)

    # 6. Layer-index labels on each Hindi centroid (just one set so it's not too noisy).
    if show_layer_labels:
        for li, layer in enumerate(layers):
            hp = centroids["hin"][li]
            ax.annotate(
                f"L{layer}",
                xy=(hp[0], hp[1]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color=LANG_COLOURS["hin"],
                fontweight="bold", alpha=0.85, zorder=8,
            )

    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(loc="best", fontsize=9, framealpha=0.95)


def make_bands(n_hidden: int):
    """Split the 0..n_hidden-1 hidden-state indices into 6 contiguous bands so
    the grid is model-agnostic (Llama=33 hidden states, himalaya/gemma4=36)."""
    splits = [list(map(int, s)) for s in np.array_split(np.arange(n_hidden), 6)]
    names = ["Early Layers", "Early-Mid Layers", "Mid Layers",
             "Mid-Late Layers", "Late Layers", "Final Layers"]
    return list(zip(names, splits))


def fig_bands(model_dir, fig_dir, model_name, *, n_neighbors, min_dist, seed):
    data = load_all(model_dir)
    bands = make_bands(data["hin"].shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.reshape(-1)
    for i, (name, layers) in enumerate(bands):
        print(f"  band: {name} (L{layers[0]}-L{layers[-1]}) ...")
        constellation_panel(
            axes[i], data, layers,
            n_neighbors=n_neighbors, min_dist=min_dist, seed=seed,
            show_legend=(i == 0),
            show_layer_labels=True,
            show_steering_dashes=True,
            title=f"{name}\nLayers {layers[0]}–{layers[-1]}",
        )
    fig.suptitle(
        f"{model_name}: per-band centroid constellations "
        f"(★ = layer centroid, ■ = first layer, dashed = hin-vs-sister gap)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = fig_dir / f"{model_name}_constellation_bands.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_single(model_dir, fig_dir, model_name, layers, suffix, title, *,
               n_neighbors, min_dist, seed):
    data = load_all(model_dir)
    fig, ax = plt.subplots(figsize=(11, 8))
    print(f"  single panel L{layers[0]}-L{layers[-1]} ({len(layers)} layers) ...")
    constellation_panel(
        ax, data, layers,
        n_neighbors=n_neighbors, min_dist=min_dist, seed=seed,
        show_legend=True,
        show_layer_labels=True,
        show_steering_dashes=True,
        title=title,
    )
    fig.tight_layout()
    out = fig_dir / f"{model_name}_constellation_{suffix}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--membank-dir", type=Path, default=DEFAULT_MEMBANK)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--n-neighbors", type=int, default=25)
    ap.add_argument("--min-dist", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-bands", action="store_true")
    ap.add_argument("--skip-headline", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.membank_dir / args.model

    # Depth-adaptive ranges: peek at one bank to get the hidden-state count.
    n_hidden = int(torch.load(model_dir / "hin.pt", map_location="cpu",
                              weights_only=False)["vectors"].shape[1])
    last = n_hidden - 1
    # Headline "language-identity band" = upper-half of the network (where the
    # hin-vs-sister gap is largest), generalized from Llama's L17-L30 of 33.
    hl_lo = round(0.52 * last)
    hl_hi = last - 2

    if not args.skip_bands:
        print("\n[band-grid constellation]")
        fig_bands(model_dir, args.fig_dir, args.model,
                  n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed)

    if not args.skip_headline:
        print(f"\n[headline constellation L{hl_lo}-L{hl_hi}, every 2nd layer]")
        step2 = list(range(hl_lo, hl_hi + 1, 2))
        fig_single(
            model_dir, args.fig_dir, args.model,
            layers=step2,
            suffix=f"headline_L{hl_lo}-L{step2[-1]}_step2",
            title=(f"{args.model}: language-identity band centroid constellation "
                   f"(L{hl_lo}..L{step2[-1]}, step 2)"),
            n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed,
        )
        print(f"\n[headline constellation L{hl_lo}-L{hl_hi}, full]")
        fig_single(
            model_dir, args.fig_dir, args.model,
            layers=list(range(hl_lo, hl_hi + 1)),
            suffix=f"headline_L{hl_lo}-L{hl_hi}",
            title=f"{args.model}: language-identity band centroid constellation (L{hl_lo}–L{hl_hi})",
            n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed,
        )

    if not args.skip_full:
        print(f"\n[full constellation L0-L{last}]")
        fig_single(
            model_dir, args.fig_dir, args.model,
            layers=list(range(0, n_hidden)),
            suffix=f"full_L0-L{last}",
            title=None,  # in-image title removed: the LaTeX \caption covers it (journal requirement)
            n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed,
        )


if __name__ == "__main__":
    main()
