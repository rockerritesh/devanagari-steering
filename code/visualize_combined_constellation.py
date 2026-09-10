"""Combined general + verb constellation, full depth (L0-L32).

Same convention as visualize_constellations.py, but instead of running on a
single memory bank we pool both `llama` (parallel-250) and `llama_verb`
(verb-pairs-152) into ONE shared UMAP and draw two trajectories per language:

  - solid line + star markers   = general (250 samples / lang / layer)
  - dashed line + diamond markers = verb   (152 samples / lang / layer)

Both share the language colour, so any drift between the two corpora is
spatially legible: where do verb activations land relative to general ones?

Output:
  results/figures/llama_combined_constellation_full_L0-L32.png
  results/figures/llama_combined_constellation_late_L17-L32.png
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
TARGETS = ["mai", "npi", "bho"]
LANG_COLOURS = {"hin": "#d62728", "mai": "#1f77b4", "npi": "#2ca02c", "bho": "#ff7f0e"}
LANG_LABELS = {"hin": "Hindi (source)", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri"}

CORPORA = ("general", "verb")
CORPUS_DIRS = {"general": "llama", "verb": "llama_verb"}
CORPUS_STYLE = {
    "general": dict(linestyle="-",  marker="*", marker_size=120, line_alpha=0.95,
                    line_width=2.0),
    "verb":    dict(linestyle="--", marker="D", marker_size=70,  line_alpha=0.85,
                    line_width=1.5),
}


def load_one(model_dir: Path) -> dict[str, torch.Tensor]:
    out = {}
    for lang in LANGS:
        payload = torch.load(model_dir / f"{lang}.pt",
                             map_location="cpu", weights_only=False)
        out[lang] = payload["vectors"].float()       # (N, L+1, H)
    return out


def panel(
    ax,
    data: dict[str, dict[str, torch.Tensor]],         # data[corpus][lang]
    layers: list[int],
    *,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    show_legend: bool,
    show_layer_labels: bool,
    show_steering_dashes: bool,
    title: str,
) -> None:
    # 1. Stack all (corpus, lang, layer) sample chunks for one shared UMAP.
    chunks = []
    counts: dict[tuple[str, str, int], int] = {}
    for corpus in CORPORA:
        for lang in LANGS:
            t = data[corpus][lang]                       # (N, L+1, H)
            for layer in layers:
                arr = t[:, layer, :].numpy()
                chunks.append(arr)
                counts[(corpus, lang, layer)] = arr.shape[0]

    combined = np.concatenate(chunks, axis=0)
    print(f"  fitting UMAP on {combined.shape[0]:,} points × {combined.shape[1]} dims ...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, metric="cosine", random_state=seed,
    )
    coords = reducer.fit_transform(combined)

    # 2. Slice back into per-(corpus, lang, layer) point clouds.
    pts: dict[tuple[str, str, int], np.ndarray] = {}
    cur = 0
    for corpus in CORPORA:
        for lang in LANGS:
            for layer in layers:
                n = counts[(corpus, lang, layer)]
                pts[(corpus, lang, layer)] = coords[cur : cur + n]
                cur += n

    # 3. Background — faded individual points, distinguish corpus by alpha only.
    for (corpus, lang, _layer), p in pts.items():
        a = 0.06 if corpus == "general" else 0.10
        ax.scatter(p[:, 0], p[:, 1], s=2.0, alpha=a,
                   color=LANG_COLOURS[lang], linewidths=0)

    # 4. Centroid trajectories — one per (corpus, lang).
    centroids: dict[tuple[str, str], np.ndarray] = {}
    for corpus in CORPORA:
        sty = CORPUS_STYLE[corpus]
        for lang in LANGS:
            cents = np.stack([pts[(corpus, lang, layer)].mean(axis=0)
                              for layer in layers])              # (L, 2)
            centroids[(corpus, lang)] = cents
            label = (f"{LANG_LABELS[lang]} ({corpus})"
                     if (lang == "hin" or corpus == "general") else None)
            ax.plot(cents[:, 0], cents[:, 1],
                    color=LANG_COLOURS[lang],
                    linewidth=sty["line_width"],
                    linestyle=sty["linestyle"],
                    alpha=sty["line_alpha"], zorder=5)
            ax.scatter(cents[:, 0], cents[:, 1],
                       color=LANG_COLOURS[lang],
                       s=sty["marker_size"], marker=sty["marker"],
                       edgecolors="black", linewidths=0.6, zorder=6,
                       label=label)
            # first-layer marker so the start is visible
            ax.scatter(cents[0, 0], cents[0, 1],
                       color=LANG_COLOURS[lang],
                       s=sty["marker_size"] + 30,
                       marker="s" if corpus == "general" else "P",
                       edgecolors="black", linewidths=0.9, zorder=7)

    # 5. Per-layer hin↔target gap dashes, drawn on GENERAL only (else clutter).
    if show_steering_dashes:
        for tgt in TARGETS:
            for li in range(len(layers)):
                hp = centroids[("general", "hin")][li]
                tp = centroids[("general", tgt)][li]
                ax.plot([hp[0], tp[0]], [hp[1], tp[1]],
                        linestyle=(0, (3, 3)), color="gray",
                        linewidth=0.6, alpha=0.35, zorder=3)

    # 6. Layer-index labels on the Hindi-general path.
    if show_layer_labels:
        hin_g = centroids[("general", "hin")]
        for li, layer in enumerate(layers):
            ax.annotate(
                f"L{layer}",
                xy=(hin_g[li, 0], hin_g[li, 1]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color=LANG_COLOURS["hin"],
                fontweight="bold", alpha=0.85, zorder=8,
            )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)
    if show_legend:
        # Build a clean legend: 4 lang colour entries + 2 corpus style entries.
        lang_handles = [
            plt.Line2D([0], [0], color=LANG_COLOURS[l], marker="*",
                       markersize=10, linewidth=2,
                       markeredgecolor="black", markeredgewidth=0.6,
                       label=LANG_LABELS[l])
            for l in LANGS
        ]
        corpus_handles = [
            plt.Line2D([0], [0], color="gray", marker="*", markersize=10,
                       linestyle="-", linewidth=2,
                       markeredgecolor="black", markeredgewidth=0.6,
                       label="general (parallel-250) ★ solid"),
            plt.Line2D([0], [0], color="gray", marker="D", markersize=8,
                       linestyle="--", linewidth=1.5,
                       markeredgecolor="black", markeredgewidth=0.6,
                       label="verb (verb-pairs-152) ◆ dashed"),
        ]
        leg1 = ax.legend(handles=lang_handles, loc="upper left",
                         fontsize=9, framealpha=0.95, title="Language")
        ax.add_artist(leg1)
        ax.legend(handles=corpus_handles, loc="upper right",
                  fontsize=9, framealpha=0.95, title="Corpus")


def fig_combined(
    membank_dir: Path, fig_dir: Path,
    layers: list[int], suffix: str, title: str,
    *, n_neighbors: int, min_dist: float, seed: int,
):
    data = {c: load_one(membank_dir / CORPUS_DIRS[c]) for c in CORPORA}
    n_gen = data["general"]["hin"].shape[0]
    n_verb = data["verb"]["hin"].shape[0]
    print(f"  general N/lang = {n_gen}, verb N/lang = {n_verb}, layers = {len(layers)}")

    fig, ax = plt.subplots(figsize=(13, 9.5))
    panel(
        ax, data, layers,
        n_neighbors=n_neighbors, min_dist=min_dist, seed=seed,
        show_legend=True, show_layer_labels=True, show_steering_dashes=True,
        title=title,
    )
    fig.tight_layout()
    out = fig_dir / f"llama_combined_constellation_{suffix}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--membank-dir", type=Path, default=DEFAULT_MEMBANK)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--n-neighbors", type=int, default=25)
    ap.add_argument("--min-dist", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-late", action="store_true")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_full:
        print("\n[combined constellation, full L0-L32]")
        fig_combined(
            args.membank_dir, args.fig_dir,
            layers=list(range(0, 33)),
            suffix="full_L0-L32",
            title=("llama: combined general + verb full-depth centroid "
                   "constellation (L0–L32)"),
            n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed,
        )

    if not args.skip_late:
        print("\n[combined constellation, late L17-L32]")
        fig_combined(
            args.membank_dir, args.fig_dir,
            layers=list(range(17, 33)),
            suffix="late_L17-L32",
            title=("llama: combined general + verb late-band centroid "
                   "constellation (L17–L32)"),
            n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed,
        )


if __name__ == "__main__":
    main()
