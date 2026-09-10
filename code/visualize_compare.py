"""Overlay general (parallel-250) vs verb (verb-pairs-152) per-layer
cosine separation curves, on a single figure.

The general corpus produces a deep U-shape valley in mid-deep layers
(language-identity band L17-L29 with hin~target dropping to ~0.7-0.85). The
verb-pairs corpus produces a nearly-flat trace at cos ~0.99 because mean-
pooling on short sentences with shared content drowns out the verb signal.
The visual contrast is the main result of the verb-axis ablation.

Two figures are produced:

  llama_compare_cosine.png       — overlay line plot
  llama_compare_constellation.png — side-by-side late-band constellations
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
MEMBANK_DIR = PROJECT_ROOT / "results" / "memory_banks"

LANGS = ["hin", "mai", "npi", "bho"]
TARGETS = ["mai", "npi", "bho"]
LANG_COLOURS = {"hin": "#d62728", "mai": "#1f77b4", "npi": "#2ca02c", "bho": "#ff7f0e"}
LANG_LABELS = {"hin": "Hindi", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri"}


def load_per_layer_cos(model_dir: Path) -> dict[str, list[float]]:
    """Read centroid_analysis.json (written by analyze_centroids.py)."""
    payload = json.loads((model_dir / "centroid_analysis.json").read_text())
    return payload["per_layer"]


def fig_cosine_overlay():
    gen = load_per_layer_cos(MEMBANK_DIR / "llama")
    verb = load_per_layer_cos(MEMBANK_DIR / "llama_verb")
    L = len(gen["hin_vs_mai"])
    layers = np.arange(L)

    fig, ax = plt.subplots(figsize=(10, 6))
    for target in TARGETS:
        c = LANG_COLOURS[target]
        ax.plot(layers, gen[f"hin_vs_{target}"], color=c, linewidth=2.0,
                marker="o", markersize=3.5, label=f"general · hin↔{target}")
        ax.plot(layers, verb[f"hin_vs_{target}"], color=c, linewidth=1.4,
                linestyle="--", marker="x", markersize=3.5,
                label=f"verb · hin↔{target}")
    ax.plot(layers, gen["hin_vs_random"], color="gray",
            linestyle=":", linewidth=1, label="random baseline (general)")

    ax.set_xlabel("Layer index (0 = embedding output, 32 = final residual)")
    ax.set_ylabel("Cosine similarity of per-layer centroids")
    ax.set_title("llama: general (parallel-250) vs verb (verb-pairs-152) "
                 "per-layer Hindi-vs-sister alignment")
    ax.set_xticks(np.arange(0, L, 2))
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=8, ncol=2)

    # annotate the deep-U for the general curves vs the flatness of verb
    ax.annotate("general U-shape\n(language-identity band)",
                xy=(26, 0.69), xytext=(22, 0.40),
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.7),
                fontsize=9, color="black", ha="center")
    ax.annotate("verb curves stay\nflat at ~0.99",
                xy=(20, 0.99), xytext=(13, 1.04),
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.7),
                fontsize=9, color="black", ha="center")

    fig.tight_layout()
    out = FIG_DIR / "llama_compare_cosine.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def constellation_panel(ax, model_subdir, layers, *, n_neighbors, min_dist, seed, title):
    data = {}
    for lang in LANGS:
        payload = torch.load(MEMBANK_DIR / model_subdir / f"{lang}.pt",
                             map_location="cpu", weights_only=False)
        data[lang] = payload["vectors"].float()
    n_per = data["hin"].shape[0]
    chunks = []
    for lang in LANGS:
        for layer in layers:
            chunks.append(data[lang][:, layer, :].numpy())
    combined = np.concatenate(chunks, axis=0)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, metric="cosine", random_state=seed)
    coords = reducer.fit_transform(combined)
    pts = {}
    cur = 0
    for lang in LANGS:
        for layer in layers:
            pts[(lang, layer)] = coords[cur : cur + n_per]
            cur += n_per
    for (lang, _layer), p in pts.items():
        ax.scatter(p[:, 0], p[:, 1], s=2.5, alpha=0.07,
                   color=LANG_COLOURS[lang], linewidths=0)
    centroids = {}
    for lang in LANGS:
        cents = np.stack([pts[(lang, layer)].mean(axis=0) for layer in layers])
        centroids[lang] = cents
        ax.plot(cents[:, 0], cents[:, 1], color=LANG_COLOURS[lang],
                linewidth=2.0, alpha=0.95, zorder=5)
        ax.scatter(cents[:, 0], cents[:, 1], color=LANG_COLOURS[lang],
                   s=110, marker="*", edgecolors="black", linewidths=0.6,
                   zorder=6, label=LANG_LABELS[lang])
        ax.scatter(cents[0, 0], cents[0, 1], color=LANG_COLOURS[lang],
                   s=140, marker="s", edgecolors="black", linewidths=0.9,
                   zorder=7)
    for lang in TARGETS:
        for li in range(len(layers)):
            hp = centroids["hin"][li]
            tp = centroids[lang][li]
            ax.plot([hp[0], tp[0]], [hp[1], tp[1]],
                    linestyle=(0, (3, 3)), color="gray",
                    linewidth=0.6, alpha=0.4, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.grid(True, alpha=0.2)


def fig_constellation_compare():
    layers = list(range(21, 26))
    fig, (ax_gen, ax_verb) = plt.subplots(1, 2, figsize=(14, 6))
    constellation_panel(
        ax_gen, "llama", layers,
        n_neighbors=25, min_dist=0.15, seed=0,
        title="general (parallel-250) — Late Layers L21-L25",
    )
    constellation_panel(
        ax_verb, "llama_verb", layers,
        n_neighbors=25, min_dist=0.15, seed=0,
        title="verb (verb-pairs-152) — Late Layers L21-L25",
    )
    handles, labels = ax_gen.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, markerscale=1.5)
    fig.suptitle(
        "llama: late-layer constellation comparison "
        "(general spreads languages; verb collapses them into one cluster)",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = FIG_DIR / "llama_compare_constellation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_cosine_overlay()
    fig_constellation_compare()


if __name__ == "__main__":
    main()
