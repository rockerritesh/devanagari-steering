"""Per-layer language-separation visualisations from centroid_analysis.json.

Uses only the lightweight per-layer cosine curves (no memory-bank tensors), so it
runs offline without the GPU VM. Separation := 1 - cosine(centroid_i, centroid_j);
higher = more linguistically distinct at that layer.

Produces, per model (llama, aya):
  results/figures/<model>/<model>_separation_heatmap.png
  results/figures/<model>/<model>_separation_matrices.png
  results/figures/<model>/<model>_separation_curves.png
And one cross-model figure:
  results/figures/<model>/llama_vs_aya_separation_hourglass.png   (written under llama/)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LANGS = ["hin", "mai", "npi", "bho"]
LANG_LABELS = {"hin": "Hindi", "mai": "Maithili", "npi": "Nepali", "bho": "Bhojpuri"}
PAIRS = ["hin_vs_npi", "hin_vs_mai", "hin_vs_bho", "mai_vs_npi", "mai_vs_bho", "npi_vs_bho"]
PAIR_LABELS = {p: f"{LANG_LABELS[p.split('_vs_')[0]][:3]}–{LANG_LABELS[p.split('_vs_')[1]][:3]}" for p in PAIRS}
STEER_LAYER = {"llama": 20, "aya": 22}
HIN_PAIRS = ["hin_vs_npi", "hin_vs_mai", "hin_vs_bho"]
HIN_COLOURS = {"hin_vs_npi": "#2ca02c", "hin_vs_mai": "#1f77b4", "hin_vs_bho": "#ff7f0e"}


def load(model):
    d = json.load(open(ROOT / "results" / "memory_banks" / model / "centroid_analysis.json"))
    pl = {k: np.array(v) for k, v in d["per_layer"].items()}
    return pl, len(next(iter(pl.values())))


def sep(pl, pair):
    return 1.0 - pl[pair]


def matrix_at(pl, layer):
    """4x4 separation matrix at one layer."""
    m = np.zeros((4, 4))
    for i, a in enumerate(LANGS):
        for j, b in enumerate(LANGS):
            if i == j:
                continue
            key = f"{a}_vs_{b}" if f"{a}_vs_{b}" in pl else f"{b}_vs_{a}"
            m[i, j] = 1.0 - pl[key][layer]
    return m


def fig_heatmap(model, pl, nL, out):
    data = np.vstack([sep(pl, p) for p in PAIRS])           # (6 pairs, nL layers)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    im = ax.imshow(data, aspect="auto", cmap="magma", origin="upper",
                   vmin=0, vmax=max(0.25, data.max()))
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels([PAIR_LABELS[p] for p in PAIRS])
    ax.set_xticks(range(0, nL, 2))
    ax.set_xlabel("Layer (0 = embedding … 32 = final)")
    sl = STEER_LAYER[model]
    ax.axvline(sl, color="cyan", linewidth=2.0, linestyle="--")
    ax.text(sl + 0.3, -0.7, f"steer L{sl}", color="cyan", fontsize=9, va="bottom")
    ax.set_title(f"{model}: per-layer language separation (1 − cosine of centroids). "
                 f"Dark = merged (shared semantics), bright = distinct.", fontsize=11)
    fig.colorbar(im, ax=ax, label="separation (1 − cos)", fraction=0.025, pad=0.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_matrices(model, pl, nL, out):
    sl = STEER_LAYER[model]
    layers = [0, 8, 14, sl, 26, nL - 1]
    vmax = max(matrix_at(pl, l).max() for l in layers)
    fig, axes = plt.subplots(1, len(layers), figsize=(3 * len(layers), 3.4))
    for ax, l in zip(axes, layers):
        m = matrix_at(pl, l)
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([LANG_LABELS[x][:3] for x in LANGS], fontsize=8)
        ax.set_yticklabels([LANG_LABELS[x][:3] for x in LANGS], fontsize=8)
        tag = " (steer)" if l == sl else (" (embed)" if l == 0 else (" (final)" if l == nL - 1 else ""))
        ax.set_title(f"L{l}{tag}", fontsize=10)
        for i in range(4):
            for j in range(4):
                if i != j:
                    ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                            color="white" if m[i, j] < vmax * 0.6 else "black", fontsize=7)
    fig.suptitle(f"{model}: pairwise language-separation matrices across depth — "
                 f"manifold collapses (mid) then re-expands (deep)", fontsize=11.5, y=1.04)
    fig.colorbar(im, ax=axes, label="separation (1 − cos)", fraction=0.012, pad=0.01)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_curves(model, pl, nL, out):
    sl = STEER_LAYER[model]
    x = np.arange(nL)
    fig, ax = plt.subplots(figsize=(11, 5))
    for p in HIN_PAIRS:
        ax.plot(x, sep(pl, p), marker="o", markersize=3.5, linewidth=2.2,
                color=HIN_COLOURS[p], label=f"Hindi–{LANG_LABELS[p.split('_vs_')[1]]}")
    for p in ["mai_vs_npi", "mai_vs_bho", "npi_vs_bho"]:
        ax.plot(x, sep(pl, p), linewidth=1.0, alpha=0.55, linestyle=":",
                label=PAIR_LABELS[p])
    if "hin_vs_random" in pl:
        ax.plot(x, sep(pl, "hin_vs_random"), color="gray", linewidth=1.4,
                linestyle="--", label="Hindi–random (baseline)")
    ax.axvline(sl, color="red", linewidth=1.6, linestyle="--", alpha=0.7)
    ax.text(sl - 0.4, 0.55, f"steer L{sl}\n(onset of re-divergence)",
            color="red", fontsize=9, va="center", ha="right")
    ax.set_xlabel("Layer (0 = embedding … 32 = final)")
    ax.set_ylabel("Separation (1 − cosine)")
    ax.set_title(f"{model}: the hourglass — languages distinct early, merge mid-network, "
                 f"re-separate deep", fontsize=11.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", ncol=4, fontsize=8, framealpha=0.92)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_cross_model(plL, nLl, plA, nLa, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    for pl, nL, model, style in [(plL, nLl, "llama", "-"), (plA, nLa, "aya", "--")]:
        x = np.arange(nL)
        for p in HIN_PAIRS:
            ax.plot(x, sep(pl, p), style, linewidth=2.0, color=HIN_COLOURS[p],
                    alpha=0.95 if model == "llama" else 0.6,
                    label=f"{model}: Hin–{LANG_LABELS[p.split('_vs_')[1]][:3]}")
        ax.axvline(STEER_LAYER[model], color="red" if model == "llama" else "purple",
                   linewidth=1.3, linestyle=":", alpha=0.6)
    ax.set_xlabel("Layer (0 = embedding … 32 = final)")
    ax.set_ylabel("Separation (1 − cosine)")
    ax.set_title("Cross-model hourglass: Hindi-vs-sister separation, Llama (solid) vs Aya (dashed). "
                 "Aya's valley is deeper/wider.", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", ncol=3, fontsize=8, framealpha=0.92)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    banks = {}
    for model in ["llama", "aya"]:
        pl, nL = load(model)
        banks[model] = (pl, nL)
        d = ROOT / "results" / "figures" / model
        fig_heatmap(model, pl, nL, d / f"{model}_separation_heatmap.png")
        fig_matrices(model, pl, nL, d / f"{model}_separation_matrices.png")
        fig_curves(model, pl, nL, d / f"{model}_separation_curves.png")
    (plL, nLl), (plA, nLa) = banks["llama"], banks["aya"]
    fig_cross_model(plL, nLl, plA, nLa,
                    ROOT / "results" / "figures" / "llama" / "llama_vs_aya_separation_hourglass.png")


if __name__ == "__main__":
    main()
