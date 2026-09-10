"""Four researcher-grade analysis figures from local data (no GPU).

1. vnorm_ladder.png         — |v_l| across layers per target, both models. The
   steering-vector magnitude encodes linguistic distance (npi>mai>bho).
2. pareto_scatter.png       — unified adherence-vs-fluency scatter (both models ×
   3 targets × all α). The fundamental single-layer trade-off + the Nepali-Llama
   outlier that crosses into "mostly target".
3. judge_profile.png        — 5-dim judge rubric at control vs adherence-peak α
   (Llama, per target): steering buys language identity at the cost of fluency/
   faithfulness/coherence.
4. geometry_steerability.png— peak adherence vs anchor-target proximity at the
   steering layer (6 points). Steerability tracks geometric closeness.

  uv run --with pandas --with pyarrow --with torch --with numpy --with matplotlib \
      python code/visualize_paper_analyses.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
TARGETS = ["npi", "mai", "bho"]
TLAB = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}
TCOL = {"npi": "#2ca02c", "mai": "#1f77b4", "bho": "#ff7f0e"}
STEER = {"llama": 20, "aya": 22}
DIMS = ["language_adherence", "fluency", "faithfulness", "coherence", "overall_quality"]
DLAB = ["Lang.\nadherence", "Fluency", "Faithful.", "Coherence", "Overall"]


def scored(model):
    # Use the 3-judge panel mean (results/judge_panel/*_finer30_panel.parquet) so
    # figures match the panel-based numbers in the paper tables.
    df = pd.read_parquet(FIG.parent / "judge_panel" / f"{model}_finer30_panel.parquet")
    for d in DIMS:
        df[d] = pd.to_numeric(df[f"panel_mean__{d}"], errors="coerce")
    return df


def fig_vnorm():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    for ax, model in zip(axes, ["llama", "aya"]):
        for t in TARGETS:
            p = torch.load(ROOT / "results" / "constellations" / model / f"{t}.pt",
                           map_location="cpu", weights_only=False)
            norms = np.array(p["per_layer_norm"])
            ax.plot(range(len(norms)), norms, marker="o", markersize=3, linewidth=2,
                    color=TCOL[t], label=f"Hindi→{TLAB[t]}")
        ax.axvline(STEER[model], color="black", ls="--", lw=1.2, alpha=0.6)
        ax.text(STEER[model] - 0.5, ax.get_ylim()[1] * 0.9, f"steer L{STEER[model]}",
                rotation=90, ha="right", fontsize=8)
        ax.set_title(f"{model}", fontsize=12); ax.set_xlabel("Layer")
        ax.set_ylabel("|v_l|  (steering-vector L2 norm)"); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    # in-image overall title removed: the LaTeX \caption covers it (journal requirement)
    fig.tight_layout(); fig.savefig(FIG / "llama" / "vnorm_ladder.png", dpi=190, bbox_inches="tight")
    plt.close(fig); print("wrote vnorm_ladder.png")


def fig_pareto_scatter():
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for model, mk in [("llama", "o"), ("aya", "s")]:
        g = scored(model).groupby(["target_lang", "alpha"])[["language_adherence", "fluency"]].mean()
        for t in TARGETS:
            sub = g.loc[t]
            ax.plot(sub["fluency"], sub["language_adherence"], "-", color=TCOL[t], alpha=0.4, zorder=2)
            ax.scatter(sub["fluency"], sub["language_adherence"], marker=mk, s=70, color=TCOL[t],
                       edgecolor="black", linewidths=0.6, zorder=4)
    ax.axhline(3.0, color="green", ls="--", lw=1.2, alpha=0.7)
    ax.text(4.6, 3.05, '"mostly target" (≥3)', color="green", fontsize=9, ha="right")
    from matplotlib.lines import Line2D
    leg = ([Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=9, label="Llama (L20)"),
            Line2D([0],[0], marker="s", color="w", markerfacecolor="gray", markersize=9, label="Aya (L22)")]
           + [Line2D([0],[0], marker="o", color="w", markerfacecolor=TCOL[t], markersize=9, label=TLAB[t]) for t in TARGETS])
    ax.legend(handles=leg, fontsize=9, loc="upper right")
    ax.set_xlabel("Fluency (mean, 1–5)"); ax.set_ylabel("Language adherence (mean, 1–5)")
    ax.set_title("The single-layer adherence–fluency frontier (both models × 3 targets × α)\n"
                 "Only Llama-Nepali crosses into 'mostly target'; everything else trades fluency for partial adherence",
                 fontsize=11.5, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "llama" / "pareto_scatter_combined.png", dpi=190, bbox_inches="tight")
    plt.close(fig); print("wrote pareto_scatter_combined.png")


def fig_judge_profile():
    df = scored("llama")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, t in zip(axes, TARGETS):
        g = df[df.target_lang == t].groupby("alpha")[DIMS].mean()
        ctrl = g.loc[0.0]
        peak_a = g["language_adherence"].idxmax()
        peak = g.loc[peak_a]
        x = np.arange(len(DIMS)); w = 0.38
        ax.bar(x - w/2, ctrl[DIMS].values, w, label="control (α=0)", color="#1f77b4")
        ax.bar(x + w/2, peak[DIMS].values, w, label=f"peak α={peak_a}", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(DLAB, fontsize=8)
        ax.set_title(TLAB[t], color=TCOL[t], fontsize=12); ax.set_ylim(0, 5); ax.grid(True, axis="y", alpha=0.25)
        if t == "npi":
            ax.set_ylabel("Mean score (1–5)"); ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("5-dimension judge profile, Llama L20: steering buys language adherence at the cost of "
                 "fluency, faithfulness, and coherence", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "llama" / "judge_profile_control_vs_peak.png", dpi=190, bbox_inches="tight")
    plt.close(fig); print("wrote judge_profile_control_vs_peak.png")


def sep_ratio(model, t):
    """Separability at the steering layer: ||mu_hin - mu_tgt|| / pooled within-language
    spread (mean of the two std-vector norms). Higher = more linearly separable."""
    sl = STEER[model]
    def bank(lg):
        return torch.load(ROOT / "results/memory_banks" / model / f"{lg}.pt",
                          map_location="cpu", weights_only=False)["vectors"].float()[:, sl]
    Xh, Xt = bank("hin"), bank(t)
    dist = (Xh.mean(0) - Xt.mean(0)).norm().item()
    spread = 0.5 * (Xh.std(0).norm().item() + Xt.std(0).norm().item())
    return dist / spread


def fig_geometry_steerability():
    """Two panels: peak adherence vs (a) cosine proximity -- does NOT predict; and
    (b) linear separability -- DOES predict. Llama = circles, Aya = squares."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    llama_prox, llama_sep = [], []
    for model, mk in [("llama", "o"), ("aya", "s")]:
        ca = json.load(open(ROOT / "results" / "memory_banks" / model / "centroid_analysis.json"))
        pl = ca["per_layer"]; sl = STEER[model]
        g = scored(model).groupby(["target_lang", "alpha"])["language_adherence"].mean()
        for t in TARGETS:
            prox = pl[f"hin_vs_{t}"][sl]                  # cosine (higher = closer)
            sep = sep_ratio(model, t)
            peak = float(g.loc[t].max())
            if model == "llama":
                llama_prox.append((prox, peak)); llama_sep.append((sep, peak))
            for ax, x in [(a1, prox), (a2, sep)]:
                ax.scatter(x, peak, marker=mk, s=130, color=TCOL[t], edgecolor="black",
                           linewidths=0.8, zorder=4)
                ax.annotate(f"{model[:1].upper()}·{TLAB[t][:3]}", (x, peak), xytext=(5, 4),
                            textcoords="offset points", fontsize=8)
    # Llama within-model trend: proximity reverses, separability rises
    for ax, pts in [(a1, llama_prox), (a2, llama_sep)]:
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="#888",
                lw=1.4, alpha=0.7, zorder=2, label="Llama trend")
    a1.set_xlabel(r"cosine proximity  cos($\mu_{\mathrm{hin}},\mu_{\mathrm{tgt}}$) at steer layer")
    a2.set_xlabel(r"separability  $\|\mu_{\mathrm{hin}}-\mu_{\mathrm{tgt}}\|\,/\,$within-lang spread")
    a1.set_ylabel("Peak language adherence (1--5)")
    a1.text(0.5, 1.02, "proximity does not predict", transform=a1.transAxes, ha="center",
            fontsize=10, style="italic")
    a2.text(0.5, 1.02, "separability predicts", transform=a2.transAxes, ha="center",
            fontsize=10, style="italic")
    for ax in (a1, a2):
        ax.axhline(3.0, color="green", ls="--", lw=1.0, alpha=0.6)
        ax.grid(True, alpha=0.25); ax.set_ylim(0.9, 4.4)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=9, label="Llama"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=9, label="Aya")],
              fontsize=8.5, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIG / "llama" / "geometry_vs_steerability.png", dpi=190, bbox_inches="tight")
    plt.close(fig); print("wrote geometry_vs_steerability.png (2-panel proximity vs separability)")


def main():
    fig_vnorm(); fig_pareto_scatter(); fig_judge_profile(); fig_geometry_steerability()


if __name__ == "__main__":
    main()
