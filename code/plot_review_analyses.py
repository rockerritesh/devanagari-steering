"""Plots from the reviewer-response analyses (slides / repo / appendix; the 8pp
submission keeps these as tables+text). Writes to results/figures/review/.

  (A) ICL vs single-layer steering: grouped bars, adherence + fluency, 3 targets.
  (B) bf16 vs NF4 geometry: per-layer probe + silhouette overlay (hourglass holds).
  (C) Judge vs independent GlotLID: agreement scatter (r=0.88) + Jaccard-vs-alpha.

  uv run --with pandas --with pyarrow --with numpy --with matplotlib \
         --with torch --with scikit-learn --with "numpy<2" \
         --with fasttext-wheel --with huggingface_hub \
         python code/plot_review_analyses.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
OUT = ROOT / "results" / "figures" / "review"
OUT.mkdir(parents=True, exist_ok=True)
TARGS = ["npi", "mai", "bho"]
TLAB = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}
C_STEER, C_ICL = "#c44e52", "#4c72b0"


def num(df, *cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----- steering peaks (from the finer30 sweep) -----
STEER_PEAK = {  # target -> (adh, flu) at the reported peak alpha
    "npi": (3.23, 1.33), "mai": (1.80, 2.07), "bho": (1.80, 1.80)}


def fig_icl_vs_steering():
    d = num(pd.read_parquet(GEN / "llama_all_icl_scored.parquet"),
            "language_adherence", "fluency")
    icl = d[d.condition == "icl3"].groupby("target_lang")[
        ["language_adherence", "fluency"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(TARGS)); w = 0.38
    for ax, metric, title in [(axes[0], "language_adherence", "Language adherence"),
                              (axes[1], "fluency", "Fluency")]:
        steer = [STEER_PEAK[t][0 if metric == "language_adherence" else 1] for t in TARGS]
        iclv = [icl.loc[t, metric] for t in TARGS]
        ax.bar(x - w/2, steer, w, label="single-layer steering (peak)", color=C_STEER)
        ax.bar(x + w/2, iclv, w, label="3-shot ICL (no steering)", color=C_ICL)
        ax.set_xticks(x); ax.set_xticklabels([TLAB[t] for t in TARGS])
        ax.set_ylim(0, 5); ax.set_title(title, fontsize=11)
        ax.set_ylabel("score (1–5)"); ax.axhline(3, color="#999", lw=0.7, ls=":")
        for xi, (s, i) in enumerate(zip(steer, iclv)):
            ax.text(xi - w/2, s + 0.08, f"{s:.2f}", ha="center", fontsize=8)
            ax.text(xi + w/2, i + 0.08, f"{i:.2f}", ha="center", fontsize=8)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Few-shot ICL beats single-layer steering on both axes and breaks "
                 "the Maithili/Bhojpuri ceiling", fontsize=11, y=1.02)
    fig.tight_layout()
    p = OUT / "icl_vs_steering.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_bf16_geometry():
    import torch
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    LANGS = ["hin", "mai", "npi", "bho"]

    def load(model, limit=None):
        mats, lab = [], []
        for li, lg in enumerate(LANGS):
            v = torch.load(ROOT / "results/memory_banks" / model / f"{lg}.pt",
                           map_location="cpu", weights_only=False)["vectors"].float().numpy()
            if limit: v = v[:limit]
            mats.append(v); lab.append(np.full(v.shape[0], li))
        return np.transpose(np.concatenate(mats), (1, 0, 2)), np.concatenate(lab)

    def curve(Xl, y):
        pr, si = [], []
        for l in range(Xl.shape[0]):
            Z = StandardScaler().fit_transform(Xl[l])
            Z = PCA(n_components=min(50, Z.shape[0]-1), random_state=0).fit_transform(Z)
            pr.append(cross_val_score(LogisticRegression(max_iter=2000), Z, y, cv=5).mean())
            si.append(silhouette_score(Z, y))
        return np.array(pr), np.array(si)

    Xn, yn = load("llama", 80); Xb, yb = load("llama_bf16")
    pn, sn = curve(Xn, yn); pb, sb = curve(Xb, yb)
    x = np.arange(len(pn))
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax2 = ax1.twinx()
    ax1.plot(x, pn, color="#6a3d9a", lw=2, label="probe acc (NF4)")
    ax1.plot(x, pb, color="#6a3d9a", lw=2, ls="--", label="probe acc (bf16)")
    ax2.plot(x, sn, color="#2ca02c", lw=2, marker="o", ms=3, label="silhouette (NF4)")
    ax2.plot(x, sb, color="#2ca02c", lw=2, ls="--", marker="s", ms=3, label="silhouette (bf16)")
    ax1.axvline(20, color="#c44e52", lw=1, ls=":"); ax1.text(20.3, 0.5, "steer L20", color="#c44e52", fontsize=8)
    ax1.set_xlabel("layer"); ax1.set_ylabel("probe accuracy (4-way)", color="#6a3d9a")
    ax2.set_ylabel("silhouette", color="#2ca02c"); ax1.set_ylim(0, 1.05)
    r = np.corrcoef(sn, sb)[0, 1]
    fig.suptitle(f"Hourglass is not a 4-bit artifact: NF4 vs full-bf16 curves "
                 f"coincide (silhouette r={r:.3f})", fontsize=10.5, y=1.02)
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, la1+la2, fontsize=7.5, loc="center right")
    fig.tight_layout()
    p = OUT / "bf16_vs_nf4_geometry.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_judge_vs_glotlid():
    import fasttext
    from huggingface_hub import hf_hub_download
    ft = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
    GLOT = {"npi": "npi_Deva", "mai": "mai_Deva", "bho": "bho_Deva"}
    d = num(pd.read_parquet(GEN / "llama_all_l20_finer30_scored.parquet"),
            "language_adherence")
    rows = []
    for t in TARGS:
        for a in sorted(d[d.target_lang == t].alpha.unique()):
            sub = d[(d.target_lang == t) & np.isclose(d.alpha, a)]
            preds = [ft.predict(re.sub(r"\s+", " ", str(g)).strip(), 1)[0][0].replace("__label__", "")
                     for g in sub.generation if str(g).strip()]
            if not preds: continue
            rows.append((t, a, np.mean([p == GLOT[t] for p in preds]),
                         sub.language_adherence.mean()))
    df = pd.DataFrame(rows, columns=["t", "a", "glot", "adh"])
    r = np.corrcoef(df.adh, df.glot)[0, 1]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    cmap = {"npi": C_STEER, "mai": "#55a868", "bho": "#8172b3"}
    for t in TARGS:
        s = df[df.t == t]
        ax.scatter(s.adh, s.glot*100, c=cmap[t], s=55, label=TLAB[t], edgecolors="white")
    ax.set_xlabel("LLM-judge adherence (1–5)")
    ax.set_ylabel("GlotLID → target (%)  [independent, non-LLM]")
    ax.set_title(f"Judge vs independent GlotLID agree (Pearson r={r:.2f})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "judge_vs_glotlid.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_icl_vs_steering()
    fig_bf16_geometry()
    fig_judge_vs_glotlid()
    print("\nall plots in", OUT)
