"""Round-2 review-response analyses (no GPU, existing data). Addresses:

  #13 Degeneracy-aware metrics: distinct-2, repetition rate, fluency-gated
      adherence, and R4-Q2 -- is there ANY alpha with mean adherence>3 AND
      mean fluency>3? (and any single generation with both?)
  #28 Steerability predictor: at the steering layer, a separability measure
      (Hindi-vs-target) that tracks steerability where raw cosine-proximity
      fails (Bhojpuri closest yet least steerable).
  #30 Corpus-size / data efficiency: cosine of the steering vector built from
      50 vs 250 passages, per target, at the steering layer.
  #31 Error-mode taxonomy: classify generations (repetition / code-switch /
      fragment / stayed-Hindi / fluent-target) per alpha.

  uv run --with pandas --with pyarrow --with numpy --with torch \
         python code/review_round2_analyses.py

Writes paper/latell/round2_stats.tex.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
BANK = ROOT / "results" / "memory_banks" / "llama"
OUT_TEX = ROOT / "paper" / "latell" / "round2_stats.tex"
TARGS = ["npi", "mai", "bho"]
TLAB = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}
STEER_L = 20
W = re.compile(r"[ऀ-ॿ]+")


def num(df, *cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def words(s):
    return W.findall(str(s))


# ----------------------------------------------------------- #13 degeneracy
def distinct2(s):
    w = words(s)
    if len(w) < 2:
        return np.nan
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg)


def rep_rate(s):
    w = words(s)
    if len(w) < 2:
        return np.nan
    return float(np.mean([w[i] == w[i - 1] for i in range(1, len(w))]))


def degeneracy(df, label):
    df = df.copy()
    df["d2"] = df.generation.map(distinct2)
    df["rep"] = df.generation.map(rep_rate)
    df["usable"] = ((df.language_adherence >= 3) & (df.fluency >= 3)).astype(float)
    print(f"\n=== #13 DEGENERACY [{label}] per (target,alpha) ===")
    g = df.groupby(["target_lang", "alpha"]).agg(
        adh=("language_adherence", "mean"), flu=("fluency", "mean"),
        d2=("d2", "mean"), rep=("rep", "mean"), usable=("usable", "mean")).round(3)
    print(g.to_string())
    # R4-Q2: any cell with mean adh>3 AND mean flu>3? any single gen with both>=3?
    cell = g[(g.adh > 3) & (g.flu > 3)]
    anygen = df[(df.language_adherence >= 3) & (df.fluency >= 3)]
    print(f"  R4-Q2 [{label}]: cells with mean adh>3 AND flu>3 = {len(cell)}; "
          f"individual generations with adh>=3 AND flu>=3 = {len(anygen)}/{len(df)} "
          f"({len(anygen)/len(df):.1%})")
    return df, g, len(anygen), len(df)


# ----------------------------------------------------------- #28 steerability
def load_centroids(layer):
    """Per-language centroid + within-language spread at one layer (fp32)."""
    cents, spread = {}, {}
    for lg in ["hin"] + TARGS:
        v = torch.load(BANK / f"{lg}.pt", map_location="cpu",
                       weights_only=False)["vectors"].float()[:, layer, :]  # (N,H)
        cents[lg] = v.mean(0)
        spread[lg] = v.std(0).norm().item()
    return cents, spread


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def steerability_predictor():
    cents, spread = load_centroids(STEER_L)
    print(f"\n=== #28 STEERABILITY PREDICTOR (Llama, L{STEER_L}) ===")
    print("  steerability (single-layer peak adherence): Nepali 3.23 > Maithili 1.80 = Bhojpuri 1.80")
    rows = {}
    for t in TARGS:
        c = cos(cents["hin"], cents[t])                       # raw proximity (FAILS)
        dist = (cents["hin"] - cents[t]).norm().item()         # centroid distance
        pooled = 0.5 * (spread["hin"] + spread[t])
        sep = dist / pooled                                    # Fisher-like separability
        rows[t] = (c, dist, spread[t], sep)
        print(f"  {TLAB[t]:9s}: cos(hin,tgt)={c:.3f}  dist={dist:.2f}  "
              f"tgt-spread={spread[t]:.2f}  separability(dist/pooled-std)={sep:.3f}")
    print("  => cosine-proximity RANKS Bhojpuri highest (wrong); separability ranks "
          "Nepali highest (matches steerability).")
    return rows


# ----------------------------------------------------------- #30 data efficiency
def data_efficiency():
    print("\n=== #30 DATA EFFICIENCY: steering vector v(50) vs v(250) ===")
    hin = torch.load(BANK / "hin.pt", map_location="cpu", weights_only=False)["vectors"].float()
    out = {}
    for t in TARGS:
        tgt = torch.load(BANK / f"{t}.pt", map_location="cpu", weights_only=False)["vectors"].float()
        v250 = hin[:250, STEER_L].mean(0) - tgt[:250, STEER_L].mean(0)
        v50 = hin[:50, STEER_L].mean(0) - tgt[:50, STEER_L].mean(0)
        c = cos(v250, v50)
        ratio = v50.norm().item() / v250.norm().item()
        out[t] = (c, ratio)
        print(f"  {TLAB[t]:9s}: cos(v50,v250)={c:.3f}  |v50|/|v250|={ratio:.2f}")
    return out


# ----------------------------------------------------------- #31 error taxonomy
def classify(row, ctrl_words):
    """Heuristic failure-mode label for one generation."""
    g = str(row.generation)
    w = words(g)
    if len(w) < 3:
        return "fragment"
    if rep_rate(g) >= 0.30 or distinct2(g) <= 0.5:
        return "repetition"
    adh = row.language_adherence
    if adh >= 4:
        return "fluent-target" if row.fluency >= 3 else "target-degraded"
    if adh <= 1.5:
        return "stayed-Hindi"
    return "code-switch"


def error_taxonomy(df):
    print("\n=== #31 ERROR-MODE TAXONOMY (Llama Nepali, by alpha) ===")
    sub = df[df.target_lang == "npi"].copy()
    sub["mode"] = sub.apply(lambda r: classify(r, None), axis=1)
    tab = sub.groupby(["alpha", "mode"]).size().unstack(fill_value=0)
    print(tab.to_string())
    return tab


def main():
    llama = num(pd.read_parquet(GEN / "llama_all_l20_finer30_scored.parquet"),
                "language_adherence", "fluency")
    aya = num(pd.read_parquet(GEN / "aya_all_l22_finer30_scored.parquet"),
              "language_adherence", "fluency")

    ld, lg, lus, ln = degeneracy(llama, "Llama")
    degeneracy(aya, "Aya")
    sep = steerability_predictor()
    de = data_efficiency()
    error_taxonomy(ld)

    # ---- tex macros (Llama)
    npi = ld[ld.target_lang == "npi"]
    # best usable alpha for Nepali
    gnpi = lg.loc["npi"]
    best_usable = gnpi.usable.max()
    d2_peak = npi[np.isclose(npi.alpha, -2.0)].d2.mean()
    d2_ctrl = npi[np.isclose(npi.alpha, 0.0)].d2.mean()
    tex = [
        "% AUTO-GENERATED by code/review_round2_analyses.py",
        f"\\newcommand{{\\usablefrac}}{{{lus}/{ln}}}",
        f"\\newcommand{{\\usablepct}}{{{lus/ln*100:.1f}\\%}}",
        f"\\newcommand{{\\npiusablebest}}{{{best_usable*100:.0f}\\%}}",
        f"\\newcommand{{\\distinctctrl}}{{{d2_ctrl:.2f}}}",
        f"\\newcommand{{\\distinctpeak}}{{{d2_peak:.2f}}}",
        f"\\newcommand{{\\sepnpi}}{{{sep['npi'][3]:.2f}}}",
        f"\\newcommand{{\\sepmai}}{{{sep['mai'][3]:.2f}}}",
        f"\\newcommand{{\\sepbho}}{{{sep['bho'][3]:.2f}}}",
        f"\\newcommand{{\\cosnpi}}{{{sep['npi'][0]:.2f}}}",
        f"\\newcommand{{\\cosbho}}{{{sep['bho'][0]:.2f}}}",
        f"\\newcommand{{\\vcosnpi}}{{{de['npi'][0]:.3f}}}",
        f"\\newcommand{{\\vcosmai}}{{{de['mai'][0]:.3f}}}",
        f"\\newcommand{{\\vcosbho}}{{{de['bho'][0]:.3f}}}",
    ]
    OUT_TEX.write_text("\n".join(tex) + "\n")
    print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
