"""Inter-judge agreement / panel-reliability statistics for the multi-LLM jury.

Consumes the wide panel parquet(s) written by code/multi_judge.py (columns
`<judge>__<dim>`), and for each rubric dimension computes the reliability
statistics a reviewer expects for a 3-judge, 1-5 ordinal rubric:

  - Krippendorff's alpha (ORDINAL)      : headline, chance-corrected, >=2 raters,
                                          ordinal distance, tolerates missing data.
  - ICC(2,k) and ICC(2,1)               : reliability of the averaged panel score
                                          / of a single average judge (two-way
                                          random, absolute agreement).
  - Pairwise quadratic-weighted Cohen's kappa : which pair drives disagreement.
  - Pairwise Spearman rho               : rank agreement between judges.
  - Exact- and within-1 agreement %     : reviewer-friendly raw agreement.
  - Gwet's AC1 (multi-rater)            : robust when a dimension is skewed
                                          (e.g. Maithili/Bhojpuri near-all 1-2).

Emits a console table for every dimension and writes LaTeX macros for the two
reported dimensions (adherence, fluency) to paper/mlj/judge_stats.tex.

Usage:
    .venv/bin/python code/judge_agreement.py                     # all panel files
    .venv/bin/python code/judge_agreement.py results/judge_panel/llama_finer30_panel.parquet
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import krippendorff
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
DIMS = ["language_adherence", "fluency", "faithfulness", "coherence", "overall_quality"]
REPORTED = ["language_adherence", "fluency"]          # -> LaTeX macros
TEX_OUT = ROOT / "paper" / "mlj" / "judge_stats.tex"
K_CATS = 5                                            # 1..5 rubric


def judge_names(df):
    return sorted({c.split("__")[0] for c in df.columns
                   if c.endswith("__language_adherence") and not c.startswith("panel_")})


def ratings_matrix(df, judges, dim):
    """(n_items, n_judges) float matrix; NaN where a judge has no score."""
    cols = [f"{j}__{dim}" for j in judges]
    return np.vstack([pd.to_numeric(df[c], errors="coerce").values for c in cols]).T


def icc(mat):
    """ICC(2,1) and ICC(2,k): two-way random effects, absolute agreement.
    mat: (n subjects, k raters), listwise-complete rows only."""
    X = mat[~np.isnan(mat).any(axis=1)]
    n, k = X.shape
    if n < 2 or k < 2:
        return float("nan"), float("nan")
    grand = X.mean()
    row_m, col_m = X.mean(1), X.mean(0)
    SSR = k * ((row_m - grand) ** 2).sum()
    SSC = n * ((col_m - grand) ** 2).sum()
    SST = ((X - grand) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    icc1 = (MSR - MSE) / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE))
    icck = (MSR - MSE) / (MSR + (MSC - MSE) / n)
    return icc1, icck


def gwet_ac1(mat):
    """Multi-rater Gwet's AC1 on categories 1..K_CATS (rows may have varying raters)."""
    rows = []
    for r in mat:
        r = r[~np.isnan(r)]
        if len(r) >= 2:
            rows.append(r)
    if not rows:
        return float("nan")
    cats = list(range(1, K_CATS + 1))
    Pa_terms, pi = [], {c: [] for c in cats}
    for r in rows:
        ri = len(r)
        counts = {c: int((r == c).sum()) for c in cats}
        Pa_terms.append(sum(v * (v - 1) for v in counts.values()) / (ri * (ri - 1)))
        for c in cats:
            pi[c].append(counts[c] / ri)
    Pa = float(np.mean(Pa_terms))
    pk = {c: float(np.mean(pi[c])) for c in cats}
    Pe = sum(pk[c] * (1 - pk[c]) for c in cats) / (K_CATS - 1)
    return (Pa - Pe) / (1 - Pe) if (1 - Pe) else float("nan")


def pairwise(mat, judges):
    """Quadratic-weighted kappa, Spearman, exact% and within-1% per judge pair."""
    out = []
    for a in range(len(judges)):
        for b in range(a + 1, len(judges)):
            x, y = mat[:, a], mat[:, b]
            m = ~(np.isnan(x) | np.isnan(y))
            xi, yi = x[m].astype(int), y[m].astype(int)
            if len(xi) < 2:
                continue
            try:
                qk = cohen_kappa_score(xi, yi, weights="quadratic",
                                       labels=list(range(1, K_CATS + 1)))
            except Exception:
                qk = float("nan")
            rho = spearmanr(xi, yi).correlation if np.std(xi) and np.std(yi) else float("nan")
            exact = float((xi == yi).mean())
            within1 = float((np.abs(xi - yi) <= 1).mean())
            out.append((f"{judges[a]}~{judges[b]}", qk, rho, exact, within1))
    return out


def kripp_ordinal(mat):
    # krippendorff expects reliability_data as (raters, units)
    data = mat.T
    try:
        return krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
    except Exception as e:  # e.g. all-identical -> undefined
        print(f"    (krippendorff alpha undefined: {e})")
        return float("nan")


def main():
    args = sys.argv[1:]
    files = args if args else sorted(glob.glob(str(ROOT / "results/judge_panel/*_panel.parquet")))
    if not files:
        sys.exit("no panel parquet(s) found in results/judge_panel/ — run multi_judge.py first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    judges = judge_names(df)
    print(f"Loaded {len(df)} rows from {len(files)} file(s); judges = {judges}\n")

    tex = {}
    for dim in DIMS:
        mat = ratings_matrix(df, judges, dim)
        alpha = kripp_ordinal(mat)
        icc1, icck = icc(mat)
        ac1 = gwet_ac1(mat)
        pw = pairwise(mat, judges)
        print(f"== {dim} ==")
        print(f"  Krippendorff ordinal alpha : {alpha:.3f}")
        print(f"  ICC(2,k)={icck:.3f}   ICC(2,1)={icc1:.3f}   Gwet AC1={ac1:.3f}")
        for name, qk, rho, ex, w1 in pw:
            print(f"  {name:28} wkappa={qk:.3f}  spearman={rho:.3f}  "
                  f"exact={ex*100:4.1f}%  within1={w1*100:5.1f}%")
        # panel-mean vs old single-judge headline (sanity): mean of panel_mean
        if f"panel_mean__{dim}" in df.columns:
            pm = pd.to_numeric(df[f"panel_mean__{dim}"], errors="coerce").mean()
            print(f"  panel-mean grand mean      : {pm:.3f}")
        print()
        if dim in REPORTED:
            tag = "adh" if dim == "language_adherence" else "flu"
            tex[f"panelalpha{tag}"] = f"{alpha:.2f}"
            tex[f"panelicc{tag}"] = f"{icck:.2f}"
            tex[f"panelac{tag}"] = f"{ac1:.2f}"
            if pw:
                mean_exact = np.mean([p[3] for p in pw]) * 100
                mean_w1 = np.mean([p[4] for p in pw]) * 100
                tex[f"panelexact{tag}"] = f"{mean_exact:.0f}"
                tex[f"panelwithinone{tag}"] = f"{mean_w1:.0f}"

    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% AUTO-GENERATED by code/judge_agreement.py -- do not edit by hand.",
             f"% Panel judges: {', '.join(judges)}. n={len(df)} generations."]
    for k, v in tex.items():
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    TEX_OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {TEX_OUT}  ({len(tex)} macros)")


if __name__ == "__main__":
    main()
