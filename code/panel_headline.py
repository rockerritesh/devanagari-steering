"""Consolidate ALL panel-scored files into the paper's judge-based headline numbers.

Reads results/judge_panel/*_panel.parquet (written by multi_judge.py) and re-derives
every judge-dependent number the paper reports, from the panel MEAN (with MEDIAN and a
bootstrap CI as robustness), replacing the old single-judge values. Geometry numbers
(silhouette, probe, separability, vector norms, bf16) are judge-INDEPENDENT and are
NOT touched here.

Prints a before/after table and writes LaTeX macros to paper/mlj/panel_headline.tex.

    .venv/bin/python code/panel_headline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "results" / "judge_panel"
TEX = ROOT / "paper" / "mlj" / "panel_headline.tex"
ADH, FLU = "panel_mean__language_adherence", "panel_mean__fluency"
MED = "panel_median__language_adherence"
TL = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}
RNG = np.random.default_rng(0)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def boot_ci(vals, n=10000):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def peak_cell(df, t, col=ADH):
    """Return (peak_alpha, peak_mean, the row-values at that alpha) for target t."""
    sub = df[df.target_lang == t]
    g = sub.groupby("alpha")
    m = g[col].apply(lambda s: num(s).mean())
    a = m.idxmax()
    return a, float(m.max()), num(sub[sub.alpha == a][col]).values


def main():
    macros = {}
    print("=" * 74)
    print("HEADLINE (single-judge OLD  ->  PANEL mean [median], flu@peak)")
    print("=" * 74)

    OLD = {("llama", "npi"): 3.23, ("llama", "mai"): 1.80, ("llama", "bho"): 1.80,
           ("aya", "npi"): 1.47, ("aya", "mai"): 1.57, ("aya", "bho"): 1.57}
    for model in ["llama", "aya"]:
        f = PANEL / f"{model}_finer30_panel.parquet"
        if not f.exists():
            print(f"  (missing {f.name})")
            continue
        df = pd.read_parquet(f)
        print(f"\n{model.upper()} finer30:")
        for t in TL:
            a, pk, vals = peak_cell(df, t)
            lo, hi = boot_ci(vals)
            med = num(df[df.target_lang == t].groupby("alpha")[MED].mean()).loc[a]
            fl = num(df[(df.target_lang == t) & (df.alpha == a)][FLU]).mean()
            ctrl = num(df[(df.target_lang == t) & (df.alpha == 0.0)][ADH]).mean()
            print(f"  {TL[t]:9} old={OLD[(model,t)]:.2f}  ->  {pk:.2f} "
                  f"[{lo:.2f},{hi:.2f}] (med {med:.2f})  flu@peak={fl:.2f}  "
                  f"ctrl={ctrl:.2f}  d={pk-ctrl:+.2f}  a={a}")
            tag = f"{model}{t}"
            macros[f"pan{tag}adh"] = f"{pk:.2f}"
            macros[f"pan{tag}adhCI"] = f"[{lo:.2f}, {hi:.2f}]"
            macros[f"pan{tag}flu"] = f"{fl:.2f}"

    # usable fraction (adherence AND fluency both >=3), per-generation panel mean
    print("\n" + "-" * 74 + "\nUsable fraction (adherence & fluency both >=3):")
    for model in ["llama", "aya"]:
        f = PANEL / f"{model}_finer30_panel.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        usable = (num(df[ADH]) >= 3) & (num(df[FLU]) >= 3)
        df2 = df.copy(); df2["u"] = usable.values
        best = df2.groupby(["target_lang", "alpha"])["u"].mean().max()
        frac = usable.mean()
        print(f"    {model}: {100*frac:.1f}% ({int(usable.sum())}/{len(df)}); best cell {100*best:.0f}%")
        if model == "llama":
            macros["panusablepct"] = f"{100*frac:.1f}\\%"
            macros["panusablefrac"] = f"{int(usable.sum())}/{len(df)}"
            macros["panusablebest"] = f"{100*best:.0f}\\%"

    # Aya ceiling = best Aya peak across targets
    af = PANEL / "aya_finer30_panel.parquet"
    if af.exists():
        df = pd.read_parquet(af)
        ceil = max(peak_cell(df, t)[1] for t in TL)
        macros["panayaceil"] = f"{ceil:.2f}"
        print(f"    Aya ceiling (best peak) = {ceil:.2f}")

    # bf16 robustness (Nepali L20), panel
    bf = PANEL / "llama_bf16_panel.parquet"
    if bf.exists():
        df = pd.read_parquet(bf)
        _, pk, _ = peak_cell(df, "npi")
        v2 = num(df[df.alpha == -2.0][ADH]).mean()
        macros["panbfpeak"] = f"{pk:.2f}"
        macros["panbftwo"] = f"{v2:.2f}"
        print(f"    bf16 panel Nepali: peak={pk:.2f}, @a=-2.0={v2:.2f}")

    # ensrc: Hindi(finer30) vs English(ensrc)
    print("\n" + "-" * 74 + "\nSOURCE: Hindi vs English (panel peak adherence)")
    for model in ["llama", "aya"]:
        hi_f = PANEL / f"{model}_finer30_panel.parquet"
        en_f = PANEL / f"{model}_ensrc_panel.parquet"
        if not (hi_f.exists() and en_f.exists()):
            continue
        hi, en = pd.read_parquet(hi_f), pd.read_parquet(en_f)
        print(f"  {model}:")
        for t in TL:
            _, hpk, _ = peak_cell(hi, t)
            _, epk, _ = peak_cell(en, t)
            print(f"    {TL[t]:9} Hindi={hpk:.2f}  English={epk:.2f}  d={hpk-epk:+.2f}")
            macros[f"pan{model}{t}en"] = f"{epk:.2f}"

    # icl
    icl_f = PANEL / "llama_icl_panel.parquet"
    if icl_f.exists():
        df = pd.read_parquet(icl_f)
        print("\n" + "-" * 74 + "\nICL (k=3) panel adh/flu (Llama):")
        for t in TL:
            adh = num(df[df.target_lang == t][ADH]).mean()
            fl = num(df[df.target_lang == t][FLU]).mean()
            print(f"    {TL[t]:9} adh={adh:.2f}  flu={fl:.2f}")
            macros[f"panicl{t}adh"] = f"{adh:.2f}"
            macros[f"panicl{t}flu"] = f"{fl:.2f}"

    # random vs real
    rnd_f = PANEL / "llama_random_panel.parquet"
    real_f = PANEL / "llama_finer30_panel.parquet"
    if rnd_f.exists() and real_f.exists():
        rnd = pd.read_parquet(rnd_f)
        real = pd.read_parquet(real_f)
        real = real[real.target_lang == "npi"]
        print("\n" + "-" * 74 + "\nDirection-specificity (Nepali L20):")
        rmean = num(rnd[ADH]).mean()
        for a in [-1.5, -2.0]:
            rv = num(rnd[rnd.alpha == a][ADH]).mean()
            rl = num(real[real.alpha == a][ADH]).mean()
            print(f"    a={a}: random={rv:.2f}  real={rl:.2f}  d={rl-rv:+.2f}")
        print(f"    random overall mean adh={rmean:.2f}, frac>=3={100*(num(rnd[ADH])>=3).mean():.1f}%")
        macros["panrandmean"] = f"{rmean:.2f}"
        rl2 = num(real[real.alpha == -2.0][ADH]).mean()
        macros["panrealnpitwo"] = f"{rl2:.2f}"

    # layersweep (Nepali per layer)
    ls_f = PANEL / "llama_layersweep_panel.parquet"
    if ls_f.exists():
        df = pd.read_parquet(ls_f)
        print("\n" + "-" * 74 + "\nPer-layer sweep, Nepali (panel peak adherence):")
        WORD = {8: "eight", 14: "fourteen", 20: "twenty", 26: "twentysix", 31: "thirtyone"}
        if "layer" in df.columns:
            for L in sorted(df["layer"].dropna().unique()):
                sub = df[df.layer == L]
                m = num(sub.groupby("alpha")[ADH].mean()).max()
                print(f"    L{int(L):<3} peak={m:.2f}")
                macros[f"panlsL{WORD.get(int(L), int(L))}"] = f"{m:.2f}"

    # multi-layer composite
    for name, f in [("multi_eval(norm-eq)", "llama_multi_eval_panel.parquet"),
                    ("multi_raw", "llama_multi_raw_panel.parquet")]:
        p = PANEL / f
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        print(f"\n{name} panel peak adherence:")
        for t in TL:
            _, pk, _ = peak_cell(df, t)
            print(f"    {TL[t]:9} {pk:.2f}")

    TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% AUTO-GENERATED by code/panel_headline.py -- do not edit by hand.",
             "% 3-judge panel (gemini-3.1-pro, gpt-5.5, gemini-3.5-flash) MEAN, "
             "with bootstrap CIs. Geometry numbers are judge-independent (elsewhere)."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()]
    TEX.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {TEX}  ({len(macros)} macros)")


if __name__ == "__main__":
    main()
