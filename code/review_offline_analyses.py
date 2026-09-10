"""Offline review-response analyses (no GPU, existing data only).

Answers three reviewer attacks using data already on disk:
  (4) Bootstrap CIs on the n=30 headline adherence/fluency cells -> is the
      3.23 vs 1.57 gap statistically real, not n=30 noise?
  (5) Independent automatic language-ID (GlotLID/fastText) on the generations
      -> the paper's `language_detected` is the SAME Gemini judge; an
      independent non-LLM metric is the strongest rebuttal to "the judge is
      hallucinating Nepali".
  (6) Word-overlap of steered output vs its own Hindi control -> does the
      content vocabulary actually leave Hindi, or only get re-flavoured?

  uv run --with pandas --with pyarrow --with numpy \
         --with fasttext-wheel --with huggingface_hub \
         python code/review_offline_analyses.py

Writes paper/latell/review_stats.tex (CI macros) and prints a report.
GlotLID needs a one-time model download (~1.2 GB) from HuggingFace; if
offline, parts (5)/(6-langid) are skipped with a notice.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
OUT_TEX = ROOT / "paper" / "latell" / "review_stats.tex"
RNG = np.random.default_rng(20260609)
B = 10000  # bootstrap resamples

# GlotLID label for each target/source code we use.
GLOT = {"npi": "npi_Deva", "mai": "mai_Deva", "bho": "bho_Deva",
        "hin": "hin_Deva", "eng": "eng_Latn"}


# ---------------------------------------------------------------- bootstrap
def boot_ci(x, b=B, lo=2.5, hi=97.5):
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(),
                   dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(b, len(x)))
    means = x[idx].mean(axis=1)
    return x.mean(), np.percentile(means, lo), np.percentile(means, hi)


def cell_report(df, model, label, cells):
    """cells: list of (target, alpha, tag). Prints CI for adherence+fluency."""
    rows = []
    for tgt, a, tag in cells:
        sub = df[(df.target_lang == tgt) & (np.isclose(df.alpha, a))]
        if len(sub) == 0:
            continue
        am, alo, ahi = boot_ci(sub.language_adherence)
        fm, flo, fhi = boot_ci(sub.fluency)
        rows.append((tgt, a, tag, len(sub), am, alo, ahi, fm, flo, fhi))
        print(f"  {label:6s} {tgt} a={a:+.1f} {tag:10s} n={len(sub):2d}  "
              f"adh {am:.2f} [{alo:.2f},{ahi:.2f}]  "
              f"flu {fm:.2f} [{flo:.2f},{fhi:.2f}]")
    return rows


def diff_ci(x, y, b=B):
    """Bootstrap CI for mean(x)-mean(y), independent resampling."""
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), float)
    y = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce").dropna(), float)
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan
    dx = x[RNG.integers(0, len(x), (b, len(x)))].mean(1)
    dy = y[RNG.integers(0, len(y), (b, len(y)))].mean(1)
    d = dx - dy
    return (x.mean() - y.mean()), np.percentile(d, 2.5), np.percentile(d, 97.5)


# ---------------------------------------------------------------- glotlid
_FT = None


def load_glotlid():
    global _FT
    if _FT is not None:
        return _FT
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        _FT = fasttext.load_model(p)
        return _FT
    except Exception as e:  # noqa: BLE001
        print(f"  [glotlid unavailable: {e}]")
        return None


def glot_pred(model, text):
    t = re.sub(r"\s+", " ", str(text)).strip()
    if not t:
        return None
    lab, _ = model.predict(t, k=1)
    return lab[0].replace("__label__", "")


def langid_report(df, model_name, label, target_alphas, show=None):
    """Run GlotLID over every (target,alpha) cell. `show` (set of alphas) prints
    a subset; correlation uses ALL cells. Returns rows for correlation."""
    ft = load_glotlid()
    if ft is None:
        return []
    rows = []
    for tgt in target_alphas:
        want = GLOT[tgt]
        alphas = sorted(df[df.target_lang == tgt].alpha.unique())
        for a in alphas:
            sub = df[(df.target_lang == tgt) & (np.isclose(df.alpha, a))]
            if len(sub) == 0:
                continue
            preds = [glot_pred(ft, g) for g in sub.generation]
            preds = [p for p in preds if p]
            frac_tgt = np.mean([p == want for p in preds]) if preds else np.nan
            frac_hin = np.mean([p == "hin_Deva" for p in preds]) if preds else np.nan
            adh = pd.to_numeric(sub.language_adherence, errors="coerce").mean()
            rows.append((label, tgt, a, len(preds), frac_tgt, frac_hin, adh))
            if show is None or a in show:
                print(f"  {label:6s} {tgt} a={a:+.1f}  GlotLID->target {frac_tgt:5.0%}"
                      f"  ->Hindi {frac_hin:5.0%}   (judge adh {adh:.2f})")
    return rows


# ---------------------------------------------------------------- ngram overlap
_WORD = re.compile(r"[ऀ-ॿ]+")  # Devanagari word run


def words(s):
    return _WORD.findall(str(s))


def overlap_vs_control(df, tgt, alphas):
    """Word-level Jaccard of each steered gen vs its own a=0 Hindi control,
    matched by eval_id. High overlap => vocabulary still Hindi."""
    ctrl = df[(df.target_lang == tgt) & (np.isclose(df.alpha, 0.0))]
    ctrl = ctrl.set_index("eval_id").generation.to_dict()
    out = []
    for a in alphas:
        sub = df[(df.target_lang == tgt) & (np.isclose(df.alpha, a))]
        js = []
        for _, r in sub.iterrows():
            c = ctrl.get(r.eval_id)
            if c is None:
                continue
            ws, wc = set(words(r.generation)), set(words(c))
            if not ws and not wc:
                continue
            j = len(ws & wc) / max(1, len(ws | wc))
            js.append(j)
        m = float(np.mean(js)) if js else np.nan
        out.append((tgt, a, len(js), m))
        print(f"  {tgt} a={a:+.1f}  mean Jaccard vs Hindi control = {m:.3f}  (n={len(js)})")
    return out


# ---------------------------------------------------------------- main
def main():
    llama = pd.read_parquet(GEN / "llama_all_l20_finer30_scored.parquet")
    aya = pd.read_parquet(GEN / "aya_all_l22_finer30_scored.parquet")
    rand = pd.read_parquet(GEN / "llama_npi_l20_random_scored.parquet")
    ensrc = pd.read_parquet(GEN / "llama_all_l20_ensrc_scored.parquet")

    print("\n=== (4) BOOTSTRAP CIs (10k resamples, 95% percentile) ===")
    cell_report(llama, "llama", "Llama", [
        ("npi", 0.0, "control"), ("npi", -1.0, "readable"), ("npi", -2.0, "peak"),
        ("mai", -2.5, "peak"), ("bho", -2.5, "peak")])
    cell_report(aya, "aya", "Aya", [
        ("npi", 0.0, "control"), ("npi", -0.4, "peak"),
        ("mai", -0.5, "peak"), ("bho", -1.0, "peak")])

    print("\n  -- headline gap: Llama-Nepali peak adh  -  Aya-Nepali peak adh --")
    lx = llama[(llama.target_lang == "npi") & (np.isclose(llama.alpha, -2.0))].language_adherence
    ax = aya[(aya.target_lang == "npi") & (np.isclose(aya.alpha, -0.4))].language_adherence
    d, dlo, dhi = diff_ci(lx, ax)
    print(f"    diff = {d:+.2f}  95% CI [{dlo:+.2f}, {dhi:+.2f}]  "
          f"({'EXCLUDES 0' if dlo > 0 or dhi < 0 else 'includes 0'})")

    print("\n  -- random-direction baseline gap (Llama Nepali) --")
    # peak steering adherence vs matched-magnitude random direction
    rb = rand.language_adherence
    rm, rlo, rhi = boot_ci(rb)
    print(f"    random-dir adherence: {rm:.2f} [{rlo:.2f},{rhi:.2f}] (n={len(rb.dropna())})")

    print("\n=== (5) INDEPENDENT LANGUAGE-ID (GlotLID/fastText, non-LLM) ===")
    glot = langid_report(llama, "llama", "Llama",
                         ["npi", "mai", "bho"], show={0.0, -1.0, -2.0, -2.5})
    # judge adherence vs independent GlotLID target-fraction, all cells
    glot_adh = np.array([r[6] for r in glot if not np.isnan(r[4])])
    glot_frac = np.array([r[4] for r in glot if not np.isnan(r[4])])
    if len(glot_adh) > 2:
        r_pear = float(np.corrcoef(glot_adh, glot_frac)[0, 1])
        print(f"\n  judge-adherence vs GlotLID target-fraction (all {len(glot_adh)} "
              f"Llama cells): Pearson r = {r_pear:.2f}")
    else:
        r_pear = np.nan

    print("\n=== (6) WORD-OVERLAP vs HINDI CONTROL (Llama Nepali) ===")
    ov = overlap_vs_control(llama, "npi", [-0.5, -1.0, -1.5, -2.0])

    # ---- emit tex macros for the headline CIs
    def m(name, lo, hi, mean):
        return (f"\\newcommand{{\\{name}}}{{{mean:.2f}~[{lo:.2f},\\,{hi:.2f}]}}")
    npi_peak = llama[(llama.target_lang == "npi") & (np.isclose(llama.alpha, -2.0))]
    am, alo, ahi = boot_ci(npi_peak.language_adherence)
    fm, flo, fhi = boot_ci(npi_peak.fluency)
    # GlotLID numbers for the validation paragraph
    def gfrac(tgt, a):
        for lab, t, aa, n, ft_, fh_, adh in glot:
            if t == tgt and np.isclose(aa, a):
                return ft_, fh_
        return np.nan, np.nan
    npi_pk_t, _ = gfrac("npi", -2.0)
    npi_ct_t, npi_ct_h = gfrac("npi", 0.0)
    ov_lo = next((mn for (t, a, n, mn) in ov if np.isclose(a, -0.5)), np.nan)
    ov_hi = next((mn for (t, a, n, mn) in ov if np.isclose(a, -2.0)), np.nan)
    tex = [
        "% AUTO-GENERATED by code/review_offline_analyses.py",
        m("npipeakadhCI", alo, ahi, am),
        m("npipeakfluCI", flo, fhi, fm),
        f"\\newcommand{{\\headlinediffCI}}{{{d:+.2f}~[{dlo:+.2f},\\,{dhi:+.2f}]}}",
        f"\\newcommand{{\\glotnpipeak}}{{{npi_pk_t*100:.0f}\\%}}",
        f"\\newcommand{{\\glotnpictrl}}{{{npi_ct_t*100:.0f}\\%}}",
        f"\\newcommand{{\\glotcorr}}{{{r_pear:.2f}}}",
        f"\\newcommand{{\\jacctrllo}}{{{ov_lo:.2f}}}",
        f"\\newcommand{{\\jacctrlhi}}{{{ov_hi:.2f}}}",
    ]
    OUT_TEX.write_text("\n".join(tex) + "\n")
    print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
