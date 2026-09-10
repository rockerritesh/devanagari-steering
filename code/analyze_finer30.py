"""Optimal-alpha / Pareto analysis for a *_all_l*_finer30_scored.parquet sweep.

Prints, per target language:
  - control (alpha=0) means
  - the adherence-optimal alpha and the overall-optimal alpha, with deltas vs control

Usage:
    uv run --with pandas --with pyarrow python code/analyze_finer30.py <scored.parquet>
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results/generations/llama_all_l20_finer30_scored.parquet"
df = pd.read_parquet(path)

DIMS = ["language_adherence", "fluency", "faithfulness", "coherence", "overall_quality"]
for d in DIMS:
    df[d] = pd.to_numeric(df[d], errors="coerce")

print(f"== {path} ==  rows={len(df)}  model={df['model'].unique().tolist()}")
g = df.groupby(["target_lang", "alpha"])[DIMS].mean().round(3)

for t in ["npi", "mai", "bho"]:
    sub = g.loc[t]
    ctrl = sub.loc[0.0]
    # adherence-optimal alpha
    adh_a = sub["language_adherence"].idxmax()
    ovr_a = sub["overall_quality"].idxmax()
    print(f"\n### {t} ###")
    print(sub.to_string())
    print(f"  control(0):        adh={ctrl.language_adherence:.2f} flu={ctrl.fluency:.2f} ovr={ctrl.overall_quality:.2f}")
    a = sub.loc[adh_a]
    print(f"  adherence-opt a={adh_a:>5}: adh={a.language_adherence:.2f} "
          f"(Δ{a.language_adherence-ctrl.language_adherence:+.2f}) "
          f"flu={a.fluency:.2f} (Δ{a.fluency-ctrl.fluency:+.2f}) "
          f"ovr={a.overall_quality:.2f} (Δ{a.overall_quality-ctrl.overall_quality:+.2f})")
    o = sub.loc[ovr_a]
    print(f"  overall-opt   a={ovr_a:>5}: adh={o.language_adherence:.2f} "
          f"(Δ{o.language_adherence-ctrl.language_adherence:+.2f}) "
          f"flu={o.fluency:.2f} (Δ{o.fluency-ctrl.fluency:+.2f}) "
          f"ovr={o.overall_quality:.2f} (Δ{o.overall_quality-ctrl.overall_quality:+.2f})")

print(f"\n== peak language_adherence per target ==")
for t in ["npi", "mai", "bho"]:
    sub = g.loc[t]
    print(f"  {t}: max adherence {sub['language_adherence'].max():.2f} at a={sub['language_adherence'].idxmax()}")
