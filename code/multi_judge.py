"""Step 4 (journal revision): PANEL / JURY LLM-as-judge harness.

Replaces the single-judge eval_judge.py with a *panel of judges from disjoint
model families* (PoLL; Verga et al., COLM 2024). The generators under study are
Llama-3.1 (Meta) and Aya-23 (Cohere); the panel deliberately uses neither family,
so self-preference / self-enhancement bias is structurally avoided.

Default panel (one seat per API we hold; verify ids with --list-models):
    gemini_pro    google  gemini-3.1-pro-preview   (flagship reasoning)
    gpt           openai  gpt-5.5                  (strongest available on this key)
    gemini_flash  google  gemini-3.5-flash         (different Google tier)

NB this panel is family-imbalanced (2 Google : 1 OpenAI) because we do not hold an
Anthropic key; state this as a limitation. Adding a claude-* seat (needs
ANTHROPIC_API_KEY) would make it a true 3-family jury.

Each judge scores every generation independently on the SAME 5-dimension rubric as
eval_judge.py (reused verbatim: language_adherence, fluency, faithfulness,
coherence, overall_quality, 1-5). Output is a WIDE parquet: for each judge a set of
`<judge>__<dim>` columns, plus panel aggregates `panel_mean__<dim>`,
`panel_median__<dim>`, and `panel_sd__<dim>` (across-judge dispersion = a
per-item disagreement / low-confidence signal). Inter-judge agreement statistics
(Krippendorff alpha, ICC, weighted kappa, Spearman) are computed separately by
code/judge_agreement.py from this wide parquet.

Per-judge checkpointing: each judge's scores are cached to
<out>.<judge>.ckpt.json so a crash / rate-limit resumes without re-scoring.

Usage:
    # smoke test all three judges on 4 rows
    .venv/bin/python code/multi_judge.py \
        --in results/generations/llama_all_l20_finer30_scored.parquet \
        --out results/judge_panel/llama_finer30_panel.parquet --limit 4

    # full run
    .venv/bin/python code/multi_judge.py \
        --in results/generations/llama_all_l20_finer30_scored.parquet \
        --out results/judge_panel/llama_finer30_panel.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Reuse the exact rubric + input formatting + schema from the single-judge harness
# so the panel scores the identical prompt the paper already documents (App A).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_judge import (  # noqa: E402
    JUDGE_PROMPT_TEMPLATE,
    JudgeBatch,
    format_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIMS = ["language_adherence", "fluency", "faithfulness", "coherence", "overall_quality"]
AUX = ["language_detected", "notes"]

# (judge_name, provider, model_id). judge_name is the column prefix.
DEFAULT_PANEL = [
    ("gemini_pro", "gemini", "gemini-3.1-pro-preview"),
    ("gpt", "openai", "gpt-5.5"),
    ("gemini_flash", "gemini", "gemini-3.5-flash"),
]


# --------------------------------------------------------------------------- #
# Provider-specific batch scoring. Each returns a list[dict] of len(rows),
# one dict per row with the 5 int dims + language_detected + notes.
# --------------------------------------------------------------------------- #
async def gemini_score_batch(client, rows, model):
    from google.genai import types

    request = JUDGE_PROMPT_TEMPLATE.format(n=len(rows), numbered_inputs=format_inputs(rows))
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JudgeBatch,
        temperature=0.0,
    )
    resp = await client.aio.models.generate_content(model=model, contents=request, config=config)
    parsed: JudgeBatch = resp.parsed
    if parsed is None or len(parsed.scores) != len(rows):
        got = 0 if parsed is None else len(parsed.scores)
        raise ValueError(f"gemini {model}: expected {len(rows)} scores, got {got}")
    return [s.model_dump() for s in parsed.scores]


async def openai_score_batch(client, rows, model):
    request = JUDGE_PROMPT_TEMPLATE.format(n=len(rows), numbered_inputs=format_inputs(rows))
    # GPT-5.x are reasoning models: use the Responses API structured parse; omit
    # temperature (reasoning models reject it) and set a moderate reasoning effort.
    kwargs = dict(
        model=model,
        input=[{"role": "user", "content": request}],
        text_format=JudgeBatch,
    )
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["reasoning"] = {"effort": "medium"}
    else:
        kwargs["temperature"] = 0.0
    resp = await client.responses.parse(**kwargs)
    parsed = resp.output_parsed
    if parsed is None or len(parsed.scores) != len(rows):
        got = 0 if parsed is None else len(parsed.scores)
        raise ValueError(f"openai {model}: expected {len(rows)} scores, got {got}")
    return [s.model_dump() for s in parsed.scores]


SCORERS = {"gemini": gemini_score_batch, "openai": openai_score_batch}


def make_client(provider):
    if provider == "gemini":
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY missing from .env")
        return genai.Client(api_key=key)
    if provider == "openai":
        from openai import AsyncOpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY missing from .env")
        return AsyncOpenAI(api_key=key)
    raise ValueError(f"unknown provider {provider}")


async def score_with_retry(scorer, client, rows, model, max_retries=4, label=""):
    for attempt in range(max_retries):
        try:
            return await scorer(client, rows, model)
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [{label}] retry {attempt + 1} after {wait}s — {type(e).__name__}: {e}")
            await asyncio.sleep(wait)


async def run_one_judge(df, judge_name, provider, model, batch_size, concurrency, out_path):
    """Score every row of df with one judge; checkpoint; return {id: score_dict}."""
    ckpt = out_path.with_suffix(f".{judge_name}.ckpt.json")
    scores_by_id: dict[str, dict] = {}
    if ckpt.exists():
        scores_by_id = json.loads(ckpt.read_text())
        print(f"[{judge_name}] resuming: {len(scores_by_id)} rows cached")

    todo = df[~df["id"].isin(scores_by_id.keys())].reset_index(drop=True)
    if len(todo) == 0:
        print(f"[{judge_name}] all rows cached; skipping API calls.")
        return scores_by_id

    client = make_client(provider)
    scorer = SCORERS[provider]
    sem = asyncio.Semaphore(concurrency)
    n_batches = (len(todo) + batch_size - 1) // batch_size

    async def one(b):
        chunk = todo.iloc[b * batch_size:(b + 1) * batch_size]
        async with sem:
            scores = await score_with_retry(
                scorer, client, chunk.to_dict("records"), model, label=f"{judge_name}:b{b}")
        return chunk["id"].tolist(), scores

    print(f"[{judge_name}] scoring {len(todo)} rows via {provider}:{model} "
          f"({n_batches} batches, {concurrency} concurrent) ...")
    pbar = tqdm(total=n_batches, desc=judge_name, unit="batch")
    for fut in asyncio.as_completed([one(b) for b in range(n_batches)]):
        ids, scores = await fut
        for _id, sc in zip(ids, scores):
            scores_by_id[_id] = sc
        ckpt.write_text(json.dumps(scores_by_id, ensure_ascii=False))
        pbar.update(1)
    pbar.close()
    return scores_by_id


async def run(args):
    load_dotenv(PROJECT_ROOT / ".env")
    df = pd.read_parquet(args.in_path)
    required = {"id", "model", "target_lang", "prompt", "generation"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"input parquet missing required columns: {missing}")
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    panel = DEFAULT_PANEL
    if args.judges:
        want = set(args.judges.split(","))
        panel = [p for p in DEFAULT_PANEL if p[0] in want]
    print(f"Loaded {len(df)} rows; panel = {[p[0] + ':' + p[2] for p in panel]}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_scores: dict[str, dict] = {}
    for judge_name, provider, model in panel:
        panel_scores[judge_name] = await run_one_judge(
            df, judge_name, provider, model,
            args.batch_size, args.concurrency, args.out_path)

    # ---- aggregate (inline, no placeholder) ----
    import numpy as np
    judge_names = [p[0] for p in panel]
    for judge_name in judge_names:
        sbi = panel_scores[judge_name]
        for col in DIMS + AUX:
            df[f"{judge_name}__{col}"] = df["id"].map(
                lambda i, s=sbi, c=col: s.get(i, {}).get(c))
    for dim in DIMS:
        stack = np.vstack([
            pd.to_numeric(df[f"{jn}__{dim}"], errors="coerce").values for jn in judge_names
        ]).astype(float)
        df[f"panel_mean__{dim}"] = np.nanmean(stack, axis=0)
        df[f"panel_median__{dim}"] = np.nanmedian(stack, axis=0)
        df[f"panel_sd__{dim}"] = np.nanstd(stack, axis=0, ddof=0)

    df.to_parquet(args.out_path, index=False)
    print(f"\nWrote {args.out_path}  ({len(df)} rows, {len(judge_names)} judges)")

    # quick console summary of the headline dims
    for dim in ["language_adherence", "fluency"]:
        cols = [f"{jn}__{dim}" for jn in judge_names] + [f"panel_mean__{dim}"]
        means = {c.split('__')[0]: round(pd.to_numeric(df[c], errors='coerce').mean(), 2)
                 for c in cols}
        print(f"  {dim:20} " + "  ".join(f"{k}={v}" for k, v in means.items()))


def list_models():
    """Print judge-relevant model ids available on each key (verification helper)."""
    load_dotenv(PROJECT_ROOT / ".env")
    from google import genai
    gc = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print("GEMINI:", sorted({m.name.split('/')[-1] for m in gc.models.list()
                             if any(k in m.name for k in ("pro", "flash"))}))
    from openai import OpenAI
    oc = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print("OPENAI:", sorted({m.id for m in oc.models.list()
                             if m.id.startswith(("gpt-5", "o3", "o4"))}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path)
    ap.add_argument("--out", dest="out_path", type=Path)
    ap.add_argument("--judges", default=None,
                    help="comma list to subset panel, e.g. 'gpt' or 'gemini_pro,gpt'")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="score only first N rows (smoke test)")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()
    if args.list_models:
        list_models()
        return
    if not args.in_path or not args.out_path:
        ap.error("--in and --out are required unless --list-models")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
