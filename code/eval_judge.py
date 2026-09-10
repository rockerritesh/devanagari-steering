"""Step 4 of devanagari-steering: LLM-as-judge harness for steered generations.

Reads a generations parquet (one row per (model, target_lang, prompt, alpha, layer)
combination plus the model output), and asks Gemini to score each output against
a 5-dimensional rubric:

  1. language_adherence (1-5): is the output in the requested target language?
  2. fluency           (1-5): grammatical, natural-sounding text?
  3. faithfulness      (1-5): does the output answer the prompt?
  4. coherence         (1-5): readable, sensible discourse?
  5. overall_quality   (1-5): overall quality

Plus a free-text `notes` field and a `language_detected` string (judge's guess).

The judge call is batched (default 8 rows per call) and async-concurrent up to
`--concurrency` parallel calls to keep wall-clock low. Each batch is a single
JSON-mode Gemini call returning a list of `JudgeScore` objects.

Input parquet schema (required columns):
    id            : str  - unique row id (eval_xxxx_<model>_<target>_<alpha>_<layer>)
    model         : str  - "llama" / "aya" / etc.
    target_lang   : str  - "hin" / "mai" / "npi" / "bho"
    prompt        : str  - the input prompt (any language)
    generation    : str  - the model's generated text to be judged
    alpha         : float (optional) - steering coefficient (NaN for unsteered)
    layer         : int   (optional) - layer steered at
    prompt_lang   : str   (optional) - language the prompt is in (defaults to target)

Output parquet schema:  input columns + score columns (5 floats) + notes + lang_detected.

Usage:
    .venv/bin/python code/eval_judge.py \\
        --in results/generations/aya_npi_l22_sweep.parquet \\
        --out results/judge/aya_npi_l22_sweep_scored.parquet
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
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.1-pro-preview"

LANG_FULL = {
    "hin": "Hindi (hin_Deva)",
    "mai": "Maithili (mai_Deva)",
    "npi": "Nepali (npi_Deva)",
    "bho": "Bhojpuri (bho_Deva)",
}


class JudgeScore(BaseModel):
    language_adherence: int = Field(ge=1, le=5)
    fluency: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    coherence: int = Field(ge=1, le=5)
    overall_quality: int = Field(ge=1, le=5)
    language_detected: str
    notes: str


class JudgeBatch(BaseModel):
    scores: list[JudgeScore]


JUDGE_PROMPT_TEMPLATE = """You are an expert linguistic evaluator for Indo-Aryan Devanagari languages: Hindi (hin), Maithili (mai), Nepali (npi), Bhojpuri (bho). Your job is to score model outputs on a 5-dimensional rubric.

The four target languages are closely related — Maithili, Nepali, and Bhojpuri are NOT just "Hindi with a different word." They have distinct grammar markers:

- Hindi (hin):       copula `है/हैं`, postpositions `का/की/के`, present `करता है`.
- Maithili (mai):    copula `अछि/छथि`, particle `थिक`, oblique `केर`, present `करैत अछि`.
- Nepali (npi):      copula `छ/हो/हुन्छ`, postposition `को/लाई`, present `गर्छ/हुन्छ`.
- Bhojpuri (bho):    copula `बा/ह`, postposition `के/में`, present `करेला/करेलें`, plurals like `बच्चन/लोगन`.

For each output below, rate on these dimensions, using INTEGERS 1-5. Be strict.

1. **language_adherence** (1-5): To what extent is the output written in the *requested* target language?
   - 5 = fully in target language; vocabulary, grammar markers, and pronouns all match.
   - 4 = mostly target language with minor Hindi-isms or one or two foreign words.
   - 3 = a recognisable mix of target language and Hindi (or another sister), neither dominant.
   - 2 = mostly Hindi (or another non-target language) with only a few target-language tokens.
   - 1 = entirely the wrong language, or unintelligible characters.

2. **fluency** (1-5): Is the output grammatical and natural-sounding (regardless of which language it ended up in)?
   - 5 = perfectly fluent native-like prose. 1 = broken / repetitive / nonsensical word salad.

3. **faithfulness** (1-5): Does the output address the prompt?
   - 5 = directly and fully answers the prompt's question/instruction.
   - 1 = ignores the prompt entirely, or talks about something unrelated.

4. **coherence** (1-5): Is the output readable, on-topic across sentences, and free of repetition loops?
   - 5 = coherent multi-sentence discourse. 1 = repetition of one phrase, garbage, or empty.

5. **overall_quality** (1-5): Holistic judgment combining the above.

Also fill:
- **language_detected**: a short string ("Hindi", "Nepali", "Maithili", "Bhojpuri", "Mixed Hindi-Nepali", "Garbage", etc).
- **notes**: a one-sentence justification (max 25 words).

Below are {n} (prompt, target_language, output) triples. Return a JSON object with key "scores", a list of exactly {n} score objects in the same order.

INPUTS:
{numbered_inputs}
"""


def format_inputs(rows):
    chunks = []
    for i, row in enumerate(rows):
        target = LANG_FULL.get(row["target_lang"], row["target_lang"])
        prompt_lang = row.get("prompt_lang") or row["target_lang"]
        chunks.append(
            f"[{i+1}]\n"
            f"REQUESTED_TARGET_LANGUAGE: {target}\n"
            f"PROMPT (in {prompt_lang}): {row['prompt']}\n"
            f"OUTPUT_TO_JUDGE: {row['generation']}"
        )
    return "\n\n---\n\n".join(chunks)


async def judge_batch(client, rows, model):
    request = JUDGE_PROMPT_TEMPLATE.format(
        n=len(rows), numbered_inputs=format_inputs(rows)
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JudgeBatch,
        temperature=0.0,  # deterministic — important for reproducible scoring
    )
    resp = await client.aio.models.generate_content(
        model=model, contents=request, config=config
    )
    parsed: JudgeBatch = resp.parsed
    if parsed is None or len(parsed.scores) != len(rows):
        got = 0 if parsed is None else len(parsed.scores)
        raise ValueError(f"expected {len(rows)} scores, got {got}")
    return parsed.scores


async def judge_with_retry(client, rows, model, max_retries=4, label=""):
    for attempt in range(max_retries):
        try:
            return await judge_batch(client, rows, model)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            print(f"  [{label}] retry {attempt + 1} after {wait}s — "
                  f"{type(e).__name__}: {e}")
            await asyncio.sleep(wait)


async def run(args):
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY missing from .env")

    df = pd.read_parquet(args.in_path)
    required = {"id", "model", "target_lang", "prompt", "generation"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"input parquet missing required columns: {missing}")

    print(f"Loaded {len(df)} rows from {args.in_path}")
    print(f"  unique models: {df['model'].nunique()}, "
          f"targets: {df['target_lang'].unique().tolist()}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_path.with_suffix(".ckpt.json")
    done_ids: set[str] = set()
    scores_by_id: dict[str, dict] = {}
    if ckpt_path.exists() and not args.force:
        ckpt = json.loads(ckpt_path.read_text())
        scores_by_id = ckpt
        done_ids = set(scores_by_id.keys())
        print(f"  resuming: {len(done_ids)} rows already scored")

    todo = df[~df["id"].isin(done_ids)].reset_index(drop=True)
    if len(todo) == 0:
        print("nothing to do — all rows already scored.")
    else:
        print(f"  scoring {len(todo)} rows in batches of {args.batch_size} "
              f"with {args.concurrency} concurrent calls ...")

    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    async def score_one_batch(batch_idx, batch_rows):
        rows_payload = batch_rows.to_dict("records")
        async with sem:
            scores = await judge_with_retry(
                client, rows_payload, args.model,
                label=f"batch{batch_idx}",
            )
        return batch_rows["id"].tolist(), [s.model_dump() for s in scores]

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    tasks = []
    for b in range(n_batches):
        chunk = todo.iloc[b * args.batch_size : (b + 1) * args.batch_size]
        tasks.append(score_one_batch(b, chunk))

    pbar = tqdm(total=n_batches, desc="judge", unit="batch")
    for fut in asyncio.as_completed(tasks):
        ids, score_dicts = await fut
        for _id, sc in zip(ids, score_dicts):
            scores_by_id[_id] = sc
        ckpt_path.write_text(json.dumps(scores_by_id, ensure_ascii=False))
        pbar.update(1)
    pbar.close()

    # Stitch scores onto the input dataframe.
    score_cols = ["language_adherence", "fluency", "faithfulness",
                  "coherence", "overall_quality",
                  "language_detected", "notes"]
    for col in score_cols:
        df[col] = df["id"].map(lambda i: scores_by_id.get(i, {}).get(col))
    df.to_parquet(args.out_path, index=False)

    print(f"\nWrote {args.out_path}  ({len(df)} rows)")
    if len(df) > 0:
        means = df[["language_adherence", "fluency", "faithfulness",
                    "coherence", "overall_quality"]].mean()
        print("\nOverall mean scores:")
        for k, v in means.items():
            print(f"  {k:>20}: {v:.2f}")
        # group by (model, target_lang, alpha) if those exist
        group_cols = [c for c in ["model", "target_lang", "alpha", "layer"]
                      if c in df.columns]
        if group_cols:
            agg = (df.groupby(group_cols)[
                ["language_adherence", "fluency", "faithfulness",
                 "coherence", "overall_quality"]
            ].mean().round(2))
            print(f"\nMean by {group_cols}:")
            print(agg.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True,
                    help="Input parquet of generations to judge.")
    ap.add_argument("--out", dest="out_path", type=Path, required=True,
                    help="Output parquet path with score columns appended.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Gemini model to use as judge.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Number of (prompt, output) pairs scored per API call.")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Max parallel Gemini calls.")
    ap.add_argument("--force", action="store_true",
                    help="Ignore checkpoint, re-score everything.")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
