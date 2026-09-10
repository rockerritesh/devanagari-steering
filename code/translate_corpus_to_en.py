"""Task 5 prep — add an English (`en`) column to the parallel corpus.

EN-as-source baseline needs English text parallel to the existing hin/mai/npi/bho.
We translate the Hindi `hin` passages → English with Gemini (English is high-resource,
single-model translation is sufficient). Async, batched-by-1 with concurrency,
checkpointed so it resumes.

  uv run --with pandas --with pyarrow --with google-genai --with python-dotenv \
      python code/translate_corpus_to_en.py \
        --in data/parallel/parallel_devanagari_250.parquet \
        --out data/parallel/parallel_devanagari_250_en.parquet
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-2.5-flash"

PROMPT = """Translate the following Hindi passage into fluent, faithful English.
Preserve meaning, names, and structure. Output ONLY the English translation, no
preamble, no notes, no quotation marks.

HINDI PASSAGE:
{text}"""


async def translate_one(client, model, text, sem, max_retries=4):
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=PROMPT.format(text=text),
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return resp.text.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)


async def run(args):
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY missing from .env")

    df = pd.read_parquet(args.in_path)
    if args.text_col not in df.columns:
        sys.exit(f"input parquet has no '{args.text_col}' column")
    ids = df["id"].astype(str).tolist()
    texts = df[args.text_col].astype(str).tolist()
    print(f"loaded {len(df)} rows from {args.in_path} (col={args.text_col} -> {args.out_col})")

    ckpt = args.out_path.with_suffix(".ckpt.json")
    done = json.loads(ckpt.read_text()) if (ckpt.exists() and not args.force) else {}
    if done:
        print(f"  resuming: {len(done)} already translated")

    client = genai.Client(api_key=key)
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(i):
        if ids[i] in done:
            return
        en = await translate_one(client, args.model, texts[i], sem)
        done[ids[i]] = en

    todo = [i for i in range(len(ids)) if ids[i] not in done]
    pbar = tqdm(total=len(todo), desc="translate", unit="psg")
    tasks = [asyncio.ensure_future(worker(i)) for i in todo]
    for fut in asyncio.as_completed(tasks):
        await fut
        ckpt.write_text(json.dumps(done, ensure_ascii=False))
        pbar.update(1)
    pbar.close()

    df[args.out_col] = df["id"].astype(str).map(done)
    n_missing = df[args.out_col].isna().sum()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"wrote {args.out_path}  shape={df.shape}  missing={n_missing}")
    print("sample EN:", (df[args.out_col].dropna().iloc[0][:160] if df[args.out_col].notna().any() else "NONE"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=PROJECT_ROOT / "data/parallel/parallel_devanagari_250.parquet")
    ap.add_argument("--out", dest="out_path", type=Path,
                    default=PROJECT_ROOT / "data/parallel/parallel_devanagari_250_en.parquet")
    ap.add_argument("--text-col", default="hin", help="Source Hindi column to translate.")
    ap.add_argument("--out-col", default="en", help="New English column name.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
