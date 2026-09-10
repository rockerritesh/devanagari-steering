"""Step 0a of devanagari-steering: build the 4-way parallel translation corpus.

Sample 1000 long Hindi passages from XLSum, translate each to Maithili (mai_Deva),
Nepali (npi_Deva), and Bhojpuri (bho_Deva) via Gemini 3.1 Pro Preview using batched
calls (one API call carries many passages, since the model has a 1M token context).

The three target languages are translated **concurrently per batch round** via
asyncio.gather, cutting wall-clock time by ~3x vs sequential per-language passes.

Output: data/parallel/parallel_devanagari_1k.parquet with columns
{id, url, title, hin, mai, npi, bho, n_tokens}.

Usage:
    python code/build_parallel_corpus.py                       # full 1000-passage run
    python code/build_parallel_corpus.py --n 50 --batch-size 10  # quick smoke test

Requirements: GEMINI_API_KEY in .env. Llama tokenizer requires `huggingface-cli login`
(or pass --tokenizer to use a public alternative).

Resumes from per-language JSON checkpoints in data/parallel/.checkpoints/ on rerun;
each language is independent, so a partial failure for one doesn't lose progress on
the others.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "parallel" / "parallel_devanagari_1k.parquet"
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

LENGTH_TOK_MIN = 256
LENGTH_TOK_MAX = 1024
TARGET_LANGS: dict[str, str] = {
    "mai": "Maithili",
    "npi": "Nepali",
    "bho": "Bhojpuri",
}


class BatchTranslations(BaseModel):
    translations: list[str]


PROMPT_TEMPLATE = """You are a professional translator specialising in Indo-Aryan languages.

Translate each of the following Hindi (hin_Deva) passages into {target_full} ({target_code}_Deva). Produce one translation per input, preserving meaning, register, and paragraph structure. Use the Devanagari script. Do not summarise. Do not add commentary or numbering. Do not include the original Hindi.

Return a JSON object with a single key "translations" whose value is a list of strings. The i-th string must be the {target_full} translation of the i-th Hindi passage below.

INPUT_PASSAGES:
{numbered_passages}
"""


def sample_hindi_passages(n_target, length_min, length_max, tokenizer, seed):
    """Sample long Hindi passages from XLSum, filtered by token length."""
    ds = load_dataset(
        "csebuetnlp/xlsum", "hindi", split="train", trust_remote_code=True
    ).shuffle(seed=seed)
    out: list[dict] = []
    for ex in ds:
        text = ex["text"]
        if not text or not text.strip():
            continue
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if length_min <= n_tokens <= length_max:
            out.append(
                {
                    "id": ex["id"],
                    "url": ex["url"],
                    "title": ex["title"],
                    "hin": text,
                    "n_tokens": n_tokens,
                }
            )
        if len(out) >= n_target:
            break
    return out


def format_batch(passages):
    return "\n\n".join(f"[{i + 1}]\n{p['hin']}" for i, p in enumerate(passages))


async def translate_batch(client, passages, target_code, target_full, model):
    prompt = PROMPT_TEMPLATE.format(
        target_full=target_full,
        target_code=target_code,
        numbered_passages=format_batch(passages),
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BatchTranslations,
        temperature=0.2,
    )
    resp = await client.aio.models.generate_content(
        model=model, contents=prompt, config=config
    )
    parsed: BatchTranslations = resp.parsed
    if parsed is None or len(parsed.translations) != len(passages):
        got = 0 if parsed is None else len(parsed.translations)
        raise ValueError(f"expected {len(passages)} translations, got {got}")
    return parsed.translations


async def translate_with_retry(
    client, passages, target_code, target_full, model, max_retries=4
):
    for attempt in range(max_retries):
        try:
            return await translate_batch(client, passages, target_code, target_full, model)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            print(f"  [{target_code}] retry {attempt + 1} after {wait}s — {type(e).__name__}: {e}")
            await asyncio.sleep(wait)


def save_checkpoint(ckpt_path, translations):
    ckpt_path.write_text(json.dumps({"translations": translations}, ensure_ascii=False))


async def run(args):
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY missing from .env")

    print(f"Loading tokenizer {args.tokenizer} for length filtering...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Sampling {args.n} Hindi passages from XLSum (length {args.length_min}–{args.length_max} tokens)...")
    passages = sample_hindi_passages(
        n_target=args.n,
        length_min=args.length_min,
        length_max=args.length_max,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    if len(passages) < args.n:
        print(f"  WARNING: only {len(passages)}/{args.n} passages matched the length filter")
    mean_tok = sum(p["n_tokens"] for p in passages) / max(1, len(passages))
    print(f"  got {len(passages)} passages, mean length {mean_tok:.0f} tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out.parent / ".checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    translations: dict[str, list[str]] = {}
    ckpt_paths: dict[str, Path] = {}
    for code in TARGET_LANGS:
        ckpt_paths[code] = ckpt_dir / f"{code}.json"
        if ckpt_paths[code].exists():
            translations[code] = json.loads(ckpt_paths[code].read_text())["translations"]
            if translations[code]:
                print(f"  resuming {code}: {len(translations[code])}/{len(passages)} already done")
        else:
            translations[code] = []

    client = genai.Client(api_key=api_key)
    print(f"Translating with {args.model}, batch size {args.batch_size}, 3 langs concurrent per round...")

    n_batches = (len(passages) + args.batch_size - 1) // args.batch_size
    pbar = tqdm(range(n_batches), desc="rounds", unit="round")
    for batch_idx in pbar:
        i = batch_idx * args.batch_size
        batch = passages[i : i + args.batch_size]
        if not batch:
            break

        tasks = []
        codes_in_flight = []
        for code, full in TARGET_LANGS.items():
            if i < len(translations[code]):
                continue
            tasks.append(
                translate_with_retry(client, batch, code, full, args.model)
            )
            codes_in_flight.append(code)

        if not tasks:
            pbar.set_postfix_str("all langs done for this round")
            continue

        results = await asyncio.gather(*tasks, return_exceptions=True)

        first_error = None
        succeeded = []
        for code, result in zip(codes_in_flight, results):
            if isinstance(result, Exception):
                if first_error is None:
                    first_error = (code, result)
                continue
            translations[code].extend(result)
            save_checkpoint(ckpt_paths[code], translations[code])
            succeeded.append(code)

        pbar.set_postfix_str(f"round {batch_idx + 1}/{n_batches} done={','.join(succeeded) or 'none'}")

        if first_error is not None:
            code, err = first_error
            raise RuntimeError(f"[{code}] failed at batch {batch_idx}: {type(err).__name__}: {err}") from err

    for code in TARGET_LANGS:
        if len(translations[code]) != len(passages):
            print(f"  WARNING: {code} has {len(translations[code])}/{len(passages)} translations")
        for p, t in zip(passages, translations[code]):
            p[code] = t

    print(f"\nWriting parquet to {args.out}...")
    df = pd.DataFrame(passages)
    df.to_parquet(args.out, index=False)
    print(f"done — {len(df)} rows, {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="passages to sample")
    parser.add_argument("--batch-size", type=int, default=25, help="passages per Gemini call")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer id used for length filter",
    )
    parser.add_argument(
        "--length-min", type=int, default=LENGTH_TOK_MIN, help="min tokens per passage"
    )
    parser.add_argument(
        "--length-max", type=int, default=LENGTH_TOK_MAX, help="max tokens per passage"
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
