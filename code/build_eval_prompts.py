"""Step 0b of devanagari-steering: 4-way parallel evaluation set.

Generate N Hindi instruction prompts (default 200), spread across diverse domains,
then translate each prompt + a reference response into Maithili (mai), Nepali (npi),
and Bhojpuri (bho). The result is a parquet with one row per prompt and 8 text
columns: {hin, mai, npi, bho} × {prompt, response}.

This set is the held-out eval corpus consumed by `eval_judge.py`. It is *not* the
same as the parallel translation corpus used to build steering vectors — the eval
prompts are short instructions that elicit ~3-6 sentence responses, whereas the
training corpus is longer narrative passages.

Usage:
    .venv/bin/python code/build_eval_prompts.py                       # 200 prompts
    .venv/bin/python code/build_eval_prompts.py --n 8 --batch-size 4  # smoke test

Requirements: GEMINI_API_KEY in .env.
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
from pydantic import BaseModel
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "parallel" / "eval_prompts_devanagari_200.parquet"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

TARGET_LANGS: dict[str, str] = {
    "mai": "Maithili",
    "npi": "Nepali",
    "bho": "Bhojpuri",
}

# Four diverse domains, each contributing 1/4 of the prompts. The mix gives the
# judge harness varied content (cultural, factual, creative, instructional) rather
# than a single register, which keeps the language-adherence signal honest.
DOMAINS: dict[str, str] = {
    "culture": (
        "Cultural and regional knowledge — festivals, food, traditions, music, "
        "clothing, customs, religious practices, regional history. Prompts can be "
        "either questions ('explain X') or instructions ('describe X')."
    ),
    "knowledge": (
        "General factual knowledge — geography, history, science, literature, "
        "biography, technology. Prompts should be answerable in 3-6 sentences "
        "with neutral, non-controversial information."
    ),
    "creative": (
        "Short creative writing — story prompts, descriptive passages, imaginative "
        "scenarios, character sketches. Each prompt names 1-3 concrete elements "
        "(e.g. nouns or themes) the response should weave in."
    ),
    "advice": (
        "Practical instructions / how-to / advice — recipes, simple craft or "
        "everyday-task instructions, life advice, study tips. The response should "
        "be a short, concrete, action-oriented answer."
    ),
}


class HindiPrompts(BaseModel):
    prompts: list[str]


class HindiResponses(BaseModel):
    responses: list[str]


class TranslatedPair(BaseModel):
    prompt: str
    response: str


class TranslatedBatch(BaseModel):
    pairs: list[TranslatedPair]


# ---------- Step 1 — generate Hindi prompts per domain --------------------- #

PROMPT_GEN_TEMPLATE = """You are designing a Hindi (hin_Deva) evaluation set for a multilingual language model.

Produce {n} **distinct** instruction-style prompts in standard Hindi (Devanagari script) for the following domain:

DOMAIN: {domain_name}
DOMAIN DESCRIPTION: {domain_desc}

Constraints:
- Each prompt must be a single self-contained instruction or question, 8-30 Hindi words.
- Prompts should be answerable in 3-6 sentences (~50-120 words).
- Prompts must be culturally and topically diverse within the domain — avoid repetition. Aim for variety in subject matter, sub-genres, and phrasing.
- Use natural conversational Hindi — the kind a layperson would type into a chatbot.
- Do NOT add numbering, bullet points, quote marks, or commentary. Each prompt is just the instruction text.
- Do NOT mention any specific language name (e.g. don't say "in Hindi" or "in English").

Return a JSON object with a single key "prompts" whose value is a list of exactly {n} strings.
"""


async def generate_prompts_for_domain(client, domain_name, domain_desc, n, model):
    prompt = PROMPT_GEN_TEMPLATE.format(n=n, domain_name=domain_name, domain_desc=domain_desc)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=HindiPrompts,
        temperature=0.95,  # high — diversity matters more than precision here
    )
    resp = await client.aio.models.generate_content(
        model=model, contents=prompt, config=config
    )
    parsed: HindiPrompts = resp.parsed
    if parsed is None or not parsed.prompts:
        raise ValueError(f"empty prompts for domain {domain_name}")
    return parsed.prompts[:n]


async def generate_prompts_with_retry(client, domain_name, domain_desc, n, model,
                                      max_retries=8):
    for attempt in range(max_retries):
        try:
            out = await generate_prompts_for_domain(client, domain_name, domain_desc, n, model)
            if len(out) < n:
                raise ValueError(f"got only {len(out)}/{n} prompts for {domain_name}")
            return out
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            print(f"  [prompts:{domain_name}] retry {attempt + 1} after {wait}s — {type(e).__name__}: {e}")
            await asyncio.sleep(wait)


# ---------- Step 2 — generate Hindi responses for each prompt ------------- #

RESPONSE_GEN_TEMPLATE = """You are answering Hindi instruction prompts. Below are {n} prompts. Produce a high-quality response **in Hindi (hin_Deva)** for each.

Constraints:
- Each response is 3-6 sentences (~50-120 words), in clear, fluent Hindi.
- Stay on-topic. Be informative but concise.
- Do NOT add numbering, headers, or commentary. Just the response text.
- The i-th response must answer the i-th prompt.

PROMPTS:
{numbered_prompts}

Return a JSON object with a single key "responses" whose value is a list of exactly {n} strings.
"""


def format_numbered(items, prefix=""):
    return "\n\n".join(f"[{i + 1}] {prefix}{x}" for i, x in enumerate(items))


async def generate_hin_responses(client, prompts_batch, model):
    request = RESPONSE_GEN_TEMPLATE.format(
        n=len(prompts_batch),
        numbered_prompts=format_numbered(prompts_batch),
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=HindiResponses,
        temperature=0.4,
    )
    resp = await client.aio.models.generate_content(
        model=model, contents=request, config=config
    )
    parsed: HindiResponses = resp.parsed
    if parsed is None or len(parsed.responses) != len(prompts_batch):
        got = 0 if parsed is None else len(parsed.responses)
        raise ValueError(f"expected {len(prompts_batch)} responses, got {got}")
    return parsed.responses


async def with_retry(coro_fn, *args, label, max_retries=8):
    for attempt in range(max_retries):
        try:
            return await coro_fn(*args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            print(f"  [{label}] retry {attempt + 1} after {wait}s — {type(e).__name__}: {e}")
            await asyncio.sleep(wait)


# ---------- Step 3 — translate (prompt, response) pairs to mai/npi/bho ----- #

TRANSLATE_TEMPLATE = """You are a professional translator specialising in Indo-Aryan languages.

Translate each of the following {n} Hindi (hin_Deva) prompt+response pairs into {target_full} ({target_code}_Deva). Preserve meaning, register, and the prompt-vs-response structure exactly. Use the Devanagari script. Do not summarise.

Return a JSON object with a single key "pairs" whose value is a list of exactly {n} objects, each with two string keys "prompt" and "response", giving the {target_full} translation of the i-th input pair.

INPUT_PAIRS (Hindi):
{numbered_pairs}
"""


def format_pairs(pairs):
    out_chunks = []
    for i, (p, r) in enumerate(pairs):
        out_chunks.append(f"[{i + 1}]\nPROMPT: {p}\nRESPONSE: {r}")
    return "\n\n".join(out_chunks)


async def translate_pairs_batch(client, pairs, target_code, target_full, model):
    request = TRANSLATE_TEMPLATE.format(
        n=len(pairs),
        target_full=target_full,
        target_code=target_code,
        numbered_pairs=format_pairs(pairs),
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TranslatedBatch,
        temperature=0.2,
    )
    resp = await client.aio.models.generate_content(
        model=model, contents=request, config=config
    )
    parsed: TranslatedBatch = resp.parsed
    if parsed is None or len(parsed.pairs) != len(pairs):
        got = 0 if parsed is None else len(parsed.pairs)
        raise ValueError(f"[{target_code}] expected {len(pairs)} translations, got {got}")
    return [(tp.prompt, tp.response) for tp in parsed.pairs]


# ---------- Orchestration ------------------------------------------------- #

async def run(args):
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY missing from .env")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out.parent / ".checkpoints_eval"
    ckpt_dir.mkdir(exist_ok=True)

    client = genai.Client(api_key=api_key)

    # Step 1: per-domain Hindi prompt generation, concurrent across domains.
    n_per_domain = args.n // len(DOMAINS)
    if n_per_domain * len(DOMAINS) != args.n:
        print(f"  note: rounding to {n_per_domain * len(DOMAINS)} prompts "
              f"(n_per_domain={n_per_domain} × {len(DOMAINS)} domains)")

    prompts_ckpt = ckpt_dir / "hin_prompts.json"
    if prompts_ckpt.exists():
        domain_prompts = json.loads(prompts_ckpt.read_text())
        print(f"  resuming Hindi prompts from checkpoint "
              f"({sum(len(v) for v in domain_prompts.values())} prompts)")
    else:
        print(f"\n[Step 1/3] generating {n_per_domain} Hindi prompts per "
              f"domain × {len(DOMAINS)} domains via {args.model} ...")
        tasks = [
            generate_prompts_with_retry(client, dn, dd, n_per_domain, args.model)
            for dn, dd in DOMAINS.items()
        ]
        results = await asyncio.gather(*tasks)
        domain_prompts = dict(zip(DOMAINS.keys(), results))
        prompts_ckpt.write_text(json.dumps(domain_prompts, ensure_ascii=False))
        print(f"  wrote {prompts_ckpt}")

    flat_prompts: list[tuple[str, str]] = []  # (domain, hindi_prompt)
    for dn in DOMAINS:
        for p in domain_prompts.get(dn, []):
            flat_prompts.append((dn, p))
    print(f"  total Hindi prompts: {len(flat_prompts)}")

    # Step 2: Hindi reference responses, batched.
    hin_resp_ckpt = ckpt_dir / "hin_responses.json"
    if hin_resp_ckpt.exists():
        hin_responses = json.loads(hin_resp_ckpt.read_text())
        print(f"\n[Step 2/3] resuming Hindi responses from checkpoint "
              f"({len(hin_responses)} done)")
    else:
        hin_responses = []
    if len(hin_responses) < len(flat_prompts):
        print(f"\n[Step 2/3] generating Hindi reference responses, "
              f"batch size {args.batch_size} ...")
        n_batches = (len(flat_prompts) + args.batch_size - 1) // args.batch_size
        for b in tqdm(range(n_batches), desc="hin responses", unit="batch"):
            i = b * args.batch_size
            if i < len(hin_responses):
                continue
            chunk = [p for _, p in flat_prompts[i : i + args.batch_size]]
            out = await with_retry(
                generate_hin_responses, client, chunk, args.model,
                label=f"hin_resp/batch{b}",
            )
            hin_responses.extend(out)
            hin_resp_ckpt.write_text(json.dumps(hin_responses, ensure_ascii=False))
        print(f"  wrote {hin_resp_ckpt}  ({len(hin_responses)} total)")

    assert len(hin_responses) == len(flat_prompts), \
        f"hin_responses {len(hin_responses)} vs prompts {len(flat_prompts)}"

    # Step 3: parallel translation to mai/npi/bho, concurrent across langs per round.
    pair_list = [(p, r) for (_, p), r in zip(flat_prompts, hin_responses)]
    translations: dict[str, list[tuple[str, str]]] = {}
    ckpt_paths: dict[str, Path] = {}
    for code in TARGET_LANGS:
        ckpt_paths[code] = ckpt_dir / f"{code}_pairs.json"
        if ckpt_paths[code].exists():
            translations[code] = [tuple(x) for x in json.loads(ckpt_paths[code].read_text())]
            print(f"  resuming {code}: {len(translations[code])}/{len(pair_list)}")
        else:
            translations[code] = []

    print(f"\n[Step 3/3] translating prompt+response pairs to "
          f"{list(TARGET_LANGS.keys())} (concurrent per round) ...")
    n_batches = (len(pair_list) + args.batch_size - 1) // args.batch_size
    pbar = tqdm(range(n_batches), desc="rounds", unit="round")
    for batch_idx in pbar:
        i = batch_idx * args.batch_size
        batch = pair_list[i : i + args.batch_size]
        if not batch:
            break

        tasks, codes_in_flight = [], []
        for code, full in TARGET_LANGS.items():
            if i < len(translations[code]):
                continue
            tasks.append(with_retry(
                translate_pairs_batch, client, batch, code, full, args.model,
                label=f"{code}/batch{batch_idx}",
            ))
            codes_in_flight.append(code)
        if not tasks:
            continue
        results = await asyncio.gather(*tasks)
        for code, res in zip(codes_in_flight, results):
            translations[code].extend(res)
            ckpt_paths[code].write_text(json.dumps(
                [list(t) for t in translations[code]], ensure_ascii=False
            ))

    # Sanity checks before saving.
    for code in TARGET_LANGS:
        assert len(translations[code]) == len(pair_list), \
            f"{code}: {len(translations[code])} vs {len(pair_list)}"

    # Assemble parquet.
    rows = []
    for idx, ((domain, hin_p), hin_r) in enumerate(zip(flat_prompts, hin_responses)):
        row = {
            "id": f"eval_{idx:04d}",
            "domain": domain,
            "hin_prompt": hin_p,
            "hin_response": hin_r,
        }
        for code in TARGET_LANGS:
            tp, tr = translations[code][idx]
            row[f"{code}_prompt"] = tp
            row[f"{code}_response"] = tr
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} rows, "
          f"{len(df.columns)} cols)")
    print(df.head(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="Total Hindi prompts (split evenly across 4 domains).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Prompts per Gemini call for response/translation steps.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
