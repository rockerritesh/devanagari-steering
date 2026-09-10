"""ICL baseline (no steering) — the comparison reviewers asked for.

Can plain prompting reach the target language as well as steering, and with
better fluency? For each test prompt we build a chat with:
  - an instruction (written in the TARGET language) to answer in that language;
  - optionally k few-shot exemplars: (Hindi prompt -> gold TARGET response),
    drawn from eval rows OUTSIDE the first --n-prompts test set (no leakage).
Then generate unsteered and emit a parquet that eval_judge.py can score, so the
adherence/fluency numbers are directly comparable to the steering tables.

Usage on VM:
    .venv/bin/python code/run_icl_baseline.py \
        --model llama --target npi --shots 3 \
        --n-prompts 30 --batch-size 4 --max-new-tokens 90 --device cuda \
        --out results/generations/llama_npi_icl3.parquet
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = PROJECT_ROOT / "data" / "parallel" / "eval_prompts_devanagari_200.parquet"
MODEL_IDS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "aya": "CohereForAI/aya-23-8B"}

# "Answer the following question in <Language>." written natively per target.
INSTR = {
    "npi": "कृपया तलको प्रश्नको उत्तर नेपाली भाषामा मात्र दिनुहोस्।",
    "mai": "कृपया निम्न प्रश्नक उत्तर मैथिली भाषा मे देल जाउ।",
    "bho": "कृपया नीचे दिहल सवाल के जवाब भोजपुरी भाषा में दीं।",
}
LANG_NAME = {"npi": "Nepali", "mai": "Maithili", "bho": "Bhojpuri"}


def load_model(model_key, device, hf_token):
    mid = MODEL_IDS[model_key]
    tok = AutoTokenizer.from_pretrained(mid, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = dict(token=hf_token, low_cpu_mem_usage=True)
    if device == "cuda":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}; kw["dtype"] = torch.bfloat16
    else:
        kw["dtype"] = torch.float16; kw["device_map"] = {"": device}
    m = AutoModelForCausalLM.from_pretrained(mid, **kw)
    m.eval()
    return tok, m


def build_messages(tok, instr, exemplars, user_q):
    """exemplars: list of (hin_prompt, tgt_response). Multi-turn few-shot."""
    msgs = []
    for q, a in exemplars:
        msgs.append({"role": "user", "content": f"{instr}\n\n{q}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"{instr}\n\n{user_q}"})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS), required=True)
    ap.add_argument("--target", choices=["mai", "npi", "bho"], required=True)
    ap.add_argument("--shots", type=int, default=0, help="k few-shot exemplars.")
    ap.add_argument("--eval-parquet", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.eval_parquet)
    test = df.head(args.n_prompts).copy()
    # exemplars from rows AFTER the test block -> no leakage
    pool = df.iloc[args.n_prompts:args.n_prompts + max(args.shots, 0)]
    exemplars = [(r.hin_prompt, getattr(r, f"{args.target}_response"))
                 for r in pool.itertuples()]
    instr = INSTR[args.target]
    print(f"target={args.target} shots={args.shots} test={len(test)} "
          f"exemplars={len(exemplars)}")

    load_dotenv(PROJECT_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")
    t0 = time.time(); tok, model = load_model(args.model, args.device, hf_token)
    print(f"loaded {args.model} in {time.time()-t0:.1f}s")

    chats = [build_messages(tok, instr, exemplars, q) for q in test.hin_prompt]
    nb = (len(chats) + args.batch_size - 1) // args.batch_size
    rows = []
    for b in tqdm(range(nb), desc="icl"):
        i = b * args.batch_size
        bc = chats[i:i + args.batch_size]
        if not bc:
            continue
        enc = tok(bc, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(args.device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False, temperature=1.0,
                             pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        for j, text in enumerate(dec):
            r = test.iloc[i + j]
            rows.append(dict(
                id=f"{r.id}__{args.model}__{args.target}__icl{args.shots}",
                eval_id=r.id, domain=r.domain, model=args.model,
                target_lang=args.target, prompt_lang="hin",
                layer=-1, alpha=0.0, norm_equalized=False,
                condition=f"icl{args.shots}",
                prompt=r.hin_prompt, generation=text))
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"wrote {args.out} shape={out_df.shape}")
    for r in rows[:2]:
        print("  >", r["generation"][:120])


if __name__ == "__main__":
    main()
