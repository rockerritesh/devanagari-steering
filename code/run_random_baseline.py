"""Task 2 — random-direction baseline (defends the headline steering result).

Replaces the real steering vector v = mu_hin - mu_target with RANDOM unit
directions scaled to the SAME L2 norm |v| at the same layer, then runs the same
batched eval generation + alpha sweep. If random directions of equal magnitude do
NOT reproduce the language-adherence lift the real vector produces, the steering
effect is attributable to the specific direction, not generic perturbation.

Schema matches eval_judge.py (id, model, target_lang, prompt, generation, alpha,
layer, prompt_lang) plus random_seed / is_random for grouping.

Usage on the VM (after syncing repo):
    .venv/bin/python code/run_random_baseline.py \\
        --model llama --target npi --layer 20 --prompt-lang hin \\
        --alphas -2.0 -1.5 --n-random 10 --n-prompts 30 \\
        --batch-size 4 --max-new-tokens 90 --device cuda \\
        --out results/generations/llama_npi_l20_random.parquet
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
DEFAULT_STEERING_ROOT = PROJECT_ROOT / "results" / "constellations"
MODEL_IDS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "aya": "CohereForAI/aya-23-8B"}


def build_chat(tok, user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)


def make_hook(vec, alpha):
    def hook(_m, _i, output):
        if alpha == 0.0:
            return output
        if isinstance(output, tuple):
            h = output[0] + alpha * vec.to(output[0].device, output[0].dtype)
            return (h,) + output[1:]
        return output + alpha * vec.to(output.device, output.dtype)
    return hook


def load_model(model_key, device, hf_token):
    model_id = MODEL_IDS[model_key]
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
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
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    return tok, model


def real_vec_norm(steering_dir, target, layer):
    payload = torch.load(steering_dir / f"{target}.pt", map_location="cpu", weights_only=False)
    v = payload["steering"].float()[layer + 1]          # (H,)
    return v, float(v.norm())


def random_unit(H, seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(H, generator=g)
    return (r / r.norm()).to(device=device, dtype=dtype)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS), required=True)
    ap.add_argument("--target", choices=["mai", "npi", "bho"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--prompt-lang", choices=["hin", "mai", "npi", "bho"], required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--n-random", type=int, default=10, help="Number of random directions.")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--eval-parquet", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--steering-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.steering_dir is None:
        args.steering_dir = DEFAULT_STEERING_ROOT / args.model
    args.out.parent.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_parquet(args.eval_parquet).head(args.n_prompts).copy()
    col = f"{args.prompt_lang}_prompt"
    prompts = eval_df[col].tolist()
    eval_ids = eval_df["id"].tolist()
    domains = eval_df["domain"].tolist()
    print(f"loaded {len(prompts)} prompts; {args.n_random} random dirs × {len(args.alphas)} alphas")

    load_dotenv(PROJECT_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")

    t0 = time.time()
    tok, model = load_model(args.model, args.device, hf_token)
    H = model.config.hidden_size
    print(f"loaded {args.model} in {time.time()-t0:.1f}s; hidden={H}")

    tgt_device = "cuda:0" if args.device == "cuda" else args.device
    tgt_dtype = torch.bfloat16 if args.device == "cuda" else torch.float16
    _, vnorm = real_vec_norm(args.steering_dir, args.target, args.layer)
    print(f"real |v_{args.target}@L{args.layer}| = {vnorm:.3f} — random dirs scaled to this")

    chats = [build_chat(tok, p) for p in prompts]
    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    rows = []

    for s_i in range(args.n_random):
        seed = args.seed_base + s_i
        rvec = random_unit(H, seed, tgt_device, tgt_dtype) * vnorm
        for alpha in args.alphas:
            for b in tqdm(range(n_batches), desc=f"seed{seed} a={alpha:+.1f}", leave=False):
                i = b * args.batch_size
                bc = chats[i:i + args.batch_size]
                if not bc:
                    continue
                enc = tok(bc, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024).to(args.device)
                h = model.model.layers[args.layer].register_forward_hook(make_hook(rvec, alpha))
                try:
                    gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, temperature=1.0,
                                         pad_token_id=tok.pad_token_id)
                finally:
                    h.remove()
                dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for j, text in enumerate(dec):
                    k = i + j
                    rows.append(dict(
                        id=f"{eval_ids[k]}__{args.model}__{args.target}__L{args.layer}"
                           f"__rand{seed}__a{alpha:+.3f}",
                        eval_id=eval_ids[k], domain=domains[k], model=args.model,
                        target_lang=args.target, prompt_lang=args.prompt_lang,
                        layer=args.layer, alpha=alpha, random_seed=seed, is_random=True,
                        prompt=prompts[k], generation=text))

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  shape={out_df.shape}")
    print(out_df.groupby(["alpha", "random_seed"]).size().to_string())


if __name__ == "__main__":
    main()
