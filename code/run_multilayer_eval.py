"""Task 3 — multi-layer composite steering on the eval set.

Hooks SEVERAL decoder layers simultaneously, each with that layer's steering
vector v_l = mu_hin - mu_target, optionally norm-equalised (each layer's vector
divided by its L2 norm so every hooked layer contributes equal magnitude alpha).
Goal: test whether composite multi-layer steering breaks the ~1.8 single-layer
language-adherence wall for mai/bho. Output schema matches eval_judge.py.

Usage on VM:
    .venv/bin/python code/run_multilayer_eval.py \\
        --model llama --target bho --layers 18 20 22 --prompt-lang hin \\
        --alphas 0 -2 -3 -4 --norm-equalize \\
        --n-prompts 30 --batch-size 4 --max-new-tokens 90 --device cuda \\
        --out results/generations/llama_bho_multiL_eval.parquet
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


def build_chat(tok, t):
    return tok.apply_chat_template([{"role": "user", "content": t}],
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


def layer_vecs(steering_dir, target, layers, norm_eq, device, dtype):
    payload = torch.load(steering_dir / f"{target}.pt", map_location="cpu", weights_only=False)
    v_all = payload["steering"].float()                  # (L+1, H)
    out = {}
    for k in layers:
        v = v_all[k + 1]
        if norm_eq:
            v = v / max(v.norm().item(), 1e-6)
        out[k] = v.to(device=device, dtype=dtype)
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS), required=True)
    ap.add_argument("--target", choices=["mai", "npi", "bho"], required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--prompt-lang", choices=["hin", "mai", "npi", "bho"], required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--norm-equalize", action="store_true")
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
    prompts = eval_df[col].tolist(); eval_ids = eval_df["id"].tolist(); domains = eval_df["domain"].tolist()
    print(f"loaded {len(prompts)} prompts; layers={args.layers} norm_eq={args.norm_equalize}")

    load_dotenv(PROJECT_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")
    t0 = time.time(); tok, model = load_model(args.model, args.device, hf_token)
    print(f"loaded {args.model} in {time.time()-t0:.1f}s; decoder layers={len(model.model.layers)}")

    dev = "cuda:0" if args.device == "cuda" else args.device
    dt = torch.bfloat16 if args.device == "cuda" else torch.float16
    vecs = layer_vecs(args.steering_dir, args.target, args.layers, args.norm_equalize, dev, dt)
    print("  |v| per layer: " + ", ".join(f"L{k}={vecs[k].float().norm():.2f}" for k in args.layers))

    chats = [build_chat(tok, p) for p in prompts]
    nb = (len(prompts) + args.batch_size - 1) // args.batch_size
    rows = []
    for alpha in args.alphas:
        print(f"\n=== alpha={alpha:+.2f} on layers {args.layers} ===")
        for b in tqdm(range(nb), desc=f"a={alpha:+.1f}", leave=False):
            i = b * args.batch_size
            bc = chats[i:i + args.batch_size]
            if not bc:
                continue
            enc = tok(bc, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(args.device)
            handles = [model.model.layers[k].register_forward_hook(make_hook(vecs[k], alpha))
                       for k in args.layers]
            try:
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     temperature=1.0, pad_token_id=tok.pad_token_id)
            finally:
                for h in handles:
                    h.remove()
            dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            lstr = "-".join(map(str, args.layers))
            for j, text in enumerate(dec):
                k = i + j
                rows.append(dict(
                    id=f"{eval_ids[k]}__{args.model}__{args.target}__multiL{lstr}"
                       f"__{'ne' if args.norm_equalize else 'raw'}__a{alpha:+.3f}",
                    eval_id=eval_ids[k], domain=domains[k], model=args.model,
                    target_lang=args.target, prompt_lang=args.prompt_lang,
                    layers=lstr, norm_equalized=bool(args.norm_equalize), alpha=alpha,
                    prompt=prompts[k], generation=text))

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  shape={out_df.shape}")
    for a in sorted(out_df["alpha"].unique()):
        s = out_df[out_df.alpha == a].iloc[0]
        print(f"  a={a:+.2f}: {s['generation'][:110]}")


if __name__ == "__main__":
    main()
