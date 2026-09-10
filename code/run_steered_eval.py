"""Step 4 glue: run steered inference over the eval prompt set, write a generations
parquet that `eval_judge.py` can score.

Reads `data/parallel/eval_prompts_devanagari_200.parquet`, picks the
`--prompt-lang` column for the prompt content, hooks the requested decoder layer
with the steering vector for `--target`, sweeps over `--alphas`, and saves one
row per (prompt × alpha) combination to a parquet shaped exactly the way
`eval_judge.py` expects:

    columns: id, model, target_lang, prompt_lang, prompt, generation,
             alpha, layer, eval_id, domain

`id` is `eval_<idx>__<model>__<target>__L<layer>__a<alpha>` so each row is a
stable unique key reusable across re-runs / judging / aggregation.

Designed for the GCP T4 VM (4-bit NF4 / bf16). Generation is *batched* — we
group prompts into mini-batches of `--batch-size` and run one
`model.generate()` call per (alpha, batch) pair. Hooks are attached/detached
per alpha, so within a batch all prompts see the same alpha.

Usage on VM:
    .venv/bin/python code/run_steered_eval.py \\
        --model aya --target npi --layer 22 --prompt-lang hin \\
        --alphas 0 -0.6 -0.7 \\
        --eval-parquet data/parallel/eval_prompts_devanagari_200.parquet \\
        --out results/generations/aya_npi_l22_eval200.parquet \\
        --n-prompts 50 --batch-size 4 --max-new-tokens 90 --device cuda

Smoke test on Mac (mps, fp16, very small):
    .venv/bin/python code/run_steered_eval.py \\
        --model llama --target npi --layer 20 --prompt-lang hin \\
        --alphas 0 -1 \\
        --n-prompts 2 --batch-size 1 --max-new-tokens 30 --device mps
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "generations"

MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "aya": "CohereForAI/aya-23-8B",
}


def build_chat(tok, user_text: str) -> str:
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_hook(steer_vec: torch.Tensor, alpha: float):
    """Forward hook that adds alpha * steer_vec to the layer output residual."""
    def hook(_module, _inputs, output):
        if alpha == 0.0:
            return output
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * steer_vec.to(device=h.device, dtype=h.dtype)
            return (h,) + output[1:]
        return output + alpha * steer_vec.to(device=output.device, dtype=output.dtype)
    return hook


def load_model(model_key: str, device: str, hf_token: str | None,
               no_quant: bool = False):
    model_id = MODEL_IDS[model_key]
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # causal LM convention; required for batched generate

    load_kwargs = dict(token=hf_token, low_cpu_mem_usage=True)
    if device == "cuda" and not no_quant:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["dtype"] = torch.bfloat16
    elif device == "cuda" and no_quant:
        # full bf16; 8B won't fit on a 16 GB T4, so let accelerate offload the
        # overflow to CPU RAM. Slow but verifies the effect is not an NF4 artifact.
        load_kwargs["device_map"] = "auto"
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = torch.float16
        load_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return tok, model


def get_steer_vec(steering_dir: Path, target: str, layer_index_in_decoder: int,
                  device: str, dtype: torch.dtype) -> torch.Tensor:
    """Load v_{target}.pt and pull out the row matching decoder layer k.

    The .pt holds (L+1, H): index 0 = embedding output, index k+1 = output of
    decoder layer k. So a hook on `model.model.layers[k]` should add v_all[k+1].
    """
    payload = torch.load(steering_dir / f"{target}.pt", map_location="cpu",
                         weights_only=False)
    v_all = payload["steering"].float()                # (L+1, H)
    return v_all[layer_index_in_decoder + 1].to(device=device, dtype=dtype)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS.keys()), required=True)
    ap.add_argument("--target", choices=["mai", "npi", "bho"], required=True,
                    help="Target language for the steering vector "
                         "(v = mu_hin - mu_target).")
    ap.add_argument("--layer", type=int, required=True,
                    help="Decoder block index to hook (0-indexed).")
    ap.add_argument("--prompt-lang", choices=["hin", "mai", "npi", "bho", "en"], required=True,
                    help="Which language column to read prompts from in the "
                         "eval parquet. For hin->target steering, set "
                         "--prompt-lang hin and pass NEGATIVE alphas.")
    ap.add_argument("--alphas", type=float, nargs="+", required=True,
                    help="Alpha values to sweep (e.g. 0 -1 -1.5 -2.5).")
    ap.add_argument("--eval-parquet", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--steering-dir", type=Path, default=None,
                    help="Defaults to results/constellations/{model}/.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output parquet path. Parent dir is created if missing.")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="Limit to first N rows of the eval parquet (smoke test).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-quant", action="store_true",
                    help="Load in full bf16 (device_map=auto, CPU offload) "
                         "instead of 4-bit NF4 — quantization-artifact check.")
    ap.add_argument("--norm-equalize", action="store_true",
                    help="Divide steering vector by its layer-wise L2 norm "
                         "before scaling — alpha then is in unit-direction units.")
    args = ap.parse_args()

    if args.steering_dir is None:
        args.steering_dir = DEFAULT_STEERING_ROOT / args.model

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load eval prompts.
    eval_df = pd.read_parquet(args.eval_parquet)
    if args.n_prompts is not None:
        eval_df = eval_df.head(args.n_prompts).copy()
    prompt_col = f"{args.prompt_lang}_prompt"
    if prompt_col not in eval_df.columns:
        raise SystemExit(f"column {prompt_col} not in {args.eval_parquet}")
    print(f"loaded {len(eval_df)} eval prompts from {args.eval_parquet}")

    load_dotenv(PROJECT_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")

    # 2. Load model.
    print(f"loading {args.model} on {args.device} ...")
    t0 = time.time()
    tok, model = load_model(args.model, args.device, hf_token, no_quant=args.no_quant)
    n_decoder = len(model.model.layers)
    if args.layer < 0 or args.layer >= n_decoder:
        raise SystemExit(f"--layer {args.layer} out of range [0, {n_decoder - 1}]")
    print(f"  loaded in {time.time() - t0:.1f}s; "
          f"decoder layers = {n_decoder}")

    # 3. Load steering vector.
    target_device = "cuda:0" if args.device == "cuda" else args.device
    target_dtype = torch.bfloat16 if args.device == "cuda" else torch.float16
    v = get_steer_vec(args.steering_dir, args.target, args.layer,
                      target_device, target_dtype)
    if args.norm_equalize:
        v = v / max(v.float().norm().item(), 1e-6)
    print(f"steering vector for {args.target} at L{args.layer}: |v|={v.float().norm():.3f}")

    # 4. Sweep over alphas, batched generation per (alpha, batch).
    rows = []
    eval_ids = eval_df["id"].tolist()
    domains = eval_df["domain"].tolist()
    prompts = eval_df[prompt_col].tolist()
    chats = [build_chat(tok, p) for p in prompts]

    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

    for alpha in args.alphas:
        print(f"\n=== alpha = {alpha:+.3f} ===")
        for b in tqdm(range(n_batches), desc=f"a={alpha:+.2f}", unit="batch"):
            i = b * args.batch_size
            batch_chats = chats[i : i + args.batch_size]
            if not batch_chats:
                continue
            enc = tok(batch_chats, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(args.device)

            handle = model.model.layers[args.layer].register_forward_hook(
                make_hook(v, alpha)
            )
            try:
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            finally:
                handle.remove()

            input_len = enc["input_ids"].shape[1]
            new_tokens = gen[:, input_len:]
            decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)

            for j, text in enumerate(decoded):
                idx_in_eval = i + j
                rows.append(dict(
                    id=f"{eval_ids[idx_in_eval]}__{args.model}__{args.target}"
                       f"__L{args.layer}__a{alpha:+.3f}",
                    eval_id=eval_ids[idx_in_eval],
                    domain=domains[idx_in_eval],
                    model=args.model,
                    target_lang=args.target,
                    prompt_lang=args.prompt_lang,
                    layer=args.layer,
                    alpha=alpha,
                    norm_equalized=bool(args.norm_equalize),
                    prompt=prompts[idx_in_eval],
                    generation=text,
                ))

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  shape={out_df.shape}")
    print(f"  unique alphas: {sorted(out_df['alpha'].unique().tolist())}")
    print(f"  bytes: {args.out.stat().st_size / 1024:.1f} KB")
    # quick sanity: show first row per alpha
    for a in sorted(out_df["alpha"].unique()):
        sub = out_df[out_df["alpha"] == a]
        print(f"\n--- sample @ alpha={a:+.2f} ---")
        print(f"  prompt: {sub.iloc[0]['prompt'][:80]} ...")
        print(f"  gen:    {sub.iloc[0]['generation'][:120]} ...")


if __name__ == "__main__":
    main()
