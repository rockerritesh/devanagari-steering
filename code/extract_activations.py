"""Step 1 of devanagari-steering: per-(model, language) memory-bank construction.

For each model M in {Llama-3.1-8B-Instruct, Aya-23-8B} and each language L in
{hin, mai, npi, bho}:
  - load the 250-passage parallel corpus,
  - run every passage through M with output_hidden_states=True,
  - mean-pool the non-pad token positions at every layer (incl. embedding layer),
  - save the (N, num_layers+1, hidden_dim) tensor to
    results/memory_banks/{model_short}/{lang}.pt.

Loaded in 4-bit NF4 (bnb_4bit_compute_dtype=bfloat16) so that two 8B models can
fit on a single T4 (16 GB). The forward pass dequantizes per-tile so captured
hidden states are bf16 — same precision both models use throughout, so the
relative cross-language geometry that drives the steering vector is preserved.

Usage on the GCP T4 VM:
    HF_TOKEN=$HF_TOKEN python code/extract_activations.py
    # smoke test:
    python code/extract_activations.py --models llama --langs hin --limit 4

Resumable: if the output .pt for (model, lang) already exists it is skipped.
"""

from __future__ import annotations

import argparse
import gc
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
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "parallel" / "parallel_devanagari_250.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "memory_banks"

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "aya": "CohereForAI/aya-23-8B",
    # himalaya-gemma-4-e2b-it: a Gemma4ForConditionalGeneration (multimodal)
    # decoder fine-tuned on English+Nepali. We only use its TEXT decoder; see
    # the text_lm branch in load_model. 35 text layers -> 36 hidden states.
    "himalaya": "himalaya-ai/himalaya-gemma-4-e2b-it",
}

# Short names whose HF id is a *ForConditionalGeneration multimodal wrapper: we
# run only the inner text language model so output_hidden_states yields the text
# decoder stream (no pixel/audio inputs needed). These default to bf16 (small
# enough for the T4) rather than 4-bit, sidestepping bitsandbytes+multimodal.
MULTIMODAL_TEXT_LM = {"himalaya"}

LANGS = ["hin", "mai", "npi", "bho", "en"]
MAX_LEN = 1024
DEFAULT_BATCH = 8


def _get_text_decoder(model):
    """Return the inner text decoder of a *ForConditionalGeneration multimodal
    model, so a forward with input_ids alone produces text hidden states.
    Tries the known transformers attribute layouts for Gemma3/Gemma4-style
    models and verifies the submodule accepts output_hidden_states."""
    candidates = [
        ("model", "language_model"),   # Gemma3/4Model.language_model (usual)
        ("language_model",),           # some wrappers expose it at top level
        ("model",),                    # last resort: the base model
    ]
    for path in candidates:
        sub = model
        ok = True
        for attr in path:
            if hasattr(sub, attr):
                sub = getattr(sub, attr)
            else:
                ok = False
                break
        if ok and hasattr(sub, "forward"):
            return sub
    raise RuntimeError(
        "Could not locate a text decoder submodule on "
        f"{type(model).__name__}; inspect model structure.")


def load_model(model_id: str, hf_token: str | None, no_quant: bool = False,
               text_lm: bool = False):
    """Load a causal LM. Default: 4-bit NF4 with bf16 compute on cuda:0.
    With no_quant=True: full bf16 via device_map='auto' (the 16 GB T4 cannot
    hold an 8B model in bf16, so accelerate offloads the overflow to CPU RAM).
    Used to verify the geometry is not a quantization artifact.

    With text_lm=True: load a multimodal *ForConditionalGeneration model in bf16
    and return only its inner text decoder (for Gemma4/himalaya). The full
    tokenizer is returned; the decoder consumes input_ids/attention_mask."""
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if text_lm:
        from transformers import AutoModelForImageTextToText
        full = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map={"": 0},
            dtype=torch.bfloat16,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
        full.eval()
        decoder = _get_text_decoder(full)
        decoder.eval()
        return tok, decoder

    if no_quant:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map={"": 0},
            dtype=torch.bfloat16,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
    model.eval()
    return tok, model


@torch.inference_mode()
def extract_for_lang(
    tok,
    model,
    texts: list[str],
    max_len: int = MAX_LEN,
    batch_size: int = DEFAULT_BATCH,
) -> torch.Tensor:
    """Forward in mini-batches; mean-pool non-pad positions per layer per sample.

    Returns a tensor of shape (N, num_layers+1, hidden_dim) on CPU in bf16.
    The +1 is HF's embedding-layer hidden state at index 0 of `hidden_states`.

    Padding is left-side (causal LM convention) so that the pad mask, applied
    to the per-token hidden states before mean-pooling, drops only positions
    that contributed nothing to the autoregressive context.
    """
    device = next(model.parameters()).device
    if tok.padding_side != "left":
        tok.padding_side = "left"

    out_chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="forward", leave=False):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        ).to(device)
        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        mask = enc["attention_mask"].to(torch.bfloat16)            # (B, S)
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)         # (B, 1)
        per_layer = []
        for hs in out.hidden_states:                               # (B, S, H)
            masked_sum = (hs * mask.unsqueeze(-1)).sum(dim=1)      # (B, H)
            per_layer.append(masked_sum / denom)                   # (B, H)
        # stack -> (B, L+1, H) -> CPU
        out_chunks.append(torch.stack(per_layer, dim=1).to("cpu"))
        del out, enc, mask, denom, per_layer
    return torch.cat(out_chunks, dim=0)                            # (N, L+1, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N rows (smoke testing).")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                    help="Mini-batch size for forward passes. T4 fits 8 at "
                         "max_len=1024 with 4-bit 8B models; drop to 4 on OOM.")
    ap.add_argument("--no-quant", action="store_true",
                    help="Full bf16 (device_map=auto, CPU offload) instead of "
                         "4-bit NF4 — to verify the geometry is not a "
                         "quantization artifact.")
    ap.add_argument("--out-suffix", default="",
                    help="If set, appended to each model subdir name. e.g. "
                         "--out-suffix verb produces "
                         "results/memory_banks/llama_verb/{lang}.pt instead of "
                         "the default results/memory_banks/llama/.")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        print("WARNING: HF_TOKEN not set; gated repos will fail.")

    df = pd.read_parquet(args.corpus)
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"Corpus rows: {len(df)} from {args.corpus}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = df["id"].astype(str).tolist()

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    for short in args.models:
        model_id = MODELS[short]
        out_subdir = args.out_dir / f"{short}{suffix}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        # skip if every requested lang is already done for this model
        if all((out_subdir / f"{lg}.pt").exists() for lg in args.langs):
            print(f"[{short}{suffix}] all targets exist; skipping load.")
            continue

        print(f"\n=== Loading {short}{suffix}: {model_id} ===")
        t0 = time.time()
        tok, model = load_model(model_id, hf_token, no_quant=args.no_quant,
                                text_lm=(short in MULTIMODAL_TEXT_LM))
        print(f"loaded in {time.time() - t0:.1f}s "
              f"({torch.cuda.memory_allocated() / 1e9:.2f} GB on cuda:0)")

        for lg in args.langs:
            out_path = out_subdir / f"{lg}.pt"
            if out_path.exists():
                print(f"[{short}{suffix}/{lg}] exists, skipping.")
                continue
            texts = df[lg].astype(str).tolist()
            print(f"[{short}{suffix}/{lg}] extracting {len(texts)} samples...")
            t1 = time.time()
            vecs = extract_for_lang(
                tok, model, texts,
                max_len=args.max_len,
                batch_size=args.batch_size,
            )
            elapsed = time.time() - t1
            payload = {
                "vectors": vecs,                  # (N, L+1, H), bf16, cpu
                "ids": ids,
                "model": model_id,
                "lang": lg,
                "pooling": "mean_nonpad",
                "compute_dtype": "bfloat16",
                "load_dtype": "bf16" if args.no_quant else "nf4",
                "max_len": args.max_len,
                "n_samples": len(texts),
                "n_layers_plus_embed": vecs.shape[1],
                "hidden_dim": vecs.shape[2],
            }
            torch.save(payload, out_path)
            sidecar = out_path.with_suffix(".json")
            sidecar.write_text(json.dumps(
                {k: v for k, v in payload.items() if k not in {"vectors", "ids"}},
                indent=2,
            ))
            print(f"[{short}{suffix}/{lg}] saved {out_path}  shape={tuple(vecs.shape)}  "
                  f"in {elapsed/60:.1f} min")

        # free model before loading the next one
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
