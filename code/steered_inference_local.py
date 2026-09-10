"""Step 3 quick-look: steered generation on Mac (mps) for Llama-3.1-8B.

Designed for a fast qualitative gut-check while Aya is still extracting on
the GCP VM. Loads Llama in fp16 on mps, hooks one chosen layer, and for each
target language generates a few continuations at alpha in {0, 1, 2} so we can
eyeball whether the steering vector moves outputs in the right direction.

Hook: h_l <- h_l + alpha * v_{T->hin, l}, broadcast across batch and seq.
The vector is added to the LayerNorm-output residual stream at the chosen
decoder block's output. We register a forward hook on
`model.model.layers[layer_idx]` so that the modification flows to all
subsequent layers and into the LM head.

Usage:
    .venv/bin/python code/steered_inference_local.py \\
        --target npi --layer 20 --alphas 0 1 2 --max-new-tokens 80
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STEERING_ROOT = PROJECT_ROOT / "results" / "constellations"
MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "aya": "CohereForAI/aya-23-8B",
}

# A tiny set of in-target-language instructions for the qualitative gut-check.
# Two per target keeps memory pressure low on a 24 GB Mac.
PROMPTS = {
    "mai": [
        "नीचाँ देल सब्द सँ एकटा दु-तीन वाक्यक छोट कथा लिखू: गाम, बच्चा, नदी।",
        "मिथिलाक संस्कृति आ खान-पान केर बारे मे संक्षेप मे बताउ।",
    ],
    "npi": [
        "तलका शब्दहरू प्रयोग गरेर दुई-तीन वाक्यको छोटो कथा लेख्नुहोस्: गाउँ, बच्चा, नदी।",
        "नेपालको सांस्कृतिक विविधताको बारेमा छोटो परिचय दिनुहोस्।",
    ],
    "bho": [
        "नीचे दिहल शब्दन के इस्तेमाल कर के दू-तीन वाक्य के छोट कहानी लिखीं: गाँव, बच्चा, नदी।",
        "भोजपुरी भाषा आ संस्कृति के बारे में थोड़ा-बहुत बताईं।",
    ],
    # Hindi-source prompts for the reversed test (push hin -> target via -alpha).
    # Same content as the target prompts above, in standard Hindi.
    "hin": [
        "नीचे दिए गए शब्दों का उपयोग करके दो-तीन वाक्यों की एक छोटी कहानी लिखें: गाँव, बच्चा, नदी।",
        "अपने क्षेत्र की संस्कृति और खान-पान के बारे में संक्षेप में बताएँ।",
    ],
}


def build_chat(tok, user_text: str) -> str:
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_hook(steer_vec: torch.Tensor, alpha: float):
    """Forward hook that adds alpha * steer_vec to the layer output residual.

    The decoder block returns a tuple `(hidden, present_kv, ...)`; we modify
    only the hidden-states tensor.
    """
    def hook(_module, _inputs, output):
        if alpha == 0.0:
            return output
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * steer_vec.to(device=h.device, dtype=h.dtype)
            return (h,) + output[1:]
        return output + alpha * steer_vec.to(device=output.device, dtype=output.dtype)
    return hook


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["mai", "npi", "bho"], default="npi",
                    help="The target language for the steering vector. "
                         "v = mu_hin - mu_target.")
    ap.add_argument("--prompt-lang", choices=["mai", "npi", "bho", "hin"], default=None,
                    help="Which language the test prompts are in. Defaults to "
                         "--target (push target -> hin). Set to 'hin' and pass "
                         "negative alphas to run the reversed direction "
                         "(push hin -> target).")
    ap.add_argument("--layer", type=int, default=None,
                    help="Inject at a single decoder block (0-indexed). Mutually "
                         "exclusive with --layers.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Multi-layer injection: pass a list of decoder-block "
                         "indices, e.g. --layers 20 22 24 26. Each gets the same "
                         "alpha. Mutually exclusive with --layer.")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--norm-equalize", action="store_true",
                    help="Divide each layer's steering vector by its L2 norm "
                         "before scaling by alpha. Critical for multi-layer "
                         "steering over a wide band (e.g. L15-L32) because "
                         "raw |v_l| grows ~25x from L15 to L32; without this, "
                         "the same alpha over-perturbs deep layers and "
                         "under-perturbs shallow ones.")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--model", choices=list(MODEL_IDS.keys()), default="llama",
                    help="Which base model to load. Picks the corresponding "
                         "steering-dir under results/constellations/{model}/.")
    ap.add_argument("--steering-dir", type=Path, default=None,
                    help="Override steering-vector dir. Defaults to "
                         "results/constellations/{model}/.")
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--prompt-start", type=int, default=0,
                    help="Index of the first prompt in PROMPTS[lang] to use. "
                         "Lets us pick the second/third prompt without re-running "
                         "the earlier ones.")
    ap.add_argument("--device", default="mps",
                    help="mps for Mac GPU, cpu for safety, cuda elsewhere.")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")

    model_id = MODEL_IDS[args.model]
    if args.steering_dir is None:
        args.steering_dir = DEFAULT_STEERING_ROOT / args.model

    print(f"Loading {args.model} ({model_id}) on {args.device} "
          f"({'4-bit NF4 / bf16' if args.device == 'cuda' else 'fp16'})...")
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(token=hf_token, low_cpu_mem_usage=True)
    if args.device == "cuda":
        # Match the 4-bit recipe used to extract the memory banks so that
        # inference-time hidden states sit in the same representation regime
        # as the centroids the steering vector was built from.
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = torch.float16
        load_kwargs["device_map"] = {"": args.device}

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    print(f"loaded.  num decoder layers = {len(model.model.layers)}")

    # Load steering vectors for chosen target. The .pt holds a (L+1, H) tensor
    # where index 0 is the embedding-layer output and indices 1..L correspond to
    # the outputs of decoder blocks 0..L-1. So a hook attached to
    # model.model.layers[k] receives index (k+1) in the captured stack.
    steering_path = args.steering_dir / f"{args.target}.pt"
    payload = torch.load(steering_path, map_location="cpu", weights_only=False)
    v_all = payload["steering"].float()                        # (L+1, H)
    target_device = "cuda:0" if args.device == "cuda" else args.device
    target_dtype = torch.bfloat16 if args.device == "cuda" else torch.float16

    if args.layer is not None and args.layers is not None:
        ap.error("--layer and --layers are mutually exclusive")
    if args.layer is None and args.layers is None:
        args.layer = 20  # back-compat default
    if args.layer is not None:
        layers_to_hook = [args.layer]
    else:
        layers_to_hook = sorted(set(args.layers))

    # Clamp to valid model.model.layers indices [0, num_layers-1].
    # v_all has shape (num_layers+1, H): v_all[0] is the embedding output and
    # v_all[k+1] is the output of decoder layer k. So a hook on layers[k] uses
    # v_all[k+1], valid only for k in [0, num_layers-1].
    num_decoder_layers = len(model.model.layers)
    valid = [l for l in layers_to_hook if 0 <= l < num_decoder_layers]
    if valid != layers_to_hook:
        dropped = sorted(set(layers_to_hook) - set(valid))
        print(f"WARNING: layers {dropped} are out of range [0, {num_decoder_layers - 1}]; skipped.")
    layers_to_hook = valid
    if not layers_to_hook:
        ap.error("No valid layers to hook after clamping.")

    layer_vecs: dict[int, torch.Tensor] = {}
    for k in layers_to_hook:
        v_k = v_all[k + 1]
        if args.norm_equalize:
            n = v_k.float().norm()
            v_k = v_k / max(n.item(), 1e-6)
        layer_vecs[k] = v_k.to(device=target_device, dtype=target_dtype)
    norm_tag = " [norm-equalized to unit]" if args.norm_equalize else ""
    print(f"steering for {args.target} at layers {layers_to_hook}{norm_tag}: "
          + ", ".join(f"|v_{k}|={layer_vecs[k].float().norm():.2f}"
                      for k in layers_to_hook))

    prompt_lang = args.prompt_lang or args.target
    s = args.prompt_start
    prompts = PROMPTS[prompt_lang][s : s + args.n_prompts]
    direction = "hin -> target" if prompt_lang == "hin" else "target -> hin"
    print(f"prompt language = {prompt_lang}, steering target = {args.target}, "
          f"direction (with positive alpha) = target -> hin; "
          f"effective direction this run = {direction} "
          f"({'use negative alphas' if prompt_lang == 'hin' else 'use positive alphas'}).")

    for i, user_text in enumerate(prompts):
        print(f"\n========== PROMPT {i+1}/{len(prompts)} "
              f"(prompt_lang={prompt_lang}, target={args.target}) ==========")
        print(user_text)
        chat = build_chat(tok, user_text)
        enc = tok(chat, return_tensors="pt").to(args.device)

        for alpha in args.alphas:
            handles = []
            try:
                for k in layers_to_hook:
                    h = model.model.layers[k].register_forward_hook(
                        make_hook(layer_vecs[k], alpha)
                    )
                    handles.append(h)
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            finally:
                for h in handles:
                    h.remove()
            new_tokens = gen[0, enc["input_ids"].shape[1]:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            n_layers_str = f"L={len(layers_to_hook)}" if len(layers_to_hook) > 1 else f"L{layers_to_hook[0]}"
            label = "UNSTEERED" if alpha == 0 else f"alpha={alpha:+.2f} ({n_layers_str})"
            print(f"\n--- {label} ---\n{text}")


if __name__ == "__main__":
    main()
