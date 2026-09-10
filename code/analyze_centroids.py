"""Step 2a: centroid sanity check — go/no-go gate before steered inference.

For a given model, load `results/memory_banks/{model}/{lang}.pt` for each
language, compute per-layer centroids (mean over the 250 samples), and report
how close Hindi sits to its sister languages compared to a random baseline.

If the sister-language cosines clearly exceed the random baseline at a band of
mid layers, the Hindi manifold reaches the sister languages and steering has
something to grip — proceed to Step 2b. Otherwise, stop and rethink.

Usage:
    python code/analyze_centroids.py --model llama
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMBANK = PROJECT_ROOT / "results" / "memory_banks"
LANGS = ["hin", "mai", "npi", "bho"]


def load_centroids(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load each {lang}.pt and return per-layer centroids in fp32.

    .pt payload was saved as bf16; centroid math is done in fp32 for stable
    cosine computation.
    """
    cents: dict[str, torch.Tensor] = {}
    for lang in LANGS:
        path = model_dir / f"{lang}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        v = payload["vectors"].float()                          # (N, L+1, H)
        cents[lang] = v.mean(dim=0)                             # (L+1, H)
    return cents


def cos_per_layer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity per layer between two (L, H) tensors."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)  # (L,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama",
                    help="Subdir under results/memory_banks/, e.g. llama or aya.")
    ap.add_argument("--membank-dir", type=Path, default=DEFAULT_MEMBANK)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_dir = args.membank_dir / args.model
    print(f"Loading centroids from {model_dir}")
    cents = load_centroids(model_dir)

    L_plus_1, H = cents["hin"].shape
    print(f"Layers (incl. embedding) = {L_plus_1},  hidden dim = {H}\n")

    # random baseline: a unit-norm gaussian vector at each layer
    g = torch.Generator().manual_seed(args.seed)
    rand = torch.randn(L_plus_1, H, generator=g)
    rand = rand / rand.norm(dim=-1, keepdim=True)

    sims = {
        target: cos_per_layer(cents["hin"], cents[target]) for target in ["mai", "npi", "bho"]
    }
    sims["random"] = cos_per_layer(cents["hin"], rand)

    # also report inter-target cosines, useful for the linguistic-distance story
    extra = {
        "mai_vs_npi": cos_per_layer(cents["mai"], cents["npi"]),
        "mai_vs_bho": cos_per_layer(cents["mai"], cents["bho"]),
        "npi_vs_bho": cos_per_layer(cents["npi"], cents["bho"]),
    }

    header = f"{'layer':>5} | {'hin~mai':>8} {'hin~npi':>8} {'hin~bho':>8} {'hin~rnd':>8} || {'mai~npi':>8} {'mai~bho':>8} {'npi~bho':>8}"
    print(header)
    print("-" * len(header))
    for l in range(L_plus_1):
        row = (
            f"{l:>5} | "
            f"{sims['mai'][l]:>+8.4f} {sims['npi'][l]:>+8.4f} {sims['bho'][l]:>+8.4f} "
            f"{sims['random'][l]:>+8.4f} || "
            f"{extra['mai_vs_npi'][l]:>+8.4f} {extra['mai_vs_bho'][l]:>+8.4f} {extra['npi_vs_bho'][l]:>+8.4f}"
        )
        print(row)

    # summary: where is the gap (hin~target − hin~random) largest?
    print("\n=== Layer with maximum (hin~target − hin~random) gap ===")
    for target in ["mai", "npi", "bho"]:
        gap = sims[target] - sims["random"]
        best = int(gap.argmax())
        print(f"  hin -> {target}: layer {best:>2}  cos={sims[target][best]:+.4f}  "
              f"baseline={sims['random'][best]:+.4f}  gap={gap[best]:+.4f}")

    # save a JSON summary for downstream tooling
    out = model_dir / "centroid_analysis.json"
    payload = {
        "model": args.model,
        "n_layers_plus_embed": L_plus_1,
        "hidden_dim": H,
        "per_layer": {
            **{f"hin_vs_{k}": v.tolist() for k, v in sims.items()},
            **{k: v.tolist() for k, v in extra.items()},
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
