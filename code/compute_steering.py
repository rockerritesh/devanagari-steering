"""Step 2b: per-(model, target language, layer) steering vectors.

For each model and each target language T in {mai, npi, bho}:
    v_{T->hin, l} = mu_{hin, l} - mu_{T, l}

Saves to results/constellations/{model}/{T}.pt as a fp32 tensor of shape
(num_layers_plus_embed, hidden_dim) plus a sidecar json with the per-layer L2
norms (useful for picking sensible alpha at inference time).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMBANK = PROJECT_ROOT / "results" / "memory_banks"
DEFAULT_OUT = PROJECT_ROOT / "results" / "constellations"
LANGS_TARGETS = ["mai", "npi", "bho"]


def load_centroid(model_dir: Path, lang: str) -> torch.Tensor:
    payload = torch.load(model_dir / f"{lang}.pt", map_location="cpu",
                         weights_only=False)
    return payload["vectors"].float().mean(dim=0)  # (L+1, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--source", default="hin",
                    help="High-resource anchor language (hin or en). "
                         "v = mu_source - mu_target.")
    ap.add_argument("--membank-dir", type=Path, default=DEFAULT_MEMBANK)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    model_dir = args.membank_dir / args.model
    out_dir = args.out_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_hin = load_centroid(model_dir, args.source)      # (L+1, H)  [anchor centroid]
    L_plus_1, H = mu_hin.shape
    print(f"[{args.model}] L+1={L_plus_1}, H={H}, source={args.source}")

    summary = {"model": args.model, "source": args.source, "targets": {}}

    for target in LANGS_TARGETS:
        mu_t = load_centroid(model_dir, target)          # (L+1, H)
        v = mu_hin - mu_t                                # (L+1, H), points from T -> source
        norms = v.norm(dim=-1).tolist()                  # per-layer L2 norm
        out_path = out_dir / f"{target}.pt"
        torch.save({
            "steering": v,                               # add alpha*v to h_l to nudge T -> source
            "source": args.source,
            "target": target,
            "model": args.model,
            "n_layers_plus_embed": L_plus_1,
            "hidden_dim": H,
            "per_layer_norm": norms,
        }, out_path)
        # find layer with largest steering magnitude — heuristic for first alpha sweep
        peak_layer = int(torch.tensor(norms).argmax())
        try:
            rel_path = str(out_path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = str(out_path)
        summary["targets"][target] = {
            "path": rel_path,
            "peak_layer": peak_layer,
            "peak_norm": norms[peak_layer],
            "norms_first5": norms[:5],
            "norms_mid": norms[L_plus_1 // 2 - 2 : L_plus_1 // 2 + 3],
            "norms_last5": norms[-5:],
        }
        print(f"  -> {target}: saved {out_path.name}  peak |v| at layer "
              f"{peak_layer} ({norms[peak_layer]:.2f})")

    summary_path = out_dir / "steering_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
