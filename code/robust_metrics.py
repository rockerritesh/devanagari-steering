"""Thread 2: robust, reference-free automatic metrics for the steered generations.

Complements the LLM-judge panel with three tiers of automatic metrics, all
reference-free and computed from the generation TEXT (so they are independent of any
judge). The generation text is identical across the single-judge and panel parquets;
we read the raw *_scored / *_panel parquet only for its text + (eval_id, target_lang,
alpha) keys.

Tiers (select with --tiers):
  degeneracy : distinct-1/2/3, repetition rate, self-BLEU (per cell). No deps.
  semantic   : meaning preservation vs the alpha=0 (Hindi) control answer, cosine in a
               multilingual sentence-embedding space (LaBSE; optional 2nd encoder).
               High cosine = the steered answer keeps the control's meaning while the
               *language* changes -> evidence the shift is real language transfer, not
               gibberish-in-script.
  ppl        : native-LM fluency FLOOR = pseudo-perplexity under IndicBERT v2
               (masked-LM scoring; covers Nepali, Maithili AND Bhojpuri). Lower = more
               native-like. This is the "fluency is the floor" signal.
  codemix    : Code-Mixing Index (CMI), M-index, I-index from per-token language tags
               (GlotLID/fastText). Quantifies the Hindi<->target mixing. NOTE: per-word
               LID on closely-related Devanagari sisters is noisy; reported with that
               caveat. Requires fasttext (numpy<2) + a GlotLID model.

Device auto-selects cuda -> mps -> cpu.

    .venv/bin/python code/robust_metrics.py \
        --in results/judge_panel/llama_finer30_panel.parquet \
        --out results/robust_metrics/llama_finer30_metrics.parquet \
        --tiers degeneracy,semantic,ppl
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEVAN = re.compile(r"[ऀ-ॿ]+")


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def words(s: str) -> list[str]:
    """Whitespace tokens that contain Devanagari (drops pure punctuation/latin)."""
    return [w for w in str(s).split() if DEVAN.search(w)]


# --------------------------------------------------------------------------- #
# Tier: degeneracy (no deps)
# --------------------------------------------------------------------------- #
def distinct_n(ws, n):
    if len(ws) < n:
        return float("nan")
    grams = list(zip(*[ws[i:] for i in range(n)]))
    return len(set(grams)) / max(1, len(grams))


def rep_rate(ws):
    if len(ws) < 2:
        return float("nan")
    return sum(a == b for a, b in zip(ws, ws[1:])) / (len(ws) - 1)


def _bleu2(hyp, refs):
    """Cheap BLEU-ish: mean of unigram+bigram precision of hyp against a set of refs."""
    if len(hyp) < 2:
        return float("nan")
    best = 0.0
    for r in refs:
        if len(r) < 2:
            continue
        p1 = _prec(hyp, r, 1)
        p2 = _prec(hyp, r, 2)
        best = max(best, math.sqrt(max(p1, 1e-9) * max(p2, 1e-9)))
    return best


def _prec(hyp, ref, n):
    hg = Counter(zip(*[hyp[i:] for i in range(n)]))
    rg = Counter(zip(*[ref[i:] for i in range(n)]))
    if not hg:
        return 0.0
    overlap = sum(min(c, rg[g]) for g, c in hg.items())
    return overlap / sum(hg.values())


def tier_degeneracy(df):
    ws = df["generation"].map(words)
    df["distinct1"] = ws.map(lambda w: distinct_n(w, 1))
    df["distinct2"] = ws.map(lambda w: distinct_n(w, 2))
    df["distinct3"] = ws.map(lambda w: distinct_n(w, 3))
    df["rep_rate"] = ws.map(rep_rate)
    # self-BLEU per (target_lang, alpha) cell: each gen vs the rest of its cell
    df["self_bleu"] = np.nan
    for _, idx in df.groupby(["target_lang", "alpha"]).groups.items():
        cell = [words(df.loc[i, "generation"]) for i in idx]
        for j, i in enumerate(idx):
            refs = cell[:j] + cell[j + 1:]
            df.loc[i, "self_bleu"] = _bleu2(cell[j], refs)
    return df


# --------------------------------------------------------------------------- #
# Tier: semantic preservation (LaBSE etc.)
# --------------------------------------------------------------------------- #
def tier_semantic(df, models, device):
    """Cosine(control, steered) in a multilingual sentence space, via transformers
    directly (no sentence-transformers dep). For LaBSE the sentence embedding is the
    L2-normalized pooler output (dense+tanh on CLS), matching the ST pipeline."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    ctrl = (df[df.alpha == 0.0]
            .set_index(["target_lang", "eval_id"])["generation"].to_dict())
    keys = list(zip(df["target_lang"], df["eval_id"]))
    ctrl_text = [ctrl.get(k, "") for k in keys]

    def embed(texts, name):
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModel.from_pretrained(name).to(device).eval()
        outs = []
        with torch.inference_mode():
            for i in range(0, len(texts), 64):
                b = [t if str(t).strip() else " " for t in texts[i:i + 64]]
                enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                          max_length=128).to(device)
                o = mdl(**enc)
                emb = o.pooler_output if getattr(o, "pooler_output", None) is not None \
                    else o.last_hidden_state[:, 0]
                emb = torch.nn.functional.normalize(emb.float(), dim=-1)
                outs.append(emb.cpu().numpy())
        del mdl
        return np.concatenate(outs)

    for tag, name in models.items():
        g = embed(df["generation"].astype(str).tolist(), name)
        c = embed(ctrl_text, name)
        cos = (g * c).sum(1)
        cos[[not bool(str(t).strip()) for t in ctrl_text]] = np.nan
        df[f"sem_{tag}"] = cos
    return df


# --------------------------------------------------------------------------- #
# Tier: fluency floor = IndicBERT v2 pseudo-perplexity
# --------------------------------------------------------------------------- #
def tier_ppl(df, model_name, device, max_tok=96):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    mask_id = tok.mask_token_id

    @torch.inference_mode()
    def pll(text, chunk=16):
        ids = tok(str(text), return_tensors="pt", truncation=True,
                  max_length=max_tok)["input_ids"][0]
        if ids.numel() < 3:
            return float("nan")
        special = set(tok.all_special_ids)
        pos = [i for i in range(ids.numel()) if int(ids[i]) not in special]
        if not pos:
            return float("nan")
        tot = 0.0
        # Chunk the masked positions to cap memory: only materialize the
        # masked-position logits (c, V), never the full (P, L, V) tensor.
        for s in range(0, len(pos), chunk):
            cpos = pos[s:s + chunk]
            batch = ids.unsqueeze(0).repeat(len(cpos), 1).clone()
            for r, p in enumerate(cpos):
                batch[r, p] = mask_id
            logits = model(batch.to(device)).logits                 # (c, L, V)
            rows = torch.arange(len(cpos))
            ml = logits[rows, torch.tensor(cpos), :].float()        # (c, V)
            lp = torch.log_softmax(ml, dim=-1)
            tgt = torch.tensor([int(ids[p]) for p in cpos], device=lp.device)
            tot += lp[rows, tgt].sum().item()
            del logits, ml, lp
        return math.exp(-tot / len(pos))                            # pseudo-perplexity

    df["indicbert_ppl"] = df["generation"].map(pll)
    del model
    return df


# --------------------------------------------------------------------------- #
# Tier: code-mixing indices (CMI, M-index, I-index) via per-token LID
# --------------------------------------------------------------------------- #
def tier_codemix(df, glotlid_path):
    import fasttext
    import fasttext.FastText as _ft
    import numpy as _np
    # fasttext-wheel calls np.array(probs, copy=False), which numpy>=2 rejects.
    # Patch only fasttext's np reference to use asarray (copy allowed).
    _orig = _ft.np.array
    def _safe_array(obj, *a, **k):
        k.pop("copy", None)
        return _np.asarray(obj)
    _ft.np.array = _safe_array

    lid = fasttext.load_model(glotlid_path)
    keep = {"hin", "npi", "mai", "bho"}

    def tag(tokens):
        labs = []
        for w in tokens:
            lab = lid.predict(w.replace("\n", " "))[0][0]       # __label__xxx_Deva
            code = lab.replace("__label__", "").split("_")[0]
            labs.append(code if code in keep else "oth")
        return labs

    def indices(tokens):
        labs = [l for l in tag(tokens) if l != "oth"]
        n = len(labs)
        if n < 2:
            return (float("nan"),) * 3
        counts = Counter(labs)
        mx = max(counts.values())
        cmi = 100.0 * (1 - mx / n)                              # % non-matrix tokens
        ps = [c / n for c in counts.values()]
        k = len(counts)
        s2 = sum(p * p for p in ps)
        m_index = (1 - s2) / ((k - 1) * s2) if k > 1 and s2 > 0 else 0.0
        switches = sum(labs[i] != labs[i + 1] for i in range(n - 1))
        i_index = switches / (n - 1)
        return cmi, m_index, i_index

    res = df["generation"].map(lambda s: indices(words(s)))
    df["cmi"] = [r[0] for r in res]
    df["m_index"] = [r[1] for r in res]
    df["i_index"] = [r[2] for r in res]
    return df


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--tiers", default="degeneracy,semantic,ppl",
                    help="comma list: degeneracy,semantic,ppl,codemix")
    ap.add_argument("--labse", default="sentence-transformers/LaBSE")
    ap.add_argument("--enc2", default="", help="optional 2nd sentence encoder id")
    ap.add_argument("--indicbert", default="ai4bharat/IndicBERTv2-MLM-only")
    ap.add_argument("--glotlid", default=str(ROOT / "models" / "glotlid" / "model.bin"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    tiers = set(t.strip() for t in args.tiers.split(","))
    df = pd.read_parquet(args.in_path)
    need = {"generation", "target_lang", "alpha", "eval_id"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"input missing columns: {miss}")
    df = df.reset_index(drop=True)
    dev = args.device or pick_device()
    print(f"Loaded {len(df)} rows; device={dev}; tiers={sorted(tiers)}")

    if "degeneracy" in tiers:
        print("  [degeneracy] distinct-n / rep-rate / self-BLEU ...")
        df = tier_degeneracy(df)
    if "semantic" in tiers:
        models = {"labse": args.labse}
        if args.enc2:
            models["enc2"] = args.enc2
        print(f"  [semantic] {list(models.values())} on {dev} ...")
        df = tier_semantic(df, models, dev)
    if "ppl" in tiers:
        print(f"  [ppl] {args.indicbert} pseudo-PPL on {dev} ...")
        df = tier_ppl(df, args.indicbert, dev)
    if "codemix" in tiers:
        print(f"  [codemix] GlotLID {args.glotlid} ...")
        df = tier_codemix(df, args.glotlid)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"wrote {args.out_path} ({len(df)} rows)")

    # quick console summary of the new metric columns by target/alpha peak region
    newcols = [c for c in ["distinct2", "rep_rate", "self_bleu", "sem_labse",
                           "sem_enc2", "indicbert_ppl", "cmi", "m_index", "i_index"]
               if c in df.columns]
    if newcols:
        agg = df.groupby("target_lang")[newcols].mean().round(3)
        print(agg.to_string())


if __name__ == "__main__":
    main()
