# How Far Can a Single Vector Carry a Language?

**Mechanistic Limits of Inference-Time Steering for Low-Resource Devanagari Languages**

Sumit Yadav, Santosh Giri, Ganesh Gautam — IOE, Pulchowk Campus, Tribhuvan University

**Project page:** <https://rockerritesh.github.io/devanagari-steering/> ·
**Preprint:** [doi.org/10.21203/rs.3.rs-10978527/v1](https://doi.org/10.21203/rs.3.rs-10978527/v1)

Code for the paper (under review at *Machine Learning*, Springer). We steer two 8B
multilingual decoders (Llama-3.1-8B-Instruct, Aya-23-8B) from **Hindi** toward its
lower-resource Devanagari sister languages (**Nepali, Maithili, Bhojpuri**) with a
single inference-time direction — no fine-tuning, no new data, no weight updates —
and use steering as a *causal probe* of where language identity is manipulable.

**Method in one line:** collect per-language activation memory banks, form the
centroid-difference vector `v_ℓ = μ_src(ℓ) − μ_tgt(ℓ)` at layer ℓ, and add `α·v_ℓ`
to the residual stream at every token position during generation.

## Setup

```bash
uv sync                      # Python 3.10–3.12; installs torch, transformers, ...
export HF_TOKEN=...          # gated models (Llama-3.1)
export GEMINI_API_KEY=...    # LLM-judge panel
export OPENAI_API_KEY=...    # LLM-judge panel (gpt judge)
```

Steered generation runs on a single 16 GB GPU (T4) with NF4 quantization; a bf16
geometry replication is included. Analysis/plot scripts run on CPU/MPS.

## Pipeline

Scripts are in `code/`, ordered by stage. Each writes its outputs under
`data/` or `results/` (paths at the top of each file).

### 1. Corpus
| script | what it does |
|---|---|
| `build_parallel_corpus.py` | build the 250-passage Hindi↔{Nepali, Maithili, Bhojpuri} parallel corpus (3-LLM consensus translation) |
| `translate_corpus_to_en.py` | add the English side (for the source-language control) |
| `build_eval_prompts.py` | build the 200 held-out Hindi/English evaluation prompts |

### 2. Activations → steering vectors
| script | what it does |
|---|---|
| `extract_activations.py` | run the corpus through a model, save per-layer hidden states (memory banks) |
| `compute_steering.py` | centroid-difference steering vectors `v_ℓ` per target language |
| `analyze_centroids.py` | layer-wise geometry: silhouettes, per-layer probes, centroid distances/cosines |
| `compare_bf16_geometry.py` | bf16 vs NF4 replication of the geometry |

### 3. Steered generation (and controls)
| script | what it does |
|---|---|
| `run_steered_eval.py` | main α-sweep steered generation at the chosen layer (Llama L20, Aya L22) |
| `steered_inference_local.py` | the generation hook itself (add `α·v_ℓ` at all positions) |
| `run_multilayer_eval.py` | multi-layer composites and the depth-trajectory control (`--layers 8..20`) |
| `run_random_baseline.py` | matched-magnitude random-direction control |
| `run_icl_baseline.py` | few-shot in-context-learning baseline (competence control) |

### 4. Evaluation
| script | what it does |
|---|---|
| `multi_judge.py` | 3-judge cross-family LLM panel (Gemini Pro / GPT / Gemini Flash), per-judge checkpoints |
| `eval_judge.py` | single-judge scoring harness (5-dim rubric) |
| `judge_agreement.py` | inter-judge reliability: Krippendorff α, ICC(2,k), Gwet AC1, weighted κ |
| `judge_sensitivity.py` | leave-one-judge-out and family-balanced panel robustness |
| `robust_metrics.py` | reference-free metrics: LaBSE semantic preservation, IndicBERT pseudo-perplexity, code-mixing indices (GlotLID), degeneracy |
| `convergent_validity.py` | per-generation correlations between judge scores and automatic metrics |
| `panel_headline.py`, `robust_metrics_stats.py`, `traj_stats.py`, `analyze_finer30.py` | aggregate the above into the paper's numbers (LaTeX macros) |

### 5. Figures and paper statistics
`visualize_*.py` and `plot_*.py` reproduce every figure; `review_offline_analyses.py`,
`review_round2_analyses.py`, and `make_examples_table.py` reproduce the remaining
statistics and the qualitative-examples table.

## Data availability

The four-way parallel Devanagari corpus is **machine-translated (synthetic — not
native-speaker-validated)** and will be public at
[`rockerritesh/devanagari-parallel-250`](https://huggingface.co/datasets/rockerritesh/devanagari-parallel-250)
upon acceptance; until then it is available from the corresponding author on
reasonable request. Scored generations and memory banks are likewise available on
request (they are large).

## Citation

```bibtex
@article{yadav2026singlevector,
  title  = {How Far Can a Single Vector Carry a Language? Mechanistic Limits of
            Inference-Time Steering for Low-Resource Devanagari Languages},
  author = {Yadav, Sumit and Giri, Santosh and Gautam, Ganesh},
  note   = {Preprint (under review at Machine Learning, Springer)},
  doi    = {10.21203/rs.3.rs-10978527/v1},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`). The bundled corpus/model licenses are those of their upstream
sources (Llama 3.1 Community License, CC-BY-NC for Aya weights, OFL for fonts).
