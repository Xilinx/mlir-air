---
name: phase-0-build-cpu-reference
description: Phase 0 of LLM deployment — produce `<model>_weights.py` (HF weight loader) and `<model>_cpu_helpers.py` (the few NumPy helpers production prefill/decode import), then confirm the HF bf16 reference baseline loads and runs via the shared `llms/verify/` subsystem's HfRunner. Downstream phases compare NPU against HF transformers in bf16 directly; there is no hand-written full-model FP32 oracle.
---

## Purpose

Phase 0 establishes the inputs every downstream phase needs. It does NOT
build a full-model CPU oracle — the reference is HuggingFace transformers
in **bf16**, accessed through the `verify/` subsystem's `HfRunner`. Phase 0
produces three things:

1. **`<model>_weights.py`** — Config dataclass + HF weight loader. Maps HF
   safetensors names to the per-layer `LayerWeights` the NPU pipeline
   consumes.
2. **`<model>_cpu_helpers.py`** — the small set of NumPy helpers that
   *production* prefill/decode import (default: `rms_norm`,
   `attention_reference`, `softmax`). These are NOT a per-kernel oracle
   catalog — each leaf kernel ships its own NumPy reference inside its
   `llama_kernel_builder/<kernel>/run.py` harness (Phase 1 uses those).
3. **HF bf16 baseline confirmation** — verify the target model loads with
   `torch_dtype=torch.bfloat16` and runs the canonical prompt through
   `HfRunner`, producing a sane top-1 and non-degenerate logits. This is
   the Phase 0 gate.

### Why HF bf16 directly, and what still needs a NumPy helper

The reference is HF transformers in bf16 — same dtype as the NPU, so
NPU-vs-reference is a fair fight (bf16 vs bf16), and there is no
hand-written 480-line full-model forward to keep correct. HF's per-layer
hidden states (used by Phase 2/3) are captured by `HfRunner` in diagnosis
mode (`lite_mode=False`), which returns `layer_intermediates[].ffn_out`
plus `final_hidden_normed`.

NumPy helpers are still required for two narrow cases that HF (a black box
that only exposes end-to-end forward + per-layer hidden states) cannot
serve:

1. **Per-kernel references (Phase 1)**: HF cannot expose a single kernel's
   output (RMSNorm alone, RoPE alone, etc.). Each leaf kernel's standalone
   harness (`llama_kernel_builder/<kernel>/run.py`) carries its own NumPy F32
   reference for that kernel. `<model>_cpu_helpers.py` only holds the
   helpers production code itself imports at runtime.
2. **CPU fallbacks (production)**: e.g. prefill `cpu_attn=True` uses
   `attention_reference` when the NPU FlashAttention kernel is unavailable
   for the configured head_dim; the LM-head final norm uses `rms_norm`.

## Phase 0 PASS criteria (HARD GATES)

Three checks, each catching a different bug class:

1. **Weights loadable** (catches weight-name / shape mismatches):
   `<model>_weights.py` loads every expected tensor; every layer index
   0..n_layers-1 has q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
   down_proj, attn_norm, ffn_norm (plus q/k/v bias if `qkv_bias=true`).
2. **HF baseline runnable** (catches HF auth / config / dtype issues):
   `HfRunner(model_name, config, max_seq, lite_mode=True).prefill(tokens)`
   on the canonical prompt runs without error, returns a sane top-1 token,
   and produces logits with no NaN and a non-degenerate distribution.
3. **Config consistency** (catches config-extraction errors): every field
   extracted in Step 1 (n_layers, emb_dim, n_heads, n_kv_heads, head_dim,
   hidden_dim, vocab_size, rope_theta, tie_word_embeddings) matches HF's
   `config.json` exactly.

The per-layer and final-logits cosine checks that used to live in Phase 0
now live in **Phase 2 (single-block) and Phase 3 (full-model)**, comparing
NPU against HF bf16 intermediates via the `verify/` diagnosis path. Phase 0
only confirms the HF baseline and helpers/weights are in place.

## Knowledge base references

Read these BEFORE acting:

- `programming_examples/llms/llama32_1b/llama32_1b_weights.py` — reference Config
  dataclass + HF weight loading pattern to copy.
- `programming_examples/llms/llama32_1b/llama32_1b_cpu_helpers.py` — the canonical
  small NumPy helper file; mirror its scope (only production-imported
  helpers), not a per-kernel catalog.
- `programming_examples/llms/verify/runners/hf_runner.py` — the bf16
  HF reference runner; this is how Phase 0 confirms the baseline and how
  Phase 2/3 obtain per-layer intermediates.
- `programming_examples/llms/verify/README.md` — the verify subsystem
  methodology (HF bf16 reference, top-k token-set gate, cosine as
  diagnosis).

## Workflow

### Step 1: Read the HF config

Fetch `config.json` for the target model from HuggingFace. Extract:

- `num_hidden_layers` → `n_layers`
- `hidden_size` → `emb_dim`
- `num_attention_heads` → `n_heads`
- `num_key_value_heads` → `n_kv_heads` (default to `n_heads` if absent → MHA)
- `intermediate_size` → `hidden_dim`
- `vocab_size` → `vocab_size`
- `rope_theta` → `rope_base` (default 10000.0 if absent)
- `head_dim` (compute as `emb_dim // n_heads` if absent)
- `tie_word_embeddings` (affects whether to load `lm_head.weight`)

### Step 2: Architecture compatibility check

Confirm the model is in-scope before scaffolding anything downstream:

- Architecture must be in `["LlamaForCausalLM", "MistralForCausalLM"
  (only if no sliding window), "Qwen2ForCausalLM", "Qwen3ForCausalLM"]`
  — i.e., a decoder-only with RMSNorm + SwiGLU + RoPE + GQA/MHA
- Reject if: MoE layers, sliding-window attention, MLA, encoder-decoder
- Reject explicitly with a clear message; don't scaffold a model the
  rest of the pipeline can't handle

### Step 3: Generate `<model>_weights.py`

Copy `programming_examples/llms/llama32_1b/llama32_1b_weights.py` to
`<model>/<model>_weights.py`. Modify:

- `LlamaConfig` dataclass defaults → match Step 1 values
- HF weight name remapping in `load_weights()` — most LLAMA-derived
  models share the same names (`model.layers.<i>.self_attn.q_proj.weight`
  etc.); confirm via inspecting the safetensors index. If different,
  write an explicit mapping.
- `generate_rope_lut()` — verify `rope_base` is parameterized and uses
  the new value
- LM head: load `lm_head.weight` only if `tie_word_embeddings` is False
- **If `qkv_bias=true` (Qwen2 / Qwen3 family)**: add `bq, bk, bv`
  fields to `LayerWeights`, parallel to wq/wk/wv. The bias is
  loaded just like the projection weights but with a different HF key
  (`q_proj.bias` etc.). The bias is applied on the HOST around the
  bias-free NPU kernels (exploiting RoPE linearity:
  `RoPE(q + bq) = RoPE(q) + RoPE(bq)`); the application detail belongs to
  Phase 2 (`phase-2-single-block-validation` Step 2) — surface it in TODO.md as a
  Phase 2 prerequisite. Re-derive the bias-on-host wrapper from the HF
  reference impl.

### Step 4: Generate `<model>_cpu_helpers.py`

Copy `programming_examples/llms/llama32_1b/llama32_1b_cpu_helpers.py` to
`<model>/<model>_cpu_helpers.py`. Keep ONLY the helpers your model's
production prefill/decode actually import:

- `rms_norm` — almost always needed (LM-head final norm in inference).
- `attention_reference` — needed if prefill exposes a `cpu_attn=True`
  fallback (GQA attention in F32 on host).
- `softmax` — keep only if `attention_reference` is kept.

Rule for what belongs here: a helper goes in `<model>_cpu_helpers.py` ONLY
if production code imports it at runtime. Per-kernel verification references
do NOT go here — they live in each `llama_kernel_builder/<kernel>/run.py`. If a
later phase (4/5) promotes a new CPU fallback op, add its helper then, not
preemptively.

Modify:

- Imports / config references → use the new model's names and config.
- For unfamiliar architectures (different attention masking, post-norm vs
  pre-norm) — adapt `attention_reference` carefully and cross-check against
  HF's `modeling_<arch>.py` for the exact computation order.

### Step 5: Confirm the HF bf16 baseline (HARD GATE)

Confirm the reference baseline using the `verify/` subsystem's `HfRunner`,
NOT a hand-written full-model comparison. Minimal confirmation:

```python
# from programming_examples/llms/<model>/, with the shared llms/verify/ reachable via verify_adapter
from verify.runners.hf_runner import HfRunner
from <model>_weights import LlamaConfig  # or your Config class

config = LlamaConfig()                     # Step-1 values
runner = HfRunner(hf_model_id, config, max_seq=64, lite_mode=True)
rec = runner.prefill(tokenize(canonical_prompt))
# Assert: rec.top1_token is a sane token; rec.logits_at_pred has no NaN;
# config fields match HF config.json.
```

**Canonical prompt**: use one of the `verify/prompts/{base,instruct}.txt`
prompts (keep Phase 0 consistent with the gate the later phases run). For a
base model use `base.txt`; for an instruct model use `instruct.txt`.

PASS = all three §"Phase 0 PASS criteria" gates hold. If any fails, see
"Failure modes".

(The per-layer cosine sanity that HF can provide via `lite_mode=False` is
not run here — Phase 2/3 own that comparison against NPU output.)

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| HF model won't load | Missing `transformers`/`torch`, or HF auth needed for gated model | `pip install -r requirements.txt`; for gated models `huggingface-cli login` |
| Weights load raises KeyError / shape mismatch | HF weight names differ from the llama32_1b remap, or `head_dim`/`n_kv_heads` wrong | Inspect the safetensors index; print weight shapes after load; re-check Step 1 config |
| Config field mismatch vs HF config.json | Step 1 extraction error (e.g. `head_dim` defaulted wrong, `tie_word_embeddings` missed) | Re-read `config.json`; do not assume defaults |
| `HfRunner.prefill` returns NaN logits | dtype/precision issue in HF load, or a corrupt download | Re-download the HF snapshot; confirm `torch_dtype=bfloat16` path |
| top-1 token looks nonsensical | tokenizer mismatch (wrong chat template / BOS handling) | Confirm tokenizer matches the model; check base vs instruct prompt set |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

(Per-layer cosine drops — RoPE convention, norm order, KV layout — are no
longer a Phase 0 concern; they surface in Phase 2/3 when NPU output is
compared against HF bf16 intermediates. The debug recipes live there.)

## Update protocol

On Phase 0 PASS, append a brief Phase 0 entry to
`<model>/docs/development_progress/progress.md` recording: HF model id,
resolved config (n_layers / emb_dim / n_heads / n_kv_heads / head_dim /
hidden_dim / vocab / rope_base / tie_word_embeddings), which `cpu_helpers`
were kept, and the HF baseline confirmation result (top-1 token on the
canonical prompt, logits OK).

Mark Phase 0 in `<model>/TODO.md`.

`<model>_weights.py` and `<model>_cpu_helpers.py` are now the stable inputs
for Phases 1-6. The correctness oracle for downstream cosine/token checks
is HF transformers bf16 via the `verify/` subsystem — there is no
hand-written reference file to keep in sync.
