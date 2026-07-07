# Llama-3.2-3B BF16 Inference on AMD NPU2 (MLIR-AIR)

End-to-end Llama-3.2-3B (3B parameter, BF16) inference running on AMD NPU2
(AIE2P) hardware via MLIR-AIR. Supports both prefill (seq_len=2048) and
autoregressive decode with a KV cache. Built kernel-first on the shared LLM
infrastructure (`../shared/`, `../verify/`) and the `../llama32_1b/` reference
exemplar — pure Llama (no QK-norm, no bias), so the prefill builders are reused
config-driven; only dims, head_dim, layer count, and eps differ.

## Performance

Measured on NPU2 (AIE2P), `make profile N_TOKENS=32`, 2026-06-28.

| Phase | Measured | Notes |
|-------|----------|-------|
| Prefill / TTFT (2048 tokens) | **3.70 s wall** | head_dim=128 → host head-first FA seq↔head transpose included in wall; NPU-kernel time ~2.85 s |
| Decode / TPOT (steady-state) | **4.7 tok/s** | 28 layers; full decode O+FFN now on NPU |

## Model Config

28 layers, emb_dim=3072, n_heads=24, head_dim=128, n_kv_heads=8 (GQA group=3),
q_dim=3072 (square O), kv_dim=1024, hidden_dim=8192, vocab_size=128256, BF16,
rope_theta=500000, eps=1e-5, tied embeddings (lm_head = embed_tokens).
Pure Llama: **no QK-norm, no bias** (unlike the Qwen forks). Note rope_theta and
eps follow Llama defaults (5e5 / 1e-5), not Qwen's (1e6 / 1e-6).

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2 hardware, Peano compiler, the
   project's standard env: `source utils/env_setup.sh ...`

2. **Extra Python packages** (on top of the base):
   ```bash
   pip install -r requirements.txt
   ```
   Installs `safetensors`, `huggingface_hub`, `transformers`, and `torch`
   (used by `make verify` for the HuggingFace bf16 reference comparison).

3. **HuggingFace model access** (one-time setup):
   - Llama-3.2-3B is **gated** — accept Meta's license:
     - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct  (default; `MODEL=instruct`)
     - https://huggingface.co/meta-llama/Llama-3.2-3B           (`MODEL=base`)
   - Create a read token at https://huggingface.co/settings/tokens and
     authenticate locally: `huggingface-cli login` (or `export HF_TOKEN=<token>`).
   - Weights (~6 GB per model) are auto-downloaded on first `make run` and cached
     under `~/.cache/huggingface/hub/`.

## Quick Start

```bash
# One-time: compile all kernels (cached to disk)
make compile

# Run inference (instruct model by default; up to 1000 tokens, stops early on EOT)
make run

# Run with a custom prompt
make run PROMPT="How does photosynthesis work?"

# Run base (completion) model with a longer context
make run MODEL=base PROMPT="In 1969, the first man to walk on" N_TOKENS=200

# Run with profiling breakdown (prefill + decode kernel tables)
make profile

# Top-k token-level correctness gate (NPU bf16 vs HF transformers bf16,
# 2 prompts × 32 greedy tokens, k=5) — the production-readiness gate
make verify

# Per-layer cosine diagnosis lens (informational, single prompt)
make diagnosis
```

## Verification

`make verify` is the PASS/FAIL gate: it greedily decodes 32 tokens on the NPU
and on HF transformers (bf16) for each prompt and checks that every NPU token is
in HF's top-5 set at that position. Current status: **PASS, exit 0, 2/2 prompts**.

## Key Files

| File | Purpose |
|------|---------|
| `llama32_3b_inference.py` | Unified prefill + decode driver (`prepare_runtime` does all one-time init outside the timed region) |
| `llama32_3b_prefill.py` | Prefill: reuses the llama32_1b builders verbatim (config-driven) |
| `llama32_3b_decode.py` | Decode: NPU rms_gemv_rope + CPU attention + NPU O/Gate/Up/Down GEMVs |
| `llama32_3b_weights.py` | Weight loading from HuggingFace safetensors |
| `llama32_3b_cpu_helpers.py` | NumPy helpers shared by production + verify: `rms_norm`, `attention_reference`, `softmax` |
| `verify_adapter.py` | Hooks this model into the shared `../verify/` subsystem |
| `Makefile` | compile / run / profile / chat / verify / verify-full / diagnosis / clean |
| `ARCHITECTURE.md` | Per-layer kernel sequence, NPU/CPU mapping, runtime flow, deltas |
