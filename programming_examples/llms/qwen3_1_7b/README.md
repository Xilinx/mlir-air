# Qwen3-1.7B BF16 Inference on AMD NPU2 (MLIR-AIR)

End-to-end Qwen3-1.7B (1.7B parameter, BF16) inference running on AMD NPU2
(AIE2P) hardware via MLIR-AIR. Supports both prefill (seq_len=2048) and
autoregressive decode with a KV cache. Built kernel-first on the shared LLM
infrastructure (`../shared/`, `../verify/`) and the `../qwen3_0_6b/` sibling
(identical Qwen3 topology); the only deltas are size (emb_dim=2048,
hidden_dim=6144) and a **square** O-projection (q_dim=emb_dim=2048).

## Performance

Measured on NPU2 (AIE2P), `make profile N_TOKENS=32`, 2026-06-28.

| Phase | Measured | Notes |
|-------|----------|-------|
| Prefill / TTFT (2048 tokens) | **2.08 s wall** | head_dim=128 → host head-first FA seq↔head transpose included in wall; NPU-kernel time is lower (~1.82 s) |
| Decode / TPOT (steady-state) | **7.4 tok/s** | 28 layers, NPU-compute-bound; only cheap single-token glue stays on host |

## Model Config

28 layers, emb_dim=2048, n_heads=16, head_dim=128, n_kv_heads=8 (GQA group=2),
q_dim=2048 (**square O**, q_dim==emb_dim), kv_dim=1024, hidden_dim=6144,
vocab_size=151936, BF16, rope_theta=1000000, eps=1e-6, tied embeddings
(lm_head = embed_tokens), **per-head QK-norm** (RMSNorm over head_dim before
RoPE — the key Qwen3 delta vs Llama).

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2 hardware, Peano compiler, the
   project's standard env: `source utils/env_setup.sh ...`

2. **Extra Python packages** (on top of the base):
   ```bash
   pip install -r requirements.txt
   ```
   Installs `safetensors`, `huggingface_hub`, `transformers`, and `torch`
   (used by `make verify` for the HuggingFace bf16 reference comparison).

3. **HuggingFace model access** (one-time):
   - Qwen3-1.7B is openly licensed: https://huggingface.co/Qwen/Qwen3-1.7B
   - Weights are auto-downloaded on the first `make run` and cached under
     `~/.cache/huggingface/hub/`.

## Quick Start

```bash
# One-time: compile all kernels (cached to disk)
make compile

# Run inference (instruct model by default; up to 1000 tokens, stops early on EOT)
make run

# Custom prompt / token budget
make run PROMPT="How does photosynthesis work?" N_TOKENS=64

# Run with profiling breakdown (per-kernel + per-phase)
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
| `qwen3_1_7b_inference.py` | Unified prefill + decode driver (`prepare_runtime` does all one-time init outside the timed region) |
| `qwen3_1_7b_prefill.py` | Prefill kernel builders + `run_transformer_block_qwen3` + `preload_prefill_weights` |
| `qwen3_1_7b_decode.py` | Decode kernel builders + `run_decode_block` (KV cache) |
| `qwen3_1_7b_weights.py` | Weight loading from HuggingFace safetensors (incl. q_norm/k_norm, tied lm_head) |
| `qwen3_1_7b_cpu_helpers.py` | NumPy helpers shared by production + verify: `rms_norm`, `qk_norm_per_head` (Qwen3 delta), `attention_reference`, `softmax` |
| `verify_adapter.py` | Hooks this model's prefill/decode into the shared `../verify/` subsystem |
| `Makefile` | compile / run / profile / chat / verify / verify-full / diagnosis / clean |
| `ARCHITECTURE.md` | Per-layer kernel sequence, NPU/CPU mapping, runtime flow, deltas |
