# SmolLM2-1.7B BF16 Inference on NPU2 (MLIR-AIR)

End-to-end decoder-only LLM inference for `HuggingFaceTB/SmolLM2-1.7B` on AMD
NPU2 (Strix, AIE2P), BF16, validated against the HuggingFace bf16 reference.

## Quick start

```bash
make compile                  # compile all kernels (one-time, cached)
make run                      # NPU prefill + decode, prints TTFT + TPS
make run MODEL=base PROMPT="The capital of France is" N_TOKENS=50
make verify                   # top-5 token-set gate vs HF bf16 (production gate)
make profile                  # per-phase + per-kernel breakdown
```

## Performance (NPU2, seq_len=2048, 24 layers)

| Metric | Value |
|---|---|
| TTFT (prefill + LM head) | ~2.0 s |
| Decode | ~124 ms/token (8.1 tokens/sec) |
| Prefill kernel | ~1.70 s (rms_gemms_rope 11.5 + flash_attn 22.0 + o_ffn 37.2 ms/layer) |

## Correctness

`make verify` PASSES the top-5 token-set inclusion gate vs HF transformers bf16
(2 prompts × 32 tokens). Per-layer cosine vs HF bf16 ≥ 0.998 across all 24
layers (`make diagnosis`).

## Model config

24 layers, emb_dim=2048, n_heads=32, **n_kv_heads=32 (pure MHA)**, head_dim=64,
hidden_dim=8192, vocab=49152, rope_theta=130000, tied embeddings.

## Relationship to llama32_1b

SmolLM2 is a bit-for-bit llama kernel sequence, so it reuses the `llama32_1b`
prefill/decode/inference machinery directly via Python imports — no SmolLM2
fork. The one architectural delta is pure MHA vs llama's GQA: at
kv_dim==emb_dim the K/V GEMMs also resolve to fused-cast and need their own f32
C-scratch args. The shared `llama32_1b_prefill.{run_transformer_block,
preload_prefill_weights}` allocate those scratch args registry-driven (querying
`gemm_registry_config` per shape), so they are correct for both GQA (1 scratch)
and MHA (3 scratch) with no model-specific code. See
[ARCHITECTURE.md](ARCHITECTURE.md).

## File map

| File | Role |
|---|---|
| `smollm2_1_7b_inference.py` | production runner (setup → prefill → decode); reuses `llama32_1b_inference` Session + loops |
| `smollm2_1_7b_weights.py` | HF weight loader + Config + RoPE LUT |
| `smollm2_1_7b_cpu_helpers.py` | rms_norm / attention_reference / softmax |
| `verify_adapter.py` | hooks into shared `../verify/` subsystem |
| `Makefile` | run / verify / verify-full / diagnosis / profile / compile / clean |
