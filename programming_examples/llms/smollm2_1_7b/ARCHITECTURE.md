# SmolLM2-1.7B BF16 Inference — Architecture

Companion to [README.md](README.md). Focuses on the per-layer kernel chain
and how the runtime is organized. Mirrors the reference exemplar
`../llama32_1b/` — deltas called out below.

## Model Config

24 layers, emb_dim=2048, n_heads=32, head_dim=64, **n_kv_heads=32 (pure MHA)**,
hidden_dim=8192, vocab_size=49152, BF16, rope_theta=130000, tied embeddings.

## Deltas vs llama32_1b (the reference exemplar)

| Axis | SmolLM2-1.7B | llama32_1b | Impact |
|---|---|---|---|
| n_layers | 24 | 16 | more layers, same per-layer chain |
| n_kv_heads | **32 (MHA)** | 8 (GQA) | KV cache + attention head layout; no GQA broadcast |
| vocab | 49152 | 128256 | smaller LM-head GEMM/GEMV |
| rope_theta | 130000 | 500000 | RoPE LUT values only (same kernel) |
| emb_dim / head_dim / hidden_dim | 2048 / 64 / 8192 | same | kernel shapes reusable |

The MHA (n_kv_heads == n_heads) delta is the one that changes kernel
*shapes* — Q and KV both have 32 heads, so attention has no GQA broadcast.
This is actually simpler than llama's GQA; the seq-first FA kernel handles
it via the head-count parameter.

## Per-Layer Kernel Sequence (validated Phases 1-3)

```
Prefill (per layer, 3 fused-ELF XRT calls):
  rms_gemms_rope.elf  (RMSNorm + Q/K/V GEMM + RoPE Q/K)
    → flash_attn.elf  (MHA 32q/32kv, seq-first, causal)
    → o_ffn.elf       (O proj + add + RMSNorm + Gate/Up + SwiGLU + Down + add)

Decode (per token per layer, 2 fused-ELF XRT calls + CPU attention):
  rms_gemv_rope.elf → CPU attention (KV cache) → o_gemv_ffn.elf
Final: lm_head_gemv.elf (49152×2048)
```

All shapes verified in Phase 1 against the kernel_registry. The only NEW shapes
vs llama32_1b: FlashAttention 32q/**32kv** (MHA, not 32q/8kv GQA) and LM-head
GEMV 49152×2048 (vocab). Because MHA makes kv_dim==emb_dim, the K/V projection
GEMMs reuse the already-covered 2048×2048×2048 shape.

## The MHA fork (why smollm2_1_7b_prefill.py exists)

llama32_1b's `rms_gemms_rope` builder is registry-driven: each GEMM picks
fused-cast vs drain by shape. For GQA, only Q (N=emb_dim) is fused-cast (needs 1
f32 C-scratch arg); K/V (N=kv_dim=512) are drain. For SmolLM2's MHA,
kv_dim=2048==emb_dim, so K/V ALSO resolve to fused-cast → the builder emits 16
func args (Q,K,V scratch). But the reference Python callers
(`run_transformer_block` AND `preload_prefill_weights`) hardcode the GQA case (1
scratch) — and preload pre-allocates the per-layer BO set, which every later
call reuses by bo_key. Result: the 16-arg ELF ran against 14 BOs → K/V output =
zero → garbage. The fork makes both callers registry-driven
(`_rms_scratch_specs`). GQA-bit-identical; MHA-correct. Decode (GEMV, FP32
accumulate) has no such issue and needed no fork.

## Runtime Flow

```
build_session()            ← one-time: compile/load kernels, load weights,
                             prepare_runtime() (preload per-layer BOs, MHA-safe)
  ↓
run_once() → generate()    ← TTFT-timed: tokenize + EOS-pad + NPU prefill
                             (24 layers × 3 ELFs) + final RMSNorm + LM head
  ↓
decode loop                ← per token: 24 layers × 2 ELFs + CPU attention + LM head
```

## Performance
TTFT ~2.0s, decode ~124 ms/token (8.1 tok/s). `make verify` 2/2 token-set PASS
vs HF bf16. See docs/development_progress/ for full per-phase data.
