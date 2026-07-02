# Qwen2.5-0.5B BF16 Inference — Architecture

Companion to [README.md](README.md). Built kernel-first on the shared LLM infra
(`../shared/`, `../verify/`) and the `../llama32_1b/` exemplar. The only Qwen2.5
model with head_dim=64, so it uses the **seq-first FlashAttention** ELF (no host
transpose).

## Model Config

24 layers, emb_dim=896, n_heads=14, head_dim=64, n_kv_heads=2 (GQA group=7),
q_dim=896, kv_dim=128, hidden_dim=4864, vocab_size=151936, BF16,
rope_theta=1000000, eps=1e-6, tied embeddings, **QKV bias** (no QK-norm).

Topology: **Qwen2.5** (QKV bias fused into the attention-input ELF). hidden=4864
is non-aligned (not ÷512 cleanly for the fused cascade) → O+FFN splits into 2
ELFs in prefill. head_dim=64 → seq-first FA.

## Per-Layer Kernel Sequence

**Prefill — 4 NPU ELFs/layer:**

```
x ─[NPU elf:rms_qkv_bias_rope]   FUSED, 1 ELF
      { RMSNorm + Q/K/V GEMM + bias-add(Q) + bias-add(K) + bias-add(V) + RoPE-Q + RoPE-K }
      bias-add = out = in + bias[D] (bias broadcast across rows); V gets bias, no RoPE
      (K/V N-padded in the GEMM then sliced for the non-aligned kv_dim=128)
      → q_roped, k_roped, v
  ─[NPU elf:flash_attn]   attn_npu2_seqfirst (SEQ-FIRST, hd=64, NO host transposes) → attn_out[seq,896]
  ─[NPU elf:o_ffn_head]   FUSED, 1 ELF
      { O GEMM + Add + RMSNorm + Gate + Up + SwiGLU }
  ─[NPU elf:down_add]     { Down GEMM (launch 0) + Add } → layer_out[seq,896]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv] (19 partitions ×8192, vocab 151936)
```

**Decode — 5 NPU ELFs/layer (+ lm_head once/token):**

```
x ─[NPU elf:rms_qkv_bias_rope_gemv]   FUSED, 1 ELF
      { RMSNorm + Q/K/V GEMV + bias(Q,K,V) + RoPE-Q/K }   (RoPE LUT per-position, NOT static)
  (HOST) KV-cache write → (HOST) decode_attention_cpu
  ─[NPU elf:o_gemv]{O} → (HOST) Add + RMSNorm
  ─[NPU elf:gate_gemv] → [NPU elf:up_gemv] → (HOST) SwiGLU
  ─[NPU elf:down_gemv]{Down} → (HOST) Add → layer_out[896]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv]
```

## NPU vs CPU Mapping

**On NPU (all heavy compute):** every GEMM / GEMV (Q/K/V, O, Gate, Up, Down,
LM-head), RMSNorm, **QKV bias-add**, RoPE, prefill FlashAttention (seq-first),
and prefill SwiGLU (fused inside `o_ffn_head`). All four decode projections
(O/Gate/Up/Down) are standalone NPU GEMV ELFs.

**On CPU (cheap single-token glue, evidence-backed):**
- **Decode attention** (single token): NPU FA launch overhead > compute.
- **Decode residual Add / FFN RMSNorm / SwiGLU** (M=1): dispatch > compute
  (~0.13 ms/layer).
- **Final RMSNorm / KV-cache write / embed lookup**: single-row dispatch >
  compute.

The decode O+FFN cannot collapse into the fused `o_gemv_ffn` 2-ELF cascade: the
`matvec_2tile_add` down-proj primitive is numerically correct only at K÷512, and
K=896/4864 are not multiples of 512. So decode stays at 5 standalone GEMV ELFs.
Decode is NPU-compute-bound anyway (host glue ~0.13 ms/layer), so little
headroom is lost.

## Runtime Flow

```
build_session → prepare_runtime()   ← one-time, OUTSIDE timed region
  · load weights, transpose decode GEMV weights, tag per-layer index
  · preload_prefill_weights (warm-up XRT call per prefill ELF → static weight BOs)
  · preload decode + LM-head BOs
  ↓
run_once():  prefill (24 layers × 4 ELFs + final RMSNorm + LM-head)   ← TTFT clock
  ↓
generate() decode loop:  per token 24 layers × 5 ELFs + LM-head GEMV  ← TPOT clock
```

`static_input_indices` + per-layer `bo_key` skip every timed weight write. RoPE
LUT is position-dependent → NON-static.

## Key Design Patterns / Deltas

- **QKV bias fused into `rms_qkv_bias_rope`** — the Qwen2.5 delta. The bias-add
  is an NPU broadcast-add slice fused into the attention-input ELF (Q/K bias
  before RoPE; V bias, no RoPE).
- **head_dim=64 → seq-first FlashAttention** (`attn_npu2_seqfirst`): the only
  model here with NO host seq↔head transpose.
- **Non-aligned dims (896 / 4864 / 128)**: per-shape GEMM tile configs (shrink
  TILE_N, keep HERD_N=4); K/V N-padded then sliced.
- **`down_add` split out** (K=4864 Down must be launch 0 of its ELF, else NaN) →
  prefill O+FFN is 2 ELFs.
- **Multi-launch ELF + text-based MLIR stitching** (shared infra).
