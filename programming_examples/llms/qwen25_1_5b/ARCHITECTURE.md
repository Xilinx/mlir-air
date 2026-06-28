# Qwen2.5-1.5B BF16 Inference — Architecture

Companion to [README.md](README.md). Built kernel-first on the shared LLM infra
(`../shared/`, `../verify/`) and the `../qwen25_0_5b/` sibling. head_dim=128 →
head-first FlashAttention; large non-aligned hidden=8960 → SwiGLU runs on NPU as
a standalone ELF and the O+FFN path is split.

## Model Config

28 layers, emb_dim=1536, n_heads=12, head_dim=128, n_kv_heads=2 (GQA group=6),
q_dim=1536, kv_dim=256, hidden_dim=8960, vocab_size=151936, BF16,
rope_theta=1000000, eps=1e-6, tied embeddings, **QKV bias** (no QK-norm).

Topology: **Qwen2.5** (QKV bias fused into the attention-input ELF; SwiGLU on
NPU; gate+up fused). head_dim=128 → head-first FA.

## Per-Layer Kernel Sequence

**Prefill — 5 NPU ELFs/layer:**

```
x ─[NPU elf:rms_qkv_bias_rope]   FUSED, 1 ELF
      { RMSNorm + Q/K/V GEMM + bias-add(Q,K,V) + RoPE-Q + RoPE-K }   → q_roped, k_roped, v
  ─[NPU elf:flash_attn]   npu_fa_headfirst (head-first, hd=128)
      (HOST) seq→head transpose → NPU FA → (HOST) head→seq transpose → attn_out[seq,1536]
  ─[NPU elf:o_res_norm]   { O GEMM + Add + RMSNorm }
  ─[NPU elf:gate_up]      { Gate GEMM + Up GEMM }   (gate+up fused into ONE ELF, flag QWEN25_FUSE_GATE_UP)
  ─[NPU elf:swiglu]       { SwiGLU }   (NPU SwiGLU, tile_n=5120)
  ─[NPU elf:down_add]     { Down GEMM (launch 0) + Add } → layer_out[seq,1536]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv] (19 partitions ×8192, vocab 151936)
```

Prefill is 5 NPU dispatches/layer: `rms_qkv_bias_rope`, `flash_attn`,
`o_res_norm`, `gate_up` (fused), `swiglu`+`down_add`. (Without the gate+up fuse
it would be 6 ELFs.)

**Decode — 5 NPU ELFs/layer (+ lm_head once/token):**

```
x ─[NPU elf:rms_qkv_bias_rope_gemv]   FUSED, 1 ELF
      { RMSNorm + Q/K/V GEMV + bias(Q,K,V) + RoPE-Q/K }   (RoPE LUT per-position, NOT static)
  (HOST) KV-cache write → (HOST) decode_attention_cpu
  ─[NPU elf:o_gemv]{O} → (HOST) Add + RMSNorm
  ─[NPU elf:gate_gemv] → [NPU elf:up_gemv] → (HOST) SwiGLU
  ─[NPU elf:down_gemv]{Down} → (HOST) Add → layer_out[1536]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv]
```

## NPU vs CPU Mapping

**On NPU (all heavy compute):** every GEMM / GEMV (Q/K/V, O, Gate, Up, Down,
LM-head), RMSNorm, **QKV bias-add**, RoPE, prefill FlashAttention, **prefill
SwiGLU** (standalone NPU ELF — moved CPU→NPU, the single biggest prefill win:
host SwiGLU was a hidden untimed ~5s cost). All four decode projections are
standalone NPU GEMV ELFs.

**On CPU (cheap glue + one transpose, evidence-backed):**
- **Head-first FA seq↔head transpose** (prefill, hd=128): BF16 DMA stride-1
  requirement + seq-first `dk_chunks>1` upstream FA bug.
- **Decode attention** (single token): NPU FA launch overhead > compute.
- **Decode residual Add / FFN RMSNorm / SwiGLU** (M=1): dispatch > compute
  (~0.13 ms/layer).
- **Final RMSNorm / KV-cache write / embed lookup**: single-row dispatch >
  compute.

Decode cannot reach the lean fused `o_gemv_ffn` 2-ELF cascade: the
`matvec_2tile_add` down-proj is numerically correct only at K÷512, and
hidden=8960 is not a multiple of 512 → it stays at 5 standalone GEMV ELFs.
Decode is NPU-compute-bound anyway.

## Runtime Flow

```
build_session → prepare_runtime()   ← one-time, OUTSIDE timed region
  · load weights, transpose decode GEMV weights, tag per-layer index
  · preload_prefill_weights → static weight BOs; preload decode + LM-head BOs
  ↓
run_once():  prefill (28 layers × 5 ELFs + final RMSNorm + LM-head)   ← TTFT clock
  ↓
generate() decode loop:  per token 28 layers × 5 ELFs + LM-head GEMV  ← TPOT clock
```

`static_input_indices` + per-layer `bo_key` skip every timed weight write. RoPE
LUT position-dependent → NON-static.

## Key Design Patterns / Deltas

- **QKV bias fused into `rms_qkv_bias_rope`** — the Qwen2.5 delta (NPU
  broadcast-add slice inside the attention-input ELF).
- **head_dim=128 → head-first FlashAttention** + host seq↔head transposes.
- **Prefill SwiGLU on NPU as a standalone ELF** (tile_n=5120) — large hidden
  forces O+FFN to split, but SwiGLU stays on-device.
- **gate+up fused** into one ELF (`QWEN25_FUSE_GATE_UP`) → 5 prefill ELFs (vs 6).
- **Non-aligned dims (1536 / 8960 / 256)** → per-shape GEMM tile configs;
  `down_add` Down must be launch 0 of its ELF (else NaN).
- **Multi-launch ELF + text-based MLIR stitching** (shared infra).
