# Llama-3.2-3B BF16 Inference — Architecture

Companion to [README.md](README.md). Pure Llama — bit-for-bit the llama32_1b
kernel sequence, only dims + head_dim differ. Built kernel-first on the shared
LLM infra (`../shared/`, `../verify/`) and the `../llama32_1b/` exemplar; the
prefill builders are reused config-driven.

## Model Config

28 layers, emb_dim=3072, n_heads=24, head_dim=128, n_kv_heads=8 (GQA group=3),
q_dim=3072 (square O), kv_dim=1024, hidden_dim=8192, vocab_size=128256, BF16,
rope_theta=500000, eps=1e-5, tied embeddings. **No QK-norm, no bias.**

Topology: **Llama** (RMS+QKV+RoPE already one ELF; O+FFN fused). head_dim=128 →
head-first FlashAttention. All dims 1024-aligned → stock GEMM tiles.

## Per-Layer Kernel Sequence

**Prefill — 3 NPU ELFs/layer:**

```
x ─[NPU elf:rms_gemms_rope]   FUSED, 6 launches, 1 ELF
      { RMSNorm + Q/K/V GEMM + RoPE-Q + RoPE-K }
      → q_roped[seq,3072], k_roped[seq,1024], v[seq,1024]
  ─[NPU elf:flash_attn]   npu_fa_headfirst (head-first, dv_chunks=2, hd=128)
      (HOST) seq→head transpose → NPU FA → (HOST) head→seq transpose → attn_out[seq,3072]
  ─[NPU elf:o_ffn]   FUSED, 8 launches, 1 ELF
      { O GEMM + Add + RMSNorm + Gate + Up + SwiGLU + Down + Add } → layer_out[seq,3072]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv] (8 partitions ×16384, vocab 128256)
```

**Decode — 5 NPU ELFs/layer (+ lm_head once/token):**

```
x ─[NPU elf:rms_gemv_rope]   FUSED, 6 launches, 1 ELF
      { RMSNorm + Q/K/V GEMV + RoPE-Q/K }   (RoPE LUT per-position, NOT static)
  (HOST) KV-cache write → (HOST) decode_attention_cpu (single-token GQA over KV cache)
  ─[NPU elf:o_gemv]{O} → (HOST) Add + RMSNorm
  ─[NPU elf:gate_gemv] → [NPU elf:up_gemv] → (HOST) SwiGLU
  ─[NPU elf:down_gemv]{Down (K=8192, dedicated mv_k8192.o renamed externs)}
  → (HOST) Add → layer_out[3072]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv]
```

## NPU vs CPU Mapping

**On NPU (all heavy compute):** every GEMM / GEMV (Q/K/V, O, Gate, Up, Down,
LM-head), RMSNorm, RoPE, prefill FlashAttention, SwiGLU (fused inside `o_ffn` in
prefill). **All four decode projections (O/Gate/Up/Down) are now standalone NPU
GEMV ELFs** — the "K=3072 BD-1023 limit" that once forced decode O+FFN onto the
CPU was a *fused-cascade* artifact, not a limit on the standalone GEMVs. Moving
them onto NPU lifted decode 0.17 → 4.7 tok/s (~28× cumulative).

**On CPU (cheap glue + one transpose, evidence-backed):**
- **Head-first FA seq↔head transpose** (prefill, hd=128): BF16 DMA stride-1
  requirement (sub-32b types) + seq-first `dk_chunks>1` upstream FA bug. Would
  need an upstream FA fix.
- **Decode attention** (single token): NPU FA launch overhead > compute.
- **Decode residual Add / FFN RMSNorm / SwiGLU** (M=1): dispatch > compute
  (~0.13 ms/layer).
- **Final RMSNorm / KV-cache write / embed lookup**: single-row dispatch >
  compute.

Decode cannot reach the lean fused `o_gemv_ffn` 2-ELF cascade: emb=3072 (≥ 2560)
needs n_cascade ≥ 5 cascade stages, but each stage needs one **core row** and the
device has only **4 core rows** → n_cascade ≤ 4 ceiling. So decode stays at 5
standalone GEMV ELFs. Decode is NPU-compute-bound anyway.

## Runtime Flow

```
build_session → prepare_runtime()   ← one-time, OUTSIDE timed region
  · external kernel compile, weight transpose, tag per-layer index
  · preload_prefill_weights → static weight BOs; preload decode + LM-head BOs
  ↓
run_once():  prefill (28 layers × 3 ELFs + final RMSNorm + LM-head)   ← TTFT clock
  ↓
generate() decode loop:  per token 28 layers × 5 ELFs + LM-head GEMV  ← TPOT clock
```

`static_input_indices` + per-layer `bo_key` skip every timed weight write
(profile: static weight BOs → 0.0 MB written/call on decode). RoPE LUT
position-dependent → NON-static.

## Key Design Patterns / Deltas

- **Pure Llama** — no QK-norm, no bias → inherits the llama32_1b prefill builders
  config-driven (the 3-ELF prefill, the cleanest topology in the set).
- **head_dim=128 → head-first FlashAttention** (dv_chunks=2) + host seq↔head
  transposes — the only Llama delta vs llama32_1b's hd=64 seq-first FA.
- **rope_theta=500000, eps=1e-5** — Llama defaults (NOT Qwen's 1e6 / 1e-6).
- **External kernel rename**: the K=8192 Down GEMV uses `mv_k8192.o` (compiled
  with `-D` renamed symbols) so it can coexist with K=3072 GEMVs in one ELF.
- **Square O-projection** (q_dim=emb_dim=3072); all dims 1024-aligned → stock
  GEMM tiles.
- **Multi-launch ELF + text-based MLIR stitching** (shared infra). Half-split
  RoPE LUT `[cos..., sin...]` matches HF Llama's `(d[i], d[i+head_dim/2])`
  rotation convention.
