# Qwen3-1.7B BF16 Inference — Architecture

Companion to [README.md](README.md). Same Qwen3 topology as the `../qwen3_0_6b/`
sibling (per-head QK-norm fused into the attention-input ELF; O+FFN fused). The
only deltas are size and a **square** O-projection (q_dim=emb_dim=2048).

## Model Config

28 layers, emb_dim=2048, n_heads=16, head_dim=128, n_kv_heads=8 (GQA group=2),
q_dim=2048 (square O), kv_dim=1024, hidden_dim=6144, vocab_size=151936, BF16,
rope_theta=1000000, eps=1e-6, tied embeddings, **per-head QK-norm** (no bias).

Topology: **Qwen3, O+FFN fused** (hidden=6144, all dims 1024-aligned → stock
GEMM tiles, no padding / low-precision tricks). head_dim=128 → head-first
FlashAttention.

## Per-Layer Kernel Sequence

**Prefill — 3 NPU ELFs/layer:**

```
x ─[NPU elf:rms_qkv_qknorm_rope]   FUSED, 8 launches, 1 ELF
      { RMSNorm + Q/K/V GEMM + QK-norm(Q) + QK-norm(K) + RoPE-Q + RoPE-K }
      QK-norm = per-head RMSNorm over head_dim=128, eps=1e-6
      → q_roped[seq,2048], k_roped[seq,1024], v[seq,1024]
  ─[NPU elf:flash_attn]   npu_fa_headfirst (head-first, hd=128)
      (HOST) seq→head transpose → NPU FA → (HOST) head→seq transpose → attn_out[seq,2048]
  ─[NPU elf:o_ffn_qwen]   FUSED, 1 ELF
      { O GEMM (square 2048→2048) + Add + RMSNorm + Gate + Up + SwiGLU + Down + Add } → layer_out[seq,2048]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv] (19 partitions ×8192, vocab 151936)
```

**Decode — 2 NPU ELFs/layer (+ lm_head once/token):**

```
x ─[NPU elf:rms_qkv_qknorm_rope_gemv]   FUSED, 1 ELF
      { RMSNorm + Q/K/V GEMV + QK-norm(Q,K) + RoPE-Q/K }   (RoPE LUT per-position, NOT a static BO)
  (HOST) KV-cache write → (HOST) decode_attention_cpu (single-token GQA over KV cache)
  ─[NPU elf:o_gemv_ffn]   FUSED cascade, 1 ELF
      { O GEMV + Add + RMSNorm + Gate/Up cascade + SwiGLU + Down } → layer_out[2048]
once: (HOST) embed/final RMSNorm → [NPU elf:lm_head_gemv]
```

The fused 2-ELF decode cascade is reachable here because emb=2048 (< 2560) and
hidden=6144 is divisible by 512 — the two limits that wall the bigger models off
this lean form (see NPU vs CPU below).

## NPU vs CPU Mapping

**On NPU (all heavy compute):** every GEMM / GEMV (Q/K/V, O, Gate, Up, Down,
LM-head), RMSNorm, **per-head QK-norm**, RoPE, prefill FlashAttention, SwiGLU
(fused inside `o_ffn_qwen` / `o_gemv_ffn`). Prefill folds Q/K/V + QK-norm + RoPE
into ONE ELF, and the whole O+FFN into a second; attention is the third.

**On CPU (cheap glue + one transpose, evidence-backed):**
- **Head-first FA seq↔head transpose** (prefill, hd=128): BF16 DMA stride-1
  requirement (sub-32b types) + seq-first `dk_chunks>1` upstream FA bug. Would
  need an upstream FA fix.
- **Decode attention** (single token): NPU FA launch overhead > compute.
- **KV-cache write / embed lookup / final RMSNorm**: single-row dispatch >
  compute.

## Runtime Flow

```
build_session → prepare_runtime()   ← one-time, OUTSIDE timed region
  · load weights, transpose decode GEMV weights, tag per-layer index
  · preload_prefill_weights (warm-up XRT call per prefill ELF → static weight BOs)
  · preload decode + LM-head BOs
  ↓
run_once():  prefill (28 layers × 3 ELFs + final RMSNorm + LM-head)   ← TTFT clock
  ↓
generate() decode loop:  per token 28 layers × 2 ELFs + LM-head GEMV  ← TPOT clock
```

`static_input_indices` + per-layer `bo_key` make the timed kernels skip every
weight host→device write. The RoPE LUT is position-dependent → NON-static.

## Key Design Patterns / Deltas

- **Per-head QK-norm fused into `rms_qkv_qknorm_rope`** — the Qwen3 delta
  (nonlinear per-head RMSNorm over head_dim=128, eps=1e-6, between Q/K/V GEMM and
  RoPE; an NPU slice fused into the attention-input ELF).
- **Square O-projection** (q_dim=emb_dim=2048) — vs qwen3_0_6b's decoupled
  2048→1024.
- **All dims 1024-aligned** → stock GEMM tiles, no padding or low-precision
  Gate/Up tier (unlike the Qwen2.5 non-aligned models).
- **O+FFN fused into one ELF** (prefill + decode), so wall-clock is close to NPU
  kernel time.
- **head_dim=128 → head-first FlashAttention** + host seq↔head transposes.
- **Multi-launch ELF + text-based MLIR stitching** (shared infra).
