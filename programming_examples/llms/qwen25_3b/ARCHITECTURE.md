# Qwen2.5-3B BF16 Inference — Architecture

Companion to [README.md](README.md). Built kernel-first on the shared LLM infra
(`../shared/`, `../verify/`) and the `../qwen25_1_5b/` sibling. Largest Qwen2.5
model: 36 layers, hidden=11008. head_dim=128 → head-first FlashAttention;
SwiGLU and the decode Down GEMV both run on NPU.

## Model Config

36 layers, emb_dim=2048, n_heads=16, head_dim=128, n_kv_heads=2 (GQA group=8),
q_dim=2048 (square O), kv_dim=256, hidden_dim=11008, vocab_size=151936, BF16,
rope_theta=1000000, eps=1e-6, tied embeddings, **QKV bias** (no QK-norm).

Topology: **Qwen2.5** (QKV bias fused into the attention-input ELF; SwiGLU on
NPU; decode Down on NPU). head_dim=128 → head-first FA.

## Per-Layer Kernel Sequence

**Prefill — 6 NPU ELFs/layer:**

```
x ─[NPU elf:rms_qkv_bias_rope]   FUSED, 1 ELF
      { RMSNorm + Q/K/V GEMM + bias-add(Q,K,V) + RoPE-Q + RoPE-K }   → q_roped, k_roped, v
  ─[NPU elf:flash_attn]   npu_fa_headfirst (head-first, hd=128)
      (HOST) seq→head transpose → NPU FA → (HOST) head→seq transpose → attn_out[seq,2048]
  ─[NPU elf:o_res_norm]   { O GEMM (square 2048×2048) + Add + RMSNorm }
  ─[NPU elf:gate]{Gate GEMM} → [NPU elf:up]{Up GEMM}
  ─[NPU elf:swiglu]       { SwiGLU }   (NPU SwiGLU, tile_n=4096)
  ─[NPU elf:down_add]     { Down GEMM (launch 0) + Add } → layer_out[seq,2048]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv] (19 partitions ×8192, vocab 151936)
```

Prefill is 6 NPU dispatches/layer: `rms_qkv_bias_rope`, `flash_attn`,
`o_res_norm`, `gate`, `up`, `swiglu`+`down_add`.

**Decode — 5 NPU ELFs/layer (+ lm_head once/token):**

```
x ─[NPU elf:rms_qkv_bias_rope_gemv]   FUSED, 1 ELF
      { RMSNorm + Q/K/V GEMV + bias(Q,K,V) + RoPE-Q/K }   (RoPE LUT per-position, NOT static)
  (HOST) KV-cache write → (HOST) decode_attention_cpu
  ─[NPU elf:o_gemv]{O} → (HOST) Add + RMSNorm
  ─[NPU elf:gate_gemv] → [NPU elf:up_gemv] → (HOST) SwiGLU
  ─[NPU elf:down_gemv]{Down}   ← MOVED TO NPU (omit_pingpong=all + dedicated down_mv.o)
  → (HOST) Add → layer_out[2048]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv]
```

## NPU vs CPU Mapping

**On NPU (all heavy compute):** every GEMM / GEMV (Q/K/V, O, Gate, Up, Down,
LM-head), RMSNorm, **QKV bias-add**, RoPE, prefill FlashAttention, **prefill
SwiGLU** (standalone NPU ELF). **Decode Down GEMV (K=11008) is on NPU** — moved
CPU→NPU with `omit_pingpong=all` + a dedicated `down_mv.o`, the +16% decode win.
All four decode projections are NPU GEMV ELFs.

**On CPU (cheap glue + one transpose, evidence-backed):**
- **Head-first FA seq↔head transpose** (prefill, hd=128): BF16 DMA stride-1
  requirement + seq-first `dk_chunks>1` upstream FA bug.
- **Decode attention** (single token): NPU FA launch overhead > compute.
- **Decode residual Add / FFN RMSNorm / SwiGLU** (M=1): dispatch > compute.
- **Final RMSNorm / KV-cache write / embed lookup**: single-row dispatch >
  compute.

Decode cannot reach the lean fused `o_gemv_ffn` 2-ELF cascade: `matvec_2tile_add`
is numerically correct only at K÷512, and hidden=11008 is not a multiple of 512
→ decode stays at 5 standalone GEMV ELFs (with Down on NPU).

## Runtime Flow

```
build_session → prepare_runtime()   ← one-time, OUTSIDE timed region
  · load weights, transpose decode GEMV weights, tag per-layer index
  · preload_prefill_weights → static weight BOs; preload decode + LM-head BOs
  ↓
run_once():  prefill (36 layers × 6 ELFs + final RMSNorm + LM-head)   ← TTFT clock
  ↓
generate() decode loop:  per token 36 layers × 5 ELFs + LM-head GEMV  ← TPOT clock
```

`static_input_indices` + per-layer `bo_key` skip every timed weight write. RoPE
LUT position-dependent → NON-static.

## Key Design Patterns / Deltas

- **QKV bias fused into `rms_qkv_bias_rope`** — the Qwen2.5 delta.
- **head_dim=128 → head-first FlashAttention** + host seq↔head transposes.
- **Square O-projection** (q_dim=emb_dim=2048).
- **Prefill SwiGLU on NPU as a standalone ELF** (tile_n=4096).
- **Decode Down GEMV on NPU** (K=11008) — the model-specific decode delta
  (qwen25_1_5b keeps it on a generic GEMV ELF; here it needed a dedicated
  `down_mv.o` + `omit_pingpong=all`).
- **Non-aligned dims (hidden=11008, kv_dim=256)** → per-shape GEMM tile configs;
  `down_add` Down must be launch 0.
- **Deepest model** (36 layers, tied with qwen3_4b).
- **Multi-launch ELF + text-based MLIR stitching** (shared infra).
