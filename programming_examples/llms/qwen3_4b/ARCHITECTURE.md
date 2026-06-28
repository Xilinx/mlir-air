# Qwen3-4B BF16 Inference — Architecture

Companion to [README.md](README.md). Built kernel-first on the shared LLM infra
(`../shared/`, `../verify/`) and the `../qwen3_0_6b/` (Qwen3 QK-norm + decoupled
O) + `../qwen25_3b/` (large-hidden un-merge) exemplars. The widest+deepest model
in the set: 36 layers, q_dim=4096, hidden=9728. Its prefill has the most NPU
ELFs/layer (**7**) of any model here.

## Model Config

36 layers, emb_dim=2560, n_heads=32, head_dim=128, n_kv_heads=8 (GQA group=4),
q_dim=4096 (decoupled, ≠ emb), kv_dim=1024, hidden_dim=9728, vocab_size=151936,
BF16, rope_theta=1000000, eps=1e-6, tied embeddings, **per-head QK-norm** (no
bias).

Topology: **Qwen3, SwiGLU on NPU, O+FFN un-merged** (hidden=9728 large/non-1024
→ Gate/Up cannot fuse-cast; SwiGLU is its own NPU ELF). head_dim=128 →
head-first FlashAttention.

## Per-Layer Kernel Sequence

**Prefill — 7 NPU ELFs/layer:**

```
x ─[NPU elf:rms_qkv_qknorm_rope]   FUSED, 8 launches, 1 ELF
      { RMSNorm + Q/K/V GEMM + QK-norm(Q) + QK-norm(K) + RoPE-Q + RoPE-K }
      QK-norm = per-head RMSNorm over head_dim=128, eps=1e-6
      → q_roped[seq,4096], k_roped[seq,1024], v[seq,1024]
  ─[NPU elf:flash_attn]   npu_fa_headfirst (head-first, dv_chunks=2, hd=128)
      (HOST) seq→head transpose → NPU FA → (HOST) head→seq transpose → attn_out[seq,4096]
  ─[NPU elf:o_res_norm]   { O GEMM (decoupled 4096→2560) + Add + RMSNorm }
  ─[NPU elf:gate]{Gate GEMM} → [NPU elf:up]{Up GEMM}
  ─[NPU elf:swiglu]       { SwiGLU }   (NPU SwiGLU, tile_n=4864)
  ─[NPU elf:down_add]     { Down GEMM (K=9728, launch 0) + Add } → layer_out[seq,2560]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv] (19 partitions ×8192, vocab 151936)
```

Prefill is 7 NPU dispatches/layer: `rms_qkv_qknorm_rope`, `flash_attn`,
`o_res_norm`, `gate`, `up`, `swiglu`, `down_add` — the most of any model here,
because hidden=9728 forces a full O+FFN un-merge AND SwiGLU is its own ELF.

**Decode — 5 NPU ELFs/layer (+ lm_head once/token):**

```
x ─[NPU elf:rms_qkv_qknorm_rope_gemv]   FUSED, 1 ELF
      { RMSNorm + Q/K/V GEMV + QK-norm(Q,K) + RoPE-Q/K }   (RoPE LUT per-position, NOT static)
  (HOST) KV-cache write → (HOST) decode_attention_cpu
  ─[NPU elf:o_gemv]{O} → (HOST) Add + RMSNorm
  ─[NPU elf:gate_gemv] → [NPU elf:up_gemv] → (HOST) SwiGLU
  ─[NPU elf:down_gemv]{Down}   ← MOVED TO NPU (omit_pingpong=all + dedicated down_mv.o, m_input=2)
  → (HOST) Add → layer_out[2560]
once: (HOST) final RMSNorm → [NPU elf:lm_head_gemv]
```

## NPU vs CPU Mapping

**On NPU (all heavy compute):** every GEMM / GEMV (Q/K/V, O, Gate, Up, Down,
LM-head), RMSNorm, **per-head QK-norm**, RoPE, prefill FlashAttention, **prefill
SwiGLU** (standalone NPU ELF, tile_n=4864 — moved CPU→NPU, the prefill
12.0s→6.06s 2.0× win). **Decode Down GEMV (K=9728) is on NPU** (dedicated
`down_mv.o`, `m_input=2` to keep the push_queue repeat ≤255). All four decode
projections are NPU GEMV ELFs.

**On CPU (cheap glue + one transpose, evidence-backed):**
- **Head-first FA seq↔head transpose** (prefill, hd=128): BF16 DMA stride-1
  requirement + seq-first `dk_chunks>1` upstream FA bug.
- **Decode attention** (single token): NPU FA launch overhead > compute.
- **Decode residual Add / FFN RMSNorm / SwiGLU** (M=1): dispatch > compute.
- **Final RMSNorm / KV-cache write / embed lookup**: single-row dispatch >
  compute.

Decode cannot reach the lean fused 2-ELF cascade: emb=2560 (≥ 2560) needs
n_cascade ≥ 5 cascade stages, but each stage needs one **core row** and the
device has only **4 core rows** → n_cascade ≤ 4 ceiling. So decode stays at 5
standalone GEMV ELFs. Decode is NPU-compute-bound anyway.

## Runtime Flow

```
build_session → prepare_runtime()   ← one-time (~18 s), OUTSIDE timed region
  · load weights, transpose decode GEMV weights, tag per-layer index
  · preload_prefill_weights → static weight BOs; preload decode + LM-head BOs
  ↓
run_once():  prefill (36 layers × 7 ELFs + final RMSNorm + LM-head)   ← TTFT clock
  ↓
generate() decode loop:  per token 36 layers × 5 ELFs + LM-head GEMV  ← TPOT clock
```

`static_input_indices` + per-layer `bo_key` skip every timed weight write
(profile: 0 weight bytes on prefill, 0.0 MB/call on decode). RoPE LUT
position-dependent → NON-static.

## Key Design Patterns / Deltas

- **Per-head QK-norm fused into `rms_qkv_qknorm_rope`** — the Qwen3 delta.
- **Decoupled O-projection** (q_dim=4096 ≠ emb=2560) → O GEMM is 4096→2560, the
  largest non-square O in the set.
- **7-ELF prefill** — the most of any model here: hidden=9728 forces a full
  O+FFN un-merge (Gate/Up cannot fuse-cast at N=9728: the f32-out B-tile DMA hits
  `Stride exceeds [1:1048576]`), plus SwiGLU is its own NPU ELF.
- **head_dim=128 → head-first FlashAttention** (dv_chunks=2) + host transposes.
- **Decode Down GEMV on NPU** (dedicated `down_mv.o`, m_input=2).
- **emb=2560=512×5**, **hidden=9728=512×19** → 512-aligned (not 1024); proj N
  divisible by 4·TILE_N=512 → stock TILE_N=128 HERD_N=4 placement.
- **Multi-launch ELF + text-based MLIR stitching** (shared infra).
