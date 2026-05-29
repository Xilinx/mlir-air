# Performance Profile: LLAMA-3.2-1B BF16 on NPU2

**Model**: LLAMA-3.2-1B, BF16, 16 layers, emb_dim=2048, hidden_dim=8192, vocab=128256

## Performance Summary

| Phase | AIR (NPU2) | IRON | Speedup |
|-------|------------|------|---------|
| **Prefill / TTFT** (seq_len=2048) | **1.27s wall** | 2.744s | **2.17x** |
| **Decode / TPOT** (steady-state) | **92ms/token (10.8 tok/s)** | 370ms/token (2.7 tok/s) | **4.0x** |

- **TTFT** (time-to-first-token): end-to-end from `make run` invocation to
  first decoded token — includes tokenize + EOS-pad + embed + 16 layers
  + final RMSNorm + LM head GEMV. Matches the vLLM / TGI / TRT-LLM TTFT
  definition. With tokenize added back in, current measured TTFT is
  ~1.28&nbsp;s (the 1.27&nbsp;s row above is the NPU-only fraction used
  in the IRON comparison, since IRON does not bundle the tokenizer).
- **TPOT** (time-per-output-token): steady-state per-token decode latency
  (excludes prefill / first-token cost). Drift across 30 decode tokens is
  <1% — see `Per-Token Wall Trend` in `make profile` output.
- **IRON baseline**: measured against the IRON reference at commit
  [`2b62dc7`](https://github.com/amd/IRON/commit/2b62dc77ecc72f0fa8fb3381b05579ab84778d27)
  of `amd/IRON`, same NPU2 hardware (Strix), same LLAMA-3.2-1B BF16 model,
  same `seq_len=2048`.

For the visual end-to-end dataflow with per-step measured timing and the
BO Write / NPU Run / BO Read concept walkthrough, see
[`PROFILE.html`](PROFILE.html). This file is the textual reference
(per-kernel tables, optimization history, vs IRON comparison).

**Recent optimizations** (vs. an earlier 1.54s wall headline):
1. Last-token-only LM Head: drop full-sequence NPU rmsnorm + 8-partition GEMM
   in prefill; do CPU rmsnorm on the 1×emb_dim last row (<1 ms) and reuse the
   decode-side `lm_head_gemv.elf` for the single-position projection (~14 ms NPU
   GEMV). Saves ~150 ms — autoregressive generation only needs that one row.
2. Eliminate per-layer numpy heap churn in `run_transformer_block` via
   `astype(bfloat16, copy=False)`. Saves ~110 ms (mostly trial-1 amortization).

---

## End-to-End Inference Workflow

The unified pipeline (`llama32_1b_inference.py`) runs through these phases:

```
Phase 1: Compilation  (one-time, ~3 min, cached to disk)
  ┌──────────────────────────────────────────────────────────────┐
  │  compile_all_external_kernels()                              │
  │    silu_and_mul.o  ← ffn_swiglu/silu_and_mul.cc              │
  │    rope.o          ← kernel_builder/rope_halfsplit.cc        │
  │    attn_npu2.o     ← flash_attention/attn_npu2.cc            │
  │    mv.o            ← matrix_vector_multiplication/mv.cc      │
  │    mv_k8192.o      ← mv.cc with -D renamed symbols           │
  │                                                              │
  │  compile_all_kernels() → prefill_kernel_cache/               │
  │    rms_gemms_rope.elf   (6 launches, 33s)                    │
  │    flash_attn.elf       (1 launch,  46s)                     │
  │    o_ffn.elf            (8 launches, 50s)                    │
  │                                                              │
  │  compile_decode_kernels() → decode_kernel_cache/             │
  │    rms_gemv_rope.elf    (6 launches, 3s)                     │
  │    o_gemv_ffn.elf       (8 launches, 7s)                     │
  │    lm_head_gemv.elf     (8 launches, 13s)                    │
  └──────────────────────────────────────────────────────────────┘

Phase 2: Prepare Runtime  (one-time, before inference)
  ┌──────────────────────────────────────────────────────────────┐
  │  Load model weights from safetensors                   ~3s   │
  │  Pre-transpose decode GEMV weights                     ~2s   │
  │  Pre-load prefill weights into per-layer BOs           ~5s   │
  │    16 layers × (wq, wk, wv, wo, w_gate, w_up, w_down)       │
  │    + LM Head: 8 partitions × 64MB = 512MB                   │
  │  Pre-load decode weights into per-layer BOs            ~8s   │
  │    16 layers × (wq_t, wk_t, wv_t, wo_t, wgate_t, etc.)     │
  │    + LM Head GEMV: 8 partitions × 64MB = 512MB              │
  └──────────────────────────────────────────────────────────────┘

  ══════════════════ PROFILED SCOPE ════════════════════════

Phase 3: Inference
  ┌──────────────────────────────────────────────────────────────┐
  │  PREFILL: 16 layers + CPU RMSNorm (last row) + LM Head GEMV │
  │    Wall time:   1.27s  (kernel + minimal Python overhead)   │
  │  DECODE:  per token (16 layers + LM Head GEMV)    → 92ms    │
  └──────────────────────────────────────────────────────────────┘

  ═════════════════════════════════════════════════════════════
```

**Profiled scope matches IRON**: Both frameworks pre-load weights before timing.
IRON reports wall time (end-to-end timed section). AIR wall time (1.27s) includes
minimal Python host overhead (KV cache extraction, embedding lookup, numpy views)
that IRON's C++ runtime avoids.

Key differences favoring AIR:
- AIR skips intermediate BO syncs (`intermediate_indices`) — IRON syncs ALL BOs
- AIR only reads `output_indices` — IRON reads ALL BOs back after each kernel

---

## Prefill Breakdown (seq_len=2048, 16 layers)

### Wall Time Breakdown: 1.27s (NPU-only) / ~1.28s TTFT

| Component | Time | Notes |
|-----------|------|-------|
| **NPU XRT calls** (sum of `load_and_run`) | ~1.12s | BO Write + NPU Run + BO Read across 49 calls: 16×3 transformer + 1 lm_head_gemv |
| **CPU host ops** (profiled) | ~37ms | tokenize + eos_pad + embed_lookup + 16×kv_cache_extract + final_rms_norm |
| **Python / numpy scheduling** | ~125ms | Per-layer dict access, numpy view setup, loop overhead (`layer-loop wall − inside-layer NPU − inside-layer CPU`) |
| **Total TTFT** (incl. tokenize) | **~1.28s** | matches `make run` Time-to-First-Token line |
| Total wall (NPU-only fraction, vs IRON) | ~1.27s | excludes tokenize; the row used in the IRON comparison |

Overhead reduced from 0.67s → 0.24s by:
- Suppressing print I/O in non-profile mode (4 prints × 16 layers)
- Removing dead `x_f32` dual-precision code and `output_f32` conversion
- Assembling only the prediction row for LM Head (not full 2048×128K)
- Skipping `bf16→f32` conversion on full logits array
- Skipping intermediate dict storage when not verifying
- Removing redundant `.astype(bfloat16)` on already-bf16 kernel results

### Per-Kernel Timing (NPU XRT calls only)

| Kernel | Launches | Per-call | x Calls | Total | % of NPU |
|--------|----------|----------|---------|-------|---|
| **o_ffn** | 8 (stitched) | 41.0ms | 16 | **656ms** | **59%** |
| **flash_attn** | 1 (separate ELF) | 21.6ms | 16 | **346ms** | **31%** |
| **rms_gemms_rope** | 6 (stitched) | 7.3ms | 16 | **117ms** | **10%** |
| **lm_head_gemv** | 8 partitions (stitched) | 13.6ms | 1 | **14ms** | **1%** |

Per-CPU-op:

| CPU op | Per-call | x Calls | Total |
|--------|----------|---------|-------|
| tokenize | ~10 ms | 1 | ~10 ms |
| eos_pad | <0.1 ms | 1 | <0.1 ms |
| embed_lookup | 5.8 ms | 1 | 5.8 ms |
| kv_cache_extract | 1.1 ms | 16 | 17.6 ms |
| final_rms_norm | 3.1 ms | 1 | 3.1 ms |

### Host vs NPU Breakdown (XRT calls only — `cache.load_and_run` internals)

| | BO Write | NPU Run | BO Read | Total |
|---|----------|---------|---------|-------|
| **Sum** | 46ms | 1062ms | 5ms | 1113ms |
| **%** | **4%** | **95%** | **0%** | 100% |

(BO Read is zero-copy view construction — see PROFILE.html Part C for what
these three segments actually measure.)

### Per-Layer Data Flow

```
Layer input: x_bf16 (2048x2048, 8MB)

┌─ KERNEL 1: rms_gemms_rope (7.3ms/layer) ───────────────────────┐
│                                                                 │
│  WRITE: x_in (8MB)              ← activation, changes/layer    │
│  SKIP:  norm_w, wq, wk, wv     ← STATIC (per-layer BO)        │
│  SKIP:  lut_q, lut_k           ← STATIC (same across layers)  │
│  SKIP:  normed, q, k, v,       ← INTERMEDIATE                  │
│         q_roped, k_roped                                        │
│                                                                 │
│  NPU (6 launches):                                              │
│    RMSNorm [8,1] → Q GEMM [8,4] → K GEMM [8,4] →              │
│    V GEMM [8,4] → RoPE Q [8,1] → RoPE K [8,1]                 │
│    (intermediates stay in DDR, no CPU round-trip)              │
│                                                                 │
│  READ: v (2MB), q_roped (8MB), k_roped (2MB)                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─ KERNEL 2: flash_attn (21.6ms/layer) ──────────────────────────┐
│                                                                 │
│  WRITE: q_roped (8MB), k_roped (2MB), v (2MB)                  │
│  SKIP:  attn_out                ← INTERMEDIATE                  │
│                                                                 │
│  NPU (1 launch):                                                │
│    FlashAttention GQA (32Q/8KV heads, causal, seq-first)        │
│                                                                 │
│  READ: attn_out (8MB)                                           │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─ KERNEL 3: o_ffn (41ms/layer) ─────────────────────────────────┐
│                                                                 │
│  WRITE: attn_out (8MB),         ← from kernel 2                │
│         x_residual (8MB)        ← skip connection from input    │
│  SKIP:  wo, ffn_norm_w,        ← STATIC (per-layer BO)         │
│         w_gate, w_up, w_down                                    │
│  SKIP:  proj, res1, normed2,   ← INTERMEDIATE                   │
│         gate, up, swiglu,                                       │
│         down, output                                            │
│                                                                 │
│  NPU (8 launches):                                              │
│    O GEMM [8,4] → Add [8,1] → RMSNorm [8,1] →                 │
│    Gate GEMM [8,4] → Up GEMM [8,4] → SiLU×mul [8,1] →         │
│    Down GEMM [8,4] → Add [8,1]                                 │
│    (all intermediates stay in DDR, no CPU round-trip)                            │
│                                                                 │
│  READ: output (8MB) → next layer's x_in                        │
└─────────────────────────────────────────────────────────────────┘

× 16 layers, then:
  final_rms_norm (CPU, 3.1ms): RMSNorm on single prediction-position row
  lm_head_gemv (NPU, 13.6ms): 8-partition GEMV → vocab logits → argmax → first token
                              (reuses the decode-side 8-partition ELF; see
                               A7 in IMPLEMENTATION_GUIDE.html for why
                               full-seq GEMM was dropped in favor of single-row GEMV)
```

---

## Decode Breakdown (per token): 92ms

### Per-Component Timing

| Component | Per-token | % |
|-----------|-----------|---|
| **o_gemv_ffn** (NPU, 3.6ms × 16 layers) | **58ms** | **63%** |
| **rms_gemv_rope** (NPU, 0.9ms × 16 layers) | **14ms** | **15%** |
| **lm_head_gemv** (NPU, 1 call) | **14ms** | **15%** |
| CPU (attention + RMSNorm + host) | **5ms** | **5%** |
| BO overhead | **<1ms** | **<1%** |

### Per-Layer Data Flow

```
Token input: x_bf16 (2048 elements, 4KB)

┌─ KERNEL 1: rms_gemv_rope (0.9ms/layer) ────────────────────────┐
│                                                                  │
│  WRITE: x_in (4KB), lut_q (4KB), lut_k (1KB)                    │
│  SKIP:  norm_w, wq, wk, wv        ← STATIC (per-layer BO)      │
│  SKIP:  normed, q, k, v,          ← INTERMEDIATE                │
│         q_roped, k_roped                                         │
│                                                                  │
│  NPU (6 launches):                                               │
│    RMSNorm [1,1] → Q GEMV [8,1] → K GEMV [8,1] →               │
│    V GEMV [8,1] → RoPE Q [1,1] → RoPE K [1,1]                  │
│                                                                  │
│  READ: v (1KB), q_roped (4KB), k_roped (1KB)                    │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌─ CPU ATTENTION (0.3ms/layer) ──────────────────────────────────┐
│  GQA: 32 Q heads attend to KV cache (8 KV heads)               │
│  Update KV cache at current_pos                                 │
│  Output: attn_out (4KB)                                         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─ KERNEL 2: o_gemv_ffn (3.6ms/layer) ───────────────────────────┐
│                                                                  │
│  WRITE: attn_out (4KB), x_residual (4KB)                         │
│  SKIP:  wo, ffn_norm_w, wgate,     ← STATIC (per-layer BO)      │
│         wup, wdown                                               │
│  SKIP:  proj, res1, normed2,       ← INTERMEDIATE                │
│         gate, up, swiglu,                                        │
│         down, output                                             │
│                                                                  │
│  NPU (8 launches):                                               │
│    O GEMV [8,1] → Add [8,1] → RMSNorm [1,1] →                  │
│    Gate GEMV [8,1] → Up GEMV [8,1] → SiLU×mul [8,1] →           │
│    Down GEMV [8,1] → Add [8,1]                                   │
│                                                                  │
│  READ: output (4KB)                                              │
└──────────────────────────────────────────────────────────────────┘

× 16 layers, then:
  CPU RMSNorm (0.01ms): normalize 2048-element vector
  lm_head_gemv (13.5ms): 8-partition GEMV → vocab logits → argmax → next token
```

### 100-Token Profile

```
Token  1:  92ms  ← steady from first token (NPU prefill keeps NPU warm)
Token  2:  91ms  ┐
Token  3:  91ms  │
...              ├ steady state: 92ms ± 2ms
Token 99:  92ms  │
Token100:  92ms  ┘

Total: 9.21s for 100 tokens = 10.86 tok/s
```

---

## Multi-Launch Memory Model

Each `air.launch` within a multi-launch ELF reads inputs from DDR and writes outputs
back to DDR. Intermediates do **not** stay in L1/L2 between launches — they go through
DDR (L3 memory). What multi-launch saves is the **CPU round-trip**, not the DDR access:

```
SEPARATE XRT CALLS (before multi-launch merging):
  Launch 1 output → DDR
    → bo.sync(FROM_DEVICE)        CPU pulls data from DDR into host cache
    → numpy array in host memory  CPU processes/reshapes
    → bo.map() + memcpy           CPU writes to new BO's mapped memory
    → bo.sync(TO_DEVICE)          CPU pushes data back to DDR
  Launch 2 input ← DDR

MULTI-LAUNCH ELF (current):
  Launch 1 output → DDR
    (no sync, no memcpy, no CPU involvement)
  Launch 2 input ← DDR            NPU DMA reads same DDR buffer directly
```

The DDR is shared physical memory accessible by both CPU and NPU. The difference is:
- **Separate calls**: CPU orchestrates data movement (cache sync + memcpy + cache sync)
- **Multi-launch**: NPU DMA engines handle DDR reads/writes autonomously within
  one `xrt.run()` invocation. The CPU only writes actual input activations and reads
  final outputs.

This is why `intermediate_indices` (SKIP) is effective: these DDR buffers are written
by one launch and read by a subsequent launch — the CPU never needs to see them.

---

## BO Write Categories

Every BO argument falls into one of three categories, controlling whether data is
synced to device on each kernel invocation:

```python
for i, array in enumerate(inputs):
    if i in static_input_indices and not first_call:
        continue    # STATIC: weight pre-loaded, skip
    if i in intermediate_indices and not first_call:
        continue    # INTERMEDIATE: kernel overwrites, skip
    # WRITE: activation data that changes each call
    bo.map() → memcpy → bo.sync(TO_DEVICE)
```

| Category | When Written | Examples |
|----------|-------------|---------|
| **WRITE** | Every call | x_in, attn_out, x_residual, LUTs |
| **STATIC** | First call only | wq, wk, wv, wo, w_gate, w_up, w_down, norm_w |
| **INTERMEDIATE** | First call only | normed, q, k, v, proj, gate, up, swiglu, down |

**Per-layer BOs** (`bo_key=f"kernel_L{layer_idx}"`): Each of 16 layers gets its own
BO set. Weights written once during pre-load, reused forever.

---

## NPU Power Management

The AMD NPU enters low-power state after ~10 seconds of inactivity:

| Idle Duration | Penalty on Next Kernel |
|--------------|----------------------|
| 0-5 seconds | None |
| 10+ seconds | +150ms |

In the unified pipeline (`llama32_1b_inference.py`), this is not an issue: the NPU
prefill (~1.27s of continuous NPU activity) keeps the hardware warm right up until
decode starts. No explicit warmup pass is needed.

---

## Key Optimizations

| Optimization | How it Works | Impact |
|-------------|-------------|--------|
| Multi-launch ELF | Stitch multiple kernels into one air.launch func via text-based MLIR stitching | 10→3 prefill calls/layer, 10→3 decode calls/block |
| Per-layer weight BOs | Each layer has dedicated BOs; weights written once, reused | -240ms prefill (14%→4% BO overhead) |
| `intermediate_indices` | Skip host→device sync for buffers the kernel overwrites | -150ms prefill, <1% decode overhead |
| NPU LM Head GEMV | 8-partition GEMV replaces CPU matmul for decode LM Head | 258ms→13.5ms per token |
| External kernel rename | Compile mv.cc with `-D` defines for renamed symbols | Enables K=2048+K=8192 GEMV in one ELF |
| Seq-first layout | RoPE + FlashAttention natively accept (seq, heads×dim) | Zero host transposes in prefill |
| `collapse_shape`/`expand_shape` | 2D↔1D type aliasing inside launch bodies | Enables shape-incompatible kernel merging |
| All .o from source | `compile_all_external_kernels()` builds all C++ kernels fresh | No stale pre-compiled artifacts |
