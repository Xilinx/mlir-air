<!---//===- FlashAttention_bf16.md ----------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# FlashAttention (BF16, GQA) — Kernel Detail

> Fused scaled-dot-product attention (online-softmax FlashAttention). Grouped-query attention (GQA), optional causal masking, BF16 in/out, FP32 softmax-scale and MMA accumulators. Verified on NPU2 across a range of shapes (head dim 64 / 128, MHA & GQA, short & long sequences, causal & non-causal — see [Tested shapes](#tested-shapes)).
> The kernel is `attn_npu2.cc` → `attn_npu2.o`, driven by the **heads-first** harness `attn_npu2.py`: `Q[num_heads, lq, dk]`, `K[num_kv_heads, lk, dk]`, `V[num_kv_heads·dv_chunks, lk, dv_tile]` → `O[num_heads·dv_chunks, lq, dv_tile]`.
> A second harness, `attn_npu2_seqfirst.py`, drives the **same `attn_npu2.o`** with a seq-first L3 layout (`[seq, heads·dk]`); it was derived for llama-3.2-1B prefill (no transpose needed in that pipeline) and is numerically **bit-identical** to heads-first (proven in [Layout equivalence](#layout-equivalence)). All shapes below are measured with the heads-first harness; the seq-first variant produces identical results.
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, June 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/flash_attention/kernel_fusion_based/
  attn_npu2.py            # heads-first harness (primary)
    build_module(lk, lkp, lq, lqp, dk, dv,
                 num_q_tiles, num_cascade_stages,
                 num_heads, num_kv_heads, causal,
                 num_heads_per_unroll)
  attn_npu2_seqfirst.py   # seq-first variant (same .o; llama-3.2-1B prefill uses it)
```

Driven by the harness CLI; the example also has a `Makefile`. The compute kernel is a hand-written microkernel `attn_npu2.cc` → `attn_npu2.o`, linked into the herd via `link_with`. Its tile parameters (`lqp`, `lkp`, `dk`, `dv`, derived `tile_size_q`) are **compile-time** `-D` macros baked into `attn_npu2.o` — sweeping them requires recompiling the `.o` (like GEMM's `mm.o`).

**Tile usage (full chip).** The work is laid out as `segment(num_heads_per_unroll=2) × herd(num_q_tiles × num_cascade_stages)` compute tiles. At a 32-tile config (`num_q_tiles=4, num_cascade_stages=4`) this is `2 × 4 × 4 = 32` tiles = **the full NPU2 8×4 array**. Columns = `num_heads_per_unroll(2) × num_q_tiles(4)` = 8; rows = `num_cascade_stages(4)` = 4. The outer `launch` is sequential in time — it does not consume extra tiles.

**`dv_chunks` (value-dim chunking).** When `dv > lkp` (e.g. `dk = dv = 128`, `lkp = 64`), the value dimension is split into `dv_chunks = dv/lkp` chunks; the heads-first layout handles this directly. (The seq-first variant asserts `dv_chunks == 1` — its interleaved-layout dv-chunked DMA strides are not test-covered — so head-dim-128 shapes use the heads-first harness.)

---

## Layout equivalence

The two variants share `attn_npu2.o`, so the same logical Q/K/V must yield the same numerics regardless of layout. Verified directly: the same `randn` per-head tensors were packed two ways (heads-first `[heads,seq,dk]` and seq-first `[seq,heads·dk]`), each compiled + run on NPU2, and both outputs de-permuted to a common `[num_heads, lq, dv]` frame and compared element-wise.

| config | max abs diff | exact-equal frac |
|---|---|---|
| 512×512, 2 heads, non-causal | **0.0** | 100% |
| 2048×2048, 32q/8kv, causal+GQA | **0.0** | 100% |

Every element is bit-identical, and each variant's error vs the FP32 reference matches to 6 significant figures — concrete evidence that the two are the *same compute kernel* under two L3 layouts.

---

## Numerical datapath (what "BF16 FlashAttention" means here)

```
Q,K,V bf16 → S = Q@Kᵀ (8×8×8 MMA, BFP16-emulated) → online softmax (running max/sum, exp2) → O = P@V (8×8×8 MMA, BFP16-emulated) → bf16
```

- **Both matmuls use the 8×8×8 MMA with BFP16 emulation** (`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, same datapath as BF16 GEMM): bf16 operands are cast to block-floating-point for the matrix unit, MMA accumulator is FP32. So FA inherits GEMM's per-matmul block-quantization error — and chains **two** such matmuls (Q@Kᵀ then P@V).
- **The softmax scale `1/√dk` is folded into the `log2e/√dk` exp2 constant at FP32 precision** (`attn_npu2.cc`), avoiding a bf16 truncation of Q. `exp2` is evaluated through an `accfloat` (FP32) accumulator; `conv_even` rounding keeps softmax intermediates unbiased.
- **The online-softmax running statistics (running max, running denominator) and the cross-K-block output accumulator are kept in bf16** (each individual reduce uses an `accfloat`/FP32 accumulator, but the values carried *between* K-chunks / cascade stages are bf16). This is the one place the NPU kernel diverges from the GPU FlashAttention-2 convention — see [Precision: datapath vs the GPU standard](#precision-datapath-vs-the-gpu-standard).
- **Output is cast to bf16** at the end (`div_gp_sp`: divide accumulated `P@V` by the softmax denominator).

The net effect (two BFP16-emulated MMAs + bf16 online-softmax) is a measured `mean_rel_L1 ≈ 3.9e-2` — about 4× the BF16 GEMM tier (~9.3e-3), as expected for a chained, non-linear kernel.

---

## Numerical accuracy

Verified element-wise over the full output against an FP32 SDPA reference (softmax in f32 — matching PyTorch `_scaled_dot_product_attention_math` / vLLM FA reference):

| Metric (llama config: lq=lk=2048, dk=dv=64, 32 heads / 8 KV, causal, randn inputs) | Measured |
|---|---|
| `mean_rel_L1 = mean|o−ref| / mean|ref|` | **3.9e-2** |
| `rel_err max` | ~1e+5–1e+6 (near-zero-ref blowup, not a signal) |
| `abs_err max` | 8.4e-2 |

- **`mean_rel_L1 = 3.9e-2`** — ~4× the GEMM tier, the expected cost of chaining two BFP16-emulated MMAs plus a bf16 online-softmax. Identical for both layouts (same kernel). Consistent with GPU practice being looser for fused attention than for a single matmul (PyTorch's own fused-vs-math SDPA bf16 test uses `rtol=atol=5e-2`).
- **`rel_err max` large** is the usual near-zero-reference blowup (causal attention produces many near-zero output elements) — not a meaningful failure signal; read `mean_rel_L1` and `abs_err max`.
- **Accuracy is set by the datapath (dtype + BFP16 MMA + bf16 softmax), not by the tile config** — changing `num_cascade_stages` etc. (within what places) does not move accuracy.

---

## Precision: datapath vs the GPU standard

The accuracy gap to a single matmul is **structural**, not a bug. A standard GPU BF16 FlashAttention kernel carries the online-softmax intermediates in **FP32** across the whole K loop and casts to bf16 only at the boundaries (verified in vLLM's CUDA attention kernel: `qk_max`, `exp_sum`, the `logits`/probability buffer, and the `accs[]` output accumulator are all `float`). The NPU kernel keeps the **single-step** accumulators in FP32 (each MMA's ACC, each reduce's `accfloat`) but stores the **cross-K-block** carried values in bf16:

| Step | symbol | GPU (vLLM CUDA) | NPU (`attn_npu2.cc`) | aligned? |
|---|---|---|---|---|
| Q@Kᵀ MMA accumulator | — | FP32 | **FP32** | ✅ |
| Q,K into MMA | — | bf16 | bf16 **+ BFP16-emulate** (block quant, GPU has none) | ⚠️ |
| scores S (pre-softmax) | `g` | **FP32** | **bf16** | ❌ |
| running max | `m` / `up` | **FP32** | **bf16** | ❌ |
| probabilities P | `g` | **FP32** | **bf16** | ❌ |
| running denominator | `l` / `sp` | **FP32** | **bf16** | ❌ |
| **output accumulator (cross-K-block)** | `O` / `gp` | **FP32** (`accs[]`) | **bf16** | ❌ |
| final O cast | — | bf16 | bf16 | ✅ |

Two sources of error stack: the **bf16 carry** of the softmax statistics/accumulator (the NPU rounds them to bf16 once per K-chunk / cascade stage, where the GPU keeps f32), plus the **BFP16-emulated MMAs** (which the GPU does not have). Together they put FA at `mean_rel_L1 ≈ 3.9e-2`, ~4× a single GEMM — which is **why the absolute tolerance is sized looser than a single-matmul kernel** (`atol = 1e-1` here vs `4e-3` for GEMM). The relative tolerance is unchanged at the canonical bf16 `rtol = 1.6e-2`. GPU FA is likewise looser than GEMM (PyTorch fused SDPA bf16 uses `5e-2`).

> **Note — FP32 accumulator path not implemented yet.** Fully matching the GPU standard would keep the output accumulator (and ideally the running max/denominator) in FP32 across the K loop. That variant is **not currently implemented**: the kernel carries these values in bf16, a deliberate L1-budget choice (an FP32 accumulator doubles its on-chip footprint). The numbers in this document are for the shipping **bf16** kernel. An FP32-accumulator variant is future work.

---

## Parameters & constraints

`build_module` takes two kinds of parameters.

**Shape parameters** — fixed by the model / input, not knobs: `lq`, `lk`, `dk`, `dv`, `num_heads`, `num_kv_heads`, `causal`.

**Tile parameters** — in principle tunable, but in practice **constrained to a near-unique value** (see [tile choice vs performance](#tile-choice-vs-performance) for the sweep that establishes this). Each constraint traces to a concrete source:

| Param | Value | Maps to | Constraint → where it comes from |
|---|---|---|---|
| `num_cascade_stages` | 4 | physical **rows** | **= 4**: NPU2 has exactly 4 compute rows; `< 4` under-pipelines / explodes `aircc` routing on long seq, `> 4` cannot place |
| `num_q_tiles` | 4 | physical **columns** (× `num_heads_per_unroll`) | `num_heads_per_unroll × num_q_tiles = 8`: NPU2 has 8 columns |
| `num_heads_per_unroll` | 2 | column multiplier | **≤ 2**: a shim-DMA-channel limit (hpu=4 runs out of channels) |
| `lqp` | 256 | Q chunk per launch | `lq % lqp == 0`; pins `tile_q = lqp/num_q_tiles` (below) |
| `lkp` | 64 | K/V chunk (MMA k-dim) | `= dk` (shared-buffer mode); `dk % lkp == 0`, `dv % lkp == 0` (`dv > lkp` → `dv_chunks > 1`, heads-first only) |
| `tile_q` | 64 | Q rows per tile | **= lkp**: causal *asserts* it (the block-wise mask needs square Q-tile/K-chunk alignment — see [Tolerances & reference](#tolerances--reference)); non-causal has no assert but `tile_q ≠ 64` hangs (32) or overflows L1 (≥128) anyway |

**Where the constraints come from** (three sources, all hardware/algorithm, not arbitrary):
- **NPU2 fabric = 8 columns × 4 compute rows.** The herd's two axes map directly onto it: `num_heads_per_unroll × num_q_tiles` → columns (≤ 8), `num_cascade_stages` → rows (≤ 4). Filling the chip (anything less wastes it) forces columns = 8, rows = 4.
- **Shim DMA channels** cap `num_heads_per_unroll ≤ 2`.
- **L1 budget (64 KB/tile) + the causal mask** pin `tile_q = lkp = 64`.

These hold for **both layouts** (same `attn_npu2.o`). heads-first additionally allows `dv_chunks > 1` (`dv > lkp`, e.g. `dk=dv=128`), which seq-first rejects.

---

## Tolerances & reference

Element-wise over the **full output** against an FP32 reference: every element must pass `|a−e| ≤ atol + rtol·|e|`.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 1e-1 |

- **Reference** = CPU FP32 scaled-dot-product-attention (per head: `Q@Kᵀ/√dk` in f32, causal mask, softmax in f32, `P@V` in f32), cast to bf16 — matching PyTorch `_scaled_dot_product_attention_math` and vLLM's FlashAttention reference. Inputs are `randn`.
- `rtol = 1.6e-2` is PyTorch / vLLM's canonical bf16 tolerance. `atol = 1e-1` is sized to this kernel's measured worst-case absolute error (looser than a single matmul because FA chains two BFP16-emulated MMAs and a bf16 online-softmax — see [Precision: datapath vs the GPU standard](#precision-datapath-vs-the-gpu-standard)).

---

## Tested shapes

Shapes verified on NPU2 (heads-first harness; the seq-first variant is bit-identical). **All use the same tiling `lqp=256, num_q_tiles=4, num_heads_per_unroll=2, num_cascade_stages=4`** (32 tiles, `tile_size_q = lkp = 64`) — this is the near-unique working config for every shape, not a per-shape choice (see [tile choice vs performance](#tile-choice-vs-performance) for why FA has almost no tile freedom). Throughput is GFLOP/s (compute-bound; FLOPs = `2·num_heads·lq·lk·(dk+dv)` — Q@Kᵀ scales with `dk`, P@V with `dv` — halved for causal). `mean_rel_L1` is vs an FP32 SDPA reference.

| lq×lk | dk/dv | heads q/kv | causal | dv_chunks | latency | GFLOP/s | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|---|---|---|---|
| 2048×2048 | 64/64 | 32/8 | ✓ | 1 | 15.4–16.1 ms | **1065–1116** | 3.9e-2 | 8.4e-2 | ✅ |
| 2048×2048 | 64/64 | 32/32 | ✓ | 1 | 16.9 ms | 2031 | 3.9e-2 | 9.4e-2 | ✅ |
| 512×512 | 64/64 | 2/2 | ✗ | 1 | 0.73 ms | 184 | 4.4e-2 | 4.7e-2 | ✅ |
| 512×512 | 64/64 | 12/6 | ✗ | 1 | 1.22 ms | 661 | 4.6e-2 | 3.9e-2 | ✅ |
| 512×512 | 64/64 | 64/8 | ✗ | 1 | 3.79 ms | 1135 | 4.6e-2 | 5.9e-2 | ✅ |
| 512×512 | 128/128 | 32/8 | ✗ | 2 | 4.38 ms | 980 | 4.4e-2 | 5.7e-2 | ✅ |
| 512×512 | 128/128 | 28/4 | ✗ | 2 | 4.05 ms | 928 | 4.4e-2 | 3.6e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✓ | 1 | 39.6 ms | 1734 | 4.5e-2 | 6.8e-2 | ✅ |
| 16384×16384 | 64/64 | 2/2 | ✗ | 1 | 40.1 ms | **3427** | 5.5e-2 | 5.9e-3 | ✅ |
| 2048×2048 | 128/128 | 16/8 | ✓ | 2 | — | — | 3.8e-2 | 7.1e-2 | ✅ Qwen3-0.6B + Qwen3-1.7B prefill (head_dim=128, 16q/8kv — identical FA config; re-PASSED on Qwen3-1.7B run) |
| 2048×2048 | 64/64 | 14/2 | ✓ | 1 | — | — | 3.8e-2 | 7.0e-2 | ✅ Qwen2.5-0.5B prefill (head_dim=64, 2 KV heads) |
| 2048×2048 | 128/128 | 12/2 | ✓ | 2 | — | — | 3.8e-2 | 7.0e-2 | ✅ Qwen2.5-1.5B prefill (head_dim=128, 12q/2kv GQA) |

> The **2048×2048, 32q/8kv, causal** row is the config llama-3.2-1B prefill imports (`attn_npu2_seqfirst.py`'s `build_module` → `flash_attn` ELF); its two-harness GFLOP/s range reflects run-to-run timing variation (~5%). The **2048×2048, 32q/32kv, causal** row is SmolLM2-1.7B's prefill config (pure MHA — every Q head has its own KV head, no GQA broadcast); same near-unique tiling, same `mean_rel_L1` (FA error is datapath-bound, independent of kv-head count), but ~2× the GFLOP/s because MHA does ~2× the attention FLOPs of 8-KV GQA in similar wall-time. The other rows are additional NPU2-verified shapes (head dim 64/128, MHA & GQA ratios, short & long sequences, causal & non-causal) — they record what the kernel is known to run, independent of any specific model. `head_dim = 128` shapes use `dv_chunks = 2` (heads-first only). The **2048×2048, 16q/8kv, causal, head_dim=128** row is Qwen3-0.6B's prefill attention config (lq=lk=2048, full-chip `lqp=256/num_q_tiles=4/heads_per_unroll=2/cascade=4`, `dv_chunks=2`), verified PASS at 3.8e-2. Note: long-sequence `head_dim=128` FA has been flaky (`ERT_CMD_STATE_TIMEOUT` / NaN) on some NPU2 setups; this run completed cleanly, but a deployment hitting the hang can fall back to CPU attention (`cpu_attn`). The **2048×2048, 14q/2kv, causal, head_dim=64** row is Qwen2.5-0.5B's prefill attention config (lq=lk=2048, `dv_chunks=1`); head_dim=64 has no hang risk, verified PASS at 3.8e-2. The **2048×2048, 12q/2kv, causal, head_dim=128** row is Qwen2.5-1.5B's prefill attention config (lq=lk=2048, `dv_chunks=2`); like Qwen3-0.6B's head_dim=128 it carries the long-sequence hang risk but completed cleanly this run (3.8e-2), with `cpu_attn` available as the fallback.

**Reading the table**:
- **Compute-bound**: FA is dominated by the two matmuls (Q@Kᵀ, P@V); throughput is GFLOP/s.
- **Accuracy** `mean_rel_L1 ≈ 3.9e-2` at the reference shape — ~4× the GEMM tier from chaining two BFP16-emulated MMAs + bf16 online-softmax; set by the datapath, not the tile config.

---

## tile choice vs performance

Unlike GEMM (which has a rich `tile_m × tile_k × tile_n` space where the swept-best tile beats the default by a wide margin), **FlashAttention has almost no tunable headroom**: the hardware + algorithm constraints collapse the config to a near-unique point. A sweep over all four knobs (`lqp, num_q_tiles, num_heads_per_unroll, num_cascade_stages`) on NPU2 establishes this — accuracy is unchanged across configs (set by the datatype path), so this is purely about placement and throughput.

**1. Full-chip is mandatory; only 2 herd shapes fill 32 tiles.** The herd's two axes map directly to the fabric grid: columns = `num_heads_per_unroll × num_q_tiles ≤ 8`, rows = `num_cascade_stages ≤ 4`, and `num_heads_per_unroll ≤ 2` (shim-DMA limit). To fill the full 8×4 array (anything less wastes the chip), `cols = 8` and `rows = ncs = 4` are forced, leaving only **two** herd shapes:

| lqp | nqt | hpu | ncs | tile_q | GFLOP/s (2048 causal) | note |
|---|---|---|---|---|---|---|
| 256 | 4 | 2 | 4 | 64 | 1065–1116 | 2 heads × 4 Q-tiles (llama default) |
| 512 | 8 | 1 | 4 | 64 | **1054–1131** | 1 head × 8 Q-tiles |

The two are within **~1–3%** across all shapes measured — the 2-heads-vs-finer-Q-tiling tradeoff barely moves throughput.

**2. `num_cascade_stages` must be 4** (= physical rows). Reducing it under-pipelines (cs=2 → ~25% slower); cs=8 needs 8 rows and does not place. On long sequences, low `ncs` also makes `aircc` routing explode (each tile serially scans `seq/lkp/ncs` chunks).

**3. `tile_q` is pinned to `lkp` (= 64), even when not required by the assert.** Under causal masking `tile_q = lqp/num_q_tiles == lkp` is enforced (see [Tolerances & reference](#tolerances--reference) — the block-wise causal mask needs square Q-tile/K-chunk alignment). For **non-causal** the assert is absent, but a sweep of `tile_q ∈ {32, 64, 128, 256, 512}` (varying `lqp` at fixed 32-tile) shows the same pin in practice:

| tile_q | result (512 & 16384 non-causal) |
|---|---|
| 32 | ❌ runtime hang (too few Q rows per tile vs the chunk schedule) |
| **64** | ✅ the only working point |
| ≥ 128 | ❌ does not place (L1 budget) |

**Bottom line.** Across the whole sweep the only working configs are `(lqp=256, nqt=4, hpu=2, ncs=4)` and the ~equivalent `(lqp=512, nqt=8, hpu=1, ncs=4)`; everything else either fails to place, hangs, or under-utilizes. The numbers in [Tested shapes](#tested-shapes) use `256/4/2/4`, which is within ~1–3% of the best placeable config for every shape. FlashAttention's tile config is **determined, not tuned**.

---

## How to reproduce (correctness + performance)

Each harness (compile-and-run mode, the default) runs the **correctness** check via `XRTRunner`: full-output element-wise compare against the FP32 reference; prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...` and `PASS!` / `failed.`

```bash
cd programming_examples/flash_attention/kernel_fusion_based

# correctness — any tested shape (heads-first harness). Set LK/LQ/DK/DV/NUM_HEADS/
# NUM_KV_HEADS to the row you want; add --causal for causal shapes.
# Examples:
make run SCRIPT=attn_npu2.py \
  LK=2048 LQ=2048 LKP=64 LQP=256 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8 \
  EXTRA_PY_FLAGS=--causal PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR     # 2048, dk=64, 32q/8kv causal

make run SCRIPT=attn_npu2.py \
  LK=512 LQ=512 LKP=64 LQP=256 DK=128 DV=128 NUM_HEADS=32 NUM_KV_HEADS=8 \
  PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR                             # dk=128 → dv_chunks=2
```

For **performance**, add `--perf-iters N` (e.g. `EXTRA_PY_FLAGS="--causal --perf-iters 20"`).

> The **seq-first** variant (what llama-3.2-1B prefill uses) runs the same shapes via `SCRIPT=attn_npu2_seqfirst.py` and gives bit-identical results — but it asserts `dv_chunks == 1`, so `head_dim = 128` shapes must use the heads-first harness shown above.

> ⚠️ **Run precision and performance separately.** The `--perf-iters` timing loop re-runs the kernel without re-syncing input buffers; because FlashAttention writes intermediate state into its output/scratch buffers, the precision number printed *in the same run as `--perf-iters`* is a buffer-reuse artifact (not the real accuracy). Take `mean_rel_L1` from a clean run (no `--perf-iters`) and latency from the `--perf-iters` run. An alternative C++ timing path exists via `make profile`.

Notes:
- Each config recompiles `attn_npu2.o` + the ELF.
- If the NPU is shared with other jobs, serialize on-device runs (e.g. with `flock`) so timing measurements aren't perturbed.
