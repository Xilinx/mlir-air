<!---//===- EltwiseAdd_bf16.md --------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Element-wise Add (BF16) — Kernel Detail

> Element-wise tensor addition `c = a + b`, the residual adds of a decoder-only LLM. BF16 in/out, per-element (no reduction, no non-linearity). The simplest leaf kernel — included to anchor the memory-bound, near-exact end of the registry.
> Shape is a flat length `N` (a 2-D `M×N` tensor is just `N = M·N` flattened): `a[N]`, `b[N]` → `c[N]`.
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, June 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/eltwise_add/eltwise_add.py
  build_module(n, tile_n, np_dtype, vector_size=16, num_tiles, herd_x, herd_y)
```

Driven by `eltwise_add.py`'s CLI; the example also has a `Makefile`. Single code-generation path: direct-codegen MLIR (vectorized `vector.transfer_read/write` + `arith.addf`), no external `.cc`. The input `[N]` is split into `tile_n`-sized chunks streamed through L3→L1→L3 DMA; the chunks are spread across an `herd_x × herd_y` AIE tile grid (each tile loops over its share). The herd is effectively 1-D for this kernel (rows are independent), so `herd_x` (AIE columns) is the scaling knob.

**Note on llama usage.** llama-3.2-1B's prefill residual add is **not** this standalone kernel — it is inlined into the fused `o_ffn` ELF (`o_ffn_multi.py`'s `_build_add_2d_to_2d`, a 2-D `[2048,2048]` add, herd 8×1, collapsed to 1-D inside the launch). The math is identical (`c = a + b`); only the L3 layout (2-D vs flat 1-D) and the surrounding fusion differ. This registry entry measures the **standalone** `eltwise_add` (a reusable, independently-reproducible building block); the fused variant has the same numerics.

---

## Numerical datapath (what "BF16 Element-wise Add" means here)

```
a,b bf16 → c = a + b (vectorized bf16 add) → bf16
```

- **Pure per-element add, no accumulation.** Each `vector_size`-wide bf16 block of `a` and `b` is added and written to `c`. There is no reduction, no running accumulator, and no non-linearity — so unlike GEMM (BFP16-emulated MMA), RMSNorm (FP32 reduction) or FlashAttention (chained MMAs + online-softmax), there is **nothing to lose precision in** beyond the single bf16 rounding of each output element.
- **The only quantization step is the bf16 rounding of `a + b`.** `a` and `b` are exact bf16; their f32 sum rounded back to bf16 is the result. This is the same single rounding a GPU bf16 add does.

The net effect is a measured `mean_rel_L1 ≈ 1.9e-3` — the cleanest kernel in the registry (below GEMM ~9.3e-3, RMSNorm ~4.2e-3), exactly as expected for a single-rounding elementwise op.

---

## Numerical accuracy

Verified element-wise over the full output against an FP32 reference (`a,b` upcast to f32, added, cast to bf16):

| Metric (N = 4194304 = 2048×2048, randn inputs) | Measured |
|---|---|
| `mean_rel_L1 = mean|c−ref| / mean|ref|` | **1.9e-3** |
| `rel_err max` | 7.8e-3 |
| `abs_err max` | 3.1e-2 |

- **`mean_rel_L1 = 1.9e-3`** — the cleanest tier in the registry. A single bf16 add rounds each output once; there is no accumulation to amplify error.
- **`rel_err max = 7.8e-3`** is small (no near-zero-reference blowup — `a + b` rarely cancels to near zero), so unlike FA/RMSNorm even the worst relative element is well within `rtol`.
- **`abs_err max = 3.1e-2`** is one bf16 ULP at the largest-magnitude sums — covered by `atol = 5e-2`.
- **Accuracy is identical across `herd_x` and across `N`** (bit-for-bit `1.9e-3` at herd_x ∈ {1,2,4,8} and N ∈ {1M,2M,4M,8M}) — set only by the datatype, not the tiling or size.

---

## Parameters & constraints

Element-wise Add is **memory-bound** (it streams `a`, `b` in and `c` out for an O(N) op, zero arithmetic intensity). The herd is 2-D (`sizes=[herd_x, herd_y]`), but a hardware limit pins the usable config:

| Knob | Value | Constraint → source |
|---|---|---|
| `herd_x` | **8** | AIE columns (≤ 8); `n % (tile_n · herd_x · herd_y) == 0` |
| `herd_y` | **1** | **must be 1** — each tile uses 3 independent L3↔L1 DMAs (a in, b in, c out); `herd_y > 1` exhausts the shim DMA channels (`aircc`: *"air.channel.put failed to map to shim dma channels: out of channels"*) |
| `tile_n` | 2048 | `n % (tile_n · herd_x · herd_y) == 0`; L1 chunk size; **≤ 4096** (≥ 8192 overflows L1 with ping-pong buffering) |
| `vector_size` | 16 | `tile_n % vector_size == 0`; AIE2 bf16 vector lane count |

**The kernel cannot use the full 32-tile array** — the 3-DMA-per-tile shim-channel demand caps it at `herd_y = 1`, i.e. **one row of 8 columns (8 tiles max)**. A sweep confirmed `herd_x = 8, herd_y = 1` is the best placeable config; every `herd_y > 1` config (8×2, 8×4, 4×4) fails to place with "out of channels". `tile_n` was swept with 3 repeats each (median bandwidth): it has a **small monotonic effect** — `256` → 55.4 GB/s up to `2048`/`4096` → ~57.5 GB/s (~4%, the larger block amortizes DMA-launch overhead, saturating by 2048), so **`tile_n = 2048` is the best (and the default)**. `vector_size` is not a tuning target. Accuracy is independent of all of them.

> **`Makefile` default**: `N=4194304, TILE_N=2048, HERD_X=8, HERD_Y=1, VECTOR_SIZE=16` (NPU2 bf16) — the best config.

---

## Tolerances & reference

Element-wise over the **full output** against an FP32 reference: every element must pass `|a−e| ≤ atol + rtol·|e|`.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 5e-2 |

- **Reference** = CPU FP32 add (`a.astype(f32) + b.astype(f32)`, cast to bf16) — the standard way a GPU/HF bf16 elementwise op is verified. Inputs are `randn` (seed 0).
- **Matches the GPU op exactly.** PyTorch's `torch.add` on bf16 tensors routes through `TensorIterator`, which for a pure element-wise op computes each output as the sum of the two corresponding inputs (no cross-element accumulation) and stores it back in bf16 — i.e. an f32 sum of two exact-bf16 operands, rounded once to bf16. The NPU kernel does the same single rounding, so it is numerically equivalent to `torch.add` (the residual add in a HF/vLLM decoder is exactly this: `residual + hidden_states`, a plain bf16 `torch.add`). The FP32-accumulator concern in PyTorch's `AccumulateType.h` applies only to *reductions* (`torch.sum`), not to element-wise add — there is nothing to accumulate here.
- `rtol = 1.6e-2` is PyTorch / vLLM's canonical bf16 tolerance. `atol = 5e-2` covers the worst-case single-element bf16 output rounding (`abs_err max ≈ 3.1e-2`, one ULP at the largest sums). With no accumulation, `mean_rel_L1 = 1.9e-3` sits far inside `rtol`.

---

## Tested shapes

Shapes verified on NPU2 (bf16). **Best config is `herd_x=8, herd_y=1, tile_n=2048` for every shape** (the shim-DMA-channel limit caps the herd at one 8-column row — see [Parameters & constraints](#parameters--constraints)). `N = 4194304` is the 2048×2048 residual-add scale of llama-3.2-1B prefill (flattened). Throughput is bandwidth (memory-bound). `mean_rel_L1` is vs an FP32 reference.

| N | (as 2-D) | best config (herd_x/herd_y/tile_n) | latency | bandwidth | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|---|---|
| 1048576 | 1024×1024 | 8/1/2048 | 175 µs | 36.0 GB/s | 1.9e-3 | 3.1e-2 | ✅ |
| 2097152 | — | 8/1/2048 | 277 µs | 45.4 GB/s | 1.9e-3 | 3.1e-2 | ✅ |
| 4194304 | 2048×2048 | 8/1/2048 | 437 µs | 57.7 GB/s | 1.9e-3 | 3.1e-2 | ✅ |
| 8388608 | — | 8/1/2048 | 798 µs | **63.0 GB/s** | 1.9e-3 | 3.1e-2 | ✅ |
| 1835008 | 2048×896 | 8/1/2048 | — | (mem-bound) | 1.9e-3 | 3.1e-2 | ✅ Qwen2.5-0.5B residual (seq·emb) |
| 3145728 | 2048×1536 | 8/1/2048 | — | (mem-bound) | 1.9e-3 | 3.1e-2 | ✅ Qwen2.5-1.5B residual (seq·emb) |

> The 1835008 row is Qwen2.5-0.5B's prefill residual-add scale (seq·emb = 2048·896); the 3145728 row is Qwen2.5-1.5B's (seq·emb = 2048·1536). Same best config, bit-identical 1.9e-3.

> The 4194304 row is the llama-3.2-1B prefill residual-add scale (the fused `o_ffn` variant does the same math on a 2-D `[2048,2048]` layout — see Builder). All shapes use the same best config; bandwidth rises with N as fixed launch overhead amortizes (36 → 63 GB/s). Accuracy is bit-identical across all shapes and herd configs.

**Reading the table**:
- **Memory-bound**: latency is gated by DMA. The kernel moves `3·N·2` bytes (a+b in, c out, bf16); at N=4M that is 25.2 MB in 437 µs ≈ 57.7 GB/s — the highest bandwidth in the registry (purest streaming, no reduction/broadcast). Throughput is bandwidth, not GFLOP/s.
- **Accuracy** `mean_rel_L1 = 1.9e-3` — the cleanest kernel; set by the single bf16 rounding, not the tile config.

---

## Tunable space & performance

The full tunable space is `(herd_x ≤ 8, herd_y ≤ 4, tile_n, vector_size)`, but two things collapse it to a single best config:

**1. `herd_y > 1` cannot place** — the 3-DMA-per-tile shim-channel demand exceeds the hardware limit. So the herd is capped at one row (`herd_y = 1`), max 8 tiles — the kernel **cannot fill the 32-tile array** (unlike GEMM/FA). Sweep at N=4194304:

| herd | tiles | result |
|---|---|---|
| 8×1 | 8 | ✅ **57.7 GB/s** (best) |
| 2×4 | 8 | ✅ 55.0 GB/s |
| 8×2 | 16 | ❌ out of shim DMA channels |
| 8×4 | 32 | ❌ out of shim DMA channels |
| 4×4 | 16 | ❌ out of shim DMA channels |

**2. Within `herd_y = 1`, more columns scale near-linearly** (each column streams its chunk through its own DMA). Sweep of `herd_x` at N=4194304:

| herd_x | latency | bandwidth | speedup vs herd_x=1 |
|---|---|---|---|
| 1 | 2759 µs | 9.1 GB/s | 1.0× |
| 2 | 1327 µs | 19.0 GB/s | 2.1× |
| 4 | 706 µs | 35.7 GB/s | 3.9× |
| 8 | **437 µs** | **57.7 GB/s** | **6.3×** |

**3. `tile_n` has a small monotonic effect** (3-repeat median bandwidth, N=4194304): `256` → 55.4, `512` → 56.2, `1024` → 56.7, `2048` → 57.5, `4096` → 57.2 GB/s — larger blocks amortize DMA-launch overhead, saturating by `tile_n = 2048` (~4% over the smallest). `tile_n ≥ 8192` overflows L1 (ping-pong). So `tile_n = 2048` is best (also the default).

So **`herd_x = 8, herd_y = 1, tile_n = 2048` is the best config** for every shape — full chip *width* (the most the shim DMA channels allow) with the DMA-saturating block size. Accuracy is bit-identical across all configs.

---

## How to reproduce (correctness + performance)

`eltwise_add.py` (compile-and-run mode, the default) runs the **correctness** check via `XRTRunner`: full-output element-wise compare against the FP32 reference; prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...` and `PASS!` / `failed.` Add `--perf-iters N` for latency → bandwidth (10 warmup iters excluded, N timed iters averaged, kernel-only — buffer sync not counted). Every tested-shapes row reproduces by setting `N` (all use the same best config `herd_x=8, herd_y=1, tile_n=2048`):

```bash
cd programming_examples/eltwise_add

# correctness — main shape (N=4194304), best config; compiles + runs on NPU2
make run N=4194304 PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# any tested shape — change N (1048576 / 2097152 / 4194304 / 8388608)
make run N=8388608 PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# performance (latency → bandwidth) — run the script directly with --perf-iters
mkdir -p build_peano && cd build_peano
python3 ../eltwise_add.py --n 4194304 --tile-n 2048 --dtype bf16 \
  --vector-size 16 --herd-x 8 --herd-y 1 --perf-iters 20
# bandwidth = 3·N·2 bytes / latency  (a+b in, c out, bf16)
```

Notes:
- `herd_y` must be 1 (shim DMA channel limit); `herd_x` ≤ 8. An alternative C++ timing path exists via `make profile`.
- If the NPU is shared with other jobs, serialize on-device runs (e.g. with `flock`) so timing measurements aren't perturbed.
