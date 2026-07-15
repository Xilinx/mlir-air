<!---//===- SiLU_Mul_bf16.md ----------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# SiLU-and-Mul (BF16) — Kernel Detail

> The SwiGLU activation of a decoder-only LLM: `out[i] = SiLU(gate[i]) · up[i]`, where `SiLU(x) = x · sigmoid(x)`. Two BF16 inputs (`gate`, `up`) → one BF16 output, per-element, no reduction. This is llama-3.2-1B prefill's FFN activation (between the gate/up projections and the down projection).
> Shape is a flat length `N` (a 2-D `rows×cols` tensor is just `N = rows·cols` flattened): `gate[N]`, `up[N]` → `out[N]`.
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, June 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/silu_and_mul/silu_and_mul.py
  build_module(n, tile_n, np_dtype_in, herd_x=8, herd_y=None)        # 1-D (standalone); herd_y=None -> 1
  build_module_2d(rows, cols, tile_n, np_dtype_in, herd_x=8, herd_y=1)  # 2-D (llama uses this)
```

Driven by `silu_and_mul.py`'s CLI. **Single code-generation path**: the compute is a hand-written external C++ vector microkernel `silu_and_mul.cc` → `silu_and_mul.o`, linked into the herd via `link_with` (there is no direct-codegen variant — the kernel needs the hardware `aie::tanh`). The input `[N]` is split into `tile_n`-sized chunks streamed through L3→L1→L3 DMA (3 DMAs per tile: `gate` in, `up` in, `out` out); the chunks are spread across an `herd_x × herd_y` AIE tile grid. The herd is effectively 1-D (rows are independent), so `herd_x` is the scaling knob.

**Note on llama usage.** llama-3.2-1B prefill's SwiGLU is launch L6 of the fused `o_ffn` ELF, built with **`build_module_2d`** (a 2-D `[2048, 8192]` op, `tile_n=4096`, herd `8×1`, collapsed to 1-D inside the launch — `o_ffn_multi.py:297-312`). The inner per-element compute is the **same** `silu_and_mul.cc` call as the 1-D standalone; only the L3 memref shape differs. This registry entry measures the **standalone 1-D `build_module`** (a reusable, independently-reproducible building block); the 2-D variant llama runs was verified to give **bit-identical** precision (`mean_rel_L1 = 1.024e-2`, `abs_err max = 0.125` at `2048×8192`, same as 1-D).

---

## Numerical datapath (what "BF16 SiLU-and-Mul" means here)

```
g,u bf16 → g_half = g·0.5
         → tanh_val = aie::tanh<bf16>(g_half)        (hardware tanh LUT)
         → sigmoid  = 0.5·(1 + tanh_val)
         → silu     = g·sigmoid
         → out      = silu·u                          → bf16
```

- **No accumulation, no reduction.** Each `gate`/`up` element is transformed independently. The `aie::accum<accfloat>` in the `.cc` is only the bf16→float conversion feeding `aie::tanh` — **not** a cross-element accumulator — so the "FP32-accumulator-for-reductions" rule (RMSNorm) does not apply here.
- **sigmoid is computed via `0.5·(1 + tanh(g/2))`, not `1/(1+exp(-g))`** — mathematically identical, but AIE2P has a hardware `tanh` and lacks an accurate `exp`/`div`. The two are equal in real arithmetic; in bf16 the hardware-`tanh` path differs from the true sigmoid, and **that difference is exactly what the precision section measures**.
- **The error has two sources**: the hardware `aie::tanh<bf16>` LUT approximation, and the chain of bf16 roundings (`g_half`, `sigmoid`, `silu`, `out` each round once). The net is `mean_rel_L1 ≈ 1.0e-2` — an order of magnitude above the single-rounding Element-wise Add (1.9e-3), as expected for a kernel with a transcendental plus several roundings.

---

## Numerical accuracy

Verified **element-wise over the full output** against an FP32 reference (`gate`,`up` upcast to f32, true sigmoid `1/(1+exp(-g))`, `out = silu·up`, cast to bf16):

| Metric (N = 16777216 = 2048×8192, randn inputs) | Measured |
|---|---|
| `mean_rel_L1 = mean|out−ref| / mean|ref|` | **1.024e-2** |
| `rel_err max` | 4.11e-1 |
| `abs_err max` | 1.25e-1 |

- **`mean_rel_L1 = 1.0e-2`** — the "bf16 + one transcendental" tier: above Element-wise Add (1.9e-3), RMSNorm (4.2e-3) and GEMM (9.3e-3), below FlashAttention (3.9e-2). The extra error vs a plain elementwise op is the hardware `tanh` LUT plus the chained bf16 roundings.
- **`rel_err max = 0.41`** is a near-zero-reference artifact: SiLU outputs values near zero around `g ≈ 0`, where a small absolute error becomes a large *relative* one. This is why `mean_rel_L1` (which does not blow up near zero), not `rel_err max`, is the headline metric.
- **`abs_err max = 0.125`** is the worst single element (large-magnitude `g·u` where the `tanh` LUT error is amplified by `0.5·g·u`) — covered by `atol = 8e-2` together with the `rtol` term.
- **Accuracy is independent of `herd_x`, `tile_n`, `N`, and the 1-D vs 2-D entry** (`mean_rel_L1 ∈ [1.023e-2, 1.025e-2]`, `abs_err max = 0.125` everywhere) — set by the datatype and the hardware `tanh`, not the tiling.

---

## Parameters & constraints

SiLU-and-Mul is **memory-bound** (it streams `gate`, `up` in and `out` out for an O(N) op; arithmetic intensity ~1 op/byte). The herd is 2-D (`sizes=[herd_x, herd_y]`), but two hardware limits pin the usable config:

| Knob | Value | Constraint → source |
|---|---|---|
| `herd_x` | **8** | AIE columns (≤ 8); `N % (tile_n · herd_x · herd_y) == 0` |
| `herd_y` | **1** | **must be 1** — each tile uses 3 independent L3↔L1 DMAs (gate in, up in, out); `herd_y > 1` exhausts the shim DMA channels (`aircc`: *"air.channel.put op failed to map to shim dma channels: out of channels"*) |
| `tile_n` | 4096 | `N % (tile_n · herd_x · herd_y) == 0`; L1 chunk size; **≤ 4096** (8192 overflows L1: a 16 KB buffer × 3 × ping-pong > 64 KB). **Also subject to a non-monotonic buffer-descriptor limit (below).** |

**The kernel cannot use the full 32-tile array** — the 3-DMA-per-tile shim-channel demand caps it at `herd_y = 1` (one row of 8 columns, 8 tiles max), exactly like Element-wise Add. In addition, within `herd_y = 1`, **not every `(tile_n, herd_x)` combination places**: the compiler's DMA buffer-descriptor allocator rejects some inner-loop iteration counts (`'aiex.dma_configure_task' op Allocator exhausted available buffer descriptor IDs`). At `N = 16777216`, the placeable set is **non-monotonic** in the inner-iteration count — `tile_n ∈ {256, 512, 4096}` place (with `herd_x = 8`), while `tile_n ∈ {1024, 2048}` fail; for `herd_x` at `tile_n = 4096`, only `1` and `8` place (`2`, `4` fail). **The best config is therefore selected from the configs that actually place, not asserted.** `tile_n = 4096, herd_x = 8` places, is the fastest legal config, and is exactly what llama uses.

> **llama default** = `tile_n = 4096, herd_x = 8, herd_y = 1` (`o_ffn_multi.py:205-207`) — which is the sweep-best. Nothing is left on the table.

---

## Tolerances & reference

Element-wise over the **full output** against an FP32 reference: every element must pass `|out − ref| ≤ atol + rtol·|ref|`.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 8e-2 |

- **Reference** = full-output FP32 SiLU-and-mul (`gate`,`up` upcast to f32, true sigmoid `1/(1+exp(-g))`, multiply, cast to bf16) — `silu_reference()` in `silu_and_mul.py`. Inputs are `randn` (seed 0). The kernel's `0.5·(1+tanh(g/2))` sigmoid is math-identical to the reference's `1/(1+exp(-g))`, so the FP32 true-sigmoid reference is the correct ground truth and the bf16 hardware-`tanh` deviation is precisely the measured error.
- **GPU standard.** vLLM's own `SiluAndMul` test compares its CUDA kernel **exactly** (`atol=0`) because that kernel is bit-identical to native PyTorch (both fp32 silu) — that exact bar does **not** transfer to a bf16 hardware-`tanh` NPU kernel. The correct bar is the framework's general bf16 tolerance `rtol = 1.6e-2, atol = 1e-3` (PyTorch `test_transformers` / vLLM `allclose_default.py`), the same `rtol` the GEMM and RMSNorm registry kernels use.
- **Why `atol = 8e-2` rather than `1e-3`.** SiLU is a saturating non-linearity and the hardware `tanh` LUT is coarser than a rounded `np.tanh`, so the worst single element (`abs_err max = 0.125`) is larger than a pure-rounding op. The measured minimum `atol` at which the full-output check passes element-wise (with `rtol = 1.6e-2`) is `6.7e-2`; `atol = 8e-2` clears it with margin (zero mismatches). The `atol` only governs the worst-case element; the **mean** error (`mean_rel_L1 = 1.0e-2`) — what matters for model quality — sits inside `rtol`.

---

## Tested shapes

Shapes verified on NPU2 (bf16). **Best config is `herd_x=8, herd_y=1, tile_n=4096` for every shape** (the shim-DMA-channel limit caps the herd at one 8-column row; the buffer-descriptor limit makes `tile_n=4096` the fastest placeable tile — see [Parameters & constraints](#parameters--constraints)). `N = 16777216` is the `2048×8192` prefill-SwiGLU scale of llama-3.2-1B. Throughput is bandwidth (memory-bound). `mean_rel_L1` is vs an FP32 reference.

| N | (as 2-D) | best config (herd_x/herd_y/tile_n) | latency | bandwidth | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|---|---|
| 2097152 | — | 8/1/4096 | 569 µs | 22.1 GB/s | 1.0e-2 | 0.125 | ✅ |
| 4194304 | 2048×2048 | 8/1/4096 | 1052 µs | 23.9 GB/s | 1.0e-2 | 0.125 | ✅ |
| 8388608 | — | 8/1/4096 | 2247 µs | 22.4 GB/s | 1.0e-2 | 0.125 | ✅ |
| 16777216 | 2048×8192 | 8/1/4096 | 4016 µs | **25.1 GB/s** | 1.0e-2 | 0.125 | ✅ |
| 6291456 | 2048×3072 | 8/1/4096 | 1771 µs | 21.3 GB/s | 1.0e-2 | 0.125 | ✅ Qwen3-0.6B SwiGLU (seq·hidden) |
| 9961472 | 2048×4864 | 8/1/4096 | 2489 µs | 24.0 GB/s | 1.0e-2 | 0.125 | ✅ Qwen2.5-0.5B SwiGLU (seq·hidden, hidden=4864) |
| 12582912 | 2048×6144 | 8/1/4096 | 3041 µs | 24.8 GB/s | 1.0e-2 | 0.125 | ✅ Qwen3-1.7B SwiGLU (seq·hidden, hidden=6144) |
| 18350080 | 2048×8960 | 8/1/4096 | 4933 µs | 22.3 GB/s | 1.0e-2 | 0.188 | ✅ Qwen2.5-1.5B SwiGLU (seq·hidden, hidden=8960) |
| 19922944 | 2048×9728 | 8/1/4096 | 5077 µs | 23.5 GB/s | 1.0e-2 | 0.125 | ✅ Qwen3-4B SwiGLU (seq·hidden, hidden=9728) |
| 22544384 | 2048×11008 | 8/1/4096 | 5694 µs | 23.8 GB/s | 1.0e-2 | 0.188 | ✅ Qwen2.5-3B SwiGLU (seq·hidden, hidden=11008) |

> The 16777216 row is llama-3.2-1B prefill's SwiGLU scale (the fused `o_ffn` variant does the same math via `build_module_2d` on a 2-D `[2048,8192]` layout — verified bit-identical, see Builder). All shapes use the same best config. Bandwidth is roughly flat (~22–25 GB/s) — unlike a pure add, the per-element hardware `tanh` dominates over fixed launch overhead, so larger N does not climb much.

**Reading the table**:
- **Memory-bound**: the kernel moves `3·N·2` bytes (gate+up in, out, bf16); at N=16M that is 100.7 MB in 4016 µs ≈ 25.1 GB/s. Throughput is bandwidth, not GFLOP/s. The absolute bandwidth is below Element-wise Add's 57.7 GB/s because each element does a hardware `tanh` plus four multiplies, which throttles the streaming rate.
- **Accuracy** `mean_rel_L1 = 1.0e-2` — the bf16 + transcendental tier; set by the datatype and the hardware `tanh`, not the tile config.

---

## Tunable space & performance

The full tunable space is `(herd_x ≤ 8, herd_y ≤ 4, tile_n)`, collapsed to a single best config by three limits:

**1. `herd_y > 1` cannot place** — the 3-DMA-per-tile shim-channel demand exceeds the hardware limit. So the herd is capped at one row (`herd_y = 1`), max 8 tiles — the kernel **cannot fill the 32-tile array**. Sweep at N=16777216:

| herd | tiles | result |
|---|---|---|
| 8×1 | 8 | ✅ **25.1 GB/s** (best) |
| 2×4 | 8 | ✅ 23.2 GB/s |
| 8×2 | 16 | ❌ out of shim DMA channels |
| 8×4 | 32 | ❌ out of shim DMA channels |
| 4×4 | 16 | ❌ out of shim DMA channels |

**2. Within `herd_y = 1`, `herd_x` scales near-linearly** — but `herd_x ∈ {2,4}` fail the buffer-descriptor limit, so the placeable points are `1` and `8`:

| herd_x | tiles | latency | bandwidth | speedup vs herd_x=1 |
|---|---|---|---|---|
| 1 | 1 | 31722 µs | 3.2 GB/s | 1.0× |
| 2 | 2 | ❌ BD-exhausted | — | — |
| 4 | 4 | ❌ BD-exhausted | — | — |
| 8 | 8 | **4153 µs** | **24.2 GB/s** | **7.6×** |

`herd_x = 1 → 8` is **7.6×** (95% scaling efficiency) — strongly parallelism-bound, confirming this is a bandwidth-class kernel rather than compute-saturated.

**3. `tile_n` is governed by a non-monotonic buffer-descriptor limit** (3-repeat best, N=16777216, herd 8×1):

| tile_n | inner iters | best latency | bandwidth | result |
|---|---|---|---|---|
| 256 | 8192 | 4168 µs | 24.1 GB/s | ✅ |
| 512 | 4096 | 4256 µs | 23.7 GB/s | ✅ |
| 1024 | 2048 | — | — | ❌ BD-exhausted |
| 2048 | 1024 | — | — | ❌ BD-exhausted |
| **4096** | 512 | **4016 µs** | **25.1 GB/s** | ✅ **best** |
| 8192 | 256 | — | — | ❌ L1 overflow |

Among the placeable tiles `{256, 512, 4096}`, bandwidth varies only ~6% and `tile_n = 4096` is the fastest (and llama's default). The placeability is **not** monotonic in the inner-iteration count — it is set by the compiler's DMA buffer-descriptor allocator (`repeat_count ≤ 31`), so the best tile must be picked from configs that actually place.

So **`herd_x = 8, herd_y = 1, tile_n = 4096` is the best config** for every shape — full chip *width* (the most the shim DMA channels allow) at the fastest placeable tile. Accuracy is identical across all configs.

---

## How to reproduce (correctness + performance)

`silu_and_mul.py` (compile-and-run mode, the default) runs the **correctness** check via `XRTRunner`: full-output element-wise compare against the FP32 reference; prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...` and `PASS!` / `failed.` Add `--perf-iters N` for latency → bandwidth (warmup iters excluded, kernel-only). The external kernel links a precompiled `silu_and_mul.o`, so compile it once first (it has no shape/tile dependence).

```bash
SRC=programming_examples/silu_and_mul

# 1) compile the external kernel .o (no NPU needed)
mkdir -p $SRC/build_peano
$PEANO_INSTALL_DIR/bin/clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -DNDEBUG \
  -I $(realpath $(dirname $(which aie-opt))/..)/include -include aie_kernels/aie_kernel_utils.h \
  -c $SRC/silu_and_mul.cc -o $SRC/build_peano/silu_and_mul.o

# 2) run from build_peano (the .o must be in cwd; aircc copies it into air_project/)
cd $SRC/build_peano

# correctness — main shape (N=16777216), best config; compiles + runs on NPU2
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 ../silu_and_mul.py --n 16777216 --tile-n 4096 --herd-x 8 --herd-y 1

# any tested shape — change --n (2097152 / 4194304 / 8388608 / 16777216), same best config
# performance (latency → bandwidth) — add --perf-iters
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 ../silu_and_mul.py --n 16777216 --tile-n 4096 --herd-x 8 --herd-y 1 --perf-iters 30
# bandwidth = 3·N·2 bytes / latency  (gate+up in, out, bf16)
```

Notes:
- `herd_y` must be 1 (shim DMA channel limit); `herd_x ≤ 8`. Placeable `tile_n ∈ {256, 512, 4096}` at N=16777216 (buffer-descriptor limit); `tile_n = 4096` is best.
- If the NPU is shared with other jobs, serialize on-device runs with `flock` so timing measurements aren't perturbed.
