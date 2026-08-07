<!---//===- GELU_bf16.md --------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# GELU — tanh approximation (BF16) — Kernel Detail

> The GELU activation in its **tanh approximation** (GELUTanh):
> `out[i] = 0.5 · x[i] · (1 + tanh(√(2/π) · (x[i] + 0.044715 · x[i]³)))`.
> One BF16 input → one BF16 output, per-element, no reduction. This is the SmolVLA
> vision encoder (SigLIP) MLP activation, applied between the `fc1` and `fc2`
> projections. Shape is a flat length `N` (a 2-D `rows×cols` tensor is just
> `N = rows·cols` flattened): `x[N]` → `out[N]`.
>
> Companion: [`../supported_kernels.md`](../supported_kernels.md) · [`../README.md`](../README.md)
> **Scope: NPU2 (Strix / AIE2P) only.** Measured on real NPU2, July 2026. Reproduce commands in "How to reproduce" below.

---

## Builder

```
programming_examples/gelu/gelu.py
  build_module(n, tile_n, np_dtype_in, vector_size=16, herd_x=1, herd_y=None)   # 1-D; herd_y=None -> 2 (original 1x2 herd, preserved for the lit test)
```

Driven by `gelu.py`'s CLI. **Single code-generation path**: the compute is emitted
**inline in MLIR** — the `tanh` maps to the hardware `__builtin_aie2p_tanh` via
`math.tanh`, so there is **no external `.o`** (unlike SiLU-and-Mul, which links a
C++ microkernel for `aie::tanh`; GELU is direct-codegen like Element-wise Add).
The input `[N]` is split into `tile_n`-sized chunks streamed through L3→L1→L3 DMA
(**2 DMAs per tile**: `x` in, `out` out); the chunks are spread across an
`herd_x × herd_y` AIE tile grid.

**Note on SmolVLA usage.** SmolVLA's SigLIP vision encoder uses `nn.GELU(approximate="tanh")`
(GELUTanh) as its MLP activation. It acts on the `fc1` output `(seq=1024, intermediate=3072)`,
i.e. `N = 1024·3072 = 3145728` flattened. The `fc1` GEMM itself is the `1024×768×3072`
row in the GEMM registry; this kernel is the activation immediately after it. The
reference is verified against the **tanh** approximation (not the exact erf-GELU) to
match what SmolVLA runs — an erf reference would introduce a ~1e-3 systematic bias.

---

## Numerical datapath (what "BF16 GELU-tanh" means here)

```
x bf16 → x²         = x·x
       → x³         = x·x²
       → inner      = x + 0.044715·x³
       → scaled     = √(2/π)·inner
       → tanh_val   = tanh(scaled)      [__builtin_aie2p_tanh]   (hardware tanh LUT)
       → out        = 0.5·x·(1 + tanh_val)                       → bf16
```

- **No accumulation, no reduction.** Each element is transformed independently, so
  the "FP32-accumulator-for-reductions" rule (RMSNorm) does not apply.
- **tanh is the hardware `__builtin_aie2p_tanh`** — AIE2P has an accurate hardware
  `tanh` and lacks an accurate `exp`/`erf`, so the tanh-approximation GELU is the
  natural hardware fit (and is exactly the variant SmolVLA / HF SigLIP use).
- **The error has two sources**: the hardware `tanh` LUT approximation, and the
  chain of bf16 roundings. The net is `mean_rel_L1 ≈ 8.4e-3` — the "bf16 + one
  transcendental" tier (the same class as SiLU-and-Mul, slightly cleaner because
  the tanh argument `√(2/π)·(x+βx³)` is scaled down, so the LUT error is smaller,
  and there is no `0.5·g·u` amplification term).

---

## Numerical accuracy

Verified **element-wise over the full output** against an FP32 reference (`x` upcast
to f32, tanh-approximation GELU evaluated in f32, cast to bf16):

| Metric (N = 3145728 = 1024×3072, randn inputs) | Measured |
|---|---|
| `mean_rel_L1 = mean|out−ref| / mean|ref|` | **8.41e-3** |
| `rel_err max` | 1.0 |
| `abs_err max` | 1.56e-2 |

- **`mean_rel_L1 = 8.4e-3`** — the "bf16 + one transcendental" tier: below
  SiLU-and-Mul (1.0e-2), above RMSNorm (4.2e-3) and Element-wise Add (1.9e-3). The
  error is the hardware `tanh` LUT plus the chained bf16 roundings.
- **`rel_err max = 1.0`** is a near-zero-reference artifact: GELU outputs values near
  zero around `x ≈ 0` and for large-negative `x`, where a small absolute error becomes
  a large *relative* one. This is why `mean_rel_L1` (which does not blow up near zero),
  not `rel_err max`, is the headline metric.
- **`abs_err max = 1.56e-2`** is the worst single element — a single bf16 output ULP at
  large `|x|` (mean|ref|≈0.4, elements up to ~4 whose bf16 ULP ≈ 1.5e-2). Much smaller
  than SiLU's 0.125 (GELU has no `0.5·g·u` amplification) — covered by `atol = 5e-2`.
- **Accuracy is independent of `herd_x`, `herd_y`, `tile_n`, and `N`**
  (`mean_rel_L1 ∈ [8.406e-3, 8.412e-3]`, `abs_err max = 1.56e-2` everywhere) — set by
  the datatype and the hardware `tanh`, not the tiling.

---

## Parameters & constraints

GELU is **memory-bound** (it streams `x` in and `out` out for an O(N) op; arithmetic
intensity ~1.5 op/byte). The herd is 2-D (`sizes=[herd_x, herd_y]`); the key property
is that GELU is a **single-input** op, so each tile uses **only 2 DMAs** (x in, out out):

| Knob | Value | Constraint → source |
|---|---|---|
| `herd_x` | **8** | AIE columns (≤ 8); `N % (tile_n · herd_x · herd_y) == 0` |
| `herd_y` | **2** | AIE rows (≤ 4). **`herd_y > 1` places** — with only 2 DMAs/tile the shim-channel budget allows a second row, so the herd reaches **16 tiles** (8×2 or 4×4). This is unlike Element-wise Add / SiLU-and-Mul, whose 3-DMA/tile demand caps them at one 8-tile row. `herd_y = 4` places only up to 16 tiles (4×4); the full 32-tile 8×4 does **not** place. |
| `tile_n` | 4096 | `N % (tile_n · herd_x · herd_y) == 0`; L1 chunk size. Placeable across `{512…6144}` (bandwidth varies < 3%). |

**GELU can use 16 tiles — twice the 8-tile ceiling of the 3-DMA elementwise
kernels** — because its single input halves the per-tile shim-DMA demand. The best
config is `herd_x=8, herd_y=2, tile_n=4096` (16 tiles, 24.1 GB/s), selected from the
configs that actually place. `herd_x` scales near-linearly (7.6× from 1→8 within
`herd_y=1`); adding the second row (8×1 → 8×2) is a further ~1.8×.

> **SmolVLA integration default** should be `herd_x=8, herd_y=2` (the sweep-best).
> The builder's own default (`herd_x=1, herd_y=2`) exists only to preserve the
> original 1×2 herd for the lit test — integrating SmolVLA vision must pass `8×2`
> explicitly, or it leaves up to 1.8× bandwidth on the table.

---

## Tolerances & reference

Element-wise over the **full output** against an FP32 reference: every element must
pass `|out − ref| ≤ atol + rtol·|ref|`.

| Output dtype | rtol | atol |
|---|---|---|
| bf16 | 1.6e-2 | 5e-2 |

- **Reference** = full-output FP32 GELU-tanh (`x` upcast to f32, `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))`,
  cast to bf16) — `gelu_reference()` in `gelu.py`. Inputs are `randn` (seed 0). The
  reference uses the **tanh** approximation (matching the NPU kernel and SmolVLA's
  GELUTanh), not the exact erf-GELU.
- **GPU standard.** HF SigLIP's MLP uses `nn.GELU(approximate="tanh")`; the framework's
  general bf16 activation tolerance is `rtol = 1.6e-2, atol = 1e-3` (PyTorch
  `test_transformers` / vLLM `allclose_default.py`), the same `rtol` the GEMM, RMSNorm,
  and SiLU-and-Mul registry kernels use.
- **Why `atol = 5e-2` rather than `1e-3`.** The hardware `tanh` LUT plus bf16 output
  rounding produces a worst single element of `abs_err max = 1.56e-2` (size/config
  independent). `atol = 5e-2` (the same convention as RoPE / RMSNorm / Element-wise
  Add) clears it with ~3× margin (zero mismatches). It is **tighter than SiLU's 8e-2**
  because GELU has no `0.5·g·u` amplification, so its worst element (0.0156) is far
  smaller than SiLU's (0.125). The `atol` only governs the worst-case element; the
  **mean** error (`mean_rel_L1 = 8.4e-3`) sits inside `rtol`.

---

## Tested shapes

Shapes verified on NPU2 (bf16). **Best config is `herd_x=8, herd_y=2, tile_n=4096`
(16 tiles) for every shape** — GELU's single input (2 DMAs/tile) lets the herd use a
second row where the 3-DMA elementwise kernels cannot (see
[Parameters & constraints](#parameters--constraints)). `N = 3145728` is the
`1024×3072` SmolVLA SigLIP vision MLP activation scale. Throughput is bandwidth
(memory-bound). `mean_rel_L1` is vs an FP32 reference.

| N | (as 2-D) | best config (herd_x/herd_y/tile_n) | latency | bandwidth | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|---|---|
| 1048576 | — | 8/2/4096 | 235 µs | 17.8 GB/s | 8.4e-3 | 1.56e-2 | ✅ |
| 2097152 | — | 8/2/4096 | 379 µs | 22.1 GB/s | 8.4e-3 | 1.56e-2 | ✅ |
| 3145728 | 1024×3072 | 8/2/4096 | 522 µs | 24.1 GB/s | 8.4e-3 | 1.56e-2 | ✅ SmolVLA vision MLP GELU (seq 1024 · intermediate 3072) |
| 4194304 | 2048×2048 | 8/2/4096 | 672 µs | 25.0 GB/s | 8.4e-3 | 1.56e-2 | ✅ |
| 8388608 | — | 8/2/4096 | 1244 µs | **27.0 GB/s** | 8.4e-3 | 1.56e-2 | ✅ |

> The 3145728 row is SmolVLA's SigLIP vision MLP activation scale (seq 1024 ·
> intermediate 3072). Bandwidth climbs with N (17.8 → 27.0 GB/s) as the fixed launch
> overhead is amortized. All shapes use the same best config; accuracy is bit-identical
> (`mean_rel_L1 ≈ 8.4e-3`, `abs_err max = 1.56e-2`) across every config and shape.

**Reading the table**:
- **Memory-bound**: the kernel moves `2·N·2` bytes (x in, out; bf16); at N=3145728 that
  is 12.6 MB in 522 µs ≈ 24.1 GB/s. Throughput is bandwidth, not GFLOP/s. The absolute
  bandwidth is close to SiLU-and-Mul's (~25 GB/s) — the hardware `tanh` per element
  throttles the streaming rate — but GELU reaches it with **16 tiles** (SiLU is capped
  at 8).
- **Accuracy** `mean_rel_L1 = 8.4e-3` — the bf16 + transcendental tier; set by the
  datatype and the hardware `tanh`, not the tile config.

---

## Tunable space & performance

The tunable space is `(herd_x ≤ 8, herd_y ≤ 4, tile_n)`. The distinguishing property
vs the other elementwise kernels: **GELU's single input means 2 DMAs/tile, so
`herd_y > 1` places**.

**1. `herd_y = 2` places → 16 tiles (2× the 3-DMA-kernel ceiling).** Sweep at N=3145728:

| herd | tiles | latency | bandwidth | result |
|---|---|---|---|---|
| 8×1 | 8 | 947 µs | 13.3 GB/s | ✅ |
| **8×2** | 16 | **522 µs** | **24.2 GB/s** | ✅ **best** |
| 4×4 | 16 | 527 µs | 23.8 GB/s | ✅ (= 8×2, same 16 tiles) |
| 8×4 | 32 | — | — | ❌ does not place (32 tiles) |
| 2×8 / 1×8 | — | — | — | ❌ `herd_y > 4` (only 4 AIE rows) |

**2. Within `herd_y = 1`, `herd_x` scales near-linearly** — and unlike SiLU, **every**
`herd_x ∈ {1,2,4,8}` places (2 DMAs/tile is easy on the buffer-descriptor allocator):

| herd_x | tiles | latency | bandwidth | speedup vs herd_x=1 |
|---|---|---|---|---|
| 1 | 1 | 7181 µs | 1.8 GB/s | 1.0× |
| 2 | 2 | 3794 µs | 3.3 GB/s | 1.9× |
| 4 | 4 | 1834 µs | 6.9 GB/s | 3.9× |
| 8 | 8 | 947 µs | 13.3 GB/s | **7.6×** |

`herd_x = 1 → 8` is **7.6×** (95% scaling efficiency); adding the second row
(8×1 → 8×2) is a further ~1.8×, for **13.6×** total (1×1 → 8×2).

**3. `tile_n` is nearly inert** (herd 8×2, N=3145728, 3-repeat best):

| tile_n | best latency | bandwidth | result |
|---|---|---|---|
| 512 | 522 µs | 24.1 GB/s | ✅ |
| 1024 | 520 µs | 24.2 GB/s | ✅ |
| 2048 | 520 µs | 24.2 GB/s | ✅ |
| **4096** | 522 µs | 24.1 GB/s | ✅ best (builder convention) |
| 6144 | 527 µs | 23.6 GB/s | ✅ |

All placeable; bandwidth varies < 3%. `tile_n = 4096` matches the registry convention.

So **`herd_x = 8, herd_y = 2, tile_n = 4096` is the best config** for every shape —
16 tiles, twice the ceiling of the 3-DMA elementwise kernels, at the fastest tile.
Accuracy is identical across all configs.

---

## How to reproduce (correctness + performance)

`gelu.py` (compile-and-run mode, the default) runs the **correctness** check via
`XRTRunner`: full-output element-wise compare against the FP32 tanh-GELU reference;
prints `[precision] mean_rel_L1=... | rel_err max=... | abs_err max=... | rtol=... atol=...`
and `PASS!` / `failed.` Add `--perf-iters N` for latency → bandwidth (warmup iters
excluded, kernel-only). GELU is direct-codegen (no external `.o` to precompile).

```bash
SRC=programming_examples/gelu
mkdir -p $SRC/build_peano
# run from build_peano (aircc writes air_project/ there)
cd $SRC/build_peano

# correctness — SmolVLA vision shape (N=3145728), best config; compiles + runs on NPU2
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 ../gelu.py --n 3145728 --tile-n 4096 --herd-x 8 --herd-y 2

# any tested shape — change --n (1048576 / 2097152 / 3145728 / 4194304 / 8388608), same best config
# performance (latency → bandwidth) — add --perf-iters
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 ../gelu.py --n 3145728 --tile-n 4096 --herd-x 8 --herd-y 2 --perf-iters 30
# bandwidth = 2·N·2 bytes / latency  (x in, out, bf16)

# full sweep (herd / tile_n / per-shape):
bash programming_examples/kernel_registry/details/internal_GELU_bf16/scripts/gelu_sweep.sh
bash programming_examples/kernel_registry/details/internal_GELU_bf16/scripts/gelu_refine.sh
```

Notes:
- `herd_y` can be 2 (GELU's 2-DMA/tile allows a second row → 16 tiles); `herd_x ≤ 8`,
  `herd_y ≤ 4`. `herd_x·herd_y = 32` (8×4) does not place.
- The builder default `herd_x=1, herd_y=2` (the original 1×2 herd) is kept for the lit
  test; pass `--herd-x 8 --herd-y 2` for the best config.
- If the NPU is shared with other jobs, serialize on-device runs with `flock` so timing
  measurements aren't perturbed.
