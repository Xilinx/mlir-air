# Matrix Multiplication (bfp16ebs8)

Matrix multiplication with bfp16ebs8 inputs, a port of mlir-aie's
[block-datatype GEMM reference design][ref] to MLIR-AIR. It runs the reference
kernel [`mm_bfp.cc`][kernel] **unmodified** (resolved from the mlir-aie install via `AIEOPT_DIR`) -- this directory contributes only
the AIR data movement around it, plus a small output-conversion kernel.

`A[M,K] x B[K,N] -> C[M,N]`. A and B are bfp16ebs8 (8 scalars share one 8-bit
exponent, so 8 elements occupy 9 bytes); B is consumed transposed. C is
accumulated in bfp16ebs8 by the kernel and converted to bf16 or f32 on the way
out.

NPU2 (Strix) only, and Peano only -- `xchesscc` has no `bfp16ebs8` codegen path.

[ref]: https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/ml/block_datatypes/matrix_multiplication
[kernel]: https://github.com/Xilinx/mlir-aie/blob/main/aie_kernels/aie2p/mm_bfp.cc

## Available Make Targets

- `make run4x4` - Compile and run on NPU with a 4x4 herd (512x512x512 matrix)
- `make run8x4` - 8x4 herd (1024x512x512)
- `make run3x3` - 3x3 herd (576x576x576)
- `make run2x4` - 2x4 herd (512x512x512)
- `make run2x2` - 2x2 herd (512x512x512)
- `make run1x1` - single core (64x64x64)
- `make run_llama_4x4` - LLAMA-3.2-1B GEMM shapes (`_qo`, `_kv`, `_gate_up`; see the K limit below for `_down`)
- `make profile` - Run on hardware and report latency + GFLOPs
- `make sweep4x4` - Measure end-to-end latencies across a range of shapes (256-2048)
- `make print` - Print the generated MLIR without running

## Tile Size Configuration

```bash
TILE_M      # M dimension tile size (default: 64)
TILE_K_L2   # K dimension L2 tile size (default: 128)
TILE_K_L1   # K dimension L1 tile size (default: 64)
TILE_N      # N dimension tile size (default: 64)
```

**Example:**
```bash
make run4x4 TILE_M=128 TILE_K_L2=256 TILE_K_L1=64 TILE_N=64
```

bfp16ebs8 constrains these more than bf16 does:

| constraint | source |
|---|---|
| `TILE_M % 16 == 0` | `mm_bfp.cc` (2x register blocking on `r`) |
| `TILE_M >= 32` | `mm_bfp.cc` runs out of registers at `TILE_M=16` on AIE2P |
| `TILE_N % 64 == 0` | host shuffle (`tileWidth % 64`) + kernel |
| `TILE_K_L1 % 64 == 0` | host shuffle (`tileWidth % 64`) + kernel |
| `K % 32 == 0` | `K*9/8` must be 4-byte aligned for shim BDs |

## Output Data Type

```bash
make run4x4                   # f32 output (default on aie2p)
make run4x4 OUTPUT_DTYPE=bf16 # bf16 output
```

Both go through `bfp_cvt.cc`, which converts the bfp16ebs8 L1 accumulator in
8x8-block order; the de-blocking permute is left to the drain DMA.

## Target Architecture Selection

`AIE_TARGET` exists for symmetry with the bf16 example, but `aie2p` is the only
legal value -- `v8bfp16ebs8` and `mac_8x8_8x8T` are AIE2P-only and NPU1 has no
block-floating-point datapath. `make run4x4 AIE_TARGET=aie2` exits non-zero with
a diagnostic rather than emitting IR that cannot be lowered.

## Note on Numerical Tolerances

The check uses `randn/sqrt(K)` inputs (the bf16 example's generator) and
compares against an f32 matmul on the ORIGINAL floats, so it covers the bfp16
input quantization. Because 8 elements share an exponent, an element next to a
much larger neighbour loses mantissa bits, and that -- not the output dtype --
dominates the error. The reference magnitude is ~`1/sqrt(K)`, which shrinks
with K while the bfp16 error floor does not, so relative error grows with K:

| K | measured `mean_rel_L1` |
|---|---|
| 64 | 2.1e-2 |
| 512 | 6.9e-2 |
| 2048 | 2.7e-1 |

`abs_err max` stays in 8.5e-3 .. 1.5e-2 across every target and both output
dtypes, so the gate is absolute-error driven (`atol=2.5e-2`, ~1.7x headroom).
mlir-aie's own `bfp_test.cpp` avoids this regime by drawing inputs from
`rand()%16`; this example keeps the bf16 generator so the numbers above are the
honest bfp16 error on high-dynamic-range data.

## Accumulator Depth Limit (max usable K)

`mm_bfp.cc` keeps C in bfp16ebs8 *between* K-l1 chunks: after every chunk the
accumulator is re-quantized to an 8-bit mantissa with one shared exponent per 8
elements. Error therefore grows with the number of round-trips,
`K / TILE_K_L1`, and it does so under any input distribution:

| round-trips (`K/TILE_K_L1`) | `mean_rel_L1`, randn | `mean_rel_L1`, `rand()%16` |
|---|---|---|
| 32 (K=2048) | 0.27 | 0.09 |
| 128 (K=8192) | 1.23 | 0.41 |

Past roughly 64 round-trips the output is not usable at any sane tolerance, so
`run_llama_4x4_down` (K=8192, 128 round-trips) is excluded from the aggregate
`run_llama_4x4` target and has no lit test. It is kept as a target for
experimentation, and `run.py` prints a warning above 64 round-trips. Raising
`TILE_K_L1` reduces the count but costs L1 (both A and B tiles scale with it).

This is a property of the reference kernel, not of the AIR port.

## Design Notes

**No bfp16 type in AIR.** mlir-aie has `!aiex.bfp<"v8bfp16ebs8">`, but
`aie-transform-bfp-types` lowers it to `i72` inside `aie.device` -- after every
AIR pass -- and AIR's BD lowering assumes an int-or-float element type. So A and
B cross the AIR boundary as plain `i8` and the kernel reinterprets via
`aie::block_vector<bfp16ebs8>`, the same type-pun
[`bf16_x_bfp16`](../bf16_x_bfp16/) uses for its weights.

**The host shuffle makes the DMAs trivial.** `shuffleMatrixForBfp16ebs8()`
reorders 8x8 sub-tiles *within* each tile box only, leaving the global matrix
row-major. So L3 A/B are plain 2D byte matrices (`[M, K*9/8]`, `[N, K*9/8]`),
L3->L2 is a strided box gather, and L2->L1 is a contiguous copy that lands in
exactly the sub-tile-major order `mm_bfp.cc`'s
`block_vector_input_buffer_stream.seek(z * colA)` expects. Neither of the two
data-layout transforms the reference README calls out as impossible for blocked
types is needed on the AIR side. `bfp16_utils.py` is a NumPy port of those host
helpers, cross-checked byte-for-byte against `helper.h`.

**Three herds.** `herd_init` zeroes the bfp16 L1 accumulator, `herd_compute`
runs the K loop, `herd_drain` converts and copies out. The accumulator is a
segment-shared L1 buffer so it survives across the three invocations.

**Herd geometry is transposed relative to IRON.** AIR's `--herd-m` maps to
columns (M-parallel) and `--herd-n` to rows (N-parallel), so on NPU2
`herd_m <= 8` and `herd_n <= 4`. mlir-aie's `whole_array` is the other way round
(`n_aie_rows = 4` is M-parallel). Irrelevant for square GEMMs; it matters once
`M != N`.

**Small herds need `--runtime-loop-tiling 1`.** `runtime_loop_tiling_sizes`
defaults to `[2, 2]` and is worth ~26% throughput (183us vs 232us at 512^3,
4x4). It is not safe when the herd is small enough that AIR gives the C drain a
single shim channel: the shim-BD fold then collapses that drain across every
launch iteration into one task with one await, so only the first launch
iteration's output is written while all the input fills still run. Observed at
herd 2x2 / 2x1 / 1x2 with more than one launch iteration; herd 4x2 and larger
split C over four channels and are unaffected. `run.py` detects this and drops
to `[1, 1]` with a printed note. Same class as the shim-BD stride-fold and
wait-pairing fixes in mlir-air #1810 / #1815.

## Differences from the bf16 example

- No `--direct-codegen`: there is no MLIR bfp16 type, so the kernel must be
  external.
- No Chess path and no `aie2` path (see above).
- `profile` uses `XRTRunner`'s built-in timing (`--perf-iters`) rather than a
  separate C++ harness, so there is no `test.cpp` / `build-test-exe`.
- No `runner` (air-runner simulation) target.
- The LLAMA targets are 4x4, not 8x4: `M=128` with `herd_m=8` forces
  `TILE_M=16`, below this kernel's register-allocation floor.
