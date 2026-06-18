<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# BF16-in / BF16-out GEMM (NPU2 / AIE2P)

`C[M,N] = A[M,K] @ B[K,N]`, bf16 inputs **and** bf16 output. Unlike f32 output,
bf16 output forces a choice about **where the precision is spent** — that is the
one thing this example makes explicit, via `--high-precision`.

## The precision contract: `--high-precision {true,false}`

The MMA accumulator is always FP32 in hardware. The question is what happens to
the partial sum **between K-tiles**, because the L1 C buffer's dtype = output dtype:

- **`--high-precision true` (default)** — keep an **FP32** accumulator across the
  whole K reduction and cast to bf16 **once** at the end. This is the GPU standard
  (no mid-reduction rounding). `mean_rel_L1 ≈ 9.3–9.7e-3`.
- **`--high-precision false`** — direct-codegen bf16: the L1 C buffer is bf16, so
  the partial sum is **truncated to bf16 once per L2 tile** (`K / tile_k_l2` times).
  Fastest single-launch path but a precision tier worse: `mean_rel_L1 ≈ 1.3–1.9e-2`
  (grows with K / number of L2 tiles). Equivalent to the legacy
  `--direct-codegen --output-dtype bf16`.

## How high-precision is achieved: `--method {auto,fused-cast,drain}`

When `--high-precision true`, the FP32-accumulate result must be cast to bf16. Two
ways, with a measured shape cross-over:

| method | how | best for | tile_m |
|---|---|---|---|
| `fused-cast` | external GEMM writes f32 + a **separate cast launch** (one ELF) | large shapes | 64 (full speed) |
| `drain` | cast inside the GEMM's drain herd (single self-contained launch) | small shapes | 32 (drain buffer caps it) |
| `auto` (default) | `fused-cast` if `M*K*N ≥ 4e9` else `drain` | — | — |

The cross-over exists because the cast is a fixed-cost launch: large GEMMs amortize
it (fused-cast wins, e.g. Down +23%), small GEMMs don't (drain wins, e.g. 512³,
1024³). The `4e9` threshold separates all swept shapes correctly.

## Measured (NPU2, herd 8×4, registry-best tiles, this example's `run.py`)

GFLOPS, all PASS. high-precision GFLOPS for fused-cast includes the cast launch.
**Bold** = faster of the two high-precision methods, which `auto` picks.

| (M,K,N) | high-prec fused-cast | high-prec drain | low-prec direct | mean_rel_L1 (high / low) |
|---|---|---|---|---|
| 2048×8192×2048 (Down) | **8898** | 7234 | 5592 | 9.7e-3 / 1.9e-2 |
| 2048×2048×8192 (Gate/Up) | **6893** | 5784 | 5287 | 9.7e-3 / 1.3e-2 |
| 2048×2048×2048 (O) | **6215** | 6025 | 5230 | 9.7e-3 / 1.3e-2 |
| 4096×4096×4096 | **8423** | 7002 | 5509 | 9.9e-3 / 1.5e-2 |
| 2048×2048×512 (K/V) | 4083 | **5626** | 4765 | 9.7e-3 / 1.3e-2 |
| 1024×1024×1024 | 2502 | **4637** | 4456 | 9.9e-3 / 1.1e-2 |
| 512×512×512 | 482 | **1703** | 1750 | 9.7e-3 / 1.0e-2 |

The `auto` threshold `M*K*N ≥ 4e9` picks the faster high-precision method for all 7
shapes above (fused-cast for Down/Gate/Up/O/4096³; drain for K/V/1024³/512³).
Note at the tiniest shape (512³) low-precision direct (1750) edges drain (1703) —
but it's a precision tier worse; `--high-precision true` stays the safe default.

## Run

```bash
# default: high-precision true, method=auto (Down -> fused-cast)
make run M=2048 K=8192 N=2048

# force drain (small shape); pass TILE_M=32 (drain caps tile_m at 32)
make run METHOD=drain M=1024 K=1024 N=1024 TILE_M=32

# low-precision direct-codegen bf16 (fastest single-launch, lower precision)
make run HIGH_PRECISION=false M=2048 K=8192 N=2048
```

If you want a **f32** output (always FP32-accumulate, no method choice), use the
sibling `bf16_in_fp32_out/`.

## Tolerances (tier-aware)

Element-wise `np.isclose`, 0% mismatch allowed. `rtol` anchors to PyTorch's bf16
standard (`torch.testing.assert_close`: bf16 `rtol=1.6e-2`) for **both** tiers —
the output is bf16, so per-element relative error is bounded by bf16 rounding
(~2⁻⁸) regardless of accumulator precision. **`atol` is what encodes the precision
tier** (measured worst-case `abs_err`, Down 2048×8192×2048):

| `--high-precision` | tier | rtol | atol | worst abs_err | margin |
|---|---|---|---|---|---|
| `true` (fused-cast / drain) | FP32-accumulate | 1.6e-2 | 1.5e-3 | ~6.1e-4 | ~2.5× |
| `false` (direct, per-L2-tile trunc) | bf16-accumulate | 1.6e-2 | 4e-3 | ~2.4e-3 | ~1.6× |

The high-precision `atol` (1.5e-3) is deliberately **below** the low-precision
worst-case (2.4e-3): a high-precision path that silently regressed to bf16
truncation would **fail** the gate. `mean_rel_L1` (printed, 9.7e-3 high / 1.9e-2
low) is diagnostic only — it does not gate.

Legacy flags (`--direct-codegen`, `--emit-external`, `--fused-bf16-cast`,
`--drain-chunks`) are kept in `run.py` as advanced aliases. See
`kernel_registry/details/GEMM_bf16_in_bf16_out.md`.
