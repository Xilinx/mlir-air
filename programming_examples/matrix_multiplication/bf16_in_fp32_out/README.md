<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# BF16-in / FP32-out GEMM (NPU2 / AIE2P)

`C[M,N] = A[M,K] @ B[K,N]`, with **bf16 inputs** and an **f32 output**.

## Why this is always "high precision"

Because the output is f32, the L1 C **accumulator IS the output** — the K
reduction stays FP32 the entire way and there is **no intermediate bf16
truncation anywhere**. This matches the GPU standard (FP32 accumulate, no
mid-reduction rounding). So there is no `high_precision` knob here: f32 output is
unconditionally high precision. (If you want a **bf16** output with the same
FP32-accumulate precision, use the sibling `bf16_in_bf16_out/` with
`--high-precision true`.)

Measured `mean_rel_L1 ≈ 9.3e-3` (the residual is the BFP16-emulated 8×8×8 MMA's
block-quantization + the bf16 *input* rounding, both of which a GPU bf16 GEMM also
has; it is **not** from accumulation).

## The one knob: `--codegen`

Both backends compute identical math; pick by speed vs. self-containedness:

| `--codegen` | compute kernel | speed (Down 2048×8192×2048) | needs |
|---|---|---|---|
| `external` (default) | hand-tuned `mm_aie2p.cc` → `mm.o` | ~9.8K GFLOPS | `mm.o` (make compile-kernel) |
| `direct` | compiler codegen from a transform script | ~6.0K GFLOPS | PEANO only (self-contained) |

### Measured (NPU2, herd 8×4, registry-best tiles, this example's `run.py`)

GFLOPS, all PASS, `mean_rel_L1` 9.3–9.5e-3 (FP32 accumulate confirmed). external
beats direct 1.5–1.7× across the board.

| (M,K,N) | external | direct |
|---|---|---|
| 2048×8192×2048 (Down) | 9797 | 6010 |
| 2048×2048×8192 (Gate/Up) | 8278 | 5582 |
| 2048×2048×2048 (O) | 8508 | 5516 |
| 4096×4096×4096 | 9329 | 5791 |
| 2048×2048×512 (K/V) | 7342 | 4896 |
| 1024×1024×1024 | 6256 | 4413 |
| 512×512×512 | 1791 | 1536 |

## Run

```bash
# external (default), Down shape, full-chip 8x4 herd, registry-best tiles
make run M=2048 K=8192 N=2048

# direct codegen (self-contained, no mm.o)
make run CODEGEN=direct M=2048 K=8192 N=2048

# or drive run.py directly (cwd = build_peano so mm.o is found)
make compile-kernel
cd build_peano
python3 ../run.py --arch aie2p --codegen external \
  --m 2048 --k 8192 --n 2048 \
  --tile-m 64 --tile-k-l2 256 --tile-k-l1 32 --tile-n 128 \
  --herd-m 8 --herd-n 4 --compile-mode compile-and-run --perf-iters 20
```

## Tolerances

`rtol=1.6e-2, atol=1.5e-3` (element-wise `np.isclose`, 0% mismatch allowed).

`rtol` anchors to PyTorch's bf16 standard (`torch.testing.assert_close`: bf16
`rtol=1.6e-2`) — even though the output is *stored* as f32, the GEMM **computes**
in bf16 (bf16 inputs + BFP16-emulated MMA), so the per-element error floor is
bf16's, not f32's `1.3e-6`. `atol=1.5e-3` encodes the FP32-accumulate tier:
measured worst-case `abs_err ≈ 5.8e-4` (Down 2048×8192×2048, ~2.5× margin), tight
enough to reject the bf16-truncation tier (`abs_err ≈ 2.4e-3`). `mean_rel_L1`
(printed, ~9.3e-3) is diagnostic only — it does not gate.

## Tile config

`tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=128`, herd `8×4` (full chip) is
registry-best for large GEMMs. `tile_n=128` is the dominant performance knob;
`tile_k_l2` is secondary. See `kernel_registry/details/GEMM_bf16_in_fp32_out.md`.
