# Flash Attention val_range=4 Investigation Status

## Problem

Flash attention passes at val_range≤2.5 but fails at val_range=4 with 63%
element-wise errors against PyTorch SDPA golden (atol=0.15, rtol=0.04).
IRON's MHA passed 5/5 on this machine (Feb 17, 2026, commit 84d3478) at
val_range=4, seq_len=16384.

## Current kernel state

`attn.cc` is at the git HEAD commit (9fa2e88d). Changes tested but NOT
committed include conv_even rounding mode (`::aie::set_rounding(conv_even)`)
on all 19 extern C functions. The committed code has NO rounding mode set.

## What has been proven

### 1. BFP16 matmul with conv_even is precise enough
The standalone bf16 matmul example (`programming_examples/matrix_multiplication/
bf16/`) with IRON's `mm.cc` kernel (which has `set_rounding(conv_even)` via
`-DROUND_CONV_EVEN`) **PASSES** at val_range=4 with tight tolerance (rtol=0.04,
atol=0.15) when compiled with **numpy random inputs**.

Note: our `mm_aie2p.cc` has NO rounding mode code. IRON's `mm.cc` has it at
lines 223-226 and 286, 314, 340, 358.

### 2. Simulations give 0 errors
- Pure bf16 simulation (per-element rounding): **0 errors** at val_range=4
- BFP16 ebs8 simulation (8-element shared exponent blocks): **0 errors**
- Both row-wise and column-wise BFP16 block orientations: **0 errors**

### 3. Softmax amplifies matmul noise
A score perturbation of σ=1.2 per element reproduces ~63% attention output
errors. Softmax with peaked distribution amplifies BFP16 matmul noise:
- Score perturbation 2.0 (within matmul tolerance 2.56 at score ~32)
- Shifts softmax max weight from 0.28 → 0.74
- Causes output diff ~1.0 (exceeds attention tolerance 0.16)

### 4. DMA tiling matches matmul template
Verified from generated `aie.air.mlir`:
- Q BD1: `[8×8, 64×64, 8×1]` → column-major tiles
- K BD2: `[8×512, 8×8, 8×64, 8×1]` → column-major tiles
- V BD: `[8×8, 64×64, 8×1]` → column-major tiles
- Compute tile S2MM: flat (no receiver-side tiling)
- Output DMA: `[64×8, 8×512, 8×1]` → untiles to row-major
- All match the matmul template's column-major A access pattern

### 5. PASSTHROUGH mode works
With softmax bypassed (all G=1.0), output = mean(V) ≈ 2.0 — correct.
Proves the data flow pipeline is functional.

### 6. conv_even rounding reduces bias but not variance
- Without conv_even: mean bias = -2.50 (direct-codegen), -0.38 (external kernel)
- With conv_even: mean bias = -0.01
- Per-element variance unchanged: still 63% errors

### 7. IRON MHA compilation fails on current setup
IRON's compilation framework (`AieccCompilationRule`) enters an infinite loop
generating `mha.bin`. The xclbin compiles successfully but NPU instruction
generation loops indefinitely. Cannot re-verify IRON's result with current
mlir-aie (March 16 build).

## What remains unresolved

### The 12-36× simulation-to-hardware gap
BFP16 simulation predicts max 0.25 error per matmul element. Hardware
produces 3-9 error per element. No simulation reproduces this.

### transpose_b correctness
First-principles analysis of the matmul template vs DMA tile layout suggests:
- Q@K^T: `transpose_b=false` computes Q@K^T (correct)
- G@V: `transpose_b=true` computes G@V (correct)

But the current code has the OPPOSITE settings (true for Q@K, false for G@V).
Empirically, both settings produce similar error rates (~63% vs ~69%), and
val_range=1 passes with either setting. The first-principles analysis may
have incorrect assumptions about within-tile element ordering.

**This needs empirical verification** by dumping actual matmul output from
the hardware and comparing element-by-element.

### BFP16 input-dependent noise
The standalone matmul test passes with numpy.random inputs but fails at 51%
with torch.rand inputs (same val_range=4, same dimensions). The BFP16 noise
is input-dependent in ways the simulation doesn't capture.

## Next steps (recommended)

### 1. Dump matmul output from minimal flash attention

Create a flash attention config with **1 cascade stage and 1 Q-tile** (to
eliminate cascade complexity):
- LK=64, LKP=64 (single K chunk)
- LQ=64, LQP=64, num_q_tiles=1 (single Q tile) — but min is 4 tiles
- Or: LQ=256, LQP=256, num_q_tiles=4, num_cascade_stages=4 with LK=256

Use DUMP_QK_SCORES mode (NOP softmax, copy G→Gp) to capture the raw
Q@K^T matmul output. Compare element-by-element against expected scores
for the KNOWN K-chunk.

This will definitively answer:
- Is Q@K^T or Q@K being computed?
- What is the actual per-element BFP16 error?
- Does the error match the 63% attention error pattern?

### 2. Verify transpose_b empirically

With the dumped matmul output, check which transpose_b setting produces
output matching the expected Q@K^T product. The DMA tile element ordering
can be verified by checking specific known input/output pairs.

### 3. Fix IRON compilation

Fix the `AieccCompilationRule` infinite loop to re-run IRON's MHA test on
the current mlir-aie build. This verifies whether the gap is specific to
our kernel or affects IRON too.

### 4. Add -DROUND_CONV_EVEN to mm_aie2p.cc

Our `mm_aie2p.cc` has NO rounding mode. Add support matching IRON's `mm.cc`
pattern. This fixes the standalone matmul example for AIE2P.

## Files involved

| File | Status | Notes |
|------|--------|-------|
| `attn.cc` | Modified (uncommitted) | conv_even rounding added via script |
| `attn.py` | Modified (uncommitted) | PyTorch SDPA golden, val_range=2.5, error_threshold=0.005 |
| `xrt_runner.py` | Modified (uncommitted) | error_threshold parameter, per-dim error stats |
| `test_precision/` | Created (untracked) | Standalone tests — DMA tiling issues, unreliable |
| `TILE_ORDER_BUG.md` | Created (untracked) | Investigation notes (tile order hypothesis disproven) |
| `PRECISION_ANALYSIS.md` | Created (untracked) | IRON comparison analysis |
| `BUGFIX_SUMMARY.md` | Created (untracked) | BUG1-4 summary from earlier session |
| `/tmp/iron_mha_precision_analysis.md` | Exists | IRON precision techniques documentation |
| `/tmp/iron_vs_mlirair_precision_comparison.md` | Exists | IRON vs mlir-air comparison |

## Key reference: IRON's approach

From `/tmp/iron_mha_precision_analysis.md`:
1. `set_rounding(conv_even)` in every matmul function
2. `accfloat` accumulators for ALL softmax intermediates
3. Combined `inv_scale * log2e` applied once per row inside softmax
4. Two-pass softmax: pass 1 finds max of SCALED scores, pass 2 computes exp+sum
5. `exp2` input promoted to f32 via `.to_vector<float>()`
6. Rescaling `exp(m_{i-1} - m_i)` computed in f32

IRON MHA test: seq_len=16384, d=64, num_heads=1, val_range=4,
rel_tol=4%, abs_tol=0.15, error_threshold=0.5%.
Result: **5/5 PASS** (Feb 17, 2026).
