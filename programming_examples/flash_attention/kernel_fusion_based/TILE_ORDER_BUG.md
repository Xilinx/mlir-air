# Flash Attention val_range=4 Failure: Investigation Notes

**UPDATE: The tile order hypothesis was DISPROVEN.** Re-analysis of the AIE BD
dimension iteration order showed that the DMA DOES produce column-major tiles,
matching the original `attn.cc` template. The row-major fix made things worse
(89% errors vs 63%). The original code was reverted.

The root cause of the 63% error at val_range=4 remains under investigation.

---

# Original (incorrect) analysis below — kept for reference

## Summary

The flash attention kernel produces 63% errors at val_range=4 due to a
**tile ordering mismatch** between the DMA tiling descriptor and the matmul
kernel template. The DMA delivers Q/K tiles in **row-major** order (K-blocks
contiguous), but the matmul template in `attn.cc` reads them in
**column-major** order (row-blocks contiguous). This causes the matmul to
compute wrong dot products by reading K-tiles from the wrong row group.

## The Bug

### attn.cc matmul template — column-major A access

```cpp
// attn.cc: matmul_vectorized_2x2_mmul
for (unsigned z = 0; z < rowA; z += 2) {
    pA1 = pA + (z) * MMUL::size_A;           // row-tile offset
    for (unsigned j = 0; j < colB; j += 2) {
        for (unsigned i = 0; i < colA; ++i) {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += rowA * MMUL::size_A;       // K-step: rowA * 64 = 512
```

Access pattern (z=row tile, i=K tile):
```
z=0, i=0: offset   0  → expects Q[rows 0-7, cols 0-7]
z=0, i=1: offset 512  → expects Q[rows 0-7, cols 8-15]   ← READS WRONG TILE
z=1, i=0: offset  64  → expects Q[rows 8-15, cols 0-7]
z=1, i=1: offset 576  → expects Q[rows 8-15, cols 8-15]
```

### AIE documentation reference matmul — row-major A access

```cpp
// AIE API documentation example (also used by IRON mm.cc)
pA1 = pA + (z * colA) * MMUL::size_A;       // row-tile offset
pA1 += MMUL::size_A;                         // K-step: 64
```

Access pattern:
```
z=0, i=0: offset   0  → Q[rows 0-7, cols 0-7]
z=0, i=1: offset  64  → Q[rows 0-7, cols 8-15]    ← CORRECT
z=1, i=0: offset 512  → Q[rows 8-15, cols 0-7]
z=1, i=1: offset 576  → Q[rows 8-15, cols 8-15]
```

### DMA tiling — row-major tiles

The L2→L1 DMA for Q uses descriptor `sizes=[dk/8, lqp/8, 8, 8]`,
`strides=[8, dk*8, dk, 1]` which produces **row-major** tile ordering
(K-blocks contiguous within each row group):

```
L1[  0: 63] = Q[rows 0-7,  cols 0-7]    (tile 0)
L1[ 64:127] = Q[rows 0-7,  cols 8-15]   (tile 1)  ← K-block 1
L1[128:191] = Q[rows 0-7,  cols 16-23]  (tile 2)
...
L1[448:511] = Q[rows 0-7,  cols 56-63]  (tile 7)
L1[512:575] = Q[rows 8-15, cols 0-7]    (tile 8)  ← row-block 1
```

### The mismatch

| K-step 1 for rows 0-7 | DMA offset | attn.cc reads | AIE ref reads |
|------------------------|-----------|---------------|---------------|
| Q[rows 0-7, cols 8-15] | **64** | **512** (WRONG) | **64** (correct) |

The `attn.cc` template reads offset 512 for the second K-tile, which
contains Q[rows 8-15, cols 0-7] — the wrong row group's first K-block
instead of the current row group's second K-block.

## Why val_range=1 passes

At val_range=1, Q_scaled values are in [0, 0.12] (uniform, low variance).
Reading the wrong tile (Q[rows 8-15] instead of Q[rows 0-7, next K-block])
produces similar-magnitude products because all Q values are small and
similar. The accumulated dot products are close enough to the correct
result to pass the tolerance check.

At val_range=4, Q_scaled values are in [0, 0.5] with higher variance.
Reading the wrong tile produces significantly different products,
causing 10-60% per-element matmul errors that propagate through softmax
to produce 63% attention output errors.

## Evidence

### DUMP_QK_SCORES mode

With softmax bypassed (all softmax functions NOP'd, G@V replaced by
G→Gp copy), the raw Q@K^T matmul output was compared against expected
scores for each K chunk:

```
K[0:64]:   max_diff=19.13, mean_diff=4.41, errors=2663/4096 (65.0%)
K[64:128]: max_diff=17.25, mean_diff=4.07, errors=2597/4096 (63.4%)
K[128:192]:max_diff=15.63, mean_diff=4.30, errors=2656/4096 (64.8%)
K[192:256]:max_diff=11.88, mean_diff=3.26, errors=2272/4096 (55.5%)
```

Mean matmul error of 3-4 per element at score magnitude ~32 (10-12%
relative). This is NOT BFP16 quantization noise — it's fundamentally
wrong products from reading transposed tiles.

### Standalone BFP16 matmul passes

The `programming_examples/matrix_multiplication/bf16/` example uses
IRON's `mm_aie2p.cc` with row-major A access (matching the DMA) and
**passes at val_range=4** with tight tolerance (rtol=0.04, atol=0.15).
This proves BFP16 matmul is precise enough when the tile ordering
is correct.

### IRON MHA passes on this machine

IRON's MHA test passed 5/5 on this machine (Feb 17, 2026, commit
84d3478) with seq_len=16384, val_range=4. IRON's `mm.cc` uses
row-major A access matching the DMA tile order.

## Fix

Change the A access pattern in `attn.cc`'s `matmul_vectorized_2x2_mmul`
from column-major to row-major:

```cpp
// BEFORE (column-major, WRONG for row-major DMA tiles):
pA1 = pA + (z) * MMUL::size_A;
pA1 += rowA * MMUL::size_A;        // K-step

// AFTER (row-major, matches DMA and AIE documentation):
pA1 = pA + (z * colA) * MMUL::size_A;
pA1 += MMUL::size_A;                // K-step
```

The same change applies to the C output stores and B access, following
the AIE API documentation reference matmul pattern used by IRON's mm.cc.

Also need to update `max_g_bf16`, `exp_g_minus_u`, `sum_g`, and
`div_gp_sp` which read the G buffer — they must use the same tile
ordering as the matmul output.

Alternatively, change the DMA tiling to produce column-major tiles
matching the current template. This requires changing the `sizes`/`strides`
in the `ChannelPut` descriptors in `attn.py`.

## Investigation timeline

This bug was found after extensive investigation that ruled out:
- BFP16 quantization noise (standalone matmul passes)
- Softmax precision (all precision fixes had zero impact)
- conv_even rounding mode (correctly set, verified in binary)
- f32 intermediate buffers (zero impact)
- Softmax amplification theory (simulation showed both approaches equivalent)
- PyTorch SDPA golden reference (bf16 simulation gives 0 errors)

The breakthrough came from using DUMP_QK_SCORES mode to inspect the
actual matmul output from the running pipeline, revealing 10-12%
per-element errors in the Q@K^T scores themselves.

## Files involved

| File | Role |
|------|------|
| `attn.cc:50-131` | `matmul_vectorized_2x2_mmul` template with column-major A access |
| `attn.py:632` | DMA tiling descriptor for K (row-major tiles) |
| `attn.py:570` | DMA tiling descriptor for Q (row-major tiles) |
| `IRON/aie_kernels/aie2p/mm.cc:84` | IRON's template with row-major A access |
