# BUG3: Head Interleaving in Segment Unroll Output

**Status: FIXED** (branch `fix-segment-unroll-metadata`, commit `87322907`)

## Summary

The segment unroll spatially partitions the 8 AIE columns between 2 heads
(4 columns each). The `metadataArray` on launch-body channel ops was
populated in device-iteration order but looked up using a different
linearization, causing `channel_0[qt, head]` to resolve to the wrong
shim DMA allocation. Head 0's tiles 2-3 were routed to head 1's shim
columns and vice versa.

## Evidence (before fix)

Test config: LK=256, LKP=64, LQ=256, LQP=256, DK=DV=64, NUM_HEADS=2, val_range=2.

Cross-head correlation matrix:

| Tile | h0 NPU vs h0 gold | h0 NPU vs h1 gold | h1 NPU vs h0 gold | h1 NPU vs h1 gold |
|------|----|----|----|----|
| 0 | **0.99** | -0.00 | **0.86** | 0.01 |
| 1 | **0.87** | 0.00 | **0.86** | -0.02 |
| 2 | 0.01 | **0.88** | 0.00 | **0.88** |
| 3 | -0.01 | **0.88** | -0.02 | **0.99** |

Tiles 0-1 always produce head 0 output, tiles 2-3 always produce head 1
output, regardless of which head is being processed.

## Verification (after fix)

Same config, val_range=4:

| Tile | h0 NPU vs h0 gold | h0 NPU vs h1 gold | h1 NPU vs h0 gold | h1 NPU vs h1 gold |
|------|----|----|----|----|
| 0 | **0.998** | 0.02 | 0.02 | **0.998** |
| 1 | **0.998** | 0.04 | 0.04 | **0.998** |
| 2 | **0.997** | 0.05 | 0.06 | **0.998** |
| 3 | **0.997** | 0.03 | 0.04 | **0.998** |

0% IRON errors. No cross-head contamination.

## Root Cause

### The metadataArray ordering mismatch

The `metadataArray` attribute on launch-body channel ops contains one entry
per shim DMA allocation. For `channel_0 [4, 2]` (4 qt tiles × 2 heads),
there are 8 entries. Two separate orderings are involved:

**Population order** (in `createShimDMAAllocationOpsImpl`, AIRToAIEPass.cpp):
The allocations vector iterates device 0 tiles first, then device 1 tiles:
```
Position 0: air_channel_0_0_0_0 (device 0, tile 0)  → shim 0
Position 1: air_channel_0_0_0_1 (device 0, tile 1)  → shim 1
Position 2: air_channel_0_0_0_2 (device 0, tile 2)  → shim 2
Position 3: air_channel_0_0_0_3 (device 0, tile 3)  → shim 3
Position 4: air_channel_0_1_0_0 (device 1, tile 0)  → shim 4
Position 5: air_channel_0_1_0_1 (device 1, tile 1)  → shim 5
Position 6: air_channel_0_1_0_2 (device 1, tile 2)  → shim 6
Position 7: air_channel_0_1_0_3 (device 1, tile 3)  → shim 7
```
This is column-major: `position = qt + 4*head`.

**Lookup order** (in `getIteratorFromMDVector`, Util.cpp):
For `channel_0[qt, head]` with dims `[4, 2]`, the function computes:
`index = dims[1]*qt + head = 2*qt + head`.
This is row-major: the head dimension varies fastest.

**The mismatch:** For `channel_0[2, 0]` (head=0, tile=2):
- Lookup: `2*2 + 0 = 4` → metadataArray[4] = `air_channel_0_1_0_0` (device 1, tile 0) ✗
- Expected: position 2 = `air_channel_0_0_0_2` (device 0, tile 2) ✓

Full mapping showing the scramble:

| Channel | Indices | Lookup index | metadataArray entry | Correct? |
|---------|---------|-------------|-------------------|----------|
| `[0,0]` | qt=0,h=0 | 0 | dev0,tile0 | ✓ |
| `[1,0]` | qt=1,h=0 | 2 | dev0,tile2 | ✗ (should be tile1) |
| `[2,0]` | qt=2,h=0 | 4 | dev1,tile0 | ✗ (should be dev0,tile2) |
| `[3,0]` | qt=3,h=0 | 6 | dev1,tile2 | ✗ (should be dev0,tile3) |
| `[0,1]` | qt=0,h=1 | 1 | dev0,tile1 | ✗ (should be dev1,tile0) |
| `[1,1]` | qt=1,h=1 | 3 | dev0,tile3 | ✗ (should be dev1,tile1) |
| `[2,1]` | qt=2,h=1 | 5 | dev1,tile1 | ✗ (should be dev1,tile2) |
| `[3,1]` | qt=3,h=1 | 7 | dev1,tile3 | ✓ |

Only `[0,0]` and `[3,1]` are correct by coincidence.

### Why this was undetected

For square channel dimensions (e.g., `[4, 4]` herds), `dims[i]` is the
same for all dimensions, so row-major and column-major linearizations
produce identical results. The bug only manifests with non-square
dimensions like `[4, 2]`, which segment unroll with NUM_HEADS≠NUM_Q_TILES
triggers.

### Where the metadataArray is consumed

The metadataArray is NOT consumed in `AIRRtToNpuPass.cpp` as initially
suspected. Instead:

1. `AIRToAIEPass.cpp:4205-4268` — builds metadataArray (appends entries sequentially)
2. `AIRLoweringPass.cpp:604-632` — consumes metadataArray during `air-to-std` lowering:
   calls `getIndexToMetadataArrayFromChannelIndices()` → `getIteratorFromMDVector()`
   to compute linearized index, then picks `metadataArray[index]` to set the
   `metadata` attribute on the `airrt.dma_memcpy_nd` op.

## Fix

**File:** `mlir/lib/Conversion/AIRToAIEPass.cpp` (lines 4268+)

After the existing metadataArray population loop, added a post-processing
step that reorders entries to match the `getIteratorFromMDVector` linearization.
The fix:

1. Detects segment-unrolled metadataArray entries by checking for the extra
   underscore segments in allocation names (`air_channel_0_X_Y_Z` has 6
   underscore-separated parts vs 4 without unroll)
2. Parses each entry's `base` name to extract unroll copy index (X) and
   tile index (Z)
3. Computes the correct linearized position using `getIteratorFromMDVector`
   with position `[tileIdx, unrollCopy]`
4. Reorders the array so `metadataArray[linearized_index]` points to the
   correct allocation

This approach preserves backward compatibility: the `getIteratorFromMDVector`
function and all its callers are unchanged. Only the population-side ordering
is adjusted.

### Why not fix `getIteratorFromMDVector` instead?

The `getIteratorFromMDVector` function uses a specific linearization
convention (`dims[i]^i * reversed_position[i]`) that happens to match
the allocation order for square dimensions. Changing it would require
updating all callers and existing tests. The post-processing reorder is
a safer, more targeted fix.

## Related Bugs

- **BUG1** (same commit): `chansMappedToEquivalentBDs` in
  `AIRToAIESchedulingUtils.cpp` collapsed Q and K BDs when they shared
  the same L1 buffer (shared-buffer mode), losing the K BD → DMA deadlock.
  Fix: check channel name before comparing BD equivalence.

- **BUG2** (separate commit): G@V matmul in `attn.cc` applied
  `aie::transpose` to V blocks, but V's DMA inner layout is already
  `[k_in, n_in]` (correct for hardware). Fix: `transpose_b=false` for
  G@V, `transpose_b=true` for Q@K.
