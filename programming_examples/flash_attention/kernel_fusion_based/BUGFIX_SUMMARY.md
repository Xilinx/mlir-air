# Flash Attention Kernel Fusion Bug Fixes Summary

## Overview

Multiple bugs were identified in the kernel-fusion-based flash attention.
The final working fix retains only BUG2 (kernel transpose) and BUG3
(metadataArray reordering). BUG1 and BUG4 were reverted — they introduced
`repeat_count` on compute tile DMAs which triggered `load_pdi` between
launches, causing non-deterministic output corruption.

**Branch:** `fix-segment-unroll-metadata`
**Validation:** IRON-style (atol=0.15, rtol=0.04), input range [0, 4)

## Bug Summary

| Bug | Status | Description |
|-----|--------|-------------|
| BUG2 | **FIXED** (kept) | G@V double-transpose in attn.cc kernel |
| BUG3 | **FIXED** (kept) | metadataArray ordering for segment unroll |
| BUG1 | **REVERTED** | Channel name check in chansMappedToEquivalentBDs — broke infinite BD loops |
| BUG4 | **REVERTED** | Prefix+suffix same-channel check — redundant given BUG1, both unnecessary |

## Why BUG1 and BUG4 Were Reverted

BUG1 added a channel name check in `chansMappedToEquivalentBDs` to prevent
Q and K from being treated as equivalent BDs. This was introduced alongside
a per-Q-tile channel split in `attn.py` (from one shared `L2ToL1Chan2` to
separate `L2ToL1ChanQ0-Q3` channels). Together, these changes created:

1. **Separate Q and K BDs** with different packet IDs on compute tile S2MM
2. **DMA task queue** with `repeat_count=7` (Task 0: Q×1, Task 1: K×8)
3. **`load_pdi`** inserted between launches (because `deviceHasRepeatCountDMAs`)
4. **Non-deterministic tile 3 corruption** from load_pdi interacting with
   packet-switched S2MM DMAs

The original `main` branch design used a **single BD infinite loop** on the
compute tile S2MM (no packet filtering, no repeat_count, no load_pdi). The
stream switch handles packet routing — the S2MM BD doesn't need packet IDs.
BUG1's channel name check broke this design.

BUG4 was redundant given BUG1: the prefix+suffix handler requires
`memcpyIOps.size() > 4`, but after deduplication only 2 unique ops remain
(Q and K), so the handler never triggers regardless of the same-channel check.

**Resolution:** Reverted BUG1 (channel name check), BUG4 (same-channel
check), the per-Q-tile channel split in attn.py, and the S2MM packet_info
assignment block in AIRToAIEPass.cpp. This restores the infinite BD loop
architecture from `main` while keeping BUG2 and BUG3.

## BUG2: G@V Double-Transpose (KEPT)

**File:** `programming_examples/flash_attention/kernel_fusion_based/attn.cc`

**Problem:** The matmul kernel applied `aie::transpose()` to every B matrix
block. Correct for Q@K (K needs transpose) but wrong for G@V (V is already
in correct layout). The double-transpose produced the transpose of the
correct result.

**Fix:** Added `transpose_b` template parameter:
- `transpose_b=true` (default): Q@K — K blocks need software transpose
- `transpose_b=false`: G@V — V blocks already in correct layout

## BUG3: Head Interleaving in Segment Unroll (KEPT)

**File:** `mlir/lib/Conversion/AIRToAIEPass.cpp`

**Problem:** The `metadataArray` on launch-body channel ops was populated
in device-iteration order but `getIteratorFromMDVector` uses a different
linearization. For non-square channel dimensions (e.g., `[4, 2]` from
segment unroll), the mismatch routed head 0 output to head 1 shim columns.

**Fix:** Post-processing step that reorders metadataArray entries to match
the `getIteratorFromMDVector` linearization order.

## Test Results

All tests use IRON-style validation: `abs_diff > (0.15 + 0.04 * |golden|)`

| Config | Mode | Result |
|--------|------|--------|
| LK=2048 LKP=64 LQ=2048 LQP=256 NH=2 | Shared-buffer | **PASS** (0 errors / 262K) |
| LK=2048 LKP=64 LQ=2048 LQP=256 NH=12 | Shared-buffer | **PASS** (0 errors / 1.57M) |
| LK=2048 LKP=64 LQ=2048 LQP=256 NH=12 causal | Shared-buffer | **235 / 1.57M** (0.015%) |
| LK=1536 LKP=96 LQ=1536 LQP=256 NH=12 | Non-shared | **PASS** (0 errors / 1.18M) |

## Performance

```
Config: NH=12 LQ=2048 LK=2048 DK=64 DV=64
Min latency: 5.7ms
Peak throughput: 2.27 TFLOP/s (matmul FLOPs only)
```

## Key Design Principle

Shared-buffer mode requires **infinitely looping DMA BDs** at the compute
tile level. The while-true core loop time-multiplexes across both LQ tiles
and head groups. Any design that introduces `repeat_count` on compute tile
DMAs will trigger `load_pdi` between launches, which corrupts packet-switched
S2MM channels. The IRON reference avoids this entirely by using separate
DMA channels and RTP-controlled loops.

## Files Modified (vs main)

- `mlir/lib/Conversion/AIRToAIEPass.cpp` — BUG3 metadataArray reordering
  (+100 lines), S2MM packet_info block removed (-45 lines)
- `mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp` — reverted to main
  (BUG1+BUG4 changes removed)
- `programming_examples/flash_attention/kernel_fusion_based/attn.cc` — BUG2
  transpose_b parameter
- `programming_examples/flash_attention/kernel_fusion_based/attn.py` —
  reverted channel architecture to main, added IRON-style validation
