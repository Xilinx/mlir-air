# Flash Attention Direct L3→L1 Architecture Progress

## Summary

This document tracks progress on the direct shim→tile (L3→L1) flash attention architecture for NPU2. The goal is to bypass memtile staging for Q/K input data, using packet-switched routing from shim DMA directly to compute tiles, to reduce shim channel usage and scale to 4x4 herds.

## What Works

### 1. Direct L3→L1 for Q/K (per-stage QK channels)

**Design:** Per-stage `QK_s [NQ, 1]` dma_packet channels. Each stage has its own set of 4 shim DMA channels (one per q_tile). Q sent once, then K chunks on the same channel. V and Gp output still through memtile.

**Results:**
| Config | Correlation | Status |
|--------|------------|--------|
| 4x1 CHUNKS=1 | 0.9987 | PASS |
| 4x1 CHUNKS=2 | 0.9983 | PASS |
| 4x2 CHUNKS=2 | 0.9977 | PASS |
| 4x3 CHUNKS=2 | 0.9966 | PASS |
| 4x4 CHUNKS=2 | N/A | FAILS: out of shim channels (needs 16+) |

**How to run:**
```bash
cd programming_examples/flash_attention/kernel_fusion_based/test_cascade
source /home/strixminipc/mlir-air/sandbox/bin/activate
export MLIR_AIR_INSTALL_DIR=/home/strixminipc/mlir-air/install
export MLIR_AIE_INSTALL_DIR=/home/strixminipc/mlir-aie/install
export PEANO_INSTALL_DIR=/home/strixminipc/mlir-air/sandbox/lib/python3.13/site-packages/llvm-aie
export LLVM_INSTALL_DIR=/home/strixminipc/mlir-air/my_install/mlir
export PATH=$MLIR_AIR_INSTALL_DIR/bin:$MLIR_AIE_INSTALL_DIR/bin:$LLVM_INSTALL_DIR/bin:/opt/xilinx/xrt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=$MLIR_AIR_INSTALL_DIR/python:$MLIR_AIE_INSTALL_DIR/python:/opt/xilinx/xrt/python
export LD_LIBRARY_PATH=$MLIR_AIR_INSTALL_DIR/lib:$MLIR_AIE_INSTALL_DIR/lib:$LLVM_INSTALL_DIR/lib:/opt/xilinx/xrt/lib

# Run specific test:
python3 step_test.py --test full_4x1_direct --val-range 3.0
python3 step_test.py --test full_4x2_direct --val-range 3.0
```

**Key design parameters in `step_test.py` `build_full_4x2_direct()`:**
- `NS = 2` (cascade stages — change to 3 or 4 for scaling)
- `CHUNKS = 2` (K/V chunks per stage)
- `omit_while_true_loop=False` (shared-buffer mode)
- `copy_tile` saves Q before K overwrites the shared DMA buffer

### 2. Compiler Fixes Applied

**Fix 1: Shim DMA BD packet headers** (`mlir/lib/Conversion/AIRRtToNpuPass.cpp`)
- Transfers `packet` attribute from AIR-level DMA ops to shim DMA BD ops
- Without this, direct L3→L1 packet flows have no packet headers → misrouting
- Located at line ~470, after `AIE::DMABDOp::create`

**Fix 2: Memtile lock count deduplication** (`mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp`)
- `getLockValuePair` counts unique channel declarations instead of raw users
- Prevents lock count inflation from loop-unrolled K puts (e.g., 32 instead of 4)
- This fix has been reverted (not needed for per-stage design, caused test regressions for broadcast design)

## What Doesn't Work

### 4x4 with per-stage channels
**Problem:** 4 stages × 4 tiles/stage = 16 QK shim channels + 4 V + 4 Gp = 24 total. Only 8 MM2S channels available on 4 shim tiles.

**Error:** `'air.channel.put' op failed to map to shim dma channels: out of channels.`

### Q/K broadcast architecture (WIP)
**Goal:** Reduce QK shim channels from 16 to 4 by broadcasting:
- Q: vertical broadcast (per-column to all cascade stages) — 4 channels
- K: horizontal broadcast (per-stage to all columns) — 2 channels (reusing Q's shim ports)
- Total: 4 shim channels for QK + 4 for V = 8 (fits!)

**Issues encountered:**

1. **Q/K arrival ordering** — Q and K from different shim channels arrive at the same compute tile S2MM in non-deterministic order. The core expects Q first, then K. Proposed fix: `dma_await_task` barrier between Q and K phases in the runtime sequence.

2. **Packet ID assignment** — When Q and K channels are merged onto the same shim port, `labelMemcpyOpsWithPacketFlow` assigns the wrong packet ID (same ID for both Q and K). Q should get vertical broadcast pkt_id, K should get horizontal broadcast pkt_id.

3. **Runtime sequence BD ordering** — The compiler reorders Q and K puts (groups all Q first, then all K). The async token dependencies between Q and K are lost during `airrt-to-npu` conversion.

4. **Manual IR editing test** — Manually edited `npu.air.mlir` with broadcast packet flows + Q-await-K barrier + correct pkt_ids still deadlocks. Root cause under investigation.

**How to reproduce the broadcast deadlock:**
```bash
# 1. Compile per-stage QK design (working baseline)
python3 step_test.py --test full_4x2_direct --val-range 3.0  # PASS

# 2. The modified npu.air.mlir is at /tmp/npu_broadcast_final.mlir
# It has:
#   - Broadcast packet flows (Q vertical, K horizontal)
#   - Q-await-K barrier in runtime sequence
#   - Correct pkt_ids (Q: 0-3, K: 4-5)

# 3. Compile with aiecc:
cd build_cascade
aiecc -v --no-aiesim --no-xchesscc --no-xbridge --no-compile-host \
  --tmpdir=air_project --generate-full-elf --expand-load-pdis \
  --full-elf-name=air.elf -O 3 --peano $PEANO_INSTALL_DIR \
  /tmp/npu_broadcast_final.mlir

# 4. Test with pyxrt:
python3 -c "
import pyxrt as xrt; import numpy as np; from ml_dtypes import bfloat16
# ... (see test script in step_test.py run_full_4x2_direct_test)
"
# Result: ERT_CMD_STATE_TIMEOUT (deadlock)
```

## Architecture Details

### Per-stage QK (working, doesn't scale to 4x4)
```
Shim 0 (MM2S:0) ──pkt=0──→ tile(0,2)  [Q_col0 + K_stage0]
Shim 1 (MM2S:0) ──pkt=1──→ tile(1,2)  [Q_col1 + K_stage0]
Shim 2 (MM2S:0) ──pkt=2──→ tile(2,2)  [Q_col2 + K_stage0]
Shim 3 (MM2S:0) ──pkt=3──→ tile(3,2)  [Q_col3 + K_stage0]
Shim 0 (MM2S:1) ──pkt=4──→ tile(0,3)  [Q_col0 + K_stage1]
Shim 1 (MM2S:1) ──pkt=5──→ tile(1,3)  [Q_col1 + K_stage1]
...
V: shim 4,5 → memtile 4,5 → broadcast to tiles (per-stage)
Gp: tiles → memtile 0-3 → shim 0-3
```

### Broadcast QK (target, uses 4 shim channels)
```
Shim 0 (MM2S:0) ──pkt=0──→ tile(0,2)+tile(0,3)   [Q_col0 vert broadcast]
                 ──pkt=4──→ tile(0..3,2)           [K_stage0 horiz broadcast]
Shim 1 (MM2S:0) ──pkt=1──→ tile(1,2)+tile(1,3)   [Q_col1 vert broadcast]
                 ──pkt=5──→ tile(0..3,3)           [K_stage1 horiz broadcast]
Shim 2 (MM2S:0) ──pkt=2──→ tile(2,2)+tile(2,3)   [Q_col2 vert broadcast]
Shim 3 (MM2S:0) ──pkt=3──→ tile(3,2)+tile(3,3)   [Q_col3 vert broadcast]

Runtime sequence: all Q → await barrier → all K → V → Gp output
```

### Compute tile core pattern (both designs)
```
// Q once (outside chunk loop)
ChannelGet("QK", qk)     // Q arrives
copy_tile(qk, q_saved)   // Save Q before K overwrites

// K per chunk (inside loop)
for chunk in range(CHUNKS):
    ChannelGet("QK", qk) // K arrives (overwrites DMA buffer)
    ChannelGet("V", v)
    matmul(q_saved, qk, G)  // Use saved Q + current K
    softmax_update(G, ...)
    Gp += softmax(G) @ V
Gp /= sp
output(Gp)
```

## Next Steps

1. **Debug broadcast deadlock** — The manually edited IR with broadcast packet flows + Q-await-K barrier deadlocks. Need to determine if:
   - `dma_await_task` on MM2S actually ensures data reaches compute tiles (not just shim completion)
   - The packet flow reconfiguration via aiecc properly routes multicast packets
   - The compute tile S2MM accepts packets from multiple packet flows on the same DMA port

2. **Compiler changes needed for broadcast** (once concept proven):
   - `ShimDMAAllocator::allocNewDmaChannel`: merge dma_packet channels onto same shim port (partially implemented, reverted)
   - `createShimDMAAllocationOpsImpl`: link multiple channel names to merged allocations (partially implemented)
   - `labelMemcpyOpsWithPacketFlow`: assign correct pkt_id per channel when multiple packet flows share same source
   - `airrt-to-npu`: preserve Q→K ordering from async tokens in runtime sequence generation

3. **Alternative: segment_unroll** — The user mentioned using `segment_unroll` to stamp out the 4x4 design twice across 8 columns. This could use the per-stage QK design (which works) within each 4-column segment, avoiding the broadcast complexity entirely.

## Files Modified

| File | Status | Purpose |
|------|--------|---------|
| `mlir/lib/Conversion/AIRRtToNpuPass.cpp` | **Applied** | Shim DMA BD packet headers |
| `mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp` | Reverted | Lock count dedup + shim port merge |
| `mlir/lib/Conversion/AIRToAIEPass.cpp` | Reverted | Allocation linking for merged channels |
| `mlir/test/Conversion/AIRToAIE/air_shimcpy_to_npu.mlir` | Reverted | Test update for lock count change |
| `test_cascade/step_test.py` | **Modified** | All test functions |
