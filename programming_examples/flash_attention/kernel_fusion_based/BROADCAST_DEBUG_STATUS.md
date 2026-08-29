# Broadcast Debug Status - March 22, 2026

## Files saved:
- `broadcast_ir_fixed.mlir` - Full broadcast IR with correct core code + runtime sequence (BEFORE aiecc)
- `broadcast_ir_fixed_v2.mlir` - Same after aiecc address assignment (full runtime seq)
- `debug_ir.mlir` - Original user reference (wrong core code)

## Core code fix applied (all 8 cores):
- copy_tile moved OUTSIDE chunk loop (Q received once, K received per chunk)
- Dead lock acquires/releases removed
- Row 3: Q→copy_tile→zero_fills→release→chunk_loop(K+V→matmul)→cascade
- Row 2: cascade_recv→Q→copy_tile→release→chunk_loop(K+V→matmul)→output

## Runtime sequence fix:
- Phase 1: 4 Q BDs (pkt 0-3), one per shim, issue_token + await_task
- Phase 2: 4 K BDs (pkt 4-5), K2L1_0 for stage0, K2L1_1 for stage1
- Phase 3: V + output in parallel

## Test results:
- Q-only broadcast: **COMPLETED** (no deadlock)
- Full broadcast (Q→await→K→V→output): **TIMEOUT** (deadlock)
- Next: test Q+K only (no V/output) to isolate K broadcast

## How to compile:
```bash
cd build_cascade
source sandbox, set env vars (see DIRECT_L3_L1_PROGRESS.md)
aiecc.py --no-aiesim --no-xchesscc --no-xbridge --no-compile-host \
  --tmpdir=air_project --generate-full-elf --expand-load-pdis \
  --full-elf-name=air_broadcast.elf -O 3 \
  --peano $PEANO_INSTALL_DIR \
  air_project/input_with_addresses.mlir
```

## How to run:
```python
device = xrt.device(0)
elf = xrt.elf('air_broadcast.elf')
context = xrt.hw_context(device, elf)
kernel = xrt.ext.kernel(context, 'main:full_4x2_direct')
```

## Updated test results:
- Q only broadcast: COMPLETED
- Q+K broadcast: COMPLETED  
- Q+K+V (no output): testing next
- Q+K+V+output: TIMEOUT (deadlock)


## CORRECTED test results (with proper await):
- Q only broadcast (awaited): COMPLETED
- Q+K broadcast (awaited): **TIMEOUT** -- K deadlocks!
- Previous Q+K/Q+K+V "COMPLETED" were false positives (no await)

## Root cause hypothesis:
K BDs use repeat_count=7 (8 sub-tiles of 512 elements = 4096 total).
Compute tile S2MM:0 BD transfers 4096 elements in one shot.
The shim fires 8 small transfers but the tile expects 1 large transfer.
Lock signaling mismatch: shim releases lock 8 times, tile acquires once.
This may cause backpressure deadlock on the multicast packet flow.

## Next step:
Try K BDs WITHOUT repeat_count -- send full 4096-element tiles instead of 8x512.

## Final K broadcast isolation:
- 1-chunk K + 1-chunk V (awaited): **TIMEOUT**
- Even a SINGLE K BD multicasting to 4 tiles deadlocks
- Q 2-way multicast (vertical, 2 tiles): WORKS
- K 4-way multicast (horizontal, 4 tiles): FAILS
- Root cause: 4-way packet multicast from shim to compute tiles deadlocks
- This may be a HW routing limitation or stream switch backpressure issue
- Next: try K through memtile (like V) instead of direct L3→L1 broadcast
