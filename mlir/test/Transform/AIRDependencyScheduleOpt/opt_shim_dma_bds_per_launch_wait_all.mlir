//===- opt_shim_dma_bds_per_launch_wait_all.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Per-launch coverage of the `air.launch_end` wait_all fallback in
// `air-opt-shim-dma-bds`. Each async launch must end with EXACTLY ONE such
// wait_all, whether it came from the tiling fixup (tiled path) or from the
// end-of-block fallback (sentinel / no-for path). Positive CHECKs require
// the wait_all to carry at least one operand (`[%...]`); a regression to
// the old empty-operand fallback would fail them.

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1 shim-dma-tile-sizes=0" \
// RUN:   | FileCheck %s --check-prefix=SENTINEL
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
// RUN: not air-opt %s -air-opt-shim-dma-bds="device=npu1 shim-dma-tile-sizes=0,1" \
// RUN:   2>&1 | FileCheck %s --check-prefix=REJECT
// REJECT: shim-dma-tile-sizes=0 is only valid as a single-value sentinel

module {

  // Sentinel `=0` skips tiling for every launch. Each of the two launches
  // gets a `launch_end` from the per-launch fallback; the func body also
  // gets one (preserved from the original `forLoopsToUnroll.empty()`
  // path). Three total, no more.

  // SENTINEL-LABEL: func.func @sentinel_two_launches
  // SENTINEL:       air.launch
  // SENTINEL:       air.wait_all{{.*\[%[^]]+\].*}}{air.launch_end}
  // SENTINEL-NOT:   air.wait_all{{.*}}{air.launch_end}
  // SENTINEL:       air.launch
  // SENTINEL:       air.wait_all{{.*\[%[^]]+\].*}}{air.launch_end}
  // SENTINEL-NOT:   air.wait_all{{.*}}{air.launch_end}
  // SENTINEL:       air.wait_all{{.*\[%[^]]+\].*}}{air.launch_end}
  // SENTINEL-NOT:   air.wait_all{{.*}}{air.launch_end}
  // SENTINEL:       return
  func.func @sentinel_two_launches(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0) : memref<512x512xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %1 = air.wait_all async
      %2 = scf.for %i = %c0 to %c512 step %c256 iter_args(%tok = %1) -> (!air.async.token) {
        %3 = air.channel.put async [%tok]  @channel_0[%c0, %c0] (%buf[%c0, %i] [%c256, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    %4 = air.launch async [%0] (%ty) in (%sy=%c1) args(%buf2=%arg1) : memref<512x512xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %5 = air.wait_all async
      %6 = scf.for %j = %c0 to %c512 step %c256 iter_args(%tok2 = %5) -> (!air.async.token) {
        %7 = air.channel.get async [%tok2]  @channel_1[%c0, %c0] (%buf2[%c0, %j] [%c256, %c64] [%c512, %c1_0]) {id = 2 : i32} : (memref<512x512xbf16>)
        scf.yield %7 : !air.async.token
      }
    }
    return
  }

  // Mixed: launch A has a shim scf.for (gets tiled under default tile=1, so
  // the launch_end wait_all comes from the tiling fixup). Launch B has only
  // direct channel ops (no scf.for at shim), so it falls through to the
  // per-launch fallback. Each must still end up with exactly one launch_end.

  // DEFAULT-LABEL: func.func @mixed_tiled_and_no_for
  // DEFAULT:       air.launch
  // DEFAULT:       air.wait_all{{.*\[%[^]]+\].*}}{air.launch_end}
  // DEFAULT-NOT:   air.wait_all{{.*}}{air.launch_end}
  // DEFAULT:       air.launch
  // DEFAULT:       air.wait_all{{.*\[%[^]]+\].*}}{air.launch_end}
  // DEFAULT-NOT:   air.wait_all{{.*}}{air.launch_end}
  // DEFAULT:       return
  func.func @mixed_tiled_and_no_for(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0) : memref<512x512xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %1 = air.wait_all async
      %2 = scf.for %i = %c0 to %c512 step %c256 iter_args(%tok = %1) -> (!air.async.token) {
        %3 = air.channel.put async [%tok]  @channel_0[%c0, %c0] (%buf[%c0, %i] [%c256, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    %4 = air.launch async [%0] (%ty) in (%sy=%c1) args(%buf2=%arg1) : memref<512x512xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %5 = air.channel.get async @channel_1[%c0, %c0] (%buf2[%c0, %c0] [%c512, %c512] [%c512, %c1_0]) {id = 2 : i32} : (memref<512x512xbf16>)
    }
    return
  }

  // Tiled-only sanity check: under default tile=1, the single launch's shim
  // scf.for goes through the tiling fixup, which inserts the launch_end
  // wait_all. The per-launch fallback MUST skip this launch, so the count
  // stays at one (a regression that re-added the fallback unconditionally
  // would produce two).

  // DEFAULT-LABEL: func.func @tiled_only_no_double_launch_end
  // DEFAULT:       air.launch
  // DEFAULT:       air.wait_all{{.*\[%[^]]+\].*}}{air.launch_end}
  // DEFAULT-NOT:   air.wait_all{{.*}}{air.launch_end}
  // DEFAULT:       return
  func.func @tiled_only_no_double_launch_end(%arg0: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0) : memref<512x512xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %1 = air.wait_all async
      %2 = scf.for %i = %c0 to %c512 step %c256 iter_args(%tok = %1) -> (!air.async.token) {
        %3 = air.channel.put async [%tok]  @channel_0[%c0, %c0] (%buf[%c0, %i] [%c256, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    return
  }
}
