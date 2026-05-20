//===- opt_shim_dma_bds_per_launch_wait_all.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// `air-opt-shim-dma-bds` must emit an end-of-launch `air.wait_all` carrying
// `air.launch_end` in EVERY async launch that did not get its shim DMA for
// loops tiled. The legacy fallback only ran when *no* launch in the module
// was tiled, which left un-tiled launches in mixed modules without a
// wait_all -- their DMAs were fire-and-forget across multi-launch
// transitions.
//
// The `shim-dma-tile-sizes=0` sentinel forces skip-tiling globally and is
// the easiest way to exercise the per-launch fallback path.

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1 shim-dma-tile-sizes=0" \
// RUN:   | FileCheck %s

module {

  // Two async launches in one func. Both contain shim scf.for loops with
  // air.channel.put/get; both must be skipped under sentinel=0 and both
  // must end up with an `air.wait_all` annotated `air.launch_end`.

  // CHECK-LABEL: func.func @mixed_launches
  func.func @mixed_launches(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    // CHECK: air.launch
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
      // CHECK: air.wait_all{{.*}}{air.launch_end}
    }
    // CHECK: air.launch
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
      // CHECK: air.wait_all{{.*}}{air.launch_end}
    }
    return
  }
}
