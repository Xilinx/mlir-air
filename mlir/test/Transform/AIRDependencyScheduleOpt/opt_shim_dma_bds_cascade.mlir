//===- opt_shim_dma_bds_cascade.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Reproducer for the original failure: cascade-style launch (two distinct
// shim BD patterns per iter on one channel — R + A_bulk on different
// memrefs, not collapsible via repeat_count) exhausted the per-tile BD
// allocator under the previous module-wide tile=[16,16] preset. With the
// new default of tile=1 per level, the same IR compiles and emits
// 8 iters * 2 puts = 16 channel.puts on @cascade.

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" | FileCheck %s

module {
  air.channel @cascade [1]

  // CHECK-LABEL: func.func @cascade_two_BDs_per_iter
  // CHECK-COUNT-16: air.channel.put async{{.*}}@cascade
  // CHECK-NOT: air.channel.put async{{.*}}@cascade
  // CHECK: air.wait_all{{.*}}{air.launch_end}
  func.func @cascade_two_BDs_per_iter(%arg0: memref<512xbf16>,
                                      %arg1: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1)
        args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @cascade[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyR} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @cascade[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyA} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    return
  }

  // Multi-launch in one func mirrors the stitched o_gemv_ffn topology that
  // triggered the original bug: a cascade-shape launch (B=2 → wants small
  // tile) sits alongside an absorbable launch (B=1 → tile setting doesn't
  // matter). The pass must handle each surviving shim loop independently;
  // a cross-launch interaction would break one of the two.
  air.channel @cas2 [1]
  air.channel @abs1 [1]

  // CHECK-LABEL: func.func @two_launches_heterogeneous_shapes
  // Cascade launch: 8 iters * 2 puts = 16 puts on @cas2, fully unrolled.
  // CHECK-COUNT-16: air.channel.put async{{.*}}@cas2
  // CHECK-NOT: air.channel.put async{{.*}}@cas2
  // Absorbable launch: loop folds into one BD's wrap-and-stride.
  // CHECK-COUNT-1: air.channel.put async{{.*}}@abs1
  // CHECK-NOT: air.channel.put async{{.*}}@abs1
  func.func @two_launches_heterogeneous_shapes(%arg0: memref<512xbf16>,
                                                %arg1: memref<512xbf16>,
                                                %arg2: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    // Launch 1: cascade-shape (B=2 on @cas2).
    %0 = air.launch async (%i) in (%n=%c1)
        args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @cas2[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyR2} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @cas2[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyA2} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    // Launch 2: absorbable (B=1 on @abs1).
    %1 = air.launch async (%i) in (%n=%c1)
        args(%c=%arg2) : memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %2 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %3 = air.channel.put async [%tok] @abs1[]
            (%c[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyAbs} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    return
  }
}
