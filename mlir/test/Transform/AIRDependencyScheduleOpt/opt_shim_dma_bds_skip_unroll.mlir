//===- opt_shim_dma_bds_skip_unroll.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1 shim-dma-tile-sizes=1,1" | FileCheck %s --check-prefix=TILE1
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1 shim-dma-tile-sizes=2" | FileCheck %s --check-prefix=TILE2
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" | FileCheck %s --check-prefix=NOTILE

// Test: all-1 tile sizes and empty tile sizes skip tiling, unrolling, and BD
// folding for the runtime loop. The scf.for from the launch conversion is
// preserved. Non-trivial tile sizes (e.g. 2) still tile and unroll.

// TILE1-LABEL: func @multi_iter_with_segment
// With tile-sizes=1,1: launch→scf.for, NOT tiled/unrolled, NO BD folding.
// The scf.for survives with channel ops and segment inside.
// The air.launch_end barrier depends on the scf.for result token.
// TILE1: air.launch async () in ()
// TILE1-SAME: dummyLaunch
// TILE1: %[[FOR_RESULT:.*]] = scf.for
// TILE1: air.channel.put async
// TILE1: air.segment @seg
// TILE1: air.channel.get
// TILE1: scf.yield
// TILE1: air.wait_all [%[[FOR_RESULT]]] {air.launch_end}

// TILE2-LABEL: func @multi_iter_with_segment
// With tile-sizes=2: outer loop (trip=2) is unrolled into 2 copies.
// Each copy has an inner scf.for (trip=2) with segment + channel.get.
// TILE2: air.launch async () in ()
// TILE2-SAME: dummyLaunch
// TILE2: air.channel.put async
// TILE2: scf.for
// TILE2: air.segment @seg
// TILE2: air.channel.get
// TILE2: scf.yield
// TILE2: air.wait_all {{.*}} {air.launch_end}
// The second unrolled copy:
// TILE2: air.channel.put async
// TILE2: scf.for
// TILE2: air.segment @seg
// TILE2: air.channel.get
// TILE2: scf.yield
// TILE2: air.wait_all {{.*}} {air.launch_end}

// NOTILE-LABEL: func @multi_iter_with_segment
// No tile sizes: same behavior as all-1 (fast path).
// NOTILE: air.launch async () in ()
// NOTILE-SAME: dummyLaunch
// NOTILE: %[[NT_FOR:.*]] = scf.for
// NOTILE: air.channel.put async
// NOTILE: air.segment @seg
// NOTILE: air.channel.get
// NOTILE: scf.yield
// NOTILE: air.wait_all [%[[NT_FOR]]] {air.launch_end}

module {
  air.channel @input_ch [1, 1]
  air.channel @output_ch [1, 1]
  func.func @multi_iter_with_segment(%arg0: memref<256x64xbf16>, %arg1: memref<256x64xbf16>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%arg2) in (%arg3=%c4) args(%arg4=%arg0, %arg5=%arg1) : memref<256x64xbf16>, memref<256x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %1 = affine.apply affine_map<()[s0] -> (s0 * 4096)>()[%arg2]
      %2 = air.channel.put async @input_ch[%c0, %c0] (%arg4[%c0, %1] [%c64, %c64] [%c64, %c1]) {metadata = @airMemcpyId1} : (memref<256x64xbf16>)
      %3 = air.segment @seg async {
        %alloc = memref.alloc() : memref<64x64xbf16, 1>
        memref.dealloc %alloc : memref<64x64xbf16, 1>
      }
      %4 = air.channel.get async [%3] @output_ch[%c0, %c0] (%arg5[%c0, %1] [%c64, %c64] [%c64, %c1]) {metadata = @airMemcpyId2} : (memref<256x64xbf16>)
      %5 = air.wait_all async [%2, %4]
    }
    return
  }
}
