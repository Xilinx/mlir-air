//===- air_channel_too_many_dims.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std 2>&1 | FileCheck %s

// Five dims that neither drop nor merge do not fit a shim descriptor; the
// lowering reports it rather than dropping the outer dim.

// CHECK: error: 'air.channel.get' op access pattern needs 5 dimensions; a shim DMA takes 4

module {
  air.channel @c5 [1, 1]
  func.func @five_real_dims(%a0: memref<256x384xi32>) {
    %c1_0 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_0) args(%arg0=%a0) : memref<256x384xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c384 = arith.constant 384 : index
      %c49152 = arith.constant 49152 : index
      %c1536 = arith.constant 1536 : index
      %0 = air.channel.get async @c5[%c0, %c0] (%arg0[%c0, %c0, %c0, %c0, %c0] [%c2, %c2, %c2, %c2, %c64] [%c49152, %c1536, %c384, %c128, %c1]) {id = 1 : i32} : (memref<256x384xi32>)
      air.segment @segment_0 {
        %c1_1 = arith.constant 1 : index
        air.herd @herd_0  tile (%x, %y) in (%sx=%c1_1, %sy=%c1_1) {
          %alloc = memref.alloc() : memref<2x128x64xi32, 2>
          air.channel.put @c5[%x, %y] (%alloc[] [] []) {id = 2 : i32} : (memref<2x128x64xi32, 2>)
          memref.dealloc %alloc : memref<2x128x64xi32, 2>
        }
      }
    }
    return
  }
}
