//===- air_channel_view_of_buffer_invalid.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-to-aie='device=npu2 row-offset=2' 2>&1 | FileCheck %s

// CHECK: error: 'air.channel.put' op moves a view of its buffer that a BD cannot address

// A strided subview: every other element. One contiguous BD on the buffer
// cannot move it.

module {
  air.channel @chan_in [1, 1]
  air.channel @chan_out [1, 1]
  func.func @strided(%arg0: memref<64xi32>, %arg1: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%a=%arg0, %b=%arg1) : memref<64xi32>, memref<32xi32> {
      air.channel.put @chan_in[] (%a[] [] []) : (memref<64xi32>)
      air.channel.get @chan_out[] (%b[] [] []) : (memref<32xi32>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @chan_in[%tx, %ty] (%buf[] [] []) : (memref<64xi32, 2>)
          %half = memref.subview %buf[0] [32] [2] : memref<64xi32, 2> to memref<32xi32, strided<[2]>, 2>
          air.channel.put @chan_out[%tx, %ty] (%half[] [] []) : (memref<32xi32, strided<[2]>, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
      }
    }
    return
  }
}
