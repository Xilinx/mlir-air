//===- air_channel_view_of_buffer.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' --split-input-file | FileCheck %s

// A core sends out part of a buffer through a view of it. The BD goes on the
// buffer under the view, and starts where the view does.

// A subview behind a reshape: the lookup walks the chain to the buffer.
// CHECK-LABEL: aie.mem(%{{.*}}tile_0_2)
// CHECK: aie.dma_bd(%[[BUF:.*]] : memref<4x16xi32, 2> offset = 32 len = 32)
// CHECK: aie.dma_bd(%[[BUF]] : memref<4x16xi32, 2> offset = 0 len = 64)

module {
  air.channel @chan_in [1, 1]
  air.channel @chan_out [1, 1]
  func.func @view_chain(%arg0: memref<64xi32>, %arg1: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%a=%arg0, %b=%arg1) : memref<64xi32>, memref<32xi32> {
      air.channel.put @chan_in[] (%a[] [] []) : (memref<64xi32>)
      air.channel.get @chan_out[] (%b[] [] []) : (memref<32xi32>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<4x16xi32, 2>
          air.channel.get @chan_in[%tx, %ty] (%buf[] [] []) : (memref<4x16xi32, 2>)
          %flat = memref.collapse_shape %buf [[0, 1]] : memref<4x16xi32, 2> into memref<64xi32, 2>
          %half = memref.subview %flat[32] [32] [1] : memref<64xi32, 2> to memref<32xi32, strided<[1], offset: 32>, 2>
          air.channel.put @chan_out[%tx, %ty] (%half[] [] []) : (memref<32xi32, strided<[1], offset: 32>, 2>)
          memref.dealloc %buf : memref<4x16xi32, 2>
        }
      }
    }
    return
  }
}

// -----

// A subview of the buffer itself.
// CHECK-LABEL: aie.mem(%{{.*}}tile_0_2)
// CHECK: aie.dma_bd(%[[BUF:.*]] : memref<64xi32, 2> offset = 32 len = 32)
// CHECK: aie.dma_bd(%[[BUF]] : memref<64xi32, 2> offset = 0 len = 64)

module {
  air.channel @chan_in [1, 1]
  air.channel @chan_out [1, 1]
  func.func @subview(%arg0: memref<64xi32>, %arg1: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%a=%arg0, %b=%arg1) : memref<64xi32>, memref<32xi32> {
      air.channel.put @chan_in[] (%a[] [] []) : (memref<64xi32>)
      air.channel.get @chan_out[] (%b[] [] []) : (memref<32xi32>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @chan_in[%tx, %ty] (%buf[] [] []) : (memref<64xi32, 2>)
          %half = memref.subview %buf[32] [32] [1] : memref<64xi32, 2> to memref<32xi32, strided<[1], offset: 32>, 2>
          air.channel.put @chan_out[%tx, %ty] (%half[] [] []) : (memref<32xi32, strided<[1], offset: 32>, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
      }
    }
    return
  }
}

// -----

// A reinterpret_cast moves the start too, here behind a reshape.
// CHECK-LABEL: aie.mem(%{{.*}}tile_0_2)
// CHECK: aie.dma_bd(%[[BUF:.*]] : memref<64xi32, 2> offset = 32 len = 32)

module {
  air.channel @chan_in [1, 1]
  air.channel @chan_out [1, 1]
  func.func @reinterpret(%arg0: memref<64xi32>, %arg1: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%a=%arg0, %b=%arg1) : memref<64xi32>, memref<32xi32> {
      air.channel.put @chan_in[] (%a[] [] []) : (memref<64xi32>)
      air.channel.get @chan_out[] (%b[] [] []) : (memref<32xi32>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @chan_in[%tx, %ty] (%buf[] [] []) : (memref<64xi32, 2>)
          %r = memref.reinterpret_cast %buf to offset: [32], sizes: [32], strides: [1] : memref<64xi32, 2> to memref<32xi32, strided<[1], offset: 32>, 2>
          %half = memref.expand_shape %r [[0, 1]] output_shape [2, 16] : memref<32xi32, strided<[1], offset: 32>, 2> into memref<2x16xi32, strided<[16, 1], offset: 32>, 2>
          air.channel.put @chan_out[%tx, %ty] (%half[] [] []) : (memref<2x16xi32, strided<[16, 1], offset: 32>, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
      }
    }
    return
  }
}

