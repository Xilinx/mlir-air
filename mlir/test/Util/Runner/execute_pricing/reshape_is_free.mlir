//===- reshape_is_free.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// memref.expand_shape / collapse_shape / reshape cost nothing.
//
// They change how a buffer is indexed and lower to no instructions. Each one
// still becomes its own vertex in the dependence graph, though, and the vertex
// used to take the runner's one-cycle default. That is not nothing: this is
// the shape air-dependency emits, a reshape sunk into the region ahead of the
// op it feeds, and a pass will emit them by the hundred.
//
// Twenty-one of them in one region ahead of a linalg.matvec priced at 200
// cycles. A metadata vertex must be free rather than priced as the region it
// sits in: the vertex's op IS the enclosing region, so pricing it by the
// region's compute charges the matvec once per reshape. That is 21 x 210
// cycles here, 4.410us against 0.231us -- which is why this is one fix with
// the region-pricing change beside it and not an independent nicety.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// CHECK: Latency (all-iterations mode): 0.231us

module {
  func.func @test(%arg0: memref<64xi8>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%la=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %1 = air.segment async args(%sa=%la) : memref<64xi8> attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_s = arith.constant 1 : index
        %2 = air.herd @reshapes async tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c1_s) attributes {id = 3 : i32} {
          %tok_f, %f = air.execute -> (memref<64xi8, 2>) {
            %alloc = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %alloc : memref<64xi8, 2>
          }
          %tok_v, %v = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_o, %o = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_c = air.execute [%tok_f, %tok_v, %tok_o] {
            %e0 = memref.expand_shape %f [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e1 = memref.collapse_shape %e0 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e2 = memref.expand_shape %e1 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e3 = memref.collapse_shape %e2 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e4 = memref.expand_shape %e3 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e5 = memref.collapse_shape %e4 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e6 = memref.expand_shape %e5 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e7 = memref.collapse_shape %e6 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e8 = memref.expand_shape %e7 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e9 = memref.collapse_shape %e8 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e10 = memref.expand_shape %e9 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e11 = memref.collapse_shape %e10 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e12 = memref.expand_shape %e11 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e13 = memref.collapse_shape %e12 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e14 = memref.expand_shape %e13 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e15 = memref.collapse_shape %e14 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e16 = memref.expand_shape %e15 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e17 = memref.collapse_shape %e16 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %e18 = memref.expand_shape %e17 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            %e19 = memref.collapse_shape %e18 [[0, 1]] : memref<8x8xi8, 2> into memref<64xi8, 2>
            %w = memref.expand_shape %e19 [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            linalg.matvec {air.op_cost = "priced"} ins(%w, %v : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o : memref<8xi8, 2>)
          }
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
