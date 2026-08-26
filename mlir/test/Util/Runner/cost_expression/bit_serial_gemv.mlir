//===- bit_serial_gemv.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json -g core | FileCheck %s

// A cost the throughput model cannot express.
//
// The built-in formula is scalar ops divided by a rate, which is right for a
// vector unit. A weight-stationary block streams an activation through fixed
// weights: it costs bit-planes per weight tile and does not care how many MACs
// that nominally is. This arch says so, over the weight operand:
//
//     "cycles": "ceildiv(volume1, 4096) * (4 + 12*bits1)"
//
// Two matmuls, identical in shape and therefore in MAC count, differing only
// in the width of their weights:
//
//     i8  weights, 64x64 = 1 tile:  1 * (4 + 12*8) = 100 cycles
//     i4  weights, 64x64 = 1 tile:  1 * (4 + 12*4) = 52 cycles
//
// A rate cannot produce that pair: same ops, same divisor, same answer. Note
// also that 52 is below the 100-cycle kernel invocation overhead the runner
// used to add unconditionally, so the narrow case was not merely mispriced
// before, it was unreachable.

// CHECK: "name": "LinalgOp(linalg.matmul)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.00[[#%d,T0:]],
// CHECK: "name": "LinalgOp(linalg.matmul)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.[[#T0 + 100]],

// CHECK: "name": "LinalgOp(linalg.matmul)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.[[#%d,T1:]],
// CHECK: "name": "LinalgOp(linalg.matmul)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.[[#T1 + 52]],

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

module {
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%hx, %hy) in (%hsx=%c1_0, %hsy=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %tok_a, %act8 = air.execute -> (memref<64x64xi8, 2>) {
            %alloc = memref.alloc() : memref<64x64xi8, 2>
            air.execute_terminator %alloc : memref<64x64xi8, 2>
          }
          %tok_w8, %w8 = air.execute -> (memref<64x64xi8, 2>) {
            %alloc = memref.alloc() : memref<64x64xi8, 2>
            air.execute_terminator %alloc : memref<64x64xi8, 2>
          }
          %tok_o8, %out8 = air.execute -> (memref<64x64xi8, 2>) {
            %alloc = memref.alloc() : memref<64x64xi8, 2>
            air.execute_terminator %alloc : memref<64x64xi8, 2>
          }
          %mm8 = air.execute [%tok_a, %tok_w8, %tok_o8] {
            linalg.matmul ins(%act8, %w8 : memref<64x64xi8, 2>, memref<64x64xi8, 2>) outs(%out8 : memref<64x64xi8, 2>)
          }
          %tok_a4, %act4 = air.execute [%mm8] -> (memref<64x64xi4, 2>) {
            %alloc = memref.alloc() : memref<64x64xi4, 2>
            air.execute_terminator %alloc : memref<64x64xi4, 2>
          }
          %tok_w4, %w4 = air.execute [%mm8] -> (memref<64x64xi4, 2>) {
            %alloc = memref.alloc() : memref<64x64xi4, 2>
            air.execute_terminator %alloc : memref<64x64xi4, 2>
          }
          %tok_o4, %out4 = air.execute [%mm8] -> (memref<64x64xi4, 2>) {
            %alloc = memref.alloc() : memref<64x64xi4, 2>
            air.execute_terminator %alloc : memref<64x64xi4, 2>
          }
          %mm4 = air.execute [%tok_a4, %tok_w4, %tok_o4] {
            linalg.matmul ins(%act4, %w4 : memref<64x64xi4, 2>, memref<64x64xi4, 2>) outs(%out4 : memref<64x64xi4, 2>)
          }
        }
      }
    }
    return
  }
}
