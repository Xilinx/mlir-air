//===- bad_expression.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/bad_expr_arch.json -g core 2>&1 | FileCheck %s

// An unusable cost expression names the op, quotes the expression and says
// what is wrong with it.
//
// And the run then declines to report a latency. The simulation still walks to
// the end -- there is nothing to stop it -- but an op whose cost could not be
// evaluated leaves a hole, and the time reached is a number for a design that
// was not the one described. It reads exactly like a real answer, which is the
// failure mode worth refusing.

// CHECK: error: 'linalg.matmul' op in cost expression "4 + 12*wbits": unknown variable 'wbits'
// CHECK: No latency reported
// CHECK-NOT: Latency (

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
