//===- missing_kernel.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json -g core 2>&1 | FileCheck %s

// Naming an entry the model does not define is an error, and it has to be one
// whatever the op's body holds.
//
// linalg.copy has no priced scalar arithmetic in it, so the op count is zero
// and the throughput model has nothing to do. That must not be a reason to
// stop checking: gating the check on the op count lets a copy or a fill name a
// kernel that does not exist and fall back to the default rate in silence,
// which is the one thing this attribute exists to prevent.

// CHECK: error: 'linalg.copy' op names kernel 'does_not_exist', which the model does not define
// CHECK: No latency reported
// CHECK-NOT: Latency (

module {
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%hx, %hy) in (%hsx=%c1_0, %hsy=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %ta, %a = air.execute -> (memref<64xi8, 2>) {
            %alloc = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %alloc : memref<64xi8, 2>
          }
          %tb, %b = air.execute -> (memref<64xi8, 2>) {
            %alloc = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %alloc : memref<64xi8, 2>
          }
          %e = air.execute [%ta, %tb] {
            linalg.copy {air.kernel = "does_not_exist"} ins(%a : memref<64xi8, 2>) outs(%b : memref<64xi8, 2>)
          }
        }
      }
    }
    return
  }
}
