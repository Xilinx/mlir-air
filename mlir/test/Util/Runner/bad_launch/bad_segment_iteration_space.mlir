//===- bad_segment_iteration_space.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json 2>&1 | FileCheck %s

// Check that a segment's iteration space is counted when allocating DUs.
//
// One instance of this segment costs a single DU, and the arch has three, so
// the segment fits if the iteration space is ignored. All four instances are
// co-resident for the segment's lifetime, so their demands sum: four DUs
// against three available. Before getResourceCost accounted for the trip
// count this ran to completion and reported a plausible latency.

// CHECK: error: 'air.segment' op isn't allocated with enough resources to run
// CHECK-NOT: "name": "LaunchTerminator",

module {
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %c4 = arith.constant 4 : index
      %1 = air.segment async unroll(%l) in (%ls=%c4) attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %async_token, %results = air.execute -> (memref<64xbf16, 1>) {
          %alloc = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %alloc : memref<64xbf16, 1>
        }
        %async_token_0 = air.execute [%async_token] {
          memref.dealloc %results : memref<64xbf16, 1>
        }
      }
    }
    return
  }
}
