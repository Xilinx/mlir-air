//===- hierarchy.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Test air hierarchy support

// CHECK: "name": "SegmentOp",
// CHECK: "ph": "B",
// CHECK: "ts": 0.001,
// CHECK: "name": "SegmentOp",
// CHECK: "ph": "E",
// CHECK: "ts": 0.002,

// CHECK: "name": "HerdOp(herd_0)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.002,
// CHECK: "name": "HerdOp(herd_0)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.003,

// CHECK: "name": "AllocOp(L1)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.003,
// CHECK: "name": "AllocOp(L1)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.004,

// CHECK: "name": "DeallocOp(L1)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.005,
// CHECK: "name": "DeallocOp(L1)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.006,

// CHECK: "name": "HerdTerminator",
// CHECK: "ph": "B",
// CHECK: "ts": 0.007,

// CHECK: "name": "SegmentTerminator",
// CHECK: "ph": "B",
// CHECK: "ts": 0.008,

// CHECK: "name": "HerdTerminator",
// CHECK: "ph": "E",
// CHECK: "ts": 0.008,

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",
// CHECK: "ts": 0.009,

// CHECK: "name": "SegmentTerminator",
// CHECK: "ph": "E",
// CHECK: "ts": 0.009,

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",
// CHECK: "ts": 0.010,

module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @test(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) -> memref<256x1024xbf16> {
    %c1 = arith.constant 1 : index
    %async_token_1, %results_2 = air.execute -> (memref<256x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x1024xbf16>
      air.execute_terminator %alloc : memref<256x1024xbf16>
    }
    %0 = air.launch async [%async_token_1] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
      %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %async_token_3, %results_4 = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %async_token_5 = air.execute [%async_token_3] {
            memref.dealloc %results_4 : memref<32x32xbf16, 2>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return %results_2 : memref<256x1024xbf16>
  }
}

