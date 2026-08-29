//===- fallback_params.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// How a linalg body is priced used to be three constants compiled into the
// runner, chosen for AIE: a herd body instance is one core, entering a kernel
// costs 100 cycles because it is an external function call, and an unlisted
// datatype runs 8 lanes wide. Those are properties of a machine. This arch
// says otherwise, in "cost_model.fallback".
//
// Same kernel as cost_function/mac_bf16.mlir, which takes the defaults and
// gets 256 + 100. Here 32x32x32 bf16 is 65536 scalar ops, the model declares
// 128 macs/core/cycle (so 256 ops/cycle) across 4 cores, and entering the
// kernel is free:
//
//     65536 / (4 x 256 x 1) = 64 cycles, + 0 overhead
//
// Reading the arch's cost_model.fallback is the whole assertion: ignore it and the
// defaults give 356.

// CHECK: "name": "LinalgOp(linalg.matmul)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.00[[#%d,TIME0:]],
// CHECK: "name": "LinalgOp(linalg.matmul)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.0[[#TIME0 + 64]],

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

module {
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
          %async_token_5, %results_6 = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %async_token_7, %results_8 = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %async_token_9 = air.execute [%async_token_5, %async_token_7] {
            linalg.matmul ins(%results_4, %results_6 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%results_8 : memref<32x32xbf16, 2>)
          }
          %async_token_10 = air.execute [%async_token_9] {
            memref.dealloc %results_4 : memref<32x32xbf16, 2>
          }
          %async_token_11 = air.execute [%async_token_9] {
            memref.dealloc %results_6 : memref<32x32xbf16, 2>
          }
          %async_token_12 = air.execute [%async_token_9] {
            memref.dealloc %results_8 : memref<32x32xbf16, 2>
          }
        }
      }
    }
    return %results_2 : memref<256x1024xbf16>
  }
}
