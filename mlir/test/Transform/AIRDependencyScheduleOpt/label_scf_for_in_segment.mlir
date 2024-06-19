//===- label_scf_for_in_segment.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-in-segment | FileCheck %s

// Label scf.for in air.segment for (full) unrolling.
// CHECK: air.segment
// CHECK: scf.for
// CHECK: scf.yield
// CHECK-NEXT: } {unroll = 8 : i32}
// CHECK: scf.for
// CHECK: air.herd
// CHECK: scf.yield
// CHECK-NOT: } {unroll = 8 : i32}

module {
  func.func @test(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
      %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c4 = arith.constant 4 : index
        %async_token_0 = air.wait_all async
        %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
          %async_token_3 = air.wait_all async [%arg11]
          scf.yield %async_token_3 : !air.async.token
        }
        %4 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %3) -> (!air.async.token) {
          %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          }
          scf.yield %2 : !air.async.token
        }
      }
    }
    return
  }
}
