//===- air_channel_get_put.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std | FileCheck %s

// CHECK: airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
air.channel @channel_1 [1, 1]
air.channel @channel_0 [1, 1]
func.func @graph(%arg0: memref<32x16xi32>, %arg1: memref<32x16xi32>) {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %async_token = air.execute {
    air.channel.put  @channel_0[%c0, %c0] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c0]) : (memref<32x16xi32>)
  }
  %async_token_0 = air.execute {
    air.channel.get  @channel_1[%c0, %c0] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c0]) : (memref<32x16xi32>)
  }
  air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) attributes {x_loc = 7 : i64, y_loc = 2 : i64} {
    %c0_1 = arith.constant 0 : index
    %c32_2 = arith.constant 32 : index
    %c16_3 = arith.constant 16 : index
    %c8_4 = arith.constant 8 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
    %alloc_5 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
    air.channel.get  @channel_0[%arg2, %arg3] (%alloc[%c0_1, %c0_1] [%c8_4, %c16_3] [%c32_2, %c0_1]) : (memref<16x8xi32, 2>)
    affine.for %arg6 = 0 to 8 {
      affine.for %arg7 = 0 to 16 {
        %0 = affine.load %alloc[%arg7, %arg6] : memref<16x8xi32, 2>
        affine.store %0, %alloc_5[%arg7, %arg6] : memref<16x8xi32, 2>
      }
    }
    air.channel.put  @channel_1[%arg2, %arg3] (%alloc_5[%c0_1, %c0_1] [%c8_4, %c16_3] [%c32_2, %c0_1]) : (memref<16x8xi32, 2>)
    memref.dealloc %alloc_5 : memref<16x8xi32, 2>
    memref.dealloc %alloc : memref<16x8xi32, 2>
    air.herd_terminator
  }
  return
}
