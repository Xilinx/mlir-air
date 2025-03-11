//===- opt_memtile_dma_bds.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu1_4col" -split-input-file | FileCheck %s

// RUN: not air-opt %s 2>&1 -air-opt-memtile-dma-bds="device=xcvc1902" -split-input-file | FileCheck %s --check-prefix=AIE1

// Optimize logical air.channel.put/get op into efficient shim dma block descriptor (BD).

// CHECK-LABEL: @func0
// CHECK: air.channel.put async {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c4{{.*}}, %c32{{.*}}, %c8{{.*}}] [%c1024{{.*}}, %c8{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put async {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c4{{.*}}, %c32{{.*}}, %c8{{.*}}] [%c1024{{.*}}, %c8{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put async {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c512{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c4{{.*}}, %c32{{.*}}, %c8{{.*}}] [%c1024{{.*}}, %c8{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put async {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c512{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c4{{.*}}, %c32{{.*}}, %c8{{.*}}] [%c1024{{.*}}, %c8{{.*}}, %c32{{.*}}, %c1{{.*}}])

// AIE1: error{{.*}}'func.func' op AIE1 architecture does not come with memtiles.

module {
  air.channel @channel_0 [1, 1]
  func.func @func0() {
    %0 = air.launch async () in () {
      %1 = air.segment @segment_0 async  {
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %c128 = arith.constant 128 : index
        %c1024 = arith.constant 1024 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c16384 = arith.constant 16384 : index
        %c32 = arith.constant 32 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %async_token, %results = air.execute -> (memref<2x16x32x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<2x16x32x32xbf16, 1>
          air.execute_terminator %alloc : memref<2x16x32x32xbf16, 1>
        }
        %2 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %async_token) -> (!air.async.token) {
          %3 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %arg1) -> (!air.async.token) {
            %4 = scf.for %arg4 = %c0 to %c16 step %c1 iter_args(%arg5 = %arg3) -> (!air.async.token) {
              %5 = air.channel.put async [%arg5]  @channel_0[] (%results[%arg0, %arg4, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) {id = 21 : i32} : (memref<2x16x32x32xbf16, 1>)
              scf.yield %5 : !air.async.token
            }
            scf.yield %4 : !air.async.token
          }
          scf.yield %3 : !air.async.token
        }
        %async_token_0 = air.execute {
          memref.dealloc %results : memref<2x16x32x32xbf16, 1>
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @func1
// CHECK: air.channel.get async {{.*}} @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c1024{{.*}}, %c8192{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: air.channel.get async {{.*}} @channel_0[%c0{{.*}}, %c1{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c2048{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c1024{{.*}}, %c8192{{.*}}, %c32{{.*}}, %c1{{.*}}])

// AIE1: error{{.*}}'func.func' op AIE1 architecture does not come with memtiles.

#map = affine_map<()[s0] -> (s0 + 1)>
module {
  air.channel @channel_0 [1, 2]
  func.func @func1() {
    %0 = air.launch async () in () {
      %1 = air.segment @segment_0 async  {
        %c2 = arith.constant 2 : index
        %c1024 = arith.constant 1024 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %c2048 = arith.constant 2048 : index
        %async_token, %results = air.execute -> (memref<8x2x32x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<8x2x32x32xbf16, 1>
          air.execute_terminator %alloc : memref<8x2x32x32xbf16, 1>
        }
        %2 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %async_token) -> (!air.async.token) {
          %4 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %arg1) -> (!air.async.token) {
            %5 = air.channel.get async [%arg3]  @channel_0[%c0, %c0] (%results[%arg2, %arg0, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1]) {id = 54 : i32} : (memref<8x2x32x32xbf16, 1>)
            scf.yield %5 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        %3 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %async_token) -> (!air.async.token) {
          %4 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %arg1) -> (!air.async.token) {
            %async_token_1, %results_2 = air.execute -> (index) {
              %6 = affine.apply #map()[%arg2]
              air.execute_terminator %6 : index
            }
            %5 = air.channel.get async [%async_token_1, %arg3]  @channel_0[%c0, %c1] (%results[%results_2, %arg0, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1]) {id = 55 : i32} : (memref<8x2x32x32xbf16, 1>)
            scf.yield %5 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        %async_token_0 = air.execute {
          memref.dealloc %results : memref<8x2x32x32xbf16, 1>
        }
      }
    }
    return
  }
}
