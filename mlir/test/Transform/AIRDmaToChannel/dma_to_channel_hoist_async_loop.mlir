//===- dma_to_channel_hoist_async_loop.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -canonicalize -cse -split-input-file | FileCheck %s

// A loop hoisted with a DMA from async IR.

// The loop carries two tokens and passes the second on unchanged. The hoisted
// loop's yield waits on that token as well as on the DMA's put.

// CHECK-LABEL: func.func @forwarded_token
// CHECK: air.segment @seg
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %c0{{.*}}, %{{.*}} = %{{.*}}, %[[T2:.*]] = %{{.*}}) -> (index, !air.async.token, !air.async.token)
// CHECK: %[[PUT:.*]] = air.channel.put
// CHECK: %[[W:.*]] = air.wait_all async [%[[PUT]], %[[T2]]]
// CHECK-NEXT: scf.yield %{{.*}}, %[[W]], %[[W]] : index, !air.async.token, !air.async.token
// CHECK: air.herd @h

#map = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
module {
  func.func @forwarded_token(%arg0: memref<256xf32>) {
    %0 = air.launch async () in () args(%arg1=%arg0) : memref<256xf32> {
      %1 = air.segment @seg async  args(%arg2=%arg1) : memref<256xf32> {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<256xf32, 1>) {
          %alloc = memref.alloc() : memref<256xf32, 1>
          air.execute_terminator %alloc : memref<256xf32, 1>
        }
        %2 = air.dma_memcpy_nd async [%async_token] (%results[] [] [], %arg2[] [] []) : (memref<256xf32, 1>, memref<256xf32>)
        %3 = air.herd @h async [%2]  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c1) args(%arg7=%results) : memref<256xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_1 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %async_token_2, %results_3 = air.execute -> (memref<16xf32, 2>) {
            %alloc = memref.alloc() : memref<16xf32, 2>
            air.execute_terminator %alloc : memref<16xf32, 2>
          }
          %4 = air.wait_all async [%async_token_2]
          %other = air.wait_all async
          %5:3 = scf.for %arg8 = %c0 to %c4 step %c1_0 iter_args(%arg9 = %c0, %arg10 = %4, %t2 = %other) -> (index, !air.async.token, !air.async.token) {
            %c16_5 = arith.constant 16 : index
            %c1_6 = arith.constant 1 : index
            %6 = affine.apply #map(%arg3)[%arg9]
            %7 = air.dma_memcpy_nd async [%arg10] (%results_3[] [] [], %arg7[%6] [%c16_5] [%c1_6]) : (memref<16xf32, 2>, memref<256xf32, 1>)
            %8 = affine.apply #map(%arg8)[%arg9]
            %9 = air.wait_all async [%7]
            scf.yield %8, %9, %t2 : index, !air.async.token, !air.async.token
          }
          %async_token_4 = air.execute [%5#1, %5#2] {
            memref.dealloc %results_3 : memref<16xf32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// The loop updates the DMA's offset in a pure air.execute. The hoisted loop
// computes the update with the whole air.execute.

// CHECK-LABEL: func.func @execute_update
// CHECK: air.segment @seg
// CHECK: scf.for {{.*}} iter_args(%[[BASE:.*]] = %c0{{.*}}, %{{.*}} = %{{.*}}) -> (index, !air.async.token)
// CHECK: air.channel.put
// CHECK: %{{.*}}, %[[NEXT:.*]] = air.execute -> (index) {
// CHECK-NEXT: %[[M:.*]] = arith.muli
// CHECK-NEXT: %[[A:.*]] = arith.addi %[[M]], %[[BASE]] : index
// CHECK-NEXT: air.execute_terminator %[[A]] : index
// CHECK: scf.yield %[[NEXT]], %{{.*}} : index, !air.async.token
// CHECK: air.herd @h

#map = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
module {
  func.func @execute_update(%arg0: memref<256xf32>) {
    %0 = air.launch async () in () args(%arg1=%arg0) : memref<256xf32> {
      %1 = air.segment @seg async  args(%arg2=%arg1) : memref<256xf32> {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<256xf32, 1>) {
          %alloc = memref.alloc() : memref<256xf32, 1>
          air.execute_terminator %alloc : memref<256xf32, 1>
        }
        %2 = air.dma_memcpy_nd async [%async_token] (%results[] [] [], %arg2[] [] []) : (memref<256xf32, 1>, memref<256xf32>)
        %3 = air.herd @h async [%2]  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c1) args(%arg7=%results) : memref<256xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_1 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %async_token_2, %results_3 = air.execute -> (memref<16xf32, 2>) {
            %alloc = memref.alloc() : memref<16xf32, 2>
            air.execute_terminator %alloc : memref<16xf32, 2>
          }
          %4 = air.wait_all async [%async_token_2]
          %5:2 = scf.for %arg8 = %c0 to %c4 step %c1_0 iter_args(%arg9 = %c0, %arg10 = %4) -> (index, !air.async.token) {
            %c16_5 = arith.constant 16 : index
            %c1_6 = arith.constant 1 : index
            %6 = affine.apply #map(%arg3)[%arg9]
            %7 = air.dma_memcpy_nd async [%arg10] (%results_3[] [] [], %arg7[%6] [%c16_5] [%c1_6]) : (memref<16xf32, 2>, memref<256xf32, 1>)
            %tk, %8 = air.execute -> (index) {
              %m = arith.muli %arg8, %c16_5 : index
              %a = arith.addi %m, %arg9 : index
              air.execute_terminator %a : index
            }
            %9 = air.wait_all async [%7]
            scf.yield %8, %9 : index, !air.async.token
          }
          %async_token_4 = air.execute [%5#1] {
            memref.dealloc %results_3 : memref<16xf32, 2>
          }
        }
      }
    }
    return
  }
}
