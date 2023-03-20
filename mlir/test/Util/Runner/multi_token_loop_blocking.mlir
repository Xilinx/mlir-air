//===- multi_token_loop_blocking.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Check for the blocking behaviour of loop-carried tokens

// CHECK-COUNT-104: "name": "WaitAllOp",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<()[s0] -> (s0 mod 64)>
module {
  func.func @test(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %async_token, %results = air.execute -> (memref<512x512xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<512x512xbf16>
      air.execute_terminator %alloc : memref<512x512xbf16>
    }
    %async_token_0 = air.execute [%async_token] {
      memref.copy %arg2, %results : memref<512x512xbf16> to memref<512x512xbf16>
    }
    %0 = air.launch async [%async_token_0] (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) {
      %async_token_1, %results_2 = air.execute -> (index) {
        %2 = affine.apply #map()[%arg3]
        air.execute_terminator %2 : index
      }
      %async_token_3, %results_4 = air.execute -> (index) {
        %2 = affine.apply #map()[%arg4]
        air.execute_terminator %2 : index
      }
      %1 = air.partition async attributes {column_usage = [4, 1]} {
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c128 = arith.constant 128 : index
        %async_token_5, %results_6 = air.execute -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %2 = scf.for %arg7 = %c0 to %c512 step %c128 iter_args(%arg8 = %async_token_5) -> (!air.async.token) {
          %async_token_7, %results_8 = air.execute [%arg8] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %async_token_9, %results_10 = air.execute [%arg8] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %3 = air.herd @herd_0 async [%arg8]  tile (%arg9, %arg10) in (%arg11=%c4, %arg12=%c4) {
            %c0_11 = arith.constant 0 : index
            %c128_12 = arith.constant 128 : index
            %c32 = arith.constant 32 : index
            %async_token_13, %results_14 = air.execute -> (memref<32x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc : memref<32x32xbf16, 2>
            }
            %async_token_15, %results_16 = air.execute -> (memref<64x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<64x32xbf16, 2>
              air.execute_terminator %alloc : memref<64x32xbf16, 2>
            }
            %async_token_17, %results_18 = air.execute -> (memref<64x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<64x32xbf16, 2>
              air.execute_terminator %alloc : memref<64x32xbf16, 2>
            }
            %4 = air.wait_all async [%async_token_15, %async_token_17] 
            %5:3 = scf.for %arg13 = %c0_11 to %c128_12 step %c32 iter_args(%arg14 = %4, %arg15 = %4, %arg16 = %4) -> (!air.async.token, !air.async.token, !air.async.token) {
              %async_token_19, %results_20 = air.execute [%arg14] -> (index) {
                %9 = affine.apply #map1()[%arg13]
                air.execute_terminator %9 : index
              }
              %6 = air.wait_all async [%async_token_19, %arg16] 
              %7 = air.wait_all async [%async_token_19, %arg16] 
              %8 = air.wait_all async [%6, %7] 
              %async_token_21, %results_22 = air.execute [%arg15] -> (index) {
                %9 = affine.apply #map1()[%arg13]
                air.execute_terminator %9 : index
              }
              %subview = memref.subview %results_16[%results_20, 0] [32, 32] [1, 1] : memref<64x32xbf16, 2> to memref<32x32xbf16, strided<[32, 1], offset: ?>, 2>
              %subview_23 = memref.subview %results_18[%results_20, 0] [32, 32] [1, 1] : memref<64x32xbf16, 2> to memref<32x32xbf16, strided<[32, 1], offset: ?>, 2>
              %async_token_24 = air.execute [%async_token_21, %8] {
                linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%subview, %subview_23 : memref<32x32xbf16, strided<[32, 1], offset: ?>, 2>, memref<32x32xbf16, strided<[32, 1], offset: ?>, 2>) outs(%results_14 : memref<32x32xbf16, 2>)
              }
              scf.yield %8, %async_token_24, %async_token_24 : !air.async.token, !air.async.token, !air.async.token
            }
            %async_token_25 = air.execute [%5#2] {
              memref.dealloc %results_14 : memref<32x32xbf16, 2>
            }
            %async_token_26 = air.execute [%5#2] {
              memref.dealloc %results_16 : memref<64x32xbf16, 2>
            }
            %async_token_27 = air.execute [%5#2] {
              memref.dealloc %results_18 : memref<64x32xbf16, 2>
            }
            air.herd_terminator
          }
          %async_token_28 = air.execute [%3] {
            memref.dealloc %results_8 : memref<128x128xbf16, 1>
          }
          %async_token_29 = air.execute [%3] {
            memref.dealloc %results_10 : memref<128x128xbf16, 1>
          }
          %async_token_30 = air.wait_all async [%async_token_28, %async_token_29]
          scf.yield %async_token_30 : !air.async.token
        }
        %async_token_31 = air.execute [%2] {
          memref.dealloc %results_6 : memref<128x128xbf16, 1>
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return
  }
}

