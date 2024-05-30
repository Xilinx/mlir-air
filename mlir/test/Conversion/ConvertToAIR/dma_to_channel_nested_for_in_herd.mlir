//===- dma_to_channel_nested_for_in_herd.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -canonicalize -cse | FileCheck %s

// Hoisting external channel put/get op out of a herd with nested for loops

#map = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
module {
// CHECK-LABEL: func.func @sync
  func.func @sync(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64x64xi32>)
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    memref.copy %alloc, %alloc_0 : memref<64x64xi32> to memref<64x64xi32>
// CHECK: scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) =
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_0[%[[VALUE0]], %[[VALUE1]]]
// CHECK: scf.parallel (%[[VALUE2:.*]], %[[VALUE3:.*]]) =
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_1[%[[VALUE2]], %[[VALUE3]]]
// CHECK: scf.parallel (%[[VALUE4:.*]], %[[VALUE5:.*]]) =
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_2[%[[VALUE4]], %[[VALUE5]]]
// CHECK: scf.parallel (%[[VALUE6:.*]], %[[VALUE7:.*]]) =
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get @channel_3[%[[VALUE6]], %[[VALUE7]]]
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc_0) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.for %newarg0 = %c0 to %c64 step %c32 {
        scf.for %newarg1 = %c0 to %c64 step %c32 {
            scf.for %arg9 = %c0 to %c64 step %c32 {
                %alloc_1 = memref.alloc() : memref<32x32xi32, 2>
                %alloc_2 = memref.alloc() : memref<32x32xi32, 2>
                %alloc_3 = memref.alloc() : memref<32x32xi32, 2>
                air.dma_memcpy_nd (%alloc_1[] [] [], %arg6[%newarg0, %arg9] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                air.dma_memcpy_nd (%alloc_2[] [] [], %arg7[%arg9, %newarg1] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                air.dma_memcpy_nd (%alloc_3[] [] [], %arg8[%newarg0, %newarg1] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                linalg.matmul ins(%alloc_1, %alloc_2 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_3 : memref<32x32xi32, 2>)
                air.dma_memcpy_nd (%arg8[%newarg0, %newarg1] [%c32, %c32] [%c64, %c1], %alloc_3[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
                memref.dealloc %alloc_1 : memref<32x32xi32, 2>
                memref.dealloc %alloc_2 : memref<32x32xi32, 2>
                memref.dealloc %alloc_3 : memref<32x32xi32, 2>
            }
        }
      }
      air.herd_terminator
    }
    return %alloc_0 : memref<64x64xi32>
  }

// CHECK-LABEL: func.func @async
// CHECK: %[[EVENT0:.*]] = scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_4[%[[VALUE0]], %[[VALUE1]]]
// CHECK: %[[EVENT1:.*]] = scf.parallel (%[[VALUE2:.*]], %[[VALUE3:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_5[%[[VALUE2]], %[[VALUE3]]]
// CHECK: %[[EVENT2:.*]] = scf.parallel (%[[VALUE4:.*]], %[[VALUE5:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_6[%[[VALUE4]], %[[VALUE5]]]
// CHECK: %[[EVENT3:.*]] = scf.parallel (%[[VALUE6:.*]], %[[VALUE7:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get async{{.*}}@channel_7[%[[VALUE6]], %[[VALUE7]]]  
  func.func @async(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %async_token, %results = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%c0_i32 : i32) outs(%results : memref<64x64xi32>)
    }
    %async_token_1, %results_2 = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<64x64xi32> to memref<64x64xi32>
    }
    %0 = air.herd @herd_0 async [%async_token_3]  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%results_2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 1 : i32} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg9 = %c0 to %c64 step %c32 iter_args(%arg10 = %1) -> (!air.async.token) {
        %3 = scf.for %arg11 = %c0 to %c64 step %c32 iter_args(%arg12 = %arg10) -> (!air.async.token) {
          %4 = scf.for %arg13 = %c0 to %c64 step %c32 iter_args(%arg14 = %arg12) -> (!air.async.token) {
            %async_token_4, %results_5 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_6, %results_7 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_8, %results_9 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %5 = air.dma_memcpy_nd async [%async_token_4, %arg14] (%results_5[] [] [], %arg6[%arg9, %arg13] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %6 = air.dma_memcpy_nd async [%async_token_6, %arg14] (%results_7[] [] [], %arg7[%arg13, %arg11] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %7 = air.dma_memcpy_nd async [%async_token_8, %arg14] (%results_9[] [] [], %arg8[%arg9, %arg11] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %async_token_10 = air.execute [%7, %6, %5] {
              linalg.matmul ins(%results_5, %results_7 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_9 : memref<32x32xi32, 2>)
            }
            %8 = air.dma_memcpy_nd async [%async_token_10] (%arg8[%arg9, %arg11] [%c32, %c32] [%c64, %c1], %results_9[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
            %async_token_11 = air.execute [%async_token_10] {
              memref.dealloc %results_5 : memref<32x32xi32, 2>
            }
            %async_token_12 = air.execute [%async_token_10] {
              memref.dealloc %results_7 : memref<32x32xi32, 2>
            }
            %async_token_13 = air.execute [%8] {
              memref.dealloc %results_9 : memref<32x32xi32, 2>
            }
            scf.yield %8 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        scf.yield %3 : !air.async.token
      }
      air.herd_terminator
    }
    return %results_2 : memref<64x64xi32>
  }

// CHECK-LABEL: func.func @gemm_k_dim_tiling
// CHECK: air.launch
// CHECK: scf.for{{.*}}
// CHECK: air.channel.put{{.*}}@channel_8
// CHECK: air.segment @segment_0
// CHECK: scf.for{{.*}}
// CHECK: air.channel.get{{.*}}@channel_8
// CHECK: air.herd @herd_0

  func.func @gemm_k_dim_tiling(%arg0: memref<2048x2048xi32>) {
    %c32 = arith.constant 32 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) args(%arg8=%arg0) : memref<2048x2048xi32> {
      %1 = air.segment @segment_0 async  args(%arg10=%arg3, %arg11=%arg4, %arg13=%arg8) : index, index, memref<2048x2048xi32> {
        %c1 = arith.constant 1 : index
        %c2048 = arith.constant 2048 : index
        %c64 = arith.constant 64 : index
        %c2 = arith.constant 2 : index
        %async_token_1, %results_2 = air.execute -> (index) {
          %4 = affine.apply #map()[%arg10]
          air.execute_terminator %4 : index
        }
        %async_token_3, %results_4 = air.execute -> (index) {
          %4 = affine.apply #map()[%arg11]
          air.execute_terminator %4 : index
        }
        %2 = air.herd @herd_0 async tile (%arg15, %arg16) in (%arg17=%c2, %arg18=%c2) args(%arg20=%arg13, %arg21=%results_2, %arg23=%results_4) : memref<2048x2048xi32>, index, index attributes {id = 1 : i32} {
          %c1_8 = arith.constant 1 : index
          %c64_9 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c32_10 = arith.constant 32 : index
          %c2048_11 = arith.constant 2048 : index
          %4 = air.wait_all async
          %5 = scf.for %arg24 = %c0 to %c2048_11 step %c256 iter_args(%arg25 = %4) -> (!air.async.token) {
            %async_token_20, %results_21 = air.execute -> (memref<64x256xi32, 1>) {
              %alloc = memref.alloc() : memref<64x256xi32, 1>
              air.execute_terminator %alloc : memref<64x256xi32, 1>
            }
            %7 = air.dma_memcpy_nd async [%arg25, %async_token_20] (%results_21[] [] [], %arg20[%arg21, %arg24] [%c64_9, %c256] [%c2048_11, %c1_8]) {id = 1 : i32} : (memref<64x256xi32, 1>, memref<2048x2048xi32>)
            %async_token_24 = air.execute [%7] {
              memref.dealloc %results_21 : memref<64x256xi32, 1>
            }
            %11 = air.wait_all async [%7]
            scf.yield %11 : !air.async.token
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }

// CHECK-LABEL: func.func @l2_l3_dma_hoisting
// CHECK: air.launch
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_9
// CHECK: air.segment @segment_0
// CHECK: scf.parallel
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@channel_9
// CHECK: air.herd @herd_0

  func.func @l2_l3_dma_hoisting(%arg0: memref<2048x2048xi32>, %arg1: memref<2048x2048xi32>, %arg2: memref<2048x2048xi32>) {
    %c32 = arith.constant 32 : index
    %async_token, %results = air.execute -> (memref<2048x2048xi32>) {
      %alloc = memref.alloc() : memref<2048x2048xi32>
      air.execute_terminator %alloc : memref<2048x2048xi32>
    }
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) args(%arg7=%arg0) : memref<2048x2048xi32> attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  args(%arg8=%arg3, %arg9=%arg7) : index, memref<2048x2048xi32> attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %async_token_1, %results_2 = air.execute -> (index) {
          %3 = affine.apply #map1()[%arg8]
          air.execute_terminator %3 : index
        }
        %async_token_3, %results_4 = air.execute -> (memref<32x256xi32, 1>) {
          %alloc = memref.alloc() : memref<32x256xi32, 1>
          air.execute_terminator %alloc : memref<32x256xi32, 1>
        }
        %2 = air.herd @herd_0 async [%async_token_3]  tile (%arg10, %arg11) in (%arg12=%c2, %arg13=%c2) args(%arg14=%results_2, %arg15=%arg9, %arg16=%results_4) : index, memref<2048x2048xi32>, memref<32x256xi32, 1> attributes {id = 3 : i32} {
          %c1 = arith.constant 1 : index
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c32_6 = arith.constant 32 : index
          %c2048 = arith.constant 2048 : index
          %async_token_7, %results_8 = air.execute -> (index) {
            %4 = affine.apply #map()[%arg10]
            air.execute_terminator %4 : index
          }
          %async_token_9, %results_10 = air.execute [%async_token_7] -> (index) {
            %4 = arith.addi %arg14, %results_8 : index
            air.execute_terminator %4 : index
          }
          %3 = scf.for %arg17 = %c0 to %c2048 step %c256 iter_args(%arg18 = %async_token_9) -> (!air.async.token) {
            %4 = air.dma_memcpy_nd async [%arg18] (%arg16[] [] [], %arg15[%results_10, %arg17] [%c32_6, %c256] [%c2048, %c1]) {id = 1 : i32} : (memref<32x256xi32, 1>, memref<2048x2048xi32>)
            scf.yield %4 : !air.async.token
          }
          air.herd_terminator
        }
        %async_token_5 = air.execute [%2] {
          memref.dealloc %results_4 : memref<32x256xi32, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    %async_token_0 = air.execute [%async_token] {
      memref.copy %results, %arg2 : memref<2048x2048xi32> to memref<2048x2048xi32>
    }
    return
  }
}
