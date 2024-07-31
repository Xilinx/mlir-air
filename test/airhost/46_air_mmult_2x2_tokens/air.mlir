//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
module {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c1 = arith.constant 1 : index
    %async_token, %results = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }
    %0 = air.launch async [%async_token] (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%results, %arg8=%arg0, %arg9=%arg1) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  args(%arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %2 = air.herd @herd_0 async  tile (%arg15, %arg16) in (%arg17=%c2, %arg18=%c2) args(%arg21=%arg12, %arg22=%arg13, %arg23=%arg14) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 3 : i32} {
          %c1_5 = arith.constant 1 : index
          %c0_i32 = arith.constant 0 : i32
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %async_token_6, %results_7 = air.execute -> (index) {
            %6 = affine.apply #map1()[%arg15]
            air.execute_terminator %6 : index
          }
          %async_token_8, %results_9 = air.execute -> (index) {
            %6 = affine.apply #map1()[%arg16]
            air.execute_terminator %6 : index
          }
          %async_token_10, %results_11 = air.execute [%async_token_6] -> (index) {
            %6 = arith.addi %c64, %results_7 : index
            air.execute_terminator %6 : index
          }
          %async_token_12, %results_13 = air.execute [%async_token_8] -> (index) {
            %6 = arith.addi %c64, %results_9 : index
            air.execute_terminator %6 : index
          }
          %async_token_14, %results_15 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %async_token_16 = air.execute [%async_token_14] {
            linalg.fill ins(%c0_i32 : i32) outs(%results_15 : memref<32x32xi32, 2>)
          }
          %3 = air.wait_all async [%async_token_10, %async_token_12, %async_token_16] 
          %4 = scf.for %arg24 = %c0 to %c64 step %c32 iter_args(%arg25 = %3) -> (!air.async.token) {
            %async_token_18, %results_19 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_20, %results_21 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %6 = affine.if #set()[%arg15, %arg16] -> !air.async.token {
              %8 = air.dma_memcpy_nd async [%arg25, %async_token_18] (%results_19[] [] [], %arg22[%results_11, %arg24] [%c32, %c32] [%c64, %c1_5]) {broadcast_set = #set, id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
              affine.yield %8 : !air.async.token
            } else {
              %8 = air.dma_memcpy_nd async [%arg25, %async_token_18] (%results_19[] [] [], %arg22[%results_11, %arg24] [%c32, %c32] [%c64, %c1_5]) {broadcast_set = #set1, id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
              affine.yield %8 : !air.async.token
            }
            %7 = affine.if #set2()[%arg15, %arg16] -> !air.async.token {
              %8 = air.dma_memcpy_nd async [%arg25, %async_token_20] (%results_21[] [] [], %arg23[%arg24, %results_13] [%c32, %c32] [%c64, %c1_5]) {broadcast_set = #set2, id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
              affine.yield %8 : !air.async.token
            } else {
              %8 = air.dma_memcpy_nd async [%arg25, %async_token_20] (%results_21[] [] [], %arg23[%arg24, %results_13] [%c32, %c32] [%c64, %c1_5]) {broadcast_set = #set3, id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
              affine.yield %8 : !air.async.token
            }
            %async_token_22 = air.execute [%7, %6] {
              linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_19, %results_21 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_15 : memref<32x32xi32, 2>)
            }
            %async_token_23 = air.execute [%async_token_22] {
              memref.dealloc %results_19 : memref<32x32xi32, 2>
            }
            %async_token_24 = air.execute [%async_token_22] {
              memref.dealloc %results_21 : memref<32x32xi32, 2>
            }
            scf.yield %async_token_22 : !air.async.token
          }
          %5 = air.dma_memcpy_nd async [%4] (%arg21[%results_11, %results_13] [%c32, %c32] [%c64, %c1_5], %results_15[] [] []) {id = 5 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
          %async_token_17 = air.execute [%5] {
            memref.dealloc %results_15 : memref<32x32xi32, 2>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    %async_token_0 = air.execute [%0] {
      memref.copy %results, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    }
    return
  }
}

