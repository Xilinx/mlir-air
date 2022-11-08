//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %async_token, %results = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    } {id = 1 : i32}
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%c0_i32 : i32) outs(%results : memref<64x64xi32>)
    } {id = 2 : i32}
    %async_token_1, %results_2 = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    } {id = 3 : i32}
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<64x64xi32> to memref<64x64xi32>
    } {id = 4 : i32}
    %0 = air.herd @herd_0 async [%async_token_3]  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%results_2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 1 : i32} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %async_token_5, %results_6 = air.execute -> (index) {
        %3 = affine.apply #map()[%arg3]
        air.execute_terminator %3 : index
      } {id = 5 : i32}
      %async_token_7, %results_8 = air.execute -> (index) {
        %3 = affine.apply #map()[%arg4]
        air.execute_terminator %3 : index
      } {id = 6 : i32}
      %1 = air.wait_all async [%async_token_5, %async_token_7]  {id = 2 : i32}
      %2 = scf.for %arg10 = %c0 to %c64 step %c32 iter_args(%arg11 = %1) -> (!air.async.token) {
        %c32_9 = arith.constant 32 : index
        %c64_10 = arith.constant 64 : index
        %c1_11 = arith.constant 1 : index
        %async_token_12, %results_13 = air.execute [%arg11] -> (memref<32x32xi32, 2>) {
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          air.execute_terminator %alloc : memref<32x32xi32, 2>
        } {id = 7 : i32}
        %async_token_14, %results_15 = air.execute [%arg11] -> (memref<32x32xi32, 2>) {
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          air.execute_terminator %alloc : memref<32x32xi32, 2>
        } {id = 8 : i32}
        %async_token_16, %results_17 = air.execute [%arg11] -> (memref<32x32xi32, 2>) {
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          air.execute_terminator %alloc : memref<32x32xi32, 2>
        } {id = 9 : i32}
        %3 = air.dma_memcpy_nd async [%async_token_12, %arg11] (%results_13[] [] [], %arg7[%results_6, %arg10] [%c32_9, %c32_9] [%c64_10, %c1_11]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        %4 = air.dma_memcpy_nd async [%async_token_14, %arg11] (%results_15[] [] [], %arg8[%arg10, %results_8] [%c32_9, %c32_9] [%c64_10, %c1_11]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        %5 = air.dma_memcpy_nd async [%async_token_16, %arg11, %arg11] (%results_17[] [] [], %arg9[%results_6, %results_8] [%c32_9, %c32_9] [%c64_10, %c1_11]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        %async_token_18 = air.execute [%4, %5, %3] {
          linalg.matmul ins(%results_13, %results_15 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_17 : memref<32x32xi32, 2>)
        } {id = 10 : i32}
        %6 = air.dma_memcpy_nd async [%async_token_18] (%arg9[%results_6, %results_8] [%c32_9, %c32_9] [%c64_10, %c1_11], %results_17[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
        %async_token_19 = air.execute [%async_token_18] {
          memref.dealloc %results_13 : memref<32x32xi32, 2>
        } {id = 11 : i32}
        %async_token_20 = air.execute [%async_token_18] {
          memref.dealloc %results_15 : memref<32x32xi32, 2>
        } {id = 12 : i32}
        %async_token_21 = air.execute [%6] {
          memref.dealloc %results_17 : memref<32x32xi32, 2>
        } {id = 13 : i32}
        %7 = air.wait_all async [%async_token_19, %async_token_20, %async_token_21]  {id = 1 : i32}
        scf.yield %7 : !air.async.token
      }
      air.herd_terminator
    }
    %async_token_4 = air.execute [%0] {
      memref.copy %results_2, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    } {id = 14 : i32}
    return
  }
}

