// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt -air-prune-linalg-generic-input-dma %s | FileCheck %s

// Remove the redundant DMA copying into linalg.generic

#map0 = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: module
// CHECK: func.func @forward
module attributes {torch.debug_module_name = "model"} {
  func.func @forward(%arg0: memref<256x256xi32>, %arg1: memref<256x256xi32>) -> memref<256x256xi32> {
    %c2 = arith.constant 2 : index
    %asyncToken, %valOut = air.region async  {
      %1 = memref.alloc() {alignment = 128 : i64} : memref<256x256xi32>
      air.region_terminator %1 : memref<256x256xi32>
    } {id = 1 : i32} : (memref<256x256xi32>)
    %0 = air.launch_herd async [%asyncToken]  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%valOut) : memref<256x256xi32>, memref<256x256xi32>, memref<256x256xi32> attributes {id = 1 : i32, sym_name = "herd_0"} {
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %asyncToken_0, %valOut_1 = air.region async  {
        %3 = affine.apply #map0()[%arg2]
        air.region_terminator %3 : index
      } {id = 2 : i32} : (index)
      %asyncToken_2, %valOut_3 = air.region async  {
        %3 = affine.apply #map0()[%arg3]
        air.region_terminator %3 : index
      } {id = 3 : i32} : (index)
      %1 = air.wait_all async [%asyncToken_0, %asyncToken_2] 
      %2 = scf.for %arg9 = %c0 to %c128 step %c64 iter_args(%arg10 = %1) -> (!air.async.token) {
        %3 = scf.for %arg11 = %c0 to %c128 step %c32 iter_args(%arg12 = %arg10) -> (!air.async.token) {
          %asyncToken_4, %valOut_5 = air.region async [%arg12]  : (!air.async.token) {
            %9 = arith.addi %valOut_1, %arg9 : index
            air.region_terminator %9 : index
          } {id = 4 : i32} : (index)
          %asyncToken_6, %valOut_7 = air.region async [%arg12]  : (!air.async.token) {
            %9 = arith.addi %valOut_3, %arg11 : index
            air.region_terminator %9 : index
          } {id = 5 : i32} : (index)
          %asyncToken_8, %valOut_9 = air.region async [%arg12]  : (!air.async.token) {
            %9 = memref.alloc() : memref<64x32xi32, 2>
            air.region_terminator %9 : memref<64x32xi32, 2>
          } {id = 6 : i32} : (memref<64x32xi32, 2>)
          %asyncToken_10, %valOut_11 = air.region async [%arg12]  : (!air.async.token) {
            %9 = memref.alloc() : memref<64x32xi32, 2>
            air.region_terminator %9 : memref<64x32xi32, 2>
          } {id = 7 : i32} : (memref<64x32xi32, 2>)
          %asyncToken_12, %valOut_13 = air.region async [%arg12]  : (!air.async.token) {
            %9 = memref.alloc() : memref<64x32xi32, 2>
            air.region_terminator %9 : memref<64x32xi32, 2>
          } {id = 8 : i32} : (memref<64x32xi32, 2>)

          // CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd async 
          // CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd async 
          // CHECK: %[[EVENT2:.*]] = air.region async [%[[EVENT1]]{{.*}}%[[EVENT0]]]
          %4 = air.dma_memcpy_nd async [%asyncToken_8, %asyncToken_6, %asyncToken_4] (%valOut_9[] [] [], %arg6[%valOut_5, %valOut_7] [%c64, %c32] [%c256, %c1]) {id = 1 : i32} : (memref<64x32xi32, 2>, memref<256x256xi32>)
          %5 = air.dma_memcpy_nd async [%asyncToken_10, %asyncToken_6, %asyncToken_4] (%valOut_11[] [] [], %arg7[%valOut_5, %valOut_7] [%c64, %c32] [%c256, %c1]) {id = 2 : i32} : (memref<64x32xi32, 2>, memref<256x256xi32>)
          %6 = air.dma_memcpy_nd async [%asyncToken_12, %asyncToken_6, %asyncToken_4] (%valOut_13[] [] [], %arg8[%valOut_5, %valOut_7] [%c64, %c32] [%c256, %c1]) {id = 3 : i32} : (memref<64x32xi32, 2>, memref<256x256xi32>)
          %asyncToken_14 = air.region async [%6, %5, %4]  : (!air.async.token, !air.async.token, !air.async.token) {
            linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%valOut_9, %valOut_11 : memref<64x32xi32, 2>, memref<64x32xi32, 2>) outs(%valOut_13 : memref<64x32xi32, 2>) {
            ^bb0(%arg13: i32, %arg14: i32, %arg15: i32):
              %9 = arith.addi %arg13, %arg14 : i32
              linalg.yield %9 : i32
            }
            air.region_terminator
          } {id = 10 : i32}
          %7 = air.dma_memcpy_nd async [%asyncToken_14] (%arg8[%valOut_5, %valOut_7] [%c64, %c32] [%c256, %c1], %valOut_13[] [] []) {id = 4 : i32} : (memref<256x256xi32>, memref<64x32xi32, 2>)
          %asyncToken_15 = air.region async [%asyncToken_14]  : (!air.async.token) {
            memref.dealloc %valOut_9 : memref<64x32xi32, 2>
            air.region_terminator
          } {id = 11 : i32}
          %asyncToken_16 = air.region async [%asyncToken_14]  : (!air.async.token) {
            memref.dealloc %valOut_11 : memref<64x32xi32, 2>
            air.region_terminator
          } {id = 12 : i32}
          %asyncToken_17 = air.region async [%7]  : (!air.async.token) {
            memref.dealloc %valOut_13 : memref<64x32xi32, 2>
            air.region_terminator
          } {id = 13 : i32}
          %8 = air.wait_all async [%asyncToken_15, %asyncToken_16, %asyncToken_17] 
          scf.yield %8 : !air.async.token
        }
        scf.yield %3 : !air.async.token
      }
      air.herd_terminator
    }
    return %valOut : memref<256x256xi32>
  }
}