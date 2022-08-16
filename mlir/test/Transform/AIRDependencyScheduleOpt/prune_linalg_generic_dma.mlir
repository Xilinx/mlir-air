// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt -air-dependency -air-prune-linalg-generic-input-dma %s | FileCheck %s

// Remove the redundant DMA copying into linalg.generic

// CHECK-LABEL: module
// CHECK: func.func @forward
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd async 
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd async 
// CHECK: %[[EVENT2:.*]] = air.region async [%[[EVENT1]]{{.*}}%[[EVENT0]]]

#map0 = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "model"} {
  func.func @forward(%arg0: memref<256x256xi32>, %arg1: memref<256x256xi32>) -> memref<256x256xi32> {
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256xi32>
    air.launch_herd  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%0) : memref<256x256xi32>, memref<256x256xi32>, memref<256x256xi32> attributes {sym_name = "herd_0"} {
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %1 = affine.apply #map0()[%arg2]
      %2 = affine.apply #map0()[%arg3]
      scf.for %arg9 = %c0 to %c128 step %c64 {
        scf.for %arg10 = %c0 to %c128 step %c32 {
          %3 = arith.addi %1, %arg9 : index
          %4 = arith.addi %2, %arg10 : index
          %5 = memref.alloc() : memref<64x32xi32, 2>
          %6 = memref.alloc() : memref<64x32xi32, 2>
          %7 = memref.alloc() : memref<64x32xi32, 2>
          air.dma_memcpy_nd (%5[] [] [], %arg6[%3, %4] [%c64, %c32] [%c256, %c1]) {id = 1 : i32} : (memref<64x32xi32, 2>, memref<256x256xi32>)
          air.dma_memcpy_nd (%6[] [] [], %arg7[%3, %4] [%c64, %c32] [%c256, %c1]) {id = 2 : i32} : (memref<64x32xi32, 2>, memref<256x256xi32>)
          air.dma_memcpy_nd (%7[] [] [], %arg8[%3, %4] [%c64, %c32] [%c256, %c1]) {id = 3 : i32} : (memref<64x32xi32, 2>, memref<256x256xi32>)
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : memref<64x32xi32, 2>, memref<64x32xi32, 2>) outs(%7 : memref<64x32xi32, 2>) {
          ^bb0(%arg11: i32, %arg12: i32, %arg13: i32):
            %8 = arith.addi %arg11, %arg12 : i32
            linalg.yield %8 : i32
          }
          air.dma_memcpy_nd (%arg8[%3, %4] [%c64, %c32] [%c256, %c1], %7[] [] []) {id = 4 : i32} : (memref<256x256xi32>, memref<64x32xi32, 2>)
          memref.dealloc %5 : memref<64x32xi32, 2>
          memref.dealloc %6 : memref<64x32xi32, 2>
          memref.dealloc %7 : memref<64x32xi32, 2>
        }
      }
      air.herd_terminator
    }
    return %0 : memref<256x256xi32>
  }
}