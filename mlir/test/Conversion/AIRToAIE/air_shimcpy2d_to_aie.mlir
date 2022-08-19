// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-to-aie | FileCheck %s
// CHECK: module
#map = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 128)>
module  {
  func.func @myAddOne(%arg0: tensor<256x256xi32>) -> tensor<256x256xi32> {
    %0 = memref.alloc() : memref<256x256xi32>
    %1 = bufferization.to_memref %arg0 : memref<256x256xi32>
    affine.for %arg1 = 0 to 2 {
      affine.for %arg2 = 0 to 2 {
        %c2 = arith.constant 2 : index
        air.herd tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg1, %arg8=%arg2, %arg9=%1, %arg10=%0) : index,index,memref<256x256xi32>,memref<256x256xi32> {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c256 = arith.constant 256 : index
          %c4096 = arith.constant 4096 : index
          %3 = memref.alloc() : memref<64x64xi32, 2>
          %4 = affine.apply #map()[%arg3, %arg7]
          %5 = affine.apply #map()[%arg4, %arg8]
          air.dma_memcpy_2d (%3, %arg9, [%c0, %c0], [%4, %5], %c4096, %c256, %c64) {id = 2 : i32} : (memref<64x64xi32, 2>, memref<256x256xi32>, [index, index], [index, index], index, index, index) -> ()
          %6 = memref.alloc() : memref<64x64xi32, 2>
          affine.for %arg11 = 0 to 64 {
            %c1_i32 = arith.constant 1 : i32
            affine.for %arg12 = 0 to 64 {
              %7 = affine.load %3[%arg11, %arg12] : memref<64x64xi32, 2>
              %8 = arith.addi %7, %c1_i32 : i32
              affine.store %8, %6[%arg11, %arg12] : memref<64x64xi32, 2>
            }
          }
          air.dma_memcpy_2d (%arg10, %6, [%4, %5], [%c0, %c0], %c4096, %c256, %c64) {id = 3 : i32} : (memref<256x256xi32>, memref<64x64xi32, 2>, [index, index], [index, index], index, index, index) -> ()
          memref.dealloc %6 : memref<64x64xi32, 2>
          memref.dealloc %3 : memref<64x64xi32, 2>
          air.herd_terminator
        }
      }
    } {affine_opt_label = "affine_opt"}
    %2 = bufferization.to_tensor %0 : memref<256x256xi32>
    return %2 : tensor<256x256xi32>
  }
}