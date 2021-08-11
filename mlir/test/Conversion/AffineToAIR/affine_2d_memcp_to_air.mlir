// RUN: air-opt %s -affine-to-air -simplify-affine-structures -cse | FileCheck %s
// CHECK: air.dma_memcpy_2d (%3, %arg9, [%c0, %c0], [%4, %5], %c4096, %c256, %c64) {id = 1 : i32} : (memref<64x64xi32, 2>, memref<256x256xi32>, [index, index], [index, index], index, index, index) -> ()
// CHECK: air.dma_memcpy_2d (%arg10, %6, [%4, %5], [%c0, %c0], %c4096, %c256, %c64) {id = 2 : i32} : (memref<256x256xi32>, memref<64x64xi32, 2>, [index, index], [index, index], index, index, index) -> ()
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 * 2)>
module  {
  func @myAddOne(%arg0: tensor<256x256xi32>) -> tensor<256x256xi32> {
    %c4096 = constant 4096 : index
    %c256 = constant 256 : index
    %c64 = constant 64 : index
    %c0 = constant 0 : index
    %0 = memref.alloc() : memref<256x256xi32>
    %1 = "aten.type_cast"(%arg0) : (tensor<256x256xi32>) -> memref<256x256xi32>
    affine.for %arg1 = 0 to 2 {
      affine.for %arg2 = 0 to 2 {
        affine.parallel (%arg3, %arg4) = (0, 0) to (2, 2) {
          %3 = affine.apply #map0(%arg3)
          %4 = affine.apply #map0(%arg4)
          %5 = affine.apply #map1(%3, %arg1)
          %6 = affine.apply #map1(%4, %arg2)
          %7 = memref.alloc() : memref<64x64xi32, 2>
          %8 = memref.alloc() : memref<1xi32>
          affine.dma_start %1[%5 * 64, %6 * 64], %7[%c0, %c0], %8[%c0], %c4096, %c256, %c64 : memref<256x256xi32>, memref<64x64xi32, 2>, memref<1xi32>
          affine.dma_wait %8[%c0], %c4096 : memref<1xi32>
          %9 = memref.alloc() : memref<64x64xi32, 2>
          %10 = memref.alloc() : memref<1xi32>
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %11 = affine.load %7[%arg5, %arg6] : memref<64x64xi32, 2>
              %c1_i32 = constant 1 : i32
              %12 = addi %11, %c1_i32 : i32
              affine.store %12, %9[%arg5, %arg6] : memref<64x64xi32, 2>
            }
          }
          affine.dma_start %9[%c0, %c0], %0[%5 * 64, %6 * 64], %10[%c0], %c4096, %c256, %c64 : memref<64x64xi32, 2>, memref<256x256xi32>, memref<1xi32>
          affine.dma_wait %10[%c0], %c4096 : memref<1xi32>
          memref.dealloc %10 : memref<1xi32>
          memref.dealloc %9 : memref<64x64xi32, 2>
          memref.dealloc %8 : memref<1xi32>
          memref.dealloc %7 : memref<64x64xi32, 2>
        }
      }
    } {affine_opt_label = "affine_opt"}
    %2 = "aten.type_cast"(%0) : (memref<256x256xi32>) -> tensor<256x256xi32>
    return %2 : tensor<256x256xi32>
  }
}