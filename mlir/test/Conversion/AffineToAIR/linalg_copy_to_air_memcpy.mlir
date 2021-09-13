// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -affine-to-air -cse
// CHECK: air.dma_memcpy_2d (%5, %arg6, [%c0, %c0], [%2, %c0], %c1024, %c64, %c64) : (memref<16x64xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
// CHECK: air.dma_memcpy_2d (%6, %arg7, [%c0, %c0], [%c0, %4], %c1024, %c64, %c16) : (memref<64x16xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
// CHECK: air.dma_memcpy_2d (%7, %arg8, [%c0, %c0], [%2, %4], %c256, %c64, %c16) : (memref<16x16xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
// CHECK: air.dma_memcpy_2d (%arg8, %7, [%2, %4], [%c0, %c0], %c256, %c64, %c16) : (memref<64x64xf32>, memref<16x16xf32, 2>, [index, index], [index, index], index, index, index) -> ()
#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
module  {
  func @myFunc(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> memref<64x64xf32> {
    %c64 = constant 64 : index
    %c16 = constant 16 : index
    %c0 = constant 0 : index
    %0 = memref.alloc() : memref<64x64xf32>
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c64) step (%c16, %c16) {
      %1 = memref.subview %arg0[%arg2, 0] [16, 64] [1, 1] : memref<64x64xf32> to memref<16x64xf32, #map>
      %2 = memref.subview %arg1[0, %arg3] [64, 16] [1, 1] : memref<64x64xf32> to memref<64x16xf32, #map>
      %3 = memref.subview %0[%arg2, %arg3] [16, 16] [1, 1] : memref<64x64xf32> to memref<16x16xf32, #map>
      %4 = memref.alloc() : memref<16x64xf32, 2>
      %5 = memref.alloc() : memref<64x16xf32, 2>
      %6 = memref.alloc() : memref<16x16xf32, 2>
      linalg.copy(%1, %4) : memref<16x64xf32, #map>, memref<16x64xf32, 2>
      linalg.copy(%2, %5) : memref<64x16xf32, #map>, memref<64x16xf32, 2>
      linalg.copy(%3, %6) : memref<16x16xf32, #map>, memref<16x16xf32, 2>
      linalg.matmul ins(%4, %5 : memref<16x64xf32, 2>, memref<64x16xf32, 2>) outs(%6 : memref<16x16xf32, 2>)
      linalg.copy(%6, %3) : memref<16x16xf32, 2>, memref<16x16xf32, #map>
      memref.dealloc %4 : memref<16x64xf32, 2>
      memref.dealloc %5 : memref<64x16xf32, 2>
      memref.dealloc %6 : memref<16x16xf32, 2>
      scf.yield
    }
    return %0 : memref<64x64xf32>
  }
}
