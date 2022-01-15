// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-linalg-codegen=test-patterns
// CHECK: %1 = memref.subview %arg0[%arg2, 0] [16, 64] [1, 1] : memref<64x64xf32> to memref<16x64xf32, #map>
// CHECK: %2 = memref.subview %arg1[0, %arg3] [64, 16] [1, 1] : memref<64x64xf32> to memref<64x16xf32, #map>
// CHECK: %3 = memref.subview %0[%arg2, %arg3] [16, 16] [1, 1] : memref<64x64xf32> to memref<16x16xf32, #map>
// CHECK: %4 = memref.alloc() : memref<16x64xf32, 2>
// CHECK: %5 = memref.alloc() : memref<64x16xf32, 2>
// CHECK: %6 = memref.alloc() : memref<16x16xf32, 2>
// CHECK: linalg.copy(%1, %4) : memref<16x64xf32, #map>, memref<16x64xf32, 2>
// CHECK: linalg.copy(%2, %5) : memref<64x16xf32, #map>, memref<64x16xf32, 2>
// CHECK: linalg.copy(%3, %6) : memref<16x16xf32, #map>, memref<16x16xf32, 2>
// CHECK: linalg.matmul ins(%4, %5 : memref<16x64xf32, 2>, memref<64x16xf32, 2>) outs(%6 : memref<16x16xf32, 2>)
// CHECK: linalg.copy(%6, %3) : memref<16x16xf32, 2>, memref<16x16xf32, #map>
// CHECK: memref.dealloc %4 : memref<16x64xf32, 2>
// CHECK: memref.dealloc %5 : memref<64x16xf32, 2>
// CHECK: memref.dealloc %6 : memref<16x16xf32, 2>
#map0 = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 64 + d1)>
#map2 = affine_map<(d0, d1) -> (d0 * 16 + d1)>
module  {
  func @myFunc(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> memref<64x64xf32> {
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %0 = memref.alloc() : memref<64x64xf32>
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c64) step (%c16, %c16) {
      %1 = memref.subview %arg0[%arg2, 0] [16, 64] [1, 1] : memref<64x64xf32> to memref<16x64xf32, #map0>
      %2 = memref.subview %arg1[0, %arg3] [64, 16] [1, 1] : memref<64x64xf32> to memref<64x16xf32, #map0>
      %3 = memref.subview %0[%arg2, %arg3] [16, 16] [1, 1] : memref<64x64xf32> to memref<16x16xf32, #map0>
      %4 = memref.alloc(%c4096) : memref<?xi8>
      %5 = memref.view %4[%c0][] : memref<?xi8> to memref<16x64xf32>
      %6 = memref.subview %5[0, 0] [16, 64] [1, 1] : memref<16x64xf32> to memref<16x64xf32, #map1>
      %7 = memref.alloc(%c4096) : memref<?xi8>
      %8 = memref.view %7[%c0][] : memref<?xi8> to memref<64x16xf32>
      %9 = memref.subview %8[0, 0] [64, 16] [1, 1] : memref<64x16xf32> to memref<64x16xf32, #map2>
      %10 = memref.alloc(%c1024) : memref<?xi8>
      %11 = memref.view %10[%c0][] : memref<?xi8> to memref<16x16xf32>
      %12 = memref.subview %11[0, 0] [16, 16] [1, 1] : memref<16x16xf32> to memref<16x16xf32, #map2>
      linalg.copy(%1, %6) : memref<16x64xf32, #map0>, memref<16x64xf32, #map1>
      linalg.copy(%2, %9) : memref<64x16xf32, #map0>, memref<64x16xf32, #map2>
      linalg.copy(%3, %12) : memref<16x16xf32, #map0>, memref<16x16xf32, #map2>
      linalg.matmul ins(%6, %9 : memref<16x64xf32, #map1>, memref<64x16xf32, #map2>) outs(%12 : memref<16x16xf32, #map2>)
      linalg.copy(%12, %3) : memref<16x16xf32, #map2>, memref<16x16xf32, #map0>
      memref.dealloc %4 : memref<?xi8>
      memref.dealloc %7 : memref<?xi8>
      memref.dealloc %10 : memref<?xi8>
      scf.yield
    }
    return %0 : memref<64x64xf32>
  }
}