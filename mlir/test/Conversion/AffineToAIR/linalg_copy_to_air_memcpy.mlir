// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -affine-to-air -cse
// CHECK: func @myFunc
// CHECK: air.dma_memcpy_2d (%3, %arg6, [%c0, %c0], [%1, %c0], %c1024, %c64, %c64) {id = 1 : i32} : (memref<16x64xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
// CHECK: air.dma_memcpy_2d (%4, %arg7, [%c0, %c0], [%c0, %2], %c1024, %c64, %c16) {id = 2 : i32} : (memref<64x16xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
// CHECK: air.dma_memcpy_2d (%5, %arg8, [%c0, %c0], [%1, %2], %c256, %c64, %c16) {id = 3 : i32} : (memref<16x16xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
// CHECK: air.dma_memcpy_2d (%arg8, %5, [%1, %2], [%c0, %c0], %c256, %c64, %c16) {id = 4 : i32} : (memref<64x64xf32>, memref<16x16xf32, 2>, [index, index], [index, index], index, index, index) -> ()
#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
#map0 = affine_map<(d0, d1, d2)[s0] -> (d0 * 524288 + s0 + d1 * 512 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module  {
  func @myFunc(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> memref<64x64xf32> {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
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
  // CHECK: func @call_linalg_generic
  // CHECK: air.dma_memcpy_nd (%4[] [] [], %arg12[%2, %3, %arg13] [%c32, %c32, %c128_0] [%c524288, %c512_1, %c1]) {id = 5 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
  // CHECK: air.dma_memcpy_nd (%5[] [] [], %arg14[%2, %3, %arg13] [%c32, %c32, %c128_0] [%c524288, %c512_1, %c1]) {id = 6 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
  // CHECK: air.dma_memcpy_nd (%6[] [] [], %arg15[%2, %3, %arg13] [%c32, %c32, %c128_0] [%c524288, %c512_1, %c1]) {id = 7 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
  func @call_linalg_generic(%arg0: memref<4096x1024x512xi32>, %arg1: memref<4096x1024x512xi32>, %arg2: memref<4096x1024x512xi32>) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c128 = arith.constant 128 : index
    scf.for %arg3 = %c0 to %c4096 step %c128 {
      scf.for %arg4 = %c0 to %c1024 step %c128 {
        scf.for %arg5 = %c0 to %c512 step %c128 {
          scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c128, %c128) step (%c32, %c32) {
            %0 = arith.addi %arg3, %arg6 : index
            %1 = arith.addi %arg4, %arg7 : index
            %2 = memref.subview %arg0[%0, %1, %arg5] [32, 32, 128] [1, 1, 1] : memref<4096x1024x512xi32> to memref<32x32x128xi32, #map0>
            %3 = memref.subview %arg1[%0, %1, %arg5] [32, 32, 128] [1, 1, 1] : memref<4096x1024x512xi32> to memref<32x32x128xi32, #map0>
            %4 = memref.subview %arg2[%0, %1, %arg5] [32, 32, 128] [1, 1, 1] : memref<4096x1024x512xi32> to memref<32x32x128xi32, #map0>
            %5 = memref.alloc() : memref<32x32x128xi32, 2>
            %6 = memref.alloc() : memref<32x32x128xi32, 2>
            %7 = memref.alloc() : memref<32x32x128xi32, 2>
            linalg.copy(%2, %5) : memref<32x32x128xi32, #map0>, memref<32x32x128xi32, 2> 
            linalg.copy(%3, %6) : memref<32x32x128xi32, #map0>, memref<32x32x128xi32, 2> 
            linalg.copy(%4, %7) : memref<32x32x128xi32, #map0>, memref<32x32x128xi32, 2> 
            linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6 : memref<32x32x128xi32, 2>, memref<32x32x128xi32, 2>) outs(%7 : memref<32x32x128xi32, 2>) {
            ^bb0(%arg8: i32, %arg9: i32, %arg10: i32):  // no predecessors
              %8 = arith.muli %arg8, %arg9 : i32
              linalg.yield %8 : i32
            }
            linalg.copy(%7, %4) : memref<32x32x128xi32, 2>, memref<32x32x128xi32, #map0> 
            memref.dealloc %5 : memref<32x32x128xi32, 2>
            memref.dealloc %6 : memref<32x32x128xi32, 2>
            memref.dealloc %7 : memref<32x32x128xi32, 2>
            scf.yield
          }
        }
      }
    }
    return
  }
}
