// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-dependency | FileCheck %s

#map0 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
module attributes {torch.debug_module_name = "model"} {
  func.func @forward(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() : memref<1024x1024xf32>
    // CHECK: = air.region async
    // CHECK: air.region_terminator
    linalg.fill ins(%cst : f32) outs(%0 : memref<1024x1024xf32>)
    // CHECK: = air.region async
    // CHECK: air.region_terminator
    %1 = memref.cast %arg2 : memref<?x?xf32> to memref<1024x1024xf32>
    // CHECK: = air.region async
    // CHECK: air.region_terminator
    linalg.copy ins(%0 : memref<1024x1024xf32>) outs(%1 : memref<1024x1024xf32>)
    // CHECK: = air.region async
    // CHECK: air.region_terminator
    scf.for %arg3 = %c0 to %c1024 step %c64 {
    // CHECK: = scf.for
      scf.for %arg4 = %c0 to %c1024 step %c64 {
      // CHECK: = scf.for
        scf.for %arg5 = %c0 to %c1024 step %c64 {
        // CHECK: = scf.for
          %2 = memref.subview %arg0[%arg3, %arg5] [64, 64] [1, 1] : memref<1024x1024xf32> to memref<64x64xf32, #map0>
          %3 = memref.subview %arg1[%arg5, %arg4] [64, 64] [1, 1] : memref<1024x1024xf32> to memref<64x64xf32, #map0>
          %4 = memref.subview %1[%arg3, %arg4] [64, 64] [1, 1] : memref<1024x1024xf32> to memref<64x64xf32, #map0>
          %5 = memref.alloc() : memref<64x64xf32, 1>
          // CHECK: = air.region async
          // CHECK: air.region_terminator
          %6 = memref.alloc() : memref<64x64xf32, 1>
          // CHECK: = air.region async
          // CHECK: air.region_terminator
          %7 = memref.alloc() : memref<64x64xf32, 1>
          // CHECK: = air.region async
          // CHECK: air.region_terminator
          %c0_0 = arith.constant 0 : index
          %c1024_1 = arith.constant 1024 : index
          %c64_2 = arith.constant 64 : index
          %c4096 = arith.constant 4096 : index
          air.dma_memcpy_2d (%5, %arg0, [%c0_0, %c0_0], [%arg3, %arg5], %c4096, %c1024_1, %c64_2) {id = 1 : i32} : (memref<64x64xf32, 1>, memref<1024x1024xf32>, [index, index], [index, index], index, index, index) -> ()
          // CHECK: = air.dma_memcpy_2d async
          %c0_3 = arith.constant 0 : index
          %c1024_4 = arith.constant 1024 : index
          %c64_5 = arith.constant 64 : index
          %c4096_6 = arith.constant 4096 : index
          air.dma_memcpy_2d (%6, %arg1, [%c0_3, %c0_3], [%arg5, %arg4], %c4096_6, %c1024_4, %c64_5) {id = 2 : i32} : (memref<64x64xf32, 1>, memref<1024x1024xf32>, [index, index], [index, index], index, index, index) -> ()
          // CHECK: = air.dma_memcpy_2d async
          %c0_7 = arith.constant 0 : index
          %c1024_8 = arith.constant 1024 : index
          %c64_9 = arith.constant 64 : index
          %c4096_10 = arith.constant 4096 : index
          air.dma_memcpy_2d (%7, %1, [%c0_7, %c0_7], [%arg3, %arg4], %c4096_10, %c1024_8, %c64_9) {id = 3 : i32} : (memref<64x64xf32, 1>, memref<1024x1024xf32>, [index, index], [index, index], index, index, index) -> ()
          // CHECK: = air.dma_memcpy_2d async
          air.launch_herd tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%5, %arg11=%6, %arg12=%7) : memref<64x64xf32, 1>, memref<64x64xf32, 1>, memref<64x64xf32, 1> attributes {sym_name = "herd_0"} {
          // CHECK: = air.launch_herd @herd_0 async
            %c0_15 = arith.constant 0 : index
            %c64_16 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %8 = affine.apply #map1()[%arg6]
            // CHECK: = air.region async
            // CHECK: air.region_terminator
            %9 = affine.apply #map1()[%arg7]
            // CHECK: = air.region async
            // CHECK: air.region_terminator
            // CHECK: = air.wait_all async
            scf.for %arg13 = %c0_15 to %c64_16 step %c32 {
              %10 = memref.subview %arg10[%8, %arg13] [32, 32] [1, 1] : memref<64x64xf32, 1> to memref<32x32xf32, #map2, 1>
              %11 = memref.subview %arg11[%arg13, %9] [32, 32] [1, 1] : memref<64x64xf32, 1> to memref<32x32xf32, #map2, 1>
              %12 = memref.subview %arg12[%8, %9] [32, 32] [1, 1] : memref<64x64xf32, 1> to memref<32x32xf32, #map2, 1>
              %13 = memref.alloc() : memref<32x32xf32, 2>
              // CHECK: = air.region async
              // CHECK: air.region_terminator
              %14 = memref.alloc() : memref<32x32xf32, 2>
              // CHECK: = air.region async
              // CHECK: air.region_terminator
              %15 = memref.alloc() : memref<32x32xf32, 2>
              // CHECK: = air.region async
              // CHECK: air.region_terminator
              %c0_17 = arith.constant 0 : index
              %c64_18 = arith.constant 64 : index
              %c32_19 = arith.constant 32 : index
              %c1024_20 = arith.constant 1024 : index
              air.dma_memcpy_2d (%13, %arg10, [%c0_17, %c0_17], [%8, %arg13], %c1024_20, %c64_18, %c32_19) {id = 4 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32, 1>, [index, index], [index, index], index, index, index) -> ()
              // CHECK: = air.dma_memcpy_2d async
              %c0_21 = arith.constant 0 : index
              %c64_22 = arith.constant 64 : index
              %c32_23 = arith.constant 32 : index
              %c1024_24 = arith.constant 1024 : index
              air.dma_memcpy_2d (%14, %arg11, [%c0_21, %c0_21], [%arg13, %9], %c1024_24, %c64_22, %c32_23) {id = 5 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32, 1>, [index, index], [index, index], index, index, index) -> ()
              // CHECK: = air.dma_memcpy_2d async
              %c0_25 = arith.constant 0 : index
              %c64_26 = arith.constant 64 : index
              %c32_27 = arith.constant 32 : index
              %c1024_28 = arith.constant 1024 : index
              air.dma_memcpy_2d (%15, %arg12, [%c0_25, %c0_25], [%8, %9], %c1024_28, %c64_26, %c32_27) {id = 6 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32, 1>, [index, index], [index, index], index, index, index) -> ()
              // CHECK: = air.dma_memcpy_2d async
              linalg.matmul ins(%13, %14 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%15 : memref<32x32xf32, 2>)
              // CHECK: = air.region async
              // CHECK: air.region_terminator
              %c0_29 = arith.constant 0 : index
              %c32_30 = arith.constant 32 : index
              %c64_31 = arith.constant 64 : index
              %c1024_32 = arith.constant 1024 : index
              air.dma_memcpy_2d (%arg12, %15, [%8, %9], [%c0_29, %c0_29], %c1024_32, %c64_31, %c32_30) {id = 7 : i32} : (memref<64x64xf32, 1>, memref<32x32xf32, 2>, [index, index], [index, index], index, index, index) -> ()
              // CHECK: = air.dma_memcpy_2d async
              memref.dealloc %13 : memref<32x32xf32, 2>
              // CHECK: = air.region async
              // CHECK: air.region_terminator
              memref.dealloc %14 : memref<32x32xf32, 2>
              // CHECK: = air.region async
              // CHECK: air.region_terminator
              memref.dealloc %15 : memref<32x32xf32, 2>
              // CHECK: = air.region async
              // CHECK: air.region_terminator
            }
            air.herd_terminator
            // CHECK: air.herd_terminator
          }
          %c0_11 = arith.constant 0 : index
          %c64_12 = arith.constant 64 : index
          %c1024_13 = arith.constant 1024 : index
          %c4096_14 = arith.constant 4096 : index
          air.dma_memcpy_2d (%1, %7, [%arg3, %arg4], [%c0_11, %c0_11], %c4096_14, %c1024_13, %c64_12) {id = 8 : i32} : (memref<1024x1024xf32>, memref<64x64xf32, 1>, [index, index], [index, index], index, index, index) -> ()
          // CHECK: = air.dma_memcpy_2d async
          memref.dealloc %5 : memref<64x64xf32, 1>
          // CHECK: = air.region async
          // CHECK: air.region_terminator
          memref.dealloc %6 : memref<64x64xf32, 1>
          // CHECK: = air.region async
          // CHECK: air.region_terminator
          memref.dealloc %7 : memref<64x64xf32, 1>
          // CHECK: = air.region async
          // CHECK: air.region_terminator
        }
        // CHECK: = air.wait_all async
        // CHECK: scf.yield
      }
      // CHECK: scf.yield
    }
    // CHECK: scf.yield
    return
  }
}
