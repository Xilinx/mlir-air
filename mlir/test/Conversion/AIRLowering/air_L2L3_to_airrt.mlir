// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-to-std | FileCheck %s
// CHECK: airrt.memcpy_nd({{.*}}) : (memref<64x64xi32, 1>, memref<64x64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) -> ()
// CHECK: airrt.memcpy_nd({{.*}}) : (memref<64x64xi32, 1>, memref<64x64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) -> ()
// CHECK: airrt.memcpy_nd({{.*}}) : (memref<64x64xi32, 1>, memref<64x64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) -> ()
// CHECK: airrt.memcpy_nd({{.*}}) : (memref<64x64xi32>, memref<64x64xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) -> ()
module attributes {torch.debug_module_name = "mmult"}  {
  func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<?x?xi32>) {
    %c0_i32 = constant 0 : i32
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %c64 = constant 64 : index
    %c1 = constant 1 : index
    %0 = memref.alloc() : memref<64x64xi32>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.store %c0_i32, %0[%arg3, %arg4] : memref<64x64xi32>
      }
    }
    %1 = memref.cast %arg2 : memref<?x?xi32> to memref<64x64xi32>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        %5 = affine.load %0[%arg3, %arg4] : memref<64x64xi32>
        affine.store %5, %arg2[%arg3, %arg4] : memref<?x?xi32>
      }
    }
    %2 = memref.alloc() : memref<64x64xi32, 1>
    %3 = memref.alloc() : memref<64x64xi32, 1>
    %4 = memref.alloc() : memref<64x64xi32, 1>
    air.dma_memcpy_nd (%2[] [] [], %arg0[%c0, %c0] [%c64, %c64] [%c64, %c1]) {id = 1 : i32} : (memref<64x64xi32, 1>, memref<64x64xi32>)
    air.dma_memcpy_nd (%3[] [] [], %arg1[%c0, %c0] [%c64, %c64] [%c64, %c1]) {id = 2 : i32} : (memref<64x64xi32, 1>, memref<64x64xi32>)
    air.dma_memcpy_nd (%4[] [] [], %1[%c0, %c0] [%c64, %c64] [%c64, %c1]) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<64x64xi32>)
    air.launch_herd tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%2, %arg8=%3, %arg9=%4) : memref<64x64xi32, 1>,memref<64x64xi32, 1>,memref<64x64xi32, 1>attributes {sym_name = "herd_0"} {
      %c32 = constant 32 : index
      %c0_0 = constant 0 : index
      %c64_1 = constant 64 : index
      %c1_2 = constant 1 : index
      %5 = muli %arg3, %c32 : index
      %6 = muli %arg4, %c32 : index
      scf.for %arg10 = %c0_0 to %c64_1 step %c32 {
        %7 = memref.alloc() : memref<32x32xi32, 2>
        %8 = memref.alloc() : memref<32x32xi32, 2>
        %9 = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%7[] [] [], %arg7[%5, %arg10] [%c32, %c32] [%c64_1, %c1_2]) {id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        air.dma_memcpy_nd (%8[] [] [], %arg8[%arg10, %6] [%c32, %c32] [%c64_1, %c1_2]) {id = 5 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        air.dma_memcpy_nd (%9[] [] [], %arg9[%5, %6] [%c32, %c32] [%c64_1, %c1_2]) {id = 6 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        affine.for %arg11 = 0 to 32 {
          affine.for %arg12 = 0 to 32 {
            affine.for %arg13 = 0 to 32 {
              %10 = affine.load %7[%arg11, %arg13] : memref<32x32xi32, 2>
              %11 = affine.load %8[%arg13, %arg12] : memref<32x32xi32, 2>
              %12 = affine.load %9[%arg11, %arg12] : memref<32x32xi32, 2>
              %13 = muli %10, %11 : i32
              %14 = addi %12, %13 : i32
              affine.store %14, %9[%arg11, %arg12] : memref<32x32xi32, 2>
            }
          }
        }
        air.dma_memcpy_nd (%arg9[%5, %6] [%c32, %c32] [%c64_1, %c1_2], %9[] [] []) {id = 7 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
        memref.dealloc %7 : memref<32x32xi32, 2>
        memref.dealloc %8 : memref<32x32xi32, 2>
        memref.dealloc %9 : memref<32x32xi32, 2>
      }
      air.herd_terminator
    }
    air.dma_memcpy_nd (%1[%c0, %c0] [%c64, %c64] [%c64, %c1], %4[] [] []) {id = 8 : i32} : (memref<64x64xi32>, memref<64x64xi32, 1>)
    memref.dealloc %2 : memref<64x64xi32, 1>
    memref.dealloc %3 : memref<64x64xi32, 1>
    memref.dealloc %4 : memref<64x64xi32, 1>
    return
  }
}