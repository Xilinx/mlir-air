// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Baseline transformed IR from Step 59 (current pipeline output)
// This file can be loaded with: python run.py --pre-transformed-ir baseline_transformed.mlir

module {
  func.func @softmax_kernel(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.muli %arg5, %c4_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %1, %c1024 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [4, 1024], strides: [1024, 1] : memref<*xbf16> to memref<4x1024xbf16, strided<[1024, 1], offset: ?>>
    %alloc = memref.alloc() : memref<4x1024xbf16, 1 : i32>
    memref.copy %reinterpret_cast, %alloc : memref<4x1024xbf16, strided<[1024, 1], offset: ?>> to memref<4x1024xbf16, 1 : i32>
    %alloc_0 = memref.alloc() : memref<4x1024xbf16, 1>
    air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c1) args(%arg12=%alloc, %arg13=%alloc_0) : memref<4x1024xbf16, 1 : i32>, memref<4x1024xbf16, 1> {
      %cst = arith.constant 1.000000e+00 : f32
      %cst_2 = arith.constant 0.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %3 = ub.poison : bf16
      %c1_4 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024_5 = arith.constant 1024 : index
      %c32 = arith.constant 32 : index
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xbf16, 2>
      air.dma_memcpy_nd (%alloc_6[] [] [], %arg12[%arg8, %c0] [%c1_4, %c1024_5] [%c1024_5, %c1_4]) {id = 1 : i32} : (memref<1x1024xbf16, 2>, memref<4x1024xbf16, 1 : i32>)
      %alloc_7 = memref.alloc() : memref<1xf32, 2>
      memref.store %cst_3, %alloc_7[%c0] : memref<1xf32, 2>
      scf.for %arg14 = %c0 to %c1024_5 step %c32 {
        %subview = memref.subview %alloc_6[0, %arg14] [1, 32] [1, 1] : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
        %4 = vector.transfer_read %subview[%c0, %c0], %3 {in_bounds = [true]} : memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>, vector<32xbf16>
        %5 = memref.load %alloc_7[%c0] : memref<1xf32, 2>
        %6 = arith.truncf %5 : f32 to bf16
        %7 = vector.reduction <maxnumf>, %4, %6 : vector<32xbf16> into bf16
        %8 = arith.extf %7 : bf16 to f32
        memref.store %8, %alloc_7[%c0] : memref<1xf32, 2>
      }
      %alloc_8 = memref.alloc() : memref<1xf32, 2>
      memref.store %cst_2, %alloc_8[%c0] : memref<1xf32, 2>
      scf.for %arg14 = %c0 to %c1024_5 step %c32 {
        %subview = memref.subview %alloc_6[0, %arg14] [1, 32] [1, 1] : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
        %4 = vector.transfer_read %subview[%c0, %c0], %3 {in_bounds = [true]} : memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>, vector<32xbf16>
        %5 = memref.load %alloc_7[%c0] : memref<1xf32, 2>
        %6 = memref.load %alloc_8[%c0] : memref<1xf32, 2>
        %7 = arith.extf %4 : vector<32xbf16> to vector<32xf32>
        %8 = vector.broadcast %5 : f32 to vector<32xf32>
        %9 = arith.subf %7, %8 : vector<32xf32>
        %10 = arith.truncf %9 : vector<32xf32> to vector<32xbf16>
        %11 = math.exp %10 : vector<32xbf16>
        %12 = arith.truncf %6 : f32 to bf16
        %13 = vector.reduction <add>, %11, %12 : vector<32xbf16> into bf16
        %14 = arith.extf %13 : bf16 to f32
        memref.store %14, %alloc_8[%c0] : memref<1xf32, 2>
      }
      %alloc_9 = memref.alloc() : memref<1x1024xbf16, 2>
      scf.for %arg14 = %c0 to %c1024_5 step %c32 {
        %subview = memref.subview %alloc_6[0, %arg14] [1, 32] [1, 1] : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
        %subview_10 = memref.subview %alloc_9[0, %arg14] [1, 32] [1, 1] : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
        %4 = vector.transfer_read %subview[%c0, %c0], %3 {in_bounds = [true]} : memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>, vector<32xbf16>
        %5 = memref.load %alloc_7[%c0] : memref<1xf32, 2>
        %6 = memref.load %alloc_8[%c0] : memref<1xf32, 2>
        %7 = arith.divf %cst, %6 : f32
        %8 = arith.truncf %7 : f32 to bf16
        %9 = arith.extf %4 : vector<32xbf16> to vector<32xf32>
        %10 = vector.broadcast %5 : f32 to vector<32xf32>
        %11 = arith.subf %9, %10 : vector<32xf32>
        %12 = arith.truncf %11 : vector<32xf32> to vector<32xbf16>
        %13 = math.exp %12 : vector<32xbf16>
        %14 = vector.broadcast %8 : bf16 to vector<32xbf16>
        %15 = arith.mulf %13, %14 : vector<32xbf16>
        vector.transfer_write %15, %subview_10[%c0, %c0] {in_bounds = [true]} : vector<32xbf16>, memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
      }
      memref.dealloc %alloc_7 : memref<1xf32, 2>
      memref.dealloc %alloc_8 : memref<1xf32, 2>
      air.dma_memcpy_nd (%arg13[%arg8, %c0] [%c1_4, %c1024_5] [%c1024_5, %c1_4], %alloc_9[] [] []) {id = 2 : i32} : (memref<4x1024xbf16, 1>, memref<1x1024xbf16, 2>)
      memref.dealloc %alloc_9 : memref<1x1024xbf16, 2>
    }
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [4, 1024], strides: [1024, 1] : memref<*xbf16> to memref<4x1024xbf16, strided<[1024, 1], offset: ?>>
    memref.copy %alloc_0, %reinterpret_cast_1 : memref<4x1024xbf16, 1> to memref<4x1024xbf16, strided<[1024, 1], offset: ?>>
    memref.dealloc %alloc_0 : memref<4x1024xbf16, 1>
    return
  }
}
