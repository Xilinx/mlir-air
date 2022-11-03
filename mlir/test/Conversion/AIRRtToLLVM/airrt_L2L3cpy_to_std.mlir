//===- airrt_L2L3cpy_to_std.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// CHECK: call @air_nd_memcpy_2d1i32_2d0i32({{.*}}) : (!llvm.ptr<i64>, memref<?x?xi32, 1>, memref<?x?xi32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
// CHECK: call @air_nd_memcpy_2d1i32_2d0i32({{.*}}) : (!llvm.ptr<i64>, memref<?x?xi32, 1>, memref<?x?xi32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
// CHECK: call @air_nd_memcpy_2d1i32_2d0i32({{.*}}) : (!llvm.ptr<i64>, memref<?x?xi32, 1>, memref<?x?xi32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
// CHECK: call @air_nd_memcpy_2d0i32_2d1i32({{.*}}) : (!llvm.ptr<i64>, memref<?x?xi32>, memref<?x?xi32, 1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
module attributes {torch.debug_module_name = "mmult"}  {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<?x?xi32>) {
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c32_i64 = arith.constant 32 : i64
    %c4_i32 = arith.constant 4 : i32
    %c32 = arith.constant 32 : index
    %c64_i64 = arith.constant 64 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() : memref<64x64xi32>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.store %c0_i32, %0[%arg3, %arg4] : memref<64x64xi32>
      }
    }
    %1 = memref.cast %arg2 : memref<?x?xi32> to memref<64x64xi32>
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        %6 = affine.load %0[%arg3, %arg4] : memref<64x64xi32>
        affine.store %6, %arg2[%arg3, %arg4] : memref<?x?xi32>
      }
    }
    %2 = airrt.alloc : memref<64x64xi32, 1>
    %3 = airrt.alloc : memref<64x64xi32, 1>
    %4 = airrt.alloc : memref<64x64xi32, 1>
    airrt.memcpy_nd(%2, %arg0, [%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c64_i64, %c64_i64], [%c0_i64, %c0_i64, %c64_i64]) : (memref<64x64xi32, 1>, memref<64x64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    airrt.memcpy_nd(%3, %arg1, [%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c64_i64, %c64_i64], [%c0_i64, %c0_i64, %c64_i64]) : (memref<64x64xi32, 1>, memref<64x64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    airrt.memcpy_nd(%4, %1, [%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c64_i64, %c64_i64], [%c0_i64, %c0_i64, %c64_i64]) : (memref<64x64xi32, 1>, memref<64x64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    %5 = airrt.herd_load "herd_0" : i64
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %6 = arith.muli %arg3, %c32 : index
        %7 = arith.muli %arg4, %c32 : index
        scf.for %arg5 = %c0 to %c64 step %c32 {
          %8 = memref.alloc() : memref<32x32xi32, 2>
          %9 = memref.alloc() : memref<32x32xi32, 2>
          %10 = memref.alloc() : memref<32x32xi32, 2>
          %11 = arith.index_cast %arg4 : index to i64
          %12 = arith.index_cast %arg3 : index to i64
          %13 = arith.index_cast %6 : index to i64
          %14 = arith.index_cast %arg5 : index to i64
          airrt.dma_memcpy_nd(%c4_i32, %11, %12, %2[%c0_i64, %c0_i64, %13, %14], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c64_i64]) : (i32, i64, i64, memref<64x64xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
          %15 = arith.index_cast %7 : index to i64
          airrt.dma_memcpy_nd(%c5_i32, %11, %12, %3[%c0_i64, %c0_i64, %14, %15], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c64_i64]) : (i32, i64, i64, memref<64x64xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
          airrt.dma_memcpy_nd(%c6_i32, %11, %12, %4[%c0_i64, %c0_i64, %13, %15], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c64_i64]) : (i32, i64, i64, memref<64x64xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
          affine.for %arg6 = 0 to 32 {
            affine.for %arg7 = 0 to 32 {
              affine.for %arg8 = 0 to 32 {
                %16 = affine.load %8[%arg6, %arg8] : memref<32x32xi32, 2>
                %17 = affine.load %9[%arg8, %arg7] : memref<32x32xi32, 2>
                %18 = affine.load %10[%arg6, %arg7] : memref<32x32xi32, 2>
                %19 = arith.muli %16, %17 : i32
                %20 = arith.addi %18, %19 : i32
                affine.store %20, %10[%arg6, %arg7] : memref<32x32xi32, 2>
              }
            }
          }
          airrt.dma_memcpy_nd(%c7_i32, %11, %12, %4[%c0_i64, %c0_i64, %13, %15], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c64_i64]) : (i32, i64, i64, memref<64x64xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
          memref.dealloc %8 : memref<32x32xi32, 2>
          memref.dealloc %9 : memref<32x32xi32, 2>
          memref.dealloc %10 : memref<32x32xi32, 2>
        }
      } {air.herd_launch = "inner"}
    } {air.herd_launch = "outer"}
    airrt.memcpy_nd(%1, %4, [%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c64_i64, %c64_i64], [%c0_i64, %c0_i64, %c64_i64]) : (memref<64x64xi32>, memref<64x64xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    airrt.dealloc %2 : memref<64x64xi32, 1>
    airrt.dealloc %3 : memref<64x64xi32, 1>
    airrt.dealloc %4 : memref<64x64xi32, 1>
    return
  }
}

