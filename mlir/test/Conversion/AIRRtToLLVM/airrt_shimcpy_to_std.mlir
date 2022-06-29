// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// CHECK: call @air_shim_memcpy({{.*}}, %c1_i32, {{.*}}) : (!llvm.ptr<i64>, i32, i64, i64, memref<?xi32>, i64, i64) -> ()
// CHECK: call @air_shim_memcpy2d({{.*}}, %c2_i32, {{.*}}) : (!llvm.ptr<i64>, i32, i64, i64, memref<?x?xi32>, i64, i64, i64, i64, i64) -> ()
// CHECK: call @air_shim_memcpy({{.*}}, %c3_i32, {{.*}}) : (!llvm.ptr<i64>, i32, i64, i64, memref<?xi32>, i64, i64) -> ()
// CHECK: call @air_shim_memcpy2d({{.*}}, %c4_i32, {{.*}}) : (!llvm.ptr<i64>, i32, i64, i64, memref<?x?xi32>, i64, i64, i64, i64, i64) -> ()
// CHECK: call @air_shim_memcpy4d({{.*}}, %c1_i32_0, {{.*}}) : (!llvm.ptr<i64>, i32, i64, i64, memref<?x?x?x?xi32>, i64, i64, i64, i64, i64, i64, i64) -> ()
// CHECK: call @air_shim_memcpy4d({{.*}}, %c2_i32, {{.*}}) : (!llvm.ptr<i64>, i32, i64, i64, memref<?x?x?x?xi32>, i64, i64, i64, i64, i64, i64, i64) -> ()
// CHECK: call @air_dma_nd_memcpy_2d0i32(
// CHECK: call @air_dma_nd_memcpy_1d1f32(
// CHECK: call @air_dma_nd_memcpy_1d0f32(
#map0 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 128)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0 * 16)>
module  {
  func.func @task0(%arg0: tensor<256x256xi32>, %arg1: tensor<256xi32>) -> tensor<256x256xi32> {
    %0 = memref.alloc() : memref<256x256xi32>
    %1 = memref.alloc() : memref<256xi32>
    %2 = bufferization.to_memref %arg0 : memref<256x256xi32>
    %3 = bufferization.to_memref %arg1 : memref<256xi32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 2 {
        %c2 = arith.constant 2 : index
        affine.for %arg4 = 0 to 2 {
          affine.for %arg5 = 0 to 2 {
            %c0 = arith.constant 0 : index
            %c4096 = arith.constant 4096 : index
            %c64 = arith.constant 64 : index
            %c256 = arith.constant 256 : index
            %5 = memref.alloc() : memref<64x64xi32, 2>
            %6 = memref.alloc() : memref<64xi32, 2>
            %7 = memref.alloc() : memref<32xi32, 2>
            %8 = affine.apply #map0()[%arg4, %arg2]
            %9 = affine.apply #map0()[%arg5, %arg3]
            %c1_i32 = arith.constant 1 : i32
            %10 = arith.index_cast %arg5 : index to i64
            %11 = arith.index_cast %arg4 : index to i64
            %12 = arith.index_cast %8 : index to i64
            %13 = arith.index_cast %c64 : index to i64
            airrt.dma_memcpy(%c1_i32, %10, %11, %3[%12], %13) : (i32, i64, i64, memref<256xi32>, [i64], i64)
            %c2_i32 = arith.constant 2 : i32
            %14 = arith.index_cast %arg5 : index to i64
            %15 = arith.index_cast %arg4 : index to i64
            %16 = arith.index_cast %8 : index to i64
            %17 = arith.index_cast %9 : index to i64
            %18 = arith.index_cast %c4096 : index to i64
            %19 = arith.index_cast %c256 : index to i64
            %20 = arith.index_cast %c64 : index to i64
            airrt.dma_memcpy_2d(%c2_i32, %14, %15, %2[%16, %17], %18, %19, %20) : (i32, i64, i64, memref<256x256xi32>, [i64, i64], i64, i64, i64)
            %21 = memref.alloc() : memref<64x64xi32, 2>
            %22 = memref.alloc() : memref<64xi32, 2>
            affine.for %arg6 = 0 to 64 {
              %c1_i32_0 = arith.constant 1 : i32
              %34 = affine.load %6[%arg6] : memref<64xi32, 2>
              %35 = arith.addi %34, %c1_i32_0 : i32
              affine.store %35, %22[%arg6] : memref<64xi32, 2>
              affine.for %arg7 = 0 to 64 {
                %36 = affine.load %5[%arg6, %arg7] : memref<64x64xi32, 2>
                %37 = arith.addi %36, %c1_i32_0 : i32
                affine.store %37, %21[%arg6, %arg7] : memref<64x64xi32, 2>
              }
            }
            %c3_i32 = arith.constant 3 : i32
            %23 = arith.index_cast %arg5 : index to i64
            %24 = arith.index_cast %arg4 : index to i64
            %25 = arith.index_cast %8 : index to i64
            %26 = arith.index_cast %c64 : index to i64
            airrt.dma_memcpy(%c3_i32, %23, %24, %1[%25], %26) : (i32, i64, i64, memref<256xi32>, [i64], i64)
            %c4_i32 = arith.constant 4 : i32
            %27 = arith.index_cast %arg5 : index to i64
            %28 = arith.index_cast %arg4 : index to i64
            %29 = arith.index_cast %8 : index to i64
            %30 = arith.index_cast %9 : index to i64
            %31 = arith.index_cast %c4096 : index to i64
            %32 = arith.index_cast %c256 : index to i64
            %33 = arith.index_cast %c64 : index to i64
            airrt.dma_memcpy_2d(%c4_i32, %27, %28, %0[%29, %30], %31, %32, %33) : (i32, i64, i64, memref<256x256xi32>, [i64, i64], i64, i64, i64)
            memref.dealloc %21 : memref<64x64xi32, 2>
            memref.dealloc %5 : memref<64x64xi32, 2>
            memref.dealloc %22 : memref<64xi32, 2>
            memref.dealloc %6 : memref<64xi32, 2>
          } {air.herd_launch = "inner"}
        } {air.herd_launch = "outer"}
      }
    } {affine_opt_label = "affine_opt"}
    %4 = bufferization.to_tensor %0 : memref<256x256xi32>
    return %4 : tensor<256x256xi32>
  }
  func.func @task1(%arg0: tensor<32x32x32x32xi32>) -> tensor<32x32x32x32xi32> {
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() : memref<32x32x32x32xi32>
    %1 = bufferization.to_memref %arg0 : memref<32x32x32x32xi32>
    %2 = airrt.herd_load "herd_0" : i64
    affine.for %arg1 = 0 to 2 {
      affine.for %arg2 = 0 to 2 {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1_i32 = arith.constant 1 : i32
        affine.for %arg3 = 0 to 16 {
          %4 = affine.apply #map1(%arg3)[%arg1]
          affine.for %arg4 = 0 to 16 {
            %5 = affine.apply #map1(%arg4)[%arg2]
            %6 = memref.alloc() : memref<1x1x32x32xi32, 2>
            %c1_i32_0 = arith.constant 1 : i32
            %7 = arith.index_cast %arg2 : index to i64
            %8 = arith.index_cast %arg1 : index to i64
            %9 = arith.index_cast %4 : index to i64
            %10 = arith.index_cast %5 : index to i64
            %11 = arith.index_cast %c0 : index to i64
            %12 = arith.index_cast %c0 : index to i64
            %13 = arith.index_cast %c1024 : index to i64
            %14 = arith.index_cast %c1024 : index to i64
            %15 = arith.index_cast %c1024 : index to i64
            airrt.dma_memcpy_4d(%c1_i32_0, %7, %8, %1[%9, %10, %11, %12], %13, %14, %15) : (i32, i64, i64, memref<32x32x32x32xi32>, [i64, i64, i64, i64], i64, i64, i64)
            %16 = memref.alloc() : memref<1x1x32x32xi32, 2>
            affine.for %arg5 = 0 to 2 {
              affine.for %arg6 = 0 to 2 {
                affine.for %arg7 = 0 to 16 {
                  affine.for %arg8 = 0 to 16 {
                    %26 = affine.load %6[0, 0, %arg7 + %arg5 * 16, %arg8 + %arg6 * 16] : memref<1x1x32x32xi32, 2>
                    %27 = arith.addi %26, %c1_i32 : i32
                    affine.store %27, %16[0, 0, %arg7 + %arg5 * 16, %arg8 + %arg6 * 16] : memref<1x1x32x32xi32, 2>
                  }
                }
              }
            }
            %c2_i32 = arith.constant 2 : i32
            %17 = arith.index_cast %arg2 : index to i64
            %18 = arith.index_cast %arg1 : index to i64
            %19 = arith.index_cast %4 : index to i64
            %20 = arith.index_cast %5 : index to i64
            %21 = arith.index_cast %c0 : index to i64
            %22 = arith.index_cast %c0 : index to i64
            %23 = arith.index_cast %c1024 : index to i64
            %24 = arith.index_cast %c1024 : index to i64
            %25 = arith.index_cast %c1024 : index to i64
            airrt.dma_memcpy_4d(%c2_i32, %17, %18, %0[%19, %20, %21, %22], %23, %24, %25) : (i32, i64, i64, memref<32x32x32x32xi32>, [i64, i64, i64, i64], i64, i64, i64)
            memref.dealloc %16 : memref<1x1x32x32xi32, 2>
            memref.dealloc %6 : memref<1x1x32x32xi32, 2>
          }
        }
      } {air.herd_launch = "inner"}
    } {air.herd_launch = "outer"}
    %3 = bufferization.to_tensor %0 : memref<32x32x32x32xi32>
    return %3 : tensor<32x32x32x32xi32>
  }
  func.func @ndfoo(%arg0: memref<256x256xi32>, %arg1: memref<256xf32>) {
    %L2 = airrt.alloc : memref<512xf32, 1>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        %c16 = arith.constant 16 : index
        %c0_i64 = arith.constant 0 : i64
        %c1_i64 = arith.constant 1 : i64
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c128_i64 = arith.constant 128 : i64
        %c256_i64 = arith.constant 256 : i64
        %0 = arith.index_cast %arg3 : index to i64
        %1 = arith.index_cast %arg2 : index to i64
        %2 = arith.index_cast %c16 : index to i64
        airrt.dma_memcpy_nd(%c1_i32, %0, %1, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %2, %2], [%c0_i64, %c0_i64, %c256_i64]) : (i32, i64, i64, memref<256x256xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
        airrt.dma_memcpy_nd(%c2_i32, %0, %1, %L2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %2, %2], [%c0_i64, %c0_i64, %c128_i64]) {attr = "attr"} : (i32, i64, i64, memref<512xf32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
        airrt.dma_memcpy_nd(%c2_i32, %0, %1, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %2, %2], [%c0_i64, %c0_i64, %c128_i64]) : (i32, i64, i64, memref<256xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
      } {air.herd_launch = "inner"}
    } {air.herd_launch = "outer"}
    return
  }

}

