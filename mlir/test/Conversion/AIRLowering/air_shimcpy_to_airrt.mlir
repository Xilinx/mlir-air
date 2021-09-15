// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-to-std | FileCheck %s
// CHECK: airrt.dma_memcpy(%c1_i32, %{{.*}}, %{{.*}}, %3[%{{.*}}], %{{.*}}) : (i32, i64, i64, memref<256xi32>, [i64], i64) -> ()
// CHECK: airrt.dma_memcpy_2d(%c2_i32, %{{.*}}, %{{.*}}, %2[%{{.*}}, %{{.*}}], %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i64, i64, memref<256x256xi32>, [i64, i64], i64, i64, i64) -> ()
// CHECK: airrt.dma_memcpy(%c3_i32, %{{.*}}, %{{.*}}, %1[%{{.*}}], %{{.*}}) : (i32, i64, i64, memref<256xi32>, [i64], i64) -> ()
// CHECK: airrt.dma_memcpy_2d(%c4_i32, %{{.*}}, %{{.*}}, %0[%{{.*}}, %{{.*}}], %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i64, i64, memref<256x256xi32>, [i64, i64], i64, i64, i64) -> ()
// CHECK: airrt.dma_memcpy_4d(%c1_i32_0, %{{.*}}, %{{.*}}, %1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i64, i64, memref<32x32x32x32xi32>, [i64, i64, i64, i64], i64, i64, i64) -> ()
// CHECK: airrt.dma_memcpy_4d(%c2_i32, %{{.*}}, %{{.*}}, %0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i64, i64, memref<32x32x32x32xi32>, [i64, i64, i64, i64], i64, i64, i64) -> ()
#map0 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 128)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0 * 16)>
module  {
  func @task0(%arg0: tensor<256x256xi32>, %arg1: tensor<256xi32>) -> tensor<256x256xi32> {
    %0 = memref.alloc() : memref<256x256xi32>
    %1 = memref.alloc() : memref<256xi32>
    %2 = memref.buffer_cast %arg0 : memref<256x256xi32>
    %3 = memref.buffer_cast %arg1 : memref<256xi32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 2 {
        %c2 = constant 2 : index
        air.launch_herd tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%arg2, %arg9=%arg3, %arg10=%2, %arg11=%0, %arg12=%3, %arg13=%1) : index,index,memref<256x256xi32>,memref<256x256xi32>,memref<256xi32>,memref<256xi32> {
          %c0 = constant 0 : index
          %c4096 = constant 4096 : index
          %c64 = constant 64 : index
          %c256 = constant 256 : index
          %5 = memref.alloc() : memref<64x64xi32, 2>
          %6 = memref.alloc() : memref<64xi32, 2>
          %7 = memref.alloc() : memref<32xi32, 2>
          %8 = affine.apply #map0()[%arg4, %arg8]
          %9 = affine.apply #map0()[%arg5, %arg9]
          air.dma_memcpy (%6, %arg12, [%c0], [%8], %c64) {id = 1 : i32} : (memref<64xi32, 2>, memref<256xi32>, [index], [index], index) -> ()
          air.dma_memcpy_2d (%5, %arg10, [%c0, %c0], [%8, %9], %c4096, %c256, %c64) {id = 2 : i32} : (memref<64x64xi32, 2>, memref<256x256xi32>, [index, index], [index, index], index, index, index) -> ()
          %10 = memref.alloc() : memref<64x64xi32, 2>
          %11 = memref.alloc() : memref<64xi32, 2>
          affine.for %arg14 = 0 to 64 {
            %c1_i32 = constant 1 : i32
            %12 = affine.load %6[%arg14] : memref<64xi32, 2>
            %13 = addi %12, %c1_i32 : i32
            affine.store %13, %11[%arg14] : memref<64xi32, 2>
            affine.for %arg15 = 0 to 64 {
              %14 = affine.load %5[%arg14, %arg15] : memref<64x64xi32, 2>
              %15 = addi %14, %c1_i32 : i32
              affine.store %15, %10[%arg14, %arg15] : memref<64x64xi32, 2>
            }
          }
          air.dma_memcpy (%arg13, %11, [%8], [%c0], %c64) {id = 3 : i32} : (memref<256xi32>, memref<64xi32, 2>, [index], [index], index) -> ()
          air.dma_memcpy_2d (%arg11, %10, [%8, %9], [%c0, %c0], %c4096, %c256, %c64) {id = 4 : i32} : (memref<256x256xi32>, memref<64x64xi32, 2>, [index, index], [index, index], index, index, index) -> ()
          memref.dealloc %10 : memref<64x64xi32, 2>
          memref.dealloc %5 : memref<64x64xi32, 2>
          memref.dealloc %11 : memref<64xi32, 2>
          memref.dealloc %6 : memref<64xi32, 2>
          air.herd_terminator
        }
      }
    } {affine_opt_label = "affine_opt"}
    %4 = memref.tensor_load %0 : memref<256x256xi32>
    return %4 : tensor<256x256xi32>
  }
  func @task1(%arg0: tensor<32x32x32x32xi32>) -> tensor<32x32x32x32xi32> {
    %c2 = constant 2 : index
    %0 = memref.alloc() : memref<32x32x32x32xi32>
    %1 = memref.buffer_cast %arg0 : memref<32x32x32x32xi32>
    air.launch_herd tile (%arg1, %arg2) in (%arg3=%c2, %arg4=%c2) args(%arg5=%1, %arg6=%0) : memref<32x32x32x32xi32>,memref<32x32x32x32xi32>attributes {sym_name = "herd_0"} {
      %c0 = constant 0 : index
      %c1024 = constant 1024 : index
      %c1_i32 = constant 1 : i32
      affine.for %arg7 = 0 to 16 {
        %3 = affine.apply #map1(%arg7)[%arg1]
        affine.for %arg8 = 0 to 16 {
          %4 = affine.apply #map1(%arg8)[%arg2]
          %5 = memref.alloc() : memref<1x1x32x32xi32, 2>
          air.dma_memcpy_4d (%5, %arg5, [%c0, %c0, %c0, %c0], [%3, %4, %c0, %c0], %c1024, %c1024, %c1024) {id = 1 : i32} : (memref<1x1x32x32xi32, 2>, memref<32x32x32x32xi32>, [index, index, index, index], [index, index, index, index], index, index, index) -> ()
          %6 = memref.alloc() : memref<1x1x32x32xi32, 2>
          affine.for %arg9 = 0 to 2 {
            affine.for %arg10 = 0 to 2 {
              affine.for %arg11 = 0 to 16 {
                affine.for %arg12 = 0 to 16 {
                  %7 = affine.load %5[0, 0, %arg11 + %arg9 * 16, %arg12 + %arg10 * 16] : memref<1x1x32x32xi32, 2>
                  %8 = addi %7, %c1_i32 : i32
                  affine.store %8, %6[0, 0, %arg11 + %arg9 * 16, %arg12 + %arg10 * 16] : memref<1x1x32x32xi32, 2>
                }
              }
            }
          }
          air.dma_memcpy_4d (%arg6, %6, [%3, %4, %c0, %c0], [%c0, %c0, %c0, %c0], %c1024, %c1024, %c1024) {id = 2 : i32} : (memref<32x32x32x32xi32>, memref<1x1x32x32xi32, 2>, [index, index, index, index], [index, index, index, index], index, index, index) -> ()
          memref.dealloc %6 : memref<1x1x32x32xi32, 2>
          memref.dealloc %5 : memref<1x1x32x32xi32, 2>
        }
      }
      air.herd_terminator
    }
    %2 = memref.tensor_load %0 : memref<32x32x32x32xi32>
    return %2 : tensor<32x32x32x32xi32>
  }
}
