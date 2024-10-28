//===- broadcast_detection.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-broadcast-detection --split-input-file | FileCheck %s

// Detects broadcast pattern for DMAs
// CHECK: [[$SET0:#set[0-9]*]] = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
// CHECK: [[$SET1:#set[0-9]*]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
// CHECK-LABEL: func.func @matmul
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd {{.*}}broadcast_pattern = [[$SET0]]{{.*}}
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}broadcast_pattern = [[$SET1]]{{.*}}

#map = affine_map<()[s0] -> (s0 * 32)>
func.func @matmul(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512xbf16>
  memref.copy %arg2, %0 : memref<512x512xbf16> to memref<512x512xbf16>
  scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c64, %c64) {
    scf.for %arg5 = %c0 to %c512 step %c64 {
      %1 = memref.alloc() : memref<64x64xbf16, 1>
      %2 = memref.alloc() : memref<64x64xbf16, 1>
      %3 = memref.alloc() : memref<64x64xbf16, 1>
      air.dma_memcpy_nd (%1[] [] [], %arg0[%arg3, %arg5] [%c64, %c64] [%c512, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
      air.dma_memcpy_nd (%2[] [] [], %arg1[%arg5, %arg4] [%c64, %c64] [%c512, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
      air.dma_memcpy_nd (%3[] [] [], %0[%arg3, %arg4] [%c64, %c64] [%c512, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
      air.herd  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%1, %arg11=%2, %arg12=%3) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {sym_name = "herd_0"} {
        %c1_0 = arith.constant 1 : index
        %c0_1 = arith.constant 0 : index
        %c64_2 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %4 = affine.apply #map()[%arg6]
        %5 = affine.apply #map()[%arg7]
        scf.for %arg13 = %c0_1 to %c64_2 step %c32 {
          %6 = memref.alloc() : memref<32x32xbf16, 2>
          %7 = memref.alloc() : memref<32x32xbf16, 2>
          %8 = memref.alloc() : memref<32x32xbf16, 2>
          air.dma_memcpy_nd (%6[] [] [], %arg10[%4, %arg13] [%c32, %c32] [%c64_2, %c1_0]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          air.dma_memcpy_nd (%7[] [] [], %arg11[%arg13, %5] [%c32, %c32] [%c64_2, %c1_0]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          air.dma_memcpy_nd (%8[] [] [], %arg12[%4, %5] [%c32, %c32] [%c64_2, %c1_0]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%6, %7 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%8 : memref<32x32xbf16, 2>)
          air.dma_memcpy_nd (%arg12[%4, %5] [%c32, %c32] [%c64_2, %c1_0], %8[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          memref.dealloc %6 : memref<32x32xbf16, 2>
          memref.dealloc %7 : memref<32x32xbf16, 2>
          memref.dealloc %8 : memref<32x32xbf16, 2>
        }
      }
      air.dma_memcpy_nd (%0[%arg3, %arg4] [%c64, %c64] [%c512, %c1], %3[] [] []) {id = 8 : i32} : (memref<512x512xbf16>, memref<64x64xbf16, 1>)
      memref.dealloc %1 : memref<64x64xbf16, 1>
      memref.dealloc %2 : memref<64x64xbf16, 1>
      memref.dealloc %3 : memref<64x64xbf16, 1>
    }
  }
  return
}

// -----

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0, s0 >= 0, -s0 >= 0)>
// CHECK-LABEL: func.func @func0
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd {{.*}} {id = 1 : i32} : (memref<256x64xbf16, 1>, memref<1024x256xbf16>)
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}broadcast_pattern = [[$SET0]]{{.*}}

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func @func0(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x256xbf16>, %arg2: memref<256x256xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg1) : memref<1024x256xbf16> attributes {id = 3 : i32} {
      air.segment @segment_0  args(%arg8=%arg4, %arg9=%arg7) : index, memref<1024x256xbf16> attributes {id = 2 : i32} {
        %c4 = arith.constant 4 : index
        %0 = affine.apply #map()[%arg8]
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c4, %arg13=%c4) args(%arg14=%0, %arg15=%arg9) : index, memref<1024x256xbf16> attributes {id = 1 : i32} {
          %c1_0 = arith.constant 1 : index
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c64 = arith.constant 64 : index
          %c1024 = arith.constant 1024 : index
          %1 = affine.apply #map1()[%arg11]
          %2 = arith.addi %arg14, %1 : index
          scf.for %arg16 = %c0 to %c1024 step %c256 {
            %alloc = memref.alloc() : memref<256x64xbf16, 1>
            air.dma_memcpy_nd (%alloc[] [] [], %arg15[%arg16, %2] [%c256, %c64] [%c256, %c1_0]) {id = 2 : i32} : (memref<256x64xbf16, 1>, memref<1024x256xbf16>)
            scf.for %arg17 = %c0 to %c256 step %c64 {
              %alloc_1 = memref.alloc() : memref<64x64xbf16, 2>
              air.dma_memcpy_nd (%alloc_1[] [] [], %alloc[%arg17, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 4 : i32} : (memref<64x64xbf16, 2>, memref<256x64xbf16, 1>)
              memref.dealloc %alloc_1 : memref<64x64xbf16, 2>
            }
            memref.dealloc %alloc : memref<256x64xbf16, 1>
          }
        }
      }
    }
    return
  }
}

// -----

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 >= 0)>
// CHECK-LABEL: func.func @func1
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd {{.*}}broadcast_pattern = [[$SET0]]{{.*}} : (memref<4x2x4x8xi32, 2 : i32>, memref<8x2048xi32, 1 : i32>)
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}} {id = 2 : i32} : (memref<8x4x8x4xi32, 2 : i32>, memref<2048x64xi32, 1 : i32>)

module {
  func.func @func1() {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    air.launch (%arg0, %arg1) in (%arg2=%c1, %arg3=%c32) {
      air.segment @segment_0  {
        %c64 = arith.constant 64 : index
        %c8 = arith.constant 8 : index
        %c2 = arith.constant 2 : index
        %c1_0 = arith.constant 1 : index
        %c0_1 = arith.constant 0 : index
        %c2048 = arith.constant 2048 : index
        %c256 = arith.constant 256 : index
        %alloc = memref.alloc() : memref<8x2048xi32, 1 : i32>
        %alloc_2 = memref.alloc() : memref<2048x64xi32, 1 : i32>
        air.herd @herd_0  tile (%arg12, %arg13) in (%arg14=%c1_0, %arg15=%c2) args(%arg16=%alloc, %arg17=%alloc_2) : memref<8x2048xi32, 1 : i32>, memref<2048x64xi32, 1 : i32> {
          %c64_4 = arith.constant 64 : index
          %c32_5 = arith.constant 32 : index
          %c1024 = arith.constant 1024 : index
          %c2048_6 = arith.constant 2048 : index
          %c8_7 = arith.constant 8 : index
          %c1_8 = arith.constant 1 : index
          %c0_i32 = arith.constant 0 : i32
          %c0_9 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c256_10 = arith.constant 256 : index
          %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg12]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg13]
          scf.for %arg19 = %c0_9 to %c256_10 step %c4 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg19]
            %alloc_12 = memref.alloc() : memref<4x2x4x8xi32, 2 : i32>
            air.dma_memcpy_nd (%alloc_12[%c0_9] [%c256_10] [%c1_8], %arg16[%c0_9, %5, %7] [%c4, %c8_7, %c8_7] [%c8_7, %c2048_6, %c1_8]) : (memref<4x2x4x8xi32, 2 : i32>, memref<8x2048xi32, 1 : i32>)
            %alloc_13 = memref.alloc() : memref<8x4x8x4xi32, 2 : i32>
            air.dma_memcpy_nd (%alloc_13[%c0_9] [%c1024] [%c1_8], %arg17[%c0_9, %7, %6] [%c8_7, %c32_5, %c4] [%c4, %c64_4, %c1_8]) : (memref<8x4x8x4xi32, 2 : i32>, memref<2048x64xi32, 1 : i32>)
            memref.dealloc %alloc_12 : memref<4x2x4x8xi32, 2 : i32>
            memref.dealloc %alloc_13 : memref<8x4x8x4xi32, 2 : i32>
          }
        }
        memref.dealloc %alloc : memref<8x2048xi32, 1 : i32>
        memref.dealloc %alloc_2 : memref<2048x64xi32, 1 : i32>
      }
    }
    return
  }
}

// -----

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 3 >= 0, d1 - s0 == 0, s0 >= 0, -s0 >= 0)>
// CHECK-LABEL: func.func @func2
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd {{.*}}broadcast_pattern = [[$SET0]]{{.*}} : (memref<4x8x3x3xi32, 2>, memref<4x32x3x3xi32, 1>)

module {
  func.func @func2() {
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c3 = arith.constant 3 : index
    air.launch (%arg0, %arg1, %arg2, %arg3) in (%arg4=%c2, %arg5=%c16, %arg6=%c3, %arg7=%c3) {
      air.segment @segment_0  {
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %alloc = memref.alloc() : memref<4x32x3x3xi32, 1>
        air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c1) args(%arg12=%alloc) : memref<4x32x3x3xi32, 1> {
          %c9 = arith.constant 9 : index
          %c288 = arith.constant 288 : index
          %c4_0 = arith.constant 4 : index
          %c3_1 = arith.constant 3 : index
          %c1_2 = arith.constant 1 : index
          %c0 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          scf.for %arg13 = %c0 to %c32 step %c8 {
            %alloc_3 = memref.alloc() : memref<4x8x3x3xi32, 2>
            air.dma_memcpy_nd (%alloc_3[] [] [], %arg12[%c0, %arg13, %c0, %c0] [%c4_0, %c8, %c3_1, %c3_1] [%c288, %c9, %c3_1, %c1_2]) {id = 4 : i32} : (memref<4x8x3x3xi32, 2>, memref<4x32x3x3xi32, 1>)
            memref.dealloc %alloc_3 : memref<4x8x3x3xi32, 2>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// 2D broadcasting to all cores in a herd.

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, s0 >= 0, -s0 >= 0)>
// CHECK-LABEL: func.func @func3
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd {{.*}}broadcast_pattern = [[$SET0]]{{.*}} : (memref<1x1x8x4xi32, 2 : i32>, memref<3x3x32x4xi32, 1 : i32>)

module {
  func.func @func3() {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    air.launch (%arg3, %arg4, %arg5) in (%arg6=%c3, %arg7=%c3, %arg8=%c16) {
      air.segment @segment_0  {
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %alloc = memref.alloc() : memref<3x3x32x4xi32, 1 : i32>
        air.herd @herd_0  tile (%arg9, %arg10) in (%arg11=%c2, %arg12=%c4) args(%arg13=%alloc) : memref<3x3x32x4xi32, 1 : i32> {
          %c128 = arith.constant 128 : index
          %c384 = arith.constant 384 : index
          %c4_0 = arith.constant 4 : index
          %c0 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          %c3_1 = arith.constant 3 : index
          %c1 = arith.constant 1 : index
          scf.for %arg14 = %c0 to %c3_1 step %c1 {
            scf.for %arg15 = %c0 to %c3_1 step %c1 {
              scf.for %arg16 = %c0 to %c32 step %c8 {
                %alloc_2 = memref.alloc() : memref<1x1x8x4xi32, 2 : i32>
                air.dma_memcpy_nd (%alloc_2[] [] [], %arg13[%arg14, %arg15, %arg16, %c0] [%c1, %c1, %c8, %c4_0] [%c384, %c128, %c4_0, %c1]) : (memref<1x1x8x4xi32, 2 : i32>, memref<3x3x32x4xi32, 1 : i32>)
              }
            }
          }
        }
        memref.dealloc %alloc : memref<3x3x32x4xi32, 1 : i32>
      }
    }
    return
  }
}
