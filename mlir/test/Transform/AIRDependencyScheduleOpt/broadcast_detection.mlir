// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-dependency -air-broadcast-detection | FileCheck %s

// Detects broadcast pattern for DMAs
// CHECK: [[$SET0:#set[0-9]+]] = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd {{.*}}broadcast_pattern = [[$SET0]]{{.*}}
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}broadcast_pattern = [[$SET1]]{{.*}}

#map = affine_map<()[s0] -> (s0 * 32)>
module {
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
        air.launch_herd  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%1, %arg11=%2, %arg12=%3) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {sym_name = "herd_0"} {
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
          air.herd_terminator
        }
        air.dma_memcpy_nd (%0[%arg3, %arg4] [%c64, %c64] [%c512, %c1], %3[] [] []) {id = 8 : i32} : (memref<512x512xbf16>, memref<64x64xbf16, 1>)
        memref.dealloc %1 : memref<64x64xbf16, 1>
        memref.dealloc %2 : memref<64x64xbf16, 1>
        memref.dealloc %3 : memref<64x64xbf16, 1>
      }
      scf.yield
    }
    return
  }
}