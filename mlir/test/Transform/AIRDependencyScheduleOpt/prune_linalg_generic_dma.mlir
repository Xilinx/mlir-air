//===- prune_linalg_generic_dma.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-dependency -air-prune-linalg-generic-input-dma %s | FileCheck %s

// Remove the redundant DMA copying into linalg.generic

// CHECK-LABEL: module
// CHECK: func.func @forward

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "model"} {
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<24576x1024xbf16>) {
    %c384 = arith.constant 384 : index
    %c16 = arith.constant 16 : index
    air.launch (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1) : memref<24576x1024xbf16>, memref<24576x1024xbf16> {
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %3 = affine.apply #map0()[%arg2]
      %4 = affine.apply #map0()[%arg3]
      %5 = memref.alloc() : memref<64x64xbf16, 1>
      %6 = memref.alloc() : memref<64x64xbf16, 1>
      air.dma_memcpy_nd (%5[] [] [], %arg6[%3, %4] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
      air.dma_memcpy_nd (%6[] [] [], %arg7[%3, %4] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
      air.partition args(%arg9=%arg2, %arg10=%arg3, %arg11=%arg4, %arg12=%arg5, %arg13=%5, %arg14=%6) : index, index, index, index, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
      // CHECK: %[[EVENT0:.*]], %[[VALUE0:.*]] = air.execute
      // CHECK: %[[EVENT1:.*]], %[[VALUE1:.*]] = air.execute
      // CHECK: %[[EVENT2:.*]], %[[VALUE2:.*]] = air.execute
      // CHECK: %[[EVENT3:.*]], %[[VALUE3:.*]] = air.execute
      // CHECK: %[[EVENT4:.*]] = air.dma_memcpy_nd async 
      // CHECK: %[[EVENT5:.*]] = air.partition async [%[[EVENT0]], %[[EVENT1]], %[[EVENT3]], %[[EVENT4]]]
        %c1_1 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        %c64_2 = arith.constant 64 : index
        %c1024_0 = arith.constant 1024 : index
        %new_0 = memref.alloc() : memref<64x64xbf16, 1>
        %new_1 = memref.alloc() : memref<64x64xbf16, 1>
        air.dma_memcpy_nd (%new_0[] [] [], %arg13[%arg9, %arg10] [%c1_1, %c1_1] [%c1_1, %c1_1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<64x64xbf16, 1>)
        air.dma_memcpy_nd (%new_1[] [] [], %arg14[%arg9, %arg10] [%c1_1, %c1_1] [%c1_1, %c1_1]) {id = 4 : i32} : (memref<64x64xbf16, 1>, memref<64x64xbf16, 1>)
        air.herd  tile (%arg22, %arg23) in (%arg16=%c2_0, %arg17=%c2_0) args(%arg18=%new_0, %arg19=%new_1) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {sym_name = "herd_1"} {
        // CHECK: %[[EVENT6:.*]], %[[VALUE4:.*]] = air.execute
        // CHECK: %[[EVENT7:.*]], %[[VALUE5:.*]] = air.execute
        // CHECK: %[[EVENT8:.*]] = air.dma_memcpy_nd async 
        // CHECK: %[[EVENT9:.*]] = air.herd @herd_1 async [%[[EVENT7]], %[[EVENT8]]]
          %c1_0 = arith.constant 1 : index
          %c64_1 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %cst_2 = arith.constant 2.000000e+00 : bf16
          %cst_3 = arith.constant 1.000000e+00 : bf16
          %cst_4 = arith.constant 5.000000e-01 : bf16
          %7 = affine.apply #map1()[%arg22]
          %8 = affine.apply #map1()[%arg23]
          %9 = memref.alloc() : memref<32x32xbf16, 2>
          %10 = memref.alloc() : memref<32x32xbf16, 2>
          // CHECK: %[[EVENT10:.*]] = air.dma_memcpy_nd async 
          air.dma_memcpy_nd (%9[] [] [], %arg18[%7, %8] [%c32, %c32] [%c64_1, %c1_0]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          air.dma_memcpy_nd (%10[] [] [], %arg19[%7, %8] [%c32, %c32] [%c64_1, %c1_0]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%9 : memref<32x32xbf16, 2>) outs(%10 : memref<32x32xbf16, 2>) {
          ^bb0(%arg20: bf16, %arg21: bf16):
            %11 = math.sqrt %cst_2 : bf16
            %12 = arith.divf %arg20, %11 : bf16
            %13 = math.erf %12 : bf16
            %14 = arith.addf %13, %cst_3 : bf16
            %15 = arith.mulf %14, %cst_4 : bf16
            %16 = arith.mulf %arg20, %15 : bf16
            linalg.yield %16 : bf16
          }
          // CHECK: %[[EVENT11:.*]] = air.dma_memcpy_nd async 
          // CHECK: %[[EVENT12:.*]] = air.execute [%[[EVENT11]]]
          air.dma_memcpy_nd (%arg19[%7, %8] [%c32, %c32] [%c64_1, %c1_0], %10[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          memref.dealloc %9 : memref<32x32xbf16, 2>
          memref.dealloc %10 : memref<32x32xbf16, 2>
          air.herd_terminator
        }
        air.dma_memcpy_nd (%arg14[%arg9, %arg10] [%c64_2, %c64_2] [%c1024_0, %c1_1], %new_1[] [] []) {id = 8 : i32} : (memref<64x64xbf16, 1>, memref<64x64xbf16, 1>)
        memref.dealloc %new_0 : memref<64x64xbf16, 1>
        memref.dealloc %new_1 : memref<64x64xbf16, 1>
        air.partition_terminator
      }
      memref.dealloc %5 : memref<64x64xbf16, 1>
      memref.dealloc %6 : memref<64x64xbf16, 1>
      air.launch_terminator
    }
    return
  }
}