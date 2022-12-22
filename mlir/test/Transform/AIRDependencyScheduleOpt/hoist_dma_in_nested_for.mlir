//===- hoist_dma_in_nested_for.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-hoist-dma-in-accum-pattern | FileCheck %s

// Hoisting redundant data movement in nested for loops

module attributes {torch.debug_module_name = "mmult"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<256x1024xbf16>, memref<1024x1024xbf16>, memref<256x1024xbf16> {
      air.partition  args(%arg10=%arg7, %arg11=%arg8, %arg12=%arg9) : memref<256x1024xbf16>, memref<1024x1024xbf16>, memref<256x1024xbf16> {
        %c1_0 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
// CHECK: %[[EVENT0:.*]] = scf.for
        scf.for %arg13 = %c0 to %c256 step %c128 {
// CHECK: %[[EVENT1:.*]] = scf.for
          scf.for %arg14 = %c0 to %c1024 step %c128 {
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd async 
// CHECK: %[[EVENT3:.*]] = scf.for
            scf.for %arg15 = %c0 to %c1024 step %c128 {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              %alloc_1 = memref.alloc() : memref<128x128xbf16, 1>
              %alloc_2 = memref.alloc() : memref<128x128xbf16, 1>
              air.dma_memcpy_nd (%alloc[] [] [], %arg10[%arg13, %arg15] [%c128, %c128] [%c1024, %c1_0]) {id = 1 : i32} : (memref<128x128xbf16, 1>, memref<256x1024xbf16>)
// CHECK: %[[EVENT4:.*]] = air.dma_memcpy_nd async 
              air.dma_memcpy_nd (%alloc_1[] [] [], %arg11[%arg15, %arg14] [%c128, %c128] [%c1024, %c1_0]) {id = 2 : i32} : (memref<128x128xbf16, 1>, memref<1024x1024xbf16>)
// CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd async 
              air.dma_memcpy_nd (%alloc_2[] [] [], %arg12[%arg13, %arg14] [%c128, %c128] [%c1024, %c1_0]) {id = 3 : i32} : (memref<128x128xbf16, 1>, memref<256x1024xbf16>)
// CHECK: %[[EVENT6:.*]] = air.herd @herd_0 async 
              air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c1_0, %arg19=%c1_0) args(%arg20=%alloc, %arg21=%alloc_1, %arg22=%alloc_2) : memref<128x128xbf16, 1>, memref<128x128xbf16, 1>, memref<128x128xbf16, 1> {
                %c1_3 = arith.constant 1 : index
                %c0_4 = arith.constant 0 : index
                %c2 = arith.constant 2 : index
                %c128_5 = arith.constant 128 : index
                %c32 = arith.constant 32 : index
// CHECK: %[[EVENT7:.*]] = scf.for
                scf.for %arg23 = %c0_4 to %c2 step %c1_3 {
// CHECK: %[[EVENT8:.*]] = scf.for
                  scf.for %arg24 = %c0_4 to %c2 step %c1_3 {
// CHECK: %[[EVENT9:.*]] = air.dma_memcpy_nd async 
// CHECK: %[[EVENT10:.*]] = scf.for
                    scf.for %arg25 = %c0_4 to %c128_5 step %c32 {
                      %alloc_6 = memref.alloc() : memref<32x32xbf16, 2>
                      %alloc_7 = memref.alloc() : memref<32x32xbf16, 2>
                      %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
                      air.dma_memcpy_nd (%alloc_6[] [] [], %arg20[%arg23, %arg25] [%c32, %c32] [%c128_5, %c1_3]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<128x128xbf16, 1>)
// CHECK: %[[EVENT11:.*]] = air.dma_memcpy_nd async 
                      air.dma_memcpy_nd (%alloc_7[] [] [], %arg21[%arg25, %arg24] [%c32, %c32] [%c128_5, %c1_3]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<128x128xbf16, 1>)
// CHECK: %[[EVENT12:.*]] = air.dma_memcpy_nd async 
                      air.dma_memcpy_nd (%alloc_8[] [] [], %arg22[%arg23, %arg24] [%c32, %c32] [%c128_5, %c1_3]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<128x128xbf16, 1>)
// CHECK: %[[EVENT13:.*]] = air.execute
                      linalg.matmul ins(%alloc_6, %alloc_7 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_8 : memref<32x32xbf16, 2>)
                      air.dma_memcpy_nd (%arg22[%arg23, %arg24] [%c32, %c32] [%c128_5, %c1_3], %alloc_8[] [] []) {id = 7 : i32} : (memref<128x128xbf16, 1>, memref<32x32xbf16, 2>)
                      memref.dealloc %alloc_6 : memref<32x32xbf16, 2>
                      memref.dealloc %alloc_7 : memref<32x32xbf16, 2>
                      memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
                    }
// CHECK: scf.yield
// CHECK: %[[EVENT14:.*]] = air.dma_memcpy_nd async 
                  }
// CHECK: scf.yield
                }
// CHECK: scf.yield
                air.herd_terminator
              }
              air.dma_memcpy_nd (%arg12[%arg13, %arg14] [%c128, %c128] [%c1024, %c1_0], %alloc_2[] [] []) {id = 8 : i32} : (memref<256x1024xbf16>, memref<128x128xbf16, 1>)
              memref.dealloc %alloc : memref<128x128xbf16, 1>
              memref.dealloc %alloc_1 : memref<128x128xbf16, 1>
              memref.dealloc %alloc_2 : memref<128x128xbf16, 1>
            }
// CHECK: scf.yield
// CHECK: %[[EVENT15:.*]] = air.dma_memcpy_nd async 
          }
// CHECK: scf.yield
        }
// CHECK: scf.yield
        air.partition_terminator
      }
      air.launch_terminator
    }
    return
  }
}

