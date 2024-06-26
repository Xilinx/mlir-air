//===- create_and_outline.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='test-patterns=to-aie-mlir' --split-input-file | FileCheck %s
// RUN: air-opt %s -air-to-aie='test-patterns=to-aie-mlir emit-while-loop' --split-input-file | FileCheck %s --check-prefix=EMITWHILE

// CHECK: aie.device
// CHECK: [[T00:%.*]] = aie.tile(1, 1)
// CHECK: [[T10:%.*]] = aie.tile(2, 1)
// CHECK: [[T01:%.*]] = aie.tile(1, 2)
// CHECK: [[T11:%.*]] = aie.tile(2, 2)
// CHECK: aie.core([[T11]])
// CHECK: aie.core([[T01]])
// CHECK: aie.core([[T10]])
// CHECK: aie.core([[T00]])
// EMITWHILE: aie.device
// EMITWHILE: [[T00:%.*]] = aie.tile(1, 1)
// EMITWHILE: [[T10:%.*]] = aie.tile(2, 1)
// EMITWHILE: [[T01:%.*]] = aie.tile(1, 2)
// EMITWHILE: [[T11:%.*]] = aie.tile(2, 2)
// EMITWHILE: aie.core([[T11]])
// EMITWHILE: aie.core([[T01]])
// EMITWHILE: aie.core([[T10]])
// EMITWHILE: aie.core([[T00]])
#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%a0: memref<64x64xi32>, %a1: memref<64x64xi32>, %a2: memref<64x64xi32>) {
    air.segment @segment0 args(%arg0=%a0, %arg1=%a1, %arg2=%a2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<64x64xi32>)
      %1 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      memref.copy %0, %1 : memref<64x64xi32> to memref<64x64xi32>
      air.herd @herd_0  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%1) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %2 = affine.apply #map()[%arg3]
        %3 = affine.apply #map()[%arg4]
        scf.for %arg10 = %c0 to %c64 step %c32 {
          %4 = memref.alloc() : memref<32x32xi32, 2>
          %5 = memref.alloc() : memref<32x32xi32, 2>
          %6 = memref.alloc() : memref<32x32xi32, 2>
          air.dma_memcpy_nd (%4[] [] [], %arg7[%2, %arg10] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          air.dma_memcpy_nd (%5[] [] [], %arg8[%arg10, %3] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          air.dma_memcpy_nd (%6[] [] [], %arg9[%2, %3] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          linalg.matmul ins(%4, %5 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%6 : memref<32x32xi32, 2>)
          air.dma_memcpy_nd (%arg9[%2, %3] [%c32, %c32] [%c64, %c1], %6[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
          memref.dealloc %4 : memref<32x32xi32, 2>
          memref.dealloc %5 : memref<32x32xi32, 2>
          memref.dealloc %6 : memref<32x32xi32, 2>
        }
      }
      memref.copy %1, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    }
    return
  }
}

// -----

// CHECK: aie.device
// CHECK: air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
// EMITWHILE: aie.device
// EMITWHILE: air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
func.func @f1() -> () {
  %cst1 = arith.constant 1 : index
  %cst4 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst4, %size_y = %cst1) {
    %src0 = memref.alloc() : memref<1xi32, 2>
    air.channel.put @channel_0[] (%src0[] [] []) : (memref<1xi32, 2>)
  }
  return
}

// -----

// Outlining multiple herds to the same aie.core

// CHECK: aie.device
// CHECK: aie.core
// CHECK-NEXT: cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK: cf.br ^bb2
// CHECK-NEXT: ^bb2:
// CHECK: air.channel.put  @channel_1
// CHECK: cf.br ^bb3
// CHECK-NEXT: ^bb3:
// CHECK: cf.br ^bb4
// CHECK-NEXT: ^bb4:
// CHECK: air.channel.put  @channel_1
// CHECK: aie.end
// EMITWHILE: aie.device
// EMITWHILE: aie.core
// EMITWHILE-NEXT: cf.br ^bb1
// EMITWHILE-NEXT: ^bb1:
// EMITWHILE: cf.br ^bb2
// EMITWHILE-NEXT: ^bb2:
// EMITWHILE: air.channel.put  @channel_1
// EMITWHILE: cf.br ^bb3
// EMITWHILE-NEXT: ^bb3:
// EMITWHILE: cf.br ^bb4
// EMITWHILE-NEXT: ^bb4:
// EMITWHILE: air.channel.put  @channel_1
// EMITWHILE: cf.br ^bb1
air.channel @channel_1 [1, 1]
func.func @f2() -> () {
  air.segment @segment0 attributes {x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
    %cst1 = arith.constant 1 : index
    air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
      %src0 = memref.alloc() : memref<1xi32, 2>
      air.channel.put @channel_1[] (%src0[] [] []) : (memref<1xi32, 2>)
      memref.dealloc %src0 : memref<1xi32, 2>
    }
    air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
      %src0 = memref.alloc() : memref<1xi32, 2>
      air.channel.put @channel_1[] (%src0[] [] []) : (memref<1xi32, 2>)
      memref.dealloc %src0 : memref<1xi32, 2>
    }
  }
  return
}
