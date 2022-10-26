//===- create_and_outline.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='test-patterns=to-aie-mlir' -o /dev/null | FileCheck %s

// CHECK: [[T00:%.*]] = AIE.tile(1, 1)
// CHECK: [[T10:%.*]] = AIE.tile(2, 1)
// CHECK: [[T01:%.*]] = AIE.tile(1, 2)
// CHECK: [[T11:%.*]] = AIE.tile(2, 2)
// CHECK: AIE.core([[T11]])
// CHECK: AIE.core([[T01]])
// CHECK: AIE.core([[T10]])
// CHECK: AIE.core([[T00]])
#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%a0: memref<64x64xi32>, %a1: memref<64x64xi32>, %a2: memref<64x64xi32>) {
    air.partition @partition0 args(%arg0=%a0, %arg1=%a1, %arg2=%a2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
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
        air.herd_terminator
      }
      memref.copy %1, %arg2 : memref<64x64xi32> to memref<64x64xi32>
      air.partition_terminator
    }
    return
  }
}
