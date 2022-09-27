//===- air_to_cpu.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
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

// RUN: air-opt %s -air-to-cpu | FileCheck %s

// Check that the herd body was outlined and spot check the ops were turned into calls
// CHECK: affine.for %{{.*}} = 0 to 32 {
// CHECK:   affine.for %{{.*}} = 0 to 32 {
// CHECK:     call @herd_0_body_fn(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (index, index, memref<1024x1024xi32>, memref<1024x1024xi32>, memref<1024x1024xi32>) -> ()
// CHECK:   } {air.herd = "inner"}
// CHECK: } {air.herd = "outer"}
// CHECK: func.func @herd_0_body_fn(%{{.*}}: index, %{{.*}}: index, %{{.*}}: memref<1024x1024xi32>, %{{.*}}: memref<1024x1024xi32>, %{{.*}}: memref<1024x1024xi32>) attributes {llvm.emit_c_interface} {
// CHECK: call @air_alloc_rM2D2I32_I64_I64
// CHECK: call @air_memcpy_nd_I32_I64_I64_M2D2I32_M0D2I32_I64_I64_I64_I64_I64_I64
// CHECK  linalg.matmul
// CHECK: call @air_memcpy_nd_I32_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64_M2D2I32
// CHECK: call @air_dealloc_I64_I64_M2D2I32
module  {
  func.func @forward(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<1024x1024xi32>) {
    %c32 = arith.constant 32 : index
    air.herd tile (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<1024x1024xi32>, memref<1024x1024xi32>, memref<1024x1024xi32> attributes {sym_name = "herd_0"} {
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %c32_0 = arith.constant 32 : index
      %0 = arith.muli %arg3, %c32_0 : index
      %1 = arith.muli %arg4, %c32_0 : index
      %2 = memref.alloc() : memref<32x32xi32, 2>
      %3 = memref.alloc() : memref<32x32xi32, 2>
      %4 = memref.alloc() : memref<32x32xi32, 2>
      air.dma_memcpy_nd (%2[] [] [], %arg7[%0, %c0] [%c32_0, %c32_0] [%c1024, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<1024x1024xi32>)
      air.dma_memcpy_nd (%3[] [] [], %arg8[%c0, %1] [%c32_0, %c32_0] [%c1024, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<1024x1024xi32>)
      air.dma_memcpy_nd (%4[] [] [], %arg9[%0, %1] [%c32_0, %c32_0] [%c1024, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<1024x1024xi32>)
      linalg.matmul ins(%2, %3 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%4 : memref<32x32xi32, 2>)
      air.dma_memcpy_nd (%arg9[%0, %1] [%c32_0, %c32_0] [%c1024, %c1], %4[] [] []) {id = 4 : i32} : (memref<1024x1024xi32>, memref<32x32xi32, 2>)
      memref.dealloc %2 : memref<32x32xi32, 2>
      memref.dealloc %3 : memref<32x32xi32, 2>
      memref.dealloc %4 : memref<32x32xi32, 2>
      air.herd_terminator
    }
    return
  }
}
