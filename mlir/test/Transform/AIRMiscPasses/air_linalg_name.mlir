//===- air_linalg_name.mlir ------------------------------------*- MLIR -*-===//
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

// RUN: air-opt -air-linalg-name %s | FileCheck %s
// CHECK: linalg.fill {__internal_linalg_transform__ = "linalg.fill0"} {{.*}}
// CHECK: linalg.copy {__internal_linalg_transform__ = "linalg.copy1"} {{.*}}
// CHECK: linalg.matmul {__internal_linalg_transform__ = "linalg.matmul2"}
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<2304x1024xi32>, %arg1: memref<1024x1024xi32>) -> memref<?x?xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() : memref<2304x1024xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<2304x1024xi32>)
    %1 = memref.alloc() : memref<2304x1024xi32>
    linalg.copy ins(%0 : memref<2304x1024xi32>) outs(%1 : memref<2304x1024xi32>)
    linalg.matmul ins(%arg0, %arg1 : memref<2304x1024xi32>, memref<1024x1024xi32>) outs(%1 : memref<2304x1024xi32>)
    %2 = memref.cast %1 : memref<2304x1024xi32> to memref<?x?xi32>
    return %2 : memref<?x?xi32>
  }
}

