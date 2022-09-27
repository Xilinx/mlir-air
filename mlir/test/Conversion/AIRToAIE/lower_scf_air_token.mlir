//===- lower_scf_air_token.mlir --------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-to-aie='test-patterns=lower-scf-tokens' | FileCheck %s

module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<32x32xi32, 2>, %arg1: memref<32x32xi32, 2>, %arg2: memref<32x32xi32, 2>) {
    %c1 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<32x32xi32, 2>, memref<32x32xi32, 2>, memref<32x32xi32, 2> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %0 = air.wait_all async
      // CHECK: scf.for %{{.*}} = %c0 to %c64 step %c32 {
      %1 = scf.for %arg10 = %c0 to %c64 step %c32 iter_args(%arg11 = %0) -> (!air.async.token) {
        %asyncToken = air.execute async [%arg11]  : (!air.async.token) {
          linalg.matmul ins(%arg7, %arg8 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%arg9 : memref<32x32xi32, 2>)
          air.execute_terminator
        } {id = 1 : i32}
        %2 = air.wait_all async [%asyncToken] 
        scf.yield %2 : !air.async.token
      }
      air.wait_all [%1] 
      air.herd_terminator
    }
    return
  }
}
