//===- air.mlir ------------------------------------------------*- MLIR -*-===//
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

#map0 = affine_map<(d0) -> (d0)>
module  {
  func.func @launch(%m0: memref<1024xi32>, %m1: memref<1024xi32>, %m2: memref<1024xi32>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    air.herd tile (%x, %y) in (%sx=%c4, %sy=%c1) args(%op0=%m0, %op1=%m1, %op2=%m2) : memref<1024xi32>,memref<1024xi32>,memref<1024xi32> attributes {sym_name="herd_0"} {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c1_1 = arith.constant 1 : index

      air.pipeline attributes {direction = "horiz"} {

        %output1 = air.pipeline.stage {
          %a = memref.alloc() : memref<1024xi32, 2>
          %b = memref.alloc() : memref<1024xi32, 2>
          air.dma_memcpy_nd (%a[][][], %op0[%c0][%c1024][%c1_1]) {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>)
          air.dma_memcpy_nd (%b[][][], %op1[%c0][%c1024][%c1_1]) {id = 2 : i32} : (memref<1024xi32, 2>, memref<1024xi32>)
          %init = tensor.empty () : tensor<1024xi32>
          %ta = bufferization.to_tensor %a : memref<1024xi32, 2>
          %tb = bufferization.to_tensor %b : memref<1024xi32, 2>
          %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%ta, %tb : tensor<1024xi32>, tensor<1024xi32>) outs(%init : tensor<1024xi32>) {
          ^bb0(%a2: i32, %a3: i32, %a4: i32):  // no predecessors
            %6 = arith.muli %a2, %a3 : i32
            linalg.yield %6 : i32
          } -> tensor<1024xi32>
          air.pipeline.yield %5 : tensor<1024xi32>
        } : tensor<1024xi32>

        %output2 = air.pipeline.stage args(%in = %output1) : tensor<1024xi32> {
          %init = tensor.empty () : tensor<1024xi32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xi32>) outs(%init : tensor<1024xi32>) {
          ^bb0(%a2: i32, %a3: i32):  // no predecessors
            %one = arith.constant 1 : i32
            %6 = arith.addi %a2, %one : i32
            linalg.yield %6 : i32
          } -> tensor<1024xi32>
          air.pipeline.yield %5 : tensor<1024xi32>
        } : tensor<1024xi32>

        %output3 = air.pipeline.stage args(%in = %output2) : tensor<1024xi32> {
          %init = tensor.empty () : tensor<1024xi32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xi32>) outs(%init : tensor<1024xi32>) {
          ^bb0(%a2: i32, %a3: i32):  // no predecessors
            %two = arith.constant 2 : i32
            %6 = arith.addi %a2, %two : i32
            linalg.yield %6 : i32
          } -> tensor<1024xi32>
          air.pipeline.yield %5 : tensor<1024xi32>
        } : tensor<1024xi32>

        air.pipeline.stage args(%in = %output3) : tensor<1024xi32> {
          %init = tensor.empty () : tensor<1024xi32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xi32>) outs(%init : tensor<1024xi32>) {
          ^bb0(%a2: i32, %a3: i32):  // no predecessors
            %three = arith.constant 3 : i32
            %6 = arith.addi %a2, %three : i32
            linalg.yield %6 : i32
          } -> tensor<1024xi32>
          %c = bufferization.to_memref %5 : memref<1024xi32, 2>
          air.dma_memcpy_nd (%op2[%c0][%c1024][%c1_1], %c[][][]) {id = 3 : i32} : (memref<1024xi32>, memref<1024xi32, 2>)
          air.pipeline.yield
        }
        air.pipeline.terminator
      }
      air.herd_terminator
    }
    return
  }
}