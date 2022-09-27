//===- parallel_herds.mlir -------------------------------------*- MLIR -*-===//
//
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

// RUN: air-opt %s -air-dependency | FileCheck %s

// The second herd should not depend on the first

// CHECK: %[[EVENT0:.*]] = scf.for
// CHECK: } {id = 14 : i32} : (index)
// CHECK-NOT: %[[EVENT1:.*]] = air.wait_all async [{{.*}}%[[EVENT0]]
// CHECK: %[[EVENT2:.*]] = scf.for

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
#map2 = affine_map<()[s0] -> (s0 * 512 + 64)>
module {
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) {
    %c16 = arith.constant 16 : index
    %c48 = arith.constant 48 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    linalg.fill ins(%cst : bf16) outs(%0 : memref<24576x1024xbf16>)
    %1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    memref.copy %0, %1 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    air.launch @launch_0 (%arg2, %arg3) in (%arg4=%c48, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%1, %arg9=%2) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 6 : i64, size_y = 2 : i64} {
      air.partition @partition_0  args(%arg10=%arg2, %arg11=%arg3, %arg12=%arg4, %arg13=%arg5, %arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 3 : i64, size_y = 2 : i64} {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %3 = affine.apply #map0()[%arg11]
        %4 = affine.apply #map1()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %6 = memref.alloc() : memref<64x64xbf16, 1>
          %7 = memref.alloc() : memref<64x64xbf16, 1>
          %8 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%6[] [] [], %arg14[%4, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%7[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%8[] [] [], %arg16[%4, %3] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) {
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%4, %3] [%c64, %c64] [%c1024, %c1], %8[] [] []) {id = 4 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %6 : memref<64x64xbf16, 1>
          memref.dealloc %7 : memref<64x64xbf16, 1>
          memref.dealloc %8 : memref<64x64xbf16, 1>
        }
        %5 = affine.apply #map2()[%arg10]
        scf.for %arg18 = %c0 to %c1024 step %c64 {
          %6 = memref.alloc() : memref<64x64xbf16, 1>
          %7 = memref.alloc() : memref<64x64xbf16, 1>
          %8 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%6[] [] [], %arg14[%5, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 5 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%7[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 6 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%8[] [] [], %arg16[%5, %3] [%c64, %c64] [%c1024, %c1]) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) {
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg16[%5, %3] [%c64, %c64] [%c1024, %c1], %8[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %6 : memref<64x64xbf16, 1>
          memref.dealloc %7 : memref<64x64xbf16, 1>
          memref.dealloc %8 : memref<64x64xbf16, 1>
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return
  }
}

