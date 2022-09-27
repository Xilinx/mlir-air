//===- air_linalg_codegen_filters.mlir -------------------------*- MLIR -*-===//
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

// check unchanged with no filter options
// RUN: air-opt %s -air-linalg-codegen | FileCheck %s -check-prefix=UNCH
// UNCH: linalg.matmul {__internal_linalg_transform__ = "mmult"} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>)
// UNCH: linalg.generic {{.*}} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic0"} {
// UNCH: linalg.generic {{.*}} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic1"} {

// check generic1 options applied
// RUN: air-opt %s -air-linalg-codegen='input-filter=generic1' | FileCheck %s -check-prefix=GENERIC1
// GENERIC1: linalg.matmul {__internal_linalg_transform__ = "mmult"} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>)
// GENERIC1: linalg.generic {{.*}} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic0"} {
// GENERIC1: linalg.generic {{.*}} ins({{.*}} : memref<64x32xi32, 2>, memref<64x32xi32, 2>) outs({{.*}} : memref<64x32xi32, 2>) {

#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<128x128xi32>, %arg1: memref<128x128xi32>, %arg2: memref<128x128xi32>, %arg3: memref<128x128xi32>) -> memref<?x?xi32> {
    %0 = memref.alloc() : memref<128x128xi32>
    %1 = memref.alloc() : memref<128x128xi32>
    linalg.matmul {__internal_linalg_transform__ = "mmult"} ins(%arg2, %arg3 : memref<128x128xi32>, memref<128x128xi32>) outs(%1 : memref<128x128xi32>)
    %2 = memref.alloc() : memref<128x128xi32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %1 : memref<128x128xi32>, memref<128x128xi32>) outs(%2 : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic0"} {
    ^bb0(%arg4: i32, %arg5: i32, %arg6: i32):
      %5 = arith.muli %arg4, %arg5 : i32
      linalg.yield %5 : i32
    }
    %3 = memref.alloc() : memref<128x128xi32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : memref<128x128xi32>, memref<128x128xi32>) outs(%3 : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic1"} {
    ^bb0(%arg4: i32, %arg5: i32, %arg6: i32):
      %5 = arith.addi %arg4, %arg5 : i32
      linalg.yield %5 : i32
    }
    %4 = memref.cast %3 : memref<128x128xi32> to memref<?x?xi32>
    return %4 : memref<?x?xi32>
  }
}

