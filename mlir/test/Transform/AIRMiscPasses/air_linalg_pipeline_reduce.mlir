// (c) Copyright 2022 AMD, Inc.

// RUN: air-opt -air-pipeline-reduce %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
module  {
  func @launch(%m0: memref<64x256xbf16, 1>, %m1: memref<256x64xbf16, 1>, %m2: memref<64x64xbf16, 1>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    air.launch_herd tile (%x, %y) in (%sx=%c1, %sy=%c12) args(%a=%m0, %b=%m1, %c=%m2) : memref<64x256xbf16, 1>,memref<256x64xbf16, 1>,memref<64x64xbf16, 1> {
      %ta = bufferization.to_tensor %a : memref<64x256xbf16, 1>
      %tb = bufferization.to_tensor %b : memref<256x64xbf16, 1>
      %tc_in = bufferization.to_tensor %c : memref<64x64xbf16, 1>
      %tc_out = linalg.matmul ins(%ta, %tb : tensor<64x256xbf16>, tensor<256x64xbf16>) outs(%tc_in : tensor<64x64xbf16>) -> tensor<64x64xbf16>
      %out = bufferization.to_memref %tc_out : memref<64x64xbf16, 1>
      memref.copy %out, %c : memref<64x64xbf16, 1> to  memref<64x64xbf16, 1>
      air.herd_terminator
    }
    return
  }
}
