//===- air_fuse_into_containing_op.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN:  air-opt %s -test-transform-dialect-interpreter


#map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: test_fuse_into_par
// CHECK: scf.parallel
// CHECK: linalg.fill {__producer__} {{.*}} memref<32x32xf32, strided<[128, 1], offset: ?>>
// CHECK: linalg.generic
// CHECK: scf.yield
func.func @test_fuse_into_par(%D: memref<128x128xf32>) -> memref<128x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill {__producer__} ins(%cst : f32) outs(%D : memref<128x128xf32>)
  linalg.generic
      {__consumer__,
        indexing_maps = [#map0, #map0],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%D : memref<128x128xf32>) outs(%D : memref<128x128xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    linalg.yield %arg2 : f32
  }
  return %D : memref<128x128xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Find the consumer and producer.
  %consumer = transform.structured.match attributes{"__consumer__"} in %arg1
  %producers = transform.structured.match attributes{"__producer__"} in %arg1
  %1, %loops:2 = transform.air.linalg_tile %consumer [32, 32]
  transform.air.fuse_into_containing_op %producers into %loops#0
}
