// RUN: air-opt %s -air-linalg-codegen=test-patterns | FileCheck %s
// CHECK: func @forward(%[[VAL_0:.*]]: memref<4096xi32>, %[[VAL_1:.*]]: memref<4096xi32>, %[[VAL_2:.*]]: memref<?xi32>) {
// CHECK:   %[[VAL_3:.*]] = memref.cast %[[VAL_2]] : memref<?xi32> to memref<4096xi32>
// CHECK:   linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : memref<4096xi32>, memref<4096xi32>) outs(%[[VAL_3]] : memref<4096xi32>) {
// CHECK:   ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:     %[[VAL_7:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:     linalg.yield %[[VAL_7]] : i32
// CHECK:   }
// CHECK:   return
// CHECK: }
// XFAIL: *
module attributes {torch.debug_module_name = "model"}  {
  func @forward(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<?xi32>) {
    %0 = memref.alloc() : memref<4096xi32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<4096xi32>, memref<4096xi32>) outs(%0 : memref<4096xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
      %2 = arith.muli %arg3, %arg4 : i32
      linalg.yield %2 : i32
    }
    %1 = memref.cast %0 : memref<4096xi32> to memref<?xi32>
    memref.copy %1, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}