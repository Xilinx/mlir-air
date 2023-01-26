//===- air_linalg_rm_copy.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-linalg-codegen=test-patterns | FileCheck %s

// Check that the copy is removed
// CHECK-LABEL: test_copy_remove
// CHECK:       %[[A0:.*]] = memref.alloc() : memref<2560xf32, 2>
// CHECK-NEXT:  %[[A1:.*]] = memref.alloc() : memref<2560xf32, 2>
// CHECK-NEXT:  memref.copy %{{.*}}, %[[A0]]
// CHECK-NEXT:  linalg.generic {{.*}} ins(%[[A0]] {{.*}} outs(%[[A1]]
// CHECK:       memref.copy %[[A1]], %{{.*}}
#map = affine_map<(d0) -> (d0)>

func.func @test_copy_remove(%arg0: memref<10240xf32>) -> memref<10240xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<10240xf32>
  %c2560 = arith.constant 2560 : index
  %c0 = arith.constant 0 : index
  %c10240 = arith.constant 10240 : index
  %c5120 = arith.constant 5120 : index
  scf.parallel (%arg1) = (%c0) to (%c10240) step (%c5120) {
    scf.for %arg2 = %c0 to %c5120 step %c2560 {
      %0 = arith.addi %arg1, %arg2 : index
      %subview = memref.subview %arg0[%0] [2560] [1] : memref<10240xf32> to memref<2560xf32, strided<[1], offset: ?>>
      %1 = arith.addi %arg1, %arg2 : index
      %subview_0 = memref.subview %alloc[%1] [2560] [1] : memref<10240xf32> to memref<2560xf32, strided<[1], offset: ?>>
      %alloc_1 = memref.alloc() : memref<2560xf32, 2>
      %alloc_2 = memref.alloc() : memref<2560xf32, 2>
      memref.copy %subview, %alloc_1 : memref<2560xf32, strided<[1], offset: ?>> to memref<2560xf32, 2>
      memref.copy %subview_0, %alloc_2 : memref<2560xf32, strided<[1], offset: ?>> to memref<2560xf32, 2>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_1 : memref<2560xf32, 2>) outs(%alloc_2 : memref<2560xf32, 2>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.cmpf ugt, %in, %cst : f32
        %3 = arith.select %2, %in, %cst : f32
        linalg.yield %3 : f32
      }
      memref.copy %alloc_2, %subview_0 : memref<2560xf32, 2> to memref<2560xf32, strided<[1], offset: ?>>
      memref.dealloc %alloc_1 : memref<2560xf32, 2>
      memref.dealloc %alloc_2 : memref<2560xf32, 2>
    }
    scf.yield
  }
  return %alloc : memref<10240xf32>
}

// Check that the copy is not removed
// CHECK-LABEL: test_copy_reduce
// CHECK:       %[[A0:.*]] = memref.alloc() : memref<2560xf32, 2>
// CHECK-NEXT:  %[[A1:.*]] = memref.alloc() : memref<1280xf32, 2>
// CHECK-NEXT:  memref.copy %{{.*}}, %[[A0]]
// CHECK-NEXT:  memref.copy %{{.*}}, %[[A1]]
// CHECK-NEXT:  linalg.generic {{.*}} ins(%[[A0]] {{.*}} outs(%[[A1]]
// CHECK:       memref.copy %[[A1]], %{{.*}}
#map1 = affine_map<(d0) -> (d0 floordiv 2)>
func.func @test_copy_reduce(%arg0: memref<10240xf32>) -> memref<10240xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<10240xf32>
  %c2560 = arith.constant 2560 : index
  %c0 = arith.constant 0 : index
  %c10240 = arith.constant 10240 : index
  %c5120 = arith.constant 5120 : index
  scf.parallel (%arg1) = (%c0) to (%c10240) step (%c5120) {
    scf.for %arg2 = %c0 to %c5120 step %c2560 {
      %0 = arith.addi %arg1, %arg2 : index
      %subview = memref.subview %arg0[%0] [2560] [1] : memref<10240xf32> to memref<2560xf32, strided<[1], offset: ?>>
      %1 = arith.addi %arg1, %arg2 : index
      %subview_0 = memref.subview %alloc[%1] [1280] [1] : memref<10240xf32> to memref<1280xf32, strided<[1], offset: ?>>
      %alloc_1 = memref.alloc() : memref<2560xf32, 2>
      %alloc_2 = memref.alloc() : memref<1280xf32, 2>
      memref.copy %subview, %alloc_1 : memref<2560xf32, strided<[1], offset: ?>> to memref<2560xf32, 2>
      memref.copy %subview_0, %alloc_2 : memref<1280xf32, strided<[1], offset: ?>> to memref<1280xf32, 2>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_1 : memref<2560xf32, 2>) outs(%alloc_2 : memref<1280xf32, 2>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.cmpf ugt, %in, %out : f32
        %3 = arith.select %2, %in, %out : f32
        linalg.yield %3 : f32
      }
      memref.copy %alloc_2, %subview_0 : memref<1280xf32, 2> to memref<1280xf32, strided<[1], offset: ?>>
      memref.dealloc %alloc_1 : memref<2560xf32, 2>
      memref.dealloc %alloc_2 : memref<1280xf32, 2>
    }
    scf.yield
  }
  return %alloc : memref<10240xf32>
}

// Check that the copy is not removed
// CHECK-LABEL: test_copy_use
// CHECK:       %[[A0:.*]] = memref.alloc() : memref<2560xf32, 2>
// CHECK-NEXT:  %[[A1:.*]] = memref.alloc() : memref<2560xf32, 2>
// CHECK-NEXT:  memref.copy %{{.*}}, %[[A0]]
// CHECK-NEXT:  memref.copy %{{.*}}, %[[A1]]
// CHECK-NEXT:  func.call @external
// CHECK-NEXT:  linalg.generic {{.*}} ins(%[[A0]] {{.*}} outs(%[[A1]]
// CHECK:       memref.copy %[[A1]], %{{.*}}
func.func @test_copy_use(%arg0: memref<10240xf32>) -> memref<10240xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<10240xf32>
  %c2560 = arith.constant 2560 : index
  %c0 = arith.constant 0 : index
  %c10240 = arith.constant 10240 : index
  %c5120 = arith.constant 5120 : index
  scf.parallel (%arg1) = (%c0) to (%c10240) step (%c5120) {
    scf.for %arg2 = %c0 to %c5120 step %c2560 {
      %0 = arith.addi %arg1, %arg2 : index
      %subview = memref.subview %arg0[%0] [2560] [1] : memref<10240xf32> to memref<2560xf32, strided<[1], offset: ?>>
      %1 = arith.addi %arg1, %arg2 : index
      %subview_0 = memref.subview %alloc[%1] [2560] [1] : memref<10240xf32> to memref<2560xf32, strided<[1], offset: ?>>
      %alloc_1 = memref.alloc() : memref<2560xf32, 2>
      %alloc_2 = memref.alloc() : memref<2560xf32, 2>
      memref.copy %subview, %alloc_1 : memref<2560xf32, strided<[1], offset: ?>> to memref<2560xf32, 2>
      memref.copy %subview_0, %alloc_2 : memref<2560xf32, strided<[1], offset: ?>> to memref<2560xf32, 2>
      func.call @external(%alloc_2) : (memref<2560xf32, 2>) -> ()
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_1 : memref<2560xf32, 2>) outs(%alloc_2 : memref<2560xf32, 2>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.cmpf ugt, %in, %cst : f32
        %3 = arith.select %2, %in, %cst : f32
        linalg.yield %3 : f32
      }
      memref.copy %alloc_2, %subview_0 : memref<2560xf32, 2> to memref<2560xf32, strided<[1], offset: ?>>
      memref.dealloc %alloc_1 : memref<2560xf32, 2>
      memref.dealloc %alloc_2 : memref<2560xf32, 2>
    }
    scf.yield
  }
  return %alloc : memref<10240xf32>
}
func.func private @external(%arg0: memref<2560xf32, 2>) -> ()