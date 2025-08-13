//===- air_fuse_nested_herds.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-nested-herd --split-input-file | FileCheck %s
// RUN: air-opt %s -air-fuse-nested-herd="order=inner-outer"  --split-input-file | FileCheck %s --check-prefix=INOUT

// CHECK-LABEL: test0
// CHECK: air.herd @herd_1  tile (%[[arg0:.*]], %[[arg1:.*]]) in (%[[arg2:.*]]=%c2{{.*}}, %[[arg3:.*]]=%c4{{.*}})
// INOUT-LABEL: test0
// INOUT: air.herd @herd_1  tile (%[[arg0:.*]], %[[arg1:.*]]) in (%[[arg2:.*]]=%c4{{.*}}, %[[arg3:.*]]=%c2{{.*}})
func.func @test0() {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  air.herd @herd_1  tile (%arg0, %arg1) in (%arg2=%c2, %arg3=%c1) {
    %c4 = arith.constant 4 : index
    %c1_0 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg4, %arg5) in (%arg6=%c4, %arg7=%c1_0) {
      %c0_i32 = arith.constant 0 : i32
      %alloc = memref.alloc() : memref<32xi32, 2 : i32>
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<32xi32, 2 : i32>)
    }
  }
  return
}

// -----

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s1 - 3 == 0, s0 >= 0, -s0 + 1 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 1 >= 0)>
// CHECK-LABEL: test1
// CHECK: air.herd @herd_1  tile (%[[arg0:.*]], %[[arg1:.*]]) in (%[[arg2:.*]]=%c2{{.*}}, %[[arg3:.*]]=%c4{{.*}})
// CHECK: affine.if [[$SET0]]()[%[[arg0]], %[[arg1]]]
// CHECK: affine.if [[$SET1]]()[%[[arg0]], %[[arg1]]]
// INOUT: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 1 >= 0)>
// INOUT: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 - 1 >= 0, -s0 + 2 >= 0, s1 >= 0, -s1 + 1 >= 0)>
// INOUT-LABEL: test1
// INOUT: air.herd @herd_1  tile (%[[arg0:.*]], %[[arg1:.*]]) in (%[[arg2:.*]]=%c4{{.*}}, %[[arg3:.*]]=%c2{{.*}})
// INOUT: affine.if [[$SET0]]()[%[[arg0]], %[[arg1]]]
// INOUT: affine.if [[$SET1]]()[%[[arg0]], %[[arg1]]]
#set = affine_set<()[s0] : (s0 - 3 == 0)>
#set1 = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
func.func @test1() {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  air.herd @herd_1  tile (%arg0, %arg1) in (%arg2=%c2, %arg3=%c1) {
    %c4 = arith.constant 4 : index
    %c1_0 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg4, %arg5) in (%arg6=%c4, %arg7=%c1_0) {
      %c0_i32 = arith.constant 0 : i32
      %alloc = memref.alloc() : memref<32xi32, 2 : i32>
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<32xi32, 2 : i32>)
      affine.if #set()[%arg4] {
        %alloc_1 = memref.alloc() : memref<32xi32, 2 : i32>
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<32xi32, 2 : i32>)
      } else {
        affine.if #set1()[%arg4] {
          %alloc_1 = memref.alloc() : memref<32xi32, 2 : i32>
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<32xi32, 2 : i32>)
        } else {
          %alloc_1 = memref.alloc() : memref<32xi32, 2 : i32>
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<32xi32, 2 : i32>)
        }
      }
    }
  }
  return
}
