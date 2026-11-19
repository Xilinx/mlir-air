//===- fusion_cases.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// The first op has another user: it stays for that user, and the fused op
// does not return its result.

// CHECK-LABEL: func.func @first_op_has_other_user
// CHECK: %[[FIRST:.*]] = linalg.generic {{.*}} ins(%arg0 : tensor<4x8xf16>)
// CHECK: %[[FUSED:.*]] = linalg.generic {{.*}} ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf32>)
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f16, %[[B:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[EXT:.*]] = arith.extf %[[IN]]
// CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[EXT]], %[[B]]
// CHECK-NEXT: linalg.yield %[[ADD]]
// CHECK-NEXT: } -> tensor<4x8xf32>
// CHECK: return %[[FIRST]], %[[FUSED]]
func.func @first_op_has_other_user(%a: tensor<4x8xf16>, %b: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %e = tensor.empty() : tensor<4x8xf32>
  %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<4x8xf16>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%in: f16, %o: f32):
    %0 = arith.extf %in : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  %c = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%p, %b : tensor<4x8xf32>, tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.addf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  return %p, %c : tensor<4x8xf32>, tensor<4x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "first_op_has_other_user"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %ops = transform.structured.match interface{LinalgOp} in %f : (!transform.any_op) -> !transform.any_op
    %first, %second = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.air.fuse_multi_op_linalg %first, %second : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// The second op reads the result twice: only the first read is fused.

// CHECK-LABEL: func.func @result_read_twice
// CHECK: %[[FIRST:.*]] = linalg.generic {{.*}} ins(%arg0 : tensor<4x8xf16>)
// CHECK: linalg.generic {{.*}} ins(%arg0, %[[FIRST]] : tensor<4x8xf16>, tensor<4x8xf32>)
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f16, %[[R:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[EXT:.*]] = arith.extf %[[IN]]
// CHECK-NEXT: arith.mulf %[[EXT]], %[[R]]
func.func @result_read_twice(%a: tensor<4x8xf16>) -> tensor<4x8xf32> {
  %e = tensor.empty() : tensor<4x8xf32>
  %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<4x8xf16>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%in: f16, %o: f32):
    %0 = arith.extf %in : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  %c = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%p, %p : tensor<4x8xf32>, tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.mulf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "result_read_twice"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %ops = transform.structured.match interface{LinalgOp} in %f : (!transform.any_op) -> !transform.any_op
    %first, %second = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.air.fuse_multi_op_linalg %first, %second : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// The fused input is not the first one: the first op's inputs take its place,
// and the other inputs keep their order.

// CHECK-LABEL: func.func @fused_input_in_the_middle
// CHECK: linalg.generic {indexing_maps = [#map, #map, #map1, #map1, #map]
// CHECK-SAME: ins(%arg2, %arg0, %arg1, %arg1 : tensor<4x8xf32>, tensor<4x8xf16>, tensor<4xf32>, tensor<4xf32>)
// CHECK-NEXT: ^bb0(%[[B:.*]]: f32, %[[IN:.*]]: f16, %[[M:.*]]: f32, %[[Z:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[EXT:.*]] = arith.extf %[[IN]]
// CHECK-NEXT: %[[SUB:.*]] = arith.subf %[[EXT]], %[[M]]
// CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[B]], %[[SUB]]
// CHECK-NEXT: arith.addf %[[MUL]], %[[Z]]
func.func @fused_input_in_the_middle(%a: tensor<4x8xf16>, %m: tensor<4xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %e = tensor.empty() : tensor<4x8xf32>
  %p = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%a, %m : tensor<4x8xf16>, tensor<4xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%in: f16, %mm: f32, %o: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.subf %0, %mm : f32
    linalg.yield %1 : f32
  } -> tensor<4x8xf32>
  %c = linalg.generic {indexing_maps = [#id, #id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%b, %p, %m : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32, %o: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %0, %z : f32
    linalg.yield %1 : f32
  } -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "fused_input_in_the_middle"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %ops = transform.structured.match interface{LinalgOp} in %f : (!transform.any_op) -> !transform.any_op
    %first, %second = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.air.fuse_multi_op_linalg %first, %second : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// A named second op is generalized as part of the fusion.

// CHECK-LABEL: func.func @named_second_op
// CHECK-NOT: linalg.reduce
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4x8xf16>)
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f16, %[[ACC:.*]]: f32):
// CHECK-NEXT: %[[EXT:.*]] = arith.extf %[[IN]]
// CHECK-NEXT: arith.addf %[[EXT]], %[[ACC]]
func.func @named_second_op(%a: tensor<4x8xf16>) -> tensor<4xf32> {
  %e = tensor.empty() : tensor<4x8xf32>
  %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<4x8xf16>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%in: f16, %o: f32):
    %0 = arith.extf %in : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  %e2 = tensor.empty() : tensor<4xf32>
  %c = linalg.reduce ins(%p : tensor<4x8xf32>) outs(%e2 : tensor<4xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
      %0 = arith.addf %in, %acc : f32
      linalg.yield %0 : f32
    }
  return %c : tensor<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "named_second_op"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %ops = transform.structured.match interface{LinalgOp} in %f : (!transform.any_op) -> !transform.any_op
    %first, %second = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.air.fuse_multi_op_linalg %first, %second : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// A named first op is fused through a generic copy; the named op itself is
// left for its other user.

// CHECK-LABEL: func.func @named_first_op
// CHECK: %[[FIRST:.*]] = linalg.elementwise <add> ins(%arg0, %arg1
// CHECK: %[[FUSED:.*]] = linalg.generic {{.*}} ins(%arg0, %arg1, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[A]], %[[B]]
// CHECK-NEXT: arith.mulf %[[ADD]], %[[C]]
// CHECK: return %[[FUSED]], %[[FIRST]]
func.func @named_first_op(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>, %c: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %e = tensor.empty() : tensor<4x8xf32>
  %p = linalg.elementwise <add> ins(%a, %b : tensor<4x8xf32>, tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) -> tensor<4x8xf32>
  %e2 = tensor.empty() : tensor<4x8xf32>
  %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%p, %c : tensor<4x8xf32>, tensor<4x8xf32>) outs(%e2 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.mulf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  return %r, %p : tensor<4x8xf32>, tensor<4x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "named_first_op"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %ops = transform.structured.match interface{LinalgOp} in %f : (!transform.any_op) -> !transform.any_op
    %first, %second = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.air.fuse_multi_op_linalg %first, %second : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
