//===- forall_init.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// Each iteration reads the tile of %init that it then writes into %o: the
// read takes it from %o. The read of a tile at other offsets, which another
// iteration writes, still takes it from %init.

// CHECK-LABEL: func.func @own_tile
// CHECK-SAME: (%[[INIT:.*]]: tensor<64x1xf32>
// CHECK: scf.forall (%[[I:.*]]) in (4) shared_outs(%[[O:.*]] = %[[INIT]])
// CHECK: %[[OFF:.*]] = affine.apply
// CHECK: %[[OLD:.*]] = tensor.extract_slice %[[O]][%[[OFF]], 0] [16, 1] [1, 1]
// CHECK: %[[NEXT:.*]] = tensor.extract_slice %[[INIT]][%{{.*}}, 0] [16, 1] [1, 1]
// CHECK: arith.addf %[[OLD]], %[[NEXT]]
// CHECK: tensor.parallel_insert_slice %{{.*}} into %[[O]][%[[OFF]], 0] [16, 1] [1, 1]
func.func @own_tile(%init: tensor<64x1xf32>) -> tensor<64x1xf32> {
  %r = scf.forall (%i) in (4) shared_outs(%o = %init) -> (tensor<64x1xf32>) {
    %off = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)
    %noff = affine.apply affine_map<(d0) -> ((d0 * 16 + 16) mod 64)>(%i)
    %old = tensor.extract_slice %init[%off, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    %next = tensor.extract_slice %init[%noff, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    %new = arith.addf %old, %next : tensor<16x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %new into %o[%off, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
    }
  }
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Each iteration writes two tiles of %o, so no tile read from %init is the
// one the iteration writes: the read stays on %init.

// CHECK-LABEL: func.func @two_inserts
// CHECK-SAME: (%[[INIT:.*]]: tensor<64x1xf32>
// CHECK: tensor.extract_slice %[[INIT]]
func.func @two_inserts(%init: tensor<64x1xf32>) -> tensor<64x1xf32> {
  %r = scf.forall (%i) in (2) shared_outs(%o = %init) -> (tensor<64x1xf32>) {
    %off = affine.apply affine_map<(d0) -> (d0 * 32)>(%i)
    %off2 = affine.apply affine_map<(d0) -> (d0 * 32 + 16)>(%i)
    %old = tensor.extract_slice %init[%off, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    %new = arith.addf %old, %old : tensor<16x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %new into %o[%off, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
      tensor.parallel_insert_slice %old into %o[%off2, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
    }
  }
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Two shared outputs. The iteration reads its own tile of %init but not of
// %init2, whose tile it reads at other offsets: only the read of %init moves
// to its shared output.

// CHECK-LABEL: func.func @two_shared_outs
// CHECK-SAME: (%[[INIT:.*]]: tensor<64x1xf32>, %[[INIT2:.*]]: tensor<64x1xf32>
// CHECK: scf.forall (%[[I:.*]]) in (4) shared_outs(%[[O:.*]] = %[[INIT]], %[[O2:.*]] = %[[INIT2]])
// CHECK: %[[OFF:.*]] = affine.apply
// CHECK: tensor.extract_slice %[[O]][%[[OFF]], 0] [16, 1] [1, 1]
// CHECK: tensor.extract_slice %[[INIT2]][%{{.*}}, 0] [16, 1] [1, 1]
// CHECK: tensor.parallel_insert_slice %{{.*}} into %[[O]][%[[OFF]], 0] [16, 1] [1, 1]
// CHECK: tensor.parallel_insert_slice %{{.*}} into %[[O2]][%[[OFF]], 0] [16, 1] [1, 1]
func.func @two_shared_outs(%init: tensor<64x1xf32>, %init2: tensor<64x1xf32>) -> (tensor<64x1xf32>, tensor<64x1xf32>) {
  %r:2 = scf.forall (%i) in (4) shared_outs(%o = %init, %o2 = %init2) -> (tensor<64x1xf32>, tensor<64x1xf32>) {
    %off = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)
    %noff = affine.apply affine_map<(d0) -> ((d0 * 16 + 16) mod 64)>(%i)
    %old = tensor.extract_slice %init[%off, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    %next = tensor.extract_slice %init2[%noff, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    %new = arith.addf %old, %next : tensor<16x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %new into %o[%off, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
      tensor.parallel_insert_slice %next into %o2[%off, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
    }
  }
  return %r#0, %r#1 : tensor<64x1xf32>, tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The read of the iteration's own tile sits in a loop nested in the forall
// body: it moves to the shared output all the same.

// CHECK-LABEL: func.func @read_in_nested_loop
// CHECK-SAME: (%[[INIT:.*]]: tensor<64x1xf32>
// CHECK: scf.forall (%[[I:.*]]) in (4) shared_outs(%[[O:.*]] = %[[INIT]])
// CHECK: %[[OFF:.*]] = affine.apply
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[O]][%[[OFF]], 0] [16, 1] [1, 1]
// CHECK: tensor.parallel_insert_slice %{{.*}} into %[[O]][%[[OFF]], 0] [16, 1] [1, 1]
func.func @read_in_nested_loop(%init: tensor<64x1xf32>) -> tensor<64x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %r = scf.forall (%i) in (4) shared_outs(%o = %init) -> (tensor<64x1xf32>) {
    %off = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)
    %e = tensor.empty() : tensor<16x1xf32>
    %acc = scf.for %k = %c0 to %c3 step %c1 iter_args(%a = %e) -> tensor<16x1xf32> {
      %old = tensor.extract_slice %init[%off, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
      %s = arith.addf %a, %old : tensor<16x1xf32>
      scf.yield %s : tensor<16x1xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %acc into %o[%off, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
    }
  }
  return %r : tensor<64x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The read has the tile's offsets but not its sizes: it takes more than the
// iteration writes, so it stays on %init.

// CHECK-LABEL: func.func @other_sizes
// CHECK-SAME: (%[[INIT:.*]]: tensor<64x1xf32>
// CHECK: scf.forall
// CHECK: tensor.extract_slice %[[INIT]][%{{.*}}, 0] [32, 1] [1, 1]
func.func @other_sizes(%init: tensor<64x1xf32>) -> (tensor<64x1xf32>, tensor<32x1xf32>) {
  %e = tensor.empty() : tensor<32x1xf32>
  %r:2 = scf.forall (%i) in (2) shared_outs(%o = %init, %w = %e) -> (tensor<64x1xf32>, tensor<32x1xf32>) {
    %off = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)
    %wide = tensor.extract_slice %init[%off, 0] [32, 1] [1, 1] : tensor<64x1xf32> to tensor<32x1xf32>
    %old = tensor.extract_slice %wide[0, 0] [16, 1] [1, 1] : tensor<32x1xf32> to tensor<16x1xf32>
    %new = arith.addf %old, %old : tensor<16x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %new into %o[%off, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
      tensor.parallel_insert_slice %wide into %w[0, 0] [32, 1] [1, 1] : tensor<32x1xf32> into tensor<32x1xf32>
    }
  }
  return %r#0, %r#1 : tensor<64x1xf32>, tensor<32x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A read of %init outside the forall is not a tile of any iteration: it
// stays. Inside, the iteration's own tile moves to %o; the target is the
// function.

// CHECK-LABEL: func.func @outside_the_forall
// CHECK-SAME: (%[[INIT:.*]]: tensor<64x1xf32>
// CHECK: %[[BEFORE:.*]] = tensor.extract_slice %[[INIT]][0, 0] [16, 1] [1, 1]
// CHECK: scf.forall (%{{.*}}) in (4) shared_outs(%[[O:.*]] = %[[INIT]])
// CHECK: tensor.extract_slice %[[O]]
func.func @outside_the_forall(%init: tensor<64x1xf32>) -> (tensor<64x1xf32>, tensor<16x1xf32>) {
  %before = tensor.extract_slice %init[0, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
  %r = scf.forall (%i) in (4) shared_outs(%o = %init) -> (tensor<64x1xf32>) {
    %off = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)
    %old = tensor.extract_slice %init[%off, 0] [16, 1] [1, 1] : tensor<64x1xf32> to tensor<16x1xf32>
    %new = arith.addf %old, %old : tensor<16x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %new into %o[%off, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<64x1xf32>
    }
  }
  return %r, %before : tensor<64x1xf32>, tensor<16x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_extract_slice_of_insert_slice %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
