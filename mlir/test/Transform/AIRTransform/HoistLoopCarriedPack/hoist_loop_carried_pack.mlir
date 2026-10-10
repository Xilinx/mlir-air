//===- hoist_loop_carried_pack.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// A K loop whose accumulator is packed and unpacked every iteration carries
// the packed value instead: one pack before, one unpack after.

// CHECK-LABEL: func.func @accumulator
// CHECK-SAME: (%[[INIT:.*]]: tensor<16x16xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x2x8x8xf32>
// CHECK: %[[P0:.*]] = linalg.pack %[[INIT]] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %[[EMPTY]]
// CHECK: %[[R:.*]]:2 = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[PACKED:.*]] = %[[P0]])
// CHECK-NOT: linalg.pack
// CHECK: %[[X:.*]] = math.exp %[[PACKED]]
// CHECK-NOT: linalg.unpack
// CHECK: scf.yield %[[ACC]], %[[X]]
// CHECK: %[[U:.*]] = linalg.unpack %[[R]]#1 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %[[R]]#0
// CHECK: return %[[U]]
func.func @accumulator(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %x = math.exp %p : tensor<2x2x8x8xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    scf.yield %u : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Other loop-carried values pass through untouched; the pack's destination
// may be defined outside the loop.

// CHECK-LABEL: func.func @with_offset
// CHECK: %[[P0:.*]] = linalg.pack %{{.*}} into %{{.*}}
// CHECK: %[[R:.*]]:3 = scf.for {{.*}} iter_args(%[[OFF:.*]] = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %[[P0]])
// CHECK: %[[NEXT:.*]] = arith.addi %[[OFF]]
// CHECK: scf.yield %[[NEXT]],
// CHECK: linalg.unpack %[[R]]#2
func.func @with_offset(%init: tensor<16x16xf32>, %dest: tensor<2x2x8x8xf32>, %off0: index) -> (tensor<16x16xf32>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%off = %off0, %acc = %init) -> (index, tensor<16x16xf32>) {
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %x = math.exp %p : tensor<2x2x8x8xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    %next = arith.addi %off, %c1 : index
    scf.yield %next, %u : index, tensor<16x16xf32>
  }
  return %r#1, %r#0 : tensor<16x16xf32>, index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The accumulator is also read directly in the loop, so it is not only
// packed and unpacked: nothing to hoist.

func.func @other_use(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %u = linalg.unpack %p inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    %s = arith.addf %u, %acc : tensor<16x16xf32>
    scf.yield %s : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no iteration argument is packed and unpacked in the loop}}
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The pack and the unpack disagree on the layout.

func.func @layout_mismatch(%init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xf32> -> tensor<2x2x8x8xf32>
    %u = linalg.unpack %p outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<16x16xf32>
    scf.yield %u : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no iteration argument is packed and unpacked in the loop}}
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The pack pads. Carrying the padded value would keep whatever the loop
// writes into the padding, which repacking every iteration resets.

func.func @padded(%init: tensor<15x15xf32>) -> tensor<15x15xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<15x15xf32> {
    %e = tensor.empty() : tensor<2x2x8x8xf32>
    %p = linalg.pack %acc padding_value(%zero : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<15x15xf32> -> tensor<2x2x8x8xf32>
    %x = math.exp %p : tensor<2x2x8x8xf32>
    %u = linalg.unpack %x inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %acc : tensor<2x2x8x8xf32> -> tensor<15x15xf32>
    scf.yield %u : tensor<15x15xf32>
  }
  return %r : tensor<15x15xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{no iteration argument is packed and unpacked in the loop}}
    %new, %packs = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
