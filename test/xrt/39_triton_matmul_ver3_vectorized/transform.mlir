// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// AIE2 (NPU1) single-pack matmul codegen via the C++ air-matmul-codegen
// orchestrator. mmul=4x4x8, launch-tile=64x64. No L3->L2 copy tiling,
// no fuse-output-truncf (output is f32), no prologue/epilogue tiling.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {

    %m1 = transform.apply_registered_pass "air-matmul-codegen" with options = {
        "launch-tile" = [64, 64],
        "bufferize-output-l2" = true,
        "l2-pack-sizes" = [4, 4, 8],
        "l2-lhs-outer-perm" = [1, 0],
        "l2-rhs-outer-perm" = [1, 0], "l2-rhs-inner-perm" = [1, 0],
        "l2-acc-outer-perm" = [1, 0],
        "outer-k-tile-factor" = 4, "outer-k-iter-index" = 2,
        "one-shot-bufferize" = true,
        "matmul-vec-tile" = [1, 1, 1, 0, 0, 0],
        "matmul-unroll-factor" = 1,
        "fill-vec-tile" = [1, 1]
    } to %arg1 : (!transform.any_op) -> !transform.any_op

    %func1 = transform.structured.match ops{["func.func"]} in %m1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "scf-forall-to-parallel" to %func1
        : (!transform.any_op) -> !transform.any_op
    %m2 = transform.apply_registered_pass "air-par-to-herd" to %m1
        : (!transform.any_op) -> !transform.any_op
    %func2 = transform.structured.match ops{["func.func"]} in %m2
        : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "air-herd-vectorize" to %func2
        : (!transform.any_op) -> !transform.any_op

    %func3a = transform.structured.match ops{["func.func"]} in %m2
        : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "canonicalize" to %func3a
        : (!transform.any_op) -> !transform.any_op
    %func3b = transform.structured.match ops{["func.func"]} in %m2
        : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "cse" to %func3b
        : (!transform.any_op) -> !transform.any_op
    %func3c = transform.structured.match ops{["func.func"]} in %m2
        : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "fold-memref-alias-ops" to %func3c
        : (!transform.any_op) -> !transform.any_op

    // Final cleanup orchestrator pass (Phase 0 unit-extent fold + Phase N
    // vec-prep no-ops on already-cleaned IR).
    transform.apply_registered_pass "air-matmul-codegen" to %m2
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
