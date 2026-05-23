// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Drives the C++ air-matmul-codegen orchestrator through the transform
// dialect. The matmul-specific tile/pack/bufferize/vectorize work is
// delegated to the orchestrator; the transform script keeps the
// non-matmul plumbing (scf.forall->herd, herd-vectorize, cleanup).

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {

    // Phase 1: matmul codegen orchestrator (pre-vectorize half).
    %m1 = transform.apply_registered_pass "air-matmul-codegen" with options = {
        "bufferize-output-l2" = true,
        "tile-l3-to-l2-copies" = true, "k-l2-tile" = 64,
        "l2-pack-sizes" = [8, 8, 8],
        "l2-lhs-outer-perm" = [1, 0], "l2-lhs-inner-perm" = [0, 1],
        "l2-rhs-outer-perm" = [1, 0], "l2-rhs-inner-perm" = [1, 0],
        "l2-acc-outer-perm" = [1, 0], "l2-acc-inner-perm" = [0, 1],
        "outer-k-tile-factor" = 8, "outer-k-iter-index" = 2,
        "core-tile" = [8, 8, 0],
        "prologue-tile" = [8, 8], "epilogue-tile" = [64, 64],
        "fill-iter-perm" = [1, 0, 2, 3],
        "one-shot-bufferize" = true,
        "post-bufferize-cleanup-first" = true,
        "matmul-vec-tile" = [2, 2, 1, 0, 0, 0],
        "matmul-unroll-vec-tile" = [1, 1, 0, 0, 0, 0],
        "matmul-unroll-factor" = 2,
        "fill-vec-tile" = [1, 1, 0, 0]
    } to %arg1 : (!transform.any_op) -> !transform.any_op

    // Phase 2: scf.forall -> scf.parallel -> air.herd, then vectorize herds.
    %func1 = transform.structured.match ops{["func.func"]} in %m1
        : (!transform.any_op) -> !transform.any_op
    %m2 = transform.apply_registered_pass "scf-forall-to-parallel" to %func1
        : (!transform.any_op) -> !transform.any_op
    %m3 = transform.apply_registered_pass "air-par-to-herd" to %m1
        : (!transform.any_op) -> !transform.any_op
    %func2 = transform.structured.match ops{["func.func"]} in %m3
        : (!transform.any_op) -> !transform.any_op
    %m4 = transform.apply_registered_pass "air-herd-vectorize" to %func2
        : (!transform.any_op) -> !transform.any_op

    // Cleanup between vectorize and vec-prep.
    %func3a = transform.structured.match ops{["func.func"]} in %m3

        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "canonicalize" to %func3a

        : (!transform.any_op) -> !transform.any_op

    %func3b = transform.structured.match ops{["func.func"]} in %m3

        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "cse" to %func3b

        : (!transform.any_op) -> !transform.any_op

    %func3c = transform.structured.match ops{["func.func"]} in %m3

        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "fold-memref-alias-ops" to %func3c

        : (!transform.any_op) -> !transform.any_op

    // Phase 3: matmul codegen orchestrator (vec-prep half).
    %m5 = transform.apply_registered_pass "air-matmul-codegen" with options = {
        "vec-prep-cast1-target-element-type" = "f32",
        "vec-prep-cast1-input-indices" = [2],
        "vec-prep-cast1-output-indices" = [0]
    } to %m3 : (!transform.any_op) -> !transform.any_op

    // Final cleanup.
    %func4a = transform.structured.match ops{["func.func"]} in %m5

        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "canonicalize" to %func4a

        : (!transform.any_op) -> !transform.any_op

    %func4b = transform.structured.match ops{["func.func"]} in %m5

        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "cse" to %func4b

        : (!transform.any_op) -> !transform.any_op

    %func4c = transform.structured.match ops{["func.func"]} in %m5

        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "fold-memref-alias-ops" to %func4c

        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
