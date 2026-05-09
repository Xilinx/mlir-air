// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// AIE2P (Strix) two-pack-level matmul codegen via the C++
// air-matmul-codegen orchestrator. M=512 N=512 K=1024.
// Per-launch matmul: 256x256x1024.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {

    transform.apply_registered_pass "air-matmul-codegen" with options = {
        "launch-tile" = [256, 256],
        "l2-pack-sizes" = [64, 64, 64],
        "l2-lhs-outer-perm" = [0, 1], "l2-lhs-inner-perm" = [0, 1],
        "l2-rhs-outer-perm" = [1, 0], "l2-rhs-inner-perm" = [1, 0],
        "l2-acc-outer-perm" = [0, 1], "l2-acc-inner-perm" = [0, 1],
        "bufferize-output-l2" = true,
        "l1-pack-sizes" = [0, 0, 0, 8, 8, 8],
        "l1-lhs-outer-perm" = [0, 1, 3, 2],
        "l1-rhs-outer-perm" = [0, 1, 3, 2], "l1-rhs-inner-perm" = [1, 0],
        "l1-acc-outer-perm" = [0, 1, 3, 2],
        "outer-k-tile-factor" = 1, "outer-k-iter-index" = 2,
        "core-tile" = [1, 1, 0, 0, 0, 0, 0, 0, 0],
        "inner-k-tile-factor" = 8, "inner-k-iter-index" = 5,
        "prologue-tile" = [1, 1], "epilogue-tile" = [1, 1],
        "hoist-static-alloc-first" = true,
        "one-shot-bufferize" = true,
        "post-bufferize-cleanup-first" = true,
        "matmul-vec-tile" = [1, 1, 1, 1, 1, 1, 0, 0, 0],
        "matmul-unroll-vec-tile" = [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "matmul-unroll-factor" = 1,
        "fill-vec-tile" = [1, 1, 1, 1]
    } to %arg1 : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
