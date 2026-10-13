# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Matrix multiplication with a ragged K, written the way a Triton GEMM is.

A Triton GEMM walks K in BLOCK_K steps, masks the loads of the last, partial
step to zero, and accumulates with ``acc = tl.dot(a, b, acc)``. Here the K steps
run on a column of cores, one step per core, and the accumulator is passed down
the column over the cascade:

    ty == n-1   acc = 0                  first K step
    ty <  n-1   acc = cascade from ty+1
    every core  acc += A[:, step] @ B[step, :]
    ty >  0     cascade acc to ty-1
    ty == 0     store acc to C           last K step

The last step's missing columns of A and rows of B are zeros added by the
memtile DMA on the way to L1 (``pad_after``), where Triton masks the load.

Each launch point computes one TILE_M x TILE_N block of C.

The memtile pads only the inner three dimensions of a transfer. B's blocked
walk has the K block axis among them. A's has it outermost, so A is walked with
the M block axis outside the K one and written to L1 in the order the matmul
reads it.
"""

import argparse
import math
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import Module

# aie2p matmul intrinsic, (m, k, n).
MM_M, MM_K, MM_N = 8, 8, 8


def build_module(m, n, k, tile_m, tile_n, block_k):
    n_steps = math.ceil(k / block_k)
    last = k - (n_steps - 1) * block_k
    # One K step per core of a column; npu2 has four core rows.
    assert 2 <= n_steps <= 4, f"K / BLOCK_K gives {n_steps} steps; need 2 to 4"
    assert m % tile_m == 0 and n % tile_n == 0
    assert tile_m % MM_M == 0 and tile_n % MM_N == 0 and block_k % MM_K == 0
    # The pad is counted in whole K blocks of the intrinsic.
    assert last % MM_K == 0, f"K ({k}) must be a multiple of {MM_K}"
    # The pad sits on the memtile BD's third dimension, which takes at most
    # 15 (16 is rejected by aie-rt).
    pad_blocks = (block_k - last) // MM_K
    assert pad_blocks <= 15, (
        f"the last K step has {last} of {block_k}; padding {pad_blocks} blocks "
        "of 8 is more than the memtile can add on that dimension (15)"
    )

    A = air.tensor([m, k], bf16)
    B = air.tensor([k, n], bf16)
    C = air.tensor([m, n], f32)

    cascade = air.channel("cascade", size=[1, n_steps - 1], channel_type="npu_cascade")

    with air.launch([range(m // tile_m), range(n // tile_n)], name="matmul") as launch:

        @launch.body
        def _(li, lj):
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    row, col = li * tile_m, lj * tile_n
                    l2_a = air.alloc([tile_m, k], bf16, scope=seg.private())
                    l2_b = air.alloc([k, tile_n], bf16, scope=seg.private())
                    l2_c = air.alloc([tile_m, tile_n], f32, scope=seg.private())
                    ops.load(l2_a, A[row : row + tile_m, :])
                    ops.load(l2_b, B[:, col : col + tile_n])

                    acc = air.alloc(
                        [
                            1,
                            n_steps,
                            tile_n // MM_N,
                            tile_m // MM_M,
                            MM_M,
                            MM_N,
                        ],
                        f32,
                        scope=seg.shared(),
                    )

                    with air.herd(
                        [range(1), range(n_steps)],
                        name="herd_0",
                        shape=(1, n_steps),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            l1_a = air.alloc(
                                [1, 1, block_k // MM_K, tile_m // MM_M, MM_M, MM_K],
                                bf16,
                                scope=h.private(),
                            )
                            l1_b = air.alloc(
                                [1, 1, tile_n // MM_N, block_k // MM_K, MM_K, MM_N],
                                bf16,
                                scope=h.private(),
                            )
                            k0 = (n_steps - 1 - ty) * block_k
                            # This core's slab of the accumulator.
                            my_acc = acc[tx, ty, :, :, :, :]

                            def load_step(width):
                                kb = width // MM_K
                                pad = (block_k - width) // MM_K
                                pads = {}
                                if pad:
                                    pads = dict(pad_after=[0, 0, 0, pad, 0, 0])
                                # A as [M/m, K/k, m, k], written to L1 as
                                # [K/k, M/m, m, k].
                                ops.load(
                                    l1_a.transpose(0, 1, 3, 2, 4, 5),
                                    l2_a[:, k0 : k0 + width]
                                    .reshape(1, 1, tile_m // MM_M, MM_M, kb, MM_K)
                                    .transpose(0, 1, 2, 4, 3, 5),
                                    **pads,
                                )
                                ops.load(
                                    l1_b,
                                    l2_b[k0 : k0 + width, :]
                                    .reshape(1, 1, kb, MM_K, tile_n // MM_N, MM_N)
                                    .transpose(0, 1, 4, 2, 3, 5),
                                    **pads,
                                )

                            if last == block_k:
                                load_step(block_k)
                            else:
                                with ops.branch(ty == 0) as ragged:
                                    load_step(last)
                                with ragged.otherwise():
                                    load_step(block_k)

                            with ops.branch(ty == n_steps - 1) as head:
                                ops.fill(acc, 0.0)
                            with head.otherwise():
                                cascade.get(my_acc, indices=[tx, ty])

                            ops.dot(l1_a, l1_b, acc=acc)

                            with ops.branch(ty == 0) as tail:
                                ops.store(
                                    my_acc.transpose(0, 1, 3, 4, 2, 5),
                                    l2_c,
                                )
                            with tail.otherwise():
                                cascade.put(my_acc, indices=[tx, ty - 1])

                    ops.store(l2_c, C[row : row + tile_m, col : col + tile_n])

    return launch


# Direct codegen for the blocked matmul, as in matrix_multiplication/bf16 for
# an f32 accumulator, with one herd.
TRANSFORM_IR = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %func_fold_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op

    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_most_matmul, %vec_loops:3 =
      transform.structured.tile_using_for %matmul tile_sizes [2, 2, 1, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
      transform.structured.tile_using_for %inner_most_matmul tile_sizes [1, 1, 0, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.loop.unroll %vec_loops_to_unroll#1 factor = 2 : !transform.any_op
    transform.loop.unroll %vec_loops_to_unroll#0 factor = 2 : !transform.any_op

    %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_most_fills, %vec_fill_loops:2 =
      transform.structured.tile_using_for %linalg_fills tile_sizes [0, 0, 1, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op

    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

    %func1_rematch = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func1_rematch : (!transform.any_op) -> !transform.any_op

    %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vectorized_herds_1 = transform.air.herd_vectorize %herds_1 : (!transform.any_op) -> !transform.any_op
    %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %result11 = transform.air.vector_type_cast %vector_contracts <{target_element_type = f32, input_indices = [2], output_indices = [0]}> : (!transform.any_op) -> !transform.any_op

    %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %vectorized_herds_1 : (!transform.any_op) -> !transform.any_op
    %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 overflow_result = 1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %vectorized_herds_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op
    %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
    %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    %func_fold_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_folded_3 = transform.air.fold_unit_extent_dims %func_fold_3 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests a bf16 matmul whose K is not a "
        "multiple of the K step",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=128, help="M dimension")
    parser.add_argument("--n", type=int, default=128, help="N dimension")
    parser.add_argument("--k", type=int, default=960, help="K dimension")
    parser.add_argument(
        "--tile-m", type=int, default=32, dest="tile_m", help="M size of a C tile"
    )
    parser.add_argument(
        "--tile-n", type=int, default=32, dest="tile_n", help="N size of a C tile"
    )
    parser.add_argument(
        "--block-k",
        type=int,
        default=256,
        dest="block_k",
        help="K step, one per core of the column",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-xclbin", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="compile-only (no XRT, no xclbin), compile-and-xclbin (requires "
        "XRT, generates xclbin), or compile-and-run (requires XRT, generates "
        "xclbin and runs)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2p",
        help="Target AIE architecture (aie2p only)",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after 10 warmup) and "
        "print Latency + GFLOPs in addition to the correctness check",
    )
    args = parser.parse_args()

    if args.arch != "aie2p":
        print(
            f"Error: --arch {args.arch} is not supported by this example.",
            file=sys.stderr,
        )
        print(
            "Its matmul blocks and transform script are written for the aie2p "
            "8x8x8 bf16 intrinsic. Re-run with --arch aie2p on an NPU2 device.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.compile_mode != "compile-only" and not os.environ.get("PEANO_INSTALL_DIR"):
        print(
            "Error: PEANO_INSTALL_DIR environment variable is not set.",
            file=sys.stderr,
        )
        print("Peano is needed for direct code generation.", file=sys.stderr)
        sys.exit(1)

    launch = build_module(
        args.m, args.n, args.k, args.tile_m, args.tile_n, args.block_k
    )
    mlir_module = launch.build(target="npu2")
    run_transform(Module.parse(TRANSFORM_IR, context=mlir_module.context), mlir_module)
    if args.print_module_only:
        print(mlir_module)
        return 0

    backend_kwargs = {
        "verbose": args.verbose,
        "omit_while_true_loop": False,
        "runtime_loop_tiling_sizes": [2, 2],
        "stack_size": 2048,
    }

    if args.compile_mode == "compile-and-run":
        # randn / sqrt(K), as in the bf16 example, so the output has unit
        # variance whatever K is.
        np.random.seed(42)
        scale = 1.0 / math.sqrt(args.k)
        a = (np.random.randn(args.m, args.k) * scale).astype(bfloat16)
        b = (np.random.randn(args.k, args.n) * scale).astype(bfloat16)
        reference = a.astype(np.float32) @ b.astype(np.float32)
        runner = XRTRunner(
            **backend_kwargs,
            instance_name="matmul_bf16_ragged_k",
            report_precision=True,
            n_perf_iters=args.perf_iters,
            perf_flops=(
                (2.0 * args.m * args.k * args.n) if args.perf_iters > 0 else None
            ),
        )
        return runner.run_test(
            mlir_module,
            inputs=[a, b],
            expected_outputs=[reference],
            rtol=2e-3,
            atol=2e-3,
        )

    if args.compile_mode == "compile-only":
        backend_kwargs.update(target_device="npu2", output_format="none")
    backend = XRTBackend(**backend_kwargs)
    backend.compile(mlir_module)
    backend.unload()
    if args.compile_mode == "compile-only":
        print("Compilation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
