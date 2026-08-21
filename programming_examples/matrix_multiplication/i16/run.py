# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tiled int16 matrix multiplication on air.api.

The schedule is unchanged from the raw-bindings version this replaces, because
it is the schedule the AIE2 matmul intrinsic and the transform script both
expect:

    air.launch (m/tile_m/herd_m, n/tile_n/herd_n)   one segment per output tile
      air.segment
        L2: flat staging buffers, each exactly the L3 region it holds
        L1: an accumulator shared by the herd, micro-tiled -- seg.shared()
        herd            zero the accumulator
        for k2 in air.sequential(0, k, tile_k_l2)
            L3 -> L2 for A and B
            herd
                L1: micro-tiled A and B tiles
                for k1 in air.sequential(0, tile_k_l2, tile_k_l1)
                    L2 -> L1, then acc += a @ b
        herd            drain the accumulator to L2
        L2 -> L3

One layout does the work, and it is not a memref layout attribute -- it is
carried by the *shape*, with the reordering derived into the DMA access pattern:

  * ``air.micro_tile(m, k, n)`` gives the L1 tiles their micro-blocked shape,
    ``[1, 1, K/k, M/m, m, k]`` and friends. This is what makes the contraction
    expressible as the 6-D ``block_matmul`` the intrinsic wants, and what
    ``mm.cc`` was compiled against (``-DDIM_M``/``-DDIM_K``/``-DDIM_N``).
  * The L2 staging buffers are *flat* -- each is exactly the L3 region it
    holds. Filling and draining them are then plain shape-matching transfers,
    and a core slices its own sub-region out with ordinary arithmetic on its
    tile coordinate.

Both lowering routes are supported, exactly as before and chosen by the same
flag. ``--direct-codegen`` runs the transform script below on the built module,
vectorising the contraction in place; without it, ``lower_linalg_to_func`` swaps
the contraction for a call into ``mm.o``. The DSL needs no special support for
either: ``launch.build()`` hands back the module for the script to rewrite, and
the backend kwargs reach ``XRTBackend`` untouched.
"""

import argparse
import os
import sys

from air.ir import Module
from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend
from air.compiler.util import run_transform

from air import api as air
from air.api import i16

import numpy as np

np.random.seed(42)

# The AIE2 matmul intrinsic's operand shape, per architecture. The int16 MAC is
# 4x4x4 on aie2 and 8x2x8 on aie2p.
MMUL_MKN = {"aie2": (4, 4, 4), "aie2p": (8, 2, 8)}

# air.api dtypes, keyed by the numpy dtype the harness works in.
DTYPE = {np.int16: i16}


def build_module(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m,
    herd_n,
    np_dtype_in,
    np_dtype_out,
    arch="aie2",
):
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0

    dt_in, dt_out = DTYPE[np_dtype_in], DTYPE[np_dtype_out]
    mm = air.micro_tile(*MMUL_MKN[arch])

    # One segment covers herd_m x herd_n output tiles; the launch grid covers
    # the rest of M and N. The outer tiling has to sit here rather than in the
    # herd because the L2 staging buffers are refilled per launch point.
    seg_m, seg_n = tile_m * herd_m, tile_n * herd_n

    A = air.tensor([m, k], dt_in)
    B = air.tensor([k, n], dt_in)
    C = air.tensor([m, n], dt_out)

    # Unlike the elementwise conversions, the target is *pinned* from --arch by
    # the caller (see launch.build below). It has to be: --arch already selects
    # the micro-tile, so the module is architecture-specific before the herd is
    # even sized, and letting the two disagree would build for one generation
    # while shaped for the other.
    with air.launch(
        [range(0, m, seg_m), range(0, n, seg_n)], name="matmul_i16"
    ) as launch:

        @launch.body
        def _(si, sj):
            with air.segment(name="matmul_seg") as seg:

                @seg.body
                def _():
                    row, col = si * seg_m, sj * seg_n

                    # L2 staging is flat: each buffer is exactly the region of
                    # L3 it holds, so filling and draining it are plain
                    # shape-matching transfers and each core slices its own
                    # sub-region out. For A this is also byte-identical to the
                    # predecessor's [herd_m, 1, tile_m, tile_k_l2] -- row-major
                    # [a, 1, b, c] and [a*b, c] are the same buffer.
                    l2_a = air.alloc([seg_m, tile_k_l2], dt_in, scope=seg.private())
                    l2_b = air.alloc([tile_k_l2, seg_n], dt_in, scope=seg.private())
                    l2_c = air.alloc([seg_m, seg_n], dt_out, scope=seg.private())
                    # The accumulator outlives each entry into the compute herd,
                    # because the k2 reduction is at segment scope, so it is
                    # allocated here and carries one slab per core.
                    acc = air.alloc(
                        mm.c(tile_m, tile_n, lead=(herd_m, herd_n)),
                        dt_out,
                        scope=seg.shared(),
                    )

                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as zero_herd:

                        @zero_herd.body
                        def _(tx, ty):
                            acc[:] = 0

                    for k2 in air.sequential(0, k, tile_k_l2):
                        air.ops.load(l2_a, A[row : row + seg_m, k2 : k2 + tile_k_l2])
                        air.ops.load(l2_b, B[k2 : k2 + tile_k_l2, col : col + seg_n])

                        with air.herd(
                            [range(herd_m), range(herd_n)],
                            name="herd_0",
                            shape=(herd_m, herd_n),
                        ) as h:

                            @h.body
                            def _(tx, ty):
                                l1_a = air.alloc(
                                    mm.a(tile_m, tile_k_l1), dt_in, scope=h.private()
                                )
                                l1_b = air.alloc(
                                    mm.b(tile_k_l1, tile_n), dt_in, scope=h.private()
                                )
                                for k1 in air.sequential(0, tile_k_l2, tile_k_l1):
                                    air.ops.load(
                                        l1_a,
                                        l2_a[
                                            tx * tile_m : tx * tile_m + tile_m,
                                            k1 : k1 + tile_k_l1,
                                        ],
                                    )
                                    air.ops.load(
                                        l1_b,
                                        l2_b[
                                            k1 : k1 + tile_k_l1,
                                            ty * tile_n : ty * tile_n + tile_n,
                                        ],
                                    )
                                    air.ops.dot(l1_a, l1_b, acc=acc)

                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as drain_herd:

                        @drain_herd.body
                        def _(tx, ty):
                            air.ops.store(
                                acc[tx, ty, :, :],
                                l2_c[
                                    tx * tile_m : tx * tile_m + tile_m,
                                    ty * tile_n : ty * tile_n + tile_n,
                                ],
                            )

                    air.ops.store(l2_c, C[row : row + seg_m, col : col + seg_n])

    return launch


if __name__ == "__main__":
    # Default values.
    M = 512
    K = 512
    N = 512
    TILE_M = 64
    TILE_K_L2 = 128
    TILE_K_L1 = 64
    TILE_N = 64
    HERD_M = 4
    HERD_N = 4
    INPUT_DATATYPE = np.int16
    OUTPUT_DATATYPE = np.int16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--m", type=int, default=M, help="M dimension size in a (MxK) * (KxN) matmul"
    )
    parser.add_argument(
        "--k", type=int, default=K, help="K dimension size in a (MxK) * (KxN) matmul"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="N dimension size in a (MxK) * (KxN) matmul",
    )
    parser.add_argument(
        "--tile-m", type=int, default=TILE_M, help="M dimension size of each L1 tile"
    )
    parser.add_argument(
        "--tile-k-l2",
        type=int,
        default=TILE_K_L2,
        help="K dimension size of each L2 tile",
    )
    parser.add_argument(
        "--tile-k-l1",
        type=int,
        default=TILE_K_L1,
        help="K dimension size of each L1 tile",
    )
    parser.add_argument(
        "--tile-n", type=int, default=TILE_N, help="N dimension size of each L1 tile"
    )
    parser.add_argument(
        "--herd-m",
        type=int,
        default=HERD_M,
        help="Number of L1 tiles along the M dimension",
    )
    parser.add_argument(
        "--herd-n",
        type=int,
        default=HERD_N,
        help="Number of L1 tiles along the N dimension",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-xclbin", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure compilation mode: compile-only (no XRT, no xclbin), compile-and-xclbin (requires XRT, generates xclbin), or compile-and-run (requires XRT, generates xclbin and runs)",
    )
    parser.add_argument(
        "--direct-codegen",
        action="store_true",
        help="Enable direct code generation mode (compiles directly without extra kernel library)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2",
        help="Target AIE architecture (aie2 or aie2p)",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after 10 warmup) and "
        "print Latency (us) in addition to the correctness check",
    )
    args = parser.parse_args()

    # Check for PEANO_INSTALL_DIR if direct codegen is enabled
    if args.direct_codegen:
        if not os.environ.get("PEANO_INSTALL_DIR"):
            print(
                "Error: PEANO_INSTALL_DIR environment variable is not set.",
                file=sys.stderr,
            )
            print("Peano is needed for direct code generation mode.", file=sys.stderr)
            sys.exit(1)

    launch = build_module(
        args.m,
        args.k,
        args.n,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
        args.arch,
    )

    # --arch already fixes the micro-tile, so the module is architecture-specific
    # either way; pinning the target here keeps the herd sized for the part it
    # will be compiled for.
    mlir_module = launch.build(target="npu2" if args.arch == "aie2p" else "npu1")

    # Vectorization - only run if direct codegen mode is enabled
    if args.direct_codegen:
        # Architecture-specific accumulator type for vector intrinsics
        # aie2: 4x4x4 matrix multiply produces i64 accumulator
        # aie2p: 8x2x8 matrix multiply produces i32 accumulator
        # This is due to different vector intrinsic shapes between architectures
        vector_acc_type = "i64" if args.arch == "aie2" else "i32"

        transform_ir_string = f"""
            module attributes {{transform.with_named_sequence}} {{
              transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{

                %func0 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                transform.apply_patterns to %func0 {{
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                }} : !transform.any_op
                %func_fold_1 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op


                %matmul = transform.structured.match ops{{["linalg.generic"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %inner_most_matmul, %vec_loops:3 =
                  transform.structured.tile_using_for %matmul tile_sizes [2, 2, 1, 0, 0, 0]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
                %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
                  transform.structured.tile_using_for %inner_most_matmul tile_sizes [1, 1, 0, 0, 0, 0]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                transform.loop.unroll %vec_loops_to_unroll#1 {{factor = 2}} : !transform.any_op
                transform.loop.unroll %vec_loops_to_unroll#0 {{factor = 2}} : !transform.any_op

                %linalg_fills = transform.structured.match ops{{["linalg.fill"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                %inner_most_fills, %vec_fill_loops:2 =
                  transform.structured.tile_using_for %linalg_fills tile_sizes [0, 0, 1, 1]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)


                %herds = transform.structured.match ops{{["air.herd"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op

                %herd1, %herd2, %herd3 = transform.split_handle %vectorized_herds : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                %scf_fors = transform.structured.match ops{{["scf.for"]}} in %herd2 : (!transform.any_op) -> !transform.any_op

                %func1 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                transform.apply_patterns to %func1 {{
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                    transform.apply_patterns.memref.fold_memref_alias_ops
                }} : !transform.any_op
                %func_fold_2 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

                // Eliminate redundant vector.transfer_read operations
                %func1_rematch = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func1_rematch : (!transform.any_op) -> !transform.any_op

                // Hoist loop-invariant vector transfers out of innermost loop
                %herds_1 = transform.structured.match ops{{["air.herd"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %herds_1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

                %scf_fors_1 = transform.structured.match ops{{["scf.for"]}} in %herd2_1 : (!transform.any_op) -> !transform.any_op
                %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {{overflow_result = 1}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

                %vector_contracts = transform.structured.match ops{{["vector.contract"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                %result11 = transform.air.vector_type_cast %vector_contracts {{target_element_type = {vector_acc_type}, input_indices = [2], output_indices = [0]}} : (!transform.any_op) -> !transform.any_op

                // Hoist all accumulator transfer pairs from the innermost loop
                %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op
                %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
                %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

                %fors_to_hoist_ptrs = transform.structured.match ops{{["scf.for"]}} in %herd2_1 : (!transform.any_op) -> !transform.any_op
                %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist_ptrs {{overflow_result = 1}}: (!transform.any_op) -> (!transform.any_op, !transform.any_op)

                // Hoist the 4 extsi/trunci pairs from the innermost loop
                %all_extsi_loop = transform.structured.match ops{{["arith.extsi"]}} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
                %all_trunci_loop = transform.structured.match ops{{["arith.trunci"]}} in %innermost_for1 : (!transform.any_op) -> !transform.any_op

                // Split to get individual operations
                %extsi_i16_1, %extsi_i16_2, %extsi_i16_3, %extsi_i16_4 = transform.split_handle %all_extsi_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

                // The 4 trunci ops correspond to the 4 vector.contract results
                %trunci_1, %trunci_2, %trunci_3, %trunci_4 = transform.split_handle %all_trunci_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

                // Hoist first pair (arg29 - index 2)
                %for1_1_hoisted_1 = transform.air.hoist_cast_pair %extsi_i16_1, %trunci_1, %innermost_for1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
                %all_extsi_loop_2 = transform.structured.match ops{{["arith.extsi"]}} in %for1_1_hoisted_1 : (!transform.any_op) -> !transform.any_op
                %all_trunci_loop_2 = transform.structured.match ops{{["arith.trunci"]}} in %for1_1_hoisted_1 : (!transform.any_op) -> !transform.any_op
                %extsi_i16_2_new, %e2_5, %e2_6 = transform.split_handle %all_extsi_loop_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                %trunci_2_1, %trunci_2_2, %trunci_2_3 = transform.split_handle %all_trunci_loop_2 {{num_result_handles = 3}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                %for1_1_hoisted_2 = transform.air.hoist_cast_pair %extsi_i16_2_new, %trunci_2_1, %for1_1_hoisted_1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

                // Re-match and hoist third pair
                %all_extsi_loop_3 = transform.structured.match ops{{["arith.extsi"]}} in %for1_1_hoisted_2 : (!transform.any_op) -> !transform.any_op
                %all_trunci_loop_3 = transform.structured.match ops{{["arith.trunci"]}} in %for1_1_hoisted_2 : (!transform.any_op) -> !transform.any_op
                %extsi_i16_3_new, %e3_7 = transform.split_handle %all_extsi_loop_3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
                %trunci_3_1, %trunci_3_2 = transform.split_handle %all_trunci_loop_3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
                %for1_1_hoisted_3 = transform.air.hoist_cast_pair %extsi_i16_3_new, %trunci_3_1, %for1_1_hoisted_2 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

                // Re-match and hoist fourth pair
                %all_extsi_loop_4 = transform.structured.match ops{{["arith.extsi"]}} in %for1_1_hoisted_3 : (!transform.any_op) -> !transform.any_op
                %all_trunci_loop_4 = transform.structured.match ops{{["arith.trunci"]}} in %for1_1_hoisted_3 : (!transform.any_op) -> !transform.any_op
                %for1_1_hoisted_final = transform.air.hoist_cast_pair %all_extsi_loop_4, %all_trunci_loop_4, %for1_1_hoisted_3 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

                %func2 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                transform.apply_patterns to %func2 {{
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                    transform.apply_patterns.memref.fold_memref_alias_ops
                }} : !transform.any_op
                %func_fold_3 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func_folded_3 = transform.air.fold_unit_extent_dims %func_fold_3 : (!transform.any_op) -> !transform.any_op

              transform.yield
            }}
            }}
        """
        transform_ir = Module.parse(transform_ir_string, context=mlir_module.context)
        run_transform(transform_ir, mlir_module)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, args.m * args.k, dtype=np.int64).reshape(args.m, args.k) % 7
    input_a = input_a.astype(INPUT_DATATYPE)
    input_b = np.arange(0, args.k * args.n, dtype=np.int64).reshape(args.k, args.n) % 7
    input_b = input_b.astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        # Full reference, every element. The predecessor sampled 100 (i, j) pairs
        # and summed each in OUTPUT_DATATYPE; a whole-array matmul in the same
        # dtype wraps identically -- verified element-for-element against the
        # sampled reference on every shape the Makefile builds -- so widening the
        # check to the whole output does not change what counts as correct.
        reference = input_a.astype(OUTPUT_DATATYPE) @ input_b.astype(OUTPUT_DATATYPE)

        ###### Compile and test
        # No tolerance is passed, and none applies: XRTRunner compares integer
        # outputs with np.array_equal, so the gate is bit-exact equality. That is
        # the same bar the predecessor's stochastic check ran under -- it took
        # the same exact-equality branch -- now over the whole output instead of
        # 100 sampled points of it.
        runner_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
            "n_perf_iters": args.perf_iters,
        }
        # Only use external kernel library if NOT in direct codegen mode
        if not args.direct_codegen:
            runner_kwargs["lower_linalg_to_func"] = "mm.o"

        runner = XRTRunner(**runner_kwargs, instance_name="matmul_i16")
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                expected_outputs=[reference],
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        ###### Compile and generate xclbin (requires XRT, no execution)
        backend_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
        }
        # Only use external kernel library if NOT in direct codegen mode
        if not args.direct_codegen:
            backend_kwargs["lower_linalg_to_func"] = "mm.o"

        backend = XRTBackend(**backend_kwargs)
        module_function = backend.compile(mlir_module)

        backend.unload()

    elif args.compile_mode == "compile-only":
        ###### Compile only (without XRT dependencies)
        # Map architecture to target device
        target_device = "npu2" if args.arch == "aie2p" else "npu1"

        backend_kwargs = {
            "verbose": args.verbose,
            "target_device": target_device,  # Explicit target based on arch (no xrt dependencies)
            "output_format": "none",  # Skip xclbin generation (no xrt dependencies)
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
        }
        # Only use external kernel library if NOT in direct codegen mode
        if not args.direct_codegen:
            backend_kwargs["lower_linalg_to_func"] = "mm.o"

        backend = XRTBackend(**backend_kwargs)
        module_function = backend.compile(mlir_module)

        backend.unload()

        print("Compilation completed successfully!")
        sys.exit(0)
