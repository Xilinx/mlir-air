# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tiled matrix multiplication on air.api.

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
import math
import os
import sys
from ml_dtypes import bfloat16

from air.ir import Module
from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend
from air.compiler.util import run_transform

from air import api as air
from air.api import bf16, f32

import numpy as np

np.random.seed(42)

# The AIE2 matmul intrinsic's operand shape, per architecture. aie2p emulates
# bf16 with BFP16 (-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 in the aie_api
# path), which is why its micro-tile is wider.
MMUL_MKN = {"aie2": (4, 8, 4), "aie2p": (8, 8, 8)}

# air.api dtypes, keyed by the numpy dtype the harness works in.
DTYPE = {bfloat16: bf16, np.float32: f32}


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
    direct_codegen=False,
):
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0

    dt_in, dt_out = DTYPE[np_dtype_in], DTYPE[np_dtype_out]
    mm = air.micro_tile(*MMUL_MKN[arch])

    # The extent of one L2 staging tile: a segment covers herd_m x herd_n
    # output tiles, and l2_a/l2_b/l2_c below are sized from it. The launch steps
    # by exactly that, one L2 tile per point, so the outer tiling sits on the
    # launch rather than the herd -- the staging buffers are refilled per point.
    # Named for what it dimensions, not for the hierarchy level that consumes
    # it: air.launch, air.segment and air.herd each own a separate iteration
    # space, and a launch point is not a synonym for a segment.
    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n

    A = air.tensor([m, k], dt_in)
    B = air.tensor([k, n], dt_in)
    C = air.tensor([m, n], dt_out)

    # Unlike the elementwise conversions, the target is *pinned* from --arch by
    # the caller (see launch.build below). It has to be: --arch already selects
    # the micro-tile, so the module is architecture-specific before the herd is
    # even sized, and letting the two disagree would build for one generation
    # while shaped for the other.
    with air.launch(
        [range(0, m, l2_m), range(0, n, l2_n)], name="matmul_bf16"
    ) as launch:

        @launch.body
        def _(li, lj):
            with air.segment(name="matmul_seg") as seg:

                @seg.body
                def _():
                    row, col = li * l2_m, lj * l2_n

                    # L2 staging is flat: each buffer is exactly the region of
                    # L3 it holds, so filling and draining it are plain
                    # shape-matching transfers and each core slices its own
                    # sub-region out. For A this is also byte-identical to the
                    # predecessor's [herd_m, 1, tile_m, tile_k_l2] -- row-major
                    # [a, 1, b, c] and [a*b, c] are the same buffer.
                    l2_a = air.alloc([l2_m, tile_k_l2], dt_in, scope=seg.private())
                    l2_b = air.alloc([tile_k_l2, l2_n], dt_in, scope=seg.private())
                    l2_c = air.alloc([l2_m, l2_n], dt_out, scope=seg.private())
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
                            acc[:] = 0.0

                    for k2 in air.sequential(0, k, tile_k_l2):
                        air.ops.load(l2_a, A[row : row + l2_m, k2 : k2 + tile_k_l2])
                        air.ops.load(l2_b, B[k2 : k2 + tile_k_l2, col : col + l2_n])

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

                    air.ops.store(l2_c, C[row : row + l2_m, col : col + l2_n])

    return launch


if __name__ == "__main__":
    # Default values.
    M = 512
    K = 512
    N = 512
    TILE_M = 128
    TILE_K_L2 = 128
    TILE_K_L1 = 32
    TILE_N = 64
    HERD_M = 4
    HERD_N = 4
    INPUT_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16

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
        "--output-dtype",
        type=str,
        choices=["bf16", "f32"],
        default=None,
        dest="output_dtype",
        help="Override output data type (default: bf16 for aie2, f32 for aie2p)",
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

    # aie2p defaults to f32 accumulation output for better precision.
    # Can be overridden with --output-dtype.
    if args.output_dtype == "bf16":
        pass  # keep default bfloat16
    elif args.output_dtype == "f32":
        OUTPUT_DATATYPE = np.float32
    elif args.arch == "aie2p":
        OUTPUT_DATATYPE = np.float32

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
        args.direct_codegen,
    )

    # --arch already fixes the micro-tile, so the module is architecture-specific
    # either way; pinning the target here keeps the herd sized for the part it
    # will be compiled for.
    mlir_module = launch.build(target="npu2" if args.arch == "aie2p" else "npu1")

    # Vectorization - only run if direct codegen mode is enabled
    if args.direct_codegen:
        transform_ir_string = (
            """
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


                %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op

                %inner_most_matmul, %vec_loops:3 =
                  transform.structured.tile_using_for %matmul tile_sizes [2, 2, 1, 0, 0, 0]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)  
                %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
                  transform.structured.tile_using_for %inner_most_matmul tile_sizes [1, 1, 0, 0, 0, 0]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)  
                transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !transform.any_op
                transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !transform.any_op

                %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %inner_most_fills, %vec_fill_loops:2 =
                  transform.structured.tile_using_for %linalg_fills tile_sizes [0, 0, 1, 1]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

                %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op
                
                %herd1, %herd2, %herd3 = transform.split_handle %vectorized_herds : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                %scf_fors = transform.structured.match ops{["scf.for"]} in %herd2 : (!transform.any_op) -> !transform.any_op

                %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                transform.apply_patterns to %func1 {
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                    transform.apply_patterns.memref.fold_memref_alias_ops
                } : !transform.any_op
                %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

                // Eliminate redundant vector.transfer_read operations
                %func1_rematch = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func1_rematch : (!transform.any_op) -> !transform.any_op
                
                // Hoist loop-invariant vector transfers out of innermost loop
                %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %vectorized_herds_1 = transform.air.herd_vectorize %herds_1 : (!transform.any_op) -> !transform.any_op
                %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %vectorized_herds_1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                
                %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
                %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
                
                %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]} : (!transform.any_op) -> !transform.any_op

                // Hoist all accumulator transfer pairs from the innermost loop
                %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op

                %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
                %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

                %fors_to_hoist_ptrs = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
                %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist_ptrs {overflow_result = 1}: (!transform.any_op) -> (!transform.any_op, !transform.any_op)

                """
            + (
                """
                // Hoist the 4 extf/truncf pairs from the innermost loop
                // (only applicable when output is bf16, producing paired extf/truncf ops)
                %all_extf_loop = transform.structured.match ops{["arith.extf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op
                %all_truncf_loop = transform.structured.match ops{["arith.truncf"]} in %innermost_for1 : (!transform.any_op) -> !transform.any_op

                // Split to get individual operations (4 extf total)
                %extf_bf16_1, %extf_bf16_2, %extf_bf16_3, %extf_bf16_4 = transform.split_handle %all_extf_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

                // The 4 truncf ops correspond to the 4 vector.contract results
                %truncf_1, %truncf_2, %truncf_3, %truncf_4 = transform.split_handle %all_truncf_loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

                // Hoist first pair
                %for1_1_hoisted_1 = transform.air.hoist_cast_pair %extf_bf16_1, %truncf_1, %innermost_for1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

                // Re-match and hoist second pair
                %all_extf_loop_2 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_1 : (!transform.any_op) -> !transform.any_op
                %all_truncf_loop_2 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_1 : (!transform.any_op) -> !transform.any_op
                %extf_bf16_2_new, %e2_5, %e2_6 = transform.split_handle %all_extf_loop_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                %truncf_2_1, %truncf_2_2, %truncf_2_3 = transform.split_handle %all_truncf_loop_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                %for1_1_hoisted_2 = transform.air.hoist_cast_pair %extf_bf16_2_new, %truncf_2_1, %for1_1_hoisted_1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

                // Re-match and hoist third pair
                %all_extf_loop_3 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_2 : (!transform.any_op) -> !transform.any_op
                %all_truncf_loop_3 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_2 : (!transform.any_op) -> !transform.any_op
                %extf_bf16_3_new, %e3_7 = transform.split_handle %all_extf_loop_3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
                %truncf_3_1, %truncf_3_2 = transform.split_handle %all_truncf_loop_3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
                %for1_1_hoisted_3 = transform.air.hoist_cast_pair %extf_bf16_3_new, %truncf_3_1, %for1_1_hoisted_2 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op

                // Re-match and hoist fourth pair
                %all_extf_loop_4 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_3 : (!transform.any_op) -> !transform.any_op
                %all_truncf_loop_4 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_3 : (!transform.any_op) -> !transform.any_op
                %for1_1_hoisted_final = transform.air.hoist_cast_pair %all_extf_loop_4, %all_truncf_loop_4, %for1_1_hoisted_3 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
                """
                if OUTPUT_DATATYPE == bfloat16
                else ""
            )
            + """

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
        )
        transform_ir = Module.parse(transform_ir_string, context=mlir_module.context)
        run_transform(transform_ir, mlir_module)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Variance-normalized inputs following PyTorch's
    # random_matrix_with_scaled_reduction_dim: randn / sqrt(K).
    # This keeps output variance ~1 regardless of K, so relative
    # tolerance behaves consistently across matrix sizes.
    scale = 1.0 / math.sqrt(args.k)
    input_a = (np.random.randn(args.m, args.k) * scale).astype(INPUT_DATATYPE)
    input_b = (np.random.randn(args.k, args.n) * scale).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        # Full reference: f32 matmul over the whole output, cast to the output
        # dtype. Computing the reference in f32 then casting matches PyTorch's
        # approach and isolates the device's quantization error. A full (not
        # sampled) check matches GPU practice (cuBLAS/CUTLASS verify every
        # element) and avoids missing worst-case elements; with an optimized
        # BLAS the numpy matmul reference is typically fast relative to the
        # compile + on-device run.
        reference = (input_a.astype(np.float32) @ input_b.astype(np.float32)).astype(
            OUTPUT_DATATYPE
        )

        # Tolerances are sized to the kernel's measured worst-case error, which
        # depends on the architecture's bf16 matmul datapath:
        # - f32 output: rtol=2e-3, atol=2e-3 (no bf16 output rounding).
        # - bf16 output on aie2p: rtol=1.6e-2 (PyTorch's default bf16 rtol),
        #   atol=4e-3 (sized to the BFP16-emulated path's worst-case abs error
        #   ~3e-3 from block quantization + bf16 output rounding).
        # - bf16 output on aie2: the native 4x8x4 bf16 MAC path has larger error
        #   (measured mean_rel_L1 ~3e-2, abs_err ~7e-3) than aie2p's BFP16+conv
        #   path, so it needs looser tolerances.
        if OUTPUT_DATATYPE == np.float32:
            test_rtol, test_atol = 2e-3, 2e-3
        elif args.arch == "aie2p":
            test_rtol, test_atol = 1.6e-2, 4e-3
        else:  # aie2 (NPU1) native bf16 MAC, larger error
            test_rtol, test_atol = 5e-2, 1e-2

        ###### Compile and test
        runner_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
            "stack_size": 2048,
            "report_precision": True,
            "n_perf_iters": args.perf_iters,
            "perf_flops": (
                (2.0 * args.m * args.k * args.n) if args.perf_iters > 0 else None
            ),
        }
        # Only use external kernel library if NOT in direct codegen mode
        if not args.direct_codegen:
            runner_kwargs["lower_linalg_to_func"] = "mm.o"

        runner = XRTRunner(**runner_kwargs, instance_name="matmul_bf16")
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                expected_outputs=[reference],
                rtol=test_rtol,
                atol=test_atol,
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        ###### Compile and generate xclbin (requires XRT, no execution)
        backend_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
            "stack_size": 2048,
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
            "stack_size": 2048,
        }
        # Only use external kernel library if NOT in direct codegen mode
        if not args.direct_codegen:
            backend_kwargs["lower_linalg_to_func"] = "mm.o"

        backend = XRTBackend(**backend_kwargs)
        module_function = backend.compile(mlir_module)

        backend.unload()

        print("Compilation completed successfully!")
        sys.exit(0)
