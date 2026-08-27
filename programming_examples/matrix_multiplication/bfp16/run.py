# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""bfp16ebs8 A x bfp16ebs8 B -> bf16/f32 C GEMM on NPU2, on air.api.

Runs the mlir-aie reference kernel ``mm_bfp.cc`` unmodified, so every bit of
arithmetic is on the far side of an ``air.extern`` call and what moves onto the
DSL is the schedule and the data movement.

AIR has no bfp16ebs8 element type -- mlir-aie's ``!aiex.bfp<>`` lowers to i72
only inside ``aie.device``, i.e. after every AIR pass, and AIR's BD lowering
assumes an int-or-float element type. So A and B cross the AIR boundary as plain
``i8`` and the kernel reinterprets them via ``aie::block_vector<bfp16ebs8>``, the
same type-pun ``bf16_x_bfp16`` uses for its weights.

Layout: bfp16ebs8 packs 8 scalars into 9 bytes, so a logical ``[H, W]`` matrix is
an ``[H, W*9//8]`` byte matrix. mlir-aie's host-side
``shuffleMatrixForBfp16ebs8()`` reorders 8x8 sub-tiles WITHIN each tile box only,
leaving the global matrix row-major -- so A and B here are byte-for-byte what
mlir-aie's own ``bfp_test.cpp`` produces. B is consumed transposed, i.e. stored
``[N, K*9//8]``.

C is accumulated in bfp16ebs8 by the kernel and converted to bf16 or f32 on the
way out (``bfp_cvt.cc``), so the output is an ordinary float tensor that
XRTRunner checks against a float reference exactly like the bf16 example::

    air.launch (m_outer, n_outer)
      air.segment
        L2: A [herd_m, k_per_l2, a_l1_b]      (packed bytes)
            B [herd_n, k_per_l2, b_l1_b]      (packed bytes, transposed)
            C [herd_m, herd_n, tile_m, tile_n](bf16 or f32)
        L1, segment lifetime, one slab per core:
            acc [herd_m, herd_n, c_l1_b]      (bfp16ebs8 bytes)
            out [herd_m, herd_n, c_elems]     (bf16 or f32)
        herd            zero the accumulator
        for k2 in air.sequential(k / tile_k_l2)
            L3 -> L2 for A and B
            herd
                for j in air.sequential(k_per_l2)
                    L2 -> L1, then one mmul into the accumulator
        herd            convert bfp16 -> output dtype and drain to L2
        L2 -> L3

**The accumulator is shared, and a core is handed its own slab.** It has to
survive the three separate herd invocations -- zero, compute, drain -- so it is
allocated at segment scope with ``<segment>.shared()``, carrying one leading
dimension per herd axis. The predecessor wrote a ``memref.subview`` at each of
the four call sites to pick out the calling core's slab, and a
``StridedLayoutAttr`` per kernel declaration to spell the resulting memref type;
both are derivable, so ``zero_kernel(acc)`` is the whole of it.

**Every access pattern here is a reshape or a transpose of a plain region.** The
L3 operands are byte matrices and the L2 staging buffers are per-core tile
boxes, so filling them splits the row axis into (core, within-tile) and brings
the K-chunk axis outside -- ``reshape(...).transpose(0, 2, 1, 3)``. The drain is
the inverse: the L1 side is one contiguous run in 8x8-block order, and only the
L2 side needs the permute, which is why the reshape lands on the destination.

The ping-pong opt-out is gone. This example used to carry
``air.disable_ping_pong`` on the K-l1 fill loop purely to supply a fact the pass
could not see; ``air-label-scf-for-to-ping-pong`` now derives the L1 budget
itself (#1928).
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32, i8
from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend

from bfp16_utils import float_to_bfp16ebs8, nbytes, shuffle_bfp16ebs8

np.random.seed(42)

KERNEL_OBJ_NAME = "mm_bfp.o"

# bfp16ebs8 MMUL is 8x8x8 (mac_8x8_8x8T).
MMUL_R = MMUL_S = MMUL_T = 8


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
    np_dtype_out,
    arch="aie2p",
):
    r, s, t = MMUL_R, MMUL_S, MMUL_T

    # bfp16ebs8 is an AIE2P type; there is no aie2 lowering for it.
    assert arch == "aie2p", "bfp16ebs8 requires arch aie2p"

    # mm_bfp.cc static_asserts
    assert tile_m % (2 * r) == 0, "tile_m must be a multiple of 16"
    # mm_bfp.cc's static_assert allows tile_m == 16, but at that size rowA is 2
    # while the z loop still carries AIE_LOOP_MIN_ITERATION_COUNT(4), and Peano
    # fails with "ran out of registers during register allocation". 32 is the
    # smallest tile_m the reference kernel actually compiles at on AIE2P.
    assert tile_m >= 32, (
        f"tile_m ({tile_m}) must be >= 32: mm_bfp.cc exhausts the register "
        "file at tile_m=16 on AIE2P"
    )
    assert tile_n % (2 * t) == 0, "tile_n must be a multiple of 16"
    assert tile_k_l1 % s == 0, "tile_k_l1 must be a multiple of 8"
    # shuffleMatrixForBfp16ebs8 asserts (tile width is in elements)
    assert tile_k_l1 % 64 == 0, "tile_k_l1 must be a multiple of 64 (host shuffle)"
    assert tile_n % 64 == 0, "tile_n must be a multiple of 64 (host shuffle)"
    # 4-byte shim BD granularity on the L3 row strides of the packed operands
    assert k % 32 == 0, "K must be a multiple of 32 (K*9/8 must be 4-byte aligned)"

    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0

    k_per_l2 = tile_k_l2 // tile_k_l1
    dt_out = bf16 if np_dtype_out == bfloat16 else f32

    KB = nbytes(k)  # L3 A/B row stride, bytes
    a_row_b = nbytes(tile_k_l1)  # one L3 row of an A/B tile box
    a_l1_b = nbytes(tile_m * tile_k_l1)
    b_l1_b = nbytes(tile_n * tile_k_l1)
    c_l1_b = nbytes(tile_m * tile_n)  # bfp16 accumulator, bytes
    c_elems = tile_m * tile_n  # converted output, elements
    MB, NB = tile_m // r, tile_n // t  # 8x8 blocks per tile

    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n

    # A and B are packed bytes; C is a real float tensor.
    A = air.tensor([m, KB], i8)
    B = air.tensor([n, KB], i8)  # B stored transposed
    C = air.tensor([m, n], dt_out)

    zero_kernel = air.extern("zero_kernel", link_with=KERNEL_OBJ_NAME)
    matmul = air.extern("matmul_vectorized_bfp16", link_with=KERNEL_OBJ_NAME)
    cvt_name = "bfp16_to_bf16_mn" if np_dtype_out == bfloat16 else "bfp16_to_f32_mn"
    convert = air.extern(cvt_name, link_with=KERNEL_OBJ_NAME)

    with air.launch(
        [range(m // l2_m), range(n // l2_n)], name="matmul_bfp16"
    ) as launch_ctx:

        @launch_ctx.body
        def _(ivx, ivy):
            with air.segment(name="matmul_seg") as seg:

                @seg.body
                def _():
                    l2_a = air.alloc(
                        [herd_m, k_per_l2, a_l1_b], i8, scope=seg.private()
                    )
                    l2_b = air.alloc(
                        [herd_n, k_per_l2, b_l1_b], i8, scope=seg.private()
                    )
                    l2_c = air.alloc(
                        [herd_m, herd_n, tile_m, tile_n], dt_out, scope=seg.private()
                    )
                    # Segment-shared so the accumulator survives across the
                    # zero / compute / drain herd invocations.
                    acc = air.alloc([herd_m, herd_n, c_l1_b], i8, scope=seg.shared())
                    out = air.alloc(
                        [herd_m, herd_n, c_elems], dt_out, scope=seg.shared()
                    )

                    off_x_rows = ivx * l2_m  # A/C row offset
                    off_y_rows = ivy * l2_n  # B row offset
                    off_y_cols = ivy * l2_n  # C column offset

                    # ---- Herd #1: zero the bfp16 L1 accumulator.
                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as zero_herd:

                        @zero_herd.body
                        def _(tx, ty):
                            zero_kernel(acc)

                    # ---- Segment-level K-l2 loop.
                    for i in air.sequential(k // tile_k_l2):
                        k_off_b = i * nbytes(tile_k_l2)

                        # L3 -> L2: gather tile boxes. Split the row axis into
                        # (core, within-tile) and the byte axis into
                        # (K-chunk, within-chunk), then bring the K-chunk axis
                        # outside the row one -- which is the order the per-core
                        # tile boxes sit in.
                        ops.load(
                            l2_a,
                            A[
                                off_x_rows : off_x_rows + l2_m,
                                k_off_b : k_off_b + k_per_l2 * a_row_b,
                            ]
                            .reshape(herd_m, tile_m, k_per_l2, a_row_b)
                            .transpose(0, 2, 1, 3),
                        )
                        ops.load(
                            l2_b,
                            B[
                                off_y_rows : off_y_rows + l2_n,
                                k_off_b : k_off_b + k_per_l2 * a_row_b,
                            ]
                            .reshape(herd_n, tile_n, k_per_l2, a_row_b)
                            .transpose(0, 2, 1, 3),
                        )

                        # ---- Herd #2: accumulate over the K-l1 chunks.
                        with air.herd(
                            [range(herd_m), range(herd_n)],
                            name="herd_0",
                            shape=(herd_m, herd_n),
                        ) as h:

                            @h.body
                            def _(tx, ty):
                                # A/B tile boxes are contiguous in
                                # sub-tile-major order, which is exactly what
                                # mm_bfp.cc's block_vector_input_buffer_stream
                                # .seek(z*colA) expects -- no strided view here.
                                a1 = air.alloc([a_l1_b], i8, scope=h.private())
                                b1 = air.alloc([b_l1_b], i8, scope=h.private())
                                for j in air.sequential(k_per_l2):
                                    ops.load(a1, l2_a[tx, j, :])
                                    ops.load(b1, l2_b[ty, j, :])
                                    matmul(a1, b1, acc)

                    # ---- Herd #3: convert bfp16 -> output dtype, then drain.
                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as drain_herd:

                        @drain_herd.body
                        def _(tx, ty):
                            convert(acc, out)
                            # L1 (8x8-block order) -> L2 (row-major). The L1
                            # side is a single contiguous run, so only the L2
                            # side needs the permute -- which is why the
                            # transpose lands on the destination.
                            ops.store(
                                out[tx, ty, :].reshape(1, 1, MB, NB, r, t),
                                l2_c[tx, ty, :, :]
                                .reshape(1, 1, MB, r, NB, t)
                                .transpose(0, 1, 2, 4, 3, 5),
                            )

                    # ---- L2 -> L3.
                    ops.store(
                        l2_c.transpose(0, 2, 1, 3),
                        C[
                            off_x_rows : off_x_rows + l2_m,
                            off_y_cols : off_y_cols + l2_n,
                        ].reshape(herd_m, tile_m, herd_n, tile_n),
                    )

    # bfp16ebs8 is an AIE2P datatype and every lit is REQUIRES: ryzen_ai_npu2.
    return launch_ctx.build(target="npu2")


def pack_operands(A, B, tile_m, tile_k_l1, tile_n):
    """Float A[M,K], B[K,N] -> the packed+shuffled device buffers.

    Byte-for-byte what mlir-aie's bfp_test.cpp builds: quantize with
    floatToBfp16(), then shuffleMatrixForBfp16ebs8() with the per-core tile
    box. B is quantized transposed because the kernel consumes it that way.
    """
    A_dev = shuffle_bfp16ebs8(float_to_bfp16ebs8(A), tile_m, tile_k_l1)
    B_dev = shuffle_bfp16ebs8(
        float_to_bfp16ebs8(np.ascontiguousarray(B.T)), tile_n, tile_k_l1
    )
    return A_dev, B_dev


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
    OUTPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the bfp16ebs8 matmul example",
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
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2p",
        help="Target AIE architecture (aie2p only; bfp16ebs8 is an AIE2P type)",
    )
    parser.add_argument(
        "--output-dtype",
        type=str,
        choices=["bf16", "f32"],
        default=None,
        dest="output_dtype",
        help="Override output data type (default: f32 for aie2p)",
    )
    parser.add_argument(
        "--runtime-loop-tiling",
        type=int,
        default=None,
        dest="runtime_loop_tiling",
        help="Tile factor for the runtime launch loop (default: auto -- 2, or 1 "
        "for small herds, see the note in the source)",
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

    # bfp16ebs8 (v8bfp16ebs8 / mac_8x8_8x8T) exists only on AIE2P. There is no
    # aie2 fallback -- neither the type nor the MMUL intrinsic is available, and
    # xchesscc has no bfp16ebs8 codegen path either. Fail loudly rather than
    # emitting IR that cannot be lowered.
    if args.arch != "aie2p":
        print(
            f"Error: --arch {args.arch} is not supported by this example.",
            file=sys.stderr,
        )
        print(
            "bfp16ebs8 (v8bfp16ebs8) and its mac_8x8_8x8T MMUL are AIE2P-only; "
            "NPU1/aie2 has no block-floating-point datapath. Re-run with "
            "--arch aie2p on an NPU2 (Strix) device.",
            file=sys.stderr,
        )
        sys.exit(1)

    # aie2p defaults to f32 output for better precision.
    # Can be overridden with --output-dtype.
    if args.output_dtype == "bf16":
        pass  # keep default bfloat16
    elif args.output_dtype == "f32":
        OUTPUT_DATATYPE = np.float32
    elif args.arch == "aie2p":
        OUTPUT_DATATYPE = np.float32

    # bfp16ebs8 kernels are Peano-only; xchesscc has no codegen path for them.
    if args.compile_mode != "compile-only" and not os.environ.get("PEANO_INSTALL_DIR"):
        print(
            "Error: PEANO_INSTALL_DIR environment variable is not set.",
            file=sys.stderr,
        )
        print(
            "Peano is required for bfp16ebs8; the Chess toolchain has no "
            "bfp16ebs8 codegen path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Runtime launch-loop tiling. [2, 2] batches the per-launch DMA task
    # triplets and is worth ~26% throughput (183us vs 232us at 512^3 / 4x4).
    #
    # It is NOT safe on small herds: when the herd is small enough that AIR
    # gives the C drain a single shim channel, the shim-BD fold collapses that
    # drain across every launch iteration into one task with one await, so only
    # the first launch iteration's output is ever written while all the input
    # fills still run. Observed at herd 2x2 / 2x1 / 1x2 with more than one
    # launch iteration (herd 4x2 and larger split C over 4 channels and are
    # fine). Same class as the shim-BD stride-fold / wait-pairing fixes in
    # mlir-air #1810 and #1815. Override with --runtime-loop-tiling.
    n_launch = (args.m // (args.tile_m * args.herd_m)) * (
        args.n // (args.tile_n * args.herd_n)
    )
    n_cores = args.herd_m * args.herd_n
    if args.runtime_loop_tiling is not None:
        runtime_loop_tiling_sizes = [args.runtime_loop_tiling] * 2
    elif n_launch > 1 and n_cores < 8:
        runtime_loop_tiling_sizes = [1, 1]
        print(
            f"note: herd {args.herd_m}x{args.herd_n} ({n_cores} cores) with "
            f"{n_launch} launch iterations -- using runtime_loop_tiling_sizes="
            "[1, 1]; [2, 2] drops all but the first launch iteration's output "
            "on herds this small. Override with --runtime-loop-tiling."
        )
    else:
        runtime_loop_tiling_sizes = [2, 2]

    # mm_bfp.cc keeps C in bfp16ebs8 between K-l1 chunks, re-quantizing the
    # accumulator to an 8-bit mantissa with a shared exponent on every chunk.
    # Error therefore grows with the number of round-trips, K/tile_k_l1,
    # independently of the input distribution (measured mean_rel_L1 at
    # M=128,N=2048: 32 round-trips -> 0.27 randn / 0.09 integer; 128
    # round-trips -> 1.23 randn / 0.41 integer). Past ~64 round-trips the
    # result is not usable at any sane tolerance.
    n_roundtrips = args.k // args.tile_k_l1
    if n_roundtrips > 64:
        print(
            f"warning: K/tile_k_l1 = {n_roundtrips} accumulator round-trips. "
            "mm_bfp.cc re-quantizes C to bfp16ebs8 on every K-l1 chunk, so this "
            "shape will lose too much precision to pass the check. Raise "
            "--tile-k-l1 or use a shallower K.",
            file=sys.stderr,
        )

    mlir_module = build_module(
        args.m,
        args.k,
        args.n,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
        OUTPUT_DATATYPE,
        args.arch,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Variance-normalized inputs following PyTorch's
    # random_matrix_with_scaled_reduction_dim: randn / sqrt(K). This keeps
    # output variance ~1 regardless of K, so relative tolerance behaves
    # consistently across matrix sizes. Same generator as the bf16 example.
    scale = 1.0 / np.sqrt(args.k)
    input_a = (np.random.randn(args.m, args.k) * scale).astype(np.float32)
    input_b = (np.random.randn(args.k, args.n) * scale).astype(np.float32)

    if args.compile_mode == "compile-and-run":
        # Device operands are the bfp16ebs8-quantized, sub-tile-shuffled bytes.
        packed_a, packed_b = pack_operands(
            input_a, input_b, args.tile_m, args.tile_k_l1, args.tile_n
        )

        # Full reference: f32 matmul over the whole output, cast to the output
        # dtype. Computing the reference from the ORIGINAL floats (not the
        # de-quantized operands) means the check covers the bfp16 input
        # quantization too, which is what mlir-aie's bfp_test.cpp does.
        reference = (input_a @ input_b).astype(OUTPUT_DATATYPE)

        # Tolerances are sized to the kernel's measured worst-case error.
        #
        # bfp16ebs8 shares one 8-bit exponent across 8 elements, so an element
        # sitting in a block with a much larger neighbour loses mantissa bits.
        # On Gaussian inputs that is the dominant error term -- it comes from
        # quantizing A and B, not from the output dtype, which is why bf16 and
        # f32 output need the same tolerance here.
        #
        # The check is therefore absolute-error driven: with randn/sqrt(K)
        # inputs the reference has std ~1/sqrt(K), which shrinks with K while
        # the bfp16 error floor does not, so mean_rel_L1 grows with K
        # (measured: 2.1e-2 at K=64, 6.9e-2 at K=512, 2.7e-1 at K=2048).
        # Measured abs_err max across every Makefile target and both output
        # dtypes is 8.5e-3 .. 1.5e-2, so atol below leaves ~1.7x headroom.
        #
        # For reference, mlir-aie's own bfp_test.cpp sidesteps this by drawing
        # inputs from rand()%16 ("limiting to 16 to avoid precision loss
        # issues") -- integers of similar magnitude share an exponent almost
        # losslessly. This example keeps the bf16 example's randn/sqrt(K)
        # generator instead, so the numbers above are the honest bfp16 error
        # on high-dynamic-range data.
        test_rtol, test_atol = 5e-2, 2.5e-2

        ###### Compile and test
        runner_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": runtime_loop_tiling_sizes,
            "stack_size": 2048,
            "report_precision": True,
            "n_perf_iters": args.perf_iters,
            "perf_flops": (
                (2.0 * args.m * args.k * args.n) if args.perf_iters > 0 else None
            ),
        }

        runner = XRTRunner(**runner_kwargs, instance_name="matmul_bfp16")
        exit(
            runner.run_test(
                mlir_module,
                inputs=[packed_a, packed_b],
                expected_outputs=[reference],
                rtol=test_rtol,
                atol=test_atol,
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        ###### Compile and generate xclbin (requires XRT, no execution)
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=runtime_loop_tiling_sizes,
            stack_size=2048,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()

    elif args.compile_mode == "compile-only":
        ###### Compile only (without XRT dependencies)
        backend = XRTBackend(
            verbose=args.verbose,
            target_device="npu2",
            output_format="none",
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=runtime_loop_tiling_sizes,
            stack_size=2048,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()

        print("Compilation completed successfully!")
        sys.exit(0)
