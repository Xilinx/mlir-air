# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Block-quantized vector-matrix multiplication on air.api.

``C[N] = sum_g (A_i8[g] @ B_i8[g,N]) * a_scale[g] * b_scale[g,N]``: the operands
are int8, accumulated in int32, and rescaled to f32 once per block of ``bs``
elements along K. So every operand travels as a *pair* -- the quantized data and
its per-block scale -- and the pair shares a channel:

    A:  L3 --aL3ToL2--> L2 --aL2ToL1--> L1     (data, then scale)
    B:  L3 --bL3ToL2--> L2 --bL2ToL1--> L1     (data, then scale)
    C:  L1 --cL1ToL2--> L2 --cL2ToL3--> L3

Two puts into one channel and two gets out of it, in the same order. That is
ordinary channel semantics rather than a trick: a channel is a stream, each get
takes the next chunk, and nothing requires the chunks to be the same shape --
which is what lets ``[k]`` of data and ``[k/bs]`` of scale share ``aL3ToL2``.

The B side is micro-tiled for the AIE2 int8 vecmat intrinsic (``m x k`` by
``k x n``, here ``(1, 16, 8)``), so it reaches L1 as ``[N/8, K/16, 16, 8]``
rather than row-major. The DMA does the pack, by walking the flat L2 buffer out
of order, and ``pack=`` names that walk::

    b_l2_l1.put(l2_b[kk : kk + tile_k, :], pack=mm.b(tile_k, tile_n, lead=()))

The B *scale* is packed the same way but with one scale per block instead of
per element, which is exactly a micro-tile of ``k=1`` -- so it reuses the same
derivation with a different micro-tile rather than a special case.

Unchanged from the raw-bindings version this replaces, except for three things
the DSL requires and one it fixes:

* The L3-side puts and gets sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The L1 tiles are allocated above the K loop and reused across trips, rather
  than a fresh set per trip.
* ``linalg_fill_i32_view16x8xi32as2`` is declared with the signature ``vm.cc``
  actually defines -- ``void linalg_fill_i32_view16x8xi32as2(float *)``, the
  buffer alone. The predecessor declared it ``(f32, memref)`` and passed a zero
  the callee never reads; harmless, since the C ABI drops the extra argument,
  but untrue.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import f32, i8

# air.api dtypes, keyed by the numpy dtype the harness works in.
DTYPE = {np.int8: i8, np.float32: f32}

# The AIE2 int8 vecmat intrinsic's operand shape: m x k by k x n.
MMUL_MKN = (1, 16, 8)


def build_module(k, n, bs, tile_k, tile_n, np_dtype_in, np_dtype_out, link_with="vm.o"):
    assert k % tile_k == 0
    assert n % tile_n == 0
    assert tile_k % bs == 0

    dt_in = DTYPE[np_dtype_in]
    dt_out = DTYPE[np_dtype_out]

    m_u, k_u, n_u = MMUL_MKN
    mm = air.micro_tile(m_u, k_u, n_u)
    # One scale per block of bs elements along K, laid out in the same
    # micro-tile order as the data it scales -- a k=1 micro-tile.
    mm_scale = air.micro_tile(m_u, 1, n_u)

    A = air.tensor([k], dt_in)
    A_s = air.tensor([k // bs], dt_out)
    B = air.tensor([k, n], dt_in)
    B_s = air.tensor([k // bs, n], dt_out)
    C = air.tensor([n], dt_out)

    a_l3_l2 = air.channel("aL3ToL2")
    b_l3_l2 = air.channel("bL3ToL2")
    a_l2_l1 = air.channel("aL2ToL1")
    b_l2_l1 = air.channel("bL2ToL1")
    c_l1_l2 = air.channel("cL1ToL2")
    c_l2_l3 = air.channel("cL2ToL3")

    vecmat = air.extern(f"vecmat_i8_f32_i32_{bs}", object=link_with)
    fill = air.extern("linalg_fill_i32_view16x8xi32as2", object=link_with)

    with air.launch(name="vecmat_i8") as launch:

        @launch.body
        def _():
            with air.segment([range(0, n, tile_n)], name="vecmat_i8_0") as seg:

                @seg.body
                def _(sj):
                    col = sj * tile_n

                    l2_a = air.alloc([k], dt_in, scope=seg.private())
                    l2_a_s = air.alloc([k // bs], dt_out, scope=seg.private())
                    l2_b = air.alloc([k, tile_n], dt_in, scope=seg.private())
                    l2_b_s = air.alloc([k // bs, tile_n], dt_out, scope=seg.private())
                    l2_c = air.alloc([tile_n], dt_out, scope=seg.private())

                    a_l3_l2.put(A)
                    a_l3_l2.put(A_s)
                    b_l3_l2.put(B[0:k, col : col + tile_n])
                    b_l3_l2.put(B_s[0 : k // bs, col : col + tile_n])
                    a_l3_l2.get(l2_a)
                    a_l3_l2.get(l2_a_s)
                    b_l3_l2.get(l2_b)
                    b_l3_l2.get(l2_b_s)

                    # One slice per K tile, data then scale, on both sides.
                    for i in air.sequential(0, k // tile_k):
                        kk = i * tile_k
                        ks = i * (tile_k // bs)
                        a_l2_l1.put(l2_a[kk : kk + tile_k])
                        a_l2_l1.put(l2_a_s[ks : ks + tile_k // bs])

                    for i in air.sequential(0, k // tile_k):
                        kk = i * tile_k
                        ks = i * (tile_k // bs)
                        b_l2_l1.put(
                            l2_b[kk : kk + tile_k, 0:tile_n],
                            pack=mm.b(tile_k, tile_n, lead=()),
                        )
                        b_l2_l1.put(
                            l2_b_s[ks : ks + tile_k // bs, 0:tile_n],
                            pack=mm_scale.b(tile_k // bs, tile_n, lead=()),
                        )

                    with air.herd([range(1)], name="herd_0", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            l1_a = air.alloc(
                                [tile_k // k_u, k_u], dt_in, scope=h.private()
                            )
                            l1_a_s = air.alloc(
                                [tile_k // bs], dt_out, scope=h.private()
                            )
                            l1_b = air.alloc(
                                [tile_n // n_u, tile_k // k_u, k_u, n_u],
                                dt_in,
                                scope=h.private(),
                            )
                            l1_b_s = air.alloc(
                                [tile_n // n_u, tile_k // bs, n_u],
                                dt_out,
                                scope=h.private(),
                            )
                            l1_c = air.alloc(
                                [tile_n // n_u, n_u], dt_out, scope=h.private()
                            )

                            fill(l1_c)
                            for _j in air.sequential(0, k, tile_k):
                                a_l2_l1.get(l1_a)
                                a_l2_l1.get(l1_a_s)
                                b_l2_l1.get(l1_b)
                                b_l2_l1.get(l1_b_s)
                                vecmat(l1_a, l1_a_s, l1_b, l1_b_s, l1_c)

                            c_l1_l2.put(l1_c)

                    c_l1_l2.get(l2_c)
                    c_l2_l3.put(l2_c)
                    c_l2_l3.get(C[col : col + tile_n])

    return launch


if __name__ == "__main__":
    # Default values.
    K = 288
    N = 48
    BS = 32
    TILE_K = 96
    TILE_N = 48
    INPUT_DATATYPE = np.int8
    ACC_DATATYPE = np.int32
    OUTPUT_DATATYPE = np.float32

    np.random.seed(42)

    parser = argparse.ArgumentParser(
        prog="single_core.py",
        description="Builds, runs, and tests the block-quantized int8 vector-matrix multiplication example",
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
        "--k", type=int, default=K, help="K dimension size in a (1xK) * (KxN) matmul"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="N dimension size in a (1xK) * (KxN) matmul",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=BS,
        help="Block size, for blocked quantization",
    )
    parser.add_argument(
        "--tile-k", type=int, default=TILE_K, help="K dimension size of each L1 tile"
    )
    parser.add_argument(
        "--tile-n", type=int, default=TILE_N, help="N dimension size of each L1 tile"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )

    args = parser.parse_args()

    launch = build_module(
        args.k,
        args.n,
        args.bs,
        args.tile_k,
        args.tile_n,
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.random.randint(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=(args.k),
        dtype=INPUT_DATATYPE,
    )
    input_a_s = np.random.randn(
        args.k // args.bs,
    ).astype(OUTPUT_DATATYPE)
    input_b = np.random.randint(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=(args.k, args.n),
        dtype=INPUT_DATATYPE,
    )
    input_b_s = np.random.rand(
        args.k // args.bs,
        args.n,
    ).astype(OUTPUT_DATATYPE)
    output_c = np.zeros(shape=(args.n), dtype=OUTPUT_DATATYPE)
    for n1 in range(args.n):
        output_c[n1] = OUTPUT_DATATYPE(0.0)
        ival = ACC_DATATYPE(0.0)
        for g in range(0, args.k - args.bs + 1, args.bs):
            for k1 in range(args.bs):
                ival += ACC_DATATYPE(input_a[g + k1]) * ACC_DATATYPE(
                    input_b[g + k1][n1]
                )
            output_c[n1] += (
                OUTPUT_DATATYPE(ival)
                * input_a_s[g // args.bs]
                * input_b_s[g // args.bs][n1]
            )
            ival = 0

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="vecmat_i8",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_a, input_a_s, input_b, input_b_s],
            expected_outputs=[output_c],
        )
    )
