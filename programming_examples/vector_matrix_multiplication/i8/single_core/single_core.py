# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""int8 vector-matrix multiplication on air.api: C[N] = A[K] @ B[K,N], in i32.

Six channels carry the whole schedule -- there is not a single
``air.dma_memcpy_nd`` in it:

    A:  L3 --aL3ToL2--> L2 --aL2ToL1--> L1
    B:  L3 --bL3ToL2--> L2 --bL2ToL1--> L1
    C:  L1 --cL1ToL2--> L2 --cL2ToL3--> L3

and the herd names none of them as an operand: ``air.herd`` is
``IsolatedFromAbove``, but a channel is a module-level symbol and resolves by
name from any depth.

What differs from the bf16 sibling is the layout. The AIE2 int8 vecmat
intrinsic wants its operands micro-tiled -- ``m x k`` by ``k x n`` with
``(1, 16, 8)`` -- so the B tile reaches L1 as ``[N/8, K/16, 16, 8]`` rather than
row-major ``[K, N]``. Nothing reorders it in a core: the *DMA* does the pack,
by walking the flat L2 buffer out of order. That walk is what ``pack=`` names::

    b_l2_l1.put(l2_b[kk : kk + tile_k, :], pack=mm.b(tile_k, tile_n, lead=()))

``ops.load`` derives the same walk from its destination buffer, which it can
see. A channel cannot -- put and get are separate ops and the packed side is at
the far end of the stream -- so the pack is named at the put.

A is micro-tiled too, but with ``m=1`` the pack is the identity: ``[K/16, 16]``
is just ``[K]`` re-shaped, so its put is an ordinary contiguous slice.

Unchanged from the raw-bindings version this replaces, except for three things
the DSL requires and one it fixes:

* The L3-side puts and gets sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The L1 tiles are allocated above the K loop and reused across trips, rather
  than a fresh set per trip.
* ``linalg_fill_i32_view16x8xi32as2`` is declared with the signature ``vm.cc``
  actually defines -- ``void linalg_fill_i32_view16x8xi32as2(int *)``, the
  buffer alone. The predecessor declared it ``(i32, memref)`` and passed a zero
  the callee never reads; harmless, since the C ABI drops the extra argument,
  but untrue.
"""

import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import i8, i32

# air.api dtypes, keyed by the numpy dtype the harness works in.
DTYPE = {np.int8: i8, np.int32: i32}

# The AIE2 int8 vecmat intrinsic's operand shape: m x k by k x n.
MMUL_MKN = (1, 16, 8)


def build_module(k, n, tile_k, tile_n, np_dtype_in, np_dtype_out, link_with="vm.o"):
    assert k % tile_k == 0
    assert n % tile_n == 0

    dt_in = DTYPE[np_dtype_in]
    dt_out = DTYPE[np_dtype_out]

    m_u, k_u, n_u = MMUL_MKN
    mm = air.micro_tile(m_u, k_u, n_u)

    A = air.tensor([k], dt_in)
    B = air.tensor([k, n], dt_in)
    C = air.tensor([n], dt_out)

    a_l3_l2 = air.channel("aL3ToL2")
    b_l3_l2 = air.channel("bL3ToL2")
    a_l2_l1 = air.channel("aL2ToL1")
    b_l2_l1 = air.channel("bL2ToL1")
    c_l1_l2 = air.channel("cL1ToL2")
    c_l2_l3 = air.channel("cL2ToL3")

    vecmat = air.extern("vecmat_i8_i32", object=link_with)
    fill = air.extern("linalg_fill_i32_view16x8xi32as2", object=link_with)

    with air.launch(name="vecmat_i8") as launch:

        @launch.body
        def _():
            with air.segment([range(0, n, tile_n)], name="vecmat_i8_0") as seg:

                @seg.body
                def _(sj):
                    col = sj * tile_n

                    l2_a = air.alloc([k], dt_in, scope=seg.private())
                    l2_b = air.alloc([k, tile_n], dt_in, scope=seg.private())
                    l2_c = air.alloc([tile_n], dt_out, scope=seg.private())

                    a_l3_l2.put(A)
                    b_l3_l2.put(B[0:k, col : col + tile_n])
                    a_l3_l2.get(l2_a)
                    b_l3_l2.get(l2_b)

                    # One slice per K tile, on both sides. Unlike the bf16
                    # sibling -- where a single whole-buffer put feeds every get
                    # because the stream is consumed in order -- B is packed
                    # here, and the pack is per tile: the walk visits all of N
                    # before advancing in K, so the tiles are not prefixes of
                    # one whole-buffer walk and have to be sent one at a time.
                    for i in air.sequential(0, k // tile_k):
                        kk = i * tile_k
                        a_l2_l1.put(l2_a[kk : kk + tile_k])

                    for i in air.sequential(0, k // tile_k):
                        kk = i * tile_k
                        b_l2_l1.put(
                            l2_b[kk : kk + tile_k, 0:tile_n],
                            pack=mm.b(tile_k, tile_n, lead=()),
                        )

                    with air.herd([range(1)], name="herd_0", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            l1_a = air.alloc(
                                [tile_k // k_u, k_u], dt_in, scope=h.private()
                            )
                            l1_b = air.alloc(
                                [tile_n // n_u, tile_k // k_u, k_u, n_u],
                                dt_in,
                                scope=h.private(),
                            )
                            l1_c = air.alloc(
                                [tile_n // n_u, n_u], dt_out, scope=h.private()
                            )

                            fill(l1_c)
                            for _j in air.sequential(0, k, tile_k):
                                a_l2_l1.get(l1_a)
                                b_l2_l1.get(l1_b)
                                vecmat(l1_a, l1_b, l1_c)

                            c_l1_l2.put(l1_c)

                    c_l1_l2.get(l2_c)
                    c_l2_l3.put(l2_c)
                    c_l2_l3.get(C[col : col + tile_n])

    return launch


if __name__ == "__main__":
    # Default values.
    K = 288
    N = 48
    TILE_K = 96
    TILE_N = 48
    INPUT_DATATYPE = np.int8
    OUTPUT_DATATYPE = np.int32

    np.random.seed(42)

    parser = argparse.ArgumentParser(
        prog="single_core.py",
        description="Builds, runs, and tests the int8 vector-matrix multiplication example",
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
    input_b = np.random.randint(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=(args.k, args.n),
        dtype=INPUT_DATATYPE,
    )
    output_c = np.dot(input_a.astype(OUTPUT_DATATYPE), input_b.astype(OUTPUT_DATATYPE))

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
            inputs=[input_a, input_b],
            expected_outputs=[output_c],
        )
    )
