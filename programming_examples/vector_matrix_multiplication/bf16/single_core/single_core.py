# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Vector-matrix multiplication (GEMV^T) on air.api: C[N] = A[K] @ B[K,N].

Six channels carry the whole schedule -- there is not a single
``air.dma_memcpy_nd`` in it:

    A:  L3 --aL3ToL2--> L2 --aL2ToL1--> L1
    B:  L3 --bL3ToL2--> L2 --bL2ToL1--> L1
    C:  L1 --cL1ToL2--> L2 --cL2ToL3--> L3

and the herd names none of them as an operand. That is the point of a channel:
``air.herd`` is ``IsolatedFromAbove``, so a buffer has to be threaded in, but a
channel is a module-level symbol and resolves by name from any depth.

The L2 -> L1 hops show the other half of it. Each puts the *whole* staged
buffer, once per K tile, while the core gets one tile per trip -- ``[k]`` put
against ``[tile_k]`` got. A channel is a stream and the get takes the next
chunk, so the two sides are deliberately not the same shape; air.api validates
them independently for exactly this reason.

Unchanged from the raw-bindings version this replaces, except for three things
the DSL requires and one it fixes:

* The L3-side puts and gets sit inside the segment. Reaching L3 needs a shim DMA
  allocation, and outside a segment there is none to link to.
* The L1 tiles are allocated above the K loop and reused across trips, rather
  than a fresh set per trip.
* ``linalg_fill_bf16`` is declared with the signature ``vm.cc`` actually
  defines -- ``void linalg_fill_bf16(bfloat16 *)``, the buffer alone. The
  predecessor declared it ``(bf16, memref)`` and passed a zero the callee never
  read; harmless, since the C ABI drops the extra argument, but untrue.
"""

import argparse
from ml_dtypes import bfloat16
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import bf16

# air.api dtypes, keyed by the numpy dtype the harness works in.
DTYPE = {bfloat16: bf16}


def build_module(k, n, tile_k, tile_n, np_dtype_in, np_dtype_out, link_with="vm.o"):
    assert k % tile_k == 0
    assert n % tile_n == 0

    dt_in = DTYPE[np_dtype_in]
    dt_out = DTYPE[np_dtype_out]

    A = air.tensor([k], dt_in)
    B = air.tensor([k, n], dt_in)
    C = air.tensor([n], dt_out)

    a_l3_l2 = air.channel("aL3ToL2")
    b_l3_l2 = air.channel("bL3ToL2")
    a_l2_l1 = air.channel("aL2ToL1")
    b_l2_l1 = air.channel("bL2ToL1")
    c_l1_l2 = air.channel("cL1ToL2")
    c_l2_l3 = air.channel("cL2ToL3")

    vecmat = air.extern("vecmat_bf16_bf16", object=link_with)
    fill = air.extern("linalg_fill_bf16", object=link_with)

    with air.launch(name="vecmat_bf16") as launch:

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

                    # One put each. A channel is a stream, so the single
                    # [k] put feeds all k/tile_k gets of [tile_k] the core
                    # makes, and likewise for b.
                    a_l2_l1.put(l2_a)
                    b_l2_l1.put(l2_b)

                    with air.herd([range(1)], name="herd_0", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            l1_a = air.alloc([tile_k], dt_in, scope=h.private())
                            l1_b = air.alloc([tile_k, tile_n], dt_in, scope=h.private())
                            l1_c = air.alloc([tile_n], dt_out, scope=h.private())

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

    input_a = np.arange(0, args.k, dtype=INPUT_DATATYPE)
    input_b = np.arange(0, args.k * args.n, dtype=INPUT_DATATYPE).reshape(
        args.k, args.n
    )
    output_c = np.dot(input_a.astype(OUTPUT_DATATYPE), input_b.astype(OUTPUT_DATATYPE))

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="vecmat_bf16",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_a, input_b],
            expected_outputs=[output_c],
            rtol=0.04,
        )
    )
