# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Matrix-vector multiplication (GEMV) on air.api: C[M] = A[M,K] @ B[K].

BF16 input/output, accfloat accumulation inside the kernel.

The schedule is unchanged from the raw-bindings version this replaces:

    air.launch (m / (tile_m * herd_m))    one segment per chunk of output rows
      air.segment
        L2: A panel and C panel for the whole chunk
        herd [herd_m, 1]                  one AIE column per row sub-chunk
            zero C
            for j in air.sequential(0, tile_m, m_input)
                L3 -> L1 for B, L2 -> L1 for A, then the kernel
            L1 -> L2 for C
        L2 -> L3 for C

Two things about it are worth stating, because both look like omissions:

* **B skips L2.** It is loaded L3 -> L1 from inside the herd loop, so that
  ``air-dma-to-channel`` can hoist it into a channel with a repeat count. The
  whole vector is the same for every core and every iteration; staging it in a
  memtile would buy nothing and cost a copy.
* **The compute is an external kernel, not ``ops.dot``.** ``mv.cc``'s
  ``matvec_vectorized_bf16_bf16`` takes three ``i32`` scalars before its
  buffers -- rows, K, and the row offset within the tile -- so it is reached
  through ``air.extern`` rather than through a contraction. A bare
  ``ops.dot`` would emit ``linalg.matvec``, which lowers through
  ``convert-linalg-to-loops`` and comes out scalar; and the
  ``lower_linalg_to_func`` route cannot pass the three scalars. The zero fill
  is the same kernel object for the same reason.

The L2 buffers are flat here -- ``[herd_m * tile_m, k]`` rather than
``[herd_m, tile_m, k]``. Row-major, those are the same bytes; keeping it flat
makes the fill a plain shape-matching transfer and lets a core slice its own
window out with arithmetic on its column index.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend

from air import api as air
from air.api import bf16, i32

# L2 (MemTile) capacity the A and C panels have to fit inside. air.api applies
# its own device-wide bound as well; this one is per-memtile and is the figure
# the kernel registry documents as the binding constraint on tile_m.
L2_CAPACITY = 512 * 1024

# air.api dtypes, keyed by the numpy dtype the harness works in.
DTYPE = {bfloat16: bf16}


def build_module(
    m,
    k,
    tile_m,
    m_input,
    herd_m,
    np_dtype_in,
    np_dtype_out,
    link_with="mv.o",
    target="auto",
):
    """Build the GEMV module. Returns an ``air.ir.Module``.

    Returning a *module* rather than the air.api ``LaunchContext`` is part of
    this function's contract, not an implementation detail: ten call sites in
    ``programming_examples/llms/`` do ``str(build_gemv(...))`` and parse the
    result as MLIR text. Returning the context stringified it as
    ``<air.api._compile.LaunchContext object at 0x...>``, which those parsers
    accepted silently and then failed on with an unrelated TypeError.
    """
    assert (
        m % (tile_m * herd_m) == 0
    ), f"M ({m}) must be divisible by tile_m * herd_m ({tile_m * herd_m})"
    assert (
        tile_m % m_input == 0
    ), f"tile_m ({tile_m}) must be divisible by m_input ({m_input})"
    assert k % 64 == 0, f"K ({k}) must be divisible by 64 (vector width)"

    # Guard MemTile/L2 capacity for staged A and C tiles.
    a_l2_bytes = herd_m * tile_m * k * np.dtype(np_dtype_in).itemsize
    c_l2_bytes = herd_m * tile_m * np.dtype(np_dtype_out).itemsize
    assert a_l2_bytes + c_l2_bytes <= L2_CAPACITY, (
        f"L2 capacity exceeded: A={a_l2_bytes}B + C={c_l2_bytes}B = "
        f"{a_l2_bytes + c_l2_bytes}B > {L2_CAPACITY}B. "
        f"Reduce herd_m ({herd_m}), tile_m ({tile_m}), or k ({k})."
    )

    dt_in, dt_out = DTYPE[np_dtype_in], DTYPE[np_dtype_out]

    # The extent of one L2 staging tile, which the launch steps by: each launch
    # point owns herd_m * tile_m output rows. Named for what it dimensions
    # rather than for the hierarchy level that consumes it -- air.launch,
    # air.segment and air.herd each own a separate iteration space.
    l2_m = tile_m * herd_m

    # The kernel and the fill live in the same object file and are linked into
    # the herd by air.extern. The three leading i32s are rows, K and the row
    # offset within the C tile; the last of those is a loop variable, which
    # air.extern index_casts.
    matvec = air.extern(
        "matvec_vectorized_bf16_bf16", object=link_with, scalars=[i32, i32, i32]
    )
    # No scalar: mv.cc defines `void linalg_fill_bf16(bfloat16 *c_out)`, which
    # takes the buffer alone and gets its extent from -DDIM_M_OUTPUT. The
    # raw-bindings predecessor declared it `(bf16, memref)` and passed a zero
    # constant the callee never read -- harmless, since the C ABI drops the
    # extra leading argument, but the declaration did not describe the symbol.
    fill = air.extern("linalg_fill_bf16", object=link_with)

    A = air.tensor([m, k], dt_in)
    B = air.tensor([k], dt_in)
    C = air.tensor([m], dt_out)

    with air.launch([range(0, m, l2_m)], name="matvec_bf16") as launch:

        @launch.body
        def _(li):
            with air.segment(name="matvec_bf16_0") as seg:

                @seg.body
                def _():
                    row = li * l2_m

                    # L3 -> L2: the A panel for this chunk, and the C panel it
                    # will drain into. Flat, so each is exactly the L3 region it
                    # holds; a core takes its own window with tx * tile_m.
                    l2_a = air.alloc([l2_m, k], dt_in, scope=seg.private())
                    l2_c = air.alloc([l2_m], dt_out, scope=seg.private())

                    air.ops.load(l2_a, A[row : row + l2_m, :])

                    # shape=(herd_m,) rather than letting air.api pick: an
                    # herd_m wider than the part has columns should fail in the
                    # placer, as it did before, not silently strip-mine onto
                    # fewer cores and run at a fraction of the speed.
                    with air.herd([range(herd_m)], name="herd_0", shape=(herd_m,)) as h:

                        @h.body
                        def _(tx):
                            l1_a = air.alloc([m_input, k], dt_in, scope=h.private())
                            l1_b = air.alloc([k], dt_in, scope=h.private())
                            l1_c = air.alloc([tile_m], dt_out, scope=h.private())

                            fill(l1_c)

                            base = tx * tile_m
                            for j in air.sequential(0, tile_m, m_input):
                                # B is the same vector for every core and every
                                # trip; loading it here is what lets
                                # air-dma-to-channel give it a repeat count.
                                air.ops.load(l1_b, B[:])
                                air.ops.load(
                                    l1_a, l2_a[base + j : base + j + m_input, :]
                                )
                                matvec(m_input, k, j, l1_a, l1_b, l1_c)

                            air.ops.store(l1_c, l2_c[base : base + tile_m])

                    air.ops.store(l2_c, C[row : row + l2_m])

    return launch.build(target=target)


if __name__ == "__main__":
    # Default values (M=2048, K=8192, 4 AIE columns)
    M = 2048
    K = 8192
    TILE_M = 4
    M_INPUT = 1
    HERD_M = 4
    INPUT_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="matvec.py",
        description="Builds, runs, and tests the bf16 matrix-vector multiplication (GEMV) example",
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
        "--m",
        type=int,
        default=M,
        help="M dimension (matrix rows / output size)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="K dimension (matrix columns / vector length)",
    )
    parser.add_argument(
        "--tile-m",
        "--tile-m-l2",  # backward compat alias
        type=int,
        default=TILE_M,
        dest="tile_m",
        help="Number of output rows per tile per column",
    )
    parser.add_argument(
        "--m-input",
        type=int,
        default=M_INPUT,
        help="Number of matrix rows per kernel call",
    )
    parser.add_argument(
        "--herd-m",
        type=int,
        default=HERD_M,
        help="Number of AIE columns (parallel compute tiles along M dimension)",
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
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-and-xclbin"],
        dest="compile_mode",
        default="compile-and-run",
        help="compile-and-run (default): compile and validate; compile-and-xclbin: generate xclbin only",
    )
    parser.add_argument(
        "--debug-ir",
        action="store_true",
        dest="debug_ir",
        help="Emit IR after each pass into debug_ir/ directory",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after 10 warmup) and "
        "print Latency + GFLOPs in addition to the correctness check",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )

    args = parser.parse_args()

    if args.perf_iters < 0:
        parser.error("--perf-iters must be >= 0")

    mlir_module = build_module(
        args.m,
        args.k,
        args.tile_m,
        args.m_input,
        args.herd_m,
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
        target=args.target,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        np.random.seed(42)
        input_a = (np.random.randn(args.m, args.k) * 4).astype(INPUT_DATATYPE)
        input_b = (np.random.randn(args.k) * 4).astype(INPUT_DATATYPE)
        output_c = np.dot(
            input_a.astype(np.float32), input_b.astype(np.float32)
        ).astype(OUTPUT_DATATYPE)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            runtime_loop_tiling_sizes=[4, 4],
            output_format=args.output_format,
            instance_name="matvec_bf16",
            debug_ir=args.debug_ir,
            use_lock_race_condition_fix=True,
            report_precision=True,
            n_perf_iters=args.perf_iters,
            perf_flops=((2.0 * args.m * args.k) if args.perf_iters > 0 else None),
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                expected_outputs=[output_c],
                rtol=1.6e-2,
                atol=1e-3,
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            runtime_loop_tiling_sizes=[4, 4],
            use_lock_race_condition_fix=True,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
