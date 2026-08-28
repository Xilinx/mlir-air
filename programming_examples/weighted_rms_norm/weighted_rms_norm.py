# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Weighted RMS normalisation over [M, N], on air.api.

Row by row, following the GPU/PyTorch standard (torch rms_norm_composite /
HF LlamaRMSNorm):

    ms   = sum(x * x) / N                   accumulated in f32
    rstd = rsqrt(ms + eps)                  scalar, f32
    y    = x * rstd * weight

The reduction and the rsqrt run in f32 because that is where the accuracy
lives; the per-element epilogue runs in bf16 vectors, because the AIE vector
unit does not legalize f32 vector elementwise mul. `ops.cast` marks the
boundary, so the region each step runs in is visible in the expression.

N is 2048 by default -- LLAMA's embedding dim -- so the reduction is read in
vector-width steps with the partials accumulated through an L1 scratch buffer.
That is the same structure the predecessor wrote by hand, and the reason it is
not one 2048-lane `vector.reduction`.

Two things the predecessor needed and this does not: the L1 scratch it used to
break the `mulf`->`addf` def-use chain that aievec rejects (the multiply here
feeds a `vector.reduction`, not an add), and the hand-built `vector.broadcast`
of the scalar `rstd` (a [1, 1] buffer against a [1, N] row is numpy
broadcasting, which the DSL lowers to `memref.load` + `vector.broadcast`).

`build_module` returns the **module**, not the launch, unlike most converted
examples: `llms/llama32_1b_int4/multi_launch_builder/rms_gemms_rope_bfp16_multi`
imports it and stitches the result, so the signature and the return type are a
contract.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

EPS = 1e-5


def build_launch(M, N, dtype=bf16, vector=16, herd_x=1):
    """The launch; `build_module` wraps this and returns the module."""
    if vector and N % vector:
        raise ValueError(f"N ({N}) must be divisible by the vector width ({vector})")
    if M % herd_x:
        raise ValueError(
            f"M ({M}) must be divisible by herd_x ({herd_x}): every tile takes "
            "the same number of rows, and there is no remainder path."
        )
    rows_per_tile = M // herd_x

    X = air.tensor([M, N], dtype)
    W = air.tensor([N], dtype)
    Y = air.tensor([M, N], dtype)

    with air.launch(name="weighted_rms_norm") as launch:

        @launch.body
        def _():
            with air.herd([range(herd_x)], name="herd_0", shape=(herd_x,)) as herd:

                @herd.body
                def _(tx):
                    row = air.alloc([1, N], dtype, scope=herd.private(), vector=vector)
                    out = air.alloc([1, N], dtype, scope=herd.private(), vector=vector)
                    # Shared by every row of this tile, so it is fetched once.
                    weight = air.alloc(
                        [1, N], dtype, scope=herd.private(), vector=vector
                    )
                    acc = air.alloc([1, 1], f32, scope=herd.private(), vector=vector)
                    rstd = air.alloc([1, 1], dtype, scope=herd.private(), vector=vector)

                    ops.load(weight, W[:])

                    for it in air.sequential(rows_per_tile):
                        r = it + tx * rows_per_tile
                        ops.load(row, X[r : r + 1, :])

                        # The square is bf16 and the accumulation f32, which is
                        # the predecessor's split.
                        acc[:] = ops.reduce_add(ops.cast(row[:] * row[:], f32))
                        rstd[:] = ops.cast(ops.rsqrt(acc[:] * (1.0 / N) + EPS), dtype)

                        out[:] = row[:] * rstd[:] * weight[:]

                        ops.store(out, Y[r : r + 1, :])

    return launch


def build_module(M, N, np_dtype=bfloat16, vector_size=16, herd_x=1, target="npu2"):
    """The MLIR module. Signature and return type are the llms/ builders' contract."""
    if np_dtype is not bfloat16:
        raise NotImplementedError(
            f"weighted_rms_norm is bf16 only, got {np_dtype!r}: the epilogue "
            "runs in bf16 vectors because the AIE vector unit does not legalize "
            "f32 vector elementwise mul."
        )
    return build_launch(M, N, bf16, vector_size, herd_x).build(target=target)


def rms_norm_reference(x, weight, eps=1e-5):
    """CPU F32 reference for weighted RMS norm."""
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * weight.astype(np.float32)).astype(x.dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weighted RMS Normalization — multi-tile with profiling",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Profile kernel execution"
    )
    parser.add_argument(
        "--M", type=int, default=2048, help="Rows (default: LLAMA seq_len)"
    )
    parser.add_argument(
        "--N", type=int, default=2048, help="Cols (default: LLAMA emb_dim)"
    )
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument(
        "--target",
        type=str,
        default="npu2",
        help="NPU generation to build for (npu2; the epilogue is AIE2P bf16)",
    )
    parser.add_argument(
        "--herd-x",
        type=int,
        default=1,
        help="Number of tiles (1=original, 8=multi-tile)",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Profiling iterations"
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after 10 warmup) and "
        "print Latency in addition to the correctness check",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
    )
    args = parser.parse_args()

    M, N = args.M, args.N
    herd_x = args.herd_x if hasattr(args, "herd_x") else 1
    print(f"Weighted RMSNorm: M={M}, N={N}, herd=[{herd_x},1]")

    mlir_module = build_module(
        M, N, bfloat16, args.vector_size, herd_x=herd_x, target=args.target
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x_input = np.random.randn(M, N).astype(bfloat16)
    weight = np.random.randn(N).astype(bfloat16)
    y_expected = rms_norm_reference(x_input, weight)

    # Function signature is (input, weight, output) for both single- and
    # multi-tile modes. Per-tile intermediate buffers in the multi-tile path
    # are allocated internally (in L1) and not exposed at the L3 boundary.

    if args.profile:
        import time
        import pyxrt as xrt
        import filelock

        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="weighted_rms_norm",
        )
        artifact = backend.compile(mlir_module)

        # Hold the NPU lock for the entire NPU-touching scope: load, BO
        # setup, warmup, and the timed loop. Keeping load() under a separate
        # lock would let other processes interleave on the device during the
        # measurement and pollute the timings.
        with filelock.FileLock("/tmp/npu.lock"):
            backend.load(artifact)

            out_buf = np.zeros((M, N), dtype=bfloat16)
            inputs = [x_input, weight, out_buf]
            sizes = [a.size * a.itemsize for a in inputs]
            bos = [
                xrt.bo(
                    backend.device, s, xrt.bo.host_only, backend.kernel.group_id(i + 3)
                )
                for i, s in enumerate(sizes)
            ]

            # Warmup
            for i, a in enumerate(inputs):
                bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            backend.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
            h.wait()

            times = []
            for _ in range(args.iterations):
                for i, a in enumerate(inputs):
                    bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
                    bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                t0 = time.perf_counter()
                h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
                h.wait()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        backend.unload()

        # Profile mode reports timing/bandwidth only. Correctness is covered
        # by `make run` / the compile-and-run path's correlation check.
        data_mb = (M * N * 2 * 2 + N * 2) / 1e6  # 2 matrices + 1 weight vector
        print(
            f"\n  Kernel: avg={np.mean(times):.1f}ms  min={np.min(times):.1f}ms  max={np.max(times):.1f}ms"
        )
        print(f"  Bandwidth: {data_mb / (np.min(times)/1000) / 1000:.2f} GB/s")

    elif args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="weighted_rms_norm",
            runtime_loop_tiling_sizes=[4, 4],
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[x_input, weight],
                expected_outputs=[y_expected],
                rtol=1.6e-2,
                atol=5e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
