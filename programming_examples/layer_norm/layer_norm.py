# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Affine LayerNorm over [M, N], on air.api.

Row by row, following the GPU/PyTorch standard:

    mean = sum(x) / N                       accumulated in f32
    var  = sum((x - mean)^2) / N            accumulated in f32
    rstd = rsqrt(var + eps)                 scalar, f32
    y    = (x - mean) * rstd * weight + bias

The two reductions and the rsqrt run in f32 because that is where LayerNorm's
accuracy lives; the per-element epilogue runs in bf16 vectors, because the AIE
vector unit does not legalize f32 vector elementwise ops. `ops.cast` marks that
boundary, and everything below a cast is read and computed in the source type,
so the region each step runs in is visible in the expression rather than left to
a convention.

**weight and bias share one flat [2N] buffer**, `[0:N]` weight and `[N:2N]`
bias, because AIE per-tile routing caps at about three L3 streams and
in + weight + bias + out is four. The halves are read straight out of it:

    out[:] = (row[:] - mean[:]) * rstd[:] * param[0:N] + param[N : 2 * N]

Each half is a plain region of the packed buffer, which is an ordinary
elementwise operand, so the packing costs nothing to read. The predecessor
carved the same two halves out with `memref.subview` plus a hand-added
`arith.addi(j, N)`.

`mean` and `rstd` are [1, 1] buffers multiplied against a [1, N] row -- numpy
broadcasting along the innermost axis, which the DSL lowers to `memref.load`
plus `vector.broadcast` where the predecessor built the broadcast by hand and
threaded it through three loop nests.

N is 768 by default, far longer than one vector, so both reductions are read in
vector-width steps with the partials accumulated through an L1 scratch buffer --
the same structure the predecessor wrote by hand, and the reason it is not one
768-lane `vector.reduction` (that does not legalize).

`build_module` returns the **module**, not the launch, unlike the other
converted examples: smolvla's vision builders do
`str(build_layer_norm(seq_len, emb_dim, bfloat16, 16, herd_x=8))`, so the
signature and the return type are a contract.
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
    # weight || bias, one DMA rather than two.
    PARAM = air.tensor([2 * N], dtype)
    Y = air.tensor([M, N], dtype)

    with air.launch(name="layer_norm") as launch:

        @launch.body
        def _():
            with air.herd([range(herd_x)], name="herd_0", shape=(herd_x,)) as herd:

                @herd.body
                def _(tx):
                    row = air.alloc([1, N], dtype, scope=herd.private(), vector=vector)
                    out = air.alloc([1, N], dtype, scope=herd.private(), vector=vector)
                    # Shared by every row of this tile, so it is fetched once,
                    # above the loop.
                    param = air.alloc(
                        [2 * N], dtype, scope=herd.private(), vector=vector
                    )
                    acc = air.alloc([1, 1], f32, scope=herd.private(), vector=vector)
                    mean = air.alloc([1, 1], dtype, scope=herd.private(), vector=vector)
                    rstd = air.alloc([1, 1], dtype, scope=herd.private(), vector=vector)

                    ops.load(param, PARAM[:])

                    for it in air.sequential(rows_per_tile):
                        r = it + tx * rows_per_tile
                        ops.load(row, X[r : r + 1, :])

                        # mean, accumulated in f32
                        acc[:] = ops.reduce_add(ops.cast(row[:], f32))
                        mean[:] = ops.cast(acc[:] * (1.0 / N), dtype)

                        # variance: the difference and its square are bf16 and
                        # the accumulation f32 -- the predecessor's split.
                        acc[:] = ops.reduce_add(
                            ops.cast((row[:] - mean[:]) * (row[:] - mean[:]), f32)
                        )
                        rstd[:] = ops.cast(ops.rsqrt(acc[:] * (1.0 / N) + EPS), dtype)

                        out[:] = (row[:] - mean[:]) * rstd[:] * param[0:N] + param[
                            N : 2 * N
                        ]

                        ops.store(out, Y[r : r + 1, :])

    return launch


def build_module(M, N, np_dtype=bfloat16, vector_size=16, herd_x=1, target="npu2"):
    """The MLIR module. Signature and return type are smolvla's contract."""
    if np_dtype is not bfloat16:
        raise NotImplementedError(
            f"layer_norm is bf16 only, got {np_dtype!r}: the epilogue runs in "
            "bf16 vectors because the AIE vector unit does not legalize f32 "
            "vector elementwise ops."
        )
    return build_launch(M, N, bf16, vector_size, herd_x).build(target=target)


def layer_norm_reference(x, weight, bias, eps=EPS):
    """CPU F32 reference for affine layer norm (HF / PyTorch nn.LayerNorm)."""
    x_f32 = x.astype(np.float32)
    mean = np.mean(x_f32, axis=-1, keepdims=True)
    variance = np.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(variance + eps)
    y = (x_f32 - mean) * rstd * weight.astype(np.float32) + bias.astype(np.float32)
    return y.astype(x.dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the affine layer normalization example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--M", type=int, default=1024, help="Rows (default: SigLIP num patches)"
    )
    parser.add_argument(
        "--N", type=int, default=768, help="Cols (default: SigLIP hidden dim)"
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
        help="Number of tiles (1=single-tile, 8=multi-tile full chip width)",
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
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    M, N = args.M, args.N
    herd_x = args.herd_x
    print(f"LayerNorm (affine): M={M}, N={N}, herd=[{herd_x},1]")

    mlir_module = build_module(
        M, N, bfloat16, args.vector_size, herd_x=herd_x, target=args.target
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    x_input = np.random.randn(M, N).astype(bfloat16)
    weight = np.random.randn(N).astype(bfloat16)
    bias = np.random.randn(N).astype(bfloat16)
    # Pack weight (gamma) and bias (beta) into a single flat [2N] buffer so the
    # kernel needs only ONE param DMA channel per tile (see build_module).
    param = np.concatenate([weight, bias]).astype(bfloat16)
    y_expected = layer_norm_reference(x_input, weight, bias)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="layer_norm",
            runtime_loop_tiling_sizes=[4, 4],
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[x_input, param],
                expected_outputs=[y_expected],
                rtol=1.6e-2,
                # atol=6e-2 (vs RMSNorm's 5e-2): LayerNorm's epilogue has one
                # extra bf16 rounding step — the bias add in
                # (x-mean)*rstd*weight + bias — so the worst-case bf16 *output*
                # granularity is ~4 ULP instead of ~3. Not a reduction-precision
                # relaxation; the reductions are FP32 (mean_rel_L1 ~4.4e-3).
                atol=6e-2,
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
