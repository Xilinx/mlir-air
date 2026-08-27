# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""RoPE (Rotary Position Embedding) -- half-split, matching HuggingFace Llama, on air.api

Applies rotary position embeddings to a 2D input [rows, head_dim] using the
*half-split* convention (HuggingFace Llama `rotate_half`), pairing
(x[i], x[i + head_dim/2]):

    out[i]        = x[i] * cos[i] - x[i + half] * sin[i]
    out[i + half] = x[i] * sin[i] + x[i + half] * cos[i]

where half = head_dim/2 and the per-row cos/sin are streamed in as a
precomputed look-up table (LUT) in the concatenated half-split layout
    LUT[row] = [cos_0, ..., cos_{half-1}, sin_0, ..., sin_{half-1}]
matching `llama32_1b_weights.py:generate_rope_lut` and the kernel
`rope_halfsplit.cc` that llama-3.2-1B actually links (`external_kernels.py:
compile_rope`). This is **NOT** the interleaved variant in `rope_lut/` or
`rope_sincos/` (those are decoys; their math does not match llama).

The rotation itself stays in C++: `air.extern("rope", link_with="rope.o")`
declares the microkernel and stamps `link_with` on both the declaration and the
herd, so the body here is pure dataflow. That is the point of the example --
the arithmetic is the kernel's, and what AIR contributes is getting one row to
each core and the result back.

Each row's RoPE is independent -- no cross-row dependency -- so rows are spread
across an `herd_x x herd_y` AIE grid. Each tile uses 3 independent shim DMAs
(input in, lut in, output out); NPU2 has 8 shim DMA channels, so the herd is
capped at herd_x * herd_y <= 8 tiles (8x1 / 2x4 place, 8x2 / 4x4 / 8x4 do not).
The best config is herd_x=8, herd_y=1 (full chip width).

Each tile streams its rows one head_dim row per DMA / per kernel call (matching
rope_halfsplit.cc's single-row signature and the llama prefill/decode
builders). `herd_x` (AIE columns) is the scaling knob; the rows are interleaved
across the herd so tile t handles rows t, t+total_tiles, ... A batched-DMA
variant (multiple rows per L3<->L1 transfer with per-row subview kernel calls)
was investigated but the air dependency pass mis-schedules the per-row subview
writes under a single bulk output DMA (half the rows come back zero / NaN at
rows_per_dma=2,4), so the faithful one-row-per-DMA structure is used -- see the
performance notes.

One difference from the raw-bindings version this replaces: the row offset is
written as ordinary Python arithmetic on the coordinates,

    (row + tx * herd_y + ty) * head_dim

where the predecessor built the same expression as a three-symbol `AffineMap`
by hand. It reaches the IR as one `affine.apply` either way.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner


def build_module(rows, head_dim, herd_x=8, herd_y=1, dtype=bf16):
    total = rows * head_dim
    if head_dim % 16:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by 16: that is the "
            "kernel's vector width N."
        )
    total_tiles = herd_x * herd_y
    if rows % total_tiles:
        raise ValueError(
            f"rows ({rows}) must be divisible by herd_x * herd_y "
            f"({total_tiles}): every core takes the same number of rows, and "
            "there is no remainder path."
        )

    src = air.tensor([total], dtype)
    lut = air.tensor([total], dtype)
    dst = air.tensor([total], dtype)

    # rope(input_row, lut_row, output_row, dims) from rope.o. The trailing
    # dims argument is a scalar, so its element type has to be stated: the
    # buffer types come from the buffers, an i32 does not.
    rope = air.extern("rope", link_with="rope.o", scalars=[i32])

    with air.launch(name="rope_halfsplit") as launch:

        @launch.body
        def _():
            with air.segment(name="rope_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(herd_y)],
                        name="herd_0",
                        shape=(herd_x, herd_y),
                    ) as herd:

                        @herd.body
                        def _(tx, ty):
                            l1_in = air.alloc([head_dim], dtype, scope=herd.private())
                            l1_lut = air.alloc([head_dim], dtype, scope=herd.private())
                            l1_out = air.alloc([head_dim], dtype, scope=herd.private())

                            # Strides by total_tiles rows; each core picks its
                            # own row out of the group, so tile t handles rows
                            # t, t + total_tiles, ...
                            for row in air.sequential(0, rows, total_tiles):
                                lo = (row + tx * herd_y + ty) * head_dim

                                ops.load(l1_in, src[lo : lo + head_dim])
                                ops.load(l1_lut, lut[lo : lo + head_dim])

                                rope(l1_in, l1_lut, l1_out, head_dim)

                                ops.store(l1_out, dst[lo : lo + head_dim])

    return launch


def rope_halfsplit_reference(input_flat, lut_flat, head_dim):
    """Full-output FP32 half-split RoPE reference (HuggingFace Llama rotate_half /
    apply_rotary_pos_emb), matching rope_halfsplit.cc.

    LUT row layout: [cos_0..cos_{half-1}, sin_0..sin_{half-1}].
        out[i]        = x[i]*cos[i] - x[i+half]*sin[i]
        out[i+half]   = x[i]*sin[i] + x[i+half]*cos[i]
    bf16 inputs are upcast to f32, rotated, cast back to bf16 (the bf16-rounded
    reference a GPU/HF RoPE op is verified against). This is the half-split math,
    NOT the interleaved decoy in rope_lut/.
    """
    half = head_dim // 2
    x = input_flat.astype(np.float32).reshape(-1, head_dim)
    lut = lut_flat.astype(np.float32).reshape(-1, head_dim)
    cos_v = lut[:, :half]
    sin_v = lut[:, half:]
    x1 = x[:, :half]
    x2 = x[:, half:]
    out = np.empty_like(x)
    out[:, :half] = x1 * cos_v - x2 * sin_v
    out[:, half:] = x1 * sin_v + x2 * cos_v
    return out.astype(input_flat.dtype).flatten()


def generate_rope_lut(rows, head_dim, dtype=bfloat16, theta=500000.0):
    """Generate the half-split [cos..., sin...] RoPE LUT (concatenated layout),
    matching llama32_1b_weights.py:generate_rope_lut and rope_halfsplit.cc.

    For position pos and dimension index i (i < head_dim/2):
        freq_i = 1 / (theta ^ (2*i / head_dim));  angle = pos * freq_i
        LUT[pos, i]        = cos(angle)
        LUT[pos, i + half] = sin(angle)

    theta default 500000.0 = llama-3.2 rope_base.
    """
    half = head_dim // 2
    i_vals = np.arange(half, dtype=np.float64)
    freqs = 1.0 / (theta ** (2.0 * i_vals / head_dim))
    pos = np.arange(rows, dtype=np.float64)
    angles = np.outer(pos, freqs)  # (rows, half)
    lut = np.empty((rows, head_dim), dtype=np.float32)
    lut[:, :half] = np.cos(angles)
    lut[:, half:] = np.sin(angles)
    return lut.astype(dtype)


if __name__ == "__main__":
    ROWS = 65536  # prefill RoPE Q: n_heads(32) * seq(2048)
    HEAD_DIM = 64  # llama-3.2-1B head_dim
    THETA = 500000.0  # llama-3.2 rope_base
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="rope_halfsplit.py",
        description="Builds, runs, and tests the standalone half-split RoPE kernel "
        "(HuggingFace Llama convention, matching rope_halfsplit.cc)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--rows", type=int, default=ROWS, help="Number of rows")
    parser.add_argument(
        "--head-dim", type=int, default=HEAD_DIM, help="Head dimension (per-row width)"
    )
    parser.add_argument(
        "--herd-x",
        type=int,
        default=8,
        help="Herd x dimension (AIE columns, default: 8 — full chip width)",
    )
    parser.add_argument(
        "--herd-y",
        type=int,
        default=1,
        help="Herd y dimension (AIE rows, default: 1). NPU2 caps the herd at "
        "herd_x * herd_y <= 8 tiles (3 shim DMAs/tile, 8 shim channels)",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after warmup) and "
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

    launch = build_module(
        args.rows,
        args.head_dim,
        herd_x=args.herd_x,
        herd_y=args.herd_y,
    )
    # build() resolves --target auto to the installed generation, so it has to
    # run before launch.target is read.
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # randn input (matching the GPU/HF reference and other registry kernels).
    rng = np.random.default_rng(0)
    input_data = rng.standard_normal(
        (args.rows, args.head_dim), dtype=np.float32
    ).astype(INPUT_DATATYPE)
    lut = generate_rope_lut(args.rows, args.head_dim, INPUT_DATATYPE, THETA)

    # Reference: full-output FP32 half-split rotate (HF rotate_half), cast bf16.
    expected = rope_halfsplit_reference(
        input_data.flatten(), lut.flatten(), args.head_dim
    )

    if args.compile_mode == "compile-and-run":
        # bf16 half-split RoPE: rtol = canonical bf16 1.6e-2; atol set from the
        # measured worst-case single-element error (see kernel_registry/details/
        # RoPE_bf16.md). RoPE is a handful of bf16 mul + one add/sub per element
        # with NO reduction, so it is near the cleanest tier (close to EltwiseAdd).
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope_halfsplit",
            report_precision=True,
            n_perf_iters=args.perf_iters,
            target_device=launch.target,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_data.flatten(), lut.flatten()],
                expected_outputs=[expected],
                rtol=1.6e-2,
                atol=5e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            target_device=launch.target,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
