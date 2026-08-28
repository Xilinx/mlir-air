# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""RoPE (Rotary Position Embedding) from a precomputed LUT, on air.api.

Applies rotary position embeddings to [seq_len, embed_dim] using the
*interleaved* convention, pairing (x[2i], x[2i+1]) against a LUT laid out
[cos_0, sin_0, cos_1, sin_1, ...]. That is not the convention llama uses --
`rope_halfsplit/` is -- so this is the general-purpose variant, and the two are
deliberately kept apart.

The rotation stays in C++: `air.extern("rope", link_with="rope.o")` declares the
microkernel and stamps link_with on both the declaration and the herd, so the
body here is dataflow only. Rows are handed out in contiguous blocks -- tile t
takes rows [t * rows_per_tile, (t+1) * rows_per_tile) -- one embed_dim row per
DMA and per kernel call, matching the kernel's single-row signature.

`generate_lut` and `rope_reference` below are unchanged and stay module-level:
the llms/ shared builders import `generate_lut` from here.

One difference from the raw-bindings version this replaces: the row offset is
ordinary Python arithmetic on the coordinate,

    (row + tx * rows_per_tile) * embed_dim

where the predecessor built the same expression as a two-symbol AffineMap. It
reaches the IR as one affine.apply either way.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner


def build_module(seq_len, embed_dim, herd_x=1, dtype=bf16):
    total = seq_len * embed_dim
    if embed_dim % 16:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by 16: that is the "
            "kernel's vector width."
        )
    if seq_len % herd_x:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by herd_x ({herd_x}): every "
            "tile takes the same number of rows, and there is no remainder path."
        )
    rows_per_tile = seq_len // herd_x

    src = air.tensor([total], dtype)
    lut = air.tensor([total], dtype)
    dst = air.tensor([total], dtype)

    # rope(input_row, lut_row, output_row, dims). The trailing dims argument is
    # a scalar, so its element type has to be stated.
    rope = air.extern("rope", link_with="rope.o", scalars=[i32])

    with air.launch(name="rope_lut") as launch:

        @launch.body
        def _():
            with air.herd([range(herd_x)], name="herd_0", shape=(herd_x,)) as herd:

                @herd.body
                def _(tx):
                    l1_in = air.alloc([embed_dim], dtype, scope=herd.private())
                    l1_lut = air.alloc([embed_dim], dtype, scope=herd.private())
                    l1_out = air.alloc([embed_dim], dtype, scope=herd.private())

                    for row in air.sequential(rows_per_tile):
                        lo = (row + tx * rows_per_tile) * embed_dim

                        ops.load(l1_in, src[lo : lo + embed_dim])
                        ops.load(l1_lut, lut[lo : lo + embed_dim])

                        rope(l1_in, l1_lut, l1_out, embed_dim)

                        ops.store(l1_out, dst[lo : lo + embed_dim])

    return launch


def rope_reference(input_data, lut, embed_dim):
    """CPU F32 reference for RoPE with precomputed LUT (vectorized)."""
    x = input_data.astype(np.float32).reshape(-1, embed_dim)
    l = lut.astype(np.float32).reshape(-1, embed_dim)
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    cos_v = l[:, 0::2]
    sin_v = l[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos_v - x_odd * sin_v
    out[:, 1::2] = x_even * sin_v + x_odd * cos_v
    return out.astype(input_data.dtype)


def generate_lut(seq_len, embed_dim, dtype=bfloat16, theta=10000.0):
    """Generate interleaved [cos, sin, cos, sin, ...] RoPE LUT (vectorized)."""
    i_vals = np.arange(embed_dim // 2, dtype=np.float64)
    freqs = 1.0 / (theta ** (2.0 * i_vals / embed_dim))
    rows = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(rows, freqs)  # (seq_len, embed_dim//2)
    lut = np.empty((seq_len, embed_dim), dtype=np.float32)
    lut[:, 0::2] = np.cos(angles)
    lut[:, 1::2] = np.sin(angles)
    return lut.astype(dtype)


if __name__ == "__main__":
    THETA = 10000.0

    parser = argparse.ArgumentParser(
        description="RoPE (LUT-based) — build, run, profile",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=64, help="Number of rows")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--herd-x",
        type=int,
        default=1,
        help="Number of tiles (1=single, 8=multi-tile)",
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

    seq_len = args.seq_len
    embed_dim = args.embed_dim
    herd_x = args.herd_x
    print(f"RoPE LUT: seq_len={seq_len}, embed_dim={embed_dim}, herd=[{herd_x},1]")

    launch = build_module(seq_len, embed_dim, herd_x=herd_x)
    # build() resolves --target auto, so it runs before launch.target is read.
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        np.random.seed(0)
        input_data = np.random.uniform(-4.0, 4.0, (seq_len, embed_dim)).astype(bfloat16)
        lut = generate_lut(seq_len, embed_dim, bfloat16, THETA)
        y_expected = rope_reference(input_data, lut, embed_dim)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope",
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_data.flatten(), lut.flatten()],
                expected_outputs=[y_expected.flatten()],
                rtol=5e-2,
                atol=5e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
