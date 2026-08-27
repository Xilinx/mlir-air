# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Causal depthwise 1-D convolution, on air.api.

    y[s, c] = sum over t of x_pad[s + t, c] * w[t, c]        t = 0, 1, 2

Depthwise, so every channel has its own 3-tap filter and there is no reduction
across channels. The convolution itself stays in C++:
`air.extern("conv1d_depthwise_bf16", link_with="conv1d_depthwise.o")` declares
the microkernel and stamps link_with on both the declaration and the herd, so
the body here is dataflow only.

The input is **pre-padded** by HALO rows on the host, so `x_pad[row]` is already
the oldest sample feeding `y[row]`: no negative indexing and no masking, just a
read of `tile_s + HALO` rows where the output is `tile_s`. That overlap is the
whole reason the two slices differ.

Weights are sequence-independent, so this tile's channel slice is loaded once
above the sequence loop rather than per trip.

The herd tiles both axes: `tx` picks a channel block and `ty` a slice of the
sequence, with the sequence loop striding by `tile_s * herd_y`. Both offsets are
ordinary Python arithmetic on the coordinates,

    chan = tx * tile_c
    row  = iv + ty * tile_s

where the predecessor built each as its own AffineMap.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

K_TAPS = 3
HALO = K_TAPS - 1  # 2 rows of left context
L1_BYTES = 65536  # per-compute-tile L1 on AIE2P


def build_module(seq, channels, tile_s, np_dtype_in=bfloat16, herd_x=8, herd_y=1):
    assert (
        channels % herd_x == 0
    ), f"channels ({channels}) must be divisible by herd_x ({herd_x})"
    tile_c = channels // herd_x

    # MEASURED-BAD CONFIGS (NPU2, swept on an idle device). Both are silent or
    # near-silent, so they are rejected here rather than left to the caller.
    #
    # herd_x == 2 is broken: at tile_s=8 it COMPILES CLEANLY AND RETURNS WRONG
    # RESULTS (mean_rel_L1 = 5.0e-1 vs the correct 2.8e-3), and at tile_s=4 it
    # fails inside aircc. This is NOT the L1 budget -- 2x1/ts=8 needs only
    # 42 KB of the 64 KB L1, and both herd_x=1 (52 KB) and herd_x=4 (37 KB)
    # are correct. The bad axis is herd_x=2 itself, non-monotonically.
    assert herd_x != 2, (
        "herd_x=2 is measured-broken for this kernel on NPU2: it either fails "
        "in aircc (tile_s=4) or COMPILES CLEANLY AND RETURNS WRONG RESULTS "
        "(tile_s=8, mean_rel_L1=5.0e-1). It is not an L1 issue -- herd_x=1 and "
        "herd_x=4 both pass at larger L1 footprints. Use herd_x in {1, 4, 8}."
    )
    #
    # herd_y > 1 is single-shot only: it places, and a SINGLE invocation is
    # numerically correct (2.813e-3, identical to herd_y=1), but the SECOND
    # invocation deadlocks (ERT_CMD_STATE_TIMEOUT) -- reproduced with as few as
    # 2 perf iterations. Any real deployment invokes this kernel once per conv
    # layer (10x per prefill for LFM2-1.2B), so a single-shot config is
    # unusable. Suspected cause: the weight DMA is hoisted outside the sequence
    # loop, so on re-invocation the extra herd row's weight producer never
    # re-fires -- unproven, so do not treat that mechanism as established.
    assert herd_y == 1, (
        f"herd_y={herd_y} is single-shot only on NPU2: one invocation is "
        f"numerically correct, the SECOND deadlocks (ERT_CMD_STATE_TIMEOUT, "
        f"reproduced at 2 iterations). Production invokes this kernel in a "
        f"loop, so only herd_y=1 is usable. Gate any change to this with a "
        f"REPEATED-invocation run (`make profile`), never a single `make run`."
    )
    assert (
        tile_c % 16 == 0
    ), f"tile_c ({tile_c}) must be a multiple of the 16-lane vector"
    assert (
        seq % (tile_s * herd_y) == 0
    ), f"seq ({seq}) must be divisible by tile_s*herd_y ({tile_s * herd_y})"

    # L1 BUDGET GUARD. Each tile holds three live buffers:
    #   x halo window (tile_s + 2) x tile_c, weights 3 x tile_c, out tile_s x tile_c
    # Overflowing the 64 KB compute-tile L1 does NOT reliably fail the build --
    # measured on NPU2, herd_x=4 / tile_s=32 (70.6 KB) compiles cleanly and then
    # returns SILENTLY WRONG results, while herd_x<=2 at the same tile_s fails
    # inside aircc. Assert here so the corrupting configs are rejected up front
    # rather than producing a plausible-looking wrong answer.
    l1_bytes = (
        (tile_s + HALO) * tile_c + K_TAPS * tile_c + tile_s * tile_c
    ) * np.dtype(np_dtype_in).itemsize
    assert l1_bytes <= L1_BYTES, (
        f"L1 over-allocation: herd_x={herd_x} (tile_c={tile_c}) with "
        f"tile_s={tile_s} needs {l1_bytes / 1024:.1f} KB > {L1_BYTES // 1024} KB. "
        f"Reduce tile_s to <= {(L1_BYTES // (np.dtype(np_dtype_in).itemsize * tile_c) - HALO - K_TAPS) // 2} "
        f"or increase herd_x. NOTE: this config may still COMPILE and return "
        f"silently wrong results -- do not remove this assert."
    )

    dtype = bf16

    # x is pre-padded by HALO rows; w is tap-major.
    X = air.tensor([seq + HALO, channels], dtype)
    W = air.tensor([K_TAPS, channels], dtype)
    Y = air.tensor([seq, channels], dtype)

    conv = air.extern(
        "conv1d_depthwise_bf16",
        link_with="conv1d_depthwise.o",
        scalars=[i32, i32],
    )

    with air.launch(name="conv1d_depthwise") as launch:

        @launch.body
        def _():
            with air.segment(name="conv1d_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(herd_y)],
                        name="herd_0",
                        shape=(herd_x, herd_y),
                    ) as herd:

                        @herd.body
                        def _(tx, ty):
                            l1_x = air.alloc(
                                [tile_s + HALO, tile_c], dtype, scope=herd.private()
                            )
                            l1_w = air.alloc(
                                [K_TAPS, tile_c], dtype, scope=herd.private()
                            )
                            l1_y = air.alloc(
                                [tile_s, tile_c], dtype, scope=herd.private()
                            )

                            chan = tx * tile_c

                            # Sequence-independent, so once per tile rather
                            # than once per trip.
                            ops.load(l1_w, W[:, chan : chan + tile_c])

                            for iv in air.sequential(0, seq, tile_s * herd_y):
                                row = iv + ty * tile_s

                                # tile_s + HALO in, tile_s out: the overlap is
                                # the taps' history.
                                ops.load(
                                    l1_x,
                                    X[row : row + tile_s + HALO, chan : chan + tile_c],
                                )

                                conv(l1_x, l1_w, l1_y, tile_s, tile_c)

                                ops.store(
                                    l1_y, Y[row : row + tile_s, chan : chan + tile_c]
                                )

    return launch


def conv1d_reference(x_pad, w_tapmajor, seq, channels):
    """FP32 reference for the causal depthwise conv.

    Mirrors the kernel exactly: bf16 inputs upcast to f32, the 3 tap products
    accumulated in f32, result cast back to bf16 — the same standard a GPU /
    HF depthwise-conv op is verified against.

    Args:
        x_pad: (seq+2, channels) bf16, pre-padded input.
        w_tapmajor: (3, channels) bf16 taps, oldest-tap-first.
        seq, channels: output dims.

    Returns:
        (seq, channels) bf16.
    """
    x32 = x_pad.astype(np.float32)
    w32 = w_tapmajor.astype(np.float32)
    acc = np.zeros((seq, channels), dtype=np.float32)
    for j in range(K_TAPS):
        acc += x32[j : j + seq] * w32[j]
    return acc.astype(bfloat16)


if __name__ == "__main__":
    # LFM2-1.2B ShortConv prefill scale: seq=2048, conv_dim=2048.
    SEQ = 2048
    CHANNELS = 2048
    TILE_S = 32

    parser = argparse.ArgumentParser(
        prog="conv1d_depthwise.py",
        description=(
            "Builds, runs, and tests the standalone causal depthwise 1-D "
            "convolution kernel (k=3, the LFM2 ShortConv convolution)"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--seq", type=int, default=SEQ, help="Sequence length")
    parser.add_argument(
        "--channels", type=int, default=CHANNELS, help="Channels (conv_dim)"
    )
    parser.add_argument(
        "--tile-s", type=int, default=TILE_S, help="Sequence rows per tile iteration"
    )
    parser.add_argument(
        "--herd-x",
        type=int,
        default=8,
        help="Herd x dimension (AIE columns) — splits the CHANNEL axis",
    )
    parser.add_argument(
        "--herd-y",
        type=int,
        default=1,
        help="Herd y dimension (AIE rows) — splits the SEQUENCE axis",
    )
    parser.add_argument(
        "--zero-pad",
        action="store_true",
        help="Zero the 2 leading (conv-state) rows — the prefill / "
        "start-of-sequence case. Default is a NONZERO random pad, i.e. the "
        "decode case where the pad is the state carried from the previous "
        "chunk; that is the general case and the stricter test of tap order.",
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
        args.seq,
        args.channels,
        args.tile_s,
        bfloat16,
        herd_x=args.herd_x,
        herd_y=args.herd_y,
    )
    # build() resolves --target auto, so it runs before launch.target is read.
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    rng = np.random.default_rng(0)
    # N(0,1) activations, and taps at N(0, 0.5) — depthwise conv taps in a
    # trained LFM2 checkpoint are O(0.1-1), so this keeps the accumulate in a
    # realistic range rather than an artificially tiny one.
    x_body = rng.standard_normal((args.seq, args.channels), dtype=np.float32)
    x_pad = np.zeros((args.seq + HALO, args.channels), dtype=np.float32)
    x_pad[HALO:] = x_body
    if not args.zero_pad:
        # DEFAULT: nonzero leading rows — the DECODE case, where the pad is the
        # conv state carried from the previous chunk. This is the general case;
        # a zero pad (prefill / sequence start) is the special case of it, and
        # is the weaker test: with zeros the tap-0 and tap-1 contributions to
        # the first two outputs vanish, so a wrong tap ORDER can still pass on
        # most rows. A nonzero pad exercises every tap on every row.
        x_pad[:HALO] = rng.standard_normal((HALO, args.channels), dtype=np.float32)
    x_pad = x_pad.astype(bfloat16)
    w = (rng.standard_normal((K_TAPS, args.channels), dtype=np.float32) * 0.5).astype(
        bfloat16
    )

    expected = conv1d_reference(x_pad, w, args.seq, args.channels)

    if args.compile_mode == "compile-and-run":
        # bf16 depthwise conv: rtol = canonical bf16 1.6e-2. atol = 5e-2,
        # matching the other f32-accumulate elementwise kernels (RoPE,
        # Element-wise Add) — this is a 3-term f32 accumulation with a single
        # bf16 output rounding, the cleanest tier, so it needs no SiLU-style
        # transcendental slack.
        runner = XRTRunner(
            target_device=launch.target,
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="conv1d_depthwise",
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[x_pad, w],
                expected_outputs=[expected],
                rtol=1.6e-2,
                atol=5e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            target_device=launch.target,
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
