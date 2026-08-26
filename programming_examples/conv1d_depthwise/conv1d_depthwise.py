# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Causal Depthwise 1-D Convolution Kernel (kernel size 3)

The convolution inside LFM2's `Lfm2ShortConv` operator:

    y[t, c] = w[0, c]*x[t+0, c] + w[1, c]*x[t+1, c] + w[2, c]*x[t+2, c]

Depthwise — each channel has its own 3 taps, no cross-channel mixing — so the
channel axis is the vectorization axis and is contiguous in memory.

**Causality is expressed by pre-padding, not by masking.** The input `x` has
`seq + 2` rows and the output `y` has `seq`: row `t` of `x` is the sample at
original position `t - 2`, so `x` row `t` is the oldest sample feeding `y[t]`
and pairs with tap 0 (oldest-first, matching `nn.Conv1d` cross-correlation
over a left-padded input). The two leading rows are the **conv state**: zeros
at the start of a sequence (prefill) or the carried tail of the previous chunk
(decode). Prefill and decode are therefore the same kernel with a different
pad — no separate decode variant.

`w` is passed TAP-MAJOR, shape `(3, C)`, so each tap's channel slice is
contiguous (HF stores it channel-major as `(C, 1, 3)`; the host transposes
once at load).

Decomposition: the herd splits the **channel** axis across `herd_x` columns
(each column owns a contiguous `C/herd_x` channel slice) and the **sequence**
axis across `herd_y` rows (each row owns every `herd_y`-th `tile_s` block).
Each tile issues 3 independent shim DMAs (x in, w in, y out).

Note the halo: a tile producing `tile_s` output rows must read `tile_s + 2`
input rows, so small `tile_s` pays a `2/tile_s` read-amplification.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

K_TAPS = 3
HALO = K_TAPS - 1  # 2 rows of left context
L1_BYTES = 65536  # per-compute-tile L1 on AIE2P


@module_builder
def build_module(seq, channels, tile_s, np_dtype_in, herd_x=8, herd_y=1):
    xrt_dtype = type_mapper(np_dtype_in)

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

    # L3 types. x is pre-padded with HALO rows; y is not.
    l3XTy = MemRefType.get([seq + HALO, channels], xrt_dtype)
    l3WTy = MemRefType.get([K_TAPS, channels], xrt_dtype)
    l3YTy = MemRefType.get([seq, channels], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1XTy = MemRefType.get(
        shape=[tile_s + HALO, tile_c], element_type=xrt_dtype, memory_space=l1_mem_space
    )
    l1WTy = MemRefType.get(
        shape=[K_TAPS, tile_c], element_type=xrt_dtype, memory_space=l1_mem_space
    )
    l1YTy = MemRefType.get(
        shape=[tile_s, tile_c], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    conv_func = FuncOp(
        "conv1d_depthwise_bf16",
        ([l1XTy, l1WTy, l1YTy, T.i32(), T.i32()], []),
        visibility="private",
    )
    conv_func.attributes["link_with"] = StringAttr.get("conv1d_depthwise.o")
    conv_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # global_channel = tx * tile_c
    chan_map = AffineMap.get(
        0,
        1,
        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_c))],
    )
    # global_row = loop_iv + ty * tile_s
    row_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineSymbolExpr.get(0),
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(1), AffineConstantExpr.get(tile_s)
                ),
            )
        ],
    )

    @FuncOp.from_py_func(l3XTy, l3WTy, l3YTy)
    def conv1d_depthwise(arg0, arg1, arg2):
        # arg0 = x padded [seq+2, C], arg1 = w tap-major [3, C], arg2 = y [seq, C]

        @launch(operands=[arg0, arg1, arg2])
        def conv_launch(l_x, l_w, l_y):

            @segment(name="conv1d_seg", operands=[l_x, l_w, l_y])
            def conv_seg(s_x, s_w, s_y):

                @herd(
                    name="herd_0",
                    sizes=[herd_x, herd_y],
                    operands=[s_x, s_w, s_y],
                )
                def herd_body(_tx, _ty, _sx, _sy, l3_x, l3_w, l3_y):
                    l1_x = AllocOp(l1XTy, [], [])
                    l1_w = AllocOp(l1WTy, [], [])
                    l1_y = AllocOp(l1YTy, [], [])

                    ts_i32 = ConstantOp(T.i32(), tile_s)
                    tc_i32 = ConstantOp(T.i32(), tile_c)

                    chan = affine_apply(chan_map, [_tx])

                    # Weights are seq-independent: load this tile's channel
                    # slice once, outside the sequence loop.
                    dma_memcpy_nd(
                        l1_w,
                        l3_w,
                        src_offsets=[0, chan],
                        src_sizes=[K_TAPS, tile_c],
                        src_strides=[channels, 1],
                    )

                    for loop_iv in range_(0, seq, tile_s * herd_y):
                        row = affine_apply(row_map, [loop_iv, _ty])

                        # Read tile_s + HALO rows starting at `row`. Because x
                        # is pre-padded, x[row] is already the oldest sample
                        # feeding y[row] — no negative indexing, no masking.
                        dma_memcpy_nd(
                            l1_x,
                            l3_x,
                            src_offsets=[row, chan],
                            src_sizes=[tile_s + HALO, tile_c],
                            src_strides=[channels, 1],
                        )

                        CallOp(conv_func, [l1_x, l1_w, l1_y, ts_i32, tc_i32])

                        dma_memcpy_nd(
                            l3_y,
                            l1_y,
                            dst_offsets=[row, chan],
                            dst_sizes=[tile_s, tile_c],
                            dst_strides=[channels, 1],
                        )
                        yield_([])

                    DeallocOp(l1_x)
                    DeallocOp(l1_w)
                    DeallocOp(l1_y)

                herd_body.attributes["link_with"] = StringAttr.get("conv1d_depthwise.o")


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
    args = parser.parse_args()

    if args.perf_iters < 0:
        parser.error("--perf-iters must be >= 0")

    mlir_module = build_module(
        args.seq,
        args.channels,
        args.tile_s,
        bfloat16,
        herd_x=args.herd_x,
        herd_y=args.herd_y,
    )
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
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
