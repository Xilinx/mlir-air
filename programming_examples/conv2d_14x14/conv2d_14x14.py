# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""mlir-air port of upstream mlir-aie programming_examples/ml/conv2d_14x14.

Targets both NPU2 (Strix, 8x4 herd, 32 cores) and NPU1 (Phoenix, 4x4 herd,
16 cores). On NPU1 each column processes twice as many oc-groups (18 vs 9)
to keep total work identical at 72 oc-groups across CO=1152.

Both paths execute 14x14 stride-14 conv on 896x896 input, 4 in-channels,
1152 out-channels, uint8 act / int8 wts / int8 out.

Every transfer here is a *view*, not a copy: the buffer keeps its declared
shape and the reshape/transpose describes the order the hardware walks it.
The three that are not simply contiguous:

* **L3 -> L2 activations.** The input is declared ``[4, 802816]`` and read as
  ``[4, 14, 64, 56]`` over strides ``[802816, 3584, 56, 1]`` -- a band of 14
  rows per memtile. ``I[:, y0:y0+50176].reshape(4, 14, 64, 56)`` is that,
  and it is the case ``python/test/api/tensor_views.py`` pins.
* **The L2 side of the same transfer** wants ``[50176, 56, 784, 1]``, whose
  middle two strides are swapped relative to row-major. Splitting the memtile
  band the other way round and permuting gets there:
  ``reshape(4, 64, 14, 56).transpose(0, 2, 1, 3)``.
* **L2 -> L1 activations** is the 6-D gather the kernel expects,
  ``[1, 1, 2, 98, 8, 8]`` over ``[50176, 12544, 6272, 8, 784, 1]``. Same idiom
  one rank up: ``reshape(4, 4, 2, 8, 98, 8).transpose(0, 1, 2, 4, 3, 5)``,
  then subscript the two leading axes. An integer subscript keeps its axis in
  the transfer's sizes, which is where the two leading 1s come from.

All compute is in ``conv2dk14.o``; this file moves data and calls it.
"""

import argparse
import numpy as np
import torch

from air import api as air
from air.api import ops
from air.api.types import i8, i32
from air.backend.xrt_runner import XRTRunner

KERNEL = "conv2dk14.o"


def build_launch(n_cols=8, num_g=9):
    """An n_cols x 4 herd processing num_g oc-groups per column.

    n_cols * num_g == 72, the total oc-groups across CO=1152.
    """
    assert (
        n_cols * num_g == 72
    ), f"Expected n_cols * num_g == 72, got {n_cols} * {num_g} = {n_cols * num_g}"
    wts_per_col = num_g * 12544  # bytes per col in the L3 weight slab
    out_per_col = num_g * 65536  # bytes per col in the L3 output slab

    I = air.tensor([4, 802816], i8)
    W = air.tensor([n_cols, wts_per_col], i8)
    O = air.tensor([n_cols, out_per_col], i8)

    conv = air.extern("conv2dk14_i8", link_with=KERNEL, scalars=[i32] * 5)

    with air.launch(name="conv2dk14_test") as launch:

        @launch.body
        def _():
            with air.segment(name="conv2dk14_seg") as seg:

                @seg.body
                def _():
                    # One full 14-row image band per memtile, and the output
                    # staging slab the herd drains into.
                    l2_act = air.alloc([4, 14, 3584], i8, scope=seg.private())
                    l2_out = air.alloc([n_cols, 4, 64, 256], i8, scope=seg.private())

                    # L3 -> L2: scatter the band. The destination walk swaps
                    # the two middle axes relative to row-major, hence the
                    # transpose on the L2 side.
                    for _gp in air.sequential(num_g):
                        for yp in air.sequential(16):
                            y14 = yp * 14
                            ops.load(
                                l2_act.reshape(4, 64, 14, 56).transpose(0, 2, 1, 3),
                                I.reshape(4, 224, 64, 56)[:, y14 : y14 + 14, :, :],
                            )

                    with air.herd(
                        [range(n_cols), range(4)],
                        name="conv2dk14_herd",
                        shape=(n_cols, 4),
                        link_with=KERNEL,
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            l1_in = air.alloc([12544], i8, scope=h.private())
                            l1_wts = air.alloc([12544], i8, scope=h.private())
                            l1_out = air.alloc([256], i8, scope=h.private())

                            # The 6-D gather, subscripted at (memtile, quarter).
                            act = l2_act.reshape(4, 4, 2, 8, 98, 8).transpose(
                                0, 1, 2, 4, 3, 5
                            )

                            for g in air.sequential(num_g):
                                w0 = g * 12544
                                ops.load(l1_wts, W[tx, w0 : w0 + 12544])
                                for y in air.sequential(16):
                                    for xb in air.sequential(4):
                                        ops.load(l1_in, act[ty, xb, :, :, :, :])
                                        conv(l1_in, l1_wts, l1_out, 224, 4, 16, 14, 14)
                                        ops.store(
                                            l1_out, l2_out[tx, ty, y * 4 + xb, 0:256]
                                        )

                    # L2 -> L3: one flat 65536-B/col drain per g iteration.
                    for gg in air.sequential(num_g):
                        g0 = gg * 65536
                        ops.store(l2_out.reshape(n_cols, 65536), O[:, g0 : g0 + 65536])

    return launch


# Problem dims (fixed to match the IR string)
WIDTH = 896
HEIGHT = 896
CI = 4
CO = 1152
KSZ = 14
WIDTH_OUT = WIDTH // KSZ  # 64
HEIGHT_OUT = HEIGHT // KSZ  # 64
CLIP_MIN, CLIP_MAX = -128, 127

# (n_cols, num_g) per target device. n_cols * num_g must == 72.
DEVICE_LAYOUTS = {
    "npu2": (8, 9),
    "npu1": (4, 18),
}


def build_inputs_and_golden(n_cols, num_g):
    """Generate the inputs and the byte-permuted golden expected by the NPU
    output layout.

    The NPU writes 4,718,592 raw bytes laid out as
        [col=n_cols][g=num_g][row=4][yxb=64][oc=2][nt=2][nt8=8][oc8=8]
    where each 256-B call's inner [oc, nt, nt8, oc8] is the kernel-write
    order. The host-expected layout (which matches PyTorch's (CO, HOUT,
    WOUT)) is [nt, nt8, oc, oc8]; we permute the golden's bytes here so
    XRTRunner's bit-exact compare succeeds end-to-end.
    """
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    int_inp = torch.randint(0, 255, (1, CI, HEIGHT, WIDTH)).type(torch.FloatTensor)
    int_weight = torch.randint(2, 20, (CO, CI, KSZ, KSZ)).type(torch.FloatTensor)

    # Golden via pure integer arithmetic so it matches the kernel's
    # `(sum + sign*8192) >> 14` rounding (round half away from zero). Using
    # torch.round() introduces banker's rounding mismatches on ties.
    inp_t = int_inp.to(torch.int32).float()
    wt_t = int_weight.float()
    sum_int = (
        torch.nn.functional.conv2d(inp_t, wt_t, stride=KSZ, padding=0)
        .squeeze(0)
        .numpy()
        .astype(np.int32)
    )
    bias = np.where(sum_int >= 0, 8192, -8192).astype(np.int32)
    quantized = (sum_int + bias) // 16384
    golden_int = np.clip(quantized, CLIP_MIN, CLIP_MAX).astype(np.int8)
    # golden_int shape: (CO, HEIGHT_OUT, WIDTH_OUT) = (1152, 64, 64)

    # Host act buffer must be YXC-packed per col so the per-pixel (CI=4)
    # bytes are contiguous - which is what the L3->L2 scatter wrap and the
    # kernel's gather wrap both assume. The raw torch tensor is (CI, H, W);
    # transpose to (H, W, CI) then flatten gives the YXC stream.
    int_inp_np = int_inp.squeeze().data.numpy().astype(np.uint8)  # (CI, H, W)
    in1 = (
        np.transpose(int_inp_np, (1, 2, 0)).reshape(CI, HEIGHT * WIDTH).astype(np.uint8)
    )
    # Host expects (CI, HEIGHT * WIDTH) = (4, 802816); air.func arg shape
    # matches.

    # Weights need to be reshaped to OIYX -> OYXIO8 grouped form so the
    # (n_cols, num_g*12544) buffer matches what the kernel reads. Upstream
    # uses DataShaper.reorder_mat("OYXIO8", "OIYX"). Total CO//8 = 144
    # 8-OC blocks packed contiguously and then split across n_cols columns;
    # the kernel consumes them in pairs (one "oc-group" / kernel call =
    # 16 OC = two CO//8 blocks = 12544 bytes).
    wts_np = int_weight.data.numpy().astype(np.int8)  # (CO, CI, KSZ, KSZ)
    # OYXIO8 layout: split CO into (CO//8, 8). Source label "OIYX" means
    # axes (CO, CI, Y, X) which is what we have. Target label "OYXIO8"
    # means (CO//8, Y, X, CI, 8 (= oc inner)). Reorder:
    #   OIYX -> OYXIO8:
    #   step 1: split O into (O_outer = CO//8, O_inner = 8): (O_outer, 8,
    #           CI, Y, X)
    #   step 2: permute to (O_outer, Y, X, CI, 8):
    co_outer = CO // 8
    wts_split = wts_np.reshape(co_outer, 8, CI, KSZ, KSZ)
    wts_oyxio8 = np.transpose(wts_split, (0, 3, 4, 2, 1))  # (co_outer, Y, X, CI, 8)
    # Per-col block: num_g oc-groups (kernel calls), 12544 B each (= 2
    # CO//8 slices). Each CO//8 slice = KSZ*KSZ*CI*8 = 6272 B. Total
    # buffer = CO * CI * KSZ * KSZ = 903168 B = n_cols * (num_g * 12544).
    in2 = wts_oyxio8.tobytes()
    in2_arr = np.frombuffer(in2, dtype=np.int8).reshape(n_cols, num_g * 12544)

    # Construct the byte-permuted golden expected at the L3 output.
    # NPU raw bytes (4,718,592) reshape as
    #   (col=n_cols, g=num_g, row=4, yxb=64, 2, 2, 8, 8) - last 4 dims
    #   are the kernel write order [oc, nt, nt8, oc8].
    #
    # Spatial layout from golden: (CO=1152, HOUT=64, WOUT=64).
    # Co-group decomposition: CO = n_cols * num_g * 16 oc-per-group =
    #   col*num_g*16 + g*16 + oc_in_group, where oc_in_group = oc*8 + oc8.
    # Output decomposition: HOUT = 4 rows * 16 y = row*16 + y. WOUT =
    #   4 xb * 16 (nt*8 + nt8) = xb*16 + nt*8 + nt8.
    g_arr = golden_int.reshape(n_cols, num_g, 2, 8, 4, 16, 4, 2, 8)
    # axes: (col, g, oc=2, oc8=8, row=4, y=16, xb=4, nt=2, nt8=8)
    g_arr = np.transpose(g_arr, (0, 1, 4, 5, 6, 2, 7, 8, 3))
    # now: (col, g, row=4, y=16, xb=4, oc=2, nt=2, nt8=8, oc8=8)
    g_arr = g_arr.reshape(n_cols, num_g, 4, 16 * 4, 2, 2, 8, 8)
    # (col, g, row, yxb=64, oc, nt, nt8, oc8) - matches NPU byte layout.
    expected_out_npu = g_arr.reshape(n_cols, num_g * 65536).astype(np.int8)

    return in1, in2_arr, expected_out_npu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="conv2d_14x14.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    parser.add_argument(
        "--target-device",
        type=str,
        choices=sorted(DEVICE_LAYOUTS.keys()),
        default="npu2",
        dest="target_device",
        help="Target NPU device. npu2: 8x4 herd, 9 oc-groups/col. "
        "npu1: 4x4 herd, 18 oc-groups/col. Same total work.",
    )
    args = parser.parse_args()

    n_cols, num_g = DEVICE_LAYOUTS[args.target_device]

    launch = build_launch(n_cols=n_cols, num_g=num_g)
    if args.print_module_only:
        print(launch.mlir())
        exit(0)

    in1, in2, expected_out = build_inputs_and_golden(n_cols, num_g)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="conv2dk14_test",
        lower_linalg_to_func="conv2dk14.o",
        target_device=args.target_device,
    )
    exit(
        runner.run_test(
            launch.build(target=args.target_device),
            inputs=[in1, in2],
            expected_outputs=[expected_out],
        )
    )
